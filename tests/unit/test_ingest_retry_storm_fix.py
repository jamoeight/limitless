"""Regression tests for the ingest retry storm (BUG 3).

What the storm looked like before the fix:
  - A 500KB WebSearch result lands in cortex.ingest as one episode.
  - add_episode forwards it to the extractor.
  - Extractor's HTTP call times out (LM Studio choked, or anthropic throttled).
  - tenacity retries 2x with 1-4s backoff → ~60s per failed extraction.
  - Many such episodes queue at once → cortex.log fills with `extractor
    failed; recording episode with 0 facts error='RetryError[...]'` lines
    and CPU/RAM stays pinned.

What the fix does:
  - Episodes larger than `extractor_skip_threshold_chars` (default 30_000)
    skip the LLM extractor entirely. The episode is still recorded and
    embedded; it just doesn't burn an LLM call to triple-distil a giant
    payload into facts.
  - tenacity is capped at 1 attempt (no retries) — fail-fast.
  - A hard `asyncio.wait_for(extractor_call_timeout_s)` wraps every call
    so a hung HTTP body can't outlive the configured budget.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from timegraph.config import get_settings
from timegraph.llm.extractor import ExtractorClient


# ---------- size-based skip (the main fix) ----------


@pytest.mark.asyncio
async def test_large_payload_skips_extractor_in_add_episode(monkeypatch) -> None:
    """A 100KB content does NOT call the extractor at all — even when the
    extractor would have succeeded. Confirms the size short-circuit."""
    from timegraph.ops import add_episode as mod
    from timegraph.types import AddEpisodeIn

    extractor_calls = 0

    class _FailIfCalled:
        def __init__(self) -> None:
            pass

        async def extract_facts(self, *a, **kw):
            nonlocal extractor_calls
            extractor_calls += 1
            raise AssertionError("extractor should be skipped for large payloads")

        async def close(self) -> None:
            pass

    create_calls: list[dict] = []
    upsert_calls: list[dict] = []

    async def _fake_create_episode_node(**kw):
        create_calls.append(kw)

    async def _fake_upsert_point(**kw):
        upsert_calls.append(kw)

    class _FakeEmbedder:
        async def embed_one(self, text: str) -> list[float]:
            return [0.0] * 768

        async def embed_many(self, texts):
            return [[0.0] * 768 for _ in texts]

        async def close(self) -> None:
            pass

    monkeypatch.setattr(mod, "ExtractorClient", _FailIfCalled)
    monkeypatch.setattr(mod, "_create_episode_node", _fake_create_episode_node)
    monkeypatch.setattr(mod, "upsert_point", _fake_upsert_point)
    monkeypatch.setattr(mod, "EmbedderClient", lambda: _FakeEmbedder())

    big = "x " * 60_000  # ~120 KB, well above default 30 KB threshold

    out = await mod.add_episode(
        AddEpisodeIn(
            content=big,
            source="tool:web_search",
            group_id="g1",
            session_id="s1",
            event_time=datetime(2026, 5, 25, tzinfo=timezone.utc),
        )
    )

    assert extractor_calls == 0
    assert out.extracted_facts == []
    assert len(create_calls) == 1
    # The episode IS embedded so vector recall still surfaces it.
    assert any(c.get("collection") == get_settings().qdrant_episodes_collection for c in upsert_calls)


@pytest.mark.asyncio
async def test_small_payload_still_calls_extractor(monkeypatch) -> None:
    """A short message keeps the existing behavior — extractor runs."""
    from timegraph.ops import add_episode as mod
    from timegraph.types import AddEpisodeIn

    extractor_calls = 0

    class _Extractor:
        async def extract_facts(self, *a, **kw):
            nonlocal extractor_calls
            extractor_calls += 1
            return [], 12.3

        async def close(self) -> None:
            pass

    async def _fake_create_episode_node(**kw):
        pass

    async def _fake_upsert_point(**kw):
        pass

    class _FakeEmbedder:
        async def embed_one(self, text: str) -> list[float]:
            return [0.0] * 768

        async def embed_many(self, texts):
            return [[0.0] * 768 for _ in texts]

        async def close(self) -> None:
            pass

    monkeypatch.setattr(mod, "ExtractorClient", lambda: _Extractor())
    monkeypatch.setattr(mod, "_create_episode_node", _fake_create_episode_node)
    monkeypatch.setattr(mod, "upsert_point", _fake_upsert_point)
    monkeypatch.setattr(mod, "EmbedderClient", lambda: _FakeEmbedder())

    await mod.add_episode(
        AddEpisodeIn(
            content="Alice lives in Paris.",
            source="msg:user",
            group_id="g1",
            session_id="s1",
            event_time=datetime(2026, 5, 25, tzinfo=timezone.utc),
        )
    )

    assert extractor_calls == 1


# ---------- retries capped at 1 attempt ----------


@pytest.mark.asyncio
async def test_extract_facts_does_not_retry_on_failure(monkeypatch) -> None:
    """tenacity is now stop_after_attempt(1). One failure → ValueError up,
    no double-tap, no 60s of exponential backoff."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-T")
    monkeypatch.setenv("TG_EXTRACTOR_BACKEND", "anthropic_api")

    call_count = 0
    extractor = ExtractorClient()

    async def _always_fail(prompt: str) -> tuple[str, float]:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("upstream is angry")

    extractor._extract_via_anthropic_api = _always_fail  # type: ignore[method-assign]

    try:
        with pytest.raises(RuntimeError, match="upstream is angry"):
            await extractor.extract_facts(
                episode_content="short content",
                event_time=datetime(2026, 5, 25, tzinfo=timezone.utc),
                session_id="s1",
                source="msg:user",
            )
    finally:
        await extractor.close()

    assert call_count == 1, "tenacity must not retry — one attempt only"


# ---------- hard outer timeout ----------


@pytest.mark.asyncio
async def test_extract_facts_enforces_hard_timeout(monkeypatch) -> None:
    """If the inner backend hangs past extractor_call_timeout_s, the outer
    asyncio.wait_for must fire and surface a ValueError. This is the
    belt-and-suspenders guarantee against a misbehaving HTTP client."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-T")
    monkeypatch.setenv("TG_EXTRACTOR_BACKEND", "anthropic_api")
    monkeypatch.setenv("TG_EXTRACTOR_CALL_TIMEOUT_S", "0.05")

    extractor = ExtractorClient()

    async def _hangs(prompt: str) -> tuple[str, float]:
        await asyncio.sleep(5)
        return "{}", 0.0

    extractor._extract_via_anthropic_api = _hangs  # type: ignore[method-assign]

    try:
        with pytest.raises(ValueError, match="hard timeout"):
            await extractor.extract_facts(
                episode_content="short content",
                event_time=datetime(2026, 5, 25, tzinfo=timezone.utc),
                session_id="s1",
                source="msg:user",
            )
    finally:
        await extractor.close()
