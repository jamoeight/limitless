"""End-to-end test for MVP-3 virtualization wiring.

We enable virtualization in CortexSettings, inject a fake provider that
RECORDS the upstream request it receives, and assert:
  - The provider sees fewer messages than the client sent.
  - The provider sees the original system prompt PLUS a recap block.
  - The `X-Cortex-Virtualized: true` response header is set.
  - The X-Cortex-Disable-Virtualize request header bypasses virtualization.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest

from cortex.canonical import (
    ChunkContentBlockStart,
    ChunkContentBlockStop,
    ChunkMessageDelta,
    ChunkMessageStart,
    ChunkMessageStop,
    ChunkTextDelta,
    CortexChunk,
    CortexRequest,
    TextBlock,
)
from cortex.config import CortexSettings
from cortex.ingest import SessionRegistry
from cortex.server import ProviderRegistry, _build_app


class RecordingProvider:
    name = "anthropic"

    def __init__(self, scripted: list[CortexChunk]) -> None:
        self._scripted = scripted
        self.last_request: CortexRequest | None = None

    async def stream(
        self,
        req: CortexRequest,
        api_key: str,
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncIterator[CortexChunk]:
        self.last_request = req
        for c in self._scripted:
            yield c

    async def aclose(self) -> None:
        pass


def _text_response(text: str) -> list[CortexChunk]:
    return [
        ChunkMessageStart(message_id="msg_v", model="claude-opus-4-7", input_tokens=12),
        ChunkContentBlockStart(index=0, block=TextBlock(text="")),
        ChunkTextDelta(index=0, text=text),
        ChunkContentBlockStop(index=0),
        ChunkMessageDelta(stop_reason="end_turn", output_tokens=4),
        ChunkMessageStop(),
    ]


async def _stub_recall(query: str, group_id: str, token_budget: int) -> str:
    return (
        "(fact 1: alice likes tea)\n"
        "(fact 2: bob lives in seattle)\n"
        "(fact 3: project deadline 2026-06-15)"
    )


async def _stub_verbatim_recall(query, cold_groups, k, token_budget):
    """Stub that returns the first cold group's text verbatim, simulating a
    successful inline-embedding retrieval pass without needing LM Studio."""
    if not cold_groups:
        return ""
    g = cold_groups[0]
    lines = [f"[turn 0.{i} {m.role}]\n" + (m.content[0].text if m.content else "")
             for i, m in enumerate(g)]
    return "Verbatim retrieved (stub):\n" + "\n".join(lines)


async def _no_ingest(content, source, group_id, session_id, event_time):
    return ""


def _setup_app(
    *,
    enable_virtualization: bool = True,
    last_k_spans: int = 2,
    recall_fn=_stub_recall,
    verbatim_recall_fn=None,
    enable_verbatim_recall: bool = False,
    # Small context limit + zero safety margin forces virtualization on the
    # synthetic conversations in this suite (each ~50 char msg → ~12 tokens).
    # 150 + 0 - 64 (max_tokens) = 86-token budget. 4 verbatim msgs ≈ 40 tokens
    # (fits); 12 msgs ≈ 102 tokens (overflows → trims). Real production
    # setups use the model's true context window. The short-circuit-when-fits
    # behavior is exercised in test_virtualize.py.
    upstream_context_limit: int = 150,
):
    settings = CortexSettings(
        enable_virtualization=enable_virtualization,
        enable_auto_ingest=False,  # focus this suite on virtualize only
        last_k_spans=last_k_spans,
        upstream_context_limit=upstream_context_limit,
        safety_margin_tokens=0,
        # Default off: this suite exercises the cold-summary + graph-recall
        # code paths. Verbatim recall is covered by a dedicated test below.
        enable_verbatim_recall=enable_verbatim_recall,
    )
    provider = RecordingProvider(_text_response("ok"))
    registry = ProviderRegistry()
    registry.register(provider)
    session_registry = SessionRegistry(settings, ingest_fn=_no_ingest)
    app = _build_app(
        settings=settings,
        registry=registry,
        session_registry=session_registry,
        recall_fn=recall_fn,
        verbatim_recall_fn=verbatim_recall_fn,
    )
    return app, provider


@asynccontextmanager
async def _live(app):
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


def _long_conversation(n_groups: int) -> list[dict]:
    msgs: list[dict] = []
    for i in range(n_groups):
        msgs.append({"role": "user", "content": f"user message {i} long enough to virtualize"})
        msgs.append({"role": "assistant", "content": f"assistant reply {i} long enough"})
    return msgs


# ---------- Virtualization actually shrinks the upstream request ----------


@pytest.mark.asyncio
async def test_virtualization_reduces_message_count_when_enabled() -> None:
    app, provider = _setup_app(enable_virtualization=True, last_k_spans=2)
    async with _live(app) as client:
        resp = await client.post(
            "/v1/messages",
            headers={"x-api-key": "sk-x", "x-cortex-group-id": "g", "x-cortex-session-id": "s"},
            json={
                "model": "claude-opus-4-7",
                "max_tokens": 64,
                "system": "You are a careful assistant.",
                "messages": _long_conversation(8),
                "stream": False,
            },
        )

    assert resp.status_code == 200
    assert provider.last_request is not None
    # Client sent 16 messages (8 groups × 2). Last 2 groups → 4 messages kept.
    assert len(provider.last_request.messages) == 4
    # Cortex headers report what happened.
    assert resp.headers.get("x-cortex-virtualized") == "true"
    assert resp.headers.get("x-cortex-original-messages") == "16"
    assert resp.headers.get("x-cortex-kept-messages") == "4"
    assert int(resp.headers.get("x-cortex-recap-tokens", "0")) > 0


@pytest.mark.asyncio
async def test_virtualization_injects_recap_into_system_prompt() -> None:
    app, provider = _setup_app(enable_virtualization=True, last_k_spans=2)
    async with _live(app) as client:
        await client.post(
            "/v1/messages",
            headers={"x-api-key": "sk-x", "x-cortex-group-id": "g", "x-cortex-session-id": "s"},
            json={
                "model": "claude-opus-4-7",
                "max_tokens": 64,
                "system": "You are a careful assistant.",
                "messages": _long_conversation(6),
                "stream": False,
            },
        )

    assert provider.last_request is not None
    sys = provider.last_request.system or ""
    # Original prompt preserved verbatim at the start.
    assert sys.startswith("You are a careful assistant.")
    # Recap block appended.
    assert "<cortex_memory>" in sys
    assert "Older conversation context" in sys
    assert "Relevant retrieved knowledge" in sys
    # The stub recall content made it in.
    assert "alice likes tea" in sys
    assert "project deadline 2026-06-15" in sys


# ---------- Disable virtualization ----------


@pytest.mark.asyncio
async def test_disable_virtualize_header_bypasses_virtualization() -> None:
    app, provider = _setup_app(enable_virtualization=True, last_k_spans=2)
    async with _live(app) as client:
        resp = await client.post(
            "/v1/messages",
            headers={
                "x-api-key": "sk-x",
                "x-cortex-group-id": "g",
                "x-cortex-session-id": "s",
                "x-cortex-disable-virtualize": "1",
            },
            json={
                "model": "claude-opus-4-7",
                "max_tokens": 64,
                "system": "You are a careful assistant.",
                "messages": _long_conversation(8),
                "stream": False,
            },
        )

    assert provider.last_request is not None
    # All 16 messages forwarded verbatim.
    assert len(provider.last_request.messages) == 16
    # System unchanged.
    assert provider.last_request.system == "You are a careful assistant."
    # Virtualized header should not be set (or set to a no-op value).
    assert "x-cortex-virtualized" not in {k.lower() for k in resp.headers}


@pytest.mark.asyncio
async def test_settings_off_skips_virtualization() -> None:
    app, provider = _setup_app(enable_virtualization=False, last_k_spans=2)
    async with _live(app) as client:
        await client.post(
            "/v1/messages",
            headers={"x-api-key": "sk-x"},
            json={
                "model": "claude-opus-4-7",
                "max_tokens": 64,
                "system": "sys",
                "messages": _long_conversation(8),
                "stream": False,
            },
        )

    assert provider.last_request is not None
    assert len(provider.last_request.messages) == 16  # unchanged
    assert provider.last_request.system == "sys"


# ---------- Verbatim recall path ----------


@pytest.mark.asyncio
async def test_verbatim_recall_injects_retrieved_history_block() -> None:
    """When verbatim_recall_fn returns content, it goes into a dedicated
    Verbatim section and supersedes the cold-summary (which mostly wastes
    budget once we have the exact text)."""
    app, provider = _setup_app(
        enable_virtualization=True,
        last_k_spans=2,
        enable_verbatim_recall=True,
        verbatim_recall_fn=_stub_verbatim_recall,
    )
    async with _live(app) as client:
        await client.post(
            "/v1/messages",
            headers={"x-api-key": "sk-x", "x-cortex-group-id": "g", "x-cortex-session-id": "s"},
            json={
                "model": "claude-opus-4-7",
                "max_tokens": 64,
                "system": "You are a careful assistant.",
                "messages": _long_conversation(6),
                "stream": False,
            },
        )

    assert provider.last_request is not None
    sys = provider.last_request.system or ""
    # Verbatim block landed in the recap with its own header.
    assert "Relevant verbatim turns" in sys
    assert "Verbatim retrieved (stub)" in sys
    # When verbatim succeeds, cold_summary is suppressed (verbatim is strictly
    # better for content-faithful retrieval; cold_summary would burn budget
    # without adding signal).
    assert "Older conversation context" not in sys
    # Original system prompt still preserved at the start.
    assert sys.startswith("You are a careful assistant.")


@pytest.mark.asyncio
async def test_verbatim_recall_failure_falls_back_to_cold_summary() -> None:
    """If verbatim_recall_fn raises, the recap still gets built — cold_summary
    fills in as the fallback."""

    async def broken_verbatim(query, cold_groups, k, token_budget):
        raise RuntimeError("embedder backend exploded")

    app, provider = _setup_app(
        enable_virtualization=True,
        last_k_spans=2,
        enable_verbatim_recall=True,
        verbatim_recall_fn=broken_verbatim,
    )
    async with _live(app) as client:
        resp = await client.post(
            "/v1/messages",
            headers={"x-api-key": "sk-x"},
            json={
                "model": "claude-opus-4-7",
                "max_tokens": 64,
                "system": "sys",
                "messages": _long_conversation(6),
                "stream": False,
            },
        )

    # Request still succeeds despite verbatim recall failing.
    assert resp.status_code == 200
    assert provider.last_request is not None
    sys = provider.last_request.system or ""
    # Verbatim section absent; cold-summary present as fallback.
    assert "Relevant verbatim turns" not in sys
    assert "Older conversation context" in sys


# ---------- Recall failures don't break the request ----------


@pytest.mark.asyncio
async def test_recall_failure_degrades_to_cold_summary_only() -> None:
    async def broken_recall(query, group_id, token_budget):
        raise RuntimeError("graph backend exploded")

    app, provider = _setup_app(
        enable_virtualization=True, last_k_spans=2, recall_fn=broken_recall
    )
    async with _live(app) as client:
        resp = await client.post(
            "/v1/messages",
            headers={"x-api-key": "sk-x"},
            json={
                "model": "claude-opus-4-7",
                "max_tokens": 64,
                "system": "sys",
                "messages": _long_conversation(6),
                "stream": False,
            },
        )

    # Response still succeeds.
    assert resp.status_code == 200
    # Provider still got a virtualized request (cold_summary made it in,
    # recall_text empty).
    assert provider.last_request is not None
    assert "<cortex_memory>" in (provider.last_request.system or "")
    # But the recall section is absent because recall_fn raised.
    assert "Relevant retrieved knowledge" not in (provider.last_request.system or "")
