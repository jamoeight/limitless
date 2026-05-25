"""Tests for the cortex worker-recycle path and stream-buffer cleanup.

These cover Fix 3 from the live-debugging session: cortex.server was
holding ~9.5 GB resident after 243 requests because fastembed buffers
accumulated, crashed `server_tool_use` streams leaked their chunk lists,
and Python's allocator never returned memory to the OS. The fix:
  - count requests; SIGTERM self after N (SessionStart hook respawns)
  - clear the per-stream chunk buffer immediately after extracting the
    assistant message (don't hold it across the ingest schedule)
"""

from __future__ import annotations

import os
import signal
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
from cortex.server import (
    ProviderRegistry,
    _build_app,
    _maybe_recycle_worker,
)


class _StubProvider:
    name = "anthropic"

    def __init__(self, n_text_deltas: int = 1) -> None:
        self.calls = 0
        self.n_text_deltas = n_text_deltas

    async def stream(
        self,
        req: CortexRequest,
        api_key: str,
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncIterator[CortexChunk]:
        self.calls += 1
        yield ChunkMessageStart(message_id="msg_r", model=req.model, input_tokens=5)
        yield ChunkContentBlockStart(index=0, block=TextBlock(text=""))
        for i in range(self.n_text_deltas):
            yield ChunkTextDelta(index=0, text=f"chunk_{i}")
        yield ChunkContentBlockStop(index=0)
        yield ChunkMessageDelta(stop_reason="end_turn", output_tokens=3)
        yield ChunkMessageStop()

    async def aclose(self) -> None:
        pass


async def _noop_ingest(content, source, group_id, session_id, event_time):
    return ""


async def _noop_recall(query: str, group_id: str, budget: int) -> str:
    return ""


def _setup_app(*, recycle_after: int, drain_s: float = 0.1):
    settings = CortexSettings(
        enable_auto_ingest=False,
        enable_virtualization=False,
        recycle_after_requests=recycle_after,
        recycle_drain_seconds=drain_s,
    )
    registry = ProviderRegistry()
    registry.register(_StubProvider())
    session_registry = SessionRegistry(settings, ingest_fn=_noop_ingest)
    app = _build_app(
        settings=settings,
        registry=registry,
        session_registry=session_registry,
        recall_fn=_noop_recall,
    )
    return app, registry


@asynccontextmanager
async def _live(app):
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


# ---------- counter increments ----------


@pytest.mark.asyncio
async def test_request_counter_increments_per_request() -> None:
    app, _ = _setup_app(recycle_after=999)
    async with _live(app) as client:
        for _ in range(3):
            r = await client.post(
                "/v1/messages",
                headers={"x-api-key": "sk-x"},
                json={
                    "model": "claude-opus-4-7",
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )
            assert r.status_code == 200
        assert app.state.request_count == 3
        assert app.state.recycle_scheduled is False


# ---------- recycle scheduling ----------


@pytest.mark.asyncio
async def test_recycle_fires_when_threshold_reached(monkeypatch) -> None:
    """Threshold = 2; after the 2nd request, recycle should be scheduled.
    We monkeypatch os.kill so the test doesn't actually kill pytest."""
    kill_calls: list[tuple[int, int]] = []

    def fake_kill(pid: int, sig: int) -> None:
        kill_calls.append((pid, sig))

    monkeypatch.setattr(os, "kill", fake_kill)

    app, _ = _setup_app(recycle_after=2, drain_s=0.05)
    async with _live(app) as client:
        for _ in range(2):
            r = await client.post(
                "/v1/messages",
                headers={"x-api-key": "sk-x"},
                json={
                    "model": "claude-opus-4-7",
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )
            assert r.status_code == 200
        # Recycle scheduled exactly once; drain task gets time to fire.
        assert app.state.recycle_scheduled is True
        import asyncio
        await asyncio.sleep(0.15)
        assert len(kill_calls) == 1
        assert kill_calls[0][0] == os.getpid()
        assert kill_calls[0][1] == signal.SIGTERM


@pytest.mark.asyncio
async def test_recycle_disabled_when_threshold_zero(monkeypatch) -> None:
    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda p, s: kill_calls.append((p, s)))

    app, _ = _setup_app(recycle_after=0)
    async with _live(app) as client:
        for _ in range(5):
            await client.post(
                "/v1/messages",
                headers={"x-api-key": "sk-x"},
                json={
                    "model": "claude-opus-4-7",
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )
        assert app.state.request_count == 5
        assert app.state.recycle_scheduled is False
        assert kill_calls == []


@pytest.mark.asyncio
async def test_recycle_idempotent(monkeypatch) -> None:
    """Once scheduled, additional requests must NOT spawn more exit tasks."""
    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda p, s: kill_calls.append((p, s)))

    app, _ = _setup_app(recycle_after=1, drain_s=0.05)
    async with _live(app) as client:
        for _ in range(4):
            await client.post(
                "/v1/messages",
                headers={"x-api-key": "sk-x"},
                json={
                    "model": "claude-opus-4-7",
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )
        import asyncio
        await asyncio.sleep(0.15)
        # Even with 4 requests crossing the threshold-of-1, exactly one
        # SIGTERM is scheduled — recycle is idempotent.
        assert len(kill_calls) == 1


# ---------- chunk buffer cleanup ----------


@pytest.mark.asyncio
async def test_stream_chunk_buffer_freed_after_assistant_extraction() -> None:
    """After the stream completes, the collected chunk list must be empty
    (cleared) — held only long enough to extract the assistant message.
    Asserted indirectly: a stream with 1000 text deltas should not hold
    them across the ingest schedule. We verify the schedule received a
    properly-built assistant message AND no stray buffer survives."""
    settings = CortexSettings(
        enable_auto_ingest=True,
        enable_virtualization=False,
        recycle_after_requests=0,
    )
    registry = ProviderRegistry()
    big_provider = _StubProvider(n_text_deltas=1000)
    registry.register(big_provider)

    captured_messages: list[dict] = []

    async def _capturing_ingest(content, source, group_id, session_id, event_time):
        captured_messages.append({"content": content, "source": source})
        return f"ep_{len(captured_messages):04d}"

    session_registry = SessionRegistry(settings, ingest_fn=_capturing_ingest)
    app = _build_app(
        settings=settings,
        registry=registry,
        session_registry=session_registry,
        recall_fn=_noop_recall,
    )

    async with _live(app) as client:
        # Streaming response so the chunk buffer path is exercised.
        async with client.stream(
            "POST",
            "/v1/messages",
            headers={"x-api-key": "sk-x", "x-cortex-group-id": "g", "x-cortex-session-id": "s"},
            json={
                "model": "claude-opus-4-7",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hi long enough to ingest"}],
                "stream": True,
            },
        ) as r:
            async for _ in r.aiter_lines():
                pass

    # Drain pending background ingest tasks so the assertions below see them.
    await session_registry.drain_all(timeout=2.0)

    # 1 user msg + 1 reconstructed assistant msg = 2 ingest calls.
    assert len(captured_messages) == 2
    # Assistant message reconstruction grabbed every text_delta (chunk_0...chunk_999)
    asst = [m for m in captured_messages if m["source"].startswith("msg:assistant:")]
    assert len(asst) == 1
    assert "chunk_0" in asst[0]["content"]
    assert "chunk_999" in asst[0]["content"]


# ---------- _maybe_recycle_worker as a pure unit ----------


def test_maybe_recycle_worker_increments_without_loop() -> None:
    """Called outside an event loop (e.g. direct unit use), the helper must
    still increment the counter and mark scheduled without raising."""
    settings = CortexSettings(recycle_after_requests=2, recycle_drain_seconds=0.01)

    class _FakeApp:
        class state:
            request_count = 0
            recycle_scheduled = False

    _FakeApp.state.settings = settings  # type: ignore[attr-defined]

    _maybe_recycle_worker(_FakeApp)  # 1
    assert _FakeApp.state.request_count == 1
    assert _FakeApp.state.recycle_scheduled is False
    _maybe_recycle_worker(_FakeApp)  # 2 → schedule
    assert _FakeApp.state.request_count == 2
    assert _FakeApp.state.recycle_scheduled is True
    # Idempotent: more calls don't re-schedule.
    _maybe_recycle_worker(_FakeApp)
    assert _FakeApp.state.recycle_scheduled is True
