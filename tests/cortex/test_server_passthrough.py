"""End-to-end test for the MVP-1 passthrough proxy.

We don't hit the real Anthropic API. Instead, we inject a `FakeProvider` into
the registry that returns scripted CortexChunks. This exercises the full
server pipeline: request parsing → translation → provider dispatch → chunk
serialization → SSE output.

Two flows tested:
  - Non-streaming (`stream: false`): chunks are aggregated server-side into
    an Anthropic Messages response body.
  - Streaming (`stream: true`): chunks are re-emitted as Anthropic SSE
    events; we parse them on the client side and verify ordering.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest

from cortex.canonical import (
    ChunkContentBlockStart,
    ChunkContentBlockStop,
    ChunkError,
    ChunkMessageDelta,
    ChunkMessageStart,
    ChunkMessageStop,
    ChunkTextDelta,
    ChunkToolUseDelta,
    CortexChunk,
    CortexRequest,
    TextBlock,
    ToolUseBlock,
)
from cortex.providers.base import Provider
from cortex.server import ProviderRegistry, _build_app


class FakeProvider:
    """Records the last request and replays a scripted chunk sequence."""

    name = "anthropic"

    def __init__(self, scripted: list[CortexChunk]) -> None:
        self._scripted = scripted
        self.last_request: CortexRequest | None = None
        self.last_api_key: str | None = None
        self.last_extra_headers: dict[str, str] | None = None

    async def stream(
        self,
        req: CortexRequest,
        api_key: str,
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncIterator[CortexChunk]:
        self.last_request = req
        self.last_api_key = api_key
        self.last_extra_headers = extra_headers
        for c in self._scripted:
            yield c

    async def aclose(self) -> None:
        pass


def _app_with(provider: Provider):
    registry = ProviderRegistry()
    registry.register(provider)
    return _build_app(registry=registry)


@asynccontextmanager
async def _live_client(app):
    """ASGI client with lifespan triggered (httpx.ASGITransport skips lifespan by default)."""
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


def _text_chunks(text: str) -> list[CortexChunk]:
    return [
        ChunkMessageStart(message_id="msg_test_001", model="claude-opus-4-7", input_tokens=12),
        ChunkContentBlockStart(index=0, block=TextBlock(text="")),
        ChunkTextDelta(index=0, text=text),
        ChunkContentBlockStop(index=0),
        ChunkMessageDelta(stop_reason="end_turn", output_tokens=4),
        ChunkMessageStop(),
    ]


# ---------- Non-streaming flow ----------


@pytest.mark.asyncio
async def test_non_streaming_text_response() -> None:
    fake = FakeProvider(_text_chunks("Hello back"))
    app = _app_with(fake)

    async with _live_client(app) as client:
        h = await client.get("/health")
        assert h.status_code == 200
        resp = await client.post(
            "/v1/messages",
            headers={"x-api-key": "sk-ant-test"},
            json={
                "model": "claude-opus-4-7",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "ping"}],
                "stream": False,
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "claude-opus-4-7"
    assert body["content"] == [{"type": "text", "text": "Hello back"}]
    assert body["stop_reason"] == "end_turn"
    assert body["usage"]["output_tokens"] == 4

    # The provider saw the right thing.
    assert fake.last_api_key == "sk-ant-test"
    assert fake.last_request is not None
    assert fake.last_request.model == "claude-opus-4-7"
    # Upstream is always streamed even when client wanted non-streaming.
    assert fake.last_request.stream is True


@pytest.mark.asyncio
async def test_missing_api_key_returns_401() -> None:
    fake = FakeProvider([])
    app = _app_with(fake)

    async with _live_client(app) as client:
        resp = await client.post(
            "/v1/messages",
            json={
                "model": "claude-opus-4-7",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

    assert resp.status_code == 401
    assert fake.last_request is None  # we never invoked the provider


@pytest.mark.asyncio
async def test_upstream_error_surfaces_as_502() -> None:
    fake = FakeProvider([ChunkError(error_type="overloaded_error", message="busy")])
    app = _app_with(fake)

    async with _live_client(app) as client:
        resp = await client.post(
            "/v1/messages",
            headers={"x-api-key": "sk-ant-test"},
            json={
                "model": "claude-opus-4-7",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

    assert resp.status_code == 502
    body = resp.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "overloaded_error"


@pytest.mark.asyncio
async def test_invalid_json_body_returns_400() -> None:
    fake = FakeProvider([])
    app = _app_with(fake)

    async with _live_client(app) as client:
        resp = await client.post(
            "/v1/messages",
            headers={"x-api-key": "sk-ant-test", "content-type": "application/json"},
            content=b"{not json",
        )

    assert resp.status_code == 400


# ---------- Streaming flow ----------


def _parse_sse(raw: str) -> list[tuple[str, dict]]:
    """Parse a raw SSE bytestream into [(event_name, data_dict)]."""
    events: list[tuple[str, dict]] = []
    current_event: str | None = None
    data_lines: list[str] = []
    for line in raw.splitlines():
        if line == "":
            if current_event is not None and data_lines:
                try:
                    payload = json.loads("".join(data_lines))
                except json.JSONDecodeError:
                    payload = {}
                events.append((current_event, payload))
            current_event = None
            data_lines = []
            continue
        if line.startswith("event:"):
            current_event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
    return events


@pytest.mark.asyncio
async def test_streaming_text_response() -> None:
    fake = FakeProvider(_text_chunks("streamed!"))
    app = _app_with(fake)

    async with _live_client(app) as client:
        async with client.stream(
            "POST",
            "/v1/messages",
            headers={"x-api-key": "sk-ant-test"},
            json={
                "model": "claude-opus-4-7",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "go"}],
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            raw = ""
            async for chunk in resp.aiter_text():
                raw += chunk

    events = _parse_sse(raw)
    names = [name for name, _ in events]
    assert names[0] == "message_start"
    assert "content_block_start" in names
    assert "content_block_delta" in names
    assert "content_block_stop" in names
    assert "message_delta" in names
    assert names[-1] == "message_stop"

    text_deltas = [d for n, d in events if n == "content_block_delta"]
    assert any(d.get("delta", {}).get("text") == "streamed!" for d in text_deltas)


@pytest.mark.asyncio
async def test_streaming_with_tool_use() -> None:
    chunks: list[CortexChunk] = [
        ChunkMessageStart(message_id="msg_t", model="claude-opus-4-7", input_tokens=15),
        ChunkContentBlockStart(
            index=0,
            block=ToolUseBlock(tool_use_id="toolu_001", tool_name="search", tool_input={}),
        ),
        ChunkToolUseDelta(index=0, partial_input_json='{"q":'),
        ChunkToolUseDelta(index=0, partial_input_json='"weather"}'),
        ChunkContentBlockStop(index=0),
        ChunkMessageDelta(stop_reason="tool_use", output_tokens=3),
        ChunkMessageStop(),
    ]
    fake = FakeProvider(chunks)
    app = _app_with(fake)

    async with _live_client(app) as client:
        async with client.stream(
            "POST",
            "/v1/messages",
            headers={"x-api-key": "sk-ant-test"},
            json={
                "model": "claude-opus-4-7",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "weather?"}],
                "stream": True,
                "tools": [
                    {
                        "name": "search",
                        "description": "search",
                        "input_schema": {
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                        },
                    }
                ],
            },
        ) as resp:
            assert resp.status_code == 200
            raw = ""
            async for chunk in resp.aiter_text():
                raw += chunk

    events = _parse_sse(raw)
    # Find the content_block_start for tool_use
    starts = [d for n, d in events if n == "content_block_start"]
    assert any(s.get("content_block", {}).get("type") == "tool_use" for s in starts)
    # Verify the partial_json deltas survived
    json_deltas = [
        d for n, d in events
        if n == "content_block_delta" and d.get("delta", {}).get("type") == "input_json_delta"
    ]
    assert len(json_deltas) == 2
    accumulated = "".join(d["delta"]["partial_json"] for d in json_deltas)
    assert json.loads(accumulated) == {"q": "weather"}


@pytest.mark.asyncio
async def test_cortex_session_headers_propagate_to_request() -> None:
    fake = FakeProvider(_text_chunks("ok"))
    app = _app_with(fake)

    async with _live_client(app) as client:
        await client.post(
            "/v1/messages",
            headers={
                "x-api-key": "sk-ant-test",
                "x-cortex-group-id": "proj-alpha",
                "x-cortex-session-id": "sess-2026-05-21",
                "x-cortex-time-anchor": "2026-05-01T00:00:00Z",
                "x-cortex-disable-virtualize": "1",
            },
            json={
                "model": "claude-opus-4-7",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )

    assert fake.last_request is not None
    assert fake.last_request.cortex_group_id == "proj-alpha"
    assert fake.last_request.cortex_session_id == "sess-2026-05-21"
    assert fake.last_request.cortex_time_anchor_iso == "2026-05-01T00:00:00Z"
    assert fake.last_request.cortex_disable_virtualize is True
    assert fake.last_request.cortex_disable_ingest is False
