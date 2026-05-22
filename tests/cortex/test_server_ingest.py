"""Server-level test for the MVP-2 auto-ingest behavior.

We inject both a FakeProvider (no real Anthropic) and a SessionRegistry with
a stub `ingest_fn` (no real Neo4j/Qdrant). The proxy should:
  - Schedule every new inbound message for ingest.
  - After the upstream stream completes, schedule the assistant turn too.
  - Skip duplicates across requests (content-hash idempotency).
  - Honor the X-Cortex-Disable-Ingest header.
  - Derive a stable group_id from the API key when no header is provided.
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
from cortex.ingest import SessionRegistry, derive_group_id
from cortex.server import ProviderRegistry, _build_app


class FakeProvider:
    name = "anthropic"

    def __init__(self, scripted: list[CortexChunk]) -> None:
        self._scripted = scripted

    async def stream(
        self,
        req: CortexRequest,
        api_key: str,
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncIterator[CortexChunk]:
        for c in self._scripted:
            yield c

    async def aclose(self) -> None:
        pass


def _text_response(text: str) -> list[CortexChunk]:
    return [
        ChunkMessageStart(message_id="msg_resp", model="claude-opus-4-7", input_tokens=12),
        ChunkContentBlockStart(index=0, block=TextBlock(text="")),
        ChunkTextDelta(index=0, text=text),
        ChunkContentBlockStop(index=0),
        ChunkMessageDelta(stop_reason="end_turn", output_tokens=4),
        ChunkMessageStop(),
    ]


def _setup_app(response_text: str = "ack"):
    calls: list[dict] = []

    async def recorder(content, source, group_id, session_id, event_time):
        calls.append(
            {
                "content": content,
                "source": source,
                "group_id": group_id,
                "session_id": session_id,
            }
        )
        return f"ep_{len(calls):04d}"

    settings = CortexSettings()
    registry = ProviderRegistry()
    registry.register(FakeProvider(_text_response(response_text)))
    session_registry = SessionRegistry(settings, ingest_fn=recorder)
    app = _build_app(settings=settings, registry=registry, session_registry=session_registry)
    return app, session_registry, calls


@asynccontextmanager
async def _live_client(app):
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


# ---------- Ingest happens on user messages ----------


@pytest.mark.asyncio
async def test_user_message_gets_ingested() -> None:
    app, session_registry, calls = _setup_app("ack")
    async with _live_client(app) as client:
        await client.post(
            "/v1/messages",
            headers={
                "x-api-key": "sk-ant-aaa",
                "x-cortex-group-id": "proj-X",
                "x-cortex-session-id": "sess-1",
            },
            json={
                "model": "claude-opus-4-7",
                "max_tokens": 32,
                "messages": [
                    {
                        "role": "user",
                        "content": "this is a long enough user message to clear the min-chars filter",
                    }
                ],
                "stream": False,
            },
        )
        # Drain any inflight background tasks scheduled by the request.
        await session_registry.drain_all(timeout=2.0)

    user_calls = [c for c in calls if c["source"].startswith("msg:user:")]
    assert len(user_calls) == 1
    assert "long enough user message" in user_calls[0]["content"]
    assert user_calls[0]["group_id"] == "proj-X"
    assert user_calls[0]["session_id"] == "sess-1"


# ---------- Assistant turn also ingested after stream completes ----------


@pytest.mark.asyncio
async def test_assistant_response_also_ingested() -> None:
    app, session_registry, calls = _setup_app("a sufficiently long assistant response yes indeed")
    async with _live_client(app) as client:
        await client.post(
            "/v1/messages",
            headers={
                "x-api-key": "sk-ant-aaa",
                "x-cortex-group-id": "proj-X",
                "x-cortex-session-id": "sess-2",
            },
            json={
                "model": "claude-opus-4-7",
                "max_tokens": 32,
                "messages": [
                    {
                        "role": "user",
                        "content": "long enough user prompt for ingestion threshold",
                    }
                ],
                "stream": False,
            },
        )
        await session_registry.drain_all(timeout=2.0)

    roles_ingested = sorted(c["source"].split(":")[1] for c in calls)
    assert roles_ingested == ["assistant", "user"]
    assistant_call = next(c for c in calls if c["source"].startswith("msg:assistant:"))
    assert "sufficiently long assistant response" in assistant_call["content"]


# ---------- Idempotency across requests ----------


@pytest.mark.asyncio
async def test_repeated_identical_request_does_not_re_ingest() -> None:
    app, session_registry, calls = _setup_app("ack ack ack ack ack ack")
    body = {
        "model": "claude-opus-4-7",
        "max_tokens": 32,
        "messages": [
            {
                "role": "user",
                "content": "this exact text appears multiple times across requests",
            }
        ],
        "stream": False,
    }
    headers = {
        "x-api-key": "sk-ant-aaa",
        "x-cortex-group-id": "proj-X",
        "x-cortex-session-id": "sess-dup",
    }
    async with _live_client(app) as client:
        for _ in range(3):
            await client.post("/v1/messages", headers=headers, json=body)
        await session_registry.drain_all(timeout=2.0)

    user_calls = [c for c in calls if c["source"].startswith("msg:user:")]
    # The user content is identical across all three requests → ingested once.
    assert len(user_calls) == 1


# ---------- Header-level opt-out ----------


@pytest.mark.asyncio
async def test_disable_ingest_header_skips_session_creation() -> None:
    app, session_registry, calls = _setup_app("ack")
    async with _live_client(app) as client:
        await client.post(
            "/v1/messages",
            headers={
                "x-api-key": "sk-ant-aaa",
                "x-cortex-group-id": "proj-X",
                "x-cortex-session-id": "sess-N",
                "x-cortex-disable-ingest": "true",
            },
            json={
                "model": "claude-opus-4-7",
                "max_tokens": 32,
                "messages": [
                    {"role": "user", "content": "long enough message that we do not want stored"}
                ],
                "stream": False,
            },
        )
        await session_registry.drain_all(timeout=2.0)

    assert calls == []


# ---------- Group derivation from API key ----------


@pytest.mark.asyncio
async def test_group_id_derived_from_api_key_when_no_header() -> None:
    app, session_registry, calls = _setup_app("ack")
    async with _live_client(app) as client:
        await client.post(
            "/v1/messages",
            headers={"x-api-key": "sk-ant-deadbeef-xyz"},
            json={
                "model": "claude-opus-4-7",
                "max_tokens": 32,
                "messages": [
                    {"role": "user", "content": "another long enough message for ingestion to fire"}
                ],
                "stream": False,
            },
        )
        await session_registry.drain_all(timeout=2.0)

    expected_group = derive_group_id("sk-ant-deadbeef-xyz")
    user_calls = [c for c in calls if c["source"].startswith("msg:user:")]
    assert all(c["group_id"] == expected_group for c in user_calls)
    assert all(c["session_id"] == "default" for c in user_calls)


# ---------- Streaming path also ingests assistant turn ----------


@pytest.mark.asyncio
async def test_streaming_response_triggers_assistant_ingest() -> None:
    app, session_registry, calls = _setup_app("a streamed assistant response with enough length")
    async with _live_client(app) as client:
        async with client.stream(
            "POST",
            "/v1/messages",
            headers={
                "x-api-key": "sk-ant-aaa",
                "x-cortex-group-id": "proj-X",
                "x-cortex-session-id": "sess-stream",
            },
            json={
                "model": "claude-opus-4-7",
                "max_tokens": 32,
                "messages": [
                    {"role": "user", "content": "a long enough streaming user prompt to ingest"}
                ],
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            async for _ in resp.aiter_text():
                pass
        await session_registry.drain_all(timeout=2.0)

    roles = sorted({c["source"].split(":")[1] for c in calls})
    assert roles == ["assistant", "user"]
