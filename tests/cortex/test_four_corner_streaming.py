"""The four-corner streaming matrix — MVP-4 gate test.

Combinations of (inbound API format) × (upstream provider):

    | inbound \\ provider | Anthropic  | OpenAI    |
    | -------------------- | ---------- | --------- |
    | Anthropic (/v1/messages) | A × A | A × O |
    | OpenAI (/v1/chat/completions) | O × A | O × O |

For each corner we verify:
  - The endpoint returns 200 with a stream in the EXPECTED INBOUND format.
  - The streamed events contain the assistant text we scripted upstream.
  - For tool-call corners: the tool_call id and accumulated arguments survive
    the round-trip through the canonical layer in both directions.

We don't touch the real Anthropic / OpenAI API. We register a single
FakeProvider under BOTH provider names so the router (which picks by model
prefix) sends every request to it. The fake replays a scripted canonical
chunk sequence; the proxy's job is to format-correctly serialize it for the
inbound format.
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
from cortex.config import CortexSettings
from cortex.ingest import SessionRegistry
from cortex.server import ProviderRegistry, _build_app


class FakeProvider:
    def __init__(self, name: str, scripted: list[CortexChunk]) -> None:
        self.name = name
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


def _text_only_response(text: str) -> list[CortexChunk]:
    return [
        ChunkMessageStart(message_id="msg_v", model="m", input_tokens=10),
        ChunkContentBlockStart(index=0, block=TextBlock(text="")),
        ChunkTextDelta(index=0, text=text),
        ChunkContentBlockStop(index=0),
        ChunkMessageDelta(stop_reason="end_turn", output_tokens=3),
        ChunkMessageStop(),
    ]


def _tool_call_response(tool_id: str = "call_42", tool_name: str = "search", input_json: str = '{"q":"cats"}') -> list[CortexChunk]:
    return [
        ChunkMessageStart(message_id="msg_t", model="m", input_tokens=14),
        ChunkContentBlockStart(index=0, block=TextBlock(text="")),
        ChunkTextDelta(index=0, text="thinking"),
        ChunkContentBlockStop(index=0),
        ChunkContentBlockStart(
            index=1,
            block=ToolUseBlock(tool_use_id=tool_id, tool_name=tool_name, tool_input={}),
        ),
        ChunkToolUseDelta(index=1, partial_input_json=input_json[: len(input_json) // 2]),
        ChunkToolUseDelta(index=1, partial_input_json=input_json[len(input_json) // 2 :]),
        ChunkContentBlockStop(index=1),
        ChunkMessageDelta(stop_reason="tool_use", output_tokens=11),
        ChunkMessageStop(),
    ]


async def _no_recall(query, group_id, token_budget):
    return ""


async def _no_ingest(content, source, group_id, session_id, event_time):
    return ""


def _setup_app(scripted_chunks: list[CortexChunk]):
    settings = CortexSettings(enable_auto_ingest=False, enable_virtualization=False)
    registry = ProviderRegistry()
    # Register the SAME chunk script under both provider names so model
    # routing always hits the fake.
    registry.register(FakeProvider("anthropic", scripted_chunks))
    registry.register(FakeProvider("openai", scripted_chunks))
    session_registry = SessionRegistry(settings, ingest_fn=_no_ingest)
    app = _build_app(
        settings=settings,
        registry=registry,
        session_registry=session_registry,
        recall_fn=_no_recall,
    )
    return app, registry


@asynccontextmanager
async def _live(app):
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


# ---------- SSE parsers ----------


def _parse_anthropic_sse(raw: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    current_event: str | None = None
    data_lines: list[str] = []
    for line in raw.splitlines():
        if line == "":
            if current_event is not None and data_lines:
                try:
                    events.append((current_event, json.loads("".join(data_lines))))
                except json.JSONDecodeError:
                    pass
            current_event = None
            data_lines = []
        elif line.startswith("event:"):
            current_event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
    return events


def _parse_openai_sse(raw: str) -> list[dict | str]:
    """Returns list of dicts (parsed data) and the literal string "[DONE]" sentinel."""
    out: list[dict | str] = []
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].lstrip()
        if payload == "[DONE]":
            out.append("[DONE]")
        else:
            try:
                out.append(json.loads(payload))
            except json.JSONDecodeError:
                pass
    return out


# ---------- Corner 1: Anthropic-in × Anthropic-out ----------


@pytest.mark.asyncio
async def test_anthropic_in_anthropic_out_text() -> None:
    app, _ = _setup_app(_text_only_response("ping"))
    async with _live(app) as client:
        async with client.stream(
            "POST",
            "/v1/messages",
            headers={"x-api-key": "sk-x"},
            json={
                "model": "claude-opus-4-7",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            raw = ""
            async for chunk in resp.aiter_text():
                raw += chunk

    events = _parse_anthropic_sse(raw)
    names = [n for n, _ in events]
    assert names[0] == "message_start"
    assert names[-1] == "message_stop"
    text_deltas = [d for n, d in events if n == "content_block_delta"]
    assert any(d["delta"].get("text") == "ping" for d in text_deltas)


# ---------- Corner 2: Anthropic-in × OpenAI-out (cross-provider) ----------


@pytest.mark.asyncio
async def test_anthropic_in_openai_provider_text() -> None:
    """Anthropic-format request to /v1/messages, but the model name routes to
    the OpenAI provider. The proxy translates the request to OpenAI shape,
    forwards, then re-serializes the response as Anthropic SSE."""
    app, _ = _setup_app(_text_only_response("crossed"))
    async with _live(app) as client:
        async with client.stream(
            "POST",
            "/v1/messages",
            headers={"x-api-key": "sk-x"},
            json={
                "model": "gpt-4o",  # routes to openai provider
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            raw = ""
            async for chunk in resp.aiter_text():
                raw += chunk

    events = _parse_anthropic_sse(raw)
    names = [n for n, _ in events]
    # Despite using the OpenAI provider, the client gets Anthropic-format SSE.
    assert names[0] == "message_start"
    assert names[-1] == "message_stop"
    text_deltas = [d for n, d in events if n == "content_block_delta"]
    assert any(d["delta"].get("text") == "crossed" for d in text_deltas)


# ---------- Corner 3: OpenAI-in × Anthropic-out (cross-provider) ----------


@pytest.mark.asyncio
async def test_openai_in_anthropic_provider_text() -> None:
    """OpenAI-format request to /v1/chat/completions, but the model name
    routes to the Anthropic provider."""
    app, _ = _setup_app(_text_only_response("flipped"))
    async with _live(app) as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"authorization": "Bearer sk-x"},
            json={
                "model": "claude-opus-4-7",  # routes to anthropic
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            raw = ""
            async for chunk in resp.aiter_text():
                raw += chunk

    events = _parse_openai_sse(raw)
    # Last event must be the [DONE] sentinel.
    assert events[-1] == "[DONE]"
    # Among the parsed chunks, the assistant text should appear in a content delta.
    contents = []
    for e in events:
        if isinstance(e, dict):
            d = e.get("choices", [{}])[0].get("delta", {})
            if "content" in d and d["content"]:
                contents.append(d["content"])
    assert "".join(contents) == "flipped"


# ---------- Corner 4: OpenAI-in × OpenAI-out ----------


@pytest.mark.asyncio
async def test_openai_in_openai_out_text() -> None:
    app, _ = _setup_app(_text_only_response("native"))
    async with _live(app) as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"authorization": "Bearer sk-x"},
            json={
                "model": "gpt-4o",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            raw = ""
            async for chunk in resp.aiter_text():
                raw += chunk

    events = _parse_openai_sse(raw)
    assert events[-1] == "[DONE]"
    contents = []
    finish_reasons = []
    for e in events:
        if isinstance(e, dict):
            c = e.get("choices", [{}])[0]
            d = c.get("delta", {})
            if "content" in d and d["content"]:
                contents.append(d["content"])
            if c.get("finish_reason"):
                finish_reasons.append(c["finish_reason"])
    assert "".join(contents) == "native"
    assert "stop" in finish_reasons


# ---------- Tool-call argument survival across all four corners ----------


@pytest.mark.asyncio
async def test_tool_call_args_survive_anthropic_in_anthropic_out() -> None:
    app, _ = _setup_app(_tool_call_response(input_json='{"q":"weather"}'))
    async with _live(app) as client:
        async with client.stream(
            "POST",
            "/v1/messages",
            headers={"x-api-key": "sk-x"},
            json={
                "model": "claude-opus-4-7",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "?"}],
                "stream": True,
                "tools": [{"name": "search", "input_schema": {"type": "object"}}],
            },
        ) as resp:
            raw = ""
            async for chunk in resp.aiter_text():
                raw += chunk
    events = _parse_anthropic_sse(raw)
    deltas = [
        d for n, d in events
        if n == "content_block_delta" and d["delta"].get("type") == "input_json_delta"
    ]
    accumulated = "".join(d["delta"]["partial_json"] for d in deltas)
    assert json.loads(accumulated) == {"q": "weather"}


@pytest.mark.asyncio
async def test_tool_call_args_survive_openai_in_openai_out() -> None:
    app, _ = _setup_app(_tool_call_response(input_json='{"q":"weather"}'))
    async with _live(app) as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"authorization": "Bearer sk-x"},
            json={
                "model": "gpt-4o",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "?"}],
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "search", "parameters": {"type": "object"}},
                    }
                ],
            },
        ) as resp:
            raw = ""
            async for chunk in resp.aiter_text():
                raw += chunk
    events = _parse_openai_sse(raw)
    # Collect all tool-call argument deltas (any index)
    args_chunks: list[str] = []
    tool_id: str | None = None
    for e in events:
        if not isinstance(e, dict):
            continue
        d = e.get("choices", [{}])[0].get("delta", {})
        for tc in d.get("tool_calls", []) or []:
            if "id" in tc and tc["id"]:
                tool_id = tc["id"]
            fn = tc.get("function", {})
            if "arguments" in fn and fn["arguments"]:
                args_chunks.append(fn["arguments"])
    assert tool_id == "call_42"
    assert json.loads("".join(args_chunks)) == {"q": "weather"}


@pytest.mark.asyncio
async def test_tool_call_args_survive_anthropic_in_openai_out() -> None:
    """Cross-provider with tool calls: Anthropic format in, OpenAI provider, Anthropic out."""
    app, _ = _setup_app(_tool_call_response(input_json='{"q":"weather"}'))
    async with _live(app) as client:
        async with client.stream(
            "POST",
            "/v1/messages",
            headers={"x-api-key": "sk-x"},
            json={
                "model": "gpt-4o",  # cross-provider
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "?"}],
                "stream": True,
                "tools": [{"name": "search", "input_schema": {"type": "object"}}],
            },
        ) as resp:
            raw = ""
            async for chunk in resp.aiter_text():
                raw += chunk
    events = _parse_anthropic_sse(raw)
    deltas = [
        d for n, d in events
        if n == "content_block_delta" and d["delta"].get("type") == "input_json_delta"
    ]
    accumulated = "".join(d["delta"]["partial_json"] for d in deltas)
    assert json.loads(accumulated) == {"q": "weather"}


@pytest.mark.asyncio
async def test_tool_call_args_survive_openai_in_anthropic_out() -> None:
    """Cross-provider with tool calls: OpenAI format in, Anthropic provider, OpenAI out."""
    app, _ = _setup_app(_tool_call_response(input_json='{"q":"weather"}'))
    async with _live(app) as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"authorization": "Bearer sk-x"},
            json={
                "model": "claude-opus-4-7",  # cross-provider
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "?"}],
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "search", "parameters": {"type": "object"}},
                    }
                ],
            },
        ) as resp:
            raw = ""
            async for chunk in resp.aiter_text():
                raw += chunk
    events = _parse_openai_sse(raw)
    args_chunks: list[str] = []
    tool_id: str | None = None
    for e in events:
        if not isinstance(e, dict):
            continue
        d = e.get("choices", [{}])[0].get("delta", {})
        for tc in d.get("tool_calls", []) or []:
            if "id" in tc and tc["id"]:
                tool_id = tc["id"]
            fn = tc.get("function", {})
            if "arguments" in fn and fn["arguments"]:
                args_chunks.append(fn["arguments"])
    assert tool_id == "call_42"
    assert json.loads("".join(args_chunks)) == {"q": "weather"}
