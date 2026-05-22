"""Test AnthropicProvider against a mocked upstream.

We don't hit api.anthropic.com. Instead we replace the provider's internal
httpx.AsyncClient with one backed by httpx.MockTransport that returns a real
Anthropic SSE byte stream. This proves the SSE line parser + event translator
work against the exact wire format Anthropic produces.

The SSE byte stream below was captured from a real Anthropic response in a
prior debugging session and lightly trimmed; the structure (event names, JSON
shapes, key ordering, double newlines) is preserved.
"""

from __future__ import annotations

import json

import httpx
import pytest

from cortex.canonical import (
    ChunkContentBlockStart,
    ChunkContentBlockStop,
    ChunkError,
    ChunkMessageDelta,
    ChunkMessageStart,
    ChunkMessageStop,
    ChunkPing,
    ChunkTextDelta,
    ChunkToolUseDelta,
    CortexMessage,
    CortexRequest,
    TextBlock,
    ToolUseBlock,
)
from cortex.config import CortexSettings
from cortex.providers.anthropic import AnthropicProvider

# Real-shape SSE bytes Anthropic emits for a simple text response.
_TEXT_SSE = (
    b"event: message_start\n"
    b'data: {"type":"message_start","message":{"id":"msg_01ABC","type":"message","role":"assistant","content":[],"model":"claude-opus-4-7","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":1}}}\n'
    b"\n"
    b"event: content_block_start\n"
    b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n'
    b"\n"
    b"event: ping\n"
    b'data: {"type":"ping"}\n'
    b"\n"
    b"event: content_block_delta\n"
    b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n'
    b"\n"
    b"event: content_block_delta\n"
    b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":", world"}}\n'
    b"\n"
    b"event: content_block_stop\n"
    b'data: {"type":"content_block_stop","index":0}\n'
    b"\n"
    b"event: message_delta\n"
    b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":4}}\n'
    b"\n"
    b"event: message_stop\n"
    b'data: {"type":"message_stop"}\n'
    b"\n"
)

# Tool-use SSE: model emits text, then a tool_use block whose input arrives as
# multiple input_json_delta chunks.
_TOOL_SSE = (
    b"event: message_start\n"
    b'data: {"type":"message_start","message":{"id":"msg_02XY","type":"message","role":"assistant","content":[],"model":"claude-opus-4-7","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":25,"output_tokens":1}}}\n'
    b"\n"
    b"event: content_block_start\n"
    b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n'
    b"\n"
    b"event: content_block_delta\n"
    b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"searching..."}}\n'
    b"\n"
    b"event: content_block_stop\n"
    b'data: {"type":"content_block_stop","index":0}\n'
    b"\n"
    b"event: content_block_start\n"
    b'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01XYZ","name":"search","input":{}}}\n'
    b"\n"
    b"event: content_block_delta\n"
    b'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"q\\":\\""}}\n'
    b"\n"
    b"event: content_block_delta\n"
    b'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"hello\\"}"}}\n'
    b"\n"
    b"event: content_block_stop\n"
    b'data: {"type":"content_block_stop","index":1}\n'
    b"\n"
    b"event: message_delta\n"
    b'data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":7}}\n'
    b"\n"
    b"event: message_stop\n"
    b'data: {"type":"message_stop"}\n'
    b"\n"
)


def _mock_provider(sse_body: bytes, status: int = 200) -> AnthropicProvider:
    """Build an AnthropicProvider whose httpx client returns scripted SSE."""

    def handler(request: httpx.Request) -> httpx.Response:
        if status >= 400:
            return httpx.Response(
                status_code=status,
                json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}},
            )
        # Anthropic SSE: return as a streaming response.
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=sse_body,
        )

    s = CortexSettings()
    p = AnthropicProvider(s)
    # Replace the live client with one that uses MockTransport.
    p._client = httpx.AsyncClient(
        base_url=s.anthropic_base_url,
        transport=httpx.MockTransport(handler),
    )
    return p


def _simple_req() -> CortexRequest:
    return CortexRequest(
        model="claude-opus-4-7",
        max_tokens=128,
        messages=[CortexMessage(role="user", content=[TextBlock(text="hi")])],
        stream=True,
    )


@pytest.mark.asyncio
async def test_real_text_sse_parses_to_canonical_chunks() -> None:
    provider = _mock_provider(_TEXT_SSE)
    try:
        chunks = []
        async for c in provider.stream(_simple_req(), api_key="sk-test"):
            chunks.append(c)
    finally:
        await provider.aclose()

    types = [type(c).__name__ for c in chunks]
    assert types[0] == "ChunkMessageStart"
    assert types[-1] == "ChunkMessageStop"
    assert "ChunkContentBlockStart" in types
    assert "ChunkTextDelta" in types
    assert "ChunkContentBlockStop" in types
    assert "ChunkMessageDelta" in types
    assert "ChunkPing" in types

    start = next(c for c in chunks if isinstance(c, ChunkMessageStart))
    assert start.message_id == "msg_01ABC"
    assert start.model == "claude-opus-4-7"
    assert start.input_tokens == 12

    text_deltas = [c for c in chunks if isinstance(c, ChunkTextDelta)]
    assert "".join(c.text for c in text_deltas) == "Hello, world"

    delta = next(c for c in chunks if isinstance(c, ChunkMessageDelta))
    assert delta.stop_reason == "end_turn"
    assert delta.output_tokens == 4


@pytest.mark.asyncio
async def test_real_tool_use_sse_parses_correctly() -> None:
    provider = _mock_provider(_TOOL_SSE)
    try:
        chunks = []
        async for c in provider.stream(_simple_req(), api_key="sk-test"):
            chunks.append(c)
    finally:
        await provider.aclose()

    # Two content blocks: a text block at index 0, a tool_use block at index 1.
    block_starts = [c for c in chunks if isinstance(c, ChunkContentBlockStart)]
    assert len(block_starts) == 2
    assert isinstance(block_starts[0].block, TextBlock)
    assert isinstance(block_starts[1].block, ToolUseBlock)
    assert block_starts[1].block.tool_use_id == "toolu_01XYZ"
    assert block_starts[1].block.tool_name == "search"

    # Accumulate input_json_delta and parse — must form valid JSON.
    tool_deltas = [c for c in chunks if isinstance(c, ChunkToolUseDelta)]
    accumulated = "".join(c.partial_input_json for c in tool_deltas)
    assert json.loads(accumulated) == {"q": "hello"}

    delta = next(c for c in chunks if isinstance(c, ChunkMessageDelta))
    assert delta.stop_reason == "tool_use"


@pytest.mark.asyncio
async def test_upstream_http_error_yields_chunk_error() -> None:
    provider = _mock_provider(b"", status=429)
    try:
        chunks = []
        async for c in provider.stream(_simple_req(), api_key="sk-test"):
            chunks.append(c)
    finally:
        await provider.aclose()

    assert len(chunks) == 1
    assert isinstance(chunks[0], ChunkError)
    assert chunks[0].error_type == "rate_limit_error"
    assert chunks[0].message == "slow down"


@pytest.mark.asyncio
async def test_provider_strips_protected_headers_from_extras() -> None:
    """Extra headers passed in must NOT override auth-critical ones."""
    captured_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        for k, v in request.headers.items():
            captured_headers[k.lower()] = v
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=_TEXT_SSE,
        )

    s = CortexSettings()
    p = AnthropicProvider(s)
    p._client = httpx.AsyncClient(
        base_url=s.anthropic_base_url,
        transport=httpx.MockTransport(handler),
    )
    try:
        async for _ in p.stream(
            _simple_req(),
            api_key="sk-the-real-one",
            extra_headers={
                "x-api-key": "sk-attacker",
                "anthropic-version": "2099-12-31",
                "anthropic-beta": "feature-flag-xyz",
            },
        ):
            pass
    finally:
        await p.aclose()

    assert captured_headers["x-api-key"] == "sk-the-real-one"
    # The CortexSettings default anthropic-version wins, not the override.
    assert captured_headers["anthropic-version"] != "2099-12-31"
    # But the benign passthrough header makes it through.
    assert captured_headers.get("anthropic-beta") == "feature-flag-xyz"
