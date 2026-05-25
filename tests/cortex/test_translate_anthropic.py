"""Unit tests for canonical ↔ Anthropic translation.

These tests do NOT touch the network or any provider. They verify that the
in-memory transformation between Anthropic wire format and CortexRequest is
faithful for the cases that matter most: text, images, tool_use / tool_result,
and the streaming event shapes.
"""

from __future__ import annotations

import json

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
    CortexServerTool,
    CortexTool,
    CortexToolChoice,
    ImageBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from cortex.translate.anthropic import (
    chunk_to_anthropic_sse,
    from_anthropic_request,
    parse_anthropic_event,
    response_from_chunks,
    to_anthropic_request,
)

# ---------- request round-trips ----------


def test_simple_text_request_roundtrip() -> None:
    body = {
        "model": "claude-opus-4-7",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    req = from_anthropic_request(body)
    assert req.model == "claude-opus-4-7"
    assert req.max_tokens == 1024
    assert len(req.messages) == 1
    assert req.messages[0].role == "user"
    assert isinstance(req.messages[0].content[0], TextBlock)
    assert req.messages[0].content[0].text == "Hello"

    # Round-trip — re-serialize and ensure key fields are preserved.
    out = to_anthropic_request(req)
    assert out["model"] == "claude-opus-4-7"
    assert out["max_tokens"] == 1024
    assert out["messages"][0]["role"] == "user"
    # String-form input becomes a list of text blocks on egress — that's still
    # a valid Anthropic shape, just normalized.
    assert out["messages"][0]["content"] == [{"type": "text", "text": "Hello"}]


def test_system_as_list_collapses_to_string() -> None:
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 100,
        "system": [
            {"type": "text", "text": "You are precise."},
            {"type": "text", "text": "Cite sources."},
        ],
        "messages": [{"role": "user", "content": "hi"}],
    }
    req = from_anthropic_request(body)
    assert req.system == "You are precise.\n\nCite sources."

    out = to_anthropic_request(req)
    assert out["system"] == "You are precise.\n\nCite sources."


def test_tool_use_and_tool_result_blocks_roundtrip() -> None:
    body = {
        "model": "claude-opus-4-7",
        "max_tokens": 512,
        "messages": [
            {"role": "user", "content": "search for cats"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "ok"},
                    {
                        "type": "tool_use",
                        "id": "toolu_01ABCD",
                        "name": "search",
                        "input": {"q": "cats"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01ABCD",
                        "content": "found 42 results",
                    }
                ],
            },
        ],
        "tools": [
            {
                "name": "search",
                "description": "Web search",
                "input_schema": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            }
        ],
        "tool_choice": {"type": "auto"},
    }
    req = from_anthropic_request(body)

    assert len(req.messages) == 3
    asst = req.messages[1]
    assert isinstance(asst.content[1], ToolUseBlock)
    assert asst.content[1].tool_use_id == "toolu_01ABCD"
    assert asst.content[1].tool_input == {"q": "cats"}

    user_with_result = req.messages[2]
    assert isinstance(user_with_result.content[0], ToolResultBlock)
    assert user_with_result.content[0].tool_use_id == "toolu_01ABCD"
    assert user_with_result.content[0].content == "found 42 results"

    assert len(req.tools) == 1
    assert req.tools[0].name == "search"

    out = to_anthropic_request(req)
    assert out["tools"][0]["input_schema"]["properties"]["q"]["type"] == "string"
    assert out["messages"][1]["content"][1]["type"] == "tool_use"
    assert out["messages"][1]["content"][1]["input"] == {"q": "cats"}
    assert out["messages"][2]["content"][0]["type"] == "tool_result"
    assert out["messages"][2]["content"][0]["tool_use_id"] == "toolu_01ABCD"


def test_image_block_base64_roundtrip() -> None:
    body = {
        "model": "claude-opus-4-7",
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBORw0KGgo=",
                        },
                    },
                    {"type": "text", "text": "what is this?"},
                ],
            }
        ],
    }
    req = from_anthropic_request(body)
    img = req.messages[0].content[0]
    assert isinstance(img, ImageBlock)
    assert img.media_type == "image/png"
    assert img.data_b64 == "iVBORw0KGgo="

    out = to_anthropic_request(req)
    assert out["messages"][0]["content"][0]["source"]["type"] == "base64"
    assert out["messages"][0]["content"][0]["source"]["data"] == "iVBORw0KGgo="


def test_tool_choice_variants() -> None:
    base = {
        "model": "claude-opus-4-7",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "t", "input_schema": {"type": "object"}}],
    }

    for raw, expected_mode, expected_name in [
        ({"type": "auto"}, "auto", None),
        ({"type": "any"}, "any", None),
        ({"type": "none"}, "none", None),
        ({"type": "tool", "name": "t"}, "tool", "t"),
    ]:
        body = {**base, "tool_choice": raw}
        req = from_anthropic_request(body)
        assert req.tool_choice.mode == expected_mode
        assert req.tool_choice.name == expected_name


def test_web_search_server_tool_roundtrip() -> None:
    """Anthropic server tools (web_search, computer, bash, text_editor) carry
    `type` and have NO `input_schema`. They must round-trip opaquely; the
    legacy `t["input_schema"]` lookup KeyError'd the whole request, which
    Claude Code surfaced as 4 retries then a dropped WebSearch."""
    body = {
        "model": "claude-opus-4-7",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "search for cats"}],
        "tools": [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
            {"type": "bash_20250124", "name": "bash"},
            {
                "type": "text_editor_20250124",
                "name": "str_replace_editor",
            },
            {
                "type": "computer_20250124",
                "name": "computer",
                "display_width_px": 1920,
                "display_height_px": 1080,
                "display_number": 1,
            },
            {
                "name": "custom_lookup",
                "description": "Look up a user-defined fact",
                "input_schema": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            },
        ],
    }

    req = from_anthropic_request(body)
    assert len(req.tools) == 5
    assert isinstance(req.tools[0], CortexServerTool)
    assert req.tools[0].server_type == "web_search_20250305"
    assert req.tools[0].name == "web_search"
    assert req.tools[0].extras == {"max_uses": 5}
    assert isinstance(req.tools[1], CortexServerTool)
    assert req.tools[1].server_type == "bash_20250124"
    assert isinstance(req.tools[2], CortexServerTool)
    assert isinstance(req.tools[3], CortexServerTool)
    assert req.tools[3].extras["display_width_px"] == 1920
    # User function tools must still parse normally.
    assert isinstance(req.tools[4], CortexTool)
    assert req.tools[4].name == "custom_lookup"
    assert req.tools[4].json_schema["properties"]["q"]["type"] == "string"

    out = to_anthropic_request(req)
    out_tools = out["tools"]
    assert out_tools[0] == {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 5,
    }
    assert out_tools[1] == {"type": "bash_20250124", "name": "bash"}
    assert out_tools[2] == {
        "type": "text_editor_20250124",
        "name": "str_replace_editor",
    }
    assert out_tools[3] == {
        "type": "computer_20250124",
        "name": "computer",
        "display_width_px": 1920,
        "display_height_px": 1080,
        "display_number": 1,
    }
    # User function tool comes out in the function shape, not the server shape.
    assert out_tools[4]["name"] == "custom_lookup"
    assert out_tools[4]["input_schema"]["properties"]["q"]["type"] == "string"
    assert "type" not in out_tools[4]


def test_unknown_role_rejected() -> None:
    body = {
        "model": "claude-opus-4-7",
        "max_tokens": 10,
        "messages": [{"role": "tool", "content": "x"}],
    }
    with pytest.raises(ValueError):
        from_anthropic_request(body)


# ---------- streaming event parsing ----------


def test_parse_message_start() -> None:
    chunk = parse_anthropic_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_01ABCDEF",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-opus-4-7",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 25, "output_tokens": 1},
            },
        },
    )
    assert isinstance(chunk, ChunkMessageStart)
    assert chunk.message_id == "msg_01ABCDEF"
    assert chunk.model == "claude-opus-4-7"
    assert chunk.input_tokens == 25


def test_parse_text_delta() -> None:
    chunk = parse_anthropic_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        },
    )
    assert isinstance(chunk, ChunkTextDelta)
    assert chunk.index == 0
    assert chunk.text == "Hello"


def test_parse_tool_use_input_delta() -> None:
    chunk = parse_anthropic_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"q":"ca'},
        },
    )
    assert isinstance(chunk, ChunkToolUseDelta)
    assert chunk.index == 1
    assert chunk.partial_input_json == '{"q":"ca'


def test_parse_ping_and_message_stop() -> None:
    p = parse_anthropic_event("ping", {"type": "ping"})
    assert isinstance(p, ChunkPing)
    s = parse_anthropic_event("message_stop", {"type": "message_stop"})
    assert isinstance(s, ChunkMessageStop)


def test_parse_error_event() -> None:
    chunk = parse_anthropic_event(
        "error",
        {
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Try again later"},
        },
    )
    assert isinstance(chunk, ChunkError)
    assert chunk.error_type == "overloaded_error"
    assert chunk.message == "Try again later"


# ---------- streaming event serialization ----------


def test_serialize_text_delta() -> None:
    chunk = ChunkTextDelta(index=0, text="Hi")
    name, data = chunk_to_anthropic_sse(chunk)
    assert name == "content_block_delta"
    assert data["delta"] == {"type": "text_delta", "text": "Hi"}


def test_serialize_message_start() -> None:
    chunk = ChunkMessageStart(message_id="msg_x", model="claude-opus-4-7", input_tokens=10)
    name, data = chunk_to_anthropic_sse(chunk)
    assert name == "message_start"
    assert data["message"]["id"] == "msg_x"
    assert data["message"]["model"] == "claude-opus-4-7"
    assert data["message"]["usage"]["input_tokens"] == 10


# ---------- aggregation: chunks → non-streaming response ----------


def test_response_from_chunks_text_only() -> None:
    chunks = [
        ChunkMessageStart(message_id="msg_xy", model="claude-opus-4-7", input_tokens=20),
        ChunkContentBlockStart(index=0, block=TextBlock(text="")),
        ChunkTextDelta(index=0, text="Hello"),
        ChunkTextDelta(index=0, text=" world"),
        ChunkContentBlockStop(index=0),
        ChunkMessageDelta(stop_reason="end_turn", output_tokens=5),
        ChunkMessageStop(),
    ]
    resp = response_from_chunks(chunks, model="claude-opus-4-7")
    assert resp["id"] == "msg_xy"
    assert resp["content"] == [{"type": "text", "text": "Hello world"}]
    assert resp["stop_reason"] == "end_turn"
    assert resp["usage"] == {"input_tokens": 20, "output_tokens": 5}


def test_response_from_chunks_with_tool_use() -> None:
    chunks = [
        ChunkMessageStart(message_id="msg_ab", model="claude-opus-4-7", input_tokens=10),
        ChunkContentBlockStart(index=0, block=TextBlock(text="")),
        ChunkTextDelta(index=0, text="Using a tool."),
        ChunkContentBlockStop(index=0),
        ChunkContentBlockStart(
            index=1,
            block=ToolUseBlock(tool_use_id="toolu_X", tool_name="search", tool_input={}),
        ),
        ChunkToolUseDelta(index=1, partial_input_json='{"q":"'),
        ChunkToolUseDelta(index=1, partial_input_json='dogs"}'),
        ChunkContentBlockStop(index=1),
        ChunkMessageDelta(stop_reason="tool_use", output_tokens=8),
        ChunkMessageStop(),
    ]
    resp = response_from_chunks(chunks, model="claude-opus-4-7")
    assert len(resp["content"]) == 2
    assert resp["content"][0] == {"type": "text", "text": "Using a tool."}
    assert resp["content"][1]["type"] == "tool_use"
    assert resp["content"][1]["id"] == "toolu_X"
    assert resp["content"][1]["name"] == "search"
    assert resp["content"][1]["input"] == {"q": "dogs"}
    assert resp["stop_reason"] == "tool_use"


# ---------- canonical round-trip on chunks ----------


def test_chunk_roundtrip_text_delta() -> None:
    """Serialize → parse → equal."""
    original = ChunkTextDelta(index=2, text="abc")
    name, data = chunk_to_anthropic_sse(original)
    parsed = parse_anthropic_event(name, data)
    assert isinstance(parsed, ChunkTextDelta)
    assert parsed.index == 2
    assert parsed.text == "abc"


def test_chunk_roundtrip_tool_use_open_close() -> None:
    block_start = ChunkContentBlockStart(
        index=1, block=ToolUseBlock(tool_use_id="t1", tool_name="x", tool_input={})
    )
    name, data = chunk_to_anthropic_sse(block_start)
    parsed = parse_anthropic_event(name, data)
    assert isinstance(parsed, ChunkContentBlockStart)
    assert isinstance(parsed.block, ToolUseBlock)
    assert parsed.block.tool_use_id == "t1"
    assert parsed.block.tool_name == "x"
