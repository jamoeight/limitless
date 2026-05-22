"""Unit tests for canonical ↔ OpenAI Chat Completions translation."""

from __future__ import annotations

import json

import pytest

from cortex.canonical import (
    ChunkContentBlockStart,
    ChunkContentBlockStop,
    ChunkMessageDelta,
    ChunkMessageStart,
    ChunkMessageStop,
    ChunkTextDelta,
    ChunkToolUseDelta,
    CortexMessage,
    CortexRequest,
    CortexTool,
    CortexToolChoice,
    ImageBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from cortex.translate.openai import (
    chunk_to_openai_sse,
    from_openai_request,
    new_openai_egress_state,
    new_openai_stream_state,
    openai_response_from_chunks,
    parse_openai_stream_chunk,
    to_openai_request,
)


# ---------- request: OpenAI → Canonical ----------


def test_simple_text_openai_request() -> None:
    body = {
        "model": "gpt-4o",
        "max_tokens": 100,
        "messages": [
            {"role": "system", "content": "You are precise."},
            {"role": "user", "content": "Hi"},
        ],
    }
    req = from_openai_request(body)
    assert req.system == "You are precise."
    assert len(req.messages) == 1
    assert req.messages[0].role == "user"
    assert isinstance(req.messages[0].content[0], TextBlock)
    assert req.messages[0].content[0].text == "Hi"


def test_openai_tool_call_and_tool_result_roundtrip() -> None:
    body = {
        "model": "gpt-4o",
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "search for cats"},
            {
                "role": "assistant",
                "content": "let me search",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q":"cats"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "found 3"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "search the web",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }
        ],
    }
    req = from_openai_request(body)
    # 3 canonical messages: user, assistant(text + tool_use), user(tool_result)
    assert len(req.messages) == 3
    asst = req.messages[1]
    assert asst.role == "assistant"
    assert isinstance(asst.content[0], TextBlock)
    assert isinstance(asst.content[1], ToolUseBlock)
    assert asst.content[1].tool_use_id == "call_abc"
    assert asst.content[1].tool_input == {"q": "cats"}

    user_tr = req.messages[2]
    assert user_tr.role == "user"
    assert isinstance(user_tr.content[0], ToolResultBlock)
    assert user_tr.content[0].tool_use_id == "call_abc"

    out = to_openai_request(req)
    # System came back as a separate message at index 0... well, we had no system.
    # Assistant entry has both content text and tool_calls.
    asst_out = next(m for m in out["messages"] if m["role"] == "assistant")
    assert asst_out["content"] == "let me search"
    assert asst_out["tool_calls"][0]["id"] == "call_abc"
    assert asst_out["tool_calls"][0]["function"]["arguments"] == '{"q": "cats"}'
    # Tool result went back to role=tool entry
    tool_out = next(m for m in out["messages"] if m["role"] == "tool")
    assert tool_out["tool_call_id"] == "call_abc"
    assert tool_out["content"] == "found 3"


def test_openai_multimodal_image_url_parses_base64_data_url() -> None:
    data_url = "data:image/png;base64,iVBORw0KGgo="
    body = {
        "model": "gpt-4o",
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what?"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }
    req = from_openai_request(body)
    blocks = req.messages[0].content
    assert isinstance(blocks[1], ImageBlock)
    assert blocks[1].media_type == "image/png"
    assert blocks[1].data_b64 == "iVBORw0KGgo="


def test_openai_tool_choice_variants() -> None:
    base = {
        "model": "gpt-4o",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"type": "function", "function": {"name": "t", "parameters": {"type": "object"}}}
        ],
    }
    assert from_openai_request({**base, "tool_choice": "auto"}).tool_choice.mode == "auto"
    assert from_openai_request({**base, "tool_choice": "none"}).tool_choice.mode == "none"
    assert from_openai_request({**base, "tool_choice": "required"}).tool_choice.mode == "any"
    forced = from_openai_request(
        {**base, "tool_choice": {"type": "function", "function": {"name": "t"}}}
    )
    assert forced.tool_choice.mode == "tool"
    assert forced.tool_choice.name == "t"


# ---------- request: Canonical → OpenAI ----------


def test_canonical_with_system_becomes_first_message() -> None:
    req = CortexRequest(
        model="gpt-4o",
        max_tokens=64,
        system="be brief",
        messages=[CortexMessage(role="user", content=[TextBlock(text="hi")])],
    )
    body = to_openai_request(req)
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    assert body["messages"][1]["role"] == "user"
    assert body["messages"][1]["content"] == "hi"


def test_canonical_tool_choice_required_maps_to_required() -> None:
    req = CortexRequest(
        model="gpt-4o",
        max_tokens=10,
        messages=[CortexMessage(role="user", content=[TextBlock(text="x")])],
        tools=[CortexTool(name="t", json_schema={"type": "object"})],
        tool_choice=CortexToolChoice(mode="any"),
    )
    body = to_openai_request(req)
    assert body["tool_choice"] == "required"


def test_canonical_image_block_becomes_image_url_part() -> None:
    req = CortexRequest(
        model="gpt-4o",
        max_tokens=10,
        messages=[
            CortexMessage(
                role="user",
                content=[
                    TextBlock(text="describe"),
                    ImageBlock(media_type="image/png", data_b64="iVBOR"),
                ],
            )
        ],
    )
    body = to_openai_request(req)
    user = body["messages"][0]
    parts = user["content"]
    assert isinstance(parts, list)
    assert parts[0]["type"] == "text"
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


# ---------- streaming: OpenAI → Canonical ----------


def test_parse_openai_stream_text_only() -> None:
    state = new_openai_stream_state()
    chunks: list = []

    # Open chunk: role + empty content
    chunks += parse_openai_stream_chunk(state, {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    })
    # Content chunks
    chunks += parse_openai_stream_chunk(state, {
        "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]
    })
    chunks += parse_openai_stream_chunk(state, {
        "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}]
    })
    # Final chunk
    chunks += parse_openai_stream_chunk(state, {
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 4},
    })

    types = [type(c).__name__ for c in chunks]
    assert types[0] == "ChunkMessageStart"
    assert "ChunkContentBlockStart" in types
    text_deltas = [c for c in chunks if isinstance(c, ChunkTextDelta)]
    assert "".join(c.text for c in text_deltas) == "Hello world"
    md = next(c for c in chunks if isinstance(c, ChunkMessageDelta))
    assert md.stop_reason == "end_turn"
    assert md.output_tokens == 4
    assert types[-1] == "ChunkMessageStop"


def test_parse_openai_stream_with_tool_call() -> None:
    state = new_openai_stream_state()
    chunks: list = []
    chunks += parse_openai_stream_chunk(state, {
        "id": "chatcmpl-2", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    })
    # Text content arrives first
    chunks += parse_openai_stream_chunk(state, {
        "choices": [{"index": 0, "delta": {"content": "searching"}, "finish_reason": None}]
    })
    # First tool_call: id+name + first args fragment
    chunks += parse_openai_stream_chunk(state, {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_z",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"q":"'},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ]
    })
    # Subsequent args fragment for same tool_call (index 0)
    chunks += parse_openai_stream_chunk(state, {
        "choices": [
            {"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'dogs"}'}}]}, "finish_reason": None}
        ]
    })
    # Finish
    chunks += parse_openai_stream_chunk(state, {
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        "usage": {"completion_tokens": 7},
    })

    # Should have a TextBlock open + closed, then a ToolUseBlock open + delta x2 + closed
    block_starts = [c for c in chunks if isinstance(c, ChunkContentBlockStart)]
    assert len(block_starts) == 2
    assert isinstance(block_starts[0].block, TextBlock)
    assert isinstance(block_starts[1].block, ToolUseBlock)
    assert block_starts[1].block.tool_use_id == "call_z"

    tool_deltas = [c for c in chunks if isinstance(c, ChunkToolUseDelta)]
    assert json.loads("".join(d.partial_input_json for d in tool_deltas)) == {"q": "dogs"}

    md = next(c for c in chunks if isinstance(c, ChunkMessageDelta))
    assert md.stop_reason == "tool_use"


# ---------- streaming: Canonical → OpenAI ----------


def test_chunk_to_openai_sse_text_delta_carries_content() -> None:
    state = new_openai_egress_state("gpt-4o")
    chunk_to_openai_sse(state, ChunkMessageStart(message_id="msg_x", model="gpt-4o"))
    out = chunk_to_openai_sse(state, ChunkTextDelta(index=0, text="Hello"))
    assert out is not None
    assert out["choices"][0]["delta"] == {"content": "Hello"}


def test_chunk_to_openai_sse_tool_use_emits_id_and_name() -> None:
    state = new_openai_egress_state("gpt-4o")
    chunk_to_openai_sse(state, ChunkMessageStart(message_id="msg_x", model="gpt-4o"))
    out = chunk_to_openai_sse(
        state,
        ChunkContentBlockStart(
            index=1,
            block=ToolUseBlock(tool_use_id="call_abc", tool_name="search", tool_input={}),
        ),
    )
    assert out is not None
    tc = out["choices"][0]["delta"]["tool_calls"][0]
    assert tc["id"] == "call_abc"
    assert tc["function"]["name"] == "search"
    assert tc["function"]["arguments"] == ""


def test_chunk_to_openai_sse_finish_reason_mapping() -> None:
    state = new_openai_egress_state("gpt-4o")
    chunk_to_openai_sse(state, ChunkMessageStart(message_id="x", model="gpt-4o"))
    out = chunk_to_openai_sse(state, ChunkMessageDelta(stop_reason="tool_use"))
    assert out["choices"][0]["finish_reason"] == "tool_calls"


# ---------- aggregation ----------


def test_openai_response_from_chunks_text() -> None:
    chunks = [
        ChunkMessageStart(message_id="msg", model="gpt-4o", input_tokens=10),
        ChunkContentBlockStart(index=0, block=TextBlock(text="")),
        ChunkTextDelta(index=0, text="Hello"),
        ChunkTextDelta(index=0, text=" world"),
        ChunkContentBlockStop(index=0),
        ChunkMessageDelta(stop_reason="end_turn", output_tokens=5),
        ChunkMessageStop(),
    ]
    body = openai_response_from_chunks(chunks, model="gpt-4o")
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Hello world"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] == 10
    assert body["usage"]["completion_tokens"] == 5
    assert body["usage"]["total_tokens"] == 15


def test_openai_response_from_chunks_with_tool_call() -> None:
    chunks = [
        ChunkMessageStart(message_id="msg", model="gpt-4o", input_tokens=10),
        ChunkContentBlockStart(index=0, block=TextBlock(text="")),
        ChunkTextDelta(index=0, text="ok"),
        ChunkContentBlockStop(index=0),
        ChunkContentBlockStart(
            index=1,
            block=ToolUseBlock(tool_use_id="call_x", tool_name="search", tool_input={}),
        ),
        ChunkToolUseDelta(index=1, partial_input_json='{"q":"'),
        ChunkToolUseDelta(index=1, partial_input_json='cats"}'),
        ChunkContentBlockStop(index=1),
        ChunkMessageDelta(stop_reason="tool_use", output_tokens=8),
        ChunkMessageStop(),
    ]
    body = openai_response_from_chunks(chunks, model="gpt-4o")
    assert body["choices"][0]["message"]["content"] == "ok"
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    tcs = body["choices"][0]["message"]["tool_calls"]
    assert len(tcs) == 1
    assert tcs[0]["id"] == "call_x"
    assert tcs[0]["function"]["arguments"] == '{"q":"cats"}'
