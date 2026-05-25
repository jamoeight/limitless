"""Canonical ↔ Anthropic Messages API.

Coverage notes:
  - request: model, system (string or text-block list), messages (string or
    block list), tools, tool_choice (auto|any|tool|none), max_tokens,
    temperature, top_p, stop_sequences, stream. Other fields are preserved
    via `CortexRequest.extras` and re-attached on egress.
  - blocks: text, image (base64 + url sources), tool_use, tool_result.
  - streaming events: message_start, content_block_start, content_block_delta
    (text_delta + input_json_delta), content_block_stop, message_delta,
    message_stop, ping, error.

What we deliberately drop or lose on round-trip:
  - System blocks with `cache_control` collapse to a single string. Caching
    support comes in v1.5.
  - Anthropic's `signature` field on `thinking` blocks (extended thinking) is
    not yet modeled — those blocks are passed through as text.
"""

from __future__ import annotations

import json
from typing import Any

from cortex.canonical import (
    ChunkContentBlockStart,
    ChunkContentBlockStop,
    ChunkCortexNotice,
    ChunkError,
    ChunkMessageDelta,
    ChunkMessageStart,
    ChunkMessageStop,
    ChunkPing,
    ChunkTextDelta,
    ChunkToolUseDelta,
    CortexBlock,
    CortexChunk,
    CortexMessage,
    CortexRequest,
    CortexServerTool,
    CortexTool,
    CortexToolChoice,
    CortexToolDef,
    ImageBlock,
    OpaqueBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


def _tool_from_anthropic(t: dict[str, Any]) -> CortexToolDef:
    """Parse one entry from Anthropic's `tools` array.

    Anthropic server tools (web_search_20250305, computer_20250124,
    bash_20250124, text_editor_20250124, etc.) carry a `type` field and have
    no `input_schema`. They're invoked by the model with provider-specific
    parameters and must be forwarded opaquely — stripping them downgrades the
    assistant (no WebSearch in research turns) AND returns a 400 to the
    caller. User-defined function tools have a `name` + `input_schema` and no
    `type` field (or `type: "custom"` in newer requests).
    """
    type_field = t.get("type")
    has_input_schema = "input_schema" in t
    is_server_tool = bool(type_field) and type_field != "custom" and not has_input_schema
    if is_server_tool:
        extras = {k: v for k, v in t.items() if k not in ("type", "name")}
        return CortexServerTool(
            name=t.get("name", type_field),
            server_type=type_field,
            extras=extras,
        )
    return CortexTool(
        name=t["name"],
        description=t.get("description"),
        json_schema=t["input_schema"],
    )


def _tool_to_anthropic(t: CortexToolDef) -> dict[str, Any]:
    if isinstance(t, CortexServerTool):
        out: dict[str, Any] = {"type": t.server_type, "name": t.name}
        for k, v in t.extras.items():
            if k in ("type", "name"):
                continue
            out[k] = v
        return out
    return {
        "name": t.name,
        "description": t.description or "",
        "input_schema": t.json_schema,
    }


# ---------- Request: Anthropic → Canonical ----------


def from_anthropic_request(body: dict[str, Any]) -> CortexRequest:
    """Parse an Anthropic `/v1/messages` request body into a CortexRequest.

    Raises:
        KeyError / ValueError if required fields are missing or malformed.
    """
    system = body.get("system")
    if isinstance(system, list):
        # Anthropic allows system as a list of text blocks (sometimes with
        # cache_control). Flatten to a single string; cache_control is dropped.
        system = "\n\n".join(
            b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"
        )
    elif system is not None and not isinstance(system, str):
        system = str(system)

    messages: list[CortexMessage] = []
    for m in body.get("messages", []):
        role = m["role"]
        if role not in ("user", "assistant"):
            raise ValueError(f"unsupported role: {role}")
        raw_content = m["content"]
        if isinstance(raw_content, str):
            blocks: list[CortexBlock] = [TextBlock(text=raw_content)]
        else:
            blocks = [_block_from_anthropic(b) for b in raw_content]
        messages.append(CortexMessage(role=role, content=blocks))

    tools: list[CortexToolDef] = [_tool_from_anthropic(t) for t in body.get("tools", [])]

    tc_raw = body.get("tool_choice")
    if tc_raw is None:
        tc = CortexToolChoice()  # default: auto
    else:
        tc_type = tc_raw.get("type", "auto")
        if tc_type == "tool":
            tc = CortexToolChoice(mode="tool", name=tc_raw.get("name"))
        elif tc_type in ("auto", "any", "none"):
            tc = CortexToolChoice(mode=tc_type)
        else:
            tc = CortexToolChoice()  # fallback

    extras: dict[str, Any] = {}
    for k in ("metadata", "service_tier"):
        if k in body:
            extras[k] = body[k]

    return CortexRequest(
        model=body["model"],
        system=system,
        messages=messages,
        max_tokens=body.get("max_tokens", 4096),
        stream=bool(body.get("stream", False)),
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        stop_sequences=list(body.get("stop_sequences", [])),
        tools=tools,
        tool_choice=tc,
        extras=extras,
    )


def _block_from_anthropic(b: dict[str, Any]) -> CortexBlock:
    t = b.get("type")
    if t == "text":
        return TextBlock(text=b.get("text", ""))
    if t == "image":
        src = b.get("source", {})
        st = src.get("type")
        if st == "base64":
            return ImageBlock(
                media_type=src.get("media_type", "image/jpeg"),
                data_b64=src.get("data", ""),
            )
        if st == "url":
            return ImageBlock(
                media_type=src.get("media_type", "image/jpeg"),
                url=src.get("url", ""),
            )
        raise ValueError(f"unknown image source type: {st!r}")
    if t == "tool_use":
        return ToolUseBlock(
            tool_use_id=b["id"],
            tool_name=b["name"],
            tool_input=b.get("input", {}) or {},
        )
    if t == "tool_result":
        raw = b.get("content", "")
        if isinstance(raw, list):
            content: str | list[CortexBlock] = [_block_from_anthropic(sub) for sub in raw]
        elif isinstance(raw, str):
            content = raw
        else:
            content = str(raw)
        return ToolResultBlock(
            tool_use_id=b["tool_use_id"],
            content=content,
            is_error=bool(b.get("is_error", False)),
        )
    if t == "thinking":
        # Extended-thinking blocks: preserve the text payload but lose the
        # signature. Models tolerate this on retry; full fidelity comes later.
        return TextBlock(text=b.get("thinking", ""))
    # Anything else (server_tool_use, web_search_tool_result,
    # redacted_thinking, future block kinds the API adds) rides through as
    # OpaqueBlock. Raising used to 502 the whole request — the legacy
    # behavior broke WebSearch end-to-end because every web_search result
    # streamed back a `server_tool_use` block and crashed the SSE parser.
    return OpaqueBlock(original_type=str(t) if t is not None else "unknown", payload=dict(b))


# ---------- Request: Canonical → Anthropic ----------


def to_anthropic_request(req: CortexRequest) -> dict[str, Any]:
    """Serialize a CortexRequest to an Anthropic `/v1/messages` body."""
    body: dict[str, Any] = {
        "model": req.model,
        "max_tokens": req.max_tokens,
        "messages": [_message_to_anthropic(m) for m in req.messages],
        "stream": req.stream,
    }
    if req.system is not None and req.system != "":
        body["system"] = req.system
    if req.temperature is not None:
        body["temperature"] = req.temperature
    if req.top_p is not None:
        body["top_p"] = req.top_p
    if req.stop_sequences:
        body["stop_sequences"] = list(req.stop_sequences)
    if req.tools:
        body["tools"] = [_tool_to_anthropic(t) for t in req.tools]
        tc = req.tool_choice
        if tc.mode == "tool" and tc.name:
            body["tool_choice"] = {"type": "tool", "name": tc.name}
        elif tc.mode != "auto":
            body["tool_choice"] = {"type": tc.mode}
    # Re-attach extras (don't clobber fields already set).
    for k, v in req.extras.items():
        body.setdefault(k, v)
    return body


def _message_to_anthropic(m: CortexMessage) -> dict[str, Any]:
    return {
        "role": m.role,
        "content": [_block_to_anthropic(b) for b in m.content],
    }


def _block_to_anthropic(b: CortexBlock) -> dict[str, Any]:
    if isinstance(b, TextBlock):
        return {"type": "text", "text": b.text}
    if isinstance(b, ImageBlock):
        if b.data_b64:
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": b.media_type,
                    "data": b.data_b64,
                },
            }
        if b.url:
            return {
                "type": "image",
                "source": {"type": "url", "media_type": b.media_type, "url": b.url},
            }
        raise ValueError("image block missing both data_b64 and url")
    if isinstance(b, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": b.tool_use_id,
            "name": b.tool_name,
            "input": b.tool_input,
        }
    if isinstance(b, ToolResultBlock):
        if isinstance(b.content, list):
            content_out: Any = [_block_to_anthropic(sub) for sub in b.content]
        else:
            content_out = b.content
        out: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": b.tool_use_id,
            "content": content_out,
        }
        if b.is_error:
            out["is_error"] = True
        return out
    if isinstance(b, OpaqueBlock):
        # Byte-equivalent passthrough; the payload was captured verbatim on
        # ingress so the upstream sees what it would have seen direct.
        return dict(b.payload)
    raise TypeError(f"cannot serialize block type {type(b).__name__}")


# ---------- Streaming: Anthropic SSE → Canonical chunks ----------


def parse_anthropic_event(event_name: str, data: dict[str, Any]) -> CortexChunk | None:
    """Parse one SSE event from an Anthropic stream.

    Returns None for unknown or non-actionable events.
    """
    t = data.get("type") or event_name

    if t == "message_start":
        m = data.get("message", {})
        usage = m.get("usage", {}) or {}
        return ChunkMessageStart(
            message_id=m.get("id", ""),
            model=m.get("model", ""),
            input_tokens=usage.get("input_tokens"),
        )
    if t == "content_block_start":
        return ChunkContentBlockStart(
            index=data["index"],
            block=_block_from_anthropic(data["content_block"]),
        )
    if t == "content_block_delta":
        d = data.get("delta", {})
        dt = d.get("type")
        if dt == "text_delta":
            return ChunkTextDelta(index=data["index"], text=d.get("text", ""))
        if dt == "input_json_delta":
            return ChunkToolUseDelta(
                index=data["index"],
                partial_input_json=d.get("partial_json", ""),
            )
        return None
    if t == "content_block_stop":
        return ChunkContentBlockStop(index=data["index"])
    if t == "message_delta":
        d = data.get("delta", {}) or {}
        usage = data.get("usage", {}) or {}
        return ChunkMessageDelta(
            stop_reason=d.get("stop_reason"),
            stop_sequence=d.get("stop_sequence"),
            output_tokens=usage.get("output_tokens"),
        )
    if t == "message_stop":
        return ChunkMessageStop()
    if t == "ping":
        return ChunkPing()
    if t == "error":
        err = data.get("error", {}) or {}
        return ChunkError(
            error_type=err.get("type", "unknown_error"),
            message=err.get("message", ""),
        )
    return None


# ---------- Streaming: Canonical chunks → Anthropic SSE ----------


def chunk_to_anthropic_sse(chunk: CortexChunk) -> tuple[str, dict[str, Any]] | None:
    """Serialize a canonical chunk as (event_name, data_dict) for Anthropic SSE."""
    if isinstance(chunk, ChunkMessageStart):
        return "message_start", {
            "type": "message_start",
            "message": {
                "id": chunk.message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": chunk.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": chunk.input_tokens or 0,
                    "output_tokens": 0,
                },
            },
        }
    if isinstance(chunk, ChunkContentBlockStart):
        return "content_block_start", {
            "type": "content_block_start",
            "index": chunk.index,
            "content_block": _block_to_anthropic(chunk.block),
        }
    if isinstance(chunk, ChunkTextDelta):
        return "content_block_delta", {
            "type": "content_block_delta",
            "index": chunk.index,
            "delta": {"type": "text_delta", "text": chunk.text},
        }
    if isinstance(chunk, ChunkToolUseDelta):
        return "content_block_delta", {
            "type": "content_block_delta",
            "index": chunk.index,
            "delta": {"type": "input_json_delta", "partial_json": chunk.partial_input_json},
        }
    if isinstance(chunk, ChunkContentBlockStop):
        return "content_block_stop", {"type": "content_block_stop", "index": chunk.index}
    if isinstance(chunk, ChunkMessageDelta):
        out: dict[str, Any] = {"type": "message_delta", "delta": {}}
        if chunk.stop_reason is not None:
            out["delta"]["stop_reason"] = chunk.stop_reason
        if chunk.stop_sequence is not None:
            out["delta"]["stop_sequence"] = chunk.stop_sequence
        if chunk.output_tokens is not None:
            out["usage"] = {"output_tokens": chunk.output_tokens}
        return "message_delta", out
    if isinstance(chunk, ChunkMessageStop):
        return "message_stop", {"type": "message_stop"}
    if isinstance(chunk, ChunkPing):
        return "ping", {"type": "ping"}
    if isinstance(chunk, ChunkError):
        return "error", {
            "type": "error",
            "error": {"type": chunk.error_type, "message": chunk.message},
        }
    if isinstance(chunk, ChunkCortexNotice):
        # Custom event name; clients that don't subscribe ignore it.
        return "cortex.notice", {
            "type": "cortex_notice",
            "notice_kind": chunk.notice_kind,
            "message": chunk.message,
            "metadata": chunk.metadata,
        }
    return None


# ---------- Aggregation: chunks → non-streaming Anthropic response ----------


def response_from_chunks(chunks: list[CortexChunk], model: str) -> dict[str, Any]:
    """Aggregate a chunk stream into an Anthropic non-streaming response body.

    Used when the client sent `stream: false` — we always stream from upstream,
    then collect into the response shape the client expects.
    """
    message_id = ""
    content_blocks: dict[int, dict[str, Any]] = {}
    block_order: list[int] = []
    partial_tool_json: dict[int, str] = {}
    stop_reason: str | None = "end_turn"
    stop_sequence: str | None = None
    input_tokens = 0
    output_tokens = 0

    for chunk in chunks:
        if isinstance(chunk, ChunkMessageStart):
            message_id = chunk.message_id or message_id
            if chunk.input_tokens is not None:
                input_tokens = chunk.input_tokens
        elif isinstance(chunk, ChunkContentBlockStart):
            blk = _block_to_anthropic(chunk.block)
            content_blocks[chunk.index] = blk
            block_order.append(chunk.index)
            if blk.get("type") == "tool_use":
                partial_tool_json[chunk.index] = ""
        elif isinstance(chunk, ChunkTextDelta):
            blk = content_blocks.get(chunk.index)
            if blk is not None and blk.get("type") == "text":
                blk["text"] = blk.get("text", "") + chunk.text
        elif isinstance(chunk, ChunkToolUseDelta):
            partial_tool_json[chunk.index] = partial_tool_json.get(chunk.index, "") + chunk.partial_input_json
        elif isinstance(chunk, ChunkContentBlockStop):
            blk = content_blocks.get(chunk.index)
            if blk is not None and blk.get("type") == "tool_use":
                raw = partial_tool_json.pop(chunk.index, "")
                try:
                    blk["input"] = json.loads(raw) if raw.strip() else {}
                except json.JSONDecodeError:
                    blk["input"] = {"_partial_json": raw}
        elif isinstance(chunk, ChunkMessageDelta):
            if chunk.stop_reason is not None:
                stop_reason = chunk.stop_reason
            if chunk.stop_sequence is not None:
                stop_sequence = chunk.stop_sequence
            if chunk.output_tokens is not None:
                output_tokens = chunk.output_tokens

    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "content": [content_blocks[i] for i in block_order if i in content_blocks],
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }
