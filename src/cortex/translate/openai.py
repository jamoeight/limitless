"""Canonical ↔ OpenAI Chat Completions API.

Coverage:
  - Request: messages (string + multimodal content arrays), system (extracted
    from a leading role=system entry), tools, tool_choice (auto|none|required|
    {type:function,function:{name}}), max_tokens / max_completion_tokens,
    temperature, top_p, stop, stream.
  - Blocks: text, image (via image_url with data URL or remote URL),
    tool_use (assistant.tool_calls), tool_result (role=tool).
  - Streaming: parses chunked deltas with role-first, content-string, and
    indexed tool_calls. Tool-call argument accumulation is intrinsic to the
    canonical format — we just forward partial JSON deltas.

Key design notes:
  - OpenAI's stream collapses text + tool_calls into one flat delta. We
    manufacture canonical content_block_start / content_block_stop events
    around them so downstream code (e.g. response_from_chunks) doesn't have
    to special-case which provider emitted what.
  - OpenAI splits role=tool messages out into their own entry per
    tool_call_id. Anthropic groups them into ONE user message with multiple
    tool_result blocks. The egress translator handles either input.
  - Stop reasons are translated bidirectionally via small maps.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from typing import Any

from cortex.canonical import (
    ChunkContentBlockStart,
    ChunkContentBlockStop,
    ChunkError,
    ChunkMessageDelta,
    ChunkMessageStart,
    ChunkMessageStop,
    ChunkTextDelta,
    ChunkToolUseDelta,
    CortexBlock,
    CortexChunk,
    CortexMessage,
    CortexRequest,
    CortexTool,
    CortexToolChoice,
    ImageBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


# ---------- Stop-reason maps ----------


def anthropic_to_openai_stop_reason(reason: str | None) -> str | None:
    if reason is None:
        return None
    return {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }.get(reason, "stop")


def openai_to_anthropic_stop_reason(reason: str | None) -> str | None:
    if reason is None:
        return None
    return {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "end_turn",
    }.get(reason, "end_turn")


# ---------- Request: OpenAI → Canonical ----------


def from_openai_request(body: dict[str, Any]) -> CortexRequest:
    """Parse an OpenAI `/v1/chat/completions` body into a CortexRequest."""
    system: str | None = None
    messages: list[CortexMessage] = []
    # Track open assistant tool_calls so subsequent role=tool messages know
    # which assistant turn they belong to.
    for entry in body.get("messages", []):
        role = entry.get("role")
        if role == "system":
            # Concatenate multiple system entries (OpenAI allows this).
            content = _flatten_text(entry.get("content"))
            system = (system + "\n\n" + content) if system else content
        elif role == "developer":
            # Newer OpenAI "developer" role: treat as system.
            content = _flatten_text(entry.get("content"))
            system = (system + "\n\n" + content) if system else content
        elif role == "user":
            messages.append(CortexMessage(role="user", content=_openai_user_blocks(entry)))
        elif role == "assistant":
            messages.append(
                CortexMessage(role="assistant", content=_openai_assistant_blocks(entry))
            )
        elif role == "tool":
            # OpenAI puts each tool result in its own role=tool message. We
            # need to collapse it into a user message with a tool_result
            # block. If the previous canonical message is already a user
            # message containing only tool_result blocks, append to it.
            tr = ToolResultBlock(
                tool_use_id=entry.get("tool_call_id", ""),
                content=_flatten_text(entry.get("content", "")),
                is_error=False,
            )
            if (
                messages
                and messages[-1].role == "user"
                and all(isinstance(b, ToolResultBlock) for b in messages[-1].content)
            ):
                messages[-1] = CortexMessage(
                    role="user", content=[*messages[-1].content, tr]
                )
            else:
                messages.append(CortexMessage(role="user", content=[tr]))
        else:
            # function/unknown roles — skip (legacy `function` is replaced by
            # `tool` since OpenAI deprecated function-calling).
            continue

    tools = [
        CortexTool(
            name=t["function"]["name"],
            description=t["function"].get("description"),
            json_schema=t["function"].get("parameters", {"type": "object"}),
        )
        for t in body.get("tools", [])
        if t.get("type") == "function"
    ]

    tc_raw = body.get("tool_choice")
    if tc_raw is None:
        tool_choice = CortexToolChoice()
    elif isinstance(tc_raw, str):
        # "auto" | "none" | "required"
        mode = {"required": "any"}.get(tc_raw, tc_raw)
        if mode in ("auto", "any", "none"):
            tool_choice = CortexToolChoice(mode=mode)
        else:
            tool_choice = CortexToolChoice()
    elif isinstance(tc_raw, dict):
        if tc_raw.get("type") == "function":
            tool_choice = CortexToolChoice(
                mode="tool", name=tc_raw.get("function", {}).get("name")
            )
        else:
            tool_choice = CortexToolChoice()
    else:
        tool_choice = CortexToolChoice()

    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 4096
    stop_sequences = body.get("stop", [])
    if isinstance(stop_sequences, str):
        stop_sequences = [stop_sequences]

    extras: dict[str, Any] = {}
    for k in ("seed", "presence_penalty", "frequency_penalty", "logit_bias", "user", "response_format", "stream_options"):
        if k in body:
            extras[k] = body[k]

    return CortexRequest(
        model=body["model"],
        system=system,
        messages=messages,
        max_tokens=max_tokens,
        stream=bool(body.get("stream", False)),
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        stop_sequences=list(stop_sequences) if stop_sequences else [],
        tools=tools,
        tool_choice=tool_choice,
        extras=extras,
    )


def _flatten_text(content: Any) -> str:
    """Convert OpenAI's content field (string OR list of parts) to a string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
        return "".join(parts)
    return str(content)


def _openai_user_blocks(entry: dict[str, Any]) -> list[CortexBlock]:
    content = entry.get("content")
    if isinstance(content, str):
        return [TextBlock(text=content)]
    if not isinstance(content, list):
        return [TextBlock(text=str(content) if content else "")]
    blocks: list[CortexBlock] = []
    for p in content:
        t = p.get("type")
        if t == "text":
            blocks.append(TextBlock(text=p.get("text", "")))
        elif t == "image_url":
            url = p.get("image_url", {}).get("url", "")
            if url.startswith("data:"):
                # data:image/png;base64,xxxx
                try:
                    head, b64 = url.split(",", 1)
                    media_type = head[len("data:") :].split(";", 1)[0] or "image/jpeg"
                except ValueError:
                    media_type, b64 = "image/jpeg", ""
                blocks.append(ImageBlock(media_type=media_type, data_b64=b64))
            else:
                blocks.append(ImageBlock(media_type="image/jpeg", url=url))
        # input_audio, file, etc. → skipped for v1.
    return blocks or [TextBlock(text="")]


def _openai_assistant_blocks(entry: dict[str, Any]) -> list[CortexBlock]:
    blocks: list[CortexBlock] = []
    content = entry.get("content")
    text = _flatten_text(content)
    if text:
        blocks.append(TextBlock(text=text))
    for tc in entry.get("tool_calls", []) or []:
        if tc.get("type") != "function":
            continue
        fn = tc.get("function", {})
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            args = {"_partial_json": raw_args}
        blocks.append(
            ToolUseBlock(
                tool_use_id=tc.get("id", ""),
                tool_name=fn.get("name", ""),
                tool_input=args,
            )
        )
    return blocks


# ---------- Request: Canonical → OpenAI ----------


def to_openai_request(req: CortexRequest) -> dict[str, Any]:
    """Serialize a CortexRequest to an OpenAI Chat Completions body."""
    messages: list[dict[str, Any]] = []
    if req.system:
        messages.append({"role": "system", "content": req.system})

    for m in req.messages:
        messages.extend(_message_to_openai(m))

    body: dict[str, Any] = {
        "model": req.model,
        "messages": messages,
        "stream": req.stream,
    }
    if req.max_tokens:
        body["max_tokens"] = req.max_tokens
    if req.temperature is not None:
        body["temperature"] = req.temperature
    if req.top_p is not None:
        body["top_p"] = req.top_p
    if req.stop_sequences:
        body["stop"] = list(req.stop_sequences)
    if req.tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.json_schema,
                },
            }
            for t in req.tools
        ]
        tc = req.tool_choice
        if tc.mode == "tool" and tc.name:
            body["tool_choice"] = {"type": "function", "function": {"name": tc.name}}
        elif tc.mode == "any":
            body["tool_choice"] = "required"
        elif tc.mode in ("none",):
            body["tool_choice"] = tc.mode
        # mode == "auto" is the default; omit
    for k, v in req.extras.items():
        body.setdefault(k, v)
    return body


def _message_to_openai(m: CortexMessage) -> list[dict[str, Any]]:
    """A single CortexMessage may produce multiple OpenAI messages (e.g., one
    role=tool entry per tool_result block)."""
    out: list[dict[str, Any]] = []

    # Split pure tool_result user messages into their own role=tool entries.
    if m.role == "user" and m.content and all(isinstance(b, ToolResultBlock) for b in m.content):
        for b in m.content:
            assert isinstance(b, ToolResultBlock)
            content_str = (
                b.content if isinstance(b.content, str) else _blocks_to_text(b.content)
            )
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": b.tool_use_id,
                    "content": content_str,
                }
            )
        return out

    if m.role == "user":
        out.append({"role": "user", "content": _user_blocks_to_openai(m.content)})
        return out

    # Assistant: collect text + tool_uses.
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for b in m.content:
        if isinstance(b, TextBlock):
            text_parts.append(b.text)
        elif isinstance(b, ToolUseBlock):
            try:
                args = json.dumps(b.tool_input, default=str)
            except (TypeError, ValueError):
                args = "{}"
            tool_calls.append(
                {
                    "id": b.tool_use_id,
                    "type": "function",
                    "function": {"name": b.tool_name, "arguments": args},
                }
            )
    entry: dict[str, Any] = {"role": "assistant"}
    entry["content"] = "".join(text_parts) if text_parts else None
    if tool_calls:
        entry["tool_calls"] = tool_calls
    out.append(entry)
    return out


def _user_blocks_to_openai(blocks: list[CortexBlock]) -> Any:
    """A user message's content: string (text only) OR array (with images)."""
    if all(isinstance(b, TextBlock) for b in blocks):
        return "".join(b.text for b in blocks if isinstance(b, TextBlock))
    parts: list[dict[str, Any]] = []
    for b in blocks:
        if isinstance(b, TextBlock):
            parts.append({"type": "text", "text": b.text})
        elif isinstance(b, ImageBlock):
            if b.data_b64:
                url = f"data:{b.media_type};base64,{b.data_b64}"
            else:
                url = b.url or ""
            parts.append({"type": "image_url", "image_url": {"url": url}})
        elif isinstance(b, ToolResultBlock):
            # Shouldn't happen in pure-multimodal user message, but handle.
            parts.append({"type": "text", "text": _blocks_to_text([b])})
    return parts


def _blocks_to_text(blocks: list[CortexBlock]) -> str:
    out: list[str] = []
    for b in blocks:
        if isinstance(b, TextBlock):
            out.append(b.text)
        elif isinstance(b, ImageBlock):
            out.append(f"[image:{b.media_type}]")
    return "".join(out)


# ---------- Streaming: OpenAI deltas → Canonical chunks ----------


class _OpenAIStreamState:
    """Parser state machine for an OpenAI streaming response.

    OpenAI sends a flat delta stream; we manufacture canonical
    content_block_start / content_block_stop events around text and tool_use
    blocks so downstream code is provider-agnostic.
    """

    def __init__(self) -> None:
        self.message_started = False
        self.message_id = ""
        self.model = ""
        self.text_open = False
        self.text_block_idx = 0
        # openai tc index → {block_idx, name, id, sent_start}
        self.tool_state: dict[int, dict[str, Any]] = {}
        self.next_block_idx = 0
        # True once any tool_call delta has been observed. Used to suppress
        # whitespace-only content deltas that LM Studio's qwen3 chat template
        # emits between extracted tool_calls — opencode's ai-sdk parser sees
        # that whitespace as "model has resumed text generation" and discards
        # all subsequent tool_calls, ending the agent loop after one tool.
        self.tool_call_seen = False

    def _allocate_block_index(self) -> int:
        idx = self.next_block_idx
        self.next_block_idx += 1
        return idx

    def consume(self, chunk_obj: dict[str, Any]) -> Iterator[CortexChunk]:
        if not self.message_started:
            self.message_started = True
            self.message_id = chunk_obj.get("id", f"msg_{uuid.uuid4().hex[:12]}")
            self.model = chunk_obj.get("model", "")
            yield ChunkMessageStart(message_id=self.message_id, model=self.model)

        if "error" in chunk_obj:
            err = chunk_obj["error"]
            yield ChunkError(
                error_type=err.get("type", "openai_error"),
                message=err.get("message", ""),
            )
            return

        choices = chunk_obj.get("choices", [])
        if not choices:
            # Usage-only chunk at the end of stream (when stream_options.include_usage).
            usage = chunk_obj.get("usage")
            if usage:
                yield ChunkMessageDelta(output_tokens=usage.get("completion_tokens"))
            return

        choice = choices[0]
        delta = choice.get("delta", {})

        # Text content delta
        if "content" in delta and delta["content"]:
            content_text = delta["content"]
            # Drop whitespace-only content that arrives after the first
            # tool_call started. See `tool_call_seen` field comment for why.
            if not (self.tool_call_seen and content_text.strip() == ""):
                if not self.text_open:
                    self.text_block_idx = self._allocate_block_index()
                    self.text_open = True
                    yield ChunkContentBlockStart(
                        index=self.text_block_idx, block=TextBlock(text="")
                    )
                yield ChunkTextDelta(index=self.text_block_idx, text=content_text)

        # Tool-call deltas
        for tc_delta in delta.get("tool_calls", []) or []:
            self.tool_call_seen = True
            tc_idx = tc_delta.get("index", 0)
            state = self.tool_state.get(tc_idx)
            if state is None:
                # New tool call — close text block if open, emit content_block_start
                if self.text_open:
                    yield ChunkContentBlockStop(index=self.text_block_idx)
                    self.text_open = False
                block_idx = self._allocate_block_index()
                fn = tc_delta.get("function", {})
                state = {
                    "block_idx": block_idx,
                    "id": tc_delta.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                    "name": fn.get("name", ""),
                }
                self.tool_state[tc_idx] = state
                yield ChunkContentBlockStart(
                    index=block_idx,
                    block=ToolUseBlock(
                        tool_use_id=state["id"],
                        tool_name=state["name"],
                        tool_input={},
                    ),
                )

            # Forward partial argument JSON, if any.
            fn = tc_delta.get("function", {})
            partial = fn.get("arguments")
            if partial:
                yield ChunkToolUseDelta(
                    index=state["block_idx"], partial_input_json=partial
                )

        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            # Close any open blocks
            if self.text_open:
                yield ChunkContentBlockStop(index=self.text_block_idx)
                self.text_open = False
            for state in self.tool_state.values():
                yield ChunkContentBlockStop(index=state["block_idx"])
            self.tool_state.clear()

            usage = chunk_obj.get("usage") or {}
            yield ChunkMessageDelta(
                stop_reason=openai_to_anthropic_stop_reason(finish_reason),
                output_tokens=usage.get("completion_tokens"),
            )
            yield ChunkMessageStop()


def parse_openai_stream_chunk(state: _OpenAIStreamState, chunk_obj: dict[str, Any]) -> list[CortexChunk]:
    """Convenience wrapper: feed one OpenAI SSE event payload to the state
    machine and return any canonical chunks it produces."""
    return list(state.consume(chunk_obj))


def new_openai_stream_state() -> _OpenAIStreamState:
    return _OpenAIStreamState()


# ---------- Streaming: Canonical chunks → OpenAI SSE ----------


class _OpenAIEgressState:
    """Track which OpenAI tool_call index each canonical block index maps to."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.message_id = ""
        self.created = int(time.time())
        self.role_emitted = False
        # canonical block_idx → openai tool_call index
        self.tool_call_idx_for: dict[int, int] = {}
        self.next_tool_call_idx = 0
        self.block_kind: dict[int, str] = {}  # block_idx → "text" | "tool_use"
        # True once any tool_call has been emitted to the client. Used to
        # force finish_reason="tool_calls" on the closing chunk even if the
        # upstream reported "stop" — opencode's prompt loop checks this
        # field to decide whether to keep iterating, and some upstreams
        # (notably LM Studio + qwen3 when content is intermixed with the
        # extracted tool_call) misreport the finish reason.
        self.tool_call_emitted = False


def chunk_to_openai_sse(state: _OpenAIEgressState, chunk: CortexChunk) -> dict[str, Any] | None:
    """Serialize one canonical chunk as an OpenAI Chat Completions chunk.

    Returns None for chunks that have no representation (e.g., Anthropic ping).
    The caller wraps each returned dict in `data: <json>\\n\\n`.
    A separate `data: [DONE]` line is emitted by the caller AFTER the final
    chunk (mapped from ChunkMessageStop).
    """
    if isinstance(chunk, ChunkMessageStart):
        state.message_id = chunk.message_id or f"chatcmpl-{uuid.uuid4().hex[:24]}"
        state.model = chunk.model or state.model
        return _openai_chunk_skeleton(
            state,
            {"role": "assistant", "content": ""},
            finish_reason=None,
        )

    if isinstance(chunk, ChunkContentBlockStart):
        state.block_kind[chunk.index] = chunk.block.type
        if isinstance(chunk.block, ToolUseBlock):
            tc_idx = state.next_tool_call_idx
            state.next_tool_call_idx += 1
            state.tool_call_idx_for[chunk.index] = tc_idx
            state.tool_call_emitted = True
            return _openai_chunk_skeleton(
                state,
                {
                    "tool_calls": [
                        {
                            "index": tc_idx,
                            "id": chunk.block.tool_use_id,
                            "type": "function",
                            "function": {
                                "name": chunk.block.tool_name,
                                "arguments": "",
                            },
                        }
                    ]
                },
                finish_reason=None,
            )
        # Text block_start: OpenAI doesn't have an explicit start, but we
        # can emit a no-op chunk to maintain stream timing.
        return None

    if isinstance(chunk, ChunkTextDelta):
        return _openai_chunk_skeleton(state, {"content": chunk.text}, finish_reason=None)

    if isinstance(chunk, ChunkToolUseDelta):
        tc_idx = state.tool_call_idx_for.get(chunk.index, 0)
        return _openai_chunk_skeleton(
            state,
            {
                "tool_calls": [
                    {
                        "index": tc_idx,
                        "function": {"arguments": chunk.partial_input_json},
                    }
                ]
            },
            finish_reason=None,
        )

    if isinstance(chunk, ChunkContentBlockStop):
        # OpenAI doesn't emit per-block stops.
        return None

    if isinstance(chunk, ChunkMessageDelta):
        finish = anthropic_to_openai_stop_reason(chunk.stop_reason)
        # Defensive override: if any tool_call was emitted during this turn,
        # force finish_reason="tool_calls" regardless of what upstream said.
        # opencode's session loop checks finish === "tool-calls" to decide
        # whether to keep iterating; an upstream that reports "stop" while
        # also emitting tool_calls (LM Studio + qwen3 with intermixed
        # content has been observed doing this) breaks the agent loop.
        if state.tool_call_emitted and finish != "tool_calls":
            finish = "tool_calls"
        return _openai_chunk_skeleton(state, {}, finish_reason=finish)

    if isinstance(chunk, ChunkMessageStop):
        # Final OpenAI behavior: emit no extra delta; caller emits [DONE].
        return None

    if isinstance(chunk, ChunkError):
        return {
            "error": {"type": chunk.error_type, "message": chunk.message},
        }

    # ChunkPing, ChunkCortexNotice: drop for OpenAI.
    return None


def _openai_chunk_skeleton(
    state: _OpenAIEgressState,
    delta: dict[str, Any],
    finish_reason: str | None,
) -> dict[str, Any]:
    return {
        "id": state.message_id or f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": state.created,
        "model": state.model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def new_openai_egress_state(model: str) -> _OpenAIEgressState:
    return _OpenAIEgressState(model=model)


# ---------- Aggregation: chunks → non-streaming OpenAI response ----------


def openai_response_from_chunks(chunks: list[CortexChunk], model: str) -> dict[str, Any]:
    """Aggregate a canonical chunk stream into an OpenAI non-streaming
    response body (for `stream: false` requests)."""
    message_id = ""
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    partial_json: dict[int, str] = {}  # block_idx → accumulating args
    tool_blocks: dict[int, dict[str, Any]] = {}
    finish_reason = "stop"
    input_tokens = 0
    output_tokens = 0

    for chunk in chunks:
        if isinstance(chunk, ChunkMessageStart):
            message_id = chunk.message_id or message_id
            if chunk.input_tokens is not None:
                input_tokens = chunk.input_tokens
        elif isinstance(chunk, ChunkContentBlockStart):
            if isinstance(chunk.block, ToolUseBlock):
                tool_blocks[chunk.index] = {
                    "id": chunk.block.tool_use_id,
                    "type": "function",
                    "function": {"name": chunk.block.tool_name, "arguments": ""},
                }
                partial_json[chunk.index] = ""
        elif isinstance(chunk, ChunkTextDelta):
            text_parts.append(chunk.text)
        elif isinstance(chunk, ChunkToolUseDelta):
            partial_json[chunk.index] = partial_json.get(chunk.index, "") + chunk.partial_input_json
        elif isinstance(chunk, ChunkContentBlockStop):
            if chunk.index in tool_blocks:
                args = partial_json.pop(chunk.index, "")
                tool_blocks[chunk.index]["function"]["arguments"] = args
                tool_calls.append(tool_blocks.pop(chunk.index))
        elif isinstance(chunk, ChunkMessageDelta):
            if chunk.stop_reason is not None:
                finish_reason = anthropic_to_openai_stop_reason(chunk.stop_reason) or finish_reason
            if chunk.output_tokens is not None:
                output_tokens = chunk.output_tokens

    message: dict[str, Any] = {"role": "assistant"}
    content_str = "".join(text_parts)
    message["content"] = content_str if content_str else None
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": message_id or f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish_reason}
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }
