"""Canonical wire format used inside Cortex.

Every inbound request is parsed into `CortexRequest`. Every provider yields
`CortexChunk`s. Translators live in `cortex.translate.*` and convert between
the canonical types and provider-specific shapes (Anthropic Messages, OpenAI
Chat Completions, LM Studio OpenAI-compat).

Why a canonical layer:
  - One shape to reason about for virtualization, ingest, recall.
  - One shape to test the four-corner streaming matrix against.
  - Adding a fifth provider in the future = one translator, no churn elsewhere.

What we deliberately DO NOT canonicalize:
  - Provider-specific raw extras (e.g., Anthropic's `metadata`, OpenAI's
    `seed`, `logit_bias`). These ride along on `CortexRequest.extra` and are
    re-attached by the egress translator if the upstream provider matches the
    ingress format. Round-tripping the same provider is byte-equivalent.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

# ---------- Content blocks ----------


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    media_type: str  # "image/png" | "image/jpeg" | ...
    # Exactly one of `data_b64` (Anthropic style) or `url` (OpenAI style) is set.
    data_b64: str | None = None
    url: str | None = None


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    tool_use_id: str
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    # Anthropic allows `content` to be a string or a list of blocks (e.g., text
    # + image). Internally we keep it as text-or-blocks; the translator decides
    # which shape to emit for the upstream provider.
    content: str | list["CortexBlock"]
    is_error: bool = False


class OpaqueBlock(BaseModel):
    """Pass-through for block types the proxy doesn't model strictly.

    Anthropic ships new block kinds faster than we can model them — server
    tools emit `server_tool_use` + `web_search_tool_result`, extended-thinking
    adds `redacted_thinking`, etc. Rather than raise on every new shape and
    return a 502 (which is what the legacy `unknown block type` ValueError
    did, breaking WebSearch end-to-end), unrecognized blocks land here. The
    raw dict is carried verbatim and re-emitted on egress byte-equivalent.
    OpenAI egress drops them (no analogue exists there).
    """

    type: Literal["opaque"] = "opaque"
    original_type: str
    payload: dict[str, Any] = Field(default_factory=dict)


CortexBlock = Annotated[
    TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock | OpaqueBlock,
    Field(discriminator="type"),
]


# Pydantic needs the forward ref resolved.
ToolResultBlock.model_rebuild()


# ---------- Messages ----------


class CortexMessage(BaseModel):
    role: Literal["user", "assistant"]
    # System messages are extracted to CortexRequest.system; the messages list
    # holds only user/assistant turns.
    content: list[CortexBlock]


# ---------- Tools ----------
#
# Two flavors:
#   - CortexTool — user-defined "function" tools (JSON Schema input). Both
#     Anthropic and OpenAI accept these; they round-trip losslessly.
#   - CortexServerTool — provider-hosted tools (e.g., Anthropic's
#     `web_search_20250305`, `computer_20250124`, `bash_20250124`,
#     `text_editor_20250124`). These have NO input_schema; the model invokes
#     them with provider-specific extras. We carry them opaquely so any caller
#     who would have hit `api.anthropic.com` directly with the tool keeps
#     working through the proxy. OpenAI-egress drops server tools (they have
#     no equivalent there).


class CortexTool(BaseModel):
    kind: Literal["function"] = "function"
    name: str
    description: str | None = None
    # JSON Schema as a dict. Anthropic calls this `input_schema`; OpenAI calls
    # it `function.parameters`. Same content, different field name.
    json_schema: dict[str, Any]


class CortexServerTool(BaseModel):
    kind: Literal["server"] = "server"
    name: str
    # Provider-side tool identifier — e.g., "web_search_20250305". Carried
    # verbatim on egress.
    server_type: str
    # Opaque per-tool configuration (max_uses, allowed_domains,
    # display_width_px, etc.) — preserved on egress, never inspected by the
    # proxy.
    extras: dict[str, Any] = Field(default_factory=dict)


CortexToolDef = Annotated[
    CortexTool | CortexServerTool,
    Field(discriminator="kind"),
]


class CortexToolChoice(BaseModel):
    mode: Literal["auto", "any", "tool", "none"] = "auto"
    name: str | None = None  # required when mode == "tool"


# ---------- Request ----------


class CortexRequest(BaseModel):
    """Provider-agnostic representation of a chat completion request."""

    model: str
    system: str | None = None
    messages: list[CortexMessage]

    max_tokens: int = 4096
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] = Field(default_factory=list)

    tools: list[CortexToolDef] = Field(default_factory=list)
    tool_choice: CortexToolChoice = Field(default_factory=CortexToolChoice)

    # Free-form bag for provider-specific extras the translator wants to
    # preserve on egress (e.g., Anthropic's `metadata.user_id`, OpenAI's
    # `seed`, `presence_penalty`).
    extras: dict[str, Any] = Field(default_factory=dict)

    # ---------- Cortex-specific request controls (read from headers) ----------
    # The proxy populates these from `X-Cortex-*` headers; downstream code
    # (virtualizer, recall) consults them. The translators IGNORE these on
    # egress — they never leave the proxy.
    cortex_group_id: str | None = None
    cortex_session_id: str | None = None
    cortex_time_anchor_iso: str | None = None
    cortex_disable_virtualize: bool = False
    cortex_disable_ingest: bool = False


# ---------- Streaming chunks ----------
#
# These map onto the union of Anthropic's typed event stream and OpenAI's
# delta stream. The canonical chunk hierarchy is closer to Anthropic's because
# Anthropic is the stricter format — translating Anthropic → OpenAI requires
# information loss (combining content_block events into delta chunks), but
# translating OpenAI → Anthropic requires reconstruction (synthesizing the
# block_start/stop events).


class ChunkMessageStart(BaseModel):
    type: Literal["message_start"] = "message_start"
    message_id: str
    model: str
    role: Literal["assistant"] = "assistant"
    # Optional upstream usage hint (Anthropic emits input_tokens here).
    input_tokens: int | None = None


class ChunkContentBlockStart(BaseModel):
    type: Literal["content_block_start"] = "content_block_start"
    index: int
    # The shell block being opened — text starts with text="", tool_use starts
    # with empty tool_input (filled in via subsequent deltas).
    block: CortexBlock


class ChunkTextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    index: int
    text: str


class ChunkToolUseDelta(BaseModel):
    """A partial-JSON delta for an open tool_use block.

    `partial_input_json` is a string fragment that, when concatenated with all
    previous fragments for the same `index`, forms valid JSON for `tool_input`.
    The accumulation state machine lives in cortex.translate.stream_state.
    """

    type: Literal["tool_use_delta"] = "tool_use_delta"
    index: int
    partial_input_json: str


class ChunkContentBlockStop(BaseModel):
    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class ChunkMessageDelta(BaseModel):
    """Carries terminal metadata (stop_reason + final usage)."""

    type: Literal["message_delta"] = "message_delta"
    stop_reason: str | None = None
    stop_sequence: str | None = None
    output_tokens: int | None = None


class ChunkMessageStop(BaseModel):
    type: Literal["message_stop"] = "message_stop"


class ChunkPing(BaseModel):
    """SSE keepalive. Translators may drop these or re-emit, per provider."""

    type: Literal["ping"] = "ping"


class ChunkError(BaseModel):
    type: Literal["error"] = "error"
    error_type: str
    message: str


class ChunkCortexNotice(BaseModel):
    """Side-channel event the proxy emits (never originates from the upstream).

    Used for contradiction warnings, provenance citations, degradation notices.
    Translators emit these as `event: cortex.notice` on Anthropic-format
    streams and as a synthetic `choices[1]` delta on OpenAI-format streams
    (most clients ignore extra choices). Clients that don't care simply
    discard.
    """

    type: Literal["cortex_notice"] = "cortex_notice"
    notice_kind: Literal["conflict", "provenance", "degradation", "info"]
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)


CortexChunk = Annotated[
    ChunkMessageStart
    | ChunkContentBlockStart
    | ChunkTextDelta
    | ChunkToolUseDelta
    | ChunkContentBlockStop
    | ChunkMessageDelta
    | ChunkMessageStop
    | ChunkPing
    | ChunkError
    | ChunkCortexNotice,
    Field(discriminator="type"),
]
