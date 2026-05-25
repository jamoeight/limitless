"""Auto-ingest: feed conversation turns into the timegraph as episodes.

This is what MVP-2 adds on top of the passthrough spine: every message the
client sends, and every assistant response the upstream model produces, is
content-hashed and (if new) fired into `timegraph.ops.add_episode` as an
async background task. The request path never waits for ingest.

Idempotency is by content hash: an LRU-bounded `Session` cache rejects spans
we've seen before, even across restarts within process lifetime. The
SessionRegistry maps `(group_id, session_id)` → Session so multiple
conversations don't collide.

The default `_default_ingest_fn` calls `timegraph.ops.add_episode.add_episode`
directly (in-process, no MCP/stdio overhead). Tests inject a stub fn so they
don't need Neo4j/Qdrant/LM Studio up.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

from cortex.canonical import (
    ChunkContentBlockStart,
    ChunkContentBlockStop,
    ChunkTextDelta,
    ChunkToolUseDelta,
    CortexBlock,
    CortexChunk,
    CortexMessage,
    ImageBlock,
    OpaqueBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from cortex.config import CortexSettings

log = structlog.get_logger(__name__)


class IngestState(str, Enum):
    IN_FLIGHT = "in_flight"
    DONE = "done"
    FAILED = "failed"


# Crude secret detection. MVP-6 expands this (file-extension allowlists, etc.).
_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"),
    re.compile(r"sk-ant-[A-Za-z0-9_-]{40,}"),
    re.compile(r"sk-[A-Za-z0-9]{40,}"),  # OpenAI-shaped
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),  # GitHub PAT
    re.compile(r"xox[bpoa]-[A-Za-z0-9-]{20,}"),  # Slack token
]


# ---------- Hashing ----------


def span_hash(message: CortexMessage) -> str:
    """Stable SHA256 of a message turn (role + canonicalized content).

    The hash is intended to be deterministic for "the same turn re-sent."
    Two messages with identical role+content+tool_use_ids produce the same
    hash, regardless of dict key ordering.
    """
    canonical = {
        "role": message.role,
        "content": [_canonical_block(b) for b in message.content],
    }
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _canonical_block(b: CortexBlock) -> dict[str, Any]:
    if isinstance(b, TextBlock):
        return {"t": "text", "x": b.text}
    if isinstance(b, ImageBlock):
        return {
            "t": "image",
            "m": b.media_type,
            "d": b.data_b64 or "",
            "u": b.url or "",
        }
    if isinstance(b, ToolUseBlock):
        return {
            "t": "tool_use",
            "id": b.tool_use_id,
            "name": b.tool_name,
            "input": b.tool_input,
        }
    if isinstance(b, ToolResultBlock):
        if isinstance(b.content, list):
            content: Any = [_canonical_block(sub) for sub in b.content]
        else:
            content = b.content
        return {
            "t": "tool_result",
            "id": b.tool_use_id,
            "content": content,
            "err": b.is_error,
        }
    if isinstance(b, OpaqueBlock):
        # Hash needs to differentiate server_tool_use blocks across turns —
        # `id` (if present) gives stable identity, otherwise the full payload.
        return {
            "t": "opaque",
            "orig": b.original_type,
            "payload": b.payload,
        }
    return {"t": "unknown"}


# ---------- Text projection ----------


def message_to_text(message: CortexMessage, *, include_tool_blocks: bool = True) -> str:
    """Flatten a message into a single string for the extractor LLM.

    Multi-block messages (text + tool_use, etc.) are joined with structure
    markers so the extractor can still pull facts from tool calls and results.
    Images become opaque placeholders — they need a vision model to extract
    from, which the LM-Studio extractor isn't.

    When `include_tool_blocks` is False, tool_use and tool_result blocks are
    dropped entirely. The ingest path uses this to honor
    `CortexSettings.enable_tool_aware_ingest`: tool results are typically
    large, low-signal payloads (file contents, search hits) that flood the
    extractor with garbage facts and waste a shared model slot. Callers that
    need accurate token counts for the whole message (e.g. virtualization)
    keep the default.
    """
    parts: list[str] = []
    for b in message.content:
        if isinstance(b, TextBlock):
            parts.append(b.text)
        elif isinstance(b, ImageBlock):
            parts.append(f"[image: {b.media_type}]")
        elif isinstance(b, ToolUseBlock):
            if not include_tool_blocks:
                continue
            try:
                inp = json.dumps(b.tool_input, default=str)
            except (TypeError, ValueError):
                inp = str(b.tool_input)
            parts.append(f"[tool_use {b.tool_name}({inp})]")
        elif isinstance(b, ToolResultBlock):
            if not include_tool_blocks:
                continue
            if isinstance(b.content, str):
                payload = b.content
            else:
                payload = " ".join(_subblock_text(sub) for sub in b.content)
            err = " (error)" if b.is_error else ""
            parts.append(f"[tool_result{err}: {payload}]")
        elif isinstance(b, OpaqueBlock):
            # Opaque blocks (server_tool_use, web_search_tool_result, etc.)
            # have no canonical text. We include a short marker so the
            # extractor sees the turn was non-empty, but skip the payload —
            # tool-result content is captured by tool_use/tool_result blocks
            # elsewhere in the same conversation.
            if include_tool_blocks:
                parts.append(f"[{b.original_type}]")
    return "\n".join(parts)


def _subblock_text(b: CortexBlock) -> str:
    if isinstance(b, TextBlock):
        return b.text
    if isinstance(b, ImageBlock):
        return f"[image:{b.media_type}]"
    return ""


def looks_like_secret(text: str) -> bool:
    return any(p.search(text) for p in _SECRET_PATTERNS)


# ---------- LRU cache ----------


class _LRU:
    """Insertion-ordered LRU. Per-session use only; not thread-safe."""

    def __init__(self, max_entries: int) -> None:
        self._max = max(1, max_entries)
        self._d: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, k: str) -> dict[str, Any] | None:
        if k in self._d:
            self._d.move_to_end(k)
            return self._d[k]
        return None

    def put(self, k: str, v: dict[str, Any]) -> None:
        if k in self._d:
            self._d.move_to_end(k)
        self._d[k] = v
        while len(self._d) > self._max:
            self._d.popitem(last=False)

    def __contains__(self, k: str) -> bool:
        return k in self._d

    def __len__(self) -> int:
        return len(self._d)


# ---------- Session + Registry ----------


# An IngestFn is `async (content, source, group_id, session_id, event_time) -> episode_id`.
# Returning an empty string means the ingest fn declined (e.g., backend down).
IngestFn = Callable[[str, str, str, str, datetime], Awaitable[str]]


async def _default_ingest_fn(
    content: str,
    source: str,
    group_id: str,
    session_id: str,
    event_time: datetime,
) -> str:
    """Real ingest: in-process call into timegraph.ops.add_episode.

    Wrapped in a generic try/except so the proxy's request path is never
    blocked by ingest-pipeline failures (Neo4j down, extractor timeout, etc.).
    The full MVP-5 degrade-and-warn behavior layers on top.
    """
    try:
        # Local import — keeps `cortex` importable even when the timegraph
        # backends aren't installed in this venv (CI / unit tests).
        from timegraph.ops.add_episode import add_episode
        from timegraph.types import AddEpisodeIn

        result = await add_episode(
            AddEpisodeIn(
                content=content,
                source=source,
                group_id=group_id,
                session_id=session_id,
                event_time=event_time,
            )
        )
        return result.episode_id
    except Exception as e:  # noqa: BLE001
        log.warning("ingest.backend_error", source=source, error=str(e))
        return ""


class Session:
    """Per-(group_id, session_id) ingest state."""

    def __init__(
        self,
        group_id: str,
        session_id: str,
        settings: CortexSettings,
        ingest_fn: IngestFn,
    ) -> None:
        self.group_id = group_id
        self.session_id = session_id
        self._s = settings
        self._cache = _LRU(settings.hash_cache_max_entries)
        self._sem = asyncio.Semaphore(max(1, settings.ingest_max_concurrent))
        self._inflight: dict[str, asyncio.Task[None]] = {}
        self._ingest_fn = ingest_fn

    def has_seen(self, h: str) -> bool:
        return h in self._cache

    def state(self, h: str) -> IngestState | None:
        entry = self._cache.get(h)
        return entry.get("state") if entry else None

    def episode_id_for(self, h: str) -> str | None:
        entry = self._cache.get(h)
        return entry.get("episode_id") if entry else None

    def schedule(self, message: CortexMessage) -> str | None:
        """If `message` is new and worth ingesting, fire add_episode in the
        background. Returns the hash if scheduled, None if skipped."""
        h = span_hash(message)
        if h in self._cache:
            return None

        text = message_to_text(
            message, include_tool_blocks=self._s.enable_tool_aware_ingest
        )
        if len(text) < self._s.ingest_min_chars:
            return None
        if looks_like_secret(text):
            log.info("ingest.skip.secret", group=self.group_id, hash=h[:8])
            return None

        source = f"msg:{message.role}:{h[:8]}"
        self._cache.put(h, {"state": IngestState.IN_FLIGHT, "episode_id": None})
        task = asyncio.create_task(self._run(h, text, source))
        self._inflight[h] = task
        return h

    async def _run(self, h: str, text: str, source: str) -> None:
        async with self._sem:
            event_time = datetime.now(timezone.utc)
            try:
                episode_id = await self._ingest_fn(
                    text, source, self.group_id, self.session_id, event_time
                )
                if episode_id:
                    self._cache.put(
                        h, {"state": IngestState.DONE, "episode_id": episode_id}
                    )
                else:
                    self._cache.put(
                        h, {"state": IngestState.FAILED, "episode_id": None}
                    )
            except Exception as e:  # noqa: BLE001
                self._cache.put(
                    h,
                    {"state": IngestState.FAILED, "episode_id": None, "error": str(e)},
                )
                log.warning("ingest.task_error", hash=h[:8], error=str(e))
            finally:
                self._inflight.pop(h, None)

    async def drain(self, timeout: float | None = None) -> None:
        """Wait for all inflight tasks to complete. Tests use this; production
        usually doesn't."""
        if not self._inflight:
            return
        await asyncio.wait(
            list(self._inflight.values()), timeout=timeout, return_when=asyncio.ALL_COMPLETED
        )

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    @property
    def inflight_count(self) -> int:
        return len(self._inflight)


class SessionRegistry:
    """Global registry of Sessions, keyed by (group_id, session_id)."""

    def __init__(
        self,
        settings: CortexSettings,
        ingest_fn: IngestFn | None = None,
    ) -> None:
        self._s = settings
        self._ingest_fn = ingest_fn or _default_ingest_fn
        self._sessions: dict[tuple[str, str], Session] = {}

    def get_or_create(self, group_id: str, session_id: str) -> Session:
        key = (group_id, session_id)
        if key not in self._sessions:
            self._sessions[key] = Session(group_id, session_id, self._s, self._ingest_fn)
        return self._sessions[key]

    async def drain_all(self, timeout: float | None = None) -> None:
        for s in self._sessions.values():
            await s.drain(timeout=timeout)


# ---------- Ingest entry points used by the server ----------


def ingest_request_messages(session: Session, messages: list[CortexMessage]) -> list[str]:
    """Schedule ingest for every new message in an inbound request.

    Returns the hashes actually scheduled (excludes cache hits and skipped).
    """
    scheduled: list[str] = []
    for m in messages:
        h = session.schedule(m)
        if h:
            scheduled.append(h)
    return scheduled


# ---------- Reconstructing the assistant turn from streaming chunks ----------


def assistant_message_from_chunks(chunks: list[CortexChunk]) -> CortexMessage | None:
    """Rebuild a CortexMessage(role="assistant") from a chunk stream.

    Used by the server after the upstream stream completes so we can ingest
    what the model actually said.
    Returns None if the stream produced no content blocks.
    """
    blocks_by_index: dict[int, CortexBlock] = {}
    order: list[int] = []
    partial_tool_json: dict[int, list[str]] = {}

    for chunk in chunks:
        if isinstance(chunk, ChunkContentBlockStart):
            blocks_by_index[chunk.index] = chunk.block
            order.append(chunk.index)
            if isinstance(chunk.block, ToolUseBlock):
                partial_tool_json[chunk.index] = []
        elif isinstance(chunk, ChunkTextDelta):
            b = blocks_by_index.get(chunk.index)
            if isinstance(b, TextBlock):
                blocks_by_index[chunk.index] = TextBlock(text=b.text + chunk.text)
        elif isinstance(chunk, ChunkToolUseDelta):
            partial_tool_json.setdefault(chunk.index, []).append(chunk.partial_input_json)
        elif isinstance(chunk, ChunkContentBlockStop):
            if chunk.index in partial_tool_json:
                raw = "".join(partial_tool_json.pop(chunk.index, []))
                b = blocks_by_index.get(chunk.index)
                if isinstance(b, ToolUseBlock):
                    try:
                        inp = json.loads(raw) if raw.strip() else {}
                    except json.JSONDecodeError:
                        inp = {"_partial_json": raw}
                    blocks_by_index[chunk.index] = ToolUseBlock(
                        tool_use_id=b.tool_use_id,
                        tool_name=b.tool_name,
                        tool_input=inp,
                    )

    ordered: list[CortexBlock] = [blocks_by_index[i] for i in order if i in blocks_by_index]
    if not ordered:
        return None
    return CortexMessage(role="assistant", content=ordered)


def derive_group_id(api_key: str) -> str:
    """Hash the API key into a group_id when the client didn't provide one.

    We hash so the raw key never appears in storage / logs. Same key → same
    group → persistent memory across sessions.
    """
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return f"k:{digest[:24]}"
