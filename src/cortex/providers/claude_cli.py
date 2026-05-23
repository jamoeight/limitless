"""Claude CLI provider — shells out to `claude -p` instead of api.anthropic.com.

Used when cortex should drive Opus via the user's local Claude Code OAuth
session rather than an ANTHROPIC_API_KEY. Trade-off: ~10-20s per call due to
the claude-p agent loop (vs ~3s direct API), and ~$0.025/call overhead from
the cached Claude Code agent context. Wins: no API key management; usable
inside CI/benchmark setups where the user's existing subscription is the
auth path.

Registers under name="anthropic" so cortex's model-name router
(`claude-*` → anthropic) routes Opus calls here automatically.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import AsyncIterator

import structlog

from cortex.canonical import (
    ChunkContentBlockStart,
    ChunkContentBlockStop,
    ChunkError,
    ChunkMessageDelta,
    ChunkMessageStart,
    ChunkMessageStop,
    ChunkTextDelta,
    CortexChunk,
    CortexRequest,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from cortex.config import CortexSettings, get_cortex_settings

log = structlog.get_logger(__name__)


class ClaudeCliProvider:
    """Provider that spawns `claude -p` per request. Yields canonical chunks
    once the subprocess completes (not real-time streaming)."""

    name = "anthropic"

    def __init__(self, settings: CortexSettings | None = None) -> None:
        self._s = settings or get_cortex_settings()

    async def stream(
        self,
        req: CortexRequest,
        api_key: str,
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncIterator[CortexChunk]:
        prompt = self._build_prompt(req)
        system = req.system or "You are a helpful assistant."
        model = self._model_alias(req.model)

        # Windows argv has a ~32K char limit; cortex-injected recaps can blow
        # past that. Write the (potentially huge) system prompt to a tmpfile
        # and use --system-prompt-file. User prompt stays on stdin (no limit).
        with tempfile.TemporaryDirectory() as td:
            sys_path = os.path.join(td, "system.txt")
            with open(sys_path, "w", encoding="utf-8") as f:
                f.write(system)

            argv = [
                "claude",
                "-p",
                "--model",
                model,
                "--system-prompt-file",
                sys_path,
                "--tools",
                "",
                "--disable-slash-commands",
                "--no-session-persistence",
                "--output-format",
                "json",
                "--max-budget-usd",
                "20.00",
            ]

            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=td,
                    env={**os.environ, "CLAUDE_CODE_DISABLE_AUTO_UPDATER": "1"},
                )
            except FileNotFoundError as e:
                yield ChunkError(error_type="claude_cli_not_found", message=str(e))
                return

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(prompt.encode("utf-8")),
                    timeout=self._s.upstream_timeout_s,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                yield ChunkError(
                    error_type="upstream_timeout",
                    message=f"claude -p timed out after {self._s.upstream_timeout_s}s",
                )
                return

        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace")[:500]
            yield ChunkError(
                error_type=f"claude_cli_exit_{proc.returncode}",
                message=err,
            )
            return

        try:
            envelope = json.loads(stdout.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as e:
            yield ChunkError(
                error_type="claude_cli_non_json",
                message=f"non-JSON stdout: {e}",
            )
            return

        if envelope.get("is_error"):
            yield ChunkError(
                error_type=envelope.get("subtype", "claude_cli_error"),
                message=str(envelope.get("result") or envelope.get("errors") or "claude -p reported is_error"),
            )
            return

        text = envelope.get("result", "") or ""
        usage = envelope.get("usage", {}) or {}
        input_tokens = (
            (usage.get("input_tokens") or 0)
            + (usage.get("cache_read_input_tokens") or 0)
            + (usage.get("cache_creation_input_tokens") or 0)
        )
        output_tokens = usage.get("output_tokens") or 0

        yield ChunkMessageStart(
            message_id=envelope.get("session_id") or "claude-cli",
            model=req.model,
            input_tokens=input_tokens,
        )
        yield ChunkContentBlockStart(index=0, block=TextBlock(text=""))
        if text:
            yield ChunkTextDelta(index=0, text=text)
        yield ChunkContentBlockStop(index=0)
        yield ChunkMessageDelta(
            stop_reason=envelope.get("stop_reason") or "end_turn",
            output_tokens=output_tokens,
        )
        yield ChunkMessageStop()

    @staticmethod
    def _model_alias(model: str) -> str:
        m = model.lower()
        if m in ("opus", "sonnet", "haiku"):
            return m
        if "opus" in m:
            return "opus"
        if "sonnet" in m:
            return "sonnet"
        if "haiku" in m:
            return "haiku"
        return model

    @staticmethod
    def _build_prompt(req: CortexRequest) -> str:
        # Flatten messages into [ROLE]\n<content> blocks separated by blank lines.
        # cortex.virtualize has already injected the recap into req.system, so
        # the messages list here is the post-virtualization (compressed) form.
        parts: list[str] = []
        for msg in req.messages:
            tag = f"[{msg.role.upper()}]"
            for block in msg.content:
                if isinstance(block, TextBlock):
                    parts.append(f"{tag}\n{block.text}")
                elif isinstance(block, ToolUseBlock):
                    parts.append(
                        f"{tag}\n[tool_use {block.tool_name} id={block.tool_use_id}]\n"
                        f"{json.dumps(block.tool_input)}"
                    )
                elif isinstance(block, ToolResultBlock):
                    content = block.content
                    if isinstance(content, list):
                        content = "\n".join(getattr(b, "text", "") for b in content)
                    parts.append(f"{tag}\n[tool_result {block.tool_use_id}]\n{content}")
                else:
                    parts.append(f"{tag}\n[unsupported block: {getattr(block, 'type', '?')}]")
        return "\n\n".join(parts)

    async def aclose(self) -> None:
        return None
