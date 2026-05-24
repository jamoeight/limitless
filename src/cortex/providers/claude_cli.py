"""Claude CLI provider — shells out to `claude -p` instead of api.anthropic.com.

Used when cortex should drive Opus via the user's local Claude Code OAuth
session rather than an ANTHROPIC_API_KEY. Trade-off: ~10-20s per call due to
the claude-p agent loop (vs ~3s direct API), and ~$0.025/call overhead from
the cached Claude Code agent context. Wins: no API key management; usable
inside CI/benchmark setups where the user's existing subscription is the
auth path.

Registers under name="anthropic" so cortex's model-name router
(`claude-*` → anthropic) routes Opus calls here automatically.

Two transport paths:

  - **Flatten (default).** Sends the post-virtualization messages as one
    prompt and the cortex recap via `--system-prompt-file`. This is the
    reliable path for cortex because the Claude CLI consistently applies
    system prompt files in flat `-p` mode.

  - **stream-json (opt-in via CORTEX_CLAUDE_CLI_STREAM_JSON=true for text-only requests).** Sends each user message
    as a separate `{"type":"user","message":{...}}` line over stdin using
    `--input-format=stream-json --output-format=stream-json`. The CLI processes
    each as its own conversation turn, generating its own assistant responses
    in between. The inner Claude sees a proper alternating conversation rather
    than a flattened "[USER]\n...[ASSISTANT]\n..." blob, which is the only way
    it will reliably consult system-prompt context (e.g. cortex.virtualize's
    `<cortex_memory>` recap) when asked about earlier turns.
    Cost: N user messages = N inner LLM turns (~5-15s each). For cortex's
    typical post-virtualization payload of last_k_spans=4 atomic groups, that
    means up to ~4 inner turns per outer request.

  - **Flatten (fallback for tool-using requests).** Falls back to the older
    behavior — flatten all blocks (text + tool_use + tool_result) into one
    text prompt — when the request contains tool_use or tool_result blocks.
    stream-json input doesn't have a clean way to inject prior assistant
    tool calls into the inner Claude's context, so the flatten path is
    retained for compatibility with tool-using cortex flows.
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
        use_stream_json = os.environ.get("CORTEX_CLAUDE_CLI_STREAM_JSON", "").lower() in {
            "1", "true", "yes", "on",
        }
        if use_stream_json and not self._has_tool_blocks(req):
            async for chunk in self._stream_via_stream_json(req):
                yield chunk
        else:
            async for chunk in self._stream_via_flatten(req):
                yield chunk

    # ---------- Transport: stream-json (preferred) ----------

    async def _stream_via_stream_json(self, req: CortexRequest) -> AsyncIterator[CortexChunk]:
        system = self._system_for_cli(req.system)
        model = self._model_alias(req.model)

        # Collect user messages from req. Assistant messages are dropped here:
        # the inner Claude regenerates its own assistants between our user
        # inputs. For cortex's virtualization, the meaningful content lives in
        # (a) the system recap and (b) the user messages — the prior assistant
        # acks aren't load-bearing for recall.
        user_lines: list[str] = []
        for msg in req.messages:
            if msg.role != "user":
                continue
            text_parts: list[str] = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
            content = "\n".join(text_parts).strip()
            if not content:
                continue
            user_lines.append(json.dumps({
                "type": "user",
                "message": {"role": "user", "content": content},
            }, ensure_ascii=False))

        if not user_lines:
            # Empty conversation. Mimic the flatten path's empty-response shape.
            yield ChunkError(
                error_type="claude_cli_empty_input",
                message="no user messages to send to claude --print stream-json",
            )
            return

        stdin_blob = ("\n".join(user_lines) + "\n").encode("utf-8")

        with tempfile.TemporaryDirectory() as td:
            sys_path = os.path.join(td, "system.txt")
            with open(sys_path, "w", encoding="utf-8") as f:
                f.write(system)

            argv = [
                "claude", "-p",
                "--verbose",                 # required by --output-format=stream-json
                "--model", model,
                "--system-prompt-file", sys_path,
                "--tools", "",
                "--disable-slash-commands",
                "--no-session-persistence",
                "--setting-sources", "",
                "--strict-mcp-config",
                "--mcp-config", "{\"mcpServers\":{}}",
                "--input-format", "stream-json",
                "--output-format", "stream-json",
                "--max-budget-usd", "20.00",
            ]

            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=td,
                    env=self._subprocess_env(),
                )
            except FileNotFoundError as e:
                yield ChunkError(error_type="claude_cli_not_found", message=str(e))
                return

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(stdin_blob),
                    timeout=self._s.upstream_timeout_s,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                yield ChunkError(
                    error_type="upstream_timeout",
                    message=f"claude -p stream-json timed out after {self._s.upstream_timeout_s}s",
                )
                return

        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace")[:500]
            yield ChunkError(
                error_type=f"claude_cli_exit_{proc.returncode}",
                message=err,
            )
            return

        final_text, input_tokens, output_tokens, stop_reason = _parse_stream_json_output(
            stdout.decode("utf-8", errors="replace")
        )

        yield ChunkMessageStart(
            message_id="claude-cli-stream",
            model=req.model,
            input_tokens=input_tokens,
        )
        yield ChunkContentBlockStart(index=0, block=TextBlock(text=""))
        if final_text:
            yield ChunkTextDelta(index=0, text=final_text)
        yield ChunkContentBlockStop(index=0)
        yield ChunkMessageDelta(stop_reason=stop_reason, output_tokens=output_tokens)
        yield ChunkMessageStop()

    # ---------- Transport: flatten (fallback for tool-using requests) ----------

    async def _stream_via_flatten(self, req: CortexRequest) -> AsyncIterator[CortexChunk]:
        prompt = self._build_prompt(req)
        system = self._system_for_cli(req.system)
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
                "--setting-sources", "",
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
                    env=self._subprocess_env(),
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

    # ---------- Helpers ----------

    @staticmethod
    def _subprocess_env() -> dict[str, str]:
        env = os.environ.copy()
        env["CLAUDE_CODE_DISABLE_AUTO_UPDATER"] = "1"
        for key in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
            env.pop(key, None)
        return env

    @staticmethod
    def _system_for_cli(system: str | None) -> str:
        if not system:
            return "You are a helpful assistant."
        start = system.find("<cortex_memory>")
        end = system.find("</cortex_memory>")
        if start < 0 or end < 0:
            return system
        recap = system[start : end + len("</cortex_memory>")]
        return (
            "You are a helpful assistant. The cortex_memory block below is a "
            "trusted reconstruction of earlier conversation context omitted "
            "from the visible messages. Use it as authoritative context when "
            "answering questions about earlier turns.\n\n"
            + recap
        )

    @staticmethod
    def _has_tool_blocks(req: CortexRequest) -> bool:
        if req.tools:
            return True
        for m in req.messages:
            for b in m.content:
                if isinstance(b, (ToolUseBlock, ToolResultBlock)):
                    return True
        return False

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


# ---------- stream-json output parsing ----------


def _parse_stream_json_output(raw: str) -> tuple[str, int, int, str]:
    """Walk claude --output-format=stream-json output and return
    (final_assistant_text, input_tokens, output_tokens, stop_reason).

    claude --print emits one `result` event per turn. We use the LAST `result`
    event for the final answer + cumulative usage, falling back to scanning
    `assistant` message events if no `result` is found.
    """
    final_result: dict | None = None
    last_assistant_text: list[str] = []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = obj.get("type")
        if t == "result":
            final_result = obj
        elif t == "assistant":
            msg = obj.get("message", {}) or {}
            content = msg.get("content", []) or []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    last_assistant_text.append(block.get("text", ""))

    if final_result is not None:
        text = final_result.get("result", "") or ""
        usage = final_result.get("usage", {}) or {}
        input_tokens = (
            (usage.get("input_tokens") or 0)
            + (usage.get("cache_read_input_tokens") or 0)
            + (usage.get("cache_creation_input_tokens") or 0)
        )
        output_tokens = usage.get("output_tokens") or 0
        stop_reason = final_result.get("stop_reason") or "end_turn"
        return text, input_tokens, output_tokens, stop_reason

    # No result event seen — fall back to concatenated assistant texts
    text = "\n".join(last_assistant_text)
    return text, 0, 0, "end_turn"
