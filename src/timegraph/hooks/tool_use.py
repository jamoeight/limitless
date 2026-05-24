"""Claude Code PostToolUse hook — ingests every tool result as an episode.

This is the load-bearing piece of the infinite-context plugin: when Claude
reads a file in turn 3, Bash-runs a test in turn 7, or greps the codebase
in turn 12, each result is written to the timegraph with provenance. After
Claude Code auto-compacts (typically around turn 50–100), `recall` on the
next prompt surfaces these episodes directly — so the model can reference
"the auth middleware you read earlier" without re-reading the file.

Tool-shape mapping (sets the `source` field for downstream queryability):
  Read / Edit / Write / NotebookEdit  -> `file:<path>`     content = file body
  Bash / PowerShell                    -> `bash:<cwd-hash>` content = cmd + stdout
  Grep / Glob                          -> `search:<tool>`   content = pattern + hits
  any other tool                       -> `tool:<name>`     content = input + result

Ingest skips:
  - any `mcp__*` tool (avoids ingest loop on our own MCP server)
  - `TodoWrite`, `TaskCreate`, `Task*` (high-volume, low-information for recall)
  - errored tool calls (no useful payload, stale state)
  - empty or tiny results (< MIN_RESULT_CHARS)

Extraction (LLM fact extraction) is skipped for tool results — `asserted_facts=[]`
forces add_episode to embed and store the raw content without spending an LLM
call per tool. Vector recall handles "find me what I read earlier"; the fact
graph is reserved for user/assistant *statements* from the Stop hook.

Fails open: any exception exits 0 silently.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

# Route structlog + stdlib logging to stderr BEFORE any timegraph.ops import,
# so library log lines don't land on stdout and break the JSON channel.
from timegraph.hooks._log import silence_to_stderr  # noqa: E402

silence_to_stderr()

# Default to the LM-Studio-free path for plugin installs. Must run before any
# timegraph.ops import — those instantiate Settings which reads env once.
os.environ.setdefault("TG_JUDGE_BACKEND", "claude_cli")
os.environ.setdefault("TG_JUDGE_CLAUDE_MODEL", "haiku")
os.environ.setdefault("TG_EXTRACTOR_BACKEND", "claude_cli")
os.environ.setdefault("TG_EXTRACTOR_CLAUDE_MODEL", "haiku")

from timegraph.project_id import derive_group_id  # noqa: E402

MAX_RESULT_CHARS = int(os.environ.get("TG_HOOK_MAX_TOOL_CHARS", "12000"))
MIN_RESULT_CHARS = int(os.environ.get("TG_HOOK_MIN_TOOL_CHARS", "24"))

SKIP_TOOLS = {
    "TodoWrite",
    "TaskCreate",
    "TaskUpdate",
    "TaskList",
    "TaskGet",
    "TaskOutput",
    "TaskStop",
    "ExitPlanMode",
    "EnterPlanMode",
    "EnterWorktree",
    "ExitWorktree",
    "ToolSearch",
    "ScheduleWakeup",
}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--group-id", default=None)
    args, _unknown = p.parse_known_args(argv)
    return args


def _truncate(s: str, limit: int = MAX_RESULT_CHARS) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n[…truncated {len(s) - limit} chars]"


def _stringify_result(result) -> str:
    """PostToolUse `tool_response` can be a string, a dict, or an Anthropic
    content-block list. Reduce it to plain text."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        # Common shapes: {"stdout": ..., "stderr": ...} or {"output": ...} or
        # {"content": [...]}.
        if "stdout" in result or "stderr" in result:
            parts: list[str] = []
            if result.get("stdout"):
                parts.append(str(result["stdout"]))
            if result.get("stderr"):
                parts.append(f"[stderr]\n{result['stderr']}")
            return "\n".join(parts)
        if "output" in result and isinstance(result["output"], str):
            return result["output"]
        if "content" in result:
            return _stringify_result(result["content"])
        try:
            return json.dumps(result, ensure_ascii=False)[:MAX_RESULT_CHARS]
        except Exception:
            return str(result)
    if isinstance(result, list):
        out: list[str] = []
        for block in result:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    out.append(str(block["text"]))
                else:
                    out.append(_stringify_result(block))
            else:
                out.append(str(block))
        return "\n".join(out)
    return str(result)


def _classify(tool_name: str, tool_input: dict) -> tuple[str, str]:
    """Return (source, header) — source is the episode's provenance string,
    header is a single line prepended to the content for human/LLM readability
    in recall snippets."""
    name = tool_name or "unknown"
    inp = tool_input or {}

    if name in ("Read", "NotebookRead"):
        path = inp.get("file_path") or inp.get("notebook_path") or ""
        return f"file:{path}" if path else f"tool:{name}", f"# {name} {path}".rstrip()

    if name in ("Edit", "Write", "NotebookEdit"):
        path = inp.get("file_path") or inp.get("notebook_path") or ""
        return f"file:{path}" if path else f"tool:{name}", f"# {name} {path}".rstrip()

    if name in ("Bash", "PowerShell"):
        cmd = inp.get("command", "")
        desc = inp.get("description", "")
        h = hashlib.blake2b(cmd.encode("utf-8", errors="replace"), digest_size=4).hexdigest()
        head = f"# {name} {desc}" if desc else f"# {name}"
        return f"bash:{h}", f"{head}\n$ {cmd}"

    if name in ("Grep", "Glob"):
        pattern = inp.get("pattern", "")
        path = inp.get("path", "")
        return f"search:{name}", f"# {name} {pattern}" + (f"  in {path}" if path else "")

    if name == "WebFetch":
        url = inp.get("url", "")
        return f"web:{url}" if url else f"tool:{name}", f"# WebFetch {url}".rstrip()

    if name == "WebSearch":
        q = inp.get("query", "")
        return f"search:web", f"# WebSearch {q}".rstrip()

    # Fallback for everything else
    return f"tool:{name}", f"# {name}"


def _should_skip(tool_name: str) -> bool:
    if not tool_name:
        return True
    if tool_name in SKIP_TOOLS:
        return True
    if tool_name.startswith("mcp__"):
        return True
    return False


def _is_errored(payload: dict) -> bool:
    """Detect tool errors. PostToolUse fires for both success and failure;
    we don't want to memorize stale/failed state."""
    resp = payload.get("tool_response") or payload.get("tool_result") or {}
    if isinstance(resp, dict):
        if resp.get("is_error") is True:
            return True
        if resp.get("error"):
            return True
        if resp.get("interrupted") is True:
            return True
    return False


async def _ingest(content: str, source: str, session_id: str, group_id: str) -> None:
    from timegraph.ops.add_episode import add_episode
    from timegraph.types import AddEpisodeIn

    payload = AddEpisodeIn(
        content=content,
        source=source,
        group_id=group_id,
        session_id=session_id,
        event_time=datetime.now(tz=timezone.utc),
        asserted_facts=[],  # no LLM extraction — embed-only, fast
    )
    await add_episode(payload)


def main() -> None:
    args = _parse_args(sys.argv[1:])
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        return

    tool_name = payload.get("tool_name") or ""
    if _should_skip(tool_name):
        return
    if _is_errored(payload):
        return

    tool_input = payload.get("tool_input") or {}
    tool_response = payload.get("tool_response")
    if tool_response is None:
        tool_response = payload.get("tool_result")

    result_text = _stringify_result(tool_response).strip()
    if len(result_text) < MIN_RESULT_CHARS:
        return

    session_id = payload.get("session_id") or "claude-code"
    cwd = payload.get("cwd")
    group_id = args.group_id or derive_group_id(cwd)

    source, header = _classify(tool_name, tool_input)
    body = f"{header}\n\n{_truncate(result_text)}"

    try:
        asyncio.run(_ingest(body, source, session_id, group_id))
    except Exception:
        return


if __name__ == "__main__":
    main()
