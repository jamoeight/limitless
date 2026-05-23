"""Claude Code Stop hook — ingests the just-completed turn into the timegraph.

Wired by the `timegraph-cortex` plugin manifest. Reads the JSON payload Claude
Code writes to stdin, walks the transcript JSONL backwards to find the last
user prompt + last assistant response, and stores them as one episode.

Fails open: exits 0 silently on any error so the user's session is never
interrupted by a memory hiccup.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Default to the LM-Studio-free path for plugin installs. setdefault preserves
# any user override (e.g. dev with LM Studio loaded). Must run before any
# timegraph.ops import — those instantiate Settings which reads env once.
os.environ.setdefault("TG_JUDGE_BACKEND", "claude_cli")
os.environ.setdefault("TG_JUDGE_CLAUDE_MODEL", "haiku")
os.environ.setdefault("TG_USE_JUDGE_FOR_EXTRACTION", "true")

from timegraph.project_id import derive_group_id  # noqa: E402

MAX_ASSISTANT_CHARS = int(os.environ.get("TG_HOOK_MAX_ASSISTANT_CHARS", "8000"))
MIN_TOTAL_CHARS = int(os.environ.get("TG_HOOK_MIN_INGEST_CHARS", "32"))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--group-id", default=None)
    args, _unknown = p.parse_known_args(argv)
    return args


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(block.get("text") or "")
            elif btype == "tool_use":
                name = block.get("name") or "tool"
                parts.append(f"[tool_use: {name}]")
            elif btype == "tool_result":
                parts.append("[tool_result]")
        return "\n".join(p for p in parts if p)
    return ""


def _read_last_pair(transcript_path: str) -> tuple[str, str] | None:
    p = Path(transcript_path)
    if not p.exists():
        return None
    last_user: str | None = None
    last_assistant: str | None = None
    with p.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = obj.get("message") or {}
            role = msg.get("role") or obj.get("type")
            content = msg.get("content")
            if content is None:
                continue
            text = _extract_text(content).strip()
            if not text:
                continue
            if role == "user":
                last_user = text
                last_assistant = None
            elif role == "assistant":
                last_assistant = text
    if not last_user or not last_assistant:
        return None
    return last_user, last_assistant


async def _ingest(content: str, session_id: str, group_id: str) -> None:
    from datetime import datetime, timezone

    from timegraph.ops.add_episode import add_episode
    from timegraph.types import AddEpisodeIn

    payload = AddEpisodeIn(
        content=content,
        source="claude-code",
        group_id=group_id,
        session_id=session_id or "claude-code",
        event_time=datetime.now(tz=timezone.utc),
    )
    await add_episode(payload)


def main() -> None:
    args = _parse_args(sys.argv[1:])
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        return

    transcript_path = payload.get("transcript_path") or ""
    session_id = payload.get("session_id") or "claude-code"
    if not transcript_path:
        return

    cwd = payload.get("cwd")
    group_id = args.group_id or derive_group_id(cwd)

    pair = _read_last_pair(transcript_path)
    if pair is None:
        return
    user_text, assistant_text = pair
    if len(assistant_text) > MAX_ASSISTANT_CHARS:
        assistant_text = assistant_text[:MAX_ASSISTANT_CHARS] + "\n[…truncated]"

    combined = f"User: {user_text}\n\nAssistant: {assistant_text}"
    if len(combined) < MIN_TOTAL_CHARS:
        return

    try:
        asyncio.run(_ingest(combined, session_id, group_id))
    except Exception:
        return


if __name__ == "__main__":
    main()
