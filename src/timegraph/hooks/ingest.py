"""Claude Code Stop hook — high-water-mark transcript ingester.

Wired by the `timegraph-cortex` plugin manifest as a `Stop` hook. After every
assistant turn, walks the session transcript JSONL from the last offset we
ingested to EOF, turning each *new* user prompt and assistant text response
into its own timegraph episode (with fact extraction). Tool calls and tool
results are deliberately skipped — `PostToolUse` ingests those in real time
so recall sees them on the very next prompt without waiting for Stop.

The hook persists its byte-offset cursor per session at
`~/.timegraph/sessions/<session_id>.json` (see hooks/state.py). Offset is
advanced incrementally *after* each successful episode write, so a hook timeout
midway through a backfill leaves the system in a correct state — the next
Stop fire resumes from where we got to, never duplicating.

Fails open: any exception exits 0 silently. A memory hiccup must never
interrupt a Claude Code turn.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Default to the LM-Studio-free path for plugin installs. setdefault preserves
# any user override (e.g. dev with LM Studio loaded). Must run before any
# timegraph.ops import — those instantiate Settings which reads env once.
os.environ.setdefault("TG_JUDGE_BACKEND", "claude_cli")
os.environ.setdefault("TG_JUDGE_CLAUDE_MODEL", "haiku")
os.environ.setdefault("TG_EXTRACTOR_BACKEND", "claude_cli")
os.environ.setdefault("TG_EXTRACTOR_CLAUDE_MODEL", "haiku")

from timegraph.hooks.state import read_offset, write_offset  # noqa: E402
from timegraph.project_id import derive_group_id  # noqa: E402

MAX_TEXT_CHARS = int(os.environ.get("TG_HOOK_MAX_TEXT_CHARS", "12000"))
MIN_TEXT_CHARS = int(os.environ.get("TG_HOOK_MIN_TEXT_CHARS", "16"))
MAX_NEW_MESSAGES = int(os.environ.get("TG_HOOK_MAX_NEW_MESSAGES", "50"))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--group-id", default=None)
    args, _unknown = p.parse_known_args(argv)
    return args


def _is_command_echo(text: str) -> bool:
    t = text.lstrip()
    return (
        t.startswith("<command-name>")
        or t.startswith("<local-command-stdout>")
        or t.startswith("<local-command-caveat>")
        or t.startswith("<system-reminder>")
        or t.startswith("[Request interrupted")
    )


def _extract_text_blocks(content) -> str:
    """Return concatenated text from a message.content list, dropping tool_use /
    tool_result blocks (those are handled by PostToolUse)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            t = block.get("text") or ""
            if t:
                parts.append(t)
    return "\n".join(parts)


def _iter_ingestables(lines: list[str]):
    """Yield (role, text, timestamp) tuples for every new user prompt or
    assistant text message in `lines`. Skips meta entries, command echoes,
    tool plumbing, and pure-tool-call assistant messages."""
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if obj.get("isMeta"):
            continue
        if obj.get("isSidechain"):
            continue
        kind = obj.get("type")
        if kind not in ("user", "assistant"):
            continue
        msg = obj.get("message") or {}
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _extract_text_blocks(msg.get("content")).strip()
        if not text:
            continue
        if _is_command_echo(text):
            continue
        ts = obj.get("timestamp")
        try:
            event_time = (
                datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if isinstance(ts, str)
                else datetime.now(tz=timezone.utc)
            )
        except Exception:
            event_time = datetime.now(tz=timezone.utc)
        yield role, text, event_time


async def _ingest_one(text: str, role: str, session_id: str, group_id: str, event_time: datetime) -> None:
    from timegraph.ops.add_episode import add_episode
    from timegraph.types import AddEpisodeIn

    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + "\n[…truncated]"
    payload = AddEpisodeIn(
        content=text,
        source=f"claude-code:{role}",
        group_id=group_id,
        session_id=session_id,
        event_time=event_time,
    )
    await add_episode(payload)


def _read_new_lines(transcript_path: Path, start_offset: int) -> tuple[list[str], list[int]]:
    """Read JSONL from `start_offset`. Return (lines, per_line_end_offsets) so
    callers can advance the cursor per-line."""
    lines: list[str] = []
    offsets: list[int] = []
    with transcript_path.open("rb") as fh:
        fh.seek(start_offset)
        while True:
            line = fh.readline()
            if not line:
                break
            lines.append(line.decode("utf-8", errors="replace"))
            offsets.append(fh.tell())
    return lines, offsets


async def _run(transcript_path: Path, session_id: str, group_id: str, start_offset: int) -> int:
    lines, offsets = _read_new_lines(transcript_path, start_offset)
    if not lines:
        return start_offset

    ingestables: list[tuple[str, str, datetime, int]] = []
    for (role, text, event_time), line_end in zip(_iter_ingestables(lines), offsets[: len(lines)], strict=False):
        if len(text) < MIN_TEXT_CHARS:
            continue
        ingestables.append((role, text, event_time, line_end))

    # Backfill safety: if we've never ingested before and the transcript is
    # huge, only take the last MAX_NEW_MESSAGES so the first hook fire doesn't
    # hammer the extractor with hundreds of historical messages. Subsequent
    # fires will be small (one turn each).
    if start_offset == 0 and len(ingestables) > MAX_NEW_MESSAGES:
        ingestables = ingestables[-MAX_NEW_MESSAGES:]

    current_offset = start_offset
    for role, text, event_time, line_end in ingestables:
        try:
            await _ingest_one(text, role, session_id, group_id, event_time)
        except Exception:
            # Stop advancing on first failure — we'll retry from here on next
            # Stop fire. Avoids losing content if e.g. Neo4j is briefly down.
            return current_offset
        current_offset = max(current_offset, line_end)
        write_offset(session_id, current_offset)

    # Always advance to EOF even if we filtered everything out — there's no
    # value in re-reading the same skipped lines next fire.
    eof_offset = offsets[-1] if offsets else start_offset
    final = max(current_offset, eof_offset)
    write_offset(session_id, final)
    return final


def main() -> None:
    args = _parse_args(sys.argv[1:])
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        return

    transcript_path_str = payload.get("transcript_path") or ""
    session_id = payload.get("session_id") or "claude-code"
    if not transcript_path_str:
        return

    transcript_path = Path(transcript_path_str)
    if not transcript_path.exists():
        return

    cwd = payload.get("cwd")
    group_id = args.group_id or derive_group_id(cwd)
    start_offset = read_offset(session_id)

    try:
        asyncio.run(_run(transcript_path, session_id, group_id, start_offset))
    except Exception:
        return


if __name__ == "__main__":
    main()
