"""Claude Code SessionStart hook — primes the new session with cross-session memory.

Fires on `startup`, `resume`, `clear`, and `compact`. Each source has a tailored
inject:

  startup / resume  → "from your prior work in this directory" primer.
                       Top facts for this group_id, semantic search anchored on
                       the cwd basename so the most project-relevant memory floats up.

  compact           → "recovered from compaction" recap. After Claude Code
                       summarizes context, this re-injects the top facts that
                       were almost certainly in the summarized region — making
                       compaction *lossless* from the user's perspective.

  clear             → no inject. The user explicitly cleared; respect that.

Output: writes a `hookSpecificOutput.additionalContext` JSON document to stdout,
which Claude Code merges into the next prompt. Fails open: on any error or empty
recall, exits 0 with no output and the session proceeds unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Default to the LM-Studio-free path for plugin installs. Must run before any
# timegraph.ops import — those instantiate Settings which reads env once.
os.environ.setdefault("TG_JUDGE_BACKEND", "claude_cli")
os.environ.setdefault("TG_JUDGE_CLAUDE_MODEL", "haiku")

from timegraph.project_id import derive_group_id  # noqa: E402

PRIMER_K = int(os.environ.get("TG_HOOK_PRIMER_K", "8"))
PRIMER_BUDGET_TOKENS = int(os.environ.get("TG_HOOK_PRIMER_BUDGET", "1500"))
COMPACT_K = int(os.environ.get("TG_HOOK_COMPACT_K", "12"))
COMPACT_BUDGET_TOKENS = int(os.environ.get("TG_HOOK_COMPACT_BUDGET", "2500"))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--group-id", default=None)
    args, _unknown = p.parse_known_args(argv)
    return args


def _attr(item, key, default=None):
    if hasattr(item, key):
        return getattr(item, key)
    if isinstance(item, dict):
        return item.get(key, default)
    return default


def _emit(additional_context: str | None) -> None:
    if not additional_context:
        return
    out = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": additional_context,
        }
    }
    sys.stdout.write(json.dumps(out))
    sys.stdout.flush()


async def _recall(query: str, group_id: str, k: int, budget: int) -> list:
    from timegraph.ops.graph_query import graph_query
    from timegraph.types import GraphQueryIn

    payload = GraphQueryIn(
        query=query,
        scope=group_id,
        mode="nodes",
        k=k,
        budget_tokens=budget,
    )
    result = await graph_query(payload)
    return getattr(result, "results", None) or []


def _format_facts(facts: list, heading: str) -> str | None:
    lines: list[str] = [f"## {heading}"]
    for f in facts:
        subj = _attr(f, "subject") or ""
        pred = _attr(f, "predicate") or ""
        obj = _attr(f, "object") or ""
        if not (subj and pred and obj):
            continue
        valid_at = _attr(f, "valid_at")
        meta = f"  _({str(valid_at)[:10]})_" if valid_at else ""
        lines.append(f"- **{subj}** {pred} **{obj}**{meta}")
    if len(lines) == 1:
        return None
    return "\n".join(lines)


def _query_for_cwd(cwd: str | None) -> str:
    if not cwd:
        return "project status"
    name = Path(cwd).name or "project"
    return f"{name} project state and recent work"


def main() -> None:
    args = _parse_args(sys.argv[1:])
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        return

    source = (payload.get("source") or "startup").lower()
    if source == "clear":
        return

    cwd = payload.get("cwd")
    group_id = args.group_id or derive_group_id(cwd)

    query = _query_for_cwd(cwd)
    if source == "compact":
        k, budget = COMPACT_K, COMPACT_BUDGET_TOKENS
        heading = "Memory recovered after compaction (timegraph)"
    elif source == "resume":
        k, budget = PRIMER_K, PRIMER_BUDGET_TOKENS
        heading = "Resuming — relevant memory from prior work in this directory"
    else:  # startup
        k, budget = PRIMER_K, PRIMER_BUDGET_TOKENS
        heading = "Memory from prior sessions in this directory"

    try:
        facts = asyncio.run(_recall(query, group_id, k, budget))
    except Exception:
        return

    ctx = _format_facts(facts, heading)
    _emit(ctx)


if __name__ == "__main__":
    main()
