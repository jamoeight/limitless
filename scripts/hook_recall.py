"""Claude Code UserPromptSubmit hook — injects timegraph recall as additionalContext.

Wiring (.claude/settings.json):

  {
    "hooks": {
      "UserPromptSubmit": [{
        "hooks": [{
          "type": "command",
          "command": ".venv\\\\Scripts\\\\python.exe scripts/hook_recall.py",
          "timeout": 15
        }]
      }]
    }
  }

Reads JSON payload from stdin, runs a semantic recall against the local
timegraph (Neo4j + Qdrant), and emits hookSpecificOutput.additionalContext
on stdout so Claude Code injects the matching memory into the prompt.

Fails open: if anything goes wrong (backends down, empty results, decode
errors), exits 0 with no stdout so Claude Code keeps working.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEFAULT_GROUP_ID = os.environ.get("TG_HOOK_GROUP_ID", "claude-code")
RECALL_K = int(os.environ.get("TG_HOOK_RECALL_K", "6"))
RECALL_BUDGET_TOKENS = int(os.environ.get("TG_HOOK_RECALL_BUDGET", "1500"))
MIN_PROMPT_CHARS = int(os.environ.get("TG_HOOK_MIN_PROMPT_CHARS", "8"))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--group-id", default=DEFAULT_GROUP_ID)
    args, _unknown = p.parse_known_args(argv)
    return args


def _emit(additional_context: str | None) -> None:
    if not additional_context:
        return
    out = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": additional_context,
        }
    }
    sys.stdout.write(json.dumps(out))
    sys.stdout.flush()


def _attr(item, key, default=None):
    if hasattr(item, key):
        return getattr(item, key)
    if isinstance(item, dict):
        return item.get(key, default)
    return default


async def _recall(prompt: str, group_id: str) -> str | None:
    from timegraph.ops.graph_query import graph_query
    from timegraph.types import GraphQueryIn

    payload = GraphQueryIn(
        query=prompt,
        scope=group_id,
        mode="nodes",
        k=RECALL_K,
        budget_tokens=RECALL_BUDGET_TOKENS,
    )
    result = await graph_query(payload)
    facts = getattr(result, "results", None) or []
    if not facts:
        return None

    lines: list[str] = ["## Relevant memory from prior sessions"]
    for f in facts:
        subj = _attr(f, "subject") or ""
        pred = _attr(f, "predicate") or ""
        obj = _attr(f, "object") or ""
        if not (subj and pred and obj):
            continue
        valid_at = _attr(f, "valid_at")
        conf = _attr(f, "confidence")
        tier = _attr(f, "tier") or ""
        meta_bits = []
        if valid_at:
            meta_bits.append(str(valid_at)[:19])
        if tier:
            meta_bits.append(tier)
        if conf is not None:
            meta_bits.append(f"c={conf:.2f}" if isinstance(conf, float) else f"c={conf}")
        meta = f"  _({', '.join(meta_bits)})_" if meta_bits else ""
        lines.append(f"- **{subj}** {pred} **{obj}**{meta}")
    if len(lines) == 1:
        return None
    return "\n".join(lines)


def main() -> None:
    args = _parse_args(sys.argv[1:])
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        return

    prompt = (payload.get("prompt") or "").strip()
    if len(prompt) < MIN_PROMPT_CHARS:
        return

    try:
        ctx = asyncio.run(_recall(prompt, args.group_id))
    except Exception:
        return

    _emit(ctx)


if __name__ == "__main__":
    main()
