"""Smoke-test the claude_cli judge backend with Haiku.

Run:
    TG_JUDGE_BACKEND=claude_cli TG_JUDGE_CLAUDE_MODEL=haiku \
      .venv/Scripts/python.exe scripts/smoke_judge_haiku.py
"""

from __future__ import annotations

import asyncio
import io
import os
import sys

# Windows console default is cp1252; Haiku's `thinking` field may contain
# unicode arrows/dashes that crash on print. Force UTF-8 stdout.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
elif isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from timegraph.config import get_settings
from timegraph.llm.judge import JudgeClient
from timegraph.types import ConflictTriple


async def main() -> int:
    s = get_settings()
    print(f"backend={s.judge_backend} model={s.judge_claude_model} budget=${s.judge_claude_budget_usd}")

    conflicts = [
        ConflictTriple(
            e1_fact_id="fact_001",
            e2_fact_id="fact_002",
            reason="same subject+predicate (Alice lives_in), different objects (Berlin vs Lisbon)",
        ),
        ConflictTriple(
            e1_fact_id="fact_010",
            e2_fact_id="fact_011",
            reason="same subject+predicate (Alice works_at), different objects (Acme vs Globex)",
        ),
    ]
    episodes = [
        "2024-08-15: Alice posted from Berlin about her new Acme contract.",
        "2026-03-20: Alice updated her LinkedIn — moved to Lisbon, now at Globex.",
    ]
    attestations = [
        {"fact_id": "fact_002", "source": "linkedin", "confidence": 0.95},
        {"fact_id": "fact_011", "source": "linkedin", "confidence": 0.95},
    ]

    judge = JudgeClient()
    try:
        for i, c in enumerate(conflicts, start=1):
            print(f"\n--- conflict {i}: {c.e1_fact_id} vs {c.e2_fact_id} ---")
            out = await judge.judge_conflicts(
                query="where does Alice live and work?",
                conflicts=[c],
                attestations=attestations,
                source_episodes_truncated=episodes,
            )
            print(f"  resolution: {out.resolution.value}")
            print(f"  confidence: {out.confidence:.2f}")
            print(f"  latency_ms: {out.latency_ms:.0f}")
            print(f"  reason:     {out.reason}")
            print(f"  thinking:   {out.thinking[:200]}{'...' if len(out.thinking) > 200 else ''}")
    finally:
        await judge.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
