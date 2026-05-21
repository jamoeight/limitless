"""Phase 0.8 — JSON-judge eval.

Runs the Qwopus3.6-27B + GBNF judge against a hand-labeled set of contradiction
pairs from BEAM. Measures accuracy + p95 latency + JSON validity rate.

Hard gates (all three must PASS):
  - Accuracy ≥ 0.65
  - p95 latency ≤ 15s
  - JSON validity rate = 1.000  (if not, GBNF is broken — fix first)

Eval dataset format (JSONL):
  {"id": "beam_0001",
   "query": "Where does Alice live now?",
   "conflicts": [{"e1_fact_id": "f_001", "e2_fact_id": "f_002", "reason": "..."}],
   "attestations": [...],
   "episodes": ["snippet1", "snippet2"],
   "gold_resolution": "e1_correct"}

Usage:
  python phase0/eval_judge.py [--dataset data/phase0_judge_eval.jsonl] [--out results/phase0_judge.csv]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

from timegraph.llm.judge import JudgeClient
from timegraph.types import ConflictTriple, Resolution


async def run_eval(dataset_path: Path, out_csv: Path) -> int:
    if not dataset_path.exists():
        print(f"[FAIL] dataset not found: {dataset_path}")
        print("       Build it first via Phase 0.6 (Task #18).")
        return 2

    examples = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    print(f"[info] {len(examples)} examples loaded from {dataset_path}")

    client = JudgeClient()
    rows = []
    latencies: list[float] = []
    json_valid_count = 0
    correct = 0

    try:
        for i, ex in enumerate(examples):
            conflicts = [ConflictTriple(**c) for c in ex["conflicts"]]
            gold = Resolution(ex["gold_resolution"])
            try:
                out = await client.judge_conflicts(
                    query=ex["query"],
                    conflicts=conflicts,
                    attestations=ex.get("attestations"),
                    source_episodes_truncated=ex.get("episodes"),
                )
                json_valid_count += 1
                latencies.append(out.latency_ms)
                is_correct = out.resolution == gold
                if is_correct:
                    correct += 1
                rows.append({
                    "id": ex["id"], "pred": out.resolution.value, "gold": gold.value,
                    "correct": int(is_correct), "confidence": out.confidence,
                    "latency_ms": out.latency_ms, "call_count": out.call_count,
                })
                if (i + 1) % 10 == 0:
                    running_acc = correct / (i + 1)
                    print(f"  [{i + 1}/{len(examples)}] running acc={running_acc:.3f}  "
                          f"p50={statistics.median(latencies):.0f}ms")
            except Exception as e:
                print(f"  [err] example {ex['id']}: {e}")
                rows.append({"id": ex["id"], "pred": "ERROR", "gold": gold.value, "correct": 0,
                             "confidence": 0.0, "latency_ms": -1, "call_count": 0, "error": str(e)})
    finally:
        await client.close()

    # Write per-example CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as fh:
        if rows:
            keys = list(rows[0].keys())
            fh.write(",".join(keys) + "\n")
            for r in rows:
                fh.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    # Aggregate
    accuracy = correct / len(examples) if examples else 0.0
    p95 = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else float("inf")
    json_validity = json_valid_count / len(examples) if examples else 0.0

    print()
    print("=" * 60)
    print(f"Accuracy:       {accuracy:.3f}  (gate: ≥0.65)")
    print(f"p95 latency:    {p95:.0f}ms ({p95 / 1000:.1f}s)  (gate: ≤15000ms)")
    print(f"JSON validity:  {json_validity:.3f}  (gate: =1.000 — must be perfect with GBNF)")
    print()
    acc_pass = accuracy >= 0.65
    lat_pass = p95 <= 15000
    json_pass = json_validity == 1.0
    print(f"Gate accuracy:  {'PASS' if acc_pass else 'FAIL'}")
    print(f"Gate latency:   {'PASS' if lat_pass else 'FAIL'}")
    print(f"Gate JSON:      {'PASS' if json_pass else 'FAIL — debug GBNF before any other work'}")
    print("=" * 60)
    print(f"Per-example results: {out_csv}")
    return 0 if (acc_pass and lat_pass and json_pass) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("data/phase0_judge_eval.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/phase0_judge.csv"))
    args = ap.parse_args()
    return asyncio.run(run_eval(args.dataset, args.out))


if __name__ == "__main__":
    sys.exit(main())
