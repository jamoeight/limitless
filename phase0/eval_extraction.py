"""Phase 0.7 — Fact-extraction eval.

Runs the Qwen3-7B-Instruct extractor against a hand-labeled set of dialog
snippets from LongMemEval. Measures F1 vs ground truth + p95 latency.

Hard gates (must both PASS):
  - F1 ≥ 0.70
  - p95 latency ≤ 8s

Eval dataset format (JSONL, one example per line):
  {"id": "lme_0001",
   "episode": "...dialog text...",
   "event_time": "2026-03-15T14:22:00",
   "session_id": "s_001",
   "source": "user_message",
   "facts": [{"subject": "Alice", "predicate": "lives_in", "object": "Boston"}, ...]}

Usage:
  python phase0/eval_extraction.py [--dataset data/phase0_extraction_eval.jsonl] [--out results/phase0_extraction.csv]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

from timegraph.llm.extractor import ExtractorClient


def fact_tuple(f: dict) -> tuple[str, str, str]:
    return (f["subject"].strip().lower(), f["predicate"].strip().lower(), f["object"].strip().lower())


def f1(pred: set[tuple], gold: set[tuple]) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred:
        return 0.0, 0.0, 0.0
    if not gold:
        return 0.0, 1.0, 0.0
    tp = len(pred & gold)
    prec = tp / len(pred)
    rec = tp / len(gold)
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0, prec, rec


async def run_eval(dataset_path: Path, out_csv: Path) -> int:
    if not dataset_path.exists():
        print(f"[FAIL] dataset not found: {dataset_path}")
        print("       Build it first via Phase 0.5 (Task #17).")
        return 2

    examples = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    print(f"[info] {len(examples)} examples loaded from {dataset_path}")

    client = ExtractorClient()
    rows = []
    latencies = []
    per_ex_f1 = []
    try:
        for i, ex in enumerate(examples):
            try:
                preds, latency_ms = await client.extract_facts(
                    episode_content=ex["episode"],
                    event_time=datetime.fromisoformat(ex["event_time"]),
                    session_id=ex.get("session_id", "phase0"),
                    source=ex.get("source", "eval"),
                )
                pred_set = {fact_tuple(f.model_dump()) for f in preds}
                gold_set = {fact_tuple(f) for f in ex["facts"]}
                f1_, p, r = f1(pred_set, gold_set)
                rows.append({
                    "id": ex["id"], "f1": f1_, "precision": p, "recall": r,
                    "latency_ms": latency_ms, "n_pred": len(pred_set), "n_gold": len(gold_set),
                })
                latencies.append(latency_ms)
                per_ex_f1.append(f1_)
                if (i + 1) % 20 == 0:
                    print(f"  [{i + 1}/{len(examples)}] running mean F1={statistics.mean(per_ex_f1):.3f}")
            except Exception as e:
                print(f"  [err] example {ex['id']}: {e}")
                rows.append({"id": ex["id"], "f1": 0.0, "precision": 0.0, "recall": 0.0,
                             "latency_ms": -1, "n_pred": 0, "n_gold": len(ex["facts"]), "error": str(e)})
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
    mean_f1 = statistics.mean(per_ex_f1) if per_ex_f1 else 0.0
    p95 = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else float("inf")

    print()
    print("=" * 60)
    print(f"Mean F1:       {mean_f1:.3f}  (gate: ≥0.70)")
    print(f"p95 latency:   {p95:.0f}ms ({p95 / 1000:.1f}s)  (gate: ≤8000ms)")
    print()
    f1_pass = mean_f1 >= 0.70
    lat_pass = p95 <= 8000
    print(f"Gate F1:       {'PASS' if f1_pass else 'FAIL'}")
    print(f"Gate latency:  {'PASS' if lat_pass else 'FAIL'}")
    print("=" * 60)
    print(f"Per-example results: {out_csv}")
    return 0 if (f1_pass and lat_pass) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("data/phase0_extraction_eval.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/phase0_extraction.csv"))
    args = ap.parse_args()
    return asyncio.run(run_eval(args.dataset, args.out))


if __name__ == "__main__":
    sys.exit(main())
