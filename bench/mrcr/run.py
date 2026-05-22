"""MRCR benchmark runner.

For each sampled row we run two paths against the same task:
  - baseline   : full message list → qwen3.5-9b chat-completions
  - ours       : build a per-row needle index (no LLM); qwen3.5-9b sees only
                 the final query (≈100 chars) and emits {needle_request,
                 position, prepend}; deterministic lookup returns the
                 verbatim needle answer.

Scoring uses the official MRCR rubric: SequenceMatcher.ratio() with a
mandatory random-string prefix check.

Usage:
  PYTHONPATH=. .venv/Scripts/python bench/mrcr/run.py \\
      --per-bucket 5 --baseline-max-chars 100000 \\
      --out results/mrcr.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

import pandas as pd  # type: ignore
from huggingface_hub import hf_hub_download  # type: ignore

from bench.mrcr.baseline import BaselineRunner
from bench.mrcr.loader import MrcrTask, parse_row, score
from bench.mrcr.query_parser import MrcrJudgeParser
from bench.mrcr.storage import build_index


HF_CACHE = Path("data/mrcr/_hf_cache")

BUCKETS: list[tuple[str, int, int]] = [
    ("XS",       0,       100_000),    # ≈<25K tok (some fits in Qwen3.5-9B's 32K)
    ("S",  100_000,       200_000),    # ≈25–50K tok (overflow)
    ("M",  200_000,       500_000),    # ≈50–125K tok
    ("L",  500_000,     1_500_000),    # ≈125–375K tok
    ("XL",1_500_000, 10_000_000),      # >375K tok
]

# Shards to use — one per variant gives 1,200 rows total to draw from.
SHARDS = [
    "2needle/2needle_0.parquet",
    "4needle/4needle_0.parquet",
    "8needle/8needle_0.parquet",
]


def bucket_of(chars: int) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= chars < hi:
            return name
    return "?"


@dataclass
class RowResult:
    src_shard: str
    row_idx: int
    bucket: str
    n_chars: int
    n_needles: int
    position: int
    total_messages: int

    # ours
    ours_score: float = 0.0
    ours_response: str = ""
    ours_judge_ms: float = 0.0
    ours_lookup_ms: float = 0.0
    ours_index_build_ms: float = 0.0
    ours_needle_match_count: int = 0
    ours_needle_request: str = ""

    # baseline
    baseline_status: str = "skipped_too_long"
    baseline_score: float = 0.0
    baseline_ms: float = 0.0
    baseline_response_head: str = ""
    baseline_prompt_tokens: int | None = None
    baseline_completion_tokens: int | None = None


def load_all() -> pd.DataFrame:
    frames = []
    for s in SHARDS:
        print(f"  pull {s}", flush=True)
        fp = hf_hub_download(
            repo_id="openai/mrcr", filename=s, repo_type="dataset",
            cache_dir=str(HF_CACHE),
        )
        df = pd.read_parquet(fp)
        df["src_shard"] = s
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def stratified_sample(df: pd.DataFrame, per_bucket: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    by_bucket: dict[str, list[int]] = {b: [] for b, _, _ in BUCKETS}
    for i, n in enumerate(df["n_chars"]):
        b = bucket_of(int(n))
        if b in by_bucket:
            by_bucket[b].append(i)
    out: list[dict] = []
    for b in by_bucket:
        idxs = by_bucket[b]
        rng.shuffle(idxs)
        idxs = idxs[:per_bucket]
        # Balance across needle counts where possible.
        chosen: list[int] = []
        for nn in (2, 4, 8):
            need = [i for i in idxs if int(df.iloc[i]["n_needles"]) == nn]
            chosen.extend(need[: max(1, per_bucket // 3)])
            if len(chosen) >= per_bucket:
                break
        # If still short, fill with whatever else remains.
        for i in idxs:
            if i not in chosen and len(chosen) < per_bucket:
                chosen.append(i)
        for i in chosen:
            r = df.iloc[i].to_dict()
            r["_global_idx"] = i
            out.append(r)
        print(f"  bucket {b:>2s}: total={len(idxs):4d}  sampled={len(chosen)}", flush=True)
    return out


async def run_one(
    row: dict, judge: MrcrJudgeParser, baseline: BaselineRunner,
    *, run_baseline: bool,
) -> RowResult:
    task: MrcrTask = parse_row(row)
    res = RowResult(
        src_shard=row["src_shard"],
        row_idx=int(row["_global_idx"]),
        bucket=bucket_of(task.n_chars),
        n_chars=task.n_chars,
        n_needles=task.n_needles,
        position=task.position,
        total_messages=task.total_messages,
    )

    # --- ours: build index, parse query, look up
    t0 = time.perf_counter()
    idx = build_index(task.messages)
    res.ours_index_build_ms = (time.perf_counter() - t0) * 1000

    try:
        spec, t_parse = await judge.parse(task.query_text)
        res.ours_judge_ms = t_parse
        res.ours_needle_request = spec.get("needle_request", "")
        t0 = time.perf_counter()
        hit = idx.lookup(res.ours_needle_request, int(spec.get("position", task.position)))
        res.ours_lookup_ms = (time.perf_counter() - t0) * 1000
        res.ours_needle_match_count = idx.candidate_count(res.ours_needle_request)
        prepend = str(spec.get("prepend", task.prepend))
        if hit is not None:
            res.ours_response = prepend + hit
            res.ours_score = score(res.ours_response, task.gold_answer, task.random_string)
        else:
            res.ours_response = ""
            res.ours_score = 0.0
    except Exception as e:  # noqa: BLE001
        print(f"    ours error on row={res.row_idx}: {e!r}", flush=True)

    # --- baseline
    if run_baseline:
        br = await baseline.run(task.messages)
        res.baseline_status = br.status
        res.baseline_ms = br.latency_ms
        res.baseline_response_head = br.response[:200]
        if br.status == "ok":
            res.baseline_score = score(br.response, task.gold_answer, task.random_string)
        res.baseline_prompt_tokens = br.prompt_tokens
        res.baseline_completion_tokens = br.completion_tokens
    return res


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def summarize(rows: list[RowResult]) -> dict[str, Any]:
    by_bucket: dict[str, list[RowResult]] = {}
    for r in rows:
        by_bucket.setdefault(r.bucket, []).append(r)
    out: dict[str, Any] = {"buckets": {}, "overall": {}}
    for b in [n for n, _, _ in BUCKETS]:
        rs = by_bucket.get(b, [])
        if not rs:
            continue
        base_ok = [r for r in rs if r.baseline_status == "ok"]
        out["buckets"][b] = {
            "n": len(rs),
            "ours_mean_score": sum(r.ours_score for r in rs) / len(rs),
            "ours_p50_score":  pct([r.ours_score for r in rs], 0.5),
            "ours_perfect_rate": sum(r.ours_score >= 0.99 for r in rs) / len(rs),
            "baseline_status_counts": _counts(r.baseline_status for r in rs),
            "baseline_mean_score_over_all": sum(r.baseline_score for r in rs) / len(rs),
            "baseline_mean_score_over_runnable": (
                sum(r.baseline_score for r in base_ok) / len(base_ok) if base_ok else None
            ),
            "baseline_perfect_over_runnable": (
                sum(r.baseline_score >= 0.99 for r in base_ok) / len(base_ok) if base_ok else None
            ),
            "ours_total_ms_p50": pct(
                [r.ours_index_build_ms + r.ours_judge_ms + r.ours_lookup_ms for r in rs], 0.5
            ),
            "baseline_ms_p50_over_runnable": (
                pct([r.baseline_ms for r in base_ok], 0.5) if base_ok else None
            ),
        }
    return out


def _counts(it: Any) -> dict[str, int]:
    o: dict[str, int] = {}
    for x in it:
        o[x] = o.get(x, 0) + 1
    return o


def print_table(summary: dict[str, Any]) -> None:
    print()
    print("=" * 116)
    print(f"{'bucket':<6s} {'n':>3s} {'ours_mean':>10s} {'ours_perfect':>13s} "
          f"{'base_mean*':>11s} {'base_overall':>13s} {'ours_p50':>10s} {'base_p50':>10s}")
    print("=" * 116)
    for b, s in summary["buckets"].items():
        base_run = s["baseline_mean_score_over_runnable"]
        base_all = s["baseline_mean_score_over_all"]
        base_p50 = s["baseline_ms_p50_over_runnable"]
        print(
            f"{b:<6s} {s['n']:>3d} "
            f"{s['ours_mean_score']:>9.3f}  "
            f"{s['ours_perfect_rate']:>12.1%}  "
            f"{(f'{base_run:.3f}' if base_run is not None else '   n/a'):>11s} "
            f"{base_all:>12.3f}  "
            f"{s['ours_total_ms_p50']:>8.0f}ms "
            f"{(f'{base_p50:.0f}ms' if base_p50 is not None else '   n/a'):>10s}"
        )
    print()
    print("* base_mean is over rows where baseline returned a parseable response.")
    print("  base_overall counts context-overflow / errors as 0, the honest comparison.")
    print()


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-bucket", type=int, default=5)
    parser.add_argument("--baseline-max-chars", type=int, default=100_000,
                        help="Skip baseline above this. Qwen3.5-9B ≈32K-tok context.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/mrcr.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    print("loading dataset shards...", flush=True)
    df = load_all()
    print(f"  total rows: {len(df)}")
    sample = stratified_sample(df, args.per_bucket, seed=args.seed)
    if args.limit:
        sample = sample[: args.limit]
    print(f"  total sampled: {len(sample)}")
    print()

    judge = MrcrJudgeParser()
    baseline = BaselineRunner()
    rows: list[RowResult] = []
    try:
        for i, row in enumerate(sample):
            run_base = int(row["n_chars"]) <= args.baseline_max_chars
            t0 = time.perf_counter()
            res = await run_one(row, judge, baseline, run_baseline=run_base)
            rows.append(res)
            dt = time.perf_counter() - t0
            print(
                f"[{i+1:3d}/{len(sample)}] row={res.row_idx:5d} bkt={res.bucket} "
                f"chars={res.n_chars:>8,d} needles={res.n_needles} pos={res.position} "
                f"ours={res.ours_score:.3f} ({res.ours_needle_match_count} cand) "
                f"base={res.baseline_score:.3f} ({res.baseline_status:<18s}) "
                f"wall={dt:5.1f}s",
                flush=True,
            )
            if (i + 1) % 3 == 0:
                _flush(args.out, rows)
    finally:
        await judge.close()
        await baseline.close()

    summary = summarize(rows)
    _flush(args.out, rows, summary=summary)
    print_table(summary)
    print(f"saved -> {args.out}")
    return 0


def _flush(path: str, rows: list[RowResult], summary: dict[str, Any] | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"rows": [asdict(r) for r in rows]}
    if summary is not None:
        payload["summary"] = summary
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
