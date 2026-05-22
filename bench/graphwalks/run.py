"""GraphWalks benchmark runner.

For each sampled task we run three modes against the same row:
  - baseline    : qwen/qwen3.5-9b sees the full prompt (graph + question)
  - ours-judge  : qwen/qwen3.5-9b sees only the question, emits a structured
                  op via strict JSON schema; Cypher executes it
  - ours-regex  : pure regex parser → Cypher (sanity / best-case latency)

Output: results/graphwalks.json with per-row + per-bucket aggregates,
plus a printed summary table.

Usage:
  PYTHONPATH=. .venv/Scripts/python bench/graphwalks/run.py \
      --per-bucket 20 --baseline-max-chars 130000 \
      --out results/graphwalks.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

from datasets import load_dataset  # type: ignore

from bench.graphwalks.baseline import BaselineRunner
from bench.graphwalks.loader import GwTask, exact_match, f1, parse_row
from bench.graphwalks.query_parser import JudgeParser, RegexParser
from bench.graphwalks.storage import (
    cypher_bfs_frontier,
    cypher_parents,
    delete_graph,
    ensure_schema,
    load_graph,
)
from timegraph.storage.neo4j_client import close_driver


HF_CACHE = Path("data/graphwalks/_hf_cache")
_OP_HEADER_RE = re.compile(r"\nOperation:\s*\n", re.IGNORECASE)
# The prompt embeds a 3-shot example block. The REAL operation comes after
# "Here is the graph to operate on:" — searching from before that point
# returns one of the example operations, not the actual question.
_REAL_GRAPH_HEADER_RE = re.compile(
    r"Here is the graph to operate on:", re.IGNORECASE
)


# Char-count buckets. p50 = 110K, p75 = 440K, p90 = 1.75M in the full set.
# We bucket by char count rather than token count because that's what the
# dataset provides directly; rough 4-chars-per-token gives the equivalents.
BUCKETS: list[tuple[str, int, int]] = [
    ("XS",       0,       5_000),    # ~<1.3K tok
    ("S",    5_000,      15_000),    # ~1.3–4K tok
    ("M",   15_000,     130_000),    # ~4–32K tok (Qwen3.5-9B context edge)
    ("L",  130_000,     500_000),    # ~32–125K tok (baseline impossible)
    ("XL", 500_000,  10_000_000),    # ~>125K tok (baseline impossible)
]


def bucket_of(chars: int) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= chars < hi:
            return name
    return "?"


def extract_operation(prompt: str) -> str:
    # Skip past the example block to the real graph header first.
    graph_m = _REAL_GRAPH_HEADER_RE.search(prompt)
    region = prompt[graph_m.end():] if graph_m else prompt
    m = _OP_HEADER_RE.search(region)
    if not m:
        return ""
    after = region[m.end():]
    # Stop before the "You should reason through..." instructions footer.
    cut = after.find("\n\nYou should")
    return (after[:cut] if cut > 0 else after).strip()


@dataclass
class RowResult:
    idx: int
    bucket: str
    prompt_chars: int
    op: str
    start: str
    depth: int | None
    n_nodes: int
    n_edges: int
    gold_size: int

    # ours-regex
    regex_em: bool = False
    regex_f1: float = 0.0
    regex_pred_size: int = 0
    regex_parse_ms: float = 0.0
    regex_load_ms: float = 0.0
    regex_cypher_ms: float = 0.0

    # ours-judge
    judge_em: bool = False
    judge_f1: float = 0.0
    judge_pred_size: int = 0
    judge_parse_ms: float = 0.0
    judge_cypher_ms: float = 0.0
    judge_op_match: bool = False  # did the LLM emit the right op + start + depth?

    # baseline
    baseline_status: str = "skipped"
    baseline_em: bool = False
    baseline_f1: float = 0.0
    baseline_pred_size: int = 0
    baseline_ms: float = 0.0
    baseline_prompt_tokens: int | None = None
    baseline_completion_tokens: int | None = None


async def execute_op(op: dict[str, Any], gid: str) -> tuple[set[str], float]:
    t0 = time.perf_counter()
    if op["op"] == "bfs":
        pred = await cypher_bfs_frontier(gid, op["start"], int(op["depth"]))
    elif op["op"] == "parents":
        pred = await cypher_parents(gid, op["start"])
    else:
        raise ValueError(f"unknown op: {op['op']}")
    return pred, (time.perf_counter() - t0) * 1000


def stratified_sample(ds: Any, per_bucket: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    by_bucket: dict[str, list[dict]] = {b: [] for b, _, _ in BUCKETS}
    for i, r in enumerate(ds):
        b = bucket_of(int(r["prompt_chars"]))
        if b in by_bucket:
            by_bucket[b].append({"idx": i, **r})
    out: list[dict] = []
    for b in by_bucket:
        rows = by_bucket[b]
        rng.shuffle(rows)
        # balance bfs/parents within bucket
        bfs = [r for r in rows if r["problem_type"] == "bfs"][: per_bucket // 2]
        par = [r for r in rows if r["problem_type"] == "parents"][: per_bucket - len(bfs)]
        out.extend(bfs + par)
        print(f"  bucket {b:>2s}: total={len(rows):4d}  sampled={len(bfs)+len(par)}  "
              f"(bfs={len(bfs)} parents={len(par)})", flush=True)
    return out


async def run_one(
    row: dict,
    judge: JudgeParser,
    baseline: BaselineRunner,
    *,
    run_baseline: bool,
) -> RowResult:
    task: GwTask = parse_row(row)
    op_text = extract_operation(task.raw_prompt)
    gid = f"gw_run_{row['idx']}"
    res = RowResult(
        idx=row["idx"],
        bucket=bucket_of(task.prompt_chars),
        prompt_chars=task.prompt_chars,
        op=task.op,
        start=task.start_node,
        depth=task.depth,
        n_nodes=len(task.nodes),
        n_edges=len({(a, b) for a, b in task.edges}),
        gold_size=len(task.answer),
    )

    # --- regex parse + Cypher
    op_regex, t_parse_regex = RegexParser.parse(op_text)
    res.regex_parse_ms = t_parse_regex
    t0 = time.perf_counter()
    n_nodes, n_edges = await load_graph(gid, task.edges)
    res.regex_load_ms = (time.perf_counter() - t0) * 1000
    try:
        pred_regex, t_cypher_regex = await execute_op(op_regex, gid)
        res.regex_cypher_ms = t_cypher_regex
        res.regex_pred_size = len(pred_regex)
        res.regex_em = exact_match(pred_regex, task.answer)
        _, _, res.regex_f1 = f1(pred_regex, task.answer)

        # --- judge parse + Cypher (LLM call). Graph is already loaded.
        try:
            op_judge, t_parse_judge = await judge.parse(op_text)
            res.judge_parse_ms = t_parse_judge
            res.judge_op_match = (
                op_judge.get("op") == op_regex["op"]
                and op_judge.get("start") == op_regex["start"]
                and (op_judge.get("depth") or None) == (op_regex["depth"] or None)
            )
            pred_judge, t_cypher_judge = await execute_op(op_judge, gid)
            res.judge_cypher_ms = t_cypher_judge
            res.judge_pred_size = len(pred_judge)
            res.judge_em = exact_match(pred_judge, task.answer)
            _, _, res.judge_f1 = f1(pred_judge, task.answer)
        except Exception as e:  # noqa: BLE001
            res.judge_op_match = False
            res.judge_em = False
            res.judge_f1 = 0.0
            print(f"    judge parse error on row={row['idx']}: {e!r}", flush=True)

        # --- baseline (full prompt in-context)
        if run_baseline:
            br = await baseline.run(task.raw_prompt)
            res.baseline_status = br.status
            res.baseline_ms = br.latency_ms
            res.baseline_pred_size = len(br.answer)
            # Strict EM: only credit if the baseline actually emitted a final
            # answer. Otherwise the empty-set-trivially-matches-empty-set case
            # would silently inflate scores.
            if br.status == "ok":
                res.baseline_em = exact_match(br.answer, task.answer)
                _, _, res.baseline_f1 = f1(br.answer, task.answer)
            else:
                res.baseline_em = False
                res.baseline_f1 = 0.0
                # Snapshot a fail trace for offline diagnosis.
                _log_baseline_fail(row["idx"], task, br)
            res.baseline_prompt_tokens = br.prompt_tokens
            res.baseline_completion_tokens = br.completion_tokens
        else:
            res.baseline_status = "skipped_too_long"
    finally:
        await delete_graph(gid)

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
        scored_base = [r for r in rs if r.baseline_status not in ("skipped_too_long",)]
        base_runnable = [r for r in scored_base if r.baseline_status == "ok"]
        out["buckets"][b] = {
            "n": len(rs),
            "ours_regex_em": sum(r.regex_em for r in rs) / len(rs),
            "ours_regex_f1": sum(r.regex_f1 for r in rs) / len(rs),
            "ours_judge_em": sum(r.judge_em for r in rs) / len(rs),
            "ours_judge_f1": sum(r.judge_f1 for r in rs) / len(rs),
            "judge_op_match_rate": sum(r.judge_op_match for r in rs) / len(rs),
            "baseline_status_counts": _counts(r.baseline_status for r in rs),
            "baseline_em_over_runnable": (
                sum(r.baseline_em for r in base_runnable) / len(base_runnable)
                if base_runnable else None
            ),
            "baseline_f1_over_runnable": (
                sum(r.baseline_f1 for r in base_runnable) / len(base_runnable)
                if base_runnable else None
            ),
            "baseline_em_over_all": sum(r.baseline_em for r in rs) / len(rs),
            "ours_regex_total_ms_p50": pct(
                [r.regex_parse_ms + r.regex_load_ms + r.regex_cypher_ms for r in rs], 0.5
            ),
            "ours_judge_total_ms_p50": pct(
                [r.regex_load_ms + r.judge_parse_ms + r.judge_cypher_ms for r in rs], 0.5
            ),
            "baseline_ms_p50_over_runnable": (
                pct([r.baseline_ms for r in base_runnable], 0.5) if base_runnable else None
            ),
        }
    return out


def _counts(it: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


def print_table(summary: dict[str, Any]) -> None:
    print()
    print("=" * 110)
    print(f"{'bucket':<6s} {'n':>4s} {'ours_regex':>11s} {'ours_judge':>11s} "
          f"{'baseline*':>11s} {'b_overall':>10s} {'regex_p50':>10s} "
          f"{'judge_p50':>10s} {'base_p50':>10s}")
    print("=" * 110)
    for b, s in summary["buckets"].items():
        base_run = s["baseline_em_over_runnable"]
        base_all = s["baseline_em_over_all"]
        base_p50 = s["baseline_ms_p50_over_runnable"]
        print(
            f"{b:<6s} {s['n']:>4d} "
            f"{s['ours_regex_em']:>10.1%}  "
            f"{s['ours_judge_em']:>10.1%}  "
            f"{(f'{base_run:.1%}' if base_run is not None else '   n/a'):>11s} "
            f"{base_all:>9.1%}  "
            f"{s['ours_regex_total_ms_p50']:>8.0f}ms "
            f"{s['ours_judge_total_ms_p50']:>8.0f}ms "
            f"{(f'{base_p50:.0f}ms' if base_p50 is not None else '   n/a'):>10s}"
        )
    print()
    print("* baseline EM is over runnable rows only (excludes context_overflow / skipped).")
    print("  b_overall counts overflow / skip as wrong, which is the honest comparison.")
    print()


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-bucket", type=int, default=20)
    parser.add_argument("--baseline-max-chars", type=int, default=130_000,
                        help="Skip baseline above this; Qwen3.5-9B context is 32K tokens ≈ 130K chars.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/graphwalks.json")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after N rows total (for smoke). 0 = no limit.")
    args = parser.parse_args()

    print("loading dataset...", flush=True)
    ds = load_dataset("openai/graphwalks", split="train", cache_dir=str(HF_CACHE))
    print(f"  total rows: {len(ds)}")
    print(f"sampling per bucket = {args.per_bucket}...", flush=True)
    sample = stratified_sample(ds, args.per_bucket, seed=args.seed)
    if args.limit:
        sample = sample[: args.limit]
    print(f"  total sampled: {len(sample)}")
    print()

    await ensure_schema()
    judge = JudgeParser()
    baseline = BaselineRunner()

    rows: list[RowResult] = []
    try:
        for i, row in enumerate(sample):
            run_base = int(row["prompt_chars"]) <= args.baseline_max_chars
            t0 = time.perf_counter()
            res = await run_one(row, judge, baseline, run_baseline=run_base)
            rows.append(res)
            dt = time.perf_counter() - t0
            base_em = "—" if res.baseline_status == "skipped_too_long" else (
                "✓" if res.baseline_em else "✗"
            )
            print(
                f"[{i+1:3d}/{len(sample)}] row={res.idx:4d} bkt={res.bucket} "
                f"chars={res.prompt_chars:>7,d} op={res.op:7s} "
                f"|V|={res.n_nodes:5d} |E|={res.n_edges:5d} "
                f"regex={'✓' if res.regex_em else '✗'} "
                f"judge={'✓' if res.judge_em else '✗'} "
                f"base={base_em} "
                f"({res.baseline_status:<18s}) "
                f"wall={dt:5.1f}s",
                flush=True,
            )
            # Periodic write so we don't lose data if interrupted.
            if (i + 1) % 5 == 0:
                _flush(args.out, rows)
    finally:
        await judge.close()
        await baseline.close()
        await close_driver()

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


def _log_baseline_fail(idx: int, task: GwTask, br: Any) -> None:
    """Snapshot the raw model response for any non-ok baseline run, so we can
    debug format_fail vs. context_overflow vs. wrong-answer offline."""
    out_dir = Path("results/_gw_baseline_fails")
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "idx": idx,
        "op": task.op,
        "start": task.start_node,
        "depth": task.depth,
        "prompt_chars": task.prompt_chars,
        "gold": sorted(task.answer),
        "status": br.status,
        "latency_ms": br.latency_ms,
        "completion_tokens": br.completion_tokens,
        "prompt_tokens": br.prompt_tokens,
        "response_text": (br.response_text or "")[:4000],
        "error": br.error,
    }
    (out_dir / f"row_{idx:05d}_{br.status}.json").write_text(
        json.dumps(snapshot, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
