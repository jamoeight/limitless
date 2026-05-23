"""GraphWalks pilot: vanilla Opus 4.7 vs Opus 4.7 + cortex.

Two arms per row:
  A. VANILLA_OPUS  → `claude -p --model opus` on the original GraphWalks prompt
                     (3-shot examples + graph + operation + format instruction)
  B. CORTEX_OPUS   → POST cortex :8080 /v1/messages where the graph is split
                     into multi-turn chat (one user/assistant pair per node).
                     Cortex auto-ingests + virtualizes (recap injection),
                     routes to ClaudeCliProvider → claude -p with the compressed prompt.

Scoring: F1 on the answer node set (from bench/graphwalks/loader.py::f1).
Both arms return `Final Answer: [n1, n2, ...]` per the GraphWalks prompt template.

Methodology note: GraphWalks max ≈ 1.75M chars ≈ 360K tokens, which fits Opus
4.7's native 1M-token context. So vanilla doesn't overflow. The experiment tests
whether cortex's cosine-based compression preserves enough graph structure for
BFS / parent-finding.

Usage:
  PYTHONPATH=. .venv/Scripts/python bench/pilot_opus/run_graphwalks.py --per-bucket 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import httpx  # noqa: E402
from datasets import load_dataset  # type: ignore  # noqa: E402

from bench.graphwalks.loader import GwTask, exact_match, f1, parse_row  # noqa: E402
from bench.graphwalks.baseline import parse_final_answer  # noqa: E402


HF_CACHE = REPO / "data" / "graphwalks" / "_hf_cache"

# Match buckets used by bench/graphwalks/run.py.
BUCKETS: list[tuple[str, int, int]] = [
    ("XS",       0,       5_000),
    ("S",    5_000,      15_000),
    ("M",   15_000,     130_000),
    ("L",  130_000,     500_000),
    ("XL", 500_000,  10_000_000),
]


def bucket_of(chars: int) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= chars < hi:
            return name
    return "?"


@dataclass
class ArmResult:
    status: str
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    exact: bool = False
    pred_size: int = 0
    latency_ms: float = 0.0
    response_head: str = ""
    response_len: int = 0
    error: str = ""
    headers: dict[str, str] | None = None


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

    vanilla_opus: ArmResult | None = None
    cortex_opus: ArmResult | None = None


_SYS_PROMPT = (
    "You are a graph reasoning agent. You'll be shown a directed graph as "
    "an edge list, then asked to perform a graph operation (BFS from a "
    "starting node up to a given depth, or find a node's parents). "
    "Reason carefully, then end your reply with exactly one line of the form:\n"
    "Final Answer: [n1, n2, n3]\n"
    "where the list contains all and only the node IDs in your answer set. "
    "If the set is empty, write: Final Answer: []"
)


def _stratified_sample(ds: Any, per_bucket: int, buckets: tuple[str, ...], seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_bucket: dict[str, list[dict]] = {b: [] for b, _, _ in BUCKETS}
    for i, r in enumerate(ds):
        b = bucket_of(int(r["prompt_chars"]))
        if b in by_bucket:
            by_bucket[b].append({"idx": i, **r})
    out: list[dict] = []
    for b in buckets:
        rows = by_bucket.get(b, [])
        rng.shuffle(rows)
        # Balance BFS / parents within the bucket
        bfs = [r for r in rows if r["problem_type"] == "bfs"][: per_bucket // 2 + per_bucket % 2]
        par = [r for r in rows if r["problem_type"] == "parents"][: per_bucket - len(bfs)]
        out.extend(bfs + par)
    return out


# ---------- Arm A: vanilla Opus via claude -p ----------


def call_vanilla_opus(prompt: str, *, timeout_s: float = 900.0) -> ArmResult:
    with tempfile.TemporaryDirectory() as td:
        cmd = [
            "claude", "-p",
            "--tools", "",
            "--disable-slash-commands",
            "--system-prompt", _SYS_PROMPT,
            "--model", "opus",
            "--output-format", "text",
            "--no-session-persistence",
            "--setting-sources", "",
        ]
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd, input=prompt, text=True, encoding="utf-8",
                capture_output=True, cwd=td, timeout=timeout_s,
                env={**os.environ, "CLAUDE_CODE_DISABLE_AUTO_UPDATER": "1"},
            )
        except subprocess.TimeoutExpired:
            return ArmResult(status="timeout", latency_ms=(time.perf_counter() - t0) * 1000,
                             error="claude -p timed out")
        except Exception as e:  # noqa: BLE001
            return ArmResult(status="http_error", latency_ms=(time.perf_counter() - t0) * 1000,
                             error=repr(e))
        ms = (time.perf_counter() - t0) * 1000

    if proc.returncode != 0:
        err = (proc.stderr or "")[:500]
        is_overflow = any(s in err.lower() for s in ("context", "token", "length", "too large"))
        return ArmResult(status="context_overflow" if is_overflow else "http_error",
                         latency_ms=ms, error=err)
    out = (proc.stdout or "").strip()
    return ArmResult(status="ok", latency_ms=ms, response_head=out[:200], response_len=len(out),
                     pred_size=0,  # filled in by scorer
                     headers={"response": out})  # store full response for scoring


# ---------- Arm B: Opus via cortex with split-graph messages ----------


def _format_graph_as_messages(task: GwTask) -> list[dict[str, str]]:
    """Split the graph into multi-turn chat. Each user message lists one
    source node's outgoing edges; assistant ack between them. Final user
    message is the operation question (cortex will recall-rank against it)."""
    by_source: dict[str, list[str]] = {}
    for a, b in task.edges:
        by_source.setdefault(a, []).append(b)

    messages: list[dict[str, str]] = []
    messages.append({
        "role": "user",
        "content": (
            "I'll show you a directed graph one source node at a time. "
            "After each node, just acknowledge so I can continue. "
            "When the graph is complete I'll ask the operation."
        ),
    })
    messages.append({"role": "assistant", "content": "Ready."})

    for src in sorted(by_source):
        dsts = by_source[src]
        if len(dsts) == 1:
            text = f"{src} -> {dsts[0]}"
        else:
            text = f"{src} -> [{', '.join(dsts)}]"
        messages.append({"role": "user", "content": text})
        messages.append({"role": "assistant", "content": "ok"})

    # Operation question
    if task.op == "bfs":
        q = (f"Perform BFS from node {task.start_node} to depth {task.depth}. "
             f"Return all nodes reachable from {task.start_node} within {task.depth} hops "
             f"(not including {task.start_node} itself unless it has a self-loop).")
    else:
        q = (f"Find all parent nodes of node {task.start_node} — i.e., every node X "
             f"such that there is an edge X -> {task.start_node} in the graph.")
    q += "\n\nEnd with exactly: Final Answer: [list of node IDs]"
    messages.append({"role": "user", "content": q})
    return messages


async def call_cortex_opus(task: GwTask, *, group_id: str,
                            timeout_s: float = 1800.0) -> ArmResult:
    messages = _format_graph_as_messages(task)
    body: dict[str, Any] = {
        "model": "claude-opus-4-7",
        "max_tokens": 4096,
        "system": _SYS_PROMPT,
        "messages": messages,
    }
    headers = {
        "x-cortex-group-id": group_id,
        "x-api-key": "claude-cli-noauth",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                "http://127.0.0.1:8080/v1/messages",
                json=body, headers=headers,
            )
        except httpx.HTTPError as e:
            return ArmResult(status="http_error",
                             latency_ms=(time.perf_counter() - t0) * 1000,
                             error=repr(e))
        ms = (time.perf_counter() - t0) * 1000

    cortex_hdrs = {k.lower(): v for k, v in resp.headers.items()
                   if k.lower().startswith("x-cortex-")}
    if resp.status_code >= 400:
        text = resp.text[:500]
        is_overflow = any(s in text.lower() for s in ("context", "token", "length"))
        return ArmResult(status="context_overflow" if is_overflow else "http_error",
                         latency_ms=ms, error=text, headers=cortex_hdrs)

    data = resp.json()
    out_text = ""
    for blk in data.get("content", []):
        if blk.get("type") == "text":
            out_text += blk.get("text", "")
    cortex_hdrs["response"] = out_text
    return ArmResult(status="ok", latency_ms=ms,
                     response_head=out_text[:200], response_len=len(out_text),
                     headers=cortex_hdrs)


def _score(arm: ArmResult, gold: set[str]) -> None:
    if arm.status != "ok":
        return
    hdrs = arm.headers or {}
    text = hdrs.get("response", "")
    pred = parse_final_answer(text)
    arm.pred_size = len(pred)
    p, r, score = f1(pred, gold)
    arm.precision = p
    arm.recall = r
    arm.f1 = score
    arm.exact = exact_match(pred, gold)


async def run_row(task: GwTask, idx: int, *, run_vanilla: bool, run_cortex: bool,
                  group_id: str) -> RowResult:
    res = RowResult(
        idx=idx,
        bucket=bucket_of(task.prompt_chars),
        prompt_chars=task.prompt_chars,
        op=task.op,
        start=task.start_node,
        depth=task.depth,
        n_nodes=len(task.nodes),
        n_edges=len(task.edges),
        gold_size=len(task.answer),
    )

    if run_vanilla:
        a = call_vanilla_opus(task.raw_prompt)
        _score(a, task.answer)
        res.vanilla_opus = a

    if run_cortex:
        b = await call_cortex_opus(task, group_id=group_id)
        _score(b, task.answer)
        res.cortex_opus = b

    return res


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-bucket", type=int, default=2)
    parser.add_argument("--buckets", default="S,M,L",
                        help="Comma-separated buckets to sample from.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/opus_vs_cortex/graphwalks.json")
    parser.add_argument("--skip-vanilla", action="store_true")
    parser.add_argument("--skip-cortex", action="store_true")
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, only run the first N rows of the picked sample.")
    args = parser.parse_args()

    buckets = tuple(b.strip() for b in args.buckets.split(",") if b.strip())

    print("loading GraphWalks dataset...", flush=True)
    ds = load_dataset("openai/graphwalks", cache_dir=str(HF_CACHE), split="train")
    print(f"  total rows: {len(ds)}")
    sample = _stratified_sample(ds, args.per_bucket, buckets, args.seed)
    if args.limit > 0:
        sample = sample[: args.limit]
    print(f"  per_bucket={args.per_bucket} buckets={buckets}; picked: {len(sample)}")
    for r in sample:
        b = bucket_of(int(r["prompt_chars"]))
        print(f"    idx={r['idx']:>4d} bkt={b:>2s} chars={r['prompt_chars']:>8,d} type={r['problem_type']:>7s}")
    print()

    rows: list[RowResult] = []
    for i, r in enumerate(sample):
        task = parse_row(r)
        gid = f"gw-pilot-r{r['idx']}"
        t0 = time.perf_counter()
        print(
            f"[{i+1}/{len(sample)}] idx={r['idx']:>4d} "
            f"bkt={bucket_of(task.prompt_chars):>2s} chars={task.prompt_chars:>8,d} "
            f"op={task.op:>7s} start={task.start_node} depth={task.depth} "
            f"nodes={len(task.nodes)} edges={len(task.edges)} gold={len(task.answer)} running:",
            flush=True,
        )
        res = await run_row(task, r["idx"],
                            run_vanilla=not args.skip_vanilla,
                            run_cortex=not args.skip_cortex,
                            group_id=gid)
        rows.append(res)

        def fmt(a: ArmResult | None) -> str:
            if a is None:
                return "skip"
            return (f"{a.status[:12]:<12s} f1={a.f1:.3f} em={a.exact!s:>5s} "
                    f"pred={a.pred_size:>3d} {a.latency_ms/1000:.1f}s")

        print(f"    vanilla_opus : {fmt(res.vanilla_opus)}", flush=True)
        print(f"    cortex_opus  : {fmt(res.cortex_opus)}", flush=True)
        if res.cortex_opus and res.cortex_opus.headers:
            hdrs_clean = {k: v for k, v in res.cortex_opus.headers.items() if k != "response"}
            if hdrs_clean:
                print(f"                   hdrs: {hdrs_clean}", flush=True)
        print(f"    row wall     : {time.perf_counter() - t0:.1f}s", flush=True)

        _flush(args.out, rows)

    summary = summarize(rows)
    _flush(args.out, rows, summary)
    print_table(summary)
    print(f"\nsaved -> {args.out}")
    return 0


def summarize(rows: list[RowResult]) -> dict[str, Any]:
    out: dict[str, Any] = {"per_bucket": {}, "overall": {}}
    for arm_name in ("vanilla_opus", "cortex_opus"):
        scores, exacts = [], []
        for r in rows:
            a: ArmResult | None = getattr(r, arm_name)
            if a is None or a.status != "ok":
                continue
            scores.append(a.f1)
            exacts.append(1.0 if a.exact else 0.0)
        out["overall"][arm_name] = {
            "n": len(scores),
            "mean_f1": (sum(scores) / len(scores)) if scores else None,
            "exact_rate": (sum(exacts) / len(exacts)) if exacts else None,
        }
    for bucket in ("XS", "S", "M", "L", "XL"):
        bucket_rows = [r for r in rows if r.bucket == bucket]
        if not bucket_rows:
            continue
        out["per_bucket"][bucket] = {}
        for arm_name in ("vanilla_opus", "cortex_opus"):
            arms = [getattr(r, arm_name) for r in bucket_rows
                    if getattr(r, arm_name) is not None and getattr(r, arm_name).status == "ok"]
            if not arms:
                continue
            out["per_bucket"][bucket][arm_name] = {
                "n": len(arms),
                "mean_f1": sum(a.f1 for a in arms) / len(arms),
                "exact_rate": sum(1 for a in arms if a.exact) / len(arms),
            }
    return out


def _flush(path: str, rows: list[RowResult], summary: dict[str, Any] | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"rows": [asdict(r) for r in rows]}
    if summary is not None:
        payload["summary"] = summary
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def print_table(summary: dict[str, Any]) -> None:
    print()
    print("=" * 80)
    print(f"{'arm':<14s} {'n':>3s}  {'mean F1':>10s}  {'exact match':>12s}")
    print("=" * 80)
    for arm, s in summary["overall"].items():
        def fmt(v: float | None, kind: str = "ratio") -> str:
            if v is None:
                return "n/a"
            if kind == "pct":
                return f"{v*100:.0f}%"
            return f"{v:.3f}"
        print(f"{arm:<14s} {s['n']:>3d}  {fmt(s['mean_f1']):>10s}  "
              f"{fmt(s['exact_rate'], 'pct'):>12s}")
    print()
    if summary.get("per_bucket"):
        print("-- per-bucket --")
        print(f"{'bucket':<6s} {'arm':<14s} {'n':>3s}  {'mean F1':>10s}  {'exact':>8s}")
        print("-" * 50)
        for bucket in ("XS", "S", "M", "L", "XL"):
            bdata = summary["per_bucket"].get(bucket)
            if not bdata:
                continue
            for arm in ("vanilla_opus", "cortex_opus"):
                s = bdata.get(arm)
                if not s:
                    continue
                print(f"{bucket:<6s} {arm:<14s} {s['n']:>3d}  "
                      f"{s['mean_f1']:>10.3f}  {s['exact_rate']*100:>7.0f}%")
            print()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
