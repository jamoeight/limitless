"""Pilot: 9B raw vs 9B+cortex vs Claude Opus on MRCR rows that exceed 9B's
native context. Question: does cortex virtualization make a local 9B match
a frontier model's long-context behavior?

Three arms per row:
  A. RAW_9B      → POST /v1/chat/completions on LM Studio :1234 (qwen3.5-9b)
  B. CORTEX_9B   → POST /v1/chat/completions on cortex :8080 (qwen3.5-9b via proxy
                   with virtualization enabled)
  C. OPUS        → `claude -p --tools "" --system-prompt <neutral> --model opus`
                   with the conversation flattened to a single prompt

Scoring: official MRCR rubric (SequenceMatcher.ratio with random-string-prefix
check, see bench/mrcr/loader.py::score).

Usage:
  PYTHONPATH=. .venv/Scripts/python bench/pilot_cortex/run.py --rows 5
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

# Make sibling bench imports work when run via `python bench/pilot_cortex/run.py`.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import httpx  # noqa: E402
import pandas as pd  # type: ignore  # noqa: E402
from huggingface_hub import hf_hub_download  # type: ignore  # noqa: E402

from bench.mrcr.loader import MrcrTask, parse_row, score  # noqa: E402


HF_CACHE = REPO / "data" / "mrcr" / "_hf_cache"

# Stratified buckets (matches bench/mrcr/run.py).
BUCKETS: list[tuple[str, int, int]] = [
    ("XS",       0,       100_000),
    ("S",  100_000,       200_000),
    ("M",  200_000,       500_000),
    ("L",  500_000,     1_500_000),
    ("XL",1_500_000, 10_000_000),
]

SHARDS = [
    "2needle/2needle_0.parquet",
    "4needle/4needle_0.parquet",
    "8needle/8needle_0.parquet",
]


def bucket_of(n_chars: int) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= n_chars < hi:
            return name
    return "?"


@dataclass
class ArmResult:
    status: str            # "ok" | "context_overflow" | "http_error" | "timeout" | "skipped"
    score: float = 0.0              # strict MRCR rubric (random_string MUST lead)
    score_lenient: float = 0.0      # MRCR rubric after stripping leading whitespace
    latency_ms: float = 0.0
    response: str = ""              # full response text (used for scoring)
    response_head: str = ""         # 200-char preview for log output
    response_len: int = 0
    error: str = ""
    headers: dict[str, str] | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class RowResult:
    src_shard: str
    row_idx: int
    bucket: str
    n_chars: int
    n_needles: int
    position: int
    total_messages: int

    raw_9b: ArmResult | None = None
    cortex_9b: ArmResult | None = None
    opus: ArmResult | None = None


def load_dataset() -> pd.DataFrame:
    frames = []
    for s in SHARDS:
        fp = hf_hub_download(
            repo_id="openai/mrcr", filename=s, repo_type="dataset",
            cache_dir=str(HF_CACHE),
        )
        df = pd.read_parquet(fp)
        df["src_shard"] = s
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def pick_rows(df: pd.DataFrame, total: int, seed: int) -> list[dict]:
    """Pick a small stratified set across S/M/L (the buckets that stress
    long-context handling without taking forever)."""
    rng = random.Random(seed)
    by_bucket: dict[str, list[int]] = {b: [] for b, _, _ in BUCKETS}
    for i, n in enumerate(df["n_chars"]):
        by_bucket[bucket_of(int(n))].append(i)

    # Distribution: ~1 S, ~2 M, ~2 L (the most interesting overflow region).
    # For total < 5, prioritize L (hardest cases) then M then S.
    plan = {"S": 1, "M": 2, "L": 2}
    while sum(plan.values()) > total:
        for k in ("S", "M", "L"):
            if plan[k] > 0:
                plan[k] -= 1
                break
    while sum(plan.values()) < total:
        plan["L"] += 1

    out: list[dict] = []
    for b, k in plan.items():
        idxs = by_bucket[b][:]
        rng.shuffle(idxs)
        for i in idxs[:k]:
            r = df.iloc[i].to_dict()
            r["_global_idx"] = i
            out.append(r)
    return out


def pick_rows_per_bucket(df: pd.DataFrame, per_bucket: int, seed: int) -> list[dict]:
    """Pick `per_bucket` rows from each of S/M/L. Balances across needle counts
    (2/4/8) where possible — mirrors bench/mrcr/run.py's stratified sampling."""
    rng = random.Random(seed)
    by_bucket: dict[str, list[int]] = {b: [] for b, _, _ in BUCKETS}
    for i, n in enumerate(df["n_chars"]):
        by_bucket[bucket_of(int(n))].append(i)

    out: list[dict] = []
    for b in ("S", "M", "L"):
        idxs = by_bucket[b][:]
        rng.shuffle(idxs)
        # Balance across needle counts: try ⌈per_bucket/3⌉ of each, then fill.
        per_nn = max(1, (per_bucket + 2) // 3)
        chosen: list[int] = []
        for nn in (2, 4, 8):
            need = [i for i in idxs if int(df.iloc[i]["n_needles"]) == nn]
            chosen.extend(need[:per_nn])
            if len(chosen) >= per_bucket:
                break
        for i in idxs:
            if i not in chosen and len(chosen) < per_bucket:
                chosen.append(i)
        for i in chosen[:per_bucket]:
            r = df.iloc[i].to_dict()
            r["_global_idx"] = i
            out.append(r)
    return out


# ---------- Arm A: raw 9B on LM Studio ----------


# Suppresses qwen3.5's tendency to lead responses with "\n\n", which fails the
# strict MRCR prefix check despite often containing correct content. Fair to
# inject — every careful production user of this model adds a similar guard.
_NO_PREAMBLE_SYSTEM = (
    "Output exactly what the user's query asks for. No preamble, no postscript, "
    "no leading whitespace, no leading newlines, no quote marks, no markdown "
    "formatting unless the query asks for it. The first character of your reply "
    "must be the first character of the requested content."
)


def _with_system(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Prepend the no-preamble system message unless one already exists."""
    if messages and messages[0].get("role") == "system":
        return messages
    return [{"role": "system", "content": _NO_PREAMBLE_SYSTEM}, *messages]


async def call_lm_studio(messages: list[dict[str, str]], *, timeout_s: float = 900.0) -> ArmResult:
    body = {
        "model": "qwen/qwen3.5-9b",
        "messages": _with_system(messages),
        "max_tokens": 8192,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        t0 = time.perf_counter()
        try:
            resp = await client.post("http://127.0.0.1:1234/v1/chat/completions", json=body)
        except httpx.HTTPError as e:
            return ArmResult(
                status="http_error",
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=repr(e),
            )
        ms = (time.perf_counter() - t0) * 1000
    if resp.status_code >= 400:
        text = resp.text[:500]
        is_overflow = any(s in text.lower() for s in ("context", "token", "length"))
        return ArmResult(
            status="context_overflow" if is_overflow else "http_error",
            latency_ms=ms, error=text,
        )
    data = resp.json()
    msg = data["choices"][0]["message"]
    out = msg.get("content") or msg.get("reasoning_content") or ""
    usage = data.get("usage", {})
    return ArmResult(
        status="ok",
        latency_ms=ms,
        response=out,
        response_head=out[:200],
        response_len=len(out),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )


# ---------- Arm B: 9B via cortex on :8080 ----------


async def call_cortex(messages: list[dict[str, str]], *, group_id: str, timeout_s: float = 900.0) -> ArmResult:
    body = {
        "model": "qwen/qwen3.5-9b",
        "messages": _with_system(messages),
        "max_tokens": 8192,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    headers = {
        "x-cortex-group-id": group_id,
        "authorization": "Bearer lm-studio-no-key",  # LM Studio doesn't validate
    }
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                "http://127.0.0.1:8080/v1/chat/completions",
                json=body, headers=headers,
            )
        except httpx.HTTPError as e:
            return ArmResult(
                status="http_error",
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=repr(e),
            )
        ms = (time.perf_counter() - t0) * 1000
    cortex_hdrs = {k.lower(): v for k, v in resp.headers.items() if k.lower().startswith("x-cortex-")}
    if resp.status_code >= 400:
        text = resp.text[:500]
        is_overflow = any(s in text.lower() for s in ("context", "token", "length"))
        return ArmResult(
            status="context_overflow" if is_overflow else "http_error",
            latency_ms=ms, error=text, headers=cortex_hdrs,
        )
    data = resp.json()
    msg = data["choices"][0]["message"]
    out = msg.get("content") or msg.get("reasoning_content") or ""
    usage = data.get("usage", {})
    return ArmResult(
        status="ok",
        latency_ms=ms,
        response=out,
        response_head=out[:200],
        response_len=len(out),
        headers=cortex_hdrs,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )


# ---------- Arm C: Claude Opus via `claude -p` ----------


_OPUS_SYSTEM = (
    "You are a literal text retrieval assistant. The user message contains a "
    "transcript of a chat conversation followed by a final query. Identify the "
    "final query and answer it exactly as instructed in the query itself. "
    "Output ONLY what the query asks for; no preamble, no explanation."
)


def flatten_messages_for_opus(messages: list[dict[str, str]]) -> str:
    """Render the MRCR messages list as one flat prompt for Opus. The last
    user message is the actual query; everything before is context."""
    parts: list[str] = [
        "Below is a chat conversation. The LAST user turn is the query you "
        "must answer; everything before is prior context. Follow the query's "
        "instructions exactly.",
        "",
        "=== TRANSCRIPT START ===",
    ]
    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        parts.append(f"[{role}]")
        parts.append(content)
    parts.append("=== TRANSCRIPT END ===")
    parts.append("")
    parts.append("Now respond to the final user query above. Output ONLY what it asks for.")
    return "\n".join(parts)


def call_opus(messages: list[dict[str, str]], *, timeout_s: float = 600.0) -> ArmResult:
    """Invoke `claude -p` in a tmp working dir to avoid CLAUDE.md pollution."""
    prompt = flatten_messages_for_opus(messages)

    with tempfile.TemporaryDirectory() as td:
        cmd = [
            "claude", "-p",
            "--tools", "",
            "--system-prompt", _OPUS_SYSTEM,
            "--model", "opus",
            "--output-format", "text",
            "--no-session-persistence",
            "--setting-sources", "",
        ]
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                encoding="utf-8",
                capture_output=True,
                cwd=td,
                timeout=timeout_s,
                env={**os.environ, "CLAUDE_CODE_DISABLE_AUTO_UPDATER": "1"},
            )
        except subprocess.TimeoutExpired:
            return ArmResult(
                status="timeout",
                latency_ms=(time.perf_counter() - t0) * 1000,
                error="claude -p timed out",
            )
        except Exception as e:  # noqa: BLE001
            return ArmResult(
                status="http_error",
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=repr(e),
            )
        ms = (time.perf_counter() - t0) * 1000

    if proc.returncode != 0:
        err = (proc.stderr or "")[:500]
        is_overflow = any(s in err.lower() for s in ("context", "token", "length", "too large"))
        return ArmResult(
            status="context_overflow" if is_overflow else "http_error",
            latency_ms=ms, error=err,
        )
    out = (proc.stdout or "").strip()
    return ArmResult(
        status="ok",
        latency_ms=ms,
        response=out,
        response_head=out[:200],
        response_len=len(out),
    )


# ---------- per-row driver ----------


async def run_row(task: MrcrTask, row_meta: dict, *, run_raw: bool, run_cortex: bool, run_opus: bool,
                  group_id: str) -> RowResult:
    res = RowResult(
        src_shard=row_meta["src_shard"],
        row_idx=int(row_meta["_global_idx"]),
        bucket=bucket_of(task.n_chars),
        n_chars=task.n_chars,
        n_needles=task.n_needles,
        position=task.position,
        total_messages=task.total_messages,
    )

    def _both(a: ArmResult) -> None:
        if a.status == "ok":
            a.score = score(a.response, task.gold_answer, task.random_string)
            a.score_lenient = score(a.response.lstrip(), task.gold_answer, task.random_string)

    if run_raw:
        a = await call_lm_studio(task.messages)
        _both(a)
        res.raw_9b = a

    if run_cortex:
        b = await call_cortex(task.messages, group_id=group_id)
        _both(b)
        res.cortex_9b = b

    if run_opus:
        c = call_opus(task.messages)
        _both(c)
        res.opus = c

    return res


# ---------- main ----------


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--per-bucket", type=int, default=0,
                        help="If >0, overrides --rows: pick this many rows from each of S/M/L.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/pilot_cortex/pilot.json")
    parser.add_argument("--skip-raw", action="store_true",
                        help="Skip raw 9B arm (it will overflow >100K rows anyway)")
    parser.add_argument("--skip-cortex", action="store_true")
    parser.add_argument("--skip-opus", action="store_true")
    args = parser.parse_args()

    print("loading MRCR shards...", flush=True)
    df = load_dataset()
    print(f"  total rows: {len(df)}")
    if args.per_bucket > 0:
        sample = pick_rows_per_bucket(df, args.per_bucket, args.seed)
        print(f"  per_bucket={args.per_bucket}; picked: {len(sample)}; "
              f"buckets: {[bucket_of(int(r['n_chars'])) for r in sample]}")
    else:
        sample = pick_rows(df, args.rows, args.seed)
        print(f"  picked: {len(sample)}; buckets: "
              f"{[bucket_of(int(r['n_chars'])) for r in sample]}")
    print()

    rows: list[RowResult] = []
    for i, r in enumerate(sample):
        task = parse_row(r)
        gid = f"pilot-r{r['_global_idx']}"
        t0 = time.perf_counter()
        print(
            f"[{i+1}/{len(sample)}] row={r['_global_idx']:>5d} "
            f"bkt={bucket_of(task.n_chars)} chars={task.n_chars:>8,d} needles={task.n_needles} pos={task.position} "
            f"running:",
            flush=True,
        )
        res = await run_row(
            task, r,
            run_raw=not args.skip_raw,
            run_cortex=not args.skip_cortex,
            run_opus=not args.skip_opus,
            group_id=gid,
        )
        rows.append(res)

        def fmt(a: ArmResult | None) -> str:
            if a is None:
                return "skip"
            return f"{a.status[:12]:<12s} score={a.score:.3f} {a.latency_ms/1000:.1f}s"

        print(f"    raw_9b    : {fmt(res.raw_9b)}", flush=True)
        print(f"    cortex_9b : {fmt(res.cortex_9b)}", flush=True)
        if res.cortex_9b and res.cortex_9b.headers:
            print(f"                 hdrs: {res.cortex_9b.headers}", flush=True)
        print(f"    opus      : {fmt(res.opus)}", flush=True)
        print(f"    row wall  : {time.perf_counter() - t0:.1f}s", flush=True)

        _flush(args.out, rows)

    summary = summarize(rows)
    _flush(args.out, rows, summary)
    print_table(summary)
    print(f"\nsaved -> {args.out}")
    return 0


def summarize(rows: list[RowResult]) -> dict[str, Any]:
    out: dict[str, Any] = {"per_bucket": {}, "overall": {}}
    for arm_name in ("raw_9b", "cortex_9b", "opus"):
        scores, scores_l = [], []
        for r in rows:
            a: ArmResult | None = getattr(r, arm_name)
            if a is None:
                continue
            scores.append(a.score)
            scores_l.append(a.score_lenient)
        out["overall"][arm_name] = {
            "n": len(scores),
            "mean_score": (sum(scores) / len(scores)) if scores else None,
            "mean_score_lenient": (sum(scores_l) / len(scores_l)) if scores_l else None,
            "perfect_rate": (sum(s >= 0.99 for s in scores) / len(scores)) if scores else None,
            "perfect_rate_lenient": (sum(s >= 0.99 for s in scores_l) / len(scores_l)) if scores_l else None,
        }
    # Per-bucket breakdown is the load-bearing signal — L rows are where
    # frontier models pull away from raw 9B, and where cortex must hold up.
    for bucket in ("S", "M", "L"):
        bucket_rows = [r for r in rows if r.bucket == bucket]
        if not bucket_rows:
            continue
        out["per_bucket"][bucket] = {}
        for arm_name in ("raw_9b", "cortex_9b", "opus"):
            arms = [getattr(r, arm_name) for r in bucket_rows if getattr(r, arm_name) is not None]
            if not arms:
                continue
            out["per_bucket"][bucket][arm_name] = {
                "n": len(arms),
                "mean_score": sum(a.score for a in arms) / len(arms),
                "mean_score_lenient": sum(a.score_lenient for a in arms) / len(arms),
                "perfect_rate_lenient": sum(a.score_lenient >= 0.99 for a in arms) / len(arms),
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
    print("=" * 90)
    print(f"{'arm':<14s} {'n':>3s}  {'strict':>10s} {'perfect%':>9s}  {'lenient':>10s} {'perfect%':>9s}")
    print("=" * 90)
    for arm, s in summary["overall"].items():
        def fmt(v: float | None, kind: str = "ratio") -> str:
            if v is None:
                return "n/a"
            if kind == "pct":
                return f"{v*100:.0f}%"
            return f"{v:.3f}"
        print(
            f"{arm:<14s} {s['n']:>3d}  "
            f"{fmt(s['mean_score']):>10s} {fmt(s['perfect_rate'], 'pct'):>9s}  "
            f"{fmt(s['mean_score_lenient']):>10s} {fmt(s['perfect_rate_lenient'], 'pct'):>9s}"
        )
    print()
    print("strict  = official MRCR rubric (response MUST start with random_string).")
    print("lenient = strip leading whitespace before applying same rubric.")
    print()

    # Per-bucket headline if present.
    if summary.get("per_bucket"):
        print("-- per-bucket lenient (the long-context test) --")
        print(f"{'bucket':<6s} {'arm':<10s} {'n':>3s}  {'lenient':>10s} {'perfect%':>9s}")
        print("-" * 50)
        for bucket in ("S", "M", "L"):
            bdata = summary["per_bucket"].get(bucket)
            if not bdata:
                continue
            for arm in ("raw_9b", "cortex_9b", "opus"):
                s = bdata.get(arm)
                if not s:
                    continue
                print(
                    f"{bucket:<6s} {arm:<10s} {s['n']:>3d}  "
                    f"{s['mean_score_lenient']:>10.3f} {s['perfect_rate_lenient']*100:>8.0f}%"
                )
            print()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
