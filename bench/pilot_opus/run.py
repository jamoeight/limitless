"""Pilot: vanilla Opus 4.7 vs Opus + cortex on MRCR.

Two arms per row:
  A. VANILLA_OPUS  → `claude -p --model opus` on the flattened transcript
                     (matches bench/pilot_cortex/run.py's `opus` arm verbatim)
  B. CORTEX_OPUS   → POST cortex :8080 /v1/messages with model=claude-opus-4-7
                     Cortex auto-ingests + virtualizes (recap injection),
                     then routes to ClaudeCliProvider which shells out to
                     `claude -p --model opus` with the compressed prompt.

Scoring: official MRCR rubric (SequenceMatcher.ratio with random-string-prefix
check, see bench/mrcr/loader.py::score).

The hypothesis: Opus 4.7 collapses on MRCR L-bucket (~50% perfect per
bench/pilot_cortex/PAPER.md). Cortex's recap pre-locates the needles and
injects them as a focused short context — should take Opus from 50% → ~99%
on those rows.

Usage:
  PYTHONPATH=. .venv/Scripts/python bench/pilot_opus/run.py --per-bucket 5
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
import pandas as pd  # type: ignore  # noqa: E402
from huggingface_hub import hf_hub_download  # type: ignore  # noqa: E402

from bench.mrcr.loader import MrcrTask, parse_row, score  # noqa: E402


HF_CACHE = REPO / "data" / "mrcr" / "_hf_cache"

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
    status: str
    score: float = 0.0
    score_lenient: float = 0.0
    latency_ms: float = 0.0
    response: str = ""
    response_head: str = ""
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

    vanilla_opus: ArmResult | None = None
    cortex_opus: ArmResult | None = None


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


def pick_rows_by_target(
    df: pd.DataFrame,
    targets: list[int],
    seed: int,
    n_needles: int = 8,
    tolerance_pct: float = 0.20,
) -> list[dict]:
    """For each target char count, pick a real 8-needle row near it, or
    synthesize one by stitching multiple rows.

    A row is "near target" if its n_chars is within ±tolerance_pct of the target.
    If no real row qualifies, stitch ceil(target / max_avail) rows together to
    reach the target. The first stitched row supplies the gold answer + query;
    other rows contribute distractor messages spliced in before the final query.
    """
    import json as _json
    import math

    rng = random.Random(seed)
    sub = df[df["n_needles"] == n_needles].copy()
    sub["n_chars"] = sub["n_chars"].astype(int)
    max_real = int(sub["n_chars"].max())

    picked: list[dict] = []
    for t in targets:
        # Find rows within tolerance.
        lo = int(t * (1 - tolerance_pct))
        hi = int(t * (1 + tolerance_pct))
        candidates = sub[(sub["n_chars"] >= lo) & (sub["n_chars"] <= hi)]
        if len(candidates) > 0:
            # Pick the one closest to target.
            candidates = candidates.copy()
            candidates["_delta"] = (candidates["n_chars"] - t).abs()
            row = candidates.nsmallest(1, "_delta").iloc[0].to_dict()
            row["_global_idx"] = int(candidates.nsmallest(1, "_delta").index[0])
            row["_target"] = t
            row["_synthetic"] = False
            picked.append(row)
            continue

        # Synthesize by stitching.
        # Pick the biggest available row as base + N distractors to hit target.
        base_idx = int(sub.nlargest(1, "n_chars").index[0])
        base = sub.loc[base_idx].to_dict()
        base["_global_idx"] = base_idx
        base_msgs = _json.loads(base["prompt"])  # type: ignore[arg-type]
        # Distractor pool: all rows minus base, ordered by char count descending
        # so very large targets can draw from the full dataset.
        pool = sub.drop(index=base_idx).sort_values("n_chars", ascending=False)
        needed = max(1, math.ceil((t - base["n_chars"]) / max_real))
        # Cap at the pool size to avoid IndexError; oversample slightly for shuffle headroom.
        candidate_pool = list(pool.index[: min(len(pool), max(needed * 2, 40))])
        rng.shuffle(candidate_pool)
        distractor_idxs = candidate_pool[:needed]

        # Stitch: keep base's last message (the final query) at the end. Insert
        # distractor messages (all but their final query) BEFORE base's final query.
        extras: list[dict] = []
        for di in distractor_idxs:
            drow = sub.loc[di].to_dict()
            dmsgs = _json.loads(drow["prompt"])  # type: ignore[arg-type]
            extras.extend(dmsgs[:-1])

        stitched = base_msgs[:-1] + extras + [base_msgs[-1]]
        stitched_prompt = _json.dumps(stitched, ensure_ascii=False)
        # Build synthetic row.
        syn = dict(base)
        syn["prompt"] = stitched_prompt
        syn["n_chars"] = sum(len(m.get("content", "")) for m in stitched)
        syn["total_messages"] = len(stitched)
        syn["_target"] = t
        syn["_synthetic"] = True
        syn["_synth_components"] = [base_idx] + list(distractor_idxs)
        picked.append(syn)
    return picked


def pick_rows_per_bucket(df: pd.DataFrame, per_bucket: int, seed: int,
                         buckets: tuple[str, ...] = ("S", "M", "L")) -> list[dict]:
    rng = random.Random(seed)
    by_bucket: dict[str, list[int]] = {b: [] for b, _, _ in BUCKETS}
    for i, n in enumerate(df["n_chars"]):
        by_bucket[bucket_of(int(n))].append(i)

    out: list[dict] = []
    for b in buckets:
        idxs = by_bucket[b][:]
        rng.shuffle(idxs)
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


# ---------- Arm A: vanilla Opus via claude -p ----------


_OPUS_SYSTEM = (
    "You are a literal text retrieval assistant. The user message contains a "
    "transcript of a chat conversation followed by a final query. Identify the "
    "final query and answer it exactly as instructed in the query itself. "
    "Output ONLY what the query asks for; no preamble, no explanation."
)


def flatten_messages_for_opus(messages: list[dict[str, str]]) -> str:
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


def call_vanilla_opus(messages: list[dict[str, str]], *, timeout_s: float = 900.0) -> ArmResult:
    prompt = flatten_messages_for_opus(messages)

    with tempfile.TemporaryDirectory() as td:
        cmd = [
            "claude", "-p",
            "--tools", "",
            "--disable-slash-commands",
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


# ---------- Arm B: Opus via cortex /v1/messages ----------


async def call_cortex_opus(messages: list[dict[str, str]], *, group_id: str,
                            timeout_s: float = 1800.0) -> ArmResult:
    # Anthropic-format body. cortex routes claude-* to ClaudeCliProvider when
    # CORTEX_USE_CLAUDE_CLI_PROVIDER=true.
    body: dict[str, Any] = {
        "model": "claude-opus-4-7",
        "max_tokens": 4096,
        "system": _OPUS_SYSTEM,
        "messages": [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages],
    }
    headers = {
        "x-cortex-group-id": group_id,
        "x-api-key": "claude-cli-noauth",  # ClaudeCliProvider ignores it; cortex middleware requires non-empty
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
    # Anthropic response shape: {"content": [{"type":"text", "text":"..."}], "usage": {...}}
    content_blocks = data.get("content", [])
    out_text = ""
    for blk in content_blocks:
        if blk.get("type") == "text":
            out_text += blk.get("text", "")
    usage = data.get("usage", {})
    return ArmResult(
        status="ok",
        latency_ms=ms,
        response=out_text,
        response_head=out_text[:200],
        response_len=len(out_text),
        headers=cortex_hdrs,
        prompt_tokens=usage.get("input_tokens"),
        completion_tokens=usage.get("output_tokens"),
    )


# ---------- per-row driver ----------


async def run_row(task: MrcrTask, row_meta: dict, *, run_vanilla: bool, run_cortex: bool,
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

    if run_vanilla:
        a = call_vanilla_opus(task.messages)
        _both(a)
        res.vanilla_opus = a

    if run_cortex:
        b = await call_cortex_opus(task.messages, group_id=group_id)
        _both(b)
        res.cortex_opus = b

    return res


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-bucket", type=int, default=5)
    parser.add_argument("--buckets", default="S,M,L",
                        help="Comma-separated buckets to sample from (e.g. 'L' or 'S,M,L').")
    parser.add_argument("--targets", default="",
                        help="Comma-separated targets (e.g. '256000,1000000,5000000,10000000'). "
                             "When set, overrides --per-bucket: picks one 8-needle row near each target, "
                             "synthesizing (stitching multiple rows) when no real row exists at that scale.")
    parser.add_argument("--unit", default="chars", choices=["chars", "tokens"],
                        help="Unit for --targets. 'tokens' multiplies each by 4 (char/4 convention) to "
                             "match Anthropic's token-based context measurements (e.g. claude-opus-4-6 "
                             "MRCR v2 1M variant).")
    parser.add_argument("--n-needles", type=int, default=8,
                        help="When --targets is set, restrict to rows with this n_needles.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/opus_vs_cortex/mrcr.json")
    parser.add_argument("--skip-vanilla", action="store_true")
    parser.add_argument("--skip-cortex", action="store_true")
    args = parser.parse_args()

    print("loading MRCR shards...", flush=True)
    df = load_dataset()
    print(f"  total rows: {len(df)}")

    if args.targets:
        raw_targets = [int(t.strip()) for t in args.targets.split(",") if t.strip()]
        # tokens → chars via char/4 convention
        if args.unit == "tokens":
            targets = [t * 4 for t in raw_targets]
            print(f"  unit=tokens; converted to char targets (×4): {targets}")
        else:
            targets = raw_targets
        sample = pick_rows_by_target(df, targets, args.seed, n_needles=args.n_needles)
        print(f"  targets(chars)={targets} n_needles={args.n_needles}; picked: {len(sample)}")
        for r, raw_t in zip(sample, raw_targets):
            tag = "synth" if r.get("_synthetic") else "real"
            comps = r.get("_synth_components", [])
            unit_label = f"{raw_t:,} {args.unit}"
            print(f"    target={unit_label:>20s} got={r['n_chars']:>11,} chars [{tag}]"
                  + (f" components={comps}" if comps else ""))
            # Stash the raw token target on the row for chart labelling later.
            r["_target_tokens"] = raw_t if args.unit == "tokens" else None
    else:
        buckets = tuple(b.strip() for b in args.buckets.split(",") if b.strip())
        sample = pick_rows_per_bucket(df, args.per_bucket, args.seed, buckets=buckets)
        print(f"  per_bucket={args.per_bucket} buckets={buckets}; picked: {len(sample)}; "
              f"buckets: {[bucket_of(int(r['n_chars'])) for r in sample]}")
    print()

    rows: list[RowResult] = []
    for i, r in enumerate(sample):
        task = parse_row(r)
        gid = f"opus-pilot-r{r['_global_idx']}"
        t0 = time.perf_counter()
        print(
            f"[{i+1}/{len(sample)}] row={r['_global_idx']:>5d} "
            f"bkt={bucket_of(task.n_chars)} chars={task.n_chars:>8,d} "
            f"needles={task.n_needles} pos={task.position} running:",
            flush=True,
        )
        res = await run_row(
            task, r,
            run_vanilla=not args.skip_vanilla,
            run_cortex=not args.skip_cortex,
            group_id=gid,
        )
        rows.append(res)

        def fmt(a: ArmResult | None) -> str:
            if a is None:
                return "skip"
            return f"{a.status[:12]:<12s} score={a.score:.3f} {a.latency_ms/1000:.1f}s"

        print(f"    vanilla_opus : {fmt(res.vanilla_opus)}", flush=True)
        print(f"    cortex_opus  : {fmt(res.cortex_opus)}", flush=True)
        if res.cortex_opus and res.cortex_opus.headers:
            print(f"                   hdrs: {res.cortex_opus.headers}", flush=True)
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
    for bucket in ("S", "M", "L"):
        bucket_rows = [r for r in rows if r.bucket == bucket]
        if not bucket_rows:
            continue
        out["per_bucket"][bucket] = {}
        for arm_name in ("vanilla_opus", "cortex_opus"):
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
    if summary.get("per_bucket"):
        print("-- per-bucket lenient (the long-context test) --")
        print(f"{'bucket':<6s} {'arm':<14s} {'n':>3s}  {'lenient':>10s} {'perfect%':>9s}")
        print("-" * 50)
        for bucket in ("S", "M", "L"):
            bdata = summary["per_bucket"].get(bucket)
            if not bdata:
                continue
            for arm in ("vanilla_opus", "cortex_opus"):
                s = bdata.get(arm)
                if not s:
                    continue
                print(
                    f"{bucket:<6s} {arm:<14s} {s['n']:>3d}  "
                    f"{s['mean_score_lenient']:>10.3f} {s['perfect_rate_lenient']*100:>8.0f}%"
                )
            print()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
