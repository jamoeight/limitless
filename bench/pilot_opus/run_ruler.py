"""RULER pilot: vanilla Opus 4.7 vs Opus 4.7 + cortex.

Uses self-long/RULER-llama3-1M (pre-generated RULER at 4K → 1M tokens, all
13 subtasks). Defaults to `niah_multikey_3` — multiple key-value pairs hidden
in haystack noise, retrieve the value for a specified key. Most multi-needle
of the NIAH variants, closest analog to MRCR's multi-needle retrieval.

Two arms per row:
  A. VANILLA_OPUS  → `claude -p --model opus` on the stripped RULER prompt
                     (llama3 chat tokens removed, raw user content sent)
  B. CORTEX_OPUS   → POST cortex :8080 /v1/messages where context is split
                     into multi-turn chat by paragraph. Final user message
                     is the question.

Scoring: per-row "any answer found" using RULER's official rubric
(`any(ans in pred_text for ans in answers)`) plus a stricter "all answers
found" metric for multi-key tasks.

Methodology caveat: RULER is natively single-prompt. Splitting context across
turns is a format change. Cortex's measured performance bakes in that change.
We chunk by paragraph and report honestly — if the format change dominates,
that's the finding.

Usage:
  PYTHONPATH=. .venv/Scripts/python bench/pilot_opus/run_ruler.py --scales 64k,128k,256k,512k,1m --rows-per-scale 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
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

# Use default HF cache (~/.cache/huggingface) — RULER lock filenames are long
# and the embedded absolute path blows past Windows MAX_PATH if we use a
# project-local cache.
HF_CACHE = None


# llama3 chat tokens — strip these to get the raw user content for claude -p
_LLAMA3_HEADER_RE = re.compile(
    r"<\|start_header_id\|>(\w+)<\|end_header_id\|>\s*\n\n(.*?)<\|eot_id\|>",
    re.DOTALL,
)


def parse_ruler_input(raw: str) -> tuple[str, str]:
    """Return (system_text, user_text) parsed out of the llama3 chat template.
    Falls back to (empty, raw) if the template doesn't match."""
    parts: dict[str, list[str]] = {"system": [], "user": [], "assistant": []}
    for m in _LLAMA3_HEADER_RE.finditer(raw):
        role, content = m.group(1), m.group(2)
        if role in parts:
            parts[role].append(content)
    if not parts["user"]:
        return "", raw
    return "\n\n".join(parts["system"]), "\n\n".join(parts["user"])


@dataclass
class ArmResult:
    status: str
    any_answer_found: bool = False
    all_answers_found: bool = False
    n_found: int = 0
    n_expected: int = 0
    latency_ms: float = 0.0
    response_head: str = ""
    response_len: int = 0
    error: str = ""
    headers: dict[str, str] | None = None


@dataclass
class RowResult:
    scale: str
    subtask: str
    row_idx: int
    n_chars_user: int
    n_tokens_llama3: int
    n_tokens_o200k: int | None
    n_expected_answers: int

    vanilla_opus: ArmResult | None = None
    cortex_opus: ArmResult | None = None


_SYS_PROMPT = (
    "You are a retrieval assistant. Read the provided context carefully, "
    "then answer the question precisely. Output only the answer itself with "
    "no preamble, no explanation, and no surrounding text."
)


def _score(text: str, answers: list[str]) -> tuple[int, int]:
    text_l = text.lower()
    found = sum(1 for a in answers if a.lower() in text_l)
    return found, len(answers)


# ---------- Arm A: vanilla Opus via claude -p ----------


def call_vanilla_opus(user_text: str, *, timeout_s: float = 900.0) -> ArmResult:
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
                cmd, input=user_text, text=True, encoding="utf-8",
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
                     headers={"response": out})


# ---------- Arm B: Opus via cortex with paragraph-split context ----------


def _split_into_messages(user_text: str, chunk_size: int = 1) -> list[dict[str, str]]:
    """Split RULER's user prompt into multi-turn chat.

    RULER NIAH formats context as single-newline-separated lines. The first
    line is an intro ("A special magic uuid is hidden..."); the last line is
    the question ("What is the magic uuid for X?"); everything in between is
    needle lines. We split per-line — each needle becomes its own user message
    so cortex's cosine recall has fine-grained access. `chunk_size > 1` groups
    multiple consecutive lines per message (reduces total message count).
    """
    lines = [ln.strip() for ln in user_text.split("\n") if ln.strip()]
    if not lines:
        return [{"role": "user", "content": user_text}]

    # Find the question line (last line that ends with '?' or contains 'What').
    q_idx = len(lines) - 1
    for i in range(len(lines) - 1, -1, -1):
        if "?" in lines[i]:
            q_idx = i
            break
    context_lines = lines[:q_idx]
    question = "\n".join(lines[q_idx:])

    # Chunk context lines.
    chunks: list[str] = []
    for i in range(0, len(context_lines), chunk_size):
        chunks.append("\n".join(context_lines[i : i + chunk_size]))

    messages: list[dict[str, str]] = []
    messages.append({
        "role": "user",
        "content": "I'll feed you a passage one chunk at a time. Acknowledge each chunk briefly.",
    })
    messages.append({"role": "assistant", "content": "Ready."})
    for chunk in chunks:
        messages.append({"role": "user", "content": chunk})
        messages.append({"role": "assistant", "content": "ok"})
    messages.append({"role": "user", "content": question})
    return messages


async def call_cortex_opus(user_text: str, *, group_id: str,
                            timeout_s: float = 1800.0) -> ArmResult:
    messages = _split_into_messages(user_text)
    body: dict[str, Any] = {
        "model": "claude-opus-4-7",
        "max_tokens": 1024,
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


# ---------- per-row driver ----------


async def run_row(scale: str, subtask: str, row_idx: int, row: dict, *,
                  run_vanilla: bool, run_cortex: bool, group_id: str) -> RowResult:
    _, user_text = parse_ruler_input(row["input"])
    answers = list(row["answers"])

    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        n_o200k = len(enc.encode(user_text))
    except ImportError:
        n_o200k = None

    res = RowResult(
        scale=scale, subtask=subtask, row_idx=row_idx,
        n_chars_user=len(user_text),
        n_tokens_llama3=int(row.get("length", 0)),
        n_tokens_o200k=n_o200k,
        n_expected_answers=len(answers),
    )

    def _record(a: ArmResult) -> None:
        if a.status != "ok":
            return
        text = (a.headers or {}).get("response", "")
        found, expected = _score(text, answers)
        a.n_found = found
        a.n_expected = expected
        a.any_answer_found = found >= 1
        a.all_answers_found = found == expected

    if run_vanilla:
        a = call_vanilla_opus(user_text)
        _record(a)
        res.vanilla_opus = a
    if run_cortex:
        b = await call_cortex_opus(user_text, group_id=group_id)
        _record(b)
        res.cortex_opus = b

    return res


# ---------- main ----------


def _load_config(scale: str, subtask: str) -> Any:
    # Dataset uses lowercase k for thousands (64k) but capital M for millions (1M).
    s = scale.strip()
    if s.endswith("m"):
        s = s[:-1] + "M"
    config_name = f"{subtask}_{s}"
    return load_dataset(
        "self-long/RULER-llama3-1M", config_name,
        split="validation",
    )


def _split_user_text(user_text: str) -> tuple[str, list[str], str]:
    """Return (intro_line, context_lines, question_line) from a RULER user
    prompt. RULER structure: line 0 is the 'A special magic uuid is hidden...'
    intro; the last line containing '?' is the question; everything in between
    is context (needle + noise lines)."""
    lines = [ln.strip() for ln in user_text.split("\n") if ln.strip()]
    if not lines:
        return "", [], ""
    intro = lines[0]
    q_idx = len(lines) - 1
    for i in range(len(lines) - 1, -1, -1):
        if "?" in lines[i]:
            q_idx = i
            break
    context = lines[1:q_idx]
    question = "\n".join(lines[q_idx:])
    return intro, context, question


def synthesize_row(base_row: dict, distractor_rows: list[dict]) -> dict:
    """Stitch a RULER row to a larger effective context.

    Base provides the intro line + question + gold answers (its needle is
    what we'll evaluate). Distractors contribute only their context lines
    (their intros + questions are stripped). Result preserves the base's
    `answers` so the scorer still works.
    """
    _, base_user = parse_ruler_input(base_row["input"])
    base_intro, base_context, base_question = _split_user_text(base_user)

    extras: list[str] = []
    for d in distractor_rows:
        _, du = parse_ruler_input(d["input"])
        _, d_context, _ = _split_user_text(du)
        extras.extend(d_context)

    synth_user = "\n".join([base_intro] + base_context + extras + [base_question])

    # Rebuild a fake llama3-chat input so parse_ruler_input is idempotent.
    synth_input = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{synth_user}<|eot_id|>"
    )
    return {
        "index": base_row.get("index", 0),
        "input": synth_input,
        "answers": base_row["answers"],
        "length": int(base_row.get("length", 0)) * (1 + len(distractor_rows)),
        "predictions": {},
        "_synthetic": True,
        "_components": [base_row.get("index", 0)] + [d.get("index", 0) for d in distractor_rows],
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", default="niah_multikey_3",
                        help="RULER subtask name (e.g. niah_multikey_3, niah_multiquery, niah_multivalue).")
    parser.add_argument("--scales", default="",
                        help="Comma-separated llama3 token scales from the pre-gen dataset "
                             "(e.g. 64k,128k,256k,512k,1m).")
    parser.add_argument("--synth-scales", default="",
                        help="Comma-separated synthesized scales beyond the dataset's 1M ceiling "
                             "(e.g. 2m,5m,10m). Each is built by stitching N 1M-token rows: base "
                             "provides intro+question+gold, distractors contribute context lines.")
    parser.add_argument("--rows-per-scale", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/opus_vs_cortex/ruler.json")
    parser.add_argument("--skip-vanilla", action="store_true")
    parser.add_argument("--skip-cortex", action="store_true")
    args = parser.parse_args()

    scales = [s.strip() for s in args.scales.split(",") if s.strip()]
    synth_scales = [s.strip() for s in args.synth_scales.split(",") if s.strip()]
    print(f"subtask={args.subtask} scales={scales} synth={synth_scales} "
          f"rows_per_scale={args.rows_per_scale}")

    sample: list[tuple[str, int, dict]] = []  # (scale, idx, row)
    for scale in scales:
        ds = _load_config(scale, args.subtask)
        for i in range(min(args.rows_per_scale, len(ds))):
            sample.append((scale, i, ds[i]))
        print(f"  scale={scale} loaded={len(ds)} picked={min(args.rows_per_scale, len(ds))}")

    if synth_scales:
        # All synth scales draw from the 1M-token config (the dataset's max).
        ds_1m = _load_config("1m", args.subtask)
        for scale in synth_scales:
            # Parse target to a multiplier of 1M. e.g. "5m" → 5 → stitch 5 rows (1 base + 4 distractors).
            s = scale.strip().lower()
            mult = int(float(s.rstrip("m"))) if s.endswith("m") else max(2, int(s) // 1_000_000)
            mult = max(2, mult)  # need at least base + 1 distractor
            for i in range(min(args.rows_per_scale, len(ds_1m))):
                base = dict(ds_1m[i])
                distractor_idxs = [j for j in range(len(ds_1m)) if j != i][: mult - 1]
                distractors = [dict(ds_1m[j]) for j in distractor_idxs]
                synth = synthesize_row(base, distractors)
                sample.append((scale, i, synth))
            print(f"  synth-scale={scale} mult={mult} base_idx=0 distractors={mult-1}")
    print()

    rows: list[RowResult] = []
    for i, (scale, idx, row) in enumerate(sample):
        gid = f"ruler-pilot-{scale}-r{idx}"
        t0 = time.perf_counter()
        # Probe user_text length for the row header line.
        _, user_text = parse_ruler_input(row["input"])
        print(
            f"[{i+1}/{len(sample)}] scale={scale} idx={idx} user_chars={len(user_text):,} "
            f"llama3_tokens={row.get('length', 0):,} expected_answers={len(row['answers'])} running:",
            flush=True,
        )
        res = await run_row(scale, args.subtask, idx, row,
                            run_vanilla=not args.skip_vanilla,
                            run_cortex=not args.skip_cortex,
                            group_id=gid)
        rows.append(res)

        def fmt(a: ArmResult | None) -> str:
            if a is None:
                return "skip"
            return (f"{a.status[:12]:<12s} any={a.any_answer_found!s:>5s} "
                    f"all={a.all_answers_found!s:>5s} {a.n_found}/{a.n_expected} "
                    f"{a.latency_ms/1000:.1f}s")

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
    out: dict[str, Any] = {"per_scale": {}, "overall": {}}
    for arm_name in ("vanilla_opus", "cortex_opus"):
        any_ok, all_ok = [], []
        for r in rows:
            a: ArmResult | None = getattr(r, arm_name)
            if a is None or a.status != "ok":
                continue
            any_ok.append(1.0 if a.any_answer_found else 0.0)
            all_ok.append(1.0 if a.all_answers_found else 0.0)
        out["overall"][arm_name] = {
            "n": len(any_ok),
            "any_found_rate": (sum(any_ok) / len(any_ok)) if any_ok else None,
            "all_found_rate": (sum(all_ok) / len(all_ok)) if all_ok else None,
        }
    for scale in sorted({r.scale for r in rows}):
        scale_rows = [r for r in rows if r.scale == scale]
        out["per_scale"][scale] = {}
        for arm_name in ("vanilla_opus", "cortex_opus"):
            arms = [getattr(r, arm_name) for r in scale_rows
                    if getattr(r, arm_name) is not None and getattr(r, arm_name).status == "ok"]
            if not arms:
                continue
            out["per_scale"][scale][arm_name] = {
                "n": len(arms),
                "any_found_rate": sum(1 for a in arms if a.any_answer_found) / len(arms),
                "all_found_rate": sum(1 for a in arms if a.all_answers_found) / len(arms),
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
    print(f"{'arm':<14s} {'n':>3s}  {'any found':>10s}  {'all found':>10s}")
    print("=" * 80)
    for arm, s in summary["overall"].items():
        def fmt(v: float | None) -> str:
            return "n/a" if v is None else f"{v*100:.0f}%"
        print(f"{arm:<14s} {s['n']:>3d}  {fmt(s['any_found_rate']):>10s}  {fmt(s['all_found_rate']):>10s}")
    print()
    if summary.get("per_scale"):
        print("-- per-scale --")
        print(f"{'scale':<6s} {'arm':<14s} {'n':>3s}  {'any':>6s}  {'all':>6s}")
        print("-" * 50)
        for scale in summary["per_scale"]:
            for arm in ("vanilla_opus", "cortex_opus"):
                s = summary["per_scale"][scale].get(arm)
                if not s:
                    continue
                print(f"{scale:<6s} {arm:<14s} {s['n']:>3d}  "
                      f"{s['any_found_rate']*100:>5.0f}%  {s['all_found_rate']*100:>5.0f}%")
            print()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
