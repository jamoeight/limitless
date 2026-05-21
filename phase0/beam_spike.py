"""Phase 0.11 — BEAM contradiction-resolution 20-case spike.

Downloads probing_questions.json from BEAM 100K chats, pools the
`contradiction_resolution` cases, runs them through our judge interface
against TWO models (Qwen3.5-9B and Qwopus 27B), prints a comparison table.

Scoring: BEAM contradiction cases generally expect "you said contradictory
things, please clarify" — that maps cleanly to our `unresolved` resolution.
A judge that picks `e1_correct` or `e2_correct` confidently is *wrong* on
these cases (it's failing to recognize the contradiction needs clarification).

Per-case score:
  - HIT: judge returns `unresolved` (or `both_partial` for partial overlap)
  - MISS: judge confidently picks a side

This isn't the full BEAM evaluation — it's a 20-case directional signal for
model selection. Phase 3 will do the full benchmark.

Usage:
  python phase0/beam_spike.py [--n 20] [--source-chats 25]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# UTF-8 stdout on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

from timegraph.llm.judge import JudgeClient
from timegraph.types import ConflictTriple, Resolution


BEAM_BASE = "https://raw.githubusercontent.com/mohammadtavakoli78/BEAM/main"

# Models to compare
MODELS = [
    ("qwen/qwen3.5-9b", "Qwen3.5-9B"),
    ("qwopus3.6-27b-v1-preview", "Qwopus-27B"),
]


def parse_ideal_answer(ideal: str) -> tuple[str, str] | None:
    """Extract (statement_1, statement_2) from BEAM's 'You said X, but you also said Y' template.

    BEAM uses several phrasings — handle the common ones. Returns None if we
    can't parse, in which case the case is skipped.
    """
    # Strip the "I notice you've mentioned contradictory information about this. "
    # prefix and the "Could you clarify which is correct?" suffix.
    body = ideal
    for prefix in [
        r"^I notice you'?ve mentioned contradictory information(?: about this)?\.?\s*",
        r"^You'?ve provided contradictory information\.?\s*",
        r"^There is contradictory information(?: about this)?\.?\s*",
    ]:
        body = re.sub(prefix, "", body, flags=re.IGNORECASE)
    body = re.sub(r"Could you clarify.*?\??$", "", body, flags=re.IGNORECASE | re.DOTALL).strip()

    # Patterns: "You said X, but you also said Y" / "X, but Y" / "X, but you also mentioned Y"
    patterns = [
        r"You said (?P<a>.+?),?\s*but you also (?:said|mentioned) (?P<b>.+?)\.?$",
        r"You (?:said|mentioned) (?P<a>.+?),?\s*(?:but|however)(?: you also (?:said|mentioned))? (?P<b>.+?)\.?$",
        r"(?P<a>.+?),?\s*but you also (?:said|mentioned) (?P<b>.+?)\.?$",
        r"On one hand,? you (?:said|mentioned) (?P<a>.+?)\.?\s*On the other(?: hand)?,? (?P<b>.+?)\.?$",
    ]
    for p in patterns:
        m = re.search(p, body, flags=re.IGNORECASE | re.DOTALL)
        if m:
            a = m.group("a").strip().rstrip(".,;")
            b = m.group("b").strip().rstrip(".,;")
            if len(a) > 5 and len(b) > 5:
                return a, b
    return None


async def fetch_probing(client: httpx.AsyncClient, chat_id: int) -> dict[str, Any] | None:
    url = f"{BEAM_BASE}/chats/100K/{chat_id}/probing_questions/probing_questions.json"
    try:
        r = await client.get(url, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


async def build_spike_dataset(target_n: int, source_chats: int) -> list[dict]:
    """Pool contradiction_resolution cases across BEAM 100K chats."""
    print(f"[1/3] Fetching probing_questions for BEAM 100K chats 1..{source_chats}…")
    async with httpx.AsyncClient() as http:
        tasks = [fetch_probing(http, i) for i in range(1, source_chats + 1)]
        results = await asyncio.gather(*tasks)

    pool: list[dict] = []
    for chat_id, pq in enumerate(results, start=1):
        if not pq:
            continue
        for case in pq.get("contradiction_resolution", []):
            parsed = parse_ideal_answer(case.get("ideal_answer", ""))
            if not parsed:
                continue
            a, b = parsed
            pool.append({
                "spike_id": f"beam_100k_chat{chat_id}_{case.get('contradiction_type', 'unk')}_{len(pool)+1:03d}",
                "question": case["question"],
                "statement_a": a,
                "statement_b": b,
                "contradiction_type": case.get("contradiction_type"),
                "difficulty": case.get("difficulty"),
                "topic": case.get("topic_questioned"),
                "ideal_resolution": "unresolved",  # BEAM contradictions request clarification
                "source_chat_id": chat_id,
            })

    print(f"      Pool size: {len(pool)} contradiction cases (target {target_n})")
    if len(pool) < target_n:
        print(f"      WARN: pool smaller than target; using all {len(pool)}")
        target_n = len(pool)

    # Pick a diverse sample: spread across chats and contradiction_types
    seen_chats: set[int] = set()
    seen_types: dict[str, int] = {}
    picked: list[dict] = []
    for case in pool:
        if len(picked) >= target_n:
            break
        cid = case["source_chat_id"]
        ctype = case["contradiction_type"]
        # Bias toward variety: skip if we already have 3+ of this chat or type
        if cid in seen_chats and seen_types.get(ctype, 0) >= max(1, target_n // 4):
            continue
        picked.append(case)
        seen_chats.add(cid)
        seen_types[ctype] = seen_types.get(ctype, 0) + 1

    # Backfill if diversity filter cut us short
    if len(picked) < target_n:
        for case in pool:
            if case in picked:
                continue
            picked.append(case)
            if len(picked) >= target_n:
                break

    print(f"      Picked {len(picked)} cases (types: {dict((t, sum(1 for p in picked if p['contradiction_type'] == t)) for t in {p['contradiction_type'] for p in picked})})")
    return picked


def score_resolution(predicted: Resolution, ideal: str) -> tuple[bool, str]:
    """BEAM contradictions request clarification → unresolved/both_partial = HIT."""
    if predicted in (Resolution.UNRESOLVED, Resolution.BOTH_PARTIAL):
        return True, "HIT"
    return False, "MISS (picked side when contradiction needed clarification)"


async def run_model(model_id: str, label: str, dataset: list[dict]) -> dict[str, Any]:
    """Run all cases through the judge against one model."""
    import os
    os.environ["TG_JUDGE_MODEL"] = model_id
    # Force re-read of settings
    from importlib import reload
    from timegraph import config as config_mod
    from timegraph.llm import judge as judge_mod
    reload(config_mod)
    reload(judge_mod)

    print(f"\n[2/3] Running {label} (model_id={model_id}) on {len(dataset)} cases…")
    client = judge_mod.JudgeClient()
    rows = []
    try:
        for i, case in enumerate(dataset, start=1):
            t0 = time.perf_counter()
            try:
                out = await client.judge_conflicts(
                    query=case["question"],
                    conflicts=[
                        judge_mod.ConflictTriple(
                            e1_fact_id="user_statement_a",
                            e2_fact_id="user_statement_b",
                            reason=f"{case['contradiction_type']}: a='{case['statement_a']}' vs b='{case['statement_b']}'",
                        )
                    ],
                    attestations=None,
                    source_episodes_truncated=[
                        f"User earlier: {case['statement_a']}",
                        f"User later: {case['statement_b']}",
                    ],
                )
                hit, note = score_resolution(out.resolution, case["ideal_resolution"])
                rows.append({
                    "spike_id": case["spike_id"],
                    "type": case["contradiction_type"],
                    "predicted": out.resolution.value,
                    "ideal": case["ideal_resolution"],
                    "hit": hit,
                    "note": note,
                    "confidence": out.confidence,
                    "latency_ms": out.latency_ms,
                })
            except Exception as e:
                rows.append({
                    "spike_id": case["spike_id"],
                    "type": case["contradiction_type"],
                    "predicted": "ERROR",
                    "ideal": case["ideal_resolution"],
                    "hit": False,
                    "note": f"error: {type(e).__name__}: {str(e)[:80]}",
                    "confidence": 0.0,
                    "latency_ms": (time.perf_counter() - t0) * 1000,
                })
            print(f"  [{i:2}/{len(dataset)}] {rows[-1]['predicted']:12} "
                  f"{'HIT ' if rows[-1]['hit'] else 'MISS'} "
                  f"{rows[-1]['latency_ms']:6.0f}ms  ({case['contradiction_type']})")
    finally:
        await client.close()

    hits = sum(1 for r in rows if r["hit"])
    lats = [r["latency_ms"] for r in rows if r["latency_ms"] > 0]
    return {
        "label": label,
        "model_id": model_id,
        "n": len(rows),
        "hits": hits,
        "accuracy": hits / len(rows) if rows else 0.0,
        "latency_mean_ms": statistics.mean(lats) if lats else 0,
        "latency_p50_ms": statistics.median(lats) if lats else 0,
        "latency_p95_ms": sorted(lats)[int(len(lats) * 0.95)] if len(lats) >= 2 else (lats[0] if lats else 0),
        "rows": rows,
    }


def print_summary(results: list[dict], dataset: list[dict]) -> None:
    print()
    print("=" * 88)
    print("Final summary")
    print("=" * 88)
    print(f"Dataset: {len(dataset)} BEAM 100K contradiction-resolution cases")
    print(f"Scoring: ideal resolution is `unresolved` (BEAM cases request clarification)")
    print()

    header = f"{'Model':<20} {'Accuracy':<12} {'Mean lat':<12} {'p50':<10} {'p95':<10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['label']:<20} "
              f"{r['hits']:>2}/{r['n']} = {r['accuracy']*100:>4.1f}%  "
              f"{r['latency_mean_ms']/1000:>5.1f}s     "
              f"{r['latency_p50_ms']/1000:>5.1f}s    "
              f"{r['latency_p95_ms']/1000:>5.1f}s")
    print()

    # Compare which model got which case right
    if len(results) == 2:
        a, b = results
        rows_by_id_a = {r["spike_id"]: r for r in a["rows"]}
        rows_by_id_b = {r["spike_id"]: r for r in b["rows"]}
        both_hit = sum(1 for sid in rows_by_id_a if rows_by_id_a[sid]["hit"] and rows_by_id_b[sid]["hit"])
        only_a = sum(1 for sid in rows_by_id_a if rows_by_id_a[sid]["hit"] and not rows_by_id_b[sid]["hit"])
        only_b = sum(1 for sid in rows_by_id_a if not rows_by_id_a[sid]["hit"] and rows_by_id_b[sid]["hit"])
        both_miss = sum(1 for sid in rows_by_id_a if not rows_by_id_a[sid]["hit"] and not rows_by_id_b[sid]["hit"])
        print(f"Agreement matrix:")
        print(f"  Both HIT:   {both_hit}")
        print(f"  Only {a['label']}: {only_a}")
        print(f"  Only {b['label']}: {only_b}")
        print(f"  Both MISS:  {both_miss}")
        print()
        speedup = b["latency_mean_ms"] / a["latency_mean_ms"] if a["latency_mean_ms"] > 0 else 0
        print(f"Speedup: {a['label']} is {speedup:.1f}x faster than {b['label']} on mean latency")
        print()
        if a["accuracy"] >= b["accuracy"] - 0.10:  # within 10pp
            print(f"VERDICT: {a['label']} is the winner — comparable accuracy "
                  f"({a['accuracy']*100:.1f}% vs {b['accuracy']*100:.1f}%) at {speedup:.1f}x speedup.")
        else:
            gap = (b['accuracy'] - a['accuracy']) * 100
            print(f"VERDICT: {b['label']} wins on accuracy ({gap:.1f}pp lead). "
                  f"Use it for the judge call site; accept the {speedup:.1f}x latency cost.")
    print("=" * 88)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="Number of spike cases")
    ap.add_argument("--source-chats", type=int, default=25, help="BEAM chats to pull from")
    ap.add_argument("--out", type=Path, default=Path("results/beam_spike.json"))
    args = ap.parse_args()

    dataset = await build_spike_dataset(args.n, args.source_chats)
    if len(dataset) < 5:
        print("[FAIL] not enough BEAM cases parsed; check ideal_answer regex")
        return 1

    # Save dataset for reproducibility
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".dataset.json").write_text(json.dumps(dataset, indent=2))

    results = []
    for model_id, label in MODELS:
        result = await run_model(model_id, label, dataset)
        results.append(result)

    args.out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[3/3] Per-case results saved to {args.out}")

    print_summary(results, dataset)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
