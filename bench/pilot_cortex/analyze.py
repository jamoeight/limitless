"""Post-pilot analysis: per-row + per-bucket scoring with lenient rescoring
and per-arm narrative findings (hallucination detection, response-length
caps, etc.).

Run after a pilot completes:
  PYTHONPATH=. .venv/Scripts/python bench/pilot_cortex/analyze.py results/pilot_cortex/pilot5.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bench.pilot_cortex.run import load_dataset, pick_rows  # noqa: E402
from bench.mrcr.loader import parse_row, score  # noqa: E402


def reconstruct_gold(seed: int, total_rows_planned: int) -> dict[int, Any]:
    """Re-derive the MRCR tasks for the rows we ran (by re-sampling with
    the same seed and intended row count). The pilot stores `row_idx` so we
    can match back. Pass the FULL planned row count, not the partial count,
    so the sample is reproduced exactly."""
    df = load_dataset()
    sample = pick_rows(df, total_rows_planned, seed)
    out = {}
    for r in sample:
        t = parse_row(r)
        out[int(r["_global_idx"])] = t
    return out


def analyze(path: str, seed: int = 42, total_rows_planned: int | None = None) -> dict[str, Any]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = d["rows"]
    if total_rows_planned is None:
        total_rows_planned = len(rows)
    gold = reconstruct_gold(seed, total_rows_planned)

    summary: dict[str, Any] = {
        "n_rows": len(rows),
        "per_arm": {},
        "per_row": [],
    }

    by_arm: dict[str, dict[str, Any]] = {
        "raw_9b":    {"strict": [], "lenient": [], "ok": 0, "overflow": 0, "err": 0, "halluc": 0, "trunc": 0, "lat_ms": []},
        "cortex_9b": {"strict": [], "lenient": [], "ok": 0, "overflow": 0, "err": 0, "halluc": 0, "trunc": 0, "lat_ms": []},
        "opus":      {"strict": [], "lenient": [], "ok": 0, "overflow": 0, "err": 0, "halluc": 0, "trunc": 0, "lat_ms": []},
    }

    for r in rows:
        t = gold[r["row_idx"]]
        row_view = {
            "row_idx": r["row_idx"], "bucket": r["bucket"], "n_chars": r["n_chars"],
            "n_needles": r["n_needles"], "position": r["position"],
            "gold_len": len(t.gold_answer), "arms": {},
        }
        for arm_name in ("raw_9b", "cortex_9b", "opus"):
            a = r.get(arm_name)
            if a is None:
                continue
            buc = by_arm[arm_name]
            st = a["status"]
            buc["lat_ms"].append(a["latency_ms"])
            if st == "ok":
                buc["ok"] += 1
                resp = a["response"]
                strict = score(resp, t.gold_answer, t.random_string)
                lenient = score(resp.lstrip(), t.gold_answer, t.random_string)
                # Hallucination heuristic: the model produced an answer (>20
                # chars) that contains random_string but the first 80 chars
                # of the needle don't appear anywhere in the response.
                needle_head = t.gold_answer[len(t.random_string):len(t.random_string) + 80]
                found_needle = needle_head in resp
                halluc = (len(resp) >= 20) and (t.random_string in resp) and not found_needle
                # Truncation heuristic: response length < 80% of gold length AND
                # lenient score > 0 (model started the right content but cut off).
                trunc = (len(resp) < 0.8 * len(t.gold_answer)) and lenient > 0.05
                buc["strict"].append(strict)
                buc["lenient"].append(lenient)
                if halluc:
                    buc["halluc"] += 1
                if trunc:
                    buc["trunc"] += 1
                row_view["arms"][arm_name] = {
                    "status": st, "strict": strict, "lenient": lenient,
                    "resp_len": len(resp), "halluc": halluc, "trunc": trunc,
                    "lat_s": a["latency_ms"] / 1000,
                }
            elif st in ("context_overflow",):
                buc["overflow"] += 1
                row_view["arms"][arm_name] = {"status": st, "lat_s": a["latency_ms"] / 1000}
            else:
                buc["err"] += 1
                row_view["arms"][arm_name] = {
                    "status": st, "error_head": (a.get("error") or "")[:120],
                    "lat_s": a["latency_ms"] / 1000,
                }
        summary["per_row"].append(row_view)

    for arm, b in by_arm.items():
        n_scored = len(b["strict"])
        summary["per_arm"][arm] = {
            "n_ok": b["ok"], "n_overflow": b["overflow"], "n_err": b["err"],
            "strict_mean": (sum(b["strict"]) / n_scored) if n_scored else None,
            "lenient_mean": (sum(b["lenient"]) / n_scored) if n_scored else None,
            "perfect_strict": (sum(s >= 0.99 for s in b["strict"]) / n_scored) if n_scored else None,
            "perfect_lenient": (sum(s >= 0.99 for s in b["lenient"]) / n_scored) if n_scored else None,
            "hallucinations": b["halluc"],
            "truncated_responses": b["trunc"],
            "median_lat_ms": (sorted(b["lat_ms"])[len(b["lat_ms"]) // 2] if b["lat_ms"] else None),
        }

    return summary


def print_report(s: dict[str, Any]) -> None:
    print()
    print("=" * 100)
    print(f"PILOT RESULT  ({s['n_rows']} rows)")
    print("=" * 100)
    print()

    print(f"{'arm':<12s} {'n_ok':>4s} {'ovrflw':>6s} {'err':>3s}  "
          f"{'strict':>7s} {'perf%':>6s}  {'lenient':>8s} {'perf%':>6s}  "
          f"{'halluc':>6s} {'trunc':>5s}  {'med_lat_s':>9s}")
    print("-" * 100)
    for arm, a in s["per_arm"].items():
        def f(v: float | None, kind: str = "ratio") -> str:
            if v is None:
                return "n/a"
            if kind == "pct":
                return f"{v*100:.0f}%"
            if kind == "ms":
                return f"{v/1000:.1f}s"
            return f"{v:.3f}"
        print(
            f"{arm:<12s} {a['n_ok']:>4d} {a['n_overflow']:>6d} {a['n_err']:>3d}  "
            f"{f(a['strict_mean']):>7s} {f(a['perfect_strict'], 'pct'):>6s}  "
            f"{f(a['lenient_mean']):>8s} {f(a['perfect_lenient'], 'pct'):>6s}  "
            f"{a['hallucinations']:>6d} {a['truncated_responses']:>5d}  "
            f"{f(a['median_lat_ms'], 'ms'):>9s}"
        )

    print()
    print("-- per-row detail --")
    for r in s["per_row"]:
        print(f"\n[row {r['row_idx']}] bucket={r['bucket']} chars={r['n_chars']:,} "
              f"needles={r['n_needles']} pos={r['position']} gold_len={r['gold_len']}")
        for arm_name in ("raw_9b", "cortex_9b", "opus"):
            arm = r["arms"].get(arm_name)
            if arm is None:
                continue
            if arm["status"] != "ok":
                print(f"  {arm_name:<12s} : {arm['status']:<18s} ({arm.get('error_head', '')[:60]})")
                continue
            tags = []
            if arm["halluc"]:
                tags.append("HALLUC")
            if arm["trunc"]:
                tags.append("TRUNC")
            tag = f" [{','.join(tags)}]" if tags else ""
            print(
                f"  {arm_name:<12s} : strict={arm['strict']:.3f}  lenient={arm['lenient']:.3f}  "
                f"resp_len={arm['resp_len']}  ({arm['lat_s']:.1f}s){tag}"
            )

    print()
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    raw, cortex, opus = s["per_arm"]["raw_9b"], s["per_arm"]["cortex_9b"], s["per_arm"]["opus"]

    def safe_pct(v: float | None) -> str:
        return f"{v*100:.0f}%" if v is not None else "n/a"

    def safe_num(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "n/a"

    print(f"  - raw 9B    : strict perfect = {safe_pct(raw['perfect_strict'])}, lenient mean = {safe_num(raw['lenient_mean'])}")
    print(f"  - cortex 9B : strict perfect = {safe_pct(cortex['perfect_strict'])}, lenient mean = {safe_num(cortex['lenient_mean'])}")
    print(f"  - Opus 4.7  : strict perfect = {safe_pct(opus['perfect_strict'])}, lenient mean = {safe_num(opus['lenient_mean'])}")
    print()
    print(f"  Hallucinations (model invented wrong needle): raw={raw['hallucinations']} / "
          f"cortex={cortex['hallucinations']} / opus={opus['hallucinations']}")
    print(f"  Truncated responses (right content, cut off): raw={raw['truncated_responses']} / "
          f"cortex={cortex['truncated_responses']} / opus={opus['truncated_responses']}")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "results/pilot_cortex/pilot5.json"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    planned = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    s = analyze(p, seed=seed, total_rows_planned=planned)
    print_report(s)
    Path(p).with_suffix(".analysis.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
