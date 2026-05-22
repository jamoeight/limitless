"""Compare two pilot runs side-by-side. Used to A/B the bug-fix rerun
against the original pilot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bench.pilot_cortex.analyze import analyze  # noqa: E402


def fmt_n(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "n/a"


def fmt_pct(v: float | None) -> str:
    return f"{v*100:>4.0f}%" if v is not None else " n/a"


def diff(v1: float | None, v2: float | None) -> str:
    if v1 is None or v2 is None:
        return "  n/a"
    d = v2 - v1
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:+.3f}"


def main(p1: str, p2: str, seed: int = 42, planned: int = 5) -> None:
    s1 = analyze(p1, seed=seed, total_rows_planned=planned)
    s2 = analyze(p2, seed=seed, total_rows_planned=planned)

    print("=" * 110)
    print(f"COMPARISON  v1={Path(p1).name}  vs  v2={Path(p2).name}")
    print("=" * 110)
    print()
    print(f"{'arm':<14s}  {'strict':<24s}  {'lenient':<24s}  {'halluc':<14s}  {'trunc':<14s}")
    print(f"{'':<14s}  {'v1   v2   Δ':<24s}  {'v1   v2   Δ':<24s}  {'v1  v2':<14s}  {'v1  v2':<14s}")
    print("-" * 110)
    for arm in ("raw_9b", "cortex_9b", "opus"):
        a1, a2 = s1["per_arm"][arm], s2["per_arm"][arm]
        strict = f"{fmt_n(a1['strict_mean'])} → {fmt_n(a2['strict_mean'])}  {diff(a1['strict_mean'], a2['strict_mean'])}"
        lenient = f"{fmt_n(a1['lenient_mean'])} → {fmt_n(a2['lenient_mean'])}  {diff(a1['lenient_mean'], a2['lenient_mean'])}"
        halluc = f"{a1['hallucinations']:>2d} → {a2['hallucinations']:>2d}"
        trunc = f"{a1['truncated_responses']:>2d} → {a2['truncated_responses']:>2d}"
        print(f"{arm:<14s}  {strict:<24s}  {lenient:<24s}  {halluc:<14s}  {trunc:<14s}")

    print()
    print("-- per-row deltas (lenient score) --")
    rows1 = {r["row_idx"]: r for r in s1["per_row"]}
    rows2 = {r["row_idx"]: r for r in s2["per_row"]}
    common = sorted(set(rows1.keys()) & set(rows2.keys()))
    for ri in common:
        r1, r2 = rows1[ri], rows2[ri]
        print(f"\n[row {ri}] {r1['bucket']} chars={r1['n_chars']:,} gold_len={r1['gold_len']}")
        for arm in ("raw_9b", "cortex_9b", "opus"):
            a1 = r1["arms"].get(arm, {})
            a2 = r2["arms"].get(arm, {})
            if a1.get("status") != "ok" or a2.get("status") != "ok":
                print(f"  {arm:<12s}: v1={a1.get('status','?')} v2={a2.get('status','?')}")
                continue
            l1, l2 = a1.get("lenient", 0.0), a2.get("lenient", 0.0)
            delta = l2 - l1
            sign = "+" if delta > 0.001 else ("-" if delta < -0.001 else "·")
            tag_v1 = ""
            tag_v2 = ""
            if a1.get("halluc"):
                tag_v1 += "H"
            if a1.get("trunc"):
                tag_v1 += "T"
            if a2.get("halluc"):
                tag_v2 += "H"
            if a2.get("trunc"):
                tag_v2 += "T"
            print(
                f"  {arm:<12s}: lenient {l1:.3f} → {l2:.3f}  Δ{delta:+.3f} {sign}   "
                f"resp_len {a1.get('resp_len',0)}→{a2.get('resp_len',0)}   "
                f"[{tag_v1 or '·'}→{tag_v2 or '·'}]"
            )

    print()
    print("=" * 110)
    print("VERDICT — does cortex_9b now claim infinite context?")
    print("=" * 110)
    c1 = s1["per_arm"]["cortex_9b"]
    c2 = s2["per_arm"]["cortex_9b"]
    r2 = s2["per_arm"]["raw_9b"]
    o2 = s2["per_arm"]["opus"]
    print(f"  cortex_9b lenient: v1={fmt_n(c1['lenient_mean'])} → v2={fmt_n(c2['lenient_mean'])}")
    print(f"  raw_9b   lenient: v2={fmt_n(r2['lenient_mean'])}")
    print(f"  opus     lenient: v2={fmt_n(o2['lenient_mean'])}")
    print()
    if c2["lenient_mean"] and o2["lenient_mean"] and c2["lenient_mean"] >= 0.90 * o2["lenient_mean"]:
        print("  ✓ cortex_9b is within 10% of Opus lenient mean — INFINITE CONTEXT CLAIM SUPPORTED")
    elif c2["lenient_mean"] and r2["lenient_mean"] and c2["lenient_mean"] > r2["lenient_mean"] + 0.10:
        print("  ~ cortex_9b beats raw_9b by >10% — partial win on overflow cases")
    elif c2["lenient_mean"] and r2["lenient_mean"] and abs(c2["lenient_mean"] - r2["lenient_mean"]) < 0.05:
        print("  ~ cortex_9b matches raw_9b — bugs fixed but no advantage demonstrated")
    else:
        print("  ✗ cortex_9b still trails raw_9b — infinite-context claim not supported")


if __name__ == "__main__":
    v1 = sys.argv[1] if len(sys.argv) > 1 else "results/pilot_cortex/pilot5.json"
    v2 = sys.argv[2] if len(sys.argv) > 2 else "results/pilot_cortex/pilot5_v2.json"
    main(v1, v2)
