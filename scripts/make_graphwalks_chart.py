"""Render the GraphWalks README chart from results/graphwalks.json.

Single figure, two panels (top: exact-match accuracy by bucket; bottom: p50
end-to-end latency by bucket). Bars: baseline (in-context Qwen3.5-9B),
ours-judge (drop-in: 1 LLM call + Cypher). Failures (context overflow,
format failures) are scored as 0% accuracy — the honest comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def fmt_chars(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


def main() -> int:
    data = json.loads(Path("results/graphwalks.json").read_text(encoding="utf-8"))
    summary = data["summary"]
    buckets = list(summary["buckets"].keys())  # XS, S, M, L, XL in run order

    # Pull bucket char-range bounds from run.py BUCKETS for x-axis labels.
    # Hard-coded mirror to avoid the import dance; keep in sync if BUCKETS change.
    char_ranges = {
        "XS": (0, 5_000),
        "S":  (5_000, 15_000),
        "M":  (15_000, 130_000),
        "L":  (130_000, 500_000),
        "XL": (500_000, 10_000_000),
    }
    labels = []
    for b in buckets:
        lo, hi = char_ranges[b]
        labels.append(f"{b}\n{fmt_chars(lo)}–{fmt_chars(hi)} chars")

    ours_em   = [summary["buckets"][b]["ours_judge_em"]    * 100 for b in buckets]
    base_em   = [summary["buckets"][b]["baseline_em_over_all"] * 100 for b in buckets]
    ours_p50  = [summary["buckets"][b]["ours_judge_total_ms_p50"] for b in buckets]
    base_p50  = [
        summary["buckets"][b]["baseline_ms_p50_over_runnable"] or 0
        for b in buckets
    ]
    base_overflow = []
    for b in buckets:
        counts = summary["buckets"][b]["baseline_status_counts"]
        n = summary["buckets"][b]["n"]
        skipped = counts.get("skipped_too_long", 0) + counts.get("context_overflow", 0)
        base_overflow.append(skipped == n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), dpi=140,
                                   gridspec_kw={"height_ratios": [1.1, 1]})
    x = np.arange(len(buckets))
    w = 0.38

    # --- Accuracy panel ---
    b1 = ax1.bar(x - w / 2, base_em, w, label="In-context Qwen3.5-9B (baseline)", color="#bababa")
    b2 = ax1.bar(x + w / 2, ours_em, w, label="Ours: Neo4j + 1 LLM call",         color="#54a24b")
    for i, (a, b) in enumerate(zip(base_em, ours_em)):
        ax1.text(i - w / 2, a + 2, f"{a:.0f}%", ha="center", va="bottom",
                 fontsize=8, color="#444")
        ax1.text(i + w / 2, b + 2, f"{b:.0f}%", ha="center", va="bottom",
                 fontsize=8, color="#2d5e2e", fontweight="bold")
        if base_overflow[i]:
            ax1.text(i - w / 2, 5, "context\noverflow", ha="center", va="bottom",
                     fontsize=7, color="#a23", style="italic")
    ax1.set_ylim(0, 115)
    ax1.set_ylabel("Exact-match accuracy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_title("GraphWalks: in-context LLM degrades, external graph stays flat",
                  fontsize=11, pad=10)
    ax1.yaxis.set_major_formatter(lambda v, _: f"{int(v)}%")
    ax1.legend(loc="upper right", frameon=False, fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", linestyle=":", alpha=0.35)

    # --- Latency panel (log scale because the spread is large) ---
    ax2.bar(x - w / 2, base_p50, w, label="Baseline p50", color="#bababa")
    ax2.bar(x + w / 2, ours_p50, w, label="Ours p50",     color="#54a24b")
    for i, (a, b) in enumerate(zip(base_p50, ours_p50)):
        if a > 0:
            ax2.text(i - w / 2, a * 1.08, f"{a / 1000:.1f}s",
                     ha="center", va="bottom", fontsize=8, color="#444")
        else:
            ax2.text(i - w / 2, 200, "n/a", ha="center", va="bottom",
                     fontsize=7, color="#a23", style="italic")
        ax2.text(i + w / 2, b * 1.08, f"{b / 1000:.1f}s",
                 ha="center", va="bottom", fontsize=8, color="#2d5e2e",
                 fontweight="bold")
    ax2.set_yscale("log")
    ax2.set_ylim(50, max(max(base_p50 + ours_p50) * 2, 200_000))
    ax2.set_ylabel("p50 latency (ms, log)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.legend(loc="upper left", frameon=False, fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", linestyle=":", alpha=0.35)

    plt.tight_layout()
    out = "results/graphwalks.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
