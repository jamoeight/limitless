"""Render MRCR comparison chart from results/mrcr.json.

Single figure, two panels:
  - top: mean SequenceMatcher score by bucket, baseline vs ours
  - bottom: p50 end-to-end latency by bucket (log scale)

Failures (context_overflow, http_error) score 0.0 — the honest comparison.
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
    data = json.loads(Path("results/mrcr.json").read_text(encoding="utf-8"))
    summary = data["summary"]
    buckets = list(summary["buckets"].keys())

    char_ranges = {
        "XS": (0,         100_000),
        "S":  (100_000,   200_000),
        "M":  (200_000,   500_000),
        "L":  (500_000, 1_500_000),
        "XL": (1_500_000, 10_000_000),
    }
    labels = []
    for b in buckets:
        lo, hi = char_ranges[b]
        labels.append(f"{b}\n{fmt_chars(lo)}–{fmt_chars(hi)} chars")

    ours_score = [summary["buckets"][b]["ours_mean_score"]            for b in buckets]
    base_score = [summary["buckets"][b]["baseline_mean_score_over_all"] for b in buckets]
    ours_p50   = [summary["buckets"][b]["ours_total_ms_p50"]          for b in buckets]
    base_p50   = [summary["buckets"][b]["baseline_ms_p50_over_runnable"] or 0
                  for b in buckets]
    base_unreachable = []
    for b in buckets:
        c = summary["buckets"][b]["baseline_status_counts"]
        n = summary["buckets"][b]["n"]
        skipped = c.get("skipped_too_long", 0) + c.get("context_overflow", 0)
        base_unreachable.append(skipped == n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), dpi=140,
                                   gridspec_kw={"height_ratios": [1.1, 1]})
    x = np.arange(len(buckets))
    w = 0.38

    # --- Score panel ---
    ax1.bar(x - w / 2, base_score, w, label="In-context Qwen3.5-9B (baseline)", color="#bababa")
    ax1.bar(x + w / 2, ours_score, w, label="Ours: external index + 1 LLM call",         color="#54a24b")
    for i, (a, b) in enumerate(zip(base_score, ours_score)):
        ax1.text(i - w / 2, a + 0.03, f"{a:.2f}", ha="center", va="bottom",
                 fontsize=8, color="#444")
        ax1.text(i + w / 2, b + 0.03, f"{b:.2f}", ha="center", va="bottom",
                 fontsize=8, color="#2d5e2e", fontweight="bold")
        if base_unreachable[i]:
            ax1.text(i - w / 2, 0.04, "context\noverflow", ha="center", va="bottom",
                     fontsize=7, color="#a23", style="italic")
    ax1.set_ylim(0, 1.18)
    ax1.set_ylabel("Mean MRCR score (SequenceMatcher.ratio)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_title("MRCR (OpenAI): in-context degrades, external index stays perfect",
                  fontsize=11, pad=10)
    ax1.legend(loc="upper right", frameon=False, fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", linestyle=":", alpha=0.35)

    # --- Latency panel ---
    ax2.bar(x - w / 2, base_p50, w, label="Baseline p50", color="#bababa")
    ax2.bar(x + w / 2, ours_p50, w, label="Ours p50",     color="#54a24b")
    for i, (a, b) in enumerate(zip(base_p50, ours_p50)):
        if a > 0:
            ax2.text(i - w / 2, a * 1.08, f"{a / 1000:.1f}s",
                     ha="center", va="bottom", fontsize=8, color="#444")
        else:
            ax2.text(i - w / 2, 200, "n/a", ha="center", va="bottom",
                     fontsize=7, color="#a23", style="italic")
        if b > 0:
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
    out = "results/mrcr.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
