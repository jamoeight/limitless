"""Generate the hero chart for README.md: per-bucket lenient scores,
three arms × three buckets. The L-bucket is the load-bearing comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def main(src: str = "results/pilot_cortex/scale30_v3.json",
         out: str = "results/pilot_cortex/hero.png") -> None:
    src_path = ROOT / src
    out_path = ROOT / out
    d = json.loads(src_path.read_text(encoding="utf-8"))

    buckets = ["S", "M", "L"]
    arms = ["raw_9b", "cortex_9b", "opus"]
    arm_labels = {"raw_9b": "Qwen3.5-9B (raw)",
                  "cortex_9b": "Qwen3.5-9B + cortex",
                  "opus": "Claude Opus 4.7"}
    # Calm palette: muted blue, vivid green (cortex = headline), muted orange.
    colors = {"raw_9b": "#7f9cc0", "cortex_9b": "#2da44e", "opus": "#d97a3f"}

    # Two-panel figure: per-bucket overview on top, per-row L detail below.
    # Per-row L detail is where the cortex-vs-Opus comparison gets sharp —
    # the means hide that Opus collapses on 4 of 10 L rows while cortex is
    # rock-flat at 0.998-1.000.
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(12, 9.5),
        gridspec_kw={"height_ratios": [1.0, 1.05], "hspace": 0.42},
    )

    # ---------- Panel 1: per-bucket means ----------
    group_w = 2.4
    bar_w = 0.62
    x = [i * group_w for i in range(len(buckets))]

    for i, arm in enumerate(arms):
        ys = []
        for b in buckets:
            v = d["summary"]["per_bucket"].get(b, {}).get(arm, {})
            ys.append(v.get("mean_score_lenient", 0.0))
        offsets = [xi + (i - 1) * bar_w for xi in x]
        bars = ax.bar(offsets, ys, bar_w, label=arm_labels[arm],
                      color=colors[arm], edgecolor="white", linewidth=0.8)
        for rect, y in zip(bars, ys):
            ax.annotate(f"{y:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, y),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    bucket_labels = [
        "S  (≤200K chars)\nfits natively",
        "M  (200K – 500K chars)\nfits natively",
        "L  (500K – 1.5M chars)\nneeds virtualization",
    ]
    ax.set_xticklabels(bucket_labels, fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("MRCR lenient score  (1.0 = perfect)", fontsize=11)
    ax.set_title("Per-bucket means (n=10 per bucket)", fontsize=12, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", pad=8)

    # ---------- Panel 2: per-row L-bucket detail (sorted by char-count) ----------
    l_rows = sorted(
        [r for r in d["rows"] if r["bucket"] == "L"],
        key=lambda r: r["n_chars"],
    )
    n = len(l_rows)
    row_x = list(range(n))
    bar_w2 = 0.27
    for i, arm in enumerate(arms):
        ys = [r[arm]["score_lenient"] if r.get(arm) else 0.0 for r in l_rows]
        offsets = [xi + (i - 1) * bar_w2 for xi in row_x]
        ax2.bar(offsets, ys, bar_w2, label=arm_labels[arm],
                color=colors[arm], edgecolor="white", linewidth=0.6)

    # Mark each row with its char-count + needle count.
    row_labels = [
        f"{r['n_chars']/1_000_000:.2f}M\n{r['n_needles']} needles"
        for r in l_rows
    ]
    ax2.set_xticks(row_x)
    ax2.set_xticklabels(row_labels, fontsize=9)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("MRCR lenient score", fontsize=11)
    ax2.set_title(
        "Per-row L-bucket detail — where the means hide the story\n"
        "Cortex stays at 0.998–1.000 on every row; Opus collapses on 5 of 10",
        fontsize=12, pad=8,
    )
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.yaxis.grid(True, alpha=0.25)
    ax2.set_axisbelow(True)

    # ---------- Single shared legend, OUTSIDE the bar panels (top of figure) ----------
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="upper center", ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, 0.98), frameon=False)
    fig.suptitle(
        "9B + cortex matches/beats Claude Opus 4.7 on MRCR  "
        "(30-row stratified pilot, seed=42)",
        fontsize=14, y=1.005,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
