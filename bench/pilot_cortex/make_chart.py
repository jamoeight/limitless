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

    # Wider group spacing so the multi-line xtick labels don't touch.
    group_w = 2.4
    bar_w = 0.62
    x = [i * group_w for i in range(len(buckets))]

    fig, ax = plt.subplots(figsize=(12, 5.8))

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
    # Single-line xticks; status moved to a smaller second line with extra spacing.
    bucket_labels = [
        "S  (≤200K chars)\nfits natively",
        "M  (200K – 500K chars)\nfits natively",
        "L  (500K – 1.5M chars)\nneeds virtualization",
    ]
    ax.set_xticklabels(bucket_labels, fontsize=11)
    # Headroom so the "0.xxx" labels on the tallest bars don't touch the title.
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("MRCR lenient score  (1.0 = perfect)", fontsize=11)
    ax.set_title(
        "9B + cortex matches/beats Opus 4.7 on MRCR\n"
        "30-row stratified pilot, seed=42, 10 rows per bucket",
        fontsize=13, pad=14,
    )
    # Legend pinned to upper-left corner with padding away from bars.
    ax.legend(loc="upper left", fontsize=10, framealpha=0.96,
              bbox_to_anchor=(0.01, 0.99), borderpad=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", pad=8)

    # Pad below the bars so 2-line tick labels have room.
    fig.subplots_adjust(bottom=0.18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
