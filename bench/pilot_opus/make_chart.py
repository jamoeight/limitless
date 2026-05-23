"""Chart: vanilla Opus 4.7 vs Opus + cortex on MRCR per-bucket lenient."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def main(src: str = "results/opus_vs_cortex/mrcr.json",
         out: str = "results/opus_vs_cortex/hero.png") -> None:
    src_path = ROOT / src
    out_path = ROOT / out
    d = json.loads(src_path.read_text(encoding="utf-8"))

    buckets = ["S", "M", "L"]
    arms = ["vanilla_opus", "cortex_opus"]
    arm_labels = {"vanilla_opus": "Claude Opus 4.7 (vanilla)",
                  "cortex_opus": "Claude Opus 4.7 + cortex"}
    colors = {"vanilla_opus": "#d97a3f", "cortex_opus": "#2da44e"}

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(12, 9.5),
        gridspec_kw={"height_ratios": [1.0, 1.05], "hspace": 0.42},
    )

    # Panel 1: per-bucket mean lenient
    group_w = 2.0
    bar_w = 0.78
    x = [i * group_w for i in range(len(buckets))]

    for i, arm in enumerate(arms):
        ys = []
        for b in buckets:
            v = d["summary"]["per_bucket"].get(b, {}).get(arm, {})
            ys.append(v.get("mean_score_lenient", 0.0))
        offsets = [xi + (i - 0.5) * bar_w for xi in x]
        bars = ax.bar(offsets, ys, bar_w, label=arm_labels[arm],
                      color=colors[arm], edgecolor="white", linewidth=0.8)
        for rect, y in zip(bars, ys):
            ax.annotate(f"{y:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, y),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n({d['summary']['per_bucket'].get(b, {}).get('vanilla_opus', {}).get('n', 0)} rows)"
                       for b in buckets])
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("MRCR lenient score (1.0 = perfect)")
    ax.set_title("MRCR per-bucket: Opus 4.7 vs Opus 4.7 + cortex (per_bucket=5)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 2: per-row L-bucket detail
    l_rows = [r for r in d["rows"] if r["bucket"] == "L"]
    l_rows.sort(key=lambda r: r["n_needles"] * 10000 + r["n_chars"])

    n_rows = len(l_rows)
    x2 = list(range(n_rows))
    bar_w2 = 0.36

    for i, arm in enumerate(arms):
        ys = [r[arm]["score_lenient"] for r in l_rows]
        offsets = [xi + (i - 0.5) * bar_w2 for xi in x2]
        bars = ax2.bar(offsets, ys, bar_w2, label=arm_labels[arm],
                       color=colors[arm], edgecolor="white", linewidth=0.6)
        for rect, y in zip(bars, ys):
            ax2.annotate(f"{y:.2f}",
                         xy=(rect.get_x() + rect.get_width() / 2, y),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=9)

    ax2.set_xticks(x2)
    ax2.set_xticklabels([f"row {r['row_idx']}\n{r['n_chars']/1000:.0f}K chars\n{r['n_needles']}-needle"
                        for r in l_rows], fontsize=9)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("MRCR lenient score")
    ax2.set_title("L-bucket per-row: where Opus collapses, cortex rescues it",
                  fontsize=13, fontweight="bold", pad=10)
    ax2.legend(loc="lower left", framealpha=0.9)
    ax2.grid(axis="y", alpha=0.25, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
