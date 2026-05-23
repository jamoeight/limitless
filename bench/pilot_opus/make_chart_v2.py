"""Chart: Opus 4.7 vs Opus + cortex on MRCR v2 scaling (8-needle, 256K → 10M chars)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def main(src: str = "results/opus_vs_cortex/mrcr_v2.json",
         out: str = "results/opus_vs_cortex/hero_v2.png") -> None:
    src_path = ROOT / src
    out_path = ROOT / out
    d = json.loads(src_path.read_text(encoding="utf-8"))

    rows = sorted(d["rows"], key=lambda r: r["n_chars"])
    labels = [f"{r['n_chars']/1000:.0f}K" if r['n_chars'] < 1_000_000
              else f"{r['n_chars']/1_000_000:.1f}M"
              for r in rows]

    vanilla = [r["vanilla_opus"]["score_lenient"] if r["vanilla_opus"]["status"] == "ok" else 0.0
               for r in rows]
    cortex = [r["cortex_opus"]["score_lenient"] for r in rows]
    vanilla_status = [r["vanilla_opus"]["status"] for r in rows]

    arm_labels = {"vanilla_opus": "Claude Opus 4.7 (vanilla)",
                  "cortex_opus": "Claude Opus 4.7 + cortex"}
    colors = {"vanilla_opus": "#d97a3f", "cortex_opus": "#2da44e"}

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    n = len(rows)
    x = list(range(n))
    bar_w = 0.38

    bars_v = ax.bar([xi - bar_w / 2 for xi in x], vanilla, bar_w,
                    label=arm_labels["vanilla_opus"],
                    color=colors["vanilla_opus"], edgecolor="white", linewidth=0.8)
    bars_c = ax.bar([xi + bar_w / 2 for xi in x], cortex, bar_w,
                    label=arm_labels["cortex_opus"],
                    color=colors["cortex_opus"], edgecolor="white", linewidth=0.8)

    for rect, y, status in zip(bars_v, vanilla, vanilla_status):
        if status == "ok":
            label = f"{y:.3f}"
        else:
            label = "OVERFLOW"
        ax.annotate(label,
                    xy=(rect.get_x() + rect.get_width() / 2, max(y, 0.02)),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=11, fontweight="bold" if status != "ok" else "normal",
                    color="#a83232" if status != "ok" else "black")

    for rect, y in zip(bars_c, cortex):
        ax.annotate(f"{y:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, y),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x)
    xtick_labels = []
    for r, lbl in zip(rows, labels):
        syn = " (synth)" if r.get("_synthetic") or r["n_chars"] > 2_600_000 else ""
        hdrs = r["cortex_opus"].get("headers") or {}
        kept = hdrs.get("x-cortex-kept-messages", "?")
        orig = hdrs.get("x-cortex-original-messages", "?")
        xtick_labels.append(f"{lbl}{syn}\n{orig}→{kept} msgs")
    ax.set_xticklabels(xtick_labels, fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("MRCR lenient score (1.0 = perfect)", fontsize=12)
    ax.set_title("MRCR scaling: vanilla Opus 4.7 vs Opus 4.7 + cortex\n(8-needle, 256K → 10M chars)",
                 fontsize=14, fontweight="bold", pad=14)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=11)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    note = (
        "Real MRCR 8-needle rows used at 256K and 1M chars. "
        "5M and 10M synthesized by stitching MRCR 8-needle rows (dataset max = 2.5M).\n"
        "OVERFLOW = vanilla Opus rejected by API (input > 200K tokens). "
        "Cortex compresses 5K-10K messages to 7 verbatim + ~5K-token recap."
    )
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=9, style="italic",
             color="#555555")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
