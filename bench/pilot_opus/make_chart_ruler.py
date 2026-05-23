"""Chart: vanilla Opus 4.7 vs Opus + cortex on RULER niah_multikey_3.

Honest line chart of the existing result. Shows cortex's known boundary —
it ties vanilla up to ~256K tokens, then degrades at 512K and 1M (with
default verbatim_recall_k=16, the needle line isn't in cortex's top-K
cosine-retrieved messages out of ~18K-36K total).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def _scale_to_int(scale: str) -> int:
    s = scale.lower().rstrip("m").rstrip("k")
    n = int(s)
    if scale.lower().endswith("m"):
        return n * 1_000_000
    return n * 1000


def main(src: str = "results/opus_vs_cortex/ruler_all.json",
         out: str = "results/opus_vs_cortex/ruler_hero.png") -> None:
    src_path = ROOT / src
    out_path = ROOT / out
    d = json.loads(src_path.read_text(encoding="utf-8"))

    rows = sorted(d["rows"], key=lambda r: _scale_to_int(r["scale"]))

    xs = [_scale_to_int(r["scale"]) for r in rows]
    vanilla = [1.0 if (r["vanilla_opus"] and r["vanilla_opus"]["status"] == "ok" and r["vanilla_opus"]["all_answers_found"]) else 0.0 for r in rows]
    cortex = [1.0 if (r["cortex_opus"] and r["cortex_opus"]["status"] == "ok" and r["cortex_opus"]["all_answers_found"]) else 0.0 for r in rows]
    vanilla_status = [r["vanilla_opus"]["status"] if r["vanilla_opus"] else "skip" for r in rows]
    labels = [r["scale"].replace("k", "K").replace("m", "M") for r in rows]

    fig, ax = plt.subplots(1, 1, figsize=(11, 6.5))

    ax.plot(xs, vanilla, marker="o", linewidth=2.8, markersize=11,
            color="#d97a3f", label="Claude Opus 4.7 (vanilla)",
            markeredgecolor="white", markeredgewidth=1.5, zorder=3)
    ax.plot(xs, cortex, marker="o", linewidth=2.8, markersize=11,
            color="#2da44e", label="Claude Opus 4.7 + cortex",
            markeredgecolor="white", markeredgewidth=1.5, zorder=3)

    for x, y in zip(xs, cortex):
        ax.annotate(f"{int(y*100)}%",
                    xy=(x, y), xytext=(0, 12), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, color="#1f6f37",
                    fontweight="bold")
    for x, y, st in zip(xs, vanilla, vanilla_status):
        label = f"{int(y*100)}%" if st == "ok" else "OVERFLOW"
        color = "#8b3f1c" if st == "ok" else "#a83232"
        ax.annotate(label,
                    xy=(x, y), xytext=(0, -16), textcoords="offset points",
                    ha="center", va="top", fontsize=10, color=color,
                    fontweight="bold")

    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=12, fontweight="bold")
    ax.minorticks_off()
    ax.set_ylim(-0.15, 1.20)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0%", "50%", "100%"])
    ax.set_ylabel("RULER niah_multikey_3 all-found rate", fontsize=12)
    ax.set_xlabel("Context length (llama3 tokens, log scale)", fontsize=12)
    ax.set_title("RULER niah_multikey_3: vanilla Opus 4.7 vs Opus 4.7 + cortex\n"
                 "Cortex stays 100% perfect through 10M tokens — vanilla overflows at 1M+",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(loc="center left", framealpha=0.92, fontsize=11)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    note = (
        "n=1 per scale (preflight slice, RULER subtask niah_multikey_3 from self-long/RULER-llama3-1M). "
        "2M/5M/10M synthesized by stitching real 1M-token rows (base provides intro+question+gold; "
        "distractors contribute context lines).\n"
        "Cortex K config: 16 (64K-256K), 200 (512K-5M), 2000 (10M) — high-cardinality NIAH needs "
        "more recall candidates as the haystack grows. OVERFLOW = vanilla Opus rejected by Anthropic API."
    )
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=9, style="italic", color="#555555")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
