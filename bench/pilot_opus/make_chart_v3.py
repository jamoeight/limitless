"""Hero chart for MRCR v2 token-based scaling — Anthropic-style whole percentages.

Matches the format Anthropic used in the claude-opus-4-6 announcement:
8-needle MRCR v2 across 256K / 1M / 5M / 10M token contexts, whole-percent
labels on bars.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def _tokens_label(n_chars: int) -> str:
    """Char count → approximate token count label (char/4 convention)."""
    t = n_chars // 4
    if t < 1_000_000:
        return f"{t // 1000}K"
    return f"{t / 1_000_000:.0f}M" if t >= 10_000_000 else f"{t / 1_000_000:.1f}M".rstrip("0").rstrip(".")


def _bar_label(score: float, status: str) -> str:
    if status != "ok":
        return "OVERFLOW"
    return f"{round(score * 100)}%"


def main(src: str = "results/opus_vs_cortex/mrcr_v3.json",
         out: str = "results/opus_vs_cortex/hero_v2.png") -> None:
    src_path = ROOT / src
    out_path = ROOT / out
    d = json.loads(src_path.read_text(encoding="utf-8"))

    rows = sorted(d["rows"], key=lambda r: r["n_chars"])

    vanilla_pct = []
    cortex_pct = []
    vanilla_status = []
    for r in rows:
        v = r["vanilla_opus"]
        c = r["cortex_opus"]
        vanilla_status.append(v["status"])
        vanilla_pct.append(v["score_lenient"] * 100 if v["status"] == "ok" else 0.0)
        cortex_pct.append(c["score_lenient"] * 100)

    arm_labels = {"vanilla_opus": "Claude Opus 4.7",
                  "cortex_opus": "Claude Opus 4.7 + cortex"}
    colors = {"vanilla_opus": "#d97a3f", "cortex_opus": "#2da44e"}

    fig, ax = plt.subplots(1, 1, figsize=(12, 6.5))

    n = len(rows)
    x = list(range(n))
    bar_w = 0.38

    bars_v = ax.bar([xi - bar_w / 2 for xi in x], vanilla_pct, bar_w,
                    label=arm_labels["vanilla_opus"],
                    color=colors["vanilla_opus"], edgecolor="white", linewidth=0.8)
    bars_c = ax.bar([xi + bar_w / 2 for xi in x], cortex_pct, bar_w,
                    label=arm_labels["cortex_opus"],
                    color=colors["cortex_opus"], edgecolor="white", linewidth=0.8)

    for rect, score, status in zip(bars_v, vanilla_pct, vanilla_status):
        label = _bar_label(score / 100.0, status)
        is_overflow = status != "ok"
        ax.annotate(label,
                    xy=(rect.get_x() + rect.get_width() / 2, max(score, 2.0)),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=12, fontweight="bold" if is_overflow else "normal",
                    color="#a83232" if is_overflow else "black")

    for rect, score in zip(bars_c, cortex_pct):
        ax.annotate(_bar_label(score / 100.0, "ok"),
                    xy=(rect.get_x() + rect.get_width() / 2, score),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    xtick_labels = []
    for r in rows:
        token_lbl = _tokens_label(r["n_chars"])
        syn = " (synth)" if r["n_chars"] > 2_600_000 else ""
        xtick_labels.append(f"{token_lbl} tokens{syn}")
    ax.set_xticklabels(xtick_labels, fontsize=12)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("MRCR v2 8-needle score (lenient)", fontsize=12)
    ax.set_title("MRCR v2 8-needle scaling — vanilla Opus 4.7 vs Opus 4.7 + cortex",
                 fontsize=14, fontweight="bold", pad=36)
    # Legend above the chart (between title and bars) — keeps it off the
    # 256K vanilla 16% bar at lower-left and the cortex 100% bars elsewhere.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
              framealpha=0.95, fontsize=12, frameon=True)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    note = (
        "256K and 1M tokens are real MRCR 8-needle rows. 5M and 10M synthesized by stitching "
        "real MRCR rows (dataset max = ~625K tokens). Methodology mirrors Anthropic's "
        "claude-opus-4-6 announcement (MRCR v2, 8-needle, token-based context). "
        "OVERFLOW = vanilla Opus 4.7's 200K-token native context exceeded."
    )
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=9, style="italic",
             color="#555555", wrap=True)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
