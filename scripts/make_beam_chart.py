"""Generate the BEAM-500 comparison chart for the README.

Horizontal bars: our run vs spec target vs Phase 0 spike vs published baseline.
The story: bounded-LLM-call judge beats the published baseline by ~10×.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    summary_path = Path("results/beam_summary.json")
    if not summary_path.exists():
        print(f"ERR: {summary_path} not found — run bench/beam_subset/run.py first")
        return 1
    summary = json.loads(summary_path.read_text())

    our_acc = summary["accuracy"]
    n = summary["n"]

    rows = [
        ("BEAM Hindsight baseline\n(published)",  0.05,  "#bbbbbb"),
        ("Plan spec target",                      0.40,  "#cccccc"),
        ("Phase 0 spike\n(20 cases, temp=0.3)",   0.55,  "#aac8e0"),
        (f"BEAM full set — this run\n(greedy, n={n})",  our_acc,  "#54a24b"),
    ]
    labels = [r[0] for r in rows]
    accs   = [r[1] for r in rows]
    colors = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.0), dpi=140)
    y = np.arange(len(rows))
    bars = ax.barh(y, accs, color=colors, edgecolor="#333", linewidth=0.5)

    for i, (bar, acc) in enumerate(zip(bars, accs)):
        ax.text(acc + 0.01, i, f"{acc*100:.1f}%",
                va="center", ha="left", fontsize=10, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Accuracy on BEAM contradiction-resolution")
    ax.set_xlim(0, 1.0)
    ax.set_title(f"BEAM-500: judge correctly returns `unresolved`\n"
                 f"on {n} BEAM 100K contradiction-resolution cases",
                 fontsize=11, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    plt.tight_layout()
    out = "results/beam_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"saved -> {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
