"""Generate the headline latency-decomposition chart for the README.

Stacked bar chart: cypher + qdrant + judge per graph size N.
The story: total stays flat across 4 orders of magnitude.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# Measured p50 latencies (ms) post-fix. 100K skipped: only have pre-fix data.
data = [
    {"N": 100,       "cypher": 7,  "qdrant": 263, "judge": 1969, "total": 2263},
    {"N": 1_000,     "cypher": 6,  "qdrant": 264, "judge": 1954, "total": 2247},
    {"N": 10_000,    "cypher": 15, "qdrant": 271, "judge": 1968, "total": 2278},
    {"N": 1_000_000, "cypher": 10, "qdrant": 471, "judge": 2263, "total": 2799},
]

labels = [f"{d['N']:,}" for d in data]
cypher = np.array([d["cypher"] for d in data])
qdrant = np.array([d["qdrant"] for d in data])
judge  = np.array([d["judge"]  for d in data])

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
x = np.arange(len(data))
bar_w = 0.55

ax.bar(x, cypher, bar_w, label="Cypher walk",  color="#4c78a8")
ax.bar(x, qdrant, bar_w, bottom=cypher, label="Qdrant search", color="#f58518")
ax.bar(x, judge,  bar_w, bottom=cypher + qdrant, label="Judge LLM (1 call)", color="#54a24b")

for i, d in enumerate(data):
    ax.text(i, d["total"] + 80, f"{d['total']} ms",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("Graph size (# facts)")
ax.set_ylabel("p50 latency (ms)")
ax.set_title("infer() latency stays bounded across 10,000× graph growth\n"
             "judge_call_count == 1, answer accuracy = 100% at every size",
             fontsize=11, pad=12)
ax.legend(loc="upper left", frameon=False)
ax.set_ylim(0, 3400)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", alpha=0.4)

plt.tight_layout()
out = "results/latency_vs_scale.png"
plt.savefig(out, bbox_inches="tight")
print(f"saved -> {out}")
