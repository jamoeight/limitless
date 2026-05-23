"""Chart: Opus 4.7 vs Opus + cortex on MRCR 8-needle scaling.

Context axis labeled in TOKENS (o200k_base), matching Anthropic's convention.
Token counts are computed on the actual prompt text the model sees (sum of
all message contents in the conversation list).

Loads BOTH mrcr_v2.json (53K, 205K, 1M, 2M token rows) and mrcr_v3.json
(5M, 10M token rows) if both exist, and emits a single combined chart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

# Targets used by each result file. Must match the bench invocations.
V2_TARGETS = [256_000, 1_000_000, 5_000_000, 10_000_000]
V3_TARGETS = [24_350_000, 48_700_000]


def _load_dataset_rows_for(targets: list[int], seed: int = 42):
    sys.path.insert(0, str(ROOT))
    from bench.pilot_opus.run import load_dataset, pick_rows_by_target
    df = load_dataset()
    return pick_rows_by_target(df, targets, seed=seed, n_needles=8)


def _tokenize(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))
    except ImportError:
        return int(len(text) / 4.87)


def _prompt_text(sr: dict) -> str:
    prompt = sr["prompt"]
    msgs = json.loads(prompt) if isinstance(prompt, str) else prompt
    return "\n".join(m.get("content", "") for m in msgs)


def _fmt_tokens(n: int) -> str:
    if n < 1000:
        return f"{n}"
    if n < 1_000_000:
        return f"{n/1000:.0f}K"
    if n < 10_000_000:
        return f"{n/1_000_000:.2f}M".rstrip("0").rstrip(".")
    return f"{n/1_000_000:.1f}M"


def main(out: str = "results/opus_vs_cortex/hero_v2.png") -> None:
    out_path = ROOT / out

    rows: list[dict] = []
    for src, targets in (
        ("results/opus_vs_cortex/mrcr_v2.json", V2_TARGETS),
        ("results/opus_vs_cortex/mrcr_v3.json", V3_TARGETS),
    ):
        p = ROOT / src
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        sample = _load_dataset_rows_for(targets)
        sample_sorted = sorted(sample, key=lambda r: r["n_chars"])
        result_sorted = sorted(d["rows"], key=lambda r: r["n_chars"])
        for rr, sr in zip(result_sorted, sample_sorted):
            tokens = _tokenize(_prompt_text(sr))
            rows.append({**rr, "_tokens": tokens, "_synthetic": sr.get("_synthetic", False)})

    rows.sort(key=lambda r: r["_tokens"])

    vanilla = [r["vanilla_opus"]["score_lenient"] if r["vanilla_opus"]["status"] == "ok" else 0.0
               for r in rows]
    cortex = [r["cortex_opus"]["score_lenient"] for r in rows]
    vanilla_status = [r["vanilla_opus"]["status"] for r in rows]

    arm_labels = {"vanilla_opus": "Claude Opus 4.7 (vanilla)",
                  "cortex_opus": "Claude Opus 4.7 + cortex"}
    colors = {"vanilla_opus": "#d97a3f", "cortex_opus": "#2da44e"}

    n = len(rows)
    fig_w = max(12, 2 + n * 1.8)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, 7.5))

    x = list(range(n))
    bar_w = 0.38

    bars_v = ax.bar([xi - bar_w / 2 for xi in x], vanilla, bar_w,
                    label=arm_labels["vanilla_opus"],
                    color=colors["vanilla_opus"], edgecolor="white", linewidth=0.8)
    bars_c = ax.bar([xi + bar_w / 2 for xi in x], cortex, bar_w,
                    label=arm_labels["cortex_opus"],
                    color=colors["cortex_opus"], edgecolor="white", linewidth=0.8)

    for rect, y, status in zip(bars_v, vanilla, vanilla_status):
        label = f"{y:.3f}" if status == "ok" else "OVERFLOW"
        ax.annotate(label,
                    xy=(rect.get_x() + rect.get_width() / 2, max(y, 0.02)),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold" if status != "ok" else "normal",
                    color="#a83232" if status != "ok" else "black")

    for rect, y in zip(bars_c, cortex):
        ax.annotate(f"{y:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, y),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)

    # Highlight Anthropic's published 1M-token Opus 4.6 limit. Snap to the bar
    # just past 1M tokens.
    one_m_idx = next((i for i, r in enumerate(rows) if r["_tokens"] >= 1_000_000), None)
    if one_m_idx is not None:
        ax.axvline(one_m_idx - 0.5, color="#888", linestyle=":", linewidth=1.2, alpha=0.7)
        ax.text(one_m_idx - 0.5, 1.135,
                " Anthropic's published\n Opus 4.6 limit = 1M tokens",
                fontsize=9, va="top", ha="left", color="#666", style="italic")

    ax.set_xticks(x)
    xtick_labels = []
    for r in rows:
        syn = " (synth)" if r.get("_synthetic") else ""
        hdrs = r["cortex_opus"].get("headers") or {}
        kept = hdrs.get("x-cortex-kept-messages", "?")
        orig = hdrs.get("x-cortex-original-messages", "?")
        xtick_labels.append(f"{_fmt_tokens(r['_tokens'])} tok{syn}\n{orig}→{kept} msgs")
    ax.set_xticklabels(xtick_labels, fontsize=9.5)
    ax.set_ylim(0, 1.20)
    ax.set_ylabel("MRCR lenient score (1.0 = perfect)", fontsize=12)
    ax.set_xlabel("Context length (o200k tokens)", fontsize=11)
    ax.set_title("MRCR 8-needle scaling: vanilla Opus 4.7 vs Opus 4.7 + cortex\n"
                 "Anthropic publishes Opus 4.6 at 1M tokens; cortex stays perfect through 9M+",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=11)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    note = (
        "Token counts via o200k_base. ~53K and ~205K rows are real MRCR 8-needle samples; "
        "~1M, ~2M, ~5M, and ~9M-token rows are synthesized by stitching real MRCR 8-needle rows "
        "(dataset max = ~625K tokens per row).\n"
        "OVERFLOW = vanilla Opus rejected by API. Cortex compresses 1K–50K messages to 7 verbatim "
        "turns + a ~5–6K-token recap that fits inside the upstream model's window."
    )
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=9, style="italic", color="#555555")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out_path} ({n} datapoints)")


if __name__ == "__main__":
    main(*sys.argv[1:])
