"""Chart: Opus 4.7 vs Opus + cortex on MRCR 8-needle scaling.

Line chart with whole-number token labels on a log x-axis. Loads both
mrcr_v2.json and mrcr_v3.json and emits a single 6-point sweep.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

V2_TARGETS = [256_000, 1_000_000, 5_000_000, 10_000_000]
V3_TARGETS = [24_350_000, 48_700_000]
V4_TARGETS = [97_400_000, 243_500_000]


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


def _fmt_tokens_whole(n: int) -> str:
    """Whole-number labels: 53K, 205K, 1M, 2M, 5M, 9M (no decimals)."""
    if n < 1000:
        return f"{n}"
    if n < 1_000_000:
        return f"{round(n / 1000)}K"
    return f"{round(n / 1_000_000)}M"


def main(out: str = "results/opus_vs_cortex/hero_v2.png") -> None:
    out_path = ROOT / out

    rows: list[dict] = []
    for src, targets in (
        ("results/opus_vs_cortex/mrcr_v2.json", V2_TARGETS),
        ("results/opus_vs_cortex/mrcr_v3.json", V3_TARGETS),
        ("results/opus_vs_cortex/mrcr_v4.json", V4_TARGETS),
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

    xs = [r["_tokens"] for r in rows]
    vanilla = [r["vanilla_opus"]["score_lenient"] if r["vanilla_opus"]["status"] == "ok" else 0.0
               for r in rows]
    cortex = [r["cortex_opus"]["score_lenient"] for r in rows]
    vanilla_status = [r["vanilla_opus"]["status"] for r in rows]

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    ax.plot(xs, vanilla, marker="o", linewidth=2.8, markersize=11,
            color="#d97a3f", label="Claude Opus 4.7 (vanilla)",
            markeredgecolor="white", markeredgewidth=1.5, zorder=3)
    ax.plot(xs, cortex, marker="o", linewidth=2.8, markersize=11,
            color="#2da44e", label="Claude Opus 4.7 + cortex",
            markeredgecolor="white", markeredgewidth=1.5, zorder=3)

    # Annotate cortex points (always near 1.0)
    for x, y in zip(xs, cortex):
        ax.annotate(f"{y:.3f}",
                    xy=(x, y), xytext=(0, 12), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, color="#1f6f37",
                    fontweight="bold")

    # Annotate vanilla points — score for ok, "OVERFLOW" for failures
    for x, y, status in zip(xs, vanilla, vanilla_status):
        if status == "ok":
            ax.annotate(f"{y:.3f}",
                        xy=(x, y), xytext=(0, -16), textcoords="offset points",
                        ha="center", va="top", fontsize=10, color="#8b3f1c",
                        fontweight="bold")
        else:
            ax.annotate("OVERFLOW",
                        xy=(x, y), xytext=(0, -16), textcoords="offset points",
                        ha="center", va="top", fontsize=10, color="#a83232",
                        fontweight="bold")

    # Anthropic's published 1M Opus 4.6 limit
    ax.axvline(1_000_000, linestyle="--", color="#888", linewidth=1.4, alpha=0.7, zorder=1)
    ax.text(1_000_000, 0.50,
            " Anthropic's published\n Opus 4.6 limit\n (1M tokens)",
            fontsize=9.5, va="center", ha="left", color="#666", style="italic")

    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([_fmt_tokens_whole(x) for x in xs], fontsize=12, fontweight="bold")
    ax.minorticks_off()

    ax.set_ylim(-0.08, 1.15)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("MRCR lenient score (1.0 = perfect)", fontsize=12)
    ax.set_xlabel("Context length (tokens, log scale)", fontsize=12)
    ax.set_title("MRCR 8-needle: Claude Opus 4.7 vs Opus 4.7 + cortex\n"
                 "Cortex stays perfect through 29M tokens — 29× past Anthropic's published Opus 4.6 limit",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(loc="center left", framealpha=0.92, fontsize=11)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    note = (
        "Token counts via tiktoken o200k_base. 53K and 205K rows are real MRCR 8-needle samples; "
        "1M / 2M / 5M / 9M / 16M / 29M rows are synthesized by stitching real MRCR 8-needle rows "
        "(dataset max ≈ 625K tokens per row; 400-row dataset combined tops out at ~30M tokens).\n"
        "OVERFLOW = vanilla Opus rejected by the Anthropic API. Cortex compresses 1K–155K messages "
        "to 7 verbatim turns + a 5–65K-token recap (verbatim_recall_k=200 used at 16M+)."
    )
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=9, style="italic", color="#555555")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out_path} ({len(rows)} datapoints)")


if __name__ == "__main__":
    main(*sys.argv[1:])
