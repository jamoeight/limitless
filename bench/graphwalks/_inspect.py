"""Download GraphWalks and print: size distribution, problem type counts,
two full example prompts (one bfs, one parents). Cached locally so we don't
re-download on subsequent runs."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

from datasets import load_dataset  # type: ignore

CACHE = Path("data/graphwalks/_hf_cache")
CACHE.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ds = load_dataset("openai/graphwalks", split="train", cache_dir=str(CACHE))
    print(f"rows: {len(ds)}")
    print(f"columns: {ds.column_names}")
    print()

    types = Counter(r["problem_type"] for r in ds)
    print("problem_type counts:", dict(types))

    chars = sorted(r["prompt_chars"] for r in ds)
    n = len(chars)
    pct = lambda p: chars[int(n * p)]  # noqa: E731
    print(f"prompt_chars  min={chars[0]:,}  p10={pct(0.1):,}  p25={pct(0.25):,}  "
          f"p50={pct(0.5):,}  p75={pct(0.75):,}  p90={pct(0.9):,}  max={chars[-1]:,}")
    print()

    shown = {"bfs": False, "parents": False}
    for r in ds:
        t = r["problem_type"]
        if shown[t]:
            continue
        shown[t] = True
        print("=" * 80)
        print(f"problem_type = {t}    prompt_chars = {r['prompt_chars']:,}    "
              f"answer_nodes (n={len(r['answer_nodes'])}): "
              f"{r['answer_nodes'][:8]}{'...' if len(r['answer_nodes']) > 8 else ''}")
        print("=" * 80)
        p = r["prompt"]
        print(f"--- prompt head (1200 chars) ---")
        print(p[:1200])
        print()
        print(f"--- prompt tail (1200 chars) ---")
        print(p[-1200:])
        print()
        if all(shown.values()):
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
