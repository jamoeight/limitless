"""Inspect MRCR: download the 2-needle and 8-needle parquet shards, print
size distribution and one example row per needle count (head + tail + query +
gold). Cache local to data/mrcr."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

from huggingface_hub import hf_hub_download  # type: ignore
import pandas as pd

CACHE = Path("data/mrcr")
CACHE.mkdir(parents=True, exist_ok=True)


def pull(filename: str) -> pd.DataFrame:
    fp = hf_hub_download(
        repo_id="openai/mrcr",
        filename=filename,
        repo_type="dataset",
        cache_dir=str(CACHE / "_hf_cache"),
    )
    return pd.read_parquet(fp)


def main() -> int:
    print("downloading 2needle shard 0...", flush=True)
    df2 = pull("2needle/2needle_0.parquet")
    print(f"  rows: {len(df2)}  cols: {list(df2.columns)}")
    print(f"  n_chars range: {int(df2['n_chars'].min()):,} → {int(df2['n_chars'].max()):,}")
    print(f"  n_needles: {Counter(df2['n_needles'])}")
    print(f"  desired_msg_index range: {int(df2['desired_msg_index'].min())}–{int(df2['desired_msg_index'].max())}")
    print(f"  total_messages range: {int(df2['total_messages'].min())}–{int(df2['total_messages'].max())}")
    print()

    # Quantiles of n_chars
    chars = sorted(df2["n_chars"].tolist())
    n = len(chars)
    pct = lambda p: chars[int(n * p)]  # noqa: E731
    print(f"  n_chars  min={chars[0]:,}  p10={pct(0.1):,}  p25={pct(0.25):,}  "
          f"p50={pct(0.5):,}  p75={pct(0.75):,}  p90={pct(0.9):,}  max={chars[-1]:,}")
    print()

    # Look at one example
    smallest = df2.nsmallest(1, "n_chars").iloc[0]
    msgs = json.loads(smallest["prompt"])
    print("=" * 80)
    print(f"smallest 2-needle row: n_chars={int(smallest['n_chars']):,}  "
          f"total_messages={int(smallest['total_messages'])}  "
          f"desired_msg_index={int(smallest['desired_msg_index'])}")
    print(f"random_string_to_prepend: {smallest['random_string_to_prepend']!r}")
    print(f"  answer head: {smallest['answer'][:200]!r}")
    print(f"  answer tail: {smallest['answer'][-200:]!r}")
    print()
    print(f"  num messages: {len(msgs)}")
    print(f"  first 2 messages:")
    for i, m in enumerate(msgs[:2]):
        body = m["content"][:200].replace("\n", " ")
        print(f"    [{i}] {m['role']:9s} : {body!r}{'...' if len(m['content']) > 200 else ''}")
    print(f"  LAST 2 messages (last = the query):")
    for i, m in enumerate(msgs[-2:], start=len(msgs) - 2):
        body = m["content"][:300].replace("\n", " ")
        print(f"    [{i}] {m['role']:9s} : {body!r}{'...' if len(m['content']) > 300 else ''}")

    # Find the matching user turns inside this convo (needles)
    last_user = msgs[-1]["content"]
    print()
    print(f"  query text: {last_user!r}")
    user_turns = [(i, m["content"]) for i, m in enumerate(msgs) if m["role"] == "user"]
    print(f"  total user turns in conv: {len(user_turns)} (including final query)")
    counts = Counter(t[1] for t in user_turns[:-1])  # excl. last query
    top = counts.most_common(5)
    print(f"  top repeated user requests:")
    for txt, c in top:
        print(f"    {c}× : {txt!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
