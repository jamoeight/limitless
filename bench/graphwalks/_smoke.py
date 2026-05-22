"""Pre-bench smoke check. Run the 10 smallest tasks, verify Cypher answers
match ground-truth set-equality. Bails out on the first mismatch with diagnostics."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

from datasets import load_dataset  # type: ignore

from bench.graphwalks.loader import parse_row, exact_match, f1
from bench.graphwalks.storage import (
    cypher_bfs_frontier,
    cypher_parents,
    delete_graph,
    ensure_schema,
    load_graph,
)
from timegraph.storage.neo4j_client import close_driver


CACHE = Path("data/graphwalks/_hf_cache")


async def main() -> int:
    print("loading dataset...", flush=True)
    ds = load_dataset("openai/graphwalks", split="train", cache_dir=str(CACHE))

    # Sort by prompt_chars and take 10 smallest, balanced bfs+parents.
    rows = sorted(
        [{"idx": i, **r} for i, r in enumerate(ds)],
        key=lambda r: r["prompt_chars"],
    )
    bfs_rows = [r for r in rows if r["problem_type"] == "bfs"][:5]
    par_rows = [r for r in rows if r["problem_type"] == "parents"][:5]
    sample = bfs_rows + par_rows

    print(f"running {len(sample)} smoke tasks (5 bfs + 5 parents)...\n", flush=True)
    await ensure_schema()

    all_pass = True
    for r in sample:
        task = parse_row(r)
        gid = f"gw_smoke_{r['idx']}"
        n_nodes, n_edges = await load_graph(gid, task.edges)
        try:
            if task.op == "bfs":
                assert task.depth is not None
                pred = await cypher_bfs_frontier(gid, task.start_node, task.depth)
            else:
                pred = await cypher_parents(gid, task.start_node)
            ok = exact_match(pred, task.answer)
            p, rec, f = f1(pred, task.answer)
            tag = "PASS" if ok else "FAIL"
            print(f"  [{tag}] row={r['idx']:4d}  {task.op:7s}  start={task.start_node}  "
                  f"depth={task.depth}  |V|={n_nodes:4d} |E|={n_edges:4d}  "
                  f"|pred|={len(pred):3d} |gold|={len(task.answer):3d}  F1={f:.3f}",
                  flush=True)
            if not ok:
                all_pass = False
                missing = task.answer - pred
                extra = pred - task.answer
                if missing:
                    print(f"        missing (gold not in pred): {sorted(missing)[:10]}")
                if extra:
                    print(f"        extra   (pred not in gold): {sorted(extra)[:10]}")
        finally:
            await delete_graph(gid)

    print()
    if all_pass:
        print("ALL SMOKE TASKS PASS")
        rc = 0
    else:
        print("SOME TASKS FAILED — fix Cypher semantics before scaling")
        rc = 1
    await close_driver()
    return rc


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
