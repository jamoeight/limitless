"""Diagnose the failing BFS rows. Compare:
  (1) Python in-memory BFS frontier
  (2) Cypher shortestPath frontier
  (3) Dataset ground truth
on the same row. Whichever matches gold tells us the correct semantics."""

from __future__ import annotations

import asyncio
import sys
from collections import defaultdict, deque
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

from datasets import load_dataset  # type: ignore

from bench.graphwalks.loader import parse_row
from bench.graphwalks.storage import (
    cypher_bfs_frontier,
    delete_graph,
    ensure_schema,
    load_graph,
)
from timegraph.storage.neo4j_client import close_driver, get_session


CACHE = Path("data/graphwalks/_hf_cache")


def python_bfs_frontier(edges: list[tuple[str, str]], start: str, depth: int) -> set[str]:
    """Standard BFS frontier: nodes whose shortest distance from start is exactly `depth`.
    Excludes start node."""
    adj: dict[str, set[str]] = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
    dist: dict[str, int] = {start: 0}
    q: deque = deque([start])
    while q:
        u = q.popleft()
        if dist[u] >= depth:
            continue
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return {n for n, d in dist.items() if d == depth and n != start}


async def debug_cypher_bfs(gid: str, start: str, depth: int) -> dict:
    """Return diagnostic info: nodes at each shortestPath length."""
    query = (
        f"MATCH (start:GwNode {{name: $start, gid: $gid}}) "
        f"MATCH (n:GwNode {{gid: $gid}}) WHERE n.name <> $start "
        f"MATCH p = shortestPath((start)-[:EDGE*..{depth}]->(n)) "
        f"RETURN n.name AS name, length(p) AS d ORDER BY d, name"
    )
    out: dict[int, list[str]] = defaultdict(list)
    async with get_session() as s:
        result = await s.run(query, parameters={"start": start, "gid": gid, "depth": depth})
        async for rec in result:
            out[rec["d"]].append(rec["name"])
    return dict(out)


async def main() -> int:
    ds = load_dataset("openai/graphwalks", split="train", cache_dir=str(CACHE))
    # The 5 failing BFS rows from the smoke test.
    failing = [400, 401, 402, 403, 404]
    await ensure_schema()
    for idx in failing:
        r = ds[idx]
        task = parse_row(r)
        print("=" * 80)
        print(f"row={idx}  start={task.start_node}  depth={task.depth}  |V|={len(task.nodes)} |E|={len(set(task.edges))}")
        print(f"  gold (n={len(task.answer)}): {sorted(task.answer)}")
        py = python_bfs_frontier(task.edges, task.start_node, task.depth or 0)
        print(f"  python BFS frontier (n={len(py)}): {sorted(py)}")

        gid = f"gw_diag_{idx}"
        await load_graph(gid, task.edges)
        try:
            cy = await cypher_bfs_frontier(gid, task.start_node, task.depth or 0)
            print(f"  cypher  frontier   (n={len(cy)}): {sorted(cy)}")
            buckets = await debug_cypher_bfs(gid, task.start_node, task.depth or 0)
            for d in sorted(buckets):
                print(f"    cypher d={d}: {sorted(buckets[d])}")
        finally:
            await delete_graph(gid)
        print()
    await close_driver()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
