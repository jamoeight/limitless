"""Neo4j storage for GraphWalks graphs.

Each task gets its own group_id ("gid") namespace. Nodes are :GwNode {name, gid}
with a uniqueness constraint. Edges are [:EDGE]. Bulk load via UNWIND.

Schema-level setup is run once via `ensure_schema()`. Each task should:
    load_graph(gid, edges)  ->  run cypher_bfs/cypher_parents  ->  delete_graph(gid)
"""

from __future__ import annotations

from timegraph.storage.neo4j_client import get_session


_CREATE_CONSTRAINT = (
    "CREATE CONSTRAINT gw_node_unique IF NOT EXISTS "
    "FOR (n:GwNode) REQUIRE (n.name, n.gid) IS UNIQUE"
)
_CREATE_INDEX = (
    "CREATE INDEX gw_node_gid IF NOT EXISTS FOR (n:GwNode) ON (n.gid)"
)


async def ensure_schema() -> None:
    """Idempotent — apply the GwNode constraint + supporting indexes."""
    async with get_session() as s:
        await s.run(_CREATE_CONSTRAINT)
        await s.run(_CREATE_INDEX)


async def load_graph(gid: str, edges: list[tuple[str, str]]) -> tuple[int, int]:
    """Bulk-load one task's graph. Returns (n_nodes, n_unique_edges)."""
    nodes = sorted({n for pair in edges for n in pair})
    # Dedupe edges; GraphWalks dataset contains duplicates that don't affect
    # query results but do bloat the store.
    unique_edges = sorted({(a, b) for a, b in edges})

    async with get_session() as s:
        await s.run(
            "UNWIND $names AS name MERGE (:GwNode {name: name, gid: $gid})",
            parameters={"names": nodes, "gid": gid},
        )
        if unique_edges:
            await s.run(
                "UNWIND $pairs AS pair "
                "MATCH (a:GwNode {name: pair[0], gid: $gid}) "
                "MATCH (b:GwNode {name: pair[1], gid: $gid}) "
                "MERGE (a)-[:EDGE]->(b)",
                parameters={"pairs": [list(p) for p in unique_edges], "gid": gid},
            )
    return len(nodes), len(unique_edges)


async def delete_graph(gid: str) -> None:
    """Remove all nodes + relationships for one task."""
    async with get_session() as s:
        await s.run(
            "MATCH (n:GwNode {gid: $gid}) DETACH DELETE n",
            parameters={"gid": gid},
        )


async def cypher_bfs_frontier(gid: str, start: str, depth: int) -> set[str]:
    """Return nodes whose SHORTEST distance from `start` is exactly `depth`.

    Matches GraphWalks BFS semantics: 'only nodes both reachable and exactly
    at that depth (not nodes at intermediate depths), do not return the
    starting node.'

    Note: Cypher variable-length bounds (`*1..N`) cannot be parameterized,
    so `depth` is interpolated. Safe — `depth` is an int we control.
    """
    if depth < 1:
        return set()
    # shortestPath returns one path per (start, n) pair. We materialize its
    # length via WITH before filtering — Cypher otherwise pushes a bare
    # WHERE on the path into the path-finding constraints instead of treating
    # it as a post-filter, and you get all reachable nodes back.
    query = (
        f"MATCH (start:GwNode {{name: $start, gid: $gid}}) "
        f"MATCH (n:GwNode {{gid: $gid}}) WHERE n.name <> $start "
        f"MATCH p = shortestPath((start)-[:EDGE*..{depth}]->(n)) "
        f"WITH n, length(p) AS d "
        f"WHERE d = $depth "
        f"RETURN collect(DISTINCT n.name) AS names"
    )
    async with get_session() as s:
        result = await s.run(query, parameters={"start": start, "gid": gid, "depth": depth})
        rec = await result.single()
        return set(rec["names"]) if rec else set()


async def cypher_parents(gid: str, node: str) -> set[str]:
    """Return nodes with a direct edge → `node`. Excludes the node itself
    (filters self-loops per GraphWalks parents semantics)."""
    async with get_session() as s:
        result = await s.run(
            "MATCH (p:GwNode {gid: $gid})-[:EDGE]->(n:GwNode {name: $node, gid: $gid}) "
            "WHERE p.name <> $node "
            "RETURN collect(DISTINCT p.name) AS names",
            parameters={"node": node, "gid": gid},
        )
        rec = await result.single()
        return set(rec["names"]) if rec else set()
