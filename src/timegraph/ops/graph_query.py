"""graph_query — Cypher-backed read paths.

Modes:
  - "facts"     : facts whose subject, predicate, or object matches `query`
                  (exact string match on entity name).
  - "neighbors" : facts within `depth` hops of an entity whose name == `query`.
  - "nodes"     : top-k facts ranked by semantic similarity (Qdrant cosine),
                  hydrated from Neo4j for full provenance.
  - "flat"      : same as "nodes" but returns a flat string list, ready for
                  inlining into a model's active context.

Common filters:
  - time_anchor: edges with valid_at <= anchor AND (invalid_at IS NULL OR invalid_at > anchor)
  - tier_filter: only edges whose tier is in the filter list (default T1+T2)
  - include_invalidated: relaxes the time filter
  - budget_tokens: hard cap on returned-string tokens (rough char/4 approximation)
"""

from __future__ import annotations

from datetime import datetime, timezone

from timegraph.config import get_settings
from timegraph.llm.embedder import EmbedderClient
from timegraph.storage.neo4j_client import get_session
from timegraph.storage.qdrant_client import search as qdrant_search
from timegraph.types import GraphQueryIn, GraphQueryOut, Provenance


# ---- Cypher templates --------------------------------------------------

# scope="*" → no group filter; otherwise filter by group_id == scope.
# Tier filter uses IN $tier_filter (we always send the list).
# Time anchor: when include_invalidated is False, we require valid_at <= anchor
# and (invalid_at IS NULL OR invalid_at > anchor). When include_invalidated=True
# the time filter still requires valid_at <= anchor (the fact must exist by then)
# but allows already-invalidated edges through.

_FACTS_BY_STRING = """
MATCH (s:Entity)-[r:FACT]->(o:Entity)
WHERE r.tier IN $tier_filter
  AND ($scope = '*' OR r.group_id = $scope)
  AND (r.subject = $query OR r.object = $query OR r.predicate = $query)
  AND r.valid_at <= $time_anchor
  AND ($include_invalidated OR r.invalid_at IS NULL OR r.invalid_at > $time_anchor)
RETURN r.id            AS fact_id,
       r.subject       AS subject,
       r.predicate     AS predicate,
       r.object        AS object,
       r.valid_at      AS valid_at,
       r.invalid_at    AS invalid_at,
       r.confidence    AS confidence,
       r.tier          AS tier,
       r.session_id    AS session_id,
       r.source        AS source,
       r.source_episode_id AS source_episode_id
ORDER BY r.valid_at DESC
LIMIT $hard_cap
"""

# Neighbors: variable-length walk from an entity matching $query, bounded by
# $depth. The path returns the FACT edges traversed.
_NEIGHBORS = """
MATCH (start:Entity {name: $query})
WHERE $scope = '*' OR start.group_id = $scope
MATCH path = (start)-[rels:FACT*1..%(depth)d]-(end:Entity)
WHERE ALL(r IN rels WHERE
        r.tier IN $tier_filter
    AND r.valid_at <= $time_anchor
    AND ($include_invalidated OR r.invalid_at IS NULL OR r.invalid_at > $time_anchor)
)
UNWIND rels AS r
RETURN DISTINCT
       r.id            AS fact_id,
       r.subject       AS subject,
       r.predicate     AS predicate,
       r.object        AS object,
       r.valid_at      AS valid_at,
       r.invalid_at    AS invalid_at,
       r.confidence    AS confidence,
       r.tier          AS tier,
       r.session_id    AS session_id,
       r.source        AS source,
       r.source_episode_id AS source_episode_id
ORDER BY r.valid_at DESC
LIMIT $hard_cap
"""


# ---- helpers ----------------------------------------------------------


def _approx_tokens(s: str) -> int:
    """Rough char/4 token estimate. Replace with tiktoken once we care about precision."""
    return max(1, len(s) // 4)


def _to_native(x):
    """Convert neo4j.time.DateTime/Date to native datetime; pass others through."""
    # neo4j.time types expose .to_native(); native datetimes do not.
    if x is None:
        return None
    to_native = getattr(x, "to_native", None)
    return to_native() if callable(to_native) else x


def _fact_to_row(r: dict) -> dict:
    """Project a Neo4j row to a plain dict; coerce neo4j DateTime to native."""
    return {
        "fact_id": r["fact_id"],
        "subject": r["subject"],
        "predicate": r["predicate"],
        "object": r["object"],
        "valid_at": _to_native(r["valid_at"]),
        "invalid_at": _to_native(r["invalid_at"]),
        "confidence": r["confidence"],
        "tier": r["tier"],
        "session_id": r["session_id"],
    }


def _row_provenance(r: dict) -> Provenance:
    """Build a Provenance from a fact-row's metadata."""
    return Provenance(
        episode_id=r.get("source_episode_id"),
        ts=_to_native(r["valid_at"]),
        source=r["source"] or "unknown",
        session_id=r["session_id"],
    )


# ---- entrypoint -------------------------------------------------------


# Hard upper bound on rows returned by any single Cypher query; budget_tokens
# then trims this list further. Keeps Cypher latency bounded.
_HARD_ROW_CAP = 256


# Hydrate a list of fact ids back into full rows from Neo4j, preserving the
# input order (so semantic-search ranking is respected). We also apply tier +
# time filters here so post-Qdrant filtering matches the symbolic path's semantics.
_HYDRATE_FACTS = """
MATCH ()-[r:FACT]->()
WHERE r.id IN $fact_ids
  AND r.tier IN $tier_filter
  AND r.valid_at <= $time_anchor
  AND ($include_invalidated OR r.invalid_at IS NULL OR r.invalid_at > $time_anchor)
RETURN r.id            AS fact_id,
       r.subject       AS subject,
       r.predicate     AS predicate,
       r.object        AS object,
       r.valid_at      AS valid_at,
       r.invalid_at    AS invalid_at,
       r.confidence    AS confidence,
       r.tier          AS tier,
       r.session_id    AS session_id,
       r.source        AS source,
       r.source_episode_id AS source_episode_id
"""


async def graph_query(payload: GraphQueryIn) -> GraphQueryOut:
    """Dispatch on mode. Symbolic (facts/neighbors) vs semantic (nodes/flat)."""
    time_anchor = payload.time_anchor or datetime.now(timezone.utc)

    if payload.mode in ("nodes", "flat"):
        return await _semantic_query(payload, time_anchor)

    params = {
        "query": payload.query,
        "scope": payload.scope,
        "time_anchor": time_anchor,
        "include_invalidated": payload.include_invalidated,
        "tier_filter": payload.tier_filter,
        "hard_cap": _HARD_ROW_CAP,
    }

    async with get_session() as session:
        if payload.mode == "facts":
            res = await session.run(_FACTS_BY_STRING, parameters=params)
        else:  # neighbors
            # depth is a literal not a parameter — Neo4j doesn't allow parameter
            # in variable-length pattern bounds. Clamp aggressively to avoid OOM.
            depth = max(1, min(payload.depth, 4))
            res = await session.run(_NEIGHBORS % {"depth": depth}, parameters=params)
        rows = [r.data() async for r in res]

    return _project_with_budget(rows, payload, mode=payload.mode)


async def _semantic_query(payload: GraphQueryIn, time_anchor: datetime) -> GraphQueryOut:
    """Embed the query, top-k via Qdrant, hydrate from Neo4j, project with budget."""
    s = get_settings()
    # k is over-fetched a little so post-filter (tier/time/invalidated) loss doesn't
    # silently empty the result. The token budget will trim back down anyway.
    over_k = max(payload.k, 8) * 3

    embedder = EmbedderClient()
    try:
        qvec = await embedder.embed_one(payload.query)
    finally:
        await embedder.close()

    hits = await qdrant_search(
        collection=s.qdrant_facts_collection,
        vector=qvec,
        k=over_k,
        group_id=payload.scope,
        tier_filter=payload.tier_filter,
    )
    ranked_ids: list[str] = [str(h.id) for h in hits]
    if not ranked_ids:
        return GraphQueryOut(mode=payload.mode, results=[], provenance=[], tokens_used=0)

    async with get_session() as session:
        res = await session.run(
            _HYDRATE_FACTS,
            parameters={
                "fact_ids": ranked_ids,
                "tier_filter": payload.tier_filter,
                "time_anchor": time_anchor,
                "include_invalidated": payload.include_invalidated,
            },
        )
        rows_by_id: dict[str, dict] = {}
        async for r in res:
            d = r.data()
            rows_by_id[d["fact_id"]] = d

    # Preserve Qdrant ranking order.
    ordered_rows = [rows_by_id[i] for i in ranked_ids if i in rows_by_id]
    return _project_with_budget(ordered_rows, payload, mode=payload.mode)


def _project_with_budget(rows: list[dict], payload: GraphQueryIn, mode: str) -> GraphQueryOut:
    """Apply the token budget and shape the response per mode."""
    results: list[dict] = []
    provenance: list[Provenance] = []
    flat_strings: list[str] = []
    tokens_used = 0
    for r in rows:
        text = f"({r['subject']}, {r['predicate']}, {r['object']})"
        cost = _approx_tokens(text)
        if tokens_used + cost > payload.budget_tokens:
            break
        if mode == "flat":
            flat_strings.append(text)
            results.append({"text": text, "fact_id": r["fact_id"]})
        else:
            results.append(_fact_to_row(r))
        provenance.append(_row_provenance(r))
        tokens_used += cost

    return GraphQueryOut(
        mode=mode,  # type: ignore[arg-type]
        results=results,
        provenance=provenance,
        tokens_used=tokens_used,
    )
