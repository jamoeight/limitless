"""fuse — propose (and optionally apply) supersession merges.

Wave 3 ships dry_run=True only. The dry-run path scans the FACT edges in a
scope+time window for `subject+predicate` groups with multiple objects, and
proposes that the newest valid_at supersedes the others.

The actual commit path (dry_run=False, with optional LLM-validated merge) is
deferred to a follow-up: it depends on a `precomputed_supersession_cache`
written by the consolidate_background job (Wave 4).
"""

from __future__ import annotations

from datetime import datetime, timezone

from timegraph.storage.neo4j_client import get_session
from timegraph.types import FuseIn, FuseOut


# Group active FACT edges by (subject, predicate) within scope; surface groups
# with ≥2 distinct objects. For each such group we propose:
#   - merge: the newest fact survives; the older sibling(s) are tagged.
#   - supersession: the newest "supersedes" the immediately-previous one.
_SCAN = """
MATCH ()-[r:FACT]->()
WHERE r.group_id = $scope
  AND r.valid_at <= $t_end
  AND ($t_start IS NULL OR r.valid_at >= $t_start)
  AND (r.invalid_at IS NULL OR r.invalid_at > $t_start_or_min)
WITH r.subject AS subject, r.predicate AS predicate,
     collect({
       fact_id: r.id, object: r.object, valid_at: r.valid_at, confidence: r.confidence
     }) AS edges
WHERE size(edges) >= 2
RETURN subject, predicate, edges
"""


async def fuse(payload: FuseIn) -> FuseOut:
    """Dry-run scan + supersession proposals. Returns proposals, never writes."""
    if not payload.dry_run:
        # Commit path is Wave 4+; surface clearly rather than silently no-op.
        raise NotImplementedError(
            "fuse(dry_run=False) is deferred to Wave 4 (depends on consolidate_background)."
        )

    t_end = payload.time_window_end or datetime.now(timezone.utc)
    t_start = payload.time_window_start
    t_start_or_min = t_start or datetime(1900, 1, 1, tzinfo=timezone.utc)

    proposed_merges: list[dict] = []
    proposed_supersessions: list[dict] = []

    async with get_session() as session:
        res = await session.run(
            _SCAN,
            parameters={
                "scope": payload.scope,
                "t_end": t_end,
                "t_start": t_start,
                "t_start_or_min": t_start_or_min,
            },
        )
        async for r in res:
            data = r.data()
            edges = data["edges"]
            # Sort by valid_at ascending; newest is the "winner".
            edges_sorted = sorted(
                edges,
                key=lambda e: e["valid_at"].to_native() if hasattr(e["valid_at"], "to_native") else e["valid_at"],
            )
            winner = edges_sorted[-1]
            losers = edges_sorted[:-1]

            for loser in losers:
                proposed_supersessions.append({
                    "older_fact_id": loser["fact_id"],
                    "newer_fact_id": winner["fact_id"],
                    "subject": data["subject"],
                    "predicate": data["predicate"],
                    "older_object": loser["object"],
                    "newer_object": winner["object"],
                })

            # A "merge" proposal collapses the equivalence class (same s+p) into
            # the winner — informational, not yet applied.
            proposed_merges.append({
                "subject": data["subject"],
                "predicate": data["predicate"],
                "winning_fact_id": winner["fact_id"],
                "winning_object": winner["object"],
                "losing_fact_ids": [e["fact_id"] for e in losers],
            })

    return FuseOut(
        proposed_merges=proposed_merges,
        proposed_supersessions=proposed_supersessions,
        applied=False,
        cache_hit_ratio=0.0,  # no cache used in dry-run path
    )
