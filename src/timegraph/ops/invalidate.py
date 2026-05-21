"""invalidate — set invalid_at on a FACT edge.

No LLM. Sets invalid_at = now() unless already set. Records the reason +
invalidator on the edge for provenance.

This is how stale facts are tombstoned. After invalidate(), reads that don't
set include_invalidated=True will skip the edge.
"""

from __future__ import annotations

from datetime import datetime, timezone

from timegraph.storage.neo4j_client import get_session
from timegraph.types import InvalidateIn, InvalidateOut

_INVALIDATE = """
MATCH ()-[r:FACT {id: $fact_id}]->()
SET r.invalid_at = COALESCE(r.invalid_at, $now),
    r.invalidate_reason = $reason,
    r.invalidated_by = $invalidated_by,
    r.invalidated_actor = $by
RETURN r.invalid_at AS invalid_at
"""


async def invalidate(payload: InvalidateIn) -> InvalidateOut:
    now = datetime.now(timezone.utc)
    async with get_session() as session:
        res = await session.run(
            _INVALIDATE,
            fact_id=payload.fact_id,
            now=now,
            reason=payload.reason,
            invalidated_by=payload.invalidated_by,
            by=payload.by,
        )
        row = await res.single()
    if row is None:
        raise ValueError(f"no fact with id={payload.fact_id}")
    raw = row["invalid_at"]
    # Convert neo4j.time.DateTime -> native if needed
    invalid_at = raw.to_native() if hasattr(raw, "to_native") else raw
    return InvalidateOut(fact_id=payload.fact_id, invalid_at=invalid_at)
