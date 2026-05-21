"""add_fact — create a FACT edge between two Entity nodes + detect conflicts.

No LLM call. Single transaction: MERGE both entities, CREATE the edge, then
query for conflicting edges (same subject+predicate, different object,
overlapping valid windows).

Conflict windows: edge A and edge B conflict iff
  - r_a.subject == r_b.subject
  - r_a.predicate == r_b.predicate
  - r_a.object != r_b.object
  - [r_a.valid_at, r_a.invalid_at) overlaps [r_b.valid_at, r_b.invalid_at)
    where invalid_at IS NULL means +inf

This op does NOT auto-invalidate conflicts — caller (typically infer()
or higher-level add_episode flow) decides what to do with the conflict set.
"""

from __future__ import annotations

import uuid

from timegraph.storage.neo4j_client import get_session
from timegraph.types import AddFactIn, AddFactOut

# Cypher: MERGE entities, CREATE the edge with all props. RETURN the edge id.
_CREATE_FACT = """
MERGE (s:Entity {name: $subject, group_id: $group_id})
  ON CREATE SET s.last_updated = datetime()
MERGE (o:Entity {name: $object, group_id: $group_id})
  ON CREATE SET o.last_updated = datetime()
CREATE (s)-[r:FACT {
  id: $fact_id,
  subject: $subject,
  predicate: $predicate,
  object: $object,
  valid_at: $valid_at,
  invalid_at: NULL,
  confidence: $confidence,
  pinned: $pinned,
  tier: $tier,
  group_id: $group_id,
  source: $source,
  session_id: $session_id,
  source_episode_id: $source_episode_id
}]->(o)
RETURN r.id AS fact_id
"""

# Conflict detection: other edges sharing (subject, predicate) but with different
# object, where the [valid_at, invalid_at) windows overlap. Run inside the same
# transaction so we observe edges created earlier in this tx (we shouldn't, but
# defensive) and don't race with concurrent inserts (Neo4j gives us snapshot
# isolation per-tx).
_FIND_CONFLICTS = """
MATCH ()-[r:FACT]->()
WHERE r.id <> $fact_id
  AND r.group_id = $group_id
  AND r.subject = $subject
  AND r.predicate = $predicate
  AND r.object <> $object
  AND (r.invalid_at IS NULL OR r.invalid_at > $valid_at)
RETURN r.id AS conflict_id
"""


async def add_fact(payload: AddFactIn) -> AddFactOut:
    """Create a FACT edge; return its id and any conflicting fact ids."""
    fact_id = str(uuid.uuid4())
    params = {
        "fact_id": fact_id,
        "subject": payload.subject,
        "predicate": payload.predicate,
        "object": payload.object,
        "group_id": payload.group_id,
        "valid_at": payload.event_time,
        "confidence": payload.confidence,
        "pinned": payload.pinned,
        "tier": payload.tier,
        "source": payload.source,
        "session_id": payload.session_id,
        "source_episode_id": payload.source_episode_id,
    }

    async with get_session() as session:
        async def _tx(tx):
            create_res = await tx.run(_CREATE_FACT, **params)
            created = await create_res.single()
            assert created is not None, "CREATE returned no row"
            conflict_res = await tx.run(_FIND_CONFLICTS, **params)
            conflicts = [row["conflict_id"] async for row in conflict_res]
            return created["fact_id"], conflicts

        created_id, conflicts = await session.execute_write(_tx)

    return AddFactOut(
        fact_id=created_id,
        edges_created=1,
        conflicts_with=conflicts,
    )
