"""delete — remove a FACT edge or an Episode node by id.

No LLM call. For type=fact: detach-delete the edge. For type=episode: optionally
cascade-delete extracted facts (when cascade=True). Returns the list of fact
ids that were deleted as a side-effect.

Qdrant cleanup (deleting the vector for the same id) is the caller's
responsibility for now — added in Wave 2 when add_episode wires the vector path.
"""

from __future__ import annotations

from timegraph.storage.neo4j_client import get_session
from timegraph.types import DeleteIn, DeleteOut

_DELETE_FACT = """
MATCH ()-[r:FACT {id: $target_id}]->()
DELETE r
RETURN count(r) AS deleted
"""

_DELETE_EPISODE_NO_CASCADE = """
MATCH (e:Episode {id: $target_id})
DETACH DELETE e
RETURN count(e) AS deleted
"""

# Cascade: also delete any FACT edges marked as extracted from this episode.
# (The :EXTRACTED relationship is created by add_episode in Wave 2; until then
# this Cypher simply finds no extracted facts and only deletes the episode.)
_DELETE_EPISODE_CASCADE = """
MATCH (e:Episode {id: $target_id})
OPTIONAL MATCH ()-[r:FACT]->()
  WHERE r.source_episode_id = $target_id
WITH e, collect(r.id) AS fact_ids, collect(r) AS facts
FOREACH (f IN facts | DELETE f)
DETACH DELETE e
RETURN fact_ids AS cascade_affected
"""


async def delete(payload: DeleteIn) -> DeleteOut:
    """Delete a fact edge or an episode node (optionally cascading to its facts)."""
    async with get_session() as session:
        if payload.type == "fact":
            res = await session.run(_DELETE_FACT, target_id=payload.target_id)
            row = await res.single()
            ok = (row["deleted"] if row else 0) > 0
            return DeleteOut(ok=ok, cascade_affected=[])

        if payload.cascade:
            res = await session.run(_DELETE_EPISODE_CASCADE, target_id=payload.target_id)
            row = await res.single()
            affected = list(row["cascade_affected"]) if row else []
            return DeleteOut(ok=True, cascade_affected=affected)

        res = await session.run(_DELETE_EPISODE_NO_CASCADE, target_id=payload.target_id)
        row = await res.single()
        ok = (row["deleted"] if row else 0) > 0
        return DeleteOut(ok=ok, cascade_affected=[])
