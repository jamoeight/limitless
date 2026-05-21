"""claim / release — Neo4j-backed exclusive lock with TTL.

A `:Lock {resource_id, holder, expires_at}` node. claim() creates it iff none
exists or the existing one has expired. release() deletes it.

Solo-dev runtime has no real concurrency, but ship the primitive — every op
that mutates shared state goes through this in production.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from timegraph.storage.neo4j_client import get_session
from timegraph.types import ClaimIn, ClaimOut, ReleaseIn, ReleaseOut

# Atomic claim:
#   - If a non-expired Lock for this resource exists -> return granted=false
#   - Else upsert the Lock with our holder + new expiry
_CLAIM = """
OPTIONAL MATCH (l:Lock {resource_id: $resource_id})
WITH l, $now AS now
WHERE l IS NULL OR l.expires_at <= now
MERGE (m:Lock {resource_id: $resource_id})
  ON CREATE SET m.holder = $holder, m.expires_at = $expires_at, m.claimed_at = $now
  ON MATCH SET  m.holder = $holder, m.expires_at = $expires_at, m.claimed_at = $now
RETURN m.holder AS holder, m.expires_at AS expires_at
"""

_RELEASE = """
MATCH (l:Lock {resource_id: $resource_id})
DELETE l
RETURN count(l) AS deleted
"""


async def claim(payload: ClaimIn, holder: str | None = None) -> ClaimOut:
    """Try to claim an exclusive lock on a resource. Returns granted=False if held."""
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(milliseconds=payload.ttl_ms)
    holder_id = holder or str(uuid.uuid4())

    async with get_session() as session:
        res = await session.run(
            _CLAIM,
            resource_id=payload.resource_id,
            holder=holder_id,
            now=now,
            expires_at=expires_at,
        )
        row = await res.single()

    if row is None or row["holder"] != holder_id:
        return ClaimOut(granted=False, ttl_ms=0)
    return ClaimOut(granted=True, ttl_ms=payload.ttl_ms)


async def release(payload: ReleaseIn) -> ReleaseOut:
    """Release the lock if present. Idempotent."""
    async with get_session() as session:
        res = await session.run(_RELEASE, resource_id=payload.resource_id)
        row = await res.single()
        deleted = (row["deleted"] if row else 0) > 0
    return ReleaseOut(released=deleted)
