"""attest — boost confidence and (optionally) pin a FACT edge.

Wave 2 simple-attest (B.4-v2):
  - `attestation="confirmed"` -> confidence -> min(1.0, confidence + 0.2), pinned=True
  - `attestation="corrected"` -> confidence -> max(0.0, confidence - 0.4), pinned=False
  - any other string         -> confidence unchanged, pinned unchanged (no-op)

The Phase-2 `attest_quorum` op composes this with phrase classifier + agent
confidence + recency checks for tier promotion.
"""

from __future__ import annotations

from timegraph.storage.neo4j_client import get_session
from timegraph.types import AttestIn, AttestOut

_UPDATE_CONFIRMED = """
MATCH ()-[r:FACT {id: $fact_id}]->()
SET r.confidence = CASE WHEN r.confidence + 0.2 > 1.0 THEN 1.0 ELSE r.confidence + 0.2 END,
    r.pinned = true,
    r.last_attested_at = datetime(),
    r.last_attested_by = $by
RETURN r.confidence AS confidence, r.pinned AS pinned
"""

_UPDATE_CORRECTED = """
MATCH ()-[r:FACT {id: $fact_id}]->()
SET r.confidence = CASE WHEN r.confidence - 0.4 < 0.0 THEN 0.0 ELSE r.confidence - 0.4 END,
    r.pinned = false,
    r.last_attested_at = datetime(),
    r.last_attested_by = $by
RETURN r.confidence AS confidence, r.pinned AS pinned
"""

_READ_ONLY = """
MATCH ()-[r:FACT {id: $fact_id}]->()
RETURN r.confidence AS confidence, r.pinned AS pinned
"""


async def attest(payload: AttestIn) -> AttestOut:
    """Apply the attestation; return new confidence + pinned status."""
    if payload.attestation == "confirmed":
        cypher = _UPDATE_CONFIRMED
    elif payload.attestation == "corrected":
        cypher = _UPDATE_CORRECTED
    else:
        cypher = _READ_ONLY

    async with get_session() as session:
        res = await session.run(cypher, fact_id=payload.fact_id, by=payload.by)
        row = await res.single()

    if row is None:
        raise ValueError(f"no fact with id={payload.fact_id}")
    return AttestOut(new_confidence=float(row["confidence"]), pinned=bool(row["pinned"]))
