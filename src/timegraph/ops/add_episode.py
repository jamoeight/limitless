"""add_episode — write an Episode + extract facts + index for retrieval.

Flow:
  1. CREATE the :Episode node in Neo4j
  2. EITHER use `asserted_facts` from the caller, OR call the extractor (Qwen3.5-9B)
  3. For each fact, call `add_fact` (which detects conflicts in one tx)
  4. Embed episode content + every fact text via the embedder client
  5. Upsert episode embedding -> Qdrant `episodes` collection
  6. Upsert each fact embedding -> Qdrant `facts` collection

Errors in the extractor are treated as "no facts extracted" (the episode is
still recorded). Errors in the embedder propagate — vector indexing is
load-bearing for Wave 2 `graph_query` nodes/flat modes.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog

from timegraph.config import get_settings
from timegraph.llm.embedder import EmbedderClient
from timegraph.llm.extractor import ExtractorClient
from timegraph.ops.add_fact import add_fact
from timegraph.storage.neo4j_client import get_session
from timegraph.storage.qdrant_client import upsert_point
from timegraph.types import AddEpisodeIn, AddEpisodeOut, AddFactIn, Fact

log = structlog.get_logger(__name__)


_CREATE_EPISODE = """
CREATE (e:Episode {
  id: $episode_id,
  content: $content,
  source: $source,
  session_id: $session_id,
  group_id: $group_id,
  event_time: $event_time,
  created_at: datetime()
})
RETURN e.id AS episode_id
"""


async def _create_episode_node(
    episode_id: str,
    content: str,
    source: str,
    session_id: str,
    group_id: str,
    event_time: datetime,
) -> None:
    async with get_session() as session:
        await session.run(
            _CREATE_EPISODE,
            episode_id=episode_id,
            content=content,
            source=source,
            session_id=session_id,
            group_id=group_id,
            event_time=event_time,
        )


def _fact_text(f: Fact) -> str:
    """Tokenization-stable text form for embedding."""
    return f"{f.subject} {f.predicate} {f.object}"


async def add_episode(payload: AddEpisodeIn) -> AddEpisodeOut:
    s = get_settings()
    episode_id = str(uuid.uuid4())
    event_time = payload.event_time or datetime.now(timezone.utc)

    # 1) Episode node
    await _create_episode_node(
        episode_id=episode_id,
        content=payload.content,
        source=payload.source,
        session_id=payload.session_id,
        group_id=payload.group_id,
        event_time=event_time,
    )

    # 2) Source the facts: either asserted, or extract.
    extractor: ExtractorClient | None = None
    if payload.asserted_facts is not None:
        extracted: list[Fact] = list(payload.asserted_facts)
    else:
        extractor = ExtractorClient()
        try:
            facts, latency_ms = await extractor.extract_facts(
                episode_content=payload.content,
                event_time=event_time,
                session_id=payload.session_id,
                source=payload.source,
            )
            log.info("extracted", n=len(facts), latency_ms=latency_ms)
            extracted = facts
        except Exception as e:
            log.warning("extractor failed; recording episode with 0 facts", error=str(e))
            extracted = []

    # 3) add_fact for each — collect conflicts.
    persisted: list[Fact] = []
    conflicts: list[str] = []
    for f in extracted:
        out = await add_fact(AddFactIn(
            subject=f.subject,
            predicate=f.predicate,
            object=f.object,
            event_time=event_time,
            source=payload.source,
            session_id=payload.session_id,
            group_id=payload.group_id,
            confidence=f.confidence,
            tier=f.tier,
            pinned=f.pinned,
            source_episode_id=episode_id,
        ))
        # Reflect the assigned id back into the Fact for the caller's view.
        persisted.append(Fact(
            fact_id=out.fact_id,
            subject=f.subject,
            predicate=f.predicate,
            object=f.object,
            valid_at=event_time,
            invalid_at=None,
            confidence=f.confidence,
            pinned=f.pinned,
            tier=f.tier,
            source_episode_id=episode_id,
            session_id=payload.session_id,
            sources=[payload.source],
        ))
        conflicts.extend(out.conflicts_with)

    # 4) Embed + upsert vectors.
    embedder = EmbedderClient()
    try:
        ep_vec = await embedder.embed_one(payload.content)
        await upsert_point(
            collection=s.qdrant_episodes_collection,
            point_id=episode_id,
            vector=ep_vec,
            payload={
                "episode_id": episode_id,
                "group_id": payload.group_id,
                "session_id": payload.session_id,
                "source": payload.source,
                "event_time": event_time.isoformat(),
            },
        )
        if persisted:
            texts = [_fact_text(f) for f in persisted]
            vecs = await embedder.embed_many(texts)
            for f, v in zip(persisted, vecs, strict=True):
                await upsert_point(
                    collection=s.qdrant_facts_collection,
                    point_id=f.fact_id,
                    vector=v,
                    payload={
                        "fact_id": f.fact_id,
                        "group_id": payload.group_id,
                        "subject": f.subject,
                        "predicate": f.predicate,
                        "object": f.object,
                        "tier": f.tier,
                        "valid_at": event_time.isoformat(),
                        "source_episode_id": episode_id,
                    },
                )
    finally:
        await embedder.close()
        if extractor is not None:
            await extractor.close()

    return AddEpisodeOut(
        episode_id=episode_id,
        extracted_facts=persisted,
        conflicts_detected=conflicts,
    )
