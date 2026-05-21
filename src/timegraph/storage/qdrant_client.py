"""Async Qdrant client wrapper + collection bootstrapping.

Wave 2 collections:
  - episodes : one point per Episode node, payload {episode_id, group_id, session_id, source}
  - facts    : one point per FACT edge, payload {fact_id, group_id, subject, predicate, object, tier, valid_at_iso}

Dim comes from `Settings.embedder_dim`. Mismatched dim -> 400 on insert.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from timegraph.config import get_settings

log = structlog.get_logger(__name__)

_client: AsyncQdrantClient | None = None
_client_lock = asyncio.Lock()


async def get_client() -> AsyncQdrantClient:
    """Lazy singleton. Prefers gRPC for binary payloads (no JSON inflation)."""
    global _client
    if _client is not None:
        return _client
    async with _client_lock:
        if _client is None:
            s = get_settings()
            # prefer_grpc routes upsert/query through port 6334; the HTTP URL
            # is still used for collection-management ops. ~5-10× faster on
            # bulk upsert because vectors stay binary instead of float→string.
            _client = AsyncQdrantClient(url=s.qdrant_url, prefer_grpc=True)
    return _client


async def close_client() -> None:
    """Close the singleton."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None


async def ensure_collections() -> None:
    """Create episodes + facts collections if absent. Idempotent."""
    s = get_settings()
    client = await get_client()
    existing = {c.name for c in (await client.get_collections()).collections}
    vec = qm.VectorParams(size=s.embedder_dim, distance=qm.Distance.COSINE)
    if s.qdrant_episodes_collection not in existing:
        await client.create_collection(
            collection_name=s.qdrant_episodes_collection, vectors_config=vec,
        )
        log.info("created collection", name=s.qdrant_episodes_collection, dim=s.embedder_dim)
    if s.qdrant_facts_collection not in existing:
        await client.create_collection(
            collection_name=s.qdrant_facts_collection, vectors_config=vec,
        )
        log.info("created collection", name=s.qdrant_facts_collection, dim=s.embedder_dim)


async def upsert_point(
    collection: str, point_id: str, vector: list[float], payload: dict[str, Any]
) -> None:
    """Upsert a single point. Qdrant accepts str/uuid/int IDs natively."""
    client = await get_client()
    await client.upsert(
        collection_name=collection,
        points=[qm.PointStruct(id=point_id, vector=vector, payload=payload)],
    )


async def search(
    collection: str,
    vector: list[float],
    *,
    k: int = 8,
    group_id: str | None = None,
    tier_filter: list[str] | None = None,
    extra_must: list[qm.FieldCondition] | None = None,
) -> list[qm.ScoredPoint]:
    """Cosine-top-k search with optional group/tier filtering.

    Uses `query_points` (qdrant-client >=1.13). Returns the .points list, which
    is a list[ScoredPoint] matching the old `search()` shape.
    """
    client = await get_client()
    must: list[qm.FieldCondition] = list(extra_must or [])
    if group_id and group_id != "*":
        must.append(qm.FieldCondition(key="group_id", match=qm.MatchValue(value=group_id)))
    if tier_filter:
        must.append(qm.FieldCondition(key="tier", match=qm.MatchAny(any=list(tier_filter))))
    flt = qm.Filter(must=must) if must else None
    resp = await client.query_points(
        collection_name=collection,
        query=vector,
        limit=k,
        query_filter=flt,
        with_payload=True,
    )
    return resp.points


async def delete_points(collection: str, point_ids: list[str]) -> None:
    if not point_ids:
        return
    client = await get_client()
    await client.delete(
        collection_name=collection,
        points_selector=qm.PointIdsList(points=point_ids),
    )
