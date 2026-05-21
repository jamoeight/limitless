"""One-off: seed a 10K graph and print the actual pair scores stage-1 sees.

This isolates whether Fix A's threshold is working as intended or whether
the noise pair scores are competing with alice's for relevance."""

from __future__ import annotations

import asyncio
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

from bench.infer_scale import (
    GROUP, SESSION, _bulk_neo4j, _bulk_qdrant, _make_noise_facts,
    _seed_hot_via_episodes, wipe_group,
)
from timegraph.config import get_settings
from timegraph.llm.embedder import EmbedderClient
from timegraph.ops.infer import _HYDRATE_AND_SIBLINGS, _row_to_fact
from timegraph.storage.neo4j_client import close_driver, get_session
from timegraph.storage.qdrant_client import (
    close_client as close_qdrant, ensure_collections, search as qdrant_search,
)


async def main() -> None:
    t_now = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
    N = 10000
    await ensure_collections()
    await wipe_group()
    print(f"seeding N={N}...")
    await _seed_hot_via_episodes(t_now)
    rng = random.Random(0xC0DE + N)
    noise = _make_noise_facts(N - 5, t_now, rng)
    await _bulk_neo4j(noise)
    await _bulk_qdrant(noise)
    print("seeded.")

    s = get_settings()
    query = "where does alice live now"

    # Embed the query
    embedder = EmbedderClient()
    qvec = await embedder.embed_one(query)
    await embedder.close()

    # Qdrant top-16
    hits = await qdrant_search(
        collection=s.qdrant_facts_collection,
        vector=qvec, k=16, tier_filter=["T1", "T2"],
    )
    print(f"\nQdrant top-16 for '{query}':")
    for i, h in enumerate(hits):
        p = h.payload or {}
        print(f"  [{i:2d}] score={h.score:.4f}  ({p.get('subject')}, {p.get('predicate')}, {p.get('object')})")

    score_by_id = {str(h.id): float(h.score) for h in hits}
    seed_ids = list(score_by_id.keys())

    # Collect pairs
    async with get_session() as session:
        res = await session.run(
            _HYDRATE_AND_SIBLINGS,
            parameters={
                "seed_ids": seed_ids,
                "tier_filter": ["T1", "T2"],
                "time_anchor": t_now,
            },
        )
        pairs = []
        seen = set()
        async for r in res:
            seed = _row_to_fact(r["seed"])
            for sib in r["siblings"]:
                sib_f = _row_to_fact(sib)
                key = tuple(sorted([seed.fact_id, sib_f.fact_id]))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((seed, sib_f))

    scored = [
        (pair, max(score_by_id.get(pair[0].fact_id, 0.0),
                   score_by_id.get(pair[1].fact_id, 0.0)))
        for pair in pairs
    ]
    scored.sort(key=lambda x: -x[1])
    print(f"\nAll {len(scored)} pairs found, ranked by query-relevance score:")
    for i, (pair, sc) in enumerate(scored):
        a, b = pair
        print(f"  [{i:2d}] score={sc:.4f}  ({a.subject}, {a.predicate}, {a.object}) "
              f"vs ({b.subject}, {b.predicate}, {b.object})")

    if scored:
        best = scored[0][1]
        floor = best * 0.6
        print(f"\nbest={best:.4f}  floor (0.6*best)={floor:.4f}")
        passing = [(p, sc) for p, sc in scored if sc >= floor]
        print(f"pairs surviving threshold: {len(passing)}")

    await close_driver()
    await close_qdrant()


if __name__ == "__main__":
    asyncio.run(main())
