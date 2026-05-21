"""Wave 2 smoke — add_episode (asserted + extracted), attest, invalidate,
graph_query nodes/flat. Exercises the LLM extractor + embedder + Qdrant.

Run with LM Studio + Neo4j + Qdrant all up. Idempotent: writes are scoped
to group_id="smoke_wave2" and the script wipes that group at start + end.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

from timegraph.config import get_settings
from timegraph.ops.add_episode import add_episode
from timegraph.ops.attest import attest
from timegraph.ops.graph_query import graph_query
from timegraph.ops.invalidate import invalidate
from timegraph.storage.neo4j_client import close_driver, get_session
from timegraph.storage.qdrant_client import (
    close_client as close_qdrant,
    ensure_collections,
    get_client as get_qdrant,
)
from timegraph.types import (
    AddEpisodeIn,
    AttestIn,
    Fact,
    GraphQueryIn,
    InvalidateIn,
)

GROUP = "smoke_wave2"
SESSION = "smoke_w2_session"


async def _wipe_group() -> None:
    """Clean Neo4j + Qdrant for this group."""
    s = get_settings()
    async with get_session() as session:
        await session.run("MATCH (e:Episode {group_id: $g}) DETACH DELETE e", g=GROUP)
        await session.run("MATCH (n:Entity {group_id: $g}) DETACH DELETE n", g=GROUP)
        await session.run("MATCH ()-[r:FACT]->() WHERE r.group_id = $g DELETE r", g=GROUP)

    # Qdrant: delete-by-filter for the group, in both collections.
    from qdrant_client.http import models as qm
    client = await get_qdrant()
    flt = qm.Filter(must=[qm.FieldCondition(key="group_id", match=qm.MatchValue(value=GROUP))])
    for coll in (s.qdrant_facts_collection, s.qdrant_episodes_collection):
        try:
            await client.delete(collection_name=coll, points_selector=qm.FilterSelector(filter=flt))
        except Exception:
            # Collection may not exist yet on first run.
            pass


async def step_asserted_episode() -> str:
    """Bypass extractor for speed/determinism; pass facts directly."""
    print()
    print("[1/5] add_episode with asserted_facts (bypass extractor)")
    t = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
    facts = [
        Fact(fact_id="ignored1", subject="bob", predicate="works_at", object="anthropic",
             valid_at=t, confidence=0.95, tier="T2"),
        Fact(fact_id="ignored2", subject="bob", predicate="prefers", object="dark_roast",
             valid_at=t, confidence=0.85, tier="T2"),
    ]
    out = await add_episode(AddEpisodeIn(
        content="Bob works at Anthropic on the Claude team. Bob prefers dark roast coffee.",
        source="smoke", group_id=GROUP, session_id=SESSION, event_time=t,
        asserted_facts=facts,
    ))
    print(f"  -> episode_id={out.episode_id[:8]} extracted={len(out.extracted_facts)} conflicts={len(out.conflicts_detected)}")
    assert len(out.extracted_facts) == 2
    assert out.conflicts_detected == []
    print("  [PASS] asserted-facts path")
    return out.episode_id


async def step_semantic_nodes() -> None:
    print()
    print("[2/5] graph_query mode='nodes' (semantic): 'where does bob work?' → top hit should mention anthropic")
    out = await graph_query(GraphQueryIn(
        query="where does bob work",
        scope=GROUP,
        mode="nodes",
        k=4,
        tier_filter=["T1", "T2"],
        budget_tokens=256,
    ))
    print(f"  -> {len(out.results)} fact(s), tokens_used={out.tokens_used}")
    for r in out.results:
        print(f"     ({r['subject']}, {r['predicate']}, {r['object']})  tier={r['tier']}")
    assert len(out.results) >= 1, "expected at least 1 semantic hit"
    top = out.results[0]
    assert top["subject"] == "bob" and top["object"] == "anthropic", (
        f"top semantic hit should be the works_at fact; got {top}"
    )
    print("  [PASS] semantic nodes")


async def step_flat_mode() -> None:
    print()
    print("[3/5] graph_query mode='flat': coffee preference query → flat string")
    out = await graph_query(GraphQueryIn(
        query="coffee preferences",
        scope=GROUP,
        mode="flat",
        k=4,
        tier_filter=["T1", "T2"],
        budget_tokens=256,
    ))
    print(f"  -> {len(out.results)} string(s), tokens_used={out.tokens_used}")
    for r in out.results:
        print(f"     {r['text']}")
    assert len(out.results) >= 1
    # Top hit should be the dark_roast fact
    assert "dark_roast" in out.results[0]["text"], out.results[0]
    print("  [PASS] flat mode")


async def step_attest_and_invalidate() -> None:
    print()
    print("[4/5] attest 'confirmed' on the works_at fact, then invalidate it")
    out = await graph_query(GraphQueryIn(
        query="anthropic", scope=GROUP, mode="facts",
        time_anchor=datetime(2026, 12, 31, tzinfo=timezone.utc),
        tier_filter=["T1", "T2"], budget_tokens=256,
    ))
    works_at = next(r for r in out.results if r["object"] == "anthropic")
    fact_id = works_at["fact_id"]

    a = await attest(AttestIn(fact_id=fact_id, attestation="confirmed", by="user"))
    print(f"  -> attest: confidence={a.new_confidence:.2f}  pinned={a.pinned}")
    assert a.pinned is True
    assert a.new_confidence >= 0.95

    inv = await invalidate(InvalidateIn(
        fact_id=fact_id, reason="bob moved jobs",
        invalidated_by="manual", by="user",
    ))
    print(f"  -> invalidate: invalid_at={inv.invalid_at}")

    # Default time-windowed query should now drop the works_at fact.
    out2 = await graph_query(GraphQueryIn(
        query="bob", scope=GROUP, mode="facts",
        time_anchor=datetime(2026, 12, 31, tzinfo=timezone.utc),
        tier_filter=["T1", "T2"], budget_tokens=256,
    ))
    remaining_objects = [r["object"] for r in out2.results]
    print(f"  -> post-invalidate facts: {remaining_objects}")
    assert "anthropic" not in remaining_objects, "invalidated fact should be filtered"
    print("  [PASS] attest + invalidate")


async def step_extracted_episode() -> None:
    print()
    print("[5/5] add_episode WITH the extractor (LM Studio Qwen3.5-9B)")
    t = datetime(2026, 5, 20, 14, 0, 0, tzinfo=timezone.utc)
    out = await add_episode(AddEpisodeIn(
        content=(
            "Carol just joined OpenAI as a research engineer. "
            "Her favorite programming language is Rust. "
            "She lives in San Francisco."
        ),
        source="smoke_extractor",
        group_id=GROUP,
        session_id=SESSION,
        event_time=t,
    ))
    print(f"  -> episode_id={out.episode_id[:8]} extracted={len(out.extracted_facts)}")
    for f in out.extracted_facts:
        print(f"     ({f.subject}, {f.predicate}, {f.object})  conf={f.confidence:.2f}")
    assert len(out.extracted_facts) >= 2, (
        f"extractor returned only {len(out.extracted_facts)} facts; expected ≥2"
    )
    print("  [PASS] extractor end-to-end")


async def main() -> int:
    print("=" * 70)
    print("Wave 2 smoke — add_episode (asserted + extracted), attest, invalidate,")
    print("              graph_query nodes/flat, embedder + Qdrant")
    print("=" * 70)
    try:
        await ensure_collections()
        await _wipe_group()
        await step_asserted_episode()
        await step_semantic_nodes()
        await step_flat_mode()
        await step_attest_and_invalidate()
        await step_extracted_episode()
        await _wipe_group()
        print()
        print("=" * 70)
        print("ALL PASS — Wave 2 ops live: extractor + embedder + Qdrant + Neo4j")
        print("=" * 70)
        return 0
    except AssertionError as e:
        print()
        print(f"[FAIL] assertion: {e}")
        return 1
    except Exception as e:
        import traceback
        print()
        print(f"[FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()
        return 2
    finally:
        await close_driver()
        await close_qdrant()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
