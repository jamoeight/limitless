"""Wave 1 smoke test — exercises add_fact, graph_query, claim/release, delete
against a live Neo4j. No LLM, no Qdrant, no MCP.

Usage:  python scripts/smoke_wave1.py

Exits 0 on full pass, non-zero on any failed assertion. Idempotent: writes are
scoped to group_id="smoke_wave1" and the script wipes that group at start + end.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone

# Force UTF-8 stdout on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

from timegraph.ops.add_fact import add_fact
from timegraph.ops.claim_release import claim, release
from timegraph.ops.delete import delete
from timegraph.ops.graph_query import graph_query
from timegraph.storage.neo4j_client import close_driver, get_session
from timegraph.types import AddFactIn, ClaimIn, DeleteIn, GraphQueryIn, ReleaseIn

GROUP = "smoke_wave1"
SESSION = "smoke_session_1"


async def _wipe_group() -> None:
    """Delete every Entity + FACT edge + Lock for the smoke group."""
    async with get_session() as s:
        await s.run("MATCH (n:Entity {group_id: $g}) DETACH DELETE n", g=GROUP)
        await s.run(
            "MATCH ()-[r:FACT]->() WHERE r.group_id = $g DELETE r", g=GROUP
        )
        await s.run("MATCH (l:Lock) WHERE l.resource_id STARTS WITH $p DELETE l", p=f"{GROUP}:")


async def step_add_facts() -> tuple[str, str]:
    print()
    print("[1/5] add_fact: insert two facts, expect the second to conflict with the first")
    t0 = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(days=90)

    a = await add_fact(AddFactIn(
        subject="alice", predicate="lives_in", object="boston",
        event_time=t0, source="smoke", session_id=SESSION, group_id=GROUP,
    ))
    print(f"  -> fact_a id={a.fact_id[:8]}  conflicts={len(a.conflicts_with)} (expected 0)")
    assert a.edges_created == 1
    assert a.conflicts_with == [], f"first fact should have no conflicts, got {a.conflicts_with}"

    b = await add_fact(AddFactIn(
        subject="alice", predicate="lives_in", object="seattle",
        event_time=t1, source="smoke", session_id=SESSION, group_id=GROUP,
    ))
    print(f"  -> fact_b id={b.fact_id[:8]}  conflicts={len(b.conflicts_with)} (expected 1)")
    assert b.edges_created == 1, b
    assert b.conflicts_with == [a.fact_id], (
        f"second fact should conflict with first; got conflicts={b.conflicts_with}"
    )
    print("  [PASS] conflict detection")
    return a.fact_id, b.fact_id


async def step_graph_query_facts() -> None:
    print()
    print("[2/5] graph_query mode=facts: subject='alice' should return both facts")
    out = await graph_query(GraphQueryIn(
        query="alice",
        scope=GROUP,
        mode="facts",
        time_anchor=datetime(2026, 12, 31, tzinfo=timezone.utc),
        tier_filter=["T1", "T2", "T3", "T4"],  # widen so we see everything
        budget_tokens=512,
    ))
    print(f"  -> {len(out.results)} fact(s), tokens_used={out.tokens_used}")
    for r in out.results:
        print(f"     ({r['subject']}, {r['predicate']}, {r['object']})  valid_at={r['valid_at']}")
    assert len(out.results) == 2, f"expected 2 facts, got {len(out.results)}"
    objects = sorted(r["object"] for r in out.results)
    assert objects == ["boston", "seattle"], objects
    print("  [PASS] facts mode")


async def step_graph_query_neighbors() -> None:
    print()
    print("[3/5] graph_query mode=neighbors: depth=1 from 'alice' should return both facts")
    out = await graph_query(GraphQueryIn(
        query="alice",
        scope=GROUP,
        mode="neighbors",
        depth=1,
        time_anchor=datetime(2026, 12, 31, tzinfo=timezone.utc),
        tier_filter=["T1", "T2", "T3", "T4"],
        budget_tokens=512,
    ))
    print(f"  -> {len(out.results)} neighbor fact(s)")
    assert len(out.results) == 2, f"expected 2, got {len(out.results)}"
    print("  [PASS] neighbors mode")


async def step_claim_release() -> None:
    print()
    print("[4/5] claim/release: first claim wins, second is rejected, release frees it")
    rid = f"{GROUP}:resource"
    c1 = await claim(ClaimIn(resource_id=rid, ttl_ms=5000), holder="holder_A")
    print(f"  -> claim_A granted={c1.granted}")
    assert c1.granted is True

    c2 = await claim(ClaimIn(resource_id=rid, ttl_ms=5000), holder="holder_B")
    print(f"  -> claim_B granted={c2.granted} (expected False)")
    assert c2.granted is False

    r = await release(ReleaseIn(resource_id=rid))
    print(f"  -> release={r.released}")
    assert r.released is True

    c3 = await claim(ClaimIn(resource_id=rid, ttl_ms=5000), holder="holder_C")
    print(f"  -> claim_C granted={c3.granted} (expected True post-release)")
    assert c3.granted is True
    await release(ReleaseIn(resource_id=rid))
    print("  [PASS] claim/release")


async def step_delete(fact_a: str, fact_b: str) -> None:
    print()
    print("[5/5] delete: remove fact_a, verify only fact_b remains")
    d = await delete(DeleteIn(target_id=fact_a, type="fact"))
    print(f"  -> delete ok={d.ok}")
    assert d.ok is True

    out = await graph_query(GraphQueryIn(
        query="alice", scope=GROUP, mode="facts",
        time_anchor=datetime(2026, 12, 31, tzinfo=timezone.utc),
        tier_filter=["T1", "T2", "T3", "T4"], budget_tokens=512,
    ))
    assert len(out.results) == 1, out
    assert out.results[0]["fact_id"] == fact_b
    print("  [PASS] delete")


async def main() -> int:
    print("=" * 70)
    print("Wave 1 smoke — add_fact / graph_query / claim_release / delete")
    print("=" * 70)
    try:
        await _wipe_group()
        a, b = await step_add_facts()
        await step_graph_query_facts()
        await step_graph_query_neighbors()
        await step_claim_release()
        await step_delete(a, b)
        await _wipe_group()
        print()
        print("=" * 70)
        print("ALL PASS — Wave 1 ops are live against Neo4j 5.24")
        print("=" * 70)
        return 0
    except AssertionError as e:
        print()
        print(f"[FAIL] assertion: {e}")
        return 1
    except Exception as e:
        print()
        print(f"[FAIL] {type(e).__name__}: {e}")
        return 2
    finally:
        await close_driver()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
