"""Wave 3 smoke — infer() in all 3 modes + fuse() dry-run.

The load-bearing assertion: `judge_call_count == 1` when conflicts exist,
regardless of how big the graph is.

Run with LM Studio + Neo4j + Qdrant up.
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime, timedelta, timezone

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

from timegraph.config import get_settings
from timegraph.ops.add_episode import add_episode
from timegraph.ops.fuse import fuse
from timegraph.ops.infer import infer
from timegraph.storage.neo4j_client import close_driver, get_session
from timegraph.storage.qdrant_client import (
    close_client as close_qdrant,
    ensure_collections,
    get_client as get_qdrant,
)
from timegraph.types import AddEpisodeIn, Fact, FuseIn, InferIn

GROUP = "smoke_wave3"
SESSION = "smoke_w3_session"


async def _wipe_group() -> None:
    s = get_settings()
    async with get_session() as session:
        await session.run("MATCH (e:Episode {group_id: $g}) DETACH DELETE e", g=GROUP)
        await session.run("MATCH (n:Entity {group_id: $g}) DETACH DELETE n", g=GROUP)
        await session.run("MATCH ()-[r:FACT]->() WHERE r.group_id = $g DELETE r", g=GROUP)

    from qdrant_client.http import models as qm
    client = await get_qdrant()
    flt = qm.Filter(must=[qm.FieldCondition(key="group_id", match=qm.MatchValue(value=GROUP))])
    for coll in (s.qdrant_facts_collection, s.qdrant_episodes_collection):
        try:
            await client.delete(collection_name=coll, points_selector=qm.FilterSelector(filter=flt))
        except Exception:
            pass


async def seed_conflicting_episodes() -> tuple[str, str]:
    """Two episodes that produce a real conflict (Alice's residence over time)."""
    print()
    print("[setup] Seeding two conflicting episodes (Alice's residence)")

    t0 = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(days=90)

    ep_a = await add_episode(AddEpisodeIn(
        content="I just moved to Boston last week for a new job at Acme.",
        source="user_msg", group_id=GROUP, session_id=SESSION, event_time=t0,
        asserted_facts=[
            Fact(fact_id="x", subject="alice", predicate="lives_in", object="Boston",
                 valid_at=t0, confidence=0.95, tier="T2"),
        ],
    ))
    print(f"  ep_a={ep_a.episode_id[:8]}  facts={len(ep_a.extracted_facts)}")

    ep_b = await add_episode(AddEpisodeIn(
        content="Update: just moved again. I'm in Seattle now — new role at Globex.",
        source="user_msg", group_id=GROUP, session_id=SESSION, event_time=t1,
        asserted_facts=[
            Fact(fact_id="x", subject="alice", predicate="lives_in", object="Seattle",
                 valid_at=t1, confidence=0.95, tier="T2"),
        ],
    ))
    print(f"  ep_b={ep_b.episode_id[:8]}  facts={len(ep_b.extracted_facts)}")
    return ep_a.episode_id, ep_b.episode_id


async def step_infer_all() -> None:
    print()
    print("[1/4] infer mode='all' — semantic, no LLM, returns everything relevant")
    t0 = time.perf_counter()
    out = await infer(InferIn(
        query="where does alice live",
        scope=GROUP,
        mode="all",
        tier_filter=["T1", "T2"],
    ))
    dt = (time.perf_counter() - t0) * 1000
    print(f"  -> answer_facts={len(out.answer_facts)}  judge_calls={out.judge_call_count}  ({dt:.0f}ms)")
    assert out.judge_call_count == 0, "all mode must NOT call the judge"
    assert len(out.answer_facts) >= 2, f"expected >=2 facts in 'all' mode, got {len(out.answer_facts)}"
    print("  [PASS] all mode")


async def step_infer_consistent() -> None:
    print()
    print("[2/4] infer mode='consistent' — drop conflicted facts, no LLM")
    t0 = time.perf_counter()
    out = await infer(InferIn(
        query="where does alice live",
        scope=GROUP,
        mode="consistent",
        tier_filter=["T1", "T2"],
    ))
    dt = (time.perf_counter() - t0) * 1000
    print(f"  -> answer_facts={len(out.answer_facts)}  judge_calls={out.judge_call_count}  ({dt:.0f}ms)")
    for f in out.answer_facts:
        print(f"     ({f.subject}, {f.predicate}, {f.object})")
    assert out.judge_call_count == 0, "consistent mode must NOT call the judge"
    # With ONLY conflicting Boston/Seattle facts in the graph, consistent should
    # be empty (every seed has a conflict).
    objects = {f.object for f in out.answer_facts}
    assert "Boston" not in objects and "Seattle" not in objects, (
        f"consistent mode should drop conflicting facts; got {objects}"
    )
    print("  [PASS] consistent mode (correctly dropped conflicted facts)")


async def step_infer_conflict_set() -> None:
    print()
    print("[3/4] ★ infer mode='conflict_set' — LOAD-BEARING: judge_call_count MUST be ≤1")
    t0 = time.perf_counter()
    out = await infer(InferIn(
        query="where does alice live now",
        scope=GROUP,
        mode="conflict_set",
        tier_filter=["T1", "T2"],
    ))
    dt = (time.perf_counter() - t0) * 1000
    print(f"  -> judge_calls={out.judge_call_count}  (assertion: ≤1)")
    print(f"  -> resolution={out.resolution}  confidence={out.confidence:.2f}")
    print(f"  -> conflict_set_size={len(out.conflict_set or [])}")
    print(f"  -> answer_facts={len(out.answer_facts)}")
    for f in out.answer_facts:
        print(f"     ({f.subject}, {f.predicate}, {f.object})  valid_at={f.valid_at}")
    print(f"  -> total latency: {dt:.0f}ms")

    # ★★★ The structural assertion the breakthrough thesis rests on.
    assert out.judge_call_count <= 1, (
        f"BREAKTHROUGH ASSERTION VIOLATED: judge_call_count={out.judge_call_count}, must be ≤1"
    )
    assert len(out.conflict_set or []) >= 1, "expected ≥1 surfaced conflict"
    assert out.resolution is not None, "judge should return a resolution"
    print("  [PASS] conflict_set mode — judge_call_count==1 holds")


async def step_fuse() -> None:
    print()
    print("[4/4] fuse(dry_run=True) — propose supersession for (alice, lives_in)")
    out = await fuse(FuseIn(
        scope=GROUP,
        group_id=GROUP,
        dry_run=True,
    ))
    print(f"  -> proposed_merges={len(out.proposed_merges)}  "
          f"proposed_supersessions={len(out.proposed_supersessions)}")
    for m in out.proposed_merges:
        print(f"     merge: ({m['subject']}, {m['predicate']}) winner={m['winning_object']!r} "
              f"losing_n={len(m['losing_fact_ids'])}")
    assert len(out.proposed_merges) == 1
    merge = out.proposed_merges[0]
    assert merge["subject"] == "alice"
    assert merge["predicate"] == "lives_in"
    # Newest valid_at wins -> Seattle (April 2026) over Boston (Jan 2026).
    assert merge["winning_object"] == "Seattle", merge
    assert out.applied is False
    print("  [PASS] fuse dry_run")


async def main() -> int:
    print("=" * 70)
    print("Wave 3 smoke — infer() (all/consistent/conflict_set) + fuse() dry_run")
    print("=" * 70)
    try:
        await ensure_collections()
        await _wipe_group()
        await seed_conflicting_episodes()
        await step_infer_all()
        await step_infer_consistent()
        await step_infer_conflict_set()
        await step_fuse()
        await _wipe_group()
        print()
        print("=" * 70)
        print("ALL PASS — Wave 3 ops live. Breakthrough assertion holds.")
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
