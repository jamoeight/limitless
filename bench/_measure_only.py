"""Measure existing scale-test data without re-seeding. Use after a prior
bench run left data in Neo4j + Qdrant for the 'scale_test' group."""

from __future__ import annotations

import asyncio
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

from bench.infer_scale import measure_size
from timegraph.storage.neo4j_client import close_driver, get_session
from timegraph.storage.qdrant_client import close_client as close_qdrant


async def count() -> tuple[int, int]:
    """Return (fact_count, episode_count) for the scale_test group."""
    async with get_session() as s:
        r = await s.run("MATCH ()-[r:FACT]->() WHERE r.group_id = 'scale_test' RETURN count(r) AS n")
        facts = (await r.single())["n"]
        r = await s.run("MATCH (e:Episode {group_id: 'scale_test'}) RETURN count(e) AS n")
        eps = (await r.single())["n"]
    return facts, eps


async def main() -> int:
    n_facts, n_eps = await count()
    print(f"existing data: {n_facts} facts, {n_eps} episodes in group=scale_test")
    if n_facts < 100:
        print("not enough data — run bench first")
        return 1
    try:
        result = await measure_size(n_facts, warmup=5, runs=10)
        print()
        print("=" * 88)
        print(f"N = {n_facts}")
        print(f"  judge_call_count all==1: {result['judge_call_count']['all_one']}")
        print(f"  cypher p50: {result['cypher_ms']['p50']:.0f}ms")
        print(f"  qdrant p50: {result['qdrant_ms']['p50']:.0f}ms")
        print(f"  judge  p50: {result['judge_ms']['p50']:.0f}ms")
        print(f"  total  p50: {result['total_ms']['p50']:.0f}ms")
        print(f"  total  p95: {result['total_ms']['p95']:.0f}ms")
        print(f"  ans ok:     {result['answer_correct_rate']:.0%}")
        return 0
    finally:
        await close_driver()
        await close_qdrant()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
