"""Scale test for the bounded-LLM-call thesis.

For each graph size N in {100, 1000, 10000}:
  1. Seed N facts:
       - 6 hot facts: 3 conflicting (alice, lives_in, Boston/Seattle/Portland)
                      + 3 distractors (alice has other attributes)
       - N-6 cold facts: random (subject, predicate, object) noise
  2. Bulk-insert into Neo4j (batched UNWIND) + Qdrant (batched upsert)
  3. Warm up (3 calls) then measure (10 calls) of:
       infer(query='where does alice live now', mode='conflict_set')
  4. Aggregate: p50/p95 of embed_ms, qdrant_ms, cypher_ms, judge_ms, total_ms,
       judge_call_count (must be ≤1 every time), candidates_n.

The breakthrough thesis claim: judge_call_count stays at 1 regardless of N.
Quality claim: judge still picks E2_CORRECT / Seattle (newest) at every N.

Saves results/infer_scale.json + prints a summary table.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

from qdrant_client.http import models as qm

from timegraph.config import get_settings
from timegraph.llm.embedder import EmbedderClient
from timegraph.ops.add_episode import add_episode
from timegraph.ops.infer import infer
from timegraph.storage.neo4j_client import close_driver, get_session
from timegraph.storage.qdrant_client import (
    close_client as close_qdrant,
    ensure_collections,
    get_client as get_qdrant,
)
from timegraph.types import AddEpisodeIn, Fact, InferIn

GROUP = "scale_test"
SESSION = "scale_session"

# Noise vocabulary — small enough that random pairs occasionally collide, big
# enough that the (subject, predicate) index is exercised.
NOISE_SUBJECTS = [f"entity_{i:03d}" for i in range(60)]
NOISE_PREDICATES = ["knows", "owns", "likes", "visited", "writes_about", "follows",
                    "manages", "reports_to", "lives_in", "works_at"]
NOISE_OBJECTS = [f"thing_{i:03d}" for i in range(80)]

QUERY = "where does alice live now"


# ---- Seeding ----------------------------------------------------------


async def wipe_group() -> None:
    s = get_settings()
    async with get_session() as session:
        await session.run("MATCH (n:Entity {group_id: $g}) DETACH DELETE n", g=GROUP)
        await session.run("MATCH ()-[r:FACT]->() WHERE r.group_id = $g DELETE r", g=GROUP)
        await session.run("MATCH (e:Episode {group_id: $g}) DETACH DELETE e", g=GROUP)

    client = await get_qdrant()
    flt = qm.Filter(must=[qm.FieldCondition(key="group_id", match=qm.MatchValue(value=GROUP))])
    for coll in (s.qdrant_facts_collection, s.qdrant_episodes_collection):
        try:
            await client.delete(collection_name=coll, points_selector=qm.FilterSelector(filter=flt))
        except Exception:
            pass


_BULK_INSERT = """
UNWIND $facts AS f
MERGE (s:Entity {name: f.subject, group_id: $group_id})
MERGE (o:Entity {name: f.object,  group_id: $group_id})
CREATE (s)-[r:FACT {
  id: f.fact_id,
  subject: f.subject,
  predicate: f.predicate,
  object: f.object,
  valid_at: f.valid_at,
  invalid_at: NULL,
  confidence: 0.9,
  pinned: false,
  tier: 'T2',
  group_id: $group_id,
  source: 'scale_bench',
  session_id: $session_id
}]->(o)
RETURN count(r) AS n
"""


async def _seed_hot_via_episodes(t_now: datetime) -> int:
    """Seed the hot 5 facts via add_episode (so they get Episode nodes for context).

    Two episodes carry the binary conflict; three more carry distractor attributes.
    Returns the number of facts created.
    """
    t0 = t_now - timedelta(days=365)
    t_mid = t_now - timedelta(days=180)
    t_recent = t_now - timedelta(days=14)

    episodes = [
        # The binary conflict — meaningful prose so the judge has evidence to weigh.
        ("I just moved to Boston last week to start a new job.", t0,
         Fact(fact_id="x", subject="alice", predicate="lives_in",
              object="Boston", valid_at=t0, confidence=0.95, tier="T2")),
        ("Update: just moved again. I'm in Seattle now — new role at Globex.", t_recent,
         Fact(fact_id="x", subject="alice", predicate="lives_in",
              object="Seattle", valid_at=t_recent, confidence=0.95, tier="T2")),
        # Distractors — semantically near "where does alice live" but not the answer.
        ("Born and raised in Chicago.", t0,
         Fact(fact_id="x", subject="alice", predicate="born_in",
              object="Chicago", valid_at=t0, confidence=0.95, tier="T2")),
        ("I work at Acme as a software engineer.", t_mid,
         Fact(fact_id="x", subject="alice", predicate="works_at",
              object="Acme", valid_at=t_mid, confidence=0.95, tier="T2")),
        ("Native English speaker.", t0,
         Fact(fact_id="x", subject="alice", predicate="speaks",
              object="English", valid_at=t0, confidence=0.95, tier="T2")),
    ]

    n = 0
    for content, t, fact in episodes:
        await add_episode(AddEpisodeIn(
            content=content, source="scale_bench_hot",
            group_id=GROUP, session_id=SESSION, event_time=t,
            asserted_facts=[fact],
        ))
        n += 1
    return n


def _make_noise_facts(n: int, t_now: datetime, rng: random.Random) -> list[dict]:
    facts = []
    span_days = 730
    for _ in range(n):
        s = rng.choice(NOISE_SUBJECTS)
        p = rng.choice(NOISE_PREDICATES)
        o = rng.choice(NOISE_OBJECTS)
        # Random day in the last 2 years.
        valid_at = t_now - timedelta(days=rng.randint(0, span_days), hours=rng.randint(0, 23))
        facts.append({
            "fact_id": str(uuid.uuid4()),
            "subject": s, "predicate": p, "object": o, "valid_at": valid_at,
        })
    return facts


async def _bulk_neo4j(facts: list[dict], batch: int = 1000) -> float:
    """Insert facts into Neo4j in batches. Returns total wall-clock (s)."""
    t0 = time.perf_counter()
    async with get_session() as session:
        for i in range(0, len(facts), batch):
            chunk = facts[i : i + batch]
            await session.run(
                _BULK_INSERT,
                parameters={"facts": chunk, "group_id": GROUP, "session_id": SESSION},
            )
    return time.perf_counter() - t0


async def _bulk_qdrant(facts: list[dict], batch: int = 64) -> float:
    """Embed every fact text + upsert into Qdrant.

    Embedding is the bottleneck. We use the embedder client's natural batching
    and the qdrant client's PointStruct upsert.
    """
    s = get_settings()
    client = await get_qdrant()
    embedder = EmbedderClient()
    t0 = time.perf_counter()
    try:
        for i in range(0, len(facts), batch):
            chunk = facts[i : i + batch]
            texts = [f"{f['subject']} {f['predicate']} {f['object']}" for f in chunk]
            vecs = await embedder.embed_many(texts)
            points = [
                qm.PointStruct(
                    id=f["fact_id"],
                    vector=v,
                    payload={
                        "fact_id": f["fact_id"],
                        "group_id": GROUP,
                        "subject": f["subject"],
                        "predicate": f["predicate"],
                        "object": f["object"],
                        "tier": "T2",
                        "valid_at": f["valid_at"].isoformat(),
                    },
                )
                for f, v in zip(chunk, vecs, strict=True)
            ]
            await client.upsert(collection_name=s.qdrant_facts_collection, points=points)
    finally:
        await embedder.close()
    return time.perf_counter() - t0


async def _bulk_qdrant_random(facts: list[dict], batch: int = 5000) -> float:
    """Upsert noise facts to Qdrant with random unit-norm 768D vectors.

    Skips the embedder, which is the only super-linear cost at 1M scale.
    Random vectors score ~0 against a real query embedding (std ~1/sqrt(768)),
    while real hot-fact embeddings score 0.6-0.85 — ranking integrity is
    preserved. This validates Qdrant's HNSW index at scale without paying
    the LM-Studio HTTP round-trip cost on 999,995 irrelevant items.
    """
    import numpy as np
    s = get_settings()
    client = await get_qdrant()
    t0 = time.perf_counter()
    rng = np.random.default_rng(0xCAFE)
    for i in range(0, len(facts), batch):
        chunk = facts[i : i + batch]
        v = rng.standard_normal((len(chunk), s.embedder_dim)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        points = [
            qm.PointStruct(
                id=f["fact_id"],
                vector=v[j].tolist(),
                payload={
                    "fact_id": f["fact_id"],
                    "group_id": GROUP,
                    "subject": f["subject"],
                    "predicate": f["predicate"],
                    "object": f["object"],
                    "tier": "T2",
                    "valid_at": f["valid_at"].isoformat(),
                },
            )
            for j, f in enumerate(chunk)
        ]
        await client.upsert(collection_name=s.qdrant_facts_collection, points=points)
    return time.perf_counter() - t0


async def seed_size(N: int, t_now: datetime, fast_noise_threshold: int = 50_000) -> dict[str, float]:
    """Seed a graph of size N. Hot facts go through add_episode (so they get
    Episode nodes the judge can read); noise gets bulk-inserted for speed.

    For N >= fast_noise_threshold, noise gets random unit-norm vectors in Qdrant
    (skipping the embedder bottleneck). Hot facts always get real embeddings.
    """
    rng = random.Random(0xC0DE + N)

    t = time.perf_counter()
    n_hot = await _seed_hot_via_episodes(t_now)
    hot_dt = time.perf_counter() - t
    print(f"  [{N}] hot facts via add_episode: {n_hot} facts in {hot_dt:.1f}s")

    noise = _make_noise_facts(max(0, N - n_hot), t_now, rng)
    print(f"  [{N}] generated {len(noise)} noise facts")

    neo_dt = await _bulk_neo4j(noise)
    print(f"  [{N}] neo4j bulk insert (noise): {neo_dt:.1f}s")

    if N >= fast_noise_threshold:
        qd_dt = await _bulk_qdrant_random(noise)
        print(f"  [{N}] qdrant random-vec upsert: {qd_dt:.1f}s  (fast-noise mode)")
    else:
        qd_dt = await _bulk_qdrant(noise)
        print(f"  [{N}] qdrant + embed (noise):    {qd_dt:.1f}s")

    return {"hot_s": hot_dt, "neo_s": neo_dt, "qdrant_s": qd_dt,
            "total_s": hot_dt + neo_dt + qd_dt}


# ---- Measurement ------------------------------------------------------


async def measure_one() -> tuple[dict[str, float], int, str | None, int, list[str]]:
    """One infer() call. Returns (timings, judge_calls, resolution, n_conflicts, answer_objects)."""
    out = await infer(InferIn(
        query=QUERY,
        scope=GROUP,
        mode="conflict_set",
        tier_filter=["T1", "T2"],
    ))
    res = out.resolution.value if out.resolution else None
    n_conf = len(out.conflict_set or [])
    # The objects of facts the judge chose as "the answer". For our scenario,
    # "Seattle" present = correct (newest alice/lives_in wins).
    answer_objects = [
        f.object for f in out.answer_facts
        if f.subject == "alice" and f.predicate == "lives_in"
    ]
    return out.timings_ms or {}, out.judge_call_count, res, n_conf, answer_objects


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round(p / 100 * (len(s) - 1)))))
    return s[idx]


async def measure_size(N: int, warmup: int = 3, runs: int = 10) -> dict:
    print(f"  [{N}] warmup x{warmup}…", end=" ", flush=True)
    for _ in range(warmup):
        await measure_one()
    print(f"measuring x{runs}…")

    samples: list[dict[str, float]] = []
    call_counts: list[int] = []
    resolutions: list[str | None] = []
    conflict_sizes: list[int] = []
    answers: list[list[str]] = []
    for i in range(runs):
        timings, judge_calls, resolution, n_conf, answer_objects = await measure_one()
        samples.append(timings)
        call_counts.append(judge_calls)
        resolutions.append(resolution)
        conflict_sizes.append(n_conf)
        answers.append(answer_objects)
        winner = answer_objects[0] if answer_objects else "(none)"
        print(
            f"  [{N}] run {i+1:2d}/{runs}  "
            f"total={timings.get('total_ms',0):6.0f}ms  "
            f"cypher={timings.get('cypher_ms',0):5.0f}ms  "
            f"qdrant={timings.get('qdrant_ms',0):5.0f}ms  "
            f"judge={timings.get('judge_ms',0):6.0f}ms  "
            f"jc={judge_calls}  conflicts={n_conf}  res={resolution}  winner={winner}"
        )

    def stat(key: str) -> dict[str, float]:
        vals = [s.get(key, 0.0) for s in samples]
        return {"p50": _pct(vals, 50), "p95": _pct(vals, 95), "mean": sum(vals) / len(vals)}

    return {
        "n_facts": N,
        "embed_ms":  stat("embed_ms"),
        "qdrant_ms": stat("qdrant_ms"),
        "cypher_ms": stat("cypher_ms"),
        "judge_ms":  stat("judge_ms"),
        "total_ms":  stat("total_ms"),
        "judge_call_count": {
            "max": max(call_counts), "min": min(call_counts),
            "all_one": all(c == 1 for c in call_counts),
        },
        "conflicts_surfaced": {
            "p50": median(conflict_sizes), "max": max(conflict_sizes),
        },
        # Correct == judge's answer_facts include Seattle (the newest alice/lives_in).
        "answer_correct_rate": sum(1 for ans in answers if "Seattle" in ans) / len(answers),
        "answers_seen": sorted({obj for ans in answers for obj in ans}),
        "resolutions_seen": sorted(set(r for r in resolutions if r)),
    }


# ---- Main -------------------------------------------------------------


async def run(sizes: list[int]) -> None:
    print("=" * 88)
    print("Scale test — infer(conflict_set) under graph sizes:", sizes)
    print("=" * 88)
    await ensure_collections()

    t_now = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
    results = []
    for N in sizes:
        print(f"\n--- N = {N} ---")
        await wipe_group()
        seed_t = await seed_size(N, t_now)
        run_r = await measure_size(N)
        run_r["seed_timings_s"] = seed_t
        results.append(run_r)

    # Print summary table.
    print()
    print("=" * 88)
    print("SUMMARY")
    print("=" * 88)
    cols = ("N", "embed p50", "qdrant p50", "cypher p50", "judge p50", "total p50", "total p95", "jc==1", "ans ok")
    print("{:>7} {:>10} {:>11} {:>11} {:>10} {:>10} {:>10} {:>6} {:>7}".format(*cols))
    print("-" * 88)
    for r in results:
        print("{:>7} {:>9.0f}ms {:>10.0f}ms {:>10.0f}ms {:>9.0f}ms {:>9.0f}ms {:>9.0f}ms {:>6} {:>6.0%}".format(
            r["n_facts"],
            r["embed_ms"]["p50"], r["qdrant_ms"]["p50"], r["cypher_ms"]["p50"],
            r["judge_ms"]["p50"], r["total_ms"]["p50"], r["total_ms"]["p95"],
            "YES" if r["judge_call_count"]["all_one"] else "NO",
            r["answer_correct_rate"],
        ))

    print()
    print("Breakthrough assertions:")
    all_one = all(r["judge_call_count"]["all_one"] for r in results)
    print(f"  judge_call_count == 1 across every run at every size: {'HOLDS' if all_one else 'VIOLATED'}")
    flat = all(r["total_ms"]["p50"] < 5000 for r in results)
    print(f"  total p50 < 5s at every size: {'HOLDS' if flat else 'VIOLATED'}")
    cypher_flat = all(r["cypher_ms"]["p50"] < 200 for r in results)
    print(f"  cypher p50 < 200ms (plan target 50ms, generous slack): {'HOLDS' if cypher_flat else 'VIOLATED'}")

    out_path = Path("results/infer_scale.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull results -> {out_path}")

    # Cleanup
    await wipe_group()


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="100,1000,10000",
                    help="comma-separated graph sizes to test")
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]
    try:
        await run(sizes)
        return 0
    except Exception as e:
        import traceback
        print(f"\n[FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()
        return 2
    finally:
        await close_driver()
        await close_qdrant()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
