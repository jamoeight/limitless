"""infer — the load-bearing two-stage hybrid.

★ This op carries the breakthrough thesis: bounded LLM calls regardless of
  graph size. The structural assertion is `judge_call_count <= 1` per invocation.

Stages
------
Stage-1 (Cypher + Qdrant, <50ms target):
  Semantic top-k via Qdrant (query → vector → relevant fact ids), then a
  parameterized Cypher walk that:
    - hydrates those facts
    - finds sibling facts with same (subject, predicate) but different object
      whose valid windows overlap → candidate conflict pairs

Stage-2 (1 LLM call, only for mode='conflict_set'):
  Pass ≤8 candidate conflict triples + their source episodes (truncated) to
  the Qwen3.5-9B judge. Receive a single resolution + reason.

Mode behavior
-------------
  consistent  — stage-1 only; return facts where no conflicts surfaced.
  conflict_set — stage-1 + stage-2; return resolution + the surviving facts.
  all         — stage-1 only; return everything (semantically relevant).

Output contract: `InferOut.judge_call_count` MUST be ≤1. Integration tests assert
this regardless of graph size.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import structlog

from timegraph.config import get_settings
from timegraph.llm.embedder import EmbedderClient
from timegraph.llm.judge import JudgeClient
from timegraph.storage.neo4j_client import get_session
from timegraph.storage.qdrant_client import search as qdrant_search
from timegraph.types import (
    ConflictTriple,
    Fact,
    InferIn,
    InferOut,
    Resolution,
)

log = structlog.get_logger(__name__)

# Stage-1 hydration + conflict surfacing for a fixed set of seed fact ids.
# Strategy: pull each seed fact + every "sibling" fact (same subject+predicate,
# different object) whose valid window overlaps. We emit pairs as conflict
# candidates only at the Python layer to keep the Cypher simple.
_HYDRATE_AND_SIBLINGS = """
MATCH ()-[r:FACT]->()
WHERE r.id IN $seed_ids
  AND r.tier IN $tier_filter
  AND ($scope = '*' OR r.group_id = $scope)
  AND r.valid_at <= $time_anchor
  AND (r.invalid_at IS NULL OR r.invalid_at > $time_anchor)
WITH collect(r) AS seeds
UNWIND seeds AS r
OPTIONAL MATCH ()-[s:FACT]->()
WHERE s.subject = r.subject
  AND s.predicate = r.predicate
  AND s.object <> r.object
  AND s.tier IN $tier_filter
  AND s.group_id = r.group_id
  AND s.valid_at <= $time_anchor
  AND (s.invalid_at IS NULL OR s.invalid_at > $time_anchor)
  // Window overlap: [r.valid_at, r.invalid_at) overlaps [s.valid_at, s.invalid_at).
  // Two windows [a,b) and [c,d) overlap iff a < d AND c < b (with NULL = +inf).
  AND r.valid_at < COALESCE(s.invalid_at, datetime('9999-12-31T23:59:59Z'))
  AND s.valid_at < COALESCE(r.invalid_at, datetime('9999-12-31T23:59:59Z'))
WITH r, collect(s) AS siblings
RETURN r{.*}     AS seed,
       [x IN siblings | x{.*}] AS siblings
"""

# Source episode text retrieval (truncated) for stage-2 context.
_LOAD_EPISODE_SNIPPETS = """
MATCH (e:Episode)
WHERE e.id IN $episode_ids
RETURN e.id AS episode_id, e.content AS content, e.event_time AS event_time
"""

# Hard ceiling on conflicts handed to the judge. Stage-1's relevance filter
# does the real work — this is the safety net for pathological cases.
_MAX_CONFLICTS = 8

# Gap-based pruning: when noise-pair scores cluster together (because nomic
# cosine has a high "semantically coherent triple" floor of ~0.55), the largest
# adjacent-score gap reliably separates query-relevant pairs from the rest.
# Minimum gap required to trigger a cut (relative to best score) — guards
# against the degenerate case where ALL pairs are roughly equally noise-level.
_MIN_GAP_FRACTION = 0.10


def _to_native(x):
    if x is None:
        return None
    n = getattr(x, "to_native", None)
    return n() if callable(n) else x


def _row_to_fact(d: dict) -> Fact:
    """Convert a Neo4j r{.*} record to our pydantic Fact."""
    return Fact(
        fact_id=d["id"],
        subject=d["subject"],
        predicate=d["predicate"],
        object=d["object"],
        valid_at=_to_native(d["valid_at"]),
        invalid_at=_to_native(d.get("invalid_at")),
        confidence=float(d.get("confidence", 1.0)),
        pinned=bool(d.get("pinned", False)),
        tier=d.get("tier", "T2"),
        session_id=d.get("session_id"),
        source_episode_id=d.get("source_episode_id"),
        sources=[d["source"]] if d.get("source") else [],
    )


async def _stage1(
    payload: InferIn, time_anchor: datetime
) -> tuple[list[Fact], list[tuple[Fact, Fact]], dict[str, float], int]:
    """Return (seed_facts, conflict_pairs, timings, pre_truncation_pair_count).

    Pure Cypher + Qdrant; no LLM. Timings cover embed, qdrant, cypher (ms).
    """
    s = get_settings()
    timings: dict[str, float] = {}

    # Semantic seed: top-k facts via Qdrant.
    embedder = EmbedderClient()
    try:
        t = time.perf_counter()
        qvec = await embedder.embed_one(payload.query)
        timings["embed_ms"] = (time.perf_counter() - t) * 1000
    finally:
        await embedder.close()

    over_k = max(payload.max_hops * 4, 16)
    t = time.perf_counter()
    hits = await qdrant_search(
        collection=s.qdrant_facts_collection,
        vector=qvec,
        k=over_k,
        group_id=payload.scope,
        tier_filter=payload.tier_filter,
    )
    timings["qdrant_ms"] = (time.perf_counter() - t) * 1000
    if not hits:
        timings["cypher_ms"] = 0.0
        return [], [], timings, 0

    # Capture Qdrant cosine scores; used both here (seed-level pruning) and
    # later (pair-level re-ranking).
    score_by_id: dict[str, float] = {str(h.id): float(h.score) for h in hits}

    # Seed-level gap pruning. The biggest score drop in the sorted Qdrant hits
    # separates query-relevant seeds from noise (random unit-norm vectors at
    # large N score ~0.1 against a real query embedding, while real embeddings
    # for relevant facts cluster at 0.5-0.85). Collecting siblings for noise
    # seeds is the dominant cost at 1M scale — pruning here is ~20× speedup.
    sorted_hits = sorted(hits, key=lambda h: -float(h.score))
    if len(sorted_hits) > 1:
        best = float(sorted_hits[0].score)
        min_gap = best * _MIN_GAP_FRACTION
        cut_at = len(sorted_hits)
        biggest_gap = 0.0
        for i in range(len(sorted_hits) - 1):
            gap = float(sorted_hits[i].score) - float(sorted_hits[i + 1].score)
            if gap > biggest_gap and gap >= min_gap:
                biggest_gap = gap
                cut_at = i + 1
        sorted_hits = sorted_hits[:cut_at]
    seed_ids = [str(h.id) for h in sorted_hits]
    timings["seeds_kept"] = float(len(seed_ids))

    # One Cypher round-trip: hydrate seeds + collect overlapping siblings per seed.
    t = time.perf_counter()
    async with get_session() as session:
        res = await session.run(
            _HYDRATE_AND_SIBLINGS,
            parameters={
                "seed_ids": seed_ids,
                "scope": payload.scope,
                "tier_filter": payload.tier_filter,
                "time_anchor": time_anchor,
            },
        )
        seeds: list[Fact] = []
        pairs: list[tuple[Fact, Fact]] = []
        seen_pair_keys: set[tuple[str, str]] = set()
        async for r in res:
            seed = _row_to_fact(r["seed"])
            seeds.append(seed)
            for sib in r["siblings"]:
                sib_f = _row_to_fact(sib)
                # Symmetric dedup: (a,b) and (b,a) are the same conflict.
                key = tuple(sorted([seed.fact_id, sib_f.fact_id]))
                if key in seen_pair_keys:
                    continue
                seen_pair_keys.add(key)
                pairs.append((seed, sib_f))
    timings["cypher_ms"] = (time.perf_counter() - t) * 1000

    pre_truncation = len(pairs)
    if not pairs:
        return seeds, [], timings, 0

    # Rank pairs by query-relevance. A pair's score is max(seed, sibling) over
    # Qdrant cosine; siblings discovered only via Cypher get 0 by default,
    # which is correct — they have NO measured relevance to the query.
    scored = [
        (pair, max(score_by_id.get(pair[0].fact_id, 0.0),
                   score_by_id.get(pair[1].fact_id, 0.0)))
        for pair in pairs
    ]
    scored.sort(key=lambda x: -x[1])
    best = scored[0][1]

    # Gap-based truncation. Walk the sorted scores; the largest drop between
    # consecutive pairs marks the boundary between query-relevant and noise.
    # Only trigger if the gap is meaningful (≥ min_gap fraction of best).
    keep_n = len(scored)
    if len(scored) > 1:
        min_gap = best * _MIN_GAP_FRACTION
        biggest_gap = 0.0
        cut_at = len(scored)
        for i in range(len(scored) - 1):
            gap = scored[i][1] - scored[i + 1][1]
            if gap > biggest_gap and gap >= min_gap:
                biggest_gap = gap
                cut_at = i + 1
        keep_n = cut_at

    relevant = [p for p, _ in scored[:keep_n]][:_MAX_CONFLICTS]
    timings["pair_score_best"] = best
    timings["pair_score_worst_kept"] = scored[keep_n - 1][1] if keep_n > 0 else 0.0
    timings["pair_score_first_dropped"] = scored[keep_n][1] if keep_n < len(scored) else 0.0

    return seeds, relevant, timings, pre_truncation


async def _load_episode_snippets(episode_ids: list[str], max_chars: int = 300) -> list[str]:
    """Fetch + truncate source-episode content for the judge's context."""
    if not episode_ids:
        return []
    async with get_session() as session:
        res = await session.run(_LOAD_EPISODE_SNIPPETS, episode_ids=episode_ids)
        rows = [r.data() async for r in res]
    snippets: list[str] = []
    for r in rows:
        content = r["content"] or ""
        if len(content) > max_chars:
            content = content[: max_chars - 1] + "…"
        ts = _to_native(r["event_time"])
        snippets.append(f"[{ts:%Y-%m-%d}] {content}")
    return snippets


def _resolve_answer_facts(
    pairs: list[tuple[Fact, Fact]],
    seeds: list[Fact],
    resolution: Resolution | None,
) -> list[Fact]:
    """Pick which facts the caller should treat as 'the answer' given resolution."""
    if not pairs:
        return seeds  # no conflicts — seeds ARE the answer
    e1s = [a for a, _ in pairs]
    e2s = [b for _, b in pairs]
    if resolution is Resolution.E1_CORRECT:
        return e1s
    if resolution is Resolution.E2_CORRECT:
        return e2s
    # both_partial / unresolved / no-resolution — return everything for the caller.
    return e1s + e2s


async def infer(payload: InferIn) -> InferOut:
    """The bounded-LLM-call inference op."""
    t_total = time.perf_counter()
    time_anchor = payload.time_anchor or datetime.now(timezone.utc)

    seeds, pairs, timings, pre_truncation = await _stage1(payload, time_anchor)
    timings["candidates_n"] = float(pre_truncation)
    timings["judge_ms"] = 0.0

    if payload.mode == "all":
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000
        return InferOut(
            mode_used="all",
            answer_facts=seeds,
            confidence=0.5,
            conflict_set=None,
            resolution=None,
            hops_taken=1,
            judge_call_count=0,  # no LLM
            timings_ms=timings,
        )

    if payload.mode == "consistent":
        # Drop seeds that appear in any conflict pair.
        conflicted_ids = {f.fact_id for pair in pairs for f in pair}
        clean = [f for f in seeds if f.fact_id not in conflicted_ids]
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000
        return InferOut(
            mode_used="consistent",
            answer_facts=clean,
            confidence=0.7 if clean else 0.0,
            conflict_set=None,
            resolution=None,
            hops_taken=1,
            judge_call_count=0,
            timings_ms=timings,
        )

    # mode == "conflict_set" — the load-bearing path.
    if not pairs:
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000
        return InferOut(
            mode_used="conflict_set",
            answer_facts=seeds,
            confidence=0.9,
            conflict_set=[],
            resolution=None,
            hops_taken=1,
            judge_call_count=0,
            timings_ms=timings,
        )

    # Sort each pair so e1 is the older fact and e2 the newer. This (1) removes
    # the dependence on Cypher traversal order and (2) gives the judge a stable
    # convention to anchor temporal reasoning on. The reason text carries the
    # valid_at dates inline so the judge doesn't have to infer them from
    # episode snippet content.
    def _ordered(a: Fact, b: Fact) -> tuple[Fact, Fact]:
        if a.valid_at <= b.valid_at:
            return a, b
        return b, a

    triples = []
    for raw_a, raw_b in pairs:
        e1, e2 = _ordered(raw_a, raw_b)
        triples.append(ConflictTriple(
            e1_fact_id=e1.fact_id,
            e2_fact_id=e2.fact_id,
            reason=(
                f"e1 (older): ({e1.subject}, {e1.predicate}, {e1.object}) "
                f"valid_at={e1.valid_at:%Y-%m-%d}; "
                f"e2 (newer): ({e2.subject}, {e2.predicate}, {e2.object}) "
                f"valid_at={e2.valid_at:%Y-%m-%d}"
            ),
        ))
    # Replace pairs with the canonically-ordered version so downstream answer
    # resolution (e1_correct → answer e1) stays in sync.
    pairs = [(_ordered(a, b)) for a, b in pairs]
    ep_ids = list({
        f.source_episode_id
        for pair in pairs
        for f in pair
        if f.source_episode_id
    })
    snippets = await _load_episode_snippets(ep_ids)

    # ★ THE one allowed LLM call. judge.judge_conflicts() guarantees call_count=1.
    judge = JudgeClient()
    try:
        t = time.perf_counter()
        verdict = await judge.judge_conflicts(
            query=payload.query,
            conflicts=triples,
            attestations=None,
            source_episodes_truncated=snippets,
        )
        timings["judge_ms"] = (time.perf_counter() - t) * 1000
    finally:
        await judge.close()

    answer = _resolve_answer_facts(pairs, seeds, verdict.resolution)
    timings["total_ms"] = (time.perf_counter() - t_total) * 1000

    return InferOut(
        mode_used="conflict_set",
        answer_facts=answer,
        confidence=verdict.confidence,
        conflict_set=triples,
        resolution=verdict.resolution,
        hops_taken=1,
        judge_call_count=verdict.call_count,  # always 1 by judge.py contract
        timings_ms=timings,
    )
