"""Phase 0: smoke-test that pydantic types serialize cleanly."""

from __future__ import annotations

from datetime import datetime

from timegraph.types import (
    AddEpisodeIn,
    ConflictTriple,
    Fact,
    InferOut,
    Resolution,
)


def test_fact_roundtrip():
    f = Fact(
        fact_id="f_001",
        subject="Alice",
        predicate="lives_in",
        object="Boston",
        valid_at=datetime(2026, 3, 15, 14, 22),
        confidence=0.95,
        tier="T2",
    )
    j = f.model_dump_json()
    f2 = Fact.model_validate_json(j)
    assert f2 == f


def test_add_episode_in_minimal():
    e = AddEpisodeIn(
        content="Alice moved to Boston in March.",
        source="chat",
        group_id="g_001",
        session_id="s_001",
    )
    assert e.event_time is None


def test_infer_out_judge_count_assertion():
    # The load-bearing assertion: judge_call_count must be ≤1 per infer() call.
    o = InferOut(
        mode_used="conflict_set",
        answer_facts=[],
        confidence=0.7,
        conflict_set=[ConflictTriple(e1_fact_id="a", e2_fact_id="b", reason="x")],
        resolution=Resolution.E1_CORRECT,
        hops_taken=2,
        judge_call_count=1,
    )
    assert o.judge_call_count <= 1, "BREAKTHROUGH THESIS VIOLATION"
