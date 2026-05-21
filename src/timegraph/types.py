"""Pydantic payloads for every MCP op.

These are the wire contracts the MCP server registers. The op modules in
`timegraph.ops.*` consume and return these.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

Tier = Literal["T1", "T2", "T3", "T4"]


class Resolution(str, Enum):
    E1_CORRECT = "e1_correct"
    E2_CORRECT = "e2_correct"
    BOTH_PARTIAL = "both_partial"
    UNRESOLVED = "unresolved"


# ---------- Common ----------


class Provenance(BaseModel):
    episode_id: str | None = None
    ts: datetime
    source: str
    user_id: str | None = None
    session_id: str | None = None
    src_tool: str | None = None
    sig: str | None = None  # HMAC; production-required, dev-optional


class Fact(BaseModel):
    fact_id: str
    subject: str
    predicate: str
    object: str
    valid_at: datetime
    invalid_at: datetime | None = None
    confidence: float = 1.0
    pinned: bool = False
    tier: Tier = "T2"
    source_episode_id: str | None = None
    session_id: str | None = None
    sources: list[str] = Field(default_factory=list)


# ---------- B.4-v2 op payloads (capability layer) ----------


class AddEpisodeIn(BaseModel):
    content: str
    source: str
    group_id: str
    event_time: datetime | None = None
    session_id: str
    asserted_facts: list[Fact] | None = None


class AddEpisodeOut(BaseModel):
    episode_id: str
    extracted_facts: list[Fact]
    conflicts_detected: list[str] = Field(default_factory=list)


class AddFactIn(BaseModel):
    subject: str
    predicate: str
    object: str
    event_time: datetime
    source: str
    session_id: str
    group_id: str = "default"
    confidence: float = 1.0
    tier: Tier = "T2"
    pinned: bool = False
    source_episode_id: str | None = None


class AddFactOut(BaseModel):
    fact_id: str
    edges_created: int
    conflicts_with: list[str] = Field(default_factory=list)


class GraphQueryIn(BaseModel):
    query: str
    scope: str = "*"
    mode: Literal["nodes", "facts", "neighbors", "flat"] = "facts"
    time_anchor: datetime | None = None
    depth: int = 1
    include_invalidated: bool = False
    k: int = 4
    tier_filter: list[Tier] = Field(default_factory=lambda: ["T1", "T2"])
    budget_tokens: int = 512


class GraphQueryOut(BaseModel):
    mode: Literal["nodes", "facts", "neighbors", "flat"]
    results: list[dict]
    provenance: list[Provenance]
    tokens_used: int


class InferIn(BaseModel):
    query: str
    scope: str = "*"
    time_anchor: datetime | None = None
    max_hops: int = 3
    mode: Literal["consistent", "conflict_set", "all"] = "conflict_set"
    tier_filter: list[Tier] = Field(default_factory=lambda: ["T1", "T2"])


class ConflictTriple(BaseModel):
    e1_fact_id: str
    e2_fact_id: str
    reason: str


class InferOut(BaseModel):
    mode_used: Literal["consistent", "conflict_set", "all"]
    answer_facts: list[Fact]
    confidence: float
    conflict_set: list[ConflictTriple] | None = None
    resolution: Resolution | None = None
    hops_taken: int
    judge_call_count: int  # MUST be ≤1 per call — the load-bearing assertion
    # Optional per-stage instrumentation. Keys when present:
    #   embed_ms       - embedder latency for the query
    #   qdrant_ms      - vector search latency
    #   cypher_ms      - hydrate + sibling collection
    #   judge_ms       - stage-2 LLM call (0 if not invoked)
    #   total_ms       - end-to-end
    #   candidates_n   - conflicts surfaced by stage-1 before truncation
    timings_ms: dict[str, float] | None = None


class FuseIn(BaseModel):
    scope: str
    group_id: str
    time_window_start: datetime | None = None
    time_window_end: datetime | None = None
    dry_run: bool = True


class FuseOut(BaseModel):
    proposed_merges: list[dict]
    proposed_supersessions: list[dict]
    applied: bool
    cache_hit_ratio: float


class InvalidateIn(BaseModel):
    fact_id: str
    reason: str
    invalidated_by: str | None = None
    by: Literal["user", "agent"] = "agent"


class InvalidateOut(BaseModel):
    fact_id: str
    invalid_at: datetime


class AttestIn(BaseModel):
    fact_id: str
    attestation: str
    by: Literal["user", "agent"] = "user"


class AttestOut(BaseModel):
    new_confidence: float
    pinned: bool


class DeleteIn(BaseModel):
    target_id: str
    type: Literal["episode", "fact"] = "fact"
    cascade: bool = False


class DeleteOut(BaseModel):
    ok: bool
    cascade_affected: list[str] = Field(default_factory=list)


class ClaimIn(BaseModel):
    resource_id: str
    ttl_ms: int = 5000


class ClaimOut(BaseModel):
    granted: bool
    ttl_ms: int


class ReleaseIn(BaseModel):
    resource_id: str


class ReleaseOut(BaseModel):
    released: bool


# ---------- B.2-v2 op payloads (safety layer, Phase 2) ----------


class AttestQuorumIn(BaseModel):
    fact_id: str
    attestation: str
    user_phrase: str
    agent_confidence: float
    fact_recall_session: str


class AttestQuorumOut(BaseModel):
    new_tier: Tier
    promotion_granted: bool
    failed_check: Literal["phrase", "agent_confidence", "recency"] | None = None


class SubscribeSignalsIn(BaseModel):
    channel: str
    action: Literal["open", "close"] = "open"
    throttle_ms: int = 200
    min_relevance: float = 0.4
    scope_filter: list[str] | None = None


class AcceptSignalIn(BaseModel):
    signal_id: str
    budget_tokens: int = 512
    mode: Literal["snippet", "full", "summary"] = "snippet"


class DismissSignalIn(BaseModel):
    signal_id: str
    reason: Literal["irrelevant", "poisoned", "duplicate", "low_tier"]
