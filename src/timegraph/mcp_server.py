"""MCP server — exposes the timegraph capability layer to any MCP client
(opencode, Claude Desktop, Continue, custom Python client, ...).

Runs over stdio: `python -m timegraph.mcp_server`.

Tools registered (5 — the lean coding-agent surface):

  remember(content, source="agent", group_id="default", session_id=…)
      Store a piece of text as an episode; extract structured facts from it.
      Returns the episode id and any extracted facts.

  add_fact(subject, predicate, object, valid_at=…, group_id="default")
      Insert a structured (subject, predicate, object) edge directly. Use this
      when you already know the fact and don't need extraction.

  recall(query, k=8, group_id="default")
      Semantic search across stored episodes. Returns top-k chunks ordered
      by relevance, with provenance.

  query(question, group_id="default")
      Ask a question and let the system resolve any conflicting facts via the
      bounded-LLM-call judge. Returns an answer + the contradictions it
      considered.

  attest(fact_id, confirmed=True|False, attestation="...")
      Confirm or correct a stored fact. Bumps confidence and pins/unpins.

Backends required at runtime: Neo4j on bolt://localhost:7687, Qdrant on
localhost:6334 (gRPC), LM Studio on :1234 with qwen3.5-9b + nomic embedder.
Each is created lazily on first call.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from mcp.server.fastmcp import FastMCP

from timegraph.ops.add_episode import add_episode
from timegraph.ops.add_fact import add_fact
from timegraph.ops.attest import attest
from timegraph.ops.graph_query import graph_query
from timegraph.ops.infer import infer
from timegraph.types import (
    AddEpisodeIn,
    AddFactIn,
    AttestIn,
    GraphQueryIn,
    InferIn,
)

log = structlog.get_logger(__name__)

# A stable session id for the lifetime of this server process. Lets the agent
# omit `session_id` on every call; episodes/facts get tied to the same MCP
# session so we can audit later. Override per-call when you have something better.
_DEFAULT_SESSION_ID = os.environ.get("TG_MCP_SESSION_ID") or f"mcp-{uuid.uuid4().hex[:12]}"
_DEFAULT_GROUP_ID = os.environ.get("TG_MCP_GROUP_ID") or "default"


server = FastMCP("timegraph-mcp")


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _json(payload: Any) -> str:
    """Serialize a pydantic model (or anything default-serializable) for the
    MCP text content channel."""
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    return json.dumps(payload, default=str, indent=2)


@server.tool()
async def remember(
    content: str,
    source: str = "agent",
    group_id: str | None = None,
    session_id: str | None = None,
    event_time: str | None = None,
) -> str:
    """Store text as an episode and extract facts from it.

    Use for: ingesting documentation, capturing decisions, recording what
    happened ("user prefers tabs", "we ripped out the old auth middleware
    because legal flagged token storage").

    Args:
      content: free-form text to remember. Longer chunks are fine — the
        extractor will pull (subject, predicate, object) triples from it.
      source: where the text came from — e.g. "agent", "user", "file:auth.py",
        "slack:#eng". Used for provenance.
      group_id: tenant / project namespace. Defaults to env TG_MCP_GROUP_ID
        or "default".
      session_id: groups episodes inside a single working session. Defaults
        to a stable per-process id.
      event_time: ISO 8601 timestamp. Defaults to now.

    Returns: JSON with episode_id, extracted_facts (list), conflicts_detected.
    """
    ts = _now() if event_time is None else datetime.fromisoformat(event_time.replace("Z", "+00:00"))
    payload = AddEpisodeIn(
        content=content,
        source=source,
        group_id=group_id or _DEFAULT_GROUP_ID,
        session_id=session_id or _DEFAULT_SESSION_ID,
        event_time=ts,
    )
    result = await add_episode(payload)
    return _json(result)


@server.tool(name="add_fact")
async def add_fact_tool(
    subject: str,
    predicate: str,
    object: str,
    event_time: str | None = None,
    source: str = "agent",
    group_id: str | None = None,
    session_id: str | None = None,
    confidence: float = 1.0,
    pinned: bool = False,
) -> str:
    """Insert a structured (subject, predicate, object) fact directly.

    Use when you already know the fact and don't need extraction — e.g. wiring
    in a fact from an authoritative source. For free-form text, prefer `remember`.

    Args:
      subject: e.g. "alice"
      predicate: e.g. "lives_in"
      object: e.g. "Seattle"
      event_time: ISO 8601 of when this fact became valid. Defaults to now.
      source: provenance string.
      group_id: tenant / project namespace.
      session_id: working-session id.
      confidence: 0.0–1.0; higher means the system trusts it more during conflict
        resolution.
      pinned: if true, treats this fact as ground truth.

    Returns: JSON with fact_id, edges_created, conflicts_with.
    """
    ts = _now() if event_time is None else datetime.fromisoformat(event_time.replace("Z", "+00:00"))
    payload = AddFactIn(
        subject=subject,
        predicate=predicate,
        object=object,
        event_time=ts,
        source=source,
        group_id=group_id or _DEFAULT_GROUP_ID,
        session_id=session_id or _DEFAULT_SESSION_ID,
        confidence=confidence,
        pinned=pinned,
    )
    result = await add_fact(payload)
    return _json(result)


@server.tool()
async def recall(
    query: str,
    k: int = 8,
    group_id: str | None = None,
    budget_tokens: int = 1024,
) -> str:
    """Semantic search over stored episodes. Use this to load relevant context
    before answering a question or editing code.

    The LLM never sees your full memory — `recall` returns the top-k chunks
    matching `query` (by embedding similarity), with provenance and a token
    budget cap.

    Args:
      query: natural-language search string.
      k: how many top hits to return.
      group_id: tenant / project namespace.
      budget_tokens: hard cap on total tokens returned across all hits.

    Returns: JSON with mode="nodes", results (list of chunks), provenance,
    tokens_used.
    """
    payload = GraphQueryIn(
        query=query,
        scope=group_id or _DEFAULT_GROUP_ID,
        mode="nodes",
        k=k,
        budget_tokens=budget_tokens,
    )
    result = await graph_query(payload)
    return _json(result)


@server.tool()
async def query(
    question: str,
    group_id: str | None = None,
) -> str:
    """Ask a question; the system retrieves relevant facts and resolves any
    contradictions via a single LLM judgment call (the bounded-LLM-call
    pattern). Use when there might be conflicting prior facts and you want
    the system to pick the right one.

    Args:
      question: natural-language question.
      group_id: tenant / project namespace.

    Returns: JSON with answer_facts, resolution, conflict_set (the facts the
    judge considered), confidence, judge_call_count (always ≤1), timings_ms.
    """
    payload = InferIn(
        query=question,
        scope=group_id or _DEFAULT_GROUP_ID,
        mode="conflict_set",
    )
    result = await infer(payload)
    return _json(result)


@server.tool(name="attest")
async def attest_fact(
    fact_id: str,
    confirmed: bool,
    attestation: str = "",
) -> str:
    """Confirm or correct a stored fact. `confirmed=True` bumps confidence
    and pins the fact; `confirmed=False` reduces confidence and unpins.

    Args:
      fact_id: the id returned by `remember` (in extracted_facts) or `add_fact`.
      confirmed: True for "yes that's right", False for "that's wrong".
      attestation: free-form rationale, stored for audit.

    Returns: JSON with new_confidence, pinned.
    """
    payload = AttestIn(
        fact_id=fact_id,
        attestation=attestation or ("confirmed" if confirmed else "corrected"),
        by="user",
    )
    result = await attest(payload)
    return _json(result)


def main() -> None:
    """Console entrypoint — runs over stdio for MCP clients."""
    log.info(
        "timegraph-mcp starting",
        session_id=_DEFAULT_SESSION_ID,
        group_id=_DEFAULT_GROUP_ID,
    )
    server.run()


if __name__ == "__main__":
    main()
