"""Recall: ask the timegraph what's relevant to the current turn.

This module wraps the in-process timegraph ops (`graph_query` for semantic
chunks, `infer` for conflict-resolved answers) and formats the result as a
single text block ready to drop into the virtualizer's recap.

Design:
  - One async call into `graph_query(mode="nodes")` for fast semantic chunks.
  - Optionally one `infer(mode="conflict_set")` call — the bounded-1-LLM-call
    judge — when we detect the query is question-shaped. This is the load-
    bearing pattern from the BEAM benchmark. We skip it for short imperative
    inputs ("look at file X", "run tests") where conflict resolution is moot.
  - Catches all exceptions: a failure here MUST NOT break the request. We
    return an empty recap, and the proxy proceeds with whatever verbatim
    history it has. Plays into MVP-5 graceful-degradation.

Tests use a stub via the same `RecallFn` protocol the virtualizer takes.
"""

from __future__ import annotations

import structlog

from cortex.config import CortexSettings

log = structlog.get_logger(__name__)


def _looks_like_question(text: str) -> bool:
    """Cheap heuristic: question marks, or starts with 5W1H."""
    if not text:
        return False
    t = text.strip().lower()
    if "?" in t:
        return True
    return t.split(" ", 1)[0] in {"what", "why", "when", "where", "who", "how", "which", "is", "are", "do", "does", "should", "can"}


async def real_recall(
    query: str,
    group_id: str,
    token_budget: int,
    *,
    settings: CortexSettings | None = None,
) -> str:
    """Real recall: in-process call into timegraph ops.

    Returns formatted recap text. Empty string on any failure (the proxy
    treats empty recap as "no memory available" and proceeds).
    """
    if not query.strip():
        return ""
    settings = settings or CortexSettings()

    try:
        from timegraph.config import get_settings as get_tg_settings
        from timegraph.ops.graph_query import graph_query
        from timegraph.types import GraphQueryIn

        s = get_tg_settings()
        payload = GraphQueryIn(
            query=query,
            scope=group_id,
            mode="nodes",
            k=max(4, s.default_recall_k),
            budget_tokens=max(256, token_budget // 2),  # leave room for infer's answer
        )
        gq_result = await graph_query(payload)
    except Exception as e:  # noqa: BLE001
        log.warning("recall.graph_query_failed", error=str(e), group=group_id)
        return ""

    # Format semantic chunks.
    semantic_lines: list[str] = []
    for hit in gq_result.results[:8]:  # cap regardless of k
        text = (hit.get("content") or hit.get("text") or "").strip()
        if not text:
            continue
        # One-line per chunk.
        first_line = text.replace("\n", " ").strip()
        if len(first_line) > 320:
            first_line = first_line[:319] + "…"
        ep_id = hit.get("episode_id") or hit.get("id") or ""
        marker = f" [ep:{ep_id[:8]}]" if ep_id else ""
        semantic_lines.append(f"- {first_line}{marker}")

    semantic_block = ""
    if semantic_lines:
        semantic_block = "Semantic recall:\n" + "\n".join(semantic_lines)

    # Optionally fire the bounded-1-call judge for question-shaped queries.
    infer_block = ""
    if _looks_like_question(query):
        try:
            from timegraph.ops.infer import infer
            from timegraph.types import InferIn

            inf_result = await infer(InferIn(query=query, scope=group_id, mode="conflict_set"))
            if inf_result.answer_facts:
                fact_lines = [
                    f"- ({f.subject}, {f.predicate}, {f.object})"
                    + (f" [conf {f.confidence:.2f}]" if f.confidence != 1.0 else "")
                    + (" [pinned]" if f.pinned else "")
                    for f in inf_result.answer_facts[:6]
                ]
                header = f"Judged answer (resolution={inf_result.resolution.value if inf_result.resolution else 'n/a'}, conf={inf_result.confidence:.2f}):"
                infer_block = header + "\n" + "\n".join(fact_lines)
        except Exception as e:  # noqa: BLE001
            log.warning("recall.infer_failed", error=str(e), group=group_id)

    parts = [p for p in (infer_block, semantic_block) if p]
    return "\n\n".join(parts)
