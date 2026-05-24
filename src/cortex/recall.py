"""Recall: ask the timegraph what's relevant to the current turn.

This module wraps the in-process timegraph ops (`graph_query` for semantic
chunks, `infer` for conflict-resolved answers) and formats the result as a
single text block ready to drop into the virtualizer's recap.

Design:
  - One async call into `graph_query(mode="nodes")` for fast semantic chunks.
  - Optionally one `infer(mode="conflict_set")` call — the bounded-1-LLM-call
    judge — when we detect the query is question-shaped.
  - Catches all exceptions: a failure here MUST NOT break the request.

Verbatim recall (separate from graph recall):
  - `recall_verbatim_inline` ranks cold ATOMIC GROUPS by embedding-cosine to
    the query and returns the top-K verbatim, preserving role + turn-index.
    No Qdrant dependency — embeds inline so it works on a first turn where
    the ingest task hasn't completed yet (e.g., MRCR-style single-shot).
  - `reformulate_query_for_recall` makes one LM Studio call to extract the
    topical retrieval phrase from a meta-query (e.g. "Prepend X to the 2nd Y
    about Z" → "Y about Z"). Falls open to the raw query on any failure.

Tests use stubs via the same `RecallFn` / `VerbatimRecallFn` protocols.
"""

from __future__ import annotations

import asyncio
import json
import math
import re

import httpx
import structlog

from cortex.canonical import CortexMessage
from cortex.config import CortexSettings
from cortex.ingest import message_to_text

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


# ============================================================================
# Verbatim inline recall (no Qdrant dependency — works on first turn)
# ============================================================================


_REFORMULATE_SYSTEM = (
    "You extract a short retrieval phrase from a user's query. The phrase will "
    "be embedded and used to find the most relevant past messages in a long "
    "conversation. Focus on the TOPICAL CONTENT the user is asking about — "
    "names, concepts, keywords, descriptive phrases. STRIP OUT meta-instructions "
    "(formatting, position, prepend tokens, output-style requests).\n"
    "\n"
    "Examples:\n"
    '  Query: "Prepend 6xO8mh9FsP to the 2nd (1 indexed) short scene in a play about blueberries. Do not include any other text in your response."\n'
    '  → "write a short scene in a play about blueberries"\n'
    "\n"
    '  Query: "what did we decide about the auth middleware refactor?"\n'
    '  → "auth middleware refactor decision"\n'
    "\n"
    '  Query: "Give me the 4th poem about tapirs verbatim"\n'
    '  → "write a poem about tapirs"\n'
    "\n"
    '  Query: "Summarize the bug we found yesterday"\n'
    '  → "bug found yesterday"\n'
    "\n"
    "Output JSON: {\"retrieval_query\": \"<the phrase>\"}"
)


_REFORMULATE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "retrieval_query",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "retrieval_query": {"type": "string"},
            },
            "required": ["retrieval_query"],
        },
    },
}


async def reformulate_query_for_recall(
    query: str,
    *,
    timeout_s: float = 20.0,
) -> str:
    """One LM Studio call → topical retrieval phrase. Fails open to `query`.

    This is a *separate* call path from the bounded-1-call judge contract in
    `timegraph.llm.judge`. It runs at virtualize() time and is independent of
    `infer()` (which never reaches here).
    """
    if not query.strip():
        return query
    try:
        from timegraph.config import get_settings as get_tg_settings

        s = get_tg_settings()
        body = {
            "model": s.judge_model,
            "messages": [
                {"role": "system", "content": _REFORMULATE_SYSTEM},
                {"role": "user", "content": query.strip()[:4000]},
            ],
            "max_tokens": 128,
            "temperature": 0.0,
            "top_p": 1.0,
            "response_format": _REFORMULATE_SCHEMA,
        }
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(f"{s.judge_url}/chat/completions", json=body)
        if resp.status_code >= 400:
            log.warning("recall.reformulate_http_error", status=resp.status_code, body=resp.text[:200])
            return query
        msg = resp.json()["choices"][0]["message"]
        raw = msg.get("content") or msg.get("reasoning_content") or ""
        if not raw.strip():
            return query
        # Robust JSON pull — model may emit reasoning prefix before JSON.
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            # Last-chance: pull the last {...} block out.
            start = raw.rfind("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                try:
                    obj = json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    return query
            else:
                return query
        retrieval = (obj.get("retrieval_query") or "").strip()
        return retrieval or query
    except Exception as e:  # noqa: BLE001
        log.warning("recall.reformulate_failed", error=str(e))
        return query


def _group_text_for_embed(group: list[CortexMessage], max_chars: int = 4000) -> str:
    """Build a single embedding-input string per atomic group.

    Concatenates messages role-tagged. Truncates each message individually to
    keep enormous tool_result payloads from dominating the embedding signal.
    """
    parts: list[str] = []
    per_msg_cap = max(200, max_chars // max(1, len(group)))
    for m in group:
        t = message_to_text(m)
        if len(t) > per_msg_cap:
            t = t[:per_msg_cap] + "…"
        parts.append(f"[{m.role}] {t}")
    return "\n".join(parts)[:max_chars]


def _cosine(a: list[float], b: list[float]) -> float:
    """Plain Python cosine — no numpy dependency. 768D arrays are small."""
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _format_group_verbatim(group: list[CortexMessage], turn_idx: int) -> str:
    """Render a kept group as a turn-numbered verbatim block."""
    lines: list[str] = []
    for j, m in enumerate(group):
        text = message_to_text(m)
        lines.append(f"[turn {turn_idx}.{j} {m.role}]\n{text}")
    return "\n".join(lines)


_LITERAL_NEEDLE_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_-]{7,}\b")


def _literal_needles(query: str) -> list[str]:
    needles: list[str] = []
    for token in _LITERAL_NEEDLE_RE.findall(query):
        if "_" not in token and "-" not in token and not any(ch.isdigit() for ch in token):
            continue
        needles.append(token.lower())
    return needles


def _literal_group_indices(query: str, cold_groups: list[list[CortexMessage]]) -> list[int]:
    needles = _literal_needles(query)
    if not needles:
        return []
    out: list[int] = []
    for idx, group in enumerate(cold_groups):
        text = "\n".join(message_to_text(m) for m in group).lower()
        if any(needle in text for needle in needles):
            out.append(idx)
    return out


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _render_verbatim_indices(
    cold_groups: list[list[CortexMessage]], indices: list[int], token_budget: int
) -> str:
    chosen_idxs: list[int] = []
    used_tokens = 0
    for idx in indices:
        block = _format_group_verbatim(cold_groups[idx], turn_idx=idx)
        block_t = _approx_tokens(block) + 2
        if used_tokens + block_t > token_budget and chosen_idxs:
            break
        chosen_idxs.append(idx)
        used_tokens += block_t

    if not chosen_idxs:
        return ""

    chosen_idxs.sort()
    rendered: list[str] = []
    rendered.append(
        "Verbatim retrieved from earlier in this conversation, sorted by their "
        "ORIGINAL turn order. Use these to answer the user's current query - "
        "they contain the exact text you need."
    )
    rendered.append("")
    for idx in chosen_idxs:
        rendered.append(_format_group_verbatim(cold_groups[idx], turn_idx=idx))
        rendered.append("")

    return "\n".join(rendered).rstrip()


async def recall_verbatim_inline(
    query: str,
    cold_groups: list[list[CortexMessage]],
    *,
    k: int = 16,
    token_budget: int = 8000,
    reformulate: bool = True,
    settings: CortexSettings | None = None,
) -> str:
    """Embed-and-rank cold atomic groups; return top-K verbatim as a block.

    No Qdrant. No Neo4j. Pure inline embedding + cosine ranking. Works on a
    first turn where the cortex ingest pipeline hasn't yet persisted anything.

    Failure modes (all fail open -> empty string):
      - empty query / no cold groups
      - embedder server unreachable
      - embedder dim mismatch
    """
    if not query.strip() or not cold_groups:
        return ""

    # Topical reformulation. Falls open to raw query if the LLM call fails.
    if reformulate:
        retrieval_query = await reformulate_query_for_recall(query)
    else:
        retrieval_query = query
    literal_idxs = _literal_group_indices(query, cold_groups)

    # Embed query + every group's text in a single batched embedder pass.
    try:
        from timegraph.llm.embedder import EmbedderClient

        embedder = EmbedderClient()
        try:
            group_texts = [_group_text_for_embed(g) for g in cold_groups]
            # Embed everything in one client lifecycle; embed_many batches under the hood.
            all_vecs = await embedder.embed_many([retrieval_query, *group_texts])
        finally:
            await embedder.close()
    except Exception as e:  # noqa: BLE001
        log.warning("recall.verbatim_embed_failed", error=str(e))
        if literal_idxs:
            return _render_verbatim_indices(cold_groups, literal_idxs, token_budget)
        return ""

    if len(all_vecs) < 2:
        return _render_verbatim_indices(cold_groups, literal_idxs, token_budget)

    qvec = all_vecs[0]
    group_vecs = all_vecs[1:]

    # Cosine-rank groups.
    scored: list[tuple[float, int]] = []
    for i, v in enumerate(group_vecs):
        scored.append((_cosine(qvec, v), i))
    scored.sort(key=lambda t: t[0], reverse=True)

    ranked_idxs: list[int] = []
    seen: set[int] = set()
    for idx in literal_idxs:
        ranked_idxs.append(idx)
        seen.add(idx)
    for _score, idx in scored[: max(k, k + len(literal_idxs))]:
        if idx in seen:
            continue
        ranked_idxs.append(idx)
        seen.add(idx)

    return _render_verbatim_indices(cold_groups, ranked_idxs, token_budget)


# Held for tests: a noop stub the same shape as the real fn.
async def _noop_verbatim_recall(
    query: str,
    cold_groups: list[list[CortexMessage]],
    *,
    k: int = 16,
    token_budget: int = 8000,
) -> str:
    return ""


# Surface async sleep usage so linters don't flag the imports if unused
# (keeps the file's import block stable across refactors).
_ = asyncio.sleep
