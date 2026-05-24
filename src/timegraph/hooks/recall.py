"""Claude Code UserPromptSubmit hook — injects timegraph recall as additionalContext.

Wired by the `timegraph-cortex` plugin manifest. On every user prompt, runs two
parallel semantic recalls against the local timegraph and emits
`hookSpecificOutput.additionalContext`:

  1. Fact recall  — top-k entries from the `facts` collection
                    (subject/predicate/object triples extracted from prior
                    user + assistant *statements*). High-signal, dense.

  2. Episode recall — top-k entries from the `episodes` collection
                      (raw text bodies from tool results: file reads, bash
                      outputs, web fetches, etc.). This is what makes recall
                      survive Claude Code's auto-compaction — the file you
                      read in turn 3 is still retrievable in turn 50.

Both recalls run with their own token budget so the more-numerous (and lower
signal-density) episode hits don't starve facts. Results are merged into one
markdown block — the user never calls a recall tool by hand.

Fails open: any error (backends down, empty results, decode failures) exits 0
with no stdout and Claude Code proceeds as if no plugin were installed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

# Route structlog + stdlib logging to stderr BEFORE any timegraph.ops import,
# so library log lines don't land on stdout and break the JSON channel.
from timegraph.hooks._log import silence_to_stderr  # noqa: E402

silence_to_stderr()

# Default to the LM-Studio-free path for plugin installs. setdefault preserves
# any user override (e.g. dev with LM Studio loaded). Must run before any
# timegraph.ops import — those instantiate Settings which reads env once.
os.environ.setdefault("TG_JUDGE_BACKEND", "claude_cli")
os.environ.setdefault("TG_JUDGE_CLAUDE_MODEL", "haiku")

from timegraph.project_id import derive_group_id  # noqa: E402

RECALL_K = int(os.environ.get("TG_HOOK_RECALL_K", "6"))
RECALL_BUDGET_TOKENS = int(os.environ.get("TG_HOOK_RECALL_BUDGET", "1500"))
EPISODE_K = int(os.environ.get("TG_HOOK_EPISODE_K", "5"))
EPISODE_BUDGET_TOKENS = int(os.environ.get("TG_HOOK_EPISODE_BUDGET", "2500"))
EPISODE_SNIPPET_CHARS = int(os.environ.get("TG_HOOK_EPISODE_SNIPPET_CHARS", "1200"))
MIN_PROMPT_CHARS = int(os.environ.get("TG_HOOK_MIN_PROMPT_CHARS", "8"))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--group-id", default=None)
    args, _unknown = p.parse_known_args(argv)
    return args


def _emit(additional_context: str | None) -> None:
    if not additional_context:
        return
    out = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": additional_context,
        }
    }
    sys.stdout.write(json.dumps(out))
    sys.stdout.flush()


def _attr(item, key, default=None):
    if hasattr(item, key):
        return getattr(item, key)
    if isinstance(item, dict):
        return item.get(key, default)
    return default


async def _recall_facts(prompt: str, group_id: str) -> list:
    from timegraph.ops.graph_query import graph_query
    from timegraph.types import GraphQueryIn

    payload = GraphQueryIn(
        query=prompt,
        scope=group_id,
        mode="nodes",
        k=RECALL_K,
        budget_tokens=RECALL_BUDGET_TOKENS,
    )
    result = await graph_query(payload)
    return getattr(result, "results", None) or []


async def _recall_episodes(prompt: str, group_id: str) -> list[dict]:
    """Search the episodes collection directly — captures tool-result content
    (file reads, bash outputs, etc.) that doesn't generate facts. Returns
    dicts with {source, content, event_time, session_id}."""
    from timegraph.config import get_settings
    from timegraph.llm.embedder import EmbedderClient
    from timegraph.storage.neo4j_client import get_session
    from timegraph.storage.qdrant_client import search as qdrant_search

    s = get_settings()
    embedder = EmbedderClient()
    try:
        qvec = await embedder.embed_one(prompt)
    finally:
        await embedder.close()

    hits = await qdrant_search(
        collection=s.qdrant_episodes_collection,
        vector=qvec,
        k=EPISODE_K,
        group_id=group_id,
    )
    if not hits:
        return []

    episode_ids = [str(h.id) for h in hits]
    async with get_session() as session:
        res = await session.run(
            """
            MATCH (e:Episode)
            WHERE e.id IN $ids
            RETURN e.id           AS episode_id,
                   e.content      AS content,
                   e.source       AS source,
                   e.event_time   AS event_time,
                   e.session_id   AS session_id
            """,
            ids=episode_ids,
        )
        by_id: dict[str, dict] = {}
        async for row in res:
            d = row.data()
            by_id[d["episode_id"]] = d

    # Preserve Qdrant ranking
    return [by_id[i] for i in episode_ids if i in by_id]


def _format_facts(facts: list) -> list[str]:
    lines: list[str] = []
    for f in facts:
        subj = _attr(f, "subject") or ""
        pred = _attr(f, "predicate") or ""
        obj = _attr(f, "object") or ""
        if not (subj and pred and obj):
            continue
        valid_at = _attr(f, "valid_at")
        conf = _attr(f, "confidence")
        tier = _attr(f, "tier") or ""
        meta_bits = []
        if valid_at:
            meta_bits.append(str(valid_at)[:19])
        if tier:
            meta_bits.append(tier)
        if conf is not None:
            meta_bits.append(f"c={conf:.2f}" if isinstance(conf, float) else f"c={conf}")
        meta = f"  _({', '.join(meta_bits)})_" if meta_bits else ""
        lines.append(f"- **{subj}** {pred} **{obj}**{meta}")
    return lines


def _format_episodes(eps: list[dict], budget_tokens: int) -> list[str]:
    """Render episodes as truncated code-fenced snippets so the model sees the
    actual content (file body, bash stdout, etc.) — not just metadata. Stops
    appending once the rough char budget is exhausted."""
    budget_chars = budget_tokens * 4
    used = 0
    out: list[str] = []
    for e in eps:
        source = e.get("source") or "episode"
        content = (e.get("content") or "").strip()
        if not content:
            continue
        snippet = content if len(content) <= EPISODE_SNIPPET_CHARS else (
            content[:EPISODE_SNIPPET_CHARS] + "\n[…truncated]"
        )
        ts = str(e.get("event_time") or "")[:19]
        sess = e.get("session_id") or ""
        header = f"### {source}" + (f" — {ts}" if ts else "") + (f"  _(session {sess[:12]})_" if sess else "")
        block = f"{header}\n```\n{snippet}\n```"
        cost = len(block)
        if used + cost > budget_chars and out:
            break
        out.append(block)
        used += cost
    return out


def _compose(fact_lines: list[str], episode_blocks: list[str]) -> str | None:
    sections: list[str] = []
    if fact_lines:
        sections.append("## Relevant memory from prior sessions\n" + "\n".join(fact_lines))
    if episode_blocks:
        sections.append("## Recalled tool results and prior context\n" + "\n\n".join(episode_blocks))
    if not sections:
        return None
    return "\n\n".join(sections)


async def _recall_all(prompt: str, group_id: str) -> str | None:
    facts, episodes = await asyncio.gather(
        _recall_facts(prompt, group_id),
        _recall_episodes(prompt, group_id),
        return_exceptions=True,
    )
    fact_lines = _format_facts(facts) if not isinstance(facts, Exception) else []
    episode_blocks = _format_episodes(episodes, EPISODE_BUDGET_TOKENS) if not isinstance(episodes, Exception) else []
    return _compose(fact_lines, episode_blocks)


def main() -> None:
    args = _parse_args(sys.argv[1:])
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        return

    prompt = (payload.get("prompt") or "").strip()
    if len(prompt) < MIN_PROMPT_CHARS:
        return

    cwd = payload.get("cwd")
    group_id = args.group_id or derive_group_id(cwd)

    try:
        ctx = asyncio.run(_recall_all(prompt, group_id))
    except Exception:
        return

    _emit(ctx)


if __name__ == "__main__":
    main()
