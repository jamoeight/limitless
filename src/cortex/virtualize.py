"""Context virtualization — make a huge conversation fit in a small context.

The thesis: the frontier model doesn't need to see the whole conversation; it
needs to see (a) the last few turns verbatim and (b) a coherent recap of
everything older. We replace cold history with a single recap block injected
after the system prompt, leaving the role-alternating message list looking
clean and recent.

Two correctness invariants (both enforced by tests):
  1. Tool-use atomicity: a `tool_use_id` in an assistant turn MUST be paired
     with the matching `tool_result` in the next user turn within the SAME
     verbatim window. Splitting them returns a 400 from Anthropic.
  2. The system prompt is never collapsed. We only append to it (the recap).

The recap is computed by calling a `RecallFn` (the real implementation wraps
`timegraph.ops.graph_query` / `infer`). Tests inject a stub.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from cortex.canonical import (
    CortexMessage,
    CortexRequest,
    ToolResultBlock,
    ToolUseBlock,
)
from cortex.config import CortexSettings
from cortex.ingest import message_to_text

log = structlog.get_logger(__name__)


# ---------- Recall fn protocol ----------


# (query, group_id, token_budget) -> recap text. May return "" if no recall available.
RecallFn = Callable[[str, str, int], Awaitable[str]]


async def _noop_recall(query: str, group_id: str, token_budget: int) -> str:
    return ""


# ---------- Token estimation ----------


def approx_tokens(text: str) -> int:
    """Cheap upper-bound token estimator.

    char/4 is the standard rule of thumb for English text + tool-call JSON.
    For accurate counts we'd call a provider-specific tokenizer; that's a
    v1.5 optimization. For now, approx + a safety margin in the budget math
    catches the edge cases.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def message_tokens(message: CortexMessage) -> int:
    """Estimate the token cost of a single message."""
    return approx_tokens(message_to_text(message))


def messages_tokens(messages: list[CortexMessage]) -> int:
    return sum(message_tokens(m) for m in messages)


def tools_tokens(tools_serialized: list[dict[str, Any]]) -> int:
    if not tools_serialized:
        return 0
    blob = json.dumps(tools_serialized, default=str)
    return approx_tokens(blob)


# ---------- Provider context limits ----------


# Conservative defaults; overridable per request via X-Cortex-Context-Limit
# (not in v1) or by tuning here.
_CONTEXT_LIMITS = {
    "claude-opus-4-7": 200_000,
    "claude-opus-4-6": 200_000,
    "claude-opus-4-5": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
    "gpt-5": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "o3": 200_000,
    "o4-mini": 200_000,
}


def context_limit_for(model: str) -> int:
    """Return the upstream context limit for a model name."""
    m = model.lower()
    for k, v in _CONTEXT_LIMITS.items():
        if m.startswith(k):
            return v
    # Conservative default for unknown models.
    return 128_000


# ---------- Atomic-group detection (the hard invariant) ----------


def is_pure_tool_result(msg: CortexMessage) -> bool:
    """A user message that contains ONLY tool_result blocks.

    These are continuation messages — they must stay paired with the preceding
    assistant turn's `tool_use` blocks.
    """
    if msg.role != "user":
        return False
    if not msg.content:
        return False
    return all(isinstance(b, ToolResultBlock) for b in msg.content)


def has_open_tool_use(msg: CortexMessage) -> bool:
    """An assistant message containing one or more `tool_use` blocks."""
    if msg.role != "assistant":
        return False
    return any(isinstance(b, ToolUseBlock) for b in msg.content)


def compute_atomic_groups(messages: list[CortexMessage]) -> list[list[CortexMessage]]:
    """Bucket messages into atomic groups.

    A group starts with a user message that contains text (not just
    tool_result) and includes everything up to (but not including) the next
    such user message. This guarantees that any `tool_use → tool_result` chain
    started in a group stays in that group.

    A leading group of just (pure tool_result) is rare but possible (e.g.,
    resuming after a tool call) — we don't merge it into anything special;
    the virtualizer will keep it verbatim if it's in the last-K window.
    """
    groups: list[list[CortexMessage]] = []
    current: list[CortexMessage] = []
    for msg in messages:
        starts_new_group = (msg.role == "user") and (not is_pure_tool_result(msg))
        if starts_new_group and current:
            groups.append(current)
            current = []
        current.append(msg)
    if current:
        groups.append(current)
    return groups


# ---------- Recap construction ----------


def _summarize_message(msg: CortexMessage, max_chars: int = 280) -> str:
    """Single-line summary of a message used in the cold-history recap."""
    text = message_to_text(msg).strip().replace("\n", " ")
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return f"[{msg.role}] {text}"


def build_cold_summary(cold_groups: list[list[CortexMessage]], max_chars_per_msg: int = 280) -> str:
    """Brief bulleted recap of the cold (omitted) history.

    Future versions will replace this with LLM-generated summaries. For now,
    a one-line truncation per turn captures the gist without a model call.
    """
    if not cold_groups:
        return ""
    lines: list[str] = []
    for group in cold_groups:
        for msg in group:
            lines.append("- " + _summarize_message(msg, max_chars=max_chars_per_msg))
    return "\n".join(lines)


def last_user_query(messages: list[CortexMessage]) -> str:
    """Pull the most recent user-text content for use as a retrieval query."""
    for msg in reversed(messages):
        if msg.role == "user" and not is_pure_tool_result(msg):
            return message_to_text(msg)
    return ""


def assemble_recap(cold_summary: str, recall_text: str) -> str:
    """Build the recap block injected after the system prompt."""
    sections: list[str] = []
    if cold_summary:
        sections.append("Older conversation context (summarized):\n" + cold_summary)
    if recall_text:
        sections.append("Relevant retrieved knowledge:\n" + recall_text.strip())
    if not sections:
        return ""
    body = "\n\n".join(sections)
    return (
        "\n\n<cortex_memory>\n"
        "The following is reconstructed memory of context that does not appear "
        "in the visible message history. Treat it as authoritative for prior "
        "conversation and stored facts.\n\n"
        f"{body}\n"
        "</cortex_memory>"
    )


# ---------- Top-level virtualize() ----------


class VirtualizationReport:
    """Diagnostic info attached to the virtualized request — read in tests
    and surfaced as response headers in v1.5."""

    def __init__(self) -> None:
        self.original_message_count: int = 0
        self.original_token_estimate: int = 0
        self.kept_message_count: int = 0
        self.kept_token_estimate: int = 0
        self.recap_token_estimate: int = 0
        self.cold_group_count: int = 0
        self.cold_token_estimate: int = 0
        self.degraded: bool = False
        self.notes: list[str] = []

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_messages": self.original_message_count,
            "original_tokens": self.original_token_estimate,
            "kept_messages": self.kept_message_count,
            "kept_tokens": self.kept_token_estimate,
            "recap_tokens": self.recap_token_estimate,
            "cold_groups": self.cold_group_count,
            "cold_tokens": self.cold_token_estimate,
            "degraded": self.degraded,
            "notes": self.notes,
        }


async def virtualize(
    req: CortexRequest,
    settings: CortexSettings,
    *,
    recall_fn: RecallFn | None = None,
    context_limit: int | None = None,
    tools_serialized: list[dict[str, Any]] | None = None,
) -> tuple[CortexRequest, VirtualizationReport]:
    """Return a CortexRequest with virtualized history + a diagnostic report.

    The original `req` is not mutated. The returned request:
      - has the SAME `model`, `max_tokens`, `tools`, `tool_choice`, `temperature`
      - has the SAME `extras` and `cortex_*` fields
      - has its `system` field extended with a recap block (if any)
      - has its `messages` reduced to the last-K-groups window

    When the verbatim window alone already exceeds the budget (typical when a
    single message contains a giant file), virtualize gives up and returns the
    original request marked `degraded=True`. The upstream then either accepts
    it or returns 400 — the proxy doesn't pretend it can fix the impossible.
    """
    report = VirtualizationReport()
    report.original_message_count = len(req.messages)
    report.original_token_estimate = messages_tokens(req.messages)

    fn = recall_fn or _noop_recall

    # Compute budget.
    limit = context_limit if context_limit is not None else context_limit_for(req.model)
    system_t = approx_tokens(req.system or "")
    tools_t = tools_tokens(tools_serialized or [])
    M = limit - req.max_tokens - system_t - tools_t - settings.safety_margin_tokens

    groups = compute_atomic_groups(req.messages)
    if not groups:
        return req, report

    k_groups = max(1, settings.last_k_spans)  # name is historical
    verbatim_groups = groups[-k_groups:]
    cold_groups = groups[:-k_groups] if len(groups) > k_groups else []

    verbatim_msgs: list[CortexMessage] = []
    for g in verbatim_groups:
        verbatim_msgs.extend(g)

    verbatim_t = messages_tokens(verbatim_msgs)
    report.kept_message_count = len(verbatim_msgs)
    report.kept_token_estimate = verbatim_t
    report.cold_group_count = len(cold_groups)
    report.cold_token_estimate = messages_tokens([m for g in cold_groups for m in g])

    # Sanity: if verbatim is already too big, we can't help.
    if verbatim_t > M:
        report.degraded = True
        report.notes.append(
            f"verbatim_tokens={verbatim_t} exceeds budget M={M}; passing through unchanged"
        )
        return req, report

    # Build the recap. Total recap budget = remaining headroom after verbatim.
    # We ALWAYS run recall when virtualization is enabled — even when there
    # are no cold groups to summarize — because cross-session memory is the
    # main reason to enable it. Recall brings in pinned facts and prior-
    # session knowledge that has nothing to do with the current conversation
    # length.
    recall_budget = max(256, M - verbatim_t)
    cold_budget = int(recall_budget * settings.verbatim_budget_pct) if cold_groups else 0
    recall_only_budget = recall_budget - cold_budget

    cold_summary = ""
    if cold_groups:
        cold_summary = build_cold_summary(cold_groups)
        cold_summary = _truncate_to_tokens(cold_summary, cold_budget)

    query = last_user_query(verbatim_msgs)
    group_id = req.cortex_group_id or "default"
    try:
        recall_text = await fn(query, group_id, recall_only_budget)
    except Exception as e:  # noqa: BLE001
        log.warning("virtualize.recall_failed", error=str(e))
        recall_text = ""
        report.notes.append(f"recall_failed: {e}")

    # If recall AND cold-summary both produced nothing, the recap will be
    # empty — return the original request unchanged.
    if not cold_summary and not recall_text:
        report.notes.append("no recall hits and no cold history; pass-through")
        return req, report

    recap = assemble_recap(cold_summary, recall_text)
    report.recap_token_estimate = approx_tokens(recap)

    new_system = (req.system or "") + recap if recap else req.system
    new_req = req.model_copy(update={"system": new_system, "messages": verbatim_msgs})
    report.notes.append(
        f"virtualized: kept={report.kept_message_count}/{report.original_message_count} groups; "
        f"recap≈{report.recap_token_estimate}tok"
    )
    return new_req, report


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately `max_tokens` (using char/4 estimator)."""
    if max_tokens <= 0:
        return ""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"
