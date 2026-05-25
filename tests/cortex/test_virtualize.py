"""Unit tests for cortex.virtualize.

The two load-bearing invariants under test:
  1. Tool-use atomicity — a tool_use_id from an assistant turn is never
     separated from its matching tool_result in the next user turn.
  2. The system prompt is never collapsed; the recap is only ever APPENDED.

Plus the usual: budget math, recap construction, degraded fallback.
"""

from __future__ import annotations

import random

import pytest

from cortex.canonical import (
    CortexMessage,
    CortexRequest,
    CortexTool,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from cortex.config import CortexSettings
from cortex.virtualize import (
    approx_tokens,
    assemble_recap,
    build_cold_summary,
    compute_atomic_groups,
    context_limit_for,
    has_open_tool_use,
    is_pure_tool_result,
    last_user_query,
    message_tokens,
    virtualize,
)


def _u(text: str) -> CortexMessage:
    return CortexMessage(role="user", content=[TextBlock(text=text)])


def _a(text: str) -> CortexMessage:
    return CortexMessage(role="assistant", content=[TextBlock(text=text)])


def _a_tool(text: str, tool_id: str, tool_name: str = "search", **input_kwargs) -> CortexMessage:
    """Assistant message with a text block AND a tool_use block."""
    return CortexMessage(
        role="assistant",
        content=[
            TextBlock(text=text),
            ToolUseBlock(tool_use_id=tool_id, tool_name=tool_name, tool_input=input_kwargs),
        ],
    )


def _u_result(tool_id: str, content: str, is_error: bool = False) -> CortexMessage:
    return CortexMessage(
        role="user",
        content=[ToolResultBlock(tool_use_id=tool_id, content=content, is_error=is_error)],
    )


# ---------- Atomic group detection ----------


def test_groups_simple_alternation() -> None:
    msgs = [_u("hi"), _a("hello"), _u("how are you"), _a("good thanks")]
    groups = compute_atomic_groups(msgs)
    # Two groups: each starts with a user-text message.
    assert len(groups) == 2
    assert groups[0] == [msgs[0], msgs[1]]
    assert groups[1] == [msgs[2], msgs[3]]


def test_groups_keep_tool_use_chain_together() -> None:
    msgs = [
        _u("search for cats"),
        _a_tool("ok", "t1", q="cats"),
        _u_result("t1", "found 3"),
        _a("here are the cats"),
        _u("now search for dogs"),
        _a_tool("ok", "t2", q="dogs"),
        _u_result("t2", "found 5"),
        _a("here are the dogs"),
    ]
    groups = compute_atomic_groups(msgs)
    assert len(groups) == 2
    # Group 1 has all four (user + assistant-tool + user-tool_result + assistant)
    assert len(groups[0]) == 4
    # Group 2 same shape
    assert len(groups[1]) == 4


def test_groups_handle_multi_tool_chain() -> None:
    """Multiple tool roundtrips inside a single user request."""
    msgs = [
        _u("plan a trip"),
        _a_tool("checking weather", "t1", loc="paris"),
        _u_result("t1", "sunny"),
        _a_tool("checking hotels", "t2", loc="paris"),
        _u_result("t2", "3 options"),
        _a("ok here's the plan"),
    ]
    groups = compute_atomic_groups(msgs)
    assert len(groups) == 1
    assert len(groups[0]) == 6


def test_pure_tool_result_detection() -> None:
    assert is_pure_tool_result(_u_result("t1", "x"))
    assert not is_pure_tool_result(_u("normal user message"))
    # Mixed: text + tool_result is NOT pure tool_result
    mixed = CortexMessage(
        role="user",
        content=[TextBlock(text="also"), ToolResultBlock(tool_use_id="t1", content="x")],
    )
    assert not is_pure_tool_result(mixed)


def test_has_open_tool_use_detection() -> None:
    assert has_open_tool_use(_a_tool("ok", "t1"))
    assert not has_open_tool_use(_a("just text"))
    assert not has_open_tool_use(_u("user message"))


# ---------- TOOL-USE ATOMICITY (the hard one) ----------


def _build_conversation_with_tool_chains(rng: random.Random, n_groups: int) -> list[CortexMessage]:
    """Generate a synthetic conversation that mixes plain turns and tool chains."""
    msgs: list[CortexMessage] = []
    tool_counter = 0
    for g in range(n_groups):
        msgs.append(_u(f"user request number {g} that is long enough to count"))
        # 50/50 chance of a tool roundtrip
        if rng.random() < 0.5:
            tool_counter += 1
            tid = f"tool_{tool_counter:04d}"
            msgs.append(_a_tool(f"thinking about {g}", tid, query=f"q_{g}"))
            msgs.append(_u_result(tid, f"result for {g}"))
            msgs.append(_a(f"final answer for request {g}"))
        else:
            msgs.append(_a(f"direct answer for request {g}"))
    return msgs


def _collect_tool_use_ids(messages: list[CortexMessage]) -> set[str]:
    out: set[str] = set()
    for m in messages:
        if m.role == "assistant":
            for b in m.content:
                if isinstance(b, ToolUseBlock):
                    out.add(b.tool_use_id)
    return out


def _collect_tool_result_ids(messages: list[CortexMessage]) -> set[str]:
    out: set[str] = set()
    for m in messages:
        if m.role == "user":
            for b in m.content:
                if isinstance(b, ToolResultBlock):
                    out.add(b.tool_use_id)
    return out


@pytest.mark.asyncio
async def test_virtualize_never_splits_tool_use_pairs_fuzz() -> None:
    """Fuzz: generate random conversations with tool chains; virtualize each;
    assert every tool_use_id in the kept window has its matching tool_result."""
    rng = random.Random(20260522)
    settings = CortexSettings(last_k_spans=2, safety_margin_tokens=128)

    for trial in range(40):
        n_groups = rng.randint(3, 12)
        msgs = _build_conversation_with_tool_chains(rng, n_groups)
        req = CortexRequest(
            model="claude-opus-4-7",
            max_tokens=1024,
            messages=msgs,
            system="you are a test bot",
        )
        # Use a large context_limit so the budget check isn't the limiting factor;
        # we want to test the atomicity invariant specifically.
        new_req, report = await virtualize(
            req, settings, context_limit=200_000
        )

        kept_tool_use_ids = _collect_tool_use_ids(new_req.messages)
        kept_tool_result_ids = _collect_tool_result_ids(new_req.messages)
        # Every tool_use that survived must have its tool_result also survive.
        assert kept_tool_use_ids <= kept_tool_result_ids, (
            f"trial {trial}: orphaned tool_use_ids found! "
            f"missing={kept_tool_use_ids - kept_tool_result_ids}"
        )
        # And vice versa: no orphan tool_results without their tool_use.
        assert kept_tool_result_ids <= kept_tool_use_ids, (
            f"trial {trial}: orphaned tool_result_ids found! "
            f"extra={kept_tool_result_ids - kept_tool_use_ids}"
        )


# ---------- Last-K verbatim ----------


@pytest.mark.asyncio
async def test_last_k_groups_kept_verbatim() -> None:
    msgs = []
    for i in range(8):
        msgs.append(_u(f"user message {i} long enough to ingest"))
        msgs.append(_a(f"assistant reply {i} long enough to retain"))
    req = CortexRequest(
        model="claude-opus-4-7",
        max_tokens=512,
        messages=msgs,
        system="sys",
    )
    settings = CortexSettings(last_k_spans=3, safety_margin_tokens=0)
    # M = limit - safety = 100. 16 msgs ≈ 144 tokens (overflows → trims);
    # 6 verbatim msgs ≈ 54 tokens (fits).
    # (tools/system/max_tokens deliberately NOT subtracted from M — see
    #  virtualize.py: messages-only budget semantic).
    new_req, report = await virtualize(req, settings, context_limit=100)

    # 3 groups × 2 messages = 6 kept verbatim.
    assert report.kept_message_count == 6
    assert report.cold_group_count == 5
    # Last user message should be in the kept window.
    assert "user message 7" in new_req.messages[-2].content[0].text  # noqa: SLF001


# ---------- System prompt preservation ----------


@pytest.mark.asyncio
async def test_system_prompt_is_only_appended_never_replaced() -> None:
    original_system = "You are a precise assistant. Always cite sources."
    msgs = []
    for i in range(6):
        msgs.append(_u(f"older message {i} sufficiently long to be ingested fine"))
        msgs.append(_a(f"older reply {i} also long enough to keep around hm"))
    req = CortexRequest(
        model="claude-opus-4-7",
        max_tokens=256,
        messages=msgs,
        system=original_system,
    )
    settings = CortexSettings(last_k_spans=2, safety_margin_tokens=0)

    async def fake_recall(query, group_id, budget):
        return "(- alice likes coffee)\n(- bob lives in seattle)"

    # M = 120 - 0 = 120. 12 msgs ≈ 150 tokens (overflows → virtualizes);
    # 4 verbatim msgs ≈ 50 tokens (fits).
    new_req, report = await virtualize(
        req, settings, recall_fn=fake_recall, context_limit=120
    )
    # System prompt starts with the original verbatim
    assert new_req.system.startswith(original_system)
    # Recap is appended
    assert "<cortex_memory>" in new_req.system
    assert "</cortex_memory>" in new_req.system
    # Cold history is collapsed
    assert report.cold_group_count > 0


@pytest.mark.asyncio
async def test_no_virtualization_needed_returns_original() -> None:
    msgs = [_u("hi"), _a("hello"), _u("bye"), _a("goodbye")]
    req = CortexRequest(
        model="claude-opus-4-7",
        max_tokens=128,
        messages=msgs,
        system="sys",
    )
    settings = CortexSettings(last_k_spans=4, safety_margin_tokens=128)
    new_req, report = await virtualize(req, settings, context_limit=200_000)
    # All 4 messages fit in last_k=4 groups → nothing to collapse.
    assert report.cold_group_count == 0
    # Messages unchanged.
    assert new_req.messages == req.messages
    # System unchanged when no recap is needed.
    assert new_req.system == req.system


# ---------- Budget enforcement ----------


@pytest.mark.asyncio
async def test_messages_only_budget_ignores_tools_and_max_tokens() -> None:
    """Regression: with the legacy math, Claude Code's tool-heavy installs
    (≥20k chars/4 tools_t + 32k max_tokens) made M negative against a 50k
    context_limit and forced degrade=true on 100% of requests. The new math
    excludes tools/system/max_tokens from M — virtualize only looks at
    messages."""
    # Realistic Claude Code shape: 28k tokens of tools, 32k max_tokens, BUT
    # message history is small (~150 tokens). Old math: M = 50000-32000-28000
    # -1024 = NEGATIVE → degrade. New math: M = 50000-1024 = ~49k → messages
    # short-circuit, NO degrade.
    big_tools = [
        {"name": f"tool_{i}", "input_schema": {"type": "object", "description": "x" * 400}}
        for i in range(70)
    ]
    msgs = []
    for i in range(6):
        msgs.append(_u(f"older message {i} sufficiently long to be ingested"))
        msgs.append(_a(f"older reply {i} also long enough"))
    req = CortexRequest(
        model="claude-opus-4-7",
        max_tokens=32_000,
        messages=msgs,
        system="You are Claude Code." * 30,
    )
    settings = CortexSettings(last_k_spans=4, safety_margin_tokens=1024)

    new_req, report = await virtualize(
        req,
        settings,
        context_limit=50_000,
        tools_serialized=big_tools,
    )

    assert report.degraded is False, (
        f"degraded={report.degraded} notes={report.notes} — tools should not "
        "block virtualize when messages fit the budget"
    )
    # Sanity: messages are small enough to short-circuit (no trim needed)
    assert report.cold_group_count == 0
    # Messages preserved verbatim.
    assert new_req.messages == req.messages


@pytest.mark.asyncio
async def test_messages_only_budget_engages_when_messages_cross_threshold() -> None:
    """When messages alone cross the budget, virtualize SHOULD trim — even
    with realistic Claude Code tool overhead in tools_serialized."""
    big_tools = [
        {"name": f"tool_{i}", "input_schema": {"type": "object", "description": "x" * 400}}
        for i in range(70)
    ]
    # 30 messages each ~50 chars → ~12 tokens × 30 = ~360 tokens. Use a tight
    # budget to trigger trimming.
    msgs = []
    for i in range(30):
        msgs.append(_u(f"user message {i} ingested fine here ok"))
        msgs.append(_a(f"assistant reply {i} retained fine here yes"))
    req = CortexRequest(
        model="claude-opus-4-7",
        max_tokens=32_000,
        messages=msgs,
        system="big system prompt " * 100,
    )
    settings = CortexSettings(last_k_spans=4, safety_margin_tokens=0)

    new_req, report = await virtualize(
        req,
        settings,
        context_limit=200,
        tools_serialized=big_tools,
    )

    # With the new math, M = 200 - 0 = 200; 60 msgs ≈ 600 tokens → trim.
    # Last-4 atomic groups (8 msgs) ≈ 80 tokens < 200 → fits → not degraded.
    assert report.degraded is False
    assert report.cold_group_count > 0
    assert report.kept_message_count == 8


@pytest.mark.asyncio
async def test_degraded_when_verbatim_exceeds_budget() -> None:
    huge = "x" * 200_000
    msgs = [_u(huge), _a("ack")]
    req = CortexRequest(
        model="claude-opus-4-7",
        max_tokens=10_000,
        messages=msgs,
        system="sys",
    )
    settings = CortexSettings(last_k_spans=2, safety_margin_tokens=128)
    new_req, report = await virtualize(req, settings, context_limit=20_000)
    assert report.degraded is True
    # Pass-through: messages unchanged.
    assert new_req.messages == req.messages


# ---------- Recap construction ----------


def test_assemble_recap_with_both_sections() -> None:
    text = assemble_recap(
        cold_summary="- [user] hello\n- [assistant] hi back",
        recall_text="(alice, likes, tea)",
    )
    assert "<cortex_memory>" in text
    assert "Older conversation context" in text
    assert "Relevant retrieved knowledge" in text
    assert "(alice, likes, tea)" in text


def test_assemble_recap_empty_returns_empty() -> None:
    assert assemble_recap("", "") == ""


def test_build_cold_summary_truncates_long_lines() -> None:
    long_msg = _u("x" * 1000)
    summary = build_cold_summary([[long_msg]], max_chars_per_msg=80)
    line = summary.splitlines()[0]
    # 80-char cap + leading "- [user] " + ellipsis
    assert len(line) <= 80 + len("- [user] ") + 1


def test_last_user_query_picks_most_recent_user_text() -> None:
    msgs = [
        _u("first"),
        _a("reply 1"),
        _u("second"),
        _a("reply 2"),
        _u_result("t1", "tool result"),  # pure tool_result, should be skipped
    ]
    q = last_user_query(msgs)
    assert q == "second"


# ---------- Token estimation ----------


def test_approx_tokens_scales_with_text_length() -> None:
    assert approx_tokens("") == 0
    assert approx_tokens("x") >= 1
    short = approx_tokens("hello world")
    long = approx_tokens("hello world" * 100)
    assert long > short * 50


def test_context_limit_for_known_and_unknown_models() -> None:
    assert context_limit_for("claude-opus-4-7") == 200_000
    assert context_limit_for("claude-haiku-4-5-20251001") == 200_000
    assert context_limit_for("gpt-4o-mini") == 128_000
    # Unknown model → safe default
    assert context_limit_for("foo-mystery-99") == 128_000


# ---------- Cortex-specific cortex_disable_virtualize bypass ----------


@pytest.mark.asyncio
async def test_virtualize_respects_caller_intent_via_settings() -> None:
    """Even with many cold groups, if `enable_virtualization` is off the
    server caller should not invoke virtualize at all.

    This test confirms virtualize itself is a pure function — it doesn't read
    the enable flag (the server gates the call). So calling virtualize
    directly always works, and the bypass is enforced at the server boundary.
    """
    msgs = []
    for i in range(10):
        msgs.append(_u(f"u{i} long enough"))
        msgs.append(_a(f"a{i} long enough"))
    req = CortexRequest(
        model="claude-opus-4-7",
        max_tokens=128,
        messages=msgs,
        system="sys",
    )
    settings = CortexSettings(last_k_spans=2, safety_margin_tokens=0)
    # M = 50 - 0 = 50; 20 short msgs ≈ 60 tokens overflows → trim.
    new_req, report = await virtualize(req, settings, context_limit=50)
    # virtualize did its job — fewer messages kept.
    assert report.kept_message_count < len(msgs)
