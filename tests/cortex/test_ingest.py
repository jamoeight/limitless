"""Unit tests for cortex.ingest.

Pure logic tests — no Neo4j, Qdrant, or LM Studio. We pass a stub `ingest_fn`
to SessionRegistry so the call is captured in-memory instead of hitting the
real timegraph backend.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime

import pytest

from cortex.canonical import (
    ChunkContentBlockStart,
    ChunkContentBlockStop,
    ChunkMessageStart,
    ChunkMessageStop,
    ChunkTextDelta,
    ChunkToolUseDelta,
    CortexMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from cortex.config import CortexSettings
from cortex.ingest import (
    IngestState,
    Session,
    SessionRegistry,
    assistant_message_from_chunks,
    derive_group_id,
    ingest_request_messages,
    looks_like_secret,
    message_to_text,
    span_hash,
)


def _text(role: str, content: str) -> CortexMessage:
    return CortexMessage(role=role, content=[TextBlock(text=content)])


def _settings(**overrides) -> CortexSettings:
    s = CortexSettings()
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _recorder():
    """Return (ingest_fn, calls list) — calls captures every ingest invocation."""
    calls: list[dict] = []

    async def fn(content, source, group_id, session_id, event_time):
        calls.append(
            {
                "content": content,
                "source": source,
                "group_id": group_id,
                "session_id": session_id,
                "event_time": event_time,
            }
        )
        return f"ep_{len(calls):04d}"

    return fn, calls


# ---------- span_hash ----------


def test_span_hash_is_deterministic_for_equivalent_messages() -> None:
    a = _text("user", "Hello there")
    b = _text("user", "Hello there")
    assert span_hash(a) == span_hash(b)


def test_span_hash_differs_for_different_content() -> None:
    assert span_hash(_text("user", "a")) != span_hash(_text("user", "b"))


def test_span_hash_differs_by_role() -> None:
    a = _text("user", "ok")
    b = _text("assistant", "ok")
    assert span_hash(a) != span_hash(b)


def test_span_hash_includes_tool_use_id() -> None:
    a = CortexMessage(
        role="assistant",
        content=[ToolUseBlock(tool_use_id="t1", tool_name="search", tool_input={"q": "x"})],
    )
    b = CortexMessage(
        role="assistant",
        content=[ToolUseBlock(tool_use_id="t2", tool_name="search", tool_input={"q": "x"})],
    )
    assert span_hash(a) != span_hash(b)


def test_span_hash_stable_across_dict_key_orderings() -> None:
    a = CortexMessage(
        role="assistant",
        content=[
            ToolUseBlock(
                tool_use_id="t1",
                tool_name="search",
                tool_input={"a": 1, "b": 2, "c": 3},
            )
        ],
    )
    b = CortexMessage(
        role="assistant",
        content=[
            ToolUseBlock(
                tool_use_id="t1",
                tool_name="search",
                tool_input={"c": 3, "b": 2, "a": 1},
            )
        ],
    )
    assert span_hash(a) == span_hash(b)


# ---------- message_to_text ----------


def test_message_to_text_concatenates_blocks() -> None:
    m = CortexMessage(
        role="assistant",
        content=[
            TextBlock(text="thinking..."),
            ToolUseBlock(tool_use_id="t1", tool_name="search", tool_input={"q": "cats"}),
        ],
    )
    out = message_to_text(m)
    assert "thinking..." in out
    assert "tool_use search" in out
    assert "cats" in out


def test_message_to_text_handles_tool_result_string() -> None:
    m = CortexMessage(
        role="user",
        content=[ToolResultBlock(tool_use_id="t1", content="result text", is_error=False)],
    )
    assert "result text" in message_to_text(m)


def test_message_to_text_handles_tool_result_error() -> None:
    m = CortexMessage(
        role="user",
        content=[ToolResultBlock(tool_use_id="t1", content="boom", is_error=True)],
    )
    assert "(error)" in message_to_text(m)


# ---------- secrets filter ----------


@pytest.mark.parametrize(
    "needle",
    [
        "-----BEGIN PRIVATE KEY-----",
        "-----BEGIN OPENSSH PRIVATE KEY-----",
        "AKIAABCDEFGHIJKLMNOP",
        "ghp_" + "a" * 36,
    ],
)
def test_secrets_filter_flags_obvious_secrets(needle: str) -> None:
    assert looks_like_secret(f"my secret is {needle} please ignore")


def test_secrets_filter_passes_normal_text() -> None:
    assert not looks_like_secret("this is a normal sentence about Python")


# ---------- Session.schedule ----------


@pytest.mark.asyncio
async def test_session_schedules_new_message_and_records_it() -> None:
    fn, calls = _recorder()
    reg = SessionRegistry(_settings(), ingest_fn=fn)
    sess = reg.get_or_create("grp", "ses")

    h = sess.schedule(_text("user", "a sufficiently long message to clear ingest_min_chars"))
    assert h is not None
    await sess.drain(timeout=2.0)

    assert len(calls) == 1
    assert "sufficiently long message" in calls[0]["content"]
    assert calls[0]["group_id"] == "grp"
    assert calls[0]["session_id"] == "ses"
    assert sess.state(h) == IngestState.DONE
    assert sess.episode_id_for(h) == "ep_0001"


@pytest.mark.asyncio
async def test_session_idempotency_same_message_ingested_once() -> None:
    fn, calls = _recorder()
    reg = SessionRegistry(_settings(), ingest_fn=fn)
    sess = reg.get_or_create("grp", "ses")

    msg = _text("user", "this is a sufficiently long message please ingest me")
    h1 = sess.schedule(msg)
    h2 = sess.schedule(msg)

    assert h1 is not None
    assert h2 is None  # second call recognized as duplicate

    await sess.drain(timeout=2.0)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_session_skips_tiny_messages() -> None:
    fn, calls = _recorder()
    reg = SessionRegistry(_settings(ingest_min_chars=20), ingest_fn=fn)
    sess = reg.get_or_create("grp", "ses")

    h = sess.schedule(_text("user", "hi"))
    assert h is None
    await sess.drain(timeout=2.0)
    assert calls == []


@pytest.mark.asyncio
async def test_session_skips_secret_messages() -> None:
    fn, calls = _recorder()
    reg = SessionRegistry(_settings(), ingest_fn=fn)
    sess = reg.get_or_create("grp", "ses")

    h = sess.schedule(
        _text(
            "user",
            "please use -----BEGIN PRIVATE KEY----- mykey -----END PRIVATE KEY-----",
        )
    )
    assert h is None
    await sess.drain(timeout=2.0)
    assert calls == []


@pytest.mark.asyncio
async def test_session_multiple_distinct_messages_concurrent() -> None:
    fn, calls = _recorder()
    reg = SessionRegistry(_settings(), ingest_fn=fn)
    sess = reg.get_or_create("grp", "ses")

    scheduled = ingest_request_messages(
        sess,
        [
            _text("user", "first message that is long enough"),
            _text("assistant", "second message that is also long enough"),
            _text("user", "third one with enough characters"),
        ],
    )
    assert len(scheduled) == 3

    await sess.drain(timeout=2.0)
    assert len(calls) == 3
    contents = sorted(c["content"] for c in calls)
    assert contents[0].startswith("first message")
    assert contents[1].startswith("second message")
    assert contents[2].startswith("third one")


@pytest.mark.asyncio
async def test_session_handles_ingest_failure_gracefully() -> None:
    async def failing_fn(content, source, group_id, session_id, event_time):
        raise RuntimeError("backend exploded")

    reg = SessionRegistry(_settings(), ingest_fn=failing_fn)
    sess = reg.get_or_create("grp", "ses")
    h = sess.schedule(_text("user", "a sufficiently long message to ingest"))
    assert h is not None
    await sess.drain(timeout=2.0)
    assert sess.state(h) == IngestState.FAILED
    assert sess.episode_id_for(h) is None


@pytest.mark.asyncio
async def test_session_isolation_by_group_id() -> None:
    fn, calls = _recorder()
    reg = SessionRegistry(_settings(), ingest_fn=fn)
    a = reg.get_or_create("group-a", "shared-session")
    b = reg.get_or_create("group-b", "shared-session")

    msg = _text("user", "a sufficiently long shared text content here")
    ha = a.schedule(msg)
    hb = b.schedule(msg)
    # Same content hash, but different sessions → both ingest (separate caches).
    assert ha is not None
    assert hb is not None
    await asyncio.gather(a.drain(timeout=2.0), b.drain(timeout=2.0))
    assert len(calls) == 2
    seen_groups = sorted(c["group_id"] for c in calls)
    assert seen_groups == ["group-a", "group-b"]


# ---------- derive_group_id ----------


def test_derive_group_id_is_stable_and_redacts_key() -> None:
    g1 = derive_group_id("sk-ant-secret-12345")
    g2 = derive_group_id("sk-ant-secret-12345")
    g3 = derive_group_id("sk-ant-different-67890")
    assert g1 == g2
    assert g1 != g3
    assert "sk-ant" not in g1
    assert g1.startswith("k:")


# ---------- assistant_message_from_chunks ----------


def test_assistant_message_from_chunks_text_only() -> None:
    chunks = [
        ChunkMessageStart(message_id="m1", model="x", input_tokens=1),
        ChunkContentBlockStart(index=0, block=TextBlock(text="")),
        ChunkTextDelta(index=0, text="Hello "),
        ChunkTextDelta(index=0, text="world"),
        ChunkContentBlockStop(index=0),
        ChunkMessageStop(),
    ]
    msg = assistant_message_from_chunks(chunks)
    assert msg is not None
    assert msg.role == "assistant"
    assert len(msg.content) == 1
    assert isinstance(msg.content[0], TextBlock)
    assert msg.content[0].text == "Hello world"


def test_assistant_message_from_chunks_text_then_tool_use() -> None:
    chunks = [
        ChunkMessageStart(message_id="m1", model="x", input_tokens=1),
        ChunkContentBlockStart(index=0, block=TextBlock(text="")),
        ChunkTextDelta(index=0, text="searching"),
        ChunkContentBlockStop(index=0),
        ChunkContentBlockStart(
            index=1,
            block=ToolUseBlock(tool_use_id="t1", tool_name="search", tool_input={}),
        ),
        ChunkToolUseDelta(index=1, partial_input_json='{"q":"'),
        ChunkToolUseDelta(index=1, partial_input_json='cats"}'),
        ChunkContentBlockStop(index=1),
        ChunkMessageStop(),
    ]
    msg = assistant_message_from_chunks(chunks)
    assert msg is not None
    assert len(msg.content) == 2
    assert isinstance(msg.content[1], ToolUseBlock)
    assert msg.content[1].tool_input == {"q": "cats"}


def test_assistant_message_from_chunks_empty_returns_none() -> None:
    assert assistant_message_from_chunks([]) is None
    assert assistant_message_from_chunks([ChunkMessageStop()]) is None
