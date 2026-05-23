"""Unit tests for the Claude Code plugin hooks (state, ingest, tool_use, session_start).

These tests exercise the pure-logic helpers (parsing, classification, formatting)
without touching Neo4j / Qdrant / LM Studio. The end-to-end `main()` paths are
covered by smoke scripts that require backends.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from timegraph.hooks import ingest as ingest_hook
from timegraph.hooks import recall as recall_hook
from timegraph.hooks import session_start as session_start_hook
from timegraph.hooks import state as state_mod
from timegraph.hooks import tool_use as tool_use_hook


# -------- state.py ----------------------------------------------------


def test_state_read_empty_returns_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("TG_HOOK_STATE_DIR", str(tmp_path))
    assert state_mod.read_offset("session-a") == 0


def test_state_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("TG_HOOK_STATE_DIR", str(tmp_path))
    state_mod.write_offset("session-b", 1234)
    assert state_mod.read_offset("session-b") == 1234

    state_mod.write_offset("session-b", 5678)
    assert state_mod.read_offset("session-b") == 5678


def test_state_isolation_between_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("TG_HOOK_STATE_DIR", str(tmp_path))
    state_mod.write_offset("session-c", 100)
    state_mod.write_offset("session-d", 200)
    assert state_mod.read_offset("session-c") == 100
    assert state_mod.read_offset("session-d") == 200


def test_state_unsafe_session_id_sanitized(tmp_path, monkeypatch):
    monkeypatch.setenv("TG_HOOK_STATE_DIR", str(tmp_path))
    state_mod.write_offset("../etc/passwd", 42)
    # The unsafe path components should be stripped — no file outside tmp_path.
    files = list(tmp_path.iterdir())
    assert all(str(tmp_path) in str(f) for f in files)


# -------- ingest.py (transcript walker) -------------------------------


def test_is_command_echo_detects_meta_blocks():
    assert ingest_hook._is_command_echo("<command-name>/foo</command-name>")
    assert ingest_hook._is_command_echo("<local-command-stdout>x</local-command-stdout>")
    assert ingest_hook._is_command_echo("<local-command-caveat>...</local-command-caveat>")
    assert ingest_hook._is_command_echo("<system-reminder>x</system-reminder>")
    assert ingest_hook._is_command_echo("[Request interrupted by user]")
    assert not ingest_hook._is_command_echo("Hello, please refactor auth.py")


def test_extract_text_blocks_string_passthrough():
    assert ingest_hook._extract_text_blocks("plain string") == "plain string"


def test_extract_text_blocks_drops_tool_use_and_result():
    content = [
        {"type": "text", "text": "Reading the file."},
        {"type": "tool_use", "name": "Read", "input": {"file_path": "x.py"}},
        {"type": "text", "text": "Found it."},
    ]
    assert ingest_hook._extract_text_blocks(content) == "Reading the file.\nFound it."


def test_iter_ingestables_filters_meta_and_echoes():
    lines = [
        json.dumps({
            "type": "user", "isMeta": True,
            "message": {"role": "user", "content": "meta prompt"},
            "timestamp": "2026-05-23T10:00:00Z",
        }),
        json.dumps({
            "type": "user",
            "message": {"role": "user", "content": "<command-name>/foo</command-name>"},
            "timestamp": "2026-05-23T10:00:01Z",
        }),
        json.dumps({
            "type": "user",
            "message": {"role": "user", "content": "Please refactor auth.py"},
            "timestamp": "2026-05-23T10:00:02Z",
        }),
        json.dumps({
            "type": "assistant",
            "message": {"role": "assistant", "content": [
                {"type": "text", "text": "Sure, let me read it."},
                {"type": "tool_use", "name": "Read", "input": {}},
            ]},
            "timestamp": "2026-05-23T10:00:03Z",
        }),
        json.dumps({
            "type": "system", "content": "system event",
            "timestamp": "2026-05-23T10:00:04Z",
        }),
    ]
    ingestables = list(ingest_hook._iter_ingestables(lines))
    roles = [r for r, _, _ in ingestables]
    texts = [t for _, t, _ in ingestables]
    assert roles == ["user", "assistant"]
    assert texts == ["Please refactor auth.py", "Sure, let me read it."]


def test_iter_ingestables_skips_sidechain():
    lines = [
        json.dumps({
            "type": "user", "isSidechain": True,
            "message": {"role": "user", "content": "subagent prompt"},
            "timestamp": "2026-05-23T10:00:00Z",
        }),
    ]
    assert list(ingest_hook._iter_ingestables(lines)) == []


def test_iter_ingestables_skips_malformed_json():
    lines = ["{not valid json", "", json.dumps({
        "type": "user",
        "message": {"role": "user", "content": "real msg"},
        "timestamp": "2026-05-23T10:00:00Z",
    })]
    out = list(ingest_hook._iter_ingestables(lines))
    assert len(out) == 1
    assert out[0][1] == "real msg"


def test_read_new_lines_resumes_from_offset(tmp_path):
    p = tmp_path / "transcript.jsonl"
    # Write bytes to avoid platform-specific line-ending translation.
    p.write_bytes(b"line1\nline2\nline3\n")
    lines, offsets = ingest_hook._read_new_lines(p, 0)
    assert [ln.rstrip("\r\n") for ln in lines] == ["line1", "line2", "line3"]
    assert offsets[-1] == p.stat().st_size

    # Resume from after line1
    lines2, _ = ingest_hook._read_new_lines(p, offsets[0])
    assert [ln.rstrip("\r\n") for ln in lines2] == ["line2", "line3"]


# -------- tool_use.py (PostToolUse classification) --------------------


def test_should_skip_filters_mcp_and_admin_tools():
    assert tool_use_hook._should_skip("mcp__timegraph__recall")
    assert tool_use_hook._should_skip("TodoWrite")
    assert tool_use_hook._should_skip("ExitPlanMode")
    assert tool_use_hook._should_skip("")
    assert not tool_use_hook._should_skip("Read")
    assert not tool_use_hook._should_skip("Bash")


def test_is_errored_detects_error_shapes():
    assert tool_use_hook._is_errored({"tool_response": {"is_error": True}})
    assert tool_use_hook._is_errored({"tool_response": {"error": "boom"}})
    assert tool_use_hook._is_errored({"tool_response": {"interrupted": True}})
    assert not tool_use_hook._is_errored({"tool_response": {"stdout": "ok"}})
    assert not tool_use_hook._is_errored({})


def test_classify_read_uses_file_source():
    source, header = tool_use_hook._classify("Read", {"file_path": "/x/auth.py"})
    assert source == "file:/x/auth.py"
    assert "Read" in header and "/x/auth.py" in header


def test_classify_edit_uses_file_source():
    source, _ = tool_use_hook._classify("Edit", {"file_path": "/x/a.py"})
    assert source == "file:/x/a.py"


def test_classify_bash_includes_command_hash_and_body():
    source, header = tool_use_hook._classify("Bash", {"command": "ls -la", "description": "list"})
    assert source.startswith("bash:")
    assert "list" in header
    assert "$ ls -la" in header


def test_classify_grep_search_source():
    source, header = tool_use_hook._classify("Grep", {"pattern": "TODO", "path": "src/"})
    assert source == "search:Grep"
    assert "TODO" in header


def test_classify_unknown_tool_falls_through():
    source, header = tool_use_hook._classify("MysteryTool", {})
    assert source == "tool:MysteryTool"
    assert "MysteryTool" in header


def test_stringify_result_handles_shapes():
    assert tool_use_hook._stringify_result("plain") == "plain"
    assert "out" in tool_use_hook._stringify_result({"stdout": "out"})
    assert tool_use_hook._stringify_result(None) == ""
    nested = tool_use_hook._stringify_result([{"type": "text", "text": "hi"}, {"type": "text", "text": "there"}])
    assert "hi" in nested and "there" in nested


def test_truncate_appends_marker_when_over_limit():
    s = "x" * 100
    out = tool_use_hook._truncate(s, limit=20)
    assert out.startswith("x" * 20)
    assert "truncated" in out


# -------- session_start.py --------------------------------------------


def test_query_for_cwd_uses_basename():
    q = session_start_hook._query_for_cwd("/a/b/myproj")
    assert "myproj" in q


def test_query_for_cwd_none_fallback():
    q = session_start_hook._query_for_cwd(None)
    assert isinstance(q, str) and q


def test_format_facts_drops_partial_rows():
    facts = [
        {"subject": "A", "predicate": "p", "object": "B"},
        {"subject": "", "predicate": "p", "object": "B"},  # dropped
        {"subject": "C", "predicate": "p2", "object": "D", "valid_at": "2026-05-23T10:00:00"},
    ]
    out = session_start_hook._format_facts(facts, "Heading")
    assert out is not None
    assert "Heading" in out
    assert "**A**" in out and "**B**" in out
    assert "**C**" in out and "**D**" in out
    assert "2026-05-23" in out


def test_format_facts_returns_none_when_no_valid_rows():
    facts = [{"subject": "", "predicate": "", "object": ""}]
    assert session_start_hook._format_facts(facts, "h") is None


# -------- recall.py (UserPromptSubmit episode/fact formatting) -------


def test_recall_format_episodes_renders_code_block():
    eps = [{
        "source": "file:/x/auth.py",
        "content": "# Read /x/auth.py\n\nclass Auth: pass",
        "event_time": "2026-05-23T10:00:00",
        "session_id": "abcdef1234567890",
    }]
    blocks = recall_hook._format_episodes(eps, budget_tokens=1000)
    assert len(blocks) == 1
    assert "file:/x/auth.py" in blocks[0]
    assert "```" in blocks[0]
    assert "class Auth" in blocks[0]
    assert "2026-05-23" in blocks[0]


def test_recall_format_episodes_truncates_long_content():
    big = "x" * (recall_hook.EPISODE_SNIPPET_CHARS + 500)
    eps = [{"source": "file:/big.txt", "content": big, "event_time": "", "session_id": "s"}]
    blocks = recall_hook._format_episodes(eps, budget_tokens=10_000)
    assert "truncated" in blocks[0]


def test_recall_format_episodes_respects_token_budget():
    # 5 episodes each ~2000 chars; budget_tokens=500 ≈ 2000 chars total
    # should drop early before all 5 are appended.
    eps = [
        {"source": f"file:/f{i}.py", "content": "y" * 1500, "event_time": "", "session_id": "s"}
        for i in range(5)
    ]
    blocks = recall_hook._format_episodes(eps, budget_tokens=500)
    assert 1 <= len(blocks) < 5


def test_recall_compose_returns_none_when_both_empty():
    assert recall_hook._compose([], []) is None


def test_recall_compose_facts_only():
    out = recall_hook._compose(["- **A** p **B**"], [])
    assert out and "Relevant memory" in out
    assert "Recalled tool results" not in out


def test_recall_compose_both_sections():
    out = recall_hook._compose(["- **A** p **B**"], ["### file:x\n```\ncontent\n```"])
    assert out and "Relevant memory" in out and "Recalled tool results" in out


# -------- end-to-end main() smoke (mocked ingest) --------------------


def test_ingest_main_advances_offset_after_ingest(tmp_path, monkeypatch, capsys):
    """End-to-end Stop hook: feed a fake transcript + payload, mock add_episode,
    verify the offset advances past every ingested line."""
    monkeypatch.setenv("TG_HOOK_STATE_DIR", str(tmp_path))

    transcript = tmp_path / "transcript.jsonl"
    lines = [
        json.dumps({
            "type": "user",
            "message": {"role": "user", "content": "first prompt about authentication"},
            "timestamp": "2026-05-23T10:00:00Z",
        }),
        json.dumps({
            "type": "assistant",
            "message": {"role": "assistant", "content": [
                {"type": "text", "text": "Sure, let me investigate the auth module."},
            ]},
            "timestamp": "2026-05-23T10:00:01Z",
        }),
    ]
    transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ingested: list[tuple[str, str]] = []

    async def fake_ingest(text, role, session_id, group_id, event_time):
        ingested.append((role, text))

    monkeypatch.setattr(ingest_hook, "_ingest_one", fake_ingest)

    payload = {
        "transcript_path": str(transcript),
        "session_id": "test-session",
        "cwd": str(tmp_path),
    }
    monkeypatch.setattr("sys.stdin", _StringIO(json.dumps(payload)))
    ingest_hook.main()

    assert len(ingested) == 2
    assert ingested[0][0] == "user"
    assert ingested[1][0] == "assistant"
    assert state_mod.read_offset("test-session") == transcript.stat().st_size


def test_ingest_main_does_not_reingest_on_second_fire(tmp_path, monkeypatch):
    """After a successful fire, a second fire with the same transcript should
    ingest zero new items."""
    monkeypatch.setenv("TG_HOOK_STATE_DIR", str(tmp_path))

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps({
            "type": "user",
            "message": {"role": "user", "content": "the only message in here so far"},
            "timestamp": "2026-05-23T10:00:00Z",
        }) + "\n",
        encoding="utf-8",
    )

    ingested: list[tuple[str, str]] = []

    async def fake_ingest(text, role, session_id, group_id, event_time):
        ingested.append((role, text))

    monkeypatch.setattr(ingest_hook, "_ingest_one", fake_ingest)
    payload = {"transcript_path": str(transcript), "session_id": "sid", "cwd": str(tmp_path)}

    monkeypatch.setattr("sys.stdin", _StringIO(json.dumps(payload)))
    ingest_hook.main()
    assert len(ingested) == 1

    # Second fire — same transcript, nothing new.
    monkeypatch.setattr("sys.stdin", _StringIO(json.dumps(payload)))
    ingest_hook.main()
    assert len(ingested) == 1  # no new ingest


class _StringIO:
    def __init__(self, s: str):
        self._s = s

    def read(self) -> str:
        return self._s
