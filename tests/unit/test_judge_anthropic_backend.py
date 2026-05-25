"""End-to-end test: JudgeClient with backend=anthropic_api hits a mocked
api.anthropic.com and yields a valid JudgeOutput.

Covers the load-bearing `judge_call_count == 1` invariant on the new path
(we count HTTP calls, not just the field, to be sure).
"""

from __future__ import annotations

import httpx
import pytest

from timegraph.llm.anthropic_client import AnthropicJsonClient
from timegraph.llm.judge import JudgeClient
from timegraph.types import ConflictTriple, Resolution


def _make_anthropic_response(payload: dict) -> bytes:
    import json

    return json.dumps(
        {
            "id": "msg_01TEST",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01TEST",
                    "name": "record_conflict_resolution",
                    "input": payload,
                }
            ],
            "usage": {"input_tokens": 50, "output_tokens": 30},
        }
    ).encode("utf-8")


def _conflict() -> ConflictTriple:
    return ConflictTriple(
        e1_fact_id="f_alice_paris",
        e2_fact_id="f_alice_berlin",
        reason="same_subject_different_value",
    )


@pytest.mark.asyncio
async def test_judge_anthropic_api_one_call_one_response(monkeypatch) -> None:
    """The judge must do exactly one HTTP round-trip per invocation —
    propagating `judge_call_count <= 1` to the new backend."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-TEST")
    monkeypatch.setenv("TG_JUDGE_BACKEND", "anthropic_api")
    monkeypatch.setenv("TG_JUDGE_ANTHROPIC_MODEL", "haiku")

    # The Settings cache: there's no @lru_cache on get_settings, it returns a
    # fresh Settings() each call. So env mutation here is picked up.
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            status_code=200,
            content=_make_anthropic_response(
                {
                    "thinking": "Recency favors Berlin.",
                    "resolution": "e2_correct",
                    "reason": "Most recent statement names Berlin.",
                    "confidence": 0.85,
                }
            ),
            headers={"content-type": "application/json"},
        )

    judge = JudgeClient()
    # Pre-build the client and swap its transport so we don't hit the real API.
    judge._anthropic = AnthropicJsonClient(
        base_url=judge.s.anthropic_api_base_url,
        anthropic_version=judge.s.anthropic_api_version,
        timeout_s=judge.s.judge_anthropic_timeout_s,
    )
    judge._anthropic._http = httpx.AsyncClient(
        base_url=judge.s.anthropic_api_base_url,
        transport=httpx.MockTransport(handler),
    )

    try:
        out = await judge.judge_conflicts(
            query="where does Alice live?",
            conflicts=[_conflict()],
        )
    finally:
        await judge.close()

    assert request_count == 1, "judge_call_count <= 1 invariant violated"
    assert out.resolution == Resolution.E2_CORRECT
    assert out.confidence == pytest.approx(0.85)
    assert "Recency favors Berlin" in out.thinking
    assert out.call_count == 1


@pytest.mark.asyncio
async def test_judge_anthropic_api_rejects_too_many_conflicts(monkeypatch) -> None:
    """Stage-1 must truncate to <=8 conflicts — judge enforces the contract.

    Tenacity wraps the ValueError in a RetryError; we inspect the chain to
    confirm the bounded-conflict check fired.
    """
    from tenacity import RetryError

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-TEST")
    monkeypatch.setenv("TG_JUDGE_BACKEND", "anthropic_api")

    judge = JudgeClient()
    try:
        with pytest.raises((ValueError, RetryError)) as exc_info:
            await judge.judge_conflicts(
                query="x",
                conflicts=[_conflict() for _ in range(9)],
            )
    finally:
        await judge.close()

    err = exc_info.value
    if isinstance(err, RetryError):
        inner = err.last_attempt.exception()
        assert isinstance(inner, ValueError)
        assert "≤8" in str(inner)
    else:
        assert "≤8" in str(err)
