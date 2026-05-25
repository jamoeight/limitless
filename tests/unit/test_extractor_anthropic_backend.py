"""End-to-end test: ExtractorClient with backend=anthropic_api hits a mocked
api.anthropic.com and yields a valid list[Fact].
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

from timegraph.llm.anthropic_client import AnthropicJsonClient
from timegraph.llm.extractor import ExtractorClient


def _make_anthropic_response(facts: list[dict]) -> bytes:
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
                    "name": "record_extracted_facts",
                    "input": {"thinking": "", "facts": facts},
                }
            ],
            "usage": {"input_tokens": 50, "output_tokens": 30},
        }
    ).encode("utf-8")


@pytest.mark.asyncio
async def test_extractor_anthropic_api_returns_facts(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-TEST")
    monkeypatch.setenv("TG_EXTRACTOR_BACKEND", "anthropic_api")
    monkeypatch.setenv("TG_EXTRACTOR_ANTHROPIC_MODEL", "haiku")

    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            status_code=200,
            content=_make_anthropic_response(
                [
                    {
                        "subject": "Alice",
                        "predicate": "lives_in",
                        "object": "Paris",
                        "confidence": 0.95,
                    }
                ]
            ),
            headers={"content-type": "application/json"},
        )

    extractor = ExtractorClient()
    extractor._anthropic = AnthropicJsonClient(
        base_url=extractor.s.anthropic_api_base_url,
        anthropic_version=extractor.s.anthropic_api_version,
        timeout_s=extractor.s.extractor_anthropic_timeout_s,
    )
    extractor._anthropic._http = httpx.AsyncClient(
        base_url=extractor.s.anthropic_api_base_url,
        transport=httpx.MockTransport(handler),
    )

    try:
        facts, latency_ms = await extractor.extract_facts(
            episode_content="Alice lives in Paris.",
            event_time=datetime(2026, 5, 25, tzinfo=timezone.utc),
            session_id="s1",
            source="msg:user",
        )
    finally:
        await extractor.close()

    assert request_count == 1
    assert len(facts) == 1
    assert facts[0].subject == "Alice"
    assert facts[0].predicate == "lives_in"
    assert facts[0].object == "Paris"
    assert facts[0].confidence == pytest.approx(0.95)
    assert latency_ms > 0
