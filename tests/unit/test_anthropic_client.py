"""Unit tests for the timegraph anthropic_api backend.

Covers credential resolution (env var precedence, OAuth-from-credentials),
auth-header selection (Bearer vs x-api-key), and the forced-tool round-trip
that gives us structured output.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from timegraph.llm.anthropic_client import (
    AnthropicJsonClient,
    anthropic_credentials_available,
    resolve_anthropic_api_key,
    resolve_model,
)


# ---------- credential resolution ----------


def test_env_api_key_takes_precedence(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-FROM-ENV")
    monkeypatch.setenv("TG_ANTHROPIC_CREDENTIALS_PATH", str(tmp_path / "missing.json"))
    assert resolve_anthropic_api_key() == "sk-ant-api03-FROM-ENV"


def test_oauth_from_credentials_file(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    creds = tmp_path / ".credentials.json"
    creds.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "sk-ant-oat01-FROM-FILE",
                    "refreshToken": "sk-ant-ort01-IGNORED",
                    "expiresAt": 99999999999,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TG_ANTHROPIC_CREDENTIALS_PATH", str(creds))
    assert resolve_anthropic_api_key() == "sk-ant-oat01-FROM-FILE"


def test_oauth_top_level_access_token(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    creds = tmp_path / ".credentials.json"
    creds.write_text(
        json.dumps({"accessToken": "sk-ant-oat01-TOPLEVEL"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("TG_ANTHROPIC_CREDENTIALS_PATH", str(creds))
    assert resolve_anthropic_api_key() == "sk-ant-oat01-TOPLEVEL"


def test_resolve_returns_none_when_nothing_available(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("TG_ANTHROPIC_CREDENTIALS_PATH", str(tmp_path / "absent.json"))
    assert resolve_anthropic_api_key() is None
    assert anthropic_credentials_available() is False


def test_malformed_credentials_file_returns_none(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    creds = tmp_path / ".credentials.json"
    creds.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setenv("TG_ANTHROPIC_CREDENTIALS_PATH", str(creds))
    assert resolve_anthropic_api_key() is None


def test_credentials_file_with_non_oauth_token_is_skipped(monkeypatch, tmp_path) -> None:
    """An accessToken that doesn't start with sk-ant-oat (e.g. an old API key
    stored in the wrong slot) must be ignored — we only pull OAuth bearers
    from this file."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    creds = tmp_path / ".credentials.json"
    creds.write_text(
        json.dumps({"claudeAiOauth": {"accessToken": "not-a-real-prefix"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("TG_ANTHROPIC_CREDENTIALS_PATH", str(creds))
    assert resolve_anthropic_api_key() is None


# ---------- model alias resolution ----------


def test_model_aliases() -> None:
    assert resolve_model("haiku") == "claude-haiku-4-5"
    assert resolve_model("sonnet") == "claude-sonnet-4-6"
    assert resolve_model("opus") == "claude-opus-4-7"
    # Exact model ids pass through.
    assert resolve_model("claude-haiku-4-5") == "claude-haiku-4-5"
    assert resolve_model("custom-fine-tune") == "custom-fine-tune"


# ---------- forced-tool round-trip ----------


def _make_response(tool_input: dict) -> bytes:
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
                    "name": "record_facts",
                    "input": tool_input,
                }
            ],
            "usage": {"input_tokens": 100, "output_tokens": 30},
        }
    ).encode("utf-8")


@pytest.mark.asyncio
async def test_call_returns_tool_input_dict() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            status_code=200,
            content=_make_response({"facts": [{"subject": "x", "predicate": "y", "object": "z", "confidence": 0.9}]}),
            headers={"content-type": "application/json"},
        )

    client = AnthropicJsonClient(base_url="https://api.anthropic.com")
    client._http = httpx.AsyncClient(
        base_url="https://api.anthropic.com",
        transport=httpx.MockTransport(handler),
    )
    try:
        out = await client.call(
            model="haiku",
            prompt="extract facts",
            schema={"type": "object"},
            schema_name="record_facts",
            schema_description="record",
            api_key="sk-ant-api03-TEST",
        )
    finally:
        await client.close()

    assert out["facts"][0]["subject"] == "x"
    body = captured["body"]
    assert body["model"] == "claude-haiku-4-5"
    assert body["tool_choice"] == {"type": "tool", "name": "record_facts"}
    assert body["tools"][0]["name"] == "record_facts"
    # API-key auth uses x-api-key, NOT Authorization bearer.
    assert captured["headers"].get("x-api-key") == "sk-ant-api03-TEST"
    assert "authorization" not in captured["headers"]


@pytest.mark.asyncio
async def test_call_uses_oauth_bearer_and_beta_flag() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        for k, v in request.headers.items():
            captured[k.lower()] = v
        return httpx.Response(
            status_code=200,
            content=_make_response({"facts": []}),
            headers={"content-type": "application/json"},
        )

    client = AnthropicJsonClient(base_url="https://api.anthropic.com")
    client._http = httpx.AsyncClient(
        base_url="https://api.anthropic.com",
        transport=httpx.MockTransport(handler),
    )
    try:
        await client.call(
            model="haiku",
            prompt="ok",
            schema={"type": "object"},
            schema_name="record_facts",
            schema_description="record",
            api_key="sk-ant-oat01-EXAMPLE",
        )
    finally:
        await client.close()

    assert captured["authorization"] == "Bearer sk-ant-oat01-EXAMPLE"
    assert "x-api-key" not in captured
    assert captured["anthropic-beta"] == "oauth-2025-04-20"


@pytest.mark.asyncio
async def test_call_raises_when_upstream_returns_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=429,
            content=b'{"type":"error","error":{"type":"rate_limit","message":"slow"}}',
            headers={"content-type": "application/json"},
        )

    client = AnthropicJsonClient(base_url="https://api.anthropic.com")
    client._http = httpx.AsyncClient(
        base_url="https://api.anthropic.com",
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ValueError, match="429"):
            await client.call(
                model="haiku",
                prompt="ok",
                schema={"type": "object"},
                schema_name="x",
                schema_description="y",
                api_key="sk-ant-api03-T",
            )
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_call_raises_when_no_tool_use_block() -> None:
    """The model went on strike and emitted plain text instead of the forced
    tool call — must surface as ValueError so the caller's tenacity retries."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.dumps(
            {
                "content": [{"type": "text", "text": "I refuse"}],
                "stop_reason": "end_turn",
            }
        ).encode("utf-8")
        return httpx.Response(
            status_code=200, content=body, headers={"content-type": "application/json"}
        )

    client = AnthropicJsonClient(base_url="https://api.anthropic.com")
    client._http = httpx.AsyncClient(
        base_url="https://api.anthropic.com",
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ValueError, match="no record_facts tool_use"):
            await client.call(
                model="haiku",
                prompt="ok",
                schema={"type": "object"},
                schema_name="record_facts",
                schema_description="record",
                api_key="sk-ant-api03-T",
            )
    finally:
        await client.close()
