"""Direct Anthropic Messages-API client for the judge + extractor call sites.

This is the third backend (alongside `lm_studio` and `claude_cli`). It POSTs
straight to `api.anthropic.com/v1/messages` and uses the forced-tool pattern
to enforce a JSON schema on the model's output — Anthropic doesn't expose
OpenAI-style `response_format: json_schema`, so we declare the target schema
as a tool's `input_schema` and pin the model to that tool via
`tool_choice: {"type": "tool", "name": ...}`. The model's only legal response
is one `tool_use` block whose `input` matches the schema.

Auth mirrors `cortex.providers.anthropic`:
  - `sk-ant-oat...` (OAuth bearer, written by `claude login` to
    `~/.claude/.credentials.json`) → `Authorization: Bearer ...` plus the
    `anthropic-beta: oauth-2025-04-20` opt-in flag.
  - `sk-ant-api...` (classic API key from `ANTHROPIC_API_KEY` env var) →
    `x-api-key`.

The credentials file is the same one Claude Code itself uses. We read it on
each call (no caching) so a `claude login` refresh is picked up without
restarting the host process. If neither source yields a token the caller
should fall back to LM Studio.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import structlog

log = structlog.get_logger(__name__)

_OAUTH_TOKEN_PREFIX = "sk-ant-oat"
_OAUTH_BETA_FLAG = "oauth-2025-04-20"


def _default_credentials_path() -> Path:
    override = os.environ.get("TG_ANTHROPIC_CREDENTIALS_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".claude" / ".credentials.json"


def _read_oauth_token(path: Path) -> str | None:
    """Best-effort extraction of an OAuth access token from `~/.claude/.credentials.json`.

    Tries known schemas: top-level `claudeAiOauth.accessToken`, top-level
    `oauth.accessToken`, top-level `accessToken`. Returns None on any error
    (missing file, malformed JSON, no token field, expired) — the caller
    falls back to env-var auth or LM Studio.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        blob = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(blob, dict):
        return None
    candidates = [
        blob.get("claudeAiOauth", {}),
        blob.get("oauth", {}),
        blob,
    ]
    for c in candidates:
        if not isinstance(c, dict):
            continue
        token = c.get("accessToken") or c.get("access_token")
        if isinstance(token, str) and token.startswith(_OAUTH_TOKEN_PREFIX):
            return token
    return None


def resolve_anthropic_api_key(credentials_path: Path | None = None) -> str | None:
    """Return an API key for api.anthropic.com, or None if no auth is available.

    Resolution order:
      1. `ANTHROPIC_API_KEY` env var (classic key OR OAuth bearer; either works).
      2. OAuth bearer from `~/.claude/.credentials.json`.

    The `is_oauth` shape is decided by the key prefix at request time — callers
    don't need to track which source it came from.
    """
    env_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if env_key:
        return env_key
    return _read_oauth_token(credentials_path or _default_credentials_path())


def anthropic_credentials_available(credentials_path: Path | None = None) -> bool:
    """Cheap check for hook startup: do we have ANY auth path for the API?"""
    return resolve_anthropic_api_key(credentials_path) is not None


def _is_oauth(token: str) -> bool:
    return token.startswith(_OAUTH_TOKEN_PREFIX)


_MODEL_ALIASES: dict[str, str] = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
}


def resolve_model(alias_or_id: str) -> str:
    """Map a short alias ('haiku'/'sonnet'/'opus') to the current model ID.

    Pass-through for anything else so callers can pin to an exact version.
    """
    return _MODEL_ALIASES.get(alias_or_id, alias_or_id)


class AnthropicJsonClient:
    """Async client for one-shot structured-output calls against api.anthropic.com.

    Uses tool-forcing to enforce a JSON schema. Each `call()` is ONE HTTP
    round-trip — no retries (the caller's tenacity wrapper handles that),
    no streaming, no tool-loop. Preserves the bounded-LLM-call invariant.
    """

    def __init__(
        self,
        *,
        base_url: str = "https://api.anthropic.com",
        anthropic_version: str = "2023-06-01",
        timeout_s: float = 60.0,
    ) -> None:
        self._base_url = base_url
        self._version = anthropic_version
        self._http = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        )

    async def call(
        self,
        *,
        model: str,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
        schema_description: str,
        api_key: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """One forced-tool call. Returns the parsed JSON dict from `tool_use.input`.

        Raises `ValueError` for any non-JSON / no-tool-use response so the
        outer tenacity wrapper can retry within its existing budget.
        """
        body: dict[str, Any] = {
            "model": resolve_model(model),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [
                {
                    "name": schema_name,
                    "description": schema_description,
                    "input_schema": schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": schema_name},
        }

        headers: dict[str, str] = {
            "anthropic-version": self._version,
            "content-type": "application/json",
            "accept": "application/json",
        }
        if _is_oauth(api_key):
            headers["authorization"] = f"Bearer {api_key}"
            headers["anthropic-beta"] = _OAUTH_BETA_FLAG
        else:
            headers["x-api-key"] = api_key

        resp = await self._http.post("/v1/messages", json=body, headers=headers)
        if resp.status_code >= 400:
            snippet = resp.text[:400]
            raise ValueError(f"anthropic api {resp.status_code}: {snippet}")
        data = resp.json()
        for block in data.get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == schema_name:
                inp = block.get("input")
                if isinstance(inp, dict):
                    return inp
        raise ValueError(
            f"anthropic api returned no {schema_name} tool_use block: "
            f"stop_reason={data.get('stop_reason')!r}"
        )

    async def close(self) -> None:
        await self._http.aclose()
