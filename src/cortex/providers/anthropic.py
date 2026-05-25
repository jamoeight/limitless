"""Anthropic Messages API provider.

Auth mode is auto-detected from the token shape:

  - `sk-ant-api...`  -> classic API key, sent as `x-api-key`.
  - `sk-ant-oat...`  -> Claude OAuth access token (from `~/.claude/.credentials.json`,
    populated by `claude login`). Sent as `Authorization: Bearer ...` along
    with the `anthropic-beta: oauth-2025-04-20` opt-in header that the public
    /v1/messages endpoint requires for OAuth-issued tokens.

The detection is intentional: when a caller sets `ANTHROPIC_BASE_URL` to
cortex, Claude Code forwards whatever auth it would have sent to
api.anthropic.com directly, which is the OAuth bearer in the common
no-API-key case. Recognizing it here means cortex re-uses the user's Claude
subscription transparently and keeps `req.tools` flowing end-to-end (unlike
the legacy `claude -p` subprocess path, which had to strip tools).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from cortex.canonical import ChunkError, CortexChunk, CortexRequest
from cortex.config import CortexSettings, get_cortex_settings
from cortex.translate.anthropic import parse_anthropic_event, to_anthropic_request

log = structlog.get_logger(__name__)

_OAUTH_TOKEN_PREFIX = "sk-ant-oat"
_OAUTH_BETA_FLAG = "oauth-2025-04-20"


def _is_oauth_token(api_key: str) -> bool:
    return api_key.startswith(_OAUTH_TOKEN_PREFIX)


class AnthropicProvider:
    """Async streaming client for api.anthropic.com /v1/messages.

    Connection pool is shared across requests; remember to `aclose()` at
    shutdown.
    """

    name = "anthropic"

    def __init__(self, settings: CortexSettings | None = None) -> None:
        self._s = settings or get_cortex_settings()
        self._client = httpx.AsyncClient(
            base_url=self._s.anthropic_base_url,
            timeout=httpx.Timeout(
                self._s.upstream_timeout_s,
                connect=self._s.upstream_connect_timeout_s,
            ),
            # Anthropic's responses can be large; let httpx handle bytestream
            # without buffering the whole body.
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )

    async def stream(
        self,
        req: CortexRequest,
        api_key: str,
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncIterator[CortexChunk]:
        body = to_anthropic_request(req)
        body["stream"] = True

        headers: dict[str, str] = {
            "anthropic-version": self._s.anthropic_version,
            "content-type": "application/json",
            "accept": "text/event-stream",
        }
        if _is_oauth_token(api_key):
            headers["authorization"] = f"Bearer {api_key}"
            headers["anthropic-beta"] = _OAUTH_BETA_FLAG
        else:
            headers["x-api-key"] = api_key

        if extra_headers:
            for k, v in extra_headers.items():
                kl = k.lower()
                # Don't let a forwarded header overwrite auth/version/format.
                if kl in ("x-api-key", "authorization", "anthropic-version", "content-type", "accept"):
                    continue
                # `anthropic-beta` is comma-joined per Anthropic's spec; preserve
                # the OAuth flag if we set it AND honor any caller-supplied betas.
                if kl == "anthropic-beta" and kl in headers:
                    existing = headers[kl]
                    if v and v not in existing.split(","):
                        headers[kl] = f"{existing},{v}" if existing else v
                    continue
                headers[kl] = v

        try:
            async with self._client.stream(
                "POST", "/v1/messages", json=body, headers=headers
            ) as resp:
                if resp.status_code >= 400:
                    raw = await resp.aread()
                    yield _error_from_response(resp.status_code, raw)
                    return

                async for chunk in _iter_sse(resp):
                    yield chunk
        except httpx.RequestError as e:
            log.warning("anthropic stream transport error", error=str(e))
            yield ChunkError(error_type="upstream_transport_error", message=str(e))
        except Exception as e:  # noqa: BLE001
            try:
                log.exception("anthropic stream unexpected error")
            except Exception:  # noqa: BLE001
                pass
            yield ChunkError(error_type="upstream_unexpected_error", message=str(e))

    async def aclose(self) -> None:
        await self._client.aclose()


def _error_from_response(status_code: int, raw: bytes) -> ChunkError:
    try:
        body: Any = json.loads(raw)
    except json.JSONDecodeError:
        body = None
    if isinstance(body, dict):
        err = body.get("error", {}) or {}
        return ChunkError(
            error_type=err.get("type", f"http_{status_code}"),
            message=err.get("message", raw[:512].decode("utf-8", errors="replace")),
        )
    return ChunkError(
        error_type=f"http_{status_code}",
        message=raw[:512].decode("utf-8", errors="replace"),
    )


async def _iter_sse(resp: httpx.Response) -> AsyncIterator[CortexChunk]:
    """Parse Anthropic SSE lines into canonical chunks.

    Anthropic SSE shape:
        event: <name>
        data: <json>
        <blank line>

    We accumulate `event:` and `data:` until we hit a blank line, then dispatch.
    """
    current_event: str | None = None
    data_buf: list[str] = []
    async for line in resp.aiter_lines():
        if line == "":
            # Dispatch the accumulated event.
            if data_buf and current_event is not None:
                raw = "".join(data_buf)
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    payload = None
                if payload is not None:
                    chunk = parse_anthropic_event(current_event, payload)
                    if chunk is not None:
                        yield chunk
            current_event = None
            data_buf = []
            continue
        if line.startswith(":"):
            # SSE comment; skip.
            continue
        if line.startswith("event:"):
            current_event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            # Per SSE spec, multiple data: lines are concatenated with \n.
            data_buf.append(line[len("data:") :].lstrip())
