"""Anthropic Messages API provider."""

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

        headers = {
            "x-api-key": api_key,
            "anthropic-version": self._s.anthropic_version,
            "content-type": "application/json",
            "accept": "text/event-stream",
        }
        if extra_headers:
            for k, v in extra_headers.items():
                # Don't let a forwarded header overwrite auth/version.
                if k.lower() in ("x-api-key", "anthropic-version", "content-type", "accept"):
                    continue
                headers[k.lower()] = v

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
            log.exception("anthropic stream unexpected error")
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
