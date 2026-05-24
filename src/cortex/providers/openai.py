"""OpenAI Chat Completions API provider."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from cortex.canonical import ChunkError, CortexChunk, CortexRequest
from cortex.config import CortexSettings, get_cortex_settings
from cortex.translate.openai import new_openai_stream_state, parse_openai_stream_chunk, to_openai_request

log = structlog.get_logger(__name__)


class OpenAIProvider:
    """Async streaming client for api.openai.com /v1/chat/completions.

    Forwards `Authorization: Bearer <api_key>` and optional org/project
    headers. Always asks upstream for `stream: true`; the proxy aggregates
    when the original client wanted non-streaming.
    """

    name = "openai"

    def __init__(self, settings: CortexSettings | None = None) -> None:
        self._s = settings or get_cortex_settings()
        self._client = httpx.AsyncClient(
            base_url=self._s.openai_base_url,
            timeout=httpx.Timeout(
                self._s.upstream_timeout_s,
                connect=self._s.upstream_connect_timeout_s,
            ),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )

    async def stream(
        self,
        req: CortexRequest,
        api_key: str,
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncIterator[CortexChunk]:
        body = to_openai_request(req)
        body["stream"] = True

        headers = {
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
            "accept": "text/event-stream",
        }
        if extra_headers:
            for k, v in extra_headers.items():
                if k.lower() in ("authorization", "content-type", "accept"):
                    continue
                headers[k.lower()] = v

        try:
            async with self._client.stream(
                "POST", "/v1/chat/completions", json=body, headers=headers
            ) as resp:
                if resp.status_code >= 400:
                    raw = await resp.aread()
                    yield _error_from_response(resp.status_code, raw)
                    return

                state = new_openai_stream_state()
                async for line in resp.aiter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].lstrip()
                    if data == "[DONE]":
                        return
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    for chunk in parse_openai_stream_chunk(state, payload):
                        yield chunk
        except httpx.RequestError as e:
            log.warning("openai stream transport error", error=str(e))
            yield ChunkError(error_type="upstream_transport_error", message=str(e))
        except Exception as e:  # noqa: BLE001
            try:
                log.exception("openai stream unexpected error")
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
            error_type=err.get("type") or err.get("code") or f"http_{status_code}",
            message=err.get("message", raw[:512].decode("utf-8", errors="replace")),
        )
    return ChunkError(
        error_type=f"http_{status_code}",
        message=raw[:512].decode("utf-8", errors="replace"),
    )
