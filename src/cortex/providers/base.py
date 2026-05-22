"""Provider protocol."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from cortex.canonical import CortexChunk, CortexRequest


class Provider(Protocol):
    """Async streaming client for an upstream model API.

    Implementations live in `cortex.providers.{anthropic,openai,lmstudio}`.
    They take a canonical `CortexRequest`, translate it to the provider's
    wire format, open a streaming HTTP connection, parse the response, and
    yield canonical `CortexChunk` events.

    The same provider instance can be reused across requests; the underlying
    httpx.AsyncClient is pooled. Call `aclose()` at shutdown.
    """

    name: str

    def stream(
        self,
        req: CortexRequest,
        api_key: str,
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncIterator[CortexChunk]:
        """Open a streaming request and yield canonical chunks.

        Args:
            req: canonical request payload. `req.stream` is forced True by the
                provider regardless of what the caller set — even non-streaming
                clients are served via stream + accumulate at the server edge.
            api_key: provider-specific API key string. Forwarded as the
                provider expects (Anthropic: x-api-key header; OpenAI:
                Authorization: Bearer).
            extra_headers: optional pass-through headers (e.g., anthropic-beta,
                OpenAI-Organization). Provider-required headers like
                anthropic-version are set internally; do not pass them here.

        Yields:
            Canonical chunks in the order the upstream emits them. The first
            chunk is always `ChunkMessageStart`; the last is `ChunkMessageStop`
            (or `ChunkError` on failure).
        """
        ...

    async def aclose(self) -> None:
        """Close any pooled HTTP clients. Idempotent."""
        ...
