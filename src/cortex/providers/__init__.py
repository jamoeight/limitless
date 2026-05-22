"""Provider adapters — one per upstream model API.

Each provider implements the `Provider` protocol from `.base`:
  - `name`: short identifier ("anthropic", "openai", "lmstudio")
  - `stream(req, api_key, extra_headers) -> AsyncIterator[CortexChunk]`
  - `aclose()`: shut down any pooled HTTP clients

Provider selection is by model-name prefix today (see `cortex.server.route_provider`).
Eventually we'll add an explicit `provider` field in the request, or route
based on a tenant config.
"""

from __future__ import annotations

from .base import Provider

__all__ = ["Provider"]
