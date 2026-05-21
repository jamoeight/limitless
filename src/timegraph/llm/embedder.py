"""Embedding client — OpenAI-compat /embeddings (default: LM Studio nomic 768D).

The architecture is dim-agnostic as long as `embedder_dim` (config) matches
the dim the Qdrant collections were created with. To swap to BGE-M3 later,
stand up a local FastAPI wrapper that speaks /v1/embeddings and bump
`embedder_url` / `embedder_model` / `embedder_dim`.

Async by design — every other LLM/storage client in the project is async.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from timegraph.config import get_settings

log = structlog.get_logger(__name__)


class EmbedderClient:
    """Async OpenAI-compat embedding client."""

    def __init__(self) -> None:
        self.s = get_settings()
        self._http = httpx.AsyncClient(timeout=self.s.embedder_timeout_s)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def _post(self, inputs: list[str]) -> list[list[float]]:
        body: dict[str, Any] = {"model": self.s.embedder_model, "input": inputs}
        resp = await self._http.post(f"{self.s.embedder_url}/embeddings", json=body)
        resp.raise_for_status()
        data = resp.json()["data"]
        vecs = [item["embedding"] for item in data]
        if vecs and len(vecs[0]) != self.s.embedder_dim:
            raise ValueError(
                f"embedder dim mismatch: got {len(vecs[0])}, configured {self.s.embedder_dim}. "
                f"Either the wrong model is loaded or embedder_dim is stale."
            )
        return vecs

    async def embed_one(self, text: str) -> list[float]:
        return (await self._post([text]))[0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        bs = self.s.embedder_batch_size
        for i in range(0, len(texts), bs):
            out.extend(await self._post(texts[i : i + bs]))
        return out

    async def close(self) -> None:
        await self._http.aclose()
