"""Embedding client — two backends:

- `fastembed` (default): in-process ONNX, no extra server. Model downloads
  to fastembed's cache dir on first use (~90 MB for bge-small-en-v1.5).
- `openai_compat`: HTTP POST to a /v1/embeddings endpoint (LM Studio, Ollama,
  OpenAI, etc.). Same interface, just routed differently.

Backend selection is via `Settings.embedder_backend`. The fastembed model is
cached at class level so the 5 EmbedderClient instantiation sites across
ops/* share one loaded model per process.

Async by design — every other LLM/storage client is async. fastembed is
synchronous (CPU/ONNX) so we wrap calls in `asyncio.to_thread`.
"""

from __future__ import annotations

from typing import Any

import asyncio
import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from timegraph.config import get_settings

log = structlog.get_logger(__name__)


class EmbedderClient:
    """Async embedding client — backend selected by `Settings.embedder_backend`."""

    _fastembed_cache: dict[str, Any] = {}

    def __init__(self) -> None:
        self.s = get_settings()
        self._fastembed: Any = None
        self._http: httpx.AsyncClient | None = None
        if self.s.embedder_backend == "fastembed":
            self._fastembed = self._get_or_load_fastembed(self.s.embedder_model)
        else:
            self._http = httpx.AsyncClient(timeout=self.s.embedder_timeout_s)

    @classmethod
    def _get_or_load_fastembed(cls, model_name: str) -> Any:
        cached = cls._fastembed_cache.get(model_name)
        if cached is not None:
            return cached
        from fastembed import TextEmbedding

        model = TextEmbedding(model_name=model_name)
        cls._fastembed_cache[model_name] = model
        log.info("loaded fastembed model", model=model_name)
        return model

    async def _embed_fastembed(self, inputs: list[str]) -> list[list[float]]:
        model = self._fastembed

        def _sync() -> list[list[float]]:
            vecs = list(model.embed(inputs))
            return [v.tolist() for v in vecs]

        result = await asyncio.to_thread(_sync)
        if result and len(result[0]) != self.s.embedder_dim:
            raise ValueError(
                f"embedder dim mismatch: fastembed returned {len(result[0])}, "
                f"configured {self.s.embedder_dim}. embedder_dim is stale for model "
                f"{self.s.embedder_model}."
            )
        return result

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def _embed_http(self, inputs: list[str]) -> list[list[float]]:
        assert self._http is not None
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

    async def _post(self, inputs: list[str]) -> list[list[float]]:
        if self._fastembed is not None:
            return await self._embed_fastembed(inputs)
        return await self._embed_http(inputs)

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
        if self._http is not None:
            await self._http.aclose()
