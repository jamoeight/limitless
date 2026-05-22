"""In-context MRCR baseline: send the full message list to qwen/qwen3.5-9b
chat completions. The last user message is the query; the model must reply
with the prepend-prefixed verbatim text of the requested needle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from timegraph.config import get_settings


@dataclass
class BaselineResult:
    response: str
    latency_ms: float
    status: str       # "ok" | "context_overflow" | "http_error"
    error: str | None = None
    completion_tokens: int | None = None
    prompt_tokens: int | None = None


class BaselineRunner:
    def __init__(self, max_tokens: int = 8192, timeout_s: float = 600.0) -> None:
        s = get_settings()
        self._url = f"{s.judge_url}/chat/completions"
        self._model = s.judge_model
        self._max_tokens = max_tokens
        self._http = httpx.AsyncClient(timeout=timeout_s)

    async def run(self, messages: list[dict[str, str]]) -> BaselineResult:
        body = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
        }
        t0 = time.perf_counter()
        try:
            resp = await self._http.post(self._url, json=body)
        except httpx.HTTPError as e:
            return BaselineResult(
                response="", latency_ms=(time.perf_counter() - t0) * 1000,
                status="http_error", error=repr(e),
            )
        latency_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code >= 400:
            body_text = resp.text[:500]
            is_overflow = any(s in body_text.lower() for s in ("context", "token", "length"))
            return BaselineResult(
                response="", latency_ms=latency_ms,
                status="context_overflow" if is_overflow else "http_error",
                error=body_text,
            )
        data = resp.json()
        msg = data["choices"][0]["message"]
        text = msg.get("content") or msg.get("reasoning_content") or ""
        usage = data.get("usage", {})
        return BaselineResult(
            response=text, latency_ms=latency_ms, status="ok",
            completion_tokens=usage.get("completion_tokens"),
            prompt_tokens=usage.get("prompt_tokens"),
        )

    async def close(self) -> None:
        await self._http.aclose()
