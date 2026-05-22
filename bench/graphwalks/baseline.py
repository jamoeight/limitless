"""In-context baseline: send the full GraphWalks prompt to qwen/qwen3.5-9b
and parse `Final Answer: [...]` from the response.

This is the apples-to-apples comparison for the architecture pitch:
  - same LLM (qwen3.5-9b)
  - same question
  - difference: baseline sees the entire graph in context, ours sees only
    the question and queries Neo4j.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import httpx

from timegraph.config import get_settings


_FINAL_RE = re.compile(r"Final Answer:\s*\[([^\]]*)\]", re.IGNORECASE)


@dataclass
class BaselineResult:
    answer: set[str]
    latency_ms: float
    response_text: str
    status: str          # "ok" | "context_overflow" | "format_fail" | "http_error"
    error: str | None = None
    completion_tokens: int | None = None
    prompt_tokens: int | None = None


def parse_final_answer(text: str) -> set[str]:
    """Pull the last `Final Answer: [...]` block from the response."""
    matches = _FINAL_RE.findall(text)
    if not matches:
        return set()
    inner = matches[-1]
    return {n.strip() for n in inner.split(",") if n.strip()}


class BaselineRunner:
    """One-shot in-context evaluator."""

    def __init__(self, max_tokens: int = 8192, timeout_s: float = 300.0) -> None:
        s = get_settings()
        self._url = f"{s.judge_url}/chat/completions"
        self._model = s.judge_model
        self._max_tokens = max_tokens
        self._http = httpx.AsyncClient(timeout=timeout_s)

    async def run(self, prompt: str) -> BaselineResult:
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
        }
        t0 = time.perf_counter()
        try:
            resp = await self._http.post(self._url, json=body)
        except httpx.HTTPError as e:
            return BaselineResult(
                answer=set(), latency_ms=(time.perf_counter() - t0) * 1000,
                response_text="", status="http_error", error=repr(e),
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code >= 400:
            body_text = resp.text[:500]
            is_overflow = (
                "context" in body_text.lower()
                or "token" in body_text.lower()
                or "length" in body_text.lower()
            )
            return BaselineResult(
                answer=set(), latency_ms=latency_ms, response_text=body_text,
                status="context_overflow" if is_overflow else "http_error",
                error=body_text,
            )

        data = resp.json()
        msg = data["choices"][0]["message"]
        text = msg.get("content") or msg.get("reasoning_content") or ""
        usage = data.get("usage", {})
        ans = parse_final_answer(text)
        # Strict: only "ok" if a `Final Answer: [...]` block was actually emitted.
        # Otherwise we'd score `gold == set() == pred == set()` as a true positive
        # whenever the model failed to format anything — a false win.
        status = "ok" if _FINAL_RE.search(text) else "format_fail"
        return BaselineResult(
            answer=ans, latency_ms=latency_ms, response_text=text, status=status,
            completion_tokens=usage.get("completion_tokens"),
            prompt_tokens=usage.get("prompt_tokens"),
        )

    async def close(self) -> None:
        await self._http.aclose()
