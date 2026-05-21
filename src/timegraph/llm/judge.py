"""Qwopus3.6-27B JSON-judge client (via LM Studio on :1234).

★ LATENCY / ACCURACY BOTTLENECK for the bounded-LLM-call thesis.

Spec contract (B.4-v2 stage-2): EXACTLY 1 call per `infer()` invocation,
regardless of graph depth. Structured output via LM Studio's strict
`response_format: json_schema`. Qwopus's native `<think>...</think>` reasoning
is preserved by giving the schema a leading `thinking` string field — the
model reasons inside the structured payload rather than fighting strict mode.

The `thinking` field is parsed off before returning the result.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx
import structlog
from jinja2 import Environment, FileSystemLoader
from tenacity import retry, stop_after_attempt, wait_exponential

from timegraph.config import get_settings
from timegraph.llm.schemas import JUDGE_RESPONSE_FORMAT
from timegraph.types import ConflictTriple, Resolution

log = structlog.get_logger(__name__)


class JudgeOutput:
    """Parsed judge response with the load-bearing call_count instrumentation."""

    def __init__(
        self,
        resolution: Resolution,
        reason: str,
        confidence: float,
        thinking: str,
        call_count: int,
        latency_ms: float,
        raw_json: str,
    ):
        self.resolution = resolution
        self.reason = reason
        self.confidence = confidence
        self.thinking = thinking
        self.call_count = call_count
        self.latency_ms = latency_ms
        self.raw_json = raw_json


class JudgeClient:
    """Async client for the Qwopus JSON-judge call site via LM Studio."""

    def __init__(self) -> None:
        self.s = get_settings()
        self._jinja = Environment(
            loader=FileSystemLoader(Path(__file__).parent / "prompts"),
            autoescape=False,
        )
        self._http = httpx.AsyncClient(timeout=self.s.judge_timeout_s)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def judge_conflicts(
        self,
        query: str,
        conflicts: list[ConflictTriple],
        attestations: list[dict[str, Any]] | None = None,
        source_episodes_truncated: list[str] | None = None,
    ) -> JudgeOutput:
        """The load-bearing stage-2 judge call. One LLM round-trip per invocation."""
        if len(conflicts) > 8:
            raise ValueError("conflicts must be ≤8 — stage-1 must truncate")

        tpl = self._jinja.get_template("judge.j2")
        prompt = tpl.render(
            query=query,
            conflicts=[c.model_dump() for c in conflicts],
            attestations=attestations or [],
            episodes=source_episodes_truncated or [],
        )

        body: dict[str, Any] = {
            "model": self.s.judge_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.s.judge_max_tokens,
            # Greedy decoding — the judge is a classifier, not a creative task.
            # Run-to-run reproducibility matters more than diversity here.
            "temperature": 0.0,
            "top_p": 1.0,
            "response_format": JUDGE_RESPONSE_FORMAT,
            # LM Studio respects OpenAI's `stream: false` default; no need to set.
        }

        t0 = time.perf_counter()
        resp = await self._http.post(f"{self.s.judge_url}/chat/completions", json=body)
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000

        data = resp.json()
        msg = data["choices"][0]["message"]
        # LM Studio routes reasoning-model output to `reasoning_content`. For
        # Qwopus with strict json_schema, the JSON body lands there and `content`
        # is empty. Fall back across both fields so we work with either runtime.
        raw = msg.get("content") or msg.get("reasoning_content") or ""
        if not raw:
            log.error("judge returned empty content + reasoning_content", message=msg)
            raise ValueError("empty judge response (both content and reasoning_content blank)")
        parsed = self._parse_response(raw)
        return JudgeOutput(
            resolution=parsed["resolution"],
            reason=parsed["reason"],
            confidence=parsed["confidence"],
            thinking=parsed["thinking"],
            call_count=1,
            latency_ms=latency_ms,
            raw_json=raw,
        )

    @staticmethod
    def _parse_response(raw: str) -> dict[str, Any]:
        """LM Studio returns the structured JSON body verbatim (strict mode)."""
        import json

        obj = json.loads(raw)
        return {
            "thinking": obj.get("thinking", ""),
            "resolution": Resolution(obj["resolution"]),
            "reason": obj["reason"],
            "confidence": float(obj["confidence"]),
        }

    async def close(self) -> None:
        await self._http.aclose()
