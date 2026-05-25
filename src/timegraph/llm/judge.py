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

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx
import structlog
from jinja2 import Environment, FileSystemLoader
from tenacity import retry, stop_after_attempt, wait_exponential

from timegraph.config import get_settings
from timegraph.llm.anthropic_client import AnthropicJsonClient, resolve_anthropic_api_key
from timegraph.llm.schemas import JUDGE_RESPONSE_FORMAT, JUDGE_SCHEMA
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
        self._anthropic: AnthropicJsonClient | None = None

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

        backend = self.s.judge_backend
        if backend == "anthropic_api":
            return await self._judge_via_anthropic_api(prompt)
        if backend == "claude_cli":
            return await self._judge_via_claude_cli(prompt)
        if backend == "lm_studio":
            return await self._judge_via_lm_studio(prompt)
        raise ValueError(f"unknown judge_backend: {backend!r}")

    async def _judge_via_lm_studio(self, prompt: str) -> JudgeOutput:
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

    async def _judge_via_anthropic_api(self, prompt: str) -> JudgeOutput:
        # Direct POST to api.anthropic.com. Forced-tool schema gives us
        # structured output without the `claude -p` agent-loop overhead.
        # Still exactly ONE LLM round-trip — judge_call_count stays bounded
        # at 1 (the structural invariant enforced by infer()).
        api_key = resolve_anthropic_api_key()
        if not api_key:
            raise ValueError(
                "anthropic_api backend requested but no credentials found "
                "(set ANTHROPIC_API_KEY or run `claude login`)"
            )
        if self._anthropic is None:
            self._anthropic = AnthropicJsonClient(
                base_url=self.s.anthropic_api_base_url,
                anthropic_version=self.s.anthropic_api_version,
                timeout_s=self.s.judge_anthropic_timeout_s,
            )

        t0 = time.perf_counter()
        structured = await self._anthropic.call(
            model=self.s.judge_anthropic_model,
            prompt=prompt,
            schema=JUDGE_SCHEMA,
            schema_name="record_conflict_resolution",
            schema_description=(
                "Record the resolution of the candidate conflict pairs. You "
                "MUST call this tool exactly once with the chosen resolution."
            ),
            api_key=api_key,
            max_tokens=self.s.judge_anthropic_max_tokens,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        return JudgeOutput(
            resolution=Resolution(structured["resolution"]),
            reason=structured["reason"],
            confidence=float(structured["confidence"]),
            thinking=structured.get("thinking", ""),
            call_count=1,
            latency_ms=latency_ms,
            raw_json=json.dumps(structured),
        )

    async def _judge_via_claude_cli(self, prompt: str) -> JudgeOutput:
        # Shell out to `claude -p` using the caller's OAuth session.
        # The Claude Code agent context still loads (no --bare without API key),
        # so first call ~$0.05 (cache write), subsequent within 5min ~$0.005 (cache read).
        argv = [
            self.s.judge_claude_cli_path,
            "-p",
            "--no-session-persistence",
            "--model",
            self.s.judge_claude_model,
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(JUDGE_SCHEMA),
            "--tools",
            "",
            "--disable-slash-commands",
            "--max-budget-usd",
            str(self.s.judge_claude_budget_usd),
        ]

        t0 = time.perf_counter()
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(prompt.encode("utf-8")),
                timeout=self.s.judge_claude_timeout_s,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise ValueError(f"claude -p timed out after {self.s.judge_claude_timeout_s}s")
        latency_ms = (time.perf_counter() - t0) * 1000

        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace")[:500]
            raise ValueError(f"claude -p exit {proc.returncode}: {err}")

        raw = stdout.decode("utf-8", errors="replace")
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError as e:
            log.error("claude -p returned non-JSON", head=raw[:200])
            raise ValueError(f"claude -p non-JSON stdout: {e}") from e

        if envelope.get("is_error"):
            errs = envelope.get("errors") or [envelope.get("result", "unknown")]
            raise ValueError(f"claude -p error: {errs}")

        structured = envelope.get("structured_output")
        if not structured:
            log.error("claude -p missing structured_output", envelope_keys=list(envelope.keys()))
            raise ValueError("claude -p returned no structured_output")

        return JudgeOutput(
            resolution=Resolution(structured["resolution"]),
            reason=structured["reason"],
            confidence=float(structured["confidence"]),
            thinking=structured.get("thinking", ""),
            call_count=1,
            latency_ms=latency_ms,
            raw_json=json.dumps(structured),
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
        if self._anthropic is not None:
            await self._anthropic.close()
            self._anthropic = None
