"""Fact extraction client (via LM Studio on :1234).

Called on every `add_episode`. Target model: Qwen3-7B-Instruct (no thinking,
structured translation). Until Qwen3-7B is loaded in LM Studio, can fall back
to the judge model via `Settings.use_judge_for_extraction=True` — slower, but
unblocks Phase 0 evals on day one.

Spec target: F1 ≥0.70 vs hand-labeled facts at p95 ≤8s per episode.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import structlog
from jinja2 import Environment, FileSystemLoader
from tenacity import retry, stop_after_attempt, wait_exponential

from timegraph.config import get_settings
from timegraph.llm.schemas import EXTRACTOR_RESPONSE_FORMAT
from timegraph.types import Fact

log = structlog.get_logger(__name__)


class ExtractorClient:
    def __init__(self) -> None:
        self.s = get_settings()
        self._jinja = Environment(
            loader=FileSystemLoader(Path(__file__).parent / "prompts"),
            autoescape=False,
        )
        self._http = httpx.AsyncClient(timeout=self.s.extractor_timeout_s)
        # Phase 0 day-1: only Qwopus may be loaded in LM Studio. Allow override.
        self._model_name = (
            self.s.judge_model if self.s.use_judge_for_extraction else self.s.extractor_model
        )

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
    async def extract_facts(
        self,
        episode_content: str,
        event_time: datetime,
        session_id: str,
        source: str,
    ) -> tuple[list[Fact], float]:
        """Extract facts from an episode. Returns (facts, latency_ms)."""
        tpl = self._jinja.get_template("extract.j2")
        prompt = tpl.render(content=episode_content, event_time=event_time.isoformat())

        body: dict[str, Any] = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.s.extractor_max_tokens,
            "temperature": 0.0,
            "top_p": 0.95,
            "response_format": EXTRACTOR_RESPONSE_FORMAT,
        }

        t0 = time.perf_counter()
        resp = await self._http.post(f"{self.s.extractor_url}/chat/completions", json=body)
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000

        msg = resp.json()["choices"][0]["message"]
        # Reasoning models (e.g., Qwopus when serving as fallback extractor) emit
        # the structured payload in `reasoning_content`; non-reasoning models
        # (Qwen3-7B-Instruct) emit it in `content`. Accept either.
        raw = msg.get("content") or msg.get("reasoning_content") or ""
        if not raw:
            log.error("extractor returned empty content + reasoning_content", message=msg)
            raise ValueError("empty extractor response")
        facts = self._parse_to_facts(raw, event_time, session_id, source)
        return facts, latency_ms

    @staticmethod
    def _parse_to_facts(
        raw: str, event_time: datetime, session_id: str, source: str
    ) -> list[Fact]:
        import json
        import uuid

        obj = json.loads(raw)
        items = obj["facts"]  # strict-mode JSON schema guarantees this key
        out: list[Fact] = []
        for item in items:
            out.append(
                Fact(
                    fact_id=str(uuid.uuid4()),
                    subject=item["subject"],
                    predicate=item["predicate"],
                    object=item["object"],
                    valid_at=event_time,
                    confidence=float(item["confidence"]),
                    session_id=session_id,
                    sources=[source],
                )
            )
        return out

    async def close(self) -> None:
        await self._http.aclose()
