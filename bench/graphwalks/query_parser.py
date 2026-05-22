"""Two query-parser strategies for GraphWalks operations.

`RegexParser`  — pure regex, no LLM call. Fastest, exact.
`JudgeParser`  — sends the operation text to qwen/qwen3.5-9b with a strict
                 JSON schema. EXACTLY one LLM call per query.

Both return the same dict shape:
    {"op": "bfs", "start": str, "depth": int}
    {"op": "parents", "start": str}
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from bench.graphwalks.loader import _BFS_RE, _PARENTS_RE  # type: ignore[attr-defined]
from timegraph.config import get_settings


QUERY_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "graphwalks_query",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "op": {"type": "string", "enum": ["bfs", "parents"]},
                "start": {"type": "string"},
                "depth": {"type": ["integer", "null"]},
            },
            "required": ["op", "start", "depth"],
        },
    },
}


_JUDGE_SYSTEM = (
    "You convert a natural-language graph operation into a structured query. "
    "Two operations are supported:\n"
    "  1) BFS frontier at exact depth D from a starting node:\n"
    '     -> {"op":"bfs","start":"<node_id>","depth":<int>}\n'
    "  2) Parents (direct predecessors) of a node:\n"
    '     -> {"op":"parents","start":"<node_id>","depth":null}\n'
    "Extract the node ID and (for BFS) the depth exactly as written. "
    "Output the JSON object only. Do not solve the operation."
)


class RegexParser:
    """Pure-string parser. Zero LLM calls. Returns parse + always 0 latency."""

    @staticmethod
    def parse(operation_text: str) -> tuple[dict[str, Any], float]:
        t0 = time.perf_counter()
        bm = _BFS_RE.search(operation_text)
        if bm:
            out = {"op": "bfs", "start": bm.group(1), "depth": int(bm.group(2))}
            return out, (time.perf_counter() - t0) * 1000
        pm = _PARENTS_RE.search(operation_text)
        if pm:
            out = {"op": "parents", "start": pm.group(1), "depth": None}
            return out, (time.perf_counter() - t0) * 1000
        raise ValueError(f"could not regex-parse: {operation_text[:200]!r}")


class JudgeParser:
    """Sends the operation text to qwen3.5-9b with strict JSON schema.

    The model never sees the graph — only the question. That's the entire
    architectural point. Exactly one LLM call per parse.
    """

    def __init__(self) -> None:
        s = get_settings()
        self._url = f"{s.judge_url}/chat/completions"
        self._model = s.judge_model
        self._http = httpx.AsyncClient(timeout=s.judge_timeout_s)

    async def parse(self, operation_text: str) -> tuple[dict[str, Any], float]:
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": operation_text.strip()},
            ],
            "max_tokens": 256,
            "temperature": 0.0,
            "top_p": 1.0,
            "response_format": QUERY_SCHEMA,
        }
        t0 = time.perf_counter()
        resp = await self._http.post(self._url, json=body)
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000
        data = resp.json()
        msg = data["choices"][0]["message"]
        raw = msg.get("content") or msg.get("reasoning_content") or ""
        if not raw:
            raise ValueError("empty parser response")
        obj = json.loads(raw)
        return obj, latency_ms

    async def close(self) -> None:
        await self._http.aclose()
