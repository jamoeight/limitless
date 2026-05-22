"""LLM judge for MRCR queries.

The model sees ONLY the final user turn — never the conversation. It
emits a structured retrieval spec:

    {
      "needle_request": "write a short scene in a play about blueberries",
      "position": 1,
      "prepend": "6xO8mh9FsP"
    }

The LLM's job is purely to reconstruct what the original user request must
have looked like, given the query's phrasing. Exactly one LLM call per query.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from timegraph.config import get_settings


_QUERY_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "mrcr_query",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "needle_request": {
                    "type": "string",
                    "description": (
                        "The original user turn the question refers to, "
                        "VERBATIM, in the exact form 'write a/an <type> "
                        "about <topic>' (no period, no leading article fix, "
                        "no quoting)."
                    ),
                },
                "position": {"type": "integer"},
                "prepend": {"type": "string"},
            },
            "required": ["needle_request", "position", "prepend"],
        },
    },
}


_SYSTEM = (
    "You convert one MRCR query into a structured retrieval spec.\n"
    "\n"
    "Background:\n"
    "The conversation history contains many user turns of the form\n"
    '  "write a <type> about <topic>"\n'
    "and the user is now asking for the Nth one of a particular kind.\n"
    "Example query:\n"
    "  Prepend 6xO8mh9FsP to the 2nd (1 indexed) short scene in a play\n"
    "  about blueberries. Do not include any other text in your response.\n"
    "\n"
    "Your output must be JSON of the form:\n"
    '  {"needle_request":"write a short scene in a play about blueberries",\n'
    '   "position":2, "prepend":"6xO8mh9FsP"}\n'
    "\n"
    "Rules:\n"
    "- needle_request ALWAYS starts with 'write a ' (with a literal 'a' — never\n"
    "  'an', even before vowel-initial words like 'email' or 'article'; the\n"
    "  source dataset uses 'a' uniformly) and has NO trailing period.\n"
    "- position is the integer the query names (1-indexed).\n"
    "- prepend is the random token verbatim — preserve case and characters.\n"
    "- Output the JSON object only. No prose."
)


class MrcrJudgeParser:
    """One LLM call per MRCR query."""

    def __init__(self) -> None:
        s = get_settings()
        self._url = f"{s.judge_url}/chat/completions"
        self._model = s.judge_model
        self._http = httpx.AsyncClient(timeout=s.judge_timeout_s)

    async def parse(self, query_text: str) -> tuple[dict[str, Any], float]:
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": query_text.strip()},
            ],
            "max_tokens": 256,
            "temperature": 0.0,
            "top_p": 1.0,
            "response_format": _QUERY_SCHEMA,
        }
        t0 = time.perf_counter()
        resp = await self._http.post(self._url, json=body)
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000
        data = resp.json()
        msg = data["choices"][0]["message"]
        raw = msg.get("content") or msg.get("reasoning_content") or ""
        if not raw:
            raise ValueError("empty MRCR parser response")
        obj = json.loads(raw)
        return obj, latency_ms

    async def close(self) -> None:
        await self._http.aclose()
