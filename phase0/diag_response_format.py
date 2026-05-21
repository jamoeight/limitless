"""Diagnostic: send one judge call to LM Studio, print the raw response.

Tests 3 configurations to identify what works:
  A. response_format: json_schema strict (current default)
  B. response_format: json_object (looser; any JSON)
  C. no response_format (plain prompt; see what Qwopus naturally emits)

Run:  python phase0/diag_response_format.py
"""

from __future__ import annotations

import asyncio
import json

import httpx

from timegraph.config import get_settings
from timegraph.llm.schemas import JUDGE_RESPONSE_FORMAT

PROMPT = """You are a contradiction-resolution judge. Given two facts about Alice's location:
- e1 (fact_id=f_001): Alice lives in Boston (valid from 2026-01-01)
- e2 (fact_id=f_002): Alice lives in Seattle (valid from 2026-04-15)

User query: Where does Alice live now?

Decide the resolution. Return one of: e1_correct, e2_correct, both_partial, unresolved.
"""


async def call(label: str, body: dict) -> None:
    s = get_settings()
    print(f"\n{'=' * 64}\n{label}\n{'=' * 64}")
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{s.judge_url}/chat/completions", json=body)
    print(f"HTTP {r.status_code}")
    if r.status_code != 200:
        print("BODY:", r.text[:1000])
        return
    data = r.json()
    content = data["choices"][0]["message"].get("content", "")
    reasoning = data["choices"][0]["message"].get("reasoning_content")  # LM Studio's separated reasoning, if any
    finish = data["choices"][0].get("finish_reason")
    usage = data.get("usage", {})
    print(f"finish_reason: {finish}")
    print(f"usage: {usage}")
    if reasoning:
        print(f"reasoning_content (len={len(reasoning)}):\n  {reasoning[:400]!r}...")
    print(f"content (len={len(content)}):\n{content[:1500]}")
    print()
    # Try to parse content as JSON
    try:
        obj = json.loads(content)
        print(f"[JSON parse OK] keys: {list(obj.keys())}")
    except json.JSONDecodeError as e:
        print(f"[JSON parse FAILED] {e}")


async def main() -> None:
    s = get_settings()
    base = {
        "model": s.judge_model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 1024,
        "temperature": 0.3,
    }

    # A. Strict json_schema
    await call(
        "A. response_format: json_schema strict",
        {**base, "response_format": JUDGE_RESPONSE_FORMAT},
    )

    # B. Looser json_object mode
    await call(
        "B. response_format: json_object",
        {**base, "response_format": {"type": "json_object"}},
    )

    # C. No response_format — what does Qwopus emit naturally?
    await call(
        "C. NO response_format (plain Qwopus output)",
        base,
    )


if __name__ == "__main__":
    asyncio.run(main())
