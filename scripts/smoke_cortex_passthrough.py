"""Live smoke test for the Cortex MVP-1 passthrough proxy.

Usage:
    # Terminal 1
    .venv/Scripts/python -m cortex.server

    # Terminal 2 — set your real Anthropic API key
    set ANTHROPIC_API_KEY=sk-ant-...
    python scripts/smoke_cortex_passthrough.py

What it does:
    1. POSTs a tiny conversation to http://127.0.0.1:8080/v1/messages
    2. Prints the streamed events as they arrive
    3. Prints the assembled assistant text at the end

This is a manual verification you can run after wiring; it requires a real
Anthropic API key because the proxy is pure passthrough at MVP-1 (no offline
fallback yet).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import httpx


async def main() -> int:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1

    body = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 128,
        "messages": [
            {"role": "user", "content": "Say 'pong' and nothing else."},
        ],
        "stream": True,
    }

    accumulated_text = ""
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8080", timeout=60.0) as client:
        async with client.stream(
            "POST",
            "/v1/messages",
            headers={"x-api-key": api_key, "content-type": "application/json"},
            json=body,
        ) as resp:
            print(f"status: {resp.status_code}")
            if resp.status_code != 200:
                body_bytes = await resp.aread()
                print(body_bytes.decode("utf-8", errors="replace"))
                return 1

            current_event: str | None = None
            data_buf: list[str] = []
            async for line in resp.aiter_lines():
                if line == "":
                    if current_event and data_buf:
                        raw = "".join(data_buf)
                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            payload = {"_raw": raw}
                        if current_event == "content_block_delta":
                            d = payload.get("delta", {})
                            if d.get("type") == "text_delta":
                                accumulated_text += d.get("text", "")
                                print(d.get("text", ""), end="", flush=True)
                        elif current_event != "ping":
                            print(f"\n[{current_event}] {payload}")
                    current_event = None
                    data_buf = []
                    continue
                if line.startswith("event:"):
                    current_event = line[len("event:") :].strip()
                elif line.startswith("data:"):
                    data_buf.append(line[len("data:") :].lstrip())

    print(f"\n\n--- accumulated assistant text ---\n{accumulated_text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
