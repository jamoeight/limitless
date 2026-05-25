"""Live verification for BUG 1: server tools (web_search_20250305 etc.)
flow through the cortex proxy without 400ing.

Uses the real `_build_app()` with virtualization + auto-ingest enabled, but
injects a mock provider so we can assert the OUTBOUND request body Anthropic
would receive — without actually calling api.anthropic.com (no real OAuth
spend, no network flakiness in CI).

Run:
    PYTHONPATH=src .venv/Scripts/python.exe scripts/verify_web_search_roundtrip.py
Exits 0 on success, 1 on any failure.
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from typing import Any

from fastapi.testclient import TestClient

from cortex.canonical import (
    ChunkContentBlockStart,
    ChunkContentBlockStop,
    ChunkMessageDelta,
    ChunkMessageStart,
    ChunkMessageStop,
    ChunkTextDelta,
    CortexChunk,
    CortexRequest,
    TextBlock,
)
from cortex.config import CortexSettings
from cortex.server import ProviderRegistry, _build_app
from cortex.translate.anthropic import to_anthropic_request


class _RecordingProvider:
    """Capture the CortexRequest the proxy hands us, then emit a stub stream."""

    name = "anthropic"

    def __init__(self) -> None:
        self.last_request: CortexRequest | None = None

    async def stream(
        self,
        req: CortexRequest,
        api_key: str,
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncIterator[CortexChunk]:
        self.last_request = req
        # Stub a complete Anthropic-shaped stream so the server's aggregator
        # produces a clean 200 response.
        yield ChunkMessageStart(message_id="msg_TEST", model=req.model, input_tokens=42)
        yield ChunkContentBlockStart(index=0, block=TextBlock(text=""))
        yield ChunkTextDelta(index=0, text="Stub response.")
        yield ChunkContentBlockStop(index=0)
        yield ChunkMessageDelta(stop_reason="end_turn", output_tokens=3)
        yield ChunkMessageStop()

    async def aclose(self) -> None:
        pass


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    settings = CortexSettings()
    settings.enable_auto_ingest = True
    settings.enable_virtualization = True
    settings.enable_verbatim_recall = False  # avoid the embedder requiring backends

    rec = _RecordingProvider()
    registry = ProviderRegistry()
    registry.register(rec)

    # Stub recall_fn so virtualize.py doesn't try to hit Neo4j/Qdrant.
    async def _noop_recall(query: str, group_id: str, budget: int) -> str:
        return ""

    # Stub the session registry's ingest_fn so we don't need backends.
    from cortex.ingest import SessionRegistry

    async def _noop_ingest(content, source, group_id, session_id, event_time):
        return "ep_stub"

    session_registry = SessionRegistry(settings, ingest_fn=_noop_ingest)

    app = _build_app(
        settings=settings,
        registry=registry,
        session_registry=session_registry,
        recall_fn=_noop_recall,
    )

    body = {
        "model": "claude-opus-4-7",
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "What's the weather in NYC today?"}
        ],
        "tools": [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
            {"type": "bash_20250124", "name": "bash"},
            {
                "name": "my_custom_tool",
                "description": "A regular function tool",
                "input_schema": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            },
        ],
    }

    with TestClient(app) as client:
        resp = client.post(
            "/v1/messages",
            json=body,
            headers={"authorization": "Bearer sk-ant-oat01-FAKE-OAUTH-FOR-VERIFY"},
        )

    _assert(resp.status_code == 200, f"expected 200, got {resp.status_code}: {resp.text[:300]}")
    payload = resp.json()
    _assert(payload.get("type") == "message", f"unexpected response shape: {payload}")

    # Inspect the outbound request that the proxy WOULD have sent upstream.
    assert rec.last_request is not None, "provider was never called"
    outbound = to_anthropic_request(rec.last_request)
    out_tools = outbound.get("tools") or []
    _assert(len(out_tools) == 3, f"expected 3 outbound tools, got {len(out_tools)}: {out_tools}")

    web_search = out_tools[0]
    _assert(web_search.get("type") == "web_search_20250305", f"web_search lost its type: {web_search}")
    _assert(web_search.get("name") == "web_search", f"web_search lost its name: {web_search}")
    _assert(web_search.get("max_uses") == 5, f"web_search dropped max_uses: {web_search}")
    _assert("input_schema" not in web_search, f"web_search erroneously got input_schema: {web_search}")

    bash_tool = out_tools[1]
    _assert(bash_tool.get("type") == "bash_20250124", f"bash lost its type: {bash_tool}")
    _assert("input_schema" not in bash_tool, f"bash erroneously got input_schema: {bash_tool}")

    custom = out_tools[2]
    _assert(custom.get("name") == "my_custom_tool", f"custom tool name: {custom}")
    _assert(
        custom.get("input_schema", {}).get("properties", {}).get("q", {}).get("type") == "string",
        f"custom tool schema corrupted: {custom}",
    )
    _assert(
        "type" not in custom,
        f"custom function tool should not carry a server `type`: {custom}",
    )

    print("OK — server tool round-trip verified end-to-end through the FastAPI app.")
    print(json.dumps(outbound["tools"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
