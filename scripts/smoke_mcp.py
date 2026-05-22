"""Smoke-test the MCP server end-to-end via stdio.

Spawns `python -m timegraph.mcp_server`, performs the MCP handshake, lists
tools, then runs a remember → recall round-trip to confirm wiring works.
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@asynccontextmanager
async def open_session():
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "timegraph.mcp_server"],
        env=None,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def main() -> int:
    print("==> spawn + handshake", flush=True)
    async with open_session() as s:
        print("    ok", flush=True)

        print("==> list_tools", flush=True)
        tools = await s.list_tools()
        names = [t.name for t in tools.tools]
        print(f"    {names}", flush=True)
        expected = {"remember", "add_fact", "recall", "query", "attest"}
        missing = expected - set(names)
        if missing:
            print(f"FAIL: missing tools: {missing}")
            return 1

        print("==> remember", flush=True)
        rem = await s.call_tool(
            "remember",
            {
                "content": "alice moved from boston to seattle in march 2026.",
                "source": "smoke_mcp",
                "group_id": "mcp_smoke",
            },
        )
        rem_text = rem.content[0].text if rem.content else ""
        print(f"    {rem_text[:300]}", flush=True)
        rem_obj = json.loads(rem_text)
        ep_id = rem_obj.get("episode_id")
        n_facts = len(rem_obj.get("extracted_facts", []))
        print(f"    episode_id={ep_id}  extracted_facts={n_facts}", flush=True)
        if not ep_id:
            print("FAIL: no episode_id returned")
            return 1

        print("==> recall", flush=True)
        rec = await s.call_tool(
            "recall",
            {"query": "where does alice live", "k": 4, "group_id": "mcp_smoke"},
        )
        rec_text = rec.content[0].text if rec.content else ""
        print(f"    {rec_text[:500]}", flush=True)
        rec_obj = json.loads(rec_text)
        n_hits = len(rec_obj.get("results", []))
        print(f"    results={n_hits}  tokens_used={rec_obj.get('tokens_used')}",
              flush=True)
        if n_hits == 0:
            print("FAIL: recall returned no hits")
            return 1

        print("==> query (infer-mode)", flush=True)
        q = await s.call_tool(
            "query",
            {"question": "where does alice live now?", "group_id": "mcp_smoke"},
        )
        q_text = q.content[0].text if q.content else ""
        q_obj = json.loads(q_text)
        print(f"    judge_call_count={q_obj.get('judge_call_count')}  "
              f"answer_facts={len(q_obj.get('answer_facts', []))}  "
              f"resolution={q_obj.get('resolution')}", flush=True)

    print()
    print("ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
