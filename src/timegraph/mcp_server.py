"""MCP server entrypoint.

Registers all B.4-v2 + B.2-v2 ops with FastMCP. Phase 1 wires the 9 B.4-v2 ops;
Phase 2 adds B.2-v2's 5 additional ops as middleware.

Run via: `python -m timegraph.mcp_server` OR `timegraph-mcp` (console script).
"""

from __future__ import annotations

import asyncio
import sys

import structlog

log = structlog.get_logger(__name__)


def main() -> None:
    """Console-script entrypoint.

    Phase 0: this is a stub. Phase 1 will wire ops via the official `mcp` SDK
    server primitives (Server / @server.list_tools / @server.call_tool).
    """
    print("=" * 60)
    print("timegraph-mcp — Phase 0 (skeleton)")
    print("=" * 60)
    print()
    print("This is a skeleton. The MCP server is not yet wired.")
    print("Phase 1 will register the 9 B.4-v2 ops.")
    print("Phase 2 will add 5 B.2-v2 ops as middleware.")
    print()
    print("To proceed: complete Phase 0 evals, then start Phase 1 implementation.")
    sys.exit(0)


if __name__ == "__main__":
    main()
