"""Route hook logging to stderr so stdout stays clean for Claude Code.

Claude Code parses every hook's stdout strictly as JSON (or as empty = no-op).
Any non-JSON byte on stdout kills the hook with "JSON validation failed",
which is what happened to `timegraph-hook-ingest` (Stop hook) on every fire:
`add_episode` calls `log.info("extracted", ...)` via structlog, and structlog's
default `PrintLoggerFactory` writes to `sys.stdout`. The log line landed in
the hook's stdout buffer, Claude Code tried to parse it, failed.

Solution: every hook calls `silence_to_stderr()` BEFORE importing anything
from `timegraph.ops` / `timegraph.llm` (which is where structlog loggers are
instantiated). Logs still go somewhere visible (stderr — Claude Code shows
them alongside the success/error line), just not into the JSON channel.

This module must not import any timegraph code, so it can run before the
env-var `setdefault` block at the top of each hook script.
"""

from __future__ import annotations

import logging
import sys


def silence_to_stderr() -> None:
    """Reconfigure structlog + stdlib logging to write to stderr, not stdout.

    Idempotent. Safe to call multiple times. Swallows configuration errors so
    a logging glitch never blocks a hook from running.
    """
    try:
        import structlog

        structlog.configure(
            logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        )
    except Exception:
        pass

    try:
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)
        handler = logging.StreamHandler(stream=sys.stderr)
        root.addHandler(handler)
        root.setLevel(logging.WARNING)
    except Exception:
        pass
