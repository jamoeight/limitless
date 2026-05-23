"""Dev-mode shim — delegates to the packaged `timegraph.hooks.session_start:main`.

The real implementation lives in `src/timegraph/hooks/session_start.py` so the
plugin manifest can reference the `timegraph-hook-session-start` console script
(installed by `pip install -e .` or `pipx install timegraph-mcp`).

This shim keeps repo-local `.claude/settings.json` wiring working in dev
mode without needing the console script on PATH.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from timegraph.hooks.session_start import main

if __name__ == "__main__":
    main()
