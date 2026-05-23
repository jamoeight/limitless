"""Dev-mode shim — delegates to the packaged `timegraph.hooks.ingest:main`.

The real implementation now lives in `src/timegraph/hooks/ingest.py` so the
plugin manifest can reference the `timegraph-hook-ingest` console script
(installed by `pip install -e .` or `pipx install timegraph-mcp`).

This shim is kept so existing dev-mode wiring in `.claude/settings.json`
(which points at `scripts/hook_ingest.py`) keeps working without a re-wire.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from timegraph.hooks.ingest import main

if __name__ == "__main__":
    main()
