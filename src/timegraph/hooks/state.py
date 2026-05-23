"""Per-session state for the Stop hook's high-water-mark transcript scanner.

Each Claude Code session writes its transcript as an append-only JSONL file.
The Stop hook fires after every assistant turn — we track the last byte offset
we ingested so we only process *new* lines on subsequent fires, never re-ingesting
content that already lives in the timegraph.

Layout: `~/.timegraph/sessions/<session_id>.json` with `{"last_offset": int,
"updated_at": iso8601}`. Writes are atomic (write to .tmp then rename) so a
crash mid-write cannot leave a half-written file.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _state_dir() -> Path:
    override = os.environ.get("TG_HOOK_STATE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".timegraph" / "sessions"


def _state_path(session_id: str) -> Path:
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_") or "default"
    return _state_dir() / f"{safe}.json"


def read_offset(session_id: str) -> int:
    """Return the byte offset already ingested for this session (0 if none)."""
    p = _state_path(session_id)
    if not p.exists():
        return 0
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        v = obj.get("last_offset", 0)
        return int(v) if isinstance(v, (int, float, str)) else 0
    except Exception:
        return 0


def write_offset(session_id: str, offset: int) -> None:
    """Atomically persist the new offset. Best-effort: failures are swallowed."""
    p = _state_path(session_id)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "last_offset": int(offset),
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        with tempfile.NamedTemporaryFile(
            "w",
            dir=str(p.parent),
            delete=False,
            encoding="utf-8",
            suffix=".tmp",
        ) as tmp:
            json.dump(payload, tmp)
            tmp_path = tmp.name
        os.replace(tmp_path, p)
    except Exception:
        return
