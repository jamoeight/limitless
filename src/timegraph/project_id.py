"""Per-project group_id derivation from the current working directory.

Claude Code spawns hooks and MCP servers with the project's cwd, so cwd is
the natural scope for memory. Slugifies the basename and suffixes a short
hash of the absolute path so two projects with the same basename in
different locations never collide.

Stable across runs: same path always produces the same group_id. Safe to
use as a Neo4j label / Qdrant payload key / sqlite row id (lowercase
alphanumerics + hyphens only).
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_MAX_BASENAME = 32
_HASH_LEN = 6

ENV_OVERRIDE = "TG_GROUP_ID"


def _slugify(s: str) -> str:
    s = s.lower().strip()
    s = _SLUG_RE.sub("-", s).strip("-")
    return s[:_MAX_BASENAME] or "project"


def derive_group_id(cwd: Path | str | None = None) -> str:
    """Return a stable, slug-safe group_id for the given (or current) directory.

    Respects the TG_GROUP_ID env var as an override — set by ops who want
    multiple worktrees of the same repo to share memory, or want a fixed
    name across machines.
    """
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        return override.strip()
    if cwd is None:
        cwd = Path.cwd()
    p = Path(cwd).resolve()
    basename = _slugify(p.name)
    h = hashlib.blake2b(str(p).encode("utf-8"), digest_size=_HASH_LEN).hexdigest()
    return f"{basename}-{h}"
