"""Pick a sensible default judge/extractor backend at hook-launch time.

Resolution order, mirroring `cortex.providers.anthropic`:

  1. Honor any pre-set `TG_*_BACKEND` env var verbatim (operator override
     always wins — e.g. dev with LM Studio loaded).
  2. If ANTHROPIC_API_KEY is set OR `~/.claude/.credentials.json` holds a
     usable OAuth token, default to `anthropic_api` and pin the model to
     `haiku`. This is the plugin-install happy path: Claude Code's own
     OAuth session powers the judge/extractor, no LM Studio needed, no
     `claude -p` agent loop per call.
  3. Otherwise fall back to `lm_studio` so dev installs with LM Studio
     loaded keep working.

`claude_cli` is no longer a hook default — the `anthropic_api` backend
gives the same Haiku 4.5 call without the subprocess + Claude Code
agent-loop overhead. Hooks can still opt into it manually.
"""

from __future__ import annotations

import os

from timegraph.llm.anthropic_client import anthropic_credentials_available


def apply_hook_backend_defaults() -> None:
    """Idempotent. Safe to call from every hook entry point."""
    has_creds = anthropic_credentials_available()
    backend = "anthropic_api" if has_creds else "lm_studio"

    os.environ.setdefault("TG_JUDGE_BACKEND", backend)
    os.environ.setdefault("TG_EXTRACTOR_BACKEND", backend)
    if backend == "anthropic_api":
        os.environ.setdefault("TG_JUDGE_ANTHROPIC_MODEL", "haiku")
        os.environ.setdefault("TG_EXTRACTOR_ANTHROPIC_MODEL", "haiku")
    # Preserve the legacy claude_cli aliases too — if anything downstream
    # routes through them, haiku stays the chosen model.
    os.environ.setdefault("TG_JUDGE_CLAUDE_MODEL", "haiku")
    os.environ.setdefault("TG_EXTRACTOR_CLAUDE_MODEL", "haiku")
