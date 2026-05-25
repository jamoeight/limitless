"""Cortex runtime configuration.

Reuses the `TG_` prefix is intentionally avoided — Cortex settings use the
`CORTEX_` prefix so the two configs can co-exist in one `.env` file without
collisions. The timegraph layer continues to read its own `Settings()`; the
proxy reads `CortexSettings()`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CortexSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="CORTEX_", extra="ignore")

    # --- Server ---
    host: str = Field("127.0.0.1")
    port: int = Field(8080)
    log_level: str = Field("info")

    # --- Upstream providers ---
    anthropic_base_url: str = Field("https://api.anthropic.com")
    anthropic_version: str = Field("2023-06-01")
    openai_base_url: str = Field("https://api.openai.com")
    # Routes unknown model names to one of the registered providers. Useful
    # when the upstream is LM Studio or another OpenAI-compatible local model
    # server — set CORTEX_DEFAULT_PROVIDER=openai and CORTEX_OPENAI_BASE_URL
    # to the local /v1.
    default_provider: Literal["anthropic", "openai"] = Field("anthropic")

    upstream_timeout_s: float = Field(300.0, description="Long generations need headroom")
    upstream_connect_timeout_s: float = Field(10.0)

    # Deprecated as of the OAuth-aware AnthropicProvider: `AnthropicProvider`
    # now auto-detects `sk-ant-oat...` (Claude OAuth) vs `sk-ant-api...` (classic
    # API key) per request and switches between `Authorization: Bearer` +
    # `anthropic-beta: oauth-2025-04-20` and `x-api-key` accordingly. The old
    # behavior of swapping to the `claude -p` subprocess provider stripped
    # `req.tools` and made the proxy unusable for agentic callers, so this flag
    # is now treated as a no-op (the OAuth path is always available; whichever
    # auth shape the client sends wins). Kept for env-var compatibility.
    use_claude_cli_provider: bool = Field(False)

    # --- Auth mode ---
    # byo_key: client sends provider key, we forward verbatim; group_id derived from key hash.
    # tenant_key: client sends a Cortex-issued key; we look up provider keys server-side.
    # hybrid: try tenant_key lookup first; fall back to byo_key passthrough.
    auth_mode: Literal["byo_key", "tenant_key", "hybrid"] = Field("byo_key")

    # When True, the proxy includes its own diagnostic events as a sidechannel
    # (`event: cortex.notice` on Anthropic, synthetic deltas on OpenAI).
    emit_cortex_notices: bool = Field(True)

    # --- Virtualization knobs (used in MVP-3+) ---
    last_k_spans: int = Field(4, description="Most recent K message spans kept verbatim")
    verbatim_budget_pct: float = Field(0.25)
    recall_budget_pct: float = Field(0.50)
    pinned_budget_pct: float = Field(0.15)
    speculative_budget_pct: float = Field(0.10)
    safety_margin_tokens: int = Field(1024, description="Headroom subtracted from upstream context limit")
    # Messages-only budget: cortex virtualizes when user/assistant messages
    # exceed this many tokens (chars/4 estimate). Tools, system prompt, and
    # max_tokens are NOT charged against this budget — they live in
    # Anthropic's cached prefix and shouldn't trigger compaction. With the
    # default 50_000 from the SessionStart hook, a Claude Code session
    # virtualizes once its message history alone crosses ~50k tokens,
    # regardless of how many tools/skills/plugins are loaded.
    #
    # When None, virtualize.context_limit_for() chooses based on the model
    # family (currently ~200k for Sonnet/Opus, smaller for Haiku).
    upstream_context_limit: int | None = Field(
        None,
        description="Messages-only budget (chars/4 estimate) before virtualize engages — tools/system/max_tokens excluded",
    )
    cold_summary_max_chars_per_msg: int = Field(
        2000,
        description="Per-message char cap when summarizing cold history. Higher preserves more verbatim content (needed for retrieval tasks); lower yields more compact recaps.",
    )
    verbatim_recall_k: int = Field(
        16,
        description="Top-K cold atomic-groups injected verbatim into the recap. Higher = more chance of catching the needle on multi-needle retrieval tasks; lower = tighter recap budget.",
    )
    verbatim_recall_budget_pct: float = Field(
        0.70,
        description="Fraction of the post-verbatim-window recap budget reserved for verbatim-retrieved cold groups. Cold-summary takes a slice of the remainder; graph recall takes the rest.",
    )
    enable_verbatim_recall: bool = Field(
        True,
        description="Enable inline embedding-based verbatim recall of cold messages. Falls back to cold-summary on any failure.",
    )
    enable_query_reformulation: bool = Field(
        True,
        description="When True, makes one extra LM Studio call to rewrite the user's query as a topical retrieval phrase before embedding. Costs ~5s; doubles recall on meta-queries like 'Prepend X to the 2nd Y about Z'.",
    )

    # --- Ingest (used in MVP-2+) ---
    ingest_max_concurrent: int = Field(4, description="Per-session inflight ingest cap")
    hash_cache_max_entries: int = Field(10_000, description="LRU bound for span-hash → episode mapping")
    ingest_min_chars: int = Field(20, description="Skip ingest for spans smaller than this")

    # --- Feature flags ---
    # MVP-2 turns on auto-ingest by default — the proxy's defining behavior.
    # MVP-3 will flip virtualization on; MVP-6 turns on tool-aware ingest.
    enable_auto_ingest: bool = Field(True)
    enable_virtualization: bool = Field(False)
    enable_tool_aware_ingest: bool = Field(False)


def get_cortex_settings() -> CortexSettings:
    """Cache-friendly settings accessor."""
    return CortexSettings()
