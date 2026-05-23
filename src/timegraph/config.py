"""Runtime configuration.

Loads from env vars (or .env at repo root). Defaults match the local Phase 0
test environment: Neo4j + Qdrant in Docker, Qwopus on :8081, Qwen3-7B on :8082.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="TG_", extra="ignore")

    # --- LLM endpoints (LM Studio OpenAI-compat /v1) ---
    # LM Studio serves one or more loaded models on a single port; the `model`
    # field in each request selects which loaded model to route to. For Phase 0
    # we may run with only Qwopus loaded; the extractor reuses the same URL
    # once Qwen3-7B-Instruct is also loaded in LM Studio.
    lm_studio_url: str = Field("http://127.0.0.1:1234/v1", description="LM Studio OpenAI-compat base URL")

    judge_url: str = Field("http://127.0.0.1:1234/v1", description="Routes to LM Studio")
    judge_model: str = Field(
        "qwen/qwen3.5-9b",
        description="Model identifier as registered in LM Studio (check /v1/models). "
                    "Locked to qwen3.5-9b after the BEAM 20-case spike: 60% vs Qwopus 55%, "
                    "8.8x faster — accuracy gap within sample noise.",
    )
    judge_thinking_budget: int = Field(
        512,
        description="Soft target for tokens inside the schema's 'thinking' field; "
                    "Qwopus reasoning lives there since strict JSON precludes raw <think> blocks",
    )
    judge_max_tokens: int = Field(1536, description="Includes thinking field content + structured tail")
    judge_timeout_s: float = Field(60.0, description="Generous — Qwopus on 4090 ~30-50 tok/s with reasoning")

    # --- Judge backend selection ---
    # "lm_studio" (default): HTTP POST to LM Studio /v1/chat/completions (Qwen3.5-9B).
    # "claude_cli": shell out to `claude -p` (uses caller's OAuth; --bare requires API key).
    judge_backend: str = Field("lm_studio", description="lm_studio | claude_cli")
    judge_claude_cli_path: str = Field("claude", description="claude CLI binary name/path on PATH")
    judge_claude_model: str = Field(
        "haiku",
        description="Model alias passed via --model. 'haiku' = claude-haiku-4-5; 'sonnet' = claude-sonnet-4-6.",
    )
    judge_claude_budget_usd: float = Field(
        0.50,
        description="Per-call --max-budget-usd cap. ~$0.03/call after cache warms; first call ~$0.05.",
    )
    judge_claude_timeout_s: float = Field(
        90.0,
        description="Subprocess wall-clock. claude -p over OAuth runs an internal agent loop (~15s typical).",
    )

    extractor_url: str = Field("http://127.0.0.1:1234/v1", description="Same LM Studio instance (multi-model)")
    extractor_model: str = Field(
        "qwen/qwen3.5-9b",
        description="Same model as judge — single loaded model handles both call sites for now.",
    )
    extractor_max_tokens: int = Field(1024)
    extractor_timeout_s: float = Field(30.0)

    # If only Qwopus is loaded in LM Studio (Phase 0 day-1 state), set this True
    # to route extraction calls to the judge model as well. Slower, but unblocks
    # Phase 0 evals before the second model is downloaded/loaded.
    use_judge_for_extraction: bool = Field(False)

    # Extractor backend selection — mirrors judge_backend. Plugin installs
    # without LM Studio set this to "claude_cli" so the Stop hook can extract
    # facts via the caller's OAuth session instead of a phantom LM Studio.
    extractor_backend: str = Field("lm_studio", description="lm_studio | claude_cli")
    extractor_claude_model: str = Field(
        "haiku",
        description="Model alias for --model when extractor_backend=claude_cli.",
    )
    extractor_claude_budget_usd: float = Field(0.20)
    extractor_claude_timeout_s: float = Field(60.0)

    # --- Storage ---
    neo4j_uri: str = Field("bolt://localhost:7687")
    neo4j_user: str = Field("neo4j")
    neo4j_password: str = Field("dev_password_change_me")
    neo4j_database: str = Field("neo4j")

    qdrant_url: str = Field("http://localhost:6333")
    qdrant_episodes_collection: str = Field("episodes")
    qdrant_facts_collection: str = Field("facts")

    sqlite_provenance_path: str = Field("data/provenance.db")

    # --- Embedder ---
    # Two backends. Default `fastembed` runs the model in-process via ONNX
    # (no extra server, no API key, no LM Studio) — this is what the plugin
    # install path uses. `openai_compat` keeps the HTTP-to-LM-Studio path
    # for users who already have LM Studio running.
    #
    # Model + dim are deliberately UNCHANGED from the pre-plugin defaults:
    # nomic-embed-text-v1.5 at 768D is what produced the MRCR / GraphWalks /
    # BEAM benchmark numbers in the README. fastembed supports the same
    # model identifier, so the swap is HTTP → in-process with byte-identical
    # vectors. Existing Qdrant collections (768D) keep working — no migration.
    embedder_backend: str = Field(
        "fastembed",
        description="fastembed (in-process ONNX, default) | openai_compat (HTTP /v1/embeddings)",
    )
    embedder_url: str = Field(
        "http://127.0.0.1:1234/v1",
        description="OpenAI-compat /embeddings base URL — only used when backend=openai_compat",
    )
    embedder_model: str = Field(
        "nomic-ai/nomic-embed-text-v1.5",
        description="fastembed: HuggingFace ID. openai_compat: model name as registered by the server "
                    "(e.g. 'text-embedding-nomic-embed-text-v1.5' for LM Studio).",
    )
    embedder_dim: int = Field(
        768,
        description="MUST match Qdrant collections. nomic-embed-v1.5=768 (default, benchmarked).",
    )
    embedder_timeout_s: float = Field(30.0)
    embedder_batch_size: int = Field(32)

    # --- Attest classifier ---
    classifier_path: str = Field("models/minilm_attest_classifier")
    classifier_threshold: float = Field(0.5)

    # --- Op defaults ---
    default_tier_filter: list[str] = Field(["T1", "T2"])
    default_recall_k: int = Field(4)
    default_recall_budget_tokens: int = Field(512)
    active_context_budget_tokens: int = Field(2000, description="Hard cap per B.4-v2 spec")

    # --- Background jobs ---
    consolidate_interval_minutes: int = Field(60)
    superseded_cache_window_hours: int = Field(24)


def get_settings() -> Settings:
    """Cache-friendly settings accessor."""
    return Settings()
