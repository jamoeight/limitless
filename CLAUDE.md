# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

This repo is **Limitless** — the engineering project. "Limitless" is the product name; the code is organized as two stacked Python packages (`timegraph` + `cortex`) that predate the rename. Keep using the existing names in code, imports, env vars (`CORTEX_*`, `TG_*`), CLIs (`cortex-serve`, `timegraph-mcp`, `timegraph init`), and the plugin name (`timegraph-cortex`) — those are stable contract surfaces. Call the project Limitless in READMEs, docs, and positioning.

Two stacked packages sharing one venv and one backend (Neo4j + Qdrant + LM Studio):

1. **`src/timegraph/`** — the original capability layer + MCP server. Implements bounded-1-LLM-call retrieval over a temporal property graph. README.md and `bench/` are about this. Console script: `timegraph-mcp` (stdio MCP server). See README for the architecture, benchmark results (MRCR 50/50, GraphWalks 100%, BEAM 54.6%), and the gap-pruning thesis.

2. **`src/cortex/`** — an HTTP proxy that wraps timegraph to make any frontier model effectively infinite-context. Exposes OpenAI-compatible `/v1/chat/completions` and Anthropic-compatible `/v1/messages`. Auto-ingests every message into the timegraph, retrieves relevant context per turn, injects a "recap" block into the system prompt. Console entry: `python -m cortex.server`. The build plan is in `~/.claude/plans/the-idea-of-sorted-eich.md`; MVPs 1–4 are shipped, MVPs 5–6 are deferred.

The relationship: cortex calls timegraph ops **in-process** (direct Python imports — no stdio MCP roundtrip for the proxy's own retrieval).

## Load-bearing invariants — do not break these

- **`judge_call_count ≤ 1` per query is structural, not aspirational.** It's enforced in `src/timegraph/llm/judge.py` (a `ValueError` if you hand it more than 8 conflict pairs) and propagated up `infer()`. Do not add retry loops, multi-stage refinement, or "let me think again" passes in the retrieval path. Stage-1 retrieval must do the filtering.
- **Tool-use atomicity in cortex virtualization** — `compute_atomic_groups()` in `src/cortex/virtualize.py` groups messages so that an assistant `tool_use_id` is never separated from its matching user `tool_result`. Splitting them returns a 400 from Anthropic. There's a fuzz test (`tests/cortex/test_virtualize.py::test_virtualize_never_splits_tool_use_pairs_fuzz`) that runs 40 random conversations through virtualize and asserts no orphans. Keep it passing.
- **System prompt is APPENDED to, never replaced.** Cortex's recap block goes after the original system prompt. Models tolerate large system context; they get suspicious of role-alternation gaps in the messages list.
- **Fail open in cortex.** If Neo4j / Qdrant / LM Studio are down, the proxy degrades to plain passthrough — never errors the user. Search for `# noqa: BLE001` to see the swallow-and-log pattern.

## The 5 MCP tools (deliberately lean — resist adding more)

`remember`, `add_fact`, `recall`, `query`, `attest`. The full schemas and behavior are in `src/timegraph/mcp_server.py`. Adding tools dilutes the surface; if you want richer retrieval, do it *inside* an existing tool or *inside* cortex's recall fn, not as a new MCP tool.

## Backends + ports

All local. The proxy and MCP server connect lazily on first call.

- Neo4j 5.24 Community on `bolt://localhost:7687` (HTTP browser at 7474)
- Qdrant 1.12 on `:6333` (REST) + `:6334` (gRPC, preferred for vector upsert)
- LM Studio on `:1234` with `qwen/qwen3.5-9b` (judge + extractor) and `text-embedding-nomic-embed-text-v1.5` (768D) loaded
- Cortex HTTP proxy on `127.0.0.1:8080` (when running)

`docker compose up -d` brings up Neo4j + Qdrant. LM Studio is a separate desktop app — load the models manually and **pin them ("keep loaded")** so they don't auto-unload mid-extract.

## Common commands

```bash
# Backends + schema (first-time setup)
docker compose up -d
.venv/Scripts/python.exe -m timegraph.storage.schema --apply
.venv/Scripts/python.exe -c "import asyncio; from timegraph.storage.qdrant_client import ensure_collections; asyncio.run(ensure_collections())"

# Run servers (foreground)
.venv/Scripts/python.exe -m timegraph.mcp_server        # MCP over stdio
.venv/Scripts/python.exe -m cortex.server               # HTTP proxy on :8080

# Tests
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/cortex/         # cortex (102+ tests)
.venv/Scripts/python.exe -m pytest tests/                               # timegraph
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/cortex/test_virtualize.py::test_virtualize_never_splits_tool_use_pairs_fuzz -v   # one test

# Smoke (these need backends + LM Studio up)
.venv/Scripts/python.exe scripts/smoke_mcp.py
.venv/Scripts/python.exe scripts/smoke_wave1.py    # graph ops only
.venv/Scripts/python.exe scripts/smoke_wave2.py    # + LLM extractor + embedder
.venv/Scripts/python.exe scripts/smoke_wave3.py    # + infer + fuse

# Benchmarks (long-running; see README for runtime estimates)
.venv/Scripts/python.exe bench/infer_scale.py --sizes 100,1000,10000,1000000
.venv/Scripts/python.exe bench/beam_subset/run.py
.venv/Scripts/python.exe bench/graphwalks/run.py --per-bucket 10 --baseline-max-chars 130000
.venv/Scripts/python.exe bench/mrcr/run.py --per-bucket 10 --baseline-max-chars 100000

# Run cortex with full behavior (ingest + virtualization)
CORTEX_DEFAULT_PROVIDER=openai \
  CORTEX_OPENAI_BASE_URL=http://127.0.0.1:1234 \
  CORTEX_ENABLE_AUTO_INGEST=true \
  CORTEX_ENABLE_VIRTUALIZATION=true \
  .venv/Scripts/python.exe -m cortex.server
```

Notes:
- The venv is at `.venv/` and shell is Git Bash on Windows (PowerShell also fine). Forward slashes in paths work.
- `PYTHONPATH=src` is needed for `tests/cortex/` because the cortex package wasn't yet picked up by the editable install in some scenarios — if `pip install -e .` was rerun after `src/cortex/` was added, you can drop it.
- The console script `timegraph-mcp.exe` is held open by any running MCP client (e.g., opencode) — kill those before `pip install -e .` if it fails with WinError 32.

## Code conventions worth knowing

- **Async everywhere.** Every I/O path is `async` (httpx, neo4j-async, qdrant-client gRPC). Sync helpers are reserved for pure CPU work (hashing, JSON shape).
- **Settings via `pydantic-settings`.** `timegraph.config.Settings` uses prefix `TG_`; `cortex.config.CortexSettings` uses `CORTEX_`. Both read from `.env` if present.
- **Type contracts via Pydantic.** All op I/O is a Pydantic model (`src/timegraph/types.py` and `src/cortex/canonical.py`). Wire formats translate to/from these — never let raw dicts leak past the translator boundary.
- **Test injection.** `cortex.server._build_app()` takes optional `registry`, `session_registry`, `recall_fn` parameters so tests can avoid Neo4j/Qdrant/LM Studio entirely. The corresponding `*_fn` arguments on `SessionRegistry` and on `virtualize()` accept stubs. See `tests/cortex/test_server_ingest.py` for the pattern.
- **Single-line comments only.** No multi-paragraph docstrings on functions; the README + this file carry the architecture. Module-level docstrings explain *why*, not *what*.
- **Date format in commits and tests.** Absolute ISO 8601 (`2026-05-22`), not relative.

## Where to look when you need to…

- **Add a new provider** — implement `cortex.providers.base.Provider`, write a `cortex.translate.{name}` module with the four translator functions, register in `_build_app`, add tests modeled on `tests/cortex/test_translate_openai.py` + `tests/cortex/test_four_corner_streaming.py`.
- **Change retrieval behavior** — `src/cortex/recall.py` (the bridge into timegraph ops) and `src/cortex/virtualize.py` (budget, recap assembly). The actual graph queries live in `src/timegraph/ops/{infer,graph_query}.py`.
- **Tune ingestion** — `src/cortex/ingest.py` for cortex's auto-ingest pipeline (hashing, secrets filter, fire-and-forget); `src/timegraph/ops/add_episode.py` for the actual write path (extractor → add_fact → embed → upsert).
- **Understand the bounded-1-call guarantee** — read `src/timegraph/ops/infer.py` end-to-end and `src/timegraph/llm/judge.py` (the ValueError at the top is the contract).
- **Wire a client to the proxy** — `scripts/smoke_cortex_passthrough.py` is the minimal example; `~/.config/opencode/opencode.json` (on dev machines) shows the opencode wiring via `@ai-sdk/openai-compatible`.

## Not built yet (deferred from the plan)

- **MVP-5**: production auth modes (BYO-key / tenant-key / hybrid), `X-Cortex-Degraded` headers, LM Studio as a first-class registered provider (today it's just OpenAI-compat with a different base URL).
- **MVP-6**: tool-aware ingest — per-file episodes for `read_file`/`write_file`/`bash` results so `recall("the auth module")` returns file contents directly instead of "we discussed auth in turn 47."
- Query augmentation (the small LLM call that resolves coreferences before recall — see `bench/mrcr/query_parser.py` for the pattern to copy).
- Anthropic prompt-caching on the system + recap block (it's stable across turns — easy latency win).
- Persistent (SQLite-backed) hash cache for ingest idempotency across proxy restarts.
