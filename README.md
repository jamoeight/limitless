# Limitless

> **Effectively infinite context for Claude Code — and anything Anthropic-compatible. Local. Self-hosted. The proof is on the wire.**

![200 real Claude Code turns, outbound payload virtualized in green vs un-virtualized in red, with the 50K-token threshold dashed in black](docs/200_turns.png)

200 real Claude Code turns through a local proxy, single resumed session. Red is the request that would have been sent to `api.anthropic.com` without Limitless. Green is what actually went on the wire. A 500-character paragraph planted at turn 5 was recalled **verbatim** when asked at turn 200 — by real Claude, with only the green payload in front of it.

---

## The number that matters

| | Without Limitless | With Limitless |
|---|---:|---:|
| Outbound input tokens at turn 200 | 115,240 | **16,960** |
| Effective session length cap | API context window | **none observed** |
| Recall of a 500-char paragraph planted at turn 5 | turn dropped to compaction | **verbatim, all 500 chars** |
| Turns >50K tokens out of 200 | grows unbounded | **0** |

Bench: 200 real Claude Code turns through `cortex.server` on `127.0.0.1:8080`, single resumed session, anchor planted at turn 5 (random per run, no false positives from prior bench memory), verbatim recall at turn 200 verified by string match. 26 min wall-clock. Raw data: [docs/bench_summary.json](docs/bench_summary.json) (single run, n=1 — reseed before quoting).

---

## What it does

A local HTTP proxy between any Anthropic-compatible client and `api.anthropic.com`. You set one env var, the proxy does three things:

1. **Auto-ingests every turn** into a temporal property graph (Neo4j) + vector store (Qdrant). Fact-level subject/predicate/object triples plus raw episode bodies. Persistent across sessions, per project directory.
2. **Rewrites the outbound `messages` list** before it leaves your machine. Last K turns kept verbatim. Older history replaced by a recap built from three retrieval paths: verbatim semantic recall over cold turns (the heavy hitter), literal-needle scan for anchor strings in the current query, and a summary fallback. The recap is appended to the system prompt.
3. **Forwards a small request upstream.** The model has no idea anything is different. Token cost per turn stays flat regardless of session length.

This is real virtualization — not RAG bolted on top of a growing payload. The wire bytes to `api.anthropic.com` actually shrink. The `X-Cortex-Outbound-Tokens` response header is the receipt.

---

## What it works with

Anything that respects `ANTHROPIC_BASE_URL`:

- **Claude Code** (the headline; auto-launches via the `timegraph-cortex` plugin)
- **Anthropic SDK** (Python, TypeScript) directly
- **Cursor, Zed, opencode, aider, Continue** — anything pointing at an Anthropic-format endpoint
- **Your own application code**

The upstream sees a standard `/v1/messages` request. Whatever model you call, calls normally.

---

## Install

Steps 1-3 are shell-agnostic. Steps 4-6 (the env vars + launch) have three syntaxes — pick the block for the shell you're running Claude Code from.

**Why `ANTHROPIC_BASE_URL` is load-bearing:** if you skip it, the plugin install still succeeds and the hooks still fire (recall + ingest), but Claude Code talks direct to `api.anthropic.com` and **no virtualization happens** — there's no error, just flat token growth. If you're debugging "why is my session still hitting context limits", check `echo $ANTHROPIC_BASE_URL` first.

```bash
# 1. Install the engine + CLIs (cortex.server, timegraph-mcp, plugin hooks).
#    Use --force if you already have a previous version installed; entry
#    points (cortex-serve, plugin hooks) only get refreshed on --force.
pipx install --force git+https://github.com/jamoeight/cortex-mcp.git

# 2. Bring up backends (Neo4j + Qdrant), apply schema, prefetch embedder (~270 MB).
timegraph init

# 3. Install the Claude Code plugin. It auto-launches the proxy on SessionStart.
#    Run these two inside Claude Code:
#      /plugin marketplace add jamoeight/cortex-mcp
#      /plugin install timegraph-cortex
```

### 4-6. Env vars + launch — bash / zsh (Linux, macOS, WSL, Git Bash)

```bash
# Point Claude Code at the proxy. MUST be set BEFORE you launch claude —
# plugins can't inject env vars into the already-running host.
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080

# Tell the proxy how to reach Anthropic. Pick ONE:
export ANTHROPIC_API_KEY=sk-ant-...            # use an API key (forwarded as-is), OR
export CORTEX_USE_CLAUDE_CLI_PROVIDER=true     # use your existing Claude Code OAuth
                                               # (cortex shells out to `claude -p` per
                                               # request; no API key needed, ~10-20s
                                               # subprocess overhead per turn)

# Launch.
claude
```

### 4-6. Env vars + launch — PowerShell (Windows)

```powershell
# Point Claude Code at the proxy.
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8080"

# Upstream auth — pick ONE:
$env:ANTHROPIC_API_KEY = "sk-ant-..."          # API key, OR
$env:CORTEX_USE_CLAUDE_CLI_PROVIDER = "true"   # use your Claude Code OAuth (no key needed)

# Launch from the SAME shell.
claude
```

To persist across PowerShell sessions instead of just this one:

```powershell
[Environment]::SetEnvironmentVariable("ANTHROPIC_BASE_URL", "http://127.0.0.1:8080", "User")
[Environment]::SetEnvironmentVariable("CORTEX_USE_CLAUDE_CLI_PROVIDER", "true", "User")
# Open a new shell after running these — they don't apply to the current shell.
```

### 4-6. Env vars + launch — cmd.exe (Windows)

```cmd
:: Point Claude Code at the proxy.
set ANTHROPIC_BASE_URL=http://127.0.0.1:8080

:: Upstream auth -- pick ONE:
set ANTHROPIC_API_KEY=sk-ant-...
set CORTEX_USE_CLAUDE_CLI_PROVIDER=true

:: Launch from the SAME shell.
claude
```

To persist across cmd sessions (writes to user environment):

```cmd
setx ANTHROPIC_BASE_URL http://127.0.0.1:8080
setx CORTEX_USE_CLAUDE_CLI_PROVIDER true
:: Open a new cmd window -- setx doesn't apply to the current one.
```

First launch triggers the plugin's `SessionStart` hook, which spawns `cortex-serve` on `127.0.0.1:8080` if it isn't already running (logs to `~/.timegraph/cortex.log`). From that point on:

- Every turn is auto-ingested into the temporal graph.
- Every outgoing `/v1/messages` is virtualized — last K turns kept verbatim, older history replaced by a recap drawn from semantic recall + literal-needle scan + summary fallback.
- The model sees a small request. Token cost per turn stays flat regardless of how long the session has been running.

Just talk. The session never has to end.

**Verify it's working:** `curl http://127.0.0.1:8080/health` should return `{"status":"ok"}`. After a few turns, every response from `api.anthropic.com` will carry `X-Cortex-Outbound-Tokens` — that's the size of the request cortex actually forwarded. Watch it stay flat as your session grows.

**Use with anything else** (Cursor, Zed, opencode, aider, the SDK): same `ANTHROPIC_BASE_URL` env var, same upstream-auth pick. Skip step 3 — the Claude Code plugin is only for the auto-launch convenience. You can also start the proxy manually: `cortex-serve`.

---

## What this isn't

- **Not zero-loss.** Cold turns get summarized to ~320 chars by default. Verbatim semantic recall reproduces the top-K most relevant cold turns whole, but if your query doesn't surface a turn, only its summary survives. Tune `verbatim_recall_k` if you need higher recall.
- **Not magic.** Needs Neo4j + Qdrant + an embedder running locally. `timegraph init` brings them up — plan for ~1 GB disk, similar RAM, brief CPU cost per turn for embedding.
- **Not hosted.** Self-hosted only today. Multi-tenant auth, hosted backends, key management — all on the roadmap.
- **Not a replacement for a real large context window** when you genuinely need every token of one giant document in the model's attention at once. For that, just send the document. Limitless is for *sessions* that accumulate to that size.
- **n=1 on the headline bench.** Single run, single anchor. Real, reproducible, but reseed before quoting in customer conversations or investor decks.

---

## Engineering

Limitless is two stacked layers in this repo, sharing one venv and one backend (Neo4j + Qdrant + an in-process fastembed embedder):

1. **`src/timegraph/`** — the temporal-graph retrieval engine. A property graph of facts (subject / predicate / object + valid-from / valid-to) plus an episode store of raw message bodies. Bounded-1-LLM-call retrieval is structural: `judge_call_count ≤ 1` per query is enforced by a `ValueError` in `src/timegraph/llm/judge.py`. Exposes 5 MCP tools — `remember`, `add_fact`, `recall`, `query`, `attest` — through the `timegraph-mcp` stdio server.
2. **`src/cortex/`** — the HTTP proxy. Anthropic-compatible `/v1/messages` and OpenAI-compatible `/v1/chat/completions`. Auto-ingests every turn into timegraph, virtualizes outbound `messages` lists, injects the recap into the system prompt. Calls timegraph ops **in-process** (no stdio MCP roundtrip for its own retrieval). Console entry: `cortex-serve` / `python -m cortex.server`.

The Claude Code plugin (`.claude-plugin/`, installed as `timegraph-cortex`) wires it all up with four hooks + an MCP server + three slash commands:

- **`UserPromptSubmit`** — embeds every prompt, runs two parallel semantic searches (fact graph + episode store), merges into `additionalContext`. ~2-5 s per turn.
- **`Stop`** — high-water-mark transcript scanner. Ingests every new user prompt and assistant response since the last fire. Cursor at `~/.timegraph/sessions/<session_id>.json` advances incrementally so partial timeouts leave the system in a correct state.
- **`PostToolUse`** — after every `Read`/`Edit`/`Write`/`Bash`/`Grep`/`Glob`/`WebFetch`/`WebSearch`, ingests the result as an episode keyed by `source=file:<path>` (or `bash:<hash>`, etc.). Embed-only, no extractor — this is what keeps a file you read in turn 3 recallable in turn 200.
- **`SessionStart`** — idempotently starts `cortex-serve` on `:8080` if `/health` isn't responding, primes new and resumed sessions with the top facts for this `cwd`. On `source=compact`, re-injects what was just summarized away, making compaction lossless.

Per-project isolation is automatic: each `cwd` gets its own `group_id`, no per-project config. `timegraph status` reports backend health; `timegraph stats` reports per-project episode/fact counts.

**Opus generation never goes through `claude -p`.** Claude Code keeps using its native OAuth path. The only `-p` calls are the bounded Haiku 4.5 judge inside timegraph ops (fact extraction during ingest, conflict resolution during `query`). Cents per session.

### Cross-benchmark calibration

The 200-turn Claude Code bench at the top is the application-level proof. The retrieval architecture under it has been calibrated on two academic long-context benchmarks at scales past every published native context window:

#### MRCR v2 8-needle, 256K → 10M tokens

![Opus + cortex MRCR v2 8-needle scaling: 256K / 1M / 5M / 10M tokens](results/opus_vs_cortex/hero_v2.png)

Same four context lengths Anthropic measures in the [claude-opus-4-6 announcement](https://www.anthropic.com/news/claude-opus-4-6). Vanilla Opus 4.7 (200K native context) scores 16% at 256K and the Anthropic API rejects every request at 1M+. **Opus 4.7 + Limitless scores 100% at every scale**, including past Opus 4.6's 1M-token native limit, by compressing up to ~39K cold messages into 7 verbatim turns + a ~6K-token recap.

| context | vanilla Opus 4.7 | Opus 4.7 + Limitless | reference (Anthropic) |
|--------:|-----------------:|---------------------:|----------------------:|
| 256K    | 16%              | **100%**             | —                     |
| 1M      | OVERFLOW         | **100%**             | Opus 4.6: **76%**     |
| 5M      | OVERFLOW         | **100%**             | beyond Opus 4.6 limit |
| 10M     | OVERFLOW         | **100%**             | beyond Opus 4.6 limit |

n=4, seed=42, lenient rubric. 256K/1M are real MRCR 8-needle rows from `openai/mrcr` (dataset max ≈ 625K tokens per row); 5M/10M are synthesized by stitching real rows with the gold needle preserved. Raw output: [results/opus_vs_cortex/mrcr_v3.json](results/opus_vs_cortex/mrcr_v3.json). Reproduce: `PYTHONPATH=. .venv/Scripts/python.exe bench/pilot_opus/run.py --targets 256000,1000000,5000000,10000000 --unit tokens --n-needles 8 --seed 42`.

#### RULER `niah_multikey_3`, 64K → 10M llama3 tokens

![Opus + cortex on RULER stays perfect through 10M tokens](results/opus_vs_cortex/ruler_hero.png)

Different benchmark, different rubric, different needle shape. **Limitless stays 100% from 64K to 10M llama3 tokens.** Vanilla Opus 4.7 matches through 512K, then the API rejects every request at 1M+. `verbatim_recall_k` is tuned per scale (K=16 default, K=200 at 512K-5M, K=2000 at 10M) — NIAH has many short candidates, so K must grow with cardinality. The recap budget stays bounded (~65K tokens at 10M) regardless of K.

n=1 per scale (preflight slice). 64K-1M are real RULER-llama3-1M samples; 2M/5M/10M are synthesized by stitching the 1M base row with additional RULER distractors. Methodology + raw data: [bench/pilot_opus/run_ruler.py](bench/pilot_opus/run_ruler.py), [results/opus_vs_cortex/ruler_all.json](results/opus_vs_cortex/ruler_all.json).

#### A local 9B + Limitless matches Opus on MRCR

![9B + cortex matches Opus on 30 MRCR rows](results/pilot_cortex/hero.png)

Same proxy, different model. **Qwen3.5-9B + Limitless hits 100% on 30 MRCR rows. Vanilla Opus 4.7 hits 73%.** The 9B catches the frontier on retrieval-shaped tasks because Limitless pre-locates the needles — the model only has to read the recap.

| arm                       | n  | MRCR lenient mean | perfect% | L-bucket perfect |
|---------------------------|----|-------------------|----------|------------------|
| Qwen3.5-9B (raw)          | 30 | 67%               | 67%      | 0%               |
| **Qwen3.5-9B + Limitless**| 30 | **100%**          | **100%** | **100%**         |
| Claude Opus 4.7           | 30 | 76%               | 73%      | 50%              |

n=30 single-seed pilot. Strict-rubric is 0 across both 9B arms (qwen3.5 prepends `\n\n`); lenient rubric is the headline. Full caveats: [bench/pilot_cortex/PAPER.md](bench/pilot_cortex/PAPER.md).

#### Timegraph standalone

The retrieval layer also stands on its own, exposed as an MCP server. Standalone results (full README preserved in [docs/timegraph.md](docs/timegraph.md)):

- **GraphWalks**: 100% on 50 tasks across 5 size buckets (4K-1.75M chars); baseline drops to 0% at 32K+ tokens.
- **BEAM** contradiction-resolution: 54.6% on all 194 cases, ~11× over the published Hindsight baseline.
- **Scale**: 1M facts retrievable with 1 LLM call at ~2.8s p95.

### How this relates to prior work

The long-context problem has three orthogonal solution families. Limitless sits in the third.

**1. Native long-context architectures.** Training models to attend over longer sequences directly — Anthropic's [Opus 4.6](https://www.anthropic.com/news/claude-opus-4-6) (1M native), Gemini 1.5 Pro, Jamba-1.5-large, Qwen2.5-14B-1M. [RULER](https://arxiv.org/abs/2404.06654) shows even the best of these degrade past effective context; the [LongMemEval ICLR 2025 paper](https://arxiv.org/abs/2410.10813) reports 30-60% drops on GPT-4o / Llama 3.1 / Phi-3 vs. oracle retrieval at ≥115K-token chat histories. Limitless inverts this: rather than ask the model to attend over the whole history, give it a small recap that fits in its high-attention zone.

**2. KV-cache compression.** Operates inside the model — StreamingLLM ([Xiao et al. 2023](https://arxiv.org/abs/2309.17453)), H2O ([Zhang et al. 2023](https://arxiv.org/abs/2306.14048)), SnapKV ([Li et al. 2024](https://arxiv.org/abs/2404.14469)), [RocketKV](https://arxiv.org/abs/2502.14051). Requires model access and doesn't help if the input itself exceeds the window. Limitless is API-side: works on closed frontier APIs over HTTP.

**3. Retrieval / memory systems.** Standard RAG plus elaborate memory architectures: MemGPT ([Packer et al. 2024](https://arxiv.org/abs/2310.08560)), Mem0, LongMem, A-MEM, MemMachine. Most do fact extraction or summarization — useful for cross-session memory but lossy for needle tasks where the answer is a verbatim string. Limitless is the **verbatim** variant: cold message-groups are inserted without summarization, preserving exact text. Recent RAG ↔ long-context work ([Yu et al. 2024](https://arxiv.org/abs/2410.05983); [Li et al. 2025](https://arxiv.org/abs/2501.01880); [Yang et al. 2025](https://arxiv.org/abs/2502.12462)) finds the two approaches complementary, not substitutable. Anthropic's own [memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool) is a file-based markdown store with just-in-time retrieval — closer to Limitless in shape than to vector-DB RAG, but storage rather than inference-time injection.

**What's specific to Limitless:**

- **Verbatim insertion** of cold message-groups, not summary. Critical for needle tasks where summarization destroys signal.
- **One small LLM call** for query reformulation (the only extra LLM call beyond upstream); then cosine ranking on embeddings.
- **Proxy architecture** — works on closed frontier APIs and local OpenAI-compat servers with no model access required.
- **Bounded recap budget** (~5-65K tokens) regardless of haystack size; the model always sees input inside its high-attention zone.

### Stack

- **Proxy**: Python 3.11, FastAPI, httpx, sse-starlette
- **Graph**: Neo4j 5.24 Community
- **Vectors**: Qdrant 1.12.4 (HNSW, 768D cosine)
- **LLM runtime**: Anthropic (via `claude -p` or API key) for Claude Code path; LM Studio (OpenAI-compat) for local path
- **Default models**: Qwen3.5-9B (local generation/extraction) + nomic-embed-text-v1.5 (embedder, 768D)
- **Tests**: 115 cortex tests + timegraph suite, all green

Hardware: Limitless itself is CPU-light. The Claude Code path needs the backends only (~1 GB RAM). The local-model path additionally needs ~24 GB VRAM for Qwen3.5-9B at 100K context + the embedder (tested on RTX 4090).

### Status

- ✅ MVPs 1-4 shipped: passthrough, auto-ingest, virtualization (verbatim recall + reformulation), OpenAI + Anthropic translators, Claude Code plugin
- ⚠️ MVP-5 deferred: production auth modes (BYO-key / tenant-key / hybrid), `X-Cortex-Degraded` SSE channel, hosted backends
- ⚠️ MVP-6 deferred: tool-aware ingest with chunk-level retrieval inside file contents

### Honest scope

Limitless's claim is **effectively unlimited context for retrieval-shaped tasks** via inline-verbatim recall. MRCR, RULER, and the 200-turn Claude Code bench are the cleanest demonstrations: content needs to come back exactly as it appeared. For reasoning, multi-hop inference, or summarization, the technique is necessary but not sufficient — the upstream model still has to do the reasoning over the recap.

All headline pilots are single-seed (42). Cortex hits 100% lenient at the sizes shown so downside is bounded, but sampling variance is real. Reseed before quoting in customer conversations or investor decks. Per-pilot caveat docs: [bench/pilot_cortex/PAPER.md](bench/pilot_cortex/PAPER.md), and the docstrings on `bench/pilot_opus/run.py` and `bench/pilot_opus/run_ruler.py`.

---

If you're a design partner, an infra researcher, or you build something that hits Anthropic's API and run out of context window in the middle of a real session — open a conversation.

## License

Apache-2.0. Issues + PRs welcome.
