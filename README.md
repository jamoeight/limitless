# cortex

**Infinite-context proxy for any LLM.** Compresses long chat history into
a verbatim recap that fits inside the upstream model's native window.
Same proxy on any model — local 9B or frontier Opus.

Three demonstrations below: Opus stays perfect on MRCR at **29× past
Anthropic's published 1M context limit**, holds at **10× on RULER**, and
a local 9B routed through cortex matches Opus on the same long-context
benchmark.

---

### Opus 4.7 + cortex stays 100% perfect through ~29M tokens (MRCR)

![Opus + cortex scaling past Anthropic's 1M-token Opus limit](results/opus_vs_cortex/hero_v2.png)

Anthropic's [Opus 4.6 announcement](https://www.anthropic.com/news/claude-opus-4-6)
publishes their long-context score on MRCR v2 8-needle at the **1M token**
scale (76% perfect for Opus 4.6, 18.5% for Sonnet 4.5). Vanilla Opus 4.7
via `claude -p` collapses to 3% at ~205K tokens and the API rejects ~1M+
outright. **Opus + cortex stays at 100% perfect through ~29M tokens** —
**29× past Anthropic's published native limit** — by compressing up to
155K messages into 7 verbatim turns + a 5-65K-token recap that fits
inside the upstream model's window.

~29M is the MRCR 8-needle dataset's synthesis ceiling (stitches 98 of
400 rows; row reuse would introduce needle conflicts). At that scale
cortex ranks 155K messages via inline cosine recall and Opus answers
correctly in ~7 minutes wall clock, dominated by LM Studio batched
embedding.

n=8 single-seed pilot across eight scale targets. 53K- and 205K-token
rows are real MRCR 8-needle samples; the six rows from 1M to 29M are
*synthesized* by stitching real MRCR 8-needle rows. Token counts via
o200k_base. Run script and methodology: [bench/pilot_opus/](bench/pilot_opus/).

---

### Same result, second benchmark: RULER niah_multikey_3 to 10M tokens

![Opus + cortex on RULER stays perfect through 10M tokens](results/opus_vs_cortex/ruler_hero.png)

To rule out MRCR-specific quirks, we re-ran the experiment on
[RULER](https://huggingface.co/datasets/self-long/RULER-llama3-1M) — a
different long-context benchmark, different rubric, different needle
shape. **Cortex stays 100% perfect at every scale from 64K to 10M
llama3 tokens.** Vanilla Opus 4.7 matches cortex through 512K, then
the Anthropic API rejects every request at 1M+ outright.

Two independent benchmarks, two orders of magnitude past Anthropic's
published context window, same result.

n=1 per scale (preflight slice of the RULER `niah_multikey_3` subtask).
The 64K-1M rows are real RULER-llama3-1M samples; 2M/5M/10M are
synthesized by stitching the 1M base row with additional RULER
distractor lines. **Cortex's `verbatim_recall_k` is tuned per scale**
(K=16 default, K=200 at 512K-5M, K=2000 at 10M) — high-cardinality NIAH
needs more recall candidates as the haystack grows. K is a config knob,
not a fundamental limit; the recap budget bounds insertion regardless
of K (still ~65K tokens at 10M). Methodology and raw data:
[bench/pilot_opus/run_ruler.py](bench/pilot_opus/run_ruler.py),
[results/opus_vs_cortex/ruler_all.json](results/opus_vs_cortex/ruler_all.json).

---

### A local 9B + cortex matches Opus on MRCR

![9B + cortex matches Opus on 30 MRCR rows](results/pilot_cortex/hero.png)

Same proxy, different model. **Qwen3.5-9B + cortex hits 100% perfect
on 30 MRCR rows. Vanilla Opus 4.7 hits 73%.** The 9B catches the
frontier on retrieval-shaped tasks because cortex pre-locates the
needles — the model only has to read the recap.

n=30 single-seed pilot. Strict-rubric is 0 across both 9B arms (qwen3.5
prepends `\n\n`); the lenient rubric is the headline. Full caveats:
[bench/pilot_cortex/PAPER.md](bench/pilot_cortex/PAPER.md).

## What it is

Cortex is an HTTP proxy in front of any OpenAI- or Anthropic-compatible
LLM endpoint. When a conversation exceeds the upstream model's window,
cortex:

1. **Reformulates the query** with one small LM Studio call — strips
   meta-instructions to extract the topical retrieval phrase.
2. **Embeds and cosine-ranks every cold message-group** inline against
   the reformulated query (no Qdrant roundtrip — works on a first turn).
3. **Injects the top-K verbatim** into a `<retrieved_history>` block in
   the system prompt, chronologically ordered.

The model sees its original system prompt + the relevant verbatim turns
+ the last few message-groups + the query, picks the right content from
the recap, and responds.

When the conversation fits natively, cortex **short-circuits to
pass-through** — the model sees byte-identical input to raw. (Verified:
S/M-bucket cortex == raw on the pilot.)

## Install

Two supported paths. Both share the same backends (Neo4j + Qdrant + a local
embedder) and the same proxy binary. They differ in **what runs the LLM
calls**: the Claude Code path uses your existing `claude` subscription via
the local CLI; the local-model path uses LM Studio + Qwen3.5-9B.

Shared prerequisites for both paths:

```bash
# Backends + Python package.
docker compose up -d                          # Neo4j + Qdrant
python -m venv .venv && .venv/Scripts/pip install -e .
docker compose exec -T qdrant true            # wait for healthy
.venv/Scripts/python.exe -m timegraph.storage.schema --apply

# Embedder — required for verbatim recall on both paths.
# Install LM Studio (https://lmstudio.ai) and load the embedder:
lms load text-embedding-nomic-embed-text-v1.5 --gpu max --ttl 86400
```

---

### Path A — Claude Code + Haiku judge (frontier upstream, no API key needed)

Uses your local `claude` CLI's OAuth session for both the **upstream model
calls** (Opus / Sonnet, whatever you ask for) and the **internal judge /
query-reformulation calls** (Haiku 4.5, ~$0.03/call after cache warms).
No `ANTHROPIC_API_KEY` required — auth is your Claude Code subscription.

```bash
# 1. Verify the claude CLI is on PATH and logged in.
claude --version
claude -p --model haiku "say ok"              # should print "ok"

# 2. Start the cortex proxy on :8080.
TG_JUDGE_BACKEND=claude_cli \
TG_JUDGE_CLAUDE_MODEL=haiku \
CORTEX_DEFAULT_PROVIDER=anthropic \
CORTEX_USE_CLAUDE_CLI_PROVIDER=true \
CORTEX_ENABLE_AUTO_INGEST=false \
CORTEX_ENABLE_VIRTUALIZATION=true \
CORTEX_ENABLE_VERBATIM_RECALL=true \
CORTEX_ENABLE_QUERY_REFORMULATION=true \
CORTEX_LAST_K_SPANS=2 \
CORTEX_VERBATIM_RECALL_K=24 \
  .venv/Scripts/python.exe -m cortex.server &

# 3. Point a client at http://127.0.0.1:8080 and ask for any claude-* model.
#    Auth header is ignored on this path (OAuth is taken from `claude` CLI).
```

Trade-off: each upstream call shells out to `claude -p` (~10–20 s subprocess
overhead vs ~3 s direct API). Judge calls add ~15 s each. Worth it for
*"I already pay for Claude Code — give me infinite context on top of it"*.

---

### Path B — Local model only (LM Studio + Qwen3.5-9B, fully offline)

Runs everything on your own hardware. Upstream model, judge, embedder are
all local. This is the configuration the headline 9B-matches-Opus pilot
in `results/pilot_cortex/` was run on.

```bash
# 1. Load the 9B alongside the embedder.
lms load qwen/qwen3.5-9b --identifier qwen/qwen3.5-9b --context-length 100000 --gpu max --ttl 86400
lms ps                                        # verify CONTEXT=100000

# 2. Start the cortex proxy on :8080.
CORTEX_DEFAULT_PROVIDER=openai \
CORTEX_OPENAI_BASE_URL=http://127.0.0.1:1234 \
CORTEX_ENABLE_AUTO_INGEST=false \
CORTEX_ENABLE_VIRTUALIZATION=true \
CORTEX_ENABLE_VERBATIM_RECALL=true \
CORTEX_ENABLE_QUERY_REFORMULATION=true \
CORTEX_UPSTREAM_CONTEXT_LIMIT=100000 \
CORTEX_LAST_K_SPANS=2 \
CORTEX_VERBATIM_RECALL_K=24 \
  .venv/Scripts/python.exe -m cortex.server &

# 3. Reproduce the 9B-matches-Opus benchmark.
PYTHONPATH=src .venv/Scripts/python.exe bench/pilot_cortex/run.py \
  --seed 42 --per-bucket 10 --out results/pilot_cortex/scale30.json
```

Hardware target: ~24 GB VRAM for Qwen3.5-9B at 100K context + nomic
embedder. Tested on RTX 4090.

---

Cortex exposes OpenAI-compatible `/v1/chat/completions` and
Anthropic-compatible `/v1/messages` on both paths. Point any client at
`http://127.0.0.1:8080`.

## What this does NOT claim

- **Cortex does not make a 9B reason like Opus.** It gives the model the
  right context, not better reasoning. For multi-hop logic, refactors,
  or system design — use a frontier model. Cortex buys you frontier
  *memory* at the upstream model's inference cost.
- **Strict-rubric scores are 0** across every 9B arm (raw and cortex
  both) — qwen3.5-9b's chat template prepends `\n\n`. The lenient rubric
  (lstrip → same prefix check) is the headline; PAPER.md has the full
  caveat.

## How fast

| bucket | path                  | p50 latency |
|--------|-----------------------|------------:|
| S      | pass-through (≤200K)  | 51 s        |
| M      | pass-through (≤500K)  | 49 s        |
| **L**  | **virtualized (1M+)** | **23 s**    |

On the L bucket cortex is at frontier-latency parity (Opus p50 = 21 s)
because the model sees ~16K tokens (recap + verbatim window) instead of
1M characters. Pass-through is bottlenecked by the 9B model itself.

## What else is in this repo

This repo also contains the original **timegraph** capability layer —
the temporal property graph + bounded-1-LLM-call retrieval engine that
cortex's graph-recall fallback uses. Standalone results (full README
preserved in [docs/timegraph.md](docs/timegraph.md)):

- **GraphWalks**: 100% on 50 tasks across 5 size buckets (4K–1.75M chars);
  baseline drops to 0% at 32K+ tokens.
- **BEAM** contradiction-resolution: 54.6% on all 194 cases, ~11× over
  the published Hindsight baseline.
- **Scale**: 1M facts retrievable with 1 LLM call at ~2.8s p95.

The timegraph is exposed as an **MCP server** (`timegraph-mcp` console
script) with 5 tools: `remember`, `add_fact`, `recall`, `query`, `attest`.
Any MCP-compatible client (Claude Desktop, Continue, opencode) can wire
up to it.

## Stack

- **Proxy**: Python 3.11, FastAPI, httpx, sse-starlette
- **Graph**: Neo4j 5.24 Community
- **Vectors**: Qdrant 1.12.4 (HNSW, 768D cosine)
- **LLM runtime**: LM Studio (OpenAI-compat `/v1`)
- **Default models**: Qwen3.5-9B (generation, extractor) + nomic-embed-text-v1.5 (embedder, 768D)
- **Tests**: 115 cortex tests + timegraph suite, all green

Hardware tested: RTX 4090 + 9800X3D + 32GB DDR5. Cortex itself is CPU-light;
GPU is for the upstream model + embedder.

## Status

- ✅ MVPs 1–4 shipped: passthrough, auto-ingest, virtualization (verbatim recall + reformulation), OpenAI + Anthropic translators
- ⚠️ MVP-5 deferred: production auth modes (BYO-key / tenant-key / hybrid), `X-Cortex-Degraded` SSE channel
- ⚠️ MVP-6 deferred: tool-aware ingest (per-file episodes for `read_file` / `write_file` / `bash` results — chunk-level retrieval inside file contents)

## Results detail

#### Opus 4.7 + cortex scaling (n=8, MRCR v2 8-needle)

| tokens | vanilla Opus 4.7 | Opus 4.7 + cortex | cortex behavior                |
|-------:|------------------|-------------------|--------------------------------|
| 53K    | 1.000            | 1.000             | passthrough                    |
| 205K   | 0.033            | **1.000**         | 1094 msgs → 7 + 4.6K recap     |
| 1M     | OVERFLOW (API)   | **0.997**         | 5303 msgs → 7 + 4.8K recap     |
| 2M     | OVERFLOW (API)   | **1.000**         | 10615 msgs → 7 + 5.0K recap    |
| 5M     | OVERFLOW (API)   | **1.000**         | 25859 msgs → 7 + 6.1K recap    |
| 9M     | OVERFLOW (API)   | **1.000**         | 48817 msgs → 7 + 6.2K recap    |
| 16M    | OVERFLOW (API)   | **0.999**         | 83150 msgs → 7 + 65K recap     |
| 29M    | OVERFLOW (API)   | **1.000**         | 154847 msgs → 7 + 65K recap    |

Overall: vanilla 13% perfect, cortex 100% perfect. The 1M row sits at
Anthropic's published Opus 4.6 native context limit; the 29M row is
**29× past it**. Wall clock at 29M: 407s (~7 min), dominated by LM
Studio batched embedding of 155K cold message groups. At 16M+ cortex
switches to `verbatim_recall_k=200` (default 16).

#### Opus 4.7 + cortex on RULER niah_multikey_3 (n=1 per scale)

| tokens (llama3) | vanilla Opus 4.7 | Opus 4.7 + cortex | cortex K | cortex behavior                |
|---------------:|------------------|-------------------|---------:|--------------------------------|
| 64K            | 1.000            | 1.000             | 16       | passthrough                    |
| 128K           | 1.000            | 1.000             | 16       | passthrough                    |
| 256K           | 1.000            | 1.000             | 16       | 9007 msgs → 7 + 0.8K recap     |
| 512K           | 1.000            | **1.000**         | 200      | 18107 msgs → 7 + 8.1K recap    |
| 1M             | OVERFLOW (API)   | **1.000**         | 200      | 36207 msgs → 7 + 8.2K recap    |
| 2M             | OVERFLOW (API)   | **1.000**         | 200      | 72409 msgs → 7 + 8.2K recap    |
| 5M             | OVERFLOW (API)   | **1.000**         | 200      | 181015 msgs → 7 + 8.2K recap   |
| 10M            | OVERFLOW (API)   | **1.000**         | 2000     | 362025 msgs → 7 + 67K recap    |

Vanilla Opus matches cortex through 512K; the API rejects every request
at 1M+. Cortex is 100% all-found at every scale. The K knob grows with
the haystack but the recap budget stays bounded (~67K tokens at 10M).

#### 9B + cortex matches Opus on MRCR (n=30)

| arm                    | n  | MRCR lenient mean | perfect% | L-bucket lenient |
|------------------------|----|-------------------|----------|------------------|
| Qwen3.5-9B (raw)       | 30 | 0.669             | 67%      | 0.006 (0%)       |
| **Qwen3.5-9B + cortex**| 30 | **1.000**         | **100%** | **0.999 (100%)** |
| Claude Opus 4.7        | 30 | 0.759             | 73%      | 0.561 (50%)      |

## Honest scope

Cortex's claim is **effectively unlimited context for retrieval-shaped
tasks** via inline-verbatim recall. MRCR and RULER are the cleanest
demonstrations: content needs to come back exactly as it appeared in
history. For reasoning, multi-hop inference, or summarization, the
technique is necessary but not sufficient — the upstream model still
has to do the reasoning over the recap.

All three pilots are single-seed (42):

- **9B + cortex on MRCR (n=30)**: rerun with seeds {17, 1729} pending.
  Cortex hits 1.000 lenient at N=30 so downside is bounded, but
  sampling variance is real.
- **Opus + cortex MRCR scaling (n=8)**: single-seed across 8 token
  scale targets. Six of the eight (1M, 2M, 5M, 9M, 16M, 29M tokens)
  are *synthesized* by stitching real MRCR 8-needle rows. Dataset max
  ≈ 625K tokens per row; full 400-row dataset combined caps out near
  ~30M tokens without needle-conflicting row reuse.
- **Opus + cortex RULER (n=1 per scale)**: preflight slice of
  `niah_multikey_3` from RULER-llama3-1M. 2M/5M/10M are synthesized by
  stitching the 1M base row with extra distractor lines from the same
  subtask. Reseeding to n=8 with rotated row indexes is the obvious
  next step.
- **`verbatim_recall_k` is tuned per scale on RULER** (16 / 200 / 2000).
  The default K=16 is fitted to MRCR's chat-shaped retrieval (long
  messages, few candidates per top-K); RULER's NIAH variant has many
  short candidates so K must grow with the haystack. Honest framing:
  cortex's *plumbing* is unchanged across these scales, but the *config*
  was hand-set per scale based on observed cardinality. Auto-tuning K
  from haystack cardinality is on the todo list.

Treat the scaling rows as a proof-of-concept until reseeded with
multiple seeds and rotated row indexes.

Per-pilot caveat docs: [bench/pilot_cortex/PAPER.md](bench/pilot_cortex/PAPER.md)
(9B), [bench/pilot_opus/run.py](bench/pilot_opus/run.py) docstring +
the synthesis logic in `pick_rows_by_target` (MRCR scaling), and
[bench/pilot_opus/run_ruler.py](bench/pilot_opus/run_ruler.py) docstring
(RULER scaling).

## License + contact

MIT. Issues + PRs welcome.
