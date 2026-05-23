# cortex

**Infinite-context proxy for any LLM.** Compresses long chat history into
a verbatim recap that fits inside the upstream model's native window.
Same proxy on any model — local 9B or frontier Opus.

---

### Opus 4.7 + cortex stays 100% perfect through ~9M tokens

![Opus + cortex scaling past Anthropic's 1M-token Opus limit](results/opus_vs_cortex/hero_v2.png)

Anthropic's [Opus 4.6 announcement](https://www.anthropic.com/news/claude-opus-4-6)
publishes their long-context score on MRCR v2 8-needle at the **1M token**
scale (76% perfect for Opus 4.6, 18.5% for Sonnet 4.5). Vanilla Opus 4.7
via `claude -p` collapses to 3% at ~205K tokens and the API rejects ~1M+
outright. **Opus + cortex stays at 100% perfect through ~9M tokens** —
9× past Anthropic's published native limit — by compressing up to 49K
messages into 7 verbatim turns + a ~6K-token recap that fits inside the
upstream model's window.

n=6 single-seed pilot across six scale targets. ~53K- and ~205K-token rows
are real MRCR 8-needle samples; the four larger rows (~1M, ~2M, ~5M, ~9M
tokens) are *synthesized* from real MRCR 8-needle rows (dataset max =
~625K tokens per row). Token counts via o200k_base. Run script and
methodology: [bench/pilot_opus/](bench/pilot_opus/).

---

### A local 9B + cortex matches Opus on MRCR

![9B + cortex matches Opus on 30 MRCR rows](results/pilot_cortex/hero.png)

Same proxy, different demonstration. **Qwen3.5-9B + cortex hits 100% perfect
on 30 MRCR rows. Vanilla Opus 4.7 hits 73%.** The 9B catches the frontier
because cortex pre-locates the needles — the model only sees the recap.

n=30 single-seed pilot. Strict-rubric is 0 across both 9B arms (qwen3.5
prepends `\n\n`). Full caveats:
[bench/pilot_cortex/PAPER.md](bench/pilot_cortex/PAPER.md).

## What it is

Cortex is an HTTP proxy that sits in front of any OpenAI-compatible (or
Anthropic-compatible) LLM endpoint and gives it effectively unlimited context.
When a conversation exceeds the upstream model's window, cortex:

1. **Reformulates the user's query** with one small LM Studio call —
   strips meta-instructions to extract the topical retrieval phrase.
2. **Embeds and cosine-ranks every cold message-group** inline against the
   reformulated query (no Qdrant roundtrip — works on a first turn).
3. **Injects the top-K verbatim** into a `<retrieved_history>` block in the
   system prompt, chronologically ordered.

The model then sees its original system prompt + the relevant verbatim
turns + the last few message-groups + the query. It picks the right
content from the recap and responds.

When the conversation fits natively, cortex **short-circuits to
pass-through** — the model sees byte-identical input to raw and the
quality is identical. (Verified: S/M-bucket cortex == raw on the pilot.)

## Quickstart

```bash
# 1. Backends + Python.
docker compose up -d                          # Neo4j + Qdrant
python -m venv .venv && .venv/Scripts/pip install -e .
docker compose exec -T qdrant true            # wait for healthy
python -m timegraph.storage.schema --apply

# 2. Load LM Studio models at the right context (lms is the LM Studio CLI).
lms load qwen/qwen3.5-9b --identifier qwen/qwen3.5-9b --context-length 100000 --gpu max --ttl 86400
lms load text-embedding-nomic-embed-text-v1.5 --gpu max --ttl 86400
lms ps                                        # verify CONTEXT=100000

# 3. Start the cortex proxy on :8080.
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

# 4. Reproduce the benchmark.
PYTHONPATH=src .venv/Scripts/python.exe bench/pilot_cortex/run.py \
  --seed 42 --per-bucket 10 --out results/pilot_cortex/scale30.json
```

Cortex exposes OpenAI-compatible `/v1/chat/completions` and
Anthropic-compatible `/v1/messages`. Point any client at `http://127.0.0.1:8080`
and use the upstream's API key as the auth header.

## What this does NOT claim

- **It does not make 9B as smart as Opus.** Cortex gives the model the right
  context; it doesn't make the model better at reasoning. For complex code
  refactoring, multi-hop logic, or system design — use a frontier model.
  Cortex gets you frontier-scale *memory* at 9B inference cost.
- **It does not eliminate frontier APIs.** Tasks that need frontier
  intelligence still need frontier intelligence.
- **Strict-rubric scores are 0** across every 9B arm (raw and cortex
  both) — qwen3.5-9b's chat template prepends `\n\n`. The lenient rubric
  (lstrip → same prefix check) is the headline. Symmetric across raw and
  cortex; PAPER.md has the full caveat.

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

#### Opus 4.7 + cortex scaling (n=6, 8-needle, MRCR v2)

| tokens (o200k) | chars (actual) | vanilla Opus 4.7 | Opus 4.7 + cortex | cortex behavior               |
|---------------:|---------------:|------------------|-------------------|-------------------------------|
| ~53K           | 256K           | 1.000            | 1.000             | passthrough                   |
| ~205K          | 1.0M           | 0.033            | **1.000**         | 1094 msgs → 7 + 4.6K recap    |
| ~1.02M         | 5.0M (synth)   | OVERFLOW (API)   | **0.997**         | 5303 msgs → 7 + 4.8K recap    |
| ~2.01M         | 9.9M (synth)   | OVERFLOW (API)   | **1.000**         | 10615 msgs → 7 + 5.0K recap   |
| ~4.88M         | 23.8M (synth)  | OVERFLOW (API)   | **1.000**         | 25859 msgs → 7 + 6.1K recap   |
| ~9.19M         | 44.7M (synth)  | OVERFLOW (API)   | **1.000**         | 48817 msgs → 7 + 6.2K recap   |

Overall: vanilla 17% perfect, cortex 100% perfect. The ~1.02M-token row
is at Anthropic's published Opus 4.6 native context limit; the ~9.19M
row is **9× past it**. Wall clock at 9M tokens: 247s (mostly batched
embedding of 49K cold message groups via LM Studio).

#### 9B + cortex matches Opus on MRCR (n=30)

| arm                    | n  | MRCR lenient mean | perfect% | L-bucket lenient |
|------------------------|----|-------------------|----------|------------------|
| Qwen3.5-9B (raw)       | 30 | 0.669             | 67%      | 0.006 (0%)       |
| **Qwen3.5-9B + cortex**| 30 | **1.000**         | **100%** | **0.999 (100%)** |
| Claude Opus 4.7        | 30 | 0.759             | 73%      | 0.561 (50%)      |

## Honest scope

Cortex's claim is **effectively unlimited context for retrieval-shaped
tasks** via inline-verbatim recall. MRCR is the cleanest demonstration:
content needs to come back exactly as it appeared in history. For
reasoning, multi-hop inference, or summarization, the technique is
necessary but not sufficient — the upstream model still has to do the
reasoning over the recap.

Both pilots are single-seed (42):
- 9B + cortex on MRCR (n=30): rerun with seeds {17, 1729} pending. Cortex
  hits 1.000 on lenient at N=30 so downside is limited, but sampling
  variance is real.
- Opus + cortex scaling (n=6): single-seed across 6 token scale targets.
  Four of the six (~1M, ~2M, ~5M, ~9M tokens) are *synthesized* by
  stitching real MRCR 8-needle rows (dataset max = ~625K tokens per row).
  Treat as a scaling proof-of-concept until reseeded.

Per-pilot caveat docs: [bench/pilot_cortex/PAPER.md](bench/pilot_cortex/PAPER.md)
(9B story) and [bench/pilot_opus/run.py](bench/pilot_opus/run.py) docstring +
the synthesis logic in `pick_rows_by_target` (Opus scaling).

## License + contact

MIT. Issues + PRs welcome.
