# cortex

**A proxy layer for long-context retrieval on top of any LLM.** Cortex
embeds and cosine-ranks cold chat-history message-groups against a
reformulated query and injects the top-K **verbatim** into the system
prompt. Inference-time only — no model modification, no fine-tuning,
no KV-cache surgery. Drop-in proxy for the Anthropic Messages API and
the OpenAI Chat Completions API.

**Calibration**: Anthropic
[reports](https://www.anthropic.com/news/claude-opus-4-6) Opus 4.6
(1M-token native context) scoring **76%** on the MRCR v2 8-needle 1M
variant; Sonnet 4.5 scores **18.5%**. **Opus 4.7 + cortex scores 100%
at 1M tokens** on the same benchmark and **stays at 100% through 10M
tokens**. Same result on RULER `niah_multikey_3` from 64K to 10M llama3
tokens. A local Qwen3.5-9B routed through cortex hits 100% on 30
stratified MRCR rows (raw 9B: 67%, raw Opus 4.7: 73%).

---

### Demo 1 — MRCR v2 8-needle at Anthropic's published scales

![Opus + cortex MRCR v2 8-needle scaling: 256K / 1M / 5M / 10M tokens](results/opus_vs_cortex/hero_v2.png)

Same four context lengths Anthropic measures in the
[claude-opus-4-6 announcement](https://www.anthropic.com/news/claude-opus-4-6).
Vanilla Opus 4.7 (200K native context) scores **16%** at 256K and the
Anthropic API rejects every request at 1M+. **Opus 4.7 + cortex scores
100% at every scale**, including past Opus 4.6's 1M-token native limit,
by compressing up to ~39K cold messages into 7 verbatim turns + a
~6K-token recap.

| context | vanilla Opus 4.7 | Opus 4.7 + cortex | reference (Anthropic) |
|--------:|-----------------:|------------------:|----------------------:|
| 256K    | 16%              | **100%**          | —                     |
| 1M      | OVERFLOW         | **100%**          | Opus 4.6: **76%**     |
| 5M      | OVERFLOW         | **100%**          | beyond Opus 4.6 limit |
| 10M     | OVERFLOW         | **100%**          | beyond Opus 4.6 limit |

n=4, seed=42, lenient rubric (`response.lstrip()` then strict
random-string prefix check + `SequenceMatcher` ratio). 256K and 1M are
real MRCR 8-needle rows from `openai/mrcr` (dataset max ≈ 625K tokens
per row); 5M and 10M are *synthesized* by stitching real rows with the
gold needle preserved in the base row. Token counts via char/4
convention. Reproduce:

```bash
# After Path A install (below), with cortex.server running on :8080:
PYTHONPATH=. .venv/Scripts/python.exe bench/pilot_opus/run.py \
  --targets 256000,1000000,5000000,10000000 --unit tokens \
  --n-needles 8 --seed 42 \
  --out results/opus_vs_cortex/mrcr_v3.json
```

Raw output: [results/opus_vs_cortex/mrcr_v3.json](results/opus_vs_cortex/mrcr_v3.json).
Chart code: [bench/pilot_opus/make_chart_v3.py](bench/pilot_opus/make_chart_v3.py).

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

## How this relates to prior work

The long-context problem has three orthogonal solution families. Cortex
sits squarely in the third — and the SOTA literature suggests this is
the family that scales most cleanly past native window limits on
retrieval-shaped tasks.

**1. Native long-context architectures.** Training models to attend over
longer sequences directly. Recent examples include Anthropic's
[Opus 4.6](https://www.anthropic.com/news/claude-opus-4-6) (1M tokens
native), Gemini 1.5 Pro, Jamba-1.5-large, and Qwen2.5-14B-1M. RULER
([Hsieh et al., 2024](https://arxiv.org/abs/2404.06654)) shows even the
best of these degrade past their effective context — and the [LongMemEval
ICLR 2025 paper](https://arxiv.org/abs/2410.10813) finds 30–60%
performance drops on advanced long-context LLMs (GPT-4o, Llama 3.1,
Phi-3) vs. oracle retrieval at ≥115K-token chat histories. Cortex
inverts this: rather than ask the model to attend over the whole
history, give it a small recap that fits in its high-attention zone.

**2. KV-cache compression.** Operates inside the model: keep heavy-hitter
tokens, evict the rest. StreamingLLM
([Xiao et al., 2023](https://arxiv.org/abs/2309.17453)), H2O
([Zhang et al., 2023](https://arxiv.org/abs/2306.14048)), SnapKV
([Li et al., 2024](https://arxiv.org/abs/2404.14469)), and more recent
work like [RocketKV](https://arxiv.org/abs/2502.14051) reduce GPU memory
during inference but require model access and don't help if the input
itself exceeds the context window. Cortex is API-side: works on any
model you can call over HTTP, including closed frontier APIs.

**3. Retrieval / memory systems.** Standard RAG, plus more elaborate
memory architectures: MemGPT ([Packer et al., 2024](https://arxiv.org/abs/2310.08560))
(OS-style paging), Mem0, LongMem, A-MEM, MemMachine. Most do fact
extraction or summarization — useful for cross-session memory but lossy
for needle-in-haystack tasks where the answer is a verbatim string.
Cortex is the **verbatim** variant: cold message-groups are inserted
without summarization, preserving exact text for retrieval. Recent RAG
↔ long-context work
([Yu et al., 2024](https://arxiv.org/abs/2410.05983);
[Li et al., 2025](https://arxiv.org/abs/2501.01880);
[Yang et al., 2025](https://arxiv.org/abs/2502.12462)) finds the two
approaches are complementary rather than substitutable. Anthropic's own
memory product
([Sep 2025](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool))
is a file-based markdown store with just-in-time retrieval — closer to
cortex in shape than to vector-DB RAG, but storage rather than inference-
time injection.

**What's specific to cortex:**
- **Verbatim insertion** of cold message-groups, not summary. Critical
  for needle tasks where summarization destroys the signal.
- **One small LLM call** for query reformulation (the only extra LLM
  call beyond the upstream); then cosine ranking on embeddings.
- **Proxy architecture** — works on closed-frontier APIs (Anthropic,
  OpenAI) and local OpenAI-compat servers (LM Studio, vLLM, llama.cpp)
  with no model access required.
- **Bounded recap budget** (~5-65K tokens) regardless of haystack size,
  so the model always sees input inside its high-attention zone.

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

### Path A — Claude Code MCP server with Haiku judge (no API key needed)

Drop timegraph into Claude Code as an MCP server. The 5 tools
(`remember`, `add_fact`, `recall`, `query`, `attest`) become available in
every Claude Code session opened in this repo. Judge calls inside those
tools route through your local `claude -p --model haiku` — using your
Claude Code subscription, no `ANTHROPIC_API_KEY` required.

```bash
# 1. Verify the claude CLI is on PATH and logged in.
claude --version
claude -p --model haiku "say ok"              # should print "ok"

# 2. Register timegraph as a project-scoped MCP server.
claude mcp add timegraph \
  --scope project \
  -e TG_JUDGE_BACKEND=claude_cli \
  -e TG_JUDGE_CLAUDE_MODEL=haiku \
  -- "$(pwd)/.venv/Scripts/timegraph-mcp.exe"

# 3. Confirm it's connected.
claude mcp list                               # → timegraph: ✓ Connected
```

That writes `.mcp.json` at the repo root:

```json
{
  "mcpServers": {
    "timegraph": {
      "type": "stdio",
      "command": "<repo>/.venv/Scripts/timegraph-mcp.exe",
      "env": {
        "TG_JUDGE_BACKEND": "claude_cli",
        "TG_JUDGE_CLAUDE_MODEL": "haiku"
      }
    }
  }
}
```

Open a new `claude` session in this directory, approve the project MCP
server prompt, and the 5 tools are live. First tool call blocks ~15 s
while `claude -p --model haiku` warms up; subsequent calls cache.

Use `--scope user` instead of `--scope project` to make timegraph
available from any directory.

**Wiring a non-Claude-Code client to the same backend** — start the
cortex HTTP proxy on `:8080` and any OpenAI- or Anthropic-compatible
client (opencode, continue.dev, an SDK) can use it:

```bash
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
```

Trade-off on the HTTP proxy path: each upstream call shells out to
`claude -p` (~10–20 s subprocess overhead vs ~3 s direct API). Worth it
for *"I already pay for Claude Code — give me infinite context on top of
it"*; the MCP path above is faster for in-Claude-Code use because Claude
Code talks to Anthropic natively and only invokes the timegraph tools
on demand.

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

#### Opus 4.7 + cortex on MRCR v2 8-needle (n=4, Anthropic-style scales)

| tokens | vanilla Opus 4.7 | Opus 4.7 + cortex | cortex behavior                  |
|-------:|------------------|-------------------|----------------------------------|
| 256K   | 16%              | **100%**          | 1078 msgs → 7 + 11K recap        |
| 1M     | OVERFLOW (API)   | **100%**          | 5201 msgs → 7 + 5.5K recap       |
| 5M     | OVERFLOW (API)   | **100%**          | 19807 msgs → 7 + 5.9K recap      |
| 10M    | OVERFLOW (API)   | **100%**          | 38713 msgs → 7 + 5.4K recap      |

Overall: vanilla 0% perfect (collapses or overflows), cortex 100% perfect
at all four Anthropic-style scales. Wall clock at 10M: 125s, dominated
by LM Studio batched embedding of ~39K cold message groups. For reference,
Anthropic reports Opus 4.6 at **76%** on the same 1M variant (Sonnet 4.5
at 18.5%).

Extended scaling: an earlier 8-target run pushed cortex from 53K through
**29M tokens** (155K stitched messages, ~7 min wall clock at the top
end). Raw data: [results/opus_vs_cortex/mrcr_v4.json](results/opus_vs_cortex/mrcr_v4.json).
The headline 4-target chart above mirrors Anthropic's published methodology
(256K / 1M / 5M / 10M) so the numbers are directly comparable.

#### Opus 4.7 + cortex on RULER niah_multikey_3 (n=1 per scale)

| tokens (llama3) | vanilla Opus 4.7 | Opus 4.7 + cortex | cortex K | cortex behavior                |
|---------------:|------------------|-------------------|---------:|--------------------------------|
| 64K            | 100%             | 100%              | 16       | passthrough                    |
| 128K           | 100%             | 100%              | 16       | passthrough                    |
| 256K           | 100%             | 100%              | 16       | 9007 msgs → 7 + 0.8K recap     |
| 512K           | 100%             | **100%**          | 200      | 18107 msgs → 7 + 8.1K recap    |
| 1M             | OVERFLOW (API)   | **100%**          | 200      | 36207 msgs → 7 + 8.2K recap    |
| 2M             | OVERFLOW (API)   | **100%**          | 200      | 72409 msgs → 7 + 8.2K recap    |
| 5M             | OVERFLOW (API)   | **100%**          | 200      | 181015 msgs → 7 + 8.2K recap   |
| 10M            | OVERFLOW (API)   | **100%**          | 2000     | 362025 msgs → 7 + 67K recap    |

Vanilla Opus matches cortex through 512K; the API rejects every request
at 1M+. Cortex is 100% all-found at every scale. The K knob grows with
the haystack but the recap budget stays bounded (~67K tokens at 10M).

#### 9B + cortex matches Opus on MRCR (n=30)

| arm                    | n  | MRCR lenient mean | perfect% | L-bucket perfect |
|------------------------|----|-------------------|----------|------------------|
| Qwen3.5-9B (raw)       | 30 | 67%               | 67%      | 0%               |
| **Qwen3.5-9B + cortex**| 30 | **100%**          | **100%** | **100%**         |
| Claude Opus 4.7        | 30 | 76%               | 73%      | 50%              |

## Honest scope

Cortex's claim is **effectively unlimited context for retrieval-shaped
tasks** via inline-verbatim recall. MRCR and RULER are the cleanest
demonstrations: content needs to come back exactly as it appeared in
history. For reasoning, multi-hop inference, or summarization, the
technique is necessary but not sufficient — the upstream model still
has to do the reasoning over the recap.

All three pilots are single-seed (42):

- **9B + cortex on MRCR (n=30)**: rerun with seeds {17, 1729} pending.
  Cortex hits 100% lenient at N=30 so downside is bounded, but sampling
  variance is real.
- **Opus + cortex MRCR scaling (n=4 headline, n=8 extended)**: the
  hero chart shows the four Anthropic-style scales (256K / 1M / 5M /
  10M tokens); an extended single-seed run also covers 53K through
  29M tokens in [results/opus_vs_cortex/mrcr_v4.json](results/opus_vs_cortex/mrcr_v4.json).
  At 1M tokens and above the rows are *synthesized* by stitching real
  MRCR 8-needle rows (dataset max ≈ 625K tokens per row).
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
