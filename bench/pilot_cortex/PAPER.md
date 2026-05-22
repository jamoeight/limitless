# Inline verbatim recall lets a 9B model match a frontier model on long-context retrieval

**Pilot result, 30-row stratified.** Session: 2026-05-22, seed=42.

## Claim

A 9B local model (Qwen3.5-9B at LM Studio :1234, loaded at 100K context)
routed through the cortex proxy with **inline verbatim recall** matches or
beats Anthropic's Claude Opus 4.7 on the MRCR (multi-needle in haystack)
benchmark, lenient scoring, across S/M/L buckets — and dominates on L
(500K–1.5M character haystacks, up to 8 needles per task).

## Headline

30-row stratified pilot, 10 rows per bucket × {S, M, L} ⊆ openai/mrcr
(2needle + 4needle + 8needle shards), seed=42:

| arm          | n  | strict  | strict perfect% | lenient | lenient perfect% |
|--------------|----|---------|-----------------|---------|------------------|
| raw 9B       | 30 | 0.000   | 0%              | 0.669   | 67%              |
| **9B + cortex** | 30 | 0.000 | 0%              | **1.000** | **100%**        |
| Opus 4.7     | 30 | 0.759   | 73%             | 0.759   | 73%              |

Per-bucket lenient (the load-bearing test is the L bucket — 500K–1.5M
characters; rows above qwen3.5-9b's 100K-token loaded ctx):

| bucket | raw 9B            | 9B + cortex          | Opus 4.7         |
|--------|-------------------|----------------------|------------------|
| S      | 1.000 (100%)      | 1.000 (100%)         | 1.000 (100%)     |
| M      | 1.000 (100%)      | 1.000 (100%)         | 0.714 (70%)      |
| **L**  | **0.006 (0%)**    | **0.999 (100%)**     | **0.561 (50%)**  |

The L-bucket per-row detail (`results/pilot_cortex/scale30_v3.json`):

| row  | chars      | needles | raw 9B | **9B + cortex** | Opus 4.7 |
|------|-----------:|--------:|-------:|----------------:|---------:|
| 36   |   824,964  | 2 | 0.041 | **1.000** | 1.000 |
| 295  | 1,458,922  | 2 | 0.000 | **1.000** | 1.000 |
| 74   |   877,073  | 2 | 0.024 | **0.999** | 0.999 |
| 40   |   994,532  | 2 | 0.000 | **0.998** | 1.000 |
| 688  | 1,355,310  | 4 | 0.000 | **0.998** | 0.993 |
| 479  |   728,042  | 4 | 0.000 | **1.000** | 0.041 |
| 609  | 1,336,598  | 4 | 0.000 | **0.999** | 0.094 |
| 410  | 1,128,720  | 4 | 0.000 | **1.000** | 0.304 |
| 842  | 1,050,598  | 8 | 0.000 | **0.999** | 0.142 |
| 1056 | 1,388,446  | 8 | 0.000 | **0.999** | 0.040 |

cortex_9b is within `[0.998, 1.000]` on every single L row. Opus collapses
on 5 of 10 L rows (lenient < 0.4), with worst-case 0.040 on a 1.39M-char
8-needle row where cortex scored 0.999. Raw 9B is flat at zero — its native
100K-token context cannot hold the 1M+ character input.

**Goal criterion met:** cortex_9b lenient (1.000) ≥ 0.90 × Opus lenient
(0.683); cortex_9b > 0 on every L row.

## Mechanism

When the request exceeds the model's loaded context, cortex's `virtualize`
step does three things:

1. **Reformulates the query** with a single LM Studio JSON-schema call to
   `qwen3.5-9b`. Strips meta-instructions to extract the topical retrieval
   phrase:
   - `"Prepend 6xO8mh9FsP to the 2nd (1 indexed) short scene in a play
     about blueberries. Do not include any other text in your response."`
   - → `"write a short scene in a play about blueberries"`

   The literal query embeds close to other `"Prepend ..."` meta-messages
   in the haystack. The reformulated phrase embeds close to the actual
   needle content. Single LM Studio call, separate from the bounded-1-call
   judge contract in `timegraph.llm.judge`.

2. **Embeds and ranks** every cold atomic message-group via
   `text-embedding-nomic-embed-text-v1.5` (768D, already loaded for the
   ingest pipeline). One batched call. Cosine top-K (default K=24).
   Restores chronological order in the output so "the Nth thing" is
   interpretable to the model.

   No Qdrant or Neo4j roundtrip — recall runs inline over the in-flight
   message list. Works on a first-turn single-shot request before any
   background ingest has completed.

3. **Injects the top-K** as a verbatim `<retrieved_history>` block in the
   recap, between the original system prompt and the (kept) verbatim
   message window. The cold-summary path becomes the fallback when
   inline-verbatim returns empty.

The 9B model then sees: original system prompt + cortex recap (verbatim top-K
retrieved turns, chronologically ordered, with turn numbers) + last 2
verbatim message-groups + the user's query. It picks "the Nth scene about
blueberries" from the recap and reproduces it verbatim.

## What changed across iterations

| run | cortex_9b lenient (overall) | cortex_9b lenient (L bucket) | what changed |
|-----|---------------------------:|----------------------------:|--------------|
| v1  | 0.143                       | n/a (5 rows)               | initial; surfaced 3 bugs in virtualize |
| v2  | 0.542                       | 0.000                       | bug fixes — short-circuit on natural fit, reasoning_content fallback, cold-summary cap bump (`0354150`, `95d4537`) |
| v3 5-row | 1.000                  | 1.000 (2 L rows)            | inline verbatim recall + query reformulation (`210ed41`) |
| **v3 30-row** | **1.000**          | **0.999 (10 L rows)**       | confirmed at N=30 |

The v1→v2 changes were strict no-harm fixes. v3 is the first time cortex
*extends* the 9B model's capability — it lets a 9B model with a 100K-token
loaded context produce content from a 1.4M-character conversation.

## Methodology

Full harness in `bench/pilot_cortex/run.py`. Three arms per row, scored on
the official MRCR rubric (`bench/mrcr/loader.py::score`, SequenceMatcher
ratio with a random-string prefix check):

- **raw_9b**: POST `/v1/chat/completions` directly to LM Studio :1234 with
  qwen3.5-9b loaded (CONTEXT=100K). `max_tokens=8192, temperature=0`.
- **9B + cortex**: same body, POST to cortex proxy on :8080. Cortex
  virtualizes, retrieves, forwards to the same LM Studio instance.
- **Opus 4.7**: `claude -p --tools "" --system-prompt <neutral> --model
  opus --output-format text --no-session-persistence --setting-sources
  ""`. The MRCR conversation is flattened to a single prompt.

A neutral no-preamble system prompt is prepended to all 9B calls.
Insufficient — qwen3.5-9b's chat template tic still emits `\n\n` at start.
Strict scores are 0 across every 9B arm for this reason (response length
is always `gold_length + 2`). The lenient rubric (lstrip then apply the
same prefix check) is the headline.

**Stratified sampling**: per_bucket=10, balanced across needle counts
(2/4/8) where shard contents permit. seed=42. The S bucket is rows ≤200K
chars, M is 200K–500K, L is 500K–1.5M.

**Backends** (must be up):

- Neo4j 5.24 on :7687 (`docker compose up -d`)
- Qdrant 1.12 on :6333/:6334 (`docker compose up -d`)
- LM Studio :1234 with `qwen/qwen3.5-9b` loaded at CONTEXT=100000 and
  `text-embedding-nomic-embed-text-v1.5` (768D) loaded.
  Verify: `lms ps` must show CONTEXT=100000 for qwen3.5-9b.
- cortex proxy on :8080 (env vars below).

**Cortex env** for benchmark fidelity:

```
CORTEX_DEFAULT_PROVIDER=openai
CORTEX_OPENAI_BASE_URL=http://127.0.0.1:1234
CORTEX_ENABLE_AUTO_INGEST=false       # disabled to avoid extractor contention
CORTEX_ENABLE_VIRTUALIZATION=true
CORTEX_ENABLE_VERBATIM_RECALL=true
CORTEX_ENABLE_QUERY_REFORMULATION=true
CORTEX_UPSTREAM_CONTEXT_LIMIT=100000  # match LM Studio's loaded ctx
CORTEX_LAST_K_SPANS=2
CORTEX_VERBATIM_RECALL_K=24
```

`CORTEX_ENABLE_AUTO_INGEST=false` is set purely to free LM Studio's GPU
slot from extractor calls during the benchmark — the verbatim recall path
embeds inline from the request's messages list and does not depend on
ingest having completed. With ingest=true, extractor calls (qwen3.5-9b)
fight MRCR generation (also qwen3.5-9b) for the same GPU; the bench-only
flag isolates the test.

## Operational lesson from the runs

The first attempt at the 30-row pilot failed catastrophically — cortex_9b
L lenient = 0.100, all L rows returned 502 Bad Gateway. Root cause was
LM Studio having silently reloaded `qwen/qwen3.5-9b` with the **default
context length of 4096 tokens**, not the 100K the 5-row v3 used. Cortex's
virtualized request (verbatim window + ~16K recap tokens) overflowed the
upstream's 4K window. Confirmed via reproducing one L row manually and
reading the upstream error body:

```
The number of tokens to keep from the initial prompt is greater than the
context length (n_keep: 12216 >= n_ctx: 4096).
```

Fix: `lms unload qwen/qwen3.5-9b; lms load qwen/qwen3.5-9b
--context-length 100000 --gpu max --ttl 86400`. After this, the re-run hit
all numbers above.

The broken results are preserved in `results/pilot_cortex/scale30_v3_broken.json`
for the record.

## Caveats and threats to validity

1. **Strict rubric universally fails on 9B arms.** Qwen3.5-9B prepends
   `\n\n` to responses, the no-preamble system prompt doesn't fully
   suppress it, and the official MRCR rubric requires the response to
   *start* with `random_string`. Visible identically on raw_9b and
   cortex_9b — symmetric across both, so the cortex-vs-raw comparison
   stands. The cortex-vs-Opus comparison loses some force here: Opus
   scores strict==lenient on most rows because it actually starts with
   the requested string.

2. **Single seed (42).** Stratified per-bucket sampling helps but doesn't
   replace seed variance. To-do: rerun with seeds {17, 1729} and report
   mean ± stdev. With cortex hitting 1.000 on lenient at N=30 there is
   limited room for downside, but variance from sampling is real.

3. **One frontier model.** Comparison is against Opus 4.7. Cortex may
   compare differently against Sonnet 4.6, GPT-5, etc. The claim worth
   making is that this technique extends ANY model effectively past its
   native context — not that it specifically beats Opus.

4. **MRCR is a retrieval-shaped benchmark.** The needles are verbatim text
   from the haystack. The 9B model's job is to reproduce one — not
   reason about it. For tasks that require reasoning across many
   message turns (multi-hop, cross-reference, summarization),
   inline-verbatim recall is necessary but not sufficient.

5. **Reformulator failure is silent.** If the LM Studio reformulator call
   times out or returns malformed JSON, `recall_verbatim_inline` falls
   back to the raw query as the embedding key. Falls open, doesn't
   error. But for MRCR specifically, the raw-query embedding leads to
   worse retrieval — failure would be hard to spot from response
   correctness alone since cortex might silently degrade to the
   passthrough behavior.

6. **Inline embedding cost scales with cold-group count.** An L row with
   ~1,500 atomic groups requires a 1,501-input batched embedding pass.
   At LM Studio nomic throughput (~2,000 embed/s on a 4090) that's
   ~750ms — amortized fine. Will be visible at 100K-message conversations.

7. **The bench disables auto-ingest** to avoid extractor/generation
   contention on a single LM Studio instance. In a real cortex
   deployment with a dedicated embedder host or a multi-instance LM
   Studio setup, both paths run concurrently with no contention.

## Reproduction

```bash
# Start backends (one-time).
docker compose up -d
.venv/Scripts/python.exe -m timegraph.storage.schema --apply
.venv/Scripts/python.exe -c "import asyncio; from timegraph.storage.qdrant_client import ensure_collections; asyncio.run(ensure_collections())"

# Load LM Studio models at the right context size (REQUIRED).
lms load qwen/qwen3.5-9b --identifier qwen/qwen3.5-9b --context-length 100000 --gpu max --ttl 86400
lms load text-embedding-nomic-embed-text-v1.5 --gpu max --ttl 86400
lms ps  # verify CONTEXT=100000 for qwen3.5-9b

# Start cortex proxy.
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

# Run pilot.
PYTHONPATH=src .venv/Scripts/python.exe bench/pilot_cortex/run.py \
  --seed 42 --per-bucket 10 --out results/pilot_cortex/scale30_v3.json
```

## What this does not claim

- It does not claim cortex makes a 9B model "as good as" Opus *in
  general*. MRCR is verbatim-retrieval; cortex hands the 9B model the
  exact needles and the 9B model picks the Nth. For reasoning,
  multi-hop, or summarization tasks, this technique is necessary but
  not sufficient.

- It does not claim cortex is faster than a frontier model. Opus runs an
  L row in ~25 s. Cortex on 9B runs it in 15–50 s (faster than raw 9B's
  100s but not always faster than Opus).

- It does not claim "infinite context." It claims **"effectively
  unlimited context for retrieval-shaped tasks via inline-verbatim
  in-conversation recall."** Reasoning and tool-use chains are separate
  problems with separate roadmap items in cortex.

- It does not address the strict-rubric `\n\n` preamble. Closing that
  gap would require either a stronger system-prompt suppression
  technique or a post-generation lstrip in the proxy. Either would
  bring cortex_9b strict to 1.000 on these results.

## Headline you can quote

> A 9B local model (Qwen3.5-9B), routed through cortex's inline-verbatim
> recall, scored **lenient perfect on 100% of 30 stratified MRCR rows**,
> including all 10 L rows (500K–1.5M characters, up to 8 needles). Claude
> Opus 4.7 scored 73% perfect overall and 50% on L; raw Qwen3.5-9B scored
> 0% on L. The mechanism is two added components — query reformulation
> and inline embedding-based ranking of cold message groups — both
> running on the same LM Studio instance hosting the 9B model. No frontier
> model is involved in cortex's retrieval; the 9B model is doing the
> reasoning over a recap selected from its own embedding space.
