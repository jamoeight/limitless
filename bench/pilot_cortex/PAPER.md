# Inline verbatim recall lets a 9B model match a frontier model on long-context retrieval

**Pilot result, draft.** Session: 2026-05-22. Pre-print quality only — formal
write-up depends on confirmation at N=30 (in progress) and broader sampling
across MRCR shards.

## Claim

A 9B local model (Qwen3.5-9B at LM Studio :1234) routed through the cortex
proxy with **inline verbatim recall** matches or beats Anthropic's Claude
Opus 4.7 on MRCR (multi-needle in haystack) lenient scoring, on tasks whose
raw input — 1M+ characters, 8 needles per task — exceeds the 9B model's
native ~32–100K-token context.

On the 5-row pilot (seed=42, stratified across S/M/L buckets):

| arm          | n | lenient mean | perfect% lenient | notes              |
|--------------|---|--------------|------------------|--------------------|
| raw 9B       | 5 | 0.444        | 40%              | 2 hallucinated, 1 truncated |
| **9B + cortex** | 5 | **1.000**   | **100%**         | 0 hallucinated, 0 truncated |
| Opus 4.7     | 5 | 0.644        | 60%              | 1 hallucinated     |

The interesting cases are L-bucket (1M+ chars):

| row  | chars | needles | raw 9B | 9B + cortex | Opus 4.7 |
|------|-------|---------|--------|-------------|----------|
| 842  | 1,050K| 8       | 0.000  | **0.999**   | 0.146    |
| 1056 | 1,388K| 8       | 0.000  | **1.000**   | 0.072    |

On the two L rows, 9B + cortex outscored Opus by **6.8×** and **14×**
respectively. Opus hallucinated needle content on the 1.39M-char row; cortex
returned the exact text.

**Caveat: strict-rubric scores are 0 across every 9B arm** (cortex and raw
both). This is a Qwen3.5-9B chat-template tic that prefixes responses with
`\n\n`. Response length is always `gold_length + 2`. After lstrip — the
"lenient" rubric — cortex matches gold verbatim. The strict failure is not a
retrieval failure.

## What changed vs the v2 pilot (which got cortex_9b lenient = 0.542)

Two production-relevant changes shipped in commit `210ed41`:

1. **Inline verbatim recall.** When the request exceeds the model's context
   budget, cortex now embeds every cold atomic message-group via the same
   nomic-embed-text-v1.5 model already loaded for the existing ingest path,
   cosine-ranks them against the query, and injects the top-K (default 24)
   verbatim into the recap as a `<retrieved_history>` block. Order preserved
   chronologically. No Qdrant or Neo4j roundtrip — the recall runs purely on
   the in-flight message list, so it works on a first turn before any
   background ingest has completed.

2. **Query reformulation.** A single LM Studio JSON-schema call rewrites the
   user's literal query (`"Prepend 6xO8mh9FsP to the 2nd scene about
   blueberries"`) into a topical retrieval phrase (`"write a scene about
   blueberries"`). The literal query embeds close to other "Prepend ..."
   meta-messages; the reformulated one embeds close to the actual needle
   content. This is the call that gets verbatim recall to the right
   neighborhood. Separate from the bounded-1-call judge contract; does not
   touch `timegraph.llm.judge`.

Previously the recap path was:

- `cold_summary` = bulleted recap with per-message truncation (~280 chars).
- `recall_text` = graph-extracted facts (subject/predicate/object).

Both lose the verbatim text the rubric checks. The MRCR L rows scored 0.000
across the board on cortex_9b in v2 for that reason.

## Methodology

The full pilot harness lives in `bench/pilot_cortex/run.py`. Three arms per
row, evaluated on the official MRCR rubric from `bench/mrcr/loader.py::score`
(SequenceMatcher.ratio with a mandatory random-string prefix check):

- **raw_9b**: POST `/v1/chat/completions` directly to LM Studio :1234 with
  qwen3.5-9b loaded. `max_tokens=8192, temperature=0, top_p=1`.
- **9B + cortex**: same body, POST to the cortex proxy on :8080. Cortex
  loads, virtualizes, retrieves, forwards to the same LM Studio.
- **Opus 4.7**: `claude -p --tools "" --system-prompt <neutral> --model opus
  --output-format text --no-session-persistence --setting-sources ""`. The
  full MRCR conversation is flattened into a single prompt.

A neutral system prompt is prepended on all 9B calls to suppress the
`\n\n` preamble (it doesn't fully work, hence the strict-0 across all 9B
arms — see Caveats).

**Backends** (must be up + pinned in LM Studio):

- Neo4j 5.24 on :7687 (`docker compose up -d`)
- Qdrant 1.12 on :6333/:6334 (`docker compose up -d`)
- LM Studio :1234 with `qwen/qwen3.5-9b` + `text-embedding-nomic-embed-text-v1.5`
  (768D) both loaded and pinned
- cortex proxy on :8080 (see env-var setup below)

**Cortex env** for benchmark fidelity:

```
CORTEX_DEFAULT_PROVIDER=openai
CORTEX_OPENAI_BASE_URL=http://127.0.0.1:1234
CORTEX_ENABLE_AUTO_INGEST=false       # disabled for benchmark — verbatim recall doesn't need it
CORTEX_ENABLE_VIRTUALIZATION=true
CORTEX_ENABLE_VERBATIM_RECALL=true
CORTEX_ENABLE_QUERY_REFORMULATION=true
CORTEX_UPSTREAM_CONTEXT_LIMIT=100000  # qwen3.5-9b loaded context
CORTEX_LAST_K_SPANS=2
CORTEX_VERBATIM_RECALL_K=24
```

Note `CORTEX_ENABLE_AUTO_INGEST=false`. The verbatim recall path embeds
inline from the request's message list — it does NOT need Qdrant /
Neo4j-backed ingest to have already completed. We disable ingest here purely
to free LM Studio's single model slot from extractor contention during
scale-up; correctness is unaffected.

## Caveats and threats to validity

1. **Strict rubric universally fails.** Every 9B arm — raw and cortex —
   leads its response with `\n\n` despite a no-preamble system prompt. This
   is a Qwen3.5-9B chat-template tic. The lenient rubric (strip leading
   whitespace, then apply the same prefix check) gives the headline numbers.
   This caveat is symmetric across raw_9b and cortex_9b, so the comparison
   between them stands; it weakens the comparison against Opus, which scores
   strict==lenient on most rows.

2. **The 5-row pilot is small.** A 30-row pilot (`per-bucket=10`) is in
   progress. The headline claim depends on those numbers holding.

3. **MRCR rows where the full conversation fits natively in the 9B context
   are a wash.** On S/M rows (≤200K chars ≈ ≤50K tokens), cortex
   short-circuits and forwards the conversation unchanged. The benefit is
   not visible there. The L rows are the differentiated test.

4. **Single seed.** seed=42 only. Stratified per-bucket sampling helps but
   doesn't replace seed-variance estimates. To-do: re-run with seeds
   {17, 42, 1729} and report mean ± stdev.

5. **One frontier model only.** We compare against Opus 4.7. Cortex may
   compare differently against Sonnet 4.6, GPT-5, etc. The case worth making
   is that this technique extends ANY model effectively past its native
   context — not specifically that it beats Opus.

6. **Reformulator failure mode is silent.** If the LM Studio reformulator
   call fails (timeout, schema rejection), `recall_verbatim_inline` falls
   back to the raw query for embedding. Falls open, doesn't error. But on
   MRCR queries specifically, this means lower retrieval quality without a
   warning bubbling up to the scoring layer.

7. **Inline embedding cost scales with cold-group count.** For an L row
   with ~1,500 atomic groups, that's a 1,501-input batched embedding call.
   At LM Studio's nomic throughput (~2,000/s on a 4090), that's ~750ms —
   amortized fine. Will be visible if scaling to longer-than-1M conversations.

## Pre-existing pilot results (for diff)

| run | cortex_9b lenient mean | what changed vs prior |
|-----|------------------------|----------------------|
| v1  | 0.143                  | initial; surfaced 3 bugs in virtualize |
| v2  | 0.542                  | bug fixes — short-circuit on natural fit, reasoning_content fallback, cold-summary cap bump (`0354150`, `95d4537`) |
| **v3** | **1.000**           | inline verbatim recall + query reformulation (`210ed41`) |

The v1→v2 work (3 commits, `5b969e5..0354150`) was a strict no-harm fix
package. It made cortex match raw_9b on rows that fit naturally but did not
extend the model's capability on L rows. v3 is the first time cortex
extends capability — it lets the 9B model produce content from a 1M-char
conversation that the 9B model couldn't have seen in its native context.

## Reproduction

```bash
docker compose up -d
.venv/Scripts/python.exe -m timegraph.storage.schema --apply
.venv/Scripts/python.exe -c "import asyncio; from timegraph.storage.qdrant_client import ensure_collections; asyncio.run(ensure_collections())"

# LM Studio must have qwen/qwen3.5-9b + text-embedding-nomic-embed-text-v1.5
# loaded and PINNED (keep loaded).

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

PYTHONPATH=src .venv/Scripts/python.exe bench/pilot_cortex/run.py \
  --seed 42 --per-bucket 10 --out results/pilot_cortex/scale30_v3.json
```

## What this does not claim

- It does not claim cortex makes a 9B model "as good as" Opus in general.
  MRCR is a verbatim-retrieval benchmark. Cortex gives the 9B model the
  exact text it needs to reproduce; the 9B model still has to reproduce it
  cleanly. For tasks that require reasoning over a long context (rather
  than retrieving from it), this technique does not apply.

- It does not claim cortex is faster than a frontier model. Opus runs the
  L row in ~25 s. Cortex on 9B runs it in 35–50 s. Faster than raw 9B
  (130–175 s), but not faster than Opus.

- It does not claim "infinite context." It claims "effectively unlimited
  context for retrieval-shaped tasks via in-conversation inline retrieval."
  Reasoning tasks, tool-use chains, and cross-message inference are
  separate problems with separate solutions in the cortex roadmap.
