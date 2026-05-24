"""Plugin-virtualization bench server.

A minimal Anthropic-compatible reverse proxy that:

  1. Receives `/v1/messages` from the outer `claude --bare --print` invocations
     of the bench harness (one POST per turn — claude resends the WHOLE session
     each time, which is exactly the payload we want to measure).

  2. Runs the request through `cortex.virtualize()` — the same code path that
     produced the README's MRCR 50/50 numbers. This shrinks the messages list
     to last-K verbatim + a recap block injected into the system prompt.

  3. Logs per-turn metrics (raw + virtualized token estimates, message counts,
     recap size, raw byte size of the wire payload) to a JSONL file the
     harness can plot.

  4. Persists the most recent virtualized payload to `last_virtualized.json`
     so the harness (or an inspector) can see what cortex would have forwarded.

  5. Returns a canned valid Anthropic response for EVERY turn. The 200-turn
     loop measures virtualization sizes only; the recall verification is run
     by the harness AFTER the loop as a single `claude -p` invocation against
     the saved virtualized payload (see run.py). This split is deliberate:
     the in-loop upstream is irrelevant to the size proof, and it avoids a
     stream-json system-prompt limitation in claude --print that surfaced
     when forwarding mid-loop through ClaudeCliProvider.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from cortex.canonical import CortexRequest
from cortex.config import get_cortex_settings
from cortex.translate.anthropic import from_anthropic_request
from cortex.virtualize import approx_tokens, virtualize

log = structlog.get_logger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PAYLOAD_LOG = RESULTS_DIR / "payload_sizes.jsonl"
LAST_VIRTUALIZED = RESULTS_DIR / "last_virtualized.json"

RECALL_SENTINEL = "[RECALL_TEST]"

# Bench-tuned settings: aggressive virtualization, no backend deps.
# - upstream_context_limit=50_000 keeps post-virt payload under the goal threshold.
# - cold_summary_max_chars_per_msg=320 truncates each cold-turn bullet to its
#   first ~80 tokens. The secret codename planted in turn 5 sits in the FIRST
#   80 chars of its prompt (`IMPORTANT: Please remember this for later —
#   SECRET_CODENAME=GOLDFISH-3491.` is ~75 chars), so the truncation preserves
#   it while preventing 200 turns × 2KB of bench filler from drowning the
#   recap in noise that demonstrably defeats model attention.
# - last_k_spans=4 keeps the most recent 4 atomic groups (user+assistant pairs)
#   verbatim in the messages list.
settings = get_cortex_settings().model_copy(update={
    "enable_virtualization": True,
    "enable_auto_ingest": False,
    "enable_verbatim_recall": False,
    "enable_query_reformulation": False,
    "last_k_spans": 4,
    "upstream_context_limit": 50_000,
    "safety_margin_tokens": 2_000,
    "cold_summary_max_chars_per_msg": 320,
})

app = FastAPI(title="plugin-virtualization-bench")

_turn_counter = {"n": 0}


def _record(payload: dict) -> None:
    with PAYLOAD_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def _persist_virtualized(req: CortexRequest, raw_bytes: int) -> None:
    """Dump the post-virtualization request so the harness/downstream can use it."""
    blob = {
        "model": req.model,
        "system": req.system,
        "messages": [m.model_dump() for m in req.messages],
        "max_tokens": req.max_tokens,
        "raw_inbound_bytes": raw_bytes,
    }
    LAST_VIRTUALIZED.write_text(json.dumps(blob, default=str, indent=2), encoding="utf-8")


def _last_user_text(req: CortexRequest) -> str:
    for msg in reversed(req.messages):
        if msg.role != "user":
            continue
        parts: list[str] = []
        for block in msg.content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        if parts:
            return "\n".join(parts)
    return ""


@app.get("/health")
async def health():
    return {"status": "ok", "turn": _turn_counter["n"]}


@app.get("/last-virtualized")
async def last_virtualized():
    if not LAST_VIRTUALIZED.exists():
        return JSONResponse(status_code=404, content={"error": "no requests yet"})
    return JSONResponse(content=json.loads(LAST_VIRTUALIZED.read_text(encoding="utf-8")))


@app.post("/v1/messages")
async def messages(request: Request):
    raw = await request.body()
    raw_bytes = len(raw)
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=400, content={"error": f"bad json: {e}"})

    try:
        cortex_req = from_anthropic_request(body)
    except (KeyError, ValueError, TypeError) as e:
        return JSONResponse(status_code=400, content={"error": f"bad anthropic body: {e}"})

    is_streaming = bool(body.get("stream", False))
    tools_serialized = body.get("tools", []) or []

    new_req, report = await virtualize(
        cortex_req,
        settings,
        recall_fn=None,
        verbatim_recall_fn=None,
        context_limit=settings.upstream_context_limit,
        tools_serialized=tools_serialized,
    )

    _turn_counter["n"] += 1
    turn_idx = _turn_counter["n"]

    system_t = approx_tokens(new_req.system or "")
    original_system_t = approx_tokens(cortex_req.system or "")
    outbound_total_t = report.kept_token_estimate + system_t
    original_total_t = report.original_token_estimate + original_system_t

    _persist_virtualized(new_req, raw_bytes)

    last_user = _last_user_text(cortex_req)
    is_recall = RECALL_SENTINEL in last_user

    record = {
        "t": time.time(),
        "turn": turn_idx,
        "model": cortex_req.model,
        "raw_inbound_bytes": raw_bytes,
        "original_messages": report.original_message_count,
        "original_messages_tokens": report.original_token_estimate,
        "original_system_tokens": original_system_t,
        "original_total_tokens": original_total_t,
        "kept_messages": report.kept_message_count,
        "kept_messages_tokens": report.kept_token_estimate,
        "recap_tokens": report.recap_token_estimate,
        "post_virt_system_tokens": system_t,
        "outbound_total_tokens": outbound_total_t,
        "cold_groups": report.cold_group_count,
        "cold_tokens": report.cold_token_estimate,
        "degraded": report.degraded,
        "notes": report.notes,
        "is_recall_request": is_recall,
        "upstream_mode": "canned",
        "stream": is_streaming,
    }
    _record(record)
    log.info("bench.turn", **{k: v for k, v in record.items() if k != "notes"})

    model = cortex_req.model
    # Every turn returns canned. Outer claude doesn't care what the assistant
    # text says; the session's messages list grows correctly either way. The
    # harness runs the actual recall test as a separate `claude -p` call
    # against the saved /last-virtualized payload.
    canned_text = f"ack turn {turn_idx}"
    if is_streaming:
        async def _gen():
            for ev in _canned_anthropic_sse_events(model, canned_text):
                yield ev
        return EventSourceResponse(_gen())
    return JSONResponse(content=_canned_anthropic_json(model, canned_text))


def _canned_anthropic_json(model: str, text: str) -> dict[str, Any]:
    return {
        "id": f"msg_bench_{uuid.uuid4().hex[:12]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": approx_tokens(text)},
    }


def _canned_anthropic_sse_events(model: str, text: str):
    msg_id = f"msg_bench_{uuid.uuid4().hex[:12]}"
    yield {"event": "message_start", "data": json.dumps({
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })}
    yield {"event": "content_block_start", "data": json.dumps({
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""},
    })}
    yield {"event": "ping", "data": json.dumps({"type": "ping"})}
    yield {"event": "content_block_delta", "data": json.dumps({
        "type": "content_block_delta", "index": 0,
        "delta": {"type": "text_delta", "text": text},
    })}
    yield {"event": "content_block_stop", "data": json.dumps({
        "type": "content_block_stop", "index": 0,
    })}
    yield {"event": "message_delta", "data": json.dumps({
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": approx_tokens(text)},
    })}
    yield {"event": "message_stop", "data": json.dumps({"type": "message_stop"})}


def main() -> None:
    port = int(os.environ.get("BENCH_PORT", "8082"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
