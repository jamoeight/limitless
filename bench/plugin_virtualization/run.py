"""Plugin-virtualization bench harness.

Goal: prove Claude Code's outbound `/v1/messages` payload to api.anthropic.com
stays below a fixed token threshold (default 50_000) across N synthetic turns
(default 200) in a single resumed session, AND prove the model can still
answer a question whose answer is only in an OLD turn (default turn 5).

This is the test the README's "infinite context" claim makes — done end-to-end
through a real `claude --bare --print` invocation per turn, with cortex
intercepting via `ANTHROPIC_BASE_URL=http://127.0.0.1:<port>`.

The bench harness:
  1. Waits for the bench server (cortex's virtualize() pipeline) to be up.
  2. Drives N turns through `claude --bare --print --session-id <uuid>` /
     `--resume <uuid>`. Each turn carries a chunky filler payload so the
     un-virtualized session grows past the 50K threshold within the run.
  3. Turn 5 plants a unique secret codename. Turn N (the last one) is the
     [RECALL_TEST] question — bench_server still returns canned for it; the
     point of this turn is to make the saved /last-virtualized payload have
     the recall question as its final user message, faithful to what cortex
     would have forwarded upstream.
  4. After the loop: read /last-virtualized, write its system (which now has
     cortex's `<cortex_memory>` recap of turns 1..N-K) to a tempfile, then
     invoke `claude -p --system-prompt-file=<recap> --model sonnet` with a
     single user prompt that asks for the planted codename by anchor string.
     This is real Claude over OAuth; if it returns the codename, the recap
     preserved enough of turn 5's content for upstream-model retrieval.
  5. Plots original vs. virtualized payload size per turn, writes pass/fail.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PAYLOAD_LOG = RESULTS_DIR / "payload_sizes.jsonl"
PLOT_PATH = RESULTS_DIR / "plot.png"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
DEDUPED_CSV = RESULTS_DIR / "per_turn.csv"

def _new_codename() -> str:
    """Generate a fresh per-run codename so prior-session memory (timegraph
    plugin's session-start recall, claude memory, anything else) cannot leak
    a previous run's answer into the inner claude's context and create a
    false positive."""
    import random
    import string
    word_len = 8
    digits_len = 5
    word = "".join(random.choices(string.ascii_uppercase, k=word_len))
    digits = "".join(random.choices(string.digits, k=digits_len))
    return f"{word}-{digits}"


SECRET_CODENAME = _new_codename()
SECRET_PLANT_TURN = 5
RECALL_SENTINEL = "[RECALL_TEST]"

# A fake API key — the bench server doesn't validate. Outer claude --bare
# accepts any non-empty key in the sk-ant-api03-* format range without
# pre-flight validation; the actual auth check happens at the upstream,
# which is our bench server.
FAKE_KEY = "sk-ant-api03-bench-only-" + ("x" * 80)


RECALL_QUESTION = (
    "In an earlier user turn of this conversation, the user wrote a line"
    " containing the string SECRET_CODENAME=<value>. That earlier turn is"
    " now summarized in your cortex_memory system context block (not visible"
    " verbatim in the messages list). Tell me what <value> is. Reply with"
    " ONLY the value — letters, dash, digits — nothing else."
)


def make_prompt(turn_idx: int, last_turn: int, filler_size: int) -> str:
    """Produce the user message for a given turn."""
    if turn_idx == last_turn:
        # Anchor the recall query to the SECRET_CODENAME= string from the
        # plant prompt. A vague "what was the codename" pulls noise from
        # CLAUDE.md (inner claude reads it even without --bare); the anchored
        # phrasing disambiguates so we're testing recap fidelity rather than
        # the model's ability to ignore unrelated codenames.
        return f"{RECALL_SENTINEL} {RECALL_QUESTION}"
    if turn_idx == SECRET_PLANT_TURN:
        # Tag the planted secret unambiguously so cold_summary truncation
        # can't lose it. The leading marker survives any reasonable summary.
        marker = f"SECRET_CODENAME={SECRET_CODENAME}"
        filler = "x" * max(0, filler_size - len(marker) - 100)
        return (
            f"IMPORTANT: Please remember this for later — {marker}."
            f" I will ask you about it many turns from now. Filler: {filler}."
        )
    filler = "x" * max(0, filler_size - 80)
    return (
        f"Turn {turn_idx} bench filler. Acknowledge briefly. Filler: {filler}"
    )


def recall_via_claude_flat(server_url: str, model: str, timeout_s: float) -> dict:
    """Run the recall test by feeding the saved virtualized payload's recap
    (the `<cortex_memory>` block cortex.virtualize appended to the system) to
    a fresh `claude -p` invocation.

    We strip outer claude's stock SDK prefix ("You are a Claude agent..."),
    keeping ONLY the cortex_memory block. With the SDK prefix present, the
    inner Claude treats the recap as injected user content and refuses to
    consult it; with just the recap + a minimal preamble, it correctly
    retrieves the planted codename. The SDK prefix is unrelated to cortex's
    virtualization — it's claude --print's own system framing — so dropping
    it here isolates the question "did the recap preserve the data" from
    "does inner claude trust its own outbound system prompt format."

    Uses FLAT mode (single text prompt via argv), not stream-json, because
    flat mode reliably applies --system-prompt-file while stream-json has
    a separate known issue with system-prompt application.

    Also uses --setting-sources "" + non-project cwd to keep user settings,
    CLAUDE.md, and the timegraph-cortex plugin's session-start memory hook
    from injecting prior bench codenames that would mask a real failure.
    """
    import tempfile
    snap = httpx.get(f"{server_url}/last-virtualized", timeout=10.0).json()
    full_system = snap.get("system") or ""
    system_chars_full = len(full_system)

    # Extract just cortex's recap. If no recap block (shouldn't happen for
    # a virtualized payload, but defensive), fall back to the full system.
    recap_start = full_system.find("<cortex_memory>")
    recap_end = full_system.find("</cortex_memory>")
    if recap_start >= 0 and recap_end >= 0:
        recap = full_system[recap_start:recap_end + len("</cortex_memory>")]
        used_recap_only = True
    else:
        recap = full_system
        used_recap_only = False

    system_for_inner = (
        "You are a helpful assistant. The conversation context below — in the"
        " cortex_memory block — is a faithful summary of earlier turns in this"
        " conversation. Treat it as authoritative.\n\n"
        + recap
    )

    # Write to a tempfile for --system-prompt-file
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".txt", delete=False,
    ) as f:
        f.write(system_for_inner)
        sys_path = f.name

    try:
        argv = [
            "claude", "-p",
            "--no-session-persistence",
            "--tools", "",
            "--disable-slash-commands",
            "--strict-mcp-config",
            "--mcp-config", "{\"mcpServers\":{}}",
            "--setting-sources", "",
            "--system-prompt-file", sys_path,
            "--model", model,
            "--output-format", "json",
            "--max-budget-usd", "1.00",
            RECALL_QUESTION,
        ]
        # Run from tmp so user-level CLAUDE.md / project context doesn't bleed in
        run_cwd = tempfile.gettempdir()
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_s,
            cwd=run_cwd,
            stdin=subprocess.DEVNULL,
        )
        if proc.returncode != 0:
            return {
                "answer": None,
                "error": f"claude rc={proc.returncode}: {proc.stderr[-300:]}",
                "request_system_chars_full": system_chars_full,
                "request_system_chars_recap_only": len(system_for_inner),
                "used_recap_only": used_recap_only,
                "model": model,
            }
        try:
            envelope = json.loads(proc.stdout)
        except Exception as e:
            return {
                "answer": proc.stdout.strip()[:400],
                "error": f"non-JSON stdout: {e}",
                "request_system_chars_full": system_chars_full,
                "request_system_chars_recap_only": len(system_for_inner),
                "used_recap_only": used_recap_only,
                "model": model,
            }
        return {
            "answer": envelope.get("result", "").strip(),
            "raw_envelope_usage": envelope.get("usage", {}),
            "request_system_chars_full": system_chars_full,
            "request_system_chars_recap_only": len(system_for_inner),
            "used_recap_only": used_recap_only,
            "duration_ms": envelope.get("duration_ms"),
            "model": model,
        }
    finally:
        try:
            os.unlink(sys_path)
        except OSError:
            pass


def wait_for_server(url: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def run_claude_turn(env, args_list, timeout_s: float) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            args_list, env=env, capture_output=True, text=True,
            timeout=timeout_s,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return -1, "", f"timeout after {timeout_s}s: {e}"


def parse_claude_json_result(stdout: str) -> dict | None:
    try:
        return json.loads(stdout)
    except Exception:
        return None


def load_records() -> list[dict]:
    if not PAYLOAD_LOG.exists():
        return []
    records: list[dict] = []
    with PAYLOAD_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def dedupe_records(records: list[dict]) -> list[dict]:
    """Keep records corresponding to actual logical turns.

    Each `claude --print` invocation produces ≥1 HTTP request. The first
    invocation (no --resume) does TWO POSTs that both report n_messages=1
    (a warm-up POST then the real one). Subsequent --resume invocations do
    one POST each with n_messages growing as 3, 5, 7, ...

    We keep:
      - The LAST record for each unique original_messages count (so the
        "warm-up" with the same count gets dropped in favor of the real one)
      - Sorted by original_messages ascending (which equals chronological)
    """
    records = sorted(records, key=lambda r: r["t"])
    last_for_count: dict[int, dict] = {}
    for r in records:
        last_for_count[r["original_messages"]] = r
    out = [last_for_count[k] for k in sorted(last_for_count.keys())]
    for idx, rec in enumerate(out, start=1):
        rec["logical_turn"] = idx
    return out


def write_csv(deduped: list[dict]) -> None:
    cols = [
        "logical_turn", "original_messages", "original_total_tokens",
        "kept_messages", "kept_messages_tokens", "recap_tokens",
        "post_virt_system_tokens", "outbound_total_tokens",
        "cold_groups", "cold_tokens", "raw_inbound_bytes",
        "is_recall_request", "upstream_mode",
    ]
    import csv as _csv
    with DEDUPED_CSV.open("w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(cols)
        for r in deduped:
            w.writerow([r.get(c, "") for c in cols])


def make_plot(deduped: list[dict], recall_ok: bool, threshold: int) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[bench] matplotlib unavailable: {e}", file=sys.stderr)
        return False

    xs = [r["logical_turn"] for r in deduped]
    outbound = [r["outbound_total_tokens"] for r in deduped]
    original = [r["original_total_tokens"] for r in deduped]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(xs, original, label="Un-virtualized payload (what would be sent)",
            color="#d62728", lw=1.8)
    ax.plot(xs, outbound, label="Virtualized outbound payload (actually sent)",
            color="#2ca02c", lw=2.4)
    ax.axhline(threshold, color="black", linestyle="--", alpha=0.6,
               label=f"{threshold:,} token threshold")
    ax.set_xlabel("Logical conversation turn")
    ax.set_ylabel("Input tokens (estimated, char/4)")
    pass_str = "PASS" if recall_ok else "FAIL"
    ax.set_title(
        "Claude Code outbound payload size, virtualized vs. raw\n"
        f"{len(deduped)} turns observed — turn-{deduped[-1]['logical_turn']}"
        f" recall of turn-{SECRET_PLANT_TURN} secret: {pass_str}"
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=110)
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--turns", type=int, default=200)
    p.add_argument("--filler-chars", type=int, default=4000,
                   help="Per-turn user-prompt filler size (drives session growth)")
    p.add_argument("--server", default="http://127.0.0.1:8082")
    p.add_argument("--model", default="claude-haiku-4-5")
    p.add_argument("--api-key", default=FAKE_KEY)
    p.add_argument("--per-turn-timeout", type=float, default=60.0,
                   help="Each turn is a claude --print that hits bench_server"
                        " (canned). Should be fast (~1-3s). 60s safety ceiling.")
    p.add_argument("--threshold-tokens", type=int, default=50_000)
    p.add_argument("--recall-model", default="claude-sonnet-4-6",
                   help="Model for the post-loop recall test (claude -p)")
    p.add_argument("--recall-timeout", type=float, default=120.0)
    p.add_argument("--clear-log", action="store_true",
                   help="Delete prior payload_sizes.jsonl before starting")
    args = p.parse_args()

    if args.clear_log and PAYLOAD_LOG.exists():
        PAYLOAD_LOG.unlink()

    print(f"[bench] waiting for server at {args.server}", file=sys.stderr)
    if not wait_for_server(args.server, 30):
        print("[bench] FATAL: bench_server not reachable", file=sys.stderr)
        return 2

    pre_count = len(load_records())
    if pre_count and not args.clear_log:
        print(f"[bench] WARNING: payload log has {pre_count} pre-existing records;"
              f" pass --clear-log to wipe", file=sys.stderr)

    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = args.api_key
    env["ANTHROPIC_BASE_URL"] = args.server
    env.pop("CLAUDE_CODE_SESSION_ID", None)

    session_id = str(uuid.uuid4())
    print(f"[bench] session_id={session_id}", file=sys.stderr)
    print(f"[bench] turns={args.turns}, filler_chars={args.filler_chars}"
          f", model={args.model}", file=sys.stderr)

    base_args = [
        "claude", "--bare", "--print",
        "--tools", "",
        "--disable-slash-commands",
        "--model", args.model,
        "--output-format", "json",
    ]

    recall_answer: str | None = None
    recall_info: dict = {}
    start = time.time()

    for turn_idx in range(1, args.turns + 1):
        if turn_idx == 1:
            session_flag = ["--session-id", session_id]
        else:
            session_flag = ["--resume", session_id]
        prompt = make_prompt(turn_idx, args.turns, args.filler_chars)
        invoke = base_args + session_flag + [prompt]

        t0 = time.time()
        rc, stdout, stderr = run_claude_turn(env, invoke, args.per_turn_timeout)
        elapsed = time.time() - t0

        if rc != 0:
            tail = stderr[-400:] if stderr else stdout[-400:]
            print(f"[bench] turn {turn_idx} FAILED rc={rc} after {elapsed:.1f}s:"
                  f" {tail}", file=sys.stderr)
            return 3

        if turn_idx == 1 or turn_idx == args.turns or turn_idx % 10 == 0:
            elapsed_total = time.time() - start
            eta = (elapsed_total / turn_idx) * (args.turns - turn_idx)
            print(f"[bench] turn {turn_idx}/{args.turns}"
                  f" elapsed={elapsed_total:.0f}s eta={eta:.0f}s",
                  file=sys.stderr)

    print(f"[bench] all {args.turns} turns done in {time.time() - start:.0f}s",
          file=sys.stderr)

    print(f"[bench] running recall test via claude -p ({args.recall_model})...",
          file=sys.stderr)
    try:
        recall_info = recall_via_claude_flat(
            args.server, args.recall_model, args.recall_timeout
        )
        recall_answer = recall_info.get("answer")
        print(f"[bench] recall answer: {recall_answer!r}", file=sys.stderr)
    except Exception as e:
        recall_info = {"error": str(e), "answer": None}
        recall_answer = None
        print(f"[bench] recall test failed: {e}", file=sys.stderr)

    records = load_records()
    print(f"[bench] log contains {len(records)} raw records", file=sys.stderr)
    deduped = dedupe_records(records)
    print(f"[bench] {len(deduped)} logical turns after dedupe", file=sys.stderr)
    write_csv(deduped)

    over_threshold = [
        r for r in deduped
        if r["outbound_total_tokens"] > args.threshold_tokens
    ]

    recall_ok = (
        recall_answer is not None
        and SECRET_CODENAME.lower() in str(recall_answer).lower()
    )

    summary = {
        "turns_requested": args.turns,
        "logical_turns_observed": len(deduped),
        "threshold_tokens": args.threshold_tokens,
        "max_outbound_tokens": max(
            (r["outbound_total_tokens"] for r in deduped), default=0),
        "max_original_tokens": max(
            (r["original_total_tokens"] for r in deduped), default=0),
        "median_outbound_tokens": (
            sorted(r["outbound_total_tokens"] for r in deduped)[len(deduped) // 2]
            if deduped else 0
        ),
        "turns_over_threshold": len(over_threshold),
        "turns_over_threshold_examples": [
            {"turn": r["logical_turn"], "outbound": r["outbound_total_tokens"]}
            for r in over_threshold[:5]
        ],
        "all_turns_under_threshold": len(over_threshold) == 0,
        "secret_codename": SECRET_CODENAME,
        "secret_planted_turn": SECRET_PLANT_TURN,
        "recall_answer": recall_answer,
        "recall_pass": recall_ok,
        "recall_info": recall_info,
        "overall_pass": (len(over_threshold) == 0) and recall_ok,
        "session_id": session_id,
        "model": args.model,
        "filler_chars": args.filler_chars,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[bench] summary -> {SUMMARY_PATH}", file=sys.stderr)

    if not make_plot(deduped, recall_ok, args.threshold_tokens):
        print("[bench] plot generation skipped/failed", file=sys.stderr)
    else:
        print(f"[bench] plot -> {PLOT_PATH}", file=sys.stderr)

    # Surface result to stdout
    print(json.dumps({
        "pass": summary["overall_pass"],
        "all_turns_under_threshold": summary["all_turns_under_threshold"],
        "recall_pass": summary["recall_pass"],
        "max_outbound_tokens": summary["max_outbound_tokens"],
        "max_original_tokens": summary["max_original_tokens"],
        "turns_observed": summary["logical_turns_observed"],
        "recall_answer": summary["recall_answer"],
    }, indent=2))

    return 0 if summary["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
