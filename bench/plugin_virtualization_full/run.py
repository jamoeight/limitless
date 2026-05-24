"""Full plugin virtualization proof.

Drives a real Claude Code session through cortex on :8080 and asserts the
strong path: auto-ingest writes storage, cortex virtualizes at least one
request, outbound payloads stay below 50K input tokens, and a 500-character
paragraph planted near the start is recovered verbatim at turn 200.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RESULTS_DIR = Path(__file__).parent / "results"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
CSV_PATH = RESULTS_DIR / "per_turn.csv"
PLOT_PATH = RESULTS_DIR / "plot.png"
RUN_LOG_PATH = RESULTS_DIR / "run.log"

DEFAULT_SERVER = "http://127.0.0.1:8080"
DEFAULT_HEADER_LOG = Path.home() / ".timegraph" / "cortex_headers.jsonl"
DEFAULT_EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
DEFAULT_JUDGE_MODEL = "qwen/qwen3.5-9b"
SECRET_PLANT_TURN = 5
MIN_VERBATIM_MATCH_CHARS = 100


class BenchFailure(RuntimeError):
    pass


def _log(message: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}"
    print(f"[bench-full] {message}", file=sys.stderr)
    with RUN_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def _load_compose_rows(compose_file: Path | None) -> list[dict[str, Any]]:
    cmd = ["docker", "compose"]
    if compose_file is not None:
        cmd.extend(["-f", str(compose_file)])
    cmd.extend(["ps", "--format", "json"])
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    if proc.returncode != 0:
        raise BenchFailure(f"docker compose ps failed: {(proc.stderr or proc.stdout).strip()}")
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, list):
            rows.extend(x for x in obj if isinstance(x, dict))
        elif isinstance(obj, dict):
            rows.append(obj)
    return rows


def _check_docker_compose() -> None:
    compose_candidates: list[Path | None] = [None]
    home_compose = Path.home() / ".timegraph" / "docker-compose.yml"
    if home_compose.exists():
        compose_candidates.append(home_compose)

    rows: list[dict[str, Any]] = []
    checked: list[str] = []
    for compose_file in compose_candidates:
        checked.append(str(compose_file or ROOT / "docker-compose.yml"))
        rows = _load_compose_rows(compose_file)
        if rows:
            break
    if not rows:
        raise BenchFailure("docker compose ps returned no services; checked " + ", ".join(checked))
    needed = {"neo4j": False, "qdrant": False}
    details: list[str] = []
    for row in rows:
        service = str(row.get("Service") or row.get("Name") or "").lower()
        state = str(row.get("State") or "").lower()
        health = str(row.get("Health") or "").lower()
        details.append(f"{service}: state={state or '-'} health={health or '-'}")
        for key in needed:
            if key in service:
                needed[key] = "running" in state and health == "healthy"
    missing = [name for name, ok in needed.items() if not ok]
    if missing:
        raise BenchFailure(
            "docker compose services not healthy: "
            + ", ".join(missing)
            + " ("
            + "; ".join(details)
            + ")"
        )


def _check_lm_studio(embed_model: str, judge_model: str, require_judge: bool) -> None:
    try:
        resp = httpx.get("http://127.0.0.1:1234/v1/models", timeout=5.0)
    except Exception as e:  # noqa: BLE001
        raise BenchFailure(f"LM Studio /v1/models unreachable: {e}") from e
    if resp.status_code != 200:
        raise BenchFailure(f"LM Studio /v1/models returned HTTP {resp.status_code}")
    data = resp.json().get("data", [])
    ids = [str(item.get("id", "")) for item in data if isinstance(item, dict)]
    if not any(embed_model in model_id for model_id in ids):
        raise BenchFailure(f"LM Studio is missing embedder {embed_model!r}; models={ids}")
    if require_judge and not any(judge_model in model_id for model_id in ids):
        raise BenchFailure(f"LM Studio is missing judge/extractor {judge_model!r}; models={ids}")


def _check_static_prereqs(args: argparse.Namespace) -> None:
    base = _normalize_base_url(os.environ.get("ANTHROPIC_BASE_URL", ""))
    if base != args.server:
        raise BenchFailure(
            "ANTHROPIC_BASE_URL must be "
            f"{args.server}; got {os.environ.get('ANTHROPIC_BASE_URL')!r}"
        )
    if shutil.which("cortex-serve") is None:
        raise BenchFailure("cortex-serve is not on PATH; run `pipx install -e .` from repo root")
    if shutil.which("claude") is None:
        raise BenchFailure("claude is not on PATH")
    _check_docker_compose()
    _check_lm_studio(args.embed_model, args.judge_model, args.require_judge_model)


def _server_healthy(server: str) -> bool:
    try:
        resp = httpx.get(f"{server}/health", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


def _server_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CORTEX_HOST": "127.0.0.1",
            "CORTEX_PORT": "8080",
            "CORTEX_DEFAULT_PROVIDER": "anthropic",
            "CORTEX_USE_CLAUDE_CLI_PROVIDER": "true",
            "CORTEX_ENABLE_AUTO_INGEST": "true",
            "CORTEX_ENABLE_VIRTUALIZATION": "true",
            "CORTEX_ENABLE_VERBATIM_RECALL": "true",
            "CORTEX_ENABLE_QUERY_REFORMULATION": "false",
            "CORTEX_UPSTREAM_CONTEXT_LIMIT": str(args.threshold_tokens),
            "CORTEX_LAST_K_SPANS": str(args.last_k_spans),
            "CORTEX_VERBATIM_RECALL_K": str(args.verbatim_recall_k),
            "CORTEX_HEADER_LOG": str(args.header_log),
            "TG_EMBEDDER_BACKEND": "openai_compat",
            "TG_EMBEDDER_MODEL": args.embed_model,
            "TG_EMBEDDER_URL": "http://127.0.0.1:1234/v1",
            "TG_JUDGE_BACKEND": "lm_studio",
            "TG_JUDGE_MODEL": args.judge_model,
            "TG_EXTRACTOR_BACKEND": "lm_studio",
            "TG_EXTRACTOR_MODEL": args.judge_model,
            "CLAUDE_CODE_DISABLE_AUTO_UPDATER": "1",
        }
    )
    return env


def _ensure_cortex_via_session_start(args: argparse.Namespace) -> None:
    if _server_healthy(args.server):
        _log("cortex /health already responding")
        return

    payload = json.dumps({"source": "startup", "cwd": str(ROOT)})
    _log("cortex not healthy; invoking timegraph-hook-session-start to autostart it")
    proc = subprocess.run(
        ["timegraph-hook-session-start"],
        input=payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_server_env(args),
        cwd=ROOT,
        timeout=20,
    )
    if proc.returncode != 0:
        raise BenchFailure(f"timegraph-hook-session-start failed: {proc.stderr[-500:]}")

    deadline = time.time() + 30
    while time.time() < deadline:
        if _server_healthy(args.server):
            _log("cortex became healthy after SessionStart autostart")
            return
        time.sleep(1)
    raise BenchFailure("cortex did not become healthy on :8080 after SessionStart autostart")


def _paragraph(seed: str) -> str:
    text = (
        f"At marker {seed}, the archived field report says the violet compass was wrapped "
        "in canvas, placed beside the south window, and checked at 07:42 by Mara Ives. "
        "The cabinet key stayed under the blue ceramic bowl while the rain gauge, brass "
        "ruler, and folded station map remained untouched. Later, the crew logged that "
        "the west lantern blinked twice, the spare fuse was never used, and the final "
        "notebook page ended with the phrase clear harbor before sunrise."
    )
    while len(text) < 500:
        text += " The inventory clerk initialed the margin and left the paper flat."
    return text[:500]


def _filler_text(turn: int, target_chars: int) -> str:
    words = [
        "archive", "ledger", "harbor", "canvas", "signal", "survey", "lantern",
        "notebook", "cabinet", "window", "station", "compass", "margin", "index",
        "weather", "catalog", "parcel", "ruler", "morning", "coastal", "receipt",
        "folder", "registry", "sample", "garden", "workshop", "courtyard", "museum",
    ]
    parts: list[str] = []
    i = 0
    while len(" ".join(parts)) < target_chars:
        parts.append(words[(turn + i * 7) % len(words)])
        i += 1
    return " ".join(parts)[:target_chars]


def _make_prompt(turn: int, turns: int, filler_chars: int, anchor: str, paragraph: str) -> str:
    fact = (
        f"Continuity note {turn}: archive item FULLBENCH-{turn:03d} "
        f"has ledger value {turn * 17} and was reviewed by the catalog desk."
    )
    if turn == SECRET_PLANT_TURN:
        filler = _filler_text(turn, filler_chars)
        return (
            "I am building a continuity archive for a fictional field report.\n"
            f"{fact}\n\n"
            f"Anchor code: {anchor}\n"
            "Please keep this field report paragraph available for later citation:\n"
            f"{paragraph}\n"
            "End of field report paragraph.\n\n"
            "Please acknowledge that the archive note was received.\n"
            f"Background catalog text: {filler}"
        )
    if turn == turns:
        return (
            "For the continuity archive, please quote the field report paragraph associated "
            f"with anchor code {anchor} from earlier in this session. I need a direct "
            "quotation of the paragraph as written for citation."
        )
    filler = _filler_text(turn, filler_chars)
    return (
        "Please review this routine archive update.\n"
        f"{fact}\n"
        f"Background catalog text: {filler}\n"
        "A brief acknowledgement is enough."
    )


def _fake_api_key() -> str:
    return "sk-ant-api03-fullbench-" + secrets.token_hex(48)


def _group_id_for_api_key(api_key: str) -> str:
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return f"k:{digest[:24]}"


def _run_claude_turn(
    args: argparse.Namespace,
    env: dict[str, str],
    session_id: str,
    turn: int,
    prompt: str,
) -> dict[str, Any]:
    session_flag = ["--session-id", session_id] if turn == 1 else ["--resume", session_id]
    argv = [
        "claude",
        "--bare",
        "--print",
        "--tools",
        "",
        "--disable-slash-commands",
        "--model",
        args.model,
        "--output-format",
        "json",
        *session_flag,
        prompt,
    ]
    proc = subprocess.run(
        argv,
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=args.per_turn_timeout,
    )
    if proc.returncode != 0:
        raise BenchFailure(
            f"turn {turn} claude failed rc={proc.returncode}: {(proc.stderr or proc.stdout)[-800:]}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise BenchFailure(f"turn {turn} returned non-JSON stdout: {proc.stdout[-800:]}") from e


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _read_header_records(path: Path, offset: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        f.seek(offset)
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _dedupe_header_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_message_count: dict[int, dict[str, Any]] = {}
    for record in records:
        headers = record.get("headers") or {}
        raw_count = headers.get("x-cortex-original-messages")
        if raw_count is None:
            continue
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        by_message_count[count] = record
    turns = [by_message_count[k] for k in sorted(by_message_count)]
    for idx, record in enumerate(turns, start=1):
        record["logical_turn"] = idx
    return turns


def _header_int(record: dict[str, Any], name: str, default: int = 0) -> int:
    try:
        return int((record.get("headers") or {}).get(name, default))
    except (TypeError, ValueError):
        return default


def _header_bool(record: dict[str, Any], name: str) -> bool:
    return str((record.get("headers") or {}).get(name, "")).lower() == "true"


def _write_csv(turns: list[dict[str, Any]]) -> None:
    cols = [
        "logical_turn",
        "x-cortex-original-messages",
        "x-cortex-kept-messages",
        "x-cortex-original-tokens",
        "x-cortex-outbound-tokens",
        "x-cortex-recap-tokens",
        "x-cortex-cold-groups",
        "x-cortex-virtualized",
        "x-cortex-degraded",
    ]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for record in turns:
            headers = record.get("headers") or {}
            writer.writerow([record.get("logical_turn")] + [headers.get(c, "") for c in cols[1:]])


def _longest_common_substring(a: str, b: str) -> tuple[int, str]:
    if not a or not b:
        return 0, ""
    prev = [0] * (len(b) + 1)
    best_len = 0
    best_end = 0
    for i, ca in enumerate(a, start=1):
        curr = [0] * (len(b) + 1)
        for j, cb in enumerate(b, start=1):
            if ca == cb:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best_len:
                    best_len = curr[j]
                    best_end = i
        prev = curr
    return best_len, a[best_end - best_len : best_end]


async def _storage_counts(group_id: str) -> dict[str, int]:
    from qdrant_client.http import models as qm
    from timegraph.config import get_settings
    from timegraph.storage.neo4j_client import close_driver, get_session
    from timegraph.storage.qdrant_client import close_client, get_client

    settings = get_settings()
    facts = 0
    neo4j_episodes = 0
    qdrant_episodes = 0
    try:
        async with get_session() as session:
            res = await session.run(
                """
                CALL {
                    MATCH (e:Episode {group_id: $g})
                    RETURN count(e) AS episodes
                }
                CALL {
                    MATCH ()-[r:FACT]->()
                    WHERE r.group_id = $g
                    RETURN count(r) AS facts
                }
                RETURN episodes, facts
                """,
                g=group_id,
            )
            row = await res.single()
            if row:
                neo4j_episodes = int(row["episodes"] or 0)
                facts = int(row["facts"] or 0)

        client = await get_client()
        count = await client.count(
            collection_name=settings.qdrant_episodes_collection,
            count_filter=qm.Filter(
                must=[qm.FieldCondition(key="group_id", match=qm.MatchValue(value=group_id))]
            ),
            exact=True,
        )
        qdrant_episodes = int(count.count)
    finally:
        await close_client()
        await close_driver()
    return {
        "neo4j_episodes": neo4j_episodes,
        "neo4j_facts": facts,
        "qdrant_episodes": qdrant_episodes,
    }


async def _wait_for_storage_counts(group_id: str, timeout_s: float) -> dict[str, int]:
    deadline = time.time() + timeout_s
    last = {"neo4j_episodes": 0, "neo4j_facts": 0, "qdrant_episodes": 0}
    while time.time() < deadline:
        last = await _storage_counts(group_id)
        if last["qdrant_episodes"] >= 100 and last["neo4j_facts"] > 0:
            return last
        _log(
            "waiting for ingest: "
            f"qdrant_episodes={last['qdrant_episodes']} neo4j_facts={last['neo4j_facts']}"
        )
        await asyncio.sleep(5)
    return last


def _make_plot(turns: list[dict[str, Any]], threshold: int, recall_ok: bool) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        raise BenchFailure(f"matplotlib unavailable; cannot write required plot: {e}") from e

    xs = [r["logical_turn"] for r in turns]
    original = [_header_int(r, "x-cortex-original-tokens") for r in turns]
    outbound = [_header_int(r, "x-cortex-outbound-tokens") for r in turns]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(xs, original, label="Raw conversation payload", color="#c73e1d", linewidth=1.8)
    ax.plot(xs, outbound, label="Cortex outbound payload", color="#237a57", linewidth=2.2)
    ax.axhline(threshold, color="#111111", linestyle="--", linewidth=1.0, label="50K limit")
    ax.set_title(
        "timegraph-cortex full-pipeline virtualization\n"
        f"{len(turns)} turns, recall {'PASS' if recall_ok else 'FAIL'}"
    )
    ax.set_xlabel("Logical turn")
    ax.set_ylabel("Input tokens from X-Cortex-* response headers")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=120)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=200)
    parser.add_argument("--filler-chars", type=int, default=1800)
    parser.add_argument("--threshold-tokens", type=int, default=50_000)
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--per-turn-timeout", type=float, default=180.0)
    parser.add_argument("--ingest-wait-timeout", type=float, default=600.0)
    parser.add_argument("--last-k-spans", type=int, default=4)
    parser.add_argument("--verbatim-recall-k", type=int, default=24)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--require-judge-model", action="store_true")
    parser.add_argument("--header-log", type=Path, default=DEFAULT_HEADER_LOG)
    parser.add_argument("--clear-results", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.server = _normalize_base_url(args.server)
    args.header_log = args.header_log.expanduser()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.clear_results:
        for path in (SUMMARY_PATH, CSV_PATH, PLOT_PATH, RUN_LOG_PATH):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    started_at = time.time()
    failures: list[str] = []
    summary: dict[str, Any] = {
        "started_at": started_at,
        "turns_requested": args.turns,
        "threshold_tokens": args.threshold_tokens,
        "secret_planted_turn": SECRET_PLANT_TURN,
    }

    try:
        _check_static_prereqs(args)
        _ensure_cortex_via_session_start(args)
    except Exception as e:  # noqa: BLE001
        summary["preflight_error"] = str(e)
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"pass": False, "preflight_error": str(e)}, indent=2))
        return 2
    if args.preflight_only:
        summary["pass"] = True
        summary["preflight_only"] = True
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"pass": True, "preflight_only": True}, indent=2))
        return 0

    header_offset = _file_size(args.header_log)
    api_key = _fake_api_key()
    group_id = _group_id_for_api_key(api_key)
    session_id = str(uuid.uuid4())
    anchor = f"PAYLOAD_{secrets.token_hex(8)}"
    paragraph = _paragraph(anchor)
    recall_answer = ""
    turn_usages: list[dict[str, Any]] = []

    env = os.environ.copy()
    env.update(
        {
            "ANTHROPIC_BASE_URL": args.server,
            "ANTHROPIC_API_KEY": api_key,
            "CLAUDE_CODE_DISABLE_AUTO_UPDATER": "1",
        }
    )
    env.pop("CLAUDE_CODE_SESSION_ID", None)

    _log(f"session_id={session_id} group_id={group_id} anchor={anchor}")
    _log(f"running {args.turns} turns through claude --bare --print")
    try:
        for turn in range(1, args.turns + 1):
            prompt = _make_prompt(turn, args.turns, args.filler_chars, anchor, paragraph)
            t0 = time.time()
            envelope = _run_claude_turn(args, env, session_id, turn, prompt)
            usage = envelope.get("usage") or {}
            turn_usages.append(
                {
                    "turn": turn,
                    "duration_s": round(time.time() - t0, 3),
                    "usage": usage,
                }
            )
            if turn == args.turns:
                recall_answer = str(envelope.get("result") or "").strip()
            if turn == 1 or turn % 10 == 0 or turn == args.turns:
                elapsed = time.time() - started_at
                _log(f"turn {turn}/{args.turns} done elapsed={elapsed:.0f}s")
    except Exception as e:  # noqa: BLE001
        summary.update(
            {
                "session_id": session_id,
                "group_id": group_id,
                "anchor": anchor,
                "run_error": str(e),
                "turn_usages": turn_usages,
            }
        )
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"pass": False, "run_error": str(e)}, indent=2))
        return 3

    records = _read_header_records(args.header_log, header_offset)
    turns = _dedupe_header_records(records)
    _write_csv(turns)

    outbound = [_header_int(r, "x-cortex-outbound-tokens") for r in turns]
    original = [_header_int(r, "x-cortex-original-tokens") for r in turns]
    over_threshold = [
        {"turn": r["logical_turn"], "outbound_tokens": _header_int(r, "x-cortex-outbound-tokens")}
        for r in turns
        if _header_int(r, "x-cortex-outbound-tokens") >= args.threshold_tokens
    ]
    virtualized_seen = any(_header_bool(r, "x-cortex-virtualized") for r in turns)
    lcs_len, lcs_text = _longest_common_substring(paragraph, recall_answer)
    recall_ok = lcs_len >= MIN_VERBATIM_MATCH_CHARS

    _log("waiting for auto-ingest storage assertions")
    try:
        counts = asyncio.run(_wait_for_storage_counts(group_id, args.ingest_wait_timeout))
    except Exception as e:  # noqa: BLE001
        counts = {"neo4j_episodes": 0, "neo4j_facts": 0, "qdrant_episodes": 0}
        failures.append(f"storage count query failed: {type(e).__name__}: {e}")

    if len(turns) != args.turns:
        failures.append(f"expected {args.turns} header-backed turns, observed {len(turns)}")
    if max(original, default=0) <= args.threshold_tokens:
        failures.append(
            f"raw conversation never exceeded {args.threshold_tokens} tokens "
            f"(max_original={max(original, default=0)})"
        )
    if over_threshold:
        failures.append(f"{len(over_threshold)} turns had outbound tokens >= {args.threshold_tokens}")
    if counts["qdrant_episodes"] < 100:
        failures.append(f"Qdrant episodes count {counts['qdrant_episodes']} < 100")
    if counts["neo4j_facts"] <= 0:
        failures.append("Neo4j Fact count is 0")
    if not recall_ok:
        failures.append(
            f"recall answer did not contain a {MIN_VERBATIM_MATCH_CHARS}-char verbatim substring "
            f"(best={lcs_len})"
        )
    if not virtualized_seen:
        failures.append("no response carried X-Cortex-Virtualized:true")

    try:
        _make_plot(turns, args.threshold_tokens, recall_ok)
    except BenchFailure as e:
        failures.append(str(e))

    summary.update(
        {
            "pass": not failures,
            "failures": failures,
            "session_id": session_id,
            "group_id": group_id,
            "anchor": anchor,
            "planted_paragraph": paragraph,
            "recall_answer": recall_answer,
            "verbatim_match_chars": lcs_len,
            "verbatim_match_excerpt": lcs_text,
            "virtualized_seen": virtualized_seen,
            "logical_turns_observed": len(turns),
            "max_outbound_tokens": max(outbound, default=0),
            "max_original_tokens": max(original, default=0),
            "turns_over_threshold": over_threshold[:10],
            "storage_counts": counts,
            "header_log": str(args.header_log),
            "plot": str(PLOT_PATH),
            "per_turn_csv": str(CSV_PATH),
            "turn_usages": turn_usages,
            "duration_s": round(time.time() - started_at, 3),
        }
    )
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "pass": summary["pass"],
                "failures": failures,
                "max_outbound_tokens": summary["max_outbound_tokens"],
                "logical_turns_observed": len(turns),
                "qdrant_episodes": counts["qdrant_episodes"],
                "neo4j_facts": counts["neo4j_facts"],
                "verbatim_match_chars": lcs_len,
                "virtualized_seen": virtualized_seen,
                "summary": str(SUMMARY_PATH),
                "plot": str(PLOT_PATH),
            },
            indent=2,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
