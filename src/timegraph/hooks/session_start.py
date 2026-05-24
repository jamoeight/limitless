"""Claude Code SessionStart hook — primes the new session with cross-session memory.

Fires on `startup`, `resume`, `clear`, and `compact`. Each source has a tailored
inject:

  startup / resume  → "from your prior work in this directory" primer.
                       Top facts for this group_id, semantic search anchored on
                       the cwd basename so the most project-relevant memory floats up.

  compact           → "recovered from compaction" recap. After Claude Code
                       summarizes context, this re-injects the top facts that
                       were almost certainly in the summarized region — making
                       compaction *lossless* from the user's perspective.

  clear             → no inject. The user explicitly cleared; respect that.

Output: writes a `hookSpecificOutput.additionalContext` JSON document to stdout,
which Claude Code merges into the next prompt. Fails open: on any error or empty
recall, exits 0 with no output and the session proceeds unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

# Default to the LM-Studio-free path for plugin installs. Must run before any
# timegraph.ops import — those instantiate Settings which reads env once.
os.environ.setdefault("TG_JUDGE_BACKEND", "claude_cli")
os.environ.setdefault("TG_JUDGE_CLAUDE_MODEL", "haiku")

from timegraph.project_id import derive_group_id  # noqa: E402

PRIMER_K = int(os.environ.get("TG_HOOK_PRIMER_K", "8"))
PRIMER_BUDGET_TOKENS = int(os.environ.get("TG_HOOK_PRIMER_BUDGET", "1500"))
COMPACT_K = int(os.environ.get("TG_HOOK_COMPACT_K", "12"))
COMPACT_BUDGET_TOKENS = int(os.environ.get("TG_HOOK_COMPACT_BUDGET", "2500"))
CORTEX_HOST = os.environ.get("CORTEX_HOST", "127.0.0.1")
CORTEX_PORT = int(os.environ.get("CORTEX_PORT", "8080"))
CORTEX_HEALTH_URL = f"http://{CORTEX_HOST}:{CORTEX_PORT}/health"
CORTEX_STARTUP_WAIT_S = float(os.environ.get("CORTEX_HOOK_STARTUP_WAIT_S", "8"))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--group-id", default=None)
    args, _unknown = p.parse_known_args(argv)
    return args


def _attr(item, key, default=None):
    if hasattr(item, key):
        return getattr(item, key)
    if isinstance(item, dict):
        return item.get(key, default)
    return default


def _emit(additional_context: str | None) -> None:
    if not additional_context:
        return
    out = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": additional_context,
        }
    }
    sys.stdout.write(json.dumps(out))
    sys.stdout.flush()


def _timegraph_home() -> Path:
    return Path(os.environ.get("TIMEGRAPH_HOME", str(Path.home() / ".timegraph"))).expanduser()


def _append_log(line: str) -> None:
    try:
        home = _timegraph_home()
        home.mkdir(parents=True, exist_ok=True)
        with (home / "cortex.log").open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} session_start: {line}\n")
    except Exception:
        pass


def _is_cortex_healthy(url: str = CORTEX_HEALTH_URL, timeout_s: float = 0.5) -> bool:
    try:
        with urlopen(url, timeout=timeout_s) as resp:
            return 200 <= resp.status < 300
    except (OSError, URLError, ValueError):
        return False


def _cortex_env() -> dict[str, str]:
    home = _timegraph_home()
    env = os.environ.copy()
    defaults = {
        "CORTEX_HOST": CORTEX_HOST,
        "CORTEX_PORT": str(CORTEX_PORT),
        "CORTEX_DEFAULT_PROVIDER": "anthropic",
        "CORTEX_USE_CLAUDE_CLI_PROVIDER": "true",
        "CORTEX_ENABLE_AUTO_INGEST": "true",
        "CORTEX_ENABLE_VIRTUALIZATION": "true",
        "CORTEX_ENABLE_VERBATIM_RECALL": "true",
        "CORTEX_ENABLE_QUERY_REFORMULATION": "false",
        "CORTEX_UPSTREAM_CONTEXT_LIMIT": "50000",
        "CORTEX_LAST_K_SPANS": "4",
        "CORTEX_VERBATIM_RECALL_K": "24",
        "CORTEX_HEADER_LOG": str(home / "cortex_headers.jsonl"),
        "CLAUDE_CODE_DISABLE_AUTO_UPDATER": "1",
    }
    for key, value in defaults.items():
        env.setdefault(key, value)
    return env


def _ensure_cortex() -> None:
    if os.environ.get("CORTEX_PLUGIN_AUTOSTART", "true").lower() in {"0", "false", "no", "off"}:
        return
    if _is_cortex_healthy():
        return

    exe = shutil.which("cortex-serve")
    if not exe:
        _append_log("cortex-serve not found on PATH; proxy autostart skipped")
        return

    home = _timegraph_home()
    home.mkdir(parents=True, exist_ok=True)
    log_path = home / "cortex.log"
    try:
        log_f = log_path.open("ab")
    except OSError as e:
        _append_log(f"could not open cortex log: {e}")
        return

    kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": log_f,
        "stderr": subprocess.STDOUT,
        "env": _cortex_env(),
        "cwd": str(Path.cwd()),
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        kwargs["start_new_session"] = True

    try:
        subprocess.Popen([exe], **kwargs)
    except Exception as e:  # noqa: BLE001
        _append_log(f"failed to spawn cortex-serve: {type(e).__name__}: {e}")
        return
    finally:
        try:
            log_f.close()
        except OSError:
            pass

    deadline = time.time() + CORTEX_STARTUP_WAIT_S
    while time.time() < deadline:
        if _is_cortex_healthy(timeout_s=0.75):
            _append_log(f"cortex-serve started on {CORTEX_HEALTH_URL}")
            return
        time.sleep(0.25)
    _append_log(f"spawned cortex-serve but {CORTEX_HEALTH_URL} did not become healthy")


async def _recall(query: str, group_id: str, k: int, budget: int) -> list:
    from timegraph.ops.graph_query import graph_query
    from timegraph.types import GraphQueryIn

    payload = GraphQueryIn(
        query=query,
        scope=group_id,
        mode="nodes",
        k=k,
        budget_tokens=budget,
    )
    result = await graph_query(payload)
    return getattr(result, "results", None) or []


def _format_facts(facts: list, heading: str) -> str | None:
    lines: list[str] = [f"## {heading}"]
    for f in facts:
        subj = _attr(f, "subject") or ""
        pred = _attr(f, "predicate") or ""
        obj = _attr(f, "object") or ""
        if not (subj and pred and obj):
            continue
        valid_at = _attr(f, "valid_at")
        meta = f"  _({str(valid_at)[:10]})_" if valid_at else ""
        lines.append(f"- **{subj}** {pred} **{obj}**{meta}")
    if len(lines) == 1:
        return None
    return "\n".join(lines)


def _query_for_cwd(cwd: str | None) -> str:
    if not cwd:
        return "project status"
    name = Path(cwd).name or "project"
    return f"{name} project state and recent work"


def main() -> None:
    args = _parse_args(sys.argv[1:])
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        _ensure_cortex()
        return

    _ensure_cortex()

    source = (payload.get("source") or "startup").lower()
    if source == "clear":
        return

    cwd = payload.get("cwd")
    group_id = args.group_id or derive_group_id(cwd)

    query = _query_for_cwd(cwd)
    if source == "compact":
        k, budget = COMPACT_K, COMPACT_BUDGET_TOKENS
        heading = "Memory recovered after compaction (timegraph)"
    elif source == "resume":
        k, budget = PRIMER_K, PRIMER_BUDGET_TOKENS
        heading = "Resuming — relevant memory from prior work in this directory"
    else:  # startup
        k, budget = PRIMER_K, PRIMER_BUDGET_TOKENS
        heading = "Memory from prior sessions in this directory"

    try:
        facts = asyncio.run(_recall(query, group_id, k, budget))
    except Exception:
        return

    ctx = _format_facts(facts, heading)
    _emit(ctx)


if __name__ == "__main__":
    main()
