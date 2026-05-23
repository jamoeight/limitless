"""`timegraph` CLI -- one-command setup + health check for the Claude Code plugin path.

Subcommands:
  init    -- bring up Neo4j + Qdrant, apply schema, init collections, prefetch
            the fastembed model. Idempotent: safe to re-run.
  status  -- health check Docker, Neo4j, Qdrant, fastembed cache, claude CLI,
            and whether the plugin is installed in ~/.claude/plugins/.

The CLI is designed for `pipx install timegraph-mcp` users. It does NOT touch
the repo-root docker-compose.yml; it bundles its own copy in
`src/timegraph/data/docker-compose.yml` and writes it to ~/.timegraph/ so the
backend can run independently of having the repo checked out.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.resources
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

DEFAULT_HOME = Path.home() / ".timegraph"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "dev_password_change_me"
QDRANT_URL = "http://localhost:6333"
DOCKER_READY_TIMEOUT_S = 120


# ---- helpers ----------------------------------------------------------


def _step(n: int, total: int, msg: str) -> None:
    sys.stdout.write(f"[{n}/{total}] {msg}".ljust(48))
    sys.stdout.flush()


def _ok(detail: str = "") -> None:
    sys.stdout.write(f"OK{(' ' + detail) if detail else ''}\n")
    sys.stdout.flush()


def _fail(detail: str) -> None:
    sys.stdout.write(f"FAIL\n  -> {detail}\n")
    sys.stdout.flush()


def _write_bundled_compose(home: Path) -> Path:
    """Copy the bundled docker-compose.yml from package data into `home`."""
    home.mkdir(parents=True, exist_ok=True)
    target = home / "docker-compose.yml"
    src = importlib.resources.files("timegraph") / "data" / "docker-compose.yml"
    target.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def _have_docker() -> tuple[bool, str]:
    """Return (ok, detail). Detail is the docker version string or the error."""
    try:
        r = subprocess.run(
            ["docker", "version", "--format", "{{.Client.Version}}"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return False, (r.stderr or r.stdout).strip() or "non-zero exit"
        return True, r.stdout.strip()
    except FileNotFoundError:
        return False, "docker not found on PATH"
    except subprocess.TimeoutExpired:
        return False, "docker version timed out"


def _docker_compose_up(compose_path: Path) -> tuple[bool, str]:
    """`docker compose -f <path> up -d --wait`. Returns (ok, detail)."""
    try:
        r = subprocess.run(
            ["docker", "compose", "-f", str(compose_path), "up", "-d", "--wait"],
            capture_output=True, text=True, timeout=DOCKER_READY_TIMEOUT_S,
        )
        if r.returncode != 0:
            return False, (r.stderr or r.stdout).strip()[:400]
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"docker compose up did not finish within {DOCKER_READY_TIMEOUT_S}s"


def _docker_compose_ps(compose_path: Path) -> list[dict]:
    """Return list of services with status. Empty list on error."""
    try:
        r = subprocess.run(
            ["docker", "compose", "-f", str(compose_path), "ps", "--format", "json"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return []
        lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
        out: list[dict] = []
        for ln in lines:
            try:
                obj = json.loads(ln)
                if isinstance(obj, list):
                    out.extend(obj)
                else:
                    out.append(obj)
            except json.JSONDecodeError:
                continue
        return out
    except FileNotFoundError:
        return []


async def _qdrant_ready() -> tuple[bool, str]:
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{QDRANT_URL}/readyz")
            if r.status_code == 200:
                return True, ""
            return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)[:200]


async def _neo4j_ready() -> tuple[bool, str]:
    try:
        from neo4j import AsyncGraphDatabase

        d = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        try:
            await d.verify_connectivity()
            return True, ""
        finally:
            await d.close()
    except Exception as e:
        return False, str(e)[:200]


def _have_claude_cli() -> tuple[bool, str]:
    p = shutil.which("claude") or shutil.which("claude.cmd") or shutil.which("claude.exe")
    if p:
        return True, p
    return False, "claude not on PATH (judge backend won't work)"


def _plugin_installed() -> tuple[bool, str]:
    """Best-effort check: look for timegraph-cortex under ~/.claude/plugins/."""
    base = Path.home() / ".claude" / "plugins"
    if not base.exists():
        return False, f"{base} does not exist"
    for sub in ("cache", "marketplaces"):
        candidate = base / sub
        if not candidate.exists():
            continue
        for path in candidate.rglob("plugin.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("name") == "timegraph-cortex":
                    return True, str(path.parent.parent)
            except Exception:
                continue
    return False, "not found under ~/.claude/plugins/"


def _fastembed_cached(model_name: str) -> tuple[bool, str]:
    """Heuristic: check the default fastembed cache dir for the model."""
    candidates = [
        Path(os.environ.get("FASTEMBED_CACHE_DIR", "")),
        Path.home() / ".cache" / "fastembed",
        Path.home() / "AppData" / "Local" / "fastembed" / "Cache",
    ]
    slug = model_name.replace("/", "--").lower()
    for c in candidates:
        if not c or not c.exists():
            continue
        for entry in c.iterdir():
            if slug in entry.name.lower():
                return True, str(entry)
    return False, "not yet downloaded (will pull on first embed call)"


# ---- init -------------------------------------------------------------


async def cmd_init(args: argparse.Namespace) -> int:
    home = Path(args.home).expanduser().resolve()
    total = 7

    _step(1, total, "Checking Docker")
    ok, detail = _have_docker()
    if not ok:
        _fail(detail)
        print("\nInstall Docker Desktop and re-run. https://docs.docker.com/desktop/", file=sys.stderr)
        return 1
    _ok(f"({detail})")

    _step(2, total, "Writing bundled docker-compose.yml")
    compose_path = _write_bundled_compose(home)
    _ok(f"({compose_path})")

    _step(3, total, "Starting Neo4j + Qdrant")
    t0 = time.time()
    ok, detail = _docker_compose_up(compose_path)
    if not ok:
        _fail(detail)
        return 1
    _ok(f"({time.time() - t0:.1f}s)")

    _step(4, total, "Verifying Neo4j")
    ok, detail = await _wait_for(_neo4j_ready, timeout=60)
    if not ok:
        _fail(detail)
        return 1
    _ok(f"({NEO4J_URI})")

    _step(5, total, "Verifying Qdrant")
    ok, detail = await _wait_for(_qdrant_ready, timeout=30)
    if not ok:
        _fail(detail)
        return 1
    _ok(f"({QDRANT_URL})")

    _step(6, total, "Applying schema + collections")
    try:
        from timegraph.config import get_settings
        from timegraph.storage.qdrant_client import ensure_collections
        from timegraph.storage.schema import apply_schema

        s = get_settings()
        if args.reset:
            await _reset_qdrant_collections(s)
        await apply_schema(s.neo4j_uri, s.neo4j_user, s.neo4j_password, s.neo4j_database)
        await ensure_collections()
    except Exception as e:
        _fail(f"{type(e).__name__}: {e}")
        return 1
    _ok()

    _step(7, total, "Prefetching fastembed model")
    if args.skip_model:
        _ok("(skipped)")
    else:
        try:
            from timegraph.config import get_settings
            from fastembed import TextEmbedding

            s = get_settings()
            TextEmbedding(model_name=s.embedder_model)
        except ImportError as e:
            _fail(f"fastembed not installed: {e}. Run `pip install -e .` or `pipx install timegraph-mcp`.")
            return 1
        except Exception as e:
            _fail(f"{type(e).__name__}: {e}")
            return 1
        _ok(f"({s.embedder_model})")

    print("\nBackends ready. Next steps:")
    print("  1) In Claude Code, install the plugin:")
    print("       /plugin marketplace add jamoeight/cortex-mcp")
    print("       /plugin install timegraph-cortex")
    print("  2) Restart Claude Code so hooks load.")
    print("  3) Open any project -- recall + ingest run on every turn automatically.")
    print(f"\nBackend state lives in {home}/data. Container names: timegraph_neo4j, timegraph_qdrant.")
    return 0


async def _wait_for(probe, timeout: float) -> tuple[bool, str]:
    """Poll `probe()` (async, returns (ok, detail)) until ok or timeout."""
    deadline = time.time() + timeout
    last_detail = "timed out"
    while time.time() < deadline:
        ok, detail = await probe()
        if ok:
            return True, detail
        last_detail = detail
        await asyncio.sleep(2)
    return False, last_detail


async def _reset_qdrant_collections(s) -> None:
    from timegraph.storage.qdrant_client import get_client

    client = await get_client()
    for name in (s.qdrant_episodes_collection, s.qdrant_facts_collection):
        try:
            await client.delete_collection(collection_name=name)
        except Exception:
            pass


# ---- status -----------------------------------------------------------


async def cmd_status(args: argparse.Namespace) -> int:
    home = Path(args.home).expanduser().resolve()
    compose_path = home / "docker-compose.yml"

    rows: list[tuple[str, bool, str]] = []

    ok, detail = _have_docker()
    rows.append(("Docker", ok, detail))

    if compose_path.exists():
        services = _docker_compose_ps(compose_path)
        if services:
            states = ", ".join(f"{s.get('Service')}={s.get('State')}" for s in services)
            rows.append(("Compose services", True, states))
        else:
            rows.append(("Compose services", False, "none running"))
    else:
        rows.append(("Compose file", False, "missing - run `timegraph init`"))

    ok, detail = await _neo4j_ready()
    rows.append(("Neo4j", ok, detail or NEO4J_URI))

    ok, detail = await _qdrant_ready()
    rows.append(("Qdrant", ok, detail or QDRANT_URL))

    ok, detail = _have_claude_cli()
    rows.append(("claude CLI (judge)", ok, detail))

    try:
        from timegraph.config import get_settings

        s = get_settings()
        ok, detail = _fastembed_cached(s.embedder_model)
        rows.append((f"fastembed cache ({s.embedder_model})", ok, detail))
    except Exception as e:
        rows.append(("fastembed cache", False, f"{type(e).__name__}: {e}"))

    ok, detail = _plugin_installed()
    rows.append(("Plugin installed (~/.claude/plugins/)", ok, detail))

    width = max(len(r[0]) for r in rows) + 2
    for name, ok, detail in rows:
        status = "OK  " if ok else "FAIL"
        print(f"{name.ljust(width)}{status}  {detail}")

    return 0 if all(ok for _, ok, _ in rows) else 1


# ---- entry ------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        prog="timegraph",
        description="One-command setup + health check for the timegraph-cortex Claude Code plugin.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    init_p = sub.add_parser("init", help="Bring up backends, apply schema, prefetch embedder model")
    init_p.add_argument("--home", default=str(DEFAULT_HOME),
                        help=f"Where to write compose file + data volumes (default: {DEFAULT_HOME})")
    init_p.add_argument("--reset", action="store_true",
                        help="Drop Qdrant collections before re-creating (data loss; use after embedder dim change)")
    init_p.add_argument("--skip-model", action="store_true",
                        help="Skip fastembed prefetch -- first ingest/recall will pay the download cost")

    status_p = sub.add_parser("status", help="Health check all backends and plugin install")
    status_p.add_argument("--home", default=str(DEFAULT_HOME),
                          help=f"Compose file location (default: {DEFAULT_HOME})")

    args = p.parse_args()

    if args.cmd == "init":
        sys.exit(asyncio.run(cmd_init(args)))
    elif args.cmd == "status":
        sys.exit(asyncio.run(cmd_status(args)))
    else:
        p.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
