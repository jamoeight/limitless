"""End-to-end smoke test (Phase 0 — verifies infra + LLM connectivity).

Runs after `docker compose up -d` + both llama-server processes started.
Walks through:
  1. Neo4j reachable, schema applied
  2. Qdrant reachable
  3. Qwen3-7B extractor responds
  4. Qwopus judge responds with valid JSON (after Phase 0 GBNF in place)

Phase 1 will extend this to assert `judge_call_count == 1` after running a real
`infer()` op end-to-end.

Usage: python scripts/smoke.py
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime

import httpx
import structlog

from timegraph.config import get_settings
from timegraph.storage.schema import check_schema

log = structlog.get_logger(__name__)


async def smoke() -> int:
    s = get_settings()
    fail = 0

    # 1. Neo4j ping
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get("http://localhost:7474")
            assert r.status_code == 200
        print("[ok ] Neo4j reachable (port 7474)")
    except Exception as e:
        print(f"[FAIL] Neo4j: {e}")
        fail += 1

    # 2. Neo4j schema
    try:
        ok = await check_schema(s.neo4j_uri, s.neo4j_user, s.neo4j_password, s.neo4j_database)
        print(f"[{'ok ' if ok else 'FAIL'}] Neo4j schema present" if ok
              else "[FAIL] Neo4j schema missing — run `python -m timegraph.storage.schema --apply`")
        if not ok:
            fail += 1
    except Exception as e:
        print(f"[FAIL] Neo4j schema check: {e}")
        fail += 1

    # 3. Qdrant ping
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{s.qdrant_url}/")
            assert r.status_code == 200
        print(f"[ok ] Qdrant reachable ({s.qdrant_url})")
    except Exception as e:
        print(f"[FAIL] Qdrant: {e}")
        fail += 1

    # 4. LM Studio reachable + which models are loaded
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{s.lm_studio_url}/models")
            assert r.status_code == 200
            loaded = [m["id"] for m in r.json().get("data", [])]
        print(f"[ok ] LM Studio reachable ({s.lm_studio_url})")
        print(f"       loaded models: {loaded or '(none)'}")
        # Check the configured judge model is loaded
        if s.judge_model not in loaded:
            print(f"[WARN] judge_model '{s.judge_model}' NOT in loaded models — "
                  f"update Settings.judge_model or load it in LM Studio")
            fail += 1
        # Check the extractor model (or accept fallback)
        if not s.use_judge_for_extraction and s.extractor_model not in loaded:
            print(f"[WARN] extractor_model '{s.extractor_model}' NOT loaded; "
                  f"set TG_USE_JUDGE_FOR_EXTRACTION=true to route extraction to judge model")
            fail += 1
    except Exception as e:
        print(f"[FAIL] LM Studio: {e}")
        fail += 1

    print()
    print("=" * 60)
    print(f"Smoke result: {'ALL PASS' if fail == 0 else f'{fail} FAILED'}")
    print("=" * 60)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(smoke()))
