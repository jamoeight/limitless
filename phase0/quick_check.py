"""Phase 0 fast pre-flight — verifies LM Studio + response_format integration.

Runs in ~30s. One fake judge call + one fake extraction call. Validates:
  - LM Studio reachable
  - The configured judge_model is loaded
  - response_format(json_schema, strict) works against LM Studio's API
  - Output parses into our pydantic types

If this passes, the Phase 0 evals (eval_judge.py, eval_extraction.py) will
work as soon as you have the hand-labeled datasets. If this fails, fix the
integration before building datasets.

Usage:  python phase0/quick_check.py
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime

# Force UTF-8 stdout on Windows (default cp1252 chokes on emoji / Unicode arrows).
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

import httpx

from timegraph.config import get_settings
from timegraph.llm.extractor import ExtractorClient
from timegraph.llm.judge import JudgeClient
from timegraph.types import ConflictTriple


async def check_lm_studio_reachable() -> tuple[bool, list[str]]:
    s = get_settings()
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{s.lm_studio_url}/models")
            r.raise_for_status()
            return True, [m["id"] for m in r.json().get("data", [])]
    except Exception as e:
        print(f"[FAIL] cannot reach LM Studio at {s.lm_studio_url}: {e}")
        return False, []


async def check_judge() -> bool:
    print()
    print("[1/2] Judge call (Qwopus3.6-27B + response_format strict)…")
    client = JudgeClient()
    try:
        conflicts = [
            ConflictTriple(
                e1_fact_id="f_001",
                e2_fact_id="f_002",
                reason="e1 says 'Alice lives in Boston (valid from 2026-01-01)'; "
                       "e2 says 'Alice lives in Seattle (valid from 2026-04-15)'.",
            )
        ]
        out = await client.judge_conflicts(
            query="Where does Alice live now?",
            conflicts=conflicts,
            attestations=[{"by": "user", "fact_id": "f_002", "text": "Yes she moved in April"}],
            source_episodes_truncated=[
                "[Jan 2026] Alice: I just moved to Boston for a new job.",
                "[Apr 2026] Alice: Update — moved again, in Seattle now.",
            ],
        )
        print(f"  resolution:    {out.resolution.value}")
        print(f"  reason:        {out.reason!r}")
        print(f"  confidence:    {out.confidence:.2f}")
        print(f"  call_count:    {out.call_count}  (assertion: must be 1)")
        print(f"  latency:       {out.latency_ms:.0f}ms ({out.latency_ms/1000:.1f}s)")
        print(f"  thinking_len:  {len(out.thinking)} chars")
        if out.thinking:
            preview = out.thinking[:200].replace("\n", " ")
            print(f"  thinking[:200]: {preview}...")
        assert out.call_count == 1, "call_count must be 1 — load-bearing assertion"
        print("  [PASS] JUDGE OK")
        return True
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        return False
    finally:
        await client.close()


async def check_extractor() -> bool:
    print()
    print("[2/2] Extractor call (response_format strict)…")
    s = get_settings()
    if s.use_judge_for_extraction:
        print(f"  (routing extraction to judge model: {s.judge_model})")
    else:
        print(f"  (using extractor model: {s.extractor_model})")
    client = ExtractorClient()
    try:
        facts, latency_ms = await client.extract_facts(
            episode_content=(
                "User said: I work at Anthropic on the Claude team. My email is "
                "alice@anthropic.com. I prefer dark roast coffee."
            ),
            event_time=datetime(2026, 5, 20, 14, 0, 0),
            session_id="quick_check",
            source="quick_check",
        )
        print(f"  facts extracted: {len(facts)}")
        for f in facts:
            print(f"    - ({f.subject}, {f.predicate}, {f.object})  conf={f.confidence:.2f}")
        print(f"  latency:         {latency_ms:.0f}ms ({latency_ms/1000:.1f}s)")
        print("  [PASS] EXTRACTOR OK")
        return True
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        return False
    finally:
        await client.close()


async def main() -> int:
    print("=" * 64)
    print("Phase 0 quick check — LM Studio + response_format integration")
    print("=" * 64)

    reachable, loaded = await check_lm_studio_reachable()
    if not reachable:
        print()
        print("Open LM Studio and start the local server on port 1234.")
        return 1
    s = get_settings()
    print(f"[ok ] LM Studio reachable at {s.lm_studio_url}")
    print(f"      models loaded: {loaded}")
    if s.judge_model not in loaded:
        print(f"[FAIL] judge_model '{s.judge_model}' not loaded.")
        print(f"       Available: {loaded}")
        print(f"       Either load it in LM Studio or set TG_JUDGE_MODEL=<actual id>")
        return 1

    fails = 0
    if not await check_judge():
        fails += 1
    if not await check_extractor():
        fails += 1

    print()
    print("=" * 64)
    if fails == 0:
        print("ALL PASS — Phase 0 integration is wired correctly.")
        print("Next: build datasets (Tasks #17, #18), then run full evals.")
    else:
        print(f"{fails} CHECK(S) FAILED — fix before building datasets.")
    print("=" * 64)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
