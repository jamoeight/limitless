"""Neo4j schema — constraints + indexes.

These are LOAD-BEARING for Phase 1: stage-1 of `infer(mode="conflict_set")`
performance depends entirely on `fact_valid_window` and `fact_subject_predicate`
indexes. The spec assumes ≤50ms stage-1 walk; measure early at 100K edges.

Run via: `python -m timegraph.storage.schema --apply`
Validate via: `python -m timegraph.storage.schema --check`
"""

from __future__ import annotations

import asyncio
import sys
from typing import Literal

from neo4j import AsyncGraphDatabase

CONSTRAINTS: list[str] = [
    "CREATE CONSTRAINT episode_id IF NOT EXISTS FOR (e:Episode) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT fact_id IF NOT EXISTS FOR ()-[r:FACT]-() REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT entity_name_group IF NOT EXISTS FOR (n:Entity) REQUIRE (n.name, n.group_id) IS UNIQUE",
]

INDEXES: list[str] = [
    # Stage-1 of infer(): subject+predicate is the primary filter for conflict detection.
    "CREATE INDEX fact_subject_predicate IF NOT EXISTS FOR ()-[r:FACT]-() ON (r.subject, r.predicate)",
    # Stage-1 also filters by valid_at window — needed for time-anchored queries.
    "CREATE INDEX fact_valid_window IF NOT EXISTS FOR ()-[r:FACT]-() ON (r.valid_at, r.invalid_at)",
    # Session/group lookup for fuse() and per-session filtering.
    "CREATE INDEX episode_session IF NOT EXISTS FOR (e:Episode) ON (e.session_id, e.group_id)",
    # Tier filtering — added in Phase 2 but free to create now.
    "CREATE INDEX fact_tier IF NOT EXISTS FOR ()-[r:FACT]-() ON (r.tier)",
]


async def apply_schema(uri: str, user: str, password: str, database: str = "neo4j") -> None:
    """Apply all constraints + indexes. Idempotent."""
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    async with driver.session(database=database) as session:
        for stmt in CONSTRAINTS:
            await session.run(stmt)
            print(f"[ok] {stmt[:80]}...")
        for stmt in INDEXES:
            await session.run(stmt)
            print(f"[ok] {stmt[:80]}...")
    await driver.close()


async def check_schema(uri: str, user: str, password: str, database: str = "neo4j") -> bool:
    """Verify all constraints + indexes exist. Returns True if all present."""
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    expected_constraints = {"episode_id", "fact_id", "entity_name_group"}
    expected_indexes = {
        "fact_subject_predicate",
        "fact_valid_window",
        "episode_session",
        "fact_tier",
    }
    async with driver.session(database=database) as session:
        c_res = await session.run("SHOW CONSTRAINTS YIELD name")
        present_constraints = {r["name"] for r in await c_res.data()}
        i_res = await session.run("SHOW INDEXES YIELD name WHERE name IS NOT NULL")
        present_indexes = {r["name"] for r in await i_res.data()}
    await driver.close()

    missing_c = expected_constraints - present_constraints
    missing_i = expected_indexes - present_indexes
    if missing_c:
        print(f"[MISSING constraints] {missing_c}")
    if missing_i:
        print(f"[MISSING indexes] {missing_i}")
    return not (missing_c or missing_i)


def main(action: Literal["apply", "check"]) -> int:
    # Late-import config so this module is importable without env setup.
    from timegraph.config import get_settings

    s = get_settings()
    if action == "apply":
        asyncio.run(apply_schema(s.neo4j_uri, s.neo4j_user, s.neo4j_password, s.neo4j_database))
        return 0
    elif action == "check":
        ok = asyncio.run(
            check_schema(s.neo4j_uri, s.neo4j_user, s.neo4j_password, s.neo4j_database)
        )
        return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("--apply", "--check"):
        print("Usage: python -m timegraph.storage.schema [--apply|--check]")
        sys.exit(2)
    sys.exit(main("apply" if sys.argv[1] == "--apply" else "check"))
