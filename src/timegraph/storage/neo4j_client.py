"""Async Neo4j driver wrapper.

Provides a single AsyncDriver shared across the process. Ops grab a session
via `get_session()`; the driver is built lazily on first use and closed on
shutdown.

Wave 1 surface:
  - `get_driver()` — singleton AsyncDriver
  - `get_session()` — async context manager yielding an AsyncSession
  - `close_driver()` — call on process shutdown
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from neo4j import AsyncGraphDatabase
from neo4j import AsyncDriver, AsyncSession

from timegraph.config import get_settings

_driver: AsyncDriver | None = None
_driver_lock = asyncio.Lock()


async def get_driver() -> AsyncDriver:
    """Lazy singleton. Safe under concurrent first-use."""
    global _driver
    if _driver is not None:
        return _driver
    async with _driver_lock:
        if _driver is None:
            s = get_settings()
            _driver = AsyncGraphDatabase.driver(
                s.neo4j_uri,
                auth=(s.neo4j_user, s.neo4j_password),
                # Silence the "unknown property key" warnings the server emits
                # when we project sparse properties (source_episode_id is NULL
                # on hand-added facts until add_episode wires it in Wave 2).
                notifications_min_severity="OFF",
            )
    return _driver


@asynccontextmanager
async def get_session(database: str | None = None) -> AsyncIterator[AsyncSession]:
    """Yield an AsyncSession bound to the configured database."""
    s = get_settings()
    driver = await get_driver()
    async with driver.session(database=database or s.neo4j_database) as session:
        yield session


async def close_driver() -> None:
    """Close the shared driver. Call on process shutdown."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
