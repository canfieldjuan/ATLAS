"""Standalone ``DatabasePool`` wrapper for the LLM-infrastructure package.

Slimmer than ``atlas_brain.storage.database`` -- just the methods the
scaffolded modules call:

  - ``is_initialized`` property
  - ``initialize()`` / ``close()`` lifecycle
  - delegate ``fetchrow`` / ``fetch`` / ``execute`` / ``acquire`` to the
    underlying ``asyncpg.Pool``

Configuration via environment variables (matches atlas_brain's
``DatabaseConfig`` for compatibility):

  ATLAS_DB_HOST       (default: localhost)
  ATLAS_DB_PORT       (default: 5432)
  ATLAS_DB_DATABASE   (default: atlas)
  ATLAS_DB_USER       (default: atlas)
  ATLAS_DB_PASSWORD   (required)
  ATLAS_DB_MIN_SIZE   (default: 2)
  ATLAS_DB_MAX_SIZE   (default: 10)

If ``ATLAS_DB_ENABLED`` is set to a falsy string, ``initialize()``
becomes a no-op so importing the scaffold cannot crash environments
without a configured database.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger("atlas.storage.database")


def _falsy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in ("0", "false", "no", "off")


class DatabasePool:
    """Thin wrapper around ``asyncpg.Pool`` for standalone use.

    Lazy-initialized on first ``await pool.initialize()``. ``is_initialized``
    is False until that completes successfully.
    """

    def __init__(self) -> None:
        self._pool: Any = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized and self._pool is not None

    async def initialize(self) -> None:
        if self._initialized:
            return

        if _falsy(os.environ.get("ATLAS_DB_ENABLED", "true")):
            logger.info("Database persistence disabled via ATLAS_DB_ENABLED")
            return

        try:
            import asyncpg
        except ImportError as exc:  # pragma: no cover - only triggers without dep
            raise RuntimeError(
                "asyncpg is required for the standalone database pool"
            ) from exc

        host = os.environ.get("ATLAS_DB_HOST", "localhost")
        port = int(os.environ.get("ATLAS_DB_PORT", "5432"))
        database = os.environ.get("ATLAS_DB_DATABASE", "atlas")
        user = os.environ.get("ATLAS_DB_USER", "atlas")
        password = os.environ.get("ATLAS_DB_PASSWORD", "")
        min_size = int(os.environ.get("ATLAS_DB_MIN_SIZE", "2"))
        max_size = int(os.environ.get("ATLAS_DB_MAX_SIZE", "10"))

        logger.info(
            "Initializing standalone database pool (host=%s port=%d db=%s)",
            host,
            port,
            database,
        )
        self._pool = await asyncpg.create_pool(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            min_size=min_size,
            max_size=max_size,
        )
        self._initialized = True

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._initialized = False

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Any]:
        if not self.is_initialized:
            raise RuntimeError(
                "DatabasePool is not initialized -- call await pool.initialize() first"
            )
        async with self._pool.acquire() as conn:
            yield conn

    async def fetchrow(self, query: str, *args: Any) -> Any:
        if not self.is_initialized:
            raise RuntimeError("DatabasePool is not initialized")
        return await self._pool.fetchrow(query, *args)

    async def fetch(self, query: str, *args: Any) -> Any:
        if not self.is_initialized:
            raise RuntimeError("DatabasePool is not initialized")
        return await self._pool.fetch(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        if not self.is_initialized:
            raise RuntimeError("DatabasePool is not initialized")
        return await self._pool.fetchval(query, *args)

    async def execute(self, query: str, *args: Any) -> Any:
        if not self.is_initialized:
            raise RuntimeError("DatabasePool is not initialized")
        return await self._pool.execute(query, *args)


_db_pool: Optional[DatabasePool] = None


def get_db_pool() -> DatabasePool:
    """Return the process-wide ``DatabasePool`` singleton (lazy-init)."""
    global _db_pool
    if _db_pool is None:
        _db_pool = DatabasePool()
    return _db_pool
