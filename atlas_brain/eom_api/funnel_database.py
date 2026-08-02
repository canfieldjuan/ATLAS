"""Dedicated CRM database pool for the slim EOM funnel profile."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from urllib.parse import urlsplit

import asyncpg

logger = logging.getLogger("atlas.eom.funnel_database")


class EOMFunnelDatabasePool:
    """Minimal asyncpg pool wrapper for the canonical CRM store."""

    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None
        self._dsn = ""
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized and self._pool is not None

    async def initialize(
        self,
        dsn: str,
        *,
        min_size: int = 1,
        max_size: int = 5,
        timeout: float = 10.0,
        command_timeout: float | None = 30.0,
    ) -> None:
        if self._initialized:
            logger.debug("EOM funnel database pool already initialized")
            return
        cleaned_dsn = dsn.strip()
        if not cleaned_dsn:
            raise RuntimeError(
                "ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING is required when "
                "ATLAS_EOM_FUNNEL_API_ENABLED=true"
            )
        logger.info("Initializing EOM funnel CRM pool (%s)", _target_label(cleaned_dsn))
        self._pool = await asyncpg.create_pool(
            dsn=cleaned_dsn,
            min_size=min_size,
            max_size=max_size,
            timeout=timeout,
            command_timeout=command_timeout,
        )
        self._dsn = cleaned_dsn
        self._initialized = True
        logger.info("EOM funnel CRM pool initialized")

    async def close(self) -> None:
        if self._pool is not None:
            logger.info("Closing EOM funnel CRM pool")
            await self._pool.close()
            self._pool = None
            self._dsn = ""
            self._initialized = False

    async def acquire(self) -> asyncpg.Connection:
        if not self.is_initialized:
            raise RuntimeError("EOM funnel database pool not initialized")
        return await self._pool.acquire()

    async def release(self, connection: asyncpg.Connection) -> None:
        if self._pool is not None:
            await self._pool.release(connection)

    async def execute(self, query: str, *args: Any) -> str:
        if not self.is_initialized:
            raise RuntimeError("EOM funnel database pool not initialized")
        return await self._pool.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> list:
        if not self.is_initialized:
            raise RuntimeError("EOM funnel database pool not initialized")
        return await self._pool.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        if not self.is_initialized:
            raise RuntimeError("EOM funnel database pool not initialized")
        return await self._pool.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        if not self.is_initialized:
            raise RuntimeError("EOM funnel database pool not initialized")
        return await self._pool.fetchval(query, *args)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[asyncpg.Connection]:
        if not self.is_initialized:
            raise RuntimeError("EOM funnel database pool not initialized")
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                yield connection

    @asynccontextmanager
    async def migration_pool(self) -> AsyncIterator[asyncpg.Pool]:
        if not self.is_initialized or not self._dsn:
            raise RuntimeError("EOM funnel database pool not initialized")
        pool = await asyncpg.create_pool(
            dsn=self._dsn,
            min_size=1,
            max_size=1,
            timeout=10.0,
            command_timeout=None,
        )
        try:
            yield pool
        finally:
            await pool.close()


_funnel_db_pool: EOMFunnelDatabasePool | None = None


def get_eom_funnel_db_pool() -> EOMFunnelDatabasePool:
    global _funnel_db_pool
    if _funnel_db_pool is None:
        _funnel_db_pool = EOMFunnelDatabasePool()
    return _funnel_db_pool


async def init_eom_funnel_database(dsn: str) -> None:
    await get_eom_funnel_db_pool().initialize(dsn)


async def close_eom_funnel_database() -> None:
    await get_eom_funnel_db_pool().close()


def _target_label(dsn: str) -> str:
    try:
        parsed = urlsplit(dsn)
    except ValueError:
        return "dsn=<connection-string>"
    database = parsed.path.lstrip("/") or "<database>"
    host = parsed.hostname or "connection-string"
    port = f":{parsed.port}" if parsed.port else ""
    return f"dsn={host}{port}/{database}"
