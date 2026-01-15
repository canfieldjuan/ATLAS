"""
Database connection pool management for Atlas Brain.

Uses asyncpg for high-performance async PostgreSQL access.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import asyncpg

from .config import db_settings

logger = logging.getLogger("atlas.storage.database")


class DatabasePool:
    """
    Manages the asyncpg connection pool.

    Provides:
    - Lazy initialization
    - Connection pooling
    - Graceful shutdown
    """

    def __init__(self):
        self._pool: Optional[asyncpg.Pool] = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if the pool is initialized."""
        return self._initialized and self._pool is not None

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if self._initialized:
            logger.debug("Database pool already initialized")
            return

        if not db_settings.enabled:
            logger.info("Database persistence is disabled")
            return

        logger.info(
            "Initializing database pool (host=%s, port=%d, db=%s)",
            db_settings.host,
            db_settings.port,
            db_settings.database,
        )

        try:
            self._pool = await asyncpg.create_pool(
                host=db_settings.host,
                port=db_settings.port,
                database=db_settings.database,
                user=db_settings.user,
                password=db_settings.password,
                min_size=db_settings.min_pool_size,
                max_size=db_settings.max_pool_size,
                timeout=db_settings.connect_timeout,
                command_timeout=db_settings.command_timeout,
            )
            self._initialized = True
            logger.info(
                "Database pool initialized (min=%d, max=%d)",
                db_settings.min_pool_size,
                db_settings.max_pool_size,
            )
        except Exception as e:
            logger.error("Failed to initialize database pool: %s", e)
            raise

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            logger.info("Closing database pool")
            await self._pool.close()
            self._pool = None
            self._initialized = False
            logger.info("Database pool closed")

    async def acquire(self) -> asyncpg.Connection:
        """Acquire a connection from the pool."""
        if not self.is_initialized:
            raise RuntimeError("Database pool not initialized. Call initialize() first.")
        return await self._pool.acquire()

    async def release(self, connection: asyncpg.Connection) -> None:
        """Release a connection back to the pool."""
        if self._pool is not None:
            await self._pool.release(connection)

    async def execute(self, query: str, *args) -> str:
        """Execute a query and return the status."""
        if not self.is_initialized:
            raise RuntimeError("Database pool not initialized")
        return await self._pool.execute(query, *args)

    async def fetch(self, query: str, *args) -> list:
        """Execute a query and return all rows."""
        if not self.is_initialized:
            raise RuntimeError("Database pool not initialized")
        return await self._pool.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Execute a query and return a single row."""
        if not self.is_initialized:
            raise RuntimeError("Database pool not initialized")
        return await self._pool.fetchrow(query, *args)

    async def fetchval(self, query: str, *args):
        """Execute a query and return a single value."""
        if not self.is_initialized:
            raise RuntimeError("Database pool not initialized")
        return await self._pool.fetchval(query, *args)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[asyncpg.Connection]:
        """
        Acquire a connection with an active transaction.

        Usage:
            async with pool.transaction() as conn:
                await conn.execute("INSERT INTO ...")
                await conn.execute("UPDATE ...")
                # Commits on success, rolls back on exception

        Yields:
            asyncpg.Connection with active transaction
        """
        if not self.is_initialized:
            raise RuntimeError("Database pool not initialized")

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                yield conn


# Global pool instance
_db_pool: Optional[DatabasePool] = None


def get_db_pool() -> DatabasePool:
    """Get or create the global database pool."""
    global _db_pool
    if _db_pool is None:
        _db_pool = DatabasePool()
    return _db_pool


async def init_database() -> None:
    """Initialize the database pool (call from app startup)."""
    pool = get_db_pool()
    await pool.initialize()


async def close_database() -> None:
    """Close the database pool (call from app shutdown)."""
    pool = get_db_pool()
    await pool.close()
