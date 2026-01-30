"""
Database connection pool for atlas_vision.

Uses asyncpg for async PostgreSQL access.
"""

import logging
from typing import Optional

import asyncpg

from .config import db_settings

logger = logging.getLogger("atlas.vision.storage.database")


class DatabasePool:
    """Manages the asyncpg connection pool."""

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
            return

        if not db_settings.enabled:
            logger.info("Database disabled")
            return

        logger.info(
            "Connecting to database: %s:%d/%s",
            db_settings.host,
            db_settings.port,
            db_settings.database,
        )

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
        logger.info("Database pool initialized")

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._initialized = False
            logger.info("Database pool closed")

    async def fetchrow(self, query: str, *args):
        """Execute query and fetch one row."""
        if not self._pool:
            raise RuntimeError("Database not initialized")
        return await self._pool.fetchrow(query, *args)

    async def fetch(self, query: str, *args):
        """Execute query and fetch all rows."""
        if not self._pool:
            raise RuntimeError("Database not initialized")
        return await self._pool.fetch(query, *args)

    async def execute(self, query: str, *args):
        """Execute query without returning results."""
        if not self._pool:
            raise RuntimeError("Database not initialized")
        return await self._pool.execute(query, *args)


# Global pool instance
_db_pool: Optional[DatabasePool] = None


def get_db_pool() -> DatabasePool:
    """Get the global database pool."""
    global _db_pool
    if _db_pool is None:
        _db_pool = DatabasePool()
    return _db_pool


async def init_database() -> None:
    """Initialize the database pool."""
    pool = get_db_pool()
    await pool.initialize()


async def close_database() -> None:
    """Close the database pool."""
    pool = get_db_pool()
    await pool.close()
