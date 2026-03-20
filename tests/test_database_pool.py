from unittest.mock import AsyncMock

import pytest

from atlas_brain.storage.database import DatabasePool


@pytest.mark.asyncio
async def test_executemany_delegates_to_asyncpg_pool():
    pool = DatabasePool()
    backend_pool = AsyncMock()
    pool._pool = backend_pool
    pool._initialized = True

    rows = [("zendesk",), ("hubspot",)]
    await pool.executemany("INSERT INTO test_table(name) VALUES($1)", rows)

    backend_pool.executemany.assert_awaited_once_with(
        "INSERT INTO test_table(name) VALUES($1)",
        rows,
    )


@pytest.mark.asyncio
async def test_executemany_requires_initialized_pool():
    pool = DatabasePool()

    with pytest.raises(RuntimeError, match="Database pool not initialized"):
        await pool.executemany("INSERT INTO test_table(name) VALUES($1)", [("zendesk",)])
