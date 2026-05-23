"""Host-owned shared admission for Content Ops ingestion imports."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from .storage.database import get_db_pool


PoolProvider = Callable[[], Any | Awaitable[Any]]

CONTENT_OPS_IMPORT_ADVISORY_LOCK_NAMESPACE = 1_096_047_443


class ContentOpsPostgresImportAdmissionGate:
    """Bound non-dry-run Content Ops imports across Atlas app processes."""

    def __init__(
        self,
        *,
        pool_provider: PoolProvider,
        max_concurrency: int,
        namespace: int = CONTENT_OPS_IMPORT_ADVISORY_LOCK_NAMESPACE,
    ) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self.max_concurrency = max_concurrency
        self._pool_provider = pool_provider
        self._namespace = namespace
        self._pool: Any | None = None
        self._connection: Any | None = None
        self._slot: int | None = None

    async def acquire(self) -> bool:
        """Acquire one advisory-lock slot, returning False when full."""

        if self._connection is not None:
            return True

        pool = await _resolve_pool(self._pool_provider)
        connection = await _maybe_await(pool.acquire())
        try:
            for slot in range(self.max_concurrency):
                acquired = await _maybe_await(
                    connection.fetchval(
                        "SELECT pg_try_advisory_lock($1::int, $2::int)",
                        self._namespace,
                        slot,
                    )
                )
                if bool(acquired):
                    self._pool = pool
                    self._connection = connection
                    self._slot = slot
                    return True
        except BaseException:
            await _release_connection(pool, connection)
            raise

        await _release_connection(pool, connection)
        return False

    async def release(self) -> None:
        """Release a held advisory-lock slot and return its connection."""

        if self._connection is None:
            return

        pool = self._pool
        connection = self._connection
        slot = self._slot
        self._pool = None
        self._connection = None
        self._slot = None

        unlock_error: BaseException | None = None
        try:
            unlocked = await _maybe_await(
                connection.fetchval(
                    "SELECT pg_advisory_unlock($1::int, $2::int)",
                    self._namespace,
                    slot,
                )
            )
            if not bool(unlocked):
                raise RuntimeError("Content Ops import advisory lock was not held")
        except Exception as exc:
            unlock_error = exc
            await _close_connection(connection)
        finally:
            if pool is not None:
                await _release_connection(pool, connection)
        if unlock_error is not None:
            raise unlock_error


def build_content_ops_import_admission_gate(
    *,
    max_concurrency: int,
    pool_provider: PoolProvider = get_db_pool,
) -> ContentOpsPostgresImportAdmissionGate:
    """Build a fresh host admission gate for one import request."""

    return ContentOpsPostgresImportAdmissionGate(
        pool_provider=pool_provider,
        max_concurrency=max_concurrency,
    )


async def _resolve_pool(provider: PoolProvider) -> Any:
    pool = await _maybe_await(provider())
    if pool is None:
        raise RuntimeError("Content Ops import admission database pool is unavailable")
    return pool


async def _release_connection(pool: Any, connection: Any) -> None:
    await _maybe_await(pool.release(connection))


async def _close_connection(connection: Any) -> None:
    close = getattr(connection, "close", None)
    if callable(close):
        await _maybe_await(close())


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value
