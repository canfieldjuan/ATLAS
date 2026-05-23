from __future__ import annotations

import asyncio

import pytest

from atlas_brain._content_ops_import_admission import (
    CONTENT_OPS_IMPORT_ADVISORY_LOCK_NAMESPACE,
    ContentOpsPostgresImportAdmissionGate,
    build_content_ops_import_admission_gate,
)


class _Connection:
    def __init__(self, try_results: list[bool], *, unlock_result: bool = True) -> None:
        self.try_results = try_results
        self.unlock_result = unlock_result
        self.fetchval_calls: list[tuple[str, tuple[object, ...]]] = []
        self.closed = False

    async def fetchval(self, query: str, *args: object) -> bool:
        self.fetchval_calls.append((query, args))
        if "pg_try_advisory_lock" in query:
            if not self.try_results:
                raise AssertionError("Unexpected advisory lock attempt")
            return self.try_results.pop(0)
        if "pg_advisory_unlock" in query:
            return self.unlock_result
        raise AssertionError(f"Unexpected query: {query}")

    async def close(self) -> None:
        self.closed = True


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection
        self.acquire_count = 0
        self.released: list[_Connection] = []

    async def acquire(self) -> _Connection:
        self.acquire_count += 1
        return self.connection

    async def release(self, connection: _Connection) -> None:
        self.released.append(connection)


@pytest.mark.asyncio
async def test_import_admission_gate_acquires_first_available_slot() -> None:
    connection = _Connection([False, True])
    pool = _Pool(connection)
    gate = ContentOpsPostgresImportAdmissionGate(
        pool_provider=lambda: pool,
        max_concurrency=3,
    )

    acquired = await gate.acquire()

    assert acquired is True
    assert pool.acquire_count == 1
    assert pool.released == []
    assert connection.fetchval_calls == [
        (
            "SELECT pg_try_advisory_lock($1::int, $2::int)",
            (CONTENT_OPS_IMPORT_ADVISORY_LOCK_NAMESPACE, 0),
        ),
        (
            "SELECT pg_try_advisory_lock($1::int, $2::int)",
            (CONTENT_OPS_IMPORT_ADVISORY_LOCK_NAMESPACE, 1),
        ),
    ]

    await gate.release()

    assert pool.released == [connection]
    assert connection.fetchval_calls[-1] == (
        "SELECT pg_advisory_unlock($1::int, $2::int)",
        (CONTENT_OPS_IMPORT_ADVISORY_LOCK_NAMESPACE, 1),
    )


@pytest.mark.asyncio
async def test_import_admission_gate_denies_when_all_slots_are_full() -> None:
    connection = _Connection([False, False])
    pool = _Pool(connection)
    gate = ContentOpsPostgresImportAdmissionGate(
        pool_provider=lambda: pool,
        max_concurrency=2,
    )

    acquired = await gate.acquire()

    assert acquired is False
    assert pool.acquire_count == 1
    assert pool.released == [connection]
    assert connection.fetchval_calls == [
        (
            "SELECT pg_try_advisory_lock($1::int, $2::int)",
            (CONTENT_OPS_IMPORT_ADVISORY_LOCK_NAMESPACE, 0),
        ),
        (
            "SELECT pg_try_advisory_lock($1::int, $2::int)",
            (CONTENT_OPS_IMPORT_ADVISORY_LOCK_NAMESPACE, 1),
        ),
    ]


@pytest.mark.parametrize(
    "exception",
    [
        RuntimeError("database unavailable"),
        asyncio.CancelledError(),
    ],
)
@pytest.mark.asyncio
async def test_import_admission_gate_releases_connection_when_try_fails(
    exception: BaseException,
) -> None:
    class _FailingConnection(_Connection):
        async def fetchval(self, query: str, *args: object) -> bool:
            self.fetchval_calls.append((query, args))
            raise exception

    connection = _FailingConnection([])
    pool = _Pool(connection)
    gate = ContentOpsPostgresImportAdmissionGate(
        pool_provider=lambda: pool,
        max_concurrency=1,
    )

    with pytest.raises(type(exception)):
        await gate.acquire()

    assert pool.released == [connection]


@pytest.mark.asyncio
async def test_import_admission_gate_closes_connection_when_unlock_fails() -> None:
    connection = _Connection([True], unlock_result=False)
    pool = _Pool(connection)
    gate = ContentOpsPostgresImportAdmissionGate(
        pool_provider=lambda: pool,
        max_concurrency=1,
    )

    assert await gate.acquire() is True

    with pytest.raises(RuntimeError, match="advisory lock was not held"):
        await gate.release()

    assert connection.closed is True
    assert pool.released == [connection]


def test_import_admission_gate_rejects_invalid_capacity() -> None:
    with pytest.raises(ValueError, match="max_concurrency must be positive"):
        ContentOpsPostgresImportAdmissionGate(
            pool_provider=lambda: object(),
            max_concurrency=0,
        )


def test_import_admission_provider_returns_fresh_gate_per_request() -> None:
    gate_one = build_content_ops_import_admission_gate(
        max_concurrency=2,
        pool_provider=lambda: object(),
    )
    gate_two = build_content_ops_import_admission_gate(
        max_concurrency=2,
        pool_provider=lambda: object(),
    )

    assert gate_one is not gate_two
    assert gate_one.max_concurrency == 2
    assert gate_two.max_concurrency == 2
