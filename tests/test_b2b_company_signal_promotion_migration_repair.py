"""Real-PostgreSQL proof for the named 297 company-signal source receipt.

These tests use only a disposable database supplied through
``ATLAS_MIGRATION_TEST_DATABASE_URL``. They create an empty test-owned schema
and never connect to Atlas's configured operational target.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

import pytest

asyncpg = pytest.importorskip("asyncpg")

from atlas_brain.storage.migrations import (  # noqa: E402
    PendingMigrationContentIntegrityError,
    run_migrations,
)
from atlas_brain.storage.migrations.reconciliation import (  # noqa: E402
    HISTORICAL_SOURCE_UNAVAILABLE,
    MIGRATION_297_B2B_COMPANY_SIGNAL_PROMOTION_RECONCILIATION,
    attest_known_historical_migration_reconciliations,
)


ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS = ROOT / "atlas_brain" / "storage" / "migrations"
DATABASE_URL_ENV = "ATLAS_MIGRATION_TEST_DATABASE_URL"
_PROBE_MIGRATION = "901_company_signal_promotion_attestation_probe"
_PROBE_TABLE = "company_signal_promotion_attestation_probe"


def _database_url_or_skip() -> str:
    database_url = os.environ.get(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")
    return database_url


def _quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


class _MigrationPool:
    """Use one disposable-database connection, matching the runner contract."""

    def __init__(self, conn) -> None:
        self._conn = conn

    async def acquire(self):
        return self._conn

    async def release(self, released) -> None:
        assert released is self._conn


def _test_catalog(tmp_path: Path) -> Path:
    """Create pending test SQL without inventing the unavailable 297 source."""
    catalog = tmp_path / "migrations"
    catalog.mkdir()
    (catalog / f"{_PROBE_MIGRATION}.sql").write_text(
        f"CREATE TABLE {_PROBE_TABLE} (id INTEGER PRIMARY KEY);\n"
    )
    return catalog


async def _prepare_receipt_ledger_schema(conn, schema: str) -> None:
    """Create only the test-owned NULL-digest ledger evidence."""
    record = MIGRATION_297_B2B_COMPANY_SIGNAL_PROMOTION_RECONCILIATION
    schema_ident = _quote_ident(schema)
    await conn.execute(f"CREATE SCHEMA {schema_ident}")
    await conn.execute(f"SET search_path TO {schema_ident}, public")
    await conn.execute("""
        CREATE TABLE schema_migrations (
            version INTEGER PRIMARY KEY,
            name VARCHAR(255) NOT NULL UNIQUE,
            content_sha256 VARCHAR(64),
            applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """)
    await conn.execute(
        """
        INSERT INTO schema_migrations (version, name, content_sha256, applied_at)
        VALUES ($1, $2, NULL, $3)
        """,
        record.migration_version,
        record.migration_name,
        record.observed_applied_at,
    )


async def _prepare_attestable_schema(
    conn,
    schema: str,
    *,
    index_predicate: str = "canonical_promotion_type IS NOT NULL",
) -> None:
    """Create the final metadata contract while keeping the table empty."""
    await _prepare_receipt_ledger_schema(conn, schema)
    await conn.execute(f"""
        CREATE TABLE b2b_company_signals (
            id UUID PRIMARY KEY,
            canonical_promotion_type TEXT
        );
        CREATE INDEX idx_b2b_company_signals_canonical_promotion_type
            ON b2b_company_signals (canonical_promotion_type)
            WHERE {index_predicate};
        """)


async def _prepare_leaf_partition_schema(conn, schema: str) -> None:
    """Create a leaf partition that shares the column/index shape, not identity."""
    await _prepare_receipt_ledger_schema(conn, schema)
    await conn.execute("""
        CREATE TABLE b2b_company_signals_parent (
            id INTEGER NOT NULL,
            canonical_promotion_type TEXT
        ) PARTITION BY RANGE (id);
        CREATE TABLE b2b_company_signals
            PARTITION OF b2b_company_signals_parent
            FOR VALUES FROM (0) TO (100);
        CREATE INDEX idx_b2b_company_signals_canonical_promotion_type
            ON b2b_company_signals (canonical_promotion_type)
            WHERE canonical_promotion_type IS NOT NULL;
        """)


async def _attestation(conn):
    record = MIGRATION_297_B2B_COMPANY_SIGNAL_PROMOTION_RECONCILIATION
    attestations = await attest_known_historical_migration_reconciliations(
        conn,
        sorted(MIGRATIONS.glob("*.sql")),
        candidate_names={record.migration_name},
    )
    assert len(attestations) == 1
    return attestations[0]


@pytest.mark.asyncio
async def test_297_receipt_attests_empty_real_catalog_without_company_signal_rows():
    database_url = _database_url_or_skip()
    schema = f"company_signal_297_attest_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_attestable_schema(conn, schema)

        attestation = await _attestation(conn)

        assert attestation.migration_name == (
            MIGRATION_297_B2B_COMPANY_SIGNAL_PROMOTION_RECONCILIATION.migration_name
        )
        assert attestation.ledger_version_matches_record
        assert attestation.b2b_company_signals_is_ordinary_table
        assert attestation.canonical_promotion_type_column_ready
        assert attestation.canonical_promotion_type_has_no_constraints
        assert attestation.canonical_promotion_type_partial_index_ready
        assert attestation.status == "attested"
        assert attestation.as_payload()["source_verification"] == (
            HISTORICAL_SOURCE_UNAVAILABLE
        )
        assert await conn.fetchval("SELECT COUNT(*) FROM b2b_company_signals") == 0
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {_quote_ident(schema)} CASCADE")
        await conn.close()


@pytest.mark.asyncio
async def test_297_receipt_rejects_leaf_partition_before_pending_sql(tmp_path: Path):
    database_url = _database_url_or_skip()
    schema = f"company_signal_297_partition_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_leaf_partition_schema(conn, schema)
        relation = await conn.fetchrow("""
            SELECT relkind, relispartition
            FROM pg_class
            WHERE oid = 'b2b_company_signals'::regclass
            """)
        assert relation["relkind"] == b"r"
        assert relation["relispartition"]

        attestation = await _attestation(conn)

        assert attestation.b2b_company_signals_is_ordinary_table is False
        assert attestation.status == "not_attested"
        with pytest.raises(
            PendingMigrationContentIntegrityError,
            match=(
                "missing_source="
                "297_b2b_company_signal_canonical_promotion_type"
            ),
        ):
            await run_migrations(
                _MigrationPool(conn),
                migrations_dir=_test_catalog(tmp_path),
                only={_PROBE_MIGRATION},
            )

        assert not await conn.fetchval(
            f"SELECT to_regclass('{_PROBE_TABLE}') IS NOT NULL"
        )
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM schema_migrations WHERE name = $1",
            _PROBE_MIGRATION,
        ) == 0
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {_quote_ident(schema)} CASCADE")
        await conn.close()


@pytest.mark.asyncio
async def test_297_receipt_refuses_bad_index_then_applies_pending_once_after_repair(
    tmp_path: Path,
):
    database_url = _database_url_or_skip()
    schema = f"company_signal_297_retry_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_attestable_schema(
            conn,
            schema,
            index_predicate="canonical_promotion_type IS NULL",
        )
        catalog = _test_catalog(tmp_path)
        attestation = await _attestation(conn)

        assert attestation.canonical_promotion_type_partial_index_ready is False
        assert attestation.status == "not_attested"
        with pytest.raises(
            PendingMigrationContentIntegrityError,
            match=(
                "missing_source="
                "297_b2b_company_signal_canonical_promotion_type"
            ),
        ):
            await run_migrations(
                _MigrationPool(conn),
                migrations_dir=catalog,
                only={_PROBE_MIGRATION},
            )
        assert not await conn.fetchval(
            f"SELECT to_regclass('{_PROBE_TABLE}') IS NOT NULL"
        )
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM schema_migrations WHERE name = $1",
            _PROBE_MIGRATION,
        ) == 0

        await conn.execute(
            "DROP INDEX idx_b2b_company_signals_canonical_promotion_type"
        )
        await conn.execute("""
            CREATE INDEX idx_b2b_company_signals_canonical_promotion_type
                ON b2b_company_signals (canonical_promotion_type)
                WHERE canonical_promotion_type IS NOT NULL
            """)
        assert (await _attestation(conn)).status == "attested"

        await run_migrations(
            _MigrationPool(conn),
            migrations_dir=catalog,
            only={_PROBE_MIGRATION},
        )
        await run_migrations(
            _MigrationPool(conn),
            migrations_dir=catalog,
            only={_PROBE_MIGRATION},
        )

        assert await conn.fetchval(
            f"SELECT to_regclass('{_PROBE_TABLE}') IS NOT NULL"
        )
        pending_digest = hashlib.sha256(
            (catalog / f"{_PROBE_MIGRATION}.sql").read_bytes()
        ).hexdigest()
        assert dict(
            await conn.fetchrow(
                """
                SELECT name, content_sha256
                FROM schema_migrations
                WHERE name = $1
                """,
                _PROBE_MIGRATION,
            )
        ) == {"name": _PROBE_MIGRATION, "content_sha256": pending_digest}
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM schema_migrations WHERE name = $1",
            _PROBE_MIGRATION,
        ) == 1
        assert await conn.fetchval("SELECT COUNT(*) FROM b2b_company_signals") == 0
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {_quote_ident(schema)} CASCADE")
        await conn.close()
