"""Real-PostgreSQL proof for the named 022b source-rename receipt.

These tests use only a disposable database supplied through
``ATLAS_MIGRATION_TEST_DATABASE_URL``.  They create an empty test-owned schema
and never connect to Atlas's configured operational target.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

import pytest

asyncpg = pytest.importorskip("asyncpg")

from atlas_brain.storage.migrations import run_migrations  # noqa: E402
from atlas_brain.storage.migrations.reconciliation import (  # noqa: E402
    HISTORICAL_LEDGER_DIGEST_UNAVAILABLE,
    MIGRATION_022B_PRESENCE_UNKNOWN_COUNT_RECONCILIATION,
    attest_known_historical_migration_reconciliations,
)


ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS = ROOT / "atlas_brain" / "storage" / "migrations"
DATABASE_URL_ENV = "ATLAS_MIGRATION_TEST_DATABASE_URL"


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
    """Create a minimal catalog retaining 027 but no synthetic 022b source."""
    record = MIGRATION_022B_PRESENCE_UNKNOWN_COUNT_RECONCILIATION
    catalog = tmp_path / "migrations"
    catalog.mkdir()
    (catalog / f"{record.current_packaged_migration_name}.sql").write_bytes(
        (MIGRATIONS / f"{record.current_packaged_migration_name}.sql").read_bytes()
    )
    (catalog / "901_presence_attestation_probe.sql").write_text(
        "CREATE TABLE presence_attestation_probe (id INTEGER PRIMARY KEY);\n"
    )
    return catalog


async def _prepare_receipt_ledger_schema(conn, schema: str) -> None:
    """Create only the test-owned ledger evidence for the named receipt."""
    record = MIGRATION_022B_PRESENCE_UNKNOWN_COUNT_RECONCILIATION
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
        22,
        record.migration_name,
        record.observed_applied_at,
    )


async def _prepare_attestable_schema(conn, schema: str) -> None:
    """Create the ordinary-table metadata required by the historical receipt."""
    await _prepare_receipt_ledger_schema(conn, schema)
    await conn.execute("""
        CREATE TABLE presence_events (id UUID PRIMARY KEY);
        ALTER TABLE presence_events
            ADD COLUMN IF NOT EXISTS unknown_count INT DEFAULT 0;
        """)


async def _prepare_foreign_presence_events_schema(
    conn,
    schema: str,
    *,
    foreign_data_wrapper: str,
    foreign_server: str,
) -> None:
    """Create a non-table relation with the same column signature.

    A foreign table can expose the same nullable `integer DEFAULT 0` shape as
    the historical migration but cannot be the result of its `ALTER TABLE`
    receipt. The generated server and wrapper are test-owned and removed by
    the caller.
    """
    await _prepare_receipt_ledger_schema(conn, schema)
    wrapper_ident = _quote_ident(foreign_data_wrapper)
    server_ident = _quote_ident(foreign_server)
    await conn.execute(f"""
        CREATE FOREIGN DATA WRAPPER {wrapper_ident};
        CREATE SERVER {server_ident}
            FOREIGN DATA WRAPPER {wrapper_ident};
        CREATE FOREIGN TABLE presence_events (
            id UUID,
            unknown_count INT DEFAULT 0
        ) SERVER {server_ident};
        """)


@pytest.mark.asyncio
async def test_022b_receipt_attests_empty_real_catalog_without_presence_rows():
    database_url = _database_url_or_skip()
    schema = f"presence_022b_attest_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_attestable_schema(conn, schema)
        record = MIGRATION_022B_PRESENCE_UNKNOWN_COUNT_RECONCILIATION

        attestations = await attest_known_historical_migration_reconciliations(
            conn,
            sorted(MIGRATIONS.glob("*.sql")),
            candidate_names={record.migration_name},
        )

        assert len(attestations) == 1
        attestation = attestations[0]
        assert attestation.migration_name == record.migration_name
        assert attestation.retained_packaged_digest_matches_record
        assert attestation.presence_events_is_ordinary_table
        assert attestation.unknown_count_column_ready
        assert attestation.unknown_count_has_no_constraints
        assert attestation.status == "attested"
        assert attestation.as_payload()["source_verification"] == (
            HISTORICAL_LEDGER_DIGEST_UNAVAILABLE
        )
        assert await conn.fetchval("SELECT COUNT(*) FROM presence_events") == 0
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {_quote_ident(schema)} CASCADE")
        await conn.close()


@pytest.mark.asyncio
async def test_022b_receipt_rejects_real_unknown_count_constraint():
    database_url = _database_url_or_skip()
    schema = f"presence_022b_constraint_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_attestable_schema(conn, schema)
        record = MIGRATION_022B_PRESENCE_UNKNOWN_COUNT_RECONCILIATION
        await conn.execute(
            """
            ALTER TABLE presence_events
                ADD CONSTRAINT presence_events_unknown_count_check
                CHECK (unknown_count >= 0)
            """
        )

        attestations = await attest_known_historical_migration_reconciliations(
            conn,
            sorted(MIGRATIONS.glob("*.sql")),
            candidate_names={record.migration_name},
        )

        assert len(attestations) == 1
        attestation = attestations[0]
        assert attestation.presence_events_is_ordinary_table
        assert attestation.unknown_count_column_ready
        assert attestation.unknown_count_has_no_constraints is False
        assert attestation.status == "not_attested"
        assert await conn.fetchval("SELECT COUNT(*) FROM presence_events") == 0
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {_quote_ident(schema)} CASCADE")
        await conn.close()


@pytest.mark.asyncio
async def test_022b_receipt_rejects_real_foreign_relation_before_pending_sql(
    tmp_path: Path,
):
    """A lookalike foreign relation cannot admit a pending migration."""
    database_url = _database_url_or_skip()
    suffix = uuid.uuid4().hex
    schema = f"presence_022b_foreign_{suffix}"
    foreign_data_wrapper = f"presence_022b_fdw_{suffix}"
    foreign_server = f"presence_022b_server_{suffix}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_foreign_presence_events_schema(
            conn,
            schema,
            foreign_data_wrapper=foreign_data_wrapper,
            foreign_server=foreign_server,
        )
        record = MIGRATION_022B_PRESENCE_UNKNOWN_COUNT_RECONCILIATION
        attestations = await attest_known_historical_migration_reconciliations(
            conn,
            sorted(MIGRATIONS.glob("*.sql")),
            candidate_names={record.migration_name},
        )

        assert len(attestations) == 1
        attestation = attestations[0]
        assert attestation.presence_events_is_ordinary_table is False
        assert attestation.unknown_count_column_ready
        assert attestation.unknown_count_has_no_constraints
        assert attestation.status == "not_attested"

        from atlas_brain.storage.migrations import PendingMigrationContentIntegrityError

        with pytest.raises(
            PendingMigrationContentIntegrityError,
            match=f"missing_source={record.migration_name}",
        ):
            await run_migrations(
                _MigrationPool(conn),
                migrations_dir=_test_catalog(tmp_path),
                only={"901_presence_attestation_probe"},
            )

        assert not await conn.fetchval(
            "SELECT to_regclass('presence_attestation_probe') IS NOT NULL"
        )
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM schema_migrations "
            "WHERE name = '901_presence_attestation_probe'"
        ) == 0
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {_quote_ident(schema)} CASCADE")
        await conn.execute(f"DROP SERVER IF EXISTS {_quote_ident(foreign_server)} CASCADE")
        await conn.execute(
            f"DROP FOREIGN DATA WRAPPER IF EXISTS {_quote_ident(foreign_data_wrapper)}"
        )
        await conn.close()


@pytest.mark.asyncio
async def test_022b_receipt_admits_one_real_pending_migration_without_synthetic_source(
    tmp_path: Path,
):
    database_url = _database_url_or_skip()
    schema = f"presence_022b_runner_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_attestable_schema(conn, schema)
        catalog = _test_catalog(tmp_path)

        await run_migrations(
            _MigrationPool(conn),
            migrations_dir=catalog,
            only={"901_presence_attestation_probe"},
        )

        assert await conn.fetchval(
            "SELECT to_regclass('presence_attestation_probe') IS NOT NULL"
        )
        pending_digest = hashlib.sha256(
            (catalog / "901_presence_attestation_probe.sql").read_bytes()
        ).hexdigest()
        assert dict(
            await conn.fetchrow(
                """
            SELECT name, content_sha256
            FROM schema_migrations
            WHERE name = '901_presence_attestation_probe'
            """
            )
        ) == {
            "name": "901_presence_attestation_probe",
            "content_sha256": pending_digest,
        }
        assert await conn.fetchval("SELECT COUNT(*) FROM presence_events") == 0
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {_quote_ident(schema)} CASCADE")
        await conn.close()
