"""Real-PostgreSQL proof for the named 272 watchlist-alert receipt.

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
    MIGRATION_272_B2B_WATCHLIST_ALERT_EVENTS_RECONCILIATION,
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
    """Create one pending-only probe without inventing a 272 source file."""
    catalog = tmp_path / "migrations"
    catalog.mkdir()
    (catalog / "901_watchlist_alert_attestation_probe.sql").write_text(
        "CREATE TABLE watchlist_alert_attestation_probe (id INTEGER PRIMARY KEY);\n"
    )
    return catalog


async def _prepare_receipt_ledger_schema(conn, schema: str) -> None:
    """Create only the test-owned synthetic-version, NULL-digest receipt."""
    record = MIGRATION_272_B2B_WATCHLIST_ALERT_EVENTS_RECONCILIATION
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
    delete_action: str = "CASCADE",
    account_status_direction: str = "DESC",
) -> None:
    """Create the source-era structural receipt without inserting alert rows.

    The retained 273 package corroborates this table shape, but the test never
    manufactures source bytes or a checksum for the separate 272 receipt.
    """
    await _prepare_receipt_ledger_schema(conn, schema)
    await conn.execute(f"""
        CREATE TABLE saas_accounts (id UUID PRIMARY KEY);
        CREATE TABLE b2b_watchlist_views (id UUID PRIMARY KEY);
        CREATE TABLE b2b_watchlist_alert_events (
            id UUID PRIMARY KEY,
            account_id UUID NOT NULL REFERENCES saas_accounts(id)
                ON DELETE {delete_action},
            watchlist_view_id UUID NOT NULL REFERENCES b2b_watchlist_views(id)
                ON DELETE CASCADE,
            event_type TEXT NOT NULL,
            threshold_field TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_key TEXT NOT NULL,
            vendor_name TEXT,
            company_name TEXT,
            category TEXT,
            source TEXT,
            threshold_value NUMERIC(6, 2),
            summary TEXT NOT NULL,
            payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            status TEXT NOT NULL DEFAULT 'open',
            first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            resolved_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT chk_b2b_watchlist_alert_events_event_type
                CHECK (event_type IN ('vendor_alert', 'account_alert', 'stale_data')),
            CONSTRAINT chk_b2b_watchlist_alert_events_threshold_field
                CHECK (threshold_field IN (
                    'vendor_alert_threshold',
                    'account_alert_threshold',
                    'stale_days_threshold'
                )),
            CONSTRAINT chk_b2b_watchlist_alert_events_entity_type
                CHECK (entity_type IN ('vendor', 'account', 'signal_cluster')),
            CONSTRAINT chk_b2b_watchlist_alert_events_status
                CHECK (status IN ('open', 'resolved'))
        );
        CREATE UNIQUE INDEX idx_b2b_watchlist_alert_events_view_entity
            ON b2b_watchlist_alert_events (watchlist_view_id, event_type, entity_key);
        CREATE INDEX idx_b2b_watchlist_alert_events_account_status
            ON b2b_watchlist_alert_events (
                account_id,
                status,
                last_seen_at {account_status_direction}
            );
        CREATE INDEX idx_b2b_watchlist_alert_events_view_status
            ON b2b_watchlist_alert_events (watchlist_view_id, status, last_seen_at DESC);
        """)


async def _prepare_partition_schema(conn, schema: str) -> None:
    """Create a partition relation that must never satisfy the table receipt."""
    await _prepare_receipt_ledger_schema(conn, schema)
    await conn.execute("""
        CREATE TABLE b2b_watchlist_alert_events (
            id UUID NOT NULL,
            partition_key INTEGER NOT NULL
        ) PARTITION BY RANGE (partition_key);
        CREATE TABLE b2b_watchlist_alert_events_leaf
            PARTITION OF b2b_watchlist_alert_events
            FOR VALUES FROM (0) TO (100);
        """)


@pytest.mark.asyncio
async def test_272_receipt_attests_empty_real_catalog_without_alert_rows():
    database_url = _database_url_or_skip()
    schema = f"watchlist_alert_272_attest_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_attestable_schema(conn, schema)
        record = MIGRATION_272_B2B_WATCHLIST_ALERT_EVENTS_RECONCILIATION

        attestations = await attest_known_historical_migration_reconciliations(
            conn,
            sorted(MIGRATIONS.glob("*.sql")),
            candidate_names={record.migration_name},
        )

        assert len(attestations) == 1
        attestation = attestations[0]
        assert attestation.migration_name == record.migration_name
        assert attestation.ledger_version_matches_record
        assert attestation.watchlist_alert_events_is_ordinary_table
        assert attestation.base_alert_event_columns_ready
        assert attestation.required_alert_event_constraints_ready
        assert attestation.required_alert_event_indexes_ready
        assert attestation.status == "attested"
        assert attestation.as_payload()["source_verification"] == (
            HISTORICAL_SOURCE_UNAVAILABLE
        )
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM b2b_watchlist_alert_events"
        ) == 0
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {_quote_ident(schema)} CASCADE")
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "delete_action", "account_status_direction", "field"),
    [
        (
            "wrong foreign-key action",
            "RESTRICT",
            "DESC",
            "required_alert_event_constraints_ready",
        ),
        (
            "wrong index direction",
            "CASCADE",
            "ASC",
            "required_alert_event_indexes_ready",
        ),
    ],
)
async def test_272_receipt_rejects_altered_catalog_before_pending_sql(
    tmp_path: Path,
    case: str,
    delete_action: str,
    account_status_direction: str,
    field: str,
):
    database_url = _database_url_or_skip()
    schema = f"watchlist_alert_272_reject_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_attestable_schema(
            conn,
            schema,
            delete_action=delete_action,
            account_status_direction=account_status_direction,
        )
        record = MIGRATION_272_B2B_WATCHLIST_ALERT_EVENTS_RECONCILIATION
        attestations = await attest_known_historical_migration_reconciliations(
            conn,
            sorted(MIGRATIONS.glob("*.sql")),
            candidate_names={record.migration_name},
        )

        assert len(attestations) == 1
        attestation = attestations[0]
        assert getattr(attestation, field) is False, case
        assert attestation.status == "not_attested", case

        with pytest.raises(
            PendingMigrationContentIntegrityError,
            match=f"missing_source={record.migration_name}",
        ):
            await run_migrations(
                _MigrationPool(conn),
                migrations_dir=_test_catalog(tmp_path),
                only={"901_watchlist_alert_attestation_probe"},
            )

        assert not await conn.fetchval(
            "SELECT to_regclass('watchlist_alert_attestation_probe') IS NOT NULL"
        )
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM schema_migrations "
            "WHERE name = '901_watchlist_alert_attestation_probe'"
        ) == 0
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {_quote_ident(schema)} CASCADE")
        await conn.close()


@pytest.mark.asyncio
async def test_272_receipt_rejects_partition_relation_before_pending_sql(tmp_path: Path):
    database_url = _database_url_or_skip()
    schema = f"watchlist_alert_272_partition_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_partition_schema(conn, schema)
        relation = await conn.fetchrow("""
            SELECT relkind, relispartition
            FROM pg_class
            WHERE oid = 'b2b_watchlist_alert_events'::regclass
            """)
        assert relation["relkind"] == b"p"
        assert relation["relispartition"] is False

        record = MIGRATION_272_B2B_WATCHLIST_ALERT_EVENTS_RECONCILIATION
        attestations = await attest_known_historical_migration_reconciliations(
            conn,
            sorted(MIGRATIONS.glob("*.sql")),
            candidate_names={record.migration_name},
        )

        assert len(attestations) == 1
        attestation = attestations[0]
        assert attestation.watchlist_alert_events_is_ordinary_table is False
        assert attestation.status == "not_attested"

        with pytest.raises(
            PendingMigrationContentIntegrityError,
            match=f"missing_source={record.migration_name}",
        ):
            await run_migrations(
                _MigrationPool(conn),
                migrations_dir=_test_catalog(tmp_path),
                only={"901_watchlist_alert_attestation_probe"},
            )

        assert not await conn.fetchval(
            "SELECT to_regclass('watchlist_alert_attestation_probe') IS NOT NULL"
        )
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {_quote_ident(schema)} CASCADE")
        await conn.close()


@pytest.mark.asyncio
async def test_272_receipt_admits_one_real_pending_migration_without_source(
    tmp_path: Path,
):
    database_url = _database_url_or_skip()
    schema = f"watchlist_alert_272_runner_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_attestable_schema(conn, schema)
        catalog = _test_catalog(tmp_path)

        await run_migrations(
            _MigrationPool(conn),
            migrations_dir=catalog,
            only={"901_watchlist_alert_attestation_probe"},
        )
        await run_migrations(
            _MigrationPool(conn),
            migrations_dir=catalog,
            only={"901_watchlist_alert_attestation_probe"},
        )

        assert await conn.fetchval(
            "SELECT to_regclass('watchlist_alert_attestation_probe') IS NOT NULL"
        )
        pending_digest = hashlib.sha256(
            (catalog / "901_watchlist_alert_attestation_probe.sql").read_bytes()
        ).hexdigest()
        assert dict(
            await conn.fetchrow(
                """
                SELECT name, content_sha256
                FROM schema_migrations
                WHERE name = '901_watchlist_alert_attestation_probe'
                """
            )
        ) == {
            "name": "901_watchlist_alert_attestation_probe",
            "content_sha256": pending_digest,
        }
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM schema_migrations "
            "WHERE name = '901_watchlist_alert_attestation_probe'"
        ) == 1
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM b2b_watchlist_alert_events"
        ) == 0
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {_quote_ident(schema)} CASCADE")
        await conn.close()
