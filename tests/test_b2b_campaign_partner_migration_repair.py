"""Real-PostgreSQL proof for the named 067 campaign-partner source receipt.

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
    MIGRATION_067_B2B_CAMPAIGN_PARTNER_RECONCILIATION,
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
    """Create a pending-only catalog without inventing the missing 067 source."""
    catalog = tmp_path / "migrations"
    catalog.mkdir()
    (catalog / "901_campaign_partner_attestation_probe.sql").write_text(
        "CREATE TABLE campaign_partner_attestation_probe (id INTEGER PRIMARY KEY);\n"
    )
    return catalog


async def _prepare_receipt_ledger_schema(conn, schema: str) -> None:
    """Create only the test-owned NULL-digest ledger receipt."""
    record = MIGRATION_067_B2B_CAMPAIGN_PARTNER_RECONCILIATION
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
    delete_action: str = "SET NULL",
) -> None:
    """Create the exact current structural contract for the named receipt."""
    await _prepare_receipt_ledger_schema(conn, schema)
    await conn.execute(f"""
        CREATE TABLE affiliate_partners (id UUID PRIMARY KEY);
        CREATE TABLE b2b_campaigns (
            id UUID PRIMARY KEY,
            partner_id UUID
        );
        ALTER TABLE b2b_campaigns
            ADD CONSTRAINT b2b_campaigns_partner_id_fkey
            FOREIGN KEY (partner_id)
            REFERENCES affiliate_partners(id)
            ON DELETE {delete_action};
        CREATE INDEX idx_b2b_campaigns_partner
            ON b2b_campaigns (partner_id)
            WHERE partner_id IS NOT NULL;
        """)


async def _prepare_leaf_partition_schema(conn, schema: str) -> None:
    """Create a leaf partition that shares the table signature but not identity."""
    await _prepare_receipt_ledger_schema(conn, schema)
    await conn.execute("""
        CREATE TABLE affiliate_partners (id UUID PRIMARY KEY);
        CREATE TABLE b2b_campaigns_parent (
            id INTEGER NOT NULL,
            partner_id UUID
        ) PARTITION BY RANGE (id);
        CREATE TABLE b2b_campaigns
            PARTITION OF b2b_campaigns_parent
            FOR VALUES FROM (0) TO (100);
        ALTER TABLE b2b_campaigns
            ADD CONSTRAINT b2b_campaigns_partner_id_fkey
            FOREIGN KEY (partner_id)
            REFERENCES affiliate_partners(id)
            ON DELETE SET NULL;
        CREATE INDEX idx_b2b_campaigns_partner
            ON b2b_campaigns (partner_id)
            WHERE partner_id IS NOT NULL;
        """)


@pytest.mark.asyncio
async def test_067_receipt_attests_empty_real_catalog_without_campaign_rows():
    database_url = _database_url_or_skip()
    schema = f"campaign_067_attest_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_attestable_schema(conn, schema)
        record = MIGRATION_067_B2B_CAMPAIGN_PARTNER_RECONCILIATION

        attestations = await attest_known_historical_migration_reconciliations(
            conn,
            sorted(MIGRATIONS.glob("*.sql")),
            candidate_names={record.migration_name},
        )

        assert len(attestations) == 1
        attestation = attestations[0]
        assert attestation.migration_name == record.migration_name
        assert attestation.ledger_version_matches_record
        assert attestation.b2b_campaigns_is_ordinary_table
        assert attestation.partner_id_column_ready
        assert attestation.partner_foreign_key_ready
        assert attestation.partner_partial_index_ready
        assert attestation.status == "attested"
        assert attestation.as_payload()["source_verification"] == (
            HISTORICAL_SOURCE_UNAVAILABLE
        )
        assert await conn.fetchval("SELECT COUNT(*) FROM b2b_campaigns") == 0
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {_quote_ident(schema)} CASCADE")
        await conn.close()


@pytest.mark.asyncio
async def test_067_receipt_rejects_wrong_foreign_key_action_before_pending_sql(
    tmp_path: Path,
):
    database_url = _database_url_or_skip()
    schema = f"campaign_067_delete_action_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_attestable_schema(conn, schema, delete_action="RESTRICT")
        record = MIGRATION_067_B2B_CAMPAIGN_PARTNER_RECONCILIATION
        attestations = await attest_known_historical_migration_reconciliations(
            conn,
            sorted(MIGRATIONS.glob("*.sql")),
            candidate_names={record.migration_name},
        )

        assert len(attestations) == 1
        attestation = attestations[0]
        assert attestation.partner_foreign_key_ready is False
        assert attestation.status == "not_attested"

        with pytest.raises(
            PendingMigrationContentIntegrityError,
            match=f"missing_source={record.migration_name}",
        ):
            await run_migrations(
                _MigrationPool(conn),
                migrations_dir=_test_catalog(tmp_path),
                only={"901_campaign_partner_attestation_probe"},
            )

        assert not await conn.fetchval(
            "SELECT to_regclass('campaign_partner_attestation_probe') IS NOT NULL"
        )
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM schema_migrations "
            "WHERE name = '901_campaign_partner_attestation_probe'"
        ) == 0
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {_quote_ident(schema)} CASCADE")
        await conn.close()


@pytest.mark.asyncio
async def test_067_receipt_rejects_leaf_partition_before_pending_sql(tmp_path: Path):
    database_url = _database_url_or_skip()
    schema = f"campaign_067_partition_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_leaf_partition_schema(conn, schema)
        relation = await conn.fetchrow("""
            SELECT relkind, relispartition
            FROM pg_class
            WHERE oid = 'b2b_campaigns'::regclass
            """)
        assert relation["relkind"] == b"r"
        assert relation["relispartition"]

        record = MIGRATION_067_B2B_CAMPAIGN_PARTNER_RECONCILIATION
        attestations = await attest_known_historical_migration_reconciliations(
            conn,
            sorted(MIGRATIONS.glob("*.sql")),
            candidate_names={record.migration_name},
        )

        assert len(attestations) == 1
        attestation = attestations[0]
        assert attestation.b2b_campaigns_is_ordinary_table is False
        assert attestation.status == "not_attested"

        with pytest.raises(
            PendingMigrationContentIntegrityError,
            match=f"missing_source={record.migration_name}",
        ):
            await run_migrations(
                _MigrationPool(conn),
                migrations_dir=_test_catalog(tmp_path),
                only={"901_campaign_partner_attestation_probe"},
            )

        assert not await conn.fetchval(
            "SELECT to_regclass('campaign_partner_attestation_probe') IS NOT NULL"
        )
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {_quote_ident(schema)} CASCADE")
        await conn.close()


@pytest.mark.asyncio
async def test_067_receipt_admits_one_real_pending_migration_without_source(
    tmp_path: Path,
):
    database_url = _database_url_or_skip()
    schema = f"campaign_067_runner_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_attestable_schema(conn, schema)
        catalog = _test_catalog(tmp_path)

        await run_migrations(
            _MigrationPool(conn),
            migrations_dir=catalog,
            only={"901_campaign_partner_attestation_probe"},
        )

        assert await conn.fetchval(
            "SELECT to_regclass('campaign_partner_attestation_probe') IS NOT NULL"
        )
        pending_digest = hashlib.sha256(
            (catalog / "901_campaign_partner_attestation_probe.sql").read_bytes()
        ).hexdigest()
        assert dict(
            await conn.fetchrow(
                """
                SELECT name, content_sha256
                FROM schema_migrations
                WHERE name = '901_campaign_partner_attestation_probe'
                """
            )
        ) == {
            "name": "901_campaign_partner_attestation_probe",
            "content_sha256": pending_digest,
        }
        assert await conn.fetchval("SELECT COUNT(*) FROM b2b_campaigns") == 0
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {_quote_ident(schema)} CASCADE")
        await conn.close()
