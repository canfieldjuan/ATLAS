"""Real-PostgreSQL proof for the named 272 watchlist-alert receipt.

These tests use only a disposable database supplied through
``ATLAS_MIGRATION_TEST_DATABASE_URL``. They create an empty test-owned schema
and never connect to Atlas's configured operational target.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
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
from atlas_brain.services.b2b.watchlist_alerts import (  # noqa: E402
    evaluate_watchlist_alert_events_for_view,
)


ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS = ROOT / "atlas_brain" / "storage" / "migrations"
DATABASE_URL_ENV = "ATLAS_MIGRATION_TEST_DATABASE_URL"


def test_watchlist_alert_evaluator_import_avoids_application_packages() -> None:
    """Keep migration proof independent from eager API and scheduler packages."""
    script = """
import sys


class _ForbidApplicationPackages:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"atlas_brain.api", "atlas_brain.autonomous"} or fullname.startswith(
            ("atlas_brain.api.", "atlas_brain.autonomous.")
        ):
            raise ImportError("watchlist evaluator must not import application packages")
        return None


sys.meta_path.insert(0, _ForbidApplicationPackages())
from atlas_brain.services.b2b.watchlist_alerts import evaluate_watchlist_alert_events_for_view
assert callable(evaluate_watchlist_alert_events_for_view)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


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


async def _prepare_attestable_schema(conn, schema: str) -> None:
    """Execute retained producer DDL as compatibility evidence, never 272 source.

    The test-owned 272 receipt stays synthetic-version/NULL-digest. Retained
    273 supplies the source-era shape and 281 supplies the sole known later
    writer-required column. Neither source verifies, renames, or replaces named
    272.
    """
    await _prepare_receipt_ledger_schema(conn, schema)
    await conn.execute("""
        CREATE TABLE saas_accounts (id UUID PRIMARY KEY);
        CREATE TABLE b2b_watchlist_views (id UUID PRIMARY KEY);
        """)
    await conn.execute(
        (MIGRATIONS / "273_b2b_watchlist_alert_events.sql").read_text()
    )
    await conn.execute(
        (MIGRATIONS / "281_b2b_watchlist_alert_reopen_count.sql").read_text()
    )


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


async def _assert_open_alert_writer_rejected_by_expression_index(conn) -> None:
    """Drive the real writer through the cited expression-index failure."""
    account_id = uuid.uuid4()
    view_id = uuid.uuid4()
    await conn.execute("INSERT INTO saas_accounts (id) VALUES ($1)", account_id)
    await conn.execute("INSERT INTO b2b_watchlist_views (id) VALUES ($1)", view_id)

    async def slow_burn_loader(**_kwargs):
        return {
            "signals": [{
                "vendor_name": "index proof vendor",
                "vendor_alert_hit": True,
                "avg_urgency_score": 9.0,
            }]
        }

    async def accounts_loader(**_kwargs):
        return {"accounts": []}

    with pytest.raises(asyncpg.DivisionByZeroError, match="division by zero"):
        await evaluate_watchlist_alert_events_for_view(
            conn,
            account_id=account_id,
            view_id=view_id,
            view_row={
                "id": view_id,
                "vendor_names": ["index proof vendor"],
                "vendor_alert_threshold": 1.0,
            },
            user=None,
            slow_burn_loader=slow_burn_loader,
            accounts_loader=accounts_loader,
        )

    assert await conn.fetchval(
        "SELECT COUNT(*) FROM b2b_watchlist_alert_events"
    ) == 0


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
        assert attestation.watchlist_alert_events_has_permanent_storage
        assert attestation.base_alert_event_columns_ready
        assert attestation.known_later_alert_event_columns_ready
        assert attestation.no_unlisted_alert_event_columns
        assert attestation.required_alert_event_constraints_ready
        assert attestation.no_unlisted_alert_event_constraints
        assert attestation.required_alert_event_indexes_ready
        assert attestation.no_unlisted_alert_event_indexes
        assert attestation.no_unreviewed_alert_event_write_interceptors
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
    ("case", "field"),
    [
        (
            "wrong foreign-key action",
            "required_alert_event_constraints_ready",
        ),
        (
            "wrong index direction",
            "required_alert_event_indexes_ready",
        ),
        (
            "check constraint literal collision",
            "required_alert_event_constraints_ready",
        ),
        (
            "check constraint literal case collision",
            "required_alert_event_constraints_ready",
        ),
        (
            "unlisted check constraint",
            "no_unlisted_alert_event_constraints",
        ),
        (
            "unlogged table storage",
            "watchlist_alert_events_has_permanent_storage",
        ),
        ("unlisted unique index", "no_unlisted_alert_event_indexes"),
        ("unlisted simple nonunique index", "no_unlisted_alert_event_indexes"),
        ("unlisted descending index", "no_unlisted_alert_event_indexes"),
        ("unlisted partial index", "no_unlisted_alert_event_indexes"),
        ("unlisted expression index", "no_unlisted_alert_event_indexes"),
        ("unlisted INCLUDE index", "no_unlisted_alert_event_indexes"),
        ("unlisted hash index", "no_unlisted_alert_event_indexes"),
        (
            "unlisted required no-default column",
            "no_unlisted_alert_event_columns",
        ),
        (
            "unreviewed before-insert trigger",
            "no_unreviewed_alert_event_write_interceptors",
        ),
        (
            "disabled unreviewed after-insert trigger",
            "no_unreviewed_alert_event_write_interceptors",
        ),
        (
            "unreviewed before-update trigger",
            "no_unreviewed_alert_event_write_interceptors",
        ),
        (
            "unreviewed insert rewrite rule",
            "no_unreviewed_alert_event_write_interceptors",
        ),
        (
            "row security enabled",
            "no_unreviewed_alert_event_write_interceptors",
        ),
        (
            "row security forced",
            "no_unreviewed_alert_event_write_interceptors",
        ),
        (
            "unreviewed row security policy",
            "no_unreviewed_alert_event_write_interceptors",
        ),
    ],
)
async def test_272_receipt_rejects_altered_catalog_before_pending_sql(
    tmp_path: Path,
    case: str,
    field: str,
):
    database_url = _database_url_or_skip()
    schema = f"watchlist_alert_272_reject_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_attestable_schema(conn, schema)
        if case == "wrong foreign-key action":
            await conn.execute("""
                ALTER TABLE b2b_watchlist_alert_events
                    DROP CONSTRAINT b2b_watchlist_alert_events_account_id_fkey;
                ALTER TABLE b2b_watchlist_alert_events
                    ADD CONSTRAINT b2b_watchlist_alert_events_account_id_fkey
                    FOREIGN KEY (account_id) REFERENCES saas_accounts(id)
                    ON DELETE RESTRICT;
                """)
        elif case == "wrong index direction":
            await conn.execute("""
                DROP INDEX idx_b2b_watchlist_alert_events_account_status;
                CREATE INDEX idx_b2b_watchlist_alert_events_account_status
                    ON b2b_watchlist_alert_events (
                        account_id,
                        status,
                        last_seen_at ASC
                    );
                """)
        elif case == "check constraint literal collision":
            await conn.execute("""
                ALTER TABLE b2b_watchlist_alert_events
                    DROP CONSTRAINT chk_b2b_watchlist_alert_events_status;
                ALTER TABLE b2b_watchlist_alert_events
                    ADD CONSTRAINT chk_b2b_watchlist_alert_events_status
                    CHECK (status IN ('o''pen', 'resolved'));
                """)
        elif case == "check constraint literal case collision":
            await conn.execute("""
                ALTER TABLE b2b_watchlist_alert_events
                    DROP CONSTRAINT chk_b2b_watchlist_alert_events_status;
                ALTER TABLE b2b_watchlist_alert_events
                    ADD CONSTRAINT chk_b2b_watchlist_alert_events_status
                    CHECK (status IN ('OPEN', 'resolved'));
                """)
        elif case == "unlisted check constraint":
            await conn.execute("""
                ALTER TABLE b2b_watchlist_alert_events
                    ADD CONSTRAINT chk_b2b_watchlist_alert_events_unlisted_status
                    CHECK (status = 'resolved') NOT VALID;
                """)
        elif case == "unlogged table storage":
            await conn.execute(
                "ALTER TABLE b2b_watchlist_alert_events SET UNLOGGED"
            )
        elif case == "unlisted unique index":
            await conn.execute("""
                CREATE UNIQUE INDEX
                    idx_b2b_watchlist_alert_events_unlisted_unique
                ON b2b_watchlist_alert_events (
                    account_id,
                    status,
                    event_type,
                    entity_key
                );
                """)
        elif case == "unlisted simple nonunique index":
            await conn.execute("""
                CREATE INDEX idx_b2b_watchlist_alert_events_unlisted_simple
                    ON b2b_watchlist_alert_events (summary);
                """)
        elif case == "unlisted descending index":
            await conn.execute("""
                CREATE INDEX idx_b2b_watchlist_alert_events_unlisted_descending
                    ON b2b_watchlist_alert_events (account_id DESC);
                """)
        elif case == "unlisted partial index":
            await conn.execute("""
                CREATE INDEX idx_b2b_watchlist_alert_events_unlisted_partial
                    ON b2b_watchlist_alert_events (account_id)
                    WHERE status = 'open';
                """)
        elif case == "unlisted expression index":
            await conn.execute("""
                CREATE INDEX idx_b2b_watchlist_alert_events_unlisted_expression
                    ON b2b_watchlist_alert_events
                    ((1 / CASE WHEN status = 'open' THEN 0 ELSE 1 END));
                """)
            await _assert_open_alert_writer_rejected_by_expression_index(conn)
        elif case == "unlisted INCLUDE index":
            await conn.execute("""
                CREATE INDEX idx_b2b_watchlist_alert_events_unlisted_include
                    ON b2b_watchlist_alert_events (account_id) INCLUDE (status);
                """)
        elif case == "unlisted hash index":
            await conn.execute("""
                CREATE INDEX idx_b2b_watchlist_alert_events_unlisted_hash
                    ON b2b_watchlist_alert_events USING hash (status);
                """)
        elif case == "unlisted required no-default column":
            await conn.execute("""
                ALTER TABLE b2b_watchlist_alert_events
                    ADD COLUMN unlisted_required_writer_input TEXT NOT NULL;
                """)
        elif case == "unreviewed before-insert trigger":
            await conn.execute("""
                CREATE FUNCTION reject_unreviewed_alert_event_insert()
                RETURNS trigger
                LANGUAGE plpgsql
                AS $$
                BEGIN
                    IF NEW.status = 'open' THEN
                        RAISE EXCEPTION 'unreviewed alert-event insert trigger';
                    END IF;
                    RETURN NEW;
                END;
                $$;
                CREATE TRIGGER reject_unreviewed_alert_event_insert
                    BEFORE INSERT ON b2b_watchlist_alert_events
                    FOR EACH ROW
                    EXECUTE FUNCTION reject_unreviewed_alert_event_insert();
                """)
        elif case == "disabled unreviewed after-insert trigger":
            await conn.execute("""
                CREATE FUNCTION audit_unreviewed_alert_event_insert()
                RETURNS trigger
                LANGUAGE plpgsql
                AS $$
                BEGIN
                    RETURN NEW;
                END;
                $$;
                CREATE TRIGGER audit_unreviewed_alert_event_insert
                    AFTER INSERT ON b2b_watchlist_alert_events
                    FOR EACH ROW
                    EXECUTE FUNCTION audit_unreviewed_alert_event_insert();
                ALTER TABLE b2b_watchlist_alert_events
                    DISABLE TRIGGER audit_unreviewed_alert_event_insert;
                """)
        elif case == "unreviewed before-update trigger":
            await conn.execute("""
                CREATE FUNCTION reject_unreviewed_alert_event_update()
                RETURNS trigger
                LANGUAGE plpgsql
                AS $$
                BEGIN
                    RAISE EXCEPTION 'unreviewed alert-event update trigger';
                END;
                $$;
                CREATE TRIGGER reject_unreviewed_alert_event_update
                    BEFORE UPDATE ON b2b_watchlist_alert_events
                    FOR EACH ROW
                    EXECUTE FUNCTION reject_unreviewed_alert_event_update();
                """)
        elif case == "unreviewed insert rewrite rule":
            await conn.execute("""
                CREATE RULE suppress_unreviewed_alert_event_insert AS
                    ON INSERT TO b2b_watchlist_alert_events
                    DO INSTEAD NOTHING;
                """)
        elif case == "row security enabled":
            await conn.execute(
                "ALTER TABLE b2b_watchlist_alert_events ENABLE ROW LEVEL SECURITY"
            )
        elif case == "row security forced":
            await conn.execute("""
                ALTER TABLE b2b_watchlist_alert_events ENABLE ROW LEVEL SECURITY;
                ALTER TABLE b2b_watchlist_alert_events FORCE ROW LEVEL SECURITY;
                """)
        elif case == "unreviewed row security policy":
            await conn.execute("""
                CREATE POLICY unreviewed_alert_event_insert_policy
                    ON b2b_watchlist_alert_events
                    FOR INSERT
                    WITH CHECK (true);
                """)
        else:  # pragma: no cover - parametrize keeps this exhaustive.
            raise AssertionError(f"unexpected altered catalog case: {case}")
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
