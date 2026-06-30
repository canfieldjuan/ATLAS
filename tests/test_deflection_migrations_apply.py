"""Real-database apply check for the deflection migration chain (#1462 follow-up).

The full migration set is not cleanly fresh-appliable (migration 076 references
a `product_metadata` table that no migration creates -- the app creates it
out-of-band), so a repo-wide `run_migrations` against a fresh database fails.
This check is scoped to the deflection chain, which has no such dependency:

- 328 content_ops_deflection_reports
- 332 content_ops_deflection_report_deliveries
- 336 content_ops_deflection_paid_reconciliation  (#1462 money-path table)
- 337 reconciliation NULL-session dedup (NOT NULL stripe_session_id)
- 339 content_ops_deflection_reports retention index
- 340 content_ops_deflection_deltas
- 341 content_ops_deflection_delta_deliveries
- 342 content_ops_deflection_checkout_authorization
- 343 content_ops_deflection_delta_entitlements

It applies those files in order to a fresh Postgres database and verifies the
tables exist and the reconciliation table's idempotency constraint holds --
including the NULL-session dedup gap that 337 closes. Runs in CI against the
workflow's fresh atlas_migration_tests service database; skipped unless
ATLAS_MIGRATION_TEST_DATABASE_URL points at a disposable DB. The migrations dir
is resolved by path (no atlas_brain import), so the CI job needs no application
dependencies beyond asyncpg.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = ROOT / "atlas_brain" / "storage" / "migrations"

DEFLECTION_MIGRATION_CHAIN = (
    "328_content_ops_deflection_reports.sql",
    "332_content_ops_deflection_report_deliveries.sql",
    "336_content_ops_deflection_paid_reconciliation.sql",
)
NULL_SESSION_MIGRATION = "337_content_ops_deflection_reconciliation_null_session.sql"
RETENTION_INDEX_MIGRATION = "339_content_ops_deflection_reports_retention_index.sql"
DELTA_MIGRATION = "340_content_ops_deflection_deltas.sql"
DELTA_DELIVERY_MIGRATION = "341_content_ops_deflection_delta_deliveries.sql"
CHECKOUT_AUTHORIZATION_MIGRATION = "342_content_ops_deflection_checkout_authorization.sql"
DELTA_ENTITLEMENT_MIGRATION = "343_content_ops_deflection_delta_entitlements.sql"

_TEST_ACCOUNT_ID = "acct-deflection-migration-apply-test"

_RECON_INSERT_SQL = (
    "INSERT INTO content_ops_deflection_paid_reconciliation "
    "(account_id, request_id, stripe_session_id, event_type, reason) "
    "VALUES ($1, $2, $3, $4, $5) "
    "ON CONFLICT (account_id, request_id, stripe_session_id) DO NOTHING"
)


def test_deflection_retention_index_migration_is_concurrent() -> None:
    migration_sql = (MIGRATIONS_DIR / RETENTION_INDEX_MIGRATION).read_text()

    assert "CREATE INDEX CONCURRENTLY IF NOT EXISTS" in migration_sql


def _database_url() -> str | None:
    return os.environ.get("ATLAS_MIGRATION_TEST_DATABASE_URL")


async def _apply(conn, *names: str) -> None:
    for name in names:
        await conn.execute((MIGRATIONS_DIR / name).read_text())


async def _reset(conn) -> None:
    # Both tests share the CI service database; test 1 applies 337 (NOT NULL),
    # which would break test 2's pre-337 NULL insert. Drop the deflection tables
    # so each test applies from a clean schema regardless of order. No-op on a
    # genuinely fresh database.
    await conn.execute(
        "DROP TABLE IF EXISTS "
        "content_ops_deflection_paid_reconciliation, "
        "content_ops_deflection_delta_entitlements, "
        "content_ops_deflection_delta_deliveries, "
        "content_ops_deflection_deltas, "
        "content_ops_deflection_report_deliveries, "
        "content_ops_deflection_reports CASCADE"
    )


@pytest.mark.asyncio
async def test_deflection_migration_chain_applies_to_a_fresh_database() -> None:
    asyncpg = pytest.importorskip("asyncpg")
    database_url = _database_url()
    if not database_url:
        pytest.skip("ATLAS_MIGRATION_TEST_DATABASE_URL not set")

    conn = await asyncpg.connect(database_url)
    try:
        await _reset(conn)
        await _apply(
            conn,
            *DEFLECTION_MIGRATION_CHAIN,
            NULL_SESSION_MIGRATION,
            RETENTION_INDEX_MIGRATION,
            DELTA_MIGRATION,
            DELTA_DELIVERY_MIGRATION,
            CHECKOUT_AUTHORIZATION_MIGRATION,
            DELTA_ENTITLEMENT_MIGRATION,
        )

        existing = {
            row["tablename"]
            for row in await conn.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public' "
                "AND tablename LIKE 'content_ops_deflection%'"
            )
        }
        assert {
            "content_ops_deflection_reports",
            "content_ops_deflection_report_deliveries",
            "content_ops_deflection_paid_reconciliation",
            "content_ops_deflection_deltas",
            "content_ops_deflection_delta_deliveries",
            "content_ops_deflection_delta_entitlements",
        } <= existing

        recon_columns = {
            row["column_name"]
            for row in await conn.fetch(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'content_ops_deflection_paid_reconciliation'"
            )
        }
        assert {
            "account_id",
            "request_id",
            "stripe_session_id",
            "event_type",
            "reason",
            "created_at",
        } <= recon_columns

        # 337 forbids NULL so the (account_id, request_id, stripe_session_id)
        # UNIQUE can dedup a missing-session row (NULL would be DISTINCT).
        is_nullable = await conn.fetchval(
            "SELECT is_nullable FROM information_schema.columns "
            "WHERE table_name = 'content_ops_deflection_paid_reconciliation' "
            "AND column_name = 'stripe_session_id'"
        )
        assert is_nullable == "NO"

        retention_index_exists = await conn.fetchval(
            "SELECT EXISTS ("
            "SELECT 1 FROM pg_indexes "
            "WHERE schemaname = 'public' "
            "AND tablename = 'content_ops_deflection_reports' "
            "AND indexname = 'idx_content_ops_deflection_reports_created_at'"
            ")"
        )
        assert retention_index_exists is True

        delta_columns = {
            row["column_name"]
            for row in await conn.fetch(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'content_ops_deflection_deltas'"
            )
        }
        assert {
            "account_id",
            "current_request_id",
            "baseline_request_id",
            "delta",
            "created_at",
            "updated_at",
        } <= delta_columns

        delta_delivery_columns = {
            row["column_name"]
            for row in await conn.fetch(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'content_ops_deflection_delta_deliveries'"
            )
        }
        assert {
            "account_id",
            "current_request_id",
            "baseline_request_id",
            "delivery_email",
            "delivery_status",
            "delivery_error",
            "provider_message_id",
            "created_at",
            "updated_at",
            "delivered_at",
        } <= delta_delivery_columns

        checkout_columns = {
            row["column_name"]
            for row in await conn.fetch(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'content_ops_deflection_reports'"
            )
        }
        assert {
            "checkout_price_variant",
            "checkout_amount_cents",
            "checkout_currency",
            "checkout_price_id",
            "checkout_authorized_at",
        } <= checkout_columns

        delta_entitlement_columns = {
            row["column_name"]
            for row in await conn.fetch(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'content_ops_deflection_delta_entitlements'"
            )
        }
        assert {
            "account_id",
            "stripe_subscription_id",
            "stripe_customer_id",
            "stripe_price_id",
            "stripe_subscription_status",
            "entitlement_source",
            "current_period_end",
            "granted_at",
            "revoked_at",
            "metadata",
            "created_at",
            "updated_at",
        } <= delta_entitlement_columns

        # Non-null dedup (ON CONFLICT DO NOTHING): a second event for the same
        # checkout must not create a duplicate ledger row.
        await conn.execute(
            _RECON_INSERT_SQL,
            _TEST_ACCOUNT_ID,
            "req-sess",
            "sess-test",
            "checkout.session.completed",
            "paid_report_missing",
        )
        await conn.execute(
            _RECON_INSERT_SQL,
            _TEST_ACCOUNT_ID,
            "req-sess",
            "sess-test",
            "checkout.session.async_payment_succeeded",
            "paid_report_missing",
        )
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM content_ops_deflection_paid_reconciliation "
                "WHERE account_id = $1 AND request_id = 'req-sess'",
                _TEST_ACCOUNT_ID,
            )
            == 1
        )

        # Going-forward empty-session dedup: '' is a normal conflict-eligible
        # value, so two missing-session events for the same checkout dedup
        # (the gap that NULL left open).
        for event_type in (
            "checkout.session.completed",
            "checkout.session.async_payment_succeeded",
        ):
            await conn.execute(
                _RECON_INSERT_SQL,
                _TEST_ACCOUNT_ID,
                "req-empty",
                "",
                event_type,
                "paid_report_missing",
            )
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM content_ops_deflection_paid_reconciliation "
                "WHERE account_id = $1 AND request_id = 'req-empty'",
                _TEST_ACCOUNT_ID,
            )
            == 1
        )

        await conn.execute(
            "INSERT INTO content_ops_deflection_reports "
            "(account_id, request_id, snapshot, artifact, paid) "
            "VALUES ($1, 'baseline-delta', '{}'::jsonb, '{}'::jsonb, true), "
            "($1, 'current-delta', '{}'::jsonb, '{}'::jsonb, true)",
            _TEST_ACCOUNT_ID,
        )
        await conn.execute(
            "INSERT INTO content_ops_deflection_deltas "
            "(account_id, current_request_id, baseline_request_id, delta) "
            "VALUES ($1, 'current-delta', 'baseline-delta', $2::jsonb)",
            _TEST_ACCOUNT_ID,
            '{"schema_version":"deflection_delta.v1"}',
        )
        await conn.execute(
            "INSERT INTO content_ops_deflection_delta_deliveries "
            "(account_id, current_request_id, baseline_request_id, delivery_email) "
            "VALUES ($1, 'current-delta', 'baseline-delta', 'buyer@example.com')",
            _TEST_ACCOUNT_ID,
        )
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM content_ops_deflection_deltas "
                "WHERE account_id = $1",
                _TEST_ACCOUNT_ID,
            )
            == 1
        )
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM content_ops_deflection_delta_deliveries "
                "WHERE account_id = $1 AND delivery_status = 'pending'",
                _TEST_ACCOUNT_ID,
            )
            == 1
        )
        for suffix, status in (("active", "active"), ("trial", "trialing")):
            await conn.execute(
                "INSERT INTO content_ops_deflection_delta_entitlements "
                "(account_id, stripe_subscription_id, stripe_subscription_status) "
                "VALUES ($1, $2, $3)",
                f"{_TEST_ACCOUNT_ID}-{suffix}",
                f"sub-{suffix}",
                status,
            )
        assert (
            await conn.fetchval(
                "SELECT count(DISTINCT account_id) "
                "FROM content_ops_deflection_delta_entitlements "
                "WHERE stripe_subscription_status IN ('active', 'trialing')"
            )
            == 2
        )
        delta_entitlement_indexdef = await conn.fetchval(
            "SELECT indexdef FROM pg_indexes "
            "WHERE schemaname = 'public' "
            "AND tablename = 'content_ops_deflection_delta_entitlements' "
            "AND indexname = 'idx_content_ops_deflection_delta_entitlements_active'"
        )
        assert delta_entitlement_indexdef is not None
        assert "revoked_at IS NULL" in delta_entitlement_indexdef
        with pytest.raises(Exception):
            await conn.execute(
                "INSERT INTO content_ops_deflection_delta_entitlements "
                "(account_id, stripe_subscription_id, stripe_subscription_status) "
                "VALUES ($1, 'sub-invalid', 'renewing')",
                _TEST_ACCOUNT_ID,
            )
    finally:
        await _cleanup(conn)
        await conn.close()


@pytest.mark.asyncio
async def test_deflection_337_collapses_pre_existing_null_session_duplicates() -> None:
    asyncpg = pytest.importorskip("asyncpg")
    database_url = _database_url()
    if not database_url:
        pytest.skip("ATLAS_MIGRATION_TEST_DATABASE_URL not set")

    conn = await asyncpg.connect(database_url)
    try:
        await _reset(conn)
        # 336 only: stripe_session_id is nullable and NULL is DISTINCT, so two
        # NULL-session rows for the same checkout both land (the bug).
        await _apply(conn, *DEFLECTION_MIGRATION_CHAIN)
        for event_type in (
            "checkout.session.completed",
            "checkout.session.async_payment_succeeded",
        ):
            await conn.execute(
                "INSERT INTO content_ops_deflection_paid_reconciliation "
                "(account_id, request_id, stripe_session_id, event_type, reason) "
                "VALUES ($1, $2, NULL, $3, $4)",
                _TEST_ACCOUNT_ID,
                "req-null",
                event_type,
                "paid_report_missing",
            )
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM content_ops_deflection_paid_reconciliation "
                "WHERE account_id = $1 AND request_id = 'req-null'",
                _TEST_ACCOUNT_ID,
            )
            == 2
        )

        # 337 collapses the NULL-equivalent duplicates to one row and backfills
        # the survivor's session id to ''.
        await _apply(conn, NULL_SESSION_MIGRATION)
        rows = await conn.fetch(
            "SELECT stripe_session_id FROM content_ops_deflection_paid_reconciliation "
            "WHERE account_id = $1 AND request_id = 'req-null'",
            _TEST_ACCOUNT_ID,
        )
        assert len(rows) == 1
        assert rows[0]["stripe_session_id"] == ""
    finally:
        await _cleanup(conn)
        await conn.close()


async def _cleanup(conn) -> None:
    try:
        await conn.execute(
            "DELETE FROM content_ops_deflection_delta_entitlements "
            "WHERE account_id LIKE $1",
            f"{_TEST_ACCOUNT_ID}%",
        )
    except Exception:
        pass
    try:
        await conn.execute(
            "DELETE FROM content_ops_deflection_deltas "
            "WHERE account_id = $1",
            _TEST_ACCOUNT_ID,
        )
    except Exception:
        pass
    try:
        await conn.execute(
            "DELETE FROM content_ops_deflection_paid_reconciliation "
            "WHERE account_id = $1",
            _TEST_ACCOUNT_ID,
        )
    except Exception:
        pass
    try:
        await conn.execute(
            "DELETE FROM content_ops_deflection_reports "
            "WHERE account_id = $1",
            _TEST_ACCOUNT_ID,
        )
    except Exception:
        pass
