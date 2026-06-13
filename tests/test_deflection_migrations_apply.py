"""Real-database apply check for the deflection migration chain (#1462 follow-up).

The full migration set is not cleanly fresh-appliable (migration 076 references
a `product_metadata` table that no migration creates -- the app creates it
out-of-band), so a repo-wide `run_migrations` against a fresh database fails.
This check is scoped to the deflection chain, which has no such dependency:

- 328 content_ops_deflection_reports
- 332 content_ops_deflection_report_deliveries
- 336 content_ops_deflection_paid_reconciliation  (#1462 money-path table)

It applies those files in order to a fresh Postgres database and verifies the
tables exist and the reconciliation table's idempotency constraint holds. Runs
in CI against the workflow's fresh atlas_migration_tests service database;
skipped unless ATLAS_MIGRATION_TEST_DATABASE_URL points at a disposable DB. The
migrations dir is resolved by path (no atlas_brain import), so the CI job needs
no application dependencies beyond asyncpg.
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

_TEST_ACCOUNT_ID = "acct-deflection-migration-apply-test"


@pytest.mark.asyncio
async def test_deflection_migration_chain_applies_to_a_fresh_database() -> None:
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_MIGRATION_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_MIGRATION_TEST_DATABASE_URL not set")

    conn = await asyncpg.connect(database_url)
    try:
        for name in DEFLECTION_MIGRATION_CHAIN:
            await conn.execute((MIGRATIONS_DIR / name).read_text())

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

        # The (account_id, request_id, stripe_session_id) uniqueness that
        # record_paid_report_missing relies on (ON CONFLICT DO NOTHING): a second
        # event for the same checkout must not create a duplicate ledger row.
        insert_sql = (
            "INSERT INTO content_ops_deflection_paid_reconciliation "
            "(account_id, request_id, stripe_session_id, event_type, reason) "
            "VALUES ($1, $2, $3, $4, $5) "
            "ON CONFLICT (account_id, request_id, stripe_session_id) DO NOTHING"
        )
        await conn.execute(
            insert_sql,
            _TEST_ACCOUNT_ID,
            "req-test",
            "sess-test",
            "checkout.session.completed",
            "paid_report_missing",
        )
        await conn.execute(
            insert_sql,
            _TEST_ACCOUNT_ID,
            "req-test",
            "sess-test",
            "checkout.session.async_payment_succeeded",
            "paid_report_missing",
        )
        rows = await conn.fetchval(
            "SELECT count(*) FROM content_ops_deflection_paid_reconciliation "
            "WHERE account_id = $1 AND request_id = 'req-test'",
            _TEST_ACCOUNT_ID,
        )
        assert rows == 1
    finally:
        try:
            await conn.execute(
                "DELETE FROM content_ops_deflection_paid_reconciliation "
                "WHERE account_id = $1",
                _TEST_ACCOUNT_ID,
            )
        except Exception:
            pass
        await conn.close()
