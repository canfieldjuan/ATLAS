from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import date
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

from atlas_brain.services.receivables import ReceivablesService
from atlas_brain.storage.repositories import invoice as invoice_repo_mod


class _SchemaPool:
    """Connection-scoped adapter that keeps repository calls in one test schema."""

    is_initialized = True

    def __init__(self, conn, schema: str) -> None:
        self.conn = conn
        self.schema = schema

    @asynccontextmanager
    async def transaction(self):
        async with self.conn.transaction():
            await self.conn.execute(f'SET LOCAL search_path TO "{self.schema}"')
            yield self.conn

    async def fetch(self, query, *args):
        async with self.conn.transaction():
            await self.conn.execute(f'SET LOCAL search_path TO "{self.schema}"')
            return await self.conn.fetch(query, *args)

    async def fetchrow(self, query, *args):
        async with self.conn.transaction():
            await self.conn.execute(f'SET LOCAL search_path TO "{self.schema}"')
            return await self.conn.fetchrow(query, *args)

    async def execute(self, query, *args):
        async with self.conn.transaction():
            await self.conn.execute(f'SET LOCAL search_path TO "{self.schema}"')
            return await self.conn.execute(query, *args)


def _migration_sql(name: str) -> str:
    return (
        Path(__file__).parents[1] / f"atlas_brain/storage/migrations/{name}"
    ).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_real_postgres_invoice_lookup_overdue_history_and_singular_payments():
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    schema = f"invoice_repository_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        await conn.execute("CREATE TABLE contacts (id uuid PRIMARY KEY)")
        await conn.execute(_migration_sql("045_invoices.sql"))
        await conn.execute(_migration_sql("344_receivables_payments.sql"))

        contact_id = uuid4()
        lookup_id = uuid4()
        overdue_id = uuid4()
        legacy_id = uuid4()
        contactless_id = uuid4()
        await conn.execute("INSERT INTO contacts (id) VALUES ($1)", contact_id)
        await conn.executemany(
            """
            INSERT INTO invoices (
                id, invoice_number, contact_id, customer_name, total_amount,
                amount_paid, due_date, status
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            [
                (
                    lookup_id,
                    "INV-2026-May-0185",
                    contact_id,
                    "Acme Lookup",
                    Decimal("50"),
                    Decimal("0"),
                    date(2026, 8, 1),
                    "sent",
                ),
                (
                    overdue_id,
                    "INV-OVERDUE",
                    contact_id,
                    "Acme Overdue",
                    Decimal("75"),
                    Decimal("25"),
                    date(2026, 7, 1),
                    "overdue",
                ),
                (
                    legacy_id,
                    "INV-LEGACY",
                    contact_id,
                    "Acme Legacy",
                    Decimal("100"),
                    Decimal("100"),
                    date(2026, 7, 16),
                    "paid",
                ),
                (
                    contactless_id,
                    "INV-CONTACTLESS",
                    None,
                    "Walk-in customer",
                    Decimal("100"),
                    Decimal("0"),
                    date(2026, 8, 1),
                    "sent",
                ),
            ],
        )

        legacy_payment_id = uuid4()
        await conn.execute(
            "ALTER TABLE invoice_payments DISABLE TRIGGER trg_adopt_legacy_invoice_payment"
        )
        await conn.execute(
            """
            INSERT INTO invoice_payments (
                id, invoice_id, amount, payment_date, payment_method,
                reference, created_at
            ) VALUES ($1, $2, 100, $3, 'check', 'legacy-1001', NOW())
            """,
            legacy_payment_id,
            legacy_id,
            date(2026, 7, 16),
        )
        await conn.execute(
            "ALTER TABLE invoice_payments ENABLE TRIGGER trg_adopt_legacy_invoice_payment"
        )

        pool = _SchemaPool(conn, schema)
        service = ReceivablesService(pool)
        repository = invoice_repo_mod.InvoiceRepository(
            pool=pool,
            receivables_service=service,
        )

        invoice = await repository.get_by_number(" INV-2026-MAY-0185 ")
        assert invoice is not None
        assert invoice["invoice_number"] == "INV-2026-May-0185"

        overdue = await repository.get_overdue(as_of_date=date(2026, 7, 16))
        assert [item["id"] for item in overdue] == [overdue_id]

        history = await repository.get_payments(legacy_id)
        assert len(history) == 1
        assert history[0]["id"] == legacy_payment_id
        assert history[0]["payment_status"] == "legacy"
        assert history[0]["receipt_total_amount"] == Decimal("100.00")

        behavior = await repository.get_payment_behavior(contact_id)
        assert behavior["paid_on_time"] == 1
        assert behavior["paid_late"] == 0

        first = await repository.record_payment(
            invoice_id=contactless_id,
            amount=25,
            payment_method="check",
            reference="1001",
        )
        second = await repository.record_payment(
            invoice_id=contactless_id,
            amount=25,
            payment_method="check",
            reference="1001",
        )
        assert first["payment_id"] != second["payment_id"]
        assert first["idempotency_key"].startswith("invoice-repository-")
        assert second["idempotency_key"].startswith("invoice-repository-")

        replay_first = await repository.record_payment(
            invoice_id=contactless_id,
            amount=10,
            payment_method="check",
            reference="1002",
            idempotency_key="mcp-payment-retry-1",
        )
        replay_second = await repository.record_payment(
            invoice_id=contactless_id,
            amount=10,
            payment_method="check",
            reference="1002",
            idempotency_key="mcp-payment-retry-1",
        )
        assert replay_first["payment_id"] == replay_second["payment_id"]
        assert replay_first["id"] == replay_second["id"]
        assert replay_second["idempotency_key"] == "mcp-payment-retry-1"
    finally:
        await conn.execute("SET search_path TO public")
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_real_postgres_billing_period_dedup_scoping_and_void_exclusion():
    """Cross-pipeline recurring-invoice dedup (migration 385).

    Proves, against a real Postgres constraint (not just app code):
    - get_by_contact_and_period finds a matching non-void recurring invoice
      and is a strict miss on a different contact or a different period.
    - An mcp_tool invoice in the same contact+month does NOT block a
      recurring invoice (source-scoping) -- negative control: the same
      insert with source='monthly_auto' instead raises UniqueViolationError.
    - A voided recurring invoice does NOT block re-issuance -- negative
      control: skipping the void step raises UniqueViolationError.
    - The unique index itself rejects a cross-source duplicate via raw SQL,
      independent of any app code, with its own negative controls (different
      contact, different period, mcp_tool source all succeed).
    """
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    schema = f"invoice_repository_dedup_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)

    async def _insert(
        *, contact_id, source, billing_period=None, status="draft", number=None,
    ):
        await conn.execute(
            """
            INSERT INTO invoices (
                id, invoice_number, contact_id, customer_name, due_date,
                status, source, billing_period
            ) VALUES ($1, $2, $3, 'Dedup Test Co', CURRENT_DATE, $4, $5, $6)
            """,
            uuid4(),
            number or f"INV-DEDUP-{uuid4().hex[:8]}",
            contact_id,
            status,
            source,
            billing_period,
        )

    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        await conn.execute("CREATE TABLE contacts (id uuid PRIMARY KEY)")
        await conn.execute(_migration_sql("045_invoices.sql"))
        await conn.execute(_migration_sql("385_invoices_billing_period_dedup.sql"))

        contact_a = uuid4()
        contact_b = uuid4()
        await conn.execute(
            "INSERT INTO contacts (id) VALUES ($1), ($2)", contact_a, contact_b
        )

        pool = _SchemaPool(conn, schema)
        repository = invoice_repo_mod.InvoiceRepository(pool=pool)

        # -- hit / miss on contact and period --------------------------------
        await _insert(contact_id=contact_a, source="monthly_auto", billing_period="2026-04")

        found = await repository.get_by_contact_and_period(contact_a, "2026-04")
        assert found is not None
        assert found["source"] == "monthly_auto"

        assert await repository.get_by_contact_and_period(contact_a, "2026-05") is None
        assert await repository.get_by_contact_and_period(contact_b, "2026-04") is None

        # -- mcp_tool same contact/month does not block a recurring invoice --
        await _insert(contact_id=contact_a, source="mcp_tool", billing_period=None)

        # Negative control: the same shape but source='monthly_auto' collides
        # with the existing 2026-04 monthly_auto row and is rejected.
        with pytest.raises(asyncpg.UniqueViolationError):
            async with conn.transaction():
                await _insert(contact_id=contact_a, source="monthly_auto", billing_period="2026-04")

        # -- a voided invoice does not block re-issuance ---------------------
        await conn.execute(
            "UPDATE invoices SET status = 'void' WHERE contact_id = $1 AND source = 'monthly_auto'",
            contact_a,
        )
        assert await repository.get_by_contact_and_period(contact_a, "2026-04") is None

        await _insert(contact_id=contact_a, source="eom_commercial_billing", billing_period="2026-04")
        reissued = await repository.get_by_contact_and_period(contact_a, "2026-04")
        assert reissued is not None
        assert reissued["source"] == "eom_commercial_billing"

        # Negative control: skipping the void step (i.e. two live recurring
        # rows for the same contact+period) is rejected by the same index.
        with pytest.raises(asyncpg.UniqueViolationError):
            async with conn.transaction():
                await _insert(contact_id=contact_a, source="monthly_auto", billing_period="2026-04")

        # -- DB-level negative controls, raw SQL, no app code ----------------
        await _insert(contact_id=contact_b, source="monthly_auto", billing_period="2026-06")
        with pytest.raises(asyncpg.UniqueViolationError):
            async with conn.transaction():
                await _insert(contact_id=contact_b, source="eom_commercial_billing", billing_period="2026-06")
        # different contact -> succeeds
        await _insert(contact_id=contact_a, source="eom_commercial_billing", billing_period="2026-06")
        # different period -> succeeds
        await _insert(contact_id=contact_b, source="eom_commercial_billing", billing_period="2026-07")
        # mcp_tool source, same contact+period as an existing recurring row -> succeeds
        await _insert(contact_id=contact_b, source="mcp_tool", billing_period=None)
    finally:
        await conn.execute("SET search_path TO public")
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


def test_invoices_billing_period_dedup_migration_is_additive_and_scoped():
    migration = (
        Path(__file__).parents[1]
        / "atlas_brain/storage/migrations/385_invoices_billing_period_dedup.sql"
    ).read_text()
    assert "ADD COLUMN IF NOT EXISTS billing_period" in migration
    assert "invoices_billing_period_check" in migration
    assert "idx_invoices_recurring_contact_period_source" in migration
    assert "status <> 'void'" in migration
    assert "'monthly_auto'" in migration
    assert "'eom_commercial_billing'" in migration
    # The index must key on (contact_id, billing_period) alone -- source
    # belongs only in the WHERE predicate, or the two recurring sources would
    # each get an independent slot per contact+period and never collide.
    assert "ON invoices (contact_id, billing_period)" in migration
    assert "DROP" not in "\n".join(
        line for line in migration.splitlines() if not line.lstrip().startswith("--")
    )
