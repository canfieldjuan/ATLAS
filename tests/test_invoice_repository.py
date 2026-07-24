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
