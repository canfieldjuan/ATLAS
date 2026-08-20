from __future__ import annotations

import json
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

    async def acquire(self):
        await self.conn.execute(f'SET search_path TO "{self.schema}"')
        return self.conn

    async def release(self, released) -> None:
        assert released is self.conn

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

    async def fetchval(self, query, *args):
        async with self.conn.transaction():
            await self.conn.execute(f'SET LOCAL search_path TO "{self.schema}"')
            return await self.conn.fetchval(query, *args)

    async def execute(self, query, *args):
        async with self.conn.transaction():
            await self.conn.execute(f'SET LOCAL search_path TO "{self.schema}"')
            return await self.conn.execute(query, *args)


def _migration_sql(name: str) -> str:
    return (
        Path(__file__).parents[1] / f"atlas_brain/storage/migrations/{name}"
    ).read_text(encoding="utf-8")


async def _run_migration(conn, schema: str, name: str) -> None:
    from atlas_brain.storage.migrations import run_migrations

    migrations_dir = Path(__file__).parents[1] / "atlas_brain/storage/migrations"
    await run_migrations(
        _SchemaPool(conn, schema),
        migrations_dir=migrations_dir,
        only={Path(name).stem},
    )


class _CaptureCreatePool:
    is_initialized = True

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def fetchrow(self, query, *args):
        self.queries.append(query)
        return {
            "id": args[0],
            "invoice_number": "INV-2026-Aug-0001",
            "contact_id": args[1],
            "customer_name": args[2],
            "customer_email": args[3],
            "customer_phone": args[4],
            "customer_address": args[5],
            "line_items": args[6],
            "subtotal": Decimal("10.00"),
            "tax_rate": Decimal("0"),
            "tax_amount": Decimal("0"),
            "discount_amount": Decimal("0"),
            "total_amount": Decimal("10.00"),
            "amount_paid": Decimal("0"),
            "amount_due": Decimal("10.00"),
            "issue_date": args[12],
            "due_date": args[13],
            "status": "draft",
            "source": args[14],
            "source_ref": args[15],
            "appointment_id": args[16],
            "business_context_id": args[17],
            "notes": args[19],
            "metadata": args[20],
            "invoice_for": args[21],
            "contact_name": args[22],
            "created_at": args[23],
            "updated_at": args[23],
        }


@pytest.mark.asyncio
async def test_invoice_create_omits_writer_only_billing_period_column_for_generic_invoices():
    pool = _CaptureCreatePool()
    repository = invoice_repo_mod.InvoiceRepository(pool=pool)

    await repository.create(
        customer_name="Draft Customer",
        due_date=date(2026, 8, 31),
        issue_date=date(2026, 8, 1),
        line_items=[{"description": "Cleaning", "quantity": 1, "unit_price": 10}],
        source="chatgpt_draft_writer",
        source_ref="draft:1",
    )

    assert "billing_period" not in pool.queries[0]


@pytest.mark.asyncio
async def test_invoice_create_persists_billing_period_only_for_recurring_writer():
    pool = _CaptureCreatePool()
    repository = invoice_repo_mod.InvoiceRepository(pool=pool)

    await repository.create(
        customer_name="Recurring Customer",
        due_date=date(2026, 9, 30),
        issue_date=date(2026, 9, 1),
        line_items=[{"description": "Cleaning", "quantity": 1, "unit_price": 10}],
        source="monthly_auto",
        source_ref="service_2026-08",
        billing_period=date(2026, 8, 1),
    )

    assert "billing_period" in pool.queries[0]


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
        await _run_migration(conn, schema, "385_invoices_billing_period_dedup.sql")

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


@pytest.mark.asyncio
async def test_real_postgres_billing_period_backfill_and_collision_handling():
    """Migration 385's backfill (added after Codex P1 review on ATLAS #2448):
    historical monthly_auto/eom_commercial_billing rows had billing_period =
    NULL before this migration, which is invisible to both the app-level
    pre-check and the partial unique index (NULL = 'x' is NULL, not true) --
    so a pre-deploy legacy invoice for an old period was NOT protected
    against a same-period duplicate approved after deploy, even though
    nothing bounds how far in the past an admin can approve a
    commercial-billing candidate. This backfills the period from data each
    writer already persisted, quarantining (not guessing at) any row whose
    derived period would itself collide with another row's."""
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    schema = f"invoice_repository_backfill_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)

    async def _insert(
        *, contact_id, source, number, status="draft", source_ref=None,
    ):
        await conn.execute(
            """
            INSERT INTO invoices (
                id, invoice_number, contact_id, customer_name, due_date,
                status, source, source_ref
            ) VALUES ($1, $2, $3, 'Backfill Test Co', CURRENT_DATE, $4, $5, $6)
            """,
            uuid4(), number, contact_id, status, source, source_ref,
        )

    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        await conn.execute("CREATE TABLE contacts (id uuid PRIMARY KEY)")
        await conn.execute(_migration_sql("045_invoices.sql"))

        contact_a = uuid4()   # unambiguous monthly_auto
        contact_b = uuid4()   # unambiguous eom_commercial_billing
        contact_c = uuid4()   # collision pair, both sources
        contact_d = uuid4()   # void / mcp_tool / garbage-month -- all inert
        contact_e = uuid4()   # eom_commercial_billing, sequence width > 9999
        contact_f = uuid4()   # semantically invalid 0000 year stays inert
        await conn.execute(
            "INSERT INTO contacts (id) VALUES ($1), ($2), ($3), ($4), ($5), ($6)",
            contact_a, contact_b, contact_c, contact_d, contact_e, contact_f,
        )

        # Pre-migration rows -- billing_period doesn't exist on this table yet.
        await _insert(
            contact_id=contact_a, source="monthly_auto",
            number="INV-2026-Apr-0001", source_ref="abcuuid_2026-04",
        )
        await _insert(
            contact_id=contact_b, source="eom_commercial_billing",
            number="INV-2026-Jun-0002",
        )
        await _insert(
            contact_id=contact_c, source="monthly_auto",
            number="INV-2026-May-0003", source_ref="xyzuuid_2026-05",
        )
        await _insert(
            contact_id=contact_c, source="eom_commercial_billing",
            number="INV-2026-May-0004",
        )
        await _insert(
            contact_id=contact_d, source="monthly_auto", status="void",
            number="INV-2026-Jul-0005", source_ref="voiduuid_2026-07",
        )
        await _insert(contact_id=contact_d, source="mcp_tool", number="INV-2026-Aug-0007")
        await _insert(
            contact_id=contact_d, source="eom_commercial_billing",
            number="INV-2026-Xyz-0008",
        )
        # Codex finding #4 (round 3): invoice_number's sequence segment is
        # zero-padded to a MINIMUM of 4 digits (lpad(..., 4, '0')), not a
        # fixed 4 -- once the sequence exceeds 9999, a real invoice number
        # like this one used to fail the backfill regex entirely and be
        # silently excluded from both collision detection and the partial
        # unique index.
        await _insert(
            contact_id=contact_e, source="eom_commercial_billing",
            number="INV-2026-Oct-10000",
        )
        await _insert(
            contact_id=contact_f, source="monthly_auto",
            number="INV-0000-Jan-0012", source_ref="zeroyear_0000-01",
        )
        await _insert(
            contact_id=contact_f, source="eom_commercial_billing",
            number="INV-0000-Jan-0013",
        )
        # NULL contact_id pair, same derivable period, different sources --
        # must NOT be treated as colliding: the real unique index treats
        # every NULL contact_id as distinct from every other, unlike SQL's
        # GROUP BY, which would otherwise falsely group them together.
        await _insert(
            contact_id=None, source="monthly_auto",
            number="INV-2026-Sep-0009", source_ref="nulluuid1_2026-09",
        )
        await _insert(contact_id=None, source="eom_commercial_billing", number="INV-2026-Sep-0010")

        await _run_migration(conn, schema, "385_invoices_billing_period_dedup.sql")

        rows = {
            row["invoice_number"]: row
            for row in await conn.fetch(
                "SELECT invoice_number, billing_period, billing_period_legacy_null, metadata FROM invoices"
            )
        }
        pool = _SchemaPool(conn, schema)
        repository = invoice_repo_mod.InvoiceRepository(pool=pool)

        assert rows["INV-2026-Apr-0001"]["billing_period"] == "2026-04"
        assert rows["INV-2026-Apr-0001"]["billing_period_legacy_null"] is False
        assert rows["INV-2026-Jun-0002"]["billing_period"] == "2026-06"
        assert rows["INV-2026-Jun-0002"]["billing_period_legacy_null"] is False

        # Finding #4 fix proof: a >9999 sequence number still backfills and
        # is protected, not silently excluded.
        assert rows["INV-2026-Oct-10000"]["billing_period"] == "2026-10"
        assert rows["INV-2026-Oct-10000"]["billing_period_legacy_null"] is False
        wide_seq_hit = await repository.get_by_contact_and_period(contact_e, "2026-10")
        assert wide_seq_hit is not None
        assert wide_seq_hit["invoice_number"] == "INV-2026-Oct-10000"
        with pytest.raises(asyncpg.UniqueViolationError):
            async with conn.transaction():
                await _insert(
                    contact_id=contact_e, source="monthly_auto",
                    number="INV-2026-Oct-0011", source_ref="wideseq_2026-10",
                )
                await conn.execute(
                    "UPDATE invoices SET billing_period = '2026-10' "
                    "WHERE invoice_number = 'INV-2026-Oct-0011'"
                )

        # Collision pair: both left NULL, both quarantined with the same
        # candidate period -- not deleted, not guessed at.
        for number in ("INV-2026-May-0003", "INV-2026-May-0004"):
            assert rows[number]["billing_period"] is None
            assert rows[number]["billing_period_legacy_null"] is True
            metadata = json.loads(rows[number]["metadata"])
            assert metadata["billing_period_backfill_collision"] is True
            assert metadata["billing_period_backfill_candidate_period"] == "2026-05"

        # Quarantine-reservation gap-closure proof (Codex P1, second review
        # round on ATLAS #2448): a quarantined (contact_id, period) pair is
        # left billing_period=NULL forever, which is invisible to both the
        # unique index and a naive app-level pre-check -- a THIRD invoice for
        # the same contact+period would otherwise go unblocked. Verify a
        # reservation row was recorded and the pre-check now catches it.
        reservation = await conn.fetchrow(
            "SELECT reason FROM invoices_billing_period_reservations "
            "WHERE contact_id = $1 AND billing_period = '2026-05'",
            contact_c,
        )
        assert reservation is not None
        assert reservation["reason"] == "backfill_collision"

        quarantined_hit = await repository.get_by_contact_and_period(contact_c, "2026-05")
        assert quarantined_hit is not None
        assert quarantined_hit["source"] == "quarantined_collision"

        # Negative controls: the reservation is scoped to this exact
        # contact+period, not a blanket block on the contact or the period
        # for everyone.
        assert await repository.get_by_contact_and_period(contact_c, "2026-06") is None
        assert await repository.get_by_contact_and_period(contact_a, "2026-05") is None

        # Void, mcp_tool, and garbage-month rows: untouched, no crash.
        assert rows["INV-2026-Jul-0005"]["billing_period"] is None
        assert rows["INV-2026-Jul-0005"]["billing_period_legacy_null"] is False
        assert rows["INV-2026-Aug-0007"]["billing_period"] is None
        assert rows["INV-2026-Aug-0007"]["billing_period_legacy_null"] is False
        assert rows["INV-2026-Xyz-0008"]["billing_period"] is None
        assert rows["INV-2026-Xyz-0008"]["billing_period_legacy_null"] is True
        assert rows["INV-0000-Jan-0012"]["billing_period"] is None
        assert rows["INV-0000-Jan-0012"]["billing_period_legacy_null"] is True
        assert rows["INV-0000-Jan-0013"]["billing_period"] is None
        assert rows["INV-0000-Jan-0013"]["billing_period_legacy_null"] is True

        # The NOT VALID write-time guard still permits updates to legacy rows
        # explicitly marked by this migration, but rejects fresh recurring
        # invoice rows that do not claim a billing_period.
        await conn.execute(
            "UPDATE invoices SET status = 'sent' WHERE invoice_number = 'INV-2026-Xyz-0008'"
        )
        with pytest.raises(asyncpg.CheckViolationError):
            async with conn.transaction():
                await _insert(
                    contact_id=contact_a,
                    source="monthly_auto",
                    number="INV-2026-Nov-0001",
                    source_ref="fresh_2026-11",
                )

        # NULL-contact_id pair: backfilled independently, NOT quarantined --
        # negative control for the collision-detection NULL-safety fix.
        assert rows["INV-2026-Sep-0009"]["billing_period"] == "2026-09"
        assert rows["INV-2026-Sep-0010"]["billing_period"] == "2026-09"
        assert json.loads(rows["INV-2026-Sep-0009"]["metadata"]) == {}
        assert json.loads(rows["INV-2026-Sep-0010"]["metadata"]) == {}

        # Gap-closure proof: the backfilled legacy row is now findable by the
        # app-level pre-check, and a second recurring invoice for the same
        # contact+period is rejected by the database itself.
        found = await repository.get_by_contact_and_period(contact_a, "2026-04")
        assert found is not None
        assert found["invoice_number"] == "INV-2026-Apr-0001"

        with pytest.raises(asyncpg.UniqueViolationError):
            async with conn.transaction():
                await _insert(
                    contact_id=contact_a, source="eom_commercial_billing",
                    number="INV-2026-Apr-9999",
                )
                await conn.execute(
                    "UPDATE invoices SET billing_period = '2026-04' "
                    "WHERE invoice_number = 'INV-2026-Apr-9999'"
                )

        # Idempotency: re-running the migration against the already-migrated
        # schema changes nothing and does not raise.
        before = await conn.fetch(
            "SELECT invoice_number, billing_period, metadata FROM invoices ORDER BY invoice_number"
        )
        await _run_migration(conn, schema, "385_invoices_billing_period_dedup.sql")
        after = await conn.fetch(
            "SELECT invoice_number, billing_period, metadata FROM invoices ORDER BY invoice_number"
        )
        assert [dict(r) for r in before] == [dict(r) for r in after]
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
    assert "ADD COLUMN IF NOT EXISTS billing_period_legacy_null" in migration
    assert "invoices_billing_period_check" in migration
    assert "idx_invoices_recurring_contact_period_source" in migration
    assert "CREATE UNIQUE INDEX CONCURRENTLY idx_invoices_recurring_contact_period_source" in migration
    assert "DROP INDEX CONCURRENTLY IF EXISTS idx_invoices_recurring_contact_period_source" in migration
    assert "NOT VALID" in migration
    assert "000[1-9]" in migration
    assert "status <> 'void'" in migration
    assert "'monthly_auto'" in migration
    assert "'eom_commercial_billing'" in migration
    # The index must key on (contact_id, billing_period) alone -- source
    # belongs only in the WHERE predicate, or the two recurring sources would
    # each get an independent slot per contact+period and never collide.
    assert "ON invoices (contact_id, billing_period)" in migration
    assert "billing_period_backfill_collision" in migration
    assert "invoices_billing_period_backfill_candidates" in migration
    assert "OR billing_period_legacy_null" in migration
    assert "HAVING count(*) > 1" in migration
    assert "-- atlas: atomic-bookkeeping" not in migration
    assert "CREATE TABLE IF NOT EXISTS invoices_billing_period_reservations" in migration
    assert "PRIMARY KEY (contact_id, billing_period)" in migration
