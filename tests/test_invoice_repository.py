from __future__ import annotations

import inspect
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


async def _install_recorded_initial_385_catalog(conn) -> None:
    """Reconstruct the exact first revision recorded by live migration 385.

    The deployed ledger has migration 385 recorded, but its catalog only has
    these original three objects. Keep this fixture inline rather than reading
    a historical Git object so the regression remains self-contained after
    repository history is shallow-cloned or archived.
    """
    await conn.execute(
        """
        ALTER TABLE invoices
            ADD COLUMN billing_period VARCHAR(7);

        ALTER TABLE invoices
            ADD CONSTRAINT invoices_billing_period_check
            CHECK (billing_period ~ '^[0-9]{4}-(0[1-9]|1[0-2])$');

        CREATE UNIQUE INDEX idx_invoices_recurring_contact_period_source
            ON invoices (contact_id, billing_period)
            WHERE billing_period IS NOT NULL
              AND status <> 'void'
              AND source IN ('monthly_auto', 'eom_commercial_billing');

        CREATE TABLE schema_migrations (
            version INTEGER PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            content_sha256 VARCHAR(64),
            applied_at TIMESTAMPTZ DEFAULT NOW()
        );
        INSERT INTO schema_migrations (version, name, content_sha256)
        VALUES (385, '385_invoices_billing_period_dedup', NULL);
        """
    )


async def _install_pending_gmail_replacement_trigger(conn) -> None:
    """Install migration 377's exact mutation trigger on a minimal real catalog."""
    await conn.execute(
        """
        CREATE TABLE commercial_billing_candidate_approvals (
            id UUID PRIMARY KEY,
            invoice_id UUID NOT NULL
        );
        CREATE TABLE commercial_billing_invoice_gmail_drafts (
            id UUID PRIMARY KEY,
            approval_id UUID NOT NULL,
            state VARCHAR(32) NOT NULL,
            draft_generation INTEGER NOT NULL
        );
        CREATE TABLE commercial_billing_invoice_gmail_draft_replacement_events (
            id UUID PRIMARY KEY,
            gmail_draft_record_id UUID NOT NULL,
            replacement_generation INTEGER NOT NULL
        );
        """
    )
    trigger_start = (
        "CREATE OR REPLACE FUNCTION "
        "commercial_billing_reject_invoice_mutation_while_gmail_replacement_pending()"
    )
    migration = _migration_sql("377_commercial_billing_gmail_draft_replacements.sql")
    _, separator, trigger_sql = migration.partition(trigger_start)
    assert separator
    await conn.execute(trigger_start + trigger_sql)


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
async def test_real_postgres_recurring_dedup_readiness_rejects_drifted_definitions():
    """Readiness must prove definitions, not names.

    An externally managed schema can contain same-named valid constraints or
    indexes with weaker definitions than migration 385. Same name + valid flag
    is not enough to protect cross-source recurring invoice dedup.
    """
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    schema = f"invoice_repository_ready_defs_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)

    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        await conn.execute("CREATE TABLE contacts (id uuid PRIMARY KEY)")
        await conn.execute(_migration_sql("045_invoices.sql"))
        await _run_migration(conn, schema, "385_invoices_billing_period_dedup.sql")

        assert await invoice_repo_mod.recurring_invoice_dedup_schema_ready(conn) is True

        await conn.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS idx_invoices_recurring_contact_period_source"
        )
        await conn.execute(
            """
            CREATE UNIQUE INDEX CONCURRENTLY idx_invoices_recurring_contact_period_source
            ON invoices (contact_id, billing_period, source)
            WHERE source IN ('monthly_auto', 'eom_commercial_billing')
              AND status <> 'void'
            """
        )
        assert await invoice_repo_mod.recurring_invoice_dedup_schema_ready(conn) is False

        await conn.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS idx_invoices_recurring_contact_period_source"
        )
        await conn.execute(
            """
            CREATE UNIQUE INDEX CONCURRENTLY idx_invoices_recurring_contact_period_source
            ON invoices (contact_id, billing_period)
            WHERE billing_period IS NOT NULL
              AND source IN ('monthly_auto', 'eom_commercial_billing')
              AND status <> 'void'
            """
        )
        assert await invoice_repo_mod.recurring_invoice_dedup_schema_ready(conn) is True

        await conn.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS idx_invoices_recurring_contact_period_source"
        )
        await conn.execute(
            """
            CREATE UNIQUE INDEX CONCURRENTLY idx_invoices_recurring_contact_period_source
            ON invoices (contact_id, billing_period)
            WHERE billing_period IS NOT NULL
              AND source IN ('monthly_auto', 'eom_commercial_billing')
              AND status <> 'void'
              AND source = 'never'
            """
        )
        assert await invoice_repo_mod.recurring_invoice_dedup_schema_ready(conn) is False

        await conn.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS idx_invoices_recurring_contact_period_source"
        )
        await conn.execute(
            """
            CREATE UNIQUE INDEX CONCURRENTLY idx_invoices_recurring_contact_period_source
            ON invoices (contact_id, billing_period)
            WHERE billing_period IS NOT NULL
              AND source IN ('monthly_auto', 'eom_commercial_billing')
              AND status <> 'void'
            """
        )
        assert await invoice_repo_mod.recurring_invoice_dedup_schema_ready(conn) is True

        await conn.execute(
            "ALTER TABLE invoices DROP CONSTRAINT IF EXISTS "
            "invoices_recurring_billing_period_required_check"
        )
        await conn.execute(
            """
            ALTER TABLE invoices
                ADD CONSTRAINT invoices_recurring_billing_period_required_check
                CHECK (
                    source NOT IN ('monthly_auto', 'eom_commercial_billing')
                    OR status = 'void'
                    OR billing_period IS NOT NULL
                )
                NOT VALID
            """
        )
        assert await invoice_repo_mod.recurring_invoice_dedup_schema_ready(conn) is False

        await conn.execute(
            "ALTER TABLE invoices DROP CONSTRAINT "
            "invoices_recurring_billing_period_required_check"
        )
        await conn.execute(
            """
            ALTER TABLE invoices
                ADD CONSTRAINT invoices_recurring_billing_period_required_check
                CHECK (
                    source NOT IN ('monthly_auto', 'eom_commercial_billing')
                    OR status = 'void'
                    OR billing_period IS NOT NULL
                    OR billing_period_legacy_null
                )
                NOT VALID
            """
        )
        assert await invoice_repo_mod.recurring_invoice_dedup_schema_ready(conn) is True

        # Every expected word remains present, but the added NULL branch makes
        # the check a fresh-write bypass. Exact canonical comparison must fail.
        await conn.execute(
            "ALTER TABLE invoices DROP CONSTRAINT "
            "invoices_recurring_billing_period_required_check"
        )
        await conn.execute(
            """
            ALTER TABLE invoices
                ADD CONSTRAINT invoices_recurring_billing_period_required_check
                CHECK (
                    source NOT IN ('monthly_auto', 'eom_commercial_billing')
                    OR status = 'void'
                    OR billing_period IS NOT NULL
                    OR billing_period_legacy_null
                    OR billing_period IS NULL
                )
                NOT VALID
            """
        )
        assert await invoice_repo_mod.recurring_invoice_dedup_schema_ready(conn) is False

        await conn.execute(
            "ALTER TABLE invoices DROP CONSTRAINT "
            "invoices_recurring_billing_period_required_check"
        )
        await conn.execute(
            """
            ALTER TABLE invoices
                ADD CONSTRAINT invoices_recurring_billing_period_required_check
                CHECK (
                    source NOT IN ('monthly_auto', 'eom_commercial_billing')
                    OR status = 'void'
                    OR billing_period IS NOT NULL
                    OR billing_period_legacy_null
                )
                NOT VALID
            """
        )
        assert await invoice_repo_mod.recurring_invoice_dedup_schema_ready(conn) is True

        # The period check can carry the same regex fragments while admitting
        # every non-NULL string; readiness must reject that named tautology too.
        await conn.execute(
            "ALTER TABLE invoices DROP CONSTRAINT invoices_billing_period_check"
        )
        await conn.execute(
            """
            ALTER TABLE invoices
                ADD CONSTRAINT invoices_billing_period_check
                CHECK (
                    billing_period ~
                        '^(000[1-9]|00[1-9][0-9]|0[1-9][0-9]{2}|[1-9][0-9]{3})-(0[1-9]|1[0-2])$'
                    OR billing_period IS NOT NULL
                )
                NOT VALID
            """
        )
        assert await invoice_repo_mod.recurring_invoice_dedup_schema_ready(conn) is False

        await conn.execute(
            "ALTER TABLE invoices DROP CONSTRAINT invoices_billing_period_check"
        )
        await conn.execute(
            """
            ALTER TABLE invoices
                ADD CONSTRAINT invoices_billing_period_check
                CHECK (
                    billing_period ~
                        '^(000[1-9]|00[1-9][0-9]|0[1-9][0-9]{2}|[1-9][0-9]{3})-(0[1-9]|1[0-2])$'
                )
                NOT VALID
            """
        )
        assert await invoice_repo_mod.recurring_invoice_dedup_schema_ready(conn) is True
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
                "SELECT id, invoice_number, status, billing_period, "
                "billing_period_legacy_null, metadata FROM invoices"
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
                await conn.execute(
                    """
                    INSERT INTO invoices (
                        id, invoice_number, contact_id, customer_name,
                        due_date, status, source, source_ref, billing_period
                    ) VALUES (
                        $1, 'INV-2026-Oct-0011', $2, 'Backfill Test Co',
                        CURRENT_DATE, 'draft', 'monthly_auto', 'wideseq_2026-10', '2026-10'
                    )
                    """,
                    uuid4(),
                    contact_e,
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

        # Review round 4: reservation blocking must follow the same non-void
        # invariant as the unique index. Once every quarantined invoice for
        # the contact+period is voided, the reservation remains as historical
        # evidence but no longer blocks a clean reissue.
        await conn.execute(
            """
            UPDATE invoices
            SET status = 'void'
            WHERE contact_id = $1
              AND metadata->>'billing_period_backfill_collision' = 'true'
              AND metadata->>'billing_period_backfill_candidate_period' = '2026-05'
            """,
            contact_c,
        )
        assert await repository.get_by_contact_and_period(contact_c, "2026-05") is None
        await conn.execute(
            """
            INSERT INTO invoices (
                id, invoice_number, contact_id, customer_name, due_date,
                status, source, source_ref, billing_period
            ) VALUES (
                $1, 'INV-2026-May-9998', $2, 'Backfill Test Co', CURRENT_DATE,
                'draft', 'eom_commercial_billing', 'reissue_2026-05', '2026-05'
            )
            """,
            uuid4(),
            contact_c,
        )
        reissued = await repository.get_by_contact_and_period(contact_c, "2026-05")
        assert reissued is not None
        assert reissued["invoice_number"] == "INV-2026-May-9998"
        with pytest.raises(invoice_repo_mod.DatabaseOperationError):
            await repository.update_status(rows["INV-2026-May-0003"]["id"], "sent")
        resurrected = await conn.fetchrow(
            "SELECT status FROM invoices WHERE invoice_number = 'INV-2026-May-0003'"
        )
        assert resurrected["status"] == "void"

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
                await conn.execute(
                    """
                    INSERT INTO invoices (
                        id, invoice_number, contact_id, customer_name, due_date,
                        status, source, billing_period
                    ) VALUES (
                        $1, 'INV-2026-Apr-9999', $2, 'Backfill Test Co',
                        CURRENT_DATE, 'draft', 'eom_commercial_billing', '2026-04'
                    )
                    """,
                    uuid4(),
                    contact_a,
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


@pytest.mark.asyncio
async def test_real_postgres_recorded_initial_385_recovery_converges_history():
    """Migration 387 recovers the observed recorded-initial-385 catalog.

    This starts from the actual old catalog shape, not a fresh current 385
    migration. It includes the partial-deploy state where a newer writer has
    already populated a period that a historical NULL candidate would claim.
    The recovery must quarantine that candidate rather than raise a unique
    error or silently crown either historical invoice as the winner.
    """
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    schema = f"invoice_repository_385_recovery_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)

    async def _insert(
        *,
        contact_id,
        source,
        number,
        source_ref=None,
        status="draft",
        billing_period=None,
        total_amount=Decimal("100.00"),
    ):
        invoice_id = uuid4()
        await conn.execute(
            """
            INSERT INTO invoices (
                id, invoice_number, contact_id, customer_name, due_date,
                status, source, source_ref, billing_period, total_amount
            ) VALUES ($1, $2, $3, 'Recovery Test Co', CURRENT_DATE, $4, $5, $6, $7, $8)
            """,
            invoice_id,
            number,
            contact_id,
            status,
            source,
            source_ref,
            billing_period,
            total_amount,
        )
        return invoice_id

    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        await conn.execute("CREATE TABLE contacts (id uuid PRIMARY KEY)")
        await conn.execute(_migration_sql("045_invoices.sql"))
        await _install_recorded_initial_385_catalog(conn)

        (
            contact_monthly,
            contact_commercial,
            contact_collision,
            contact_mixed,
            contact_unparseable,
            contact_void,
            contact_mcp,
            contact_zero_year,
            contact_fresh,
        ) = [uuid4() for _ in range(9)]
        await conn.executemany(
            "INSERT INTO contacts (id) VALUES ($1)",
            [
                (contact_monthly,),
                (contact_commercial,),
                (contact_collision,),
                (contact_mixed,),
                (contact_unparseable,),
                (contact_void,),
                (contact_mcp,),
                (contact_zero_year,),
                (contact_fresh,),
            ],
        )

        monthly_id = await _insert(
            contact_id=contact_monthly,
            source="monthly_auto",
            number="INV-2026-Apr-0001",
            source_ref="monthly_2026-04",
            total_amount=Decimal("50.00"),
        )
        await _insert(
            contact_id=contact_commercial,
            source="eom_commercial_billing",
            number="INV-2026-Jun-10000",
            total_amount=Decimal("60.00"),
        )
        await _insert(
            contact_id=contact_collision,
            source="monthly_auto",
            number="INV-2026-May-0003",
            source_ref="collision_2026-05",
        )
        await _insert(
            contact_id=contact_collision,
            source="eom_commercial_billing",
            number="INV-2026-May-0004",
        )
        await _insert(
            contact_id=contact_mixed,
            source="monthly_auto",
            number="INV-2026-Jul-0005",
            source_ref="populated_2026-07",
            billing_period="2026-07",
        )
        await _insert(
            contact_id=contact_mixed,
            source="eom_commercial_billing",
            number="INV-2026-Jul-0006",
        )
        await _insert(
            contact_id=contact_unparseable,
            source="eom_commercial_billing",
            number="INV-2026-Xyz-0007",
        )
        await _insert(
            contact_id=contact_zero_year,
            source="monthly_auto",
            number="INV-0000-Jan-0008",
            source_ref="zero_0000-01",
        )
        await _insert(
            contact_id=contact_void,
            source="monthly_auto",
            number="INV-2026-Aug-0009",
            source_ref="void_2026-08",
            status="void",
        )
        await _insert(
            contact_id=contact_mcp,
            source="mcp_tool",
            number="INV-2026-Sep-0010",
        )
        await _insert(
            contact_id=None,
            source="monthly_auto",
            number="INV-2026-Sep-0011",
            source_ref="contactless_a_2026-09",
        )
        await _insert(
            contact_id=None,
            source="eom_commercial_billing",
            number="INV-2026-Sep-0012",
        )
        legacy_payment_id = uuid4()
        await conn.execute(
            """
            INSERT INTO invoice_payments (id, invoice_id, amount, payment_method, reference)
            VALUES ($1, $2, 50.00, 'check', 'migration-385-recovery-proof')
            """,
            legacy_payment_id,
            monthly_id,
        )

        immutable_before = [
            dict(row)
            for row in await conn.fetch(
                """
                SELECT id, invoice_number, contact_id, status, source, source_ref,
                       total_amount, amount_paid
                FROM invoices
                ORDER BY invoice_number
                """
            )
        ]
        payments_before = [
            dict(row)
            for row in await conn.fetch(
                "SELECT id, invoice_id, amount, payment_method, reference "
                "FROM invoice_payments ORDER BY id"
            )
        ]

        assert await invoice_repo_mod.recurring_invoice_dedup_schema_ready(conn) is False
        assert await conn.fetchval(
            "SELECT content_sha256 FROM schema_migrations WHERE name = $1",
            "385_invoices_billing_period_dedup",
        ) is None

        await _run_migration(
            conn, schema, "387_eom_recurring_invoice_dedup_recovery.sql"
        )

        assert await invoice_repo_mod.recurring_invoice_dedup_schema_ready(conn) is True
        migration_rows = await conn.fetch(
            "SELECT version, name, content_sha256 FROM schema_migrations ORDER BY version"
        )
        assert [row["name"] for row in migration_rows] == [
            "385_invoices_billing_period_dedup",
            "387_eom_recurring_invoice_dedup_recovery",
        ]
        assert migration_rows[0]["content_sha256"] is None
        assert migration_rows[1]["content_sha256"] is not None

        immutable_after = [
            dict(row)
            for row in await conn.fetch(
                """
                SELECT id, invoice_number, contact_id, status, source, source_ref,
                       total_amount, amount_paid
                FROM invoices
                ORDER BY invoice_number
                """
            )
        ]
        payments_after = [
            dict(row)
            for row in await conn.fetch(
                "SELECT id, invoice_id, amount, payment_method, reference "
                "FROM invoice_payments ORDER BY id"
            )
        ]
        assert immutable_after == immutable_before
        assert payments_after == payments_before

        rows = {
            row["invoice_number"]: row
            for row in await conn.fetch(
                """
                SELECT invoice_number, billing_period, billing_period_legacy_null,
                       metadata
                FROM invoices
                """
            )
        }
        assert rows["INV-2026-Apr-0001"]["billing_period"] == "2026-04"
        assert rows["INV-2026-Apr-0001"]["billing_period_legacy_null"] is False
        assert rows["INV-2026-Jun-10000"]["billing_period"] == "2026-06"
        assert rows["INV-2026-Jun-10000"]["billing_period_legacy_null"] is False

        for number in ("INV-2026-May-0003", "INV-2026-May-0004"):
            assert rows[number]["billing_period"] is None
            assert rows[number]["billing_period_legacy_null"] is True
            metadata = json.loads(rows[number]["metadata"])
            assert metadata["billing_period_backfill_collision"] is True
            assert metadata["billing_period_backfill_candidate_period"] == "2026-05"

        # A partially deployed newer writer has already claimed July. The
        # legacy NULL row is quarantined; it is never overwritten or inserted
        # into the unique slot that belongs to the populated invoice.
        assert rows["INV-2026-Jul-0005"]["billing_period"] == "2026-07"
        assert rows["INV-2026-Jul-0005"]["billing_period_legacy_null"] is False
        assert rows["INV-2026-Jul-0006"]["billing_period"] is None
        assert rows["INV-2026-Jul-0006"]["billing_period_legacy_null"] is True
        mixed_metadata = json.loads(rows["INV-2026-Jul-0006"]["metadata"])
        assert mixed_metadata["billing_period_backfill_collision"] is True
        assert mixed_metadata["billing_period_backfill_candidate_period"] == "2026-07"

        for number in ("INV-2026-Xyz-0007", "INV-0000-Jan-0008"):
            assert rows[number]["billing_period"] is None
            assert rows[number]["billing_period_legacy_null"] is True
            assert json.loads(rows[number]["metadata"])["billing_period_legacy_null"] is True

        # Void and ad-hoc records are intentionally outside the recurring
        # recovery boundary. NULL-contact candidates backfill independently,
        # just as PostgreSQL's unique index treats each NULL key independently.
        assert rows["INV-2026-Aug-0009"]["billing_period"] is None
        assert rows["INV-2026-Aug-0009"]["billing_period_legacy_null"] is False
        assert rows["INV-2026-Sep-0010"]["billing_period"] is None
        assert rows["INV-2026-Sep-0010"]["billing_period_legacy_null"] is False
        assert rows["INV-2026-Sep-0011"]["billing_period"] == "2026-09"
        assert rows["INV-2026-Sep-0012"]["billing_period"] == "2026-09"

        reservation_rows = await conn.fetch(
            "SELECT contact_id, billing_period, reason "
            "FROM invoices_billing_period_reservations"
        )
        reservations = {
            (row["contact_id"], row["billing_period"]): row["reason"]
            for row in reservation_rows
        }
        assert reservations == {
            (contact_collision, "2026-05"): "backfill_collision",
            (contact_mixed, "2026-07"): "backfill_collision",
        }

        repository = invoice_repo_mod.InvoiceRepository(pool=_SchemaPool(conn, schema))
        collision_hit = await repository.get_by_contact_and_period(
            contact_collision, "2026-05"
        )
        assert collision_hit is not None
        assert collision_hit["source"] == "quarantined_collision"
        mixed_hit = await repository.get_by_contact_and_period(contact_mixed, "2026-07")
        assert mixed_hit is not None
        assert mixed_hit["invoice_number"] == "INV-2026-Jul-0005"

        with pytest.raises(asyncpg.CheckViolationError):
            async with conn.transaction():
                await _insert(
                    contact_id=contact_fresh,
                    source="monthly_auto",
                    number="INV-2026-Oct-0013",
                    source_ref="fresh_2026-10",
                )
        with pytest.raises(asyncpg.CheckViolationError):
            async with conn.transaction():
                await _insert(
                    contact_id=contact_fresh,
                    source="monthly_auto",
                    number="INV-0000-Jan-0014",
                    source_ref="fresh_0000-01",
                    billing_period="0000-01",
                )
        with pytest.raises(asyncpg.UniqueViolationError):
            async with conn.transaction():
                await _insert(
                    contact_id=contact_monthly,
                    source="eom_commercial_billing",
                    number="INV-2026-Apr-0015",
                    billing_period="2026-04",
                )

        recovery_before_retry = [
            dict(row)
            for row in await conn.fetch(
                """
                SELECT invoice_number, billing_period, billing_period_legacy_null, metadata
                FROM invoices
                ORDER BY invoice_number
                """
            )
        ]
        reservations_before_retry = [
            dict(row)
            for row in await conn.fetch(
                "SELECT contact_id, billing_period, reason "
                "FROM invoices_billing_period_reservations ORDER BY contact_id, billing_period"
            )
        ]
        await _run_migration(
            conn, schema, "387_eom_recurring_invoice_dedup_recovery.sql"
        )
        recovery_after_retry = [
            dict(row)
            for row in await conn.fetch(
                """
                SELECT invoice_number, billing_period, billing_period_legacy_null, metadata
                FROM invoices
                ORDER BY invoice_number
                """
            )
        ]
        reservations_after_retry = [
            dict(row)
            for row in await conn.fetch(
                "SELECT contact_id, billing_period, reason "
                "FROM invoices_billing_period_reservations ORDER BY contact_id, billing_period"
            )
        ]
        assert recovery_after_retry == recovery_before_retry
        assert reservations_after_retry == reservations_before_retry
    finally:
        await conn.execute("SET search_path TO public")
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_real_postgres_recorded_initial_385_recovery_is_atomic_on_bad_period():
    """A malformed preexisting non-NULL period must leave no partial recovery."""
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    schema = f"invoice_repository_385_recovery_atomic_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        await conn.execute("CREATE TABLE contacts (id uuid PRIMARY KEY)")
        await conn.execute(_migration_sql("045_invoices.sql"))
        await _install_recorded_initial_385_catalog(conn)
        contact_id = uuid4()
        await conn.execute("INSERT INTO contacts (id) VALUES ($1)", contact_id)
        await conn.execute(
            """
            INSERT INTO invoices (
                id, invoice_number, contact_id, customer_name, due_date,
                status, source, billing_period
            ) VALUES (
                $1, 'INV-0000-Jan-atomic', $2, 'Atomic Recovery Co', CURRENT_DATE,
                'draft', 'monthly_auto', '0000-01'
            )
            """,
            uuid4(),
            contact_id,
        )

        with pytest.raises(
            asyncpg.PostgresError,
            match="Cannot recover migration 385: invoices.billing_period",
        ):
            await _run_migration(
                conn, schema, "387_eom_recurring_invoice_dedup_recovery.sql"
            )

        assert await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'invoices'
                  AND column_name = 'billing_period_legacy_null'
            )
            """
        ) is False
        assert await conn.fetchval(
            "SELECT to_regclass('invoices_billing_period_reservations') IS NULL"
        ) is True
        assert await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM schema_migrations WHERE name = $1)",
            "387_eom_recurring_invoice_dedup_recovery",
        ) is False
        assert await conn.fetchval(
            "SELECT billing_period FROM invoices WHERE invoice_number = 'INV-0000-Jan-atomic'"
        ) == "0000-01"
    finally:
        await conn.execute("SET search_path TO public")
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_real_postgres_387_rejects_wrong_recurring_index_predicate_atomically():
    """A token-similar predicate must not strand a recorded recovery."""
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    schema = f"invoice_repository_387_wrong_index_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        await conn.execute("CREATE TABLE contacts (id uuid PRIMARY KEY)")
        await conn.execute(_migration_sql("045_invoices.sql"))
        await _install_recorded_initial_385_catalog(conn)
        await conn.execute(
            """
            DROP INDEX idx_invoices_recurring_contact_period_source;
            CREATE UNIQUE INDEX idx_invoices_recurring_contact_period_source
                ON invoices (contact_id, billing_period)
                WHERE billing_period IS NOT NULL
                  AND status = 'void'
                  AND source IN ('monthly_auto', 'eom_commercial_billing');
            """
        )

        with pytest.raises(
            asyncpg.PostgresError,
            match="recurring billing_period index is missing, invalid, or has an unexpected definition",
        ):
            await _run_migration(
                conn, schema, "387_eom_recurring_invoice_dedup_recovery.sql"
            )

        assert await conn.fetchval(
            """
            SELECT NOT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'invoices'
                  AND column_name = 'billing_period_legacy_null'
            )
            """
        ) is True
        assert await conn.fetchval(
            "SELECT to_regclass('invoices_billing_period_reservations') IS NULL"
        ) is True
        assert await conn.fetchval(
            "SELECT NOT EXISTS (SELECT 1 FROM schema_migrations WHERE name = $1)",
            "387_eom_recurring_invoice_dedup_recovery",
        ) is True
    finally:
        await conn.execute("SET search_path TO public")
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_real_postgres_387_replaces_token_similar_weak_constraints():
    """Named tautologies must be replaced before migration 387 is recorded."""
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    schema = f"invoice_repository_387_weak_constraints_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        await conn.execute("CREATE TABLE contacts (id uuid PRIMARY KEY)")
        await conn.execute(_migration_sql("045_invoices.sql"))
        await _install_recorded_initial_385_catalog(conn)
        await conn.execute(
            """
            ALTER TABLE invoices DROP CONSTRAINT invoices_billing_period_check;
            ALTER TABLE invoices
                ADD CONSTRAINT invoices_billing_period_check
                CHECK (
                    billing_period ~
                        '^(000[1-9]|00[1-9][0-9]|0[1-9][0-9]{2}|[1-9][0-9]{3})-(0[1-9]|1[0-2])$'
                        OR billing_period IS NOT NULL
                );
            ALTER TABLE invoices
                ADD COLUMN billing_period_legacy_null BOOLEAN NOT NULL DEFAULT false;
            ALTER TABLE invoices
                ADD CONSTRAINT invoices_recurring_billing_period_required_check
                CHECK (
                    source NOT IN ('monthly_auto', 'eom_commercial_billing')
                    OR status = 'void'
                    OR billing_period IS NOT NULL
                    OR billing_period_legacy_null
                    OR billing_period IS NULL
                ) NOT VALID;
            """
        )

        await _run_migration(
            conn, schema, "387_eom_recurring_invoice_dedup_recovery.sql"
        )
        assert await invoice_repo_mod.recurring_invoice_dedup_schema_ready(conn) is True
        constraints = {
            row["conname"]: invoice_repo_mod._canonicalize_catalog_constraint_expression(
                row["definition"]
            )
            for row in await conn.fetch(
                """
                SELECT conname, pg_get_expr(conbin, conrelid) AS definition
                FROM pg_constraint
                WHERE conrelid = 'invoices'::regclass
                  AND conname = ANY($1::text[])
                """,
                list(invoice_repo_mod._RECURRING_INVOICE_DEDUP_CONSTRAINTS),
            )
        }
        assert constraints == invoice_repo_mod._RECURRING_INVOICE_DEDUP_CONSTRAINT_EXPRESSIONS

        contact_id = uuid4()
        await conn.execute("INSERT INTO contacts (id) VALUES ($1)", contact_id)
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                """
                INSERT INTO invoices (
                    id, invoice_number, contact_id, customer_name, due_date,
                    status, source
                ) VALUES (
                    $1, 'INV-2026-Oct-null', $2, 'Constraint Recovery Co', CURRENT_DATE,
                    'draft', 'monthly_auto'
                )
                """,
                uuid4(),
                contact_id,
            )
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                """
                INSERT INTO invoices (
                    id, invoice_number, contact_id, customer_name, due_date,
                    status, source, billing_period
                ) VALUES (
                    $1, 'INV-0000-Jan-invalid', $2, 'Constraint Recovery Co', CURRENT_DATE,
                    'draft', 'monthly_auto', '0000-01'
                )
                """,
                uuid4(),
                contact_id,
            )
        assert await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM schema_migrations WHERE name = $1)",
            "387_eom_recurring_invoice_dedup_recovery",
        ) is True
    finally:
        await conn.execute("SET search_path TO public")
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_real_postgres_387_preflights_pending_gmail_replacement_before_updates():
    """The migration waits for migration 377's exact pending-replacement trigger."""
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    schema = f"invoice_repository_387_pending_replacement_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        await conn.execute("CREATE TABLE contacts (id uuid PRIMARY KEY)")
        await conn.execute(_migration_sql("045_invoices.sql"))
        await _install_recorded_initial_385_catalog(conn)
        contact_id, invoice_id, approval_id, draft_id = [uuid4() for _ in range(4)]
        await conn.execute("INSERT INTO contacts (id) VALUES ($1)", contact_id)
        await conn.execute(
            """
            INSERT INTO invoices (
                id, invoice_number, contact_id, customer_name, due_date,
                status, source
            ) VALUES (
                $1, 'INV-2026-Aug-0001', $2, 'Pending Replacement Co', CURRENT_DATE,
                'draft', 'eom_commercial_billing'
            )
            """,
            invoice_id,
            contact_id,
        )
        await _install_pending_gmail_replacement_trigger(conn)
        await conn.execute(
            """
            INSERT INTO commercial_billing_candidate_approvals (id, invoice_id)
            VALUES ($1, $2)
            """,
            approval_id,
            invoice_id,
        )
        await conn.execute(
            """
            INSERT INTO commercial_billing_invoice_gmail_drafts (
                id, approval_id, state, draft_generation
            ) VALUES ($1, $2, 'creating', 2)
            """,
            draft_id,
            approval_id,
        )
        await conn.execute(
            """
            INSERT INTO commercial_billing_invoice_gmail_draft_replacement_events (
                id, gmail_draft_record_id, replacement_generation
            ) VALUES ($1, $2, 2)
            """,
            uuid4(),
            draft_id,
        )

        with pytest.raises(
            asyncpg.CheckViolationError,
            match="Commercial billing invoice mutation is blocked",
        ):
            await conn.execute(
                "UPDATE invoices SET customer_name = customer_name WHERE id = $1",
                invoice_id,
            )
        with pytest.raises(
            asyncpg.PostgresError,
            match="pending Gmail draft replacement blocks a recurring legacy invoice update",
        ):
            await _run_migration(
                conn, schema, "387_eom_recurring_invoice_dedup_recovery.sql"
            )

        assert await conn.fetchval(
            """
            SELECT NOT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'invoices'
                  AND column_name = 'billing_period_legacy_null'
            )
            """
        ) is True
        assert await conn.fetchval(
            "SELECT to_regclass('invoices_billing_period_reservations') IS NULL"
        ) is True
        assert await conn.fetchval(
            "SELECT NOT EXISTS (SELECT 1 FROM schema_migrations WHERE name = $1)",
            "387_eom_recurring_invoice_dedup_recovery",
        ) is True
        assert await conn.fetchval(
            "SELECT billing_period FROM invoices WHERE id = $1", invoice_id
        ) is None

        await conn.execute(
            "UPDATE commercial_billing_invoice_gmail_drafts SET state = 'draft_created' WHERE id = $1",
            draft_id,
        )
        await _run_migration(
            conn, schema, "387_eom_recurring_invoice_dedup_recovery.sql"
        )
        assert await invoice_repo_mod.recurring_invoice_dedup_schema_ready(conn) is True
        recovered = await conn.fetchrow(
            "SELECT billing_period, billing_period_legacy_null FROM invoices WHERE id = $1",
            invoice_id,
        )
        assert dict(recovered) == {
            "billing_period": "2026-08",
            "billing_period_legacy_null": False,
        }
    finally:
        await conn.execute("SET search_path TO public")
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_real_postgres_387_rollback_fence_allows_retained_old_writer():
    """The documented constraint removal makes the retained pre-2448 writer usable."""
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    schema = f"invoice_repository_387_rollback_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        await conn.execute("CREATE TABLE contacts (id uuid PRIMARY KEY)")
        await conn.execute(_migration_sql("045_invoices.sql"))
        await _install_recorded_initial_385_catalog(conn)
        contact_id = uuid4()
        await conn.execute("INSERT INTO contacts (id) VALUES ($1)", contact_id)
        await _run_migration(
            conn, schema, "387_eom_recurring_invoice_dedup_recovery.sql"
        )
        assert await invoice_repo_mod.recurring_invoice_dedup_schema_ready(conn) is True

        legacy_insert = """
            INSERT INTO invoices (
                id, invoice_number, contact_id, customer_name, due_date,
                status, source, source_ref
            ) VALUES (
                $1, $2, $3, 'Retained Writer Co', CURRENT_DATE,
                'draft', 'monthly_auto', 'retained_2026-09'
            )
        """
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(legacy_insert, uuid4(), "INV-2026-Sep-before", contact_id)

        await conn.execute(
            """
            ALTER TABLE invoices DROP CONSTRAINT IF EXISTS
                invoices_recurring_billing_period_required_check
            """
        )
        legacy_invoice_id = uuid4()
        await conn.execute(
            legacy_insert,
            legacy_invoice_id,
            "INV-2026-Sep-after",
            contact_id,
        )
        inserted = await conn.fetchrow(
            """
            SELECT billing_period, billing_period_legacy_null
            FROM invoices
            WHERE id = $1
            """,
            legacy_invoice_id,
        )
        assert dict(inserted) == {
            "billing_period": None,
            "billing_period_legacy_null": False,
        }
        assert await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM schema_migrations WHERE name = $1)",
            "387_eom_recurring_invoice_dedup_recovery",
        ) is True
    finally:
        await conn.execute("SET search_path TO public")
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_real_postgres_387_is_data_noop_for_final_385_catalog():
    """A fresh final-385 schema needs only a new 387 ledger row."""
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    schema = f"invoice_repository_387_noop_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        await conn.execute("CREATE TABLE contacts (id uuid PRIMARY KEY)")
        await conn.execute(_migration_sql("045_invoices.sql"))
        contact_unique = uuid4()
        contact_collision = uuid4()
        await conn.execute(
            "INSERT INTO contacts (id) VALUES ($1), ($2)",
            contact_unique,
            contact_collision,
        )
        await conn.executemany(
            """
            INSERT INTO invoices (
                id, invoice_number, contact_id, customer_name, due_date,
                status, source, source_ref
            ) VALUES ($1, $2, $3, 'Final Catalog Co', CURRENT_DATE, 'draft', $4, $5)
            """,
            [
                (
                    uuid4(),
                    "INV-2026-Apr-final",
                    contact_unique,
                    "monthly_auto",
                    "final_2026-04",
                ),
                (
                    uuid4(),
                    "INV-2026-May-final-a",
                    contact_collision,
                    "monthly_auto",
                    "final_collision_2026-05",
                ),
                (
                    uuid4(),
                    "INV-2026-May-final-b",
                    contact_collision,
                    "eom_commercial_billing",
                    None,
                ),
            ],
        )
        await _run_migration(conn, schema, "385_invoices_billing_period_dedup.sql")
        assert await invoice_repo_mod.recurring_invoice_dedup_schema_ready(conn) is True

        invoices_before = [
            dict(row)
            for row in await conn.fetch(
                """
                SELECT invoice_number, billing_period, billing_period_legacy_null, metadata
                FROM invoices
                ORDER BY invoice_number
                """
            )
        ]
        reservations_before = [
            dict(row)
            for row in await conn.fetch(
                "SELECT contact_id, billing_period, reason, created_at "
                "FROM invoices_billing_period_reservations ORDER BY contact_id, billing_period"
            )
        ]
        constraints_before = [
            dict(row)
            for row in await conn.fetch(
                """
                SELECT conname, pg_get_constraintdef(oid) AS definition
                FROM pg_constraint
                WHERE conrelid = 'invoices'::regclass
                  AND conname = ANY($1::text[])
                ORDER BY conname
                """,
                [
                    "invoices_billing_period_check",
                    "invoices_recurring_billing_period_required_check",
                ],
            )
        ]
        index_before = dict(
            await conn.fetchrow(
                """
                SELECT index_state.indisunique, index_state.indisvalid,
                       index_state.indisready,
                       pg_get_indexdef(index_state.indexrelid) AS definition,
                       pg_get_expr(index_state.indpred, index_state.indrelid) AS predicate
                FROM pg_index AS index_state
                JOIN pg_class AS index_class ON index_class.oid = index_state.indexrelid
                WHERE index_class.relname = 'idx_invoices_recurring_contact_period_source'
                """
            )
        )
        comments_before = dict(
            await conn.fetchrow(
                """
                SELECT
                    MAX(col_description('invoices'::regclass, attribute.attnum))
                        FILTER (WHERE attribute.attname = 'billing_period') AS billing_period,
                    MAX(col_description('invoices'::regclass, attribute.attnum))
                        FILTER (WHERE attribute.attname = 'billing_period_legacy_null')
                        AS billing_period_legacy_null,
                    MAX(obj_description('invoices_billing_period_reservations'::regclass))
                        AS reservations
                FROM pg_attribute AS attribute
                WHERE attribute.attrelid = 'invoices'::regclass
                  AND attribute.attname IN ('billing_period', 'billing_period_legacy_null')
                """
            )
        )

        await _run_migration(
            conn, schema, "387_eom_recurring_invoice_dedup_recovery.sql"
        )
        assert await invoice_repo_mod.recurring_invoice_dedup_schema_ready(conn) is True

        invoices_after = [
            dict(row)
            for row in await conn.fetch(
                """
                SELECT invoice_number, billing_period, billing_period_legacy_null, metadata
                FROM invoices
                ORDER BY invoice_number
                """
            )
        ]
        reservations_after = [
            dict(row)
            for row in await conn.fetch(
                "SELECT contact_id, billing_period, reason, created_at "
                "FROM invoices_billing_period_reservations ORDER BY contact_id, billing_period"
            )
        ]
        constraints_after = [
            dict(row)
            for row in await conn.fetch(
                """
                SELECT conname, pg_get_constraintdef(oid) AS definition
                FROM pg_constraint
                WHERE conrelid = 'invoices'::regclass
                  AND conname = ANY($1::text[])
                ORDER BY conname
                """,
                [
                    "invoices_billing_period_check",
                    "invoices_recurring_billing_period_required_check",
                ],
            )
        ]
        index_after = dict(
            await conn.fetchrow(
                """
                SELECT index_state.indisunique, index_state.indisvalid,
                       index_state.indisready,
                       pg_get_indexdef(index_state.indexrelid) AS definition,
                       pg_get_expr(index_state.indpred, index_state.indrelid) AS predicate
                FROM pg_index AS index_state
                JOIN pg_class AS index_class ON index_class.oid = index_state.indexrelid
                WHERE index_class.relname = 'idx_invoices_recurring_contact_period_source'
                """
            )
        )
        comments_after = dict(
            await conn.fetchrow(
                """
                SELECT
                    MAX(col_description('invoices'::regclass, attribute.attnum))
                        FILTER (WHERE attribute.attname = 'billing_period') AS billing_period,
                    MAX(col_description('invoices'::regclass, attribute.attnum))
                        FILTER (WHERE attribute.attname = 'billing_period_legacy_null')
                        AS billing_period_legacy_null,
                    MAX(obj_description('invoices_billing_period_reservations'::regclass))
                        AS reservations
                FROM pg_attribute AS attribute
                WHERE attribute.attrelid = 'invoices'::regclass
                  AND attribute.attname IN ('billing_period', 'billing_period_legacy_null')
                """
            )
        )
        assert invoices_after == invoices_before
        assert reservations_after == reservations_before
        assert constraints_after == constraints_before
        assert index_after == index_before
        assert comments_after == comments_before

        migration_names = await conn.fetch(
            "SELECT name, content_sha256 FROM schema_migrations ORDER BY version"
        )
        assert [row["name"] for row in migration_names] == [
            "385_invoices_billing_period_dedup",
            "387_eom_recurring_invoice_dedup_recovery",
        ]
        assert all(row["content_sha256"] is not None for row in migration_names)
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
    assert "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS idx_invoices_recurring_contact_period_source" in migration
    assert "DROP INDEX CONCURRENTLY IF EXISTS idx_invoices_recurring_contact_period_source_invalid" in migration
    assert "NOT (index_state.indisvalid AND index_state.indisready)" in migration
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
    assert "ALTER TABLE invoices DROP CONSTRAINT IF EXISTS invoices_recurring_billing_period_required_check" in migration
    assert "CREATE TABLE IF NOT EXISTS invoices_billing_period_reservations" in migration
    assert "PRIMARY KEY (contact_id, billing_period)" in migration


def test_invoice_status_update_blocks_quarantined_void_resurrection_in_repository():
    source = inspect.getsource(invoice_repo_mod.InvoiceRepository.update_status)
    assert "billing_period_legacy_null" in source
    assert "billing_period IS NULL" in source
    assert "status = 'void'" in source
    assert "$2 <> 'void'" in source
    assert "source IN ('monthly_auto', 'eom_commercial_billing')" in source


def test_recurring_dedup_predicate_readiness_rejects_extra_clauses():
    good = (
        "((billing_period IS NOT NULL) "
        "AND ((source)::text = ANY "
        "(ARRAY['monthly_auto'::text, 'eom_commercial_billing'::text])) "
        "AND ((status)::text <> 'void'::text))"
    )
    impossible_extra_clause = (
        "((billing_period IS NOT NULL) "
        "AND ((source)::text = ANY "
        "(ARRAY['monthly_auto'::text, 'eom_commercial_billing'::text])) "
        "AND ((status)::text <> 'void'::text) "
        "AND ((source)::text = 'never'::text))"
    )

    assert invoice_repo_mod._recurring_index_predicate_ready(good) is True
    assert (
        invoice_repo_mod._recurring_index_predicate_ready(impossible_extra_clause)
        is False
    )
