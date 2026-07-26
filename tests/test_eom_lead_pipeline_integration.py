"""Real-Postgres reachability proof for the EOM lead pipeline slice."""

from __future__ import annotations

import os
import uuid
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI

asyncpg = pytest.importorskip("asyncpg")

from atlas_brain.api import leads as leads_mod  # noqa: E402
from atlas_brain.services.crm_provider import DatabaseCRMProvider  # noqa: E402
from atlas_brain.services.eom_lead_ingress import (  # noqa: E402
    resolve_or_create_eom_inbound_lead,
)


ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS = ROOT / "atlas_brain" / "storage" / "migrations"
DATABASE_URL_ENV = "ATLAS_MIGRATION_TEST_DATABASE_URL"


@pytest.mark.asyncio
async def test_intake_to_pipeline_roundtrip_preserves_managed_state(monkeypatch):
    database_url = os.environ.get(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")

    schema = f"atlas_lead_pipeline_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}", public')
        await conn.execute("CREATE TABLE appointments (id UUID PRIMARY KEY)")
        for name in (
            "035_contacts.sql",
            "256_contact_interaction_dedupe.sql",
            "346_contact_lead_pipeline.sql",
            "351_eom_lead_lifecycle_events.sql",
        ):
            await conn.execute((MIGRATIONS / name).read_text())

        import atlas_brain.storage.database as db_mod

        monkeypatch.setattr(db_mod, "get_db_pool", lambda: conn)
        from atlas_brain.config import settings

        monkeypatch.setattr(settings.email, "enabled", False)
        provider = DatabaseCRMProvider()
        app = FastAPI()
        app.include_router(leads_mod.router, prefix="/api/v1")
        app.dependency_overrides[leads_mod._crm_dependency] = lambda: provider
        app.dependency_overrides[leads_mod._email_dependency] = (
            lambda: MagicMock(send=AsyncMock())
        )

        async def zero_count(*_args):
            return 0

        app.dependency_overrides[leads_mod._daily_count_dependency] = (
            lambda: zero_count
        )
        app.dependency_overrides[leads_mod._ack_volume_dependency] = (
            lambda: zero_count
        )

        payload = {
            "name": "Pipeline Proof",
            "email": "pipeline-proof@example.com",
            "gclid": "first-click",
        }
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            created = await client.post("/api/v1/leads/intake", json=payload)
            assert created.status_code == 200

            contact = await conn.fetchrow(
                "SELECT * FROM contacts WHERE email = $1", payload["email"]
            )
            assert contact["lead_stage"] == "new"
            assert contact["business_context_id"] == "effingham_maids"
            assert await conn.fetchval(
                """
                SELECT COUNT(*) FROM eom_lead_lifecycle_events
                WHERE contact_id = $1 AND event_type = 'lead_created'
                """,
                contact["id"],
            ) == 1
            assert await conn.fetchval(
                """
                SELECT confdeltype = 'r'
                FROM pg_constraint
                WHERE conrelid = 'eom_lead_lifecycle_events'::regclass
                  AND contype = 'f'
                """
            ) is True
            with pytest.raises(asyncpg.RaiseError, match="append-only"):
                await conn.execute(
                    "DELETE FROM eom_lead_lifecycle_events WHERE contact_id = $1",
                    contact["id"],
                )

            follow_up = datetime(2026, 7, 25, 15, tzinfo=timezone.utc)
            with pytest.raises(ValueError, match="funnel transition service"):
                await provider.update_contact(
                    str(contact["id"]),
                    {
                        "lead_stage": "qualified",
                        "lead_owner": "Juan",
                        "next_follow_up_at": follow_up,
                    },
                    require_contact_type="lead",
                )
            with pytest.raises(ValueError, match="funnel transition service"):
                await provider.update_contact(
                    str(contact["id"]), {"contact_type": "customer"}
                )
            eom_customer_id = uuid.uuid4()
            await conn.execute(
                """
                INSERT INTO contacts (
                    id, full_name, business_context_id, contact_type, status
                ) VALUES ($1, 'Already Customer', 'effingham_maids', 'customer', 'active')
                """,
                eom_customer_id,
            )
            with pytest.raises(ValueError, match="funnel transition service"):
                await provider.update_contact(
                    str(eom_customer_id), {"contact_type": "lead"}
                )
            legacy_contact_id = uuid.uuid4()
            await conn.execute(
                """
                INSERT INTO contacts (id, full_name, contact_type, status)
                VALUES ($1, 'Legacy Contact', 'customer', 'active')
                """,
                legacy_contact_id,
            )
            with pytest.raises(ValueError, match="funnel transition service"):
                await provider.update_contact(
                    str(legacy_contact_id),
                    {
                        "business_context_id": "effingham_maids",
                        "contact_type": "lead",
                    },
                )
            repeated = await client.post("/api/v1/leads/intake", json=payload)
            assert repeated.status_code == 200
            changed_attribution = await client.post(
                "/api/v1/leads/intake",
                json={**payload, "gclid": "second-click"},
            )
            assert changed_attribution.status_code == 200
            assert await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM contact_interactions
                WHERE contact_id = $1 AND interaction_type = 'web_form'
                """,
                contact["id"],
            ) == 2
            captured_clicks = await conn.fetch(
                """
                SELECT metadata->'attribution'->>'gclid' AS gclid
                FROM contact_interactions
                WHERE contact_id = $1 AND interaction_type = 'web_form'
                ORDER BY occurred_at, id
                """,
                contact["id"],
            )
            assert {row["gclid"] for row in captured_clicks} == {
                "first-click",
                "second-click",
            }

        persisted = await conn.fetchrow(
            """
            SELECT contact_type, lead_stage, lead_owner, next_follow_up_at
            FROM contacts
            WHERE id = $1
            """,
            contact["id"],
        )
        assert dict(persisted) == {
            "contact_type": "lead",
            "lead_stage": "new",
            "lead_owner": None,
            "next_follow_up_at": None,
        }

        await conn.executemany(
            """
            INSERT INTO contacts (
                full_name, business_context_id, contact_type, status,
                lead_stage, next_follow_up_at
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            [
                (
                    "Foreign Earlier",
                    "churnsignals",
                    "lead",
                    "active",
                    "qualified",
                    datetime(2026, 7, 25, 12, tzinfo=timezone.utc),
                ),
                (
                    "Non-lead Earlier",
                    "effingham_maids",
                    "customer",
                    "active",
                    "qualified",
                    datetime(2026, 7, 25, 13, tzinfo=timezone.utc),
                ),
                (
                    "Legacy Due",
                    None,
                    "lead",
                    "active",
                    "qualified",
                    datetime(2026, 7, 25, 16, tzinfo=timezone.utc),
                ),
                (
                    "Archived Earlier",
                    "effingham_maids",
                    "lead",
                    "archived",
                    "qualified",
                    datetime(2026, 7, 25, 11, tzinfo=timezone.utc),
                ),
                (
                    "Tenant Future",
                    "effingham_maids",
                    "lead",
                    "active",
                    "qualified",
                    datetime(2026, 7, 25, 17, tzinfo=timezone.utc),
                ),
            ],
        )
        due = await provider.list_contacts(
            business_context_id="effingham_maids",
            include_unclaimed_legacy=True,
            contact_type="lead",
            lead_stage="qualified",
            next_follow_up_before=datetime(
                2026, 7, 25, 16, tzinfo=timezone.utc
            ),
            limit=2,
        )
        assert [row["full_name"] for row in due] == ["Legacy Due"]
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


class _TransactionPool:
    """Small asyncpg pool adapter matching Atlas' transaction boundary."""

    def __init__(self, pool):
        self._pool = pool

    @asynccontextmanager
    async def transaction(self):
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                yield connection

    async def fetchrow(self, *args):
        async with self._pool.acquire() as connection:
            return await connection.fetchrow(*args)

    @asynccontextmanager
    async def acquire(self):
        async with self._pool.acquire() as connection:
            yield connection


@pytest.mark.asyncio
async def test_atomic_eom_inbound_identity_creates_one_contact_and_one_ledger_event(monkeypatch):
    database_url = os.environ.get(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")

    schema = f"atlas_eom_inbound_{uuid.uuid4().hex}"
    setup = await asyncpg.connect(database_url)
    pool = None
    try:
        await setup.execute(f'CREATE SCHEMA "{schema}"')
        await setup.execute(f'SET search_path TO "{schema}", public')
        await setup.execute("CREATE TABLE appointments (id UUID PRIMARY KEY)")
        for name in (
            "035_contacts.sql",
            "346_contact_lead_pipeline.sql",
            "351_eom_lead_lifecycle_events.sql",
        ):
            await setup.execute((MIGRATIONS / name).read_text())

        pool = await asyncpg.create_pool(
            database_url,
            min_size=2,
            max_size=2,
            server_settings={"search_path": f'"{schema}", public'},
        )
        import atlas_brain.storage.database as db_mod

        monkeypatch.setattr(db_mod, "get_db_pool", lambda: _TransactionPool(pool))
        provider = DatabaseCRMProvider()

        async def resolve(source_ref: str):
            return await resolve_or_create_eom_inbound_lead(
                provider,
                full_name="Concurrent Inbound",
                phone="217-555-0199",
                email="concurrent@example.com",
                address=None,
                source="web",
                source_ref=source_ref,
            )

        first, second = await asyncio.gather(resolve("web-1"), resolve("web-2"))
        assert first["id"] == second["id"]
        async with pool.acquire() as check:
            assert await check.fetchval("SELECT COUNT(*) FROM contacts") == 1
            assert await check.fetchval(
                "SELECT COUNT(*) FROM eom_lead_lifecycle_events WHERE event_type = 'lead_created'"
            ) == 1
    finally:
        if pool is not None:
            await pool.close()
        await setup.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await setup.close()


@pytest.mark.asyncio
async def test_atomic_eom_inbound_resolution_is_active_phone_first_and_blocks_claimable_lifecycle_writes(monkeypatch):
    database_url = os.environ.get(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")

    schema = f"atlas_eom_resolution_{uuid.uuid4().hex}"
    setup = await asyncpg.connect(database_url)
    pool = None
    try:
        await setup.execute(f'CREATE SCHEMA "{schema}"')
        await setup.execute(f'SET search_path TO "{schema}", public')
        await setup.execute("CREATE TABLE appointments (id UUID PRIMARY KEY)")
        for name in (
            "035_contacts.sql",
            "346_contact_lead_pipeline.sql",
            "351_eom_lead_lifecycle_events.sql",
        ):
            await setup.execute((MIGRATIONS / name).read_text())

        pool = await asyncpg.create_pool(
            database_url,
            min_size=2,
            max_size=2,
            server_settings={"search_path": f'"{schema}", public'},
        )
        import atlas_brain.storage.database as db_mod

        monkeypatch.setattr(db_mod, "get_db_pool", lambda: _TransactionPool(pool))
        provider = DatabaseCRMProvider()

        eom_email_id = uuid.uuid4()
        legacy_phone_id = uuid.uuid4()
        archived_id = uuid.uuid4()
        claimable_legacy_id = uuid.uuid4()
        await setup.executemany(
            """
            INSERT INTO contacts (
                id, full_name, email, phone, business_context_id, contact_type,
                status
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            [
                (
                    eom_email_id,
                    "Shared Email",
                    "shared@example.com",
                    "2175550999",
                    "effingham_maids",
                    "lead",
                    "active",
                ),
                (
                    legacy_phone_id,
                    "Legacy Phone",
                    "legacy@example.com",
                    "2175550100",
                    None,
                    "lead",
                    "active",
                ),
                (
                    archived_id,
                    "Archived Match",
                    "archived@example.com",
                    "2175550200",
                    "effingham_maids",
                    "lead",
                    "archived",
                ),
                (
                    claimable_legacy_id,
                    "Claimable Legacy",
                    "claimable@example.com",
                    "2175550300",
                    None,
                    "lead",
                    "active",
                ),
            ],
        )

        phone_match = await resolve_or_create_eom_inbound_lead(
            provider,
            full_name="Inbound Caller",
            phone="217-555-0100",
            email="shared@example.com",
            address=None,
            source="web",
            source_ref="phone-before-email",
        )
        assert phone_match["id"] == legacy_phone_id
        assert phone_match["_was_created"] is False

        archived_match = await resolve_or_create_eom_inbound_lead(
            provider,
            full_name="Replacement Lead",
            phone="217-555-0200",
            email="archived@example.com",
            address=None,
            source="web",
            source_ref="archived-not-resolved",
        )
        assert archived_match["id"] != archived_id
        assert archived_match["_was_created"] is True
        assert archived_match["status"] == "active"

        lifecycle_update = asyncio.create_task(
            provider.update_contact(
                str(claimable_legacy_id),
                {"lead_stage": "qualified"},
                require_contact_type="lead",
            )
        )
        claimed = await provider.claim_contact(
            str(claimable_legacy_id), "effingham_maids"
        )
        with pytest.raises(ValueError, match="funnel transition service"):
            await lifecycle_update
        assert claimed is not None
        assert claimed["business_context_id"] == "effingham_maids"

        with pytest.raises(ValueError, match="funnel transition service"):
            await provider.update_contact(
                str(claimable_legacy_id), {"contact_type": "customer"}
            )
        triage_update = await provider.update_contact(
            str(claimable_legacy_id),
            {
                "lead_owner": "Juan",
                "next_follow_up_at": datetime(2026, 8, 1, 15, tzinfo=timezone.utc),
            },
            require_contact_type="lead",
        )
        assert triage_update is not None
        assert triage_update["lead_owner"] == "Juan"
    finally:
        if pool is not None:
            await pool.close()
        await setup.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await setup.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing_ledger_sql",
    [
        "DROP TABLE eom_lead_lifecycle_events CASCADE",
        "DROP TRIGGER trg_record_eom_lead_created ON contacts",
        "ALTER TABLE contacts DISABLE TRIGGER trg_record_eom_lead_created",
    ],
)
async def test_atomic_eom_inbound_rejects_when_lifecycle_ledger_is_incomplete(
    monkeypatch, missing_ledger_sql
):
    database_url = os.environ.get(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")

    schema = f"atlas_eom_unready_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}", public')
        await conn.execute("CREATE TABLE appointments (id UUID PRIMARY KEY)")
        for name in (
            "035_contacts.sql",
            "346_contact_lead_pipeline.sql",
            "351_eom_lead_lifecycle_events.sql",
        ):
            await conn.execute((MIGRATIONS / name).read_text())
        await conn.execute(missing_ledger_sql)

        import atlas_brain.storage.database as db_mod

        monkeypatch.setattr(db_mod, "get_db_pool", lambda: conn)
        provider = DatabaseCRMProvider()

        with pytest.raises(RuntimeError, match="migration 351 lifecycle ledger"):
            await resolve_or_create_eom_inbound_lead(
                provider,
                full_name="Unavailable Ledger",
                phone="217-555-0199",
                email="ledger@example.com",
                address=None,
                source="web",
                source_ref="ledger-unavailable",
            )
        assert await conn.fetchval("SELECT COUNT(*) FROM contacts") == 0
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()
