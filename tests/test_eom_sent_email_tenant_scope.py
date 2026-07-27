"""PostgreSQL proof for EOM acknowledgement history tenant isolation."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI

asyncpg = pytest.importorskip("asyncpg")

from atlas_brain.api import leads as leads_mod  # noqa: E402
from atlas_brain.services.crm_provider import DatabaseCRMProvider  # noqa: E402
from atlas_brain.storage.repositories.email import EmailRepository  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS = ROOT / "atlas_brain" / "storage" / "migrations"
DATABASE_URL_ENV = "ATLAS_MIGRATION_TEST_DATABASE_URL"
TENANT = "effingham_maids"
FOREIGN_TENANT = "churnsignals"


class _PoolAdapter:
    def __init__(self, pool):
        self._pool = pool
        self.is_initialized = True

    async def fetch(self, query, *args):
        return await self._pool.fetch(query, *args)

    async def fetchrow(self, query, *args):
        return await self._pool.fetchrow(query, *args)

    async def fetchval(self, query, *args):
        return await self._pool.fetchval(query, *args)

    async def execute(self, query, *args):
        return await self._pool.execute(query, *args)


class _ForbiddenInboxProvider:
    async def list_messages(self, **_kwargs):
        raise AssertionError("scoped customer context opened the global inbox")


class _ReassigningEmailRepository(EmailRepository):
    """Move the owning contact after the email read but before serialization."""

    def __init__(self, pool, contact_id, new_owner):
        self._pool = pool
        self._contact_id = contact_id
        self._new_owner = new_owner
        self._reassigned = False

    async def query(self, *args, **kwargs):
        rows = await super().query(*args, **kwargs)
        if not self._reassigned:
            self._reassigned = True
            await self._pool.execute(
                """
                UPDATE contacts
                SET business_context_id = $2
                WHERE id = $1
                """,
                self._contact_id,
                self._new_owner,
            )
        return rows


@pytest.mark.asyncio
async def test_intake_history_and_scoped_context_are_tenant_exact(
    monkeypatch,
    caplog,
):
    database_url = os.environ.get(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")

    schema = f"atlas_eom_sent_email_{uuid.uuid4().hex}"
    admin_conn = await asyncpg.connect(database_url)
    pool = None
    db_mod = None
    original_db_pool = None
    email_repo_mod = None
    original_email_repo = None
    try:
        caplog.set_level(logging.INFO, logger="atlas.storage.email")
        await admin_conn.execute(f'CREATE SCHEMA "{schema}"')
        await admin_conn.execute(f'SET search_path TO "{schema}", public')
        for name in (
            "001_initial_schema.sql",
            "012_appointments.sql",
            "016_sent_emails.sql",
        ):
            await admin_conn.execute((MIGRATIONS / name).read_text())

        legacy_email_id = await admin_conn.fetchval(
            """
            INSERT INTO sent_emails (to_addresses, subject, body, sent_at)
            VALUES ($1, 'legacy', 'unclassified', $2)
            RETURNING id
            """,
            ["tenant-proof@example.com"],
            datetime.now(timezone.utc) - timedelta(days=1),
        )
        migration_349 = (
            MIGRATIONS / "349_sent_emails_business_context.sql"
        ).read_text()
        await admin_conn.execute(migration_349)
        await admin_conn.execute(migration_349)
        assert (
            await admin_conn.fetchval(
                """
            SELECT business_context_id
            FROM sent_emails
            WHERE id = $1
            """,
                legacy_email_id,
            )
            is None
        )
        with pytest.raises(asyncpg.CheckViolationError):
            await admin_conn.execute(
                """
                INSERT INTO sent_emails (
                    to_addresses, subject, body, business_context_id
                )
                VALUES ($1, 'invalid', 'invalid', '   ')
                """,
                ["tenant-proof@example.com"],
            )

        for name in (
            "030_call_transcripts.sql",
            "035_contacts.sql",
            "036_call_contact_link.sql",
            "044_sms_messages.sql",
            "045_invoices.sql",
            "256_contact_interaction_dedupe.sql",
            "346_contact_lead_pipeline.sql",
            "351_eom_lead_lifecycle_events.sql",
            "352_eom_inbound_delivery_receipts.sql",
        ):
            await admin_conn.execute((MIGRATIONS / name).read_text())

        async def set_search_path(connection):
            await connection.execute(f'SET search_path TO "{schema}", public')

        pool = await asyncpg.create_pool(
            database_url,
            min_size=1,
            max_size=8,
            setup=set_search_path,
        )
        adapter = _PoolAdapter(pool)

        import atlas_brain.mcp.crm_server as crm_server
        import atlas_brain.services.crm_provider as crm_provider_mod
        import atlas_brain.services.customer_context as context_mod
        import atlas_brain.services.email_provider as email_provider_mod
        import atlas_brain.storage.database as db_mod
        import atlas_brain.storage.repositories.email as email_repo_mod
        from atlas_brain.config import settings

        original_db_pool = db_mod._db_pool
        db_mod._db_pool = adapter
        provider = DatabaseCRMProvider()
        monkeypatch.setattr(crm_provider_mod, "_crm_provider", provider)
        context_service = context_mod.CustomerContextService()
        context_service._get_inbox_emails = AsyncMock(
            side_effect=AssertionError(
                "scoped customer context invoked inbox enrichment"
            )
        )
        context_service._get_b2b_churn_signals = AsyncMock(
            side_effect=AssertionError("scoped customer context invoked B2B enrichment")
        )
        monkeypatch.setattr(
            context_mod,
            "_customer_context_service",
            context_service,
        )
        monkeypatch.setattr(
            email_provider_mod,
            "_email_provider",
            _ForbiddenInboxProvider(),
        )
        original_email_repo = email_repo_mod._email_repo
        email_repo_mod._email_repo = EmailRepository()
        monkeypatch.setattr(settings.email, "enabled", True)
        monkeypatch.setattr(
            settings.mcp,
            "crm_default_business_context",
            TENANT,
        )
        crm_server.set_provider_override(lambda: provider)
        existing_contact = await pool.fetchrow(
            """
            INSERT INTO contacts (
                full_name, email, phone, business_context_id
            )
            VALUES (
                'Tenant Proof', 'stored-address@example.com',
                '2175550188', $1
            )
            RETURNING id
            """,
            TENANT,
        )
        missing_email_contact = await pool.fetchrow(
            """
            INSERT INTO contacts (
                full_name, phone, business_context_id
            )
            VALUES ('Missing Email Proof', '2175550199', $1)
            RETURNING id
            """,
            TENANT,
        )

        transport = MagicMock()
        transport.send = AsyncMock(
            side_effect=[
                {"success": True, "message_id": "ACK-PROOF-1"},
                {"success": True, "message_id": "ACK-PROOF-2"},
            ]
        )

        async def zero(*_args):
            return 0

        app = FastAPI()
        app.include_router(leads_mod.router, prefix="/api/v1")
        app.dependency_overrides[leads_mod._crm_dependency] = lambda: provider
        app.dependency_overrides[leads_mod._email_dependency] = lambda: transport
        app.dependency_overrides[leads_mod._email_history_dependency] = EmailRepository
        app.dependency_overrides[leads_mod._daily_count_dependency] = lambda: zero
        app.dependency_overrides[leads_mod._ack_volume_dependency] = lambda: zero

        payload = {
            "name": "Tenant Proof",
            "email": "tenant-proof@example.com",
            "phone": "217-555-0188",
            "service": "office cleaning",
            "frequency": "weekly",
        }
        missing_email_payload = {
            "name": "Missing Email Proof",
            "email": "submitted-only@example.com",
            "phone": "217-555-0199",
        }
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            created = await client.post("/api/v1/leads/intake", json=payload)
            repeated = await client.post("/api/v1/leads/intake", json=payload)
            missing_email_created = await client.post(
                "/api/v1/leads/intake",
                json=missing_email_payload,
            )

        assert created.status_code == 200
        assert created.json() == {"success": True, "email_sent": True}
        assert repeated.status_code == 200
        assert repeated.json() == {"success": True, "email_sent": False}
        assert missing_email_created.status_code == 200
        assert missing_email_created.json() == {
            "success": True,
            "email_sent": True,
        }
        assert transport.send.await_count == 2
        assert payload["email"] not in caplog.text
        assert missing_email_payload["email"] not in caplog.text
        assert "We received your estimate request" not in caplog.text

        contact = await pool.fetchrow(
            """
            SELECT id, business_context_id
            FROM contacts
            WHERE id = $1
            """,
            existing_contact["id"],
        )
        assert contact["business_context_id"] == TENANT
        stored = await pool.fetchrow(
            """
            SELECT subject, business_context_id, template_type,
                   resend_message_id, metadata
            FROM sent_emails
            WHERE resend_message_id = 'ACK-PROOF-1'
            """
        )
        stored_dict = dict(stored)
        if isinstance(stored_dict["metadata"], str):
            stored_dict["metadata"] = json.loads(stored_dict["metadata"])
        assert stored_dict == {
            "subject": ("We received your estimate request - Effingham Office Maids"),
            "business_context_id": TENANT,
            "template_type": "request_acknowledgement",
            "resend_message_id": "ACK-PROOF-1",
            "metadata": {
                "source": "website_estimate_form",
                "contact_id": str(contact["id"]),
            },
        }

        now = datetime.now(timezone.utc)
        await pool.executemany(
            """
            INSERT INTO sent_emails (
                to_addresses, subject, body, sent_at, business_context_id
            )
            VALUES ($1, $2, $3, $4, $5)
            """,
            [
                (
                    [payload["email"]],
                    "FOREIGN-SECRET",
                    "foreign",
                    now + timedelta(minutes=2),
                    FOREIGN_TENANT,
                ),
                (
                    [payload["email"]],
                    "NULL-SECRET",
                    "unclassified",
                    now + timedelta(minutes=1),
                    None,
                ),
            ],
        )
        repository = EmailRepository()
        ack_id = await pool.fetchval(
            """
            SELECT id
            FROM sent_emails
            WHERE resend_message_id = 'ACK-PROOF-1'
            """
        )
        foreign_id = await pool.fetchval(
            """
            SELECT id
            FROM sent_emails
            WHERE subject = 'FOREIGN-SECRET'
            """
        )
        assert (
            await repository.get_by_id(
                ack_id,
                business_context_id=TENANT,
            )
        ).subject == stored["subject"]
        assert (
            "business_context_id"
            not in (
                await repository.get_by_id(
                    ack_id,
                    business_context_id=TENANT,
                )
            ).to_dict()
        )
        assert (
            await repository.get_by_id(
                foreign_id,
                business_context_id=TENANT,
            )
        ) is None
        assert await repository.count(business_context_id=TENANT) == 2
        recent = [
            row.subject
            for row in await repository.get_recent(
                hours=1,
                business_context_id=TENANT,
            )
        ]
        assert recent == [stored["subject"], stored["subject"]]
        today = [
            row.subject
            for row in await repository.get_today(
                business_context_id=TENANT,
            )
        ]
        assert today == [stored["subject"], stored["subject"]]
        with pytest.raises(ValueError, match="must not be blank"):
            await repository.create(
                to_addresses=[payload["email"]],
                subject="invalid",
                body="invalid",
                business_context_id="  ",
            )

        scoped = json.loads(
            await crm_server.get_customer_context(
                contact_id=str(contact["id"]),
                business_context_id=TENANT,
                max_emails=1,
            )
        )
        assert scoped["found"] is True
        assert [row["subject"] for row in scoped["sent_emails"]] == [stored["subject"]]
        assert "business_context_id" not in scoped["sent_emails"][0]
        assert scoped["inbox_emails"] == []
        assert scoped["b2b_churn_signals"] == []
        assert scoped["email_sources_omitted_under_scope"] == ["inbox_emails"]
        assert scoped["emails_omitted_under_scope"] is True
        assert scoped["b2b_enrichment_omitted_under_scope"] is True
        context_service._get_inbox_emails.assert_not_awaited()
        context_service._get_b2b_churn_signals.assert_not_awaited()

        monkeypatch.setattr(
            settings.mcp,
            "crm_default_business_context",
            None,
        )
        unscoped_contact_id = await pool.fetchval(
            """
            INSERT INTO contacts (
                full_name, email, business_context_id
            )
            VALUES ('Unscoped History', 'unscoped-history@example.com', NULL)
            RETURNING id
            """
        )
        await EmailRepository().create(
            to_addresses=["unscoped-history@example.com"],
            subject="unscoped compatibility",
            body="legacy response shape",
            business_context_id=TENANT,
        )
        unscoped_context = json.loads(
            await crm_server.get_customer_context(
                contact_id=str(unscoped_contact_id),
                max_emails=10,
            )
        )
        assert unscoped_context["found"] is True
        assert [
            row["subject"] for row in unscoped_context["sent_emails"]
        ] == ["unscoped compatibility"]
        assert all(
            "business_context_id" not in row
            for row in unscoped_context["sent_emails"]
        )
        monkeypatch.setattr(
            settings.mcp,
            "crm_default_business_context",
            TENANT,
        )

        missing_email_scoped = json.loads(
            await crm_server.get_customer_context(
                contact_id=str(missing_email_contact["id"]),
                business_context_id=TENANT,
                max_emails=1,
            )
        )
        assert missing_email_scoped["found"] is True
        assert [
            row["resend_message_id"] for row in missing_email_scoped["sent_emails"]
        ] == ["ACK-PROOF-2"]

        unscoped = await repository.query(
            to_address=payload["email"],
            limit=10,
        )
        assert {row.subject for row in unscoped} == {
            "legacy",
            "FOREIGN-SECRET",
            "NULL-SECRET",
            scoped["sent_emails"][0]["subject"],
        }

        race_contact = await pool.fetchrow(
            """
            INSERT INTO contacts (
                full_name, email, business_context_id
            )
            VALUES ('Email Race', 'race@example.com', $1)
            RETURNING id
            """,
            TENANT,
        )
        await EmailRepository().create(
            to_addresses=["race@example.com"],
            subject="race history",
            body="race",
            business_context_id=TENANT,
        )

        for new_owner, explicit, expected_found in (
            (FOREIGN_TENANT, TENANT, False),
            (None, TENANT, False),
            (None, None, True),
        ):
            await pool.execute(
                """
                UPDATE contacts
                SET business_context_id = $2
                WHERE id = $1
                """,
                race_contact["id"],
                TENANT,
            )
            email_repo_mod._email_repo = _ReassigningEmailRepository(
                adapter,
                race_contact["id"],
                new_owner,
            )
            raced = json.loads(
                await crm_server.get_customer_context(
                    contact_id=str(race_contact["id"]),
                    business_context_id=explicit,
                    max_emails=1,
                )
            )
            assert raced["found"] is expected_found
            if not expected_found:
                assert raced["context"] is None
    finally:
        try:
            import atlas_brain.mcp.crm_server as crm_server

            crm_server.set_provider_override(None)
        except Exception:
            pass
        if email_repo_mod is not None:
            email_repo_mod._email_repo = original_email_repo
        if db_mod is not None:
            db_mod._db_pool = original_db_pool
        if pool is not None:
            await pool.close()
        await admin_conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await admin_conn.close()
