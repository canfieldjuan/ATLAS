"""Real-entrypoint PostgreSQL proof for CRM contact API tenant isolation."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, HTTPException

asyncpg = pytest.importorskip("asyncpg")

from atlas_brain.api.contacts import router as contacts_router  # noqa: E402
from atlas_brain.auth.dependencies import AuthUser, require_auth  # noqa: E402
from atlas_brain.services.crm_provider import DatabaseCRMProvider  # noqa: E402
from atlas_brain.storage.repositories.call_transcript import (  # noqa: E402
    CallTranscriptRepository,
)


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

    async def execute(self, query, *args):
        return await self._pool.execute(query, *args)


class _InboxProvider:
    def __init__(self):
        self.calls = 0

    async def list_messages(self, **_kwargs):
        self.calls += 1
        return [
            {
                "id": "inbox-secret",
                "subject": "INBOX-SECRET",
                "date": datetime.now(timezone.utc),
            }
        ]


class _ReassigningCRMProvider(DatabaseCRMProvider):
    def __init__(self, pool, contact_id, new_owner):
        self._pool = pool
        self._contact_id = str(contact_id)
        self._new_owner = new_owner
        self._reassigned = False

    async def get_contact(
        self,
        contact_id: str,
        business_context_id: str | None = None,
    ):
        row = await super().get_contact(
            contact_id,
            business_context_id=business_context_id,
        )
        if (
            row
            and str(contact_id) == self._contact_id
            and not self._reassigned
        ):
            self._reassigned = True
            await self._pool.execute(
                """
                UPDATE contacts
                SET business_context_id = $2
                WHERE id = $1
                """,
                contact_id,
                self._new_owner,
            )
        return row


def _user(*, platform_admin: bool) -> AuthUser:
    return AuthUser(
        user_id=str(uuid.uuid4()),
        account_id=str(uuid.uuid4()),
        plan="pro",
        plan_status="active",
        role="owner",
        is_admin=platform_admin,
        is_platform_admin=platform_admin,
    )


@pytest.mark.asyncio
async def test_contact_routes_require_admin_and_enforce_exact_tenant_scope():
    database_url = os.environ.get(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")

    schema = f"atlas_eom_contacts_api_{uuid.uuid4().hex}"
    admin_conn = await asyncpg.connect(database_url)
    pool = None
    db_mod = None
    original_db_pool = None
    crm_provider_mod = None
    original_crm_provider = None
    email_provider_mod = None
    original_email_provider = None
    try:
        await admin_conn.execute(f'CREATE SCHEMA "{schema}"')
        await admin_conn.execute(f'SET search_path TO "{schema}", public')
        for name in (
            "001_initial_schema.sql",
            "012_appointments.sql",
            "016_sent_emails.sql",
            "030_call_transcripts.sql",
            "035_contacts.sql",
            "036_call_contact_link.sql",
            "348_appointment_operating_fields.sql",
        ):
            await admin_conn.execute((MIGRATIONS / name).read_text())
        await admin_conn.execute(
            """
            ALTER TABLE call_transcripts
            ALTER COLUMN business_context_id DROP NOT NULL
            """
        )

        async def set_search_path(connection):
            await connection.execute(f'SET search_path TO "{schema}", public')

        pool = await asyncpg.create_pool(
            database_url,
            min_size=1,
            max_size=3,
            setup=set_search_path,
        )
        adapter = _PoolAdapter(pool)

        import atlas_brain.storage.database as db_mod
        import atlas_brain.services.crm_provider as crm_provider_mod
        import atlas_brain.services.email_provider as email_provider_mod

        original_db_pool = db_mod._db_pool
        db_mod._db_pool = adapter
        original_crm_provider = crm_provider_mod._crm_provider
        crm_provider_mod._crm_provider = DatabaseCRMProvider()
        original_email_provider = email_provider_mod._email_provider
        inbox_provider = _InboxProvider()
        email_provider_mod._email_provider = inbox_provider

        eom_contact = await pool.fetchrow(
            """
            INSERT INTO contacts (full_name, email, business_context_id)
            VALUES ('Effingham Customer', 'customer@example.com', $1)
            RETURNING id
            """,
            TENANT,
        )
        archived_contact = await pool.fetchrow(
            """
            INSERT INTO contacts (full_name, business_context_id, status)
            VALUES ('Archived Customer', $1, 'archived')
            RETURNING id
            """,
            TENANT,
        )
        foreign_contact = await pool.fetchrow(
            """
            INSERT INTO contacts (full_name, business_context_id)
            VALUES ('Churn Signals Customer', $1)
            RETURNING id
            """,
            FOREIGN_TENANT,
        )
        race_contact = await pool.fetchrow(
            """
            INSERT INTO contacts (full_name, business_context_id)
            VALUES ('Ownership Race', $1)
            RETURNING id
            """,
            TENANT,
        )
        now = datetime.now(timezone.utc)
        await pool.executemany(
            """
            INSERT INTO contact_interactions (
                contact_id, interaction_type, summary, occurred_at
            )
            VALUES ($1, 'note', $2, $3)
            """,
            [
                (
                    eom_contact["id"],
                    "EOM-only note",
                    now - timedelta(hours=3),
                ),
                (
                    archived_contact["id"],
                    "Archived note",
                    now - timedelta(days=1, hours=3),
                ),
            ],
        )
        await pool.executemany(
            """
            INSERT INTO appointments (
                start_time, end_time, duration_minutes, service_type,
                customer_name, customer_phone, business_context_id, contact_id
            )
            VALUES ($1, $2, 60, 'Cleaning', $3, '217-555-0100', $4, $5)
            """,
            [
                (
                    now - timedelta(hours=2),
                    now - timedelta(hours=1),
                    "Effingham Customer",
                    TENANT,
                    eom_contact["id"],
                ),
                (
                    now - timedelta(days=1, hours=2),
                    now - timedelta(days=1, hours=1),
                    "Archived Customer",
                    TENANT,
                    archived_contact["id"],
                ),
            ],
        )
        await pool.executemany(
            """
            INSERT INTO call_transcripts (
                call_sid, from_number, to_number, business_context_id,
                transcript, summary, status, contact_id, created_at
            )
            VALUES ($1, '217-555-0100', '217-555-0101', $2,
                    $3, $4, 'ready', $5, $6)
            """,
            [
                (
                    "CA-EOM",
                    TENANT,
                    "renewal discussion for EOM",
                    "EOM renewal",
                    eom_contact["id"],
                    now - timedelta(hours=1),
                ),
                (
                    "CA-FOREIGN",
                    FOREIGN_TENANT,
                    "renewal discussion for Churn Signals",
                    "Foreign renewal",
                    foreign_contact["id"],
                    now,
                ),
                (
                    "CA-LEGACY",
                    None,
                    "legacy renewal discussion",
                    "LEGACY-SECRET",
                    eom_contact["id"],
                    now - timedelta(minutes=30),
                ),
                (
                    "CA-ARCHIVED",
                    TENANT,
                    "archived customer history",
                    "Archived call",
                    archived_contact["id"],
                    now - timedelta(days=1),
                ),
                (
                    "CA-RACE",
                    TENANT,
                    "ownership race secret",
                    "RACE-SECRET",
                    race_contact["id"],
                    now,
                ),
            ],
        )
        await pool.execute(
            """
            INSERT INTO sent_emails (to_addresses, subject, body, sent_at)
            VALUES (ARRAY['customer@example.com'], 'SENT-SECRET',
                    'tenantless email body', $1)
            """,
            now,
        )

        app = FastAPI()
        app.include_router(contacts_router)
        app.dependency_overrides[require_auth] = lambda: _user(
            platform_admin=True
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            timeline = await client.get(
                f"/contacts/{eom_contact['id']}/timeline",
                params={"business_context_id": TENANT},
            )
            assert timeline.status_code == 200
            payload = timeline.json()
            assert payload["contact_name"] == "Effingham Customer"
            assert payload["emails_omitted_under_scope"] is True
            assert {event["type"] for event in payload["events"]} == {
                "interaction",
                "appointment",
                "call",
            }
            assert [
                event["summary"]
                for event in payload["events"]
                if event["type"] == "call"
            ] == ["EOM renewal"]
            assert all(
                "Foreign" not in str(event) for event in payload["events"]
            )
            assert "SENT-SECRET" not in str(payload)
            assert "INBOX-SECRET" not in str(payload)
            assert "LEGACY-SECRET" not in str(payload)
            assert inbox_provider.calls == 0

            archived = await client.get(
                f"/contacts/{archived_contact['id']}/timeline",
                params={"business_context_id": TENANT},
            )
            assert archived.status_code == 200
            assert archived.json()["contact_name"] == "Archived Customer"
            assert {
                event["type"] for event in archived.json()["events"]
            } == {"interaction", "appointment", "call"}

            for contact_id, context in (
                (foreign_contact["id"], TENANT),
                (eom_contact["id"], FOREIGN_TENANT),
            ):
                response = await client.get(
                    f"/contacts/{contact_id}/timeline",
                    params={"business_context_id": context},
                )
                assert response.status_code == 404

            normal_provider = crm_provider_mod._crm_provider
            for new_owner in (FOREIGN_TENANT, None):
                await pool.execute(
                    """
                    UPDATE contacts
                    SET business_context_id = $2
                    WHERE id = $1
                    """,
                    race_contact["id"],
                    TENANT,
                )
                crm_provider_mod._crm_provider = _ReassigningCRMProvider(
                    adapter,
                    race_contact["id"],
                    new_owner,
                )
                race = await client.get(
                    f"/contacts/{race_contact['id']}/timeline",
                    params={"business_context_id": TENANT},
                )
                assert race.status_code == 404
                assert "RACE-SECRET" not in race.text
            crm_provider_mod._crm_provider = normal_provider

            calls = await client.get(
                "/comms/calls/search",
                params={
                    "q": "renewal",
                    "business_context_id": TENANT,
                    "limit": 1,
                },
            )
            assert calls.status_code == 200
            assert [item["call_sid"] for item in calls.json()["results"]] == [
                "CA-EOM"
            ]

            for path in (
                f"/contacts/{eom_contact['id']}/timeline",
                "/comms/calls/search",
            ):
                assert (await client.get(path)).status_code == 422
                assert (
                    await client.get(
                        path,
                        params={"business_context_id": "   "},
                    )
                ).status_code == 422
                assert (
                    await client.get(
                        path,
                        params={"business_context_id": "x" * 65},
                    )
                ).status_code == 422

            app.dependency_overrides[require_auth] = lambda: _user(
                platform_admin=False
            )
            for path in (
                f"/contacts/{eom_contact['id']}/timeline",
                "/comms/calls/search",
            ):
                response = await client.get(
                    path,
                    params={"business_context_id": TENANT},
                )
                assert response.status_code == 403

            async def unauthenticated():
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required",
                )

            app.dependency_overrides[require_auth] = unauthenticated
            for path in (
                f"/contacts/{eom_contact['id']}/timeline",
                "/comms/calls/search",
            ):
                response = await client.get(
                    path,
                    params={"business_context_id": TENANT},
                )
                assert response.status_code == 401

        linked_default = await CallTranscriptRepository().get_by_contact_id(
            str(eom_contact["id"]),
            business_context_id=TENANT,
        )
        assert {row["call_sid"] for row in linked_default} == {
            "CA-EOM",
            "CA-LEGACY",
        }
        linked_strict = await CallTranscriptRepository().get_by_contact_id(
            str(eom_contact["id"]),
            business_context_id=TENANT,
            include_unclaimed_legacy=False,
        )
        assert [row["call_sid"] for row in linked_strict] == ["CA-EOM"]

        unscoped = await CallTranscriptRepository().search(
            keyword="renewal",
            limit=10,
        )
        assert {row["call_sid"] for row in unscoped} == {
            "CA-EOM",
            "CA-FOREIGN",
            "CA-LEGACY",
        }
    finally:
        if email_provider_mod is not None:
            email_provider_mod._email_provider = original_email_provider
        if crm_provider_mod is not None:
            crm_provider_mod._crm_provider = original_crm_provider
        if db_mod is not None:
            db_mod._db_pool = original_db_pool
        if pool is not None:
            await pool.close()
        await admin_conn.execute("SET search_path TO public")
        await admin_conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await admin_conn.close()
