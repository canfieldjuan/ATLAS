"""Real-Postgres reachability proof for the EOM lead pipeline slice."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI

asyncpg = pytest.importorskip("asyncpg")

from atlas_brain.api import leads as leads_mod  # noqa: E402
from atlas_brain.services.crm_provider import DatabaseCRMProvider  # noqa: E402


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

            follow_up = datetime(2026, 7, 25, 15, tzinfo=timezone.utc)
            await provider.update_contact(
                str(contact["id"]),
                {
                    "lead_stage": "qualified",
                    "lead_owner": "Juan",
                    "next_follow_up_at": follow_up,
                },
                require_contact_type="lead",
            )
            with pytest.raises(ValueError, match="contact_type='lead'"):
                await provider.update_contact(
                    str(contact["id"]),
                    {
                        "contact_type": "customer",
                        "lead_stage": "converted",
                    },
                )
            repeated = await client.post("/api/v1/leads/intake", json=payload)
            assert repeated.status_code == 200

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
            "lead_stage": "qualified",
            "lead_owner": "Juan",
            "next_follow_up_at": follow_up,
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
        assert [row["full_name"] for row in due] == [
            "Pipeline Proof",
            "Legacy Due",
        ]
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()
