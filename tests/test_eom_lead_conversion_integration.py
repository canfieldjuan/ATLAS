"""Real-Postgres proof for the EOM customer handoff transaction."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

asyncpg = pytest.importorskip("asyncpg")

from atlas_brain.services.crm_provider import DatabaseCRMProvider  # noqa: E402
from atlas_brain.services.eom_lead_conversion import EOMLeadConversionError  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS = ROOT / "atlas_brain" / "storage" / "migrations"
DATABASE_URL_ENV = "ATLAS_MIGRATION_TEST_DATABASE_URL"


@pytest.mark.asyncio
async def test_office_handoff_is_atomic_idempotent_and_keeps_rate_schedule_out_of_atlas(
    monkeypatch,
):
    database_url = os.environ.get(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")

    schema = f"atlas_eom_conversion_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}", public')
        await conn.execute("CREATE TABLE appointments (id UUID PRIMARY KEY)")
        for name in (
            "035_contacts.sql",
            "346_contact_lead_pipeline.sql",
            "351_eom_lead_lifecycle_events.sql",
            "353_eom_customer_handoffs.sql",
        ):
            await conn.execute((MIGRATIONS / name).read_text())

        import atlas_brain.storage.database as db_mod

        monkeypatch.setattr(db_mod, "get_db_pool", lambda: conn)
        provider = DatabaseCRMProvider()
        contact_id = uuid.uuid4()
        approval_key = f"office-handoff-{uuid.uuid4().hex}"
        different_approval_key = f"office-handoff-{uuid.uuid4().hex}"
        await conn.execute(
            """
            INSERT INTO contacts (
                id, full_name, business_context_id, contact_type, lead_stage, status
            ) VALUES ($1, 'Approved Estimate', 'effingham_maids', 'lead', 'new', 'active')
            """,
            contact_id,
        )

        first = await provider.finalize_eom_customer_handoff(
            contact_id=str(contact_id),
            tracker_customer_id=101,
            tracker_site_id=202,
            approval_key=approval_key,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        retry = await provider.finalize_eom_customer_handoff(
            contact_id=str(contact_id),
            tracker_customer_id=101,
            tracker_site_id=202,
            approval_key=approval_key,
            actor_id=1,
            actor_name="Juan Canfield",
        )

        assert first["idempotent"] is False
        assert retry == {**first, "idempotent": True}
        contact = await conn.fetchrow(
            "SELECT contact_type, lead_stage FROM contacts WHERE id = $1", contact_id
        )
        assert dict(contact) == {"contact_type": "customer", "lead_stage": None}
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM eom_customer_handoffs WHERE contact_id = $1", contact_id
        ) == 1
        assert await conn.fetchval(
            """
            SELECT COUNT(*) FROM eom_lead_lifecycle_events
            WHERE contact_id = $1 AND event_type = 'customer_approved'
            """,
            contact_id,
        ) == 1

        with pytest.raises(EOMLeadConversionError, match="different customer handoff"):
            await provider.finalize_eom_customer_handoff(
                contact_id=str(contact_id),
                tracker_customer_id=101,
                tracker_site_id=202,
                approval_key=different_approval_key,
                actor_id=1,
                actor_name="Juan Canfield",
            )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()
