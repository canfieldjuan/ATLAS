"""Real-Postgres proof that reasoning events survive asyncpg-decoded values.

Producers build event payloads and entity ids from asyncpg rows, where uuid
columns decode to uuid.UUID. These tests drive the real CRM producers and the
real emit path, then read back the persisted atlas_events row. emit_if_enabled
swallows emission errors into a warning, so the row is the only proof the event
was not silently dropped.
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest

asyncpg = pytest.importorskip("asyncpg")

from atlas_brain.reasoning.events import emit_event  # noqa: E402
from atlas_brain.services.crm_provider import DatabaseCRMProvider  # noqa: E402

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS = ROOT / "atlas_brain" / "storage" / "migrations"
DATABASE_URL_ENV = "ATLAS_MIGRATION_TEST_DATABASE_URL"


@asynccontextmanager
async def _event_schema(monkeypatch):
    database_url = os.environ.get(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")

    schema = f"atlas_reasoning_events_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}", public')
        await conn.execute("CREATE TABLE appointments (id UUID PRIMARY KEY)")
        for name in (
            "035_contacts.sql",
            "256_contact_interaction_dedupe.sql",
            "346_contact_lead_pipeline.sql",
            "366_contacts_customer_type.sql",
            "071_reasoning_agent_foundation.sql",
        ):
            await conn.execute((MIGRATIONS / name).read_text())

        import atlas_brain.storage.database as db_mod
        from atlas_brain.config import settings

        # emit_event writes through the global pool; bind it to this schema.
        monkeypatch.setattr(db_mod, "get_db_pool", lambda: conn)
        monkeypatch.setattr(settings.reasoning, "enabled", True)
        yield conn
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


def _payload(row) -> dict:
    value = row["payload"]
    return json.loads(value) if isinstance(value, str) else dict(value)


@pytest.mark.asyncio
async def test_log_interaction_persists_interaction_logged_event(monkeypatch):
    async with _event_schema(monkeypatch) as conn:
        contact_id = uuid.uuid4()
        await conn.execute(
            "INSERT INTO contacts (id, full_name) VALUES ($1, 'American Red Cross')",
            contact_id,
        )
        provider = DatabaseCRMProvider(pool=conn)

        interaction = await provider.log_interaction(
            contact_id=str(contact_id),
            interaction_type="invoice",
            summary="Invoice INV-2026-Sep-0518 sent via Gmail",
        )

        # The producer really hands the emitter a row-decoded UUID.
        assert isinstance(interaction["id"], uuid.UUID)
        event = await conn.fetchrow(
            "SELECT entity_type, entity_id, payload FROM atlas_events "
            "WHERE event_type = 'crm.interaction_logged'"
        )
        assert event is not None, "crm.interaction_logged event was dropped"
        assert event["entity_type"] == "contact"
        assert event["entity_id"] == str(contact_id)
        payload = _payload(event)
        assert payload["interaction_id"] == str(interaction["id"])
        assert payload["contact_id"] == str(contact_id)
        assert payload["interaction_type"] == "invoice"


@pytest.mark.asyncio
async def test_find_or_create_contact_persists_contact_created_event(monkeypatch):
    async with _event_schema(monkeypatch) as conn:
        provider = DatabaseCRMProvider(pool=conn)

        contact = await provider.find_or_create_contact(
            full_name="AKRA Builders", phone="618-553-4628"
        )

        # Both the payload contact_id and entity_id come from the row's UUID.
        assert isinstance(contact["id"], uuid.UUID)
        event = await conn.fetchrow(
            "SELECT entity_id, payload FROM atlas_events "
            "WHERE event_type = 'crm.contact_created'"
        )
        assert event is not None, "crm.contact_created event was dropped"
        assert event["entity_id"] == str(contact["id"])
        assert _payload(event)["contact_id"] == str(contact["id"])


@pytest.mark.asyncio
async def test_emit_event_normalizes_nested_row_values_and_keeps_null_entity(
    monkeypatch,
):
    async with _event_schema(monkeypatch) as conn:
        first, second, entity = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        sent_at = datetime(2026, 9, 25, 19, 27, 46, tzinfo=timezone.utc)

        event_id = await emit_event(
            "test.row_values",
            "test",
            {"ids": [first], "ref": {"id": second}, "sent_at": sent_at},
            entity_type="contact",
            entity_id=entity,
        )
        no_entity_id = await emit_event("test.no_entity", "test", {"ok": True})

        event = await conn.fetchrow(
            "SELECT entity_id, payload FROM atlas_events WHERE id = $1", event_id
        )
        assert event["entity_id"] == str(entity)
        assert _payload(event) == {
            "ids": [str(first)],
            "ref": {"id": str(second)},
            "sent_at": str(sent_at),
        }
        # A missing entity stays NULL rather than becoming the string "None".
        assert (
            await conn.fetchval(
                "SELECT entity_id FROM atlas_events WHERE id = $1", no_entity_id
            )
            is None
        )
