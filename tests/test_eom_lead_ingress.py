"""Behavioral coverage for the EOM office-controlled lead ingress boundary."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
from types import SimpleNamespace

import pytest

_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)

from atlas_brain.services.eom_lead_ingress import (  # noqa: E402
    EOM_BUSINESS_CONTEXT_ID,
    resolve_or_create_eom_contact,
    resolve_or_create_eom_inbound_lead,
)


def _crm(rows: list[dict] | None = None, created: dict | None = None):
    rows = rows or []

    async def search_contacts(**kwargs):
        if kwargs.get("business_context_id"):
            return [row for row in rows if row.get("business_context_id") == kwargs["business_context_id"]]
        if kwargs.get("business_context_id_is_null"):
            return [row for row in rows if row.get("business_context_id") is None]
        return rows

    crm = MagicMock()
    crm.search_contacts = AsyncMock(side_effect=search_contacts)
    crm.find_or_create_contact = AsyncMock(return_value=created or {
        "id": "lead-new", "contact_type": "lead", "lead_stage": "new", "_was_created": True,
    })
    crm.log_interaction = AsyncMock(return_value={"id": "interaction-1"})
    return crm


@pytest.mark.asyncio
async def test_unmatched_eom_inbound_is_created_only_as_new_lead():
    crm = _crm()

    contact = await resolve_or_create_eom_inbound_lead(
        crm,
        full_name="  New Caller  ",
        phone="(217) 555-0199",
        email=" NEW@EXAMPLE.COM ",
        address="100 Main St",
        source="phone_call",
        source_ref="call-1",
    )

    assert contact["contact_type"] == "lead"
    kwargs = crm.find_or_create_contact.await_args.kwargs
    assert kwargs == {
        "full_name": "New Caller",
        "phone": "2175550199",
        "email": "new@example.com",
        "address": "100 Main St",
        "business_context_id": EOM_BUSINESS_CONTEXT_ID,
        "contact_type": "lead",
        "lead_stage": "new",
        "source": "phone_call",
        "source_ref": "call-1",
        "preserve_existing": True,
    }


@pytest.mark.asyncio
async def test_identityless_eom_inbound_requires_a_stable_relay_event_identity():
    crm = _crm()

    with pytest.raises(ValueError, match="stable relay event identity"):
        await resolve_or_create_eom_inbound_lead(
            crm,
            full_name="Name only",
            phone=None,
            email=None,
            address=None,
            source="web",
            source_ref="website_estimate_form",
        )

    crm.find_or_create_contact.assert_not_awaited()


@pytest.mark.asyncio
async def test_identityless_eom_inbound_uses_only_the_explicit_relay_event_key():
    crm = _crm()

    await resolve_or_create_eom_inbound_lead(
        crm,
        full_name="Relay only",
        phone=None,
        email=None,
        address=None,
        source="web",
        source_ref="untrusted-caller-metadata",
        relay_event_id="web3forms:message-1",
    )

    assert crm.find_or_create_contact.await_args.kwargs["source_ref"] == "web3forms:message-1"


def test_preferred_eom_inbound_phone_uses_full_extraction_then_transport():
    from atlas_brain.services.eom_lead_ingress import preferred_eom_inbound_phone

    assert preferred_eom_inbound_phone("217-555-0100", "+12175550199") == "217-555-0100"
    assert preferred_eom_inbound_phone("555-0100", "+12175550199") == "+12175550199"
    assert preferred_eom_inbound_phone("555-0100", "555-0199") == "555-0100"


@pytest.mark.asyncio
@pytest.mark.parametrize("contact_type", ["lead", "customer"])
async def test_matching_eom_contact_is_returned_unchanged(contact_type):
    existing = {
        "id": "already-there",
        "business_context_id": EOM_BUSINESS_CONTEXT_ID,
        "contact_type": contact_type,
        "lead_stage": "qualified" if contact_type == "lead" else None,
        "full_name": "Stored Name",
    }
    crm = _crm([existing])

    contact = await resolve_or_create_eom_inbound_lead(
        crm,
        full_name="Extractor Guess",
        phone="2175550199",
        email="new@example.com",
        address="Not stored",
        source="sms",
        source_ref="sms-1",
    )

    assert contact == {**existing, "_was_created": False}
    crm.find_or_create_contact.assert_not_awaited()


@pytest.mark.asyncio
async def test_eom_provider_does_not_promote_existing_lead_from_generic_merge():
    from atlas_brain.services.crm_provider import DatabaseCRMProvider

    provider = DatabaseCRMProvider.__new__(DatabaseCRMProvider)
    provider.search_contacts = AsyncMock(return_value=[{
        "id": "lead-1",
        "business_context_id": EOM_BUSINESS_CONTEXT_ID,
        "contact_type": "lead",
        "lead_stage": "new",
    }])
    provider.update_contact = AsyncMock()

    contact = await provider.create_contact({
        "full_name": "Wrong Extracted Name",
        "phone": "2175550199",
        "business_context_id": EOM_BUSINESS_CONTEXT_ID,
        "contact_type": "customer",
    })

    assert contact["contact_type"] == "lead"
    assert contact["_was_created"] is False
    provider.update_contact.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_eom_provider_merge_retains_existing_behavior():
    from atlas_brain.services.crm_provider import DatabaseCRMProvider

    provider = DatabaseCRMProvider.__new__(DatabaseCRMProvider)
    provider.search_contacts = AsyncMock(return_value=[{
        "id": "other-1", "business_context_id": "other", "contact_type": "lead",
    }])
    provider.update_contact = AsyncMock(return_value={"id": "other-1", "full_name": "Updated"})

    contact = await provider.create_contact({
        "full_name": "Updated", "email": "updated@example.com",
        "business_context_id": "other", "contact_type": "customer",
    })

    assert contact["id"] == "other-1"
    provider.update_contact.assert_awaited_once()


@pytest.mark.asyncio
async def test_real_call_link_uses_eom_lead_resolver(monkeypatch):
    from atlas_brain.comms import call_intelligence
    import atlas_brain.services.crm_provider as crm_provider
    import atlas_brain.storage.database as database

    crm = _crm(created={"id": "call-lead", "contact_type": "lead", "lead_stage": "new", "_was_created": True})
    pool = MagicMock(is_initialized=True)
    repo = MagicMock()
    repo.link_contact = AsyncMock()
    monkeypatch.setattr(crm_provider, "get_crm_provider", lambda: crm)
    monkeypatch.setattr(database, "get_db_pool", lambda: pool)

    contact_id, is_new = await call_intelligence._link_to_crm(
        repo, uuid4(), "call-1", "217-555-0199", EOM_BUSINESS_CONTEXT_ID,
        {
            "customer_name": "Caller",
            "customer_phone": "555-0100",
            "intent": "estimate_request",
        },
        "Need an estimate",
    )

    assert (contact_id, is_new) == ("call-lead", True)
    contact_kwargs = crm.find_or_create_contact.await_args.kwargs
    assert contact_kwargs["contact_type"] == "lead"
    assert contact_kwargs["phone"] == "2175550199"
    interaction_kwargs = crm.log_interaction.await_args.kwargs
    assert interaction_kwargs["intent"] == "estimate_request"
    assert interaction_kwargs["metadata"] == {"crm_event_id": "call:call-1"}
    await call_intelligence._link_to_crm(
        repo,
        uuid4(),
        "call-2",
        "217-555-0199",
        "another_context",
        {"customer_phone": "555-0100"},
        "Call summary",
    )
    assert crm.find_or_create_contact.await_args.kwargs["phone"] == "555-0100"


@pytest.mark.asyncio
async def test_real_sms_link_uses_eom_lead_resolver(monkeypatch):
    from atlas_brain.comms import sms_intelligence
    import atlas_brain.services.crm_provider as crm_provider
    import atlas_brain.storage.database as database

    crm = _crm(created={"id": "sms-lead", "contact_type": "lead", "lead_stage": "new", "_was_created": True})
    pool = MagicMock(is_initialized=True)
    repo = MagicMock()
    repo.link_contact = AsyncMock()
    monkeypatch.setattr(crm_provider, "get_crm_provider", lambda: crm)
    monkeypatch.setattr(database, "get_db_pool", lambda: pool)

    sms_id = uuid4()
    contact_id, is_new = await sms_intelligence._link_to_crm(
        repo, sms_id, "217-555-0199", EOM_BUSINESS_CONTEXT_ID,
        {
            "customer_name": "Texter",
            "customer_phone": "555-0100",
            "intent": "booking",
        },
        "Can I get an estimate?",
        provider_message_id="SM-provider-1",
    )

    assert (contact_id, is_new) == ("sms-lead", True)
    contact_kwargs = crm.find_or_create_contact.await_args.kwargs
    assert contact_kwargs["contact_type"] == "lead"
    assert contact_kwargs["phone"] == "2175550199"
    assert contact_kwargs["source_ref"] == "SM-provider-1"
    interaction_kwargs = crm.log_interaction.await_args.kwargs
    assert interaction_kwargs["intent"] == "estimate_request"
    assert interaction_kwargs["metadata"] == {"crm_event_id": "sms:SM-provider-1"}
    fallback_sms_id = uuid4()
    await sms_intelligence._link_to_crm(
        repo,
        fallback_sms_id,
        "217-555-0199",
        "another_context",
        {"customer_phone": "555-0100"},
        "SMS summary",
        provider_message_id="SM-non-eom-provider-1",
    )
    assert crm.find_or_create_contact.await_args.kwargs["phone"] == "555-0100"
    assert crm.find_or_create_contact.await_args.kwargs["source_ref"] == str(fallback_sms_id)
    assert crm.log_interaction.await_args.kwargs["metadata"] == {
        "crm_event_id": "sms:SM-non-eom-provider-1"
    }


@pytest.mark.asyncio
async def test_sms_link_uses_provider_identity_without_a_local_sms_row(monkeypatch):
    from atlas_brain.comms import sms_intelligence
    import atlas_brain.services.crm_provider as crm_provider
    import atlas_brain.storage.database as database

    crm = _crm(created={
        "id": "sms-provider-lead",
        "contact_type": "lead",
        "lead_stage": "new",
        "_was_created": True,
    })
    repo = MagicMock()
    repo.link_contact = AsyncMock()
    monkeypatch.setattr(crm_provider, "get_crm_provider", lambda: crm)
    db_pool = database.get_db_pool()
    original_pool = db_pool._pool
    original_initialized = db_pool._initialized
    db_pool._pool = MagicMock()
    db_pool._initialized = True

    try:
        contact_id, is_new = await sms_intelligence._link_to_crm(
            repo,
            None,
            "217-555-0199",
            EOM_BUSINESS_CONTEXT_ID,
            {"customer_name": "Retry Texter", "intent": "booking"},
            "Retry-safe estimate request",
            provider_message_id="SM-persistence-failed-1",
        )
    finally:
        db_pool._pool = original_pool
        db_pool._initialized = original_initialized

    assert (contact_id, is_new) == ("sms-provider-lead", True)
    assert crm.find_or_create_contact.await_args.kwargs["source_ref"] == (
        "SM-persistence-failed-1"
    )
    assert crm.log_interaction.await_args.kwargs["metadata"] == {
        "crm_event_id": "sms:SM-persistence-failed-1"
    }
    repo.link_contact.assert_not_awaited()


@pytest.mark.asyncio
async def test_sms_fallback_uses_eom_lead_resolver(monkeypatch):
    from atlas_brain.api.comms import webhooks
    import atlas_brain.services.crm_provider as crm_provider
    from atlas_brain.config import settings

    crm = _crm(created={"id": "fallback-lead", "contact_type": "lead", "lead_stage": "new", "_was_created": True})
    sms_repo = MagicMock()
    sms_repo.link_contact = AsyncMock()
    monkeypatch.setattr(crm_provider, "get_crm_provider", lambda: crm)
    monkeypatch.setattr(settings.alerts, "ntfy_enabled", False)

    await webhooks._sms_fallback_crm_and_notify(
        uuid4(), sms_repo, "217-555-0199", "Need an estimate",
        SimpleNamespace(id=EOM_BUSINESS_CONTEXT_ID, name="Effingham Office Maids"),
        provider_message_id="SM-fallback-1",
    )

    assert crm.find_or_create_contact.await_args.kwargs["contact_type"] == "lead"
    assert crm.find_or_create_contact.await_args.kwargs["source_ref"] == "SM-fallback-1"
    assert crm.log_interaction.await_args.kwargs["interaction_type"] == "sms"
    assert crm.log_interaction.await_args.kwargs["metadata"] == {
        "crm_event_id": "sms:SM-fallback-1"
    }

    local_sms_id = uuid4()
    await webhooks._sms_fallback_crm_and_notify(
        local_sms_id,
        sms_repo,
        "217-555-0199",
        "Other tenant message",
        SimpleNamespace(id="another_context", name="Another Context"),
        provider_message_id="SM-fallback-non-eom-1",
    )
    assert crm.find_or_create_contact.await_args.kwargs["source_ref"] == str(local_sms_id)


@pytest.mark.asyncio
async def test_web3forms_relay_uses_eom_lead_resolver(monkeypatch):
    from atlas_brain.autonomous.tasks import gmail_digest
    import atlas_brain.services.crm_provider as crm_provider

    crm = _crm(created={
        "id": "relay-lead", "contact_type": "lead", "lead_stage": "new",
        "_was_created": True,
    })
    monkeypatch.setattr(crm_provider, "get_crm_provider", lambda: crm)
    emails = [{
        "id": "web3forms-message-1",
        "category": "lead",
        "reply_to": "Relay Lead <relay@example.com>",
        "body_text": "Name: Relay Lead\nPhone: (217) 555-0199\nMessage: Estimate",
        "subject": "Estimate request",
    }]

    await gmail_digest._process_lead_emails(emails)

    kwargs = crm.find_or_create_contact.await_args.kwargs
    assert kwargs["contact_type"] == "lead"
    assert kwargs["lead_stage"] == "new"
    assert kwargs["preserve_existing"] is True
    assert kwargs["source_ref"] == "web3forms:web3forms-message-1"
    assert kwargs["tags"] == ["web3forms"]
    assert crm.log_interaction.await_args.kwargs["metadata"] == {
        "gmail_message_id": "web3forms-message-1"
    }


@pytest.mark.asyncio
async def test_web3forms_name_only_without_message_id_does_not_create_unanchored_lead(
    monkeypatch,
):
    from atlas_brain.autonomous.tasks import gmail_digest
    import atlas_brain.services.crm_provider as crm_provider

    crm = _crm()
    monkeypatch.setattr(crm_provider, "get_crm_provider", lambda: crm)

    await gmail_digest._process_lead_emails(
        [
            {
                "id": "",
                "category": "lead",
                "reply_to": "",
                "body_text": "Name: Unanchored relay",
                "subject": "Estimate request",
            }
        ]
    )

    crm.find_or_create_contact.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_scheduling_preserves_partial_phone_leads(monkeypatch):
    from atlas_brain.tools import scheduling
    import atlas_brain.services.crm_provider as crm_provider
    import atlas_brain.storage.database as database

    appointment_id = uuid4()
    contact_id = uuid4()
    crm = _crm(created={
        "id": str(contact_id), "contact_type": "lead", "lead_stage": "new",
        "_was_created": True,
    })
    context = SimpleNamespace(
        id=EOM_BUSINESS_CONTEXT_ID,
        name="Effingham Office Maids",
        hours=SimpleNamespace(timezone="America/Chicago"),
        scheduling=SimpleNamespace(calendar_id="estimate-calendar", default_duration_minutes=60),
    )
    service = MagicMock()
    service.book_appointment = AsyncMock(return_value=SimpleNamespace(id="calendar-event"))
    repo = MagicMock()
    repo.create = AsyncMock(return_value={"id": str(appointment_id)})
    pool = MagicMock()
    pool.execute = AsyncMock()

    monkeypatch.setattr(scheduling, "_get_default_context", lambda: context)
    monkeypatch.setattr(
        scheduling,
        "_parse_datetime",
        lambda *_args: datetime(2026, 8, 1, 15, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(scheduling, "_get_time_slot_class", lambda: lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(scheduling, "_get_scheduling_service", lambda: service)
    monkeypatch.setattr(scheduling, "get_appointment_repo", lambda: repo)
    monkeypatch.setattr(crm_provider, "get_crm_provider", lambda: crm)
    monkeypatch.setattr(database, "get_db_pool", lambda: pool)

    result = await scheduling.BookAppointmentTool().execute({
        "customer_name": "Legacy Estimate",
        "customer_phone": "555-0199",
        "date": "August 1",
        "time": "10:00 AM",
    })

    assert result.success is True
    kwargs = crm.find_or_create_contact.await_args.kwargs
    assert kwargs["contact_type"] == "lead"
    assert kwargs["lead_stage"] == "new"
    assert kwargs["source"] == "booking"
    assert kwargs["phone"] is None
    assert kwargs["source_ref"] == f"appointment:{appointment_id}"
    assert kwargs["preserve_existing"] is True
    pool.execute.assert_awaited_once()

    crm.find_or_create_contact.side_effect = ValueError("CRM unavailable")
    link_failure = await scheduling.BookAppointmentTool().execute({
        "customer_name": "Legacy Estimate",
        "customer_phone": "555-0199",
        "date": "August 1",
        "time": "10:00 AM",
    })
    assert link_failure.success is True


@pytest.mark.asyncio
async def test_generic_mcp_stage_writer_rejects_eom_lead(monkeypatch):
    from atlas_brain.mcp import crm_server

    existing = {
        "id": str(uuid4()),
        "business_context_id": EOM_BUSINESS_CONTEXT_ID,
        "contact_type": "lead",
        "lead_stage": "new",
    }
    monkeypatch.setattr(crm_server, "_guarded_contact", AsyncMock(return_value=(True, existing)))

    response = json.loads(await crm_server.update_contact(
        existing["id"], lead_stage="estimate_booked", business_context_id=EOM_BUSINESS_CONTEXT_ID,
    ))

    assert response == {
        "success": False,
        "error": "EOM lead stages can only change through the funnel transition service",
    }


def test_lifecycle_migration_records_create_once_in_contact_insert_transaction():
    migration = (
        Path(__file__).parents[1]
        / "atlas_brain/storage/migrations/351_eom_lead_lifecycle_events.sql"
    ).read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS eom_lead_lifecycle_events" in migration
    assert "REFERENCES contacts(id) ON DELETE RESTRICT" in migration
    assert "BEFORE UPDATE OR DELETE ON eom_lead_lifecycle_events" in migration
    assert "CREATE TRIGGER trg_prevent_eom_lead_lifecycle_event_mutation" in migration
    assert "BEFORE TRUNCATE ON eom_lead_lifecycle_events" in migration
    assert "CREATE TRIGGER trg_prevent_eom_lead_lifecycle_event_truncate" in migration
    assert "CREATE OR REPLACE FUNCTION record_eom_lead_created()" in migration
    assert "AFTER INSERT ON contacts" in migration
    assert "CREATE TRIGGER trg_record_eom_lead_created" in migration
    assert "ON CONFLICT (contact_id, event_type, operation_key)" in migration


# --- D2: email_backfill through the EOM boundary (website #125) -------------
#
# email_backfill previously called crm.find_or_create_contact directly with a
# hardcoded tenant literal and the default preserve_existing=False, so a name
# derived from an email local part could merge over a curated one.


@pytest.mark.asyncio
async def test_backfill_returns_an_existing_eom_contact_untouched():
    """The actual bug: derived data must never overwrite a curated record.

    email_backfill infers a display name from the email local part, so a merge
    here would replace a real customer's name with a guess.
    """
    existing = {
        "id": str(uuid4()),
        "full_name": "Margaret Ellsworth-Hayes",
        "email": "m.hayes@example.test",
        "business_context_id": EOM_BUSINESS_CONTEXT_ID,
        "contact_type": "customer",
    }
    crm = _crm(rows=[existing])

    result = await resolve_or_create_eom_contact(
        crm,
        full_name="M Hayes",  # what the local-part heuristic would produce
        email="m.hayes@example.test",
        source="email_backfill",
        contact_type="customer",
    )

    assert result["full_name"] == "Margaret Ellsworth-Hayes"
    assert result["_was_created"] is False
    crm.find_or_create_contact.assert_not_awaited()


@pytest.mark.asyncio
async def test_backfill_creates_with_tenant_from_the_boundary_and_customer_type():
    """Tenant comes from the boundary constant, not a caller literal.

    contact_type is honoured rather than forced to lead/new: these are people
    who already bought, and mislabelling them would corrupt the funnel.
    """
    crm = _crm(created={"id": str(uuid4()), "_was_created": True})

    await resolve_or_create_eom_contact(
        crm,
        full_name="New Person",
        email="New.Person@Example.TEST",
        source="email_backfill",
        contact_type="customer",
    )

    kwargs = crm.find_or_create_contact.await_args.kwargs
    assert kwargs["business_context_id"] == EOM_BUSINESS_CONTEXT_ID
    assert kwargs["contact_type"] == "customer"
    assert kwargs["source"] == "email_backfill"
    assert kwargs["preserve_existing"] is True
    assert kwargs["email"] == "new.person@example.test"


@pytest.mark.asyncio
async def test_backfill_claims_a_legacy_untenanted_contact_rather_than_duplicating():
    """Legacy rows have business_context_id NULL and must be claimed, not forked.

    Resolution order is the reason the resolver was lifted to module level and
    shared with ingress instead of reimplemented here.
    """
    legacy = {
        "id": str(uuid4()),
        "full_name": "Legacy Customer",
        "email": "legacy@example.test",
        "business_context_id": None,
    }
    crm = _crm(rows=[legacy])

    result = await resolve_or_create_eom_contact(
        crm,
        full_name="Legacy",
        email="legacy@example.test",
        source="email_backfill",
        contact_type="customer",
    )

    assert result["id"] == legacy["id"]
    assert result["_was_created"] is False
    crm.find_or_create_contact.assert_not_awaited()


@pytest.mark.asyncio
async def test_backfill_prefers_the_tenanted_contact_over_a_legacy_one():
    """Scoped-before-legacy, the same precedence ingress relies on."""
    scoped = {
        "id": str(uuid4()),
        "full_name": "Tenanted",
        "email": "shared@example.test",
        "business_context_id": EOM_BUSINESS_CONTEXT_ID,
    }
    legacy = {
        "id": str(uuid4()),
        "full_name": "Untenanted",
        "email": "shared@example.test",
        "business_context_id": None,
    }
    crm = _crm(rows=[legacy, scoped])

    result = await resolve_or_create_eom_contact(
        crm,
        full_name="Whoever",
        email="shared@example.test",
        source="email_backfill",
        contact_type="customer",
    )

    assert result["id"] == scoped["id"]
