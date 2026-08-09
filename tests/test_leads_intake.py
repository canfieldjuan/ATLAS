"""Tests for the EOM website lead-intake endpoint (issue #2151 Phase 1).

Covers the Review Contract in plans/PR-EOM-Lead-Intake.md:
  1. tenant/source stamping on the CRM upsert
  2. untruncated interaction logging (service/frequency/sqft + full message)
  3. honeypot silent drop (no CRM, no email)
  4. same-day duplicate does not re-send the acknowledgement
  5. email failure never fails the request
  6. neither email nor phone -> validation error
  7. template guardrails: correct phone, no price/quote language; reply-to wiring
"""

from __future__ import annotations

import json
import sys
from itertools import product
from random import Random
from unittest.mock import AsyncMock, MagicMock

import pytest

_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)

from atlas_brain.api.leads import (  # noqa: E402
    EOM_BUSINESS_CONTEXT_ID,
    MAX_DAILY_SUBMISSIONS,
    LeadIntakeRequest,
    LeadRateLimitedError,
    LeadValidationError,
    _default_lead_notifier,
    _format_phone_for_display,
    _lead_push_body,
    _lead_push_title,
    _process_lead_intake,
    _publish_lead_ntfy,
    router,
)
from atlas_brain.templates.email.request_acknowledgement import (  # noqa: E402
    format_request_acknowledgement,
)


@pytest.fixture(autouse=True)
def _email_enabled(monkeypatch):
    """The endpoint honors settings.email.enabled (round 6, R11); tests that
    exercise the send path assume it is on."""
    from atlas_brain.config import settings
    monkeypatch.setattr(settings.email, "enabled", True)


@pytest.fixture(autouse=True)
def _leads_ntfy_off(monkeypatch):
    """Force the new-lead push OFF for the whole module by default, regardless of
    the checkout's .env, so running this suite from the deployed configuration
    can never publish fake lead PII to the live topic (Codex #2332 R2/R12). Tests
    that exercise the transport re-enable it explicitly, and their monkeypatch
    (applied after this autouse) wins."""
    from atlas_brain.config import settings
    monkeypatch.setattr(settings.alerts, "leads_ntfy_topic", "")


LONG_MESSAGE = (
    "We have a three-story home with a finished basement and two dogs. "
    "Mostly interested in the main floor and the upstairs bathrooms, the "
    "basement only occasionally. Hardwood throughout the main level that "
    "needs gentle treatment, and we'd prefer visits while we're at work. "
    "Please also note the bonus room over the garage is rarely used."
)
assert len(LONG_MESSAGE) > 200  # guards the truncation regression test


def _payload(**overrides):
    base = dict(
        name="Jane Doe",
        email="jane@example.com",
        phone="217-555-0100",
        service="residential",
        frequency="bi-weekly",
        square_feet="2400",
        message=LONG_MESSAGE,
        source_page="/house-cleaning-services/",
    )
    base.update(overrides)
    return LeadIntakeRequest(**base)


def _crm(inserted: bool = True, existing: list | None = None):
    """Search mock mirrors production filters: a tenant-scoped call returns
    only that tenant's rows; a null-scoped call returns only NULL-context
    rows; an unscoped call returns everything."""
    rows = existing or []

    async def _search(**kwargs):
        if kwargs.get("business_context_id"):
            return [m for m in rows
                    if m.get("business_context_id") == kwargs["business_context_id"]]
        if kwargs.get("business_context_id_is_null"):
            return [m for m in rows if m.get("business_context_id") is None]
        return rows

    crm = MagicMock()
    crm.search_contacts = AsyncMock(side_effect=_search)
    crm.find_or_create_contact = AsyncMock(return_value={"id": "c-123"})
    crm.log_interaction = AsyncMock(return_value={"id": "i-1", "inserted": inserted})
    return crm


def _email_provider(success: bool = True):
    provider = MagicMock()
    provider.send = AsyncMock(
        return_value={"success": success, "message_id": "provider-message-1"}
    )
    return provider


def _email_history():
    history = MagicMock()
    history.create = AsyncMock(return_value=MagicMock())
    return history


# ---------------------------------------------------------------------------
# 1. Tenant + source stamping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contact_stamped_with_eom_tenant_and_web_source():
    crm, provider = _crm(), _email_provider()
    result = await _process_lead_intake(_payload(), crm=crm, email_provider=provider)

    assert result["success"] is True
    assert "contact_id" not in result  # public response must not leak CRM ids
    kwargs = crm.find_or_create_contact.call_args.kwargs
    assert kwargs["business_context_id"] == EOM_BUSINESS_CONTEXT_ID == "effingham_maids"
    assert kwargs["contact_type"] == "lead"
    assert kwargs["source"] == "web"
    assert kwargs["source_ref"] == "website_estimate_form"
    assert kwargs["tags"] == ["website", "estimate_request"]
    assert kwargs["lead_stage"] == "new"
    assert kwargs["preserve_existing"] is True
    assert kwargs["email"] == "jane@example.com"
    assert kwargs["phone"] == "2175550100"  # digits-only normalization
    assert kwargs["address"] is None  # base payload submits no address


@pytest.mark.asyncio
async def test_address_forwarded_to_crm_and_recorded_on_interaction():
    crm, provider = _crm(), _email_provider()
    await _process_lead_intake(
        _payload(address="207 Santa Fe Ave, Effingham, IL 62401"),
        crm=crm,
        email_provider=provider,
    )
    assert (
        crm.find_or_create_contact.call_args.kwargs["address"]
        == "207 Santa Fe Ave, Effingham, IL 62401"
    )
    assert (
        crm.log_interaction.call_args.kwargs["metadata"]["submitted_address"]
        == "207 Santa Fe Ave, Effingham, IL 62401"
    )


@pytest.mark.asyncio
async def test_blank_address_collapses_to_none_on_create():
    crm, provider = _crm(), _email_provider()
    await _process_lead_intake(_payload(address="   "), crm=crm, email_provider=provider)
    assert crm.find_or_create_contact.call_args.kwargs["address"] is None


# ---------------------------------------------------------------------------
# 2. Untruncated interaction logging
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interaction_summary_keeps_all_fields_untruncated():
    crm, provider = _crm(), _email_provider()
    await _process_lead_intake(_payload(), crm=crm, email_provider=provider)

    kwargs = crm.log_interaction.call_args.kwargs
    assert kwargs["interaction_type"] == "web_form"
    assert kwargs["intent"] == "estimate_request"
    summary = kwargs["summary"]
    # The gmail_digest path drops these three and truncates to 200 chars;
    # this endpoint must not.
    assert "residential" in summary
    assert "bi-weekly" in summary
    assert "2400" in summary
    assert LONG_MESSAGE in summary  # full message, no truncation
    assert kwargs["metadata"]["square_feet"] == "2400"


@pytest.mark.asyncio
async def test_intake_persists_attribution_as_interaction_evidence():
    crm, provider = _crm(), _email_provider()
    await _process_lead_intake(_payload(
        utm_source="google",
        utm_medium="cpc",
        utm_campaign="spring-residential",
        utm_term="house cleaning",
        utm_content="ad-3",
        gclid="gclid-1",
        gbraid="gbraid-1",
        wbraid="wbraid-1",
        landing_path="/house-cleaning-services/?utm_source=google",
        referrer="https://www.google.com/",
    ), crm=crm, email_provider=provider)

    assert crm.log_interaction.await_args.kwargs["metadata"]["attribution"] == {
        "utm_source": "google",
        "utm_medium": "cpc",
        "utm_campaign": "spring-residential",
        "utm_term": "house cleaning",
        "utm_content": "ad-3",
        "gclid": "gclid-1",
        "gbraid": "gbraid-1",
        "wbraid": "wbraid-1",
        "landing_path": "/house-cleaning-services/?utm_source=google",
        "referrer": "https://www.google.com/",
    }


@pytest.mark.asyncio
async def test_intake_omits_empty_attribution_metadata():
    crm, provider = _crm(), _email_provider()

    await _process_lead_intake(_payload(), crm=crm, email_provider=provider)

    assert "attribution" not in crm.log_interaction.await_args.kwargs["metadata"]


# ---------------------------------------------------------------------------
# 3. Honeypot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_honeypot_drops_silently_without_crm_or_email():
    crm, provider = _crm(), _email_provider()
    result = await _process_lead_intake(
        _payload(website="http://spam.example"), crm=crm, email_provider=provider
    )

    assert result == {"success": True, "email_sent": False}
    crm.find_or_create_contact.assert_not_awaited()
    crm.log_interaction.assert_not_awaited()
    provider.send.assert_not_awaited()


# ---------------------------------------------------------------------------
# 4. Same-day duplicate does not re-send the acknowledgement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_submission_skips_acknowledgement_email():
    crm, provider = _crm(inserted=False), _email_provider()
    result = await _process_lead_intake(_payload(), crm=crm, email_provider=provider)

    assert result["success"] is True
    assert result["email_sent"] is False
    provider.send.assert_not_awaited()


# ---------------------------------------------------------------------------
# 5. Email failure never fails the request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_email_provider_exception_still_returns_success():
    crm = _crm()
    provider = MagicMock()
    provider.send = AsyncMock(side_effect=RuntimeError("smtp down"))

    result = await _process_lead_intake(_payload(), crm=crm, email_provider=provider)

    assert result["success"] is True
    assert result["email_sent"] is False
    crm.log_interaction.assert_awaited()  # CRM write happened regardless


@pytest.mark.asyncio
async def test_phone_only_lead_skips_email_but_succeeds():
    crm, provider = _crm(), _email_provider()
    result = await _process_lead_intake(
        _payload(email=""), crm=crm, email_provider=provider
    )

    assert result["success"] is True
    assert result["email_sent"] is False
    provider.send.assert_not_awaited()
    assert crm.find_or_create_contact.call_args.kwargs["email"] is None


# ---------------------------------------------------------------------------
# 6. Validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_email_and_phone_rejected():
    crm, provider = _crm(), _email_provider()
    with pytest.raises(LeadValidationError):
        await _process_lead_intake(
            _payload(email="", phone=""), crm=crm, email_provider=provider
        )
    crm.find_or_create_contact.assert_not_awaited()


@pytest.mark.asyncio
async def test_partial_phone_without_email_rejected_before_crm_or_email_side_effects():
    crm, provider = _crm(), _email_provider()

    with pytest.raises(LeadValidationError, match="at least 10 digits"):
        await _process_lead_intake(
            _payload(email="", phone="5550100"),
            crm=crm,
            email_provider=provider,
        )

    crm.find_or_create_contact.assert_not_awaited()
    crm.log_interaction.assert_not_awaited()
    provider.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_malformed_email_rejected():
    crm, provider = _crm(), _email_provider()
    with pytest.raises(LeadValidationError):
        await _process_lead_intake(
            _payload(email="not-an-email"), crm=crm, email_provider=provider
        )


# ---------------------------------------------------------------------------
# 7. Acknowledgement template guardrails + send wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acknowledgement_send_wiring_reply_to_business_email():
    crm, provider = _crm(), _email_provider()
    await _process_lead_intake(_payload(), crm=crm, email_provider=provider)

    send_kwargs = provider.send.call_args.kwargs
    assert send_kwargs["to"] == ["jane@example.com"]
    assert send_kwargs["reply_to"] == "info@effinghamofficemaids.com"
    assert "estimate request" in send_kwargs["subject"].lower()
    # Routed through Resend so it originates from the verified brand domain
    # sender (info@...) rather than the Gmail account.
    assert send_kwargs["provider"] == "resend"
    assert send_kwargs["from_email"].startswith("Effingham Office Maids")
    assert "info@effinghamofficemaids.com" in send_kwargs["from_email"]


@pytest.mark.asyncio
async def test_successful_acknowledgement_records_tenant_history():
    crm, provider, history = _crm(), _email_provider(), _email_history()

    result = await _process_lead_intake(
        _payload(),
        crm=crm,
        email_provider=provider,
        email_history=history,
    )

    assert result["email_sent"] is True
    history.create.assert_awaited_once()
    kwargs = history.create.await_args.kwargs
    assert kwargs["to_addresses"] == ["jane@example.com"]
    assert kwargs["business_context_id"] == EOM_BUSINESS_CONTEXT_ID
    assert kwargs["template_type"] == "request_acknowledgement"
    assert kwargs["resend_message_id"] == "provider-message-1"
    assert kwargs["metadata"] == {
        "source": "website_estimate_form",
        "contact_id": "c-123",
        # ATLAS #2320 slice A1: the acknowledgement variant is recorded as
        # evidence alongside the raw submitted value. ``template_type`` stays
        # "request_acknowledgement" because this slice still renders the single
        # existing template.
        "service": "residential",
        "ack_variant": "residential",
    }


@pytest.mark.asyncio
async def test_refused_acknowledgement_does_not_record_history():
    crm, provider, history = _crm(), _email_provider(False), _email_history()

    result = await _process_lead_intake(
        _payload(),
        crm=crm,
        email_provider=provider,
        email_history=history,
    )

    assert result["email_sent"] is False
    history.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_history_failure_does_not_flip_successful_delivery():
    crm, provider, history = _crm(), _email_provider(), _email_history()
    history.create.side_effect = RuntimeError("history unavailable")

    result = await _process_lead_intake(
        _payload(),
        crm=crm,
        email_provider=provider,
        email_history=history,
    )

    assert result == {"success": True, "email_sent": True}
    provider.send.assert_awaited_once()
    history.create.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "inserted"),
    [
        (_payload(website="spam"), True),
        (_payload(), False),
    ],
)
async def test_non_send_paths_do_not_record_history(payload, inserted):
    crm, provider, history = _crm(inserted=inserted), _email_provider(), _email_history()

    await _process_lead_intake(
        payload,
        crm=crm,
        email_provider=provider,
        email_history=history,
    )

    history.create.assert_not_awaited()


def test_template_guardrails_no_prices_no_quotes_correct_phone():
    subject, body = format_request_acknowledgement(
        client_name="Jane", service="residential", frequency="bi-weekly"
    )
    combined = subject + body
    assert "(217) 207-3097" in body
    assert "(217) 821-2370" not in combined  # the stale number this PR retires
    assert "$" not in combined  # no dollar figures, ever (operator rule)
    assert "quote" not in combined.lower()  # "estimate" is the approved word
    assert "same-day" not in combined.lower()
    assert "within 24 hours" in body
    assert "Your request: residential, bi-weekly." in body


def test_template_omits_request_line_when_fields_empty():
    _, body = format_request_acknowledgement(client_name="")
    assert "Your request:" not in body
    assert "Hi there," in body  # empty name falls back gracefully


# ---------------------------------------------------------------------------
# PR #2152 review reconciliation — throttle, phone digits, scoped dedupe,
# route smoke
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_daily_cap_blocks_before_any_side_effect():
    crm, provider, history = _crm(), _email_provider(), _email_history()
    counter = AsyncMock(return_value=MAX_DAILY_SUBMISSIONS)
    with pytest.raises(LeadRateLimitedError):
        await _process_lead_intake(
            _payload(),
            crm=crm,
            email_provider=provider,
            daily_count=counter,
            email_history=history,
        )
    crm.find_or_create_contact.assert_not_awaited()
    crm.log_interaction.assert_not_awaited()
    provider.send.assert_not_awaited()
    history.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_daily_cap_under_limit_proceeds():
    crm, provider = _crm(), _email_provider()
    counter = AsyncMock(return_value=MAX_DAILY_SUBMISSIONS - 1)
    result = await _process_lead_intake(
        _payload(), crm=crm, email_provider=provider, daily_count=counter
    )
    assert result["success"] is True
    counter.assert_awaited_once()


@pytest.mark.asyncio
async def test_phone_without_dialable_digits_rejected_as_only_channel():
    crm, provider = _crm(), _email_provider()
    with pytest.raises(LeadValidationError):
        await _process_lead_intake(
            _payload(email="", phone="n/a"), crm=crm, email_provider=provider
        )
    crm.find_or_create_contact.assert_not_awaited()


@pytest.mark.asyncio
async def test_garbage_phone_dropped_when_email_present():
    crm, provider = _crm(), _email_provider()
    result = await _process_lead_intake(
        _payload(phone="—"), crm=crm, email_provider=provider
    )
    assert result["success"] is True
    assert crm.find_or_create_contact.call_args.kwargs["phone"] is None


def _provider_with(matches):
    """Kwargs-aware search mock mirroring production: a scoped call returns
    only rows in that tenant; an unscoped call returns everything."""
    from atlas_brain.services.crm_provider import DatabaseCRMProvider

    async def _search(**kwargs):
        ctx = kwargs.get("business_context_id")
        if ctx:
            return [m for m in matches if m.get("business_context_id") == ctx]
        if kwargs.get("business_context_id_is_null"):
            return [m for m in matches if m.get("business_context_id") is None]
        return matches

    provider = DatabaseCRMProvider.__new__(DatabaseCRMProvider)
    provider.search_contacts = AsyncMock(side_effect=_search)
    provider.update_contact = AsyncMock(side_effect=lambda cid, u: {"id": cid, **u})
    provider.claim_contact = AsyncMock(
        side_effect=lambda cid, ctx: {"id": cid, "business_context_id": ctx})
    return provider


@pytest.mark.asyncio
async def test_create_contact_does_not_match_foreign_context_after_eom_miss():
    """Cross-tenant regression: a stamped create must never resolve to a
    contact that belongs to a DIFFERENT business context. Provider callers that
    still own live EOM backfill/import jobs keep the existing fresh-insert path
    until those entry points move behind the canonical EOM service."""
    provider = _provider_with(
        [{"id": "b2b-1", "business_context_id": "churnsignals"}]
    )
    import atlas_brain.storage.database as db_mod
    pool = MagicMock()
    pool.fetchrow = AsyncMock(return_value={"id": "new-eom"})
    orig = db_mod.get_db_pool
    db_mod.get_db_pool = lambda: pool
    try:
        result = await provider.create_contact(
            {"full_name": "Jane", "email": "jane@example.com",
             "business_context_id": "effingham_maids"}
        )
    finally:
        db_mod.get_db_pool = orig
    assert result["id"] == "new-eom"
    assert result["_was_created"] is True
    pool.fetchrow.assert_awaited_once()
    provider.update_contact.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_contact_dedupe_claims_null_context_contact():
    """Historical contacts with NULL context must still match a stamped
    create (and get claimed), or every SMS/call linker would duplicate
    existing customers until the Phase 2 backfill."""
    provider = _provider_with(
        [{"id": "legacy-1", "business_context_id": None}]
    )
    result = await provider.create_contact(
        {"full_name": "Jane", "email": "jane@example.com",
         "business_context_id": "effingham_maids"}
    )
    assert result["id"] == "legacy-1"
    # Claimed via compare-and-set (round-5 of #2157), not a blind merge
    provider.claim_contact.assert_awaited_once_with("legacy-1", "effingham_maids")


@pytest.mark.asyncio
async def test_create_contact_dedupe_same_tenant_match_reused():
    provider = _provider_with(
        [{"id": "eom-1", "business_context_id": "effingham_maids"}]
    )
    result = await provider.create_contact(
        {"full_name": "Jane", "email": "jane@example.com",
         "business_context_id": "effingham_maids"}
    )
    assert result["id"] == "eom-1"


def test_route_smoke_mounted_path_statuses():
    """Route-level smoke: the mounted POST /api/v1/leads/intake path maps
    outcomes to 200 / 422 / 429 via FastAPI dependency overrides."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import atlas_brain.api.leads as leads_mod

    counter = {"n": 0}

    async def fake_count(email, phone):
        return counter["n"]

    app = FastAPI()
    app.include_router(leads_mod.router, prefix="/api/v1")
    app.dependency_overrides[leads_mod._crm_dependency] = lambda: _crm()
    app.dependency_overrides[leads_mod._email_dependency] = lambda: _email_provider()
    app.dependency_overrides[leads_mod._email_history_dependency] = _email_history
    app.dependency_overrides[leads_mod._daily_count_dependency] = lambda: fake_count

    async def fake_volume():
        return 0

    app.dependency_overrides[leads_mod._ack_volume_dependency] = lambda: fake_volume
    client = TestClient(app)

    ok = client.post("/api/v1/leads/intake", json={"name": "Jane", "email": "jane@example.com"})
    assert ok.status_code == 200 and ok.json()["success"] is True

    bad = client.post("/api/v1/leads/intake", json={"name": "Jane"})
    assert bad.status_code == 422

    counter["n"] = MAX_DAILY_SUBMISSIONS
    throttled = client.post(
        "/api/v1/leads/intake", json={"name": "Jane", "email": "jane@example.com"}
    )
    assert throttled.status_code == 429


def test_route_rejects_database_invalid_address_before_crm_write():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import atlas_brain.api.leads as leads_mod

    crm = _crm()

    async def fake_count(email, phone):
        return 0

    async def fake_volume():
        return 0

    app = FastAPI()
    app.include_router(leads_mod.router, prefix="/api/v1")
    app.dependency_overrides[leads_mod._crm_dependency] = lambda: crm
    app.dependency_overrides[leads_mod._email_dependency] = lambda: _email_provider()
    app.dependency_overrides[leads_mod._email_history_dependency] = _email_history
    app.dependency_overrides[leads_mod._daily_count_dependency] = lambda: fake_count
    app.dependency_overrides[leads_mod._ack_volume_dependency] = lambda: fake_volume

    client = TestClient(app)
    rng = Random(20260807)
    valid_code_point_ranges = (
        (0x20, 0x7F),
        (0xA0, 0xD800),
        (0xE000, 0x10000),
        (0x10000, 0x110000),
    )

    def generated_valid_character() -> str:
        start, stop = rng.choice(valid_code_point_ranges)
        return chr(rng.randrange(start, stop))

    valid_addresses = [
        "".join(generated_valid_character() for _ in range(rng.randint(1, 16)))
        for _ in range(6)
    ]

    for address in valid_addresses:
        response = client.post(
            "/api/v1/leads/intake",
            content=json.dumps({
                "name": "Jane",
                "email": "jane@example.com",
                "address": address,
            }),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 200

    writes_before_invalid_inputs = crm.find_or_create_contact.await_count
    interactions_before_invalid_inputs = crm.log_interaction.await_count
    surrogate_ranges = ((0xD800, 0xDC00), (0xDC00, 0xE000))
    generated_surrogates = (
        chr(rng.randrange(start, stop))
        for start, stop in surrogate_ranges
        for _ in range(4)
    )
    invalid_code_points = ("\x00", *generated_surrogates)
    for valid_address, invalid_code_point in product(
        valid_addresses,
        invalid_code_points,
    ):
        for position in range(len(valid_address) + 1):
            address = (
                valid_address[:position]
                + invalid_code_point
                + valid_address[position:]
            )
            response = client.post(
                "/api/v1/leads/intake",
                content=json.dumps({
                    "name": "Jane",
                    "email": "jane@example.com",
                    "address": address,
                }),
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 422

    assert crm.find_or_create_contact.await_count == writes_before_invalid_inputs
    assert crm.log_interaction.await_count == interactions_before_invalid_inputs


# ---------------------------------------------------------------------------
# PR #2153 review reconciliation — id non-exposure, no-mutation upsert,
# normalized throttle identity, global ack volume cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_existing_eom_contact_not_mutated_or_downgraded():
    """A returning CUSTOMER requesting another estimate must not be rewritten
    into a lead or have identity fields replaced by untrusted form input."""
    existing = [{"id": "cust-9", "contact_type": "customer",
                 "business_context_id": EOM_BUSINESS_CONTEXT_ID,
                 "full_name": "Real Name"}]
    crm, provider = _crm(existing=existing), _email_provider()

    result = await _process_lead_intake(_payload(), crm=crm, email_provider=provider)

    assert result["success"] is True
    crm.find_or_create_contact.assert_not_awaited()  # no create, no merge
    crm.log_interaction.assert_awaited()  # the request is still recorded
    assert crm.log_interaction.call_args.kwargs["contact_id"] == "cust-9"
    # resolution is PHONE-FIRST (phone is the more unique channel; a
    # shared/mistyped email must not steal the match)
    first = crm.search_contacts.await_args_list[0]
    assert "phone" in first.kwargs


@pytest.mark.asyncio
async def test_repeat_intake_does_not_reset_existing_lead_pipeline():
    existing = [{
        "id": "lead-9",
        "contact_type": "lead",
        "business_context_id": EOM_BUSINESS_CONTEXT_ID,
        "lead_stage": "qualified",
        "lead_owner": "Juan",
        "next_follow_up_at": "2026-07-24T15:00:00+00:00",
    }]
    crm, provider = _crm(existing=existing), _email_provider()

    await _process_lead_intake(_payload(), crm=crm, email_provider=provider)

    crm.find_or_create_contact.assert_not_awaited()
    crm.update_contact.assert_not_called()
    assert crm.log_interaction.call_args.kwargs["contact_id"] == "lead-9"


@pytest.mark.asyncio
async def test_throttle_receives_digit_normalized_phone():
    crm, provider = _crm(), _email_provider()
    counter = AsyncMock(return_value=0)
    await _process_lead_intake(
        _payload(email="", phone="(217) 555-0100"),
        crm=crm, email_provider=provider, daily_count=counter,
    )
    assert counter.await_args.args == ("", "2175550100")


@pytest.mark.asyncio
async def test_global_ack_volume_cap_skips_email_but_captures_lead():
    crm, provider, history = _crm(), _email_provider(), _email_history()
    volume = AsyncMock(return_value=1000)
    result = await _process_lead_intake(
        _payload(),
        crm=crm,
        email_provider=provider,
        ack_volume=volume,
        email_history=history,
    )
    assert result["success"] is True
    assert result["email_sent"] is False
    provider.send.assert_not_awaited()
    history.create.assert_not_awaited()
    crm.log_interaction.assert_awaited()  # lead capture unaffected


@pytest.mark.asyncio
async def test_disabled_acknowledgement_does_not_record_history(monkeypatch):
    from atlas_brain.config import settings

    monkeypatch.setattr(settings.email, "enabled", False)
    crm, provider, history = _crm(), _email_provider(), _email_history()

    result = await _process_lead_intake(
        _payload(),
        crm=crm,
        email_provider=provider,
        email_history=history,
    )

    assert result["email_sent"] is False
    provider.send.assert_not_awaited()
    history.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_global_ack_volume_under_cap_sends():
    crm, provider = _crm(), _email_provider()
    volume = AsyncMock(return_value=0)
    result = await _process_lead_intake(
        _payload(), crm=crm, email_provider=provider, ack_volume=volume
    )
    assert result["email_sent"] is True
    provider.send.assert_awaited()


def test_payload_caps_fit_contacts_schema():
    """migrations/035: email VARCHAR(256), phone VARCHAR(32) — API caps must
    not admit values the CRM write would then reject with a 503."""
    fields = LeadIntakeRequest.model_fields
    assert fields["email"].metadata[0].max_length <= 256
    assert fields["phone"].metadata[0].max_length <= 32


@pytest.mark.asyncio
async def test_ack_volume_failure_skips_email_never_fails_request():
    """PR #2153 round 3 (R6): a broken volume guard must not 503 a lead
    that was already captured — email is skipped, request succeeds."""
    crm, provider = _crm(), _email_provider()
    volume = AsyncMock(side_effect=RuntimeError("db pool down"))
    result = await _process_lead_intake(
        _payload(), crm=crm, email_provider=provider, ack_volume=volume
    )
    assert result == {"success": True, "email_sent": False}
    provider.send.assert_not_awaited()
    crm.log_interaction.assert_awaited()


@pytest.mark.asyncio
async def test_legacy_null_context_contact_resolved_readonly():
    """PR #2153 round 4: a legacy customer whose business_context_id is
    still NULL must be resolved read-only, not dropped into the mutating
    create path."""
    legacy = [{"id": "legacy-7", "contact_type": "customer",
               "business_context_id": None}]
    crm, provider = _crm(existing=legacy), _email_provider()
    result = await _process_lead_intake(_payload(), crm=crm, email_provider=provider)
    assert result["success"] is True
    crm.find_or_create_contact.assert_not_awaited()
    assert crm.log_interaction.call_args.kwargs["contact_id"] == "legacy-7"


@pytest.mark.asyncio
async def test_foreign_tenant_match_ignored_new_contact_created():
    """A same-channel contact belonging to another tenant is invisible to
    the read-only resolution; a fresh EOM contact is created instead."""
    foreign = [{"id": "b2b-3", "business_context_id": "churnsignals"}]
    crm, provider = _crm(existing=foreign), _email_provider()
    await _process_lead_intake(_payload(), crm=crm, email_provider=provider)
    crm.find_or_create_contact.assert_awaited()


@pytest.mark.asyncio
async def test_submitted_channels_recorded_in_interaction_metadata():
    """New callback email/phone from a returning contact must not be lost."""
    existing = [{"id": "cust-9", "business_context_id": EOM_BUSINESS_CONTEXT_ID}]
    crm, provider = _crm(existing=existing), _email_provider()
    await _process_lead_intake(_payload(), crm=crm, email_provider=provider)
    md = crm.log_interaction.call_args.kwargs["metadata"]
    assert md["submitted_email"] == "jane@example.com"
    assert md["submitted_phone"] == "2175550100"


@pytest.mark.asyncio
async def test_provider_prefers_same_tenant_over_null_context():
    """When both a same-tenant and a (newer) NULL-context row match, the
    tenant's own record wins; the claimable legacy row must not shadow it."""
    provider = _provider_with([
        {"id": "legacy-null", "business_context_id": None},
        {"id": "eom-real", "business_context_id": "effingham_maids"},
    ])
    result = await provider.create_contact(
        {"full_name": "Jane", "email": "jane@example.com",
         "business_context_id": "effingham_maids"}
    )
    assert result["id"] == "eom-real"


@pytest.mark.asyncio
async def test_email_disabled_setting_skips_send(monkeypatch):
    """R11: with settings.email.enabled=False no acknowledgement goes out even
    though Gmail OAuth may be present; the lead is still captured."""
    from atlas_brain.config import settings
    monkeypatch.setattr(settings.email, "enabled", False)
    crm, provider = _crm(), _email_provider()
    result = await _process_lead_intake(_payload(), crm=crm, email_provider=provider)
    assert result == {"success": True, "email_sent": False}
    provider.send.assert_not_awaited()
    crm.log_interaction.assert_awaited()


@pytest.mark.asyncio
async def test_partial_phone_not_used_for_contact_matching():
    """Round 6 R4/R6: 7-9 digit numbers are a valid channel but must not
    resolve to a contact via substring match; email resolution still runs."""
    existing = [{"id": "cust-9", "business_context_id": EOM_BUSINESS_CONTEXT_ID}]
    crm, provider = _crm(existing=existing), _email_provider()
    await _process_lead_intake(
        _payload(phone="5550100"), crm=crm, email_provider=provider
    )
    # no phone-based lookup for the short number; email lookups only
    for call in crm.search_contacts.await_args_list:
        assert "phone" not in call.kwargs
        assert "email" in call.kwargs


@pytest.mark.asyncio
async def test_short_phone_not_passed_to_mutating_create():
    """Round 8 R3/R4: a 7-9 digit number must not enter find_or_create's own
    substring dedupe (it could merge into an unrelated contact)."""
    crm, provider = _crm(), _email_provider()
    await _process_lead_intake(
        _payload(email="jane@example.com", phone="5550100"),
        crm=crm, email_provider=provider,
    )
    assert crm.find_or_create_contact.call_args.kwargs["phone"] is None


@pytest.mark.asyncio
async def test_readonly_resolution_queries_exact_populations():
    """Round 8 R3/R4: every resolution search names its population — tenant
    page or IS-NULL page — never an unscoped global page."""
    crm, provider = _crm(), _email_provider()
    await _process_lead_intake(_payload(), crm=crm, email_provider=provider)
    for call in crm.search_contacts.await_args_list:
        assert (call.kwargs.get("business_context_id") == EOM_BUSINESS_CONTEXT_ID
                or call.kwargs.get("business_context_id_is_null") is True)


@pytest.mark.asyncio
async def test_corrected_callback_defeats_same_day_dedupe_key():
    """Round 6 R1/R6: submitted channels ride the summary, so a corrected
    callback changes the dedupe basis instead of being swallowed."""
    crm, provider = _crm(), _email_provider()
    await _process_lead_intake(_payload(), crm=crm, email_provider=provider)
    s1 = crm.log_interaction.call_args.kwargs["summary"]
    crm2, _ = _crm(), None
    await _process_lead_intake(
        _payload(phone="217-555-9999"), crm=crm2, email_provider=provider
    )
    s2 = crm2.log_interaction.call_args.kwargs["summary"]
    assert s1 != s2
    assert "2175559999" in s2  # corrected callback visible in the summary


def test_cors_middleware_scoped_to_form_origins():
    """Rounds 6-7 R3/R12: intake CORS is a path-scoped middleware mounted
    OUTSIDE the app-wide credentialed CORSMiddleware, so real browser
    preflights (with Access-Control-Request-Method) are answered for the
    form origins, credential-free, without widening any other route."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.testclient import TestClient

    import atlas_brain.api.leads as leads_mod

    app = FastAPI()
    app.include_router(leads_mod.router, prefix="/api/v1")
    # same order as main.py: app-wide credentialed CORS first, then ours
    app.add_middleware(
        CORSMiddleware, allow_origins=["http://localhost:5174"],
        allow_credentials=True, allow_methods=["POST"], allow_headers=["*"],
    )
    app.add_middleware(leads_mod.LeadIntakeCORSMiddleware)
    client = TestClient(app)

    preflight_headers = {
        "Origin": "https://effinghamofficemaids.com",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type",
    }
    pre = client.options("/api/v1/leads/intake", headers=preflight_headers)
    assert pre.status_code == 204
    assert pre.headers["access-control-allow-origin"] == "https://effinghamofficemaids.com"
    assert "access-control-allow-credentials" not in pre.headers

    evil = client.options(
        "/api/v1/leads/intake",
        headers={**preflight_headers, "Origin": "https://evil.example"},
    )
    assert "access-control-allow-origin" not in evil.headers
    assert evil.status_code == 400

    # other routes keep the app-wide policy (dashboard preflight untouched)
    dash = client.options(
        "/api/v1/anything",
        headers={"Origin": "http://localhost:5174",
                 "Access-Control-Request-Method": "POST"},
    )
    assert dash.headers.get("access-control-allow-origin") == "http://localhost:5174"


@pytest.mark.asyncio
async def test_callback_channels_prepended_within_dedupe_prefix():
    """Round 7 R1/R6: channels must lead the summary so they sit inside the
    hashed dedupe prefix even for maximum-length messages."""
    crm, provider = _crm(), _email_provider()
    await _process_lead_intake(
        _payload(message="x" * 7900), crm=crm, email_provider=provider
    )
    summary = crm.log_interaction.call_args.kwargs["summary"]
    assert summary.startswith("Callback: jane@example.com, 2175550100")


# ---------------------------------------------------------------------------
# 8. Resend provider routing (send lead acknowledgement from info@ via Resend)
# ---------------------------------------------------------------------------


class _FakeResendResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"id": "resend-msg-1"}


class _FakeHTTPClient:
    """Stand-in for httpx.AsyncClient -- the external Resend transport."""

    posted: list = []

    def __init__(self, *args, **kwargs):
        pass

    async def post(self, url, json=None, headers=None):
        type(self).posted.append(json)
        return _FakeResendResponse()

    async def aclose(self):
        return None


@pytest.mark.asyncio
async def test_forced_resend_routes_through_real_stack(monkeypatch):
    """Integration through the REAL Composite -> ResendEmailProvider -> EmailTool
    path, mocking ONLY the external Gmail and Resend transports (Codex R2). A
    forced send must select the Resend transport from info@ -- and would fail if
    ResendEmailProvider stopped stamping force_resend or EmailTool ignored it
    (production would then route the acknowledgement back through Gmail). A
    default send must still select Gmail (unchanged for every other caller)."""
    from atlas_brain.services import google_oauth
    from atlas_brain.services.email_provider import CompositeEmailProvider
    from atlas_brain.tools import email as email_mod
    from atlas_brain.tools import gmail as gmail_mod
    from atlas_brain.tools.email import EmailTool

    # External Resend transport: give a FRESH EmailTool a fake HTTP client so the
    # Resend send never touches the network -- hermetic under any run order (the
    # shared singleton can be left holding a real httpx client + real key by an
    # earlier test, which is why patching httpx/_client on it is not enough).
    # ResendEmailProvider.send re-imports email_tool from the module, so swapping
    # the module attribute makes it use our controlled instance.
    _FakeHTTPClient.posted = []
    fresh_tool = EmailTool()
    monkeypatch.setattr(fresh_tool, "_client", _FakeHTTPClient())
    monkeypatch.setattr(fresh_tool._config, "enabled", True)
    monkeypatch.setattr(fresh_tool._config, "api_key", "re_test_key")
    monkeypatch.setattr(fresh_tool._config, "gmail_send_enabled", True)
    monkeypatch.setattr(email_mod, "email_tool", fresh_tool)

    # External Gmail transport: credentials present + a transport spy, so a
    # dropped force_resend would visibly select Gmail here.
    store = MagicMock()
    store.get_credentials = MagicMock(return_value="cred")
    monkeypatch.setattr(google_oauth, "get_google_token_store", lambda: store)
    gmail_transport = MagicMock()
    gmail_transport.send = AsyncMock(return_value={"id": "gmail-1", "threadId": "t"})
    monkeypatch.setattr(gmail_mod, "get_gmail_transport", lambda: gmail_transport)

    comp = CompositeEmailProvider()  # real GmailEmailProvider + ResendEmailProvider

    forced = await comp.send(
        to=["x@example.com"],
        subject="s",
        body="b",
        from_email="info@effinghamofficemaids.com",
        provider="resend",
    )
    assert forced["transport"] == "resend"
    gmail_transport.send.assert_not_called()
    assert len(_FakeHTTPClient.posted) == 1
    assert _FakeHTTPClient.posted[0]["from"] == "info@effinghamofficemaids.com"

    # Default send: real stack still prefers Gmail; Resend transport not touched.
    await comp.send(to=["x@example.com"], subject="s", body="b")
    gmail_transport.send.assert_awaited_once()
    assert len(_FakeHTTPClient.posted) == 1


# ---------------------------------------------------------------------------
# 9. New-lead ntfy push (dedicated eom-leads topic)
#
# Fires an instant operator heads-up on every NEW lead so real leads are not
# lost in the email noise. Gated: only on a freshly-logged lead (never on a
# honeypot or a same-day duplicate), and fire-and-forget (a push failure must
# never fail the already-captured lead). The topic is empty by default, so the
# transport stays off — and every other test in this module stays hermetic —
# until an operator sets ATLAS_ALERTS_LEADS_NTFY_TOPIC.
# ---------------------------------------------------------------------------


class _FakeNtfyResponse:
    def raise_for_status(self):
        return None


class _FakeNtfyClient:
    """Async-context-manager stand-in for httpx.AsyncClient."""

    posted: list = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, content=None, headers=None):
        type(self).posted.append({"url": url, "content": content, "headers": headers})
        return _FakeNtfyResponse()


@pytest.mark.asyncio
async def test_new_lead_fires_notification_with_channels():
    crm, provider = _crm(), _email_provider()
    notifier = AsyncMock()
    payload = _payload()

    await _process_lead_intake(
        payload, crm=crm, email_provider=provider, lead_notifier=notifier
    )

    notifier.assert_awaited_once()
    args = notifier.await_args.args
    assert args[0] is payload
    assert args[1] == "jane@example.com"  # normalized email
    assert args[2] == "2175550100"        # digits-only phone


@pytest.mark.asyncio
async def test_honeypot_does_not_notify():
    notifier = AsyncMock()
    await _process_lead_intake(
        _payload(website="http://spam.example"),
        crm=_crm(), email_provider=_email_provider(), lead_notifier=notifier,
    )
    notifier.assert_not_awaited()


@pytest.mark.asyncio
async def test_same_day_duplicate_does_not_notify():
    """A duplicate is not a new lead: no second push (mirrors the no-second-
    email guarantee)."""
    notifier = AsyncMock()
    await _process_lead_intake(
        _payload(), crm=_crm(inserted=False),
        email_provider=_email_provider(), lead_notifier=notifier,
    )
    notifier.assert_not_awaited()


@pytest.mark.asyncio
async def test_phone_only_lead_still_notifies():
    """The push is independent of the ack email, so a phone-only lead (no
    email address, hence no acknowledgement) still reaches the operator."""
    notifier = AsyncMock()
    await _process_lead_intake(
        _payload(email=""), crm=_crm(),
        email_provider=_email_provider(), lead_notifier=notifier,
    )
    notifier.assert_awaited_once()
    args = notifier.await_args.args
    assert args[1] == ""
    assert args[2] == "2175550100"


@pytest.mark.asyncio
async def test_notifier_failure_never_fails_request():
    notifier = AsyncMock(side_effect=RuntimeError("ntfy unreachable"))
    result = await _process_lead_intake(
        _payload(), crm=_crm(),
        email_provider=_email_provider(), lead_notifier=notifier,
    )
    assert result["success"] is True  # lead captured despite the push failing


def test_format_phone_for_display():
    assert _format_phone_for_display("2175550100") == "(217) 555-0100"
    assert _format_phone_for_display("12175550100") == "(217) 555-0100"  # US +1
    assert _format_phone_for_display("5550100") == "5550100"             # short: passthrough
    assert _format_phone_for_display("") == ""


def test_lead_push_body_layout_name_channels_service_address():
    body = _lead_push_body(
        _payload(address="207 Santa Fe Ave, Effingham, IL"),
        "jane@example.com", "2175550100",
    )
    lines = body.split("\n")
    assert lines[0] == "Jane Doe"  # exact name leads the UTF-8 body
    assert lines[1] == "(217) 555-0100 · jane@example.com"
    assert lines[2] == "residential · bi-weekly"
    assert lines[3] == "207 Santa Fe Ave, Effingham, IL"


def test_lead_push_body_phone_only_and_no_detail():
    body = _lead_push_body(
        _payload(email="", address="", service="", frequency=""), "", "2175550100"
    )
    assert body == "Jane Doe\n(217) 555-0100"


def test_non_ascii_name_title_is_ascii_body_keeps_exact_name():
    """A valid non-ASCII name must neither crash the HTTP Title header (httpx
    0.28 raises on non-latin-1) nor render as mojibake on ntfy (which decodes
    header bytes as UTF-8). So the title stays strictly ASCII — generic when the
    name is not fully ASCII — while the exact original name rides the UTF-8 body.
    Regresses Codex #2332 R1/R2/R13."""
    payload = _payload(name="José 王伟")
    title = _lead_push_title(payload)
    title.encode("ascii")  # would raise if any non-ASCII byte leaked into the header
    assert title == "New lead"                 # generic: name is not fully ASCII
    body = _lead_push_body(payload, "jane@example.com", "2175550100")
    assert body.split("\n")[0] == "José 王伟"   # exact name preserved in the body


def test_title_generic_when_name_not_fully_ascii():
    assert _lead_push_title(_payload(name="王伟")) == "New lead"
    assert _lead_push_title(_payload(name="José")) == "New lead"  # accents too


def test_title_strips_control_chars_from_ascii_name():
    # header-injection safe: a newline in an otherwise-ASCII name is removed
    assert _lead_push_title(_payload(name="Jane\nBcc: x")) == "New lead: JaneBcc: x"


@pytest.mark.asyncio
async def test_publish_non_ascii_name_sends_ascii_header(monkeypatch):
    """End-to-end: a non-ASCII lead name produces an ASCII Title header (accepted
    by the transport, rendered correctly by ntfy) and the exact name in the body,
    so the push is actually delivered rather than silently swallowed."""
    from atlas_brain.config import settings
    monkeypatch.setattr(settings.alerts, "ntfy_enabled", True)
    monkeypatch.setattr(settings.alerts, "leads_ntfy_topic", "eom-leads-abc")
    import httpx

    _FakeNtfyClient.posted = []
    monkeypatch.setattr(httpx, "AsyncClient", _FakeNtfyClient)

    await _default_lead_notifier(_payload(name="José 王伟"), "jane@example.com", "2175550100")

    assert len(_FakeNtfyClient.posted) == 1
    sent = _FakeNtfyClient.posted[0]
    sent["headers"]["Title"].encode("ascii")  # transport-safe + no mojibake
    assert "José 王伟".encode("utf-8") in sent["content"]  # exact name in the body


@pytest.mark.asyncio
async def test_publish_skipped_when_topic_unset(monkeypatch):
    from atlas_brain.config import settings
    monkeypatch.setattr(settings.alerts, "ntfy_enabled", True)
    monkeypatch.setattr(settings.alerts, "leads_ntfy_topic", "")
    import httpx

    def _boom(*a, **k):
        raise AssertionError("must not open an HTTP client when no topic is set")

    monkeypatch.setattr(httpx, "AsyncClient", _boom)
    await _publish_lead_ntfy("t", "b")  # returns cleanly, no transport


@pytest.mark.asyncio
async def test_publish_skipped_when_ntfy_disabled(monkeypatch):
    from atlas_brain.config import settings
    monkeypatch.setattr(settings.alerts, "ntfy_enabled", False)
    monkeypatch.setattr(settings.alerts, "leads_ntfy_topic", "eom-leads-x")
    import httpx

    def _boom(*a, **k):
        raise AssertionError("must not POST when ntfy is disabled")

    monkeypatch.setattr(httpx, "AsyncClient", _boom)
    await _publish_lead_ntfy("t", "b")


@pytest.mark.asyncio
async def test_publish_posts_to_configured_leads_topic(monkeypatch):
    from atlas_brain.config import settings
    monkeypatch.setattr(settings.alerts, "ntfy_enabled", True)
    monkeypatch.setattr(settings.alerts, "ntfy_url", "https://ntfy.example/")  # trailing slash
    monkeypatch.setattr(settings.alerts, "leads_ntfy_topic", "eom-leads-abc")
    import httpx

    _FakeNtfyClient.posted = []
    monkeypatch.setattr(httpx, "AsyncClient", _FakeNtfyClient)

    await _publish_lead_ntfy("New lead: Jane Doe", "body-here")

    assert len(_FakeNtfyClient.posted) == 1
    sent = _FakeNtfyClient.posted[0]
    assert sent["url"] == "https://ntfy.example/eom-leads-abc"  # single-slash join
    assert sent["content"] == b"body-here"
    assert sent["headers"]["Title"] == "New lead: Jane Doe"
    assert sent["headers"]["Priority"] == "high"
    assert sent["headers"]["Tags"] == "moneybag"


@pytest.mark.asyncio
async def test_publish_swallows_transport_error(monkeypatch):
    from atlas_brain.config import settings
    monkeypatch.setattr(settings.alerts, "ntfy_enabled", True)
    monkeypatch.setattr(settings.alerts, "leads_ntfy_topic", "eom-leads-abc")
    import httpx

    class _BoomClient(_FakeNtfyClient):
        async def post(self, *a, **k):
            raise RuntimeError("network down")

    monkeypatch.setattr(httpx, "AsyncClient", _BoomClient)
    await _publish_lead_ntfy("t", "b")  # must not raise


def test_lead_push_title_uses_name():
    from atlas_brain.api.leads import _lead_push_title
    assert _lead_push_title(_payload()) == "New lead: Jane Doe"


def test_lead_push_title_falls_back_when_name_blank():
    from atlas_brain.api.leads import _lead_push_title
    # name has a min_length=1 constraint, so a lone space is the blank-ish edge
    assert _lead_push_title(_payload(name=" ")) == "New lead"


@pytest.mark.asyncio
async def test_default_notifier_publishes_built_title_and_body(monkeypatch):
    """End-to-end through the real transport gate: the default notifier feeds
    the pure title/body builders into _publish_lead_ntfy, which posts them.
    Patches only the third-party httpx transport (not any first-party target)."""
    from atlas_brain.config import settings
    monkeypatch.setattr(settings.alerts, "ntfy_enabled", True)
    monkeypatch.setattr(settings.alerts, "leads_ntfy_topic", "eom-leads-abc")
    import httpx

    _FakeNtfyClient.posted = []
    monkeypatch.setattr(httpx, "AsyncClient", _FakeNtfyClient)

    await _default_lead_notifier(_payload(), "jane@example.com", "2175550100")

    assert len(_FakeNtfyClient.posted) == 1
    sent = _FakeNtfyClient.posted[0]
    assert sent["headers"]["Title"] == "New lead: Jane Doe"
    assert b"jane@example.com" in sent["content"]


def test_route_level_post_delivers_ntfy(monkeypatch):
    """POST /api/v1/leads/intake reaches the observable delivery effect: with the
    REAL notifier dependency wired (not overridden), a successful intake emits an
    ntfy POST to the configured topic. Only the third-party httpx transport is
    faked. Regresses Codex #2332 R2 (no route-level delivery proof)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import atlas_brain.api.leads as leads_mod
    from atlas_brain.config import settings

    monkeypatch.setattr(settings.alerts, "ntfy_enabled", True)
    monkeypatch.setattr(settings.alerts, "leads_ntfy_topic", "eom-leads-route")
    import httpx
    _FakeNtfyClient.posted = []
    monkeypatch.setattr(httpx, "AsyncClient", _FakeNtfyClient)

    async def fake_count(email, phone):
        return 0

    async def fake_volume():
        return 0

    app = FastAPI()
    app.include_router(leads_mod.router, prefix="/api/v1")
    app.dependency_overrides[leads_mod._crm_dependency] = lambda: _crm()
    app.dependency_overrides[leads_mod._email_dependency] = lambda: _email_provider()
    app.dependency_overrides[leads_mod._email_history_dependency] = _email_history
    app.dependency_overrides[leads_mod._daily_count_dependency] = lambda: fake_count
    app.dependency_overrides[leads_mod._ack_volume_dependency] = lambda: fake_volume
    app.dependency_overrides[leads_mod._notify_volume_dependency] = lambda: fake_volume
    # _notify_dependency is intentionally NOT overridden: the real
    # _default_lead_notifier runs, proving the route wires notifier -> transport.
    client = TestClient(app)

    ok = client.post(
        "/api/v1/leads/intake", json={"name": "Jane", "email": "jane@example.com"}
    )
    assert ok.status_code == 200 and ok.json()["success"] is True
    assert len(_FakeNtfyClient.posted) == 1
    sent = _FakeNtfyClient.posted[0]
    assert sent["url"].endswith("/eom-leads-route")
    assert sent["headers"]["Title"] == "New lead: Jane"


@pytest.mark.asyncio
async def test_notification_volume_cap_skips_push_over_ceiling():
    """A public flood of distinct identities cannot spam the phone: once the
    hourly lead volume exceeds the ceiling the push is skipped (lead still
    captured). Regresses Codex #2332 R3/R8."""
    from atlas_brain.api.leads import GLOBAL_NOTIFY_HOURLY_CAP
    notifier = AsyncMock()
    over = AsyncMock(return_value=GLOBAL_NOTIFY_HOURLY_CAP + 1)
    result = await _process_lead_intake(
        _payload(), crm=_crm(), email_provider=_email_provider(),
        lead_notifier=notifier, notify_volume=over,
    )
    assert result["success"] is True
    notifier.assert_not_awaited()
    over.assert_awaited_once()


@pytest.mark.asyncio
async def test_notification_volume_under_cap_sends():
    from atlas_brain.api.leads import GLOBAL_NOTIFY_HOURLY_CAP
    notifier = AsyncMock()
    at_cap = AsyncMock(return_value=GLOBAL_NOTIFY_HOURLY_CAP)  # == cap: still sends
    await _process_lead_intake(
        _payload(), crm=_crm(), email_provider=_email_provider(),
        lead_notifier=notifier, notify_volume=at_cap,
    )
    notifier.assert_awaited_once()


@pytest.mark.asyncio
async def test_notification_volume_check_failure_fails_closed():
    """A broken volume query must not fail the captured lead, and must fail closed
    (skip the push) rather than fire uncapped."""
    notifier = AsyncMock()
    boom = AsyncMock(side_effect=RuntimeError("db pool down"))
    result = await _process_lead_intake(
        _payload(), crm=_crm(), email_provider=_email_provider(),
        lead_notifier=notifier, notify_volume=boom,
    )
    assert result["success"] is True
    notifier.assert_not_awaited()


@pytest.mark.asyncio
async def test_direct_caller_without_notifier_does_not_notify(monkeypatch):
    """A direct _process_lead_intake caller that omits lead_notifier performs NO
    notification — even with the topic configured and the transport live, nothing
    is published. This is what keeps the rest of this module (and any caller run
    from the deployed checkout) off the live topic. Regresses Codex #2332 R2/R12."""
    from atlas_brain.config import settings
    monkeypatch.setattr(settings.alerts, "ntfy_enabled", True)
    monkeypatch.setattr(settings.alerts, "leads_ntfy_topic", "eom-leads-live")
    import httpx

    def _boom(*a, **k):
        raise AssertionError("a notifier-less direct caller must not publish")

    monkeypatch.setattr(httpx, "AsyncClient", _boom)
    result = await _process_lead_intake(
        _payload(), crm=_crm(), email_provider=_email_provider()
    )
    assert result["success"] is True
