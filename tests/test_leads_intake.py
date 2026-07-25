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

import sys
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
    _process_lead_intake,
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
    assert kwargs["email"] == "jane@example.com"
    assert kwargs["phone"] == "2175550100"  # digits-only normalization


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
async def test_create_contact_dedupe_skips_foreign_tenant_matches_up_front():
    """Cross-tenant regression: a stamped create must never resolve to a
    contact that belongs to a DIFFERENT business context."""
    provider = _provider_with(
        [{"id": "b2b-1", "business_context_id": "churnsignals"}]
    )
    # No compatible match -> falls through to the insert path, which needs a
    # pool; patch it to observe the outcome.
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
    assert result["id"] == "new-eom"  # created fresh, foreign row untouched
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
