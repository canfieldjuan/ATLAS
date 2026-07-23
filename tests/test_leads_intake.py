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
    crm = MagicMock()
    crm.search_contacts = AsyncMock(return_value=existing or [])
    crm.find_or_create_contact = AsyncMock(return_value={"id": "c-123"})
    crm.log_interaction = AsyncMock(return_value={"id": "i-1", "inserted": inserted})
    return crm


def _email_provider(success: bool = True):
    provider = MagicMock()
    provider.send = AsyncMock(return_value={"success": success})
    return provider


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
    crm, provider = _crm(), _email_provider()
    counter = AsyncMock(return_value=MAX_DAILY_SUBMISSIONS)
    with pytest.raises(LeadRateLimitedError):
        await _process_lead_intake(
            _payload(), crm=crm, email_provider=provider, daily_count=counter
        )
    crm.find_or_create_contact.assert_not_awaited()
    crm.log_interaction.assert_not_awaited()
    provider.send.assert_not_awaited()


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


@pytest.mark.asyncio
async def test_create_contact_dedupe_is_tenant_scoped():
    """Provider-level regression for the cross-tenant mutation finding:
    when data carries business_context_id, both dedupe searches must be
    scoped to that tenant."""
    from atlas_brain.services.crm_provider import DatabaseCRMProvider

    provider = DatabaseCRMProvider.__new__(DatabaseCRMProvider)
    provider.search_contacts = AsyncMock(return_value=[{"id": "existing-eom"}])
    provider.update_contact = AsyncMock(return_value={"id": "existing-eom"})

    result = await provider.create_contact(
        {
            "full_name": "Jane Doe",
            "email": "jane@example.com",
            "phone": "2175550100",
            "business_context_id": "effingham_maids",
        }
    )
    assert result["id"] == "existing-eom"
    for call in provider.search_contacts.await_args_list:
        assert call.kwargs.get("business_context_id") == "effingham_maids"


@pytest.mark.asyncio
async def test_create_contact_dedupe_unscoped_without_context():
    """Without a stamped tenant the legacy global dedupe is preserved."""
    from atlas_brain.services.crm_provider import DatabaseCRMProvider

    provider = DatabaseCRMProvider.__new__(DatabaseCRMProvider)
    provider.search_contacts = AsyncMock(return_value=[{"id": "existing-any"}])
    provider.update_contact = AsyncMock(return_value={"id": "existing-any"})

    await provider.create_contact({"full_name": "X", "email": "x@example.com"})
    for call in provider.search_contacts.await_args_list:
        assert "business_context_id" not in call.kwargs


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
    existing = [{"id": "cust-9", "contact_type": "customer", "full_name": "Real Name"}]
    crm, provider = _crm(existing=existing), _email_provider()

    result = await _process_lead_intake(_payload(), crm=crm, email_provider=provider)

    assert result["success"] is True
    crm.find_or_create_contact.assert_not_awaited()  # no create, no merge
    crm.log_interaction.assert_awaited()  # the request is still recorded
    assert crm.log_interaction.call_args.kwargs["contact_id"] == "cust-9"
    # resolution searches are tenant-scoped
    for call in crm.search_contacts.await_args_list:
        assert call.kwargs["business_context_id"] == EOM_BUSINESS_CONTEXT_ID


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
    crm, provider = _crm(), _email_provider()
    volume = AsyncMock(return_value=1000)
    result = await _process_lead_intake(
        _payload(), crm=crm, email_provider=provider, ack_volume=volume
    )
    assert result["success"] is True
    assert result["email_sent"] is False
    provider.send.assert_not_awaited()
    crm.log_interaction.assert_awaited()  # lead capture unaffected


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
