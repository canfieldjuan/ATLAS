from datetime import datetime, timezone
from uuid import uuid4

import pytest

from atlas_brain.api import auth as mod


class _FakeResponse:
    def raise_for_status(self):
        return None


class _FakeAsyncClient:
    def __init__(self, capture):
        self.capture = capture

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, headers=None, json=None):
        self.capture["url"] = url
        self.capture["headers"] = headers
        self.capture["json"] = json
        return _FakeResponse()


def _patch_async_client(monkeypatch, capture):
    monkeypatch.setattr(
        mod.httpx,
        "AsyncClient",
        lambda timeout=15.0: _FakeAsyncClient(capture),
    )


def test_onboarding_sequence_helpers_use_configured_values(monkeypatch):
    monkeypatch.setattr(
        mod.settings.campaign_sequence,
        "onboarding_product_name",
        "Atlas Growth",
    )
    monkeypatch.setattr(
        mod.settings.campaign_sequence,
        "onboarding_sender_name",
        "Customer Success",
    )
    monkeypatch.setattr(
        mod.settings.campaign_sequence,
        "onboarding_sender_company",
        "Example Intelligence",
    )
    monkeypatch.setattr(
        mod.settings.b2b_campaign,
        "default_booking_url",
        "https://app.example.com/book",
    )

    company_context = mod._onboarding_sequence_company_context(
        account_id=uuid4(),
        product="b2b_retention",
        plan="b2b_trial",
        trial_ends=datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc),
    )
    selling_context = mod._onboarding_sequence_selling_context()

    assert company_context["recipient_type"] == "onboarding"
    assert company_context["product"] == "b2b_retention"
    assert company_context["product_name"] == "Atlas Growth"
    assert selling_context == {
        "sender_name": "Customer Success",
        "sender_company": "Example Intelligence",
        "booking_url": "https://app.example.com/book",
    }


@pytest.mark.asyncio
async def test_send_password_reset_email_uses_configured_frontend_and_branding(monkeypatch):
    capture = {}
    _patch_async_client(monkeypatch, capture)

    monkeypatch.setattr(mod.settings.campaign_sequence, "resend_api_key", "key")
    monkeypatch.setattr(
        mod.settings.campaign_sequence,
        "resend_from_email",
        "auth@example.com",
    )
    monkeypatch.setattr(
        mod.settings.saas_auth,
        "frontend_base_url",
        "https://app.example.com/",
    )
    monkeypatch.setattr(
        mod.settings.saas_auth,
        "email_product_name",
        "Atlas Cloud",
    )
    monkeypatch.setattr(
        mod.settings.saas_auth,
        "email_company_name",
        "Example Intelligence",
    )

    await mod._send_password_reset_email(
        "owner@example.com",
        "Pat",
        "reset-token",
    )

    payload = capture["json"]
    assert payload["subject"] == "Reset your Atlas Cloud password"
    assert payload["from"] == "auth@example.com"
    assert "https://app.example.com/reset-password?token=reset-token" in payload["html"]
    assert "Atlas Cloud by Example Intelligence" in payload["html"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("product", "subject_name", "heading_name"),
    [
        ("b2b_retention", "Atlas B2B Suite", "Configured B2B Heading"),
        ("consumer", "Atlas Seller Suite", "Atlas Seller Suite"),
    ],
)
async def test_send_welcome_email_uses_configured_names_and_trial_days(
    monkeypatch,
    product,
    subject_name,
    heading_name,
):
    capture = {}
    _patch_async_client(monkeypatch, capture)

    monkeypatch.setattr(mod.settings.campaign_sequence, "resend_api_key", "key")
    monkeypatch.setattr(
        mod.settings.campaign_sequence,
        "resend_from_email",
        "welcome@example.com",
    )
    monkeypatch.setattr(mod.settings.saas_auth, "trial_days", 21)
    monkeypatch.setattr(
        mod.settings.saas_auth,
        "b2b_welcome_product_name",
        "Atlas B2B Suite",
    )
    monkeypatch.setattr(
        mod.settings.saas_auth,
        "b2b_welcome_heading",
        "Configured B2B Heading",
    )
    monkeypatch.setattr(
        mod.settings.saas_auth,
        "consumer_welcome_product_name",
        "Atlas Seller Suite",
    )

    await mod._send_welcome_email("owner@example.com", "Pat", product)

    payload = capture["json"]
    assert payload["subject"] == f"Welcome to {subject_name}"
    assert f"<h2>Welcome to {heading_name}!</h2>" in payload["html"]
    assert "Your 21-day trial gives you full access" in payload["html"]
