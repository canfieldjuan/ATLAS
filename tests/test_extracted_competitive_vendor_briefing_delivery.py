from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from extracted_competitive_intelligence.services.b2b import (
    vendor_briefing_delivery as delivery,
)
from extracted_competitive_intelligence.autonomous.tasks import (
    b2b_vendor_briefing as briefing_task,
)


class RecordingSender:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def send(
        self,
        *,
        to: str,
        from_email: str,
        subject: str,
        body: str,
        tags: list[dict[str, str]] | None = None,
    ) -> dict[str, str]:
        self.calls.append({
            "to": to,
            "from_email": from_email,
            "subject": subject,
            "body": body,
            "tags": tags,
        })
        return {"id": "resend-1"}


class RecordingPool:
    is_initialized = True

    def __init__(self, *, row: dict[str, Any] | None = None) -> None:
        self.row = row
        self.executed: list[tuple[Any, ...]] = []

    async def fetchrow(self, *_args: Any) -> dict[str, Any] | None:
        return self.row

    async def execute(self, *args: Any) -> None:
        self.executed.append(args)


def _settings(
    *,
    resend_api_key: str = "resend-key",
    resend_from_email: str = "sender@example.com",
) -> SimpleNamespace:
    return SimpleNamespace(
        campaign_sequence=SimpleNamespace(
            resend_api_key=resend_api_key,
            resend_from_email=resend_from_email,
        ),
        b2b_churn=SimpleNamespace(
            vendor_briefing_sender_name="Atlas Intelligence",
            vendor_briefing_standard_churn_subject_template=(
                "Churn Intelligence Briefing: {vendor_name}"
            ),
            vendor_briefing_standard_sales_subject_template=(
                "Sales Intelligence Briefing: {vendor_name}"
            ),
            vendor_briefing_prospect_churn_subject_template=(
                "{vendor_name} -- Churn Signals Detected"
            ),
            vendor_briefing_prospect_sales_subject_template=(
                "{vendor_name} -- Accounts In Motion"
            ),
            vendor_briefing_gated_churn_subject_template=(
                "Your {vendor_name} Churn Intelligence Report"
            ),
            vendor_briefing_gated_sales_subject_template=(
                "Your {vendor_name} Sales Intelligence Report"
            ),
            vendor_briefing_tag_type_name="type",
            vendor_briefing_tag_type_value="vendor_briefing",
            vendor_briefing_tag_vendor_name="vendor",
        ),
    )


@pytest.mark.parametrize(
    ("briefing_data", "expected"),
    [
        ({}, "Churn Intelligence Briefing: Acme"),
        ({"challenger_mode": True}, "Sales Intelligence Briefing: Acme"),
        ({"prospect_mode": True}, "Acme -- Churn Signals Detected"),
        (
            {"prospect_mode": True, "challenger_mode": True},
            "Acme -- Accounts In Motion",
        ),
        (
            {"is_gated_delivery": True},
            "Your Acme Churn Intelligence Report",
        ),
        (
            {"is_gated_delivery": True, "challenger_mode": True},
            "Your Acme Sales Intelligence Report",
        ),
    ],
)
def test_build_vendor_briefing_subject_matches_existing_modes(
    briefing_data: dict[str, Any],
    expected: str,
) -> None:
    assert delivery.build_vendor_briefing_subject("Acme", briefing_data) == expected


def test_build_vendor_briefing_subject_uses_configured_templates(monkeypatch) -> None:
    settings = _settings()
    settings.b2b_churn.vendor_briefing_standard_churn_subject_template = (
        "Configured briefing for {vendor_name}"
    )
    monkeypatch.setattr(delivery, "settings", settings)

    assert delivery.build_vendor_briefing_subject("Acme", {}) == (
        "Configured briefing for Acme"
    )


def test_delivery_configuration_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(delivery, "settings", _settings(resend_api_key=""))

    assert not delivery.vendor_briefing_delivery_configured()
    with pytest.raises(delivery.VendorBriefingDeliveryNotConfigured):
        delivery.require_vendor_briefing_delivery_configured()


@pytest.mark.asyncio
async def test_send_vendor_briefing_delivery_uses_shared_sender_contract(
    monkeypatch,
) -> None:
    sender = RecordingSender()
    monkeypatch.setattr(delivery, "settings", _settings())
    monkeypatch.setattr(delivery, "get_campaign_sender", lambda: sender)

    result = await delivery.send_vendor_briefing_delivery(
        to_email="buyer@example.com",
        vendor_name="Acme",
        subject="Churn Intelligence Briefing: Acme",
        briefing_html="<p>briefing</p>",
    )

    assert result.provider_message_id == "resend-1"
    assert sender.calls == [{
        "to": "buyer@example.com",
        "from_email": "Atlas Intelligence <sender@example.com>",
        "subject": "Churn Intelligence Briefing: Acme",
        "body": "<p>briefing</p>",
        "tags": [
            {"name": "type", "value": "vendor_briefing"},
            {"name": "vendor", "value": "Acme"},
        ],
    }]


@pytest.mark.asyncio
async def test_send_vendor_briefing_routes_through_shared_delivery_helper(
    monkeypatch,
) -> None:
    pool = RecordingPool()
    calls: list[dict[str, str]] = []

    async def fake_delivery(**kwargs: str) -> delivery.VendorBriefingDeliveryResult:
        calls.append(kwargs)
        return delivery.VendorBriefingDeliveryResult(provider_message_id="send-1")

    async def not_suppressed(*_args: Any, **_kwargs: Any) -> bool:
        return False

    async def resolve_name(value: str) -> str:
        return value

    monkeypatch.setattr(briefing_task, "get_db_pool", lambda: pool)
    monkeypatch.setattr(briefing_task, "is_suppressed", not_suppressed)
    monkeypatch.setattr(briefing_task, "resolve_vendor_name", resolve_name)
    monkeypatch.setattr(
        briefing_task,
        "require_vendor_briefing_delivery_configured",
        lambda: None,
    )
    monkeypatch.setattr(briefing_task, "send_vendor_briefing_delivery", fake_delivery)

    result = await briefing_task.send_vendor_briefing(
        to_email="buyer@example.com",
        vendor_name="Acme",
        briefing_html="<p>briefing</p>",
        briefing_data={"challenger_mode": True},
    )

    assert result == {
        "resend_id": "send-1",
        "status": "sent",
        "subject": "Sales Intelligence Briefing: Acme",
    }
    assert calls == [{
        "to_email": "buyer@example.com",
        "vendor_name": "Acme",
        "subject": "Sales Intelligence Briefing: Acme",
        "briefing_html": "<p>briefing</p>",
    }]
    assert pool.executed


@pytest.mark.asyncio
async def test_send_approved_briefing_routes_through_shared_delivery_helper(
    monkeypatch,
) -> None:
    pool = RecordingPool(
        row={
            "vendor_name": "Acme",
            "recipient_email": "buyer@example.com",
            "subject": "Stored Subject",
            "briefing_html": "<p>approved</p>",
            "briefing_data": {"vendor": "Acme"},
        }
    )
    calls: list[dict[str, str]] = []

    async def fake_delivery(**kwargs: str) -> delivery.VendorBriefingDeliveryResult:
        calls.append(kwargs)
        return delivery.VendorBriefingDeliveryResult(provider_message_id="send-2")

    monkeypatch.setattr(briefing_task, "get_db_pool", lambda: pool)
    monkeypatch.setattr(
        briefing_task,
        "require_vendor_briefing_delivery_configured",
        lambda: None,
    )
    monkeypatch.setattr(briefing_task, "send_vendor_briefing_delivery", fake_delivery)

    result = await briefing_task.send_approved_briefing("briefing-1")

    assert result == {
        "id": "briefing-1",
        "vendor_name": "Acme",
        "to_email": "buyer@example.com",
        "status": "sent",
        "resend_id": "send-2",
    }
    assert calls == [{
        "to_email": "buyer@example.com",
        "vendor_name": "Acme",
        "subject": "Stored Subject",
        "briefing_html": "<p>approved</p>",
    }]
    assert "approved_at = NOW()" in pool.executed[0][0]
