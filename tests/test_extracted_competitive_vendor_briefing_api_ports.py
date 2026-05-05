from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from extracted_competitive_intelligence.api import b2b_vendor_briefing as api
from extracted_competitive_intelligence.services.b2b import (
    vendor_briefing_api_ports as ports,
)


class FakeStripeError(Exception):
    pass


class FakeStripeSessionAPI:
    created: dict[str, Any] | None = None

    @classmethod
    def create(cls, **params: Any) -> SimpleNamespace:
        cls.created = params
        return SimpleNamespace(url="https://stripe.example/checkout/session")

    @classmethod
    def retrieve(cls, session_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            customer_details=SimpleNamespace(email="buyer@example.com"),
            customer_email="fallback@example.com",
            metadata={
                "vendor_name": "Acme",
                "tier": "pro",
                "source": "vendor_briefing_report",
            },
            payment_status="paid",
        )


class FakeStripe:
    StripeError = FakeStripeError
    checkout = SimpleNamespace(Session=FakeStripeSessionAPI)
    api_key = ""


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return {"id": "resend-msg-1"}


class FakeHTTPClient:
    calls: list[dict[str, Any]] = []

    def __init__(self, *, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> "FakeHTTPClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> FakeResponse:
        self.calls.append({
            "timeout": self.timeout,
            "url": url,
            "headers": headers,
            "json": json,
        })
        return FakeResponse()


class FakeAPIPort:
    def __init__(self) -> None:
        self.checkout_calls: list[dict[str, Any]] = []
        self.session_calls: list[str] = []
        self.confirmation_calls: list[dict[str, str]] = []

    async def create_checkout_session(
        self,
        *,
        vendor_name: str,
        tier: str,
        customer_email: str | None = None,
    ) -> ports.VendorCheckoutSession:
        self.checkout_calls.append({
            "vendor_name": vendor_name,
            "tier": tier,
            "customer_email": customer_email,
        })
        return ports.VendorCheckoutSession(url="https://checkout.example/session")

    async def retrieve_checkout_session(
        self,
        session_id: str,
    ) -> ports.VendorCheckoutSessionInfo:
        self.session_calls.append(session_id)
        return ports.VendorCheckoutSessionInfo(
            customer_email="buyer@example.com",
            vendor_name="Acme",
            tier="standard",
            source="vendor_briefing_report",
            payment_status="paid",
        )

    async def send_gated_report_email(
        self,
        *,
        vendor_name: str,
        recipient_email: str,
        report_data: dict[str, Any],
        briefing_data: dict[str, Any],
    ) -> ports.GatedReportDelivery:
        return ports.GatedReportDelivery(provider_message_id="email-1")

    async def send_checkout_confirmation_email(
        self,
        *,
        vendor_name: str,
        tier: str,
        customer_email: str,
    ) -> ports.CheckoutConfirmationDelivery:
        self.confirmation_calls.append({
            "vendor_name": vendor_name,
            "tier": tier,
            "customer_email": customer_email,
        })
        return ports.CheckoutConfirmationDelivery(provider_message_id="confirm-1")


class FakePool:
    def __init__(self) -> None:
        self.fetchval_calls: list[tuple[str, str]] = []
        self.execute_calls: list[tuple[Any, ...]] = []

    async def fetchval(self, query: str, value: str) -> None:
        self.fetchval_calls.append((query, value))
        return None

    async def execute(self, *args: Any) -> None:
        self.execute_calls.append(args)
        return None


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        b2b_churn=SimpleNamespace(
            vendor_briefing_report_base_url="https://reports.example/vendor",
            vendor_briefing_gate_base_url="https://gate.example/report",
            vendor_briefing_reply_to_email="reply@example.com",
            vendor_briefing_sender_name="Signals Team",
        ),
        campaign_sequence=SimpleNamespace(
            resend_api_key="resend-key",
            resend_from_email="sender@example.com",
            resend_api_url="https://resend.example/emails",
            resend_timeout_seconds=12.5,
        ),
        saas_auth=SimpleNamespace(
            stripe_secret_key="stripe-secret",
            stripe_vendor_standard_price_id="price-standard",
            stripe_vendor_pro_price_id="price-pro",
        ),
    )


def teardown_function() -> None:
    ports.configure_vendor_briefing_api_port(None)
    FakeStripeSessionAPI.created = None
    FakeHTTPClient.calls.clear()
    sys.modules.pop("stripe", None)
    sys.modules.pop("httpx", None)


@pytest.mark.asyncio
async def test_default_port_creates_checkout_from_configured_values(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "stripe", FakeStripe)
    monkeypatch.setattr(ports, "settings", _settings())

    session = await ports.DefaultVendorBriefingAPIPort().create_checkout_session(
        vendor_name="Acme Corp",
        tier="pro",
        customer_email="BUYER@EXAMPLE.COM",
    )

    assert session.url == "https://stripe.example/checkout/session"
    assert FakeStripe.api_key == "stripe-secret"
    assert FakeStripeSessionAPI.created == {
        "mode": "subscription",
        "line_items": [{"price": "price-pro", "quantity": 1}],
        "success_url": (
            "https://reports.example/vendor?vendor=Acme%20Corp"
            "&checkout=success&session_id={CHECKOUT_SESSION_ID}"
        ),
        "cancel_url": (
            "https://reports.example/vendor?vendor=Acme%20Corp&checkout=cancelled"
        ),
        "metadata": {
            "vendor_name": "Acme Corp",
            "tier": "pro",
            "source": "vendor_briefing_report",
        },
        "customer_email": "buyer@example.com",
    }


@pytest.mark.asyncio
async def test_default_port_sends_gated_report_through_configured_resend(
    monkeypatch,
) -> None:
    monkeypatch.setattr(ports, "settings", _settings())
    monkeypatch.setattr(
        ports,
        "render_vendor_full_report_pdf",
        lambda **_kwargs: b"%PDF-report",
    )
    monkeypatch.setattr(
        ports,
        "render_report_delivery_html",
        lambda vendor_name: f"<p>{vendor_name}</p>",
    )
    monkeypatch.setitem(
        sys.modules,
        "httpx",
        SimpleNamespace(AsyncClient=FakeHTTPClient),
    )

    delivery = await ports.DefaultVendorBriefingAPIPort().send_gated_report_email(
        vendor_name="Acme",
        recipient_email="buyer@example.com",
        report_data={"report": True},
        briefing_data={"briefing": True},
    )

    assert delivery.provider_message_id == "resend-msg-1"
    assert FakeHTTPClient.calls == [{
        "timeout": 12.5,
        "url": "https://resend.example/emails",
        "headers": {
            "Authorization": "Bearer resend-key",
            "Content-Type": "application/json",
        },
        "json": {
            "from": "Signals Team <sender@example.com>",
            "to": ["buyer@example.com"],
            "subject": "Your Acme Churn Intelligence Report",
            "html": "<p>Acme</p>",
            "reply_to": "reply@example.com",
            "attachments": [{
                "filename": "acme-churn-report.pdf",
                "content": "JVBERi1yZXBvcnQ=",
            }],
        },
    }]


@pytest.mark.asyncio
async def test_api_checkout_route_uses_configured_runtime_port(monkeypatch) -> None:
    fake_port = FakeAPIPort()
    ports.configure_vendor_briefing_api_port(fake_port)
    monkeypatch.setattr(api, "settings", _settings())

    result = await api.vendor_checkout(
        api.VendorCheckoutRequest(
            vendor_name=" Acme ",
            tier="standard",
            email=" BUYER@EXAMPLE.COM ",
        )
    )

    assert result == {"url": "https://checkout.example/session"}
    assert fake_port.checkout_calls == [{
        "vendor_name": "Acme",
        "tier": "standard",
        "customer_email": "BUYER@EXAMPLE.COM",
    }]


@pytest.mark.asyncio
async def test_api_checkout_session_confirmation_uses_runtime_port(
    monkeypatch,
) -> None:
    fake_port = FakeAPIPort()
    fake_pool = FakePool()
    ports.configure_vendor_briefing_api_port(fake_port)
    monkeypatch.setattr(api, "settings", _settings())
    monkeypatch.setattr(api, "get_db_pool", lambda: fake_pool)

    result = await api.checkout_session_info("cs_test_12345")

    assert result == {
        "email": "buyer@example.com",
        "vendor_name": "Acme",
        "tier": "standard",
    }
    assert fake_port.session_calls == ["cs_test_12345"]
    assert fake_port.confirmation_calls == [{
        "vendor_name": "Acme",
        "tier": "standard",
        "customer_email": "buyer@example.com",
    }]
    assert fake_pool.fetchval_calls[0][1] == "vendor_checkout_email_cs_test_12345"
    assert fake_pool.execute_calls
