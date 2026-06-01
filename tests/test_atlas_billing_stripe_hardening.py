from __future__ import annotations

import logging
import sys
import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from atlas_brain.api import billing
from atlas_brain.auth.dependencies import AuthUser


class _CustomerApi:
    created: list[dict[str, Any]] = []

    @classmethod
    def create(cls, **kwargs: Any) -> SimpleNamespace:
        cls.created.append(kwargs)
        return SimpleNamespace(id="cus_test_created")


class _CheckoutSessionApi:
    created: list[dict[str, Any]] = []

    @classmethod
    def create(cls, **kwargs: Any) -> SimpleNamespace:
        cls.created.append(kwargs)
        return SimpleNamespace(url="https://checkout.stripe.test/session")


class _Pool:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        if "FROM saas_accounts" in query:
            return {"stripe_customer_id": None, "name": "Acme Co."}
        if "FROM saas_users" in query:
            return {"email": "owner@example.com"}
        raise AssertionError(query)

    async def execute(self, query: str, *args: Any) -> str:
        self.execute_calls.append((query, args))
        return "UPDATE 1"


def _fake_stripe() -> SimpleNamespace:
    _CustomerApi.created = []
    _CheckoutSessionApi.created = []
    return SimpleNamespace(
        api_key="",
        api_version="",
        Customer=_CustomerApi,
        checkout=SimpleNamespace(Session=_CheckoutSessionApi),
    )


def test_configure_stripe_module_pins_api_version_and_warns_on_full_secret_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stripe = _fake_stripe()

    with caplog.at_level(logging.WARNING, logger="atlas.api.billing"):
        configured = billing._configure_stripe_module(stripe, "sk_test_full_secret")

    assert configured is stripe
    assert stripe.api_key == "sk_test_full_secret"
    assert stripe.api_version == billing.STRIPE_API_VERSION
    assert "restricted rk_ key" in caplog.text
    assert "sk_test_full_secret" not in caplog.text

    caplog.clear()
    billing._configure_stripe_module(stripe, "rk_test_restricted")
    assert stripe.api_key == "rk_test_restricted"
    assert stripe.api_version == billing.STRIPE_API_VERSION
    assert "restricted rk_ key" not in caplog.text


def test_stripe_webhook_secret_candidates_trim_and_drop_empty_values() -> None:
    assert billing._stripe_webhook_secret_candidates(
        " whsec_live , ,whsec_test\n, whsec_rotated "
    ) == ("whsec_live", "whsec_test", "whsec_rotated")
    assert billing._stripe_webhook_secret_candidates(" , ") == ()


class _StripeWebhookApi:
    calls: list[tuple[bytes, str, str]] = []
    valid_secret = "whsec_valid"

    @classmethod
    def construct_event(cls, body: bytes, signature: str, secret: str) -> SimpleNamespace:
        cls.calls.append((body, signature, secret))
        if secret != cls.valid_secret:
            raise ValueError(f"bad secret: {secret}")
        return SimpleNamespace(id="evt_valid", type="checkout.session.completed")


def test_construct_stripe_webhook_event_accepts_later_valid_secret() -> None:
    stripe = SimpleNamespace(Webhook=_StripeWebhookApi)
    _StripeWebhookApi.calls = []

    event = billing._construct_stripe_webhook_event(
        stripe,
        b'{"id":"evt_valid"}',
        "t=123,v1=sig",
        "whsec_old, whsec_valid",
    )

    assert event.id == "evt_valid"
    assert _StripeWebhookApi.calls == [
        (b'{"id":"evt_valid"}', "t=123,v1=sig", "whsec_old"),
        (b'{"id":"evt_valid"}', "t=123,v1=sig", "whsec_valid"),
    ]


def test_construct_stripe_webhook_event_rejects_when_all_secrets_fail() -> None:
    stripe = SimpleNamespace(Webhook=_StripeWebhookApi)
    _StripeWebhookApi.calls = []

    with pytest.raises(ValueError, match="bad secret: whsec_other"):
        billing._construct_stripe_webhook_event(
            stripe,
            b'{"id":"evt_bad"}',
            "t=123,v1=bad",
            "whsec_old, whsec_other",
        )

    assert _StripeWebhookApi.calls == [
        (b'{"id":"evt_bad"}', "t=123,v1=bad", "whsec_old"),
        (b'{"id":"evt_bad"}', "t=123,v1=bad", "whsec_other"),
    ]


def test_billing_idempotency_keys_are_stable_and_parameter_scoped() -> None:
    account_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    first = billing._stripe_checkout_idempotency_key(
        account_id=account_id,
        user_id=user_id,
        price_id="price_growth",
        success_url="https://app.example.com/success",
        cancel_url="https://app.example.com/cancel",
    )
    replay = billing._stripe_checkout_idempotency_key(
        account_id=account_id,
        user_id=user_id,
        price_id="price_growth",
        success_url="https://app.example.com/success",
        cancel_url="https://app.example.com/cancel",
    )
    changed_url = billing._stripe_checkout_idempotency_key(
        account_id=account_id,
        user_id=user_id,
        price_id="price_growth",
        success_url="https://app.example.com/success",
        cancel_url="https://app.example.com/other-cancel",
    )

    assert billing._stripe_customer_idempotency_key(account_id) == (
        f"atlas-customer:{account_id}"
    )
    assert first == replay
    assert first != changed_url
    assert first.startswith(f"atlas-checkout:{account_id}:{user_id}:")
    assert "https://" not in first


@pytest.mark.asyncio
async def test_create_checkout_passes_pinned_version_and_idempotency_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    fake_stripe = _fake_stripe()
    pool = _Pool()

    monkeypatch.setitem(sys.modules, "stripe", fake_stripe)
    monkeypatch.setattr(billing.settings.saas_auth, "stripe_secret_key", "rk_test_billing")
    monkeypatch.setattr(
        billing.settings.saas_auth,
        "stripe_growth_price_id",
        "price_growth",
    )
    monkeypatch.setattr(billing, "get_db_pool", lambda: pool)

    response = await billing.create_checkout(
        billing.CheckoutRequest(
            plan="growth",
            success_url="https://app.example.com/success",
            cancel_url="https://app.example.com/cancel",
        ),
        AuthUser(
            user_id=user_id,
            account_id=account_id,
            plan="trial",
            plan_status="trialing",
            role="owner",
        ),
    )

    assert response.checkout_url == "https://checkout.stripe.test/session"
    assert fake_stripe.api_key == "rk_test_billing"
    assert fake_stripe.api_version == billing.STRIPE_API_VERSION
    assert _CustomerApi.created[0]["idempotency_key"] == (
        f"atlas-customer:{account_id}"
    )
    assert _CustomerApi.created[0]["metadata"] == {"account_id": account_id}
    assert pool.execute_calls[0][1] == ("cus_test_created", uuid.UUID(account_id))

    checkout = _CheckoutSessionApi.created[0]
    assert checkout["idempotency_key"] == billing._stripe_checkout_idempotency_key(
        account_id=account_id,
        user_id=user_id,
        price_id="price_growth",
        success_url="https://app.example.com/success",
        cancel_url="https://app.example.com/cancel",
    )
    assert checkout["customer"] == "cus_test_created"
    assert checkout["line_items"] == [{"price": "price_growth", "quantity": 1}]
