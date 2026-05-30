from __future__ import annotations

import sys
import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from atlas_brain.api import billing


class _Pool:
    def __init__(self) -> None:
        self.is_initialized = True
        self.fetchval_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.update_result = "UPDATE 1"
        self.fail_billing_event_insert = False

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.fetchval_calls.append((query, args))
        return None

    async def execute(self, query: str, *args: Any) -> str:
        self.execute_calls.append((query, args))
        if "UPDATE content_ops_deflection_reports" in query:
            return self.update_result
        if self.fail_billing_event_insert and "INSERT INTO billing_events" in query:
            raise RuntimeError("billing event insert failed")
        return "INSERT 0 1"


def _session(
    *,
    account_id: str | None = None,
    request_id: str | None = "req-123",
    session_id: str = "cs_test_deflection",
    mode: str = "payment",
    payment_status: str = "paid",
    amount_total: int | str | None = 150000,
    currency: str = "usd",
) -> SimpleNamespace:
    metadata = {"source": "content_ops_deflection_report"}
    if account_id is not None:
        metadata["account_id"] = account_id
    if request_id is not None:
        metadata["request_id"] = request_id
    return SimpleNamespace(
        id=session_id,
        mode=mode,
        payment_status=payment_status,
        amount_total=amount_total,
        currency=currency,
        metadata=metadata,
        to_dict=lambda: {
            "id": session_id,
            "mode": mode,
            "payment_status": payment_status,
            "amount_total": amount_total,
            "currency": currency,
            "metadata": dict(metadata),
        },
    )


@pytest.mark.asyncio
async def test_deflection_checkout_completion_marks_report_paid() -> None:
    account_id = str(uuid.uuid4())
    session = _session(account_id=account_id)
    pool = _Pool()

    returned = await billing._handle_content_ops_deflection_report_checkout_completed(
        pool,
        session,
        session.metadata,
    )

    assert returned is None
    assert len(pool.execute_calls) == 1
    query, args = pool.execute_calls[0]
    assert "UPDATE content_ops_deflection_reports" in query
    assert args == (account_id, "req-123", "cs_test_deflection")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", "subscription"),
        ("payment_status", "unpaid"),
        ("account_id", None),
        ("account_id", "not-a-uuid"),
        ("request_id", None),
        ("amount_total", 149999),
        ("amount_total", None),
        ("currency", "eur"),
    ],
)
async def test_deflection_checkout_completion_fails_closed_for_invalid_sessions(
    field: str,
    value: Any,
) -> None:
    kwargs = {"account_id": str(uuid.uuid4())}
    kwargs[field] = value
    session = _session(**kwargs)
    pool = _Pool()

    returned = await billing._handle_content_ops_deflection_report_checkout_completed(
        pool,
        session,
        session.metadata,
    )

    assert returned is None
    assert pool.execute_calls == []


@pytest.mark.asyncio
async def test_deflection_checkout_completion_fails_closed_when_report_missing() -> None:
    account_id = str(uuid.uuid4())
    session = _session(account_id=account_id)
    pool = _Pool()
    pool.update_result = "UPDATE 0"

    with pytest.raises(billing.HTTPException) as exc:
        await billing._handle_content_ops_deflection_report_checkout_completed(
            pool,
            session,
            session.metadata,
        )

    assert exc.value.status_code == 409
    assert len(pool.execute_calls) == 1
    query, args = pool.execute_calls[0]
    assert "UPDATE content_ops_deflection_reports" in query
    assert args == (account_id, "req-123", "cs_test_deflection")


@pytest.mark.parametrize(
    ("amount_cents", "currency"),
    [
        (0, "usd"),
        (-1, "usd"),
        (150000, ""),
    ],
)
def test_deflection_checkout_amount_rejects_misconfigured_price_gate(
    monkeypatch: pytest.MonkeyPatch,
    amount_cents: int,
    currency: str,
) -> None:
    session = _session(account_id=str(uuid.uuid4()))
    monkeypatch.setattr(
        billing.settings.saas_auth,
        "stripe_content_ops_deflection_report_amount_cents",
        amount_cents,
    )
    monkeypatch.setattr(
        billing.settings.saas_auth,
        "stripe_content_ops_deflection_report_currency",
        currency,
    )

    assert billing._deflection_checkout_amount_is_valid(session) is False


@pytest.mark.asyncio
async def test_stripe_webhook_routes_deflection_checkout_to_paid_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account_id = str(uuid.uuid4())
    session = _session(account_id=account_id)
    event = SimpleNamespace(
        id="evt_deflection_paid",
        type="checkout.session.completed",
        data=SimpleNamespace(object=session),
    )

    class _Webhook:
        @staticmethod
        def construct_event(_body: bytes, _sig: str, _secret: str) -> Any:
            return event

    fake_stripe = SimpleNamespace(Webhook=_Webhook, api_key="")
    pool = _Pool()
    request = SimpleNamespace(
        headers={"stripe-signature": "valid"},
        body=lambda: _body(),
    )

    async def _body() -> bytes:
        return b"{}"

    monkeypatch.setitem(sys.modules, "stripe", fake_stripe)
    monkeypatch.setattr(billing.settings.saas_auth, "stripe_secret_key", "sk_test")
    monkeypatch.setattr(
        billing.settings.saas_auth,
        "stripe_webhook_secret",
        "whsec_test",
    )
    monkeypatch.setattr(billing, "get_db_pool", lambda: pool)

    response = await billing.stripe_webhook(request)

    assert response == {"status": "ok"}
    assert fake_stripe.api_key == "sk_test"
    assert pool.fetchval_calls[0][1] == ("evt_deflection_paid",)
    update_query, update_args = pool.execute_calls[0]
    insert_query, insert_args = pool.execute_calls[1]
    assert "UPDATE content_ops_deflection_reports" in update_query
    assert update_args == (account_id, "req-123", "cs_test_deflection")
    assert "INSERT INTO billing_events" in insert_query
    assert insert_args[0] is None
    assert insert_args[1] == "evt_deflection_paid"
    assert insert_args[2] == "checkout.session.completed"


@pytest.mark.asyncio
async def test_stripe_webhook_keeps_paid_unlock_when_audit_insert_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    account_id = str(uuid.uuid4())
    session = _session(account_id=account_id)
    event = SimpleNamespace(
        id="evt_deflection_paid_audit_failure",
        type="checkout.session.completed",
        data=SimpleNamespace(object=session),
    )

    class _Webhook:
        @staticmethod
        def construct_event(_body: bytes, _sig: str, _secret: str) -> Any:
            return event

    fake_stripe = SimpleNamespace(Webhook=_Webhook, api_key="")
    pool = _Pool()
    pool.fail_billing_event_insert = True
    request = SimpleNamespace(
        headers={"stripe-signature": "valid"},
        body=lambda: _body(),
    )

    async def _body() -> bytes:
        return b"{}"

    monkeypatch.setitem(sys.modules, "stripe", fake_stripe)
    monkeypatch.setattr(billing.settings.saas_auth, "stripe_secret_key", "sk_test")
    monkeypatch.setattr(
        billing.settings.saas_auth,
        "stripe_webhook_secret",
        "whsec_test",
    )
    monkeypatch.setattr(billing, "get_db_pool", lambda: pool)

    response = await billing.stripe_webhook(request)

    assert response == {"status": "ok"}
    update_query, update_args = pool.execute_calls[0]
    insert_query, insert_args = pool.execute_calls[1]
    assert "UPDATE content_ops_deflection_reports" in update_query
    assert update_args == (account_id, "req-123", "cs_test_deflection")
    assert "INSERT INTO billing_events" in insert_query
    assert insert_args[1] == "evt_deflection_paid_audit_failure"
    assert "billing_events audit insert failed" in caplog.text


@pytest.mark.asyncio
async def test_stripe_webhook_does_not_log_processed_event_when_report_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account_id = str(uuid.uuid4())
    session = _session(account_id=account_id)
    event = SimpleNamespace(
        id="evt_deflection_missing_report",
        type="checkout.session.completed",
        data=SimpleNamespace(object=session),
    )

    class _Webhook:
        @staticmethod
        def construct_event(_body: bytes, _sig: str, _secret: str) -> Any:
            return event

    fake_stripe = SimpleNamespace(Webhook=_Webhook, api_key="")
    pool = _Pool()
    pool.update_result = "UPDATE 0"
    request = SimpleNamespace(
        headers={"stripe-signature": "valid"},
        body=lambda: _body(),
    )

    async def _body() -> bytes:
        return b"{}"

    monkeypatch.setitem(sys.modules, "stripe", fake_stripe)
    monkeypatch.setattr(billing.settings.saas_auth, "stripe_secret_key", "sk_test")
    monkeypatch.setattr(
        billing.settings.saas_auth,
        "stripe_webhook_secret",
        "whsec_test",
    )
    monkeypatch.setattr(billing, "get_db_pool", lambda: pool)

    with pytest.raises(billing.HTTPException) as exc:
        await billing.stripe_webhook(request)

    assert exc.value.status_code == 409
    assert len(pool.execute_calls) == 1
    update_query, update_args = pool.execute_calls[0]
    assert "UPDATE content_ops_deflection_reports" in update_query
    assert update_args == (account_id, "req-123", "cs_test_deflection")
