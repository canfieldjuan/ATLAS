from __future__ import annotations

import logging
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
        self.processed_event_ids: set[str] = set()
        self.report_rows: dict[tuple[str, str], dict[str, Any]] = {}
        self.delivery_rows: dict[tuple[str, str], dict[str, Any]] = {}

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.fetchval_calls.append((query, args))
        if (
            "FROM billing_events" in query
            and args
            and args[0] in self.processed_event_ids
        ):
            return "billing-event-id"
        return None

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        account_id, request_id = args
        if "FROM content_ops_deflection_reports" in query:
            return self.report_rows.get((str(account_id), str(request_id)))
        raise AssertionError(query)

    async def execute(self, query: str, *args: Any) -> str:
        self.execute_calls.append((query, args))
        if "UPDATE content_ops_deflection_reports" in query:
            account_id, request_id, payment_reference = args
            row = self.report_rows.get((str(account_id), str(request_id)))
            if row is None:
                return "UPDATE 0"
            if "SET paid = false" in query:
                if payment_reference and row.get("payment_reference") not in {
                    None,
                    payment_reference,
                }:
                    return "UPDATE 0"
                row["paid"] = False
                return "UPDATE 1"
            if row is not None:
                row["paid"] = True
                row["payment_reference"] = payment_reference
            return self.update_result
        if "UPDATE content_ops_deflection_report_deliveries" in query:
            account_id, request_id, delivery_error = args
            row = self.delivery_rows.get((str(account_id), str(request_id)))
            if row is not None and row.get("delivery_status") in {"pending", "sending"}:
                row["delivery_status"] = "revoked"
                row["delivery_error"] = delivery_error
            return "UPDATE 1"
        if "INSERT INTO content_ops_deflection_report_deliveries" in query:
            account_id, request_id, payment_reference = args
            self.delivery_rows[(str(account_id), str(request_id))] = {
                "account_id": account_id,
                "request_id": request_id,
                "payment_reference": payment_reference,
                "delivery_status": "pending",
            }
            return "INSERT 0 1"
        if self.fail_billing_event_insert and "INSERT INTO billing_events" in query:
            raise RuntimeError("billing event insert failed")
        if "INSERT INTO billing_events" in query:
            self.processed_event_ids.add(str(args[1]))
        return "INSERT 0 1"

    def add_report(
        self,
        *,
        account_id: str,
        request_id: str = "req-123",
        delivery_email: str | None = "buyer@example.com",
        paid: bool = False,
        payment_reference: str | None = None,
    ) -> None:
        self.report_rows[(account_id, request_id)] = {
            "account_id": account_id,
            "request_id": request_id,
            "snapshot": {},
            "artifact": {},
            "paid": paid,
            "payment_reference": payment_reference,
            "delivery_email": delivery_email,
        }


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


def _payment_event_object(
    *,
    object_id: str = "ch_test_deflection_refund",
    payment_intent: str = "pi_test_deflection",
    metadata: dict[str, str] | None = None,
    refunded: bool | None = None,
    amount_refunded: int | None = None,
    amount_captured: int | None = None,
) -> SimpleNamespace:
    payload = {
        "id": object_id,
        "payment_intent": payment_intent,
        "metadata": dict(metadata or {}),
    }
    if refunded is not None:
        payload["refunded"] = refunded
    if amount_refunded is not None:
        payload["amount_refunded"] = amount_refunded
    if amount_captured is not None:
        payload["amount_captured"] = amount_captured
    return SimpleNamespace(
        id=object_id,
        payment_intent=payment_intent,
        metadata=metadata or {},
        refunded=refunded,
        amount_refunded=amount_refunded,
        amount_captured=amount_captured,
        to_dict=lambda: dict(payload),
    )


class _CheckoutSessionList:
    def __init__(
        self,
        sessions: list[Any] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.sessions = sessions or []
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def list(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(data=list(self.sessions))


def _stripe_module_for_event(
    event: Any,
    *,
    checkout_sessions: list[Any] | None = None,
    checkout_error: Exception | None = None,
) -> tuple[SimpleNamespace, _CheckoutSessionList]:
    session_list = _CheckoutSessionList(checkout_sessions, error=checkout_error)

    class _Webhook:
        @staticmethod
        def construct_event(_body: bytes, _sig: str, _secret: str) -> Any:
            return event

    return (
        SimpleNamespace(
            Webhook=_Webhook,
            checkout=SimpleNamespace(Session=session_list),
            api_key="",
        ),
        session_list,
    )


async def _run_stripe_webhook(
    monkeypatch: pytest.MonkeyPatch,
    *,
    event: Any,
    pool: _Pool,
    stripe_module: Any,
) -> dict[str, Any]:
    request = SimpleNamespace(
        headers={"stripe-signature": "valid"},
        body=lambda: _body(),
    )

    async def _body() -> bytes:
        return b"{}"

    monkeypatch.setitem(sys.modules, "stripe", stripe_module)
    monkeypatch.setattr(billing.settings.saas_auth, "stripe_secret_key", "sk_test")
    monkeypatch.setattr(
        billing.settings.saas_auth,
        "stripe_webhook_secret",
        "whsec_test",
    )
    monkeypatch.setattr(billing, "get_db_pool", lambda: pool)
    return await billing.stripe_webhook(request)


@pytest.mark.asyncio
async def test_deflection_checkout_completion_marks_report_paid() -> None:
    account_id = str(uuid.uuid4())
    session = _session(account_id=account_id)
    pool = _Pool()
    pool.add_report(account_id=account_id)

    returned = await billing._handle_content_ops_deflection_report_checkout_completed(
        pool,
        session,
        session.metadata,
    )

    assert returned is None
    assert len(pool.execute_calls) == 2
    query, args = pool.execute_calls[0]
    assert "UPDATE content_ops_deflection_reports" in query
    assert args == (account_id, "req-123", "cs_test_deflection")
    delivery_query, delivery_args = pool.execute_calls[1]
    assert "INSERT INTO content_ops_deflection_report_deliveries" in delivery_query
    assert "delivery_status IN ('delivered', 'sending')" in delivery_query
    assert delivery_args == (account_id, "req-123", "cs_test_deflection")
    assert "buyer@example.com" not in str(pool.delivery_rows)


@pytest.mark.asyncio
async def test_deflection_checkout_completion_skips_delivery_queue_without_email() -> None:
    account_id = str(uuid.uuid4())
    session = _session(account_id=account_id)
    pool = _Pool()
    pool.add_report(account_id=account_id, delivery_email=None)

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
    assert pool.delivery_rows == {}


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
        ("amount_total", 150001),
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


@pytest.mark.asyncio
async def test_deflection_checkout_completion_accepts_lower_allowlisted_amount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account_id = str(uuid.uuid4())
    session = _session(account_id=account_id, amount_total=120000)
    pool = _Pool()
    pool.add_report(account_id=account_id)
    monkeypatch.setattr(
        billing.settings.saas_auth,
        "stripe_content_ops_deflection_report_allowed_amount_cents",
        "120000,150000,180000",
    )

    returned = await billing._handle_content_ops_deflection_report_checkout_completed(
        pool,
        session,
        session.metadata,
    )

    assert returned is None
    query, args = pool.execute_calls[0]
    assert "UPDATE content_ops_deflection_reports" in query
    assert args == (account_id, "req-123", "cs_test_deflection")


def test_deflection_checkout_amount_requires_exact_default_amount() -> None:
    account_id = str(uuid.uuid4())

    assert (
        billing._deflection_checkout_amount_is_valid(
            _session(account_id=account_id, amount_total=150000)
        )
        is True
    )
    assert (
        billing._deflection_checkout_amount_is_valid(
            _session(account_id=account_id, amount_total=150001)
        )
        is False
    )


def test_deflection_checkout_amount_uses_allowed_amount_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account_id = str(uuid.uuid4())
    monkeypatch.setattr(
        billing.settings.saas_auth,
        "stripe_content_ops_deflection_report_allowed_amount_cents",
        "120000,150000,180000",
    )

    assert (
        billing._deflection_checkout_amount_is_valid(
            _session(account_id=account_id, amount_total=120000)
        )
        is True
    )
    assert (
        billing._deflection_checkout_amount_is_valid(
            _session(account_id=account_id, amount_total=180000)
        )
        is True
    )
    assert (
        billing._deflection_checkout_amount_is_valid(
            _session(account_id=account_id, amount_total=130000)
        )
        is False
    )


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


@pytest.mark.parametrize(
    "allowed_amounts",
    [
        "120000,,150000",
        "120000,not-cents,150000",
        "120000,0,150000",
    ],
)
def test_deflection_checkout_amount_rejects_misconfigured_allowed_amount_gate(
    monkeypatch: pytest.MonkeyPatch,
    allowed_amounts: str,
) -> None:
    session = _session(account_id=str(uuid.uuid4()))
    monkeypatch.setattr(
        billing.settings.saas_auth,
        "stripe_content_ops_deflection_report_allowed_amount_cents",
        allowed_amounts,
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
    pool.add_report(account_id=account_id)
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
    assert fake_stripe.api_version == billing.STRIPE_API_VERSION
    assert pool.fetchval_calls[0][1] == ("evt_deflection_paid",)
    update_query, update_args = pool.execute_calls[0]
    delivery_query, delivery_args = pool.execute_calls[1]
    insert_query, insert_args = pool.execute_calls[2]
    assert "UPDATE content_ops_deflection_reports" in update_query
    assert update_args == (account_id, "req-123", "cs_test_deflection")
    assert "INSERT INTO content_ops_deflection_report_deliveries" in delivery_query
    assert delivery_args == (account_id, "req-123", "cs_test_deflection")
    assert "INSERT INTO billing_events" in insert_query
    assert insert_args[0] is None
    assert insert_args[1] == "evt_deflection_paid"
    assert insert_args[2] == "checkout.session.completed"
    assert pool.processed_event_ids == {"evt_deflection_paid"}


@pytest.mark.asyncio
async def test_stripe_webhook_routes_deflection_async_success_to_paid_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account_id = str(uuid.uuid4())
    session = _session(
        account_id=account_id,
        session_id="cs_test_deflection_async_success",
    )
    event = SimpleNamespace(
        id="evt_deflection_async_success",
        type="checkout.session.async_payment_succeeded",
        data=SimpleNamespace(object=session),
    )

    class _Webhook:
        @staticmethod
        def construct_event(_body: bytes, _sig: str, _secret: str) -> Any:
            return event

    fake_stripe = SimpleNamespace(Webhook=_Webhook, api_key="")
    pool = _Pool()
    pool.add_report(account_id=account_id)
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
    delivery_query, delivery_args = pool.execute_calls[1]
    insert_query, insert_args = pool.execute_calls[2]
    assert "UPDATE content_ops_deflection_reports" in update_query
    assert update_args == (
        account_id,
        "req-123",
        "cs_test_deflection_async_success",
    )
    assert "INSERT INTO content_ops_deflection_report_deliveries" in delivery_query
    assert delivery_args == (
        account_id,
        "req-123",
        "cs_test_deflection_async_success",
    )
    assert "INSERT INTO billing_events" in insert_query
    assert insert_args[1] == "evt_deflection_async_success"
    assert insert_args[2] == "checkout.session.async_payment_succeeded"
    assert pool.processed_event_ids == {"evt_deflection_async_success"}


@pytest.mark.asyncio
async def test_stripe_webhook_deflection_completed_unpaid_stays_pending(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    account_id = str(uuid.uuid4())
    session = _session(account_id=account_id, payment_status="unpaid")
    event = SimpleNamespace(
        id="evt_deflection_pending_payment",
        type="checkout.session.completed",
        data=SimpleNamespace(object=session),
    )

    class _Webhook:
        @staticmethod
        def construct_event(_body: bytes, _sig: str, _secret: str) -> Any:
            return event

    fake_stripe = SimpleNamespace(Webhook=_Webhook, api_key="")
    pool = _Pool()
    pool.add_report(account_id=account_id)
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
    assert "checkout event arrived before funds were available" in caplog.text
    assert len(pool.execute_calls) == 1
    insert_query, insert_args = pool.execute_calls[0]
    assert "INSERT INTO billing_events" in insert_query
    assert insert_args[1] == "evt_deflection_pending_payment"
    assert insert_args[2] == "checkout.session.completed"
    assert pool.processed_event_ids == {"evt_deflection_pending_payment"}
    assert pool.report_rows[(account_id, "req-123")]["paid"] is False
    assert pool.delivery_rows == {}


@pytest.mark.asyncio
async def test_stripe_webhook_deflection_async_success_unpaid_stays_pending(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    account_id = str(uuid.uuid4())
    session = _session(account_id=account_id, payment_status="unpaid")
    event = SimpleNamespace(
        id="evt_deflection_async_success_unpaid",
        type="checkout.session.async_payment_succeeded",
        data=SimpleNamespace(object=session),
    )

    class _Webhook:
        @staticmethod
        def construct_event(_body: bytes, _sig: str, _secret: str) -> Any:
            return event

    fake_stripe = SimpleNamespace(Webhook=_Webhook, api_key="")
    pool = _Pool()
    pool.add_report(account_id=account_id)
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
    assert "checkout event arrived before funds were available" in caplog.text
    assert len(pool.execute_calls) == 1
    insert_query, insert_args = pool.execute_calls[0]
    assert "INSERT INTO billing_events" in insert_query
    assert insert_args[1] == "evt_deflection_async_success_unpaid"
    assert insert_args[2] == "checkout.session.async_payment_succeeded"
    assert pool.processed_event_ids == {"evt_deflection_async_success_unpaid"}
    assert pool.report_rows[(account_id, "req-123")]["paid"] is False
    assert pool.delivery_rows == {}


@pytest.mark.asyncio
async def test_stripe_webhook_deflection_async_failure_is_observed_without_unlock(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    account_id = str(uuid.uuid4())
    session = _session(
        account_id=account_id,
        session_id="cs_test_deflection_async_failed",
        payment_status="failed",
    )
    event = SimpleNamespace(
        id="evt_deflection_async_failed",
        type="checkout.session.async_payment_failed",
        data=SimpleNamespace(object=session),
    )

    class _Webhook:
        @staticmethod
        def construct_event(_body: bytes, _sig: str, _secret: str) -> Any:
            return event

    fake_stripe = SimpleNamespace(Webhook=_Webhook, api_key="")
    pool = _Pool()
    pool.add_report(account_id=account_id)
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
    assert "async payment failed" in caplog.text
    assert len(pool.execute_calls) == 1
    insert_query, insert_args = pool.execute_calls[0]
    assert "INSERT INTO billing_events" in insert_query
    assert insert_args[1] == "evt_deflection_async_failed"
    assert insert_args[2] == "checkout.session.async_payment_failed"
    assert pool.report_rows[(account_id, "req-123")]["paid"] is False
    assert pool.delivery_rows == {}


@pytest.mark.asyncio
async def test_stripe_webhook_refund_relocks_paid_deflection_report_via_checkout_lookup(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    account_id = str(uuid.uuid4())
    checkout_session = _session(
        account_id=account_id,
        session_id="cs_test_deflection_refunded",
    )
    charge = _payment_event_object(
        payment_intent="pi_test_refunded",
        refunded=True,
        amount_refunded=150000,
        amount_captured=150000,
    )
    event = SimpleNamespace(
        id="evt_deflection_refunded",
        type="charge.refunded",
        data=SimpleNamespace(object=charge),
    )
    fake_stripe, session_list = _stripe_module_for_event(
        event,
        checkout_sessions=[checkout_session],
    )
    pool = _Pool()
    pool.add_report(
        account_id=account_id,
        paid=True,
        payment_reference="cs_test_deflection_refunded",
    )
    pool.delivery_rows[(account_id, "req-123")] = {
        "account_id": account_id,
        "request_id": "req-123",
        "payment_reference": "cs_test_deflection_refunded",
        "delivery_status": "pending",
    }

    response = await _run_stripe_webhook(
        monkeypatch,
        event=event,
        pool=pool,
        stripe_module=fake_stripe,
    )

    assert response == {"status": "ok"}
    assert session_list.calls == [
        {"payment_intent": "pi_test_refunded", "limit": 1, "timeout": 10}
    ]
    update_calls = [
        call for call in pool.execute_calls if "UPDATE content_ops_deflection_reports" in call[0]
    ]
    delivery_update_calls = [
        call
        for call in pool.execute_calls
        if "UPDATE content_ops_deflection_report_deliveries" in call[0]
    ]
    billing_event_calls = [
        call for call in pool.execute_calls if "INSERT INTO billing_events" in call[0]
    ]
    assert update_calls[0][1] == (
        account_id,
        "req-123",
        "cs_test_deflection_refunded",
    )
    assert "SET paid = false" in update_calls[0][0]
    assert delivery_update_calls[0][1] == (
        account_id,
        "req-123",
        "payment_revoked:charge.refunded",
    )
    assert billing_event_calls[0][1][1] == "evt_deflection_refunded"
    assert billing_event_calls[0][1][2] == "charge.refunded"
    assert pool.report_rows[(account_id, "req-123")]["paid"] is False
    assert pool.delivery_rows[(account_id, "req-123")]["delivery_status"] == "revoked"
    assert "access revoked after Stripe payment reversal" in caplog.text


@pytest.mark.asyncio
async def test_stripe_webhook_partial_refund_keeps_deflection_report_paid(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="atlas.api.billing")
    account_id = str(uuid.uuid4())
    checkout_session = _session(account_id=account_id)
    charge = _payment_event_object(
        payment_intent="pi_test_partial_refund",
        refunded=False,
        amount_refunded=50000,
        amount_captured=150000,
    )
    event = SimpleNamespace(
        id="evt_deflection_partial_refund",
        type="charge.refunded",
        data=SimpleNamespace(object=charge),
    )
    fake_stripe, session_list = _stripe_module_for_event(
        event,
        checkout_sessions=[checkout_session],
    )
    pool = _Pool()
    pool.add_report(account_id=account_id, paid=True, payment_reference="cs_test")

    response = await _run_stripe_webhook(
        monkeypatch,
        event=event,
        pool=pool,
        stripe_module=fake_stripe,
    )

    assert response == {"status": "ok"}
    assert session_list.calls == []
    assert pool.report_rows[(account_id, "req-123")]["paid"] is True
    update_calls = [
        call
        for call in pool.execute_calls
        if "UPDATE content_ops_deflection_reports" in call[0]
    ]
    assert update_calls == []
    billing_event_calls = [
        call for call in pool.execute_calls if "INSERT INTO billing_events" in call[0]
    ]
    assert billing_event_calls[0][1][1] == "evt_deflection_partial_refund"
    assert "partial refund observed without revocation" in caplog.text


@pytest.mark.asyncio
async def test_stripe_webhook_refund_lookup_failure_retries_without_unlocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account_id = str(uuid.uuid4())
    charge = _payment_event_object(
        payment_intent="pi_lookup_down",
        refunded=True,
        amount_refunded=150000,
        amount_captured=150000,
    )
    event = SimpleNamespace(
        id="evt_deflection_refund_lookup_down",
        type="charge.refunded",
        data=SimpleNamespace(object=charge),
    )
    fake_stripe, session_list = _stripe_module_for_event(
        event,
        checkout_error=RuntimeError("stripe down"),
    )
    pool = _Pool()
    pool.add_report(account_id=account_id, paid=True, payment_reference="cs_test")

    with pytest.raises(billing.HTTPException) as exc:
        await _run_stripe_webhook(
            monkeypatch,
            event=event,
            pool=pool,
            stripe_module=fake_stripe,
        )

    assert exc.value.status_code == 503
    assert session_list.calls == [
        {"payment_intent": "pi_lookup_down", "limit": 1, "timeout": 10}
    ]
    assert pool.report_rows[(account_id, "req-123")]["paid"] is True
    assert pool.execute_calls == []
    assert pool.processed_event_ids == set()


@pytest.mark.asyncio
async def test_stripe_webhook_dispute_relocks_paid_deflection_report_from_direct_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account_id = str(uuid.uuid4())
    dispute = _payment_event_object(
        object_id="du_test_deflection",
        metadata={
            "source": "content_ops_deflection_report",
            "account_id": account_id,
            "request_id": "req-123",
        },
    )
    event = SimpleNamespace(
        id="evt_deflection_dispute",
        type="charge.dispute.created",
        data=SimpleNamespace(object=dispute),
    )
    fake_stripe, session_list = _stripe_module_for_event(event, checkout_sessions=[])
    pool = _Pool()
    pool.add_report(account_id=account_id, paid=True, payment_reference=None)

    response = await _run_stripe_webhook(
        monkeypatch,
        event=event,
        pool=pool,
        stripe_module=fake_stripe,
    )

    assert response == {"status": "ok"}
    assert session_list.calls == [
        {"payment_intent": "pi_test_deflection", "limit": 1, "timeout": 10}
    ]
    update_calls = [
        call for call in pool.execute_calls if "UPDATE content_ops_deflection_reports" in call[0]
    ]
    assert update_calls[0][1] == (account_id, "req-123", None)
    assert "SET paid = false" in update_calls[0][0]
    assert pool.report_rows[(account_id, "req-123")]["paid"] is False
    assert pool.delivery_rows == {}


@pytest.mark.asyncio
async def test_stripe_webhook_unmapped_refund_is_observed_without_mutating_reports(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="atlas.api.billing")
    account_id = str(uuid.uuid4())
    charge = _payment_event_object(
        payment_intent="pi_missing_metadata",
        refunded=True,
        amount_refunded=150000,
        amount_captured=150000,
    )
    event = SimpleNamespace(
        id="evt_deflection_refund_unmapped",
        type="charge.refunded",
        data=SimpleNamespace(object=charge),
    )
    fake_stripe, _session_list = _stripe_module_for_event(event, checkout_sessions=[])
    pool = _Pool()
    pool.add_report(
        account_id=account_id,
        paid=True,
        payment_reference="cs_unrelated",
    )

    response = await _run_stripe_webhook(
        monkeypatch,
        event=event,
        pool=pool,
        stripe_module=fake_stripe,
    )

    assert response == {"status": "ok"}
    assert "could not be mapped" in caplog.text
    assert pool.report_rows[(account_id, "req-123")]["paid"] is True
    assert [
        call for call in pool.execute_calls if "UPDATE content_ops_deflection_reports" in call[0]
    ] == []
    assert [
        call
        for call in pool.execute_calls
        if "UPDATE content_ops_deflection_report_deliveries" in call[0]
    ] == []
    billing_event_calls = [
        call for call in pool.execute_calls if "INSERT INTO billing_events" in call[0]
    ]
    assert billing_event_calls[0][1][1] == "evt_deflection_refund_unmapped"
    assert billing_event_calls[0][1][2] == "charge.refunded"


@pytest.mark.asyncio
async def test_stripe_webhook_skips_processed_refund_before_revocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account_id = str(uuid.uuid4())
    checkout_session = _session(account_id=account_id)
    charge = _payment_event_object()
    event = SimpleNamespace(
        id="evt_deflection_refund_duplicate",
        type="charge.refunded",
        data=SimpleNamespace(object=charge),
    )
    fake_stripe, session_list = _stripe_module_for_event(
        event,
        checkout_sessions=[checkout_session],
    )
    pool = _Pool()
    pool.add_report(
        account_id=account_id,
        paid=True,
        payment_reference="cs_test_deflection",
    )
    pool.processed_event_ids.add("evt_deflection_refund_duplicate")

    response = await _run_stripe_webhook(
        monkeypatch,
        event=event,
        pool=pool,
        stripe_module=fake_stripe,
    )

    assert response == {"status": "already_processed"}
    assert session_list.calls == []
    assert pool.report_rows[(account_id, "req-123")]["paid"] is True
    assert pool.execute_calls == []


@pytest.mark.asyncio
async def test_stripe_webhook_skips_processed_deflection_checkout_before_paid_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account_id = str(uuid.uuid4())
    session = _session(account_id=account_id)
    event = SimpleNamespace(
        id="evt_deflection_paid_duplicate",
        type="checkout.session.completed",
        data=SimpleNamespace(object=session),
    )

    class _Webhook:
        @staticmethod
        def construct_event(_body: bytes, _sig: str, _secret: str) -> Any:
            return event

    fake_stripe = SimpleNamespace(Webhook=_Webhook, api_key="")
    pool = _Pool()
    pool.processed_event_ids.add("evt_deflection_paid_duplicate")
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

    assert response == {"status": "already_processed"}
    assert pool.fetchval_calls[0][1] == ("evt_deflection_paid_duplicate",)
    assert pool.execute_calls == []


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
    pool.add_report(account_id=account_id)
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
    delivery_query, delivery_args = pool.execute_calls[1]
    insert_query, insert_args = pool.execute_calls[2]
    assert "UPDATE content_ops_deflection_reports" in update_query
    assert update_args == (account_id, "req-123", "cs_test_deflection")
    assert "INSERT INTO content_ops_deflection_report_deliveries" in delivery_query
    assert delivery_args == (account_id, "req-123", "cs_test_deflection")
    assert "INSERT INTO billing_events" in insert_query
    assert insert_args[1] == "evt_deflection_paid_audit_failure"
    assert "billing_events audit insert failed" in caplog.text
    assert pool.processed_event_ids == set()


@pytest.mark.asyncio
async def test_stripe_webhook_retry_after_audit_failure_restores_idempotency(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    account_id = str(uuid.uuid4())
    session = _session(account_id=account_id)
    event = SimpleNamespace(
        id="evt_deflection_paid_retry",
        type="checkout.session.completed",
        data=SimpleNamespace(object=session),
    )

    class _Webhook:
        @staticmethod
        def construct_event(_body: bytes, _sig: str, _secret: str) -> Any:
            return event

    fake_stripe = SimpleNamespace(Webhook=_Webhook, api_key="")
    pool = _Pool()
    pool.add_report(account_id=account_id)
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

    assert await billing.stripe_webhook(request) == {"status": "ok"}
    assert pool.processed_event_ids == set()
    assert "billing_events audit insert failed" in caplog.text

    pool.fail_billing_event_insert = False
    assert await billing.stripe_webhook(request) == {"status": "ok"}
    assert pool.processed_event_ids == {"evt_deflection_paid_retry"}

    assert await billing.stripe_webhook(request) == {"status": "already_processed"}

    update_calls = [
        call
        for call in pool.execute_calls
        if "UPDATE content_ops_deflection_reports" in call[0]
    ]
    insert_calls = [
        call for call in pool.execute_calls if "INSERT INTO billing_events" in call[0]
    ]
    assert len(update_calls) == 2
    assert all(
        call[1] == (account_id, "req-123", "cs_test_deflection")
        for call in update_calls
    )
    delivery_calls = [
        call
        for call in pool.execute_calls
        if "INSERT INTO content_ops_deflection_report_deliveries" in call[0]
    ]
    assert len(delivery_calls) == 2
    assert len(insert_calls) == 2


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
