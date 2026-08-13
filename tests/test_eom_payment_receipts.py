"""HTTP and canonical-CRM boundary tests for residential payment receipts."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

from atlas_brain.eom_api import auth as receivables_auth
from atlas_brain.eom_api import receivables as routes
from atlas_brain.services.crm_provider import DatabaseCRMProvider
from atlas_brain.services.receivables import ReceivablesReceiptContextRequiredError
from atlas_brain.storage.exceptions import DatabaseUnavailableError


class _CanonicalPool:
    def __init__(self, *, initialized: bool) -> None:
        self.is_initialized = initialized


class _CRM:
    def __init__(self, customer: dict | None) -> None:
        self.customer = customer
        self.calls: list[UUID] = []

    async def get_eom_payment_customer(self, contact_id: UUID) -> dict | None:
        self.calls.append(contact_id)
        return self.customer


class _PaymentService:
    def __init__(self, *, replay_payment: dict | None = None) -> None:
        self.replay_payment = replay_payment
        self.calls: list[dict] = []
        self.ledger_writes = 0

    async def create_payment(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["receipt_recipient"] is None:
            if self.replay_payment is not None:
                return self.replay_payment
            raise ReceivablesReceiptContextRequiredError(
                "Canonical customer data is required for a new EOM payment"
            )
        self.ledger_writes += 1
        recipient_email = kwargs["receipt_recipient"].recipient_email
        return {
            "id": "payment-1",
            "status": "received",
            "receipt_delivery": {
                "receipt_number": "EOM-RCP-payment-1",
                "recipient_email": recipient_email,
                "status": "pending" if recipient_email else "skipped",
                "skip_reason": None if recipient_email else "no_email",
            },
        }


def _app(crm: _CRM, service: _PaymentService) -> tuple[FastAPI, str]:
    generated = receivables_auth.generate_receivables_service_token()
    app = FastAPI()
    app.include_router(routes.router)
    app.state.eom_funnel_crm_provider = lambda: crm
    app.dependency_overrides[receivables_auth.get_receivables_api_config] = (
        lambda: SimpleNamespace(
            receivables_api_enabled=True,
            receivables_service_token="",
            receivables_service_token_sha256=generated.sha256,
        )
    )
    app.dependency_overrides[routes.get_receivables_service] = lambda: service
    return app, generated.token


def _headers(token: str, *, key: str = "payment-receipt-http-1") -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "X-EOM-Actor": "Juan Canfield",
        "Idempotency-Key": key,
    }


def _body(contact_id: UUID) -> dict:
    return {
        "contact_id": str(contact_id),
        "payer_name": "Riley Customer",
        "total_amount_cents": 12_500,
        "payment_method": "check",
        "received_date": "2026-08-12",
        "reference": "1042",
    }


@pytest.mark.asyncio
async def test_full_payment_route_uses_canonical_residential_snapshot(monkeypatch):
    from atlas_brain.api.invoicing import receivables as full_routes

    contact_id = uuid4()
    crm = _CRM(
        {
            "contact_id": contact_id,
            "customer_name": "Riley Customer",
            "customer_type": "residential",
            "recipient_email": "riley@example.test",
        }
    )
    service = _PaymentService()
    monkeypatch.setattr(full_routes, "get_crm_provider", lambda: crm)

    result = await full_routes.create_payment(
        full_routes.CreatePaymentRequest.model_validate(_body(contact_id)),
        actor="Juan Canfield",
        idempotency_key="full-payment-receipt-http-1",
        service=service,
    )

    assert result["receipt_delivery"]["recipient_email"] == "riley@example.test"
    assert crm.calls == [contact_id]
    assert service.ledger_writes == 1
    assert service.calls[0]["require_receipt_recipient"] is True
    assert service.calls[0]["receipt_recipient"].customer_type == "residential"


@pytest.mark.asyncio
async def test_full_payment_route_refuses_new_missing_customer_but_recovers_replay(
    monkeypatch,
):
    from fastapi import HTTPException

    from atlas_brain.api.invoicing import receivables as full_routes

    contact_id = uuid4()
    body = full_routes.CreatePaymentRequest.model_validate(_body(contact_id))
    missing_service = _PaymentService()
    monkeypatch.setattr(full_routes, "get_crm_provider", lambda: _CRM(None))

    with pytest.raises(HTTPException) as excinfo:
        await full_routes.create_payment(
            body,
            actor="Juan Canfield",
            idempotency_key="full-payment-receipt-missing-1",
            service=missing_service,
        )
    assert excinfo.value.status_code == 404
    assert missing_service.ledger_writes == 0

    class _UnavailableCRM:
        async def get_eom_payment_customer(self, _contact_id):
            raise DatabaseUnavailableError("canonical CRM")

    replay_service = _PaymentService(
        replay_payment={"id": "existing-full-payment", "status": "received"}
    )
    monkeypatch.setattr(full_routes, "get_crm_provider", lambda: _UnavailableCRM())
    replay = await full_routes.create_payment(
        body,
        actor="Juan Canfield",
        idempotency_key="full-payment-receipt-replay-1",
        service=replay_service,
    )
    assert replay == {"id": "existing-full-payment", "status": "received"}
    assert replay_service.ledger_writes == 0
    assert replay_service.calls[0]["receipt_recipient"] is None
    assert replay_service.calls[0]["require_receipt_recipient"] is True


@pytest.mark.asyncio
async def test_slim_payment_route_passes_only_canonical_residential_snapshot(
    monkeypatch,
):
    contact_id = uuid4()
    crm = _CRM(
        {
            "contact_id": contact_id,
            "customer_name": "Riley Customer",
            "customer_type": "residential",
            "recipient_email": "riley@example.test",
        }
    )
    service = _PaymentService()
    app, token = _app(crm, service)
    monkeypatch.setattr(
        routes, "get_eom_funnel_db_pool", lambda: _CanonicalPool(initialized=True)
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://receivables.test"
    ) as client:
        response = await client.post(
            "/receivables/payments",
            headers=_headers(token),
            json=_body(contact_id),
        )

    assert response.status_code == 201, response.text
    assert response.json()["receipt_delivery"]["recipient_email"] == "riley@example.test"
    assert crm.calls == [contact_id]
    assert service.ledger_writes == 1
    [call] = service.calls
    snapshot = call["receipt_recipient"]
    assert snapshot.contact_id == contact_id
    assert snapshot.customer_type == "residential"
    assert snapshot.recipient_email == "riley@example.test"
    assert call["require_receipt_recipient"] is True
    assert call["recorded_by"] == "Juan Canfield"
    assert "customer_type" not in _body(contact_id)


@pytest.mark.asyncio
async def test_slim_payment_route_refuses_new_payment_without_canonical_customer(
    monkeypatch,
):
    contact_id = uuid4()
    crm = _CRM(None)
    service = _PaymentService()
    app, token = _app(crm, service)
    monkeypatch.setattr(
        routes, "get_eom_funnel_db_pool", lambda: _CanonicalPool(initialized=True)
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://receivables.test"
    ) as client:
        response = await client.post(
            "/receivables/payments",
            headers=_headers(token),
            json=_body(contact_id),
        )

    assert response.status_code == 404, response.text
    assert response.json()["detail"]["code"] == "not_found"
    assert crm.calls == [contact_id]
    assert service.ledger_writes == 0
    assert service.calls[0]["receipt_recipient"] is None
    assert service.calls[0]["require_receipt_recipient"] is True


@pytest.mark.asyncio
async def test_slim_payment_route_allows_residential_customer_without_email(
    monkeypatch,
):
    contact_id = uuid4()
    crm = _CRM(
        {
            "contact_id": contact_id,
            "customer_name": "No Email Customer",
            "customer_type": "residential",
            "recipient_email": None,
        }
    )
    service = _PaymentService()
    app, token = _app(crm, service)
    monkeypatch.setattr(
        routes, "get_eom_funnel_db_pool", lambda: _CanonicalPool(initialized=True)
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://receivables.test"
    ) as client:
        response = await client.post(
            "/receivables/payments",
            headers=_headers(token),
            json=_body(contact_id),
        )

    assert response.status_code == 201, response.text
    assert response.json()["receipt_delivery"] == {
        "receipt_number": "EOM-RCP-payment-1",
        "recipient_email": None,
        "status": "skipped",
        "skip_reason": "no_email",
    }
    assert service.ledger_writes == 1
    assert service.calls[0]["receipt_recipient"].recipient_email is None


@pytest.mark.asyncio
async def test_slim_payment_retry_recovers_original_when_canonical_pool_is_unavailable(
    monkeypatch,
):
    contact_id = uuid4()
    crm = _CRM(None)
    service = _PaymentService(
        replay_payment={
            "id": "existing-payment",
            "status": "received",
            "receipt_delivery": {
                "receipt_number": "EOM-RCP-existing-payment",
                "recipient_email": "riley@example.test",
                "status": "pending",
                "skip_reason": None,
            },
        }
    )
    app, token = _app(crm, service)
    monkeypatch.setattr(
        routes, "get_eom_funnel_db_pool", lambda: _CanonicalPool(initialized=False)
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://receivables.test"
    ) as client:
        response = await client.post(
            "/receivables/payments",
            headers=_headers(token, key="existing-payment-key"),
            json=_body(contact_id),
        )

    assert response.status_code == 201, response.text
    assert response.json()["id"] == "existing-payment"
    assert crm.calls == []
    assert service.ledger_writes == 0
    assert service.calls[0]["receipt_recipient"] is None
    assert service.calls[0]["require_receipt_recipient"] is True


@pytest.mark.asyncio
async def test_slim_payment_route_returns_controlled_unavailable_for_new_payment(
    monkeypatch,
):
    contact_id = uuid4()
    crm = _CRM(None)
    service = _PaymentService()
    app, token = _app(crm, service)
    monkeypatch.setattr(
        routes, "get_eom_funnel_db_pool", lambda: _CanonicalPool(initialized=False)
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://receivables.test"
    ) as client:
        response = await client.post(
            "/receivables/payments",
            headers=_headers(token),
            json=_body(contact_id),
        )

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["code"] == "billing_recipients_unavailable"
    assert service.ledger_writes == 0


@pytest.mark.asyncio
async def test_canonical_payment_customer_is_tenant_scoped_and_normalizes_email():
    contact_id = uuid4()

    class _Pool:
        def __init__(self) -> None:
            self.query = ""
            self.args = ()

        async def fetchrow(self, query, *args):
            self.query = query
            self.args = args
            return {
                "id": contact_id,
                "full_name": "  Riley Customer  ",
                "customer_type": "residential",
                "email": "  RILEY@Example.TEST  ",
            }

    pool = _Pool()
    customer = await DatabaseCRMProvider(pool=pool).get_eom_payment_customer(
        contact_id
    )

    assert customer == {
        "contact_id": contact_id,
        "customer_name": "Riley Customer",
        "customer_type": "residential",
        "recipient_email": "riley@example.test",
    }
    assert "business_context_id = $2" in pool.query
    assert "status = $4" in pool.query
    assert "contact_type = $5" in pool.query
    assert pool.args[0] == contact_id
    assert pool.args[1] == "effingham_maids"


@pytest.mark.asyncio
async def test_billing_recipient_readiness_requires_customer_type_column():
    class _Pool:
        def __init__(self) -> None:
            self.query = ""

        async def fetch(self, query):
            self.query = query
            return []

    pool = _Pool()
    assert await DatabaseCRMProvider(pool=pool).billing_recipients_schema_ready()
    assert "customer_type" in pool.query
