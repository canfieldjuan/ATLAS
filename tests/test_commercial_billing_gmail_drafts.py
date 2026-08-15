"""Contract tests for durable, no-send EOM commercial Gmail invoice drafts."""

from __future__ import annotations

import asyncio
import ast
import base64
import hashlib
import inspect
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from email import message_from_bytes
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from atlas_brain.services.commercial_billing_invoice_gmail_drafts import (
    CommercialBillingGmailDraftConflictError,
    CommercialBillingGmailDraftRecoveryRequiredError,
    CommercialBillingGmailDraftUnavailableError,
    CommercialBillingGmailDraftValidationError,
    CommercialBillingInvoiceGmailDraftService,
    _request_text,
)
from atlas_brain.services.commercial_billing_invoice_pdfs import (
    CommercialBillingInvoicePDFService,
)
from atlas_brain.tools.gmail import (
    GmailDraftCreateError,
    GmailDraftLookupError,
    GmailTransport,
)


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _idempotency_key_oracle(value: object) -> bool:
    return isinstance(value, str) and 1 <= len(value.strip()) <= 128


def _idempotency_key_grammar_candidates():
    for token, wrapper, padding in product(
        ("", "a", "a" * 128, "a" * 129, " "),
        (
            lambda value: value,
            lambda value: [value],
            lambda value: {"key": value},
            lambda value: (value,),
        ),
        (lambda value: value, lambda value: f" {value} ", lambda value: f"\t{value}\n"),
    ):
        yield wrapper(padding(token))


@pytest.mark.parametrize("value", tuple(_idempotency_key_grammar_candidates()))
def test_gmail_draft_idempotency_key_matches_the_spec_derived_oracle(value: object):
    if _idempotency_key_oracle(value):
        assert _request_text(value, "Idempotency key", limit=128) == value.strip()
    else:
        with pytest.raises(CommercialBillingGmailDraftValidationError):
            _request_text(value, "Idempotency key", limit=128)


class _SchemaPool:
    is_initialized = True

    def __init__(self, connection, schema: str) -> None:
        self.connection = connection
        self.schema = schema

    @asynccontextmanager
    async def transaction(self):
        async with self.connection.transaction():
            await self.connection.execute(f'SET LOCAL search_path TO "{self.schema}"')
            yield self.connection


class _PDFRenderer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, invoice: dict) -> bytes:
        self.calls.append(str(invoice["invoice_number"]))
        return b"%PDF-1.7\nsynthetic commercial billing artifact\n%%EOF\n"


class _RecordingGateway:
    def __init__(self) -> None:
        self.create_calls: list[dict] = []
        self.lookup_calls: list[str] = []
        self.drafts: dict[str, dict] = {}
        self.create_error: Exception | None = None
        self.create_then_raise = False

    async def create_draft(self, **kwargs) -> dict:
        self.create_calls.append(kwargs)
        message_id = kwargs["headers"]["Message-ID"]
        result = {
            "id": f"draft-{len(self.create_calls)}",
            "message": {
                "id": f"message-{len(self.create_calls)}",
                "threadId": f"thread-{len(self.create_calls)}",
            },
        }
        if self.create_then_raise:
            self.drafts[message_id] = result
            raise RuntimeError("synthetic uncertain Gmail timeout")
        if self.create_error is not None:
            raise self.create_error
        self.drafts[message_id] = result
        return result

    async def find_draft_by_rfc_message_id(self, rfc_message_id: str) -> dict | None:
        self.lookup_calls.append(rfc_message_id)
        result = self.drafts.get(rfc_message_id)
        return json.loads(json.dumps(result)) if result is not None else None


class _BlockingGateway(_RecordingGateway):
    def __init__(self) -> None:
        super().__init__()
        self.create_started = asyncio.Event()
        self.release_create = asyncio.Event()

    async def create_draft(self, **kwargs) -> dict:
        self.create_calls.append(kwargs)
        self.create_started.set()
        await self.release_create.wait()
        message_id = kwargs["headers"]["Message-ID"]
        result = {
            "id": "draft-1",
            "message": {"id": "message-1", "threadId": "thread-1"},
        }
        self.drafts[message_id] = result
        return result


@asynccontextmanager
async def _gmail_draft_database():
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")
    schema = f"commercial_gmail_draft_{uuid4().hex}"
    connection = await asyncpg.connect(database_url)
    migrations = Path(__file__).parents[1] / "atlas_brain/storage/migrations"
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        await connection.execute("CREATE TABLE contacts (id UUID PRIMARY KEY)")
        for name in (
            "045_invoices.sql",
            "047_invoice_extra_fields.sql",
            "370_commercial_billing_runs.sql",
            "372_commercial_billing_candidate_approvals.sql",
            "373_commercial_billing_invoice_pdf_artifacts.sql",
            "374_commercial_billing_invoice_gmail_drafts.sql",
        ):
            await connection.execute((migrations / name).read_text())
        yield connection, schema
    finally:
        await connection.execute("SET search_path TO public")
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


async def _seed_approved_invoice(
    connection,
    schema: str,
    *,
    delivery_method: str = "gmail_pdf",
    customer_email: str = "billing@example.test",
    with_artifact: bool = True,
) -> dict:
    contact_id, run_id, candidate_id, approval_id, invoice_id = (
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
    )
    candidate_key = f"commercial-billing:acme:{approval_id}"
    source_fingerprint = _fingerprint({"approval": str(approval_id)})
    request_fingerprint = _fingerprint({"seed": str(approval_id)})
    now = datetime(2026, 4, 2, tzinfo=timezone.utc)
    await connection.execute("INSERT INTO contacts (id) VALUES ($1)", contact_id)
    await connection.execute(
        """
        INSERT INTO commercial_billing_runs (
            id, billing_period, state, candidate_contract_version,
            snapshot_fingerprint, source, idempotency_key, request_fingerprint,
            created_by, created_at, updated_at
        )
        VALUES ($1, '2026-03', 'draft', 1, $2, 'eom_admin', $3, $4, 'Juan', $5, $5)
        """,
        run_id,
        source_fingerprint,
        f"run-{approval_id}",
        request_fingerprint,
        now,
    )
    await connection.execute(
        """
        INSERT INTO commercial_billing_run_candidates (
            id, billing_run_id, candidate_key, source_fingerprint,
            display_order, snapshot, created_at
        )
        VALUES ($1, $2, $3, $4, 0, '{}'::jsonb, $5)
        """,
        candidate_id,
        run_id,
        candidate_key,
        source_fingerprint,
        now,
    )
    metadata = {
        "candidateKey": candidate_key,
        "commercialBillingRunId": str(run_id),
        "deliveryMethod": delivery_method,
        "sourceFingerprint": source_fingerprint,
    }
    await connection.execute(
        """
        INSERT INTO invoices (
            id, invoice_number, contact_id, customer_name, customer_email,
            line_items, subtotal, tax_rate, tax_amount, discount_amount,
            total_amount, issue_date, due_date, status, source, source_ref,
            business_context_id, notes, metadata, invoice_for, contact_name,
            created_at, updated_at
        )
        VALUES (
            $1, $2, $3, 'Acme Office', $4, $5::jsonb, 96.50, 0, 0, 0,
            96.50, $6, $7, 'draft', 'eom_commercial_billing', $8,
            'effingham_maids', 'Approved commercial billing candidate', $9::jsonb,
            'Office cleaning - March 2026', 'Billing Contact', $10, $10
        )
        """,
        invoice_id,
        f"INV-2026-Mar-{str(approval_id).replace('-', '')[:4]}",
        contact_id,
        customer_email,
        json.dumps(
            [
                {
                    "amount": 96.50,
                    "description": "Office cleaning",
                    "quantity": 2,
                    "unit_price": 48.25,
                }
            ]
        ),
        date(2026, 4, 2),
        date(2026, 4, 16),
        f"approval:{approval_id}",
        json.dumps(metadata),
        now,
    )
    await connection.execute(
        """
        INSERT INTO commercial_billing_candidate_approvals (
            id, billing_run_id, candidate_key, source_fingerprint, source,
            idempotency_key, request_fingerprint, invoice_id, state,
            approved_by, approved_at, created_at, updated_at
        )
        VALUES ($1, $2, $3, $4, 'eom_admin', $5, $6, $7, 'invoice_created',
                'Juan', $8, $8, $8)
        """,
        approval_id,
        run_id,
        candidate_key,
        source_fingerprint,
        f"approval-{approval_id}",
        request_fingerprint,
        invoice_id,
        now,
    )
    renderer = _PDFRenderer()
    if with_artifact:
        await CommercialBillingInvoicePDFService(
            pool=_SchemaPool(connection, schema),
            renderer=renderer,
            now=lambda: now,
        ).generate_or_reuse(
            approval_id=approval_id,
            idempotency_key=f"pdf-{approval_id}",
            actor="Juan",
        )
    return {
        "approval_id": approval_id,
        "invoice_id": invoice_id,
        "renderer": renderer,
    }


def _service(connection, schema: str, gateway: _RecordingGateway):
    now = datetime(2026, 4, 2, tzinfo=timezone.utc)
    pool = _SchemaPool(connection, schema)
    return CommercialBillingInvoiceGmailDraftService(
        pool=pool,
        pdf_service=CommercialBillingInvoicePDFService(pool=pool, now=lambda: now),
        gateway_loader=lambda: gateway,
        now=lambda: now,
    )


@pytest.mark.asyncio
async def test_real_postgres_creates_one_no_send_draft_and_reuses_it_idempotently():
    async with _gmail_draft_database() as (connection, schema):
        seed = await _seed_approved_invoice(connection, schema)
        gateway = _RecordingGateway()
        service = _service(connection, schema, gateway)

        created = await service.create_or_reuse(
            approval_id=seed["approval_id"], idempotency_key="gmail-1", actor="Juan"
        )
        replayed = await service.create_or_reuse(
            approval_id=seed["approval_id"], idempotency_key="gmail-1", actor="Mayra"
        )
        reused = await service.create_or_reuse(
            approval_id=seed["approval_id"], idempotency_key="gmail-2", actor="Mayra"
        )

        assert created["replayed"] is False
        assert created["reused"] is False
        assert created["draft"]["state"] == "draft_created"
        assert created["draft"]["gmailDraftId"] == "draft-1"
        assert created["draft"]["recipientEmail"] == "billing@example.test"
        assert created["draft"]["subject"].startswith("Invoice INV-2026-Mar-")
        assert created["draft"]["rfcMessageId"].startswith(
            "<atlas-eom-commercial-billing-"
        )
        assert replayed["replayed"] is True
        assert replayed["reused"] is True
        assert reused["replayed"] is False
        assert reused["reused"] is True
        assert len(gateway.create_calls) == 1
        assert gateway.lookup_calls == []

        call = gateway.create_calls[0]
        assert call["headers"]["Message-ID"] == created["draft"]["rfcMessageId"]
        assert call["headers"]["X-Atlas-Commercial-Billing-Approval"] == str(
            seed["approval_id"]
        )
        assert call["headers"]["X-Atlas-Commercial-Billing-Invoice"] == str(
            seed["invoice_id"]
        )
        attachment = call["attachments"][0]
        assert attachment["filename"].endswith(".pdf")
        assert base64.b64decode(attachment["content"]).startswith(b"%PDF-")
        assert call["body"]
        assert call["html"]

        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_gmail_drafts"
        ) == 1
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_gmail_draft_operations"
        ) == 2
        invoice = await connection.fetchrow(
            "SELECT status, sent_at, sent_via FROM invoices WHERE id = $1", seed["invoice_id"]
        )
        assert dict(invoice) == {"status": "draft", "sent_at": None, "sent_via": None}


@pytest.mark.asyncio
async def test_definite_gmail_rejection_stays_retryable_and_same_key_can_create_once():
    async with _gmail_draft_database() as (connection, schema):
        seed = await _seed_approved_invoice(connection, schema)
        gateway = _RecordingGateway()
        gateway.create_error = GmailDraftCreateError(
            "synthetic rejected request", definitely_not_created=True
        )
        service = _service(connection, schema, gateway)

        with pytest.raises(CommercialBillingGmailDraftUnavailableError):
            await service.create_or_reuse(
                approval_id=seed["approval_id"], idempotency_key="gmail-retry", actor="Juan"
            )
        assert await connection.fetchval(
            "SELECT state FROM commercial_billing_invoice_gmail_drafts"
        ) == "retryable"
        gateway.create_error = None

        created = await service.create_or_reuse(
            approval_id=seed["approval_id"], idempotency_key="gmail-retry", actor="Juan"
        )
        assert created["replayed"] is True
        assert created["reused"] is False
        assert created["draft"]["state"] == "draft_created"
        assert len(gateway.create_calls) == 2
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_gmail_drafts"
        ) == 1
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_gmail_draft_operations"
        ) == 1


@pytest.mark.asyncio
async def test_gmail_acceptance_with_database_confirm_failure_recovers_by_message_id():
    async with _gmail_draft_database() as (connection, schema):
        seed = await _seed_approved_invoice(connection, schema)
        gateway = _RecordingGateway()
        service = _service(connection, schema, gateway)
        await connection.execute(
            """
            CREATE FUNCTION reject_gmail_draft_confirmation()
            RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
                IF NEW.state = 'draft_created' THEN
                    RAISE EXCEPTION 'synthetic confirmation failure';
                END IF;
                RETURN NEW;
            END;
            $$
            """
        )
        await connection.execute(
            """
            CREATE TRIGGER reject_gmail_draft_confirmation_trigger
            BEFORE UPDATE ON commercial_billing_invoice_gmail_drafts
            FOR EACH ROW EXECUTE FUNCTION reject_gmail_draft_confirmation()
            """
        )
        with pytest.raises(CommercialBillingGmailDraftRecoveryRequiredError):
            await service.create_or_reuse(
                approval_id=seed["approval_id"], idempotency_key="gmail-confirm", actor="Juan"
            )
        assert len(gateway.create_calls) == 1
        assert await connection.fetchval(
            "SELECT state FROM commercial_billing_invoice_gmail_drafts"
        ) == "recovery_required"
        await connection.execute(
            "DROP TRIGGER reject_gmail_draft_confirmation_trigger "
            "ON commercial_billing_invoice_gmail_drafts"
        )
        await connection.execute("DROP FUNCTION reject_gmail_draft_confirmation()")

        recovered = await service.create_or_reuse(
            approval_id=seed["approval_id"], idempotency_key="gmail-confirm", actor="Juan"
        )
        assert recovered["replayed"] is True
        assert recovered["reused"] is True
        assert recovered["draft"]["state"] == "draft_created"
        assert len(gateway.create_calls) == 1
        assert len(gateway.lookup_calls) == 1


@pytest.mark.asyncio
async def test_concurrent_fresh_keys_never_issue_a_second_gmail_create():
    async with _gmail_draft_database() as (connection, schema):
        seed = await _seed_approved_invoice(connection, schema)
        gateway = _BlockingGateway()
        service = _service(connection, schema, gateway)
        first = asyncio.create_task(
            service.create_or_reuse(
                approval_id=seed["approval_id"], idempotency_key="gmail-concurrent-1", actor="Juan"
            )
        )
        await asyncio.wait_for(gateway.create_started.wait(), timeout=2)
        with pytest.raises(CommercialBillingGmailDraftRecoveryRequiredError):
            await service.create_or_reuse(
                approval_id=seed["approval_id"], idempotency_key="gmail-concurrent-2", actor="Mayra"
            )
        assert len(gateway.create_calls) == 1
        assert len(gateway.lookup_calls) == 1
        gateway.release_create.set()
        completed = await first
        assert completed["draft"]["state"] == "draft_created"
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_gmail_drafts"
        ) == 1
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_gmail_draft_operations"
        ) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("delivery_method", "customer_email", "with_artifact"),
    (
        ("manual_square", "billing@example.test", True),
        ("gmail_pdf", "not-an-email", True),
        ("gmail_pdf", "billing@example.test", False),
    ),
)
async def test_ineligible_or_missing_pdf_draft_inputs_leave_zero_gmail_writes(
    delivery_method: str,
    customer_email: str,
    with_artifact: bool,
):
    async with _gmail_draft_database() as (connection, schema):
        seed = await _seed_approved_invoice(
            connection,
            schema,
            delivery_method=delivery_method,
            customer_email=customer_email,
            with_artifact=with_artifact,
        )
        gateway = _RecordingGateway()
        with pytest.raises(CommercialBillingGmailDraftConflictError):
            await _service(connection, schema, gateway).create_or_reuse(
                approval_id=seed["approval_id"], idempotency_key="gmail-invalid", actor="Juan"
            )
        assert gateway.create_calls == []
        assert gateway.lookup_calls == []
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_gmail_drafts"
        ) == 0
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_gmail_draft_operations"
        ) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ("sent", "changed_after_pdf"))
async def test_sent_or_changed_invoice_rejects_new_draft_before_gmail_or_operation_write(
    mutation: str,
):
    async with _gmail_draft_database() as (connection, schema):
        seed = await _seed_approved_invoice(connection, schema)
        if mutation == "sent":
            await connection.execute(
                "UPDATE invoices SET status = 'sent' WHERE id = $1",
                seed["invoice_id"],
            )
        else:
            await connection.execute(
                "UPDATE invoices SET notes = 'Changed after PDF generation' WHERE id = $1",
                seed["invoice_id"],
            )
        gateway = _RecordingGateway()
        with pytest.raises(CommercialBillingGmailDraftConflictError):
            await _service(connection, schema, gateway).create_or_reuse(
                approval_id=seed["approval_id"], idempotency_key="gmail-sent", actor="Juan"
            )
        assert gateway.create_calls == []
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_gmail_drafts"
        ) == 0
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_gmail_draft_operations"
        ) == 0


@pytest.mark.asyncio
async def test_gmail_transport_draft_create_uses_only_drafts_endpoint_and_safe_headers():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"id": "draft-1", "message": {"id": "message-1", "threadId": "thread-1"}},
        )

    transport = GmailTransport()
    transport._access_token = "token"
    transport._token_expires = time.time() + 600
    transport._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await transport.create_draft(
            to=["billing@example.test"],
            subject="Invoice INV-1",
            body="Body",
            headers={
                "Message-ID": "<atlas-eom-commercial-billing-1@example.test>",
                "X-Atlas-Commercial-Billing-Approval": "approval-1",
            },
        )
    finally:
        await transport.close()

    assert result["id"] == "draft-1"
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert request.url.path.endswith("/users/me/drafts")
    assert "/messages/send" not in request.url.path
    payload = json.loads(request.content)
    raw = payload["message"]["raw"]
    raw_bytes = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
    message = message_from_bytes(raw_bytes)
    assert message["Message-ID"] == "<atlas-eom-commercial-billing-1@example.test>"
    assert message["X-Atlas-Commercial-Billing-Approval"] == "approval-1"


@pytest.mark.asyncio
async def test_gmail_transport_looks_up_exact_rfc_message_id_and_rejects_ambiguity():
    requests: list[httpx.Request] = []

    async def one_match(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "drafts": [
                    {
                        "id": "draft-1",
                        "message": {"id": "message-1", "threadId": "thread-1"},
                    }
                ]
            },
        )

    transport = GmailTransport()
    transport._access_token = "token"
    transport._token_expires = time.time() + 600
    transport._client = httpx.AsyncClient(transport=httpx.MockTransport(one_match))
    try:
        found = await transport.find_draft_by_rfc_message_id("<stable@example.test>")
    finally:
        await transport.close()
    assert found == {
        "id": "draft-1",
        "message": {"id": "message-1", "threadId": "thread-1"},
    }
    assert requests[0].method == "GET"
    assert requests[0].url.path.endswith("/users/me/drafts")
    assert requests[0].url.params["q"] == "rfc822msgid:<stable@example.test>"

    async def ambiguous(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "drafts": [
                    {"id": "draft-1", "message": {"id": "m-1", "threadId": "t-1"}},
                    {"id": "draft-2", "message": {"id": "m-2", "threadId": "t-2"}},
                ]
            },
        )

    transport = GmailTransport()
    transport._access_token = "token"
    transport._token_expires = time.time() + 600
    transport._client = httpx.AsyncClient(transport=httpx.MockTransport(ambiguous))
    try:
        with pytest.raises(GmailDraftLookupError, match="multiple"):
            await transport.find_draft_by_rfc_message_id("<stable@example.test>")
    finally:
        await transport.close()


class _RouteService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create_or_reuse(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return {
            "draft": {
                "approvalId": str(kwargs["approval_id"]),
                "id": "draft-record-1",
                "state": "draft_created",
            },
            "replayed": False,
            "reused": False,
        }


@pytest.mark.asyncio
async def test_full_atlas_app_gmail_draft_route_requires_existing_auth_and_never_returns_pdf_bytes():
    from atlas_brain.api.invoicing import auth as receivables_auth
    from atlas_brain.api.invoicing import receivables as routes
    from atlas_brain.eom_api.auth import generate_receivables_service_token
    from atlas_brain.main import app

    approval_id = uuid4()
    service = _RouteService()
    generated = generate_receivables_service_token()
    original_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[receivables_auth.get_receivables_api_config] = lambda: SimpleNamespace(
        receivables_api_enabled=True,
        receivables_service_token="",
        receivables_service_token_sha256=generated.sha256,
    )
    app.dependency_overrides[
        routes.get_commercial_billing_invoice_gmail_draft_service
    ] = lambda: service
    path = f"/api/v1/receivables/commercial-billing-approvals/{approval_id}/gmail-draft"
    headers = {"Authorization": f"Bearer {generated.token}"}
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.post(path)).status_code == 401
            assert (await client.post(path, headers=headers)).status_code == 422
            assert (
                await client.post(
                    path,
                    headers={**headers, "Idempotency-Key": "route-gmail-1"},
                )
            ).status_code == 422
            response = await client.post(
                path,
                headers={
                    **headers,
                    "Idempotency-Key": "route-gmail-1",
                    "X-EOM-Actor": "Juan",
                },
            )
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)
    assert response.status_code == 201
    assert response.json()["draft"] == {
        "approvalId": str(approval_id),
        "id": "draft-record-1",
        "state": "draft_created",
    }
    assert "pdf_bytes" not in response.text
    assert "pdfBytes" not in response.text
    assert service.calls == [
        {"approval_id": approval_id, "idempotency_key": "route-gmail-1", "actor": "Juan"}
    ]


def test_gmail_draft_migration_is_additive_and_never_encodes_sent_state():
    migration = (
        Path(__file__).parents[1]
        / "atlas_brain/storage/migrations/374_commercial_billing_invoice_gmail_drafts.sql"
    ).read_text()
    executable = "\n".join(
        line for line in migration.splitlines() if not line.lstrip().startswith("--")
    )
    assert "CREATE TABLE IF NOT EXISTS commercial_billing_invoice_gmail_drafts" in migration
    assert "CREATE TABLE IF NOT EXISTS commercial_billing_invoice_gmail_draft_operations" in migration
    assert migration.count("ON DELETE RESTRICT") == 3
    assert "approval_id UUID NOT NULL UNIQUE" in migration
    assert "artifact_id UUID NOT NULL UNIQUE" in migration
    assert "UNIQUE (source, idempotency_key)" in migration
    assert "'creating', 'retryable', 'recovery_required', 'draft_created'" in migration
    assert "sent_at" not in executable
    assert "UPDATE invoices" not in executable
    assert "DROP TABLE" not in executable


def test_gmail_draft_service_imports_no_sender_or_financial_state_writer():
    import atlas_brain.services.commercial_billing_invoice_gmail_drafts as drafts

    source = inspect.getsource(drafts)
    imports = {
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        {
            f"{node.module}.{alias.name}" if node.module else alias.name
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
    )
    assert not any(
        fragment in imported
        for fragment in {
            "email_provider",
            "invoicing_server",
            "monthly_invoice_generation",
            "crm_provider",
        }
        for imported in imports
    )
    assert "UPDATE invoices" not in source
    assert ".send(" not in source
    assert "float(" not in source


def test_invoicing_workflow_enrolls_the_gmail_draft_surface_and_contract():
    workflow = (Path(__file__).parents[1] / ".github/workflows/atlas_invoicing_checks.yml").read_text()
    for path in (
        "atlas_brain/services/commercial_billing_invoice_gmail_drafts.py",
        "atlas_brain/storage/migrations/374_commercial_billing_invoice_gmail_drafts.sql",
        "atlas_brain/tools/gmail.py",
        "tests/test_commercial_billing_gmail_drafts.py",
    ):
        assert workflow.count(f'      - "{path}"') == 2
    assert "tests/test_commercial_billing_gmail_drafts.py \\" in workflow
