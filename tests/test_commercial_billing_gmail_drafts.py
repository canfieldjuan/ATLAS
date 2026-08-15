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
from atlas_brain.services.commercial_billing_invoice_gmail_sent_reconciliation import (
    MAX_DELIVERY_STATE_OFFSET,
    CommercialBillingGmailDeliveryStateNotFoundError,
    CommercialBillingGmailSentReconciliationConflictError,
    CommercialBillingGmailSentReconciliationUnavailableError,
    CommercialBillingGmailSentReconciliationValidationError,
    CommercialBillingInvoiceGmailSentReconciliationService,
)
from atlas_brain.services.commercial_billing_invoice_pdfs import (
    CommercialBillingInvoicePDFService,
)
from atlas_brain.tools.gmail import (
    GmailDraftCreateError,
    GmailDraftLookupError,
    GmailSentMessageLookupError,
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


@pytest.mark.parametrize(
    ("artifact", "draft", "reconciliation", "invoice_status", "sent_at", "sent_via", "expected"),
    (
        (None, None, None, "draft", None, None, "needs_pdf"),
        ({"id": "pdf"}, None, None, "draft", None, None, "needs_gmail_draft"),
        ({"id": "pdf"}, {"state": "creating"}, {"state": "not_reconciled"}, "draft", None, None, "gmail_draft_creating"),
        ({"id": "pdf"}, {"state": "retryable"}, {"state": "not_reconciled"}, "draft", None, None, "gmail_draft_retryable"),
        ({"id": "pdf"}, {"state": "recovery_required"}, {"state": "not_reconciled"}, "draft", None, None, "gmail_draft_recovery_required"),
        ({"id": "pdf"}, {"state": "draft_created"}, {"state": "not_reconciled"}, "draft", None, None, "gmail_draft_not_reconciled"),
        ({"id": "pdf"}, {"state": "draft_created"}, {"state": "draft_present"}, "draft", None, None, "gmail_draft_present"),
        ({"id": "pdf"}, {"state": "draft_created"}, {"state": "draft_missing"}, "draft", None, None, "gmail_draft_missing"),
        ({"id": "pdf"}, {"state": "draft_created"}, {"state": "sent_confirmed"}, "sent", datetime(2026, 4, 2, tzinfo=timezone.utc), "gmail", "gmail_sent_confirmed"),
        ({"id": "pdf"}, {"state": "draft_created"}, {"state": "draft_missing"}, "sent", datetime(2026, 4, 2, tzinfo=timezone.utc), "gmail", "lifecycle_conflict"),
    ),
)
def test_delivery_state_vocabulary_is_closed_and_missing_drafts_never_become_sent(
    artifact, draft, reconciliation, invoice_status, sent_at, sent_via, expected
):
    assert CommercialBillingInvoiceGmailSentReconciliationService._delivery_state(
        context_matches=True,
        invoice_status=invoice_status,
        invoice_sent_at=sent_at,
        invoice_sent_via=sent_via,
        artifact=artifact,
        draft=draft,
        reconciliation=reconciliation,
    ) == expected


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


class _SingleStatementDeliveryStateConnection:
    def __init__(self, connection) -> None:
        self._connection = connection
        self.read_statement_count = 0

    async def fetch(self, query: str, *args):
        self.read_statement_count += 1
        if self.read_statement_count > 1:
            raise AssertionError("delivery-state pagination must use one read statement")
        return await self._connection.fetch(query, *args)

    async def fetchrow(self, *args, **kwargs):
        raise AssertionError("delivery-state pagination must not split its read")

    async def fetchval(self, *args, **kwargs):
        raise AssertionError("delivery-state pagination must not split its read")

    def __getattr__(self, name):
        return getattr(self._connection, name)


class _SingleStatementDeliveryStatePool(_SchemaPool):
    def __init__(self, connection, schema: str) -> None:
        super().__init__(connection, schema)
        self.read_connection = _SingleStatementDeliveryStateConnection(connection)

    @asynccontextmanager
    async def transaction(self):
        async with self.connection.transaction():
            await self.connection.execute(f'SET LOCAL search_path TO "{self.schema}"')
            yield self.read_connection


class _TransactionForbiddenPool:
    def transaction(self):
        raise AssertionError(
            "out-of-range delivery-state offsets must fail before opening a database transaction"
        )


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
        self.sent_lookup_calls: list[str] = []
        self.drafts: dict[str, dict] = {}
        self.sent_messages: dict[str, dict] = {}
        self.create_error: Exception | None = None
        self.sent_lookup_error: Exception | None = None
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

    async def find_sent_message_by_rfc_message_id(
        self, rfc_message_id: str
    ) -> dict | None:
        self.sent_lookup_calls.append(rfc_message_id)
        if self.sent_lookup_error is not None:
            raise self.sent_lookup_error
        result = self.sent_messages.get(rfc_message_id)
        return json.loads(json.dumps(result)) if result is not None else None

    def record_sent(
        self,
        *,
        rfc_message_id: str,
        approval_id,
        invoice_id,
        internal_date: str = "1775088000123",
    ) -> None:
        self.drafts.pop(rfc_message_id, None)
        self.sent_messages[rfc_message_id] = {
            "id": "sent-message-1",
            "threadId": "sent-thread-1",
            "labelIds": ["SENT"],
            "internalDate": internal_date,
            "headers": [
                {"name": "Message-ID", "value": rfc_message_id},
                {
                    "name": "X-Atlas-Commercial-Billing-Approval",
                    "value": str(approval_id),
                },
                {
                    "name": "X-Atlas-Commercial-Billing-Invoice",
                    "value": str(invoice_id),
                },
            ],
        }


class _BlockingGateway(_RecordingGateway):
    def __init__(self) -> None:
        super().__init__()
        self.create_started = asyncio.Event()
        self.release_create = asyncio.Event()

    async def create_draft(self, **kwargs) -> dict:
        self.create_calls.append(kwargs)
        self.create_started.set()
        await self.release_create.wait()
        if self.create_error is not None:
            raise self.create_error
        message_id = kwargs["headers"]["Message-ID"]
        result = {
            "id": "draft-1",
            "message": {"id": "message-1", "threadId": "thread-1"},
        }
        self.drafts[message_id] = result
        return result


class _BlockingSentGateway(_RecordingGateway):
    def __init__(self) -> None:
        super().__init__()
        self.sent_lookup_started = asyncio.Event()
        self.release_first_lookup = asyncio.Event()

    async def find_sent_message_by_rfc_message_id(
        self, rfc_message_id: str
    ) -> dict | None:
        self.sent_lookup_calls.append(rfc_message_id)
        if len(self.sent_lookup_calls) == 1:
            self.sent_lookup_started.set()
            await self.release_first_lookup.wait()
        result = self.sent_messages.get(rfc_message_id)
        return json.loads(json.dumps(result)) if result is not None else None


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
            "375_commercial_billing_invoice_gmail_sent_reconciliation.sql",
            "377_commercial_billing_gmail_draft_replacements.sql",
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
    billing_run_id=None,
    display_order: int = 0,
) -> dict:
    contact_id, generated_run_id, candidate_id, approval_id, invoice_id = (
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
    )
    run_id = billing_run_id or generated_run_id
    candidate_key = f"commercial-billing:acme:{approval_id}"
    source_fingerprint = _fingerprint({"approval": str(approval_id)})
    request_fingerprint = _fingerprint({"seed": str(approval_id)})
    now = datetime(2026, 4, 2, tzinfo=timezone.utc)
    await connection.execute("INSERT INTO contacts (id) VALUES ($1)", contact_id)
    if billing_run_id is None:
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
        VALUES ($1, $2, $3, $4, $5, '{}'::jsonb, $6)
        """,
        candidate_id,
        run_id,
        candidate_key,
        source_fingerprint,
        display_order,
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
        "candidate_key": candidate_key,
        "invoice_id": invoice_id,
        "renderer": renderer,
        "run_id": run_id,
        "source_fingerprint": source_fingerprint,
    }


async def _seed_equivalent_review_run(connection, seed: dict) -> dict:
    run_id = uuid4()
    candidate_id = uuid4()
    now = datetime(2026, 4, 3, tzinfo=timezone.utc)
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
        seed["source_fingerprint"],
        f"equivalent-run-{run_id}",
        _fingerprint({"equivalentRun": str(run_id)}),
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
        seed["candidate_key"],
        seed["source_fingerprint"],
        now,
    )
    return {"run_id": run_id}


def _service(connection, schema: str, gateway: _RecordingGateway):
    now = datetime(2026, 4, 2, tzinfo=timezone.utc)
    pool = _SchemaPool(connection, schema)
    return CommercialBillingInvoiceGmailDraftService(
        pool=pool,
        pdf_service=CommercialBillingInvoicePDFService(pool=pool, now=lambda: now),
        gateway_loader=lambda: gateway,
        now=lambda: now,
    )


def _sent_reconciliation_service(
    connection, schema: str, gateway: _RecordingGateway
):
    now = datetime(2026, 4, 2, tzinfo=timezone.utc)
    return CommercialBillingInvoiceGmailSentReconciliationService(
        pool=_SchemaPool(connection, schema),
        gateway_loader=lambda: gateway,
        now=lambda: now,
    )


async def _create_and_reconcile_missing_draft(
    connection,
    schema: str,
    *,
    seed: dict,
    gateway: _RecordingGateway,
    draft_key: str,
) -> dict:
    draft = await _service(connection, schema, gateway).create_or_reuse(
        approval_id=seed["approval_id"],
        idempotency_key=draft_key,
        actor="Juan",
    )
    gateway.drafts.pop(draft["draft"]["rfcMessageId"])
    reconciled = await _sent_reconciliation_service(
        connection,
        schema,
        gateway,
    ).reconcile(
        approval_id=seed["approval_id"],
        idempotency_key=f"{draft_key}-missing",
        actor="Juan",
    )
    assert reconciled["outcome"] == "draft_missing"
    return draft


@pytest.mark.asyncio
async def test_real_postgres_delivery_state_is_bounded_and_never_calls_gmail():
    async with _gmail_draft_database() as (connection, schema):
        first = await _seed_approved_invoice(connection, schema)
        second = await _seed_approved_invoice(
            connection,
            schema,
            billing_run_id=first["run_id"],
            display_order=1,
        )
        gateway_loads = 0

        def failing_gateway_loader():
            nonlocal gateway_loads
            gateway_loads += 1
            raise AssertionError("durable delivery-state reads must not load Gmail")

        service = CommercialBillingInvoiceGmailSentReconciliationService(
            pool=_SchemaPool(connection, schema),
            gateway_loader=failing_gateway_loader,
        )
        first_page = await service.list_delivery_state_for_run(
            billing_run_id=first["run_id"], limit=1, offset=0
        )
        second_page = await service.list_delivery_state_for_run(
            billing_run_id=first["run_id"], limit=1, offset=1
        )
        empty_page = await service.list_delivery_state_for_run(
            billing_run_id=first["run_id"], limit=1, offset=2
        )

        assert first_page["total"] == 2
        assert first_page["limit"] == 1
        assert first_page["offset"] == 0
        assert second_page["total"] == 2
        assert {first_page["items"][0]["approval"]["id"], second_page["items"][0]["approval"]["id"]} == {
            str(first["approval_id"]),
            str(second["approval_id"]),
        }
        assert first_page["items"][0]["deliveryState"] == "needs_gmail_draft"
        assert first_page["items"][0]["pdf"]["state"] == "ready"
        assert first_page["items"][0]["gmailDraft"] is None
        assert empty_page == {
            "billingRunId": str(first["run_id"]),
            "items": [],
            "limit": 1,
            "offset": 2,
            "total": 2,
        }
        assert "pdf_bytes" not in json.dumps(first_page).casefold()
        assert "pdfbytes" not in json.dumps(first_page).casefold()
        assert gateway_loads == 0

        with pytest.raises(CommercialBillingGmailSentReconciliationValidationError):
            await service.list_delivery_state_for_run(
                billing_run_id=first["run_id"], limit=0
            )
        with pytest.raises(CommercialBillingGmailSentReconciliationValidationError):
            await service.list_delivery_state_for_run(
                billing_run_id=first["run_id"], offset=-1
            )
        assert gateway_loads == 0


@pytest.mark.asyncio
async def test_delivery_state_rejects_postgres_out_of_range_offsets_before_opening_a_database_transaction():
    service = CommercialBillingInvoiceGmailSentReconciliationService(
        pool=_TransactionForbiddenPool(),
        gateway_loader=lambda: (_ for _ in ()).throw(
            AssertionError("delivery-state validation must not load Gmail")
        ),
    )

    with pytest.raises(CommercialBillingGmailSentReconciliationValidationError):
        await service.list_delivery_state_for_run(
            billing_run_id=uuid4(), offset=MAX_DELIVERY_STATE_OFFSET + 1
        )


@pytest.mark.asyncio
async def test_real_postgres_delivery_state_projects_cross_linked_gmail_drafts_as_lifecycle_conflicts():
    async with _gmail_draft_database() as (connection, schema):
        first = await _seed_approved_invoice(connection, schema)
        second = await _seed_approved_invoice(
            connection,
            schema,
            billing_run_id=first["run_id"],
            display_order=1,
        )
        await _service(connection, schema, _RecordingGateway()).create_or_reuse(
            approval_id=first["approval_id"],
            idempotency_key="cross-linked-draft",
            actor="Juan",
        )
        second_artifact_id = await connection.fetchval(
            "SELECT id FROM commercial_billing_invoice_pdf_artifacts WHERE approval_id = $1",
            second["approval_id"],
        )
        await connection.execute(
            """
            UPDATE commercial_billing_invoice_gmail_drafts
               SET artifact_id = $2
             WHERE approval_id = $1
            """,
            first["approval_id"],
            second_artifact_id,
        )
        service = CommercialBillingInvoiceGmailSentReconciliationService(
            pool=_SchemaPool(connection, schema),
            gateway_loader=lambda: (_ for _ in ()).throw(
                AssertionError("durable delivery-state reads must not load Gmail")
            ),
        )

        page = await service.list_delivery_state_for_run(
            billing_run_id=first["run_id"]
        )

        by_approval = {item["approval"]["id"]: item for item in page["items"]}
        conflict = by_approval[str(first["approval_id"])]
        assert page["total"] == 2
        assert conflict["deliveryState"] == "lifecycle_conflict"
        assert conflict["gmailDraft"]["approvalId"] == str(first["approval_id"])
        assert conflict["gmailDraft"]["artifactId"] == str(second_artifact_id)
        assert conflict["reconciliation"]["state"] == "not_reconciled"
        assert "recoveryAction" not in conflict["reconciliation"]
        assert by_approval[str(second["approval_id"])]["deliveryState"] == "needs_gmail_draft"


@pytest.mark.asyncio
async def test_real_postgres_delivery_state_total_and_page_share_one_snapshot():
    async with _gmail_draft_database() as (connection, schema):
        first = await _seed_approved_invoice(connection, schema)
        await _seed_approved_invoice(
            connection,
            schema,
            billing_run_id=first["run_id"],
            display_order=1,
        )
        pool = _SingleStatementDeliveryStatePool(connection, schema)
        service = CommercialBillingInvoiceGmailSentReconciliationService(
            pool=pool,
            gateway_loader=lambda: (_ for _ in ()).throw(
                AssertionError("durable delivery-state reads must not load Gmail")
            ),
        )

        page = await service.list_delivery_state_for_run(
            billing_run_id=first["run_id"], limit=1
        )

        # DatabasePool.transaction() is READ COMMITTED, so a separate count and
        # page statement could observe different committed approvals.  This
        # wrapper admits only one service read statement, which PostgreSQL
        # evaluates from one statement snapshot.
        assert pool.read_connection.read_statement_count == 1
        assert page["total"] == 2
        assert len(page["items"]) == 1


@pytest.mark.asyncio
async def test_real_postgres_delivery_state_reopens_exact_prior_approval_from_a_later_equivalent_run():
    async with _gmail_draft_database() as (connection, schema):
        seed = await _seed_approved_invoice(connection, schema)
        later_run = await _seed_equivalent_review_run(connection, seed)
        service = CommercialBillingInvoiceGmailSentReconciliationService(
            pool=_SchemaPool(connection, schema),
            gateway_loader=lambda: (_ for _ in ()).throw(
                AssertionError("durable delivery-state reads must not load Gmail")
            ),
        )

        page = await service.list_delivery_state_for_run(
            billing_run_id=later_run["run_id"]
        )

        assert page["billingRunId"] == str(later_run["run_id"])
        assert page["total"] == 1
        item, = page["items"]
        assert item["approval"] == {
            "billingRunId": str(seed["run_id"]),
            "candidateKey": seed["candidate_key"],
            "id": str(seed["approval_id"]),
            "sourceFingerprint": seed["source_fingerprint"],
            "state": "invoice_created",
        }
        assert item["candidate"] == {
            "candidateKey": seed["candidate_key"],
            "sourceFingerprint": seed["source_fingerprint"],
        }
        assert item["deliveryState"] == "needs_gmail_draft"


@pytest.mark.asyncio
async def test_real_postgres_delivery_state_never_infers_sent_from_a_missing_draft():
    async with _gmail_draft_database() as (connection, schema):
        missing = await _seed_approved_invoice(connection, schema)
        sent = await _seed_approved_invoice(
            connection,
            schema,
            billing_run_id=missing["run_id"],
            display_order=1,
        )
        missing_gateway = _RecordingGateway()
        sent_gateway = _RecordingGateway()
        missing_draft = await _service(connection, schema, missing_gateway).create_or_reuse(
            approval_id=missing["approval_id"],
            idempotency_key="missing-draft",
            actor="Juan",
        )
        sent_draft = await _service(connection, schema, sent_gateway).create_or_reuse(
            approval_id=sent["approval_id"],
            idempotency_key="sent-draft",
            actor="Juan",
        )
        missing_gateway.drafts.pop(missing_draft["draft"]["rfcMessageId"])
        missing_outcome = await _sent_reconciliation_service(
            connection, schema, missing_gateway
        ).reconcile(
            approval_id=missing["approval_id"],
            idempotency_key="missing-reconcile",
            actor="Juan",
        )
        sent_gateway.record_sent(
            rfc_message_id=sent_draft["draft"]["rfcMessageId"],
            approval_id=sent["approval_id"],
            invoice_id=sent["invoice_id"],
        )
        sent_outcome = await _sent_reconciliation_service(
            connection, schema, sent_gateway
        ).reconcile(
            approval_id=sent["approval_id"],
            idempotency_key="sent-reconcile",
            actor="Juan",
        )
        service = CommercialBillingInvoiceGmailSentReconciliationService(
            pool=_SchemaPool(connection, schema),
            gateway_loader=lambda: (_ for _ in ()).throw(
                AssertionError("durable delivery-state reads must not load Gmail")
            ),
        )

        page = await service.list_delivery_state_for_run(
            billing_run_id=missing["run_id"]
        )
        by_approval = {item["approval"]["id"]: item for item in page["items"]}
        missing_item = by_approval[str(missing["approval_id"])]
        sent_item = by_approval[str(sent["approval_id"])]

        assert missing_outcome["outcome"] == "draft_missing"
        assert missing_item["deliveryState"] == "gmail_draft_missing"
        assert missing_item["reconciliation"]["state"] == "draft_missing"
        assert missing_item["invoice"]["status"] == "draft"
        assert missing_item["invoice"]["sentAt"] is None
        assert sent_outcome["outcome"] == "sent_confirmed"
        assert sent_item["deliveryState"] == "gmail_sent_confirmed"
        assert sent_item["reconciliation"]["state"] == "sent_confirmed"
        assert sent_item["invoice"]["status"] == "sent"
        assert sent_item["invoice"]["sentVia"] == "gmail"
        assert sent_item["invoice"]["sentAt"] is not None

        await connection.execute(
            "UPDATE invoices SET status = 'sent', sent_at = $2, sent_via = 'gmail' WHERE id = $1",
            missing["invoice_id"],
            datetime(2026, 4, 4, tzinfo=timezone.utc),
        )
        conflict_page = await service.list_delivery_state_for_run(
            billing_run_id=missing["run_id"]
        )
        conflict_by_approval = {
            item["approval"]["id"]: item for item in conflict_page["items"]
        }
        assert (
            conflict_by_approval[str(missing["approval_id"])]["deliveryState"]
            == "lifecycle_conflict"
        )
        assert (
            "recoveryAction"
            not in conflict_by_approval[str(missing["approval_id"])]["reconciliation"]
        )


@pytest.mark.asyncio
async def test_real_postgres_delivery_state_omits_reconciliation_actions_for_all_lifecycle_conflicts():
    async with _gmail_draft_database() as (connection, schema):
        seed = await _seed_approved_invoice(connection, schema)
        await _service(connection, schema, _RecordingGateway()).create_or_reuse(
            approval_id=seed["approval_id"],
            idempotency_key="sent-before-reconciliation",
            actor="Juan",
        )
        await connection.execute(
            "UPDATE invoices SET status = 'sent', sent_at = $2, sent_via = 'gmail' WHERE id = $1",
            seed["invoice_id"],
            datetime(2026, 4, 4, tzinfo=timezone.utc),
        )
        service = CommercialBillingInvoiceGmailSentReconciliationService(
            pool=_SchemaPool(connection, schema),
            gateway_loader=lambda: (_ for _ in ()).throw(
                AssertionError("durable delivery-state reads must not load Gmail")
            ),
        )

        page = await service.list_delivery_state_for_run(
            billing_run_id=seed["run_id"]
        )

        item, = page["items"]
        assert item["deliveryState"] == "lifecycle_conflict"
        assert item["gmailDraft"]["state"] == "draft_created"
        assert item["reconciliation"]["state"] == "not_reconciled"
        assert "recoveryAction" not in item["reconciliation"]


@pytest.mark.asyncio
async def test_real_postgres_delivery_state_marks_a_stale_pdf_as_a_lifecycle_conflict():
    async with _gmail_draft_database() as (connection, schema):
        seed = await _seed_approved_invoice(connection, schema)
        await _service(connection, schema, _RecordingGateway()).create_or_reuse(
            approval_id=seed["approval_id"],
            idempotency_key="stale-pdf-draft",
            actor="Juan",
        )
        await connection.execute(
            "UPDATE invoices SET notes = 'Edited after PDF generation' WHERE id = $1",
            seed["invoice_id"],
        )
        service = CommercialBillingInvoiceGmailSentReconciliationService(
            pool=_SchemaPool(connection, schema),
            gateway_loader=lambda: (_ for _ in ()).throw(
                AssertionError("durable delivery-state reads must not load Gmail")
            ),
        )

        page = await service.list_delivery_state_for_run(
            billing_run_id=seed["run_id"]
        )

        item, = page["items"]
        assert item["deliveryState"] == "lifecycle_conflict"
        assert item["gmailDraft"]["state"] == "draft_created"
        assert item["reconciliation"]["state"] == "not_reconciled"
        assert "recoveryAction" not in item["reconciliation"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("draft_state", "delivery_state"),
    (
        ("creating", "gmail_draft_creating"),
        ("retryable", "gmail_draft_retryable"),
        ("recovery_required", "gmail_draft_recovery_required"),
    ),
)
async def test_real_postgres_delivery_state_omits_sent_reconciliation_until_draft_is_ready(
    draft_state: str, delivery_state: str
):
    async with _gmail_draft_database() as (connection, schema):
        seed = await _seed_approved_invoice(connection, schema)
        await _service(connection, schema, _RecordingGateway()).create_or_reuse(
            approval_id=seed["approval_id"],
            idempotency_key=f"unfinished-draft-{draft_state}",
            actor="Juan",
        )
        await connection.execute(
            """
            UPDATE commercial_billing_invoice_gmail_drafts
            SET state = $2::varchar,
                gmail_draft_id = NULL,
                gmail_message_id = NULL,
                gmail_thread_id = NULL,
                draft_created_at = NULL,
                recovery_required_at = CASE
                    WHEN $2::text = 'recovery_required' THEN created_at
                    ELSE NULL
                END,
                reconciliation_state = 'not_reconciled'
            WHERE approval_id = $1
            """,
            seed["approval_id"],
            draft_state,
        )
        service = CommercialBillingInvoiceGmailSentReconciliationService(
            pool=_SchemaPool(connection, schema),
            gateway_loader=lambda: (_ for _ in ()).throw(
                AssertionError("durable delivery-state reads must not load Gmail")
            ),
        )

        page = await service.list_delivery_state_for_run(
            billing_run_id=seed["run_id"]
        )

        item, = page["items"]
        assert item["deliveryState"] == delivery_state
        assert item["gmailDraft"]["state"] == draft_state
        assert item["reconciliation"]["state"] == "not_reconciled"
        assert "recoveryAction" not in item["reconciliation"]


@pytest.mark.asyncio
async def test_real_postgres_delivery_state_rejects_unknown_run_without_loading_gmail():
    async with _gmail_draft_database() as (connection, schema):
        gateway_loads = 0

        def failing_gateway_loader():
            nonlocal gateway_loads
            gateway_loads += 1
            raise AssertionError("durable delivery-state reads must not load Gmail")

        service = CommercialBillingInvoiceGmailSentReconciliationService(
            pool=_SchemaPool(connection, schema), gateway_loader=failing_gateway_loader
        )
        with pytest.raises(CommercialBillingGmailDeliveryStateNotFoundError):
            await service.list_delivery_state_for_run(billing_run_id=uuid4())
        assert gateway_loads == 0


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
async def test_real_postgres_reconciles_verifiable_sent_mail_once_and_replays_it():
    async with _gmail_draft_database() as (connection, schema):
        seed = await _seed_approved_invoice(connection, schema)
        gateway = _RecordingGateway()
        draft = await _service(connection, schema, gateway).create_or_reuse(
            approval_id=seed["approval_id"], idempotency_key="gmail-for-sent", actor="Juan"
        )
        rfc_message_id = draft["draft"]["rfcMessageId"]
        gateway.record_sent(
            rfc_message_id=rfc_message_id,
            approval_id=seed["approval_id"],
            invoice_id=seed["invoice_id"],
        )
        service = _sent_reconciliation_service(connection, schema, gateway)

        reconciled = await service.reconcile(
            approval_id=seed["approval_id"], idempotency_key="sent-reconcile-1", actor="Juan"
        )
        replayed = await service.reconcile(
            approval_id=seed["approval_id"], idempotency_key="sent-reconcile-1", actor="Mayra"
        )

        assert reconciled["outcome"] == "sent_confirmed"
        assert reconciled["replayed"] is False
        assert reconciled["reused"] is False
        assert reconciled["reconciliation"]["state"] == "sent_confirmed"
        assert reconciled["reconciliation"]["gmailSentMessageId"] == "sent-message-1"
        assert reconciled["reconciliation"]["gmailSentThreadId"] == "sent-thread-1"
        assert reconciled["reconciliation"]["sentReconciledBy"] == "Juan"
        assert reconciled["reconciliation"]["recoveryAction"] == "none"
        assert replayed["outcome"] == "sent_confirmed"
        assert replayed["replayed"] is True
        assert replayed["reused"] is True
        assert len(gateway.sent_lookup_calls) == 1
        assert gateway.lookup_calls == []

        invoice = await connection.fetchrow(
            "SELECT status, sent_at, sent_via FROM invoices WHERE id = $1", seed["invoice_id"]
        )
        assert invoice["status"] == "sent"
        assert invoice["sent_via"] == "gmail"
        assert invoice["sent_at"] == datetime.fromtimestamp(
            1775088000, tz=timezone.utc
        ).replace(microsecond=123000)
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_gmail_sent_reconciliation_operations"
        ) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("remove_draft", "expected_state", "expected_recovery_action"),
    (
        (False, "draft_present", "wait_for_operator_send"),
        (True, "draft_missing", "review_missing_draft"),
    ),
)
async def test_sent_reconciliation_retains_nonfinancial_draft_observations(
    remove_draft: bool, expected_state: str, expected_recovery_action: str
):
    async with _gmail_draft_database() as (connection, schema):
        seed = await _seed_approved_invoice(connection, schema)
        gateway = _RecordingGateway()
        draft = await _service(connection, schema, gateway).create_or_reuse(
            approval_id=seed["approval_id"], idempotency_key="gmail-observation", actor="Juan"
        )
        if remove_draft:
            gateway.drafts.pop(draft["draft"]["rfcMessageId"])

        result = await _sent_reconciliation_service(connection, schema, gateway).reconcile(
            approval_id=seed["approval_id"],
            idempotency_key=f"sent-observation-{expected_state}",
            actor="Mayra",
        )

        assert result["outcome"] == expected_state
        assert result["reconciliation"]["state"] == expected_state
        assert result["reconciliation"]["recoveryAction"] == expected_recovery_action
        assert result["reconciliation"]["gmailSentMessageId"] is None
        if remove_draft:
            assert result["reconciliation"]["draftMissingAt"] is not None
        else:
            assert result["reconciliation"]["draftMissingAt"] is None
        invoice = await connection.fetchrow(
            "SELECT status, sent_at, sent_via FROM invoices WHERE id = $1", seed["invoice_id"]
        )
        assert dict(invoice) == {"status": "draft", "sent_at": None, "sent_via": None}


@pytest.mark.asyncio
async def test_missing_gmail_draft_replacement_is_audited_idempotent_and_requires_new_sent_proof():
    async with _gmail_draft_database() as (connection, schema):
        seed = await _seed_approved_invoice(connection, schema)
        gateway = _RecordingGateway()
        original = await _create_and_reconcile_missing_draft(
            connection,
            schema,
            seed=seed,
            gateway=gateway,
            draft_key="replacement-original",
        )
        service = _service(connection, schema, gateway)

        replaced = await service.replace_missing(
            approval_id=seed["approval_id"],
            idempotency_key="replacement-1",
            actor="Juan",
        )
        replayed = await service.replace_missing(
            approval_id=seed["approval_id"],
            idempotency_key="replacement-1",
            actor="Mayra",
        )
        ordinary_reuse = await service.create_or_reuse(
            approval_id=seed["approval_id"],
            idempotency_key="normal-after-replacement",
            actor="Mayra",
        )

        replacement = replaced["replacement"]
        assert replaced["replayed"] is False
        assert replaced["reused"] is False
        assert replaced["draft"]["state"] == "draft_created"
        assert replacement["priorGeneration"] == 1
        assert replacement["replacementGeneration"] == 2
        assert replacement["replacedBy"] == "Juan"
        assert replacement["replacedAt"]
        assert replaced["draft"]["rfcMessageId"] != original["draft"]["rfcMessageId"]
        assert "-g2@effinghamofficemaids.com>" in replaced["draft"]["rfcMessageId"]
        assert replayed["replayed"] is True
        assert replayed["reused"] is True
        assert replayed["draft"] == replaced["draft"]
        assert replayed["replacement"] == replacement
        assert ordinary_reuse["replayed"] is False
        assert ordinary_reuse["reused"] is True
        assert ordinary_reuse["draft"] == replaced["draft"]
        assert len(gateway.create_calls) == 2

        root = await connection.fetchrow(
            """
            SELECT state, draft_generation, rfc_message_id, reconciliation_state,
                   gmail_draft_id, gmail_message_id, gmail_thread_id,
                   last_replaced_by, last_replaced_at
            FROM commercial_billing_invoice_gmail_drafts
            WHERE approval_id = $1
            """,
            seed["approval_id"],
        )
        assert dict(root) == {
            "state": "draft_created",
            "draft_generation": 2,
            "rfc_message_id": replaced["draft"]["rfcMessageId"],
            "reconciliation_state": "not_reconciled",
            "gmail_draft_id": "draft-2",
            "gmail_message_id": "message-2",
            "gmail_thread_id": "thread-2",
            "last_replaced_by": "Juan",
            "last_replaced_at": datetime(2026, 4, 2, tzinfo=timezone.utc),
        }
        event = await connection.fetchrow(
            """
            SELECT prior_generation, replacement_generation, prior_snapshot,
                   replaced_by
            FROM commercial_billing_invoice_gmail_draft_replacement_events
            """
        )
        snapshot = event["prior_snapshot"]
        if isinstance(snapshot, str):
            snapshot = json.loads(snapshot)
        assert isinstance(snapshot, dict)
        assert event["prior_generation"] == 1
        assert event["replacement_generation"] == 2
        assert event["replaced_by"] == "Juan"
        assert snapshot["draft_generation"] == 1
        assert snapshot["rfc_message_id"] == original["draft"]["rfcMessageId"]
        assert snapshot["state"] == "draft_created"
        assert snapshot["reconciliation_state"] == "draft_missing"
        assert snapshot["gmail_draft_id"] == "draft-1"
        assert snapshot["gmail_message_id"] == "message-1"
        assert snapshot["gmail_thread_id"] == "thread-1"
        invoice = await connection.fetchrow(
            "SELECT status, sent_at, sent_via FROM invoices WHERE id = $1",
            seed["invoice_id"],
        )
        assert dict(invoice) == {"status": "draft", "sent_at": None, "sent_via": None}

        gateway.record_sent(
            rfc_message_id=replaced["draft"]["rfcMessageId"],
            approval_id=seed["approval_id"],
            invoice_id=seed["invoice_id"],
        )
        sent = await _sent_reconciliation_service(
            connection, schema, gateway
        ).reconcile(
            approval_id=seed["approval_id"],
            idempotency_key="replacement-sent-proof",
            actor="Juan",
        )
        assert sent["outcome"] == "sent_confirmed"
        assert (
            sent["reconciliation"]["rfcMessageId"] == replaced["draft"]["rfcMessageId"]
        )
        invoice = await connection.fetchrow(
            "SELECT status, sent_at, sent_via FROM invoices WHERE id = $1",
            seed["invoice_id"],
        )
        assert invoice["status"] == "sent"
        assert invoice["sent_via"] == "gmail"


@pytest.mark.asyncio
async def test_missing_gmail_draft_replacement_rejects_stale_generation_replay_before_gmail():
    async with _gmail_draft_database() as (connection, schema):
        seed = await _seed_approved_invoice(connection, schema)
        gateway = _RecordingGateway()
        await _create_and_reconcile_missing_draft(
            connection,
            schema,
            seed=seed,
            gateway=gateway,
            draft_key="replacement-stale-original",
        )
        service = _service(connection, schema, gateway)

        generation_two = await service.replace_missing(
            approval_id=seed["approval_id"],
            idempotency_key="replacement-generation-two",
            actor="Juan",
        )
        gateway.drafts.pop(generation_two["draft"]["rfcMessageId"])
        missing = await _sent_reconciliation_service(
            connection,
            schema,
            gateway,
        ).reconcile(
            approval_id=seed["approval_id"],
            idempotency_key="replacement-generation-two-missing",
            actor="Mayra",
        )
        assert missing["outcome"] == "draft_missing"

        generation_three = await service.replace_missing(
            approval_id=seed["approval_id"],
            idempotency_key="replacement-generation-three",
            actor="Mayra",
        )
        before_create_calls = len(gateway.create_calls)
        before_lookup_calls = list(gateway.lookup_calls)

        with pytest.raises(
            CommercialBillingGmailDraftConflictError,
            match="replacement replay is stale",
        ):
            await service.replace_missing(
                approval_id=seed["approval_id"],
                idempotency_key="replacement-generation-two",
                actor="Juan",
            )

        assert generation_three["draft"]["state"] == "draft_created"
        assert generation_three["replacement"]["replacementGeneration"] == 3
        assert len(gateway.create_calls) == before_create_calls
        assert gateway.lookup_calls == before_lookup_calls
        root = await connection.fetchrow(
            """
            SELECT draft_generation, rfc_message_id
            FROM commercial_billing_invoice_gmail_drafts
            WHERE approval_id = $1
            """,
            seed["approval_id"],
        )
        assert dict(root) == {
            "draft_generation": 3,
            "rfc_message_id": generation_three["draft"]["rfcMessageId"],
        }
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_gmail_draft_replacement_events"
        ) == 2


@pytest.mark.asyncio
async def test_missing_gmail_draft_replacement_definite_failure_retries_the_same_generation():
    async with _gmail_draft_database() as (connection, schema):
        seed = await _seed_approved_invoice(connection, schema)
        gateway = _RecordingGateway()
        await _create_and_reconcile_missing_draft(
            connection,
            schema,
            seed=seed,
            gateway=gateway,
            draft_key="replacement-definite-original",
        )
        service = _service(connection, schema, gateway)
        gateway.create_error = GmailDraftCreateError(
            "synthetic replacement rejection",
            definitely_not_created=True,
        )

        with pytest.raises(CommercialBillingGmailDraftUnavailableError):
            await service.replace_missing(
                approval_id=seed["approval_id"],
                idempotency_key="replacement-definite",
                actor="Juan",
            )
        retryable = await connection.fetchrow(
            """
            SELECT state, draft_generation, rfc_message_id, reconciliation_state
            FROM commercial_billing_invoice_gmail_drafts
            WHERE approval_id = $1
            """,
            seed["approval_id"],
        )
        assert dict(retryable) == {
            "state": "retryable",
            "draft_generation": 2,
            "rfc_message_id": gateway.create_calls[-1]["headers"]["Message-ID"],
            "reconciliation_state": "not_reconciled",
        }
        gateway.create_error = None

        retried = await service.replace_missing(
            approval_id=seed["approval_id"],
            idempotency_key="replacement-definite",
            actor="Mayra",
        )
        assert retried["replayed"] is True
        assert retried["reused"] is False
        assert retried["draft"]["state"] == "draft_created"
        assert retried["draft"]["rfcMessageId"] == retryable["rfc_message_id"]
        assert retried["replacement"]["replacementGeneration"] == 2
        assert len(gateway.create_calls) == 3
        assert (
            gateway.create_calls[-1]["headers"]["Message-ID"]
            == retryable["rfc_message_id"]
        )
        assert (
            await connection.fetchval(
                "SELECT COUNT(*) FROM commercial_billing_invoice_gmail_draft_replacement_events"
            )
            == 1
        )


@pytest.mark.asyncio
async def test_missing_gmail_draft_replacement_definite_failure_wins_over_duplicate_recovery():
    async with _gmail_draft_database() as (connection, schema):
        seed = await _seed_approved_invoice(connection, schema)
        original_gateway = _RecordingGateway()
        await _create_and_reconcile_missing_draft(
            connection,
            schema,
            seed=seed,
            gateway=original_gateway,
            draft_key="replacement-definite-overlap-original",
        )
        gateway = _BlockingGateway()
        gateway.create_error = GmailDraftCreateError(
            "synthetic replacement rejection", definitely_not_created=True
        )
        service = _service(connection, schema, gateway)
        first = asyncio.create_task(
            service.replace_missing(
                approval_id=seed["approval_id"],
                idempotency_key="replacement-definite-overlap",
                actor="Juan",
            )
        )
        await asyncio.wait_for(gateway.create_started.wait(), timeout=2)

        with pytest.raises(CommercialBillingGmailDraftRecoveryRequiredError):
            await service.replace_missing(
                approval_id=seed["approval_id"],
                idempotency_key="replacement-definite-overlap",
                actor="Mayra",
            )
        assert (
            await connection.fetchval(
                "SELECT state FROM commercial_billing_invoice_gmail_drafts"
            )
            == "recovery_required"
        )

        gateway.release_create.set()
        with pytest.raises(CommercialBillingGmailDraftUnavailableError):
            await first
        assert (
            await connection.fetchval(
                "SELECT state FROM commercial_billing_invoice_gmail_drafts"
            )
            == "retryable"
        )

        gateway.create_error = None
        retried = await service.replace_missing(
            approval_id=seed["approval_id"],
            idempotency_key="replacement-definite-overlap",
            actor="Juan",
        )
        assert retried["draft"]["state"] == "draft_created"
        assert len(gateway.create_calls) == 2
        assert len(gateway.lookup_calls) == 1


@pytest.mark.asyncio
async def test_missing_gmail_draft_replacement_blocks_invoice_mutations_until_creation(
    monkeypatch,
):
    asyncpg = pytest.importorskip("asyncpg")
    async with _gmail_draft_database() as (connection, schema):
        seed = await _seed_approved_invoice(connection, schema)
        original_gateway = _RecordingGateway()
        await _create_and_reconcile_missing_draft(
            connection,
            schema,
            seed=seed,
            gateway=original_gateway,
            draft_key="replacement-invoice-fence-original",
        )
        gateway = _BlockingGateway()
        service = _service(connection, schema, gateway)
        original_context = service._current_context
        replacement_context_started = asyncio.Event()
        release_replacement_context = asyncio.Event()

        async def block_replacement_context(*args, **kwargs):
            replacement_context_started.set()
            await release_replacement_context.wait()
            return await original_context(*args, **kwargs)

        monkeypatch.setattr(service, "_current_context", block_replacement_context)
        writer = await asyncpg.connect(os.environ["ATLAS_RECEIVABLES_TEST_DATABASE_URL"])
        await writer.execute(f'SET search_path TO "{schema}"')
        first = None
        invoice_update = None
        try:
            first = asyncio.create_task(
                service.replace_missing(
                    approval_id=seed["approval_id"],
                    idempotency_key="replacement-invoice-fence",
                    actor="Juan",
                )
            )
            await asyncio.wait_for(replacement_context_started.wait(), timeout=2)

            invoice_update = asyncio.create_task(
                writer.execute(
                    "UPDATE invoices SET status = 'sent' WHERE id = $1",
                    seed["invoice_id"],
                )
            )
            await asyncio.sleep(0.05)
            assert not invoice_update.done()

            release_replacement_context.set()
            await asyncio.wait_for(gateway.create_started.wait(), timeout=2)
            with pytest.raises(
                asyncpg.CheckViolationError,
                match="Gmail draft replacement is pending",
            ):
                await asyncio.wait_for(invoice_update, timeout=2)
            with pytest.raises(
                asyncpg.CheckViolationError,
                match="Gmail draft replacement is pending",
            ):
                await writer.execute(
                    "UPDATE invoices SET notes = 'must remain immutable' WHERE id = $1",
                    seed["invoice_id"],
                )
            invoice = await connection.fetchrow(
                "SELECT status, notes FROM invoices WHERE id = $1", seed["invoice_id"]
            )
            assert dict(invoice) == {
                "status": "draft",
                "notes": "Approved commercial billing candidate",
            }

            gateway.release_create.set()
            completed = await first
            assert completed["draft"]["state"] == "draft_created"
            assert len(gateway.create_calls) == 1
        finally:
            release_replacement_context.set()
            gateway.release_create.set()
            if invoice_update is not None:
                await asyncio.gather(invoice_update, return_exceptions=True)
            if first is not None:
                await asyncio.gather(first, return_exceptions=True)
            await writer.close()


@pytest.mark.asyncio
async def test_missing_gmail_draft_replacement_uncertain_creation_recovers_by_new_message_id():
    async with _gmail_draft_database() as (connection, schema):
        seed = await _seed_approved_invoice(connection, schema)
        gateway = _RecordingGateway()
        await _create_and_reconcile_missing_draft(
            connection,
            schema,
            seed=seed,
            gateway=gateway,
            draft_key="replacement-uncertain-original",
        )
        service = _service(connection, schema, gateway)
        gateway.create_then_raise = True

        with pytest.raises(CommercialBillingGmailDraftRecoveryRequiredError):
            await service.replace_missing(
                approval_id=seed["approval_id"],
                idempotency_key="replacement-uncertain",
                actor="Juan",
            )
        rfc_message_id = await connection.fetchval(
            "SELECT rfc_message_id FROM commercial_billing_invoice_gmail_drafts"
        )
        assert (
            await connection.fetchval(
                "SELECT state FROM commercial_billing_invoice_gmail_drafts"
            )
            == "recovery_required"
        )
        gateway.create_then_raise = False

        recovered = await service.replace_missing(
            approval_id=seed["approval_id"],
            idempotency_key="replacement-uncertain",
            actor="Mayra",
        )
        assert recovered["replayed"] is True
        assert recovered["reused"] is True
        assert recovered["draft"]["state"] == "draft_created"
        assert recovered["draft"]["rfcMessageId"] == rfc_message_id
        assert len(gateway.create_calls) == 2
        assert gateway.lookup_calls[-1] == rfc_message_id
        assert gateway.lookup_calls.count(rfc_message_id) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "precondition",
    ("not_reconciled", "draft_present", "stale_pdf", "sent_invoice"),
)
async def test_missing_gmail_draft_replacement_rejects_unproven_or_stale_inputs(
    precondition: str,
):
    async with _gmail_draft_database() as (connection, schema):
        seed = await _seed_approved_invoice(connection, schema)
        gateway = _RecordingGateway()
        service = _service(connection, schema, gateway)
        if precondition == "not_reconciled":
            await service.create_or_reuse(
                approval_id=seed["approval_id"],
                idempotency_key="replacement-unproven",
                actor="Juan",
            )
        elif precondition == "draft_present":
            draft = await service.create_or_reuse(
                approval_id=seed["approval_id"],
                idempotency_key="replacement-present",
                actor="Juan",
            )
            assert (
                await _sent_reconciliation_service(
                    connection, schema, gateway
                ).reconcile(
                    approval_id=seed["approval_id"],
                    idempotency_key="replacement-present-reconcile",
                    actor="Juan",
                )
            )["outcome"] == "draft_present"
            assert draft["draft"]["state"] == "draft_created"
        else:
            await _create_and_reconcile_missing_draft(
                connection,
                schema,
                seed=seed,
                gateway=gateway,
                draft_key=f"replacement-{precondition}",
            )
            if precondition == "stale_pdf":
                await connection.execute(
                    "UPDATE invoices SET notes = 'changed after PDF' WHERE id = $1",
                    seed["invoice_id"],
                )
            else:
                await connection.execute(
                    """
                    UPDATE invoices
                       SET status = 'sent', sent_at = $2, sent_via = 'gmail'
                     WHERE id = $1
                    """,
                    seed["invoice_id"],
                    datetime(2026, 4, 2, tzinfo=timezone.utc),
                )

        with pytest.raises(CommercialBillingGmailDraftConflictError):
            await service.replace_missing(
                approval_id=seed["approval_id"],
                idempotency_key=f"replacement-reject-{precondition}",
                actor="Mayra",
            )
        assert len(gateway.create_calls) == 1
        assert (
            await connection.fetchval(
                """
            SELECT COUNT(*) FROM commercial_billing_invoice_gmail_draft_operations
            WHERE source = 'eom_admin_draft_replacement'
            """
            )
            == 0
        )
        assert (
            await connection.fetchval(
                "SELECT COUNT(*) FROM commercial_billing_invoice_gmail_draft_replacement_events"
            )
            == 0
        )


@pytest.mark.asyncio
async def test_concurrent_missing_gmail_draft_replacements_create_one_new_draft():
    async with _gmail_draft_database() as (connection, schema):
        seed = await _seed_approved_invoice(connection, schema)
        original_gateway = _RecordingGateway()
        await _create_and_reconcile_missing_draft(
            connection,
            schema,
            seed=seed,
            gateway=original_gateway,
            draft_key="replacement-concurrent-original",
        )
        gateway = _BlockingGateway()
        service = _service(connection, schema, gateway)
        first = asyncio.create_task(
            service.replace_missing(
                approval_id=seed["approval_id"],
                idempotency_key="replacement-concurrent-1",
                actor="Juan",
            )
        )
        await asyncio.wait_for(gateway.create_started.wait(), timeout=2)
        with pytest.raises(CommercialBillingGmailDraftConflictError):
            await service.replace_missing(
                approval_id=seed["approval_id"],
                idempotency_key="replacement-concurrent-2",
                actor="Mayra",
            )
        assert len(gateway.create_calls) == 1
        gateway.release_create.set()
        completed = await first

        assert completed["draft"]["state"] == "draft_created"
        assert completed["replacement"]["replacementGeneration"] == 2
        assert (
            await connection.fetchval(
                "SELECT COUNT(*) FROM commercial_billing_invoice_gmail_draft_replacement_events"
            )
            == 1
        )
        assert (
            await connection.fetchval(
                """
            SELECT COUNT(*) FROM commercial_billing_invoice_gmail_draft_operations
            WHERE source = 'eom_admin_draft_replacement'
            """
            )
            == 1
        )


@pytest.mark.asyncio
async def test_missing_gmail_draft_replacement_waits_for_pending_sent_mail_reconciliation():
    async with _gmail_draft_database() as (connection, schema):
        seed = await _seed_approved_invoice(connection, schema)
        original_gateway = _RecordingGateway()
        await _create_and_reconcile_missing_draft(
            connection,
            schema,
            seed=seed,
            gateway=original_gateway,
            draft_key="replacement-pending-original",
        )
        gateway = _BlockingSentGateway()
        reconciliation = _sent_reconciliation_service(connection, schema, gateway)
        pending = asyncio.create_task(
            reconciliation.reconcile(
                approval_id=seed["approval_id"],
                idempotency_key="replacement-pending-reconcile",
                actor="Mayra",
            )
        )
        await asyncio.wait_for(gateway.sent_lookup_started.wait(), timeout=2)

        with pytest.raises(CommercialBillingGmailDraftConflictError):
            await _service(connection, schema, gateway).replace_missing(
                approval_id=seed["approval_id"],
                idempotency_key="replacement-pending",
                actor="Juan",
            )
        root = await connection.fetchrow(
            """
            SELECT draft_generation, reconciliation_state
            FROM commercial_billing_invoice_gmail_drafts
            WHERE approval_id = $1
            """,
            seed["approval_id"],
        )
        assert dict(root) == {
            "draft_generation": 1,
            "reconciliation_state": "draft_missing",
        }
        assert gateway.create_calls == []
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_gmail_draft_replacement_events"
        ) == 0

        gateway.release_first_lookup.set()
        completed = await pending
        assert completed["outcome"] == "draft_missing"
        replaced = await _service(connection, schema, gateway).replace_missing(
            approval_id=seed["approval_id"],
            idempotency_key="replacement-after-pending",
            actor="Juan",
        )
        assert replaced["draft"]["state"] == "draft_created"
        assert replaced["replacement"]["replacementGeneration"] == 2
        assert len(gateway.create_calls) == 1


@pytest.mark.asyncio
async def test_sent_reconciliation_rejects_mismatched_or_ambiguous_evidence_without_financial_write():
    async with _gmail_draft_database() as (connection, schema):
        seed = await _seed_approved_invoice(connection, schema)
        gateway = _RecordingGateway()
        draft = await _service(connection, schema, gateway).create_or_reuse(
            approval_id=seed["approval_id"], idempotency_key="gmail-bad-sent", actor="Juan"
        )
        rfc_message_id = draft["draft"]["rfcMessageId"]
        gateway.record_sent(
            rfc_message_id=rfc_message_id,
            approval_id=seed["approval_id"],
            invoice_id=uuid4(),
        )
        service = _sent_reconciliation_service(connection, schema, gateway)

        with pytest.raises(CommercialBillingGmailSentReconciliationConflictError):
            await service.reconcile(
                approval_id=seed["approval_id"],
                idempotency_key="sent-mismatched", actor="Juan"
            )
        gateway.sent_lookup_error = GmailSentMessageLookupError(
            "synthetic multiple sent messages"
        )
        with pytest.raises(CommercialBillingGmailSentReconciliationConflictError):
            await service.reconcile(
                approval_id=seed["approval_id"],
                idempotency_key="sent-ambiguous", actor="Juan"
            )

        invoice = await connection.fetchrow(
            "SELECT status, sent_at, sent_via FROM invoices WHERE id = $1", seed["invoice_id"]
        )
        assert dict(invoice) == {"status": "draft", "sent_at": None, "sent_via": None}
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_gmail_sent_reconciliation_operations "
            "WHERE state = 'completed'"
        ) == 0


@pytest.mark.asyncio
async def test_sent_reconciliation_retries_a_read_failure_with_the_same_key():
    async with _gmail_draft_database() as (connection, schema):
        seed = await _seed_approved_invoice(connection, schema)
        gateway = _RecordingGateway()
        draft = await _service(connection, schema, gateway).create_or_reuse(
            approval_id=seed["approval_id"], idempotency_key="gmail-retry-sent", actor="Juan"
        )
        gateway.sent_lookup_error = RuntimeError("synthetic Gmail timeout")
        service = _sent_reconciliation_service(connection, schema, gateway)

        with pytest.raises(CommercialBillingGmailSentReconciliationUnavailableError):
            await service.reconcile(
                approval_id=seed["approval_id"], idempotency_key="sent-retry", actor="Juan"
            )
        gateway.sent_lookup_error = None
        gateway.record_sent(
            rfc_message_id=draft["draft"]["rfcMessageId"],
            approval_id=seed["approval_id"],
            invoice_id=seed["invoice_id"],
        )

        result = await service.reconcile(
            approval_id=seed["approval_id"], idempotency_key="sent-retry", actor="Juan"
        )
        assert result["outcome"] == "sent_confirmed"
        assert result["replayed"] is True
        assert len(gateway.sent_lookup_calls) == 2


@pytest.mark.asyncio
async def test_concurrent_sent_reconciliation_requests_have_one_financial_transition():
    async with _gmail_draft_database() as (connection, schema):
        seed = await _seed_approved_invoice(connection, schema)
        gateway = _BlockingSentGateway()
        draft = await _service(connection, schema, gateway).create_or_reuse(
            approval_id=seed["approval_id"], idempotency_key="gmail-concurrent-sent", actor="Juan"
        )
        gateway.record_sent(
            rfc_message_id=draft["draft"]["rfcMessageId"],
            approval_id=seed["approval_id"],
            invoice_id=seed["invoice_id"],
        )
        await connection.execute(
            "CREATE TABLE invoice_status_changes (old_status text, new_status text)"
        )
        await connection.execute(
            """
            CREATE FUNCTION capture_invoice_status_change()
            RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
                IF OLD.status IS DISTINCT FROM NEW.status THEN
                    INSERT INTO invoice_status_changes (old_status, new_status)
                    VALUES (OLD.status, NEW.status);
                END IF;
                RETURN NEW;
            END;
            $$
            """
        )
        await connection.execute(
            """
            CREATE TRIGGER capture_invoice_status_change_trigger
            BEFORE UPDATE ON invoices
            FOR EACH ROW EXECUTE FUNCTION capture_invoice_status_change()
            """
        )
        service = _sent_reconciliation_service(connection, schema, gateway)
        first = asyncio.create_task(
            service.reconcile(
                approval_id=seed["approval_id"], idempotency_key="sent-concurrent-1", actor="Juan"
            )
        )
        await asyncio.wait_for(gateway.sent_lookup_started.wait(), timeout=2)
        second = await service.reconcile(
            approval_id=seed["approval_id"], idempotency_key="sent-concurrent-2", actor="Mayra"
        )
        gateway.release_first_lookup.set()
        first_result = await first

        assert first_result["outcome"] == "sent_confirmed"
        assert second["outcome"] == "sent_confirmed"
        assert len(gateway.create_calls) == 1
        assert len(gateway.sent_lookup_calls) == 2
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM invoice_status_changes "
            "WHERE old_status = 'draft' AND new_status = 'sent'"
        ) == 1


@pytest.mark.asyncio
async def test_sent_reconciliation_rejects_a_stale_invoice_after_gmail_lookup():
    async with _gmail_draft_database() as (connection, schema):
        seed = await _seed_approved_invoice(connection, schema)
        gateway = _BlockingSentGateway()
        draft = await _service(connection, schema, gateway).create_or_reuse(
            approval_id=seed["approval_id"],
            idempotency_key="gmail-stale-sent",
            actor="Juan",
        )
        gateway.record_sent(
            rfc_message_id=draft["draft"]["rfcMessageId"],
            approval_id=seed["approval_id"],
            invoice_id=seed["invoice_id"],
        )
        reconciliation = _sent_reconciliation_service(connection, schema, gateway)
        task = asyncio.create_task(
            reconciliation.reconcile(
                approval_id=seed["approval_id"],
                idempotency_key="sent-stale",
                actor="Juan",
            )
        )
        await asyncio.wait_for(gateway.sent_lookup_started.wait(), timeout=2)
        await connection.execute(
            "UPDATE invoices SET status = 'void' WHERE id = $1", seed["invoice_id"]
        )
        gateway.release_first_lookup.set()

        with pytest.raises(CommercialBillingGmailSentReconciliationConflictError):
            await task

        invoice = await connection.fetchrow(
            "SELECT status, sent_at, sent_via FROM invoices WHERE id = $1",
            seed["invoice_id"],
        )
        assert dict(invoice) == {"status": "void", "sent_at": None, "sent_via": None}
        assert await connection.fetchval(
            "SELECT reconciliation_state FROM commercial_billing_invoice_gmail_drafts"
        ) == "not_reconciled"
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_gmail_sent_reconciliation_operations "
            "WHERE state = 'completed'"
        ) == 0


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


@pytest.mark.asyncio
async def test_gmail_transport_reads_one_sent_message_with_metadata_and_never_sends():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/users/me/messages"):
            return httpx.Response(
                200,
                json={"messages": [{"id": "sent-1", "threadId": "thread-1"}]},
            )
        if request.url.path.endswith("/users/me/messages/sent-1"):
            return httpx.Response(
                200,
                json={
                    "id": "sent-1",
                    "threadId": "thread-1",
                    "labelIds": ["SENT"],
                    "internalDate": "1775088000123",
                    "payload": {
                        "headers": [
                            {"name": "Message-ID", "value": "<stable@example.test>"},
                            {
                                "name": "X-Atlas-Commercial-Billing-Approval",
                                "value": "approval-1",
                            },
                            {
                                "name": "X-Atlas-Commercial-Billing-Invoice",
                                "value": "invoice-1",
                            },
                        ]
                    },
                },
            )
        return httpx.Response(404)

    transport = GmailTransport()
    transport._access_token = "token"
    transport._token_expires = time.time() + 600
    transport._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        found = await transport.find_sent_message_by_rfc_message_id(
            "<stable@example.test>"
        )
    finally:
        await transport.close()

    assert found == {
        "id": "sent-1",
        "threadId": "thread-1",
        "labelIds": ["SENT"],
        "internalDate": "1775088000123",
        "headers": [
            {"name": "Message-ID", "value": "<stable@example.test>"},
            {"name": "X-Atlas-Commercial-Billing-Approval", "value": "approval-1"},
            {"name": "X-Atlas-Commercial-Billing-Invoice", "value": "invoice-1"},
        ],
    }
    assert [request.method for request in requests] == ["GET", "GET"]
    assert "/messages/send" not in " ".join(request.url.path for request in requests)
    search = requests[0]
    assert search.url.params["q"] == "rfc822msgid:<stable@example.test>"
    assert search.url.params["labelIds"] == "SENT"
    metadata_headers = requests[1].url.params.get_list("metadataHeaders")
    assert metadata_headers == [
        "Message-ID",
        "X-Atlas-Commercial-Billing-Approval",
        "X-Atlas-Commercial-Billing-Invoice",
    ]

    async def ambiguous_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/users/me/messages"):
            return httpx.Response(
                200,
                json={
                    "messages": [
                        {"id": "sent-1", "threadId": "thread-1"},
                        {"id": "sent-2", "threadId": "thread-2"},
                    ]
                },
            )
        message_id = request.url.path.rsplit("/", maxsplit=1)[-1]
        return httpx.Response(
            200,
            json={
                "id": message_id,
                "threadId": f"thread-{message_id[-1]}",
                "labelIds": ["SENT"],
                "internalDate": "1775088000123",
                "payload": {"headers": []},
            },
        )

    transport = GmailTransport()
    transport._access_token = "token"
    transport._token_expires = time.time() + 600
    transport._client = httpx.AsyncClient(
        transport=httpx.MockTransport(ambiguous_handler)
    )
    try:
        with pytest.raises(GmailSentMessageLookupError, match="multiple"):
            await transport.find_sent_message_by_rfc_message_id("<stable@example.test>")
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

    async def replace_missing(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return {
            "draft": {
                "approvalId": str(kwargs["approval_id"]),
                "id": "draft-record-1",
                "state": "draft_created",
            },
            "replayed": False,
            "replacement": {
                "draftId": "draft-record-1",
                "id": "replacement-event-1",
                "priorGeneration": 1,
                "replacementGeneration": 2,
            },
            "reused": False,
        }


class _SentReconciliationRouteService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def reconcile(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return {
            "outcome": "draft_missing",
            "reconciliation": {
                "approvalId": str(kwargs["approval_id"]),
                "state": "draft_missing",
            },
            "replayed": False,
            "reused": False,
        }


class _DeliveryStateRouteService:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.error: Exception | None = None

    async def list_delivery_state_for_run(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {
            "billingRunId": str(kwargs["billing_run_id"]),
            "items": [
                {
                    "approval": {"id": "approval-1"},
                    "deliveryState": "gmail_draft_missing",
                    "pdf": {"id": "artifact-1", "state": "ready"},
                    "reconciliation": {"state": "draft_missing"},
                }
            ],
            "limit": kwargs["limit"],
            "offset": kwargs["offset"],
            "total": 1,
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


@pytest.mark.asyncio
async def test_full_atlas_app_missing_gmail_draft_replacement_route_requires_existing_auth():
    from atlas_brain.api.invoicing import auth as receivables_auth
    from atlas_brain.api.invoicing import receivables as routes
    from atlas_brain.eom_api.auth import generate_receivables_service_token
    from atlas_brain.main import app

    approval_id = uuid4()
    service = _RouteService()
    generated = generate_receivables_service_token()
    original_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[receivables_auth.get_receivables_api_config] = lambda: (
        SimpleNamespace(
            receivables_api_enabled=True,
            receivables_service_token="",
            receivables_service_token_sha256=generated.sha256,
        )
    )
    app.dependency_overrides[
        routes.get_commercial_billing_invoice_gmail_draft_service
    ] = lambda: service
    path = (
        "/api/v1/receivables/commercial-billing-approvals/"
        f"{approval_id}/gmail-draft/replace-missing"
    )
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
                    headers={**headers, "Idempotency-Key": "route-replace-1"},
                )
            ).status_code == 422
            response = await client.post(
                path,
                headers={
                    **headers,
                    "Idempotency-Key": "route-replace-1",
                    "X-EOM-Actor": "Juan",
                },
            )
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)

    assert response.status_code == 201
    assert response.json()["replacement"] == {
        "draftId": "draft-record-1",
        "id": "replacement-event-1",
        "priorGeneration": 1,
        "replacementGeneration": 2,
    }
    assert "pdf_bytes" not in response.text
    assert "pdfBytes" not in response.text
    assert service.calls == [
        {
            "approval_id": approval_id,
            "idempotency_key": "route-replace-1",
            "actor": "Juan",
        }
    ]


@pytest.mark.asyncio
async def test_full_atlas_app_gmail_delivery_state_route_requires_existing_auth_and_returns_no_delivery_side_effect():
    from atlas_brain.api.invoicing import auth as receivables_auth
    from atlas_brain.api.invoicing import receivables as routes
    from atlas_brain.eom_api.auth import generate_receivables_service_token
    from atlas_brain.main import app

    billing_run_id = uuid4()
    service = _DeliveryStateRouteService()
    generated = generate_receivables_service_token()
    original_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[receivables_auth.get_receivables_api_config] = lambda: SimpleNamespace(
        receivables_api_enabled=True,
        receivables_service_token="",
        receivables_service_token_sha256=generated.sha256,
    )
    app.dependency_overrides[
        routes.get_commercial_billing_invoice_gmail_sent_reconciliation_service
    ] = lambda: service
    path = (
        "/api/v1/receivables/commercial-billing-runs/"
        f"{billing_run_id}/gmail-delivery-state"
    )
    headers = {"Authorization": f"Bearer {generated.token}"}
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.get(path)).status_code == 401
            assert (
                await client.get(path, headers={"Authorization": "Bearer wrong"})
            ).status_code == 401
            assert (
                await client.get(f"{path}?limit=0", headers=headers)
            ).status_code == 422
            out_of_range = await client.get(
                f"{path}?offset={MAX_DELIVERY_STATE_OFFSET + 1}", headers=headers
            )
            response = await client.get(f"{path}?limit=1&offset=2", headers=headers)
            service.error = CommercialBillingGmailDeliveryStateNotFoundError(
                "Commercial billing run not found"
            )
            not_found = await client.get(path, headers=headers)
            service.error = CommercialBillingGmailSentReconciliationUnavailableError(
                "Commercial billing database unavailable"
            )
            unavailable = await client.get(path, headers=headers)
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)

    assert response.status_code == 200
    assert out_of_range.status_code == 422
    assert response.json()["billingRunId"] == str(billing_run_id)
    assert response.json()["items"][0]["deliveryState"] == "gmail_draft_missing"
    assert "pdf_bytes" not in response.text.casefold()
    assert "pdfbytes" not in response.text.casefold()
    assert not_found.status_code == 404
    assert not_found.json()["detail"]["code"] == "commercial_billing_run_not_found"
    assert unavailable.status_code == 503
    assert unavailable.json()["detail"]["code"] == (
        "commercial_billing_gmail_sent_reconciliation_unavailable"
    )
    assert service.calls == [
        {"billing_run_id": billing_run_id, "limit": 1, "offset": 2},
        {"billing_run_id": billing_run_id, "limit": 50, "offset": 0},
        {"billing_run_id": billing_run_id, "limit": 50, "offset": 0},
    ]


@pytest.mark.asyncio
async def test_full_atlas_app_gmail_sent_reconciliation_route_requires_existing_auth():
    from atlas_brain.api.invoicing import auth as receivables_auth
    from atlas_brain.api.invoicing import receivables as routes
    from atlas_brain.eom_api.auth import generate_receivables_service_token
    from atlas_brain.main import app

    approval_id = uuid4()
    service = _SentReconciliationRouteService()
    generated = generate_receivables_service_token()
    original_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[receivables_auth.get_receivables_api_config] = lambda: SimpleNamespace(
        receivables_api_enabled=True,
        receivables_service_token="",
        receivables_service_token_sha256=generated.sha256,
    )
    app.dependency_overrides[
        routes.get_commercial_billing_invoice_gmail_sent_reconciliation_service
    ] = lambda: service
    path = (
        "/api/v1/receivables/commercial-billing-approvals/"
        f"{approval_id}/gmail-draft/reconcile"
    )
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
                    headers={**headers, "Idempotency-Key": "route-sent-1"},
                )
            ).status_code == 422
            response = await client.post(
                path,
                headers={
                    **headers,
                    "Idempotency-Key": "route-sent-1",
                    "X-EOM-Actor": "Juan",
                },
            )
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)

    assert response.status_code == 200
    assert response.json()["outcome"] == "draft_missing"
    assert service.calls == [
        {"approval_id": approval_id, "idempotency_key": "route-sent-1", "actor": "Juan"}
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


def test_gmail_sent_reconciliation_migration_is_additive_and_never_marks_an_invoice_sent():
    migration = (
        Path(__file__).parents[1]
        / "atlas_brain/storage/migrations/375_commercial_billing_invoice_gmail_sent_reconciliation.sql"
    ).read_text()
    executable = "\n".join(
        line for line in migration.splitlines() if not line.lstrip().startswith("--")
    )
    assert "ADD COLUMN IF NOT EXISTS reconciliation_state" in migration
    assert "CREATE TABLE IF NOT EXISTS commercial_billing_gmail_sent_reconciliation_operations" in migration
    assert "UNIQUE (source, idempotency_key)" in migration
    assert "'draft_present', 'draft_missing', 'sent_confirmed'" in migration
    assert "UPDATE invoices" not in executable
    assert "DROP TABLE" not in executable
    assert "DROP COLUMN" not in executable


def test_missing_gmail_draft_replacement_migration_is_atomic_append_only_and_nonfinancial():
    migration = (
        Path(__file__).parents[1]
        / "atlas_brain/storage/migrations/377_commercial_billing_gmail_draft_replacements.sql"
    ).read_text()
    executable = "\n".join(
        line for line in migration.splitlines() if not line.lstrip().startswith("--")
    )
    assert migration.splitlines()[0] == "-- atlas: atomic-bookkeeping"
    assert "ADD COLUMN IF NOT EXISTS draft_generation" in migration
    assert (
        "CREATE TABLE commercial_billing_invoice_gmail_draft_replacement_events"
        in migration
    )
    assert "prior_snapshot JSONB NOT NULL" in migration
    assert "UNIQUE (gmail_draft_record_id, replacement_generation)" in migration
    assert "CREATE TRIGGER commercial_billing_reject_invoice_mutation_while_gmail_replacement_pending" in migration
    assert "BEFORE UPDATE ON invoices" in executable
    assert "sent_at" not in executable
    assert "UPDATE invoices" not in executable
    assert "DROP TABLE" not in executable
    assert "DROP COLUMN" not in executable
    assert "DELETE FROM" not in executable


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


def test_gmail_sent_reconciliation_service_only_reads_gmail_and_writes_after_proof():
    import atlas_brain.services.commercial_billing_invoice_gmail_sent_reconciliation as reconciliation

    source = inspect.getsource(reconciliation)
    assert "find_sent_message_by_rfc_message_id" in source
    assert "find_draft_by_rfc_message_id" in source
    assert "create_draft" not in source
    assert ".send(" not in source
    assert "UPDATE invoices" in source
    assert "float(" not in source


def test_invoicing_workflow_enrolls_the_gmail_draft_surface_and_contract():
    workflow = (Path(__file__).parents[1] / ".github/workflows/atlas_invoicing_checks.yml").read_text()
    for path in (
        "atlas_brain/services/commercial_billing_invoice_gmail_drafts.py",
        "atlas_brain/services/commercial_billing_invoice_gmail_sent_reconciliation.py",
        "atlas_brain/storage/migrations/374_commercial_billing_invoice_gmail_drafts.sql",
        "atlas_brain/storage/migrations/375_commercial_billing_invoice_gmail_sent_reconciliation.sql",
        "atlas_brain/storage/migrations/377_commercial_billing_gmail_draft_replacements.sql",
        "atlas_brain/tools/gmail.py",
        "tests/test_commercial_billing_gmail_drafts.py",
    ):
        assert workflow.count(f'      - "{path}"') == 2
    assert "tests/test_commercial_billing_gmail_drafts.py \\" in workflow
