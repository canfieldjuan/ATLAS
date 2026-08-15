"""Contract tests for the durable EOM manual-Square invoice queue."""

from __future__ import annotations

import asyncio
import ast
import hashlib
import inspect
import json
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from atlas_brain.services.commercial_billing_manual_square_invoices import (
    CommercialBillingManualSquareInvoiceConflictError,
    CommercialBillingManualSquareInvoiceService,
    CommercialBillingManualSquareInvoiceValidationError,
    _request_text,
)


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _safe_reference_oracle(value: object) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value.strip()) <= 256
        and "\r" not in value.strip()
        and "\n" not in value.strip()
        and "\x00" not in value.strip()
    )


def _reference_grammar_candidates():
    for token, wrapper, padding in product(
        ("", "SQ-1", "S" * 256, "S" * 257, " ", "SQ\n1", "SQ\r1", "SQ\x001"),
        (
            lambda value: value,
            lambda value: [value],
            lambda value: {"reference": value},
            lambda value: (value,),
        ),
        (
            lambda value: value,
            lambda value: f" {value} ",
            lambda value: f"\t{value}\n",
        ),
    ):
        yield wrapper(padding(token))


@pytest.mark.parametrize("value", tuple(_reference_grammar_candidates()))
def test_manual_square_reference_admission_matches_the_spec_derived_oracle(
    value: object,
):
    if _safe_reference_oracle(value):
        assert (
            _request_text(value, "Square invoice reference", limit=256) == value.strip()
        )
    else:
        with pytest.raises(CommercialBillingManualSquareInvoiceValidationError):
            _request_text(value, "Square invoice reference", limit=256)


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

    async def fetch(self, query, *args):
        return await self.connection.fetch(query, *args)


@asynccontextmanager
async def _manual_square_database():
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")
    schema = f"commercial_manual_square_{uuid4().hex}"
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
            "376_commercial_billing_manual_square_invoices.sql",
        ):
            await connection.execute((migrations / name).read_text())
        yield connection, schema, database_url
    finally:
        await connection.execute("SET search_path TO public")
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


async def _seed_approved_invoice(
    connection,
    *,
    delivery_method: str = "manual_square",
    customer_name: str = "Acme Office",
) -> dict:
    contact_id, run_id, candidate_id, approval_id, invoice_id = (
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
    )
    candidate_key = f"commercial-billing:{customer_name.casefold()}:{approval_id}"
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
            $1, $2, $3, $4, 'billing@example.test', $5::jsonb,
            96.50, 0, 0, 0, 96.50, $6, $7, 'draft',
            'eom_commercial_billing', $8, 'effingham_maids',
            'Approved commercial billing candidate', $9::jsonb,
            'Office cleaning - March 2026', 'Billing Contact', $10, $10
        )
        """,
        invoice_id,
        f"INV-2026-Mar-{str(approval_id).replace('-', '')[:4]}",
        contact_id,
        customer_name,
        json.dumps(
            [
                {
                    "amount": "96.50",
                    "description": "Office cleaning",
                    "quantity": 2,
                    "unit_price": "48.25",
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
    return {"approval_id": approval_id, "invoice_id": invoice_id, "now": now}


def _service(connection, schema: str) -> CommercialBillingManualSquareInvoiceService:
    now = datetime(2026, 4, 2, tzinfo=timezone.utc)
    return CommercialBillingManualSquareInvoiceService(
        pool=_SchemaPool(connection, schema),
        now=lambda: now,
    )


@pytest.mark.asyncio
async def test_real_postgres_lists_only_active_manual_square_work_without_creating_state():
    async with _manual_square_database() as (connection, schema, _database_url):
        manual = await _seed_approved_invoice(connection, customer_name="Acme Office")
        await _seed_approved_invoice(
            connection,
            delivery_method="gmail_pdf",
            customer_name="Gmail Customer",
        )

        result = await _service(connection, schema).list_needs_square_invoices(
            limit=1, offset=0
        )

        assert result["limit"] == 1
        assert result["offset"] == 0
        assert result["total"] == 1
        assert result["items"] == [
            {
                "approvalId": str(manual["approval_id"]),
                "id": None,
                "invoice": {
                    "dueDate": "2026-04-16",
                    "id": str(manual["invoice_id"]),
                    "invoiceNumber": result["items"][0]["invoice"]["invoiceNumber"],
                    "issueDate": "2026-04-02",
                    "sentAt": None,
                    "sentVia": None,
                    "sourceRef": f"approval:{manual['approval_id']}",
                    "status": "draft",
                    "totalCents": 9650,
                },
                "customerName": "Acme Office",
                "referenceRecordedAt": None,
                "referenceRecordedBy": None,
                "sentViaSquareAt": None,
                "sentViaSquareBy": None,
                "squareInvoiceReference": None,
                "state": "needs_square_invoice",
            }
        ]
        assert (
            await connection.fetchval(
                "SELECT COUNT(*) FROM commercial_billing_manual_square_invoices"
            )
            == 0
        )
        assert (
            await connection.fetchval(
                "SELECT COUNT(*) FROM commercial_billing_manual_square_invoice_operations"
            )
            == 0
        )
        assert dict(
            await connection.fetchrow(
                "SELECT status, sent_at, sent_via FROM invoices WHERE id = $1",
                manual["invoice_id"],
            )
        ) == {"status": "draft", "sent_at": None, "sent_via": None}


@pytest.mark.asyncio
async def test_real_postgres_records_one_square_reference_and_replays_without_sending():
    async with _manual_square_database() as (connection, schema, _database_url):
        seed = await _seed_approved_invoice(connection)
        service = _service(connection, schema)

        created = await service.record_reference(
            approval_id=seed["approval_id"],
            square_invoice_reference="SQ-INV-0001",
            idempotency_key="reference-1",
            actor="Juan",
        )
        replayed = await service.record_reference(
            approval_id=seed["approval_id"],
            square_invoice_reference="SQ-INV-0001",
            idempotency_key="reference-1",
            actor="Mayra",
        )
        with pytest.raises(CommercialBillingManualSquareInvoiceConflictError):
            await service.record_reference(
                approval_id=seed["approval_id"],
                square_invoice_reference="SQ-INV-DIFFERENT",
                idempotency_key="reference-1",
                actor="Mayra",
            )
        reused = await service.record_reference(
            approval_id=seed["approval_id"],
            square_invoice_reference="SQ-INV-0001",
            idempotency_key="reference-2",
            actor="Mayra",
        )

        assert created["replayed"] is False
        assert created["reused"] is False
        assert created["manualSquareInvoice"]["state"] == "reference_recorded"
        assert created["manualSquareInvoice"]["squareInvoiceReference"] == "SQ-INV-0001"
        assert created["manualSquareInvoice"]["referenceRecordedBy"] == "Juan"
        assert replayed == {**created, "replayed": True, "reused": True}
        assert reused["replayed"] is False
        assert reused["reused"] is True
        assert reused["manualSquareInvoice"]["referenceRecordedBy"] == "Juan"
        assert (
            await connection.fetchval(
                "SELECT COUNT(*) FROM commercial_billing_manual_square_invoices"
            )
            == 1
        )
        assert (
            await connection.fetchval(
                "SELECT COUNT(*) FROM commercial_billing_manual_square_invoice_operations"
            )
            == 2
        )
        assert dict(
            await connection.fetchrow(
                "SELECT status, sent_at, sent_via FROM invoices WHERE id = $1",
                seed["invoice_id"],
            )
        ) == {"status": "draft", "sent_at": None, "sent_via": None}


@pytest.mark.asyncio
async def test_real_postgres_marks_only_a_referenced_manual_square_draft_sent_once():
    async with _manual_square_database() as (connection, schema, _database_url):
        seed = await _seed_approved_invoice(connection)
        service = _service(connection, schema)

        with pytest.raises(CommercialBillingManualSquareInvoiceConflictError):
            await service.mark_sent(
                approval_id=seed["approval_id"],
                idempotency_key="sent-before-reference",
                actor="Juan",
            )
        await service.record_reference(
            approval_id=seed["approval_id"],
            square_invoice_reference="SQ-INV-0002",
            idempotency_key="reference-1",
            actor="Juan",
        )
        marked = await service.mark_sent(
            approval_id=seed["approval_id"],
            idempotency_key="mark-sent-1",
            actor="Juan",
        )
        replayed = await service.mark_sent(
            approval_id=seed["approval_id"],
            idempotency_key="mark-sent-1",
            actor="Mayra",
        )
        reused = await service.mark_sent(
            approval_id=seed["approval_id"],
            idempotency_key="mark-sent-2",
            actor="Mayra",
        )

        assert marked["replayed"] is False
        assert marked["reused"] is False
        assert marked["manualSquareInvoice"]["state"] == "sent_via_square"
        assert marked["manualSquareInvoice"]["sentViaSquareBy"] == "Juan"
        assert marked["manualSquareInvoice"]["invoice"]["status"] == "sent"
        assert marked["manualSquareInvoice"]["invoice"]["sentVia"] == "square"
        assert replayed == {**marked, "replayed": True, "reused": True}
        assert reused["replayed"] is False
        assert reused["reused"] is True
        invoice = await connection.fetchrow(
            "SELECT status, sent_at, sent_via FROM invoices WHERE id = $1",
            seed["invoice_id"],
        )
        assert dict(invoice) == {
            "status": "sent",
            "sent_at": seed["now"],
            "sent_via": "square",
        }
        record = await connection.fetchrow("""
            SELECT state, square_invoice_reference, reference_recorded_by,
                   sent_via_square_by, sent_via_square_at
            FROM commercial_billing_manual_square_invoices
            """)
        assert dict(record) == {
            "state": "sent_via_square",
            "square_invoice_reference": "SQ-INV-0002",
            "reference_recorded_by": "Juan",
            "sent_via_square_by": "Juan",
            "sent_via_square_at": seed["now"],
        }


@pytest.mark.asyncio
async def test_real_postgres_rejects_invalid_delivery_policy_reference_overwrite_and_stale_lifecycle_without_writes():
    async with _manual_square_database() as (connection, schema, _database_url):
        gmail = await _seed_approved_invoice(connection, delivery_method="gmail_pdf")
        manual = await _seed_approved_invoice(connection)
        service = _service(connection, schema)

        with pytest.raises(CommercialBillingManualSquareInvoiceConflictError):
            await service.record_reference(
                approval_id=gmail["approval_id"],
                square_invoice_reference="SQ-WRONG-POLICY",
                idempotency_key="wrong-policy",
                actor="Juan",
            )
        await service.record_reference(
            approval_id=manual["approval_id"],
            square_invoice_reference="SQ-IMMUTABLE",
            idempotency_key="reference-1",
            actor="Juan",
        )
        with pytest.raises(CommercialBillingManualSquareInvoiceConflictError):
            await service.record_reference(
                approval_id=manual["approval_id"],
                square_invoice_reference="SQ-CHANGED",
                idempotency_key="reference-2",
                actor="Mayra",
            )
        await connection.execute(
            """
            UPDATE invoices
               SET status = 'sent', sent_at = $2, sent_via = 'gmail'
             WHERE id = $1
            """,
            manual["invoice_id"],
            manual["now"],
        )
        with pytest.raises(CommercialBillingManualSquareInvoiceConflictError):
            await service.mark_sent(
                approval_id=manual["approval_id"],
                idempotency_key="stale-mark-sent",
                actor="Juan",
            )
        queue = await service.list_needs_square_invoices()
        assert queue["total"] == 1
        assert queue["items"][0]["approvalId"] == str(manual["approval_id"])
        assert queue["items"][0]["state"] == "lifecycle_conflict"
        assert (
            await connection.fetchval(
                "SELECT COUNT(*) FROM commercial_billing_manual_square_invoice_operations"
            )
            == 1
        )
        assert dict(
            await connection.fetchrow(
                "SELECT status, sent_via FROM invoices WHERE id = $1",
                manual["invoice_id"],
            )
        ) == {"status": "sent", "sent_via": "gmail"}


@pytest.mark.asyncio
async def test_real_postgres_serializes_same_reference_operation_key_across_connections():
    async with _manual_square_database() as (connection, schema, database_url):
        asyncpg = pytest.importorskip("asyncpg")
        seed = await _seed_approved_invoice(connection)
        second_connection = await asyncpg.connect(database_url)
        await second_connection.execute(f'SET search_path TO "{schema}"')
        try:
            first = _service(connection, schema)
            second = _service(second_connection, schema)
            results = await asyncio.gather(
                first.record_reference(
                    approval_id=seed["approval_id"],
                    square_invoice_reference="SQ-CONCURRENT",
                    idempotency_key="same-operation",
                    actor="Juan",
                ),
                second.record_reference(
                    approval_id=seed["approval_id"],
                    square_invoice_reference="SQ-CONCURRENT",
                    idempotency_key="same-operation",
                    actor="Mayra",
                ),
            )
        finally:
            await second_connection.close()

        assert sorted(result["replayed"] for result in results) == [False, True]
        assert all(
            result["manualSquareInvoice"]["squareInvoiceReference"] == "SQ-CONCURRENT"
            for result in results
        )
        assert (
            await connection.fetchval(
                "SELECT COUNT(*) FROM commercial_billing_manual_square_invoices"
            )
            == 1
        )
        assert (
            await connection.fetchval(
                "SELECT COUNT(*) FROM commercial_billing_manual_square_invoice_operations"
            )
            == 1
        )


class _RouteService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def list_needs_square_invoices(self, **kwargs) -> dict:
        self.calls.append(("list", kwargs))
        return {
            "items": [],
            "limit": kwargs["limit"],
            "offset": kwargs["offset"],
            "total": 0,
        }

    async def record_reference(self, **kwargs) -> dict:
        self.calls.append(("reference", kwargs))
        return {
            "manualSquareInvoice": {
                "approvalId": str(kwargs["approval_id"]),
                "state": "reference_recorded",
            },
            "replayed": False,
            "reused": False,
        }

    async def mark_sent(self, **kwargs) -> dict:
        self.calls.append(("sent", kwargs))
        return {
            "manualSquareInvoice": {
                "approvalId": str(kwargs["approval_id"]),
                "state": "sent_via_square",
            },
            "replayed": False,
            "reused": False,
        }


@pytest.mark.asyncio
async def test_full_atlas_app_manual_square_routes_reject_missing_auth_actor_and_key():
    from atlas_brain.api.invoicing import auth as receivables_auth
    from atlas_brain.api.invoicing import receivables as routes
    from atlas_brain.eom_api.auth import generate_receivables_service_token
    from atlas_brain.main import app

    approval_id = uuid4()
    service = _RouteService()
    generated = generate_receivables_service_token()
    original_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[receivables_auth.get_receivables_api_config] = (
        lambda: SimpleNamespace(
            receivables_api_enabled=True,
            receivables_service_token="",
            receivables_service_token_sha256=generated.sha256,
        )
    )
    app.dependency_overrides[
        routes.get_commercial_billing_manual_square_invoice_service
    ] = lambda: service
    list_path = "/api/v1/receivables/commercial-billing/manual-square-invoices"
    reference_path = (
        "/api/v1/receivables/commercial-billing-approvals/"
        f"{approval_id}/manual-square-invoice-reference"
    )
    sent_path = (
        "/api/v1/receivables/commercial-billing-approvals/"
        f"{approval_id}/manual-square-invoice/mark-sent"
    )
    headers = {"Authorization": f"Bearer {generated.token}"}
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.get(list_path)).status_code == 401
            listed = await client.get(list_path, headers=headers, params={"limit": 1})
            assert listed.status_code == 200
            assert (await client.post(reference_path)).status_code == 401
            assert (
                await client.post(reference_path, headers=headers)
            ).status_code == 422
            assert (
                await client.post(
                    reference_path,
                    headers={**headers, "Idempotency-Key": "reference-route"},
                    json={"square_invoice_reference": "SQ-ROUTE"},
                )
            ).status_code == 422
            recorded = await client.post(
                reference_path,
                headers={
                    **headers,
                    "Idempotency-Key": "reference-route",
                    "X-EOM-Actor": "Juan",
                },
                json={"square_invoice_reference": "SQ-ROUTE"},
            )
            assert (await client.post(sent_path, headers=headers)).status_code == 422
            sent = await client.post(
                sent_path,
                headers={
                    **headers,
                    "Idempotency-Key": "sent-route",
                    "X-EOM-Actor": "Juan",
                },
            )
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)

    assert listed.json() == {"items": [], "limit": 1, "offset": 0, "total": 0}
    assert recorded.status_code == 201
    assert recorded.json()["manualSquareInvoice"]["state"] == "reference_recorded"
    assert sent.status_code == 200
    assert sent.json()["manualSquareInvoice"]["state"] == "sent_via_square"
    assert service.calls == [
        ("list", {"limit": 1, "offset": 0}),
        (
            "reference",
            {
                "approval_id": approval_id,
                "square_invoice_reference": "SQ-ROUTE",
                "idempotency_key": "reference-route",
                "actor": "Juan",
            },
        ),
        (
            "sent",
            {
                "approval_id": approval_id,
                "idempotency_key": "sent-route",
                "actor": "Juan",
            },
        ),
    ]


def test_manual_square_migration_is_additive_and_performs_no_invoice_transition():
    migration = (
        Path(__file__).parents[1]
        / "atlas_brain/storage/migrations/376_commercial_billing_manual_square_invoices.sql"
    ).read_text()
    executable = "\n".join(
        line for line in migration.splitlines() if not line.lstrip().startswith("--")
    )
    assert (
        "CREATE TABLE IF NOT EXISTS commercial_billing_manual_square_invoices"
        in migration
    )
    assert (
        "CREATE TABLE IF NOT EXISTS commercial_billing_manual_square_invoice_operations"
        in migration
    )
    assert "UNIQUE (source, idempotency_key)" in migration
    assert "'reference_recorded', 'sent_via_square'" in migration
    assert "'record_reference', 'mark_sent'" in migration
    assert "UPDATE invoices" not in executable
    assert "DROP TABLE" not in executable
    assert "DROP COLUMN" not in executable


def test_manual_square_service_has_no_external_delivery_or_payment_transport():
    import atlas_brain.services.commercial_billing_manual_square_invoices as manual_square

    source = inspect.getsource(manual_square)
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
            "gmail",
            "email_provider",
            "invoice_pdf",
            "monthly_invoice_generation",
            "payment",
            "square",
        }
        for imported in imports
    )
    assert ".send(" not in source
    assert "float(" not in source


def test_invoicing_workflow_enrolls_the_manual_square_provider_contract():
    workflow = (
        Path(__file__).parents[1] / ".github/workflows/atlas_invoicing_checks.yml"
    ).read_text()
    for path in (
        "atlas_brain/services/commercial_billing_manual_square_invoices.py",
        "atlas_brain/storage/migrations/376_commercial_billing_manual_square_invoices.sql",
        "tests/test_commercial_billing_manual_square_invoices.py",
    ):
        assert workflow.count(f'      - "{path}"') == 2
    assert "tests/test_commercial_billing_manual_square_invoices.py \\" in workflow
