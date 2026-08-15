"""Contract tests for explicit EOM commercial billing candidate approval."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import os
from contextlib import asynccontextmanager
from datetime import date
from decimal import Decimal
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

from atlas_brain.services.commercial_billing_approvals import (
    CommercialBillingApprovalConflictError,
    CommercialBillingApprovalService,
    CommercialBillingApprovalStaleError,
    CommercialBillingApprovalUnavailableError,
    CommercialBillingApprovalValidationError,
)


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _candidate(
    *,
    blockers: list[dict] | None = None,
    candidate_key: str = "commercial-billing:acme:2026-03",
    description: str = "Office cleaning",
    delivery_method: str = "gmail_pdf",
    recipient_email: str | None = "billing@example.test",
) -> dict:
    candidate = {
        "billingPeriod": "2026-03",
        "blockers": blockers or [],
        "candidateKey": candidate_key,
        "customer": {
            "contactId": "00000000-0000-0000-0000-000000000001",
            "customerType": "commercial",
            "displayName": "Acme Office",
        },
        "deliveryMethod": delivery_method,
        "lineItems": [
            {
                "amountCents": 9650,
                "description": description,
                "quantity": 2,
                "rateCents": 4825,
                "sourceDate": "2026-03-03",
            }
        ],
        "recipient": {"email": recipient_email},
        "subtotalCents": 9650,
        "taxCents": 0,
        "taxRateBasisPoints": 0,
        "totalCents": 9650,
    }
    candidate["sourceFingerprint"] = _fingerprint(candidate)
    return candidate


def _refresh_fingerprint(candidate: dict) -> str:
    evidence = copy.deepcopy(candidate)
    evidence.pop("sourceFingerprint", None)
    candidate["sourceFingerprint"] = _fingerprint(evidence)
    return candidate["sourceFingerprint"]


def _tampered_snapshot(candidate: dict, family: str, value: object) -> dict:
    tampered = copy.deepcopy(candidate)
    if family == "customer":
        tampered["customer"]["displayName"] = value
    elif family == "recipient":
        tampered["recipient"]["email"] = value
    else:
        assert family == "line_item"
        tampered["lineItems"][0]["description"] = value
    return tampered


def _tamper_container(token: str, shape: str) -> object:
    if shape == "scalar":
        return token
    if shape == "list":
        return [token]
    if shape == "wrapped":
        return {"value": token}
    assert shape == "nested"
    return {"value": [token]}


class _CandidateService:
    def __init__(self, candidate: dict) -> None:
        self.candidate = candidate
        self.calls: list[str] = []

    async def preview(self, *, billing_period: str) -> dict:
        self.calls.append(billing_period)
        return {"billingPeriod": billing_period, "candidates": [copy.deepcopy(self.candidate)]}


class _MemoryConnection:
    def __init__(self, candidate: dict, run_id: UUID) -> None:
        self.candidate = candidate
        self.run_id = run_id
        self.invoices: dict[UUID, dict] = {}
        self.approvals_by_key: dict[str, dict] = {}
        self.approvals_by_candidate: dict[tuple[str, str], dict] = {}
        self.insert_attempts = 0

    async def fetchval(self, query, *_args):
        assert "pg_advisory_xact_lock" in query
        return None

    async def fetchrow(self, query, *args):
        if "FROM commercial_billing_runs AS run" in query:
            if args == (self.run_id, self.candidate["candidateKey"]):
                return {
                    "billing_period": "2026-03",
                    "source_fingerprint": self.candidate["sourceFingerprint"],
                    "snapshot": copy.deepcopy(self.candidate),
                }
            return None
        if "FROM commercial_billing_candidate_approvals AS a" in query:
            if "a.source = $1" in query:
                approval = self.approvals_by_key.get(args[1])
            else:
                approval = self.approvals_by_candidate.get((args[0], args[1]))
            return self._view_row(approval) if approval else None
        if "INSERT INTO invoices" in query:
            self.insert_attempts += 1
            invoice_id = args[0]
            invoice = {
                "id": invoice_id,
                "invoice_number": "INV-2026-Mar-0001",
                "status": "draft",
                "total_amount": args[9],
                "issue_date": args[10],
                "due_date": args[11],
                "source_ref": args[13],
                "line_items": json.loads(args[5]),
                "subtotal": args[6],
                "tax_rate": args[7],
                "tax_amount": args[8],
                "metadata": json.loads(args[16]),
            }
            if any(item["source_ref"] == invoice["source_ref"] for item in self.invoices.values()):
                return None
            self.invoices[invoice_id] = invoice
            return {"id": invoice_id}
        if "INSERT INTO commercial_billing_candidate_approvals" in query:
            approval = {
                "approval_id": args[0], "billing_run_id": args[1], "candidate_key": args[2],
                "source_fingerprint": args[3], "request_fingerprint": args[6],
                "invoice_id": args[7], "state": "invoice_created", "approved_by": args[8],
                "approved_at": args[9],
            }
            self.approvals_by_key[args[5]] = approval
            self.approvals_by_candidate[(args[2], args[3])] = approval
            return {"id": args[0]}
        raise AssertionError(query)

    def _view_row(self, approval: dict) -> dict:
        invoice = self.invoices[approval["invoice_id"]]
        return {
            **approval,
            "invoice_number": invoice["invoice_number"],
            "invoice_status": invoice["status"],
            "total_amount": invoice["total_amount"],
            "issue_date": invoice["issue_date"],
            "due_date": invoice["due_date"],
            "source_ref": invoice["source_ref"],
        }


class _MemoryPool:
    is_initialized = True

    def __init__(self, candidate: dict, run_id: UUID) -> None:
        self.conn = _MemoryConnection(candidate, run_id)

    @asynccontextmanager
    async def transaction(self):
        yield self.conn

    async def fetchrow(self, query, *args):
        return await self.conn.fetchrow(query, *args)


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

    async def fetchrow(self, query, *args):
        async with self.connection.transaction():
            await self.connection.execute(f'SET LOCAL search_path TO "{self.schema}"')
            return await self.connection.fetchrow(query, *args)


@asynccontextmanager
async def _approval_database():
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")
    schema, connection = f"commercial_approval_{uuid4().hex}", await asyncpg.connect(database_url)
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
        ):
            await connection.execute((migrations / name).read_text())
        yield connection, schema
    finally:
        await connection.execute("SET search_path TO public")
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


def _service(pool: _MemoryPool, source: _CandidateService) -> CommercialBillingApprovalService:
    return CommercialBillingApprovalService(
        pool=pool,
        candidate_service_loader=lambda: source,
        due_days_loader=lambda: 14,
        today=lambda: date(2026, 4, 2),
    )


@pytest.mark.asyncio
async def test_approval_creates_one_exact_draft_and_same_key_replays_without_source_read():
    run_id, candidate = uuid4(), _candidate()
    fingerprint = candidate["sourceFingerprint"]
    pool, source = _MemoryPool(candidate, run_id), _CandidateService(candidate)
    service = _service(pool, source)

    created = await service.approve(
        billing_run_id=run_id, candidate_key=candidate["candidateKey"],
        expected_source_fingerprint=fingerprint, idempotency_key="approve-acme-1", actor="Juan",
    )

    assert created["replayed"] is False
    assert created["approval"]["state"] == "invoice_created"
    assert created["approval"]["invoice"] == {
        "dueDate": "2026-04-16", "id": created["approval"]["invoice"]["id"],
        "invoiceNumber": "INV-2026-Mar-0001", "issueDate": "2026-04-02",
        "sourceRef": created["approval"]["invoice"]["sourceRef"], "status": "draft",
        "totalCents": 9650,
    }
    invoice = next(iter(pool.conn.invoices.values()))
    assert invoice["subtotal"] == Decimal("96.50")
    assert invoice["line_items"] == [{"amount": "96.50", "date": "2026-03-03", "description": "Office cleaning", "quantity": 2, "unit_price": "48.25"}]
    assert source.calls == ["2026-03"]

    source.candidate = _candidate(description="Changed cleaning")
    replayed = await service.approve(
        billing_run_id=run_id, candidate_key=candidate["candidateKey"],
        expected_source_fingerprint=fingerprint, idempotency_key="approve-acme-1", actor="Juan",
    )
    assert replayed == {**created, "replayed": True}
    assert len(pool.conn.invoices) == 1
    assert source.calls == ["2026-03"]


@pytest.mark.asyncio
async def test_stale_or_blocked_candidate_never_attempts_an_invoice_insert():
    run_id, candidate = uuid4(), _candidate()
    fingerprint = candidate["sourceFingerprint"]
    pool, source = _MemoryPool(candidate, run_id), _CandidateService(
        _candidate(description="Changed cleaning")
    )
    service = _service(pool, source)

    with pytest.raises(CommercialBillingApprovalStaleError):
        await service.approve(
            billing_run_id=run_id, candidate_key=candidate["candidateKey"],
            expected_source_fingerprint=fingerprint, idempotency_key="stale-1", actor="Juan",
        )
    assert pool.conn.insert_attempts == 0

    blocked = _candidate(blockers=[{"code": "missing_hours"}])
    blocked_fingerprint = blocked["sourceFingerprint"]
    blocked_pool, blocked_source = _MemoryPool(blocked, run_id), _CandidateService(blocked)
    with pytest.raises(CommercialBillingApprovalValidationError):
        await _service(blocked_pool, blocked_source).approve(
            billing_run_id=run_id, candidate_key=blocked["candidateKey"],
            expected_source_fingerprint=blocked_fingerprint, idempotency_key="blocked-1", actor="Juan",
        )
    assert blocked_pool.conn.insert_attempts == 0
    assert blocked_source.calls == []

    malformed = _candidate()
    malformed["taxCents"], malformed["totalCents"] = 1, 9651
    malformed_fingerprint = _refresh_fingerprint(malformed)
    malformed_pool, malformed_source = _MemoryPool(malformed, run_id), _CandidateService(malformed)
    with pytest.raises(CommercialBillingApprovalValidationError):
        await _service(malformed_pool, malformed_source).approve(
            billing_run_id=run_id, candidate_key=malformed["candidateKey"],
            expected_source_fingerprint=malformed_fingerprint, idempotency_key="bad-totals-1", actor="Juan",
        )
    assert malformed_pool.conn.insert_attempts == 0
    assert malformed_source.calls == []

    mismatch_pool, mismatch_source = _MemoryPool(candidate, run_id), _CandidateService(candidate)
    with pytest.raises(CommercialBillingApprovalConflictError):
        await _service(mismatch_pool, mismatch_source).approve(
            billing_run_id=run_id, candidate_key=candidate["candidateKey"],
            expected_source_fingerprint=_fingerprint("b"), idempotency_key="mismatch-1", actor="Juan",
        )
    assert mismatch_pool.conn.insert_attempts == 0
    assert mismatch_source.calls == []


@pytest.mark.asyncio
async def test_tampered_snapshot_fingerprint_never_reaches_current_source_or_invoice_insert():
    run_id, candidate = uuid4(), _candidate()
    tampered = copy.deepcopy(candidate)
    tampered["customer"]["displayName"] = "Wrong customer"
    pool, source = _MemoryPool(tampered, run_id), _CandidateService(candidate)

    with pytest.raises(CommercialBillingApprovalUnavailableError):
        await _service(pool, source).approve(
            billing_run_id=run_id,
            candidate_key=tampered["candidateKey"],
            expected_source_fingerprint=tampered["sourceFingerprint"],
            idempotency_key="tampered-snapshot-1",
            actor="Juan",
        )

    assert pool.conn.insert_attempts == 0
    assert source.calls == []


@pytest.mark.asyncio
async def test_snapshot_fingerprint_rejects_generated_tamper_tokens_containers_and_key_families():
    """Spec-derived oracle: every unsealed snapshot alteration rejects before money moves.

    Tokens x containers x key families cover scalar and JSON-wrapper shapes.
    Representation parity requires every generated container to have the same
    expected rejection; the oracle is independent of the writer because a
    retained source fingerprint covers the entire candidate before field reads.
    """

    token_stems = ("changed", "recipient")
    modifiers = ("", "-suffix", "-123")
    container_shapes = ("scalar", "list", "wrapped", "nested")
    key_families = ("customer", "recipient", "line_item")
    expected_rejection = CommercialBillingApprovalUnavailableError

    for token_stem, modifier, container_shape, key_family in product(
        token_stems, modifiers, container_shapes, key_families
    ):
        candidate = _candidate()
        tampered = _tampered_snapshot(
            candidate,
            key_family,
            _tamper_container(f"{token_stem}{modifier}", container_shape),
        )
        pool, source = _MemoryPool(tampered, uuid4()), _CandidateService(candidate)

        with pytest.raises(expected_rejection):
            await _service(pool, source).approve(
                billing_run_id=pool.conn.run_id,
                candidate_key=tampered["candidateKey"],
                expected_source_fingerprint=tampered["sourceFingerprint"],
                idempotency_key=(
                    f"tamper-{token_stem}-{modifier or 'base'}-"
                    f"{container_shape}-{key_family}"
                ),
                actor="Juan",
            )

        assert pool.conn.insert_attempts == 0
        assert source.calls == []


@pytest.mark.asyncio
async def test_manual_square_candidate_creates_draft_without_a_gmail_recipient():
    run_id, candidate = uuid4(), _candidate(
        delivery_method="manual_square", recipient_email=None
    )
    fingerprint = candidate["sourceFingerprint"]
    pool, source = _MemoryPool(candidate, run_id), _CandidateService(candidate)

    result = await _service(pool, source).approve(
        billing_run_id=run_id, candidate_key=candidate["candidateKey"],
        expected_source_fingerprint=fingerprint, idempotency_key="square-1", actor="Juan",
    )

    assert result["approval"]["invoice"]["status"] == "draft"
    assert next(iter(pool.conn.invoices.values()))["metadata"]["deliveryMethod"] == "manual_square"


@pytest.mark.asyncio
async def test_real_postgres_approval_is_atomic_and_reuses_same_candidate_across_runs():
    async with _approval_database() as (connection, schema):
        run_id, second_run_id, candidate = uuid4(), uuid4(), _candidate()
        fingerprint = candidate["sourceFingerprint"]
        contact_id = UUID(candidate["customer"]["contactId"])
        await connection.execute("INSERT INTO contacts (id) VALUES ($1)", contact_id)
        for run, key in ((run_id, "review-1"), (second_run_id, "review-2")):
            await connection.execute(
                """
                INSERT INTO commercial_billing_runs (
                    id, billing_period, state, candidate_contract_version,
                    snapshot_fingerprint, source, idempotency_key,
                    request_fingerprint, created_by
                ) VALUES ($1, '2026-03', 'draft', 2, $2, 'eom_admin', $3, $2, 'Juan')
                """,
                run, fingerprint, key,
            )
            await connection.execute(
                """
                INSERT INTO commercial_billing_run_candidates (
                    id, billing_run_id, candidate_key, source_fingerprint,
                    display_order, snapshot
                ) VALUES ($1, $2, $3, $4, 0, $5::jsonb)
                """,
                uuid4(), run, candidate["candidateKey"], fingerprint, json.dumps(candidate),
            )
        source = _CandidateService(candidate)
        service = _service(_SchemaPool(connection, schema), source)

        created = await service.approve(
            billing_run_id=run_id, candidate_key=candidate["candidateKey"],
            expected_source_fingerprint=fingerprint, idempotency_key="approve-1", actor="Juan",
        )
        replayed = await service.approve(
            billing_run_id=run_id, candidate_key=candidate["candidateKey"],
            expected_source_fingerprint=fingerprint, idempotency_key="approve-1", actor="Juan",
        )
        reused = await service.approve(
            billing_run_id=second_run_id, candidate_key=candidate["candidateKey"],
            expected_source_fingerprint=fingerprint, idempotency_key="approve-2", actor="Juan",
        )

        assert created["replayed"] is False
        assert replayed == {**created, "replayed": True}
        assert reused == {**created, "replayed": True}
        assert await connection.fetchval("SELECT COUNT(*) FROM invoices") == 1
        assert await connection.fetchval("SELECT COUNT(*) FROM commercial_billing_candidate_approvals") == 1
        invoice = await connection.fetchrow("SELECT status, total_amount, amount_due FROM invoices")
        assert dict(invoice) == {"status": "draft", "total_amount": Decimal("96.50"), "amount_due": Decimal("96.50")}
        assert source.calls == ["2026-03", "2026-03"]


@pytest.mark.asyncio
async def test_real_postgres_rolls_back_the_invoice_when_approval_audit_insert_fails():
    async with _approval_database() as (connection, schema):
        run_id, candidate = uuid4(), _candidate(
            candidate_key="commercial-billing:rollback:2026-03",
            description="Transactional approval test",
        )
        fingerprint = candidate["sourceFingerprint"]
        await connection.execute(
            "INSERT INTO contacts (id) VALUES ($1)",
            UUID(candidate["customer"]["contactId"]),
        )
        await connection.execute(
            """
            INSERT INTO commercial_billing_runs (
                id, billing_period, state, candidate_contract_version,
                snapshot_fingerprint, source, idempotency_key,
                request_fingerprint, created_by
            ) VALUES ($1, '2026-03', 'draft', 2, $2, 'eom_admin', 'review-rollback', $2, 'Juan')
            """,
            run_id,
            fingerprint,
        )
        await connection.execute(
            """
            INSERT INTO commercial_billing_run_candidates (
                id, billing_run_id, candidate_key, source_fingerprint,
                display_order, snapshot
            ) VALUES ($1, $2, $3, $4, 0, $5::jsonb)
            """,
            uuid4(),
            run_id,
            candidate["candidateKey"],
            fingerprint,
            json.dumps(candidate),
        )
        await connection.execute(
            """
            CREATE FUNCTION reject_commercial_billing_approval()
            RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
                RAISE EXCEPTION 'injected approval-audit failure';
            END;
            $$
            """
        )
        await connection.execute(
            """
            CREATE TRIGGER reject_commercial_billing_approval_trigger
            BEFORE INSERT ON commercial_billing_candidate_approvals
            FOR EACH ROW EXECUTE FUNCTION reject_commercial_billing_approval()
            """
        )

        with pytest.raises(CommercialBillingApprovalUnavailableError):
            await _service(_SchemaPool(connection, schema), _CandidateService(candidate)).approve(
                billing_run_id=run_id,
                candidate_key=candidate["candidateKey"],
                expected_source_fingerprint=fingerprint,
                idempotency_key="approve-rollback-1",
                actor="Juan",
            )

        assert await connection.fetchval("SELECT COUNT(*) FROM invoices") == 0
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_candidate_approvals"
        ) == 0


class _RouteService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def approve(self, **kwargs):
        self.calls.append(kwargs)
        return {"approval": {"id": "approval-1"}, "replayed": False}


def _route_app(service: _RouteService) -> tuple[FastAPI, str]:
    from atlas_brain.api.invoicing import auth as receivables_auth
    from atlas_brain.api.invoicing import receivables as routes
    from atlas_brain.eom_api.auth import generate_receivables_service_token

    generated = generate_receivables_service_token()
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[receivables_auth.get_receivables_api_config] = lambda: SimpleNamespace(
        receivables_api_enabled=True, receivables_service_token="", receivables_service_token_sha256=generated.sha256,
    )
    app.dependency_overrides[routes.get_commercial_billing_approval_service] = lambda: service
    return app, generated.token


@pytest.mark.asyncio
async def test_approval_route_requires_token_actor_fingerprint_and_idempotency_key():
    service, run_id = _RouteService(), uuid4()
    app, token = _route_app(service)
    path = f"/receivables/commercial-billing-runs/{run_id}/approvals"
    body = {"candidate_key": "commercial-billing:acme:2026-03", "expected_source_fingerprint": _fingerprint("a")}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.post(path, json=body)).status_code == 401
        assert (await client.post(path, json=body, headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "route-1"})).status_code == 422
        assert (await client.post(path, json={**body, "expected_source_fingerprint": "bad"}, headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "route-1", "X-EOM-Actor": "Juan"})).status_code == 422
        response = await client.post(path, json=body, headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "route-1", "X-EOM-Actor": "Juan"})
    assert response.status_code == 201
    assert service.calls == [{"billing_run_id": run_id, "candidate_key": body["candidate_key"], "expected_source_fingerprint": body["expected_source_fingerprint"], "idempotency_key": "route-1", "actor": "Juan"}]


def test_approval_migration_is_additive_and_financially_restrictive():
    migration = (Path(__file__).parents[1] / "atlas_brain/storage/migrations/372_commercial_billing_candidate_approvals.sql").read_text()
    assert "CREATE TABLE IF NOT EXISTS commercial_billing_candidate_approvals" in migration
    assert "ON DELETE RESTRICT" in migration
    assert "UNIQUE (source, idempotency_key)" in migration
    assert "UNIQUE (candidate_key, source_fingerprint)" in migration
    assert "UNIQUE (invoice_id)" in migration
    assert "idx_commercial_billing_run_candidates_exact_source" in migration
    assert "commercial_billing_candidate_approvals_snapshot_fkey" in migration
    assert "eom_commercial_billing" in migration
    assert "DROP TABLE" not in "\n".join(line for line in migration.splitlines() if not line.lstrip().startswith("--"))


def test_approval_service_does_not_import_delivery_or_legacy_monthly_writers():
    import atlas_brain.services.commercial_billing_approvals as approvals

    imports = {alias.name for node in ast.walk(ast.parse(inspect.getsource(approvals))) if isinstance(node, ast.Import) for alias in node.names}
    imports.update({f"{node.module}.{alias.name}" if node.module else alias.name for node in ast.walk(ast.parse(inspect.getsource(approvals))) if isinstance(node, ast.ImportFrom) for alias in node.names})
    assert not any(fragment in imported for fragment in {"gmail", "email_provider", "invoice_pdf", "monthly_invoice_generation", "mark_invoiced"} for imported in imports)
    assert "float(" not in inspect.getsource(approvals)


def test_invoicing_workflow_enrolls_the_approval_writer_and_contract():
    workflow = (Path(__file__).parents[1] / ".github/workflows/atlas_invoicing_checks.yml").read_text()
    for path in (
        "atlas_brain/services/commercial_billing_approvals.py",
        "atlas_brain/storage/migrations/372_commercial_billing_candidate_approvals.sql",
        "tests/test_commercial_billing_approvals.py",
    ):
        assert workflow.count(f'      - "{path}"') == 2
    assert "tests/test_commercial_billing_approvals.py \\" in workflow
