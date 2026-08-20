"""Contract tests for explicit EOM commercial billing candidate approval."""

from __future__ import annotations

import asyncio
import ast
import copy
import hashlib
import inspect
import json
import os
import threading
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
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
from atlas_brain.services.commercial_billing_invoice_pdfs import (
    CommercialBillingInvoicePDFConflictError,
    CommercialBillingInvoicePDFRenderError,
    CommercialBillingInvoicePDFService,
    CommercialBillingInvoicePDFUnavailableError,
    CommercialBillingInvoicePDFValidationError,
    _validate_key,
)
from atlas_brain.services.commercial_billing_runs import (
    CommercialBillingRunConflictError,
    CommercialBillingRunService,
    CommercialBillingRunUnavailableError,
)
from atlas_brain.services.commercial_billing_candidate_overrides import (
    decorate_line_keys,
)
from atlas_brain.storage.repositories.invoice import InvoiceRepository


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _idempotency_key_oracle(value: object) -> bool:
    """Specification-derived admission oracle for the bounded header key."""

    return isinstance(value, str) and 1 <= len(value.strip()) <= 128


def _idempotency_key_grammar_candidates():
    """Generate tokens x wrappers x families rather than enumerating examples."""

    for token, wrapper, family in product(
        ("", "a", "a" * 128, "a" * 129, " "),
        (
            lambda value: value,
            lambda value: [value],
            lambda value: {"idempotency": value},
            lambda value: (value,),
        ),
        (
            lambda value: value,
            lambda value: f" {value} ",
            lambda value: f"\t{value}\n",
        ),
    ):
        yield wrapper(family(token))


@pytest.mark.parametrize("value", tuple(_idempotency_key_grammar_candidates()))
def test_commercial_billing_invoice_pdfs_idempotency_key_matches_spec_derived_oracle(
    value: object,
):
    """Every generated shape follows the key contract without a database call."""

    if _idempotency_key_oracle(value):
        assert _validate_key(value, field="Idempotency key", limit=128) == value.strip()
    else:
        with pytest.raises(CommercialBillingInvoicePDFValidationError):
            _validate_key(value, field="Idempotency key", limit=128)


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
        self.review_decision: dict | None = None
        self.override: dict | None = None
        self.recurring_period_conflict: dict | None = None

    async def fetchval(self, query, *_args):
        if (
            "information_schema.columns" in query
            or "pg_constraint" in query
            or "pg_index" in query
        ):
            return True
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
        if "FROM commercial_billing_candidate_review_decisions" in query:
            return self.review_decision
        if "FROM commercial_billing_candidate_overrides" in query:
            return self.override
        if "SELECT source, invoice_number FROM invoices" in query:
            return self.recurring_period_conflict
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
                "source_fingerprint": args[3], "review_fingerprint": args[4],
                "request_fingerprint": args[7], "invoice_id": args[8],
                "state": "invoice_created", "approved_by": args[9], "approved_at": args[10],
            }
            self.approvals_by_key[args[6]] = approval
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

    async def acquire(self):
        await self.connection.execute(f'SET search_path TO "{self.schema}"')
        return self.connection

    async def release(self, released) -> None:
        assert released is self.connection

    @asynccontextmanager
    async def transaction(self):
        async with self.connection.transaction():
            await self.connection.execute(f'SET LOCAL search_path TO "{self.schema}"')
            yield self.connection

    async def fetchrow(self, query, *args):
        async with self.connection.transaction():
            await self.connection.execute(f'SET LOCAL search_path TO "{self.schema}"')
            return await self.connection.fetchrow(query, *args)

    async def fetch(self, query, *args):
        async with self.connection.transaction():
            await self.connection.execute(f'SET LOCAL search_path TO "{self.schema}"')
            return await self.connection.fetch(query, *args)

    async def fetchval(self, query, *args):
        async with self.connection.transaction():
            await self.connection.execute(f'SET LOCAL search_path TO "{self.schema}"')
            return await self.connection.fetchval(query, *args)

    async def execute(self, query, *args):
        async with self.connection.transaction():
            await self.connection.execute(f'SET LOCAL search_path TO "{self.schema}"')
            return await self.connection.execute(query, *args)


async def _run_migration(connection, schema: str, name: str) -> None:
    from atlas_brain.storage.migrations import run_migrations

    migrations = Path(__file__).parents[1] / "atlas_brain/storage/migrations"
    await run_migrations(
        _SchemaPool(connection, schema),
        migrations_dir=migrations,
        only={Path(name).stem},
    )


def test_approval_database_uses_runner_for_concurrent_dedup_migration():
    source = inspect.getsource(_approval_database)

    assert '"385_invoices_billing_period_dedup.sql"' not in source.split(
        "await _run_migration", maxsplit=1
    )[0]
    assert 'await _run_migration(\n            connection,\n            schema,\n            "385_invoices_billing_period_dedup.sql",' in source


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
            "380_commercial_billing_candidate_review_decisions.sql",
            "381_commercial_billing_candidate_review_decisions_recovery.sql",
            "382_commercial_billing_candidate_overrides.sql",
            "373_commercial_billing_invoice_pdf_artifacts.sql",
        ):
            await connection.execute((migrations / name).read_text())
        await _run_migration(
            connection,
            schema,
            "385_invoices_billing_period_dedup.sql",
        )
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
async def test_excluded_review_decision_never_attempts_an_invoice_insert():
    run_id, candidate = uuid4(), _candidate()
    pool, source = _MemoryPool(candidate, run_id), _CandidateService(candidate)
    pool.conn.review_decision = {"decision": "excluded"}

    with pytest.raises(CommercialBillingApprovalConflictError, match="excluded"):
        await _service(pool, source).approve(
            billing_run_id=run_id,
            candidate_key=candidate["candidateKey"],
            expected_source_fingerprint=candidate["sourceFingerprint"],
            idempotency_key="blocked1",
            actor="Juan",
        )

    assert pool.conn.insert_attempts == 0
    assert pool.conn.approvals_by_key == {}


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
async def test_real_postgres_override_requires_a_fresh_include_and_retries_without_invoice_side_effects():
    async with _approval_database() as (connection, schema):
        run_id, candidate = uuid4(), _candidate(
            candidate_key="commercial-billing:override:2026-03"
        )
        fingerprint = candidate["sourceFingerprint"]
        contact_id = UUID(candidate["customer"]["contactId"])
        await connection.execute("INSERT INTO contacts (id) VALUES ($1)", contact_id)
        await connection.execute(
            """
            INSERT INTO commercial_billing_runs (
                id, billing_period, state, candidate_contract_version,
                snapshot_fingerprint, source, idempotency_key,
                request_fingerprint, created_by
            ) VALUES ($1, '2026-03', 'draft', 2, $2, 'eom_admin', 'override-run-1', $2, 'Juan')
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

        legacy_decision_id = uuid4()
        await connection.execute(
            """
            INSERT INTO commercial_billing_candidate_review_decisions (
                id, billing_run_id, candidate_key, source_fingerprint,
                revision, decision, reason, source, idempotency_key,
                request_fingerprint, decided_by
            ) VALUES ($1, $2, $3, $4, 1, 'included',
                      'A mixed rollout legacy writer omitted the new column.',
                      'eom_admin', 'legacy-override-decision-1', $5, 'Juan')
            """,
            legacy_decision_id,
            run_id,
            candidate["candidateKey"],
            fingerprint,
            _fingerprint("legacy-override-decision"),
        )
        assert await connection.fetchval(
            "SELECT review_fingerprint FROM commercial_billing_candidate_review_decisions WHERE id = $1",
            legacy_decision_id,
        ) == fingerprint

        review_service = CommercialBillingRunService(pool=_SchemaPool(connection, schema))
        line_key = decorate_line_keys(candidate)["lineItems"][0]["lineKey"]
        created_override = await review_service.set_candidate_override(
            billing_run_id=run_id,
            candidate_key=candidate["candidateKey"],
            expected_source_fingerprint=fingerprint,
            expected_override_revision=0,
            reason_code="additional_charge",
            reason="The customer approved an after-hours access fee for this run.",
            line_overrides=[{"lineKey": line_key, "quantity": 3}],
            adjustment={
                "kind": "charge",
                "description": "After-hours access fee",
                "amountCents": 17,
            },
            recipient=None,
            delivery_method="gmail_pdf",
            idempotency_key="override-1",
            actor="Juan",
        )
        active_review_fingerprint = created_override["candidate"]["reviewFingerprint"]
        assert active_review_fingerprint != fingerprint
        assert created_override["candidate"]["lineItems"][0]["amountCents"] == 14_475
        assert created_override["candidate"]["totalCents"] == 14_492
        assert created_override["candidate"]["lineItems"][0]["lineKey"] == line_key
        assert candidate["lineItems"][0]["quantity"] == 2
        assert await connection.fetchval("SELECT COUNT(*) FROM invoices") == 0

        replayed_override = await review_service.set_candidate_override(
            billing_run_id=run_id,
            candidate_key=candidate["candidateKey"],
            expected_source_fingerprint=fingerprint,
            expected_override_revision=0,
            reason_code="additional_charge",
            reason="The customer approved an after-hours access fee for this run.",
            line_overrides=[{"lineKey": line_key, "quantity": 3}],
            adjustment={
                "kind": "charge",
                "description": "After-hours access fee",
                "amountCents": 17,
            },
            recipient=None,
            delivery_method="gmail_pdf",
            idempotency_key="override-1",
            actor="Juan",
        )
        assert replayed_override == {**created_override, "replayed": True}
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_candidate_overrides"
        ) == 1

        revised_override = await review_service.set_candidate_override(
            billing_run_id=run_id,
            candidate_key=candidate["candidateKey"],
            expected_source_fingerprint=fingerprint,
            expected_override_revision=1,
            reason_code="additional_charge",
            reason="The final approved after-hours quantity is four visits for this run.",
            line_overrides=[
                {
                    "lineKey": created_override["candidate"]["lineItems"][0]["lineKey"],
                    "quantity": 4,
                }
            ],
            adjustment={
                "kind": "charge",
                "description": "After-hours access fee",
                "amountCents": 17,
            },
            recipient=None,
            delivery_method="gmail_pdf",
            idempotency_key="override-2",
            actor="Juan",
        )
        active_review_fingerprint = revised_override["candidate"]["reviewFingerprint"]
        assert revised_override["override"]["revision"] == 2
        assert revised_override["candidate"]["lineItems"][0]["amountCents"] == 19_300
        assert revised_override["candidate"]["totalCents"] == 19_317
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_candidate_overrides"
        ) == 2

        with pytest.raises(CommercialBillingRunConflictError, match="Idempotency key"):
            await review_service.set_candidate_review_decision(
                billing_run_id=run_id,
                candidate_key=candidate["candidateKey"],
                expected_source_fingerprint=fingerprint,
                expected_review_fingerprint=active_review_fingerprint,
                decision="included",
                reason="A mixed rollout legacy writer omitted the new column.",
                idempotency_key="legacy-override-decision-1",
                actor="Juan",
            )

        with pytest.raises(CommercialBillingRunConflictError, match="override revision changed"):
            await review_service.set_candidate_override(
                billing_run_id=run_id,
                candidate_key=candidate["candidateKey"],
                expected_source_fingerprint=fingerprint,
                expected_override_revision=0,
                reason_code="additional_charge",
                reason="A stale browser tab must not replace the active review evidence.",
                line_overrides=[{"lineKey": line_key, "quantity": 4}],
                adjustment=None,
                recipient=None,
                delivery_method=None,
                idempotency_key="override-stale-1",
                actor="Juan",
            )

        current_source = _CandidateService(candidate)
        approval_service = _service(_SchemaPool(connection, schema), current_source)
        with pytest.raises(
            CommercialBillingApprovalConflictError,
            match="requires an explicit include",
        ):
            await approval_service.approve(
                billing_run_id=run_id,
                candidate_key=candidate["candidateKey"],
                expected_source_fingerprint=fingerprint,
                expected_review_fingerprint=active_review_fingerprint,
                idempotency_key="override-approval-before-include-1",
                actor="Juan",
            )
        assert await connection.fetchval("SELECT COUNT(*) FROM invoices") == 0

        with pytest.raises(CommercialBillingRunConflictError, match="review identity changed"):
            await review_service.set_candidate_review_decision(
                billing_run_id=run_id,
                candidate_key=candidate["candidateKey"],
                expected_source_fingerprint=fingerprint,
                decision="included",
                reason="A stale Include must be rejected.",
                idempotency_key="override-include-stale-1",
                actor="Juan",
            )

        included = await review_service.set_candidate_review_decision(
            billing_run_id=run_id,
            candidate_key=candidate["candidateKey"],
            expected_source_fingerprint=fingerprint,
            expected_review_fingerprint=active_review_fingerprint,
            decision="included",
            reason="The effective one-run adjustment was reviewed and approved.",
            idempotency_key="override-include-1",
            actor="Juan",
        )
        assert included["reviewDecision"]["decision"] == "included"

        changed_source = copy.deepcopy(candidate)
        changed_source["lineItems"][0]["description"] = "Changed source service evidence"
        _refresh_fingerprint(changed_source)
        current_source.candidate = changed_source
        with pytest.raises(CommercialBillingApprovalStaleError, match="regenerate"):
            await approval_service.approve(
                billing_run_id=run_id,
                candidate_key=candidate["candidateKey"],
                expected_source_fingerprint=fingerprint,
                expected_review_fingerprint=active_review_fingerprint,
                idempotency_key="override-approval-stale-source-1",
                actor="Juan",
            )
        assert await connection.fetchval("SELECT COUNT(*) FROM invoices") == 0
        current_source.candidate = candidate

        approved = await approval_service.approve(
            billing_run_id=run_id,
            candidate_key=candidate["candidateKey"],
            expected_source_fingerprint=fingerprint,
            expected_review_fingerprint=active_review_fingerprint,
            idempotency_key="override-approval-1",
            actor="Juan",
        )
        assert approved["replayed"] is False
        assert approved["approval"]["reviewFingerprint"] == active_review_fingerprint
        assert await connection.fetchval("SELECT COUNT(*) FROM invoices") == 1
        invoice = await connection.fetchrow(
            "SELECT metadata, total_amount FROM invoices"
        )
        invoice_metadata = json.loads(invoice["metadata"])
        assert invoice_metadata["reviewFingerprint"] == active_review_fingerprint
        assert invoice_metadata["commercialBillingExactLineAmounts"] is True
        assert invoice["total_amount"] == Decimal("193.17")

        with pytest.raises(CommercialBillingRunConflictError, match="cannot be overridden"):
            await review_service.set_candidate_override(
                billing_run_id=run_id,
                candidate_key=candidate["candidateKey"],
                expected_source_fingerprint=fingerprint,
                expected_override_revision=2,
                reason_code="additional_charge",
                reason="Approved candidates are immutable.",
                line_overrides=[{"lineKey": line_key, "quantity": 4}],
                adjustment=None,
                recipient=None,
                delivery_method=None,
                idempotency_key="override-after-approval-1",
                actor="Juan",
            )
        asyncpg = pytest.importorskip("asyncpg")
        with pytest.raises(asyncpg.PostgresError, match="append-only"):
            await connection.execute(
                "UPDATE commercial_billing_candidate_overrides SET reason = 'mutated'"
            )


@pytest.mark.asyncio
async def test_real_postgres_concurrent_override_replays_one_committed_revision():
    """Two same-key writers serialize at the operation lock and commit once."""

    async with _approval_database() as (connection, schema):
        run_id, candidate = uuid4(), _candidate(
            candidate_key="commercial-billing:concurrent-override:2026-03"
        )
        fingerprint = candidate["sourceFingerprint"]
        await connection.execute(
            """
            INSERT INTO commercial_billing_runs (
                id, billing_period, state, candidate_contract_version,
                snapshot_fingerprint, source, idempotency_key,
                request_fingerprint, created_by
            ) VALUES ($1, '2026-03', 'draft', 2, $2, 'eom_admin',
                      'concurrent-override-run-1', $2, 'Juan')
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

        asyncpg = pytest.importorskip("asyncpg")
        database_url = os.environ["ATLAS_RECEIVABLES_TEST_DATABASE_URL"]
        operation_key = "override-concurrent-replay-1"
        operation_lock = f"commercial-billing-run-override:eom_admin:{operation_key}"
        locker = await asyncpg.connect(database_url)
        await locker.fetchval(
            "SELECT pg_advisory_lock(hashtextextended($1, 0))", operation_lock
        )
        first_task = None
        second_task = None
        lock_released = False
        line_key = decorate_line_keys(candidate)["lineItems"][0]["lineKey"]
        request = {
            "billing_run_id": run_id,
            "candidate_key": candidate["candidateKey"],
            "expected_source_fingerprint": fingerprint,
            "expected_override_revision": 0,
            "reason_code": "additional_charge",
            "reason": "The operator documented this one-time extra visit.",
            "line_overrides": [{"lineKey": line_key, "quantity": 3}],
            "adjustment": None,
            "recipient": None,
            "delivery_method": None,
            "idempotency_key": operation_key,
            "actor": "Juan",
        }
        try:
            first_service = CommercialBillingRunService(
                pool=_IsolatedSchemaPool(database_url, schema)
            )
            second_service = CommercialBillingRunService(
                pool=_IsolatedSchemaPool(database_url, schema)
            )
            first_task = asyncio.create_task(
                first_service.set_candidate_override(**request)
            )
            second_task = asyncio.create_task(
                second_service.set_candidate_override(**request)
            )
            for _ in range(100):
                waiting = await connection.fetchval(
                    "SELECT COUNT(*) FROM pg_locks "
                    "WHERE locktype = 'advisory' AND NOT granted"
                )
                if waiting >= 2:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError(
                    "both override calls did not wait on the operation lock"
                )

            await locker.fetchval(
                "SELECT pg_advisory_unlock(hashtextextended($1, 0))", operation_lock
            )
            lock_released = True
            first_result, second_result = await asyncio.wait_for(
                asyncio.gather(first_task, second_task), timeout=5
            )
        finally:
            if not lock_released:
                await locker.fetchval(
                    "SELECT pg_advisory_unlock(hashtextextended($1, 0))", operation_lock
                )
            await locker.close()
            if first_task is not None and second_task is not None:
                await asyncio.gather(first_task, second_task, return_exceptions=True)

        assert {first_result["replayed"], second_result["replayed"]} == {False, True}
        assert {
            first_result["override"]["revision"],
            second_result["override"]["revision"],
        } == {1}
        assert (
            await connection.fetchval(
                "SELECT COUNT(*) FROM commercial_billing_candidate_overrides"
            )
            == 1
        )


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
async def test_real_postgres_approval_rejects_when_legacy_monthly_writer_already_invoiced_the_period():
    """Cross-pipeline recurring-invoice dedup (migration 385): the approval
    writer refuses to create a second recurring invoice for a contact+period
    the legacy monthly_auto cron already invoiced. Negative control: the
    same contact with a DIFFERENT billing period is not a conflict and
    succeeds normally -- proves the check discriminates on period, not just
    contact."""
    async with _approval_database() as (connection, schema):
        run_id, candidate = uuid4(), _candidate(
            candidate_key="commercial-billing:cross-pipeline-dedup:2026-03"
        )
        fingerprint = candidate["sourceFingerprint"]
        contact_id = UUID(candidate["customer"]["contactId"])
        await connection.execute("INSERT INTO contacts (id) VALUES ($1)", contact_id)
        await connection.execute(
            """
            INSERT INTO commercial_billing_runs (
                id, billing_period, state, candidate_contract_version,
                snapshot_fingerprint, source, idempotency_key,
                request_fingerprint, created_by
            ) VALUES ($1, '2026-03', 'draft', 2, $2, 'eom_admin', 'cross-pipeline-run', $2, 'Juan')
            """,
            run_id, fingerprint,
        )
        await connection.execute(
            """
            INSERT INTO commercial_billing_run_candidates (
                id, billing_run_id, candidate_key, source_fingerprint,
                display_order, snapshot
            ) VALUES ($1, $2, $3, $4, 0, $5::jsonb)
            """,
            uuid4(), run_id, candidate["candidateKey"], fingerprint, json.dumps(candidate),
        )

        # The legacy cron already invoiced this contact for this period.
        await connection.execute(
            """
            INSERT INTO invoices (
                id, invoice_number, contact_id, customer_name, due_date,
                status, source, billing_period
            ) VALUES ($1, 'INV-LEGACY-2026-03', $2, 'Acme Office', CURRENT_DATE,
                      'draft', 'monthly_auto', '2026-03')
            """,
            uuid4(), contact_id,
        )

        service = _service(_SchemaPool(connection, schema), _CandidateService(candidate))
        with pytest.raises(
            CommercialBillingApprovalConflictError, match="recurring invoice already exists"
        ):
            await service.approve(
                billing_run_id=run_id, candidate_key=candidate["candidateKey"],
                expected_source_fingerprint=fingerprint, idempotency_key="cross-pipeline-1", actor="Juan",
            )
        assert await connection.fetchval("SELECT COUNT(*) FROM invoices") == 1

        # Negative control: same contact, different period -> not a conflict.
        other_run_id = uuid4()
        other_candidate = _candidate(
            candidate_key="commercial-billing:cross-pipeline-dedup-other-period:2026-04",
        )
        other_candidate["billingPeriod"] = "2026-04"
        other_fingerprint = _refresh_fingerprint(other_candidate)
        await connection.execute(
            """
            INSERT INTO commercial_billing_runs (
                id, billing_period, state, candidate_contract_version,
                snapshot_fingerprint, source, idempotency_key,
                request_fingerprint, created_by
            ) VALUES ($1, '2026-04', 'draft', 2, $2, 'eom_admin', 'cross-pipeline-run-2', $2, 'Juan')
            """,
            other_run_id, other_fingerprint,
        )
        await connection.execute(
            """
            INSERT INTO commercial_billing_run_candidates (
                id, billing_run_id, candidate_key, source_fingerprint,
                display_order, snapshot
            ) VALUES ($1, $2, $3, $4, 0, $5::jsonb)
            """,
            uuid4(), other_run_id, other_candidate["candidateKey"], other_fingerprint,
            json.dumps(other_candidate),
        )
        other_service = _service(_SchemaPool(connection, schema), _CandidateService(other_candidate))
        approved = await other_service.approve(
            billing_run_id=other_run_id, candidate_key=other_candidate["candidateKey"],
            expected_source_fingerprint=other_fingerprint, idempotency_key="cross-pipeline-2", actor="Juan",
        )
        assert approved["replayed"] is False
        assert await connection.fetchval("SELECT COUNT(*) FROM invoices") == 2


@pytest.mark.asyncio
async def test_real_postgres_approval_rejects_a_quarantined_backfill_collision_period():
    """Codex P1, second review round on ATLAS #2448: a quarantined
    (contact_id, billing_period) -- an ambiguous historical collision the
    backfill left billing_period=NULL for both rows, rather than guess which
    is real -- has no row claiming idx_invoices_recurring_contact_period_source's
    slot, so without invoices_billing_period_reservations a third invoice for
    that same contact+period would go unblocked. Negative control: the same
    contact with a DIFFERENT period (no reservation) is not a conflict and
    succeeds normally."""
    async with _approval_database() as (connection, schema):
        run_id, candidate = uuid4(), _candidate(
            candidate_key="commercial-billing:quarantine-reservation:2026-05"
        )
        candidate["billingPeriod"] = "2026-05"
        fingerprint = _refresh_fingerprint(candidate)
        contact_id = UUID(candidate["customer"]["contactId"])
        await connection.execute("INSERT INTO contacts (id) VALUES ($1)", contact_id)
        await connection.execute(
            """
            INSERT INTO commercial_billing_runs (
                id, billing_period, state, candidate_contract_version,
                snapshot_fingerprint, source, idempotency_key,
                request_fingerprint, created_by
            ) VALUES ($1, '2026-05', 'draft', 2, $2, 'eom_admin', 'quarantine-run', $2, 'Juan')
            """,
            run_id, fingerprint,
        )
        await connection.execute(
            """
            INSERT INTO commercial_billing_run_candidates (
                id, billing_run_id, candidate_key, source_fingerprint,
                display_order, snapshot
            ) VALUES ($1, $2, $3, $4, 0, $5::jsonb)
            """,
            uuid4(), run_id, candidate["candidateKey"], fingerprint, json.dumps(candidate),
        )

        # Two ambiguous historical invoices already exist for this
        # contact+period (this is what a real collision-quarantine leaves
        # behind: billing_period=NULL on both, plus a reservation row --
        # inserted directly here rather than re-running the full backfill,
        # to isolate the approval-writer's own consumption of the
        # reservation table). Both use source='monthly_auto': a real
        # eom_commercial_billing row is subject to this schema's own
        # review-identity trigger (migration 380/381/382), which is
        # orthogonal to what this test is proving.
        await connection.execute(
            """
            INSERT INTO invoices (
                id, invoice_number, contact_id, customer_name, due_date,
                status, source, billing_period, billing_period_legacy_null, metadata
            ) VALUES
                ($1, 'INV-QUARANTINE-A', $3, 'Acme Office', CURRENT_DATE, 'draft', 'monthly_auto', NULL, true, $4::jsonb),
                ($2, 'INV-QUARANTINE-B', $3, 'Acme Office', CURRENT_DATE, 'draft', 'monthly_auto', NULL, true, $4::jsonb)
            """,
            uuid4(), uuid4(), contact_id,
            json.dumps({
                "billing_period_backfill_collision": True,
                "billing_period_backfill_candidate_period": "2026-05",
            }),
        )
        await connection.execute(
            "INSERT INTO invoices_billing_period_reservations (contact_id, billing_period) "
            "VALUES ($1, '2026-05')",
            contact_id,
        )

        service = _service(_SchemaPool(connection, schema), _CandidateService(candidate))
        with pytest.raises(
            CommercialBillingApprovalConflictError, match="recurring invoice already exists"
        ):
            await service.approve(
                billing_run_id=run_id, candidate_key=candidate["candidateKey"],
                expected_source_fingerprint=fingerprint, idempotency_key="quarantine-1", actor="Juan",
            )
        # Still exactly the 2 pre-existing ambiguous rows -- no third invoice
        # was created, and neither existing row was touched.
        assert await connection.fetchval("SELECT COUNT(*) FROM invoices") == 2

        # Negative control: same contact, a period with NO reservation ->
        # not a conflict, approves normally.
        other_run_id = uuid4()
        other_candidate = _candidate(
            candidate_key="commercial-billing:quarantine-reservation-other-period:2026-06",
        )
        other_candidate["customer"]["contactId"] = str(contact_id)
        other_candidate["billingPeriod"] = "2026-06"
        other_fingerprint = _refresh_fingerprint(other_candidate)
        await connection.execute(
            """
            INSERT INTO commercial_billing_runs (
                id, billing_period, state, candidate_contract_version,
                snapshot_fingerprint, source, idempotency_key,
                request_fingerprint, created_by
            ) VALUES ($1, '2026-06', 'draft', 2, $2, 'eom_admin', 'quarantine-run-2', $2, 'Juan')
            """,
            other_run_id, other_fingerprint,
        )
        await connection.execute(
            """
            INSERT INTO commercial_billing_run_candidates (
                id, billing_run_id, candidate_key, source_fingerprint,
                display_order, snapshot
            ) VALUES ($1, $2, $3, $4, 0, $5::jsonb)
            """,
            uuid4(), other_run_id, other_candidate["candidateKey"], other_fingerprint,
            json.dumps(other_candidate),
        )
        other_service = _service(_SchemaPool(connection, schema), _CandidateService(other_candidate))
        approved = await other_service.approve(
            billing_run_id=other_run_id, candidate_key=other_candidate["candidateKey"],
            expected_source_fingerprint=other_fingerprint, idempotency_key="quarantine-2", actor="Juan",
        )
        assert approved["replayed"] is False
        assert await connection.fetchval("SELECT COUNT(*) FROM invoices") == 3

        # Review round 4: the reservation remains as historical evidence, but
        # once every matching quarantined invoice is voided it must no longer
        # block the approval writer from cleanly reissuing that period.
        await connection.execute(
            """
            UPDATE invoices
            SET status = 'void'
            WHERE contact_id = $1
              AND metadata->>'billing_period_backfill_collision' = 'true'
              AND metadata->>'billing_period_backfill_candidate_period' = '2026-05'
            """,
            contact_id,
        )
        reissue = await service.approve(
            billing_run_id=run_id, candidate_key=candidate["candidateKey"],
            expected_source_fingerprint=fingerprint,
            idempotency_key="quarantine-void-release",
            actor="Juan",
        )
        assert reissue["replayed"] is False
        assert await connection.fetchval("SELECT COUNT(*) FROM invoices") == 4


@pytest.mark.asyncio
async def test_real_postgres_override_identity_and_final_invoice_trigger_are_scoped_to_run():
    """An override in one retained run cannot stale the same source in another."""

    asyncpg = pytest.importorskip("asyncpg")
    async with _approval_database() as (connection, schema):
        first_run_id, second_run_id = uuid4(), uuid4()
        candidate = _candidate(
            candidate_key="commercial-billing:run-scoped-override:2026-03"
        )
        fingerprint = candidate["sourceFingerprint"]
        contact_id = UUID(candidate["customer"]["contactId"])
        await connection.execute("INSERT INTO contacts (id) VALUES ($1)", contact_id)
        for run_id, operation_key in (
            (first_run_id, "run-scope-a"),
            (second_run_id, "run-scope-b"),
        ):
            await connection.execute(
                """
                INSERT INTO commercial_billing_runs (
                    id, billing_period, state, candidate_contract_version,
                    snapshot_fingerprint, source, idempotency_key,
                    request_fingerprint, created_by
                ) VALUES ($1, '2026-03', 'draft', 2, $2, 'eom_admin', $3, $2, 'Juan')
                """,
                run_id,
                fingerprint,
                operation_key,
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

        review_service = CommercialBillingRunService(pool=_SchemaPool(connection, schema))
        line_key = decorate_line_keys(candidate)["lineItems"][0]["lineKey"]
        second_override = await review_service.set_candidate_override(
            billing_run_id=second_run_id,
            candidate_key=candidate["candidateKey"],
            expected_source_fingerprint=fingerprint,
            expected_override_revision=0,
            reason_code="additional_charge",
            reason="The second run includes a documented one-time extra visit.",
            line_overrides=[{"lineKey": line_key, "quantity": 3}],
            adjustment=None,
            recipient=None,
            delivery_method=None,
            idempotency_key="run-scope-override-b",
            actor="Juan",
        )
        assert second_override["candidate"]["reviewFingerprint"] != fingerprint

        first_metadata = {
            "candidateKey": candidate["candidateKey"],
            "commercialBillingRunId": str(first_run_id),
            "reviewFingerprint": fingerprint,
            "sourceFingerprint": fingerprint,
        }
        await connection.execute(
            """
            INSERT INTO invoices (
                id, invoice_number, contact_id, customer_name, line_items,
                subtotal, tax_rate, tax_amount, total_amount, due_date,
                source, source_ref, business_context_id, metadata, billing_period
            ) VALUES (
                $1, 'INV-RUN-A-0001', $2, 'Acme Office', $3::jsonb,
                96.50, 0, 0, 96.50, $4,
                'eom_commercial_billing', 'run-scope-a-invoice', 'effingham_maids', $5::jsonb, '2026-03'
            )
            """,
            uuid4(),
            contact_id,
            json.dumps([]),
            date(2026, 4, 16),
            json.dumps(first_metadata),
        )
        assert await connection.fetchval("SELECT COUNT(*) FROM invoices") == 1

        second_metadata = {
            **first_metadata,
            "commercialBillingRunId": str(second_run_id),
        }
        with pytest.raises(asyncpg.PostgresError, match="review identity is stale"):
            await connection.execute(
                """
                INSERT INTO invoices (
                    id, invoice_number, contact_id, customer_name, line_items,
                    subtotal, tax_rate, tax_amount, total_amount, due_date,
                    source, source_ref, business_context_id, metadata, billing_period
                ) VALUES (
                    $1, 'INV-RUN-B-0001', $2, 'Acme Office', $3::jsonb,
                    96.50, 0, 0, 96.50, $4,
                    'eom_commercial_billing', 'run-scope-b-invoice', 'effingham_maids', $5::jsonb, '2026-03'
                )
                """,
                uuid4(),
                contact_id,
                json.dumps([]),
                date(2026, 4, 16),
                json.dumps(second_metadata),
            )
        assert await connection.fetchval("SELECT COUNT(*) FROM invoices") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("minutes", "rate_cents"),
    ((1, 4825), (2, 4825), (17, 1999), (31, 1045), (59, 7999)),
)
async def test_notes_only_draft_edits_preserve_exact_hourly_line_amounts(
    monkeypatch, minutes, rate_cents
):
    """Committed cents win over rounded display hours when lines are unchanged."""

    invoice_id = uuid4()
    exact_cents = int(
        (Decimal(rate_cents) * Decimal(minutes) / Decimal(60)).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )
    exact_amount = Decimal(exact_cents) / Decimal(100)
    display_hours = f"{(Decimal(minutes) / Decimal(60)):.4f}".rstrip("0").rstrip(".")
    line_item = {
        "amount": f"{exact_amount:.2f}",
        "date": "2026-03-03",
        "description": "Hourly cleaning",
        "quantity": display_hours,
        "unit_price": f"{Decimal(rate_cents) / Decimal(100):.2f}",
    }

    class _UpdatePool:
        is_initialized = True

        def __init__(self) -> None:
            self.arguments = None

        async def fetchrow(self, query, *arguments):
            assert "UPDATE invoices" in query
            self.arguments = arguments
            return {
                "id": invoice_id,
                "invoice_number": "INV-EXACT-HOURLY",
                "status": "draft",
                "line_items": arguments[1],
                "due_date": arguments[2],
                "notes": arguments[3],
                "tax_rate": Decimal(str(arguments[4])),
                "tax_amount": Decimal(str(arguments[5])),
                "discount_amount": Decimal(str(arguments[6])),
                "subtotal": Decimal(str(arguments[7])),
                "total_amount": Decimal(str(arguments[8])),
                "amount_paid": Decimal("0"),
                "amount_due": Decimal(str(arguments[8])),
                "metadata": arguments[11],
            }

    pool = _UpdatePool()
    repository = InvoiceRepository(pool=pool)

    async def _current(_invoice_id):
        return {
            "id": invoice_id,
            "source": "eom_commercial_billing",
            "status": "draft",
            "line_items": [line_item],
            "tax_rate": 0.0,
            "discount_amount": 0.0,
            "metadata": {"commercialBillingExactLineAmounts": True},
        }

    monkeypatch.setattr(repository, "get_by_id", _current)
    updated = await repository.update_invoice(
        invoice_id=invoice_id, notes="Operator added a non-financial note."
    )

    assert pool.arguments is not None
    assert Decimal(str(pool.arguments[7])) == exact_amount
    assert Decimal(str(pool.arguments[8])) == exact_amount
    assert updated["line_items"] == [line_item]
    assert Decimal(str(updated["subtotal"])) == exact_amount
    assert Decimal(str(updated["total_amount"])) == exact_amount


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "metadata"),
    (
        ("mcp_tool", {}),
        ("eom_commercial_billing", {}),
        ("mcp_tool", {"commercialBillingExactLineAmounts": True}),
    ),
)
async def test_notes_only_draft_edits_recalculate_untrusted_recorded_amounts(
    monkeypatch, source, metadata
):
    """Generic caller amounts cannot become authoritative after invoice creation."""

    invoice_id = uuid4()
    line_item = {
        "amount": "1.00",
        "description": "Caller supplied a stale amount",
        "quantity": 2,
        "unit_price": "50.00",
    }

    class _UpdatePool:
        is_initialized = True

        def __init__(self) -> None:
            self.arguments = None

        async def fetchrow(self, query, *arguments):
            assert "UPDATE invoices" in query
            self.arguments = arguments
            return {
                "id": invoice_id,
                "invoice_number": "INV-UNTRUSTED-AMOUNT",
                "status": "draft",
                "line_items": arguments[1],
                "due_date": arguments[2],
                "notes": arguments[3],
                "tax_rate": Decimal(str(arguments[4])),
                "tax_amount": Decimal(str(arguments[5])),
                "discount_amount": Decimal(str(arguments[6])),
                "subtotal": Decimal(str(arguments[7])),
                "total_amount": Decimal(str(arguments[8])),
                "amount_paid": Decimal("0"),
                "amount_due": Decimal(str(arguments[8])),
                "metadata": arguments[11],
            }

    pool = _UpdatePool()
    repository = InvoiceRepository(pool=pool)

    async def _current(_invoice_id):
        return {
            "id": invoice_id,
            "source": source,
            "status": "draft",
            "line_items": [line_item],
            "tax_rate": 0.0,
            "discount_amount": 0.0,
            "metadata": metadata,
        }

    monkeypatch.setattr(repository, "get_by_id", _current)
    updated = await repository.update_invoice(
        invoice_id=invoice_id, notes="Operator added a non-financial note."
    )

    assert pool.arguments is not None
    assert Decimal(str(pool.arguments[7])) == Decimal("100.00")
    assert Decimal(str(pool.arguments[8])) == Decimal("100.00")
    assert Decimal(str(updated["subtotal"])) == Decimal("100.00")
    assert Decimal(str(updated["total_amount"])) == Decimal("100.00")


@pytest.mark.asyncio
async def test_real_postgres_exclusion_blocks_approval_until_reincluded_and_rejects_late_review():
    async with _approval_database() as (connection, schema):
        run_id, candidate = uuid4(), _candidate(
            candidate_key="commercial-billing:excluded:2026-03"
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
            ) VALUES ($1, '2026-03', 'draft', 2, $2, 'eom_admin', 'review-excluded', $2, 'Juan')
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
            INSERT INTO commercial_billing_candidate_review_decisions (
                id, billing_run_id, candidate_key, source_fingerprint,
                revision, decision, reason, source, idempotency_key,
                request_fingerprint, decided_by
            ) VALUES ($1, $2, $3, $4, 1, 'excluded', $5, 'eom_admin', $6, $7, 'Juan')
            """,
            uuid4(),
            run_id,
            candidate["candidateKey"],
            fingerprint,
            "Resolve a customer question first.",
            "review-exclude-1",
            fingerprint,
        )

        approval_service = _service(
            _SchemaPool(connection, schema), _CandidateService(candidate)
        )
        with pytest.raises(CommercialBillingApprovalConflictError, match="excluded"):
            await approval_service.approve(
                billing_run_id=run_id,
                candidate_key=candidate["candidateKey"],
                expected_source_fingerprint=fingerprint,
                idempotency_key="blocked2",
                actor="Juan",
            )
        assert await connection.fetchval("SELECT COUNT(*) FROM invoices") == 0
        assert (
            await connection.fetchval(
                "SELECT COUNT(*) FROM commercial_billing_candidate_approvals"
            )
            == 0
        )

        await connection.execute(
            """
            INSERT INTO commercial_billing_candidate_review_decisions (
                id, billing_run_id, candidate_key, source_fingerprint,
                revision, decision, reason, source, idempotency_key,
                request_fingerprint, decided_by
            ) VALUES ($1, $2, $3, $4, 2, 'included', $5, 'eom_admin', $6, $7, 'Juan')
            """,
            uuid4(),
            run_id,
            candidate["candidateKey"],
            fingerprint,
            "Question resolved after review.",
            "review-include-1",
            fingerprint,
        )
        approved = await approval_service.approve(
            billing_run_id=run_id,
            candidate_key=candidate["candidateKey"],
            expected_source_fingerprint=fingerprint,
            idempotency_key="restored1",
            actor="Juan",
        )
        assert approved["replayed"] is False
        assert await connection.fetchval("SELECT COUNT(*) FROM invoices") == 1

        review_service = CommercialBillingRunService(
            pool=_SchemaPool(connection, schema)
        )
        with pytest.raises(CommercialBillingRunConflictError, match="cannot be reviewed"):
            await review_service.set_candidate_review_decision(
                billing_run_id=run_id,
                candidate_key=candidate["candidateKey"],
                expected_source_fingerprint=fingerprint,
                decision="excluded",
                reason="Too late: an invoice already exists.",
                idempotency_key="review-after-approved-1",
                actor="Juan",
            )
        assert (
            await connection.fetchval(
                "SELECT COUNT(*) FROM commercial_billing_candidate_review_decisions"
            )
            == 2
        )


@pytest.mark.asyncio
async def test_real_postgres_global_review_decision_fences_duplicate_runs_and_old_workers():
    asyncpg = pytest.importorskip("asyncpg")
    async with _approval_database() as (connection, schema):
        first_run_id, second_run_id = uuid4(), uuid4()
        candidate = _candidate(
            candidate_key="commercial-billing:global-review:2026-03"
        )
        fingerprint = candidate["sourceFingerprint"]
        contact_id = UUID(candidate["customer"]["contactId"])
        await connection.execute("INSERT INTO contacts (id) VALUES ($1)", contact_id)
        for run_id, run_key in ((first_run_id, "gla1"), (second_run_id, "glb1")):
            await connection.execute(
                """
                INSERT INTO commercial_billing_runs (
                    id, billing_period, state, candidate_contract_version,
                    snapshot_fingerprint, source, idempotency_key,
                    request_fingerprint, created_by
                ) VALUES ($1, '2026-03', 'draft', 2, $2, 'eom_admin', $3, $2, 'Juan')
                """,
                run_id,
                fingerprint,
                run_key,
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

        review_service = CommercialBillingRunService(pool=_SchemaPool(connection, schema))
        excluded = await review_service.set_candidate_review_decision(
            billing_run_id=first_run_id,
            candidate_key=candidate["candidateKey"],
            expected_source_fingerprint=fingerprint,
            decision="excluded",
            reason="The shared candidate must not be invoiced yet.",
            idempotency_key="global1",
            actor="Juan",
        )
        assert excluded["reviewDecision"]["revision"] == 1
        second_view = await review_service.get_run(second_run_id)
        assert second_view["candidates"][0]["reviewDecision"]["decision"] == "excluded"

        approval_service = _service(
            _SchemaPool(connection, schema), _CandidateService(candidate)
        )
        with pytest.raises(CommercialBillingApprovalConflictError, match="excluded"):
            await approval_service.approve(
                billing_run_id=second_run_id,
                candidate_key=candidate["candidateKey"],
                expected_source_fingerprint=fingerprint,
                idempotency_key="global3",
                actor="Juan",
            )

        with pytest.raises(asyncpg.PostgresError, match="excluded"):
            await connection.execute(
                """
                INSERT INTO invoices (
                    id, invoice_number, contact_id, customer_name, line_items,
                    subtotal, tax_rate, tax_amount, total_amount, due_date,
                    source, source_ref, business_context_id, metadata
                ) VALUES (
                    $1, $2, $3, 'Acme Office', $4::jsonb,
                    96.50, 0, 0, 96.50, $5,
                    'eom_commercial_billing', $6, 'effingham_maids', $7::jsonb
                )
                """,
                uuid4(),
                "INV-OLD-0001",
                contact_id,
                json.dumps([]),
                date(2026, 4, 16),
                "old1",
                json.dumps(
                    {
                        "candidateKey": candidate["candidateKey"],
                        "commercialBillingRunId": str(second_run_id),
                        "sourceFingerprint": fingerprint,
                    }
                ),
            )
        assert await connection.fetchval("SELECT COUNT(*) FROM invoices") == 0

        included = await review_service.set_candidate_review_decision(
            billing_run_id=second_run_id,
            candidate_key=candidate["candidateKey"],
            expected_source_fingerprint=fingerprint,
            decision="included",
            reason="The shared candidate is ready for approval.",
            idempotency_key="global2",
            actor="Juan",
        )
        assert included["reviewDecision"]["revision"] == 2
        first_view = await review_service.get_run(first_run_id)
        assert first_view["candidates"][0]["reviewDecision"] == included["reviewDecision"]

        approved = await approval_service.approve(
            billing_run_id=second_run_id,
            candidate_key=candidate["candidateKey"],
            expected_source_fingerprint=fingerprint,
            idempotency_key="global4",
            actor="Juan",
        )
        assert approved["replayed"] is False
        assert await connection.fetchval("SELECT COUNT(*) FROM invoices") == 1
        first_approved_view = await review_service.get_run(first_run_id)
        second_approved_view = await review_service.get_run(second_run_id)
        assert first_approved_view["candidates"][0]["approval"] == approved["approval"]
        assert second_approved_view["candidates"][0]["approval"] == approved["approval"]

        # Exercise the real public reader as well as the service: the original
        # approval belongs to the equivalent second run, but its exact durable
        # candidate/source/review identity must lock either saved review.
        from atlas_brain.api.invoicing import auth as receivables_auth
        from atlas_brain.api.invoicing import receivables as routes
        from atlas_brain.eom_api.auth import generate_receivables_service_token
        from atlas_brain.main import app

        generated = generate_receivables_service_token()
        original_overrides = dict(app.dependency_overrides)
        app.dependency_overrides[receivables_auth.get_receivables_api_config] = lambda: (
            SimpleNamespace(
                receivables_api_enabled=True,
                receivables_service_token="",
                receivables_service_token_sha256=generated.sha256,
            )
        )
        app.dependency_overrides[routes.get_commercial_billing_run_service] = (
            lambda: review_service
        )
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://atlas.test"
            ) as client:
                response = await client.get(
                    f"/api/v1/receivables/commercial-billing-runs/{first_run_id}",
                    headers={"Authorization": f"Bearer {generated.token}"},
                )
        finally:
            app.dependency_overrides.clear()
            app.dependency_overrides.update(original_overrides)
        assert response.status_code == 200
        assert response.json()["candidates"][0]["approval"] == approved["approval"]
        assert await connection.fetchval("SELECT COUNT(*) FROM invoices") == 1
        assert (
            await connection.fetchval(
                "SELECT COUNT(*) FROM commercial_billing_candidate_approvals"
            )
            == 1
        )

        # A retained approval with the wrong effective review identity is not
        # silently downgraded to an unapproved candidate after a browser reload.
        await connection.execute(
            "UPDATE commercial_billing_candidate_approvals "
            "SET review_fingerprint = $1 WHERE id = $2",
            _fingerprint("mismatched-review"),
            UUID(approved["approval"]["id"]),
        )
        with pytest.raises(CommercialBillingRunUnavailableError, match="review identity"):
            await review_service.get_run(first_run_id)
        assert await connection.fetchval("SELECT COUNT(*) FROM invoices") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "canonical_candidate_key", "canonical_source_fingerprint"),
    (
        ({}, "candidate", "a" * 64),
        ({"candidateKey": " ", "sourceFingerprint": "a" * 64}, "candidate", "a" * 64),
        ({"candidateKey": "candidate"}, "candidate", "a" * 64),
        ({"candidateKey": "candidate", "sourceFingerprint": "A" * 64}, "candidate", "a" * 64),
        ({"candidateKey": "candidate", "sourceFingerprint": "a" * 63}, "candidate", "a" * 64),
        ({"candidateKey": "x" * 513, "sourceFingerprint": "a" * 64}, "candidate", "a" * 64),
        ({"candidateKey": "\tcandidate\t", "sourceFingerprint": "a" * 64}, "candidate", "a" * 64),
        ({"candidateKey": "\ncandidate\n", "sourceFingerprint": "a" * 64}, "candidate", "a" * 64),
        ({"candidateKey": 123, "sourceFingerprint": "a" * 64}, "123", "a" * 64),
        ({"candidateKey": True, "sourceFingerprint": "a" * 64}, "true", "a" * 64),
        ({"candidateKey": ["candidate"], "sourceFingerprint": "a" * 64}, "candidate", "a" * 64),
        (
            {"candidateKey": "candidate", "sourceFingerprint": int("1" * 64)},
            "candidate",
            "1" * 64,
        ),
    ),
    ids=(
        "missing-identity",
        "blank-candidate-key",
        "missing-fingerprint",
        "uppercase-fingerprint",
        "short-fingerprint",
        "long-candidate-key",
        "tab-wrapped-candidate-key",
        "newline-wrapped-candidate-key",
        "numeric-candidate-key",
        "boolean-candidate-key",
        "array-candidate-key",
        "numeric-source-fingerprint",
    ),
)
async def test_real_postgres_invoice_trigger_rejects_noncanonical_or_nonstring_review_identity(
    metadata,
    canonical_candidate_key,
    canonical_source_fingerprint,
):
    """An old worker cannot coerce identity metadata around the final writer guard."""

    asyncpg = pytest.importorskip("asyncpg")
    async with _approval_database() as (connection, _schema):
        run_id = uuid4()
        await connection.execute(
            """
            INSERT INTO commercial_billing_runs (
                id, billing_period, state, candidate_contract_version,
                snapshot_fingerprint, source, idempotency_key,
                request_fingerprint, created_by
            ) VALUES ($1, '2026-03', 'draft', 2, $2, 'eom_admin', $3, $2, 'Juan')
            """,
            run_id,
            canonical_source_fingerprint,
            f"identity-{run_id.hex}",
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
            canonical_candidate_key,
            canonical_source_fingerprint,
            json.dumps(
                {
                    "candidateKey": canonical_candidate_key,
                    "sourceFingerprint": canonical_source_fingerprint,
                }
            ),
        )
        with pytest.raises(asyncpg.PostgresError, match="review identity is invalid"):
            await connection.execute(
                """
                INSERT INTO invoices (
                    id, invoice_number, customer_name, due_date,
                    source, source_ref, metadata
                ) VALUES (
                    $1, 'INV-GUARD-0001', 'Guarded commercial invoice', $2,
                    'eom_commercial_billing', 'guard1', $3::jsonb
                )
                """,
                uuid4(),
                date(2026, 4, 16),
                json.dumps(metadata),
            )
        assert await connection.fetchval("SELECT COUNT(*) FROM invoices") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "billing_run_identity",
    (None, "", "not-a-uuid", 7, "00000000-0000-0000-0000-000000000099"),
)
async def test_real_postgres_invoice_trigger_rejects_invalid_or_unretained_run_identity(
    billing_run_identity,
):
    """Final invoice admission cannot borrow an override scope from another run."""

    asyncpg = pytest.importorskip("asyncpg")
    async with _approval_database() as (connection, _schema):
        run_id = uuid4()
        candidate_key = "commercial-billing:run-identity-guard:2026-03"
        source_fingerprint = "b" * 64
        await connection.execute(
            """
            INSERT INTO commercial_billing_runs (
                id, billing_period, state, candidate_contract_version,
                snapshot_fingerprint, source, idempotency_key,
                request_fingerprint, created_by
            ) VALUES ($1, '2026-03', 'draft', 2, $2, 'eom_admin', $3, $2, 'Juan')
            """,
            run_id,
            source_fingerprint,
            f"run-identity-{run_id.hex}",
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
            candidate_key,
            source_fingerprint,
            json.dumps(
                {
                    "candidateKey": candidate_key,
                    "sourceFingerprint": source_fingerprint,
                }
            ),
        )
        metadata = {
            "candidateKey": candidate_key,
            "sourceFingerprint": source_fingerprint,
        }
        if billing_run_identity is not None:
            metadata["commercialBillingRunId"] = billing_run_identity
        with pytest.raises(asyncpg.PostgresError, match="review identity is invalid"):
            await connection.execute(
                """
                INSERT INTO invoices (
                    id, invoice_number, customer_name, due_date,
                    source, source_ref, metadata
                ) VALUES (
                    $1, 'INV-RUN-GUARD-0001', 'Guarded commercial invoice', $2,
                    'eom_commercial_billing', 'run-guard-1', $3::jsonb
                )
                """,
                uuid4(),
                date(2026, 4, 16),
                json.dumps(metadata),
            )
        assert await connection.fetchval("SELECT COUNT(*) FROM invoices") == 0


@pytest.mark.asyncio
async def test_real_postgres_exclusion_and_approval_serialize_at_the_same_candidate_boundary():
    """An approval waiting behind a committed exclusion cannot insert an invoice."""

    async with _approval_database() as (connection, schema):
        run_id, candidate = uuid4(), _candidate(
            candidate_key="commercial-billing:concurrent-exclusion:2026-03"
        )
        fingerprint = candidate["sourceFingerprint"]
        database_url = os.environ["ATLAS_RECEIVABLES_TEST_DATABASE_URL"]
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
            ) VALUES ($1, '2026-03', 'draft', 2, $2, 'eom_admin', 'review-concurrent', $2, 'Juan')
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

        decision_service = CommercialBillingRunService(
            pool=_IsolatedSchemaPool(database_url, schema)
        )
        approval_service = CommercialBillingApprovalService(
            pool=_IsolatedSchemaPool(database_url, schema),
            candidate_service_loader=lambda: _CandidateService(candidate),
            due_days_loader=lambda: 14,
            today=lambda: date(2026, 4, 2),
        )
        asyncpg = pytest.importorskip("asyncpg")
        candidate_lock_key = (
            "commercial-billing-approval:"
            f"candidate:{candidate['candidateKey']}:{fingerprint}"
        )
        locker = await asyncpg.connect(database_url)
        await locker.fetchval(
            "SELECT pg_advisory_lock(hashtextextended($1, 0))",
            candidate_lock_key,
        )
        decision_task = None
        approval_task = None
        lock_released = False
        try:
            decision_task = asyncio.create_task(
                decision_service.set_candidate_review_decision(
                    billing_run_id=run_id,
                    candidate_key=candidate["candidateKey"],
                    expected_source_fingerprint=fingerprint,
                    decision="excluded",
                    reason="Do not approve while the operator resolves this question.",
                    idempotency_key="review-concurrent-exclude-1",
                    actor="Juan",
                )
            )
            for _ in range(100):
                waiting = await connection.fetchval(
                    "SELECT COUNT(*) FROM pg_locks "
                    "WHERE locktype = 'advisory' AND NOT granted"
                )
                if waiting >= 1:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("review decision did not wait on the candidate lock")
            approval_task = asyncio.create_task(
                approval_service.approve(
                    billing_run_id=run_id,
                    candidate_key=candidate["candidateKey"],
                    expected_source_fingerprint=fingerprint,
                    idempotency_key="approve-concurrent-exclude-1",
                    actor="Juan",
                )
            )
            for _ in range(100):
                waiting = await connection.fetchval(
                    "SELECT COUNT(*) FROM pg_locks "
                    "WHERE locktype = 'advisory' AND NOT granted"
                )
                if waiting >= 2:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("approval did not wait behind the review decision")

            await locker.fetchval(
                "SELECT pg_advisory_unlock(hashtextextended($1, 0))",
                candidate_lock_key,
            )
            lock_released = True
            decision = await asyncio.wait_for(decision_task, timeout=5)
            assert decision["reviewDecision"]["decision"] == "excluded"
            with pytest.raises(CommercialBillingApprovalConflictError, match="excluded"):
                await asyncio.wait_for(approval_task, timeout=5)
        finally:
            if not lock_released:
                await locker.fetchval(
                    "SELECT pg_advisory_unlock(hashtextextended($1, 0))",
                    candidate_lock_key,
                )
            await locker.close()
            if decision_task is not None:
                await asyncio.gather(decision_task, return_exceptions=True)
            if approval_task is not None:
                await asyncio.gather(approval_task, return_exceptions=True)

        assert await connection.fetchval("SELECT COUNT(*) FROM invoices") == 0
        assert (
            await connection.fetchval(
                "SELECT COUNT(*) FROM commercial_billing_candidate_approvals"
            )
            == 0
        )
        assert (
            await connection.fetchval(
                "SELECT COUNT(*) FROM commercial_billing_candidate_review_decisions"
            )
            == 1
        )


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


async def _create_approved_invoice(connection, schema: str) -> dict:
    run_id, candidate = uuid4(), _candidate()
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
        ) VALUES ($1, '2026-03', 'draft', 2, $2, 'eom_admin', 'review-pdf', $2, 'Juan')
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
    result = await _service(
        _SchemaPool(connection, schema), _CandidateService(candidate)
    ).approve(
        billing_run_id=run_id,
        candidate_key=candidate["candidateKey"],
        expected_source_fingerprint=fingerprint,
        idempotency_key="approve-pdf",
        actor="Juan",
    )
    return result["approval"]


class _PDFRenderer:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.failure: Exception | None = None
        self.output = b"%PDF-1.7\nsynthetic commercial billing artifact\n%%EOF\n"

    def __call__(self, invoice: dict) -> bytes:
        self.calls.append(copy.deepcopy(invoice))
        if self.failure is not None:
            raise self.failure
        return self.output


class _BlockingPDFRenderer(_PDFRenderer):
    """Hold one real service transaction while another caller reaches its lock."""

    def __init__(self) -> None:
        super().__init__()
        self.first_render_started = threading.Event()
        self.release_first_render = threading.Event()
        self._calls_lock = threading.Lock()

    def __call__(self, invoice: dict) -> bytes:
        with self._calls_lock:
            call_number = len(self.calls) + 1
            self.calls.append(copy.deepcopy(invoice))
        if call_number == 1:
            self.first_render_started.set()
            if not self.release_first_render.wait(timeout=5):
                raise RuntimeError("timed out waiting to release synthetic PDF render")
        if self.failure is not None:
            raise self.failure
        return self.output


class _IsolatedSchemaPool:
    """Give concurrent service calls independent real PostgreSQL connections."""

    def __init__(self, database_url: str, schema: str) -> None:
        self.database_url = database_url
        self.schema = schema

    @property
    def is_initialized(self) -> bool:
        return True

    @asynccontextmanager
    async def transaction(self):
        asyncpg = pytest.importorskip("asyncpg")
        connection = await asyncpg.connect(self.database_url)
        try:
            await connection.execute(f'SET search_path TO "{self.schema}"')
            async with connection.transaction():
                yield connection
        finally:
            await connection.close()

    async def fetchrow(self, query, *args):
        asyncpg = pytest.importorskip("asyncpg")
        connection = await asyncpg.connect(self.database_url)
        try:
            await connection.execute(f'SET search_path TO "{self.schema}"')
            return await connection.fetchrow(query, *args)
        finally:
            await connection.close()


class _ApprovalLockObservingPDFService(CommercialBillingInvoicePDFService):
    def __init__(self, *, approval_lock_started: threading.Event, **kwargs) -> None:
        super().__init__(**kwargs)
        self._approval_lock_started = approval_lock_started

    async def _lock(self, conn, scope: str) -> None:
        if scope.startswith("approval:"):
            self._approval_lock_started.set()
        await super()._lock(conn, scope)


def _pdf_service(connection, schema: str, renderer: _PDFRenderer) -> CommercialBillingInvoicePDFService:
    return CommercialBillingInvoicePDFService(
        pool=_SchemaPool(connection, schema),
        renderer=renderer,
        now=lambda: datetime(2026, 4, 2, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_real_postgres_pdf_artifact_is_immutable_idempotent_and_reusable():
    async with _approval_database() as (connection, schema):
        approval = await _create_approved_invoice(connection, schema)
        renderer = _PDFRenderer()
        service = _pdf_service(connection, schema, renderer)

        created = await service.generate_or_reuse(
            approval_id=UUID(approval["id"]), idempotency_key="pdf-1", actor="Juan"
        )
        replayed = await service.generate_or_reuse(
            approval_id=UUID(approval["id"]), idempotency_key="pdf-1", actor="Juan"
        )
        reused = await service.generate_or_reuse(
            approval_id=UUID(approval["id"]), idempotency_key="pdf-2", actor="Juan"
        )

        assert created["replayed"] is False
        assert created["reused"] is False
        assert created["artifact"]["state"] == "ready"
        assert created["artifact"]["filename"] == "INV-2026-Mar-0001.pdf"
        assert "pdfBytes" not in created["artifact"]
        assert replayed == {**created, "replayed": True, "reused": True}
        assert reused == {**created, "replayed": False, "reused": True}
        assert len(renderer.calls) == 1

        artifact = await connection.fetchrow(
            "SELECT pdf_bytes, byte_size, pdf_sha256, render_fingerprint, generated_by "
            "FROM commercial_billing_invoice_pdf_artifacts"
        )
        assert artifact["pdf_bytes"] == b"%PDF-1.7\nsynthetic commercial billing artifact\n%%EOF\n"
        assert artifact["byte_size"] == len(artifact["pdf_bytes"])
        assert artifact["pdf_sha256"] == hashlib.sha256(artifact["pdf_bytes"]).hexdigest()
        assert artifact["render_fingerprint"] == created["artifact"]["renderFingerprint"]
        assert artifact["generated_by"] == "Juan"
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_pdf_artifacts"
        ) == 1
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_pdf_operations"
        ) == 2


@pytest.mark.asyncio
async def test_real_postgres_pdf_artifact_serializes_concurrent_new_keys_per_approval():
    """Every admitted distinct-key schedule creates one artifact and two receipts."""

    async with _approval_database() as (connection, schema):
        approval = await _create_approved_invoice(connection, schema)
        approval_id = UUID(approval["id"])
        database_url = os.environ["ATLAS_RECEIVABLES_TEST_DATABASE_URL"]
        renderer = _BlockingPDFRenderer()
        first_service = CommercialBillingInvoicePDFService(
            pool=_IsolatedSchemaPool(database_url, schema),
            renderer=renderer,
            now=lambda: datetime(2026, 4, 2, tzinfo=timezone.utc),
        )
        second_approval_lock_started = threading.Event()
        second_service = _ApprovalLockObservingPDFService(
            pool=_IsolatedSchemaPool(database_url, schema),
            renderer=renderer,
            now=lambda: datetime(2026, 4, 2, tzinfo=timezone.utc),
            approval_lock_started=second_approval_lock_started,
        )

        first = asyncio.create_task(
            asyncio.to_thread(
                lambda: asyncio.run(
                    first_service.generate_or_reuse(
                        approval_id=approval_id,
                        idempotency_key="pdf-concurrent-1",
                        actor="Juan",
                    )
                )
            )
        )
        assert await asyncio.to_thread(renderer.first_render_started.wait, 2)
        second = asyncio.create_task(
            asyncio.to_thread(
                lambda: asyncio.run(
                    second_service.generate_or_reuse(
                        approval_id=approval_id,
                        idempotency_key="pdf-concurrent-2",
                        actor="Juan",
                    )
                )
            )
        )
        assert await asyncio.to_thread(second_approval_lock_started.wait, 2)
        assert not second.done()

        renderer.release_first_render.set()
        first_result, second_result = await asyncio.gather(first, second)
        assert {result["reused"] for result in (first_result, second_result)} == {False, True}
        assert all(result["replayed"] is False for result in (first_result, second_result))
        assert len(renderer.calls) == 1
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_pdf_artifacts"
        ) == 1
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_pdf_operations"
        ) == 2


@pytest.mark.asyncio
async def test_pdf_new_request_rejects_changed_or_non_draft_invoice_without_a_new_write():
    async with _approval_database() as (connection, schema):
        approval = await _create_approved_invoice(connection, schema)
        renderer = _PDFRenderer()
        service = _pdf_service(connection, schema, renderer)
        created = await service.generate_or_reuse(
            approval_id=UUID(approval["id"]), idempotency_key="pdf-change-1", actor="Juan"
        )

        await connection.execute(
            "UPDATE invoices SET customer_name = 'Changed Office' WHERE id = $1",
            UUID(approval["invoice"]["id"]),
        )
        with pytest.raises(CommercialBillingInvoicePDFConflictError):
            await service.generate_or_reuse(
                approval_id=UUID(approval["id"]),
                idempotency_key="pdf-change-2",
                actor="Juan",
            )
        replayed = await service.generate_or_reuse(
            approval_id=UUID(approval["id"]),
            idempotency_key="pdf-change-1",
            actor="Juan",
        )
        assert replayed == {**created, "replayed": True, "reused": True}
        assert len(renderer.calls) == 1
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_pdf_operations"
        ) == 1

    async with _approval_database() as (connection, schema):
        approval = await _create_approved_invoice(connection, schema)
        renderer = _PDFRenderer()
        await connection.execute(
            "UPDATE invoices SET status = 'sent' WHERE id = $1",
            UUID(approval["invoice"]["id"]),
        )
        with pytest.raises(CommercialBillingInvoicePDFConflictError):
            await _pdf_service(connection, schema, renderer).generate_or_reuse(
                approval_id=UUID(approval["id"]),
                idempotency_key="pdf-sent-1",
                actor="Juan",
            )
        assert renderer.calls == []
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_pdf_artifacts"
        ) == 0
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_pdf_operations"
        ) == 0

    async with _approval_database() as (connection, schema):
        approval = await _create_approved_invoice(connection, schema)
        renderer = _PDFRenderer()
        await connection.execute(
            "UPDATE invoices SET total_amount = 'NaN'::numeric WHERE id = $1",
            UUID(approval["invoice"]["id"]),
        )
        with pytest.raises(CommercialBillingInvoicePDFConflictError):
            await _pdf_service(connection, schema, renderer).generate_or_reuse(
                approval_id=UUID(approval["id"]),
                idempotency_key="pdf-nan-1",
                actor="Juan",
            )
        assert renderer.calls == []
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_pdf_artifacts"
        ) == 0
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_pdf_operations"
        ) == 0


@pytest.mark.asyncio
async def test_pdf_render_and_operation_failures_leave_no_artifact_and_retry_safely():
    async with _approval_database() as (connection, schema):
        approval = await _create_approved_invoice(connection, schema)
        renderer = _PDFRenderer()
        renderer.failure = RuntimeError("synthetic renderer failure")
        service = _pdf_service(connection, schema, renderer)

        with pytest.raises(CommercialBillingInvoicePDFRenderError):
            await service.generate_or_reuse(
                approval_id=UUID(approval["id"]), idempotency_key="pdf-retry-1", actor="Juan"
            )
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_pdf_artifacts"
        ) == 0
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_pdf_operations"
        ) == 0

        renderer.failure = None
        renderer.output = b"not a PDF"
        with pytest.raises(CommercialBillingInvoicePDFRenderError):
            await service.generate_or_reuse(
                approval_id=UUID(approval["id"]), idempotency_key="pdf-retry-1", actor="Juan"
            )
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_pdf_artifacts"
        ) == 0
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_pdf_operations"
        ) == 0

        renderer.output = b"%PDF-1.7\nsynthetic commercial billing artifact\n%%EOF\n"
        await connection.execute(
            """
            CREATE FUNCTION reject_commercial_billing_pdf_operation()
            RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
                RAISE EXCEPTION 'injected PDF operation failure';
            END;
            $$
            """
        )
        await connection.execute(
            """
            CREATE TRIGGER reject_commercial_billing_pdf_operation_trigger
            BEFORE INSERT ON commercial_billing_invoice_pdf_operations
            FOR EACH ROW EXECUTE FUNCTION reject_commercial_billing_pdf_operation()
            """
        )
        with pytest.raises(CommercialBillingInvoicePDFUnavailableError):
            await service.generate_or_reuse(
                approval_id=UUID(approval["id"]), idempotency_key="pdf-retry-1", actor="Juan"
            )
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_pdf_artifacts"
        ) == 0
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_pdf_operations"
        ) == 0

        await connection.execute(
            "DROP TRIGGER reject_commercial_billing_pdf_operation_trigger "
            "ON commercial_billing_invoice_pdf_operations"
        )
        await connection.execute("DROP FUNCTION reject_commercial_billing_pdf_operation()")
        retried = await service.generate_or_reuse(
            approval_id=UUID(approval["id"]), idempotency_key="pdf-retry-1", actor="Juan"
        )
        assert retried["replayed"] is False
        assert retried["reused"] is False
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_pdf_artifacts"
        ) == 1
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_invoice_pdf_operations"
        ) == 1


class _RouteService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def approve(self, **kwargs):
        self.calls.append(kwargs)
        return {"approval": {"id": "approval-1"}, "replayed": False}


class _PDFRouteService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate_or_reuse(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "artifact": {
                "approvalId": str(kwargs["approval_id"]),
                "id": "artifact-1",
                "state": "ready",
            },
            "replayed": False,
            "reused": False,
        }


def _route_app(
    service: _RouteService,
    pdf_service: _PDFRouteService | None = None,
) -> tuple[FastAPI, str]:
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
    if pdf_service is not None:
        app.dependency_overrides[
            routes.get_commercial_billing_invoice_pdf_service
        ] = lambda: pdf_service
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


@pytest.mark.asyncio
async def test_full_atlas_app_pdf_route_requires_token_actor_and_idempotency_without_returning_bytes():
    """Exercise the live ``main.app -> /api/v1`` route registration chain."""

    from atlas_brain.api.invoicing import auth as receivables_auth
    from atlas_brain.api.invoicing import receivables as routes
    from atlas_brain.eom_api.auth import generate_receivables_service_token
    from atlas_brain.main import app

    approval_id = uuid4()
    service = _PDFRouteService()
    generated = generate_receivables_service_token()
    original_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[receivables_auth.get_receivables_api_config] = lambda: SimpleNamespace(
        receivables_api_enabled=True,
        receivables_service_token="",
        receivables_service_token_sha256=generated.sha256,
    )
    app.dependency_overrides[routes.get_commercial_billing_invoice_pdf_service] = lambda: service
    path = f"/api/v1/receivables/commercial-billing-approvals/{approval_id}/invoice-pdf"
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
                    headers={**headers, "Idempotency-Key": "route-pdf-1"},
                )
            ).status_code == 422
            response = await client.post(
                path,
                headers={
                    **headers,
                    "Idempotency-Key": "route-pdf-1",
                    "X-EOM-Actor": "Juan",
                },
            )
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)
    assert response.status_code == 201
    assert response.json()["artifact"] == {
        "approvalId": str(approval_id), "id": "artifact-1", "state": "ready"
    }
    assert "pdf_bytes" not in response.text
    assert service.calls == [
        {"approval_id": approval_id, "idempotency_key": "route-pdf-1", "actor": "Juan"}
    ]


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


def test_pdf_artifact_migration_is_additive_and_financially_restrictive():
    migration = (
        Path(__file__).parents[1]
        / "atlas_brain/storage/migrations/373_commercial_billing_invoice_pdf_artifacts.sql"
    ).read_text()
    assert "CREATE TABLE IF NOT EXISTS commercial_billing_invoice_pdf_artifacts" in migration
    assert "CREATE TABLE IF NOT EXISTS commercial_billing_invoice_pdf_operations" in migration
    assert migration.count("ON DELETE RESTRICT") == 2
    assert "approval_id UUID NOT NULL UNIQUE" in migration
    assert "invoice_id UUID" not in migration
    assert "pdf_bytes BYTEA NOT NULL" in migration
    assert "CHECK (state = 'ready')" in migration
    assert "CHECK (octet_length(pdf_bytes) = byte_size)" in migration
    assert "UNIQUE (source, idempotency_key)" in migration
    executable = "\n".join(
        line for line in migration.splitlines() if not line.lstrip().startswith("--")
    )
    assert "DROP TABLE" not in executable
    assert not any(token in executable.lower() for token in ("gmail", "email", "sent_via"))


def test_approval_service_does_not_import_delivery_or_legacy_monthly_writers():
    import atlas_brain.services.commercial_billing_approvals as approvals

    imports = {alias.name for node in ast.walk(ast.parse(inspect.getsource(approvals))) if isinstance(node, ast.Import) for alias in node.names}
    imports.update({f"{node.module}.{alias.name}" if node.module else alias.name for node in ast.walk(ast.parse(inspect.getsource(approvals))) if isinstance(node, ast.ImportFrom) for alias in node.names})
    assert not any(fragment in imported for fragment in {"gmail", "email_provider", "invoice_pdf", "monthly_invoice_generation", "mark_invoiced"} for imported in imports)
    assert "float(" not in inspect.getsource(approvals)


def test_pdf_artifact_service_tracks_every_renderer_field_without_delivery_imports():
    import atlas_brain.services.commercial_billing_invoice_pdfs as pdf_artifacts
    from atlas_brain.services import invoice_pdf

    renderer_fields = {
        node.args[0].value
        for node in ast.walk(ast.parse(inspect.getsource(invoice_pdf.render_invoice_pdf)))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "invoice"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    snapshot = pdf_artifacts._render_snapshot(
        {field: None for field in renderer_fields}
    )
    source = inspect.getsource(pdf_artifacts)
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
    assert renderer_fields == set(snapshot)
    assert not any(
        fragment in imported
        for fragment in {"gmail", "email_provider", "monthly_invoice_generation", "crm_provider"}
        for imported in imports
    )
    assert "UPDATE invoices" not in source
    assert "float(" not in source


def test_invoicing_workflow_enrolls_the_approval_writer_and_contract():
    workflow = (Path(__file__).parents[1] / ".github/workflows/atlas_invoicing_checks.yml").read_text()
    for path in (
        "atlas_brain/services/commercial_billing_approvals.py",
        "atlas_brain/storage/migrations/372_commercial_billing_candidate_approvals.sql",
        "atlas_brain/services/commercial_billing_invoice_pdfs.py",
        "atlas_brain/storage/migrations/373_commercial_billing_invoice_pdf_artifacts.sql",
        "tests/test_commercial_billing_approvals.py",
    ):
        assert workflow.count(f'      - "{path}"') == 2
    assert "tests/test_commercial_billing_approvals.py \\" in workflow
