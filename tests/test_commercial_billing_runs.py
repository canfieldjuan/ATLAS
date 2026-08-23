"""Contract tests for durable, pre-approval EOM commercial billing runs."""

from __future__ import annotations

import asyncio
import ast
import copy
import hashlib
import inspect
import json
import os
from contextlib import asynccontextmanager
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

from atlas_brain.services.commercial_billing_candidates import (
    CommercialBillingCandidatesUnavailableError,
    CommercialBillingCandidatesValidationError,
)
from atlas_brain.services.commercial_billing_runs import (
    CommercialBillingRunConflictError,
    CommercialBillingRunNotFoundError,
    CommercialBillingRunService,
    CommercialBillingRunUnavailableError,
    CommercialBillingRunValidationError,
    _normalize_preview,
    lock_commercial_billing_run_candidate,
)


def _fingerprint(letter: str) -> str:
    return letter * 64


def _candidate(
    key: str,
    fingerprint: str,
    *,
    blockers: list[dict] | None = None,
    service_name: str = "Office cleaning",
) -> dict:
    return {
        "billingPeriod": "2026-03",
        "blockers": blockers or [],
        "candidateKey": key,
        "customer": {
            "contactId": "00000000-0000-0000-0000-000000000001",
            "customerType": "commercial",
            "displayName": "Acme Office",
        },
        "deliveryMethod": None,
        "lineItems": [
            {
                "amountCents": 9650,
                "description": service_name,
                "eventIds": ["event-1"],
                "locations": ["100 Main St"],
                "quantity": 2,
                "quantityUnit": "visit",
                "rateCents": 4825,
                "serviceId": "service-1",
                "sourceDate": "2026-03-03",
            }
        ],
        "recipient": {
            "contactId": "00000000-0000-0000-0000-000000000001",
            "displayName": "Acme Accounts Payable",
            "email": "billing@example.test",
        },
        "services": [
            {
                "calendarId": "commercial-calendar",
                "calendarKeyword": "Acme",
                "rateCents": 4825,
                "rateLabel": "Per Visit",
                "serviceId": "service-1",
                "serviceName": service_name,
                "taxRateBasisPoints": 0,
            }
        ],
        "sourceEvents": [
            {
                "allDay": False,
                "calendarId": "commercial-calendar",
                "end": "2026-03-03T17:00:00+00:00",
                "eventId": "event-1",
                "location": "100 Main St",
                "sourceDate": "2026-03-03",
                "start": "2026-03-03T16:00:00+00:00",
                "status": "confirmed",
                "summary": "Acme Office cleaning",
            }
        ],
        "sourceFingerprint": fingerprint,
        "subtotalCents": 9650,
        "taxCents": 0,
        "taxRateBasisPoints": 0,
        "totalCents": 9650,
    }


def _preview(*candidates: dict) -> dict:
    return {
        "billingPeriod": "2026-03",
        "calendarId": "commercial-calendar",
        "candidates": list(candidates),
        "contractVersion": 2,
        "summary": {
            "blockedCandidateCount": sum(1 for candidate in candidates if candidate["blockers"]),
            "candidateCount": len(candidates),
        },
    }


class _CandidateService:
    def __init__(self, preview: dict, *, error: Exception | None = None) -> None:
        self.preview_payload = preview
        self.error = error
        self.calls: list[str] = []

    async def preview(self, *, billing_period: str) -> dict:
        self.calls.append(billing_period)
        if self.error is not None:
            raise self.error
        return copy.deepcopy(self.preview_payload)


class _BarrierCandidateService(_CandidateService):
    def __init__(self, preview: dict) -> None:
        super().__init__(preview)
        self.two_calls_started = asyncio.Event()
        self.release = asyncio.Event()

    async def preview(self, *, billing_period: str) -> dict:
        self.calls.append(billing_period)
        if len(self.calls) >= 2:
            self.two_calls_started.set()
        await self.release.wait()
        return copy.deepcopy(self.preview_payload)


class _SchemaPool:
    is_initialized = True

    def __init__(self, conn, schema: str) -> None:
        self.conn = conn
        self.schema = schema

    async def acquire(self):
        await self.conn.execute(f'SET search_path TO "{self.schema}"')
        return self.conn

    async def release(self, released) -> None:
        assert released is self.conn

    @asynccontextmanager
    async def transaction(self):
        async with self.conn.transaction():
            await self.conn.execute(f'SET LOCAL search_path TO "{self.schema}"')
            yield self.conn

    async def fetch(self, query, *args):
        async with self.conn.transaction():
            await self.conn.execute(f'SET LOCAL search_path TO "{self.schema}"')
            return await self.conn.fetch(query, *args)

    async def fetchrow(self, query, *args):
        async with self.conn.transaction():
            await self.conn.execute(f'SET LOCAL search_path TO "{self.schema}"')
            return await self.conn.fetchrow(query, *args)


@asynccontextmanager
async def _billing_run_database():
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    schema = f"commercial_billing_runs_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    migrations = Path(__file__).parents[1] / "atlas_brain/storage/migrations"
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        await conn.execute(
            "CREATE TABLE invoices ("
            "id UUID PRIMARY KEY, source TEXT, source_ref TEXT, "
            "invoice_number TEXT, status TEXT, issue_date DATE, due_date DATE, "
            "total_amount NUMERIC, business_context_id TEXT"
            ")"
        )
        for name in (
            "370_commercial_billing_runs.sql",
            "372_commercial_billing_candidate_approvals.sql",
            "380_commercial_billing_candidate_review_decisions.sql",
            "381_commercial_billing_candidate_review_decisions_recovery.sql",
            "382_commercial_billing_candidate_overrides.sql",
        ):
            await conn.execute((migrations / name).read_text(encoding="utf-8"))
        yield conn, schema, database_url
    finally:
        await conn.execute("SET search_path TO public")
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


async def _current_380_schema(conn, schema: str):
    """Build the current 370/372/380 schema through the production runner."""
    from atlas_brain.storage.migrations import run_migrations

    migrations_dir = Path(__file__).parents[1] / "atlas_brain/storage/migrations"
    pool = _SchemaPool(conn, schema)
    await run_migrations(
        pool,
        migrations_dir=migrations_dir,
        only={
            "370_commercial_billing_runs",
            "372_commercial_billing_candidate_approvals",
            "380_commercial_billing_candidate_review_decisions",
        },
    )
    await conn.execute(f'SET search_path TO "{schema}"')
    return pool, migrations_dir


async def _recorded_380_legacy_schema(conn, schema: str):
    """Build the observed recorded-380 shape before later safety DDL landed."""
    pool, migrations_dir = await _current_380_schema(conn, schema)
    assert await conn.fetchval(
        "SELECT COUNT(*) FROM schema_migrations WHERE name = $1",
        "380_commercial_billing_candidate_review_decisions",
    ) == 1

    await conn.execute(
        "DROP TRIGGER IF EXISTS "
        "trg_prevent_commercial_billing_invoice_for_excluded_candidate ON invoices"
    )
    await conn.execute(
        "DROP TRIGGER IF EXISTS "
        "trg_prevent_commercial_billing_review_decision_mutation "
        "ON commercial_billing_candidate_review_decisions"
    )
    await conn.execute(
        "DROP TRIGGER IF EXISTS "
        "trg_prevent_commercial_billing_review_decision_truncate "
        "ON commercial_billing_candidate_review_decisions"
    )
    await conn.execute(
        "DROP FUNCTION IF EXISTS "
        "prevent_commercial_billing_invoice_for_excluded_candidate()"
    )
    await conn.execute(
        "DROP FUNCTION IF EXISTS "
        "prevent_commercial_billing_review_decision_mutation()"
    )
    await conn.execute(
        "DROP INDEX IF EXISTS idx_commercial_billing_run_candidates_identity"
    )
    await conn.execute(
        "ALTER TABLE commercial_billing_candidate_review_decisions "
        "DROP CONSTRAINT commercial_billing_candidate_review_decisions_revision_key"
    )
    await conn.execute(
        "ALTER TABLE commercial_billing_candidate_review_decisions "
        "ADD CONSTRAINT commercial_billing_candidate_review_decisions_revision_key "
        "UNIQUE (billing_run_id, candidate_key, source_fingerprint, revision)"
    )
    return pool, migrations_dir


async def _revision_key_columns(conn) -> list[str] | None:
    columns = await conn.fetchval(
        """
        SELECT ARRAY_AGG(attribute.attname ORDER BY key_column.ordinality)
        FROM pg_constraint AS constraint_state
        JOIN UNNEST(constraint_state.conkey) WITH ORDINALITY
            AS key_column(attnum, ordinality)
            ON TRUE
        JOIN pg_attribute AS attribute
            ON attribute.attrelid = constraint_state.conrelid
           AND attribute.attnum = key_column.attnum
        WHERE constraint_state.conrelid =
                  'commercial_billing_candidate_review_decisions'::regclass
          AND constraint_state.conname =
                  'commercial_billing_candidate_review_decisions_revision_key'
          AND constraint_state.contype = 'u'
        """
    )
    return list(columns) if columns is not None else None


async def _review_decision_safety_catalog(conn, schema: str) -> dict:
    """Return the logical safety catalog that migration 381 must preserve."""
    identity_index = await conn.fetchval(
        """
        SELECT indexdef
        FROM pg_indexes
        WHERE schemaname = $1
          AND tablename = 'commercial_billing_run_candidates'
          AND indexname = 'idx_commercial_billing_run_candidates_identity'
        """,
        schema,
    )
    trigger_rows = await conn.fetch(
        """
        SELECT trigger_state.tgname, relation.relname,
               pg_get_triggerdef(trigger_state.oid) AS definition
        FROM pg_trigger AS trigger_state
        JOIN pg_class AS relation
          ON relation.oid = trigger_state.tgrelid
        JOIN pg_namespace AS namespace_state
          ON namespace_state.oid = relation.relnamespace
        WHERE NOT trigger_state.tgisinternal
          AND namespace_state.nspname = $1
          AND trigger_state.tgname IN (
              'trg_prevent_commercial_billing_review_decision_mutation',
              'trg_prevent_commercial_billing_review_decision_truncate',
              'trg_prevent_commercial_billing_invoice_for_excluded_candidate'
          )
        ORDER BY trigger_state.tgname
        """,
        schema,
    )
    function_rows = await conn.fetch(
        """
        SELECT routine.proname, pg_get_functiondef(routine.oid) AS definition
        FROM pg_proc AS routine
        JOIN pg_namespace AS namespace_state
          ON namespace_state.oid = routine.pronamespace
        WHERE namespace_state.nspname = $1
          AND routine.proname IN (
              'prevent_commercial_billing_review_decision_mutation',
              'prevent_commercial_billing_invoice_for_excluded_candidate'
          )
        ORDER BY routine.proname
        """,
        schema,
    )
    return {
        "revisionKeyColumns": await _revision_key_columns(conn),
        "identityIndex": identity_index,
        "triggers": [dict(row) for row in trigger_rows],
        "functions": [dict(row) for row in function_rows],
    }


def _service(pool, candidate_service: _CandidateService) -> CommercialBillingRunService:
    return CommercialBillingRunService(
        pool=pool,
        candidate_service_loader=lambda: candidate_service,
    )


async def _insert_legacy_run_candidate(
    conn,
    *,
    run_id: UUID,
    candidate: dict,
    idempotency_key: str,
) -> None:
    """Seed a pre-382 run without asking current provider code to serve it."""

    fingerprint = candidate["sourceFingerprint"]
    await conn.execute(
        """
        INSERT INTO commercial_billing_runs (
            id, billing_period, state, candidate_contract_version,
            snapshot_fingerprint, source, idempotency_key,
            request_fingerprint, created_by
        ) VALUES ($1, '2026-03', 'draft', 2, $2, 'eom_admin', $3, $2, 'Migration recovery test')
        """,
        run_id,
        fingerprint,
        idempotency_key,
    )
    await conn.execute(
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


@pytest.mark.asyncio
async def test_real_postgres_snapshot_is_immutable_and_same_key_replays_without_source_read():
    async with _billing_run_database() as (conn, schema, _database_url):
        candidate_service = _CandidateService(
            _preview(
                _candidate(
                    "commercial-billing:acme:2026-03",
                    _fingerprint("a"),
                    blockers=[
                        {
                            "code": "missing_billing_delivery_preference",
                            "eventIds": [],
                            "message": "No explicit delivery preference.",
                            "serviceId": None,
                        }
                    ],
                )
            )
        )
        service = _service(_SchemaPool(conn, schema), candidate_service)

        created = await service.create_run(
            billing_period="2026-03",
            idempotency_key="billing-run-create-1",
            actor="Juan Canfield",
        )

        assert created["replayed"] is False
        run = created["billingRun"]
        assert run["state"] == "draft"
        assert run["createdBy"] == "Juan Canfield"
        assert run["candidateContractVersion"] == 2
        assert run["summary"] == {"blockedCandidateCount": 1, "candidateCount": 1}
        assert run["candidates"][0]["lineItems"][0]["amountCents"] == 9650
        assert run["candidates"][0]["sourceEvents"][0]["location"] == "100 Main St"
        assert run["candidates"][0]["approval"] is None
        assert run["candidates"][0]["reviewDecision"] == {
            "decidedAt": None,
            "decidedBy": None,
            "decision": "included",
            "isExplicit": False,
            "reason": None,
            "revision": 0,
        }
        assert len(run["snapshotFingerprint"]) == 64
        assert candidate_service.calls == ["2026-03"]

        persisted = await conn.fetchrow(
            """
            SELECT
                created_by, billing_period, calendar_id, state, snapshot_fingerprint
            FROM commercial_billing_runs
            """
        )
        assert dict(persisted) == {
            "created_by": "Juan Canfield",
            "billing_period": "2026-03",
            "calendar_id": "commercial-calendar",
            "state": "draft",
            "snapshot_fingerprint": run["snapshotFingerprint"],
        }
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_run_candidates"
        ) == 1

        candidate_service.error = AssertionError("retry must not regenerate sources")
        replayed = await service.create_run(
            billing_period="2026-03",
            idempotency_key="billing-run-create-1",
            actor="Juan Canfield",
        )
        assert replayed == {"billingRun": run, "replayed": True}
        assert candidate_service.calls == ["2026-03"]

        listed = await service.list_runs(billing_period="2026-03", limit=10)
        assert listed["items"] == [
            {
                key: run[key]
                for key in (
                    "billingPeriod",
                    "calendarId",
                    "candidateContractVersion",
                    "createdAt",
                    "createdBy",
                    "id",
                    "snapshotFingerprint",
                    "state",
                    "summary",
                    "updatedAt",
                )
            }
        ]


@pytest.mark.asyncio
async def test_real_postgres_reconciliation_detects_changed_missing_and_new_evidence_without_write():
    async with _billing_run_database() as (conn, schema, _database_url):
        original = _preview(
            _candidate("commercial-billing:alpha:2026-03", _fingerprint("a")),
            _candidate("commercial-billing:bravo:2026-03", _fingerprint("b")),
        )
        candidate_service = _CandidateService(original)
        service = _service(_SchemaPool(conn, schema), candidate_service)
        created = await service.create_run(
            billing_period="2026-03",
            idempotency_key="billing-run-reconcile-1",
            actor="Juan Canfield",
        )
        run = created["billingRun"]
        before = await conn.fetchrow(
            "SELECT updated_at FROM commercial_billing_runs WHERE id = $1",
            UUID(run["id"]),
        )

        current = await service.reconcile_run(UUID(run["id"]))
        assert current["isStale"] is False
        assert [change["status"] for change in current["candidateChanges"]] == [
            "unchanged",
            "unchanged",
        ]

        candidate_service.preview_payload["calendarId"] = "replacement-calendar"
        changed_calendar = await service.reconcile_run(UUID(run["id"]))
        assert changed_calendar["isStale"] is True
        assert [change["status"] for change in changed_calendar["candidateChanges"]] == [
            "unchanged",
            "unchanged",
        ]

        candidate_service.preview_payload = _preview(
            _candidate("commercial-billing:alpha:2026-03", _fingerprint("c")),
            _candidate("commercial-billing:charlie:2026-03", _fingerprint("d")),
        )
        stale = await service.reconcile_run(UUID(run["id"]))

        assert stale["isStale"] is True
        assert {
            change["candidateKey"]: change["status"]
            for change in stale["candidateChanges"]
        } == {
            "commercial-billing:alpha:2026-03": "changed",
            "commercial-billing:bravo:2026-03": "missing",
            "commercial-billing:charlie:2026-03": "new",
        }
        assert stale["snapshotFingerprint"] == run["snapshotFingerprint"]
        assert stale["currentSnapshotFingerprint"] != run["snapshotFingerprint"]
        after = await conn.fetchrow(
            "SELECT updated_at FROM commercial_billing_runs WHERE id = $1",
            UUID(run["id"]),
        )
        assert after["updated_at"] == before["updated_at"]
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_run_candidates"
        ) == 2


@pytest.mark.asyncio
async def test_real_postgres_source_failure_rolls_back_then_retry_recovers():
    async with _billing_run_database() as (conn, schema, _database_url):
        candidate_service = _CandidateService(
            _preview(_candidate("commercial-billing:acme:2026-03", _fingerprint("a"))),
            error=CommercialBillingCandidatesUnavailableError("calendar unavailable"),
        )
        service = _service(_SchemaPool(conn, schema), candidate_service)

        with pytest.raises(CommercialBillingCandidatesUnavailableError):
            await service.create_run(
                billing_period="2026-03",
                idempotency_key="billing-run-source-retry-1",
                actor="Juan Canfield",
            )
        assert await conn.fetchval("SELECT COUNT(*) FROM commercial_billing_runs") == 0
        assert (
            await conn.fetchval("SELECT COUNT(*) FROM commercial_billing_run_candidates")
            == 0
        )

        candidate_service.error = None
        recovered = await service.create_run(
            billing_period="2026-03",
            idempotency_key="billing-run-source-retry-1",
            actor="Juan Canfield",
        )
        assert recovered["replayed"] is False
        assert await conn.fetchval("SELECT COUNT(*) FROM commercial_billing_runs") == 1


@pytest.mark.asyncio
async def test_real_postgres_concurrent_same_key_creates_one_snapshot_set():
    async with _billing_run_database() as (observer, schema, database_url):
        asyncpg = pytest.importorskip("asyncpg")
        first_conn = await asyncpg.connect(database_url)
        second_conn = await asyncpg.connect(database_url)
        await first_conn.execute(f'SET search_path TO "{schema}"')
        await second_conn.execute(f'SET search_path TO "{schema}"')
        candidate_service = _BarrierCandidateService(
            _preview(
                _candidate("commercial-billing:acme:2026-03", _fingerprint("a")),
                _candidate("commercial-billing:bravo:2026-03", _fingerprint("b")),
            )
        )
        try:
            first = _service(_SchemaPool(first_conn, schema), candidate_service)
            second = _service(_SchemaPool(second_conn, schema), candidate_service)
            first_task = asyncio.create_task(
                first.create_run(
                    billing_period="2026-03",
                    idempotency_key="billing-run-concurrent-1",
                    actor="Juan Canfield",
                )
            )
            second_task = asyncio.create_task(
                second.create_run(
                    billing_period="2026-03",
                    idempotency_key="billing-run-concurrent-1",
                    actor="Juan Canfield",
                )
            )
            await asyncio.wait_for(candidate_service.two_calls_started.wait(), timeout=5)
            candidate_service.release.set()
            first_result, second_result = await asyncio.gather(first_task, second_task)
        finally:
            candidate_service.release.set()
            await first_conn.close()
            await second_conn.close()

        assert {first_result["billingRun"]["id"], second_result["billingRun"]["id"]} == {
            first_result["billingRun"]["id"]
        }
        assert {first_result["replayed"], second_result["replayed"]} == {False, True}
        assert await observer.fetchval("SELECT COUNT(*) FROM commercial_billing_runs") == 1
        assert (
            await observer.fetchval(
                "SELECT COUNT(*) FROM commercial_billing_run_candidates"
            )
            == 2
        )


@pytest.mark.asyncio
async def test_same_key_with_another_period_conflicts_without_regenerating_sources():
    async with _billing_run_database() as (conn, schema, _database_url):
        candidate_service = _CandidateService(
            _preview(_candidate("commercial-billing:acme:2026-03", _fingerprint("a")))
        )
        service = _service(_SchemaPool(conn, schema), candidate_service)
        await service.create_run(
            billing_period="2026-03",
            idempotency_key="billing-run-period-conflict-1",
            actor="Juan Canfield",
        )
        with pytest.raises(CommercialBillingRunConflictError):
            await service.create_run(
                billing_period="2026-04",
                idempotency_key="billing-run-period-conflict-1",
                actor="Juan Canfield",
            )
        assert candidate_service.calls == ["2026-03"]
        assert await conn.fetchval("SELECT COUNT(*) FROM commercial_billing_runs") == 1


@pytest.mark.asyncio
async def test_real_postgres_review_decisions_are_append_only_idempotent_and_derived():
    async with _billing_run_database() as (conn, schema, _database_url):
        candidate = _candidate("commercial-billing:review:2026-03", _fingerprint("a"))
        service = _service(_SchemaPool(conn, schema), _CandidateService(_preview(candidate)))
        created_run = await service.create_run(
            billing_period="2026-03",
            idempotency_key="billing-run-review-decision-1",
            actor="Juan Canfield",
        )
        run_id = UUID(created_run["billingRun"]["id"])
        source_fingerprint = candidate["sourceFingerprint"]
        before_updated_at = await conn.fetchval(
            "SELECT updated_at FROM commercial_billing_runs WHERE id = $1",
            run_id,
        )

        excluded = await service.set_candidate_review_decision(
            billing_run_id=run_id,
            candidate_key=candidate["candidateKey"],
            expected_source_fingerprint=source_fingerprint,
            decision="excluded",
            reason="Customer is resolving a service question.",
            idempotency_key="review-exclude-1",
            actor="Juan Canfield",
        )
        assert excluded["replayed"] is False
        assert excluded["reviewDecision"] == {
            "decidedAt": excluded["reviewDecision"]["decidedAt"],
            "decidedBy": "Juan Canfield",
            "decision": "excluded",
            "id": excluded["reviewDecision"]["id"],
            "isExplicit": True,
            "reason": "Customer is resolving a service question.",
            "revision": 1,
        }

        replayed = await service.set_candidate_review_decision(
            billing_run_id=run_id,
            candidate_key=candidate["candidateKey"],
            expected_source_fingerprint=source_fingerprint,
            decision="excluded",
            reason="Customer is resolving a service question.",
            idempotency_key="review-exclude-1",
            actor="Another operator",
        )
        assert replayed == {**excluded, "replayed": True}
        with pytest.raises(CommercialBillingRunConflictError):
            await service.set_candidate_review_decision(
                billing_run_id=run_id,
                candidate_key=candidate["candidateKey"],
                expected_source_fingerprint=source_fingerprint,
                decision="excluded",
                reason="A different reason must not reuse the operation key.",
                idempotency_key="review-exclude-1",
                actor="Juan Canfield",
            )

        included = await service.set_candidate_review_decision(
            billing_run_id=run_id,
            candidate_key=candidate["candidateKey"],
            expected_source_fingerprint=source_fingerprint,
            decision="included",
            reason="Service question resolved.",
            idempotency_key="review-include-1",
            actor="Juan Canfield",
        )
        assert included["replayed"] is False
        assert included["reviewDecision"]["decision"] == "included"
        assert included["reviewDecision"]["revision"] == 2

        detail = await service.get_run(run_id)
        assert detail["candidates"][0]["reviewDecision"] == included["reviewDecision"]
        snapshot = await conn.fetchval(
            "SELECT snapshot FROM commercial_billing_run_candidates WHERE billing_run_id = $1",
            run_id,
        )
        assert "reviewDecision" not in snapshot
        history = await conn.fetch(
            """
            SELECT revision, decision, reason, decided_by
            FROM commercial_billing_candidate_review_decisions
            ORDER BY revision
            """
        )
        assert [dict(row) for row in history] == [
            {
                "revision": 1,
                "decision": "excluded",
                "reason": "Customer is resolving a service question.",
                "decided_by": "Juan Canfield",
            },
            {
                "revision": 2,
                "decision": "included",
                "reason": "Service question resolved.",
                "decided_by": "Juan Canfield",
            },
        ]
        assert await conn.fetchval("SELECT COUNT(*) FROM invoices") == 0
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM commercial_billing_candidate_approvals"
            )
            == 0
        )
        assert (
            await conn.fetchval(
                "SELECT updated_at FROM commercial_billing_runs WHERE id = $1",
                run_id,
            )
            == before_updated_at
        )


@pytest.mark.asyncio
async def test_real_postgres_review_idempotency_canonicalizes_no_override_identity_variants():
    """Omitted and explicit legacy identities are the same retry request."""

    async with _billing_run_database() as (conn, schema, _database_url):
        candidate = _candidate(
            "commercial-billing:review-idempotency:2026-03", _fingerprint("a")
        )
        service = _service(_SchemaPool(conn, schema), _CandidateService(_preview(candidate)))
        created_run = await service.create_run(
            billing_period="2026-03",
            idempotency_key="billing-run-review-idempotency-1",
            actor="Juan Canfield",
        )
        run_id = UUID(created_run["billingRun"]["id"])
        source_fingerprint = candidate["sourceFingerprint"]

        for index, (first_identity, retry_identity, decision) in enumerate(
            (
                (None, source_fingerprint, "excluded"),
                (source_fingerprint, None, "included"),
            ),
            start=1,
        ):
            created = await service.set_candidate_review_decision(
                billing_run_id=run_id,
                candidate_key=candidate["candidateKey"],
                expected_source_fingerprint=source_fingerprint,
                expected_review_fingerprint=first_identity,
                decision=decision,
                reason=f"Canonical retry identity case {index}.",
                idempotency_key=f"review-identity-{index}",
                actor="Juan Canfield",
            )
            replayed = await service.set_candidate_review_decision(
                billing_run_id=run_id,
                candidate_key=candidate["candidateKey"],
                expected_source_fingerprint=source_fingerprint,
                expected_review_fingerprint=retry_identity,
                decision=decision,
                reason=f"Canonical retry identity case {index}.",
                idempotency_key=f"review-identity-{index}",
                actor="Juan Canfield",
            )

            assert created["replayed"] is False
            assert replayed == {**created, "replayed": True}

        assert await conn.fetchval(
            "SELECT COUNT(*) FROM commercial_billing_candidate_review_decisions"
        ) == 2


@pytest.mark.asyncio
async def test_real_postgres_review_decision_history_rejects_direct_mutation():
    asyncpg = pytest.importorskip("asyncpg")
    async with _billing_run_database() as (conn, schema, _database_url):
        candidate = _candidate("commercial-billing:immutable:2026-03", _fingerprint("a"))
        service = _service(_SchemaPool(conn, schema), _CandidateService(_preview(candidate)))
        created_run = await service.create_run(
            billing_period="2026-03",
            idempotency_key="billing-run-immutable-history-1",
            actor="Juan Canfield",
        )
        run_id = UUID(created_run["billingRun"]["id"])
        recorded = await service.set_candidate_review_decision(
            billing_run_id=run_id,
            candidate_key=candidate["candidateKey"],
            expected_source_fingerprint=candidate["sourceFingerprint"],
            decision="excluded",
            reason="Keep the original review evidence intact.",
            idempotency_key="review-immutable-history-1",
            actor="Juan Canfield",
        )
        decision_id = UUID(recorded["reviewDecision"]["id"])

        mutations = (
            (
                "UPDATE commercial_billing_candidate_review_decisions "
                "SET reason = 'rewritten' WHERE id = $1",
                (decision_id,),
            ),
            (
                "DELETE FROM commercial_billing_candidate_review_decisions WHERE id = $1",
                (decision_id,),
            ),
            ("TRUNCATE commercial_billing_candidate_review_decisions", ()),
        )
        for statement, arguments in mutations:
            with pytest.raises(asyncpg.PostgresError, match="append-only"):
                await conn.execute(statement, *arguments)

        history = await conn.fetchrow(
            "SELECT id, decision, reason FROM commercial_billing_candidate_review_decisions"
        )
        assert dict(history) == {
            "id": decision_id,
            "decision": "excluded",
            "reason": "Keep the original review evidence intact.",
        }


@pytest.mark.asyncio
async def test_real_postgres_candidate_lock_does_not_serialize_other_candidates():
    asyncpg = pytest.importorskip("asyncpg")
    async with _billing_run_database() as (observer, schema, database_url):
        first_candidate = _candidate("commercial-billing:lock-one:2026-03", _fingerprint("a"))
        second_candidate = _candidate("commercial-billing:lock-two:2026-03", _fingerprint("b"))
        service = _service(
            _SchemaPool(observer, schema),
            _CandidateService(_preview(first_candidate, second_candidate)),
        )
        created_run = await service.create_run(
            billing_period="2026-03",
            idempotency_key="billing-run-lock-scope-1",
            actor="Juan Canfield",
        )
        run_id = UUID(created_run["billingRun"]["id"])
        first_conn = await asyncpg.connect(database_url)
        second_conn = await asyncpg.connect(database_url)
        first_transaction = first_conn.transaction()
        first_transaction_started = False
        second_task = None
        try:
            await first_conn.execute(f'SET search_path TO "{schema}"')
            await second_conn.execute(f'SET search_path TO "{schema}"')
            await first_transaction.start()
            first_transaction_started = True
            locked = await lock_commercial_billing_run_candidate(
                first_conn,
                billing_run_id=run_id,
                candidate_key=first_candidate["candidateKey"],
            )
            assert locked["source_fingerprint"] == first_candidate["sourceFingerprint"]

            second_task = asyncio.create_task(
                lock_commercial_billing_run_candidate(
                    second_conn,
                    billing_run_id=run_id,
                    candidate_key=second_candidate["candidateKey"],
                )
            )
            independently_locked = await asyncio.wait_for(
                asyncio.shield(second_task), timeout=0.5
            )
            assert independently_locked["source_fingerprint"] == second_candidate[
                "sourceFingerprint"
            ]
        finally:
            if first_transaction_started:
                await first_transaction.rollback()
            if second_task is not None and not second_task.done():
                second_task.cancel()
                await asyncio.gather(second_task, return_exceptions=True)
            await first_conn.close()
            await second_conn.close()


@pytest.mark.asyncio
async def test_review_decision_rejects_invalid_unknown_and_stale_input_without_writing():
    async with _billing_run_database() as (conn, schema, _database_url):
        candidate = _candidate("commercial-billing:reject:2026-03", _fingerprint("a"))
        service = _service(_SchemaPool(conn, schema), _CandidateService(_preview(candidate)))
        created_run = await service.create_run(
            billing_period="2026-03",
            idempotency_key="billing-run-review-reject-1",
            actor="Juan Canfield",
        )
        run_id = UUID(created_run["billingRun"]["id"])

        with pytest.raises(CommercialBillingRunNotFoundError):
            await service.set_candidate_review_decision(
                billing_run_id=run_id,
                candidate_key="commercial-billing:missing:2026-03",
                expected_source_fingerprint=_fingerprint("a"),
                decision="excluded",
                reason="No matching retained candidate.",
                idempotency_key="review-missing-1",
                actor="Juan Canfield",
            )
        with pytest.raises(CommercialBillingRunConflictError):
            await service.set_candidate_review_decision(
                billing_run_id=run_id,
                candidate_key=candidate["candidateKey"],
                expected_source_fingerprint=_fingerprint("b"),
                decision="excluded",
                reason="Stale browser evidence.",
                idempotency_key="review-stale-1",
                actor="Juan Canfield",
            )
        with pytest.raises(CommercialBillingRunValidationError):
            await service.set_candidate_review_decision(
                billing_run_id=run_id,
                candidate_key=candidate["candidateKey"],
                expected_source_fingerprint=candidate["sourceFingerprint"],
                decision="deleted",
                reason=" ",
                idempotency_key="review-invalid-1",
                actor="Juan Canfield",
            )

        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM commercial_billing_candidate_review_decisions"
            )
            == 0
        )


@pytest.mark.asyncio
async def test_review_decision_input_grammar_fails_closed_across_boundary_forms():
    """Generate contract grammar variants rather than a fixture list.

    Candidate key, reason, idempotency key, and actor are OPEN scalar-text
    families. The specification-derived oracle admits only a nonblank trimmed
    scalar within each field's limit and with database-encodable text; list,
    tuple, set, nested, and mapping containers reject. Fingerprints
    admit only 64 lower-hex characters. The review decision is a CLOSED
    vocabulary authored by this billing contract: included or excluded. Every
    unrecognized form must reject before the transaction starts, so it cannot
    create an audit row or invoice.
    """

    pool = _NoStoredRunPool()
    service = _service(pool, _CandidateService(_preview()))
    run_id = uuid4()
    valid_candidate = "commercial-billing:grammar:2026-03"
    valid_fingerprint = _fingerprint("a")

    async def observed(**overrides):
        before_transactions = pool.conn.transaction_count
        try:
            await service.set_candidate_review_decision(
                billing_run_id=run_id,
                candidate_key=overrides.get("candidate_key", valid_candidate),
                expected_source_fingerprint=overrides.get(
                    "expected_source_fingerprint", valid_fingerprint
                ),
                decision=overrides.get("decision", "excluded"),
                reason=overrides.get("reason", "Review boundary grammar."),
                idempotency_key=overrides.get("idempotency_key", "review-grammar-1"),
                actor=overrides.get("actor", "Juan Canfield"),
            )
        except CommercialBillingRunValidationError:
            return False, pool.conn.transaction_count == before_transactions
        except CommercialBillingRunNotFoundError:
            return True, pool.conn.transaction_count == before_transactions + 1
        raise AssertionError("the no-stored-candidate pool must not accept a decision")

    containers = (
        ("scalar", lambda value: value),
        ("list", lambda value: [value]),
        ("tuple", lambda value: (value,)),
        ("set", lambda value: {value}),
        ("nested", lambda value: [[value]]),
        ("wrapped", lambda value: {"value": value}),
        ("none", lambda _value: None),
    )
    modifiers = ("", " ", "\t", "\n")

    def database_text_is_admissible(value):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            return False
        return "\x00" not in value

    text_values = (
        "",
        "x",
        "commercial-billing:acme:2026-03",
        "x" * 128,
        "x" * 129,
        "x" * 512,
        "x" * 513,
        "x" * 1001,
        "valid\x00text",
        "\ud800",
    )
    text_families = (
        ("candidate_key", 512),
        ("reason", 1000),
        ("actor", 128),
        ("idempotency_key", 128),
    )
    for (field, limit), value, modifier, (_container_name, container) in product(
        text_families, text_values, modifiers, containers
    ):
        raw = container(f"{modifier}{value}{modifier}")
        expected = (
            isinstance(raw, str)
            and database_text_is_admissible(raw)
            and 1 <= len(raw.strip()) <= limit
        )
        accepted, prewrite_boundary_held = await observed(**{field: raw})
        assert accepted is expected
        assert prewrite_boundary_held is True

    fingerprint_tokens = ("0", "a", "f", "A", "g")
    fingerprint_lengths = (63, 64, 65)
    for token, length, modifier, (_container_name, container) in product(
        fingerprint_tokens, fingerprint_lengths, modifiers, containers
    ):
        raw = container(f"{modifier}{token * length}{modifier}")
        expected = (
            isinstance(raw, str)
            and len(raw) == 64
            and all(character in "0123456789abcdef" for character in raw)
        )
        accepted, prewrite_boundary_held = await observed(
            expected_source_fingerprint=raw
        )
        assert accepted is expected
        assert prewrite_boundary_held is True

    decision_tokens = ("included", "excluded", "deleted", "", "INCLUDED")
    for token, modifier, (_container_name, container) in product(
        decision_tokens, modifiers, containers
    ):
        raw = container(f"{modifier}{token}{modifier}")
        expected = isinstance(raw, str) and raw in {"included", "excluded"}
        accepted, prewrite_boundary_held = await observed(decision=raw)
        assert accepted is expected
        assert prewrite_boundary_held is True

    assert pool.conn.write_attempts == 0


class _NoStoredRunConnection:
    def __init__(self) -> None:
        self.transaction_count = 0
        self.write_attempts = 0

    async def fetchval(self, query, *_args):
        assert "pg_advisory_xact_lock" in query
        return None

    async def fetchrow(self, query, *_args):
        if "commercial_billing_candidate_review_decisions" in query:
            return None
        assert "FROM commercial_billing_runs" in query
        return None

    async def execute(self, *_args):
        self.write_attempts += 1
        raise AssertionError("invalid source evidence must not write a run")


class _NoStoredRunPool:
    is_initialized = True

    def __init__(self) -> None:
        self.conn = _NoStoredRunConnection()

    @asynccontextmanager
    async def transaction(self):
        self.conn.transaction_count += 1
        yield self.conn


@pytest.mark.asyncio
async def test_invalid_generated_evidence_fails_before_snapshot_write():
    candidate_service = _CandidateService(
        _preview(
            _candidate("commercial-billing:duplicate:2026-03", _fingerprint("a")),
            _candidate("commercial-billing:duplicate:2026-03", _fingerprint("b")),
        )
    )
    pool = _NoStoredRunPool()
    service = _service(pool, candidate_service)

    with pytest.raises(CommercialBillingCandidatesValidationError):
        await service.create_run(
            billing_period="2026-3",
            idempotency_key="billing-run-invalid-period-1",
            actor="Juan Canfield",
        )
    assert candidate_service.calls == []
    assert pool.conn.transaction_count == 0

    with pytest.raises(CommercialBillingRunUnavailableError, match="keys must be unique"):
        await service.create_run(
            billing_period="2026-03",
            idempotency_key="billing-run-duplicate-key-1",
            actor="Juan Canfield",
        )
    assert candidate_service.calls == ["2026-03"]
    assert pool.conn.transaction_count == 1
    assert pool.conn.write_attempts == 0

    invalid_money = _preview(
        _candidate("commercial-billing:invalid-money:2026-03", _fingerprint("c"))
    )
    invalid_money["candidates"][0]["lineItems"][0]["amountCents"] = float("nan")
    candidate_service.preview_payload = invalid_money
    with pytest.raises(CommercialBillingRunUnavailableError, match="not JSON-safe"):
        await service.create_run(
            billing_period="2026-03",
            idempotency_key="billing-run-invalid-money-1",
            actor="Juan Canfield",
        )
    assert pool.conn.write_attempts == 0


def test_generated_preview_admission_has_a_grammar_derived_oracle():
    """Exercise every admitted token/container/family combination at the seam."""

    tokens = "candidateKey sourceFingerprint blockers".split()
    containers = "mapping json-object".split()
    families = "accepted rejected".split()

    for token, container, family in product(tokens, containers, families):
        candidate = _candidate("commercial-billing:grammar:2026-03", _fingerprint("a"))
        if family == "rejected":
            if token == "candidateKey":
                candidate[token] = " commercial-billing:grammar:2026-03"
            elif token == "sourceFingerprint":
                candidate[token] = _fingerprint("g")
            else:
                candidate[token] = {"not": "a list"}

        raw_candidate = candidate if container == "mapping" else json.dumps(candidate)
        preview = {
            "billingPeriod": "2026-03",
            "calendarId": "commercial-calendar",
            "candidates": [raw_candidate],
            "contractVersion": 2,
        }
        expected = family == "accepted"
        try:
            normalized = _normalize_preview(preview, billing_period="2026-03")
        except CommercialBillingRunUnavailableError:
            observed = False
        else:
            observed = True
            assert normalized.candidates[0].candidate_key == candidate["candidateKey"]

        assert observed is expected


def test_normalize_preview_keeps_legacy_candidate_contract_versions_readable():
    """A deployed v1 billing-run snapshot remains readable after v2 preview."""
    preview = _preview(_candidate("commercial-billing:legacy:2026-03", _fingerprint("a")))
    preview["contractVersion"] = 1

    normalized = _normalize_preview(preview, billing_period="2026-03")

    assert normalized.contract_version == 1
class _RouteRunService:
    def __init__(self) -> None:
        self.create_calls: list[tuple[str, str, str]] = []
        self.decision_calls: list[dict] = []
        self.override_calls: list[dict] = []
        self.get_calls: list[UUID] = []
        self.reconcile_calls: list[UUID] = []

    async def create_run(self, *, billing_period: str, idempotency_key: str, actor: str):
        self.create_calls.append((billing_period, idempotency_key, actor))
        if billing_period != "2026-03":
            raise CommercialBillingCandidatesValidationError(
                "billing_period must use YYYY-MM"
            )
        return {
            "billingRun": {"id": "00000000-0000-0000-0000-000000000001"},
            "replayed": False,
        }

    async def list_runs(self, *, billing_period, limit, offset):
        return {"items": [], "limit": limit, "offset": offset}

    async def set_candidate_review_decision(self, **kwargs):
        self.decision_calls.append(kwargs)
        return {
            "reviewDecision": {
                "decision": kwargs["decision"],
                "id": "review-decision-1",
                "isExplicit": True,
            },
            "replayed": False,
        }

    async def set_candidate_override(self, **kwargs):
        self.override_calls.append(kwargs)
        return {
            "candidate": {"reviewFingerprint": _fingerprint("b")},
            "override": {"revision": 1},
            "replayed": False,
        }

    async def get_run(self, run_id: UUID):
        self.get_calls.append(run_id)
        return {"id": str(run_id)}

    async def reconcile_run(self, run_id: UUID):
        self.reconcile_calls.append(run_id)
        return {"billingRunId": str(run_id), "isStale": False}


def _route_app(service: _RouteRunService) -> tuple[FastAPI, str]:
    from atlas_brain.api.invoicing import auth as receivables_auth
    from atlas_brain.api.invoicing import receivables as routes
    from atlas_brain.eom_api.auth import generate_receivables_service_token

    generated = generate_receivables_service_token()
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[receivables_auth.get_receivables_api_config] = lambda: (
        SimpleNamespace(
            receivables_api_enabled=True,
            receivables_service_token="",
            receivables_service_token_sha256=generated.sha256,
        )
    )
    app.dependency_overrides[routes.get_commercial_billing_run_service] = lambda: service
    return app, generated.token


@pytest.mark.asyncio
async def test_full_provider_run_routes_authenticate_actor_and_expose_durable_reads():
    service = _RouteRunService()
    app, token = _route_app(service)
    run_id = "00000000-0000-0000-0000-000000000001"
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://receivables.test",
    ) as client:
        path = "/receivables/commercial-billing-runs"
        assert (await client.post(path, json={"billing_period": "2026-03"})).status_code == 401
        assert service.create_calls == []

        no_actor = await client.post(
            path,
            json={"billing_period": "2026-03"},
            headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "route-1"},
        )
        assert no_actor.status_code == 422
        assert service.create_calls == []

        headers = {
            "Authorization": f"Bearer {token}",
            "Idempotency-Key": "route-1",
            "X-EOM-Actor": "Juan Canfield",
        }
        malformed = await client.post(
            path,
            json={"billing_period": "2026-3"},
            headers=headers,
        )
        assert malformed.status_code == 422
        assert service.create_calls == []

        created = await client.post(
            path,
            json={"billing_period": "2026-03"},
            headers=headers,
        )
        listed = await client.get(path, headers={"Authorization": f"Bearer {token}"})
        detail = await client.get(
            f"{path}/{run_id}", headers={"Authorization": f"Bearer {token}"}
        )
        reconciliation = await client.get(
            f"{path}/{run_id}/reconciliation",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert created.status_code == 201
    assert created.json()["replayed"] is False
    assert listed.json() == {"items": [], "limit": 50, "offset": 0}
    assert detail.json() == {"id": run_id}
    assert reconciliation.json() == {"billingRunId": run_id, "isStale": False}
    assert service.create_calls == [("2026-03", "route-1", "Juan Canfield")]
    assert service.get_calls == [UUID(run_id)]
    assert service.reconcile_calls == [UUID(run_id)]


@pytest.mark.asyncio
async def test_review_decision_route_requires_auth_actor_shape_and_idempotency():
    service = _RouteRunService()
    app, token = _route_app(service)
    run_id = "00000000-0000-0000-0000-000000000001"
    candidate_key = "commercial-billing:acme:2026-03"
    path = (
        f"/receivables/commercial-billing-runs/{run_id}/candidates/"
        f"{candidate_key}/review-decision"
    )
    body = {
        "expected_source_fingerprint": _fingerprint("a"),
        "decision": "excluded",
        "reason": "Resolve a customer question before approval.",
    }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://receivables.test",
    ) as client:
        assert (await client.put(path, json=body)).status_code == 401
        assert service.decision_calls == []

        no_actor = await client.put(
            path,
            json=body,
            headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "route-1"},
        )
        assert no_actor.status_code == 422
        assert service.decision_calls == []

        headers = {
            "Authorization": f"Bearer {token}",
            "Idempotency-Key": "route-1",
            "X-EOM-Actor": "Juan Canfield",
        }
        malformed = await client.put(
            path,
            json={**body, "decision": "deleted"},
            headers=headers,
        )
        assert malformed.status_code == 422
        assert service.decision_calls == []

        blank_reason = await client.put(
            path,
            json={**body, "reason": " \t "},
            headers=headers,
        )
        assert blank_reason.status_code == 422
        assert service.decision_calls == []

        accepted = await client.put(path, json=body, headers=headers)

    assert accepted.status_code == 200
    assert accepted.json()["reviewDecision"] == {
        "decision": "excluded",
        "id": "review-decision-1",
        "isExplicit": True,
    }
    assert service.decision_calls == [
        {
            "billing_run_id": UUID(run_id),
            "candidate_key": candidate_key,
            "expected_source_fingerprint": _fingerprint("a"),
            "decision": "excluded",
            "reason": "Resolve a customer question before approval.",
            "idempotency_key": "route-1",
            "actor": "Juan Canfield",
        }
    ]


@pytest.mark.asyncio
async def test_candidate_override_route_bounds_the_operator_edit_surface_and_actor_audit():
    service = _RouteRunService()
    app, token = _route_app(service)
    run_id = "00000000-0000-0000-0000-000000000001"
    candidate_key = "commercial-billing:acme:2026-03"
    path = (
        f"/receivables/commercial-billing-runs/{run_id}/candidates/"
        f"{candidate_key}/override"
    )
    line_key = _fingerprint("c")
    body = {
        "expected_source_fingerprint": _fingerprint("a"),
        "expected_override_revision": 0,
        "reason_code": "one_time_service_variation",
        "reason": "The customer asked for 75 minutes after hours.",
        "line_overrides": [
            {
                "line_key": line_key,
                "description": "After-hours cleaning",
                "rate_cents": 4825,
                "quantity_minutes": 75,
            }
        ],
        "adjustment": {
            "kind": "charge",
            "description": "One-time access fee",
            "amount_cents": 17,
        },
        "recipient": {"display_name": "Acme AP", "email": "billing@example.test"},
        "delivery_method": "gmail_pdf",
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Idempotency-Key": "override-route-1",
        "X-EOM-Actor": "Juan Canfield",
    }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://receivables.test",
    ) as client:
        assert (await client.post(path, json=body)).status_code == 401
        assert service.override_calls == []
        assert (
            await client.post(
                path,
                json=body,
                headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "override-route-1"},
            )
        ).status_code == 422
        assert service.override_calls == []
        assert (
            await client.post(
                path,
                json={**body, "source_date": "2026-03-04"},
                headers=headers,
            )
        ).status_code == 422
        assert service.override_calls == []
        assert (
            await client.post(
                path,
                json={
                    **body,
                    "line_overrides": [{"line_key": line_key, "quantity_minutes": 75.0}],
                },
                headers=headers,
            )
        ).status_code == 422
        assert service.override_calls == []
        accepted = await client.post(path, json=body, headers=headers)

    assert accepted.status_code == 201
    assert accepted.json()["override"] == {"revision": 1}
    assert service.override_calls == [
        {
            "billing_run_id": UUID(run_id),
            "candidate_key": candidate_key,
            "expected_source_fingerprint": _fingerprint("a"),
            "expected_override_revision": 0,
            "reason_code": "one_time_service_variation",
            "reason": "The customer asked for 75 minutes after hours.",
            "line_overrides": [
                {
                    "lineKey": line_key,
                    "description": "After-hours cleaning",
                    "rateCents": 4825,
                    "quantityMinutes": 75,
                }
            ],
            "adjustment": {
                "kind": "charge",
                "description": "One-time access fee",
                "amountCents": 17,
            },
            "recipient": {"displayName": "Acme AP", "email": "billing@example.test"},
            "delivery_method": "gmail_pdf",
            "idempotency_key": "override-route-1",
            "actor": "Juan Canfield",
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("amountCents", 9650),
        ("recipient", {"email": "billing@example.test"}),
        ("deliveryMethod", "gmail_pdf"),
        ("lineItems", []),
        ("taxRateBasisPoints", 0),
        ("approved", False),
        ("operationNote", {"reason": "unsupported"}),
    ),
)
async def test_review_decision_route_rejects_unrecognized_fields_without_writer(
    field: str, value: object
):
    service = _RouteRunService()
    app, token = _route_app(service)
    run_id = "00000000-0000-0000-0000-000000000001"
    candidate_key = "commercial-billing:acme:2026-03"
    path = (
        f"/receivables/commercial-billing-runs/{run_id}/candidates/"
        f"{candidate_key}/review-decision"
    )
    body = {
        "expected_source_fingerprint": _fingerprint("a"),
        "decision": "excluded",
        "reason": "Reject unsupported billing review data.",
        field: value,
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Idempotency-Key": "route-extra-field-1",
        "X-EOM-Actor": "Juan Canfield",
    }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://receivables.test",
    ) as client:
        response = await client.put(path, json=body, headers=headers)

    assert response.status_code == 422
    assert service.decision_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_reason", "expected_status", "normalized_reason"),
    (
        (" " + "a" * 1000, 200, "a" * 1000),
        ("\t" + "a" * 1000 + "\n", 200, "a" * 1000),
        (" \t" + "a" * 999 + "\n ", 200, "a" * 999),
        (" " + "a" * 1001, 422, None),
        ("\t" + "a" * 1001 + "\n", 422, None),
        (" \t\n ", 422, None),
        ("", 422, None),
    ),
    ids=(
        "leading-space-at-limit",
        "tab-and-newline-at-limit",
        "mixed-whitespace-within-limit",
        "leading-space-over-limit",
        "tab-and-newline-over-limit",
        "whitespace-only",
        "empty",
    ),
)
async def test_review_decision_route_normalizes_reason_before_length_validation(
    raw_reason: str,
    expected_status: int,
    normalized_reason: str | None,
):
    """Exercise seven whitespace/boundary shapes through the real route model."""

    service = _RouteRunService()
    app, token = _route_app(service)
    run_id = "00000000-0000-0000-0000-000000000001"
    candidate_key = "commercial-billing:acme:2026-03"
    path = (
        f"/receivables/commercial-billing-runs/{run_id}/candidates/"
        f"{candidate_key}/review-decision"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Idempotency-Key": "reason1",
        "X-EOM-Actor": "Juan Canfield",
    }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://receivables.test",
    ) as client:
        response = await client.put(
            path,
            json={
                "expected_source_fingerprint": _fingerprint("a"),
                "decision": "excluded",
                "reason": raw_reason,
            },
            headers=headers,
        )

    assert response.status_code == expected_status
    if normalized_reason is None:
        assert service.decision_calls == []
    else:
        assert service.decision_calls[0]["reason"] == normalized_reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_idempotency_key",
    (
        " " + "a" * 128,
        "\t" + "b" * 128,
        "c" * 128 + " ",
        " \t" + "d" * 126 + " \t",
        "e" * 127 + "\t",
        "f" * 128,
    ),
    ids=(
        "leading-space-at-limit",
        "leading-tab-at-limit",
        "trailing-space-at-limit",
        "mixed-padding-within-limit",
        "trailing-tab-within-limit",
        "unmodified-at-limit",
    ),
)
async def test_review_decision_route_defers_padded_idempotency_key_length_to_service(
    raw_idempotency_key: str,
):
    """The route must not reject a key before the shared trim-then-bound guard."""

    service = _RouteRunService()
    app, token = _route_app(service)
    run_id = "00000000-0000-0000-0000-000000000001"
    candidate_key = "commercial-billing:acme:2026-03"
    path = (
        f"/receivables/commercial-billing-runs/{run_id}/candidates/"
        f"{candidate_key}/review-decision"
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://receivables.test",
    ) as client:
        response = await client.put(
            path,
            json={
                "expected_source_fingerprint": _fingerprint("a"),
                "decision": "excluded",
                "reason": "Idempotency header boundary proof.",
            },
            headers={
                "Authorization": f"Bearer {token}",
                "Idempotency-Key": raw_idempotency_key,
                "X-EOM-Actor": "Juan Canfield",
            },
        )

    assert response.status_code == 200
    assert service.decision_calls[0]["idempotency_key"] == raw_idempotency_key


@pytest.mark.asyncio
async def test_full_atlas_app_persists_review_decision_route_under_api_v1():
    from atlas_brain.api.invoicing import auth as receivables_auth
    from atlas_brain.api.invoicing import receivables as routes
    from atlas_brain.eom_api.auth import generate_receivables_service_token
    from atlas_brain.main import app

    async with _billing_run_database() as (conn, schema, _database_url):
        candidate = _candidate(
            "commercial-billing:route-persistence:2026-03", _fingerprint("a")
        )
        service = _service(
            _SchemaPool(conn, schema), _CandidateService(_preview(candidate))
        )
        created = await service.create_run(
            billing_period="2026-03",
            idempotency_key="billing-run-route-persistence-1",
            actor="Juan Canfield",
        )
        run_id = UUID(created["billingRun"]["id"])
        generated = generate_receivables_service_token()
        original_overrides = dict(app.dependency_overrides)
        path = (
            f"/api/v1/receivables/commercial-billing-runs/{run_id}/candidates/"
            f"{candidate['candidateKey']}/review-decision"
        )
        canonical_idempotency_key = "k" * 128
        padded_idempotency_key = " " + canonical_idempotency_key
        valid_body = {
            "expected_source_fingerprint": candidate["sourceFingerprint"],
            "decision": "included",
            "reason": "Reviewed and ready for explicit approval.",
        }
        app.dependency_overrides[receivables_auth.get_receivables_api_config] = lambda: (
            SimpleNamespace(
                receivables_api_enabled=True,
                receivables_service_token="",
                receivables_service_token_sha256=generated.sha256,
            )
        )
        app.dependency_overrides[routes.get_commercial_billing_run_service] = lambda: service
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://atlas.test",
            ) as client:
                assert (await client.put(path)).status_code == 401
                normally_rejected = await client.put(
                    path,
                    json={
                        "expected_source_fingerprint": candidate["sourceFingerprint"],
                        "decision": "included",
                        "reason": " ",
                    },
                    headers={
                        "Authorization": f"Bearer {generated.token}",
                        "Idempotency-Key": "full-route-invalid-blank-1",
                        "X-EOM-Actor": "Juan Canfield",
                    },
                )
                rejected = await client.put(
                    path,
                    content=json.dumps(
                        {
                            "expected_source_fingerprint": candidate["sourceFingerprint"],
                            "decision": "included",
                            "reason": "\ud800",
                        },
                        ensure_ascii=True,
                    ).encode("ascii"),
                    headers={
                        "Authorization": f"Bearer {generated.token}",
                        "Content-Type": "application/json",
                        "Idempotency-Key": "full-route-invalid-1",
                        "X-EOM-Actor": "Juan Canfield",
                    },
                )
                blank_idempotency = await client.put(
                    path,
                    json=valid_body,
                    headers={
                        "Authorization": f"Bearer {generated.token}",
                        "Idempotency-Key": " ",
                        "X-EOM-Actor": "Juan Canfield",
                    },
                )
                normalized_over_limit_idempotency = await client.put(
                    path,
                    json=valid_body,
                    headers={
                        "Authorization": f"Bearer {generated.token}",
                        "Idempotency-Key": " " + "o" * 129,
                        "X-EOM-Actor": "Juan Canfield",
                    },
                )
                accepted = await client.put(
                    path,
                    json=valid_body,
                    headers={
                        "Authorization": f"Bearer {generated.token}",
                        "Idempotency-Key": padded_idempotency_key,
                        "X-EOM-Actor": "Juan Canfield",
                    },
                )
                replayed = await client.put(
                    path,
                    json=valid_body,
                    headers={
                        "Authorization": f"Bearer {generated.token}",
                        "Idempotency-Key": canonical_idempotency_key,
                        "X-EOM-Actor": "Juan Canfield",
                    },
                )
        finally:
            app.dependency_overrides.clear()
            app.dependency_overrides.update(original_overrides)

        assert accepted.status_code == 200
        assert replayed.status_code == 200
        assert normally_rejected.status_code == 422
        assert rejected.status_code == 422
        assert blank_idempotency.status_code == 422
        assert normalized_over_limit_idempotency.status_code == 422
        response_decision = accepted.json()["reviewDecision"]
        assert accepted.json()["replayed"] is False
        assert replayed.json() == {**accepted.json(), "replayed": True}
        assert response_decision["decision"] == "included"
        assert response_decision["reason"] == "Reviewed and ready for explicit approval."
        assert response_decision["decidedBy"] == "Juan Canfield"
        persisted = await conn.fetchrow(
            """
            SELECT billing_run_id, candidate_key, source_fingerprint, revision,
                   decision, reason, idempotency_key, decided_by
            FROM commercial_billing_candidate_review_decisions
            """
        )
        assert dict(persisted) == {
            "billing_run_id": run_id,
            "candidate_key": candidate["candidateKey"],
            "source_fingerprint": candidate["sourceFingerprint"],
            "revision": 1,
            "decision": "included",
            "reason": "Reviewed and ready for explicit approval.",
            "idempotency_key": canonical_idempotency_key,
            "decided_by": "Juan Canfield",
        }
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM commercial_billing_candidate_review_decisions"
            )
            == 1
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("migration_fails", "ledger_mode"),
    (
        (True, "missing"),
        (True, "unqueryable"),
        (False, "missing"),
    ),
)
async def test_full_atlas_migration_check_blocks_enabled_receivables_without_recovery(
    migration_fails: bool,
    ledger_mode: str,
):
    from atlas_brain import main

    events: list[str] = []
    queries: list[tuple[str, tuple[object, ...]]] = []

    class _Pool:
        async def fetchval(self, query: str, *args: object) -> bool:
            events.append("ledger")
            queries.append((query, args))
            if ledger_mode == "unqueryable":
                raise RuntimeError("ledger unavailable")
            return False

    async def fail_migrations(pool: object) -> None:
        assert isinstance(pool, _Pool)
        events.append("migrate")
        if migration_fails:
            raise RuntimeError("migration failed")

    async def close_database() -> None:
        events.append("close")

    with pytest.raises(
        main.CommercialBillingReviewRecoveryUnavailableError,
        match="recovery migration must complete",
    ) as exc_info:
        await main._run_database_migration_check(
            _Pool(),
            receivables_api_enabled=True,
            run_migrations_fn=fail_migrations,
            close_database_fn=close_database,
        )

    if migration_fails:
        assert isinstance(exc_info.value.__cause__, RuntimeError)
    else:
        assert exc_info.value.__cause__ is None
    assert events == ["migrate", "ledger", "close"]
    assert queries == [
        (
            "SELECT EXISTS (SELECT 1 FROM schema_migrations WHERE name = $1)",
            (main._COMMERCIAL_BILLING_REVIEW_RECOVERY_MIGRATION,),
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "migration_fails",
        "receivables_api_enabled",
        "recovery_recorded",
        "dedup_ready",
        "expected_events",
        "expected_ledger_migrations",
    ),
    (
        # receivables_api_enabled=True requires review-recovery migration 382
        # plus recurring-writer dedup readiness, not a base receivables
        # migration-385 ledger check.
        (True, True, True, True, ["migrate", "ledger", "recurring-ready"], "recovery"),
        (False, True, True, True, ["migrate", "ledger", "recurring-ready"], "recovery"),
        (True, False, False, False, ["migrate"], "none"),
        (False, False, False, False, ["migrate"], "none"),
    ),
)
async def test_full_atlas_migration_check_allows_recovered_or_disabled_receivables(
    migration_fails: bool,
    receivables_api_enabled: bool,
    recovery_recorded: bool,
    dedup_ready: bool,
    expected_events: list[str],
    expected_ledger_migrations: str,
):
    from atlas_brain import main

    events: list[str] = []
    ledger_queries: list[str] = []

    class _Pool:
        async def fetchval(self, query: str, *args: object) -> bool:
            assert query == "SELECT EXISTS (SELECT 1 FROM schema_migrations WHERE name = $1)"
            assert args == (main._COMMERCIAL_BILLING_REVIEW_RECOVERY_MIGRATION,)
            ledger_queries.append(args[0])
            events.append("ledger")
            return recovery_recorded

    async def fail_migrations(pool: object) -> None:
        assert isinstance(pool, _Pool)
        events.append("migrate")
        if migration_fails:
            raise RuntimeError("migration failed")

    async def close_database() -> None:
        events.append("close")

    async def recurring_ready(pool: object) -> bool:
        assert isinstance(pool, _Pool)
        events.append("recurring-ready")
        return dedup_ready

    await main._run_database_migration_check(
        _Pool(),
        receivables_api_enabled=receivables_api_enabled,
        run_migrations_fn=fail_migrations,
        close_database_fn=close_database,
        recurring_dedup_ready_fn=recurring_ready,
    )

    assert events == expected_events
    if expected_ledger_migrations == "recovery":
        assert ledger_queries == [
            main._COMMERCIAL_BILLING_REVIEW_RECOVERY_MIGRATION,
        ]
    else:
        assert ledger_queries == []


@pytest.mark.asyncio
async def test_full_atlas_migration_check_blocks_enabled_auto_invoice_without_dedup_readiness():
    """The legacy monthly auto-invoice task reaches invoices.billing_period
    exactly like the receivables-mounted approval writer does, but through a
    completely independent flag (auto_invoice_enabled) that the review-
    recovery fence never checked. Positive-fence proof: this path is now
    fenced too, not only receivables_api_enabled."""
    from atlas_brain import main

    events: list[str] = []
    queries: list[tuple[str, tuple[object, ...]]] = []

    class _Pool:
        pass

    async def run_migrations(pool: object) -> None:
        assert isinstance(pool, _Pool)
        events.append("migrate")

    async def close_database() -> None:
        events.append("close")

    async def recurring_ready(pool: object) -> bool:
        assert isinstance(pool, _Pool)
        events.append("recurring-ready")
        return False

    with pytest.raises(
        main.RecurringInvoiceDedupMigrationUnavailableError,
        match="Recurring-invoice billing_period dedup schema",
    ) as exc_info:
        await main._run_database_migration_check(
            _Pool(),
            receivables_api_enabled=False,
            auto_invoice_enabled=True,
            run_migrations_fn=run_migrations,
            close_database_fn=close_database,
            recurring_dedup_ready_fn=recurring_ready,
        )

    assert exc_info.value.__cause__ is None
    assert events == ["migrate", "recurring-ready", "close"]
    assert queries == []


@pytest.mark.asyncio
async def test_full_atlas_migration_check_blocks_when_dedup_readiness_query_errors():
    """A broken readiness query is not a soft warning. If a recurring writer is
    enabled, inability to verify migration 385 is the same startup safety class
    as an explicit false readiness result: close the pool and fail closed."""
    from atlas_brain import main

    events: list[str] = []

    class _Pool:
        pass

    async def run_migrations(pool: object) -> None:
        assert isinstance(pool, _Pool)
        events.append("migrate")

    async def close_database() -> None:
        events.append("close")

    async def recurring_ready(pool: object) -> bool:
        assert isinstance(pool, _Pool)
        events.append("recurring-ready")
        raise RuntimeError("permission denied")

    with pytest.raises(
        main.RecurringInvoiceDedupMigrationUnavailableError,
        match="Recurring-invoice billing_period dedup schema",
    ) as exc_info:
        await main._run_database_migration_check(
            _Pool(),
            receivables_api_enabled=False,
            auto_invoice_enabled=True,
            run_migrations_fn=run_migrations,
            close_database_fn=close_database,
            recurring_dedup_ready_fn=recurring_ready,
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "permission denied"
    assert events == ["migrate", "recurring-ready", "close"]


@pytest.mark.asyncio
async def test_full_atlas_migration_check_allows_auto_invoice_when_dedup_schema_ready():
    """Negative control for the positive-fence test above: the same
    auto_invoice_enabled-only shape does NOT false-positive-block a healthy
    deploy once the recurring dedup schema is ready."""
    from atlas_brain import main

    events: list[str] = []

    class _Pool:
        pass

    async def run_migrations(pool: object) -> None:
        assert isinstance(pool, _Pool)
        events.append("migrate")

    async def close_database() -> None:
        events.append("close")

    async def recurring_ready(pool: object) -> bool:
        assert isinstance(pool, _Pool)
        events.append("recurring-ready")
        return True

    await main._run_database_migration_check(
        _Pool(),
        receivables_api_enabled=False,
        auto_invoice_enabled=True,
        run_migrations_fn=run_migrations,
        close_database_fn=close_database,
        recurring_dedup_ready_fn=recurring_ready,
    )

    assert events == ["migrate", "recurring-ready"]


@pytest.mark.asyncio
async def test_full_atlas_lifespan_uses_enabled_receivables_recovery_fence(monkeypatch):
    from atlas_brain import main
    from atlas_brain.eom_api.auth import generate_receivables_service_token

    events: list[str] = []

    class _Pool:
        is_initialized = True

        async def acquire(self) -> _Pool:
            events.append("acquire")
            return self

        async def release(self, connection: object) -> None:
            assert connection is self
            events.append("release")

        async def fetchval(self, query: str, *args: object) -> bool:
            if query == "SELECT pg_try_advisory_lock($1)":
                events.append("lock")
                return True
            assert query == "SELECT EXISTS (SELECT 1 FROM schema_migrations WHERE name = $1)"
            assert args == (main._COMMERCIAL_BILLING_REVIEW_RECOVERY_MIGRATION,)
            events.append("ledger")
            return False

        async def execute(self, query: str, *args: object) -> str:
            if "CREATE TABLE IF NOT EXISTS schema_migrations" in query:
                events.append("ensure")
                return "CREATE TABLE"
            if "ADD COLUMN IF NOT EXISTS content_sha256" in query:
                events.append("ensure-content-identity")
                return "ALTER TABLE"
            assert query == "SELECT pg_advisory_unlock($1)"
            assert args
            events.append("unlock")
            return "SELECT 1"

        async def fetch(self, query: str) -> list[object]:
            assert query == "SELECT name FROM schema_migrations"
            events.append("migrate")
            raise RuntimeError("migration failed")

    pool = _Pool()
    generated = generate_receivables_service_token()

    async def init_database() -> None:
        events.append("init")

    async def close_database() -> None:
        events.append("close")

    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(
            invoicing=SimpleNamespace(
                enabled=True,
                receivables_api_enabled=True,
                auto_invoice_enabled=False,
                receivables_service_token="",
                receivables_service_token_sha256=generated.sha256,
            )
        ),
    )
    monkeypatch.setattr(main, "db_settings", SimpleNamespace(enabled=True))
    monkeypatch.setattr(main, "_enforce_paid_funnel_alert_channel", lambda _settings: None)
    monkeypatch.setattr(main, "init_database", init_database)
    monkeypatch.setattr(main, "get_db_pool", lambda: pool)
    monkeypatch.setattr(main, "close_database", close_database)

    with pytest.raises(
        main.CommercialBillingReviewRecoveryUnavailableError,
        match="recovery migration must complete",
    ):
        async with main.lifespan(FastAPI()):
            raise AssertionError("unsafe receivables startup must not serve")

    assert events == [
        "init",
        "acquire",
        "lock",
        "ensure",
        "ensure-content-identity",
        "migrate",
        "unlock",
        "release",
        "ledger",
        "close",
    ]


@pytest.mark.asyncio
async def test_full_atlas_lifespan_fences_auto_invoice_only_deployment_without_dedup_readiness(
    monkeypatch,
):
    """The legacy-cron-only deployment shape (auto_invoice_enabled=True,
    receivables_api_enabled=False -- no reason to ever run the receivables
    API just to auto-invoice) is fenced end-to-end through the real
    main.lifespan(...), not only at the _run_database_migration_check unit
    level. Mirrors test_full_atlas_lifespan_uses_enabled_receivables_recovery_fence."""
    from atlas_brain import main
    from atlas_brain.storage.repositories import invoice as invoice_mod

    events: list[str] = []

    class _Pool:
        is_initialized = True

        async def acquire(self) -> _Pool:
            events.append("acquire")
            return self

        async def release(self, connection: object) -> None:
            assert connection is self
            events.append("release")

        async def fetchval(self, query: str, *args: object) -> bool:
            if query == "SELECT pg_try_advisory_lock($1)":
                events.append("lock")
                return True
            raise AssertionError(f"Unexpected fetchval query: {query}")

        async def execute(self, query: str, *args: object) -> str:
            if "CREATE TABLE IF NOT EXISTS schema_migrations" in query:
                events.append("ensure")
                return "CREATE TABLE"
            if "ADD COLUMN IF NOT EXISTS content_sha256" in query:
                events.append("ensure-content-identity")
                return "ALTER TABLE"
            assert query == "SELECT pg_advisory_unlock($1)"
            assert args
            events.append("unlock")
            return "SELECT 1"

        async def fetch(self, query: str) -> list[object]:
            assert query == "SELECT name FROM schema_migrations"
            events.append("migrate")
            raise RuntimeError("migration failed")

    pool = _Pool()

    async def init_database() -> None:
        events.append("init")

    async def close_database() -> None:
        events.append("close")

    async def recurring_ready(candidate) -> bool:
        assert candidate is pool
        events.append("recurring-ready")
        return False

    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(
            invoicing=SimpleNamespace(
                enabled=True,
                receivables_api_enabled=False,
                auto_invoice_enabled=True,
                receivables_service_token="",
                receivables_service_token_sha256="",
            )
        ),
    )
    monkeypatch.setattr(main, "db_settings", SimpleNamespace(enabled=True))
    monkeypatch.setattr(main, "_enforce_paid_funnel_alert_channel", lambda _settings: None)
    monkeypatch.setattr(main, "init_database", init_database)
    monkeypatch.setattr(main, "get_db_pool", lambda: pool)
    monkeypatch.setattr(main, "close_database", close_database)
    monkeypatch.setattr(invoice_mod, "recurring_invoice_dedup_schema_ready", recurring_ready)

    with pytest.raises(
        main.RecurringInvoiceDedupMigrationUnavailableError,
        match="Recurring-invoice billing_period dedup schema",
    ):
        async with main.lifespan(FastAPI()):
            raise AssertionError("unsafe auto-invoice startup must not serve")

    assert events == [
        "init",
        "acquire",
        "lock",
        "ensure",
        "ensure-content-identity",
        "migrate",
        "unlock",
        "release",
        "recurring-ready",
        "close",
    ]


@pytest.mark.asyncio
async def test_full_atlas_lifespan_scopes_auto_invoice_fence_to_master_invoicing_gate(
    monkeypatch,
):
    """Finding #7 regression: a stale auto_invoice_enabled=True must not
    fence startup once the master invoicing.enabled gate is off. The legacy
    monthly task checks settings.invoicing.enabled first and returns
    "Invoicing disabled" before ever reaching auto_invoice_enabled or
    billing_period (monthly_invoice_generation.py), so requiring migration
    385 in that shape is a false-positive block on a healthy,
    invoicing-disabled deployment. Negative control:
    test_full_atlas_lifespan_fences_auto_invoice_only_deployment_without_dedup_readiness
    (invoicing.enabled=True) proves the fence still fires when the master
    gate is actually on."""
    from atlas_brain import main

    captured: dict[str, object] = {}

    async def fake_run_database_migration_check(pool, **kwargs) -> None:
        captured.update(kwargs)
        raise main.RecurringInvoiceDedupMigrationUnavailableError("stop-after-capture")

    class _Pool:
        is_initialized = True

    async def init_database() -> None:
        pass

    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(
            invoicing=SimpleNamespace(
                enabled=False,
                receivables_api_enabled=False,
                auto_invoice_enabled=True,
                receivables_service_token="",
                receivables_service_token_sha256="",
            )
        ),
    )
    monkeypatch.setattr(main, "db_settings", SimpleNamespace(enabled=True))
    monkeypatch.setattr(main, "_enforce_paid_funnel_alert_channel", lambda _settings: None)
    monkeypatch.setattr(main, "init_database", init_database)
    monkeypatch.setattr(main, "get_db_pool", lambda: _Pool())
    monkeypatch.setattr(
        main, "_run_database_migration_check", fake_run_database_migration_check
    )

    with pytest.raises(
        main.RecurringInvoiceDedupMigrationUnavailableError, match="stop-after-capture"
    ):
        async with main.lifespan(FastAPI()):
            raise AssertionError("should not reach yield")

    assert captured == {
        "receivables_api_enabled": False,
        "auto_invoice_enabled": False,
    }


def test_billing_run_migration_is_additive_and_preserves_draft_evidence():
    migration = (
        Path(__file__).parents[1]
        / "atlas_brain/storage/migrations/370_commercial_billing_runs.sql"
    ).read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS commercial_billing_runs" in migration
    assert "CREATE TABLE IF NOT EXISTS commercial_billing_run_candidates" in migration
    assert "CHECK (state = 'draft')" in migration
    assert "UNIQUE (source, idempotency_key)" in migration
    assert "UNIQUE (billing_run_id, candidate_key)" in migration
    assert "ON DELETE RESTRICT" in migration
    assert "DROP TABLE" not in "\n".join(
        line for line in migration.splitlines() if not line.lstrip().startswith("--")
    )


def test_review_decision_migration_is_append_only_and_snapshot_restrictive():
    migration = (
        Path(__file__).parents[1]
        / "atlas_brain/storage/migrations/380_commercial_billing_candidate_review_decisions.sql"
    ).read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS commercial_billing_candidate_review_decisions" in migration
    assert "CHECK (decision IN ('included', 'excluded'))" in migration
    assert "UNIQUE (source, idempotency_key)" in migration
    assert "UNIQUE (candidate_key, source_fingerprint, revision)" in migration
    assert "commercial_billing_candidate_review_decisions_snapshot_fkey" in migration
    assert "ON DELETE RESTRICT" in migration
    assert "prevent_commercial_billing_review_decision_mutation" in migration
    assert "BEFORE UPDATE OR DELETE ON commercial_billing_candidate_review_decisions" in migration
    assert "BEFORE TRUNCATE ON commercial_billing_candidate_review_decisions" in migration
    assert "prevent_commercial_billing_invoice_for_excluded_candidate" in migration
    assert "BEFORE INSERT ON invoices" in migration
    assert "jsonb_typeof(NEW.metadata -> 'candidateKey')" in migration
    assert "jsonb_typeof(NEW.metadata -> 'sourceFingerprint')" in migration
    assert "FROM commercial_billing_run_candidates" in migration
    assert "idx_commercial_billing_run_candidates_identity" in migration
    assert "ON commercial_billing_run_candidates (candidate_key, source_fingerprint)" in migration
    executable = "\n".join(
        line for line in migration.splitlines() if not line.lstrip().startswith("--")
    )
    assert "DROP TABLE" not in executable
    assert "INSERT INTO invoices" not in executable
    assert "UPDATE invoices" not in executable
    assert "gmail" not in executable.lower()
    assert "email" not in executable.lower()


def test_review_decision_recovery_migration_is_atomic_and_data_preserving():
    migration = (
        Path(__file__).parents[1]
        / "atlas_brain/storage/migrations/"
        "381_commercial_billing_candidate_review_decisions_recovery.sql"
    ).read_text(encoding="utf-8")

    assert migration.startswith("-- atlas: atomic-bookkeeping")
    assert "UNIQUE (candidate_key, source_fingerprint, revision)" in migration
    assert "duplicate global revision identities" in migration
    assert "idx_commercial_billing_run_candidates_identity" in migration
    assert "prevent_commercial_billing_review_decision_mutation" in migration
    assert "prevent_commercial_billing_invoice_for_excluded_candidate" in migration
    executable = "\n".join(
        line for line in migration.splitlines() if not line.lstrip().startswith("--")
    )
    assert "DROP TABLE" not in executable
    assert "INSERT INTO commercial_billing_candidate_review_decisions" not in executable
    assert "INSERT INTO invoices" not in executable
    assert "DELETE FROM commercial_billing_candidate_review_decisions" not in executable


def test_historical_379_run_fence_recovery_is_atomic_and_data_preserving():
    migration = (
        Path(__file__).parents[1]
        / "atlas_brain/storage/migrations/"
        "391_eom_commercial_billing_run_fence_recovery.sql"
    ).read_text(encoding="utf-8")

    assert migration.startswith("-- atlas: atomic-bookkeeping")
    assert "b71db37ee1906ca26788be21deb716092052fc3197d4b72762d57892fbc77851" in migration
    assert "commercialBillingRunId" in migration
    assert "WHERE billing_run_id = candidate_identity_billing_run_id" in migration
    assert "immutable review history guards" in migration
    executable = "\n".join(
        line for line in migration.splitlines() if not line.lstrip().startswith("--")
    )
    assert "DROP TABLE" not in executable
    assert "INSERT INTO commercial_billing_candidate_review_decisions" not in executable
    assert "UPDATE commercial_billing_candidate_review_decisions" not in executable
    assert "INSERT INTO commercial_billing_candidate_overrides" not in executable
    assert "UPDATE commercial_billing_candidate_overrides" not in executable
    assert "INSERT INTO invoices" not in executable
    assert "UPDATE invoices" not in executable


@pytest.mark.asyncio
async def test_real_postgres_historical_379_commercial_billing_run_fence_recovery():
    """391 preserves history and stops a run-A override from blocking run B."""
    asyncpg = pytest.importorskip("asyncpg")
    from atlas_brain.storage.migrations import run_migrations
    from atlas_brain.storage.migrations.reconciliation import (
        MIGRATION_379_COMMERCIAL_BILLING_RUN_FENCE_FORWARD_RECOVERY,
    )

    record = MIGRATION_379_COMMERCIAL_BILLING_RUN_FENCE_FORWARD_RECOVERY
    async with _billing_run_database() as (conn, schema, _database_url):
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
        await conn.execute(
            "ALTER TABLE invoices ADD COLUMN metadata JSONB NOT NULL "
            "DEFAULT '{}'::jsonb"
        )
        legacy_source = (
            Path(__file__).parents[1]
            / "atlas_brain/storage/migrations/"
            "382_commercial_billing_candidate_overrides.sql"
        ).read_text(encoding="utf-8")
        legacy_section = legacy_source.split(
            "CREATE OR REPLACE FUNCTION "
            "prevent_commercial_billing_invoice_for_excluded_candidate()",
            1,
        )[1]
        legacy_body = (
            legacy_section.split("AS $$", 1)[1]
            .split("$$;", 1)[0]
            .replace("    candidate_identity_billing_run_id UUID;\n", "")
            .replace(
                "       OR jsonb_typeof(NEW.metadata -> 'commercialBillingRunId') "
                "IS DISTINCT FROM 'string'\n",
                "",
            )
            .replace(
                "    BEGIN\n"
                "        candidate_identity_billing_run_id :=\n"
                "            (NEW.metadata ->> 'commercialBillingRunId')::UUID;\n"
                "    EXCEPTION WHEN invalid_text_representation THEN\n"
                "        RAISE EXCEPTION "
                "'Commercial billing invoice review identity is invalid';\n"
                "    END;\n",
                "",
            )
            .replace(
                "    WHERE billing_run_id = candidate_identity_billing_run_id\n"
                "      AND candidate_key = candidate_identity_key\n",
                "    WHERE candidate_key = candidate_identity_key\n",
            )
        )
        assert hashlib.sha256(legacy_body.encode()).hexdigest() == (
            record.legacy_function_body_sha256
        )
        await conn.execute(
            "CREATE OR REPLACE FUNCTION "
            "prevent_commercial_billing_invoice_for_excluded_candidate() "
            "RETURNS TRIGGER LANGUAGE plpgsql AS $legacy$"
            f"{legacy_body}"
            "$legacy$;"
        )
        await conn.execute(
            "CREATE TABLE schema_migrations ("
            "version INTEGER PRIMARY KEY, name VARCHAR(255) NOT NULL, "
            "content_sha256 VARCHAR(64), "
            "applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        await conn.executemany(
            "INSERT INTO schema_migrations (version, name, content_sha256, applied_at) "
            "VALUES ($1, $2, $3, $4)",
            [
                (
                    record.historical_migration_version,
                    record.migration_name,
                    record.historical_ledger_sha256,
                    record.observed_applied_at,
                ),
                *[
                    (
                        receipt.migration_version,
                        receipt.migration_name,
                        None,
                        receipt.observed_applied_at,
                    )
                    for receipt in record.successor_receipts
                ],
            ],
        )
        candidate = _candidate(
            "commercial-billing:historical-379-run-fence", _fingerprint("a")
        )
        first_run_id, second_run_id = uuid4(), uuid4()
        await _insert_legacy_run_candidate(
            conn,
            run_id=first_run_id,
            candidate=candidate,
            idempotency_key=f"migration-379-first-{first_run_id}",
        )
        await _insert_legacy_run_candidate(
            conn,
            run_id=second_run_id,
            candidate=candidate,
            idempotency_key=f"migration-379-second-{second_run_id}",
        )
        override_fingerprint = _fingerprint("b")
        await conn.execute(
            """
            INSERT INTO commercial_billing_candidate_overrides (
                id, billing_run_id, candidate_key, source_fingerprint, revision,
                review_fingerprint, effective_snapshot, reason_code, reason, source,
                idempotency_key, request_fingerprint, overridden_by
            ) VALUES (
                $1, $2, $3, $4, 1, $5, $6::jsonb,
                'source_correction_pending', 'Run-A-only recovery proof.',
                'eom_admin', $7, $8, 'Migration recovery test'
            )
            """,
            uuid4(),
            first_run_id,
            candidate["candidateKey"],
            candidate["sourceFingerprint"],
            override_fingerprint,
            json.dumps(candidate),
            "historical-379-run-a-override",
            _fingerprint("c"),
        )
        review_history_before = [
            dict(row)
            for row in await conn.fetch(
                "SELECT * FROM commercial_billing_candidate_review_decisions ORDER BY id"
            )
        ]
        override_history_before = [
            dict(row)
            for row in await conn.fetch(
                "SELECT * FROM commercial_billing_candidate_overrides ORDER BY id"
            )
        ]
        second_run_metadata = json.dumps({
            "candidateKey": candidate["candidateKey"],
            "commercialBillingRunId": str(second_run_id),
            "sourceFingerprint": candidate["sourceFingerprint"],
        })
        with pytest.raises(asyncpg.PostgresError, match="stale"):
            await conn.execute(
                "INSERT INTO invoices (id, source, source_ref, metadata) "
                "VALUES ($1, 'eom_commercial_billing', 'historical-379-before', $2::jsonb)",
                uuid4(),
                second_run_metadata,
            )
        assert await conn.fetchval("SELECT COUNT(*) FROM invoices") == 0

        pool = _SchemaPool(conn, schema)
        migrations_dir = Path(__file__).parents[1] / "atlas_brain/storage/migrations"
        await run_migrations(
            pool,
            migrations_dir=migrations_dir,
            only={record.recovery_migration_name},
        )

        assert await conn.fetchval(
            "SELECT COUNT(*) FROM schema_migrations WHERE name = $1",
            record.recovery_migration_name,
        ) == 1
        assert await conn.fetchval(
            "SELECT content_sha256 FROM schema_migrations WHERE name = $1",
            record.recovery_migration_name,
        ) == record.recovery_packaged_sha256
        await conn.execute(
            "INSERT INTO invoices (id, source, source_ref, metadata) "
            "VALUES ($1, 'eom_commercial_billing', 'historical-379-after', $2::jsonb)",
            uuid4(),
            second_run_metadata,
        )
        assert await conn.fetchval("SELECT COUNT(*) FROM invoices") == 1
        assert [
            dict(row)
            for row in await conn.fetch(
                "SELECT * FROM commercial_billing_candidate_review_decisions ORDER BY id"
            )
        ] == review_history_before
        assert [
            dict(row)
            for row in await conn.fetch(
                "SELECT * FROM commercial_billing_candidate_overrides ORDER BY id"
            )
        ] == override_history_before

        await run_migrations(
            pool,
            migrations_dir=migrations_dir,
            only={record.recovery_migration_name},
        )
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM schema_migrations WHERE name = $1",
            record.recovery_migration_name,
        ) == 1
        assert await conn.fetchval("SELECT COUNT(*) FROM invoices") == 1


@pytest.mark.asyncio
async def test_current_380_schema_runs_pending_381_once_without_rewriting_catalog_or_history():
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    from atlas_brain.storage.migrations import run_migrations

    recovery_migration = "381_commercial_billing_candidate_review_decisions_recovery"
    expected_ledger = [
        "370_commercial_billing_runs",
        "372_commercial_billing_candidate_approvals",
        "380_commercial_billing_candidate_review_decisions",
        recovery_migration,
    ]
    schema = f"billing_380_current_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        await conn.execute(
            "CREATE TABLE invoices ("
            "id UUID PRIMARY KEY, source TEXT, source_ref TEXT, "
            "metadata JSONB NOT NULL DEFAULT '{}'::jsonb"
            ")"
        )
        pool, migrations_dir = await _current_380_schema(conn, schema)
        assert [
            row["name"]
            for row in await conn.fetch("SELECT name FROM schema_migrations ORDER BY name")
        ] == expected_ledger[:-1]

        candidate = _candidate("commercial-billing:current-380:2026-03", _fingerprint("a"))
        run_id = uuid4()
        await _insert_legacy_run_candidate(
            conn,
            run_id=run_id,
            candidate=candidate,
            idempotency_key="billing-run-current-380-pending-381",
        )
        decision_id = uuid4()
        await conn.execute(
            """
            INSERT INTO commercial_billing_candidate_review_decisions (
                id, billing_run_id, candidate_key, source_fingerprint,
                revision, decision, reason, source, idempotency_key,
                request_fingerprint, decided_by
            ) VALUES ($1, $2, $3, $4, 1, 'included', $5, 'eom_admin', $6, $7,
                      'Migration recovery test')
            """,
            decision_id,
            run_id,
            candidate["candidateKey"],
            candidate["sourceFingerprint"],
            "Preserve the current-schema review decision during recovery.",
            "current-380-pending-381-decision",
            _fingerprint("c"),
        )
        history_before = dict(
            await conn.fetchrow(
                """
                SELECT id, billing_run_id, candidate_key, source_fingerprint, revision,
                       decision, reason, source, idempotency_key, request_fingerprint,
                       decided_by, decided_at, created_at
                FROM commercial_billing_candidate_review_decisions
                WHERE id = $1
                """,
                decision_id,
            )
        )
        catalog_before = await _review_decision_safety_catalog(conn, schema)
        assert catalog_before["revisionKeyColumns"] == [
            "candidate_key",
            "source_fingerprint",
            "revision",
        ]
        assert catalog_before["identityIndex"] is not None
        assert len(catalog_before["triggers"]) == 3
        assert len(catalog_before["functions"]) == 2

        await run_migrations(
            pool,
            migrations_dir=migrations_dir,
            only={recovery_migration},
        )

        assert [
            row["name"]
            for row in await conn.fetch("SELECT name FROM schema_migrations ORDER BY name")
        ] == expected_ledger
        assert dict(
            await conn.fetchrow(
                """
                SELECT id, billing_run_id, candidate_key, source_fingerprint, revision,
                       decision, reason, source, idempotency_key, request_fingerprint,
                       decided_by, decided_at, created_at
                FROM commercial_billing_candidate_review_decisions
                WHERE id = $1
                """,
                decision_id,
            )
        ) == history_before
        assert await _review_decision_safety_catalog(conn, schema) == catalog_before
        assert await conn.fetchval("SELECT COUNT(*) FROM invoices") == 0

        await run_migrations(
            pool,
            migrations_dir=migrations_dir,
            only={recovery_migration},
        )

        assert [
            row["name"]
            for row in await conn.fetch("SELECT name FROM schema_migrations ORDER BY name")
        ] == expected_ledger
        assert dict(
            await conn.fetchrow(
                """
                SELECT id, billing_run_id, candidate_key, source_fingerprint, revision,
                       decision, reason, source, idempotency_key, request_fingerprint,
                       decided_by, decided_at, created_at
                FROM commercial_billing_candidate_review_decisions
                WHERE id = $1
                """,
                decision_id,
            )
        ) == history_before
        assert await _review_decision_safety_catalog(conn, schema) == catalog_before
        assert await conn.fetchval("SELECT COUNT(*) FROM invoices") == 0
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_recorded_380_recovery_restores_review_decision_enforcement():
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    from atlas_brain.storage.migrations import run_migrations

    recovery_migration = "381_commercial_billing_candidate_review_decisions_recovery"
    schema = f"billing_380_recovery_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        await conn.execute(
            "CREATE TABLE invoices ("
            "id UUID PRIMARY KEY, source TEXT, source_ref TEXT, "
            "metadata JSONB NOT NULL DEFAULT '{}'::jsonb"
            ")"
        )
        pool, migrations_dir = await _recorded_380_legacy_schema(conn, schema)

        assert await _revision_key_columns(conn) == [
            "billing_run_id",
            "candidate_key",
            "source_fingerprint",
            "revision",
        ]
        assert await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_indexes
                WHERE schemaname = $1
                  AND tablename = 'commercial_billing_run_candidates'
                  AND indexname = 'idx_commercial_billing_run_candidates_identity'
            )
            """,
            schema,
        ) is False

        candidate = _candidate("commercial-billing:recovery:2026-03", _fingerprint("a"))
        run_id = uuid4()
        await _insert_legacy_run_candidate(
            conn,
            run_id=run_id,
            candidate=candidate,
            idempotency_key="billing-run-recorded-380-recovery",
        )
        legacy_decision_id = uuid4()
        await conn.execute(
            """
            INSERT INTO commercial_billing_candidate_review_decisions (
                id, billing_run_id, candidate_key, source_fingerprint,
                revision, decision, reason, source, idempotency_key,
                request_fingerprint, decided_by
            ) VALUES ($1, $2, $3, $4, 1, 'included', $5, 'eom_admin', $6, $7,
                      'Migration recovery test')
            """,
            legacy_decision_id,
            run_id,
            candidate["candidateKey"],
            candidate["sourceFingerprint"],
            "Preserve the legacy review decision during recovery.",
            "legacy-review-1",
            _fingerprint("c"),
        )
        legacy_history_before = await conn.fetchrow(
            """
            SELECT id, billing_run_id, candidate_key, source_fingerprint, revision,
                   decision, reason, source, idempotency_key, request_fingerprint,
                   decided_by, decided_at, created_at
            FROM commercial_billing_candidate_review_decisions
            WHERE id = $1
            """,
            legacy_decision_id,
        )

        await run_migrations(
            pool,
            migrations_dir=migrations_dir,
            only={recovery_migration},
        )

        assert await conn.fetchval(
            "SELECT COUNT(*) FROM schema_migrations WHERE name = $1",
            "380_commercial_billing_candidate_review_decisions",
        ) == 1
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM schema_migrations WHERE name = $1",
            recovery_migration,
        ) == 1
        assert await _revision_key_columns(conn) == [
            "candidate_key",
            "source_fingerprint",
            "revision",
        ]
        assert dict(
            await conn.fetchrow(
                """
                SELECT id, billing_run_id, candidate_key, source_fingerprint, revision,
                       decision, reason, source, idempotency_key, request_fingerprint,
                       decided_by, decided_at, created_at
                FROM commercial_billing_candidate_review_decisions
                WHERE id = $1
                """,
                legacy_decision_id,
            )
        ) == dict(legacy_history_before)
        assert await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_indexes
                WHERE schemaname = $1
                  AND tablename = 'commercial_billing_run_candidates'
                  AND indexname = 'idx_commercial_billing_run_candidates_identity'
            )
            """,
            schema,
        ) is True
        assert {
            (row["tgname"], row["relname"])
            for row in await conn.fetch(
                """
                SELECT trigger_state.tgname, relation.relname
                FROM pg_trigger AS trigger_state
                JOIN pg_class AS relation
                  ON relation.oid = trigger_state.tgrelid
                JOIN pg_namespace AS namespace_state
                  ON namespace_state.oid = relation.relnamespace
                WHERE NOT trigger_state.tgisinternal
                  AND namespace_state.nspname = $1
                  AND trigger_state.tgname IN (
                      'trg_prevent_commercial_billing_review_decision_mutation',
                      'trg_prevent_commercial_billing_review_decision_truncate',
                      'trg_prevent_commercial_billing_invoice_for_excluded_candidate'
                  )
                """,
                schema,
            )
        } == {
            (
                "trg_prevent_commercial_billing_review_decision_mutation",
                "commercial_billing_candidate_review_decisions",
            ),
            (
                "trg_prevent_commercial_billing_review_decision_truncate",
                "commercial_billing_candidate_review_decisions",
            ),
            (
                "trg_prevent_commercial_billing_invoice_for_excluded_candidate",
                "invoices",
            ),
        }

        await run_migrations(
            pool,
            migrations_dir=migrations_dir,
            only={"382_commercial_billing_candidate_overrides"},
        )
        service = _service(_SchemaPool(conn, schema), _CandidateService(_preview(candidate)))
        recorded = await service.set_candidate_review_decision(
            billing_run_id=run_id,
            candidate_key=candidate["candidateKey"],
            expected_source_fingerprint=candidate["sourceFingerprint"],
            decision="excluded",
            reason="Restore the durable review decision fence.",
            idempotency_key="recorded-380-recovery-decision",
            actor="Migration recovery test",
        )
        decision_id = UUID(recorded["reviewDecision"]["id"])
        for statement, arguments in (
            (
                "UPDATE commercial_billing_candidate_review_decisions "
                "SET reason = 'rewritten' WHERE id = $1",
                (decision_id,),
            ),
            (
                "DELETE FROM commercial_billing_candidate_review_decisions WHERE id = $1",
                (decision_id,),
            ),
            ("TRUNCATE commercial_billing_candidate_review_decisions", ()),
        ):
            with pytest.raises(asyncpg.PostgresError, match="append-only"):
                await conn.execute(statement, *arguments)
        with pytest.raises(asyncpg.PostgresError, match="candidate is excluded"):
            await conn.execute(
                """
                INSERT INTO invoices (id, source, source_ref, metadata)
                VALUES ($1, 'eom_commercial_billing', 'recovery-fence-test', $2::jsonb)
                """,
                uuid4(),
                json.dumps(
                    {
                        "candidateKey": candidate["candidateKey"],
                        "commercialBillingRunId": str(run_id),
                        "sourceFingerprint": candidate["sourceFingerprint"],
                    }
                ),
            )
        assert await conn.fetchval("SELECT COUNT(*) FROM invoices") == 0

        history_before_rerun = await conn.fetch(
            """
            SELECT id, billing_run_id, candidate_key, source_fingerprint, revision,
                   decision, reason, source, idempotency_key, request_fingerprint,
                   decided_by, decided_at, created_at
            FROM commercial_billing_candidate_review_decisions
            ORDER BY revision
            """
        )
        await run_migrations(
            pool,
            migrations_dir=migrations_dir,
            only={recovery_migration},
        )
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM schema_migrations WHERE name = $1",
            recovery_migration,
        ) == 1
        assert await conn.fetch(
            """
            SELECT id, billing_run_id, candidate_key, source_fingerprint, revision,
                   decision, reason, source, idempotency_key, request_fingerprint,
                   decided_by, decided_at, created_at
            FROM commercial_billing_candidate_review_decisions
            ORDER BY revision
            """
        ) == history_before_rerun
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_recorded_380_recovery_rejects_ambiguous_global_revision_history():
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_RECEIVABLES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_RECEIVABLES_TEST_DATABASE_URL not set")

    from atlas_brain.storage.migrations import run_migrations

    recovery_migration = "381_commercial_billing_candidate_review_decisions_recovery"
    schema = f"billing_380_conflict_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        await conn.execute(
            "CREATE TABLE invoices ("
            "id UUID PRIMARY KEY, source TEXT, source_ref TEXT, "
            "metadata JSONB NOT NULL DEFAULT '{}'::jsonb"
            ")"
        )
        pool, migrations_dir = await _recorded_380_legacy_schema(conn, schema)
        candidate = _candidate("commercial-billing:conflict:2026-03", _fingerprint("b"))
        first_run_id, second_run_id = uuid4(), uuid4()
        for run_id, key in (
            (first_run_id, "billing-run-recorded-380-conflict-1"),
            (second_run_id, "billing-run-recorded-380-conflict-2"),
        ):
            await _insert_legacy_run_candidate(
                conn,
                run_id=run_id,
                candidate=candidate,
                idempotency_key=key,
            )
        for run_id, key, fingerprint in (
            (first_run_id, "recorded-380-conflict-decision-1", _fingerprint("c")),
            (second_run_id, "recorded-380-conflict-decision-2", _fingerprint("d")),
        ):
            await conn.execute(
                """
                INSERT INTO commercial_billing_candidate_review_decisions (
                    id, billing_run_id, candidate_key, source_fingerprint,
                    revision, decision, reason, source, idempotency_key,
                    request_fingerprint, decided_by
                ) VALUES ($1, $2, $3, $4, 1, 'excluded',
                          'Legacy per-run revision evidence.', 'eom_admin', $5,
                          $6, 'Migration recovery test')
                """,
                uuid4(),
                run_id,
                candidate["candidateKey"],
                candidate["sourceFingerprint"],
                key,
                fingerprint,
            )

        history_before = await conn.fetch(
            """
            SELECT billing_run_id, candidate_key, source_fingerprint, revision,
                   decision, reason, idempotency_key, request_fingerprint
            FROM commercial_billing_candidate_review_decisions
            ORDER BY idempotency_key
            """
        )
        with pytest.raises(
            asyncpg.PostgresError,
            match="duplicate global revision identities",
        ):
            await run_migrations(
                pool,
                migrations_dir=migrations_dir,
                only={recovery_migration},
            )

        assert await conn.fetchval(
            "SELECT COUNT(*) FROM schema_migrations WHERE name = $1",
            recovery_migration,
        ) == 0
        assert await _revision_key_columns(conn) == [
            "billing_run_id",
            "candidate_key",
            "source_fingerprint",
            "revision",
        ]
        assert await conn.fetch(
            """
            SELECT billing_run_id, candidate_key, source_fingerprint, revision,
                   decision, reason, idempotency_key, request_fingerprint
            FROM commercial_billing_candidate_review_decisions
            ORDER BY idempotency_key
            """
        ) == history_before
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_review_decision_migration_installs_trigger_identity_index():
    async with _billing_run_database() as (conn, schema, _database_url):
        indexdef = await conn.fetchval(
            """
            SELECT indexdef
            FROM pg_indexes
            WHERE schemaname = $1
              AND tablename = 'commercial_billing_run_candidates'
              AND indexname = 'idx_commercial_billing_run_candidates_identity'
            """,
            schema,
        )

    assert indexdef is not None
    assert "commercial_billing_run_candidates" in indexdef
    assert "(candidate_key, source_fingerprint)" in indexdef


def test_billing_run_service_does_not_import_financial_or_delivery_writers():
    import atlas_brain.services.commercial_billing_runs as billing_runs

    imports = {
        alias.name
        for node in ast.walk(ast.parse(inspect.getsource(billing_runs)))
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        {
            f"{node.module}.{alias.name}" if node.module else alias.name
            for node in ast.walk(ast.parse(inspect.getsource(billing_runs)))
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
    )
    forbidden_fragments = {
        "monthly_invoice_generation",
        "invoice_pdf",
        "email_provider",
        "gmail",
        "notification",
        "invoice",
    }
    assert not any(
        fragment in imported
        for fragment in forbidden_fragments
        for imported in imports
    )


def test_invoicing_workflow_enrolls_billing_run_contract_for_pr_and_main_push():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github/workflows/atlas_invoicing_checks.yml"
    ).read_text(encoding="utf-8")

    for path in (
        "atlas_brain/services/commercial_billing_runs.py",
        "atlas_brain/services/commercial_billing_candidate_overrides.py",
        "atlas_brain/storage/migrations/370_commercial_billing_runs.sql",
        "atlas_brain/storage/migrations/380_commercial_billing_candidate_review_decisions.sql",
        "atlas_brain/storage/migrations/381_commercial_billing_candidate_review_decisions_recovery.sql",
        "atlas_brain/storage/migrations/382_commercial_billing_candidate_overrides.sql",
        "tests/test_commercial_billing_runs.py",
        "tests/test_commercial_billing_candidate_overrides.py",
    ):
        assert workflow.count(f'      - "{path}"') == 2
    assert "tests/test_commercial_billing_runs.py \\" in workflow
    assert "tests/test_commercial_billing_candidate_overrides.py \\" in workflow
