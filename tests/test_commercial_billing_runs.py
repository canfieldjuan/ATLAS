"""Contract tests for durable, pre-approval EOM commercial billing runs."""

from __future__ import annotations

import asyncio
import ast
import copy
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
    CommercialBillingRunService,
    CommercialBillingRunUnavailableError,
    _normalize_preview,
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
        "contractVersion": 1,
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
    migration = (
        Path(__file__).parents[1]
        / "atlas_brain/storage/migrations/370_commercial_billing_runs.sql"
    ).read_text(encoding="utf-8")
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        await conn.execute(migration)
        yield conn, schema, database_url
    finally:
        await conn.execute("SET search_path TO public")
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


def _service(pool, candidate_service: _CandidateService) -> CommercialBillingRunService:
    return CommercialBillingRunService(
        pool=pool,
        candidate_service_loader=lambda: candidate_service,
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
        assert run["candidateContractVersion"] == 1
        assert run["summary"] == {"blockedCandidateCount": 1, "candidateCount": 1}
        assert run["candidates"][0]["lineItems"][0]["amountCents"] == 9650
        assert run["candidates"][0]["sourceEvents"][0]["location"] == "100 Main St"
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


class _NoStoredRunConnection:
    def __init__(self) -> None:
        self.transaction_count = 0
        self.write_attempts = 0

    async def fetchval(self, query, *_args):
        assert "pg_advisory_xact_lock" in query
        return None

    async def fetchrow(self, query, *_args):
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
            "contractVersion": 1,
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
class _RouteRunService:
    def __init__(self) -> None:
        self.create_calls: list[tuple[str, str, str]] = []
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
        "atlas_brain/storage/migrations/370_commercial_billing_runs.sql",
        "tests/test_commercial_billing_runs.py",
    ):
        assert workflow.count(f'      - "{path}"') == 2
    assert "tests/test_commercial_billing_runs.py \\" in workflow
