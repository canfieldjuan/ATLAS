"""Boundary proof for immutable post-clean service commitments."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest

from atlas_brain import main_eom
from atlas_brain.eom_api import funnel as funnel_mod
from atlas_brain.eom_api import funnel_auth as funnel_auth_mod
from atlas_brain.eom_api.config import EOMFunnelConfig
from atlas_brain.services.eom_card_service_commitment import (
    EOMCardServiceCommitmentConflictError,
    EOMCardServiceCommitmentNotFoundError,
    EOMCardServiceCommitmentService,
    EOMCardServiceCommitmentUnavailableError,
    EOMCardServiceCommitmentValidationError,
    eom_card_service_commitment_schema_ready,
)


_CANDIDATE_ID = UUID("11111111-1111-4111-8111-111111111111")
_CONTACT_ID = UUID("22222222-2222-4222-8222-222222222222")
_NOW = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
_SERVICE = funnel_auth_mod.generate_eom_funnel_service_token()
_MIGRATION = (
    Path(__file__).resolve().parent.parent
    / "atlas_brain/storage/migrations/399_eom_card_service_commitments.sql"
)


class _State:
    def __init__(self) -> None:
        self.schema_ready = True
        self.candidate: dict[str, Any] | None = {
            "candidate_id": _CANDIDATE_ID,
            "contact_id": _CONTACT_ID,
            "business_context_id": "effingham_maids",
            "candidate_status": "pending",
            "contact_context_id": "effingham_maids",
            "contact_type": "customer",
            "customer_type": "residential",
            "contact_status": "active",
        }
        self.rows: list[dict[str, Any]] = []
        self.inserts = 0


class _Connection:
    def __init__(self, state: _State) -> None:
        self.state = state

    async def execute(self, query: str, *args: Any) -> None:
        assert "pg_advisory_xact_lock" in query
        assert args

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        if "FROM eom_post_clean_onboarding_candidates AS candidate" in query:
            return dict(self.state.candidate) if self.state.candidate else None
        if "INSERT INTO eom_post_clean_service_commitments" in query:
            self.state.inserts += 1
            row = {
                "id": args[0],
                "candidate_id": args[1],
                "contact_id": args[2],
                "operation_key": args[3],
                "request_fingerprint": args[4],
                "service_commitment": args[5],
                "decided_by_employee_id": args[6],
                "decided_by_name": args[7],
                "decided_at": _NOW,
            }
            self.state.rows.append(row)
            return dict(row)
        raise AssertionError(f"unexpected fetchrow: {query}")

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        assert "FROM eom_post_clean_service_commitments" in query
        candidate_id = args[0]
        contact_id = args[1] if len(args) == 3 else None
        operation_key = args[-1]
        return [
            dict(row)
            for row in self.state.rows
            if row["candidate_id"] == candidate_id
            or row["contact_id"] == contact_id
            or row["operation_key"] == operation_key
        ]


class _Pool:
    is_initialized = True

    def __init__(self, state: _State) -> None:
        self.state = state
        self.connection = _Connection(state)

    async def fetchval(self, query: str) -> bool:
        assert "eom_post_clean_service_commitments" in query
        return self.state.schema_ready

    @asynccontextmanager
    async def transaction(self):
        yield self.connection


def _service(state: _State) -> EOMCardServiceCommitmentService:
    return EOMCardServiceCommitmentService(pool=_Pool(state))


@pytest.mark.asyncio
@pytest.mark.parametrize("commitment", ["recurring", "one_time"])
async def test_decision_records_once_and_replays_exactly(commitment: str) -> None:
    state = _State()
    service = _service(state)
    kwargs = {
        "candidate_id": _CANDIDATE_ID,
        "service_commitment": commitment,
        "operation_key": "eom-card-policy:stable-key",
        "actor_id": 7,
        "actor_name": "Juan",
    }

    created = await service.decide(**kwargs)
    replay = await service.decide(**kwargs)

    assert created == {
        "candidateId": str(_CANDIDATE_ID),
        "contactId": str(_CONTACT_ID),
        "serviceCommitment": commitment,
        "decidedByName": "Juan",
        "decidedAt": _NOW,
        "idempotent": False,
    }
    assert replay == {**created, "idempotent": True}
    assert state.inserts == 1


@pytest.mark.asyncio
async def test_exact_replay_survives_later_subject_ineligibility() -> None:
    state = _State()
    service = _service(state)
    kwargs = {
        "candidate_id": _CANDIDATE_ID,
        "service_commitment": "recurring",
        "operation_key": "eom-card-policy:durable-replay",
        "actor_id": 7,
        "actor_name": "Juan",
    }

    created = await service.decide(**kwargs)
    assert state.candidate is not None
    state.candidate["contact_status"] = "inactive"

    assert await service.decide(**kwargs) == {**created, "idempotent": True}
    assert state.inserts == 1


@pytest.mark.asyncio
async def test_conflicting_candidate_key_or_actor_cannot_reclassify() -> None:
    state = _State()
    service = _service(state)
    common = {
        "candidate_id": _CANDIDATE_ID,
        "operation_key": "eom-card-policy:stable-key",
        "actor_id": 7,
        "actor_name": "Juan",
    }
    await service.decide(service_commitment="one_time", **common)

    with pytest.raises(EOMCardServiceCommitmentConflictError):
        await service.decide(service_commitment="recurring", **common)
    with pytest.raises(EOMCardServiceCommitmentConflictError):
        await service.decide(
            service_commitment="one_time",
            **{**common, "actor_name": "Another operator"},
        )
    assert state.inserts == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_status", "closed"),
        ("contact_status", "inactive"),
        ("contact_type", "lead"),
        ("customer_type", "commercial"),
        ("business_context_id", "other"),
        ("contact_context_id", "other"),
    ],
)
async def test_ineligible_subjects_fail_before_decision_insert(
    field: str, value: str
) -> None:
    state = _State()
    assert state.candidate is not None
    state.candidate[field] = value

    with pytest.raises(EOMCardServiceCommitmentConflictError):
        await _service(state).decide(
            candidate_id=_CANDIDATE_ID,
            service_commitment="recurring",
            operation_key="eom-card-policy:ineligible",
            actor_id=7,
            actor_name="Juan",
        )
    assert state.inserts == 0


@pytest.mark.asyncio
async def test_missing_candidate_and_schema_fail_closed() -> None:
    missing = _State()
    missing.candidate = None
    with pytest.raises(EOMCardServiceCommitmentNotFoundError):
        await _service(missing).decide(
            candidate_id=_CANDIDATE_ID,
            service_commitment="recurring",
            operation_key="eom-card-policy:not-found",
            actor_id=7,
            actor_name="Juan",
        )

    unavailable = _State()
    unavailable.schema_ready = False
    with pytest.raises(EOMCardServiceCommitmentUnavailableError):
        await _service(unavailable).decide(
            candidate_id=_CANDIDATE_ID,
            service_commitment="recurring",
            operation_key="eom-card-policy:no-schema",
            actor_id=7,
            actor_name="Juan",
        )
    assert unavailable.inserts == 0


@pytest.mark.parametrize(
    "commitment",
    [None, "", "weekly", "RECURRING", False, 0, ["recurring"]],
)
def test_commitment_vocabulary_is_closed(commitment: object) -> None:
    with pytest.raises(EOMCardServiceCommitmentValidationError):
        import asyncio

        asyncio.run(
            _service(_State()).decide(
                candidate_id=_CANDIDATE_ID,
                service_commitment=commitment,  # type: ignore[arg-type]
                operation_key="eom-card-policy:invalid-value",
                actor_id=7,
                actor_name="Juan",
            )
        )


@pytest.mark.asyncio
async def test_schema_attestation_names_all_three_required_triggers() -> None:
    class SchemaPool:
        async def fetchval(self, query: str) -> bool:
            assert "eom_card_service_commitment_schema_ready" in query
            assert "trg_protect_eom_card_service_commitment" in query
            assert "trg_protect_eom_card_service_commitment_truncate" in query
            assert "trg_require_eom_recurring_card_commitment" in query
            assert "atlas_eom_handoff_owner" in query
            return True

    assert await eom_card_service_commitment_schema_ready(SchemaPool()) is True


def test_migration_is_append_only_and_refuses_ambiguous_existing_enrollments() -> None:
    sql = _MIGRATION.read_text()

    assert "existing EOM card enrollments require explicit reconciliation" in sql
    assert "EOM service-commitment history is append-only" in sql
    assert "service_commitment = 'recurring'" in sql
    assert "BEFORE INSERT ON eom_card_vault_enrollments" in sql
    assert "UPDATE eom_post_clean_service_commitments" not in sql
    assert "DELETE FROM eom_post_clean_service_commitments" not in sql


def test_funnel_advertises_decision_only_with_the_exact_registered_route() -> None:
    capability = "customer.post_clean_service_commitment.decide"
    route = (
        "POST",
        "/eom-funnel/post-clean-onboarding-candidates/{candidate_id}/service-commitment",
    )
    registered = {
        (method, item.path)
        for item in funnel_mod.router.routes
        for method in (getattr(item, "methods", None) or ())
    }

    assert funnel_mod._CAPABILITY_ROUTES[capability] == route
    assert route in registered
    assert capability in funnel_mod.served_capabilities()


@pytest.mark.asyncio
async def test_real_eom_entrypoint_authenticates_and_replays_decision() -> None:
    state = _State()
    config = EOMFunnelConfig(
        api_enabled=True,
        service_token_sha256=_SERVICE.sha256,
        _env_file=None,
    )
    headers = {
        "Authorization": f"Bearer {_SERVICE.token}",
        "X-EOM-Actor": "Juan",
        "X-EOM-Actor-ID": "7",
        "Idempotency-Key": "eom-card-policy:route-proof",
    }
    path = (
        "/api/v1/eom-funnel/post-clean-onboarding-candidates/"
        f"{_CANDIDATE_ID}/service-commitment"
    )
    main_eom.app.dependency_overrides[funnel_auth_mod.get_eom_funnel_api_config] = (
        lambda: config
    )
    main_eom.app.dependency_overrides[
        funnel_mod._card_service_commitment_dependency
    ] = lambda: _service(state)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main_eom.app),
            base_url="http://test",
        ) as client:
            unauthenticated = await client.post(
                path,
                headers={
                    key: value
                    for key, value in headers.items()
                    if key != "Authorization"
                },
                json={"serviceCommitment": "recurring"},
            )
            missing_actor = await client.post(
                path,
                headers={
                    key: value
                    for key, value in headers.items()
                    if key not in {"X-EOM-Actor", "X-EOM-Actor-ID"}
                },
                json={"serviceCommitment": "recurring"},
            )
            missing_idempotency_key = await client.post(
                path,
                headers={
                    key: value
                    for key, value in headers.items()
                    if key != "Idempotency-Key"
                },
                json={"serviceCommitment": "recurring"},
            )
            invalid_value = await client.post(
                path,
                headers=headers,
                json={"serviceCommitment": "weekly"},
            )
            assert state.inserts == 0
            created = await client.post(
                path,
                headers=headers,
                json={"serviceCommitment": "recurring"},
            )
            replay = await client.post(
                path,
                headers=headers,
                json={"serviceCommitment": "recurring"},
            )
    finally:
        main_eom.app.dependency_overrides.clear()

    assert unauthenticated.status_code == 401
    assert missing_actor.status_code == 422
    assert missing_idempotency_key.status_code == 422
    assert invalid_value.status_code == 422
    assert created.status_code == 201
    assert created.json() == {
        "candidateId": str(_CANDIDATE_ID),
        "contactId": str(_CONTACT_ID),
        "serviceCommitment": "recurring",
        "decidedByName": "Juan",
        "decidedAt": "2026-08-30T12:00:00Z",
        "idempotent": False,
    }
    assert replay.status_code == 200
    assert replay.json() == {**created.json(), "idempotent": True}
    assert state.inserts == 1
