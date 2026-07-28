"""Route and adapter coverage for the private EOM estimate-booking boundary."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from atlas_brain.eom_api import funnel as funnel_mod
from atlas_brain.eom_api import funnel_auth as auth_mod
from atlas_brain.eom_api.config import EOMFunnelConfig
from atlas_brain.services import eom_lead_booking as booking_mod
from atlas_brain.services.eom_lead_booking import (
    EstimateBookingCommand,
    EstimateBookingResult,
    EOMLeadBookingService,
)
from atlas_brain.tools.calendar import CalendarTool

_GENERATED_SERVICE_TOKEN = auth_mod.generate_eom_funnel_service_token()
_SERVICE_TOKEN = _GENERATED_SERVICE_TOKEN.token
_SERVICE_TOKEN_SHA256 = _GENERATED_SERVICE_TOKEN.sha256


class _RouteService:
    def __init__(self, *, idempotent: bool = False) -> None:
        self.commands: list[EstimateBookingCommand] = []
        self.idempotent = idempotent

    async def book_estimate(
        self,
        command: EstimateBookingCommand,
    ) -> EstimateBookingResult:
        self.commands.append(command)
        return EstimateBookingResult(
            operation_id=uuid4(),
            appointment_id=uuid4(),
            calendar_event_id="eom0123456789abcdef0123456789abcdef",
            status="completed",
            idempotent=self.idempotent,
        )


def _app(config: EOMFunnelConfig, service: _RouteService) -> FastAPI:
    app = FastAPI()
    app.include_router(funnel_mod.router)
    app.dependency_overrides[auth_mod.get_eom_funnel_api_config] = lambda: config
    app.dependency_overrides[funnel_mod._booking_service_dependency] = lambda: service
    return app


def _enabled_config() -> EOMFunnelConfig:
    return EOMFunnelConfig(
        api_enabled=True,
        service_token_sha256=_SERVICE_TOKEN_SHA256,
        estimate_calendar_id="estimate-calendar",
    )


def _headers(key: str = "estimate-booking-0001") -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_SERVICE_TOKEN}",
        "X-EOM-Actor": "Mayra",
        "X-EOM-Actor-ID": "7",
        "Idempotency-Key": key,
    }


def _payload() -> dict[str, object]:
    return {
        "startTime": "2026-08-01T10:00:00-05:00",
        "durationMinutes": 60,
        "serviceType": "Cleaning estimate",
        "location": "100 Main St",
        "notes": "Use side door",
    }


@pytest.mark.asyncio
async def test_full_atlas_app_serves_private_estimate_booking_route():
    from atlas_brain.main import app

    service = _RouteService()
    contact_id = uuid4()
    original_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[auth_mod.get_eom_funnel_api_config] = _enabled_config
    app.dependency_overrides[funnel_mod._booking_service_dependency] = lambda: service
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                f"/api/v1/eom-funnel/leads/{contact_id}/estimate-bookings",
                headers=_headers(),
                json=_payload(),
            )
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)

    assert response.status_code == 201
    assert service.commands[0].contact_id == contact_id


@pytest.mark.asyncio
async def test_private_route_requires_token_actor_key_and_timezone():
    service = _RouteService()
    app = _app(_enabled_config(), service)
    contact_id = uuid4()
    path = f"/eom-funnel/leads/{contact_id}/estimate-bookings"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        assert (await client.post(path, json=_payload())).status_code == 401
        assert (
            await client.post(
                path,
                headers={"Authorization": f"Bearer {_SERVICE_TOKEN}"},
                json=_payload(),
            )
        ).status_code == 422
        bad_key = dict(_headers())
        bad_key["Idempotency-Key"] = "short"
        assert (
            await client.post(path, headers=bad_key, json=_payload())
        ).status_code == 422
        bad_time = _payload()
        bad_time["startTime"] = "2026-08-01T10:00:00"
        assert (
            await client.post(path, headers=_headers(), json=bad_time)
        ).status_code == 422
        overflow_time = _payload()
        overflow_time["startTime"] = "9999-12-31T23:59:59+00:00"
        overflow_time["durationMinutes"] = 240
        assert (
            await client.post(path, headers=_headers(), json=overflow_time)
        ).status_code == 422

        created = await client.post(path, headers=_headers(), json=_payload())

    assert created.status_code == 201
    assert created.json()["status"] == "completed"
    assert created.json()["calendarEventId"] == "eom0123456789abcdef0123456789abcdef"
    assert service.commands[0].contact_id == contact_id
    assert service.commands[0].actor_id == 7
    assert service.commands[0].actor_name == "Mayra"
    assert service.commands[0].idempotency_key == "estimate-booking-0001"
    assert service.commands[0].service_type == "Cleaning estimate"


@pytest.mark.asyncio
async def test_private_route_returns_200_for_idempotent_replay():
    service = _RouteService(idempotent=True)
    app = _app(_enabled_config(), service)
    contact_id = uuid4()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/estimate-bookings",
            headers=_headers(),
            json=_payload(),
        )

    assert response.status_code == 200
    assert response.json()["idempotent"] is True


@pytest.mark.asyncio
async def test_private_route_is_unavailable_when_funnel_api_is_disabled():
    app = _app(EOMFunnelConfig(api_enabled=False), _RouteService())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{uuid4()}/estimate-bookings",
            headers=_headers(),
            json=_payload(),
        )
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_calendar_tool_recovers_exact_event_after_conflicting_retry():
    class _Client:
        def __init__(self) -> None:
            self.posts: list[tuple[str, dict[str, object]]] = []
            self.gets: list[tuple[str, dict[str, object]]] = []

        async def post(self, url: str, **kwargs):
            self.posts.append((url, kwargs))
            return httpx.Response(409, request=httpx.Request("POST", url))

        async def get(self, url: str, **kwargs):
            self.gets.append((url, kwargs))
            return httpx.Response(
                200,
                json={
                    "id": "eom0123456789abcdef0123456789abcdef",
                    "status": "confirmed",
                },
                request=httpx.Request("GET", url),
            )

    tool = CalendarTool()
    tool._config = SimpleNamespace(calendar_enabled=True, calendar_refresh_token="refresh")
    client = _Client()
    tool._ensure_client = AsyncMock(return_value=client)
    tool._get_auth_header = AsyncMock(return_value={"Authorization": "Bearer token"})

    result = await tool.create_event(
        summary="Estimate: Retry-safe",
        start=datetime(2026, 8, 1, 15, tzinfo=timezone.utc),
        end=datetime(2026, 8, 1, 16, tzinfo=timezone.utc),
        calendar_id="estimate-calendar",
        event_id="eom0123456789abcdef0123456789abcdef",
    )

    assert result.success is True
    assert client.posts[0][1]["json"]["id"] == "eom0123456789abcdef0123456789abcdef"
    assert client.gets[0][0].endswith("/eom0123456789abcdef0123456789abcdef")


@pytest.mark.asyncio
async def test_calendar_tool_treats_cancelled_conflict_as_terminal():
    class _Client:
        def __init__(self) -> None:
            self.posts: list[tuple[str, dict[str, object]]] = []
            self.gets: list[tuple[str, dict[str, object]]] = []

        async def post(self, url: str, **kwargs):
            self.posts.append((url, kwargs))
            return httpx.Response(409, request=httpx.Request("POST", url))

        async def get(self, url: str, **kwargs):
            self.gets.append((url, kwargs))
            return httpx.Response(
                200,
                json={
                    "id": "eom0123456789abcdef0123456789abcdef",
                    "status": "cancelled",
                },
                request=httpx.Request("GET", url),
            )

    tool = CalendarTool()
    tool._config = SimpleNamespace(calendar_enabled=True, calendar_refresh_token="refresh")
    client = _Client()
    tool._ensure_client = AsyncMock(return_value=client)
    tool._get_auth_header = AsyncMock(return_value={"Authorization": "Bearer token"})

    result = await tool.create_event(
        summary="Estimate: Cancelled conflict",
        start=datetime(2026, 8, 1, 15, tzinfo=timezone.utc),
        end=datetime(2026, 8, 1, 16, tzinfo=timezone.utc),
        calendar_id="estimate-calendar",
        event_id="eom0123456789abcdef0123456789abcdef",
    )

    assert result.success is False
    assert result.error == "API_ERROR"
    assert result.data == {
        "status_code": 409,
        "event_id": "eom0123456789abcdef0123456789abcdef",
        "calendar_event_status": "cancelled",
    }
    assert EOMLeadBookingService._is_terminal_calendar_failure(result) is True


def test_booking_event_id_and_fingerprint_are_payload_scoped():
    contact_id = uuid4()
    command = EstimateBookingCommand(
        contact_id=contact_id,
        idempotency_key="estimate-booking-same",
        actor_id=7,
        actor_name="Mayra",
        start_time=datetime(2026, 8, 1, 10, tzinfo=timezone.utc),
        duration_minutes=60,
        service_type="estimate",
        location="100 Main St",
    )
    changed = EstimateBookingCommand(
        contact_id=contact_id,
        idempotency_key="estimate-booking-same",
        actor_id=9,
        actor_name="Juan",
        start_time=command.start_time,
        duration_minutes=90,
        service_type="estimate",
        location="100 Main St",
    )

    assert EOMLeadBookingService._event_id(uuid4()).startswith("eom")
    assert command.actor == "employee:7:Mayra"
    assert command.request_fingerprint != changed.request_fingerprint


@pytest.mark.asyncio
async def test_create_operation_rechecks_same_key_after_contact_lock(monkeypatch):
    contact_id = uuid4()
    command = EstimateBookingCommand(
        contact_id=contact_id,
        idempotency_key="estimate-booking-race-key",
        actor_id=7,
        actor_name="Mayra",
        start_time=datetime(2026, 8, 1, 15, tzinfo=timezone.utc),
        duration_minutes=60,
        service_type="Cleaning estimate",
        location="100 Main St",
        notes="Use side door",
    )
    operation = {
        "id": uuid4(),
        "request_fingerprint": command.request_fingerprint,
        "appointment_id": None,
        "status": "calendar_rejected",
    }
    contact = {
        "id": contact_id,
        "full_name": "Estimate Lead",
        "phone": "2175550101",
        "email": "estimate@example.com",
        "address": "100 Main St",
        "lead_stage": "new",
        "status": "active",
    }

    class _Conn:
        def __init__(self) -> None:
            self.fetchrow_queries: list[str] = []
            self.same_key_lookups = 0

        async def fetchrow(self, query: str, *args):
            normalized = " ".join(query.split())
            self.fetchrow_queries.append(normalized)
            if normalized.startswith(
                "SELECT * FROM eom_lead_estimate_booking_operations"
            ):
                self.same_key_lookups += 1
                return None if self.same_key_lookups == 1 else operation
            if "FROM contacts" in normalized and "FOR UPDATE" in normalized:
                return contact
            if normalized.startswith(
                "SELECT id FROM eom_lead_estimate_booking_operations"
            ):
                raise AssertionError("active-operation query should not run")
            if normalized.startswith("INSERT INTO eom_lead_estimate_booking_operations"):
                raise AssertionError("duplicate same-key operation should not insert")
            raise AssertionError(f"unexpected fetchrow query: {normalized}")

    class _Tx:
        def __init__(self, conn: _Conn) -> None:
            self.conn = conn

        async def __aenter__(self):
            return self.conn

        async def __aexit__(self, exc_type, exc, tb):
            return False

    conn = _Conn()
    monkeypatch.setattr(
        booking_mod,
        "_transaction_connection",
        lambda _pool: _Tx(conn),
    )
    service = EOMLeadBookingService(
        pool=object(),
        calendar=object(),
        config=EOMFunnelConfig(estimate_calendar_id="estimate-calendar"),
    )

    loaded, created = await service._create_or_load_operation(command)

    assert loaded is operation
    assert created is False
    same_key_indices = [
        index
        for index, query in enumerate(conn.fetchrow_queries)
        if query.startswith("SELECT * FROM eom_lead_estimate_booking_operations")
    ]
    contact_lock_index = next(
        index
        for index, query in enumerate(conn.fetchrow_queries)
        if "FROM contacts" in query and "FOR UPDATE" in query
    )
    assert same_key_indices[0] < contact_lock_index < same_key_indices[1]


@pytest.mark.asyncio
async def test_completion_locks_contact_before_booking_operation(monkeypatch):
    contact_id = uuid4()
    operation_id = uuid4()
    appointment_id = uuid4()
    operation = {
        "id": operation_id,
        "contact_id": contact_id,
        "appointment_id": None,
        "projection_token": uuid4(),
        "contact_snapshot": {
            "full_name": "Estimate Lead",
            "phone": "2175550101",
            "email": "estimate@example.com",
            "address": "100 Main St",
        },
        "start_time": datetime(2026, 8, 1, 15, tzinfo=timezone.utc),
        "end_time": datetime(2026, 8, 1, 16, tzinfo=timezone.utc),
        "service_type": "Cleaning estimate",
        "notes": "Use side door",
        "calendar_event_id": "eom0123456789abcdef0123456789abcdef",
        "actor": "employee:7:Mayra",
        "idempotency_key": "estimate-booking-key-0001",
        "status": "projecting",
    }

    class _Conn:
        def __init__(self) -> None:
            self.fetchrow_queries: list[str] = []

        async def fetchrow(self, query: str, *args):
            normalized = " ".join(query.split())
            self.fetchrow_queries.append(normalized)
            if normalized.startswith(
                "SELECT contact_id FROM eom_lead_estimate_booking_operations"
            ):
                return {"contact_id": contact_id}
            if "SELECT id FROM contacts" in normalized and "FOR UPDATE" in normalized:
                return {"id": contact_id}
            if normalized.startswith(
                "SELECT * FROM eom_lead_estimate_booking_operations"
            ):
                return operation
            if normalized.startswith("INSERT INTO appointments"):
                return {"id": appointment_id}
            if normalized.startswith("UPDATE contacts"):
                return {"id": contact_id}
            if normalized.startswith("UPDATE eom_lead_estimate_booking_operations"):
                return {
                    **operation,
                    "appointment_id": appointment_id,
                    "status": "completed",
                }
            raise AssertionError(f"unexpected fetchrow query: {normalized}")

        async def execute(self, query: str, *args):
            return "INSERT 0 1"

    class _Tx:
        def __init__(self, conn: _Conn) -> None:
            self.conn = conn

        async def __aenter__(self):
            return self.conn

        async def __aexit__(self, exc_type, exc, tb):
            return False

    conn = _Conn()
    monkeypatch.setattr(
        booking_mod,
        "_transaction_connection",
        lambda _pool: _Tx(conn),
    )
    service = EOMLeadBookingService(
        pool=object(),
        calendar=object(),
        config=EOMFunnelConfig(estimate_calendar_id="estimate-calendar"),
    )

    completed = await service._complete_operation(
        operation_id,
        operation["projection_token"],
    )

    assert completed["status"] == "completed"
    contact_lock_index = next(
        index
        for index, query in enumerate(conn.fetchrow_queries)
        if "SELECT id FROM contacts" in query and "FOR UPDATE" in query
    )
    operation_lock_index = next(
        index
        for index, query in enumerate(conn.fetchrow_queries)
        if query.startswith("SELECT * FROM eom_lead_estimate_booking_operations")
        and "FOR UPDATE" in query
    )
    assert contact_lock_index < operation_lock_index
