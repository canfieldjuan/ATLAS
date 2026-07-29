"""Route and adapter coverage for the private EOM estimate-booking boundary."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from atlas_brain.eom_api import funnel as funnel_mod
from atlas_brain.eom_api import funnel_auth as auth_mod
from atlas_brain.mcp import calendar_server as calendar_mcp
from atlas_brain.eom_api.config import EOMFunnelConfig
from atlas_brain.services import eom_lead_booking as booking_mod
from atlas_brain.services.eom_lead_booking import (
    EstimateBookingCommand,
    EstimateBookingResult,
    EOMLeadBookingConflictError,
    EOMLeadBookingProjectionError,
    EOMLeadBookingService,
)
from atlas_brain.storage.repositories import appointment as appointment_repo_mod
from atlas_brain.tools import scheduling as scheduling_mod
from atlas_brain.tools.base import ToolResult
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
        for field_name in ("serviceType", "location", "notes"):
            nul_text = _payload()
            nul_text[field_name] = "safe\x00unsafe"
            assert (
                await client.post(path, headers=_headers(), json=nul_text)
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
                    "summary": "Estimate: Retry-safe",
                    "start": {"dateTime": "2026-08-01T15:00:00+00:00"},
                    "end": {"dateTime": "2026-08-01T16:00:00+00:00"},
                    "location": "100 Main St",
                    "description": "Use side door",
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
        location="100 Main St",
        description="Use side door",
        calendar_id="estimate-calendar",
        event_id="eom0123456789abcdef0123456789abcdef",
    )

    assert result.success is True
    assert client.posts[0][1]["json"]["id"] == "eom0123456789abcdef0123456789abcdef"
    assert client.gets[0][0].endswith("/eom0123456789abcdef0123456789abcdef")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field_name", "override"),
    [
        ("summary", {"summary": "Estimate: Edited"}),
        ("start", {"start": {"dateTime": "2026-08-01T16:00:00+00:00"}}),
        ("end", {"end": {"dateTime": "2026-08-01T17:00:00+00:00"}}),
        ("location", {"location": "200 Main St"}),
        ("description", {"description": "Edited notes"}),
    ],
)
async def test_calendar_tool_rejects_conflicting_retry_when_recovered_fields_differ(
    field_name: str,
    override: dict[str, object],
):
    class _Client:
        async def post(self, url: str, **kwargs):
            return httpx.Response(409, request=httpx.Request("POST", url))

        async def get(self, url: str, **kwargs):
            event = {
                "id": "eom0123456789abcdef0123456789abcdef",
                "summary": "Estimate: Retry-safe",
                "start": {"dateTime": "2026-08-01T15:00:00+00:00"},
                "end": {"dateTime": "2026-08-01T16:00:00+00:00"},
                "location": "100 Main St",
                "description": "Use side door",
                "status": "confirmed",
            }
            event.update(override)
            return httpx.Response(
                200,
                json=event,
                request=httpx.Request("GET", url),
            )

    tool = CalendarTool()
    tool._config = SimpleNamespace(calendar_enabled=True, calendar_refresh_token="refresh")
    tool._ensure_client = AsyncMock(return_value=_Client())
    tool._get_auth_header = AsyncMock(return_value={"Authorization": "Bearer token"})

    result = await tool.create_event(
        summary="Estimate: Retry-safe",
        start=datetime(2026, 8, 1, 15, tzinfo=timezone.utc),
        end=datetime(2026, 8, 1, 16, tzinfo=timezone.utc),
        location="100 Main St",
        description="Use side door",
        calendar_id="estimate-calendar",
        event_id="eom0123456789abcdef0123456789abcdef",
    )

    assert result.success is False
    assert result.error == "API_ERROR"
    assert result.data["event_id"] == "eom0123456789abcdef0123456789abcdef"
    assert field_name in result.data["mismatched_fields"]


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
    local_time_command = EstimateBookingCommand(
        contact_id=contact_id,
        idempotency_key="estimate-booking-same",
        actor_id=7,
        actor_name="Mayra",
        start_time=datetime(
            2026,
            8,
            1,
            10,
            tzinfo=timezone(timedelta(hours=-5)),
        ),
        duration_minutes=60,
        service_type="estimate",
        location="100 Main St",
    )
    utc_command = EstimateBookingCommand(
        contact_id=contact_id,
        idempotency_key="estimate-booking-same",
        actor_id=7,
        actor_name="Mayra",
        start_time=datetime(2026, 8, 1, 15, tzinfo=timezone.utc),
        duration_minutes=60,
        service_type="estimate",
        location="100 Main St",
    )

    assert EOMLeadBookingService._event_id(uuid4()).startswith("eom")
    assert command.actor == "employee:7:Mayra"
    assert command.request_fingerprint != changed.request_fingerprint
    assert local_time_command.request_fingerprint == utc_command.request_fingerprint


def _operation_row(**overrides):
    base = {
        "id": uuid4(),
        "contact_id": uuid4(),
        "idempotency_key": "estimate-booking-rollback",
        "request_fingerprint": "1" * 64,
        "actor": "employee:7:Mayra",
        "start_time": datetime(2026, 8, 1, 15, tzinfo=timezone.utc),
        "end_time": datetime(2026, 8, 1, 16, tzinfo=timezone.utc),
        "service_type": "Cleaning estimate",
        "location": "100 Main St",
        "notes": "Use side door",
        "status": "calendar_failed",
        "appointment_id": None,
        "calendar_event_id": "eom0123456789abcdef0123456789abcdef",
        "calendar_id": "estimate-calendar",
        "contact_snapshot": {
            "full_name": "Estimate Lead",
            "phone": "2175550101",
            "email": "estimate@example.com",
            "address": "100 Main St",
        },
    }
    base.update(overrides)
    return base


def test_rollback_drain_command_reconstructs_original_booking_request():
    contact_id = uuid4()
    original_command = EstimateBookingCommand(
        contact_id=contact_id,
        idempotency_key="estimate-booking-rollback",
        actor_id=42,
        actor_name="Mayra Ortiz",
        start_time=datetime(
            2026,
            8,
            1,
            10,
            tzinfo=timezone(timedelta(hours=-5)),
        ),
        duration_minutes=60,
        service_type="Cleaning estimate",
        location="100 Main St",
        notes="Use side door",
    )
    operation = _operation_row(
        contact_id=contact_id,
        actor="employee:42:Mayra Ortiz",
        request_fingerprint=original_command.request_fingerprint,
        start_time=datetime(2026, 8, 1, 15, tzinfo=timezone.utc),
        end_time=datetime(2026, 8, 1, 16, tzinfo=timezone.utc),
    )

    command = EOMLeadBookingService._command_from_operation(operation)

    assert command.contact_id == operation["contact_id"]
    assert command.idempotency_key == operation["idempotency_key"]
    assert command.actor_id == 42
    assert command.actor_name == "Mayra Ortiz"
    assert command.duration_minutes == 60
    assert command.location == "100 Main St"
    assert command.notes == "Use side door"
    assert command.request_fingerprint == operation["request_fingerprint"]


@pytest.mark.parametrize(
    "actor",
    ["Mayra", "employee:not-int:Mayra", "employee:7:", "employee:0:Mayra"],
)
def test_rollback_drain_rejects_unreplayable_operation_actor(actor: str):
    with pytest.raises(EOMLeadBookingConflictError):
        EOMLeadBookingService._command_from_operation(_operation_row(actor=actor))


@pytest.mark.asyncio
async def test_rollback_drain_replays_unfinished_operations_before_reporting_safe():
    operation = _operation_row()
    appointment_id = uuid4()

    class _Pool:
        async def fetch(self, *args):
            return [operation]

        async def fetchval(self, *args):
            return 0

    service = EOMLeadBookingService(
        pool=_Pool(),
        calendar=object(),
        config=EOMFunnelConfig(estimate_calendar_id="estimate-calendar"),
    )
    service.book_estimate = AsyncMock(
        return_value=EstimateBookingResult(
            operation_id=operation["id"],
            appointment_id=appointment_id,
            calendar_event_id=operation["calendar_event_id"],
            status="completed",
            idempotent=True,
        )
    )

    summary = await service.drain_unfinished_for_rollback(limit=10)

    assert summary == {
        "attempted": 1,
        "drained": [
            {
                "operation_id": str(operation["id"]),
                "status": "completed",
                "appointment_id": str(appointment_id),
            }
        ],
        "failures": [],
        "remaining": 0,
        "ok": True,
    }
    replayed = service.book_estimate.await_args.args[0]
    assert replayed.idempotency_key == operation["idempotency_key"]


@pytest.mark.asyncio
async def test_rollback_drain_treats_terminalized_absent_calendar_event_as_drained():
    operation = _operation_row()

    class _Pool:
        async def fetch(self, *args):
            return [operation]

        async def fetchrow(self, *args):
            return {
                "id": operation["id"],
                "status": "calendar_rejected",
                "appointment_id": None,
            }

        async def fetchval(self, *args):
            return 0

    service = EOMLeadBookingService(
        pool=_Pool(),
        calendar=object(),
        config=EOMFunnelConfig(estimate_calendar_id="estimate-calendar"),
    )
    service.book_estimate = AsyncMock(
        side_effect=EOMLeadBookingProjectionError("Calendar API error: 404")
    )

    summary = await service.drain_unfinished_for_rollback(limit=10)

    assert summary["ok"] is True
    assert summary["drained"] == [
        {
            "operation_id": str(operation["id"]),
            "status": "calendar_rejected",
            "appointment_id": None,
        }
    ]


@pytest.mark.asyncio
async def test_rollback_drain_refuses_ambiguous_unfinished_operations():
    operation = _operation_row(status="projecting")

    class _Pool:
        async def fetch(self, *args):
            return [operation]

        async def fetchval(self, *args):
            return 1

    service = EOMLeadBookingService(
        pool=_Pool(),
        calendar=object(),
        config=EOMFunnelConfig(estimate_calendar_id="estimate-calendar"),
    )
    service.book_estimate = AsyncMock(
        side_effect=EOMLeadBookingConflictError(
            "Estimate calendar projection is already in progress"
        )
    )

    summary = await service.drain_unfinished_for_rollback(limit=10)

    assert summary["ok"] is False
    assert summary["remaining"] == 1
    assert summary["failures"] == [
        {
            "operation_id": str(operation["id"]),
            "status": "projecting",
            "error": "Estimate calendar projection is already in progress",
        }
    ]


@pytest.mark.asyncio
async def test_booking_reconcile_rejects_mismatched_recovered_calendar_event():
    operation = {
        "id": uuid4(),
        "calendar_event_id": "eom0123456789abcdef0123456789abcdef",
        "calendar_id": "estimate-calendar",
        "start_time": datetime(2026, 8, 1, 15, tzinfo=timezone.utc),
        "end_time": datetime(2026, 8, 1, 16, tzinfo=timezone.utc),
        "location": "100 Main St",
        "notes": "Use side door",
        "contact_snapshot": {
            "full_name": "Estimate Lead",
            "phone": "2175550101",
            "email": "estimate@example.com",
            "address": "100 Main St",
        },
    }

    class _Calendar:
        async def get_event(self, **kwargs):
            return ToolResult(
                success=True,
                data={
                    "event_id": kwargs["event_id"],
                    "summary": "Estimate: Estimate Lead",
                    "start": {"dateTime": "2026-08-01T16:00:00+00:00"},
                    "end": {"dateTime": "2026-08-01T17:00:00+00:00"},
                    "location": "100 Main St",
                    "description": (
                        "EOM office estimate booking\n"
                        "Lead: Estimate Lead\n"
                        "Phone: 2175550101\n"
                        "Email: estimate@example.com\n"
                        "\n"
                        "Use side door"
                    ),
                    "calendar_event_status": "confirmed",
                },
                message="fetched",
            )

    service = EOMLeadBookingService(
        pool=object(),
        calendar=_Calendar(),
        config=EOMFunnelConfig(estimate_calendar_id="estimate-calendar"),
    )

    assert await service._reconcile_existing_calendar_event(operation) is None


@pytest.mark.asyncio
async def test_generic_cancel_rejects_eom_estimate_booking_appointments(monkeypatch):
    appointment_id = uuid4()
    appointment = {
        "id": appointment_id,
        "calendar_event_id": "eom0123456789abcdef0123456789abcdef",
        "customer_name": "Estimate Lead",
        "customer_phone": "2175550101",
        "start_time": datetime(2026, 8, 1, 15, tzinfo=timezone.utc),
        "eom_estimate_booking_operation_id": uuid4(),
    }

    class _Repo:
        def __init__(self) -> None:
            self.cancel_calls = 0

        async def get_by_id(self, loaded_id):
            assert loaded_id == appointment_id
            return appointment

        async def get_by_phone(self, *args, **kwargs):
            return [appointment]

        async def cancel(self, *args, **kwargs):
            self.cancel_calls += 1
            raise AssertionError("estimate booking appointment must not be cancelled")

    repo = _Repo()
    scheduling_service = SimpleNamespace(cancel_appointment=AsyncMock())
    tool = scheduling_mod.CancelAppointmentTool(
        context_provider=lambda: SimpleNamespace(
            scheduling=SimpleNamespace(calendar_id="estimate-calendar"),
        ),
        appointment_repo_provider=lambda: repo,
        scheduling_service_provider=lambda: scheduling_service,
    )

    for params in (
        {"appointment_id": str(appointment_id)},
        {"customer_phone": "217-555-0101"},
    ):
        scheduling_service.cancel_appointment.reset_mock()
        result = await tool.execute(params)

        assert result.success is False
        assert result.error == "EOM_ESTIMATE_BOOKING_MANAGED"
        assert repo.cancel_calls == 0
        scheduling_service.cancel_appointment.assert_not_awaited()


@pytest.mark.asyncio
async def test_generic_reschedule_rejects_eom_estimate_booking_appointments(monkeypatch):
    appointment = {
        "id": uuid4(),
        "calendar_event_id": "eom0123456789abcdef0123456789abcdef",
        "customer_name": "Estimate Lead",
        "customer_email": "estimate@example.com",
        "customer_address": "100 Main St",
        "customer_phone": "2175550101",
        "service_type": "Cleaning estimate",
        "notes": "Use side door",
        "eom_estimate_booking_operation_id": uuid4(),
    }

    class _Repo:
        def __init__(self) -> None:
            self.update_calls = 0

        async def get_by_phone(self, *args, **kwargs):
            return [appointment]

        async def update(self, *args, **kwargs):
            self.update_calls += 1
            raise AssertionError("estimate booking appointment must not be rescheduled")

    repo = _Repo()
    scheduling_service = SimpleNamespace(
        book_appointment=AsyncMock(),
        cancel_appointment=AsyncMock(),
    )
    tool = scheduling_mod.RescheduleAppointmentTool(
        context_provider=lambda: SimpleNamespace(
            scheduling=SimpleNamespace(
                calendar_id="estimate-calendar",
                default_duration_minutes=60,
            ),
            hours=SimpleNamespace(timezone="UTC"),
        ),
        appointment_repo_provider=lambda: repo,
        scheduling_service_provider=lambda: scheduling_service,
    )

    result = await tool.execute(
        {
            "customer_phone": "217-555-0101",
            "new_date": "tomorrow",
            "new_time": "2pm",
        }
    )

    assert result.success is False
    assert result.error == "EOM_ESTIMATE_BOOKING_MANAGED"
    assert repo.update_calls == 0
    scheduling_service.book_appointment.assert_not_awaited()
    scheduling_service.cancel_appointment.assert_not_awaited()


@pytest.mark.asyncio
async def test_calendar_mcp_sync_rejects_eom_estimate_booking_before_calendar(monkeypatch):
    appointment_id = uuid4()

    class _Pool:
        async def fetchrow(self, query: str, loaded_id: str):
            assert loaded_id == str(appointment_id)
            assert "eom_estimate_booking_operation_id" in query
            return {
                "id": appointment_id,
                "customer_name": "Estimate Lead",
                "customer_address": "100 Main St",
                "start_time": datetime(2026, 8, 1, 15, tzinfo=timezone.utc),
                "end_time": datetime(2026, 8, 1, 16, tzinfo=timezone.utc),
                "notes": "Use side door",
                "calendar_event_id": "eom0123456789abcdef0123456789abcdef",
                "eom_estimate_booking_operation_id": uuid4(),
            }

        async def execute(self, *args, **kwargs):
            raise AssertionError("estimate booking sync must not update appointments")

    provider = SimpleNamespace(
        create_event=AsyncMock(
            side_effect=AssertionError("estimate booking sync must not create events")
        ),
        update_event=AsyncMock(
            side_effect=AssertionError("estimate booking sync must not update events")
        ),
    )
    result = json.loads(
        await calendar_mcp._sync_appointment_with_dependencies(
            str(appointment_id),
            None,
            pool=_Pool(),
            provider=provider,
        )
    )

    assert result["success"] is False
    assert result["error"] == "EOM_ESTIMATE_BOOKING_MANAGED"
    provider.create_event.assert_not_awaited()
    provider.update_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_appointment_repository_blocks_protected_eom_estimate_mutations(monkeypatch):
    appointment_id = uuid4()

    class _Pool:
        is_initialized = True

        def __init__(self) -> None:
            self.cancel_query = ""
            self.update_query = ""
            self.update_args: tuple[object, ...] = ()

        async def execute(self, query: str, *args):
            self.cancel_query = query
            return "UPDATE 0"

        async def fetchrow(self, query: str, *args):
            self.update_query = query
            self.update_args = args
            return None

    pool = _Pool()
    repo = appointment_repo_mod.AppointmentRepository(pool_provider=lambda: pool)

    assert await repo.cancel(appointment_id, "caller cancelled") is False
    assert "eom_estimate_booking_operation_id IS NULL" in pool.cancel_query

    updated = await repo.update(
        appointment_id,
        start_time=datetime(2026, 8, 1, 16, tzinfo=timezone.utc),
        end_time=datetime(2026, 8, 1, 17, tzinfo=timezone.utc),
    )

    assert updated is None
    assert "eom_estimate_booking_operation_id IS NULL" in pool.update_query
    assert pool.update_args[0] == appointment_id
    assert pool.update_args[1] is True


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
