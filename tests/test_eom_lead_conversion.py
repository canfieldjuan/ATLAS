"""HTTP boundary proof for the private EOM office conversion API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

from atlas_brain.eom_api import funnel as funnel_mod
from atlas_brain.eom_api import funnel_auth as auth_mod
from atlas_brain.eom_api.config import EOMFunnelConfig
from atlas_brain.services.eom_estimate_booking import (
    deterministic_eom_estimate_calendar_event_id,
)
from atlas_brain.services.eom_lead_conversion import EOMLeadConversionError
from atlas_brain.tools.base import ToolResult
from atlas_brain.tools.calendar import CalendarAuthError, CalendarTool

_GENERATED_SERVICE_TOKEN = auth_mod.generate_eom_funnel_service_token()
_SERVICE_TOKEN = _GENERATED_SERVICE_TOKEN.token
_SERVICE_TOKEN_SHA256 = _GENERATED_SERVICE_TOKEN.sha256


class _CRM:
    def __init__(self, *, review_leads: list[dict[str, object]] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.review_calls: list[dict[str, object]] = []
        self.prepare_calls: list[dict[str, object]] = []
        self.complete_calls: list[dict[str, object]] = []
        self.ambiguous_calls: list[dict[str, object]] = []
        self.failed_calls: list[dict[str, object]] = []
        self.execution_lock_keys: list[str] = []
        self.review_leads = review_leads or []

    @asynccontextmanager
    async def eom_estimate_booking_execution_lock(self, *, booking_key: str):
        self.execution_lock_keys.append(booking_key)
        yield

    async def list_eom_new_lead_review_items(
        self,
        *,
        limit: int,
        cursor_created_at=None,
        cursor_contact_id=None,
    ):
        self.review_calls.append(
            {
                "limit": limit,
                "cursor_created_at": cursor_created_at,
                "cursor_contact_id": cursor_contact_id,
            }
        )
        return self.review_leads

    async def finalize_eom_customer_handoff(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "handoff_id": "b4fef3b3-a2bd-44e5-aac4-67176270c173",
            "contact_id": kwargs["contact_id"],
            "tracker_customer_id": kwargs["tracker_customer_id"],
            "tracker_site_id": kwargs["tracker_site_id"],
            "approval_key": kwargs["approval_key"],
            "idempotent": False,
        }

    async def prepare_eom_estimate_booking(self, **kwargs):
        self.prepare_calls.append(kwargs)
        return {
            "contact_id": kwargs["contact_id"],
            "lead_stage": "new",
            "status": "calendar_pending",
            "calendar_event_id": None,
            "expected_calendar_event_id": kwargs["expected_calendar_event_id"],
            "idempotent": False,
            "contact": {
                "full_name": "Review Queue Lead",
                "address": "100 Main St",
            },
        }

    async def complete_eom_estimate_booking(self, **kwargs):
        self.complete_calls.append(kwargs)
        return {
            "contact_id": kwargs["contact_id"],
            "lead_stage": "estimate_booked",
            "status": "estimate_booked",
            "calendar_event_id": kwargs["calendar_event_id"],
            "expected_calendar_event_id": kwargs["expected_calendar_event_id"],
            "idempotent": False,
        }

    async def mark_eom_estimate_booking_calendar_ambiguous(self, **kwargs):
        self.ambiguous_calls.append(kwargs)

    async def mark_eom_estimate_booking_calendar_failed(self, **kwargs):
        self.failed_calls.append(kwargs)


class _Calendar:
    def __init__(
        self,
        *,
        success: bool = True,
        event_id: str | None = None,
        error: str = "API_ERROR",
        message: str = "Calendar API error: 503",
        data: dict[str, object] | None = None,
    ) -> None:
        self.success = success
        self.event_id = event_id
        self.error = error
        self.message = message
        self.data = data or {}
        self.calls: list[dict[str, object]] = []

    async def create_event(self, **kwargs):
        self.calls.append(kwargs)
        if not self.success:
            return ToolResult(
                success=False,
                error=self.error,
                data=self.data,
                message=self.message,
            )
        return ToolResult(
            success=True,
            data={"event_id": self.event_id or kwargs["event_id"]},
            message="Created event",
        )


class _CalendarResponse:
    def __init__(self, *, status_code: int, payload: dict[str, object]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, object]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "calendar error",
                request=httpx.Request("POST", "https://calendar.example/events"),
                response=httpx.Response(self.status_code),
            )


class _CalendarClient:
    def __init__(
        self,
        *,
        post_response: _CalendarResponse,
        get_response: _CalendarResponse | None = None,
    ) -> None:
        self.post_response = post_response
        self.get_response = get_response
        self.post_calls: list[dict[str, object]] = []
        self.get_calls: list[dict[str, object]] = []

    async def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]):
        self.post_calls.append(
            {"url": url, "headers": dict(headers), "json": dict(json)}
        )
        return self.post_response

    async def get(self, url: str, *, headers: dict[str, str]):
        self.get_calls.append({"url": url, "headers": dict(headers)})
        assert self.get_response is not None
        return self.get_response


def _app(
    crm: _CRM, config: EOMFunnelConfig, calendar: _Calendar | None = None
) -> FastAPI:
    app = FastAPI()
    app.include_router(funnel_mod.router)
    app.dependency_overrides[funnel_mod._crm_dependency] = lambda: crm
    app.dependency_overrides[funnel_mod._calendar_dependency] = (
        lambda: calendar or _Calendar()
    )
    app.dependency_overrides[auth_mod.get_eom_funnel_api_config] = lambda: config
    return app


def _enabled_config() -> EOMFunnelConfig:
    return EOMFunnelConfig(
        api_enabled=True,
        service_token_sha256=_SERVICE_TOKEN_SHA256,
    )


def _approval_key() -> str:
    return f"office-handoff-{uuid4().hex}"


def _headers(
    token: str = _SERVICE_TOKEN,
    approval_key: str | None = None,
    *,
    actor: str = "Juan Canfield",
    actor_id: str = "1",
) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "X-EOM-Actor": actor,
        "X-EOM-Actor-ID": actor_id,
        "Idempotency-Key": approval_key or _approval_key(),
    }


def _booking_payload(**overrides) -> dict[str, object]:
    payload: dict[str, object] = {
        "scheduled_start": "2026-08-04T14:00:00-05:00",
        "scheduled_end": "2026-08-04T15:00:00-05:00",
        "calendar_id": "estimate-calendar",
        "notes": "Bring estimate worksheet",
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_full_atlas_app_serves_public_intake_and_private_handoff_together():
    """The actual full aggregate serves the tracker callback beside lead intake."""
    from atlas_brain.main import app

    contact_id = uuid4()
    crm = _CRM(
        review_leads=[
            {
                "contact_id": contact_id,
                "full_name": "Review Queue Lead",
                "email": "review@example.com",
                "phone": "2175550100",
                "address": "100 Main St",
                "source": "web",
                "lead_stage": "new",
                "created_at": "2026-07-27T12:00:00+00:00",
            }
        ]
    )
    original_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[funnel_mod._crm_dependency] = lambda: crm
    app.dependency_overrides[auth_mod.get_eom_funnel_api_config] = _enabled_config
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            public_response = await client.get("/api/v1/leads/intake")
            review_response = await client.get(
                "/api/v1/eom-funnel/leads",
                headers=_headers(),
            )
            response = await client.post(
                "/api/v1/eom-funnel/customer-handoffs",
                headers=_headers(),
                json={
                    "contact_id": "11111111-1111-1111-1111-111111111111",
                    "tracker_customer_id": 12,
                    "tracker_site_id": 24,
                },
            )
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)

    assert public_response.status_code == 405
    assert review_response.status_code == 200
    assert review_response.json() == {
        "leads": [
            {
                "contactId": str(contact_id),
                "fullName": "Review Queue Lead",
                "email": "review@example.com",
                "phone": "2175550100",
                "address": "100 Main St",
                "source": "web",
                "leadStage": "new",
                "createdAt": "2026-07-27T12:00:00Z",
            }
        ],
        "limit": 100,
        "cursor": None,
        "hasMore": False,
        "nextCursor": None,
    }
    assert crm.review_calls == [
        {"limit": 101, "cursor_created_at": None, "cursor_contact_id": None}
    ]
    assert response.status_code == 201
    assert crm.calls


@pytest.mark.asyncio
async def test_private_lead_review_returns_only_the_closed_projection():
    first_contact_id = uuid4()
    second_contact_id = uuid4()
    crm = _CRM(
        review_leads=[
            {
                "contact_id": first_contact_id,
                "full_name": "Review Queue Lead 1",
                "email": "review1@example.com",
                "phone": "2175550100",
                "address": "100 Main St",
                "source": "web",
                "lead_stage": "new",
                "created_at": "2026-07-27T12:00:00+00:00",
            },
            {
                "contact_id": second_contact_id,
                "full_name": "Review Queue Lead 2",
                "email": "review2@example.com",
                "phone": "2175550101",
                "address": "101 Main St",
                "source": "web",
                "lead_stage": "estimate_booked",
                "created_at": "2026-07-27T11:00:00+00:00",
            },
        ]
    )
    expected_cursor = funnel_mod._encode_lead_review_cursor(
        created_at=datetime.fromisoformat("2026-07-27T12:00:00+00:00"),
        contact_id=first_contact_id,
    )
    app = _app(crm, _enabled_config())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/eom-funnel/leads?limit=1", headers=_headers())

    assert response.status_code == 200
    assert response.json() == {
        "leads": [
            {
                "contactId": str(first_contact_id),
                "fullName": "Review Queue Lead 1",
                "email": "review1@example.com",
                "phone": "2175550100",
                "address": "100 Main St",
                "source": "web",
                "leadStage": "new",
                "createdAt": "2026-07-27T12:00:00Z",
            }
        ],
        "limit": 1,
        "cursor": None,
        "hasMore": True,
        "nextCursor": expected_cursor,
    }
    assert crm.review_calls == [
        {"limit": 2, "cursor_created_at": None, "cursor_contact_id": None}
    ]


@pytest.mark.asyncio
async def test_private_lead_review_forwards_keyset_cursor_for_continuation():
    cursor_created_at = datetime.fromisoformat("2026-07-27T12:00:00+00:00")
    cursor_contact_id = uuid4()
    cursor = funnel_mod._encode_lead_review_cursor(
        created_at=cursor_created_at,
        contact_id=cursor_contact_id,
    )
    crm = _CRM()
    app = _app(crm, _enabled_config())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            f"/eom-funnel/leads?limit=50&cursor={cursor}",
            headers=_headers(),
        )

    assert response.status_code == 200
    assert response.json() == {
        "leads": [],
        "limit": 50,
        "cursor": cursor,
        "hasMore": False,
        "nextCursor": None,
    }
    assert crm.review_calls == [
        {
            "limit": 51,
            "cursor_created_at": cursor_created_at,
            "cursor_contact_id": cursor_contact_id,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "headers", "expected_status"),
    (
        (EOMFunnelConfig(api_enabled=False), _headers(), 503),
        (_enabled_config(), {**_headers(), "Authorization": ""}, 401),
        (_enabled_config(), {**_headers(actor=" ")}, 422),
        (_enabled_config(), {**_headers(actor_id="0")}, 422),
    ),
)
async def test_private_lead_review_rejects_boundary_failures_before_crm_call(
    config: EOMFunnelConfig,
    headers: dict[str, str],
    expected_status: int,
):
    crm = _CRM()
    app = _app(crm, config)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/eom-funnel/leads", headers=headers)

    assert response.status_code == expected_status
    assert crm.review_calls == []


@pytest.mark.asyncio
async def test_private_lead_review_rejects_out_of_range_limit_before_crm_call():
    crm = _CRM()
    app = _app(crm, _enabled_config())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/eom-funnel/leads?limit=201", headers=_headers())

    assert response.status_code == 422
    assert crm.review_calls == []


@pytest.mark.asyncio
async def test_private_lead_review_rejects_malformed_cursor_before_crm_call():
    crm = _CRM()
    app = _app(crm, _enabled_config())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            "/eom-funnel/leads?cursor=not-a-real-cursor",
            headers=_headers(),
        )

    assert response.status_code == 422
    assert crm.review_calls == []


@pytest.mark.asyncio
async def test_private_estimate_booking_prepares_calendar_and_completes_in_order():
    crm = _CRM()
    calendar = _Calendar()
    app = _app(crm, _enabled_config(), calendar=calendar)
    contact_id = uuid4()
    booking_key = f"office-booking-{uuid4().hex}"
    expected_event_id = deterministic_eom_estimate_calendar_event_id(
        contact_id=str(contact_id),
        booking_key=booking_key,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/estimate-bookings",
            headers=_headers(approval_key=booking_key),
            json=_booking_payload(),
        )

    assert response.status_code == 201
    assert response.json() == {
        "success": True,
        "contact_id": str(contact_id),
        "lead_stage": "estimate_booked",
        "status": "estimate_booked",
        "calendar_event_id": expected_event_id,
        "expected_calendar_event_id": expected_event_id,
        "idempotent": False,
    }
    assert crm.prepare_calls[0]["contact_id"] == str(contact_id)
    assert crm.prepare_calls[0]["booking_key"] == booking_key
    assert crm.prepare_calls[0]["expected_calendar_event_id"] == expected_event_id
    assert calendar.calls == [
        {
            "summary": "Estimate: Review Queue Lead",
            "start": datetime.fromisoformat("2026-08-04T14:00:00-05:00"),
            "end": datetime.fromisoformat("2026-08-04T15:00:00-05:00"),
            "location": "100 Main St",
            "description": (
                "Scheduled from the private EOM lead funnel.\n\n"
                "Bring estimate worksheet"
            ),
            "calendar_id": "estimate-calendar",
            "event_id": expected_event_id,
        }
    ]
    assert crm.complete_calls[0]["calendar_event_id"] == expected_event_id
    assert crm.ambiguous_calls == []


@pytest.mark.asyncio
async def test_private_estimate_booking_uses_configured_calendar_when_payload_omits_id():
    crm = _CRM()
    calendar = _Calendar()
    calendar._config = SimpleNamespace(calendar_id="configured-estimate-calendar")
    app = _app(crm, _enabled_config(), calendar=calendar)
    contact_id = uuid4()
    booking_key = f"office-booking-{uuid4().hex}"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/estimate-bookings",
            headers=_headers(approval_key=booking_key),
            json=_booking_payload(calendar_id=None),
        )

    assert response.status_code == 201
    assert crm.prepare_calls[0]["calendar_id"] == "configured-estimate-calendar"
    assert crm.prepare_calls[0]["calendar_id_explicit"] is False
    assert calendar.calls[0]["calendar_id"] == "configured-estimate-calendar"
    assert crm.complete_calls[0]["calendar_id"] == "configured-estimate-calendar"


@pytest.mark.asyncio
async def test_private_estimate_booking_reports_explicit_calendar_id_to_prepare():
    crm = _CRM()
    calendar = _Calendar()
    app = _app(crm, _enabled_config(), calendar=calendar)
    contact_id = uuid4()
    booking_key = f"office-booking-{uuid4().hex}"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/estimate-bookings",
            headers=_headers(approval_key=booking_key),
            json=_booking_payload(calendar_id="estimate-calendar"),
        )

    assert response.status_code == 201
    assert crm.prepare_calls[0]["calendar_id"] == "estimate-calendar"
    assert crm.prepare_calls[0]["calendar_id_explicit"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "numeric_payload",
    [
        {"scheduled_start": 0, "scheduled_end": 3600},
        {"scheduled_start": 1754323200.5, "scheduled_end": 1754326800.5},
        {"scheduled_start": "0", "scheduled_end": "3600"},
        {"scheduled_start": "1754323200", "scheduled_end": "1754326800"},
        {"scheduled_start": "20260804T140000Z", "scheduled_end": "20260804T150000Z"},
        {
            "scheduled_start": "2026-08-04 19:00:00Z",
            "scheduled_end": "2026-08-04 20:00:00Z",
        },
        # Held-out shapes Pydantic accepts but RFC 3339 forbids: missing
        # seconds, date-only, and a colon-less UTC offset.
        {
            "scheduled_start": "2026-08-04T19:00Z",
            "scheduled_end": "2026-08-04T20:00Z",
        },
        {"scheduled_start": "2026-08-04", "scheduled_end": "2026-08-05"},
        {
            "scheduled_start": "2026-08-04T19:00:00+0500",
            "scheduled_end": "2026-08-04T20:00:00+0500",
        },
    ],
)
async def test_private_estimate_booking_rejects_numeric_timestamps(numeric_payload):
    """Pydantic lax mode would coerce epoch numbers -- and digit-only strings
    like "3600" -- into 1970-era UTC-aware datetimes that pass the
    timezone/ordering checks; the boundary must 422 anything that is not an
    RFC 3339 date-time string before CRM or Calendar sees it."""
    crm = _CRM()
    calendar = _Calendar()
    app = _app(crm, _enabled_config(), calendar=calendar)
    contact_id = uuid4()
    booking_key = f"office-booking-{uuid4().hex}"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/estimate-bookings",
            headers=_headers(approval_key=booking_key),
            json=_booking_payload(**numeric_payload),
        )

    assert response.status_code == 422
    assert crm.prepare_calls == []
    assert calendar.calls == []


@pytest.mark.asyncio
async def test_private_estimate_booking_holds_execution_lock_for_booking_key():
    """The whole prepare -> Calendar -> complete span runs under the same-key
    execution lock so handoff can fence in-flight attempts."""
    crm = _CRM()
    calendar = _Calendar()
    app = _app(crm, _enabled_config(), calendar=calendar)
    contact_id = uuid4()
    booking_key = f"office-booking-{uuid4().hex}"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/estimate-bookings",
            headers=_headers(approval_key=booking_key),
            json=_booking_payload(),
        )

    assert response.status_code == 201
    assert crm.execution_lock_keys == [booking_key]


@pytest.mark.asyncio
@pytest.mark.parametrize("pre_request_error", ["TOOL_DISABLED", "NOT_CONFIGURED"])
async def test_private_estimate_booking_pre_request_calendar_failure_is_terminal(
    pre_request_error,
):
    """TOOL_DISABLED/NOT_CONFIGURED happen before any Google request, so no
    event can exist: the booking must record a terminal failed attempt, never
    an ambiguous wedge (a service booted before Calendar secrets are populated
    must not permanently block the lead)."""
    crm = _CRM()
    calendar = _Calendar(
        success=False,
        error=pre_request_error,
        message="Calendar unavailable before any request",
    )
    app = _app(crm, _enabled_config(), calendar=calendar)
    contact_id = uuid4()
    booking_key = f"office-booking-{uuid4().hex}"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/estimate-bookings",
            headers=_headers(approval_key=booking_key),
            json=_booking_payload(),
        )

    assert response.status_code == 502
    assert crm.ambiguous_calls == []
    assert len(crm.failed_calls) == 1
    assert crm.failed_calls[0]["calendar_error"] == pre_request_error


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_error", ["EXECUTION_ERROR", "API_ERROR", "AUTH_ERROR"])
async def test_private_estimate_booking_auth_phase_failure_is_terminal(auth_error):
    """OAuth token acquisition happens before any Google Calendar event
    request, so a token timeout/5xx/refresh failure proves no event write
    exists: the booking must record a terminal failed attempt, never an
    ambiguous wedge that outlives the token outage."""
    data: dict[str, object] = {"request_phase": "auth"}
    if auth_error == "API_ERROR":
        data["status_code"] = 503
    crm = _CRM()
    calendar = _Calendar(
        success=False,
        error=auth_error,
        message="OAuth token endpoint unavailable",
        data=data,
    )
    app = _app(crm, _enabled_config(), calendar=calendar)
    contact_id = uuid4()
    booking_key = f"office-booking-{uuid4().hex}"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/estimate-bookings",
            headers=_headers(approval_key=booking_key),
            json=_booking_payload(),
        )

    assert response.status_code == 502
    assert crm.ambiguous_calls == []
    assert len(crm.failed_calls) == 1
    assert crm.failed_calls[0]["calendar_error"] == auth_error


@pytest.mark.asyncio
async def test_private_estimate_booking_marks_ambiguous_when_completion_rejects():
    """Once Calendar returned the expected event ID the appointment exists;
    a completion rejection (e.g. the contact mutated out-of-band mid-flight)
    must record reconciliation evidence instead of orphaning the event
    behind a forever-pending ledger."""

    class _RejectingCompletionCRM(_CRM):
        async def complete_eom_estimate_booking(self, **kwargs):
            self.complete_calls.append(kwargs)
            raise EOMLeadConversionError(409, "EOM contact is not a lead")

    crm = _RejectingCompletionCRM()
    calendar = _Calendar()
    app = _app(crm, _enabled_config(), calendar=calendar)
    contact_id = uuid4()
    booking_key = f"office-booking-{uuid4().hex}"
    expected_event_id = deterministic_eom_estimate_calendar_event_id(
        contact_id=str(contact_id),
        booking_key=booking_key,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/estimate-bookings",
            headers=_headers(approval_key=booking_key),
            json=_booking_payload(),
        )

    assert response.status_code == 409
    assert len(crm.complete_calls) == 1
    assert len(crm.ambiguous_calls) == 1
    assert crm.ambiguous_calls[0]["expected_calendar_event_id"] == expected_event_id
    assert crm.ambiguous_calls[0]["observed_calendar_event_id"] == expected_event_id


@pytest.mark.asyncio
async def test_private_estimate_booking_runs_lifecycle_on_execution_scoped_provider():
    """The execution lock may yield a provider bound to the lock's own
    session connection; every lifecycle step must run through it so one
    booking consumes exactly one pooled connection instead of reserving a
    lock connection plus a transaction connection."""

    class _ScopingCRM(_CRM):
        def __init__(self) -> None:
            super().__init__()
            self.scoped = _CRM()

        @asynccontextmanager
        async def eom_estimate_booking_execution_lock(self, *, booking_key: str):
            self.execution_lock_keys.append(booking_key)
            yield self.scoped

    crm = _ScopingCRM()
    calendar = _Calendar()
    app = _app(crm, _enabled_config(), calendar=calendar)
    contact_id = uuid4()
    booking_key = f"office-booking-{uuid4().hex}"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/estimate-bookings",
            headers=_headers(approval_key=booking_key),
            json=_booking_payload(),
        )

    assert response.status_code == 201
    assert crm.execution_lock_keys == [booking_key]
    assert crm.prepare_calls == []
    assert crm.complete_calls == []
    assert len(crm.scoped.prepare_calls) == 1
    assert len(crm.scoped.complete_calls) == 1


@pytest.mark.asyncio
async def test_database_provider_execution_lock_uses_one_pool_connection():
    """The EOM funnel pool has max_size=5; if the lock reserved one
    connection and the lifecycle steps acquired a second, five concurrent
    bookings would exhaust the pool and deadlock behind their own locks. The
    lock must yield a provider bound to the lock's own connection."""
    from atlas_brain.services.crm_provider import DatabaseCRMProvider

    class _LockConn:
        def __init__(self) -> None:
            self.statements: list[tuple[str, tuple[object, ...]]] = []

        async def fetchval(self, query: str, *args: object):
            self.statements.append((query.strip(), args))
            return True

    class _Pool:
        def __init__(self) -> None:
            self.conn = _LockConn()
            self.acquire_count = 0
            self.released: list[object] = []

        async def acquire(self):
            self.acquire_count += 1
            return self.conn

        async def release(self, conn):
            self.released.append(conn)

    pool = _Pool()
    provider = DatabaseCRMProvider(pool=pool)
    booking_key = f"office-booking-{uuid4().hex}"

    async with provider.eom_estimate_booking_execution_lock(
        booking_key=booking_key
    ) as scoped:
        assert isinstance(scoped, DatabaseCRMProvider)
        assert scoped._get_pool() is pool.conn
        assert pool.acquire_count == 1

    assert pool.acquire_count == 1
    assert pool.released == [pool.conn]
    assert "pg_try_advisory_lock" in pool.conn.statements[0][0]
    assert "pg_advisory_unlock" in pool.conn.statements[-1][0]


@pytest.mark.asyncio
async def test_private_estimate_booking_reuses_prepared_calendar_snapshot():
    class _SnapshotCRM(_CRM):
        async def prepare_eom_estimate_booking(self, **kwargs):
            prepared = await super().prepare_eom_estimate_booking(**kwargs)
            prepared["contact"] = {
                "full_name": "Mutated Lead",
                "address": "999 Changed Ave",
            }
            prepared["calendar_event"] = {
                "summary": "Estimate: Original Lead",
                "location": "100 Original St",
                "description": (
                    "Scheduled from the private EOM lead funnel.\n\n"
                    "Original prepared note"
                ),
                "calendar_id": "prepared-calendar",
                "event_id": kwargs["expected_calendar_event_id"],
            }
            return prepared

    crm = _SnapshotCRM()
    calendar = _Calendar()
    app = _app(crm, _enabled_config(), calendar=calendar)
    contact_id = uuid4()
    booking_key = f"office-booking-{uuid4().hex}"
    expected_event_id = deterministic_eom_estimate_calendar_event_id(
        contact_id=str(contact_id),
        booking_key=booking_key,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/estimate-bookings",
            headers=_headers(approval_key=booking_key),
            json=_booking_payload(),
        )

    assert response.status_code == 201
    assert calendar.calls == [
        {
            "summary": "Estimate: Original Lead",
            "start": datetime.fromisoformat("2026-08-04T14:00:00-05:00"),
            "end": datetime.fromisoformat("2026-08-04T15:00:00-05:00"),
            "location": "100 Original St",
            "description": (
                "Scheduled from the private EOM lead funnel.\n\n"
                "Original prepared note"
            ),
            "calendar_id": "prepared-calendar",
            "event_id": expected_event_id,
        }
    ]
    assert crm.complete_calls[0]["calendar_event_id"] == expected_event_id
    assert crm.ambiguous_calls == []
    assert crm.failed_calls == []


@pytest.mark.asyncio
async def test_private_estimate_booking_indeterminate_calendar_failure_marks_ambiguous():
    crm = _CRM()
    calendar = _Calendar(success=False)
    app = _app(crm, _enabled_config(), calendar=calendar)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{uuid4()}/estimate-bookings",
            headers=_headers(approval_key=f"office-booking-{uuid4().hex}"),
            json=_booking_payload(),
        )

    assert response.status_code == 502
    assert response.json()["detail"] == "Calendar API error: 503"
    assert len(crm.prepare_calls) == 1
    assert len(calendar.calls) == 1
    assert crm.complete_calls == []
    assert crm.failed_calls == []
    assert crm.ambiguous_calls == [
        {
            "contact_id": crm.prepare_calls[0]["contact_id"],
            "booking_key": crm.prepare_calls[0]["booking_key"],
            "expected_calendar_event_id": crm.prepare_calls[0][
                "expected_calendar_event_id"
            ],
            "observed_calendar_event_id": "",
            "actor_id": 1,
            "actor_name": "Juan Canfield",
        }
    ]


@pytest.mark.asyncio
async def test_private_estimate_booking_definitive_calendar_failure_does_not_complete_crm():
    crm = _CRM()
    calendar = _Calendar(
        success=False,
        error="API_ERROR",
        message="Calendar API error: 404",
        data={"status_code": 404},
    )
    app = _app(crm, _enabled_config(), calendar=calendar)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{uuid4()}/estimate-bookings",
            headers=_headers(approval_key=f"office-booking-{uuid4().hex}"),
            json=_booking_payload(),
        )

    assert response.status_code == 502
    assert response.json()["detail"] == "Calendar API error: 404"
    assert len(crm.prepare_calls) == 1
    assert len(calendar.calls) == 1
    assert crm.complete_calls == []
    assert crm.ambiguous_calls == []
    assert crm.failed_calls == [
        {
            "contact_id": crm.prepare_calls[0]["contact_id"],
            "booking_key": crm.prepare_calls[0]["booking_key"],
            "expected_calendar_event_id": crm.prepare_calls[0][
                "expected_calendar_event_id"
            ],
            "calendar_error": "API_ERROR",
            "calendar_message": "Calendar API error: 404",
            "actor_id": 1,
            "actor_name": "Juan Canfield",
        }
    ]


@pytest.mark.asyncio
async def test_private_estimate_booking_post_conflict_auth_failure_marks_ambiguous():
    crm = _CRM()
    contact_id = uuid4()
    booking_key = f"office-booking-{uuid4().hex}"
    expected_event_id = deterministic_eom_estimate_calendar_event_id(
        contact_id=str(contact_id),
        booking_key=booking_key,
    )
    calendar = _Calendar(
        success=False,
        error="AUTH_ERROR",
        data={"request_phase": "conflict_verification"},
        message="Calendar authentication failed. Refresh token needs renewal.",
    )
    app = _app(crm, _enabled_config(), calendar=calendar)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/estimate-bookings",
            headers=_headers(approval_key=booking_key),
            json=_booking_payload(),
        )

    assert response.status_code == 502
    assert crm.complete_calls == []
    assert crm.failed_calls == []
    assert crm.ambiguous_calls == [
        {
            "contact_id": str(contact_id),
            "booking_key": booking_key,
            "expected_calendar_event_id": expected_event_id,
            "observed_calendar_event_id": "",
            "actor_id": 1,
            "actor_name": "Juan Canfield",
        }
    ]


@pytest.mark.asyncio
async def test_private_estimate_booking_unexpected_calendar_id_marks_ambiguous():
    crm = _CRM()
    calendar = _Calendar(event_id="surprise-calendar-event")
    app = _app(crm, _enabled_config(), calendar=calendar)
    contact_id = uuid4()
    booking_key = f"office-booking-{uuid4().hex}"
    expected_event_id = deterministic_eom_estimate_calendar_event_id(
        contact_id=str(contact_id),
        booking_key=booking_key,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/estimate-bookings",
            headers=_headers(approval_key=booking_key),
            json=_booking_payload(),
        )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Calendar returned an unexpected event id; booking requires reconciliation"
    )
    assert crm.complete_calls == []
    assert crm.failed_calls == []
    assert crm.ambiguous_calls == [
        {
            "contact_id": str(contact_id),
            "booking_key": booking_key,
            "expected_calendar_event_id": expected_event_id,
            "observed_calendar_event_id": "surprise-calendar-event",
            "actor_id": 1,
            "actor_name": "Juan Canfield",
        }
    ]


@pytest.mark.asyncio
async def test_private_estimate_booking_idempotent_replay_skips_calendar_side_effect():
    class _BookedCRM(_CRM):
        async def prepare_eom_estimate_booking(self, **kwargs):
            self.prepare_calls.append(kwargs)
            return {
                "contact_id": UUID(kwargs["contact_id"]),
                "lead_stage": "estimate_booked",
                "status": "estimate_booked",
                "calendar_event_id": kwargs["expected_calendar_event_id"],
                "expected_calendar_event_id": kwargs["expected_calendar_event_id"],
                "idempotent": True,
                "contact": {
                    "full_name": "Review Queue Lead",
                    "address": "100 Main St",
                },
            }

    crm = _BookedCRM()
    calendar = _Calendar()
    app = _app(crm, _enabled_config(), calendar=calendar)
    contact_id = uuid4()
    booking_key = f"office-booking-{uuid4().hex}"
    expected_event_id = deterministic_eom_estimate_calendar_event_id(
        contact_id=str(contact_id),
        booking_key=booking_key,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/estimate-bookings",
            headers=_headers(approval_key=booking_key),
            json=_booking_payload(),
        )

    assert response.status_code == 200
    assert response.json()["contact_id"] == str(contact_id)
    assert response.json()["idempotent"] is True
    assert response.json()["calendar_event_id"] == expected_event_id
    assert len(crm.prepare_calls) == 1
    assert calendar.calls == []
    assert crm.complete_calls == []
    assert crm.failed_calls == []


@pytest.mark.asyncio
async def test_private_estimate_booking_provider_conflict_skips_calendar_side_effect():
    class _RejectingBookingCRM(_CRM):
        async def prepare_eom_estimate_booking(self, **kwargs):
            self.prepare_calls.append(kwargs)
            raise EOMLeadConversionError(
                409,
                "Booking key already belongs to a different estimate booking",
            )

    crm = _RejectingBookingCRM()
    calendar = _Calendar()
    app = _app(crm, _enabled_config(), calendar=calendar)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{uuid4()}/estimate-bookings",
            headers=_headers(approval_key=f"office-booking-{uuid4().hex}"),
            json=_booking_payload(),
        )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "Booking key already belongs to a different estimate booking"
    )
    assert len(crm.prepare_calls) == 1
    assert calendar.calls == []
    assert crm.complete_calls == []
    assert crm.failed_calls == []


@pytest.mark.asyncio
async def test_private_estimate_booking_calendar_conflict_marks_ambiguous():
    crm = _CRM()
    contact_id = uuid4()
    booking_key = f"office-booking-{uuid4().hex}"
    expected_event_id = deterministic_eom_estimate_calendar_event_id(
        contact_id=str(contact_id),
        booking_key=booking_key,
    )
    calendar = _Calendar(
        success=False,
        error="IDEMPOTENCY_CONFLICT",
        data={"event_id": expected_event_id},
        message=(
            "Existing calendar event does not match requested event; "
            "booking requires reconciliation"
        ),
    )
    app = _app(crm, _enabled_config(), calendar=calendar)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/estimate-bookings",
            headers=_headers(approval_key=booking_key),
            json=_booking_payload(),
        )

    assert response.status_code == 502
    assert crm.complete_calls == []
    assert crm.failed_calls == []
    assert crm.ambiguous_calls == [
        {
            "contact_id": str(contact_id),
            "booking_key": booking_key,
            "expected_calendar_event_id": expected_event_id,
            "observed_calendar_event_id": expected_event_id,
            "actor_id": 1,
            "actor_name": "Juan Canfield",
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    (
        _booking_payload(scheduled_start="2026-08-04T14:00:00"),
        _booking_payload(scheduled_end="2026-08-04T15:00:00"),
        _booking_payload(scheduled_end="2026-08-04T14:00:00-05:00"),
        _booking_payload(per_visit_rate=150),
    ),
)
async def test_private_estimate_booking_rejects_bad_body_before_side_effects(
    body: dict[str, object],
):
    crm = _CRM()
    calendar = _Calendar()
    app = _app(crm, _enabled_config(), calendar=calendar)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{uuid4()}/estimate-bookings",
            headers=_headers(approval_key=f"office-booking-{uuid4().hex}"),
            json=body,
        )

    assert response.status_code == 422
    assert crm.prepare_calls == []
    assert calendar.calls == []
    assert crm.complete_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "headers", "expected_status"),
    (
        (EOMFunnelConfig(api_enabled=False), _headers(), 503),
        (_enabled_config(), {**_headers(), "Authorization": ""}, 401),
        (_enabled_config(), {**_headers(), "Authorization": "Basic tracker"}, 401),
        (_enabled_config(), {**_headers(), "Idempotency-Key": "short"}, 422),
        (_enabled_config(), {**_headers(actor_id="not-an-id")}, 422),
        (_enabled_config(), {**_headers(actor_id="0")}, 422),
        (_enabled_config(), {**_headers(actor_id="-1")}, 422),
    ),
)
async def test_private_estimate_booking_rejects_http_guards_before_side_effects(
    config: EOMFunnelConfig,
    headers: dict[str, str],
    expected_status: int,
):
    crm = _CRM()
    calendar = _Calendar()
    app = _app(crm, config, calendar=calendar)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{uuid4()}/estimate-bookings",
            headers=headers,
            json=_booking_payload(),
        )

    assert response.status_code == expected_status
    assert crm.prepare_calls == []
    assert calendar.calls == []
    assert crm.complete_calls == []


@pytest.mark.asyncio
async def test_calendar_create_event_sends_optional_deterministic_event_id(monkeypatch):
    tool = CalendarTool()
    tool._config = SimpleNamespace(
        calendar_enabled=True, calendar_refresh_token="refresh"
    )
    client = _CalendarClient(
        post_response=_CalendarResponse(
            status_code=200,
            payload={"id": "eomestabc123", "summary": "Estimate"},
        )
    )

    async def ensure_client():
        return client

    async def auth_header(**_kwargs):
        return {"Authorization": "Bearer token"}

    monkeypatch.setattr(tool, "_ensure_client", ensure_client)
    monkeypatch.setattr(tool, "_get_auth_header", auth_header)

    result = await tool.create_event(
        summary="Estimate",
        start=datetime.fromisoformat("2026-08-04T14:00:00-05:00"),
        end=datetime.fromisoformat("2026-08-04T15:00:00-05:00"),
        calendar_id="estimate-calendar",
        event_id="eomestabc123",
    )

    assert result.success is True
    assert result.data["event_id"] == "eomestabc123"
    assert client.post_calls[0]["json"]["id"] == "eomestabc123"


@pytest.mark.asyncio
async def test_calendar_create_event_propagates_http_status_for_failure_classifier(
    monkeypatch,
):
    tool = CalendarTool()
    tool._config = SimpleNamespace(
        calendar_enabled=True, calendar_refresh_token="refresh"
    )
    client = _CalendarClient(
        post_response=_CalendarResponse(
            status_code=404,
            payload={"error": "calendar not found"},
        )
    )

    async def ensure_client():
        return client

    async def auth_header(**_kwargs):
        return {"Authorization": "Bearer token"}

    monkeypatch.setattr(tool, "_ensure_client", ensure_client)
    monkeypatch.setattr(tool, "_get_auth_header", auth_header)

    result = await tool.create_event(
        summary="Estimate",
        start=datetime.fromisoformat("2026-08-04T14:00:00-05:00"),
        end=datetime.fromisoformat("2026-08-04T15:00:00-05:00"),
        calendar_id="missing-calendar",
        event_id="eomestabc123",
    )

    assert result.success is False
    assert result.error == "API_ERROR"
    assert result.data == {"request_phase": "create", "status_code": 404}
    assert result.message == "Calendar API error: 404"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raised, expected_error",
    [
        ("http_503", "API_ERROR"),
        ("timeout", "EXECUTION_ERROR"),
        ("auth", "AUTH_ERROR"),
    ],
)
async def test_calendar_create_event_marks_auth_phase_before_any_event_request(
    monkeypatch, raised, expected_error
):
    """A token-endpoint failure happens before any Google Calendar event
    request, so create_event must surface request_phase='auth' for the
    booking classifier to prove no event write exists."""
    tool = CalendarTool()
    tool._config = SimpleNamespace(
        calendar_enabled=True, calendar_refresh_token="refresh"
    )
    client = _CalendarClient(
        post_response=_CalendarResponse(status_code=200, payload={"id": "unused"})
    )

    async def ensure_client():
        return client

    async def auth_header(**_kwargs):
        if raised == "http_503":
            raise httpx.HTTPStatusError(
                "token endpoint error",
                request=httpx.Request("POST", "https://oauth2.example/token"),
                response=httpx.Response(503),
            )
        if raised == "timeout":
            raise RuntimeError("token endpoint timed out")
        raise CalendarAuthError("refresh token rejected")

    monkeypatch.setattr(tool, "_ensure_client", ensure_client)
    monkeypatch.setattr(tool, "_get_auth_header", auth_header)

    result = await tool.create_event(
        summary="Estimate",
        start=datetime.fromisoformat("2026-08-04T14:00:00-05:00"),
        end=datetime.fromisoformat("2026-08-04T15:00:00-05:00"),
        calendar_id="estimate-calendar",
        event_id="eomestabc123",
    )

    assert result.success is False
    assert result.error == expected_error
    assert result.data["request_phase"] == "auth"
    assert client.post_calls == []


@pytest.mark.asyncio
async def test_calendar_create_event_marks_auth_error_after_conflict_verification(
    monkeypatch,
):
    tool = CalendarTool()
    tool._config = SimpleNamespace(
        calendar_enabled=True, calendar_refresh_token="refresh"
    )
    client = _CalendarClient(
        post_response=_CalendarResponse(
            status_code=409, payload={"error": "duplicate"}
        ),
        get_response=_CalendarResponse(
            status_code=401,
            payload={"error": "unauthorized"},
        ),
    )

    async def ensure_client():
        return client

    async def auth_header(**kwargs):
        if kwargs.get("force_refresh"):
            raise CalendarAuthError("refresh failed")
        return {"Authorization": "Bearer token"}

    monkeypatch.setattr(tool, "_ensure_client", ensure_client)
    monkeypatch.setattr(tool, "_get_auth_header", auth_header)

    result = await tool.create_event(
        summary="Estimate",
        start=datetime.fromisoformat("2026-08-04T14:00:00-05:00"),
        end=datetime.fromisoformat("2026-08-04T15:00:00-05:00"),
        calendar_id="estimate-calendar",
        event_id="eomestabc123",
    )

    assert result.success is False
    assert result.error == "AUTH_ERROR"
    assert result.data == {"request_phase": "conflict_verification"}


@pytest.mark.asyncio
async def test_calendar_create_event_reuses_existing_deterministic_event_id(
    monkeypatch,
):
    tool = CalendarTool()
    tool._config = SimpleNamespace(
        calendar_enabled=True, calendar_refresh_token="refresh"
    )
    client = _CalendarClient(
        post_response=_CalendarResponse(
            status_code=409, payload={"error": "duplicate"}
        ),
        get_response=_CalendarResponse(
            status_code=200,
            payload={
                "id": "eomestabc123",
                "summary": "Estimate",
                "start": {"dateTime": "2026-08-04T14:00:00-05:00"},
                "end": {"dateTime": "2026-08-04T15:00:00-05:00"},
            },
        ),
    )

    async def ensure_client():
        return client

    async def auth_header(**_kwargs):
        return {"Authorization": "Bearer token"}

    monkeypatch.setattr(tool, "_ensure_client", ensure_client)
    monkeypatch.setattr(tool, "_get_auth_header", auth_header)

    result = await tool.create_event(
        summary="Estimate",
        start=datetime.fromisoformat("2026-08-04T14:00:00-05:00"),
        end=datetime.fromisoformat("2026-08-04T15:00:00-05:00"),
        calendar_id="estimate-calendar",
        event_id="eomestabc123",
    )

    assert result.success is True
    assert result.data["event_id"] == "eomestabc123"
    assert client.post_calls[0]["json"]["id"] == "eomestabc123"
    assert client.get_calls == [
        {
            "url": (
                "https://www.googleapis.com/calendar/v3/calendars/"
                "estimate-calendar/events/eomestabc123"
            ),
            "headers": {
                "Authorization": "Bearer token",
                "Content-Type": "application/json",
            },
        }
    ]


@pytest.mark.asyncio
async def test_calendar_create_event_rejects_changed_existing_deterministic_event(
    monkeypatch,
):
    tool = CalendarTool()
    tool._config = SimpleNamespace(
        calendar_enabled=True, calendar_refresh_token="refresh"
    )
    client = _CalendarClient(
        post_response=_CalendarResponse(
            status_code=409, payload={"error": "duplicate"}
        ),
        get_response=_CalendarResponse(
            status_code=200,
            payload={
                "id": "eomestabc123",
                "summary": "Estimate",
                "start": {"dateTime": "2026-08-04T14:30:00-05:00"},
                "end": {"dateTime": "2026-08-04T15:30:00-05:00"},
            },
        ),
    )

    async def ensure_client():
        return client

    async def auth_header(**_kwargs):
        return {"Authorization": "Bearer token"}

    monkeypatch.setattr(tool, "_ensure_client", ensure_client)
    monkeypatch.setattr(tool, "_get_auth_header", auth_header)

    result = await tool.create_event(
        summary="Estimate",
        start=datetime.fromisoformat("2026-08-04T14:00:00-05:00"),
        end=datetime.fromisoformat("2026-08-04T15:00:00-05:00"),
        calendar_id="estimate-calendar",
        event_id="eomestabc123",
    )

    assert result.success is False
    assert result.error == "IDEMPOTENCY_CONFLICT"
    assert result.data == {"event_id": "eomestabc123", "status": None}


@pytest.mark.asyncio
async def test_enabled_full_atlas_funnel_requires_authoritative_data_store(monkeypatch):
    from atlas_brain import main

    class _Pool:
        def __init__(self, *, initialized: bool, schema_ready: bool) -> None:
            self.is_initialized = initialized
            self._schema_ready = schema_ready
            self.queries: list[str] = []

        async def fetchval(self, query: str) -> bool:
            self.queries.append(query)
            return self._schema_ready

    disabled = SimpleNamespace(api_enabled=False)

    def fail_if_looked_up():
        pytest.fail("disabled EOM funnel must not require a database pool")

    monkeypatch.setattr(main, "get_db_pool", fail_if_looked_up)
    await main._require_eom_funnel_data_store(disabled, database_enabled=False)

    enabled = SimpleNamespace(api_enabled=True)
    ready_pool = _Pool(initialized=True, schema_ready=True)
    monkeypatch.setattr(main, "get_db_pool", lambda: ready_pool)

    await main._require_eom_funnel_data_store(enabled, database_enabled=True)
    assert "atlas_eom_handoff_owner" in ready_pool.queries[0]
    assert "atlas_nocodb" in ready_pool.queries[0]
    assert "pg_auth_members" in ready_pool.queries[0]
    assert "rolcanlogin" in ready_pool.queries[0]
    assert "rolinherit" in ready_pool.queries[0]
    assert "rolsuper" in ready_pool.queries[0]
    assert "has_database_privilege" in ready_pool.queries[0]
    assert "has_schema_privilege" in ready_pool.queries[0]
    assert "has_table_privilege" in ready_pool.queries[0]
    assert "has_column_privilege" in ready_pool.queries[0]
    assert "pg_tables" in ready_pool.queries[0]

    with pytest.raises(RuntimeError, match="authoritative Atlas database"):
        await main._require_eom_funnel_data_store(enabled, database_enabled=False)

    monkeypatch.setattr(
        main,
        "get_db_pool",
        lambda: _Pool(initialized=False, schema_ready=True),
    )
    with pytest.raises(RuntimeError, match="initialized Atlas database pool"):
        await main._require_eom_funnel_data_store(
            enabled,
            database_enabled=True,
        )

    monkeypatch.setattr(
        main,
        "get_db_pool",
        lambda: _Pool(initialized=True, schema_ready=False),
    )
    with pytest.raises(RuntimeError, match="CRM lifecycle and handoff schema"):
        await main._require_eom_funnel_data_store(
            enabled,
            database_enabled=True,
        )


@pytest.mark.asyncio
async def test_full_app_lifespan_executes_enabled_preflight_before_handoff_request(
    monkeypatch,
):
    """The configured full-app lifespan gates the authenticated callback."""
    from atlas_brain import main
    from atlas_brain.eom_api import config as config_mod

    class _Pool:
        def __init__(self, *, initialized: bool) -> None:
            self.is_initialized = initialized

        async def fetchval(self, _query: str) -> bool:
            return True

    async def no_op(*_args, **_kwargs):
        return None

    runtime_settings = main.settings.model_copy(deep=True)
    runtime_settings.load_llm_on_startup = False
    runtime_settings.llm.model_swap_enabled = False
    runtime_settings.llm.cloud_enabled = False
    runtime_settings.intent_router.llm_fallback_enabled = False
    runtime_settings.email_draft.enabled = False
    runtime_settings.email_draft.triage_enabled = False
    runtime_settings.reasoning.enabled = False
    runtime_settings.discovery.enabled = False
    runtime_settings.alerts.enabled = False
    runtime_settings.reminder.enabled = False
    runtime_settings.autonomous.enabled = False
    runtime_settings.mqtt.enabled = False
    runtime_settings.tools.calendar_enabled = False
    runtime_settings.mcp.client_enabled = False
    runtime_settings.voice.enabled = False
    config = _enabled_config()
    monkeypatch.setattr(config_mod, "funnel_settings", config)
    monkeypatch.setattr(auth_mod, "funnel_settings", config)
    monkeypatch.setattr(main, "settings", runtime_settings)
    monkeypatch.setattr(main, "db_settings", SimpleNamespace(enabled=True))
    pools = iter((_Pool(initialized=False), _Pool(initialized=True)))
    monkeypatch.setattr(main, "get_db_pool", lambda: next(pools))
    monkeypatch.setattr(main, "init_database", no_op)
    monkeypatch.setattr(main, "close_database", no_op)
    monkeypatch.setattr(main.llm_registry, "deactivate", lambda: None)

    preflight_calls: list[str] = []
    original_preflight = main._validate_eom_funnel_startup

    async def preflight_spy():
        preflight_calls.append("enabled")
        await original_preflight()

    monkeypatch.setattr(main, "_validate_eom_funnel_startup", preflight_spy)

    app = main.app
    crm = _CRM()
    original_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[funnel_mod._crm_dependency] = lambda: crm
    app.dependency_overrides[auth_mod.get_eom_funnel_api_config] = _enabled_config
    try:
        async with app.router.lifespan_context(app):
            assert preflight_calls == ["enabled"]
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/api/v1/eom-funnel/customer-handoffs",
                    headers=_headers(),
                    json={
                        "contact_id": "11111111-1111-1111-1111-111111111111",
                        "tracker_customer_id": 12,
                        "tracker_site_id": 24,
                    },
                )
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)

    assert response.status_code == 201
    assert preflight_calls == ["enabled"]
    assert crm.calls


@pytest.mark.asyncio
async def test_private_handoff_accepts_only_ids_and_actor_evidence():
    crm = _CRM()
    app = _app(crm, _enabled_config())
    approval_key = _approval_key()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/eom-funnel/customer-handoffs",
            headers=_headers(approval_key=approval_key),
            json={
                "contact_id": "11111111-1111-1111-1111-111111111111",
                "tracker_customer_id": 12,
                "tracker_site_id": 24,
            },
        )

    assert response.status_code == 201
    assert response.json()["success"] is True
    assert crm.calls == [
        {
            "contact_id": "11111111-1111-1111-1111-111111111111",
            "tracker_customer_id": 12,
            "tracker_site_id": 24,
            "approval_key": approval_key,
            "actor_id": 1,
            "actor_name": "Juan Canfield",
        }
    ]


@pytest.mark.asyncio
async def test_private_handoff_rejects_operational_estimate_fields_before_crm_call():
    crm = _CRM()
    app = _app(crm, _enabled_config())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/eom-funnel/customer-handoffs",
            headers=_headers(),
            json={
                "contact_id": "11111111-1111-1111-1111-111111111111",
                "tracker_customer_id": 12,
                "tracker_site_id": 24,
                "per_visit_rate": 150,
            },
        )

    assert response.status_code == 422
    assert crm.calls == []


@pytest.mark.asyncio
async def test_private_handoff_rejects_bad_service_token_before_crm_call():
    crm = _CRM()
    app = _app(crm, _enabled_config())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/eom-funnel/customer-handoffs",
            headers=_headers("y" * 24),
            json={
                "contact_id": "11111111-1111-1111-1111-111111111111",
                "tracker_customer_id": 12,
                "tracker_site_id": 24,
            },
        )

    assert response.status_code == 401
    assert crm.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "headers", "expected_status"),
    (
        (EOMFunnelConfig(api_enabled=False), _headers(), 503),
        (_enabled_config(), {**_headers(), "Authorization": ""}, 401),
        (_enabled_config(), {**_headers(), "Authorization": "Basic tracker"}, 401),
        (_enabled_config(), {**_headers(), "Idempotency-Key": "short"}, 422),
        (_enabled_config(), {**_headers(actor_id="not-an-id")}, 422),
        (_enabled_config(), {**_headers(actor_id="0")}, 422),
        (_enabled_config(), {**_headers(actor_id="-1")}, 422),
    ),
)
async def test_private_handoff_rejects_each_http_boundary_guard_before_crm_call(
    config: EOMFunnelConfig,
    headers: dict[str, str],
    expected_status: int,
):
    crm = _CRM()
    app = _app(crm, config)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/eom-funnel/customer-handoffs",
            headers=headers,
            json={
                "contact_id": "11111111-1111-1111-1111-111111111111",
                "tracker_customer_id": 12,
                "tracker_site_id": 24,
            },
        )

    assert response.status_code == expected_status
    assert crm.calls == []


@pytest.mark.asyncio
async def test_private_handoff_rejects_non_ascii_bearer_before_crm_call():
    crm = _CRM()
    app = _app(crm, _enabled_config())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/eom-funnel/customer-handoffs",
            headers=[
                (b"authorization", b"Bearer \xff"),
                (b"x-eom-actor", b"Juan Canfield"),
                (b"x-eom-actor-id", b"1"),
                (b"idempotency-key", _approval_key().encode("ascii")),
            ],
            json={
                "contact_id": "11111111-1111-1111-1111-111111111111",
                "tracker_customer_id": 12,
                "tracker_site_id": 24,
            },
        )

    assert response.status_code == 401
    assert crm.calls == []


@pytest.mark.parametrize(
    "token_digest",
    (
        "",
        "f" * 63,
        "f" * 65,
        "F" * 64,
        "x" * 24,
        "eomf_v1_" + ("x" * 43),
        "eomf_v1_" + ("a" * 42),
        "eomf_v1_" + ("*" * 43),
        "eomrx_v1_" + "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789_-ABC",
        "eomf_v1_" + (("abc123" * 8)[:43]),
        "\u00e9" * 43,
    ),
)
def test_enabled_funnel_rejects_missing_or_malformed_token_digests_at_startup(
    token_digest: str,
):
    with pytest.raises(RuntimeError, match="digest|required|hex|placeholder"):
        auth_mod.validate_eom_funnel_api_config(
            EOMFunnelConfig(api_enabled=True, service_token_sha256=token_digest)
        )


def test_enabled_funnel_accepts_a_fresh_generated_service_token_at_startup():
    generated = auth_mod.generate_eom_funnel_service_token()
    auth_mod.validate_eom_funnel_api_config(
        EOMFunnelConfig(
            api_enabled=True,
            service_token_sha256=generated.sha256,
        )
    )
    assert auth_mod.eom_funnel_service_token_sha256(generated.token) == generated.sha256


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ("tracker_customer_id", "tracker_site_id"))
async def test_private_handoff_rejects_storage_overflow_before_crm_call(field: str):
    crm = _CRM()
    app = _app(crm, _enabled_config())
    body = {
        "contact_id": "11111111-1111-1111-1111-111111111111",
        "tracker_customer_id": 12,
        "tracker_site_id": 24,
    }
    body[field] = 2**63
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/eom-funnel/customer-handoffs",
            headers=_headers(),
            json=body,
        )

    assert response.status_code == 422
    assert crm.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    (
        _headers(actor=" "),
        _headers(actor="x" * 100),
        _headers(actor_id=str(2**63)),
    ),
)
async def test_private_handoff_rejects_invalid_actor_evidence_before_crm_call(
    headers: dict[str, str],
):
    crm = _CRM()
    app = _app(crm, _enabled_config())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/eom-funnel/customer-handoffs",
            headers=headers,
            json={
                "contact_id": "11111111-1111-1111-1111-111111111111",
                "tracker_customer_id": 12,
                "tracker_site_id": 24,
            },
        )

    assert response.status_code == 422
    assert crm.calls == []


@pytest.mark.asyncio
async def test_private_handoff_preserves_provider_rejection_without_side_effect_claim():
    class _RejectingCRM(_CRM):
        async def finalize_eom_customer_handoff(self, **kwargs):
            self.calls.append(kwargs)
            raise EOMLeadConversionError(409, "EOM lead is not ready for approval")

    crm = _RejectingCRM()
    app = _app(crm, _enabled_config())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/eom-funnel/customer-handoffs",
            headers=_headers(),
            json={
                "contact_id": "11111111-1111-1111-1111-111111111111",
                "tracker_customer_id": 12,
                "tracker_site_id": 24,
            },
        )

    assert response.status_code == 409
    assert response.json()["detail"] == "EOM lead is not ready for approval"
    assert len(crm.calls) == 1
