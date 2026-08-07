"""HTTP boundary proof for the private EOM office conversion API."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
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
    deterministic_eom_first_clean_calendar_event_id,
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
        self.first_clean_prepare_calls: list[dict[str, object]] = []
        self.first_clean_complete_calls: list[dict[str, object]] = []
        self.first_clean_ambiguous_calls: list[dict[str, object]] = []
        self.first_clean_failed_calls: list[dict[str, object]] = []
        self.onboarding_draft_id = "0b8db22e-16b1-4a30-a15f-6c78ee9204a5"
        self.execution_lock_keys: list[str] = []
        self.review_leads = review_leads or []
        self.draft_rows: dict[str, dict[str, object]] = {}
        self.draft_list_rows: list[dict[str, object]] = []
        self.draft_list_calls: list[dict[str, object]] = []
        self.draft_update_calls: list[dict[str, object]] = []
        self.draft_claim_calls: list[dict[str, object]] = []
        self.draft_confirm_calls: list[dict[str, object]] = []
        self.draft_revoke_calls: list[dict[str, object]] = []
        self.interaction_logs: list[dict[str, object]] = []
        self.lost_calls: list[dict[str, object]] = []
        self.reopen_calls: list[dict[str, object]] = []
        self.reopen_stage = "new"
        self.operator_contact_calls: list[object] = []
        self.operator_contact_result: dict[str, object] | None = None

    @asynccontextmanager
    async def eom_estimate_booking_execution_lock(self, *, booking_key: str):
        self.execution_lock_keys.append(booking_key)
        yield

    async def mutate_eom_operator_contact_atomic(self, *, command):
        self.operator_contact_calls.append(command)
        if self.operator_contact_result is not None:
            return self.operator_contact_result
        contact_id = command.contact_id or "4c0f1ee8-8063-4ba9-9e10-847e6c2a95ff"
        operation = (
            "contact_updated" if command.contact_id is not None else "contact_created"
        )
        return {
            "contact_id": contact_id,
            "operation": operation,
            "idempotent": False,
            "contact": {
                "id": contact_id,
                "full_name": command.fields.get("full_name") or "Stored Contact",
                "email": command.fields.get("email"),
                "phone": command.fields.get("phone"),
                "address": command.fields.get("address"),
                "city": command.fields.get("city"),
                "state": command.fields.get("state"),
                "zip": command.fields.get("zip"),
                "notes": command.fields.get("notes"),
                "contact_type": command.contact_type or "customer",
                "lead_stage": "new" if command.contact_type == "lead" else None,
                "status": "active",
                "source": command.contact_source,
                "source_ref": command.contact_source_ref,
                "created_at": datetime(2026, 8, 6, 12, tzinfo=timezone.utc),
                "updated_at": datetime(2026, 8, 6, 12, tzinfo=timezone.utc),
            },
        }

    async def mark_eom_lead_lost(self, **kwargs: object) -> dict[str, object]:
        self.lost_calls.append(kwargs)
        return {
            "contact_id": kwargs["contact_id"],
            "lead_stage": "lost",
            "status": "lost",
            "reason_code": kwargs["reason_code"],
            "from_stage": "new",
            "idempotent": False,
        }

    async def reopen_eom_lead(self, **kwargs: object) -> dict[str, object]:
        self.reopen_calls.append(kwargs)
        return {
            "contact_id": kwargs["contact_id"],
            "lead_stage": self.reopen_stage,
            "status": "active",
            "idempotent": False,
        }

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

    async def prepare_eom_first_clean_booking(self, **kwargs):
        self.first_clean_prepare_calls.append(kwargs)
        return {
            "contact_id": kwargs["contact_id"],
            "lead_stage": "estimate_booked",
            "status": "calendar_pending",
            "calendar_event_id": None,
            "expected_calendar_event_id": kwargs["expected_calendar_event_id"],
            "idempotent": False,
            "contact": {
                "full_name": "Review Queue Lead",
                "address": "100 Main St",
            },
        }

    async def complete_eom_first_clean_booking(self, **kwargs):
        self.first_clean_complete_calls.append(kwargs)
        return {
            "contact_id": kwargs["contact_id"],
            "lead_stage": "won",
            "status": "first_clean_booked",
            "calendar_event_id": kwargs["calendar_event_id"],
            "expected_calendar_event_id": kwargs["expected_calendar_event_id"],
            "idempotent": False,
            "onboarding_draft_id": self.onboarding_draft_id,
        }

    async def mark_eom_first_clean_booking_calendar_ambiguous(self, **kwargs):
        self.first_clean_ambiguous_calls.append(kwargs)

    async def mark_eom_first_clean_booking_calendar_failed(self, **kwargs):
        self.first_clean_failed_calls.append(kwargs)

    # -- Onboarding draft fake: mirrors the provider's migration-360 state
    # -- machine in memory so route+service behavior is provable without
    # -- Postgres; the SQL itself is proven by the integration suite.

    @staticmethod
    def _draft_closed(row: dict[str, object], *, idempotent: bool = False):
        def _iso(value):
            return value.isoformat() if value is not None else None

        return {
            "draft_id": str(row["id"]),
            "contact_id": str(row["contact_id"]),
            "status": str(row["status"]),
            "recipient_email": row["recipient_email"],
            "blocker": row["blocker"],
            "subject": str(row["subject"]),
            "body": str(row["body"]),
            "created_at": _iso(row["created_at"]),
            "claimed_at": _iso(row["claimed_at"]),
            "sent_at": _iso(row["sent_at"]),
            "revoked_at": _iso(row["revoked_at"]),
            "approved_by_name": row["approved_by_name"],
            "idempotent": idempotent,
        }

    def seed_draft(self, **overrides) -> dict[str, object]:
        row: dict[str, object] = {
            "id": uuid4(),
            "contact_id": uuid4(),
            "full_name": "Review Queue Lead",
            "recipient_email": "lead@example.com",
            "blocker": None,
            "subject": "Welcome aboard - Effingham Office Maids",
            "body": "Hi Review Queue Lead,\n\nWelcome to the team.",
            "status": "pending",
            "created_at": datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc),
            "claimed_at": None,
            "sent_at": None,
            "revoked_at": None,
            "approved_by_employee_id": None,
            "approved_by_name": None,
            "contact_status": "active",
        }
        row.update(overrides)
        self.draft_rows[str(row["id"])] = row
        return row

    async def list_eom_onboarding_drafts(self, **kwargs):
        self.draft_list_calls.append(kwargs)
        return list(self.draft_list_rows)

    async def get_eom_onboarding_draft(self, draft_id):
        return self.draft_rows.get(str(draft_id))

    async def update_eom_onboarding_draft(
        self, *, draft_id, subject=None, body=None, recipient_email=None
    ):
        self.draft_update_calls.append(
            {
                "draft_id": str(draft_id),
                "subject": subject,
                "body": body,
                "recipient_email": recipient_email,
            }
        )
        row = self.draft_rows.get(str(draft_id))
        if row is None:
            raise EOMLeadConversionError(404, "EOM onboarding draft not found")
        if row["status"] != "pending":
            raise EOMLeadConversionError(
                409,
                "EOM onboarding draft is "
                f"{row['status']}; only pending drafts can be edited",
            )
        if subject is not None:
            row["subject"] = subject
        if body is not None:
            row["body"] = body
        if recipient_email is not None:
            row["recipient_email"] = recipient_email
            row["blocker"] = None
        return self._draft_closed(row)

    async def claim_eom_onboarding_draft(self, *, draft_id, actor_id, actor_name):
        self.draft_claim_calls.append(
            {"draft_id": str(draft_id), "actor_id": actor_id, "actor_name": actor_name}
        )
        row = self.draft_rows.get(str(draft_id))
        if row is None:
            raise EOMLeadConversionError(404, "EOM onboarding draft not found")
        if (
            row["status"] == "pending"
            and not row["blocker"]
            and row["recipient_email"]
            and row.get("contact_status", "active") == "active"
        ):
            row["status"] = "sending"
            row["claimed_at"] = datetime.now(timezone.utc)
            row["approved_by_employee_id"] = actor_id
            row["approved_by_name"] = actor_name
            return {"claimed": True, "draft": self._draft_closed(row)}
        if row["status"] == "sent":
            return {
                "claimed": False,
                "draft": self._draft_closed(row, idempotent=True),
            }
        if row["status"] == "sending":
            raise EOMLeadConversionError(
                409,
                "EOM onboarding draft send is already in flight or requires "
                "reconciliation",
            )
        if row["status"] == "revoked":
            raise EOMLeadConversionError(409, "EOM onboarding draft is revoked")
        if row["blocker"]:
            raise EOMLeadConversionError(
                409, f"EOM onboarding draft is blocked: {row['blocker']}"
            )
        if row["recipient_email"] is None:
            raise EOMLeadConversionError(
                409, "EOM onboarding draft has no recipient email"
            )
        raise EOMLeadConversionError(
            409,
            "EOM onboarding draft contact is not an active "
            "effingham_maids contact",
        )

    @staticmethod
    def _sending_claim_is_stale(row):
        claimed_at = row.get("claimed_at")
        if claimed_at is None:
            return True
        return datetime.now(timezone.utc) - claimed_at >= timedelta(minutes=15)

    async def confirm_eom_onboarding_draft_sent(
        self, *, draft_id, require_stale=False
    ):
        self.draft_confirm_calls.append(
            {"draft_id": str(draft_id), "require_stale": require_stale}
        )
        row = self.draft_rows.get(str(draft_id))
        if row is None:
            raise EOMLeadConversionError(404, "EOM onboarding draft not found")
        if row["status"] == "sending":
            if require_stale and not self._sending_claim_is_stale(row):
                raise EOMLeadConversionError(
                    409,
                    "EOM onboarding draft send is still in flight; reconcile "
                    "only after the claim goes stale",
                )
            row["status"] = "sent"
            row["sent_at"] = datetime.now(timezone.utc)
            return self._draft_closed(row)
        if row["status"] == "sent":
            return self._draft_closed(row, idempotent=True)
        if row["status"] == "revoked":
            raise EOMLeadConversionError(
                409,
                "EOM onboarding draft was revoked while sending; reconcile "
                "against the transport log",
            )
        raise EOMLeadConversionError(
            409, "EOM onboarding draft has not been claimed for sending"
        )

    async def revoke_eom_onboarding_draft(self, *, draft_id):
        self.draft_revoke_calls.append({"draft_id": str(draft_id)})
        row = self.draft_rows.get(str(draft_id))
        if row is None:
            raise EOMLeadConversionError(404, "EOM onboarding draft not found")
        if row["status"] == "pending" or (
            row["status"] == "sending" and self._sending_claim_is_stale(row)
        ):
            row["status"] = "revoked"
            row["revoked_at"] = datetime.now(timezone.utc)
            return self._draft_closed(row)
        if row["status"] == "sending":
            raise EOMLeadConversionError(
                409,
                "EOM onboarding draft send is still in flight; reconcile "
                "only after the claim goes stale",
            )
        if row["status"] == "revoked":
            return self._draft_closed(row, idempotent=True)
        raise EOMLeadConversionError(
            409, "EOM onboarding draft was already sent and cannot be revoked"
        )

    async def log_interaction(self, contact_id, interaction_type, summary):
        self.interaction_logs.append(
            {
                "contact_id": str(contact_id),
                "interaction_type": interaction_type,
                "summary": summary,
            }
        )


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


def _operator_contact_payload(**overrides) -> dict[str, object]:
    payload: dict[str, object] = {
        "sourceChannel": "time_tracker",
        "sourceRef": "customer:42",
        "fullName": "Ada Operator",
        "email": "ADA@EXAMPLE.COM",
        "phone": "(217) 555-0100",
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
        "capabilities": list(funnel_mod.served_capabilities()),
    }
    assert crm.review_calls == [
        {"limit": 101, "cursor_created_at": None, "cursor_contact_id": None}
    ]
    assert response.status_code == 201
    assert crm.calls


@pytest.mark.asyncio
async def test_private_operator_contact_route_normalizes_and_delegates_create():
    crm = _CRM()
    app = _app(crm, _enabled_config())
    operation_key = f"office-contact-{uuid4().hex}"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/eom-funnel/operator-contacts",
            headers=_headers(approval_key=operation_key),
            json=_operator_contact_payload(address="   ", contactType="lead"),
        )

    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True
    assert body["operation"] == "contact_created"
    assert body["idempotent"] is False
    assert body["contact"]["email"] == "ada@example.com"
    assert body["contact"]["phone"] == "2175550100"
    assert body["contact"]["address"] is None
    assert body["contact"]["sourceRef"] == "time_tracker:customer:42"
    command = crm.operator_contact_calls[0]
    assert command.operation_key == operation_key
    assert command.actor_id == 1
    assert command.actor_name == "Juan Canfield"
    assert command.contact_type == "lead"
    assert dict(command.fields) == {
        "full_name": "Ada Operator",
        "email": "ada@example.com",
        "phone": "2175550100",
        "address": None,
    }
    assert len(command.request_fingerprint) == 64


@pytest.mark.asyncio
async def test_private_operator_contact_route_returns_200_for_idempotent_replay():
    contact_id = str(uuid4())
    crm = _CRM()
    crm.operator_contact_result = {
        "contact_id": contact_id,
        "operation": "contact_updated",
        "idempotent": True,
        "contact": {
            "id": contact_id,
            "full_name": "Replay Contact",
            "email": "replay@example.com",
            "phone": "2175550101",
            "address": None,
            "city": None,
            "state": None,
            "zip": None,
            "notes": None,
            "contact_type": "customer",
            "lead_stage": None,
            "status": "active",
            "source": "manual",
            "source_ref": "time_tracker:customer:51",
            "created_at": datetime(2026, 8, 6, 12, tzinfo=timezone.utc),
            "updated_at": datetime(2026, 8, 6, 12, tzinfo=timezone.utc),
        },
    }
    app = _app(crm, _enabled_config())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/eom-funnel/operator-contacts",
            headers=_headers(approval_key=f"office-contact-{uuid4().hex}"),
            json=_operator_contact_payload(
                contactId=contact_id,
                sourceRef="customer:51",
                email="replay@example.com",
            ),
        )

    assert response.status_code == 200
    assert response.json()["idempotent"] is True
    assert response.json()["contactId"] == contact_id
    assert len(crm.operator_contact_calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "headers", "expected_status"),
    (
        (EOMFunnelConfig(api_enabled=False), _headers(), 503),
        (_enabled_config(), {**_headers(), "Authorization": ""}, 401),
        (_enabled_config(), {**_headers(actor=" ")}, 422),
        (_enabled_config(), {**_headers(approval_key="short")}, 422),
    ),
)
async def test_private_operator_contact_rejects_boundary_failures_before_crm_call(
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
            "/eom-funnel/operator-contacts",
            headers=headers,
            json=_operator_contact_payload(),
        )

    assert response.status_code == expected_status
    assert crm.operator_contact_calls == []


@pytest.mark.asyncio
async def test_private_operator_contact_rejects_unknown_source_channel_before_crm_call():
    crm = _CRM()
    app = _app(crm, _enabled_config())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/eom-funnel/operator-contacts",
            headers=_headers(approval_key=f"office-contact-{uuid4().hex}"),
            json=_operator_contact_payload(sourceChannel="random"),
        )

    assert response.status_code == 422
    assert crm.operator_contact_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload_updates",
    (
        {"email": "a@@b"},
        {"email": "a b@example.com"},
        {"email": "ada\noperator@example.com"},
        {"email": f"{'a' * 65}@example.com"},
        {"phone": "٢١٧٥٥٥٠١٠٠"},
        {"phone": "2175550100 ext 123"},
    ),
)
async def test_private_operator_contact_rejects_malformed_identity_before_crm_call(
    payload_updates: dict[str, str],
):
    crm = _CRM()
    app = _app(crm, _enabled_config())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        headers = {
            **_headers(approval_key=f"office-contact-{uuid4().hex}"),
            "Content-Type": "application/json",
        }
        response = await client.post(
            "/eom-funnel/operator-contacts",
            headers=headers,
            content=json.dumps(_operator_contact_payload(**payload_updates)),
        )

    assert response.status_code == 422
    assert crm.operator_contact_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload_updates",
    (
        {"sourceRef": "customer:\x0042"},
        {"sourceRef": "customer:\ud80042"},
        {"fullName": "Ada\x00Operator"},
        {"fullName": "Ada\ud800Operator"},
        {"address": "100\x00Main"},
        {"address": "100\ud800Main"},
        {"city": "Effingham\x00"},
        {"city": "Effingham\ud800"},
        {"state": "I\x00L"},
        {"state": "I\ud800L"},
        {"zip": "624\x0001"},
        {"zip": "624\ud80001"},
        {"notes": "bring\x00supplies"},
        {"notes": "bring\ud800supplies"},
    ),
)
async def test_private_operator_contact_rejects_database_invalid_text_before_crm_call(
    payload_updates: dict[str, str],
):
    crm = _CRM()
    app = _app(crm, _enabled_config())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        headers = {
            **_headers(approval_key=f"office-contact-{uuid4().hex}"),
            "Content-Type": "application/json",
        }
        response = await client.post(
            "/eom-funnel/operator-contacts",
            headers=headers,
            content=json.dumps(_operator_contact_payload(**payload_updates)),
        )

    assert response.status_code == 422
    assert crm.operator_contact_calls == []


def test_operator_contact_capability_is_derived_from_registered_route():
    funnel_mod._served_capabilities_cache = None
    assert "contact.operator_mutation" in funnel_mod.served_capabilities()


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
        "capabilities": list(funnel_mod.served_capabilities()),
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
        "capabilities": list(funnel_mod.served_capabilities()),
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
@pytest.mark.parametrize(
    "booking_path", ["estimate-bookings", "first-clean-bookings"]
)
async def test_private_estimate_booking_rejects_numeric_timestamps(
    numeric_payload, booking_path
):
    """Pydantic lax mode would coerce epoch numbers -- and digit-only strings
    like "3600" -- into 1970-era UTC-aware datetimes that pass the
    timezone/ordering checks; the boundary must 422 anything that is not an
    RFC 3339 date-time string before CRM or Calendar sees it. Both booking
    routes share the request model, so both are held to it."""
    crm = _CRM()
    calendar = _Calendar()
    app = _app(crm, _enabled_config(), calendar=calendar)
    contact_id = uuid4()
    booking_key = f"office-booking-{uuid4().hex}"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/{booking_path}",
            headers=_headers(approval_key=booking_key),
            json=_booking_payload(**numeric_payload),
        )

    assert response.status_code == 422
    assert crm.prepare_calls == []
    assert crm.first_clean_prepare_calls == []
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
async def test_disposition_replay_supersession_uses_lifecycle_sequence():
    from atlas_brain.services.crm_provider import (
        _eom_disposition_replay_was_superseded,
    )

    class _Conn:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[object, ...]]] = []

        async def fetchval(self, query: str, *args: object) -> bool:
            self.calls.append((query, args))
            return True

    conn = _Conn()
    replay_id = uuid4()

    assert await _eom_disposition_replay_was_superseded(
        conn,
        contact_id="contact-1",
        replay_event_type="lead_lost",
        replay_event_id=replay_id,
        replay_lifecycle_sequence=42,
    )

    query, args = conn.calls[0]
    assert "lifecycle_sequence > $3" in query
    assert args == (
        "contact-1",
        ["lead_lost", "lead_reopened"],
        42,
    )


@pytest.mark.asyncio
async def test_disposition_replay_supersession_checks_legacy_rows_by_other_id():
    from atlas_brain.services.crm_provider import (
        _eom_disposition_replay_was_superseded,
    )

    class _Conn:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[object, ...]]] = []

        async def fetchval(self, query: str, *args: object) -> bool:
            self.calls.append((query, args))
            return True

    conn = _Conn()
    replay_id = uuid4()

    assert await _eom_disposition_replay_was_superseded(
        conn,
        contact_id="contact-1",
        replay_event_type="lead_lost",
        replay_event_id=replay_id,
        replay_lifecycle_sequence=None,
    )

    query, args = conn.calls[0]
    assert "id <> $3" in query
    assert "lifecycle_sequence > $3" not in query
    assert args == (
        "contact-1",
        ["lead_lost", "lead_reopened"],
        replay_id,
    )


@pytest.mark.asyncio
async def test_disposition_replay_supersession_admits_legacy_reopen_pair():
    from atlas_brain.services.crm_provider import (
        _eom_disposition_replay_was_superseded,
    )

    class _Conn:
        def __init__(self, row: dict[str, int]) -> None:
            self.row = row
            self.calls: list[tuple[str, tuple[object, ...]]] = []

        async def fetchrow(self, query: str, *args: object) -> dict[str, int]:
            self.calls.append((query, args))
            return self.row

    replay_id = uuid4()
    conn = _Conn(
        {
            "disposition_count": 2,
            "replay_reopen_count": 1,
            "legacy_loss_predecessor_count": 1,
        }
    )

    assert not await _eom_disposition_replay_was_superseded(
        conn,
        contact_id="contact-1",
        replay_event_type="lead_reopened",
        replay_event_id=replay_id,
        replay_lifecycle_sequence=None,
    )

    query, args = conn.calls[0]
    assert "COUNT(*) AS disposition_count" in query
    assert "event.from_stage = 'lost'" in query
    assert "event.from_stage = replay.to_stage" in query
    assert "event.to_stage = ANY($4::varchar[])" in query
    assert "legacy_loss_predecessor_count" in query
    assert args == (
        "contact-1",
        ["lead_lost", "lead_reopened"],
        replay_id,
        ["new", "estimate_booked"],
    )

    ambiguous_conn = _Conn(
        {
            "disposition_count": 3,
            "replay_reopen_count": 1,
            "legacy_loss_predecessor_count": 1,
        }
    )
    assert await _eom_disposition_replay_was_superseded(
        ambiguous_conn,
        contact_id="contact-1",
        replay_event_type="lead_reopened",
        replay_event_id=replay_id,
        replay_lifecycle_sequence=None,
    )

    mismatched_conn = _Conn(
        {
            "disposition_count": 2,
            "replay_reopen_count": 1,
            "legacy_loss_predecessor_count": 0,
        }
    )
    assert await _eom_disposition_replay_was_superseded(
        mismatched_conn,
        contact_id="contact-1",
        replay_event_type="lead_reopened",
        replay_event_id=replay_id,
        replay_lifecycle_sequence=None,
    )


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
async def test_private_first_clean_booking_prepares_calendar_and_completes_in_order():
    crm = _CRM()
    calendar = _Calendar()
    app = _app(crm, _enabled_config(), calendar=calendar)
    contact_id = uuid4()
    booking_key = f"office-first-clean-{uuid4().hex}"
    expected_event_id = deterministic_eom_first_clean_calendar_event_id(
        contact_id=str(contact_id),
        booking_key=booking_key,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/first-clean-bookings",
            headers=_headers(approval_key=booking_key),
            json=_booking_payload(),
        )

    assert response.status_code == 201
    assert response.json() == {
        "success": True,
        "contact_id": str(contact_id),
        "lead_stage": "won",
        "status": "first_clean_booked",
        "calendar_event_id": expected_event_id,
        "expected_calendar_event_id": expected_event_id,
        "idempotent": False,
        "onboarding_draft_id": crm.onboarding_draft_id,
    }
    assert crm.first_clean_prepare_calls[0]["contact_id"] == str(contact_id)
    assert crm.first_clean_prepare_calls[0]["booking_key"] == booking_key
    assert (
        crm.first_clean_prepare_calls[0]["expected_calendar_event_id"]
        == expected_event_id
    )
    assert calendar.calls == [
        {
            "summary": "First clean: Review Queue Lead",
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
    assert crm.first_clean_complete_calls[0]["calendar_event_id"] == expected_event_id
    # The estimate family must stay untouched by a first-clean booking.
    assert crm.prepare_calls == []
    assert crm.complete_calls == []
    assert crm.first_clean_ambiguous_calls == []


@pytest.mark.asyncio
async def test_private_first_clean_booking_shares_the_execution_lock_namespace():
    """Both families serialize through the same execution lock so the handoff
    fence sees an in-flight first-clean booking exactly like an estimate."""
    crm = _CRM()
    calendar = _Calendar()
    app = _app(crm, _enabled_config(), calendar=calendar)
    contact_id = uuid4()
    booking_key = f"office-first-clean-{uuid4().hex}"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/first-clean-bookings",
            headers=_headers(approval_key=booking_key),
            json=_booking_payload(),
        )

    assert response.status_code == 201
    assert crm.execution_lock_keys == [booking_key]


@pytest.mark.asyncio
async def test_private_first_clean_booking_runs_lifecycle_on_execution_scoped_provider():
    """The first-clean bindings must resolve on the execution-scoped provider
    the lock yields, not the outer provider, or one booking reserves two
    pooled connections."""

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
    booking_key = f"office-first-clean-{uuid4().hex}"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/first-clean-bookings",
            headers=_headers(approval_key=booking_key),
            json=_booking_payload(),
        )

    assert response.status_code == 201
    assert crm.execution_lock_keys == [booking_key]
    assert crm.first_clean_prepare_calls == []
    assert crm.first_clean_complete_calls == []
    assert len(crm.scoped.first_clean_prepare_calls) == 1
    assert len(crm.scoped.first_clean_complete_calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("pre_request_error", ["TOOL_DISABLED", "NOT_CONFIGURED"])
async def test_private_first_clean_booking_pre_request_calendar_failure_is_terminal(
    pre_request_error,
):
    """Pre-request Calendar failures prove no event write for the first-clean
    family exactly as for estimates: terminal failed attempt, never an
    ambiguous wedge."""
    crm = _CRM()
    calendar = _Calendar(
        success=False,
        error=pre_request_error,
        message="Calendar unavailable before any request",
    )
    app = _app(crm, _enabled_config(), calendar=calendar)
    contact_id = uuid4()
    booking_key = f"office-first-clean-{uuid4().hex}"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/first-clean-bookings",
            headers=_headers(approval_key=booking_key),
            json=_booking_payload(),
        )

    assert response.status_code == 502
    assert crm.first_clean_ambiguous_calls == []
    assert len(crm.first_clean_failed_calls) == 1
    assert crm.first_clean_failed_calls[0]["calendar_error"] == pre_request_error
    assert crm.failed_calls == []


@pytest.mark.asyncio
async def test_private_first_clean_booking_auth_phase_failure_is_terminal():
    """An OAuth-phase failure proves no event write for the first-clean
    family: terminal failed attempt on the first-clean markers."""
    crm = _CRM()
    calendar = _Calendar(
        success=False,
        error="AUTH_ERROR",
        message="OAuth token endpoint unavailable",
        data={"request_phase": "auth"},
    )
    app = _app(crm, _enabled_config(), calendar=calendar)
    contact_id = uuid4()
    booking_key = f"office-first-clean-{uuid4().hex}"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/first-clean-bookings",
            headers=_headers(approval_key=booking_key),
            json=_booking_payload(),
        )

    assert response.status_code == 502
    assert crm.first_clean_ambiguous_calls == []
    assert len(crm.first_clean_failed_calls) == 1
    assert crm.first_clean_failed_calls[0]["calendar_error"] == "AUTH_ERROR"


@pytest.mark.asyncio
async def test_private_first_clean_booking_marks_ambiguous_when_completion_rejects():
    """Completion rejection after a successful Calendar create must leave
    reconciliation evidence on the first-clean markers."""

    class _RejectingCompletionCRM(_CRM):
        async def complete_eom_first_clean_booking(self, **kwargs):
            self.first_clean_complete_calls.append(kwargs)
            raise EOMLeadConversionError(409, "EOM contact is not a lead")

    crm = _RejectingCompletionCRM()
    calendar = _Calendar()
    app = _app(crm, _enabled_config(), calendar=calendar)
    contact_id = uuid4()
    booking_key = f"office-first-clean-{uuid4().hex}"
    expected_event_id = deterministic_eom_first_clean_calendar_event_id(
        contact_id=str(contact_id),
        booking_key=booking_key,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/first-clean-bookings",
            headers=_headers(approval_key=booking_key),
            json=_booking_payload(),
        )

    assert response.status_code == 409
    assert len(crm.first_clean_complete_calls) == 1
    assert len(crm.first_clean_ambiguous_calls) == 1
    assert (
        crm.first_clean_ambiguous_calls[0]["expected_calendar_event_id"]
        == expected_event_id
    )
    assert (
        crm.first_clean_ambiguous_calls[0]["observed_calendar_event_id"]
        == expected_event_id
    )
    assert crm.ambiguous_calls == []


@pytest.mark.asyncio
async def test_private_first_clean_booking_idempotent_replay_skips_calendar_side_effect():
    class _BookedCRM(_CRM):
        async def prepare_eom_first_clean_booking(self, **kwargs):
            self.first_clean_prepare_calls.append(kwargs)
            return {
                "contact_id": UUID(kwargs["contact_id"]),
                "lead_stage": "won",
                "status": "first_clean_booked",
                "calendar_event_id": kwargs["expected_calendar_event_id"],
                "expected_calendar_event_id": kwargs["expected_calendar_event_id"],
                "idempotent": True,
                "onboarding_draft_id": self.onboarding_draft_id,
                "contact": {
                    "full_name": "Review Queue Lead",
                    "address": "100 Main St",
                },
            }

    crm = _BookedCRM()
    calendar = _Calendar()
    app = _app(crm, _enabled_config(), calendar=calendar)
    contact_id = uuid4()
    booking_key = f"office-first-clean-{uuid4().hex}"
    expected_event_id = deterministic_eom_first_clean_calendar_event_id(
        contact_id=str(contact_id),
        booking_key=booking_key,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/first-clean-bookings",
            headers=_headers(approval_key=booking_key),
            json=_booking_payload(),
        )

    assert response.status_code == 200
    assert response.json()["contact_id"] == str(contact_id)
    assert response.json()["idempotent"] is True
    assert response.json()["calendar_event_id"] == expected_event_id
    assert response.json()["onboarding_draft_id"] == crm.onboarding_draft_id
    assert len(crm.first_clean_prepare_calls) == 1
    assert calendar.calls == []
    assert crm.first_clean_complete_calls == []
    assert crm.first_clean_failed_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("contact_email", "latest_intake_email", "expected_recipient", "expected_blocker"),
    (
        ("lead@example.com", None, "lead@example.com", None),
        # Ingress leaves contacts.email unchanged on re-submission; the
        # latest intake projection must win over the stale contact column.
        ("stale@example.com", "fresh@example.com", "fresh@example.com", None),
        (None, "fresh@example.com", "fresh@example.com", None),
        ("", None, None, "no_email"),
        (None, None, None, "no_email"),
        ("   ", None, None, "no_email"),
    ),
)
async def test_onboarding_draft_enqueue_snapshots_recipient_or_records_blocker(
    contact_email, latest_intake_email, expected_recipient, expected_blocker
):
    """A contact without an email is enqueued with blocker='no_email' rather
    than silently skipped, and the recipient resolves through the same
    latest-intake projection the review queue shows."""
    from atlas_brain.services.crm_provider import DatabaseCRMProvider
    from atlas_brain.templates.email import format_onboarding_welcome

    class _Conn:
        def __init__(self, intake_email) -> None:
            self.intake_email = intake_email
            self.fetchval_calls: list[tuple[str, tuple[object, ...]]] = []
            self.fetchrow_calls: list[tuple[str, tuple[object, ...]]] = []

        async def fetchval(self, query: str, *args):
            self.fetchval_calls.append((query, args))
            return self.intake_email

        async def fetchrow(self, query: str, *args):
            self.fetchrow_calls.append((query, args))
            return {"id": UUID("0b8db22e-16b1-4a30-a15f-6c78ee9204a5")}

    conn = _Conn(latest_intake_email)
    contact_id = uuid4()
    draft_id = await DatabaseCRMProvider._enqueue_eom_onboarding_email_draft(
        conn,
        contact={
            "id": contact_id,
            "full_name": "Review Queue Lead",
            "email": contact_email,
        },
        operation_key="office-first-clean-abc123",
    )

    assert draft_id == "0b8db22e-16b1-4a30-a15f-6c78ee9204a5"
    intake_query, intake_args = conn.fetchval_calls[0]
    assert "contact_interactions" in intake_query
    assert "submitted_email" in intake_query
    assert intake_args == (str(contact_id),)
    query, args = conn.fetchrow_calls[0]
    assert "INSERT INTO eom_onboarding_email_drafts" in query
    assert "ON CONFLICT (operation_key) DO NOTHING" in query
    expected_subject, expected_body = format_onboarding_welcome(
        client_name="Review Queue Lead"
    )
    assert args == (
        str(contact_id),
        "office-first-clean-abc123",
        expected_recipient,
        expected_blocker,
        expected_subject,
        expected_body,
    )


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
@pytest.mark.parametrize(
    "booking_path", ["estimate-bookings", "first-clean-bookings"]
)
async def test_private_estimate_booking_rejects_bad_body_before_side_effects(
    body: dict[str, object],
    booking_path: str,
):
    crm = _CRM()
    calendar = _Calendar()
    app = _app(crm, _enabled_config(), calendar=calendar)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{uuid4()}/{booking_path}",
            headers=_headers(approval_key=f"office-booking-{uuid4().hex}"),
            json=body,
        )

    assert response.status_code == 422
    assert crm.prepare_calls == []
    assert crm.first_clean_prepare_calls == []
    assert calendar.calls == []
    assert crm.complete_calls == []
    assert crm.first_clean_complete_calls == []


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
@pytest.mark.parametrize(
    "booking_path", ["estimate-bookings", "first-clean-bookings"]
)
async def test_private_estimate_booking_rejects_http_guards_before_side_effects(
    config: EOMFunnelConfig,
    headers: dict[str, str],
    expected_status: int,
    booking_path: str,
):
    crm = _CRM()
    calendar = _Calendar()
    app = _app(crm, config, calendar=calendar)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{uuid4()}/{booking_path}",
            headers=headers,
            json=_booking_payload(),
        )

    assert response.status_code == expected_status
    assert crm.prepare_calls == []
    assert crm.first_clean_prepare_calls == []
    assert calendar.calls == []
    assert crm.complete_calls == []
    assert crm.first_clean_complete_calls == []


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

class _DraftSender:
    """Fake transport for approve-and-send tests."""

    def __init__(self, *, fail: bool = False, idempotent_replay: bool = False):
        self.fail = fail
        self.idempotent_replay = idempotent_replay
        self.calls: list[dict[str, object]] = []

    async def __call__(self, *, to, subject, body, idempotency_key):
        self.calls.append(
            {
                "to": to,
                "subject": subject,
                "body": body,
                "idempotency_key": idempotency_key,
            }
        )
        if self.fail:
            raise RuntimeError("transport unavailable")
        if self.idempotent_replay:
            return {"message_id": None, "idempotent_replay": True}
        return {"message_id": "resend-msg-1", "idempotent_replay": False}


class _SentEmailHistory:
    """Recording stand-in injected through the service's history seam."""

    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.created.append(kwargs)
        return SimpleNamespace(id=uuid4())


def _draft_app(crm: _CRM, sender=None, email_history=None) -> FastAPI:
    app = _app(crm, _enabled_config())
    app.dependency_overrides[funnel_mod._onboarding_sender_dependency] = (
        lambda: sender
    )
    app.dependency_overrides[funnel_mod._onboarding_email_history_dependency] = (
        lambda: email_history
    )
    return app


async def _post(app: FastAPI, path: str, *, json_body=None, method: str = "POST"):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.request(
            method,
            path,
            headers=_headers(approval_key=f"office-draft-{uuid4().hex}"),
            json=json_body,
        )


@pytest.mark.asyncio
async def test_private_onboarding_draft_list_returns_closed_projection():
    crm = _CRM()
    draft_id = uuid4()
    contact_id = uuid4()
    crm.draft_list_rows = [
        {
            "draft_id": draft_id,
            "contact_id": contact_id,
            "full_name": "Won Lead",
            "recipient_email": None,
            "blocker": "no_email",
            "subject": "Welcome aboard - Effingham Office Maids",
            "body": "Hi Won Lead,",
            "status": "pending",
            "created_at": datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc),
            "claimed_at": None,
            "sent_at": None,
            "revoked_at": None,
            "approved_by_name": None,
            # A leaked column must be rejected by the closed response model.
        }
    ]
    app = _draft_app(crm)

    response = await _post(
        app, "/eom-funnel/onboarding-drafts?limit=25", method="GET"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "pending"
    assert payload["limit"] == 25
    assert payload["hasMore"] is False
    assert payload["nextCursor"] is None
    assert payload["drafts"] == [
        {
            "draftId": str(draft_id),
            "contactId": str(contact_id),
            "fullName": "Won Lead",
            "recipientEmail": None,
            "blocker": "no_email",
            "subject": "Welcome aboard - Effingham Office Maids",
            "body": "Hi Won Lead,",
            "status": "pending",
            "createdAt": "2026-08-04T12:00:00Z",
            "claimedAt": None,
            "sentAt": None,
            "revokedAt": None,
            "approvedByName": None,
        }
    ]
    assert crm.draft_list_calls == [
        {
            "status": "pending",
            "limit": 26,
            "cursor_created_at": None,
            "cursor_draft_id": None,
        }
    ]


@pytest.mark.asyncio
async def test_private_onboarding_draft_list_rejects_unknown_status_before_crm_call():
    crm = _CRM()
    app = _draft_app(crm)

    response = await _post(
        app, "/eom-funnel/onboarding-drafts?status=draining", method="GET"
    )

    assert response.status_code == 422
    assert crm.draft_list_calls == []


@pytest.mark.asyncio
async def test_private_onboarding_draft_edit_sets_recipient_and_clears_blocker():
    crm = _CRM()
    row = crm.seed_draft(recipient_email=None, blocker="no_email")
    app = _draft_app(crm)

    response = await _post(
        app,
        f"/eom-funnel/onboarding-drafts/{row['id']}",
        method="PATCH",
        json_body={"recipient_email": "fixed@example.com"},
    )

    assert response.status_code == 200
    assert response.json()["recipient_email"] == "fixed@example.com"
    assert response.json()["blocker"] is None
    assert crm.draft_update_calls == [
        {
            "draft_id": str(row["id"]),
            "subject": None,
            "body": None,
            "recipient_email": "fixed@example.com",
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    (
        {},
        {"subject": "   "},
        {"body": ""},
        {"recipient_email": "not-an-email"},
        {"recipient_email": "two words@example.com"},
        # Valid pattern but 255 chars: one over the intake boundary's 254
        # cap, which office corrections must never exceed.
        {"recipient_email": ("a" * 243) + "@example.com"},
        {"subject": "ok", "unexpected": "field"},
    ),
)
async def test_private_onboarding_draft_edit_rejects_bad_body_before_crm_call(body):
    crm = _CRM()
    row = crm.seed_draft()
    app = _draft_app(crm)

    response = await _post(
        app,
        f"/eom-funnel/onboarding-drafts/{row['id']}",
        method="PATCH",
        json_body=body,
    )

    assert response.status_code == 422
    assert crm.draft_update_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("frozen_status", ("sending", "sent", "revoked"))
async def test_private_onboarding_draft_edit_rejects_non_pending(frozen_status):
    crm = _CRM()
    row = crm.seed_draft(status=frozen_status)
    app = _draft_app(crm)

    response = await _post(
        app,
        f"/eom-funnel/onboarding-drafts/{row['id']}",
        method="PATCH",
        json_body={"subject": "Edited"},
    )

    assert response.status_code == 409
    assert frozen_status in response.json()["detail"]
    assert crm.draft_rows[str(row["id"])]["subject"] != "Edited"


@pytest.mark.asyncio
async def test_private_onboarding_draft_approve_claims_sends_confirms_in_order():
    crm = _CRM()
    row = crm.seed_draft()
    sender = _DraftSender()
    app = _draft_app(crm, sender=sender)

    response = await _post(
        app, f"/eom-funnel/onboarding-drafts/{row['id']}/approve-send"
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["success"] is True
    assert payload["status"] == "sent"
    assert payload["idempotent"] is False
    assert payload["resend_message_id"] == "resend-msg-1"
    assert payload["transport_idempotent_replay"] is False
    assert crm.draft_claim_calls == [
        {
            "draft_id": str(row["id"]),
            "actor_id": 1,
            "actor_name": "Juan Canfield",
        }
    ]
    assert sender.calls == [
        {
            "to": "lead@example.com",
            "subject": row["subject"],
            "body": row["body"],
            "idempotency_key": f"eom-onboarding-draft:{row['id']}",
        }
    ]
    assert crm.draft_confirm_calls == [
        {"draft_id": str(row["id"]), "require_stale": False}
    ]
    assert crm.draft_rows[str(row["id"])]["status"] == "sent"
    assert crm.draft_rows[str(row["id"])]["approved_by_name"] == "Juan Canfield"
    assert len(crm.interaction_logs) == 1
    assert crm.interaction_logs[0]["interaction_type"] == "email"


@pytest.mark.asyncio
async def test_approve_service_records_send_evidence_through_injected_history():
    """Evidence is recorded only after confirmed delivery, through the
    injectable history seam (no internal module patching)."""
    from atlas_brain.services.eom_onboarding_drafts import (
        EOMOnboardingDraftApproval,
        approve_and_send_eom_onboarding_draft,
    )

    crm = _CRM()
    row = crm.seed_draft()
    sender = _DraftSender()
    history = _SentEmailHistory()

    result = await approve_and_send_eom_onboarding_draft(
        crm,
        EOMOnboardingDraftApproval(
            draft_id=str(row["id"]), actor_id=1, actor_name="Juan Canfield"
        ),
        sender=sender,
        email_history=history,
    )

    assert result["status"] == "sent"
    created = history.created[0]
    assert created["to_addresses"] == ["lead@example.com"]
    assert created["template_type"] == "onboarding_welcome"
    assert created["resend_message_id"] == "resend-msg-1"
    assert created["business_context_id"] == "effingham_maids"
    assert created["metadata"]["draft_id"] == str(row["id"])
    assert created["metadata"]["contact_id"] == str(row["contact_id"])
    assert len(crm.interaction_logs) == 1
    assert crm.interaction_logs[0]["interaction_type"] == "email"
    assert str(row["id"]) in crm.interaction_logs[0]["summary"]


class _EvidencePool:
    """Recording store adapter standing in for the funnel CRM pool."""

    is_initialized = True

    def __init__(self) -> None:
        self.fetchrow_calls: list[tuple[str, tuple]] = []

    async def fetchrow(self, query: str, *args):
        self.fetchrow_calls.append((query, args))
        return {"id": args[0], "sent_at": datetime.now(timezone.utc)}


@pytest.mark.asyncio
async def test_approve_service_default_history_uses_the_crm_providers_pool():
    """Without an injected history seam, the sent_emails evidence write goes
    through the CRM provider's own pool -- the store that owns the draft --
    not the global pool (which the slim funnel profile may not even point at
    the same database)."""
    from atlas_brain.services.eom_onboarding_drafts import (
        EOMOnboardingDraftApproval,
        approve_and_send_eom_onboarding_draft,
    )

    crm = _CRM()
    crm.pool = _EvidencePool()
    row = crm.seed_draft()
    sender = _DraftSender()

    result = await approve_and_send_eom_onboarding_draft(
        crm,
        EOMOnboardingDraftApproval(
            draft_id=str(row["id"]), actor_id=1, actor_name="Juan Canfield"
        ),
        sender=sender,
    )

    assert result["status"] == "sent"
    assert len(crm.pool.fetchrow_calls) == 1
    insert_query, insert_args = crm.pool.fetchrow_calls[0]
    assert "INSERT INTO sent_emails" in insert_query
    assert ["lead@example.com"] in insert_args


@pytest.mark.asyncio
async def test_private_onboarding_draft_approve_replays_sent_without_transport():
    crm = _CRM()
    row = crm.seed_draft(
        status="sent",
        sent_at=datetime(2026, 8, 4, 13, 0, tzinfo=timezone.utc),
    )
    sender = _DraftSender()
    app = _draft_app(crm, sender=sender)

    response = await _post(
        app, f"/eom-funnel/onboarding-drafts/{row['id']}/approve-send"
    )

    assert response.status_code == 200
    assert response.json()["idempotent"] is True
    assert response.json()["status"] == "sent"
    assert sender.calls == []
    assert crm.draft_confirm_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "detail_fragment"),
    (
        ({"blocker": "no_email", "recipient_email": None}, "blocked"),
        ({"recipient_email": None}, "no recipient"),
        ({"status": "sending"}, "in flight"),
        ({"status": "revoked"}, "revoked"),
        ({"contact_status": "archived"}, "not an active"),
    ),
)
async def test_private_onboarding_draft_approve_refuses_unready_states(
    overrides, detail_fragment
):
    crm = _CRM()
    row = crm.seed_draft(**overrides)
    sender = _DraftSender()
    app = _draft_app(crm, sender=sender)

    response = await _post(
        app, f"/eom-funnel/onboarding-drafts/{row['id']}/approve-send"
    )

    assert response.status_code == 409
    assert detail_fragment in response.json()["detail"]
    assert sender.calls == []
    assert crm.draft_confirm_calls == []


@pytest.mark.asyncio
async def test_private_onboarding_draft_approve_transport_failure_leaves_sending():
    crm = _CRM()
    row = crm.seed_draft()
    sender = _DraftSender(fail=True)
    app = _draft_app(crm, sender=sender)

    response = await _post(
        app, f"/eom-funnel/onboarding-drafts/{row['id']}/approve-send"
    )

    assert response.status_code == 502
    assert "reconciliation" in response.json()["detail"]
    assert len(sender.calls) == 1
    # The row stays 'sending' as operator evidence (migration 360 step 4):
    # neither auto-retried nor rolled back to pending.
    assert crm.draft_rows[str(row["id"])]["status"] == "sending"
    assert crm.draft_confirm_calls == []


@pytest.mark.asyncio
async def test_private_onboarding_draft_approve_resend_replay_confirms_sent():
    crm = _CRM()
    row = crm.seed_draft()
    sender = _DraftSender(idempotent_replay=True)
    app = _draft_app(crm, sender=sender)

    response = await _post(
        app, f"/eom-funnel/onboarding-drafts/{row['id']}/approve-send"
    )

    assert response.status_code == 201
    assert response.json()["status"] == "sent"
    assert response.json()["transport_idempotent_replay"] is True
    assert crm.draft_rows[str(row["id"])]["status"] == "sent"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enabled", "api_key"),
    (
        (False, "re_valid_key"),
        # Whitespace-only key: truthy but unusable; must 503 before any
        # claim rather than wedge the row in 'sending' at Resend.
        (True, "   "),
    ),
)
async def test_private_onboarding_draft_approve_requires_transport_before_claim(
    monkeypatch, enabled, api_key
):
    from atlas_brain.config import settings

    monkeypatch.setattr(settings.email, "enabled", enabled)
    monkeypatch.setattr(settings.email, "api_key", api_key)
    crm = _CRM()
    row = crm.seed_draft()
    app = _draft_app(crm, sender=None)

    response = await _post(
        app, f"/eom-funnel/onboarding-drafts/{row['id']}/approve-send"
    )

    assert response.status_code == 503
    assert "not claimed" in response.json()["detail"]
    assert crm.draft_claim_calls == []
    assert crm.draft_rows[str(row["id"])]["status"] == "pending"


@pytest.mark.asyncio
async def test_private_onboarding_draft_revoke_paths():
    crm = _CRM()
    pending = crm.seed_draft()
    stuck = crm.seed_draft(
        status="sending",
        claimed_at=datetime.now(timezone.utc) - timedelta(minutes=20),
    )
    active = crm.seed_draft(
        status="sending",
        claimed_at=datetime.now(timezone.utc),
    )
    sent = crm.seed_draft(status="sent")
    app = _draft_app(crm)

    fresh = await _post(app, f"/eom-funnel/onboarding-drafts/{pending['id']}/revoke")
    assert fresh.status_code == 201
    assert fresh.json()["status"] == "revoked"
    assert len(crm.interaction_logs) == 1
    assert "revoked onboarding draft" in crm.interaction_logs[0]["summary"]

    replay = await _post(app, f"/eom-funnel/onboarding-drafts/{pending['id']}/revoke")
    assert replay.status_code == 200
    assert replay.json()["idempotent"] is True
    assert len(crm.interaction_logs) == 1  # replay logs nothing

    reconciled = await _post(app, f"/eom-funnel/onboarding-drafts/{stuck['id']}/revoke")
    assert reconciled.status_code == 201

    # An ACTIVE send (fresh claim) must not be revocable: the customer email
    # may already be delivered while its confirmation is still in flight.
    in_flight = await _post(
        app, f"/eom-funnel/onboarding-drafts/{active['id']}/revoke"
    )
    assert in_flight.status_code == 409
    assert "still in flight" in in_flight.json()["detail"]
    assert crm.draft_rows[str(active["id"])]["status"] == "sending"

    refused = await _post(app, f"/eom-funnel/onboarding-drafts/{sent['id']}/revoke")
    assert refused.status_code == 409
    assert "already sent" in refused.json()["detail"]


@pytest.mark.asyncio
async def test_private_onboarding_draft_confirm_sent_paths():
    crm = _CRM()
    stuck = crm.seed_draft(
        status="sending",
        claimed_at=datetime.now(timezone.utc) - timedelta(minutes=20),
    )
    active = crm.seed_draft(
        status="sending",
        claimed_at=datetime.now(timezone.utc),
    )
    pending = crm.seed_draft()
    history = _SentEmailHistory()
    app = _draft_app(crm, email_history=history)

    confirmed = await _post(
        app, f"/eom-funnel/onboarding-drafts/{stuck['id']}/confirm-sent"
    )
    assert confirmed.status_code == 201
    assert confirmed.json()["status"] == "sent"
    # The operator route demands a stale claim; the in-flow service confirm
    # does not (it just observed transport acceptance).
    assert crm.draft_confirm_calls[-1]["require_stale"] is True
    assert len(crm.interaction_logs) == 2
    assert "transport-log" in crm.interaction_logs[0]["summary"]
    assert crm.interaction_logs[1]["interaction_type"] == "email"
    # Crash-recovery deliveries record the same sent-email history as the
    # normal approve path, with a null transport id (never observed).
    assert len(history.created) == 1
    assert history.created[0]["resend_message_id"] is None
    assert history.created[0]["template_type"] == "onboarding_welcome"
    assert history.created[0]["to_addresses"] == [stuck["recipient_email"]]
    assert history.created[0]["business_context_id"] == "effingham_maids"

    replay = await _post(
        app, f"/eom-funnel/onboarding-drafts/{stuck['id']}/confirm-sent"
    )
    assert replay.status_code == 200
    assert replay.json()["idempotent"] is True
    assert len(crm.interaction_logs) == 2
    assert len(history.created) == 1

    # A fresh claim is an active send with an unknown outcome; the operator
    # cannot record it as delivered before it settles or goes stale.
    in_flight = await _post(
        app, f"/eom-funnel/onboarding-drafts/{active['id']}/confirm-sent"
    )
    assert in_flight.status_code == 409
    assert "still in flight" in in_flight.json()["detail"]
    assert crm.draft_rows[str(active["id"])]["status"] == "sending"

    refused = await _post(
        app, f"/eom-funnel/onboarding-drafts/{pending['id']}/confirm-sent"
    )
    assert refused.status_code == 409
    assert "not been claimed" in refused.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "headers", "expected_status"),
    (
        (EOMFunnelConfig(api_enabled=False), _headers(), 503),
        (_enabled_config(), {**_headers(), "Authorization": ""}, 401),
        (_enabled_config(), {**_headers(actor_id="not-an-id")}, 422),
    ),
)
@pytest.mark.parametrize(
    ("method", "path_suffix"),
    (
        ("GET", ""),
        ("POST", "/{draft_id}/approve-send"),
        ("PATCH", "/{draft_id}"),
        ("POST", "/{draft_id}/revoke"),
        ("POST", "/{draft_id}/confirm-sent"),
    ),
)
async def test_private_onboarding_draft_routes_reject_http_guards_before_crm_call(
    config, headers, expected_status, method, path_suffix
):
    crm = _CRM()
    row = crm.seed_draft()
    app = FastAPI()
    app.include_router(funnel_mod.router)
    app.dependency_overrides[funnel_mod._crm_dependency] = lambda: crm
    app.dependency_overrides[funnel_mod._onboarding_sender_dependency] = (
        lambda: _DraftSender()
    )
    app.dependency_overrides[auth_mod.get_eom_funnel_api_config] = lambda: config
    path = "/eom-funnel/onboarding-drafts" + path_suffix.format(
        draft_id=row["id"]
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.request(
            method,
            path,
            headers=headers,
            json={"subject": "Edited"} if method == "PATCH" else None,
        )

    assert response.status_code == expected_status
    assert crm.draft_list_calls == []
    assert crm.draft_update_calls == []
    assert crm.draft_claim_calls == []
    assert crm.draft_revoke_calls == []
    assert crm.draft_confirm_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_status", "response_json", "expected"),
    (
        (200, {"id": "resend-abc"}, {"message_id": "resend-abc", "idempotent_replay": False}),
        (
            409,
            {"name": "invalid_idempotent_request", "message": "key reuse"},
            {"message_id": None, "idempotent_replay": True},
        ),
    ),
)
async def test_send_onboarding_email_maps_resend_responses(
    response_status, response_json, expected
):
    from atlas_brain.services.eom_onboarding_drafts import send_onboarding_email

    class _Response:
        status_code = response_status

        def json(self):
            return response_json

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    "resend error",
                    request=httpx.Request("POST", "https://api.resend.com/emails"),
                    response=httpx.Response(self.status_code),
                )

    class _Client:
        def __init__(self):
            self.posts = []

        async def post(self, url, *, json, headers):
            self.posts.append({"url": url, "json": json, "headers": headers})
            return _Response()

    client = _Client()
    result = await send_onboarding_email(
        to="lead@example.com",
        subject="Welcome",
        body="Hi",
        idempotency_key="eom-onboarding-draft:abc",
        http_client=client,
    )

    assert result == expected
    posted = client.posts[0]
    assert posted["headers"]["Idempotency-Key"] == "eom-onboarding-draft:abc"
    assert posted["json"]["to"] == ["lead@example.com"]
    assert posted["json"]["text"] == "Hi"
    assert posted["json"]["from"].startswith("Effingham Office Maids <")


@pytest.mark.asyncio
async def test_send_onboarding_email_strips_the_configured_api_key(monkeypatch):
    """A padded key reaches Resend stripped, matching the preflight bound."""
    from atlas_brain.config import settings
    from atlas_brain.services.eom_onboarding_drafts import send_onboarding_email

    monkeypatch.setattr(settings.email, "api_key", "  re_padded_key  ")

    class _Response:
        status_code = 200

        def json(self):
            return {"id": "resend-abc"}

        def raise_for_status(self):
            return None

    class _Client:
        def __init__(self):
            self.posts = []

        async def post(self, url, *, json, headers):
            self.posts.append({"url": url, "json": json, "headers": headers})
            return _Response()

    client = _Client()
    result = await send_onboarding_email(
        to="lead@example.com",
        subject="Welcome",
        body="Hi",
        idempotency_key="eom-onboarding-draft:abc",
        http_client=client,
    )

    assert result["message_id"] == "resend-abc"
    assert client.posts[0]["headers"]["Authorization"] == "Bearer re_padded_key"


@pytest.mark.asyncio
async def test_send_onboarding_email_raises_on_transport_error():
    from atlas_brain.services.eom_onboarding_drafts import send_onboarding_email

    class _Response:
        status_code = 500

        def json(self):
            return {"message": "server error"}

        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "resend error",
                request=httpx.Request("POST", "https://api.resend.com/emails"),
                response=httpx.Response(500),
            )

    class _Client:
        async def post(self, url, *, json, headers):
            return _Response()

    with pytest.raises(httpx.HTTPStatusError):
        await send_onboarding_email(
            to="lead@example.com",
            subject="Welcome",
            body="Hi",
            idempotency_key="eom-onboarding-draft:abc",
            http_client=_Client(),
        )


@pytest.mark.asyncio
async def test_private_mark_lead_lost_records_reason_note_and_actor():
    crm = _CRM()
    app = _app(crm, _enabled_config())
    contact_id = uuid4()
    op_key = f"office-lost-{uuid4().hex}"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/lost",
            headers=_headers(approval_key=op_key),
            json={"reason_code": "spam", "note": "  bot asked us to pay  "},
        )

    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True
    assert body["lead_stage"] == "lost"
    assert body["reason_code"] == "spam"
    assert crm.lost_calls[0]["contact_id"] == str(contact_id)
    assert crm.lost_calls[0]["reason_code"] == "spam"
    # the request model strips a whitespace-padded note
    assert crm.lost_calls[0]["note"] == "bot asked us to pay"
    assert crm.lost_calls[0]["operation_key"] == op_key
    assert crm.lost_calls[0]["actor_id"] == 1
    assert crm.lost_calls[0]["actor_name"] == "Juan Canfield"


@pytest.mark.asyncio
async def test_private_mark_lead_lost_rejects_unknown_reason_code():
    crm = _CRM()
    app = _app(crm, _enabled_config())
    contact_id = uuid4()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/lost",
            headers=_headers(),
            json={"reason_code": "banana"},
        )

    assert response.status_code == 422
    assert crm.lost_calls == []


@pytest.mark.asyncio
async def test_private_mark_lead_lost_blank_note_becomes_null():
    crm = _CRM()
    app = _app(crm, _enabled_config())
    contact_id = uuid4()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/lost",
            headers=_headers(),
            json={"reason_code": "no_response", "note": "   "},
        )

    assert response.status_code == 201
    assert crm.lost_calls[0]["note"] is None


@pytest.mark.asyncio
async def test_private_reopen_lead_surfaces_restored_stage():
    crm = _CRM()
    crm.reopen_stage = "estimate_booked"
    app = _app(crm, _enabled_config())
    contact_id = uuid4()
    op_key = f"office-reopen-{uuid4().hex}"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/leads/{contact_id}/reopen",
            headers=_headers(approval_key=op_key),
        )

    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True
    assert body["lead_stage"] == "estimate_booked"
    assert crm.reopen_calls[0]["contact_id"] == str(contact_id)
    assert crm.reopen_calls[0]["operation_key"] == op_key
    assert crm.reopen_calls[0]["actor_id"] == 1
