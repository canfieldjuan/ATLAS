"""HTTP boundary proof for the private EOM office conversion API."""

from __future__ import annotations

from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from atlas_brain.api import eom_lead_funnel as funnel_mod
from atlas_brain.api import eom_lead_funnel_auth as auth_mod
from atlas_brain.config import EOMFunnelConfig
from atlas_brain.services.eom_lead_conversion import EOMLeadConversionError


class _CRM:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

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


def _app(crm: _CRM, config: EOMFunnelConfig) -> FastAPI:
    app = FastAPI()
    app.include_router(funnel_mod.router)
    app.dependency_overrides[funnel_mod._crm_dependency] = lambda: crm
    app.dependency_overrides[auth_mod.get_eom_funnel_api_config] = lambda: config
    return app


def _approval_key() -> str:
    return f"office-handoff-{uuid4().hex}"


def _headers(token: str = "x" * 24, approval_key: str | None = None) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "X-EOM-Actor": "Juan Canfield",
        "X-EOM-Actor-ID": "1",
        "Idempotency-Key": approval_key or _approval_key(),
    }


@pytest.mark.asyncio
async def test_private_handoff_accepts_only_ids_and_actor_evidence():
    crm = _CRM()
    app = _app(crm, EOMFunnelConfig(api_enabled=True, service_token="x" * 24))
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
    app = _app(crm, EOMFunnelConfig(api_enabled=True, service_token="x" * 24))
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
    app = _app(crm, EOMFunnelConfig(api_enabled=True, service_token="x" * 24))
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
async def test_private_handoff_preserves_provider_rejection_without_side_effect_claim():
    class _RejectingCRM(_CRM):
        async def finalize_eom_customer_handoff(self, **kwargs):
            self.calls.append(kwargs)
            raise EOMLeadConversionError(409, "EOM lead is not ready for approval")

    crm = _RejectingCRM()
    app = _app(crm, EOMFunnelConfig(api_enabled=True, service_token="x" * 24))
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
