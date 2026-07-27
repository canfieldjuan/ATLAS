"""HTTP boundary proof for the private EOM office conversion API."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from atlas_brain.eom_api import funnel as funnel_mod
from atlas_brain.eom_api import funnel_auth as auth_mod
from atlas_brain.eom_api.config import EOMFunnelConfig
from atlas_brain.services.eom_lead_conversion import EOMLeadConversionError


_SERVICE_TOKEN = auth_mod.generate_eom_funnel_service_token()


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


def test_full_atlas_app_mounts_public_intake_and_private_handoff_together():
    """The tracker callback is served by the same full app that owns web leads."""
    from atlas_brain.main import app

    paths = {route.path for route in app.routes}

    assert "/api/v1/leads/intake" in paths
    assert "/api/v1/eom-funnel/customer-handoffs" in paths


@pytest.mark.asyncio
async def test_enabled_full_atlas_funnel_requires_authoritative_data_store(monkeypatch):
    from atlas_brain import main

    class _Pool:
        def __init__(self, *, initialized: bool, schema_ready: bool) -> None:
            self.is_initialized = initialized
            self._schema_ready = schema_ready

        async def fetchval(self, _query: str) -> bool:
            return self._schema_ready

    disabled = SimpleNamespace(api_enabled=False)

    def fail_if_looked_up():
        pytest.fail("disabled EOM funnel must not require a database pool")

    monkeypatch.setattr(main, "get_db_pool", fail_if_looked_up)
    await main._require_eom_funnel_data_store(disabled, database_enabled=False)

    enabled = SimpleNamespace(api_enabled=True)
    monkeypatch.setattr(
        main,
        "get_db_pool",
        lambda: _Pool(initialized=True, schema_ready=True),
    )

    await main._require_eom_funnel_data_store(enabled, database_enabled=True)

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
async def test_private_handoff_accepts_only_ids_and_actor_evidence():
    crm = _CRM()
    app = _app(crm, EOMFunnelConfig(api_enabled=True, service_token=_SERVICE_TOKEN))
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
    app = _app(crm, EOMFunnelConfig(api_enabled=True, service_token=_SERVICE_TOKEN))
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
    app = _app(crm, EOMFunnelConfig(api_enabled=True, service_token=_SERVICE_TOKEN))
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


@pytest.mark.parametrize(
    "token",
    (
        "x" * 24,
        "eomf_v1_" + ("x" * 43),
        "eomf_v1_" + ("a" * 42),
        "eomf_v1_" + ("*" * 43),
        "eomrx_v1_" + "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789_-ABC",
        "eomf_v1_" + (("abc123" * 8)[:43]),
        "\u00e9" * 43,
    ),
)
def test_enabled_funnel_rejects_weak_or_non_generated_service_tokens_at_startup(
    token: str,
):
    with pytest.raises(RuntimeError, match="generated|too short|invalid|too weak"):
        auth_mod.validate_eom_funnel_api_config(
            EOMFunnelConfig(api_enabled=True, service_token=token)
        )


def test_enabled_funnel_accepts_a_fresh_generated_service_token_at_startup():
    auth_mod.validate_eom_funnel_api_config(
        EOMFunnelConfig(
            api_enabled=True,
            service_token=auth_mod.generate_eom_funnel_service_token(),
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ("tracker_customer_id", "tracker_site_id"))
async def test_private_handoff_rejects_storage_overflow_before_crm_call(field: str):
    crm = _CRM()
    app = _app(crm, EOMFunnelConfig(api_enabled=True, service_token=_SERVICE_TOKEN))
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
    app = _app(crm, EOMFunnelConfig(api_enabled=True, service_token=_SERVICE_TOKEN))
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
    app = _app(crm, EOMFunnelConfig(api_enabled=True, service_token=_SERVICE_TOKEN))
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
