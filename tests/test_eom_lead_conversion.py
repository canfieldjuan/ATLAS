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


_GENERATED_SERVICE_TOKEN = auth_mod.generate_eom_funnel_service_token()
_SERVICE_TOKEN = _GENERATED_SERVICE_TOKEN.token
_SERVICE_TOKEN_SHA256 = _GENERATED_SERVICE_TOKEN.sha256


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


@pytest.mark.asyncio
async def test_full_atlas_app_serves_public_intake_and_private_handoff_together():
    """The actual full aggregate serves the tracker callback beside lead intake."""
    from atlas_brain.main import app

    crm = _CRM()
    original_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[funnel_mod._crm_dependency] = lambda: crm
    app.dependency_overrides[auth_mod.get_eom_funnel_api_config] = _enabled_config
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            public_response = await client.get("/api/v1/leads/intake")
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
    assert response.status_code == 201
    assert crm.calls


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
    assert "has_schema_privilege" in ready_pool.queries[0]
    assert "has_table_privilege" in ready_pool.queries[0]
    assert "has_column_privilege" in ready_pool.queries[0]

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
async def test_full_app_lifespan_executes_enabled_preflight_before_handoff_request(monkeypatch):
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
