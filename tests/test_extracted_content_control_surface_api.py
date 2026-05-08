import pytest

import extracted_content_pipeline.api.control_surfaces as api_module
from extracted_content_pipeline.api.control_surfaces import (
    ContentOpsControlSurfaceApiConfig,
    create_content_ops_control_surface_router,
)
from extracted_content_pipeline.content_ops_execution import ContentOpsExecutionServices


pytestmark = pytest.mark.skipif(
    api_module.APIRouter is None,
    reason="fastapi is not installed in this test environment",
)


def _route(router, path: str, method: str):
    for route in router.routes:
        if getattr(route, "path", None) == path and method.upper() in getattr(route, "methods", set()):
            return route
    raise AssertionError(f"route not found: {method} {path}")


class _CampaignService:
    def __init__(self):
        self.calls = []

    async def generate(self, *, scope, target_mode, limit=None, filters=None, **kwargs):
        self.calls.append({
            "scope": scope,
            "target_mode": target_mode,
            "limit": limit,
            "filters": dict(filters or {}),
            "kwargs": dict(kwargs),
        })
        return {"generated": 1, "saved_ids": ["draft-1"]}


class _FailingCampaignService:
    async def generate(self, **kwargs):
        del kwargs
        raise RuntimeError("postgres://user:secret@example/internal")


@pytest.mark.asyncio
async def test_describe_control_surfaces_route_returns_catalog_and_presets():
    router = create_content_ops_control_surface_router()

    route = _route(router, "/content-ops/control-surfaces", "GET")
    payload = await route.endpoint()

    outputs = {item["id"]: item for item in payload["outputs"]}
    output_ids = set(outputs)
    preset_ids = {item["id"] for item in payload["presets"]}
    assert "email_campaign" in output_ids
    assert "landing_page" in output_ids
    assert outputs["email_campaign"]["execution_configured"] is False
    assert outputs["email_campaign"]["can_execute"] is False
    assert outputs["email_campaign"]["estimated_unit_cost_usd"] == 0.18
    assert outputs["email_campaign"]["default_parse_retry_attempts"] == 1
    assert outputs["email_campaign"]["estimated_retry_adjusted_unit_cost_usd"] == 0.36
    assert payload["execution"] == {"configured": False, "configured_outputs": []}
    assert "email_only" in preset_ids
    assert "lead_gen_campaign" in preset_ids
    assert payload["ingestion_profiles"] == [
        "domain_specific",
        "manual",
        "existing_evidence",
    ]


@pytest.mark.asyncio
async def test_describe_control_surfaces_reports_configured_execution_services():
    router = create_content_ops_control_surface_router(
        execution_services_provider=lambda: ContentOpsExecutionServices(
            campaign=_CampaignService(),
            report=_CampaignService(),
        )
    )

    route = _route(router, "/content-ops/control-surfaces", "GET")
    payload = await route.endpoint()

    outputs = {item["id"]: item for item in payload["outputs"]}
    assert payload["execution"] == {
        "configured": True,
        "configured_outputs": ["email_campaign", "report"],
    }
    assert outputs["email_campaign"]["execution_configured"] is True
    assert outputs["email_campaign"]["can_execute"] is True
    assert outputs["report"]["execution_configured"] is True
    assert outputs["report"]["can_execute"] is True
    assert outputs["blog_post"]["execution_configured"] is False
    assert outputs["blog_post"]["can_execute"] is False


@pytest.mark.asyncio
async def test_describe_control_surfaces_requires_generate_method_for_readiness():
    router = create_content_ops_control_surface_router(
        execution_services_provider=lambda: ContentOpsExecutionServices(
            campaign=object(),
            report=_CampaignService(),
        )
    )

    route = _route(router, "/content-ops/control-surfaces", "GET")
    payload = await route.endpoint()

    outputs = {item["id"]: item for item in payload["outputs"]}
    assert payload["execution"] == {
        "configured": True,
        "configured_outputs": ["report"],
    }
    assert outputs["email_campaign"]["execution_configured"] is False
    assert outputs["email_campaign"]["can_execute"] is False
    assert outputs["report"]["execution_configured"] is True
    assert outputs["report"]["can_execute"] is True


@pytest.mark.asyncio
async def test_describe_control_surfaces_ignores_invalid_execution_provider_result():
    router = create_content_ops_control_surface_router(
        execution_services_provider=lambda: object()
    )

    route = _route(router, "/content-ops/control-surfaces", "GET")
    payload = await route.endpoint()

    outputs = {item["id"]: item for item in payload["outputs"]}
    assert payload["execution"] == {"configured": False, "configured_outputs": []}
    assert outputs["email_campaign"]["execution_configured"] is False
    assert outputs["email_campaign"]["can_execute"] is False


@pytest.mark.asyncio
async def test_preview_generation_route_returns_preflight_plan():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
    )

    route = _route(router, "/ops/preview", "POST")
    payload = await route.endpoint(
        {
            "outputs": ["email_campaign"],
            "inputs": {
                "target_account": "Acme",
                "offer": "Churn audit",
            },
            "max_cost_usd": 1.0,
        }
    )

    assert payload["can_run"] is True
    assert payload["outputs"] == ["email_campaign"]
    assert payload["estimated_cost_usd"] == 0.36
    assert payload["missing_inputs"] == []


@pytest.mark.asyncio
async def test_preview_generation_route_blocks_budget_between_base_and_retry_cost():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
    )

    route = _route(router, "/ops/preview", "POST")
    payload = await route.endpoint(
        {
            "outputs": ["email_campaign"],
            "inputs": {
                "target_account": "Acme",
                "offer": "Churn audit",
            },
            "max_cost_usd": 0.22,
        }
    )

    assert payload["can_run"] is False
    assert payload["estimated_cost_usd"] == 0.36
    assert "Estimated cost exceeds max_cost_usd: 0.36 > 0.22" in payload["warnings"]


@pytest.mark.asyncio
async def test_plan_generation_route_returns_execution_plan():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
    )

    route = _route(router, "/ops/plan", "POST")
    payload = await route.endpoint(
        {
            "outputs": ["email_campaign"],
            "inputs": {
                "target_account": "Acme",
                "offer": "Churn audit",
            },
            "max_cost_usd": 1.0,
        }
    )

    assert payload["can_execute"] is True
    assert payload["steps"][0]["runner"] == "CampaignGenerationService.generate"
    assert payload["steps"][0]["status"] == "runnable"
    assert payload["preview"]["can_run"] is True


@pytest.mark.asyncio
async def test_execute_generation_route_runs_configured_services():
    service = _CampaignService()
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        execution_services_provider=lambda: ContentOpsExecutionServices(campaign=service),
        scope_provider=lambda: {
            "account_id": " acct-1 ",
            "allowed_vendors": [" Acme ", "", "   "],
            "roles": [" admin ", ""],
        },
    )

    route = _route(router, "/ops/execute", "POST")
    payload = await route.endpoint(
        {
            "outputs": ["email_campaign"],
            "limit": 2,
            "inputs": {
                "target_account": "Acme",
                "offer": "Churn audit",
                "filters": {"status": "ready"},
            },
        }
    )

    assert payload["status"] == "completed"
    assert payload["steps"][0]["result"] == {"generated": 1, "saved_ids": ["draft-1"]}
    assert service.calls[0]["scope"].account_id == "acct-1"
    assert service.calls[0]["scope"].allowed_vendors == ("Acme",)
    assert service.calls[0]["scope"].roles == ("admin",)
    assert service.calls[0]["target_mode"] == "vendor_retention"
    assert service.calls[0]["limit"] == 2
    assert service.calls[0]["filters"] == {"status": "ready"}


@pytest.mark.asyncio
async def test_execute_generation_route_requires_configured_services():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
    )

    route = _route(router, "/ops/execute", "POST")
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint(
            {
                "outputs": ["email_campaign"],
                "inputs": {
                    "target_account": "Acme",
                    "offer": "Churn audit",
                },
            }
        )

    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_execute_generation_route_sanitizes_service_failures():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            campaign=_FailingCampaignService()
        ),
    )

    route = _route(router, "/ops/execute", "POST")
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint(
            {
                "outputs": ["email_campaign"],
                "inputs": {
                    "target_account": "Acme",
                    "offer": "Churn audit",
                },
            }
        )

    assert exc.value.status_code == 207
    assert exc.value.detail["errors"] == [
        {"output": "email_campaign", "reason": "execution_failed"}
    ]
    assert exc.value.detail["steps"][0]["error"] == "execution_failed"
    assert "postgres://" not in str(exc.value.detail)


@pytest.mark.asyncio
async def test_execute_generation_route_wraps_execution_provider_exception(caplog):
    def provider():
        raise RuntimeError("postgres://user:secret@example/internal")

    caplog.set_level("WARNING", logger="extracted_content_pipeline.api.control_surfaces")
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        execution_services_provider=provider,
    )

    route = _route(router, "/ops/execute", "POST")
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint(
            {
                "outputs": ["email_campaign"],
                "inputs": {
                    "target_account": "Acme",
                    "offer": "Churn audit",
                },
            }
        )

    assert exc.value.status_code == 503
    assert exc.value.detail == "Content Ops execution services are unavailable."
    assert "postgres://" not in caplog.text


@pytest.mark.asyncio
async def test_execute_generation_route_wraps_scope_provider_exception(caplog):
    def scope_provider():
        raise RuntimeError("postgres://user:secret@example/internal")

    caplog.set_level("WARNING", logger="extracted_content_pipeline.api.control_surfaces")
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            campaign=_CampaignService()
        ),
        scope_provider=scope_provider,
    )

    route = _route(router, "/ops/execute", "POST")
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint(
            {
                "outputs": ["email_campaign"],
                "inputs": {
                    "target_account": "Acme",
                    "offer": "Churn audit",
                },
            }
        )

    assert exc.value.status_code == 503
    assert exc.value.detail == "Content Ops scope provider is unavailable."
    assert "postgres://" not in caplog.text


@pytest.mark.asyncio
async def test_preview_generation_route_rejects_invalid_payload_shape():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
    )

    route = _route(router, "/ops/preview", "POST")
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint(
            {
                "outputs": ["email_campaign"],
                "limit": 0,
                "unexpected": "not allowed",
                "inputs": {
                    "target_account": "Acme",
                    "offer": "Churn audit",
                },
            }
        )

    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_execute_generation_route_rejects_invalid_execution_provider_result():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        execution_services_provider=lambda: object(),
    )

    route = _route(router, "/ops/execute", "POST")
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint(
            {
                "outputs": ["email_campaign"],
                "inputs": {
                    "target_account": "Acme",
                    "offer": "Churn audit",
                },
            }
        )

    assert exc.value.status_code == 503


def test_config_requires_absolute_prefix():
    with pytest.raises(ValueError, match="prefix must start with /"):
        ContentOpsControlSurfaceApiConfig(prefix="content-ops")
