import pytest

import extracted_content_pipeline.api.control_surfaces as api_module
from extracted_content_pipeline.api.control_surfaces import (
    ContentOpsControlSurfaceApiConfig,
    create_content_ops_control_surface_router,
)


pytestmark = pytest.mark.skipif(
    api_module.APIRouter is None,
    reason="fastapi is not installed in this test environment",
)


def _route(router, path: str, method: str):
    for route in router.routes:
        if getattr(route, "path", None) == path and method.upper() in getattr(route, "methods", set()):
            return route
    raise AssertionError(f"route not found: {method} {path}")


@pytest.mark.asyncio
async def test_describe_control_surfaces_route_returns_catalog_and_presets():
    router = create_content_ops_control_surface_router()

    route = _route(router, "/content-ops/control-surfaces", "GET")
    payload = await route.endpoint()

    output_ids = {item["id"] for item in payload["outputs"]}
    preset_ids = {item["id"] for item in payload["presets"]}
    assert "email_campaign" in output_ids
    assert "landing_page" in output_ids
    assert "email_only" in preset_ids
    assert "lead_gen_campaign" in preset_ids
    assert payload["ingestion_profiles"] == [
        "domain_specific",
        "manual",
        "existing_evidence",
    ]


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
    assert payload["estimated_cost_usd"] == 0.18
    assert payload["missing_inputs"] == []


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


def test_config_requires_absolute_prefix():
    with pytest.raises(ValueError, match="prefix must start with /"):
        ContentOpsControlSurfaceApiConfig(prefix="content-ops")
