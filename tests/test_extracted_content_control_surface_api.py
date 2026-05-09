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
    assert outputs["signal_extraction"]["implemented"] is True
    assert outputs["email_campaign"]["execution_configured"] is False
    assert outputs["email_campaign"]["can_execute"] is False
    assert outputs["email_campaign"]["estimated_unit_cost_usd"] == 0.18
    assert outputs["email_campaign"]["default_parse_retry_attempts"] == 1
    assert outputs["email_campaign"]["estimated_retry_adjusted_unit_cost_usd"] == 0.36
    assert outputs["email_campaign"]["reasoning_requirement"] == "optional_host_context"
    assert outputs["blog_post"]["reasoning_requirement"] == "optional_host_context"
    assert outputs["signal_extraction"]["reasoning_requirement"] == "absent"
    assert payload["execution"] == {"configured": False, "configured_outputs": []}
    assert payload["reasoning"] == {"configured": False}
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
            signal_extraction=_CampaignService(),
        )
    )

    route = _route(router, "/content-ops/control-surfaces", "GET")
    payload = await route.endpoint()

    outputs = {item["id"]: item for item in payload["outputs"]}
    assert payload["execution"] == {
        "configured": True,
        "configured_outputs": ["email_campaign", "report", "signal_extraction"],
    }
    assert outputs["email_campaign"]["execution_configured"] is True
    assert outputs["email_campaign"]["can_execute"] is True
    assert outputs["report"]["execution_configured"] is True
    assert outputs["report"]["can_execute"] is True
    assert outputs["signal_extraction"]["execution_configured"] is True
    assert outputs["signal_extraction"]["can_execute"] is True
    assert outputs["blog_post"]["execution_configured"] is False
    assert outputs["blog_post"]["can_execute"] is False
    assert payload["reasoning"] == {"configured": False}


@pytest.mark.asyncio
async def test_describe_control_surfaces_reports_reasoning_provider_status():
    router = create_content_ops_control_surface_router(
        reasoning_context_provider=lambda: object()
    )

    route = _route(router, "/content-ops/control-surfaces", "GET")
    payload = await route.endpoint()

    assert payload["execution"] == {"configured": False, "configured_outputs": []}
    assert payload["reasoning"] == {"configured": True}


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
    assert payload["reasoning"] == {"configured": False}
    assert outputs["email_campaign"]["execution_configured"] is False
    assert outputs["email_campaign"]["can_execute"] is False


# -----------------------
# PR-Describe-Control-Surfaces-Cache: static portion is cached at import.
# Verify two consecutive calls return mutually independent dicts (no
# aliasing into the cache) and that the cache is computed once.
# -----------------------


@pytest.mark.asyncio
async def test_describe_control_surfaces_returns_independent_dict_per_call():
    """Per-call mutation must not leak into the next call's response."""

    router = create_content_ops_control_surface_router()
    route = _route(router, "/content-ops/control-surfaces", "GET")

    first = await route.endpoint()
    first["outputs"][0]["label"] = "MUTATED"
    first["outputs"][0]["required_inputs"].append("injected_field")
    first["presets"][0]["outputs"].append("injected_output")
    first["ingestion_profiles"].append("injected_profile")

    second = await route.endpoint()
    assert second["outputs"][0]["label"] != "MUTATED"
    assert "injected_field" not in second["outputs"][0]["required_inputs"]
    assert "injected_output" not in second["presets"][0]["outputs"]
    assert "injected_profile" not in second["ingestion_profiles"]


@pytest.mark.asyncio
async def test_describe_control_surfaces_static_cache_is_not_rebuilt_per_request(monkeypatch):
    """``_build_static_catalog_payload`` is invoked at import, not per
    request. The spy is installed after import, so this asserts the
    builder is not re-invoked per request (not the import-time call)."""

    call_count = {"n": 0}
    original = api_module._build_static_catalog_payload

    def _spy() -> object:
        call_count["n"] += 1
        return original()

    monkeypatch.setattr(api_module, "_build_static_catalog_payload", _spy)

    router = create_content_ops_control_surface_router()
    route = _route(router, "/content-ops/control-surfaces", "GET")
    await route.endpoint()
    await route.endpoint()

    assert call_count["n"] == 0


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
async def test_plan_generation_route_rejects_invalid_signal_text_cap_as_400():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
    )

    route = _route(router, "/ops/plan", "POST")
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint(
            {
                "outputs": ["signal_extraction"],
                "inputs": {
                    "source_material": "Pricing pressure came up at renewal.",
                    "source_max_text_chars": 0,
                },
            }
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "source_max_text_chars must be at least 1; got 0"


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
async def test_execute_generation_route_rejects_invalid_signal_text_cap_as_400():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            signal_extraction=_CampaignService()
        ),
    )

    route = _route(router, "/ops/execute", "POST")
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint(
            {
                "outputs": ["signal_extraction"],
                "inputs": {
                    "source_material": "Pricing pressure came up at renewal.",
                    "source_max_text_chars": 0,
                },
            }
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "source_max_text_chars must be at least 1; got 0"


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

    assert exc.value.status_code == 502
    assert exc.value.detail["errors"] == [
        {
            "output": "email_campaign",
            "runner": "CampaignGenerationService.generate",
            "error": "execution_failed",
            "reason": "execution_failed",
        }
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


# -----------------------
# PR-ControlSurfaces-Reasoning-Provider: route-level reasoning seam
# -----------------------


class _ReasoningCapturingService:
    """Records the reasoning provider seen at generate() time."""

    def __init__(self, reasoning_context=None):
        self._reasoning_context = reasoning_context
        self.calls = []

    def with_reasoning_context(self, provider):
        return _ReasoningCapturingService(reasoning_context=provider)

    async def generate(self, *, scope, target_mode, limit=None, filters=None, **kwargs):
        self.calls.append({
            "reasoning_context": self._reasoning_context,
            "scope": scope,
            "target_mode": target_mode,
            "limit": limit,
            "kwargs": dict(kwargs),
        })
        return {"generated": 1, "saved_ids": ["draft-1"]}


@pytest.mark.asyncio
async def test_execute_route_threads_reasoning_provider_into_services():
    """A configured ``reasoning_context_provider`` reaches the service
    that the executor invokes for the request."""

    base = _ReasoningCapturingService()
    sentinel = object()  # acts as the resolved reasoning provider

    router = create_content_ops_control_surface_router(
        execution_services_provider=lambda: ContentOpsExecutionServices(
            campaign=base
        ),
        reasoning_context_provider=lambda: sentinel,
    )

    route = _route(router, "/content-ops/execute", "POST")
    await route.endpoint(
        {
            "outputs": ["email_campaign"],
            "inputs": {"target_account": "Acme", "offer": "Audit"},
        }
    )

    # The base service is unchanged (no mutation); a derived service
    # carrying the sentinel was constructed and invoked.
    assert base.calls == []  # original instance untouched
    # The derived service was constructed via with_reasoning_context;
    # we can't reach it directly, but its presence is visible through
    # the route succeeding (no service_not_configured error). The
    # route's behavior already implies the wiring; the unit-level
    # bundle test in test_extracted_content_ops_execution.py asserts
    # the mechanical detail.


@pytest.mark.asyncio
async def test_execute_route_without_reasoning_provider_passes_services_unchanged():
    """When no ``reasoning_context_provider`` is supplied, the bundle
    is not derived and the original services receive the call."""

    base = _ReasoningCapturingService()
    router = create_content_ops_control_surface_router(
        execution_services_provider=lambda: ContentOpsExecutionServices(
            campaign=base
        ),
        # No reasoning_context_provider.
    )

    route = _route(router, "/content-ops/execute", "POST")
    await route.endpoint(
        {
            "outputs": ["email_campaign"],
            "inputs": {"target_account": "Acme", "offer": "Audit"},
        }
    )

    # The original instance handled the request; no wrapping happened.
    assert len(base.calls) == 1
    assert base.calls[0]["reasoning_context"] is None


@pytest.mark.asyncio
async def test_execute_route_reasoning_provider_returning_none_rebinds_to_none():
    """Reviewer-flagged plan-vs-code divergence: when the host wires a
    ``reasoning_context_provider`` that resolves to ``None`` for
    tenant-policy reasons, the bundle is derived with reasoning
    rebound to ``None`` -- not silently bypassed. Otherwise
    construction-time reasoning would leak through.

    Verifies the fix gates derivation on whether the kwarg was
    supplied (not on the resolved value).
    """

    sentinel_construction_time = object()
    base = _ReasoningCapturingService(reasoning_context=sentinel_construction_time)

    router = create_content_ops_control_surface_router(
        execution_services_provider=lambda: ContentOpsExecutionServices(
            campaign=base
        ),
        # Kwarg IS supplied, but resolves to None per request.
        reasoning_context_provider=lambda: None,
    )

    route = _route(router, "/content-ops/execute", "POST")
    await route.endpoint(
        {
            "outputs": ["email_campaign"],
            "inputs": {"target_account": "Acme", "offer": "Audit"},
        }
    )

    # Construction-time reasoning is NOT leaked through; the derived
    # bundle was constructed with rebind-to-None. The base instance
    # itself is unchanged (cached service stays intact).
    assert base.calls == []  # original untouched, kept its construction-time reasoning
    assert base._reasoning_context is sentinel_construction_time  # preserved
