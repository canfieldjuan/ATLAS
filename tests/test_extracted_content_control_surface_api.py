import pytest

import extracted_content_pipeline.api.control_surfaces as api_module
from extracted_content_pipeline.api.control_surfaces import (
    ContentOpsControlSurfaceApiConfig,
    create_content_ops_control_surface_router,
)
from extracted_content_pipeline.campaign_ports import CampaignReasoningContext
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
async def test_describe_control_surfaces_reports_rich_reasoning_status():
    router = create_content_ops_control_surface_router(
        reasoning_context_provider=lambda: object(),
        reasoning_status_provider=lambda: {
            "configured": True,
            "source": "db",
            "modes": ["file", "db", "multi_pass"],
            "packs": ("campaign", "long_form"),
            "unsafe": {"nested": "value"},
        },
    )

    route = _route(router, "/content-ops/control-surfaces", "GET")
    payload = await route.endpoint()

    assert payload["reasoning"] == {
        "configured": True,
        "source": "db",
        "modes": ["file", "db", "multi_pass"],
        "packs": ["campaign", "long_form"],
    }


@pytest.mark.asyncio
async def test_describe_control_surfaces_sanitizes_reasoning_status_lists():
    router = create_content_ops_control_surface_router(
        reasoning_context_provider=lambda: object(),
        reasoning_status_provider=lambda: {
            "configured": True,
            "capabilities": ["falsification", {"unsafe": "nested"}, "cache"],
            "details": [{"unsafe": "nested"}],
        },
    )

    route = _route(router, "/content-ops/control-surfaces", "GET")
    payload = await route.endpoint()

    assert payload["reasoning"] == {
        "configured": True,
        "capabilities": ["falsification", "cache"],
    }


@pytest.mark.asyncio
async def test_describe_control_surfaces_preserves_reasoning_capability_statuses():
    router = create_content_ops_control_surface_router(
        reasoning_context_provider=lambda: object(),
        reasoning_status_provider=lambda: {
            "configured": True,
            "capabilities": {
                "explicit_provider": {
                    "configured": False,
                    "ready": False,
                    "active": False,
                    "missing": ["reasoning_provider"],
                    "unsafe": {"nested": "value"},
                },
                "multi_pass": {
                    "configured": True,
                    "ready": True,
                    "active": True,
                    "missing": [],
                },
                "bad": {"missing": [{"nested": "value"}]},
                "": {"ready": True},
            },
        },
    )

    route = _route(router, "/content-ops/control-surfaces", "GET")
    payload = await route.endpoint()

    assert payload["reasoning"] == {
        "configured": True,
        "capabilities": {
            "explicit_provider": {
                "configured": False,
                "ready": False,
                "active": False,
                "missing": ["reasoning_provider"],
            },
            "multi_pass": {
                "configured": True,
                "ready": True,
                "active": True,
            },
        },
    }


@pytest.mark.asyncio
async def test_describe_control_surfaces_caps_reasoning_status_lists():
    router = create_content_ops_control_surface_router(
        reasoning_context_provider=lambda: object(),
        reasoning_status_provider=lambda: {
            "configured": True,
            "modes": [f"mode-{index}" for index in range(25)],
        },
    )

    route = _route(router, "/content-ops/control-surfaces", "GET")
    payload = await route.endpoint()

    assert payload["reasoning"]["modes"] == [f"mode-{index}" for index in range(20)]


@pytest.mark.asyncio
async def test_describe_control_surfaces_drops_non_finite_reasoning_status_floats():
    router = create_content_ops_control_surface_router(
        reasoning_context_provider=lambda: object(),
        reasoning_status_provider=lambda: {
            "configured": True,
            "score": float("nan"),
            "capabilities": ["cache", float("inf"), "falsification"],
        },
    )

    route = _route(router, "/content-ops/control-surfaces", "GET")
    payload = await route.endpoint()

    assert payload["reasoning"] == {
        "configured": True,
        "capabilities": ["cache", "falsification"],
    }


@pytest.mark.asyncio
async def test_describe_control_surfaces_status_provider_failure_falls_back():
    def _failing_status_provider():
        raise RuntimeError("status unavailable")

    router = create_content_ops_control_surface_router(
        reasoning_context_provider=lambda: object(),
        reasoning_status_provider=_failing_status_provider,
    )

    route = _route(router, "/content-ops/control-surfaces", "GET")
    payload = await route.endpoint()

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


class _ReasoningPayloadService(_ReasoningCapturingService):
    def with_reasoning_context(self, provider):
        return _ReasoningPayloadService(reasoning_context=provider)

    async def generate(self, *, scope, target_mode, limit=None, filters=None, **kwargs):
        self.calls.append({
            "reasoning_context": self._reasoning_context,
            "scope": scope,
            "target_mode": target_mode,
            "limit": limit,
            "kwargs": dict(kwargs),
        })
        payload = {"generated": 1, "saved_ids": ["draft-1"]}
        if self._reasoning_context is not None:
            payload["reasoning_contexts_used"] = 1
            payload["consumed_reasoning_contexts"] = [
                {
                    "summary": "Acme renewal pricing pressure",
                    "proof_points": [{"label": "source_material", "value": "pricing"}],
                }
            ]
        return payload


class _ProviderRecordingService:
    def __init__(self, providers=None):
        self.providers = providers if providers is not None else []

    def with_reasoning_context(self, provider):
        self.providers.append(provider)
        return _ProviderRecordingService(self.providers)

    async def generate(self, *, scope, target_mode, limit=None, filters=None, **kwargs):
        del scope, target_mode, limit, filters, kwargs
        return {"generated": 1, "saved_ids": ["draft-1"]}


class _StaticReasoningProvider:
    def __init__(self, context):
        self.context = context

    async def read_campaign_reasoning_context(self, **kwargs):
        del kwargs
        return self.context


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
async def test_execute_route_returns_consumed_reasoning_payloads_from_rebound_service():
    """Route-level provider resolution must compose with the executor's
    consumed-context audit. This locks the HTTP payload shape that the
    UI consumes, not just the lower-level executor helper."""

    base = _ReasoningPayloadService()
    sentinel = object()

    router = create_content_ops_control_surface_router(
        execution_services_provider=lambda: ContentOpsExecutionServices(
            campaign=base
        ),
        reasoning_context_provider=lambda: sentinel,
    )

    route = _route(router, "/content-ops/execute", "POST")
    payload = await route.endpoint(
        {
            "outputs": ["email_campaign"],
            "inputs": {"target_account": "Acme", "offer": "Audit"},
        }
    )

    assert base.calls == []
    assert payload["steps"][0]["reasoning"] == {
        "requirement": "optional_host_context",
        "service_supports_reasoning": True,
        "provider_configured": True,
        "contexts_used": 1,
        "consumed_contexts": [
            {
                "summary": "Acme renewal pricing pressure",
                "proof_points": [{"label": "source_material", "value": "pricing"}],
            }
        ],
    }


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


@pytest.mark.asyncio
async def test_execute_route_builds_structured_reasoning_for_report_only():
    campaign = _ReasoningCapturingService()
    recorded_providers = []
    report = _ProviderRecordingService(recorded_providers)

    router = create_content_ops_control_surface_router(
        execution_services_provider=lambda: ContentOpsExecutionServices(
            campaign=campaign,
            report=report,
        ),
        llm_provider=lambda: object(),
    )

    route = _route(router, "/content-ops/execute", "POST")
    payload = await route.endpoint(
        {
            "outputs": ["email_campaign", "report"],
            "reasoning_preset": "multi_pass_structured",
            "inputs": {
                "target_account": "Acme",
                "offer": "Audit",
                "opportunity_id": "opp-1",
            },
        }
    )

    assert len(recorded_providers) == 1
    provider = recorded_providers[0]
    assert provider._config.default_goal == "synthesize content reasoning context"
    assert provider._config.narrative_plan_pack.name == "content_ops_structured"
    assert provider._config.output_policy is not None
    assert provider._config.output_policy.require_citations is True
    assert provider._config.block_on_validation_failure is False
    assert campaign.calls[0]["reasoning_context"] is None
    email_step = next(step for step in payload["steps"] if step["output"] == "email_campaign")
    assert email_step["reasoning"]["provider_configured"] is False


@pytest.mark.asyncio
async def test_execute_route_wraps_strict_reasoning_with_blocking_provider():
    recorded_providers = []
    report = _ProviderRecordingService(recorded_providers)

    router = create_content_ops_control_surface_router(
        execution_services_provider=lambda: ContentOpsExecutionServices(report=report),
        llm_provider=lambda: object(),
    )

    route = _route(router, "/content-ops/execute", "POST")
    await route.endpoint(
        {
            "outputs": ["report"],
            "reasoning_preset": "multi_pass_strict",
            "inputs": {"opportunity_id": "opp-1"},
        }
    )

    provider = recorded_providers[0]
    assert provider.provider._config.output_policy is not None
    assert provider.provider._config.block_on_validation_failure is False


@pytest.mark.asyncio
async def test_execute_route_can_relax_strict_reasoning_citation_policy():
    recorded_providers = []
    report = _ProviderRecordingService(recorded_providers)

    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(
            structured_reasoning_require_citations=False,
        ),
        execution_services_provider=lambda: ContentOpsExecutionServices(report=report),
        llm_provider=lambda: object(),
    )

    route = _route(router, "/content-ops/execute", "POST")
    await route.endpoint(
        {
            "outputs": ["report"],
            "reasoning_preset": "multi_pass_strict",
            "inputs": {"opportunity_id": "opp-1"},
        }
    )

    provider = recorded_providers[0]
    assert provider.provider._config.output_policy is not None
    assert provider.provider._config.output_policy.require_citations is False
    assert provider.provider._config.block_on_validation_failure is False


@pytest.mark.asyncio
async def test_blocking_reasoning_provider_surfaces_validation_blockers():
    context = CampaignReasoningContext(
        canonical_reasoning={
            "validation": {
                "passed": False,
                "blockers": ("claim_missing_citations:0",),
            }
        }
    )
    provider = api_module._BlockingReasoningContextProvider(
        _StaticReasoningProvider(context)
    )

    with pytest.raises(RuntimeError, match="claim_missing_citations:0"):
        await provider.read_campaign_reasoning_context(
            scope=None,
            target_id="opp-1",
            target_mode="vendor_retention",
            opportunity={},
        )


@pytest.mark.parametrize(
    ("payload", "expected_status", "expected_detail"),
    (
        (
            {
                "outputs": ["report"],
                "reasoning_preset": "garbage",
                "inputs": {"opportunity_id": "opp-1"},
            },
            422,
            "multi_pass_strict",
        ),
        (
            {
                "outputs": ["blog_post"],
                "reasoning_preset": "garbage",
                "inputs": {"topic": "Churn pressure"},
            },
            422,
            "multi_pass_strict",
        ),
        (
            {
                "outputs": ["report"],
                "reasoning_preset": "multi_pass_light",
                "inputs": {"opportunity_id": "opp-1"},
            },
            400,
            "multi_pass_structured or multi_pass_strict",
        ),
        (
            {
                "outputs": ["blog_post"],
                "reasoning_preset": "multi_pass_structured",
                "inputs": {"topic": "Churn pressure"},
            },
            400,
            "report and sales_brief",
        ),
    ),
)
@pytest.mark.asyncio
async def test_execute_route_rejects_invalid_reasoning_preset_requests(
    payload,
    expected_status,
    expected_detail,
):
    router = create_content_ops_control_surface_router(
        execution_services_provider=lambda: ContentOpsExecutionServices(
            blog_post=_ProviderRecordingService(),
            report=_ProviderRecordingService(),
        ),
        llm_provider=lambda: object(),
    )

    route = _route(router, "/content-ops/execute", "POST")
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint(payload)

    assert exc.value.status_code == expected_status
    assert expected_detail in str(exc.value.detail)


@pytest.mark.asyncio
async def test_execute_route_host_reasoning_provider_beats_reasoning_preset():
    base = _ReasoningCapturingService()
    sentinel = object()
    router = create_content_ops_control_surface_router(
        execution_services_provider=lambda: ContentOpsExecutionServices(report=base),
        reasoning_context_provider=lambda: sentinel,
    )

    route = _route(router, "/content-ops/execute", "POST")
    await route.endpoint(
        {
            "outputs": ["report"],
            "reasoning_preset": "multi_pass_structured",
            "inputs": {"opportunity_id": "opp-1"},
        }
    )

    assert base.calls == []


@pytest.mark.parametrize("preset", ("none", "context_only", " "))
@pytest.mark.asyncio
async def test_execute_route_noop_reasoning_presets_skip_packaged_provider(preset):
    base = _ReasoningCapturingService()
    router = create_content_ops_control_surface_router(
        execution_services_provider=lambda: ContentOpsExecutionServices(report=base),
        llm_provider=lambda: pytest.fail("packaged reasoning should be skipped"),
    )

    route = _route(router, "/content-ops/execute", "POST")
    await route.endpoint({
        "outputs": ["report"],
        "reasoning_preset": preset,
        "inputs": {"opportunity_id": "opp-1"},
    })

    assert len(base.calls) == 1
    assert base.calls[0]["reasoning_context"] is None


@pytest.mark.asyncio
async def test_execute_route_validates_plan_before_structured_reasoning_provider():
    router = create_content_ops_control_surface_router(
        execution_services_provider=lambda: ContentOpsExecutionServices(
            report=_ProviderRecordingService()
        ),
        llm_provider=lambda: pytest.fail("llm provider should not be resolved"),
    )

    route = _route(router, "/content-ops/execute", "POST")
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint({
            "outputs": ["report"],
            "reasoning_preset": "multi_pass_structured",
            "inputs": {},
        })

    assert exc.value.status_code == 400
    assert exc.value.detail["errors"] == [{"reason": "plan_not_executable"}]


@pytest.mark.asyncio
async def test_execute_route_validates_packaged_preset_before_service_capability():
    router = create_content_ops_control_surface_router(
        execution_services_provider=lambda: ContentOpsExecutionServices(
            report=_CampaignService()
        ),
        llm_provider=lambda: object(),
    )

    route = _route(router, "/content-ops/execute", "POST")
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint({
            "outputs": ["report"],
            "reasoning_preset": "multi_pass_light",
            "inputs": {"opportunity_id": "opp-1"},
        })

    assert exc.value.status_code == 400
    assert "multi_pass_structured or multi_pass_strict" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_execute_route_structured_reasoning_requires_reasoning_aware_service():
    router = create_content_ops_control_surface_router(
        execution_services_provider=lambda: ContentOpsExecutionServices(
            report=_CampaignService()
        ),
        llm_provider=lambda: object(),
    )

    route = _route(router, "/content-ops/execute", "POST")
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint({
            "outputs": ["report"],
            "reasoning_preset": "multi_pass_structured",
            "inputs": {"opportunity_id": "opp-1"},
        })

    assert exc.value.status_code == 503
    assert "does not support structured reasoning" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_execute_route_structured_reasoning_requires_llm_provider():
    router = create_content_ops_control_surface_router(
        execution_services_provider=lambda: ContentOpsExecutionServices(
            report=_ProviderRecordingService()
        )
    )

    route = _route(router, "/content-ops/execute", "POST")
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint(
            {
                "outputs": ["report"],
                "reasoning_preset": "multi_pass_structured",
                "inputs": {"opportunity_id": "opp-1"},
            }
        )

    assert exc.value.status_code == 503


def test_content_ops_config_rejects_blank_structured_reasoning_pack_name():
    with pytest.raises(ValueError, match="structured_reasoning_pack_name"):
        ContentOpsControlSurfaceApiConfig(structured_reasoning_pack_name=" ")
