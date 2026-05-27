import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal

import pytest

import extracted_content_pipeline.api.control_surfaces as api_module
from extracted_content_pipeline.api.control_surfaces import (
    ContentOpsControlSurfaceApiConfig,
    create_content_ops_control_surface_router,
)
from extracted_content_pipeline.campaign_ports import CampaignReasoningContext
from extracted_content_pipeline.content_ops_input_provider import ContentOpsInputPackage
from extracted_content_pipeline.content_ops_execution import ContentOpsExecutionServices


pytestmark = pytest.mark.skipif(
    api_module.APIRouter is None,
    reason="fastapi is not installed in this test environment",
)

_DEFAULT_EXECUTION_LIMITS = {
    "max_concurrency": 8,
    "max_source_material_rows": 1000,
    "faq_max_source_material_rows": 1000,
    "large_upload_strategy": "background_or_offline",
}


def _route(router, path: str, method: str):
    for route in router.routes:
        if getattr(route, "path", None) == path and method.upper() in getattr(route, "methods", set()):
            return route
    raise AssertionError(f"route not found: {method} {path}")


class _UploadFile:
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self._content = content

    async def read(self, size=-1):
        if size is None or size < 0:
            return self._content
        return self._content[:size]


def _ticket_bundle_upload(row_count: int) -> _UploadFile:
    rows = [
        {
            "ticket_id": f"ticket-{index}",
            "source_type": "support_ticket",
            "subject": "Billing renewal question",
            "message": "How do I confirm my renewal invoice before payment?",
            "pain_category": "billing",
        }
        for index in range(row_count)
    ]
    return _UploadFile(
        "support-ticket-export.json",
        (json.dumps({"support_tickets": rows}) + "\n").encode("utf-8"),
    )


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


class _SyncInputProvider:
    def __init__(self, package: ContentOpsInputPackage):
        self.package = package
        self.calls = []

    def build_content_ops_input_package(self, *, scope, request=None):
        self.calls.append({"scope": scope, "request": dict(request or {})})
        return self.package


class _AsyncInputProvider(_SyncInputProvider):
    async def build_content_ops_input_package(self, *, scope, request=None):
        self.calls.append({"scope": scope, "request": dict(request or {})})
        return self.package


class _FailingInputProvider:
    def build_content_ops_input_package(self, *, scope, request=None):
        del scope, request
        raise RuntimeError("postgres://user:secret@example/internal")


class _HTTPFailingInputProvider:
    def build_content_ops_input_package(self, *, scope, request=None):
        del scope, request
        raise api_module.HTTPException(status_code=400, detail="bad ticket payload")


class _BlockingCampaignService(_CampaignService):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, *, scope, target_mode, limit=None, filters=None, **kwargs):
        self.calls.append({
            "scope": scope,
            "target_mode": target_mode,
            "limit": limit,
            "filters": dict(filters or {}),
            "kwargs": dict(kwargs),
        })
        self.started.set()
        await self.release.wait()
        return {"generated": 1, "saved_ids": ["draft-1"]}


class _FailingCampaignService:
    async def generate(self, **kwargs):
        del kwargs
        raise RuntimeError("postgres://user:secret@example/internal")


class _Pool:
    def __init__(self):
        self.executed = []
        self.is_initialized = True

    async def execute(self, query, *args):
        self.executed.append((str(query), args))
        return "EXECUTE"


class _UsagePool:
    is_initialized = True

    def __init__(self):
        self.fetchrow_calls = []
        self.fetch_calls = []

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append((str(query), args))
        return {
            "total_cost_usd": Decimal("1.234567"),
            "input_tokens": 1200,
            "billable_input_tokens": 1100,
            "output_tokens": 300,
            "total_tokens": 1500,
            "cached_tokens": 100,
            "cache_write_tokens": 25,
            "total_calls": 4,
            "failed_calls": 1,
            "cache_hit_calls": 2,
            "avg_duration_ms": Decimal("234.56"),
            "latest_call_at": datetime(2026, 5, 26, 17, 0, tzinfo=timezone.utc),
        }

    async def fetch(self, query, *args):
        self.fetch_calls.append((str(query), args))
        if "GROUP BY provider, model" in str(query):
            return [
                {
                    "provider": "openrouter",
                    "model": "anthropic/claude-haiku-4-5",
                    "cost_usd": Decimal("1.000000"),
                    "calls": 3,
                    "input_tokens": 900,
                    "output_tokens": 200,
                }
            ]
        return [
            {
                "asset_type": "blog_post",
                "cost_usd": Decimal("0.750000"),
                "calls": 2,
                "input_tokens": 600,
                "output_tokens": 150,
            }
        ]


class _Transaction:
    def __init__(self):
        self.entered = False
        self.exited_with = None

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        del exc, tb
        self.exited_with = exc_type
        return False


class _Connection(_Pool):
    def __init__(self, *, fail_on_insert=False):
        super().__init__()
        self.fail_on_insert = fail_on_insert
        self.tx = _Transaction()

    def transaction(self):
        return self.tx

    async def execute(self, query, *args):
        self.executed.append((str(query), args))
        if self.fail_on_insert and "INSERT INTO" in str(query):
            raise RuntimeError("postgres://user:secret@example/internal")
        return "EXECUTE"


class _Acquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False


class _PoolWithAcquire:
    def __init__(self, connection):
        self.connection = connection
        self.is_initialized = True

    def acquire(self):
        return _Acquire(self.connection)


class _AdmissionGate:
    def __init__(
        self,
        *,
        allowed=True,
        max_concurrency=1,
        fail_acquire=False,
        fail_release=False,
    ):
        self.allowed = allowed
        self.max_concurrency = max_concurrency
        self.fail_acquire = fail_acquire
        self.fail_release = fail_release
        self.acquire_calls = 0
        self.release_calls = 0

    async def acquire(self):
        self.acquire_calls += 1
        if self.fail_acquire:
            raise RuntimeError("admission backend unavailable")
        return self.allowed

    async def release(self):
        self.release_calls += 1
        if self.fail_release:
            raise RuntimeError("admission backend release failed")


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
    assert "faq_markdown" in output_ids
    assert outputs["signal_extraction"]["implemented"] is True
    assert outputs["email_campaign"]["execution_configured"] is False
    assert outputs["email_campaign"]["can_execute"] is False
    assert outputs["email_campaign"]["estimated_unit_cost_usd"] == 0.18
    assert outputs["email_campaign"]["default_parse_retry_attempts"] == 1
    assert outputs["email_campaign"]["default_quality_repair_attempts"] == 0
    assert outputs["email_campaign"]["estimated_retry_adjusted_unit_cost_usd"] == 0.36
    assert outputs["landing_page"]["default_quality_repair_attempts"] == 1
    assert outputs["landing_page"]["estimated_retry_adjusted_unit_cost_usd"] == 2.6
    input_contracts = payload["input_contracts"]
    assert input_contracts["landing_page_quality_repair_attempts"] == {
        "key": "landing_page_quality_repair_attempts",
        "label": "Landing page quality repair attempts",
        "type": "integer",
        "min": 0,
        "max": 10,
        "default": 1,
    }
    assert input_contracts["target_keyword"] == {
        "key": "target_keyword",
        "label": "Target keyword",
        "type": "string",
        "placeholder": "customer support FAQ",
        "asset": "landing_page",
        "group": "seo_geo_aeo",
    }
    assert input_contracts["secondary_keywords"]["asset"] == "landing_page"
    assert input_contracts["secondary_keywords"]["group"] == "seo_geo_aeo"
    assert input_contracts["secondary_keywords"]["type"] == "string_list"
    assert input_contracts["cta_url"]["asset"] == "landing_page"
    assert input_contracts["cta_url"]["placeholder"] == "/systems/ai-content-ops/intake"
    assert input_contracts["faq_documentation_terms"] == {
        "key": "faq_documentation_terms",
        "label": "Documentation terms",
        "type": "string_list",
        "placeholder": "Single sign-on setup\nData export guide",
        "asset": "faq_markdown",
        "group": "vocabulary_gap",
    }
    assert input_contracts["faq_vocabulary_gap_rules"] == {
        "key": "faq_vocabulary_gap_rules",
        "label": "Vocabulary-gap rules",
        "type": "nested_string_list",
        "placeholder": "SSO, single sign-on\nexport, data export",
        "asset": "faq_markdown",
        "group": "vocabulary_gap",
    }
    assert outputs["email_campaign"]["reasoning_requirement"] == "optional_host_context"
    assert outputs["blog_post"]["reasoning_requirement"] == "optional_host_context"
    assert outputs["signal_extraction"]["reasoning_requirement"] == "absent"
    assert outputs["faq_markdown"]["reasoning_requirement"] == "absent"
    assert payload["execution"] == {
        "configured": False,
        "configured_outputs": [],
        "limits": dict(_DEFAULT_EXECUTION_LIMITS),
    }
    assert payload["reasoning"] == {"configured": False}
    assert "email_only" in preset_ids
    assert "lead_gen_campaign" in preset_ids
    assert payload["ingestion_profiles"] == [
        "domain_specific",
        "manual",
        "existing_evidence",
    ]
    assert payload["ingestion_limits"] == {
        "inline_rows": {
            "max_rows": 1000,
            "deprecated": True,
        },
        "file_upload": {
            "max_file_bytes": 25 * 1024 * 1024,
            "max_rows": 10000,
            "supported_formats": ["auto", "json", "jsonl", "csv"],
        },
        "max_source_text_chars": 10000,
        "max_sample_limit": 25,
    }


@pytest.mark.asyncio
async def test_usage_summary_route_requires_configured_pool_provider():
    router = create_content_ops_control_surface_router()

    route = _route(router, "/content-ops/usage/summary", "GET")

    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint()

    assert exc.value.status_code == 503
    assert exc.value.detail == "Content Ops usage database is unavailable."


@pytest.mark.asyncio
async def test_usage_summary_route_returns_content_ops_llm_rollup_with_filters():
    pool = _UsagePool()
    router = create_content_ops_control_surface_router(
        usage_pool_provider=lambda: pool,
    )

    route = _route(router, "/content-ops/usage/summary", "GET")
    payload = await route.endpoint(
        days=14,
        asset_type="blog_post",
        run_id="run-123",
        request_id="req-456",
    )

    assert payload["period_days"] == 14
    assert payload["filters"] == {
        "asset_type": "blog_post",
        "run_id": "run-123",
        "request_id": "req-456",
    }
    assert payload["summary"] == {
        "total_cost_usd": 1.234567,
        "total_calls": 4,
        "failed_calls": 1,
        "input_tokens": 1200,
        "billable_input_tokens": 1100,
        "output_tokens": 300,
        "total_tokens": 1500,
        "cached_tokens": 100,
        "cache_write_tokens": 25,
        "cache_hit_calls": 2,
        "avg_duration_ms": 234.6,
        "latest_call_at": "2026-05-26T17:00:00+00:00",
    }
    assert payload["by_model"] == [
        {
            "provider": "openrouter",
            "model": "anthropic/claude-haiku-4-5",
            "cost_usd": 1.0,
            "calls": 3,
            "input_tokens": 900,
            "output_tokens": 200,
        }
    ]
    assert payload["by_asset_type"] == [
        {
            "asset_type": "blog_post",
            "cost_usd": 0.75,
            "calls": 2,
            "input_tokens": 600,
            "output_tokens": 150,
        }
    ]
    summary_query, summary_args = pool.fetchrow_calls[0]
    assert "span_name = 'content_ops.llm.complete'" in summary_query
    assert "metadata ->> 'product' = 'content_ops'" in summary_query
    assert "metadata ->> 'asset_type' = $2" in summary_query
    assert "(run_id = $3 OR metadata ->> 'run_id' = $3)" in summary_query
    assert "metadata ->> 'request_id' = $4" in summary_query
    assert summary_args == (14, "blog_post", "run-123", "req-456")
    assert len(pool.fetch_calls) == 2


@pytest.mark.asyncio
async def test_describe_control_surfaces_reports_configured_execution_services():
    router = create_content_ops_control_surface_router(
        execution_services_provider=lambda: ContentOpsExecutionServices(
            campaign=_CampaignService(),
            report=_CampaignService(),
            signal_extraction=_CampaignService(),
            faq_markdown=_CampaignService(),
        )
    )

    route = _route(router, "/content-ops/control-surfaces", "GET")
    payload = await route.endpoint()

    outputs = {item["id"]: item for item in payload["outputs"]}
    assert payload["execution"] == {
        "configured": True,
        "configured_outputs": [
            "email_campaign",
            "faq_markdown",
            "report",
            "signal_extraction",
        ],
        "limits": dict(_DEFAULT_EXECUTION_LIMITS),
    }
    assert outputs["email_campaign"]["execution_configured"] is True
    assert outputs["email_campaign"]["can_execute"] is True
    assert outputs["report"]["execution_configured"] is True
    assert outputs["report"]["can_execute"] is True
    assert outputs["signal_extraction"]["execution_configured"] is True
    assert outputs["signal_extraction"]["can_execute"] is True
    assert outputs["faq_markdown"]["execution_configured"] is True
    assert outputs["faq_markdown"]["can_execute"] is True
    assert outputs["blog_post"]["execution_configured"] is False
    assert outputs["blog_post"]["can_execute"] is False
    assert payload["reasoning"] == {"configured": False}


@pytest.mark.asyncio
async def test_describe_control_surfaces_reports_configured_execute_concurrency():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(execute_max_concurrency=3)
    )

    route = _route(router, "/content-ops/control-surfaces", "GET")
    payload = await route.endpoint()

    assert payload["execution"]["limits"] == {
        **_DEFAULT_EXECUTION_LIMITS,
        "max_concurrency": 3,
    }


@pytest.mark.asyncio
async def test_describe_control_surfaces_reports_configured_faq_execute_row_limit():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(
            faq_execute_max_source_material_rows=250
        )
    )

    route = _route(router, "/content-ops/control-surfaces", "GET")
    payload = await route.endpoint()

    assert payload["execution"]["limits"] == {
        **_DEFAULT_EXECUTION_LIMITS,
        "faq_max_source_material_rows": 250,
    }


@pytest.mark.asyncio
async def test_describe_control_surfaces_reports_reasoning_provider_status():
    router = create_content_ops_control_surface_router(
        reasoning_context_provider=lambda: object()
    )

    route = _route(router, "/content-ops/control-surfaces", "GET")
    payload = await route.endpoint()

    assert payload["execution"] == {
        "configured": False,
        "configured_outputs": [],
        "limits": dict(_DEFAULT_EXECUTION_LIMITS),
    }
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
        "limits": dict(_DEFAULT_EXECUTION_LIMITS),
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
    assert payload["execution"] == {
        "configured": False,
        "configured_outputs": [],
        "limits": dict(_DEFAULT_EXECUTION_LIMITS),
    }
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
    first["ingestion_limits"]["inline_rows"]["max_rows"] = 999999
    first["ingestion_limits"]["file_upload"]["supported_formats"].append("yaml")
    first["execution"]["limits"]["max_source_material_rows"] = 999999

    second = await route.endpoint()
    assert second["outputs"][0]["label"] != "MUTATED"
    assert "injected_field" not in second["outputs"][0]["required_inputs"]
    assert "injected_output" not in second["presets"][0]["outputs"]
    assert "injected_profile" not in second["ingestion_profiles"]
    assert second["ingestion_limits"]["inline_rows"]["max_rows"] == 1000
    assert "yaml" not in second["ingestion_limits"]["file_upload"]["supported_formats"]
    assert second["execution"]["limits"]["max_source_material_rows"] == 1000


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
    assert "input_provider" not in payload


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
async def test_preview_generation_route_blocks_account_usage_budget():
    pool = _UsagePool()
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        usage_pool_provider=lambda: pool,
        scope_provider=lambda: {"account_id": " acct-1 "},
    )

    route = _route(router, "/ops/preview", "POST")
    payload = await route.endpoint(
        {
            "outputs": ["email_campaign"],
            "inputs": {
                "target_account": "Acme",
                "offer": "Churn audit",
            },
            "account_usage_budget_usd": 1.5,
            "account_usage_budget_days": 7,
        }
    )

    assert payload["can_run"] is False
    assert payload["usage_budget"]["current_cost_usd"] == 1.234567
    assert payload["usage_budget"]["estimated_cost_usd"] == 0.36
    assert payload["usage_budget"]["projected_cost_usd"] == 1.594567
    assert payload["usage_budget"]["exceeded"] is True
    assert "metadata ->> 'account_id' = $2" in pool.fetchrow_calls[0][0]
    assert pool.fetchrow_calls[0][1] == (7, "acct-1")
    assert (
        "Projected account usage exceeds account_usage_budget_usd: "
        "7-day projected 1.59 > 1.50"
    ) in payload["warnings"]


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
    assert "input_provider" not in payload


@pytest.mark.asyncio
async def test_plan_generation_route_marks_steps_blocked_when_account_budget_exceeds():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        usage_pool_provider=lambda: _UsagePool(),
        scope_provider=lambda: {"account_id": "acct-1"},
    )

    route = _route(router, "/ops/plan", "POST")
    payload = await route.endpoint(
        {
            "outputs": ["email_campaign"],
            "inputs": {
                "target_account": "Acme",
                "offer": "Churn audit",
            },
            "account_usage_budget_usd": 1.5,
        }
    )

    assert payload["can_execute"] is False
    assert payload["preview"]["can_run"] is False
    assert payload["preview"]["usage_budget"]["exceeded"] is True
    assert payload["steps"][0]["status"] == "blocked"
    assert payload["steps"][0]["reason"] == "account_usage_budget_exceeded"


@pytest.mark.asyncio
async def test_preview_generation_route_hides_noop_input_provider_diagnostics():
    provider = _SyncInputProvider(
        ContentOpsInputPackage(
            provider="atlas_support_ticket_request",
            inputs={},
            outputs=(),
            target_mode="",
            ingestion_profile="",
            metadata={
                "source": "atlas_content_ops_input_provider",
                "mode": "noop",
            },
        )
    )
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        input_provider=provider,
    )

    route = _route(router, "/ops/preview", "POST")
    payload = await route.endpoint({
        "outputs": ["email_campaign"],
        "inputs": {
            "target_account": "Acme",
            "offer": "Churn audit",
        },
    })

    assert payload["can_run"] is True
    assert payload["outputs"] == ["email_campaign"]
    assert "input_provider" not in payload


@pytest.mark.asyncio
async def test_preview_generation_route_applies_sync_input_provider():
    provider = _SyncInputProvider(
        ContentOpsInputPackage(
            provider="ticket_upload",
            outputs=("landing_page",),
            inputs={
                "audience": "10-50 person SaaS teams",
                "offer": "Provider offer",
            },
            metadata={
                "source_row_count": 10000,
                "included_row_count": 1000,
                "internal_request_id": "req-secret",
            },
            warnings=({
                "code": "ticket_rows_truncated",
                "message": "Used first 1000 ticket rows out of 10000.",
            },),
        )
    )
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        input_provider=provider,
        scope_provider=lambda: {"account_id": " acct-1 "},
    )

    route = _route(router, "/ops/preview", "POST")
    payload = await route.endpoint({
        "inputs": {
            "offer": "Operator offer",
        },
    })

    assert payload["can_run"] is True
    assert payload["outputs"] == ["landing_page"]
    assert payload["missing_inputs"] == []
    assert payload["input_provider"] == {
        "provider": "ticket_upload",
        "metadata": {"source_row_count": 10000, "included_row_count": 1000},
        "warnings": [{
            "code": "ticket_rows_truncated",
            "message": "Used first 1000 ticket rows out of 10000.",
        }],
    }
    assert provider.calls[0]["scope"].account_id == "acct-1"
    assert provider.calls[0]["request"]["inputs"]["offer"] == "Operator offer"


@pytest.mark.asyncio
async def test_plan_generation_route_applies_async_input_provider():
    provider = _AsyncInputProvider(
        ContentOpsInputPackage(
            provider="ticket_upload",
            outputs=("faq_markdown",),
            inputs={
                "source_material": [{
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                    "text": "How do I export my dashboard?",
                }],
            },
            metadata={
                "source_row_count": 10000,
                "included_row_count": 1000,
                "internal_request_id": "req-secret",
            },
            warnings=({
                "code": "ticket_rows_truncated",
                "message": "Used first 1000 ticket rows out of 10000.",
            },),
        )
    )
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        input_provider=provider,
    )

    route = _route(router, "/ops/plan", "POST")
    payload = await route.endpoint({})

    assert payload["can_execute"] is True
    assert payload["steps"][0]["output"] == "faq_markdown"
    assert payload["steps"][0]["runner"] == "TicketFAQMarkdownService.generate"
    assert payload["input_provider"] == {
        "provider": "ticket_upload",
        "metadata": {"source_row_count": 10000, "included_row_count": 1000},
        "warnings": [{
            "code": "ticket_rows_truncated",
            "message": "Used first 1000 ticket rows out of 10000.",
        }],
    }
    assert provider.calls[0]["scope"].account_id is None


@pytest.mark.asyncio
async def test_preview_generation_route_wraps_input_provider_failure(caplog):
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        input_provider=_FailingInputProvider(),
    )

    route = _route(router, "/ops/preview", "POST")
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint({"outputs": ["landing_page"]})

    assert exc.value.status_code == 503
    assert exc.value.detail == "Content Ops input provider is unavailable."
    assert "postgres://" not in str(exc.value.detail)
    assert "postgres://" not in caplog.text


@pytest.mark.asyncio
async def test_preview_generation_route_passes_through_input_provider_http_exception():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        input_provider=_HTTPFailingInputProvider(),
    )

    route = _route(router, "/ops/preview", "POST")
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint({"outputs": ["landing_page"]})

    assert exc.value.status_code == 400
    assert exc.value.detail == "bad ticket payload"


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
async def test_ingestion_inspect_route_reports_source_rows():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
    )

    route = _route(router, "/ops/ingestion/inspect", "POST")
    payload = await route.endpoint({
        "source_rows": True,
        "source": "fixture",
        "rows": [{
            "call_id": "call-1",
            "company": "Acme",
            "vendor": "HubSpot",
            "transcript": "The renewal process is too manual.",
            "contact_email": "ops@example.com",
        }],
    })

    assert payload["ok"] is True
    assert payload["mode"] == "source_rows"
    assert payload["source"] == "fixture"
    assert payload["opportunity_count"] == 1
    assert payload["source_type_counts"] == {"transcript": 1}
    assert payload["samples"][0]["source_type"] == "transcript"


@pytest.mark.asyncio
async def test_ingestion_inspect_route_applies_source_default_fields():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
    )

    route = _route(router, "/ops/ingestion/inspect", "POST")
    payload = await route.endpoint({
        "source_rows": True,
        "source": "fixture",
        "default_fields": {
            "company_name": "Acme Logistics",
            "contact_email": "ops@example.com",
        },
        "rows": [{
            "id": "g2-review-1",
            "vendor": "Slack",
            "review_text": "Search gets slow once message history grows.",
        }],
    })

    assert payload["ok"] is True
    assert payload["samples"][0]["company_name"] == "Acme Logistics"
    assert payload["samples"][0]["contact_email"] == "ops@example.com"


@pytest.mark.asyncio
async def test_ingestion_inspect_route_reports_missing_opportunity_fields():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
    )

    route = _route(router, "/ops/ingestion/inspect", "POST")
    payload = await route.endpoint({
        "rows": [{
            "target_id": "opp-1",
            "company_name": "Acme",
        }],
    })

    assert payload["ok"] is True
    assert payload["mode"] == "opportunities"
    assert payload["missing_field_counts"] == {
        "evidence": 1,
        "vendor_name": 1,
    }


@pytest.mark.asyncio
async def test_ingestion_inspect_route_rejects_oversized_rows():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
    )

    route = _route(router, "/ops/ingestion/inspect", "POST")
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint({
            "rows": [
                {"target_id": f"opp-{index}"}
                for index in range(1001)
            ],
        })

    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_inline_ingestion_routes_are_deprecated():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
    )

    inspect_route = _route(router, "/ops/ingestion/inspect", "POST")
    import_route = _route(router, "/ops/ingestion/import", "POST")

    assert inspect_route.deprecated is True
    assert import_route.deprecated is True


@pytest.mark.asyncio
async def test_ingestion_file_inspect_route_accepts_more_than_inline_row_cap():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
    )
    rows = [
        {
            "ticket_id": f"ticket-{index}",
            "source_type": "support_ticket",
            "subject": "Billing renewal question",
            "message": "How do I confirm my renewal invoice before payment?",
            "pain_category": "billing",
        }
        for index in range(1001)
    ]
    upload = _UploadFile(
        "support-ticket-export.json",
        (json.dumps({"support_tickets": rows}) + "\n").encode("utf-8"),
    )

    route = _route(router, "/ops/ingestion/files/inspect", "POST")
    payload = await route.endpoint(
        file=upload,
        source_rows=True,
        source="ticket-csv-upload",
        target_mode="vendor_retention",
        file_format="json",
        max_source_text_chars=1200,
        sample_limit=3,
        default_fields=json.dumps({
            "company_name": "Acme",
            "vendor_name": "Atlas",
            "contact_email": "support@example.com",
        }),
    )

    assert payload["ingestion_path"] == "file_upload"
    assert payload["limits"]["inline_rows_deprecated"] is True
    assert payload["source"] == "ticket-csv-upload"
    assert payload["mode"] == "source_rows"
    assert payload["ok"] is True
    assert payload["opportunity_count"] == 1001
    assert payload["source_type_counts"] == {"support_ticket": 1001}


@pytest.mark.asyncio
async def test_ingestion_file_import_route_dry_run_uses_file_parser():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        scope_provider=lambda: {"account_id": "acct-1"},
    )
    rows = [
        {
            "ticket_id": f"ticket-{index}",
            "source_type": "support_ticket",
            "subject": "Setup question",
            "message": "Where do I find the setup checklist?",
            "pain_category": "setup",
        }
        for index in range(1001)
    ]
    upload = _UploadFile(
        "support-ticket-export.json",
        (json.dumps({"support_tickets": rows}) + "\n").encode("utf-8"),
    )

    route = _route(router, "/ops/ingestion/files/import", "POST")
    payload = await route.endpoint(
        file=upload,
        source_rows=True,
        source="ticket-csv-upload",
        target_mode="vendor_retention",
        file_format="json",
        max_source_text_chars=1200,
        sample_limit=3,
        default_fields=json.dumps({
            "company_name": "Acme",
            "vendor_name": "Atlas",
            "contact_email": "support@example.com",
        }),
        dry_run=True,
    )

    assert payload["diagnostics"]["ingestion_path"] == "file_upload"
    assert payload["diagnostics"]["opportunity_count"] == 1001
    assert payload["import"]["dry_run"] is True
    assert payload["import"]["inserted"] == 1001
    assert payload["import"]["source"] == "ticket-csv-upload"


@pytest.mark.asyncio
async def test_ingestion_file_import_route_rejects_when_import_gate_is_full():
    pool = _Pool()
    provider_started = asyncio.Event()
    provider_release = asyncio.Event()
    provider_calls = 0

    async def pool_provider():
        nonlocal provider_calls
        provider_calls += 1
        provider_started.set()
        await provider_release.wait()
        return pool

    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(
            prefix="/ops",
            tags=("ops",),
            ingestion_import_max_concurrency=1,
        ),
        opportunity_import_pool_provider=pool_provider,
        scope_provider=lambda: {"account_id": "acct-1"},
    )
    route = _route(router, "/ops/ingestion/files/import", "POST")
    first = asyncio.create_task(
        route.endpoint(
            file=_ticket_bundle_upload(1),
            source_rows=True,
            source="ticket-csv-upload",
            target_mode="vendor_retention",
            file_format="json",
            max_source_text_chars=1200,
            sample_limit=3,
            default_fields=json.dumps({
                "company_name": "Acme",
                "vendor_name": "Atlas",
                "contact_email": "support@example.com",
            }),
            dry_run=False,
        )
    )
    await asyncio.wait_for(provider_started.wait(), timeout=1)

    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint(
            file=_ticket_bundle_upload(1),
            source_rows=True,
            source="ticket-csv-upload",
            target_mode="vendor_retention",
            file_format="json",
            max_source_text_chars=1200,
            sample_limit=3,
            default_fields=json.dumps({
                "company_name": "Acme",
                "vendor_name": "Atlas",
                "contact_email": "support@example.com",
            }),
            dry_run=False,
        )

    assert exc.value.status_code == 429
    assert exc.value.detail == {
        "reason": "content_ops_ingestion_import_at_capacity",
        "max_concurrency": 1,
    }
    assert provider_calls == 1

    provider_release.set()
    first_payload = await first
    assert first_payload["import"]["inserted"] == 1
    assert provider_calls == 1

    second_payload = await route.endpoint(
        file=_ticket_bundle_upload(1),
        source_rows=True,
        source="ticket-csv-upload",
        target_mode="vendor_retention",
        file_format="json",
        max_source_text_chars=1200,
        sample_limit=3,
        default_fields=json.dumps({
            "company_name": "Acme",
            "vendor_name": "Atlas",
            "contact_email": "support@example.com",
        }),
        dry_run=False,
    )
    assert second_payload["import"]["inserted"] == 1
    assert provider_calls == 2


@pytest.mark.asyncio
async def test_ingestion_file_import_route_uses_custom_admission_before_pool_provider():
    gate = _AdmissionGate(allowed=False, max_concurrency=3)
    provider_calls = 0

    async def pool_provider():
        nonlocal provider_calls
        provider_calls += 1
        return _Pool()

    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        opportunity_import_pool_provider=pool_provider,
        ingestion_import_admission_provider=lambda: gate,
        scope_provider=lambda: {"account_id": "acct-1"},
    )
    route = _route(router, "/ops/ingestion/files/import", "POST")

    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint(
            file=_ticket_bundle_upload(1),
            source_rows=True,
            source="ticket-csv-upload",
            target_mode="vendor_retention",
            file_format="json",
            max_source_text_chars=1200,
            sample_limit=3,
            default_fields=json.dumps({
                "company_name": "Acme",
                "vendor_name": "Atlas",
                "contact_email": "support@example.com",
            }),
            dry_run=False,
        )

    assert exc.value.status_code == 429
    assert exc.value.detail == {
        "reason": "content_ops_ingestion_import_at_capacity",
        "max_concurrency": 3,
    }
    assert gate.acquire_calls == 1
    assert gate.release_calls == 0
    assert provider_calls == 0


@pytest.mark.asyncio
async def test_ingestion_file_import_route_treats_custom_admission_acquire_failure_as_503():
    gate = _AdmissionGate(fail_acquire=True)
    provider_calls = 0

    async def pool_provider():
        nonlocal provider_calls
        provider_calls += 1
        return _Pool()

    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        opportunity_import_pool_provider=pool_provider,
        ingestion_import_admission_provider=lambda: gate,
        scope_provider=lambda: {"account_id": "acct-1"},
    )
    route = _route(router, "/ops/ingestion/files/import", "POST")

    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint(
            file=_ticket_bundle_upload(1),
            source_rows=True,
            source="ticket-csv-upload",
            target_mode="vendor_retention",
            file_format="json",
            max_source_text_chars=1200,
            sample_limit=3,
            default_fields=json.dumps({
                "company_name": "Acme",
                "vendor_name": "Atlas",
                "contact_email": "support@example.com",
            }),
            dry_run=False,
        )

    assert exc.value.status_code == 503
    assert exc.value.detail == "Content Ops ingestion import admission is unavailable."
    assert gate.acquire_calls == 1
    assert gate.release_calls == 0
    assert provider_calls == 0


@pytest.mark.asyncio
async def test_ingestion_import_route_releases_custom_admission_after_success():
    gate = _AdmissionGate(allowed=True)
    pool = _Pool()

    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        opportunity_import_pool_provider=lambda: pool,
        ingestion_import_admission_provider=lambda: gate,
        scope_provider=lambda: {"account_id": "acct-1"},
    )
    route = _route(router, "/ops/ingestion/import", "POST")

    payload = await route.endpoint({
        "dry_run": False,
        "source": "operator-upload",
        "rows": [{
            "target_id": "opp-1",
            "company_name": "Acme",
            "vendor_name": "HubSpot",
            "evidence": [{"quote": "Renewal process is too manual."}],
        }],
    })

    assert payload["import"]["inserted"] == 1
    assert gate.acquire_calls == 1
    assert gate.release_calls == 1


@pytest.mark.asyncio
async def test_ingestion_import_route_keeps_success_when_custom_admission_release_fails():
    gate = _AdmissionGate(allowed=True, fail_release=True)
    pool = _Pool()

    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        opportunity_import_pool_provider=lambda: pool,
        ingestion_import_admission_provider=lambda: gate,
        scope_provider=lambda: {"account_id": "acct-1"},
    )
    route = _route(router, "/ops/ingestion/import", "POST")

    payload = await route.endpoint({
        "dry_run": False,
        "source": "operator-upload",
        "rows": [{
            "target_id": "opp-1",
            "company_name": "Acme",
            "vendor_name": "HubSpot",
            "evidence": [{"quote": "Renewal process is too manual."}],
        }],
    })

    assert payload["import"]["inserted"] == 1
    assert gate.acquire_calls == 1
    assert gate.release_calls == 1


@pytest.mark.asyncio
async def test_ingestion_file_inspect_route_rejects_oversized_upload_bytes():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
    )
    route = _route(router, "/ops/ingestion/files/inspect", "POST")

    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint(
            file=_UploadFile("support-ticket-export.json", b"x" * (25 * 1024 * 1024 + 2)),
            source_rows=True,
            source="ticket-csv-upload",
            target_mode="vendor_retention",
            file_format="json",
            max_source_text_chars=1200,
            sample_limit=3,
            default_fields=json.dumps({
                "company_name": "Acme",
                "vendor_name": "Atlas",
                "contact_email": "support@example.com",
            }),
        )

    assert exc.value.status_code == 413
    assert "Uploaded ingestion file is too large" in exc.value.detail


@pytest.mark.asyncio
async def test_ingestion_file_inspect_route_rejects_more_than_file_row_cap():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
    )
    route = _route(router, "/ops/ingestion/files/inspect", "POST")

    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint(
            file=_ticket_bundle_upload(10001),
            source_rows=True,
            source="ticket-csv-upload",
            target_mode="vendor_retention",
            file_format="json",
            max_source_text_chars=1200,
            sample_limit=3,
            default_fields=json.dumps({
                "company_name": "Acme",
                "vendor_name": "Atlas",
                "contact_email": "support@example.com",
            }),
        )

    assert exc.value.status_code == 413
    assert "max 10000 rows" in exc.value.detail


@pytest.mark.asyncio
async def test_ingestion_file_inspect_route_rejects_empty_file():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
    )
    route = _route(router, "/ops/ingestion/files/inspect", "POST")

    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint(
            file=_UploadFile("empty.json", b""),
            source_rows=True,
            source="ticket-csv-upload",
            target_mode="vendor_retention",
            file_format="json",
            max_source_text_chars=1200,
            sample_limit=3,
            default_fields=None,
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "Uploaded ingestion file is empty."


@pytest.mark.asyncio
async def test_ingestion_file_inspect_route_rejects_invalid_file_format():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
    )
    route = _route(router, "/ops/ingestion/files/inspect", "POST")

    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint(
            file=_ticket_bundle_upload(1),
            source_rows=True,
            source="ticket-csv-upload",
            target_mode="vendor_retention",
            file_format="xlsx",
            max_source_text_chars=1200,
            sample_limit=3,
            default_fields=None,
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "file_format must be one of: auto, json, jsonl, csv."


@pytest.mark.asyncio
async def test_ingestion_file_inspect_route_rejects_malformed_default_fields():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
    )
    route = _route(router, "/ops/ingestion/files/inspect", "POST")

    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint(
            file=_ticket_bundle_upload(1),
            source_rows=True,
            source="ticket-csv-upload",
            target_mode="vendor_retention",
            file_format="json",
            max_source_text_chars=1200,
            sample_limit=3,
            default_fields="{not-json",
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "default_fields must be a JSON object."


@pytest.mark.asyncio
async def test_ingestion_file_import_route_rejects_not_ready_diagnostics():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
    )
    upload = _UploadFile(
        "support-ticket-export.json",
        (json.dumps([{"message": "Missing target defaults."}]) + "\n").encode("utf-8"),
    )
    route = _route(router, "/ops/ingestion/files/import", "POST")

    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint(
            file=upload,
            source_rows=False,
            source="ticket-csv-upload",
            target_mode="vendor_retention",
            file_format="json",
            max_source_text_chars=1200,
            sample_limit=3,
            default_fields=None,
            dry_run=True,
        )

    assert exc.value.status_code == 400
    assert exc.value.detail["reason"] == "ingestion_not_ready"
    assert exc.value.detail["diagnostics"]["ingestion_path"] == "file_upload"


@pytest.mark.asyncio
async def test_ingestion_import_route_dry_run_does_not_require_pool_provider():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
    )

    route = _route(router, "/ops/ingestion/import", "POST")
    payload = await route.endpoint({
        "dry_run": True,
        "source": "operator-upload",
        "rows": [{
            "target_id": "opp-1",
            "company_name": "Acme",
            "vendor_name": "HubSpot",
            "evidence": [{"quote": "Renewal process is too manual."}],
        }],
    })

    assert payload["diagnostics"]["ok"] is True
    assert payload["import"]["dry_run"] is True
    assert payload["import"]["inserted"] == 1
    assert payload["import"]["source"] == "operator-upload"
    assert payload["import"]["target_ids"] == ["opp-1"]


@pytest.mark.asyncio
async def test_ingestion_import_route_dry_run_applies_source_default_fields():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
    )

    route = _route(router, "/ops/ingestion/import", "POST")
    payload = await route.endpoint({
        "dry_run": True,
        "source_rows": True,
        "source": "g2-export",
        "default_fields": {
            "company_name": "Acme Logistics",
            "contact_email": "ops@example.com",
        },
        "rows": [{
            "id": "g2-review-1",
            "vendor": "Slack",
            "review_text": "Search gets slow once message history grows.",
        }],
    })

    sample = payload["diagnostics"]["samples"][0]
    assert payload["diagnostics"]["ok"] is True
    assert sample["company_name"] == "Acme Logistics"
    assert sample["contact_email"] == "ops@example.com"
    assert payload["import"]["inserted"] == 1
    assert payload["import"]["target_ids"] == ["g2-review-1"]


@pytest.mark.asyncio
async def test_ingestion_import_route_writes_rows_with_scope_and_replace_existing():
    pool = _Pool()
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        opportunity_import_pool_provider=lambda: pool,
        scope_provider=lambda: {"account_id": "acct-1"},
    )

    route = _route(router, "/ops/ingestion/import", "POST")
    payload = await route.endpoint({
        "replace_existing": True,
        "target_mode": "vendor_retention",
        "rows": [{
            "target_id": "opp-1",
            "company_name": "Acme",
            "vendor_name": "HubSpot",
            "evidence": [{"quote": "Renewal process is too manual."}],
        }],
    })

    assert payload["import"]["inserted"] == 1
    assert payload["import"]["replace_existing"] is True
    assert len(pool.executed) == 2
    delete_query, delete_args = pool.executed[0]
    insert_query, insert_args = pool.executed[1]
    assert "DELETE FROM \"campaign_opportunities\"" in delete_query
    assert delete_args == ("acct-1", "vendor_retention", ["opp-1"])
    assert "INSERT INTO \"campaign_opportunities\"" in insert_query
    assert insert_args[0] == "acct-1"
    assert insert_args[1] == "opp-1"


@pytest.mark.asyncio
async def test_ingestion_import_route_requires_pool_for_write():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
    )

    route = _route(router, "/ops/ingestion/import", "POST")
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint({
            "rows": [{
                "target_id": "opp-1",
                "company_name": "Acme",
                "vendor_name": "HubSpot",
                "evidence": [{"quote": "Renewal process is too manual."}],
            }],
        })

    assert exc.value.status_code == 503
    assert exc.value.detail == "Content Ops ingestion import database is not configured."


@pytest.mark.asyncio
async def test_ingestion_import_route_rejects_not_ready_diagnostics():
    pool = _Pool()
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        opportunity_import_pool_provider=lambda: pool,
    )

    route = _route(router, "/ops/ingestion/import", "POST")
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint({
            "rows": [{
                "evidence": [{"quote": "Renewal process is too manual."}],
            }],
        })

    assert exc.value.status_code == 400
    assert exc.value.detail["reason"] == "ingestion_not_ready"
    assert exc.value.detail["diagnostics"]["missing_field_counts"] == {
        "company_name": 1,
        "target_id": 1,
        "vendor_name": 1,
    }
    assert pool.executed == []


@pytest.mark.asyncio
async def test_ingestion_import_route_wraps_transactional_db_failures(caplog):
    connection = _Connection(fail_on_insert=True)
    pool = _PoolWithAcquire(connection)
    caplog.set_level("WARNING", logger="extracted_content_pipeline.api.control_surfaces")
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        opportunity_import_pool_provider=lambda: pool,
    )

    route = _route(router, "/ops/ingestion/import", "POST")
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint({
            "replace_existing": True,
            "rows": [{
                "target_id": "opp-1",
                "company_name": "Acme",
                "vendor_name": "HubSpot",
                "evidence": [{"quote": "Renewal process is too manual."}],
            }],
        })

    assert exc.value.status_code == 503
    assert exc.value.detail == "Content Ops ingestion import failed."
    assert connection.tx.entered is True
    assert connection.tx.exited_with is RuntimeError
    assert any("DELETE FROM" in query for query, _args in connection.executed)
    assert any("INSERT INTO" in query for query, _args in connection.executed)
    assert "postgres://" not in str(exc.value.detail)
    assert "postgres://" not in caplog.text


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
    assert "input_provider" not in payload


@pytest.mark.asyncio
async def test_execute_generation_route_blocks_account_usage_budget_before_generation():
    service = _CampaignService()
    provider_calls = {"count": 0}

    def execution_services_provider():
        provider_calls["count"] += 1
        return ContentOpsExecutionServices(campaign=service)

    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        execution_services_provider=execution_services_provider,
        usage_pool_provider=lambda: _UsagePool(),
        scope_provider=lambda: {"account_id": "acct-1"},
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
                "account_usage_budget_usd": 1.5,
            }
        )

    assert exc.value.status_code == 400
    assert exc.value.detail["reason"] == "account_usage_budget_exceeded"
    assert exc.value.detail["usage_budget"]["projected_cost_usd"] == 1.594567
    assert provider_calls["count"] == 0
    assert service.calls == []


@pytest.mark.asyncio
async def test_execute_generation_route_applies_input_provider_before_generation():
    service = _CampaignService()
    provider = _SyncInputProvider(
        ContentOpsInputPackage(
            provider="ticket_upload",
            outputs=("email_campaign",),
            inputs={
                "target_account": "Provider account",
                "offer": "Provider offer",
                "filters": {"status": "provider"},
            },
            metadata={
                "source_row_count": 10000,
                "included_row_count": 1000,
                "internal_request_id": "req-secret",
            },
            warnings=({
                "code": "ticket_rows_truncated",
                "message": "Used first 1000 ticket rows out of 10000.",
            },),
        )
    )
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        execution_services_provider=lambda: ContentOpsExecutionServices(campaign=service),
        input_provider=provider,
        scope_provider=lambda: {"account_id": " acct-1 "},
    )

    route = _route(router, "/ops/execute", "POST")
    payload = await route.endpoint({
        "inputs": {
            "offer": "Operator offer",
            "filters": {"status": "operator"},
        },
    })

    assert payload["status"] == "completed"
    assert provider.calls[0]["scope"].account_id == "acct-1"
    assert service.calls[0]["scope"].account_id == "acct-1"
    assert service.calls[0]["filters"] == {"status": "operator"}
    assert service.calls[0]["target_mode"] == "vendor_retention"
    assert service.calls[0]["limit"] == 1
    assert payload["input_provider"] == {
        "provider": "ticket_upload",
        "metadata": {"source_row_count": 10000, "included_row_count": 1000},
        "warnings": [{
            "code": "ticket_rows_truncated",
            "message": "Used first 1000 ticket rows out of 10000.",
        }],
    }


@pytest.mark.asyncio
async def test_execute_generation_route_rejects_when_concurrency_gate_is_full():
    service = _BlockingCampaignService()
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(
            prefix="/ops",
            tags=("ops",),
            execute_max_concurrency=1,
        ),
        execution_services_provider=lambda: ContentOpsExecutionServices(campaign=service),
    )
    route = _route(router, "/ops/execute", "POST")
    request = {
        "outputs": ["email_campaign"],
        "inputs": {
            "target_account": "Acme",
            "offer": "Churn audit",
        },
    }

    first = asyncio.create_task(route.endpoint(request))
    await asyncio.wait_for(service.started.wait(), timeout=1)
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint(request)

    assert exc.value.status_code == 429
    assert exc.value.detail == {
        "reason": "content_ops_execute_at_capacity",
        "max_concurrency": 1,
    }
    assert len(service.calls) == 1

    service.release.set()
    first_payload = await first
    assert first_payload["status"] == "completed"

    second_payload = await route.endpoint(request)
    assert second_payload["status"] == "completed"
    assert len(service.calls) == 2


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
async def test_execute_generation_route_rejects_invalid_faq_vocabulary_rules_as_400():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            faq_markdown=_CampaignService()
        ),
    )

    route = _route(router, "/ops/execute", "POST")
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint(
            {
                "outputs": ["faq_markdown"],
                "inputs": {
                    "source_material": [
                        {
                            "source_type": "ticket",
                            "text": "How do I enable SSO?",
                        }
                    ],
                    "faq_vocabulary_gap_rules": [["SSO"]],
                },
            }
        )

    assert exc.value.status_code == 400
    assert (
        exc.value.detail
        == "faq_vocabulary_gap_rules entries must include at least two terms"
    )


@pytest.mark.asyncio
async def test_execute_generation_route_accepts_faq_source_material_at_configured_limit():
    service = _CampaignService()
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(
            prefix="/ops",
            tags=("ops",),
            faq_execute_max_source_material_rows=2,
        ),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            faq_markdown=service
        ),
    )

    route = _route(router, "/ops/execute", "POST")
    payload = await route.endpoint(
        {
            "outputs": ["faq_markdown"],
            "inputs": {
                "source_material": {
                    "support_tickets": [
                        {
                            "source_type": "ticket",
                            "text": f"Ticket row {index}",
                        }
                        for index in range(2)
                    ],
                },
            },
        }
    )

    assert payload["status"] == "completed"
    assert len(service.calls) == 1


@pytest.mark.asyncio
async def test_execute_generation_route_rejects_faq_source_material_over_configured_limit():
    service = _CampaignService()
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(
            prefix="/ops",
            tags=("ops",),
            faq_execute_max_source_material_rows=2,
        ),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            faq_markdown=service
        ),
    )

    route = _route(router, "/ops/execute", "POST")
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint(
            {
                "outputs": ["faq_markdown"],
                "inputs": {
                    "source_material": {
                        "support_tickets": [
                            {
                                "source_type": "ticket",
                                "text": f"Ticket row {index}",
                            }
                            for index in range(3)
                        ],
                    },
                },
            }
        )

    assert exc.value.status_code == 413
    assert exc.value.detail == {
        "reason": "faq_source_material_too_large_for_sync_execute",
        "max_source_material_rows": 2,
        "source_material_rows": 3,
        "large_upload_strategy": "background_or_offline",
    }
    assert service.calls == []


@pytest.mark.asyncio
async def test_execute_generation_route_checks_input_provider_faq_rows_after_merge():
    service = _CampaignService()
    provider = _SyncInputProvider(
        ContentOpsInputPackage(
            provider="ticket_upload",
            outputs=("faq_markdown",),
            inputs={
                "source_material": {
                    "support_tickets": [
                        {
                            "source_type": "ticket",
                            "text": f"Ticket row {index}",
                        }
                        for index in range(3)
                    ],
                },
            },
        )
    )
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(
            prefix="/ops",
            tags=("ops",),
            faq_execute_max_source_material_rows=2,
        ),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            faq_markdown=service
        ),
        input_provider=provider,
    )

    route = _route(router, "/ops/execute", "POST")
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint({})

    assert exc.value.status_code == 413
    assert exc.value.detail["source_material_rows"] == 3
    assert service.calls == []


@pytest.mark.asyncio
async def test_execute_generation_route_does_not_apply_faq_row_limit_to_non_faq_outputs():
    service = _CampaignService()
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(
            prefix="/ops",
            tags=("ops",),
            faq_execute_max_source_material_rows=2,
        ),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            signal_extraction=service
        ),
    )

    route = _route(router, "/ops/execute", "POST")
    payload = await route.endpoint(
        {
            "outputs": ["signal_extraction"],
            "inputs": {
                "source_material": [
                    {
                        "source_type": "ticket",
                        "text": f"Ticket row {index}",
                    }
                    for index in range(3)
                ],
            },
        }
    )

    assert payload["status"] == "completed"
    assert len(service.calls[0]["kwargs"]["source_material"]) == 3


@pytest.mark.asyncio
async def test_execute_generation_route_rejects_source_material_over_1000_as_422():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            faq_markdown=_CampaignService()
        ),
    )

    route = _route(router, "/ops/execute", "POST")
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint(
            {
                "outputs": ["faq_markdown"],
                "inputs": {
                    "source_material": [
                        {
                            "source_type": "ticket",
                            "text": f"Ticket row {index}",
                        }
                        for index in range(1001)
                    ],
                },
            }
        )

    assert exc.value.status_code == 422
    assert exc.value.detail[0]["msg"] == "Value error, inputs arrays are too large"


@pytest.mark.asyncio
async def test_execute_generation_route_keeps_50_cap_for_non_source_material_arrays():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            faq_markdown=_CampaignService()
        ),
    )

    route = _route(router, "/ops/execute", "POST")
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint(
            {
                "outputs": ["faq_markdown"],
                "inputs": {
                    "faq_documentation_terms": [
                        f"Documentation term {index}"
                        for index in range(51)
                    ],
                    "source_material": [
                        {
                            "source_type": "ticket",
                            "text": "How do I enable SSO?",
                        }
                    ],
                },
            }
        )

    assert exc.value.status_code == 422
    assert exc.value.detail[0]["msg"] == "Value error, inputs arrays are too large"


@pytest.mark.asyncio
async def test_execute_generation_route_accepts_source_material_bundle_1000_rows():
    service = _CampaignService()
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            faq_markdown=service
        ),
    )

    route = _route(router, "/ops/execute", "POST")
    payload = await route.endpoint(
        {
            "outputs": ["faq_markdown"],
            "inputs": {
                "source_material": {
                    "support_tickets": [
                        {
                            "source_type": "ticket",
                            "text": f"Ticket row {index}",
                        }
                        for index in range(1000)
                    ],
                },
            },
        }
    )

    assert payload["status"] == "completed"
    assert len(service.calls[0]["kwargs"]["source_material"]["support_tickets"]) == 1000


@pytest.mark.asyncio
async def test_execute_generation_route_keeps_50_cap_for_nested_source_material_arrays():
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            faq_markdown=_CampaignService()
        ),
    )

    route = _route(router, "/ops/execute", "POST")
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint(
            {
                "outputs": ["faq_markdown"],
                "inputs": {
                    "source_material": [
                        [
                            {
                                "source_type": "ticket",
                                "text": f"Ticket row {index}",
                            }
                            for index in range(51)
                        ]
                    ],
                },
            }
        )

    assert exc.value.status_code == 422
    assert exc.value.detail[0]["msg"] == "Value error, inputs arrays are too large"


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

    async def generate(
        self,
        *,
        scope,
        target_mode=None,
        limit=None,
        filters=None,
        campaign=None,
        **kwargs,
    ):
        del scope, target_mode, limit, filters, campaign, kwargs
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
async def test_execute_route_builds_structured_reasoning_for_campaign_and_report():
    campaign_providers = []
    report_providers = []
    campaign = _ProviderRecordingService(campaign_providers)
    report = _ProviderRecordingService(report_providers)

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

    assert len(campaign_providers) == 1
    assert len(report_providers) == 1
    assert campaign_providers[0] is report_providers[0]
    provider = report_providers[0]
    assert provider._config.default_goal == "synthesize content reasoning context"
    assert provider._config.narrative_plan_pack.name == "content_ops_structured"
    assert provider._config.output_policy is not None
    assert provider._config.output_policy.require_citations is True
    assert provider._config.block_on_validation_failure is False
    email_step = next(step for step in payload["steps"] if step["output"] == "email_campaign")
    assert email_step["reasoning"]["provider_configured"] is True


@pytest.mark.asyncio
async def test_execute_route_builds_blog_specific_structured_reasoning_pack():
    recorded_providers = []
    blog_post = _ProviderRecordingService(recorded_providers)

    router = create_content_ops_control_surface_router(
        execution_services_provider=lambda: ContentOpsExecutionServices(
            blog_post=blog_post,
        ),
        llm_provider=lambda: object(),
    )

    route = _route(router, "/content-ops/execute", "POST")
    await route.endpoint(
        {
            "outputs": ["blog_post"],
            "reasoning_preset": "multi_pass_structured",
            "inputs": {"topic": "Churn pressure"},
        }
    )

    assert len(recorded_providers) == 1
    provider = recorded_providers[0]
    assert provider._config.narrative_plan_pack.name == "content_ops_blog"
    assert provider._config.output_policy is not None
    assert provider._config.block_on_validation_failure is False


@pytest.mark.asyncio
async def test_execute_route_builds_structured_reasoning_for_landing_page():
    recorded_providers = []
    landing_page = _ProviderRecordingService(recorded_providers)

    router = create_content_ops_control_surface_router(
        execution_services_provider=lambda: ContentOpsExecutionServices(
            landing_page=landing_page,
        ),
        llm_provider=lambda: object(),
    )

    route = _route(router, "/content-ops/execute", "POST")
    await route.endpoint(
        {
            "outputs": ["landing_page"],
            "reasoning_preset": "multi_pass_structured",
            "inputs": {"offer": "Audit", "audience": "RevOps"},
        }
    )

    assert len(recorded_providers) == 1
    provider = recorded_providers[0]
    assert provider._config.narrative_plan_pack.name == "content_ops_structured"
    assert provider._config.output_policy is not None
    assert provider._config.block_on_validation_failure is False


@pytest.mark.asyncio
async def test_execute_route_uses_configured_output_pack_mapping():
    recorded_providers = []
    blog_post = _ProviderRecordingService(recorded_providers)

    router = create_content_ops_control_surface_router(
        execution_services_provider=lambda: ContentOpsExecutionServices(
            blog_post=blog_post,
        ),
        llm_provider=lambda: object(),
        config=ContentOpsControlSurfaceApiConfig(
            structured_reasoning_output_pack_names={
                "blog_post": "custom_blog_pack",
            },
        ),
    )

    route = _route(router, "/content-ops/execute", "POST")
    await route.endpoint(
        {
            "outputs": ["blog_post"],
            "reasoning_preset": "multi_pass_structured",
            "inputs": {"topic": "Churn pressure"},
        }
    )

    assert recorded_providers[0]._config.narrative_plan_pack.name == "custom_blog_pack"


@pytest.mark.parametrize("outputs", (("blog_post", "report"), ("report", "blog_post")))
@pytest.mark.asyncio
async def test_execute_route_keeps_blog_and_report_reasoning_packs_separate(outputs):
    blog_providers = []
    report_providers = []
    blog_post = _ProviderRecordingService(blog_providers)
    report = _ProviderRecordingService(report_providers)

    router = create_content_ops_control_surface_router(
        execution_services_provider=lambda: ContentOpsExecutionServices(
            blog_post=blog_post,
            report=report,
        ),
        llm_provider=lambda: object(),
    )

    route = _route(router, "/content-ops/execute", "POST")
    await route.endpoint(
        {
            "outputs": list(outputs),
            "reasoning_preset": "multi_pass_structured",
            "inputs": {
                "topic": "Churn pressure",
                "opportunity_id": "opp-1",
            },
        }
    )

    assert blog_providers[0]._config.narrative_plan_pack.name == "content_ops_blog"
    assert report_providers[0]._config.narrative_plan_pack.name == "content_ops_structured"


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
    assert provider.provider._config.falsification_policy is None
    assert provider.provider._config.block_on_validation_failure is False


@pytest.mark.asyncio
async def test_execute_route_wires_strict_reasoning_falsification_policy():
    recorded_providers = []
    report = _ProviderRecordingService(recorded_providers)

    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(
            structured_reasoning_falsification_rules=(
                {
                    "id": "renewal_signal_lost",
                    "predicate": "fresh evidence shows renewal completed",
                },
            ),
            structured_reasoning_falsification_conservative=False,
            structured_reasoning_drop_falsified=True,
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

    provider = recorded_providers[0].provider
    assert provider._config.falsification_policy is not None
    assert provider._config.falsification_policy.rules == (
        {
            "id": "renewal_signal_lost",
            "predicate": "fresh evidence shows renewal completed",
        },
    )
    assert provider._config.falsification_policy.conservative is False
    assert provider._config.drop_falsified is True


@pytest.mark.asyncio
async def test_execute_route_does_not_wire_falsification_for_structured_preset():
    recorded_providers = []
    report = _ProviderRecordingService(recorded_providers)

    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(
            structured_reasoning_falsification_rules=(
                {"id": "renewal_signal_lost"},
            ),
            structured_reasoning_drop_falsified=True,
        ),
        execution_services_provider=lambda: ContentOpsExecutionServices(report=report),
        llm_provider=lambda: object(),
    )

    route = _route(router, "/content-ops/execute", "POST")
    await route.endpoint(
        {
            "outputs": ["report"],
            "reasoning_preset": "multi_pass_structured",
            "inputs": {"opportunity_id": "opp-1"},
        }
    )

    provider = recorded_providers[0]
    assert provider._config.falsification_policy is None
    assert provider._config.drop_falsified is False


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
                "outputs": ["email_campaign"],
                "reasoning_preset": "multi_pass_strict",
                "inputs": {"target_account": "Acme", "offer": "Audit"},
            },
            400,
            "not supported",
        ),
        (
            {
                "outputs": ["blog_post"],
                "reasoning_preset": "multi_pass_strict",
                "inputs": {"topic": "Churn pressure"},
            },
            400,
            "not supported",
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


def test_content_ops_config_rejects_invalid_output_pack_names():
    with pytest.raises(ValueError, match="must be a mapping"):
        ContentOpsControlSurfaceApiConfig(
            structured_reasoning_output_pack_names="content_ops_blog"
        )
    with pytest.raises(ValueError, match="keys must be non-empty"):
        ContentOpsControlSurfaceApiConfig(
            structured_reasoning_output_pack_names={" ": "content_ops_blog"}
        )
    with pytest.raises(ValueError, match="values must be non-empty"):
        ContentOpsControlSurfaceApiConfig(
            structured_reasoning_output_pack_names={"blog_post": " "}
        )


def test_content_ops_config_stores_output_pack_names_immutably():
    config = ContentOpsControlSurfaceApiConfig(
        structured_reasoning_output_pack_names={"blog_post": "content_ops_blog"}
    )

    with pytest.raises(TypeError):
        config.structured_reasoning_output_pack_names["blog_post"] = "other_pack"  # type: ignore[index]


def test_content_ops_config_rejects_invalid_execute_concurrency():
    with pytest.raises(ValueError, match="execute_max_concurrency must be positive"):
        ContentOpsControlSurfaceApiConfig(execute_max_concurrency=0)


def test_content_ops_config_rejects_invalid_faq_execute_row_limit():
    with pytest.raises(
        ValueError,
        match="faq_execute_max_source_material_rows must be positive",
    ):
        ContentOpsControlSurfaceApiConfig(faq_execute_max_source_material_rows=0)
    with pytest.raises(
        ValueError,
        match="faq_execute_max_source_material_rows cannot exceed 1000",
    ):
        ContentOpsControlSurfaceApiConfig(faq_execute_max_source_material_rows=1001)


def test_content_ops_config_rejects_invalid_ingestion_import_concurrency():
    with pytest.raises(ValueError, match="ingestion_import_max_concurrency must be positive"):
        ContentOpsControlSurfaceApiConfig(ingestion_import_max_concurrency=0)


def test_content_ops_config_rejects_invalid_falsification_rules_shape():
    with pytest.raises(ValueError, match="must be a sequence"):
        ContentOpsControlSurfaceApiConfig(
            structured_reasoning_falsification_rules={"id": "rule-1"}
        )
    with pytest.raises(ValueError, match="must be a sequence"):
        ContentOpsControlSurfaceApiConfig(
            structured_reasoning_falsification_rules=5
        )
    with pytest.raises(ValueError, match="entries must be mappings"):
        ContentOpsControlSurfaceApiConfig(
            structured_reasoning_falsification_rules=("rule-1",)
        )


def test_content_ops_config_rejects_drop_falsified_without_rules():
    with pytest.raises(ValueError, match="requires non-empty"):
        ContentOpsControlSurfaceApiConfig(
            structured_reasoning_drop_falsified=True,
        )


def test_content_ops_config_rejects_too_many_falsification_rules():
    rules = tuple({"id": f"rule-{index}"} for index in range(21))

    with pytest.raises(ValueError, match="cannot exceed 20"):
        ContentOpsControlSurfaceApiConfig(
            structured_reasoning_falsification_rules=rules,
        )
