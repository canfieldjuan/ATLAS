from __future__ import annotations

from typing import Any

import pytest

import extracted_content_pipeline.api.control_surfaces as api_module
from atlas_brain._content_ops_input_provider import build_content_ops_input_provider
from extracted_content_pipeline.api.control_surfaces import (
    ContentOpsControlSurfaceApiConfig,
    create_content_ops_control_surface_router,
)
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.content_ops_execution import ContentOpsExecutionServices
from extracted_content_pipeline.content_ops_execution import execute_content_ops_request
from extracted_content_pipeline.content_ops_input_provider import (
    merge_content_ops_input_package,
)
from extracted_content_pipeline.control_surfaces import (
    preview_control_surface,
    request_from_mapping,
)


class _OpportunityRepo:
    def __init__(self, pool):
        self.pool = pool

    async def read_campaign_opportunities(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        limit: int,
        filters=None,
    ):
        target_id = dict(filters or {}).get("target_id")
        self.pool["opportunity_calls"].append({
            "account_id": scope.account_id,
            "target_mode": target_mode,
            "limit": limit,
            "target_id": target_id,
        })
        return tuple(self.pool["opportunities"].get(target_id, ()))[:limit]


class _B2BDisplacementPool:
    def __init__(self, rows):
        self.rows = tuple(rows)
        self.calls = []

    async def fetch(self, query, *args):
        self.calls.append({"query": query, "args": args})
        return list(self.rows)


class _RunnableService:
    async def generate(self, **kwargs):  # pragma: no cover - plan route only.
        return {"kwargs": kwargs}


_MARKETER_OUTPUTS = ("landing_page", "blog_post", "sales_brief", "social_post")


def _route(router, path: str, method: str):
    for route in router.routes:
        if getattr(route, "path", None) == path and method.upper() in getattr(
            route,
            "methods",
            set(),
        ):
            return route
    raise AssertionError(f"Route {method.upper()} {path!r} not mounted")


def _review_row(
    target_id: str = "review-1",
    *,
    vendor_name: str = "Slack",
) -> dict[str, Any]:
    return {
        "target_id": target_id,
        "review_id": target_id,
        "source_type": "review",
        "reviewer_company": "Acme Logistics",
        "vendor_name": vendor_name,
        "review_text": "Search gets slow once message history grows.",
        "pain_category": "performance",
    }


def _generic_review_row(target_id: str = "review-1") -> dict[str, Any]:
    return {
        "target_id": target_id,
        "review_id": target_id,
        "vendor_name": "Slack",
        "reviewer_company": "Acme Logistics",
        "content": "Search gets slow once message history grows.",
        "pain_category": "performance",
    }


def _competitive_row(target_id: str = "competitive-1") -> dict[str, Any]:
    return {
        "target_id": target_id,
        "source_id": target_id,
        "source_type": "competitive_displacement",
        "from_vendor": "Slack",
        "to_vendor": "Teams",
        "competitor": "Teams",
        "content": "Admins are switching to Teams because calendar handoff is simpler.",
        "displacement_mention_count": 4,
        "primary_driver": "workflow friction",
    }


def _b2b_displacement_row(
    *,
    from_vendor: str = "Slack",
    to_vendor: str = "Teams",
) -> dict[str, Any]:
    return {
        "from_vendor": from_vendor,
        "to_vendor": to_vendor,
        "as_of_date": "2026-06-01",
        "analysis_window_days": 90,
        "schema_version": "v1",
        "dynamics": {
            "battle_summary": {
                "conclusion": "Admins choose Teams when calendar handoff matters.",
            },
            "migration_proof": {
                "proof_points": [
                    "Switching evidence clusters around bundled collaboration.",
                ],
            },
            "edge_metrics": {
                "mention_count": 7,
                "primary_driver": "workflow friction",
            },
        },
    }


def _support_ticket_row() -> dict[str, Any]:
    return {
        "ticket_id": "ticket-1",
        "subject": "Billing question",
        "message": "Why was I charged twice this month?",
    }


def test_review_source_material_builds_landing_blog_and_sales_brief_inputs() -> None:
    package = build_content_ops_input_provider().build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "inputs": {
                "source_type": "reviews",
                "source_material": [_review_row()],
            }
        },
    )
    payload = merge_content_ops_input_package({"inputs": {}}, package)
    request = request_from_mapping(payload)
    preview = preview_control_surface(request)

    assert package.provider == "atlas_review_request"
    assert request.outputs == _MARKETER_OUTPUTS
    assert request.ingestion_profile == "existing_evidence"
    assert request.inputs["source_type"] == "reviews"
    assert request.inputs["brief_type"] == "discovery"
    assert request.inputs["source_material"][0]["source_type"] == "review"
    assert request.inputs["review_source_material"][0]["source_type"] == "review"
    assert request.inputs["source_material"][0]["target_id"] == "review-1"
    assert request.inputs["target_account"] == "Slack"
    assert request.inputs["topic"] == "Customer review themes worth turning into content"
    assert package.metadata["included_row_count"] == 1
    assert preview.can_run is True


def test_review_mode_defaults_generic_rows_to_requested_review_type() -> None:
    package = build_content_ops_input_provider().build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "inputs": {
                "source_type": "reviews",
                "source_material": [_generic_review_row()],
            }
        },
    )

    assert package.provider == "atlas_review_request"
    assert package.inputs["source_material"][0]["source_type"] == "reviews"
    assert package.inputs["source_material"][0]["target_id"] == "review-1"
    assert package.metadata["source_row_count"] == 1
    assert package.metadata["included_row_count"] == 1


def test_competitive_source_material_builds_landing_and_blog_inputs() -> None:
    package = build_content_ops_input_provider().build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "inputs": {
                "source_type": "competitive",
                "source_material": [_competitive_row()],
            }
        },
    )
    payload = merge_content_ops_input_package({"inputs": {}}, package)
    request = request_from_mapping(payload)
    preview = preview_control_surface(request)

    assert package.provider == "atlas_competitive_request"
    assert request.outputs == _MARKETER_OUTPUTS
    assert request.ingestion_profile == "existing_evidence"
    assert request.inputs["source_type"] == "competitive"
    assert request.inputs["competitive_source_material"][0]["target_id"] == "competitive-1"
    assert request.inputs["brief_type"] == "displacement"
    assert request.inputs["target_account"] == "Slack"
    assert request.inputs["competitive_alternatives"] == ["Teams"]
    assert package.metadata["included_row_count"] == 1
    assert preview.can_run is True


def test_competitive_source_material_accepts_competitive_bundle_alias() -> None:
    row = _competitive_row()
    row.pop("source_type")
    package = build_content_ops_input_provider().build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "inputs": {
                "source_material_type": "competitive",
                "source_material": {"competitive_signals": [row]},
            }
        },
    )

    assert package.provider == "atlas_competitive_request"
    assert package.outputs == _MARKETER_OUTPUTS
    assert package.inputs["source_material"][0]["source_type"] == "competitive"
    assert package.inputs["source_material"][0]["target_id"] == "competitive-1"
    assert package.metadata["source_row_count"] == 1
    assert package.metadata["included_row_count"] == 1


def test_hyphenated_competitive_source_type_routes_to_competitive_provider() -> None:
    row = _competitive_row()
    row.pop("source_type")
    package = build_content_ops_input_provider().build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "inputs": {
                "source_type": "competitive-signal",
                "source_material": [row],
            }
        },
    )

    assert package.provider == "atlas_competitive_request"
    assert package.inputs["source_type"] == "competitive"
    assert package.inputs["source_material"][0]["source_type"] == "competitive_signal"
    assert package.metadata["requested_source_type"] == "competitive_signal"
    assert package.metadata["included_row_count"] == 1


def test_review_source_material_accepts_review_bundle_alias() -> None:
    package = build_content_ops_input_provider().build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "inputs": {
                "source_material_type": "review",
                "source_material": {"reviews": [_review_row()]},
            }
        },
    )

    assert package.provider == "atlas_review_request"
    assert package.outputs == _MARKETER_OUTPUTS
    assert package.inputs["source_material"][0]["source_type"] == "review"
    assert package.metadata["included_row_count"] == 1


def test_missing_source_type_keeps_review_rows_out_of_support_ticket_provider() -> None:
    package = build_content_ops_input_provider().build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "outputs": ["signal_extraction"],
            "inputs": {"source_material": [_review_row()]},
        },
    )
    payload = merge_content_ops_input_package(
        {"outputs": ["signal_extraction"], "inputs": {"source_material": [_review_row()]}},
        package,
    )
    request = request_from_mapping(payload)

    assert package.metadata["mode"] == "noop"
    assert request.outputs == ("signal_extraction",)
    assert request.inputs["source_material"][0]["source_type"] == "review"


def test_explicit_support_ticket_source_type_keeps_support_ticket_provider() -> None:
    package = build_content_ops_input_provider().build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "outputs": ["faq_markdown"],
            "inputs": {
                "source_type": "support_ticket",
                "source_material": [_support_ticket_row()],
            },
        },
    )

    assert package.provider == "atlas_support_ticket_request"
    assert package.inputs["source_material"][0]["source_type"] == "support_ticket"
    assert "faq_markdown" in package.outputs
    assert package.metadata["included_row_count"] == 1


def test_review_mode_rejects_non_review_rows() -> None:
    package = build_content_ops_input_provider().build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "inputs": {
                "source_type": "reviews",
                "source_material": [{
                    "source_type": "document",
                    "target_id": "doc-1",
                    "content": "General implementation notes.",
                }],
            }
        },
    )

    assert package.provider == "atlas_review_request"
    assert package.inputs == {}
    assert package.metadata["mode"] == "noop"
    assert package.metadata["included_row_count"] == 0
    assert package.warnings[-1]["code"] == "review_source_rows_unrecognized"


def test_competitive_mode_rejects_non_competitive_rows() -> None:
    package = build_content_ops_input_provider().build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "inputs": {
                "source_type": "competitive",
                "source_material": [_generic_review_row()],
            }
        },
    )

    assert package.provider == "atlas_competitive_request"
    assert package.inputs == {}
    assert package.metadata["mode"] == "noop"
    assert package.metadata["included_row_count"] == 0
    assert package.warnings[-1]["code"] == "competitive_source_rows_unrecognized"


def test_review_mode_rejects_saved_faq_selection() -> None:
    package = build_content_ops_input_provider().build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "inputs": {
                "source_type": "reviews",
                "source_faq_ids": ["11111111-1111-1111-1111-111111111111"],
                "source_material": [_review_row()],
            }
        },
    )

    assert package.provider == "atlas_review_request"
    assert package.inputs == {}
    assert package.metadata["mode"] == "noop"
    assert package.warnings == ({
        "code": "review_source_faq_ids_unsupported",
        "message": "Saved FAQ source selection is only supported for support-ticket runs.",
    },)


def test_competitive_mode_rejects_saved_faq_selection() -> None:
    package = build_content_ops_input_provider().build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "inputs": {
                "source_type": "competitive",
                "source_faq_ids": ["11111111-1111-1111-1111-111111111111"],
                "source_material": [_competitive_row()],
            }
        },
    )

    assert package.provider == "atlas_competitive_request"
    assert package.inputs == {}
    assert package.metadata["mode"] == "noop"
    assert package.warnings == ({
        "code": "competitive_source_faq_ids_unsupported",
        "message": "Saved FAQ source selection is only supported for support-ticket runs.",
    },)


def test_unknown_explicit_source_type_fails_closed() -> None:
    package = build_content_ops_input_provider().build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "inputs": {
                "source_type": "call_recordings",
                "source_material": [_review_row()],
            }
        },
    )

    assert package.inputs == {}
    assert package.metadata["mode"] == "noop"
    assert package.metadata["requested_source_type"] == "call_recordings"
    assert package.warnings == ({
        "code": "content_ops_source_type_unsupported",
        "message": "Unsupported Content Ops source type.",
        "source_type": "call_recordings",
    },)


@pytest.mark.asyncio
async def test_review_mode_fetches_persisted_targets_by_tenant_scope() -> None:
    pool = {
        "opportunities": {
            "review-1": [_generic_review_row("review-1")],
            "review-2": [{**_generic_review_row("review-2"), "vendor_name": "Teams"}],
        },
        "opportunity_calls": [],
    }
    provider = build_content_ops_input_provider(
        pool_provider=lambda: pool,
        opportunity_repository_factory=_OpportunityRepo,
    )

    package = await provider.build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "target_mode": "vendor_retention",
            "inputs": {
                "source_type": "reviews",
                "source_import_target_ids": ["review-1", "review-2"],
            },
        },
    )

    assert [call["target_id"] for call in pool["opportunity_calls"]] == [
        "review-1",
        "review-2",
    ]
    assert {call["account_id"] for call in pool["opportunity_calls"]} == {"acct-1"}
    assert package.provider == "atlas_review_request"
    assert package.outputs == _MARKETER_OUTPUTS
    assert package.metadata["source_target_loaded_count"] == 2
    assert package.metadata["included_row_count"] == 2
    assert {row["source_type"] for row in package.inputs["source_material"]} == {
        "reviews",
    }


@pytest.mark.asyncio
async def test_competitive_mode_fetches_persisted_targets_by_tenant_scope() -> None:
    pool = {
        "opportunities": {
            "competitive-1": [_competitive_row("competitive-1")],
            "competitive-2": [{
                **_competitive_row("competitive-2"),
                "from_vendor": "Notion",
                "to_vendor": "Confluence",
                "competitor": "Confluence",
            }],
        },
        "opportunity_calls": [],
    }
    provider = build_content_ops_input_provider(
        pool_provider=lambda: pool,
        opportunity_repository_factory=_OpportunityRepo,
    )

    package = await provider.build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "target_mode": "vendor_retention",
            "inputs": {
                "source_type": "competitive",
                "source_import_target_ids": ["competitive-1", "competitive-2"],
            },
        },
    )

    assert [call["target_id"] for call in pool["opportunity_calls"]] == [
        "competitive-1",
        "competitive-2",
    ]
    assert {call["account_id"] for call in pool["opportunity_calls"]} == {"acct-1"}
    assert package.provider == "atlas_competitive_request"
    assert package.outputs == _MARKETER_OUTPUTS
    assert package.metadata["source_target_loaded_count"] == 2
    assert package.metadata["included_row_count"] == 2
    assert package.inputs["competitive_alternatives"] == ["Teams", "Confluence"]


@pytest.mark.asyncio
async def test_competitive_mode_fetches_b2b_displacement_dynamics_by_tracked_vendor_scope() -> None:
    pool = _B2BDisplacementPool([_b2b_displacement_row()])
    provider = build_content_ops_input_provider(pool_provider=lambda: pool)

    package = await provider.build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "inputs": {
                "source_type": "competitive",
                "b2b_displacement_vendors": ["Slack", "Google Cloud"],
            },
        },
    )

    assert package.provider == "atlas_competitive_request"
    assert package.outputs == _MARKETER_OUTPUTS
    assert package.metadata["b2b_displacement_vendor_count"] == 2
    assert package.metadata["b2b_displacement_loaded_count"] == 1
    assert package.metadata["b2b_displacement_missing_vendor_count"] == 1
    assert package.metadata["included_row_count"] == 1
    row = package.inputs["competitive_source_material"][0]
    assert row["source_type"] == "competitive_displacement"
    assert row["source_id"] == "b2b_displacement:Slack->Teams:2026-06-01"
    assert row["target_id"] == "b2b_displacement:Slack->Teams:2026-06-01"
    assert row["from_vendor"] == "Slack"
    assert row["to_vendor"] == "Teams"
    assert row["competitor"] == "Teams"
    assert row["displacement_mention_count"] == 7
    assert row["primary_driver"] == "workflow friction"
    assert "Admins choose Teams" in row["text"]
    assert package.inputs["competitive_alternatives"] == ["Teams"]
    assert len(pool.calls) == 1
    query = pool.calls[0]["query"]
    assert "FROM b2b_displacement_dynamics" in query
    assert "FROM tracked_vendors tv" in query
    assert "tv.account_id = $1" in query
    assert pool.calls[0]["args"] == ("acct-1", ["slack", "google cloud"], 10)
    assert package.warnings[0]["code"] == "b2b_displacement_vendors_not_found"


@pytest.mark.asyncio
async def test_competitive_b2b_displacement_selection_requires_account_scope() -> None:
    pool = _B2BDisplacementPool([_b2b_displacement_row()])
    provider = build_content_ops_input_provider(pool_provider=lambda: pool)

    package = await provider.build_content_ops_input_package(
        scope=TenantScope(account_id=None),
        request={
            "inputs": {
                "source_type": "competitive",
                "b2b_displacement_vendors": ["Slack"],
            },
        },
    )

    assert package.provider == "atlas_competitive_request"
    assert package.inputs == {}
    assert package.metadata["mode"] == "noop"
    assert package.metadata["b2b_displacement_loaded_count"] == 0
    assert package.warnings[0]["code"] == "b2b_displacement_missing_account_scope"
    assert package.warnings[-1]["code"] == "competitive_source_material_empty"
    assert pool.calls == []


@pytest.mark.asyncio
async def test_competitive_b2b_displacement_selection_requires_configured_pool() -> None:
    provider = build_content_ops_input_provider()

    package = await provider.build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "inputs": {
                "source_type": "competitive",
                "b2b_displacement_vendors": ["Slack"],
            },
        },
    )

    assert package.inputs == {}
    assert package.metadata["mode"] == "noop"
    assert package.metadata["b2b_displacement_loaded_count"] == 0
    assert package.warnings[0]["code"] == "b2b_displacement_repository_unconfigured"
    assert package.warnings[-1]["code"] == "competitive_source_material_empty"


@pytest.mark.asyncio
async def test_competitive_b2b_displacement_selection_reports_missing_vendor() -> None:
    pool = _B2BDisplacementPool([])
    provider = build_content_ops_input_provider(pool_provider=lambda: pool)

    package = await provider.build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "inputs": {
                "source_type": "competitive",
                "b2b_displacement_vendors": ["Slack"],
            },
        },
    )

    assert package.inputs == {}
    assert package.metadata["mode"] == "noop"
    assert package.metadata["b2b_displacement_missing_vendor_count"] == 1
    assert package.warnings[0]["code"] == "b2b_displacement_vendors_not_found"
    assert package.warnings[-1]["code"] == "competitive_source_material_empty"


@pytest.mark.asyncio
async def test_review_input_evidence_reaches_landing_blog_and_sales_brief_generators() -> None:
    package = build_content_ops_input_provider().build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "inputs": {
                "source_type": "reviews",
                "source_material": [_generic_review_row()],
            }
        },
    )
    payload = merge_content_ops_input_package({"inputs": {}}, package)
    request = request_from_mapping(payload)

    result = await execute_content_ops_request(
        request,
        services=ContentOpsExecutionServices(
            blog_post=_RunnableService(),
            landing_page=_RunnableService(),
            sales_brief=_RunnableService(),
            social_post=_RunnableService(),
        ),
        scope=TenantScope(account_id="acct-1"),
    )

    landing_step = next(step for step in result.steps if step.output == "landing_page")
    landing_campaign = landing_step.result["kwargs"]["campaign"]
    assert landing_campaign.context["review_source_material"][0]["target_id"] == "review-1"

    blog_step = next(step for step in result.steps if step.output == "blog_post")
    blog_context = blog_step.result["kwargs"]["data_context"]
    assert blog_context["source"] == "review_input_provider"
    assert blog_context["review_source_material"][0]["target_id"] == "review-1"

    sales_brief_step = next(step for step in result.steps if step.output == "sales_brief")
    sales_brief_kwargs = sales_brief_step.result["kwargs"]
    assert sales_brief_kwargs["default_brief_type"] == "discovery"
    assert sales_brief_kwargs["source_material"][0]["target_id"] == "review-1"

    social_post_step = next(step for step in result.steps if step.output == "social_post")
    social_post_kwargs = social_post_step.result["kwargs"]
    assert social_post_kwargs["source_material"][0]["target_id"] == "review-1"
    assert social_post_kwargs["target_mode"] == "vendor_retention"


@pytest.mark.asyncio
async def test_competitive_input_evidence_reaches_landing_and_blog_generators() -> None:
    package = build_content_ops_input_provider().build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "inputs": {
                "source_type": "competitive",
                "source_material": [_competitive_row()],
            }
        },
    )
    payload = merge_content_ops_input_package({"inputs": {}}, package)
    request = request_from_mapping(payload)

    result = await execute_content_ops_request(
        request,
        services=ContentOpsExecutionServices(
            blog_post=_RunnableService(),
            landing_page=_RunnableService(),
            sales_brief=_RunnableService(),
            social_post=_RunnableService(),
        ),
        scope=TenantScope(account_id="acct-1"),
    )

    landing_step = next(step for step in result.steps if step.output == "landing_page")
    landing_campaign = landing_step.result["kwargs"]["campaign"]
    assert landing_campaign.context["competitive_source_material"][0]["target_id"] == "competitive-1"

    blog_step = next(step for step in result.steps if step.output == "blog_post")
    blog_context = blog_step.result["kwargs"]["data_context"]
    assert blog_context["source"] == "competitive_input_provider"
    assert blog_context["competitive_source_material"][0]["target_id"] == "competitive-1"

    sales_brief_step = next(step for step in result.steps if step.output == "sales_brief")
    sales_brief_kwargs = sales_brief_step.result["kwargs"]
    assert sales_brief_kwargs["default_brief_type"] == "displacement"
    assert sales_brief_kwargs["source_material"][0]["target_id"] == "competitive-1"

    social_post_step = next(step for step in result.steps if step.output == "social_post")
    social_post_kwargs = social_post_step.result["kwargs"]
    assert social_post_kwargs["source_material"][0]["target_id"] == "competitive-1"
    assert social_post_kwargs["target_mode"] == "vendor_retention"


@pytest.mark.skipif(
    api_module.APIRouter is None,
    reason="fastapi is not installed in this test environment",
)
@pytest.mark.asyncio
async def test_plan_route_applies_review_input_provider() -> None:
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/content-ops"),
        input_provider=build_content_ops_input_provider(),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            blog_post=_RunnableService(),
            landing_page=_RunnableService(),
            sales_brief=_RunnableService(),
            social_post=_RunnableService(),
        ),
        scope_provider=lambda: TenantScope(account_id="acct-review-route"),
    )
    route = _route(router, "/content-ops/plan", "POST")

    payload = await route.endpoint({
        "inputs": {
            "source_type": "reviews",
            "source_material": [_review_row()],
        },
    })

    assert payload["can_execute"] is True
    assert [step["output"] for step in payload["steps"]] == list(_MARKETER_OUTPUTS)
    assert payload["preview"]["outputs"] == list(_MARKETER_OUTPUTS)
