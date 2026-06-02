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


class _RunnableService:
    async def generate(self, **kwargs):  # pragma: no cover - plan route only.
        return {"kwargs": kwargs}


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


def _support_ticket_row() -> dict[str, Any]:
    return {
        "ticket_id": "ticket-1",
        "subject": "Billing question",
        "message": "Why was I charged twice this month?",
    }


def test_review_source_material_builds_landing_and_blog_inputs() -> None:
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
    assert request.outputs == ("landing_page", "blog_post")
    assert request.ingestion_profile == "existing_evidence"
    assert request.inputs["source_type"] == "reviews"
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
    assert package.outputs == ("landing_page", "blog_post")
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
    assert package.outputs == ("landing_page", "blog_post")
    assert package.metadata["source_target_loaded_count"] == 2
    assert package.metadata["included_row_count"] == 2
    assert {row["source_type"] for row in package.inputs["source_material"]} == {
        "reviews",
    }


@pytest.mark.asyncio
async def test_review_input_evidence_reaches_landing_and_blog_generators() -> None:
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
    assert [step["output"] for step in payload["steps"]] == [
        "landing_page",
        "blog_post",
    ]
    assert payload["preview"]["outputs"] == ["landing_page", "blog_post"]
