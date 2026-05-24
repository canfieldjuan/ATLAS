from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import pytest

from atlas_brain._content_ops_input_provider import build_content_ops_input_provider
from extracted_content_pipeline.api.control_surfaces import (
    ContentOpsControlSurfaceApiConfig,
    create_content_ops_control_surface_router,
)
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.content_ops_execution import ContentOpsExecutionServices


ROOT = Path(__file__).resolve().parents[1]


class _LandingPageCaptureService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        *,
        scope: TenantScope,
        campaign: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append({
            "scope": scope,
            "campaign": campaign,
            "kwargs": dict(kwargs),
        })
        return {
            "generated": 1,
            "saved_ids": ["landing-support-ticket-1"],
            "campaign_name": getattr(campaign, "name", ""),
        }


class _BlogPostCaptureService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        limit: int,
        filters: Mapping[str, Any] | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append({
            "scope": scope,
            "target_mode": target_mode,
            "limit": limit,
            "filters": dict(filters or {}),
            "kwargs": dict(kwargs),
        })
        return {
            "generated": 1,
            "saved_ids": ["blog-support-ticket-1"],
            "topic": kwargs.get("topic"),
        }


def _route(router: Any, path: str, method: str) -> Any:
    for route in router.routes:
        if getattr(route, "path", None) == path and method.upper() in getattr(
            route,
            "methods",
            set(),
        ):
            return route
    raise AssertionError(f"Route {method.upper()} {path!r} not mounted")


def _support_ticket_csv_rows() -> list[dict[str, str]]:
    path = ROOT / "extracted_content_pipeline" / "examples" / "support_ticket_sources.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _execute_router(*, service: Any, output: str) -> Any:
    services = (
        ContentOpsExecutionServices(landing_page=service)
        if output == "landing_page"
        else ContentOpsExecutionServices(blog_post=service)
    )
    return create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(),
        input_provider=build_content_ops_input_provider(),
        execution_services_provider=lambda: services,
        scope_provider=lambda: TenantScope(
            account_id="acct-support-ticket-content",
            user_id="user-support-ticket-content",
        ),
    )


@pytest.mark.asyncio
async def test_support_ticket_provider_feeds_landing_page_execute_context() -> None:
    service = _LandingPageCaptureService()
    router = _execute_router(service=service, output="landing_page")

    payload = await _route(router, "/content-ops/execute", "POST").endpoint({
        "outputs": ["landing_page"],
        "require_quality_gates": False,
        "inputs": {
            "source_material": _support_ticket_csv_rows(),
        },
    })

    assert payload["status"] == "completed"
    assert payload["plan"]["steps"][0]["runner"] == "LandingPageGenerationService.generate"
    assert payload["steps"][0]["result"]["saved_ids"] == ["landing-support-ticket-1"]

    assert len(service.calls) == 1
    call = service.calls[0]
    assert call["scope"].account_id == "acct-support-ticket-content"

    campaign = call["campaign"]
    assert campaign.name == "FAQ Report"
    assert campaign.value_prop == (
        "Turn repeat support tickets into clear FAQ answers customers can use "
        "before they email support"
    )
    assert campaign.persona == "Small teams answering repeat support questions"

    context = campaign.context
    assert context["target_keyword"] == "support ticket FAQ report"
    assert context["secondary_keywords"] == [
        "customer support FAQ",
        "reduce repeat support tickets",
        "help center answers from support tickets",
    ]
    assert context["search_intent"] == (
        "Small teams looking for a practical way to turn repeat support "
        "questions into help-center answers."
    )
    assert context["source_period"] == "Uploaded support tickets"
    assert context["cta_label"] == "Upload Ticket CSV -- Free Analysis"
    assert context["cta_url"] == "/systems/ai-content-ops/intake"
    assert context["faq_questions"] == [
        "How do I change my login email?",
        "How do we export campaign attribution data before renewal?",
    ]
    assert "source_material" not in context


@pytest.mark.asyncio
async def test_support_ticket_provider_feeds_blog_post_execute_inputs() -> None:
    service = _BlogPostCaptureService()
    router = _execute_router(service=service, output="blog_post")

    payload = await _route(router, "/content-ops/execute", "POST").endpoint({
        "outputs": ["blog_post"],
        "limit": 2,
        "require_quality_gates": False,
        "inputs": {
            "source_material": _support_ticket_csv_rows(),
        },
    })

    assert payload["status"] == "completed"
    assert payload["plan"]["steps"][0]["runner"] == "BlogPostGenerationService.generate"
    assert payload["steps"][0]["result"] == {
        "generated": 1,
        "saved_ids": ["blog-support-ticket-1"],
        "topic": "Support-ticket questions customers keep asking",
    }

    assert service.calls == [
        {
            "scope": TenantScope(
                account_id="acct-support-ticket-content",
                user_id="user-support-ticket-content",
            ),
            "target_mode": "vendor_retention",
            "limit": 2,
            "filters": {"topic_type": "content_ops_support_ticket_faq"},
            "kwargs": {
                "temperature": 0.3,
                "max_tokens": 4096,
                "parse_retry_attempts": 1,
                "parse_retry_response_excerpt_chars": 800,
                "quality_gates_enabled": False,
                "quality_repair_attempts": 2,
                "topic": "Support-ticket questions customers keep asking",
            },
        }
    ]
