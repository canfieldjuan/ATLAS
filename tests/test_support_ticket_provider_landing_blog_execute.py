from __future__ import annotations

import asyncio
import csv
from pathlib import Path
from typing import Any, Mapping

import pytest

from atlas_brain._content_ops_input_provider import build_content_ops_input_provider
import extracted_content_pipeline.api.control_surfaces as api_module
from extracted_content_pipeline.api.control_surfaces import (
    ContentOpsControlSurfaceApiConfig,
    create_content_ops_control_surface_router,
)
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.content_ops_execution import ContentOpsExecutionServices
from extracted_content_pipeline.support_ticket_input_provider import (
    SupportTicketInputProvider,
)
from extracted_content_pipeline.ticket_faq_ports import TicketFAQDraft


ROOT = Path(__file__).resolve().parents[1]
FAQ_DRAFT_ID = "11111111-1111-4111-8111-111111111111"


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


class _BlockingLandingPageCaptureService(_LandingPageCaptureService):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

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
        self.started.set()
        await self.release.wait()
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


class _FAQRepo:
    def __init__(self, pool: dict[str, Any]) -> None:
        self.pool = pool

    async def get_draft(self, faq_id: str, *, scope: TenantScope) -> TicketFAQDraft | None:
        self.pool["calls"].append((faq_id, scope.account_id))
        return self.pool["drafts"].get(faq_id)


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


def _saved_faq_draft() -> TicketFAQDraft:
    return TicketFAQDraft(
        id=FAQ_DRAFT_ID,
        target_id="ticket-faq-report",
        target_mode="support_ticket_faq",
        title="Saved FAQ report",
        markdown="# Saved FAQ report",
        items=(
            {
                "topic": "billing confusion",
                "question": "Why was I charged twice?",
                "summary": "Customers ask why duplicate-looking invoices appear.",
                "steps": (
                    "Open Billing, choose invoice history, and compare pending charges.",
                    "Contact support with the invoice ID if both charges settled.",
                ),
                "answer_evidence_status": "resolution_evidence",
                "source_ids": ("ticket-billing-1", "ticket-billing-2"),
            },
        ),
        source_count=2,
        ticket_source_count=2,
        output_checks={"has_action_items": True},
        status="draft",
    )


def _generated_support_ticket_rows(row_count: int) -> list[dict[str, str]]:
    return [
        {
            "ticket_id": f"ticket-{index}",
            "source_type": "support_ticket",
            "subject": (
                "How do I change my login email?"
                if index % 2 == 0
                else "How do I export renewal billing data?"
            ),
            "description": (
                "Customer cannot find the account email setting before launch."
                if index % 2 == 0
                else "Customer needs renewal invoice data before finance review."
            ),
            "pain_category": "account" if index % 2 == 0 else "billing",
        }
        for index in range(row_count)
    ]


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


def _selected_faq_execute_router(
    *,
    service: Any,
    output: str,
    pool: dict[str, Any],
) -> Any:
    services = (
        ContentOpsExecutionServices(landing_page=service)
        if output == "landing_page"
        else ContentOpsExecutionServices(blog_post=service)
    )
    return create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(),
        input_provider=build_content_ops_input_provider(
            pool_provider=lambda: pool,
            faq_repository_factory=_FAQRepo,
        ),
        execution_services_provider=lambda: services,
        scope_provider=lambda: TenantScope(
            account_id="acct-support-ticket-content",
            user_id="user-support-ticket-content",
        ),
    )


def _loader_backed_router(
    *,
    rows: list[dict[str, str]],
    services: ContentOpsExecutionServices | None = None,
    execute_max_concurrency: int = 8,
) -> Any:
    return create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(
            execute_max_concurrency=execute_max_concurrency,
        ),
        execution_services_provider=(lambda: services) if services is not None else None,
        input_provider=SupportTicketInputProvider(
            source_material_loader=lambda scope, request: rows,
        ),
        scope_provider=lambda: TenantScope(
            account_id="acct-support-ticket-stress",
            user_id="user-support-ticket-stress",
        ),
    )


def _assert_provider_counts(
    payload: Mapping[str, Any],
    *,
    source_row_count: int,
    included_row_count: int,
) -> None:
    diagnostics = payload["input_provider"]
    assert diagnostics["provider"] == "support_ticket_input_provider"
    assert diagnostics["metadata"]["source_row_count"] == source_row_count
    assert diagnostics["metadata"]["included_row_count"] == included_row_count
    assert diagnostics["metadata"]["truncated_row_count"] == max(
        0,
        source_row_count - included_row_count,
    )
    if source_row_count > included_row_count:
        assert diagnostics["warnings"] == [
            {
                "code": "ticket_rows_truncated",
                "message": f"Used first {included_row_count} ticket rows out of {source_row_count}.",
                "row_count": source_row_count,
                "max_rows": included_row_count,
                "truncated_row_count": source_row_count - included_row_count,
            }
        ]
    else:
        assert diagnostics["warnings"] == []


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
        "repeat support questions",
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
        "How do we export the campaign reporting dashboard before renewal?",
    ]
    assert context["has_dated_window"] is False
    assert context["support_ticket_resolution_evidence_present"] is False
    assert context["support_ticket_resolution_evidence_count"] == 0
    assert context["has_measured_outcomes"] is False
    assert context["measured_outcome_count"] == 0
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

    assert len(service.calls) == 1
    call = service.calls[0]
    assert call["scope"] == TenantScope(
        account_id="acct-support-ticket-content",
        user_id="user-support-ticket-content",
    )
    assert call["target_mode"] == "vendor_retention"
    assert call["limit"] == 2
    assert call["filters"] == {"topic_type": "content_ops_support_ticket_faq"}
    kwargs = dict(call["kwargs"])
    data_context = kwargs.pop("data_context")
    assert kwargs == {
        "temperature": 0.3,
        "max_tokens": 4096,
        "parse_retry_attempts": 1,
        "parse_retry_response_excerpt_chars": 800,
        "quality_gates_enabled": False,
        "quality_repair_attempts": 2,
        "topic": "Support-ticket questions customers keep asking",
    }
    assert data_context["source"] == "support_ticket_provider"
    assert data_context["included_ticket_row_count"] == 4
    assert data_context["top_clusters"] == [
        {"label": "email and profile updates", "count": 2},
        {"label": "reporting friction", "count": 2},
    ]
    assert data_context["customer_wording_examples"][0]["source_id"] == "ticket-acme-1"
    assert data_context["has_dated_window"] is False
    assert data_context["support_ticket_resolution_evidence_present"] is False
    assert data_context["support_ticket_resolution_evidence_count"] == 0
    assert "support_ticket_resolution_examples" not in data_context
    assert data_context["has_measured_outcomes"] is False
    assert data_context["measured_outcome_count"] == 0
    assert "measured_outcome_examples" not in data_context


@pytest.mark.asyncio
async def test_support_ticket_provider_threads_resolution_evidence_into_blog_context() -> None:
    service = _BlogPostCaptureService()
    router = _execute_router(service=service, output="blog_post")

    await _route(router, "/content-ops/execute", "POST").endpoint({
        "outputs": ["blog_post"],
        "require_quality_gates": False,
        "inputs": {
            "source_material": [
                {
                    "ticket_id": "ticket-1",
                    "subject": "How do I export reports?",
                    "description": "Where do I export the dashboard?",
                    "created_at": "2026-05-01",
                    "resolution_text": "Open Reports, choose Export, then select CSV.",
                    "measured_outcome": "Repeat reporting tickets fell from 9 to 4.",
                }
            ],
        },
    })

    data_context = service.calls[0]["kwargs"]["data_context"]
    assert data_context["has_dated_window"] is True
    assert data_context["source_period"] == "Last 90 days of support tickets"
    assert data_context["support_ticket_resolution_evidence_present"] is True
    assert data_context["support_ticket_resolution_evidence_count"] == 1
    assert data_context["support_ticket_resolution_examples"] == [
        {
            "source_id": "ticket-1",
            "source_title": "How do I export reports?",
            "text": "Open Reports, choose Export, then select CSV.",
        }
    ]
    assert data_context["has_measured_outcomes"] is True
    assert data_context["measured_outcome_count"] == 1
    assert data_context["measured_outcome_examples"] == [
        {
            "source_id": "ticket-1",
            "source_title": "How do I export reports?",
            "text": "Repeat reporting tickets fell from 9 to 4.",
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("row_count", [1_000, 10_000, 50_000])
async def test_loader_backed_support_ticket_provider_surfaces_scale_diagnostics(
    row_count: int,
) -> None:
    rows = _generated_support_ticket_rows(row_count)
    router = _loader_backed_router(rows=rows)

    preview = await _route(router, "/content-ops/preview", "POST").endpoint({
        "outputs": ["landing_page"],
    })
    plan = await _route(router, "/content-ops/plan", "POST").endpoint({
        "outputs": ["blog_post"],
    })

    assert preview["can_run"] is True
    assert preview["outputs"] == ["landing_page"]
    _assert_provider_counts(
        preview,
        source_row_count=row_count,
        included_row_count=min(row_count, 1_000),
    )

    assert plan["can_execute"] is True
    assert plan["steps"][0]["output"] == "blog_post"
    _assert_provider_counts(
        plan,
        source_row_count=row_count,
        included_row_count=min(row_count, 1_000),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ["landing_page", "blog_post"])
async def test_loader_backed_support_ticket_provider_bounds_large_execute_inputs(
    output: str,
) -> None:
    rows = _generated_support_ticket_rows(50_000)
    service = (
        _LandingPageCaptureService()
        if output == "landing_page"
        else _BlogPostCaptureService()
    )
    services = (
        ContentOpsExecutionServices(landing_page=service)
        if output == "landing_page"
        else ContentOpsExecutionServices(blog_post=service)
    )
    router = _loader_backed_router(rows=rows, services=services)

    payload = await _route(router, "/content-ops/execute", "POST").endpoint({
        "outputs": [output],
        "require_quality_gates": False,
    })

    assert payload["status"] == "completed"
    _assert_provider_counts(
        payload,
        source_row_count=50_000,
        included_row_count=1_000,
    )
    assert len(service.calls) == 1

    if output == "landing_page":
        context = service.calls[0]["campaign"].context
        assert "source_material" not in context
        assert context["source_row_count"] == 50_000
        assert context["included_ticket_row_count"] == 1_000
        assert context["truncated_ticket_row_count"] == 49_000
        assert context["has_dated_window"] is False
        assert context["support_ticket_resolution_evidence_present"] is False
        assert context["support_ticket_resolution_evidence_count"] == 0
        assert context["has_measured_outcomes"] is False
        assert context["measured_outcome_count"] == 0
        assert len(context["customer_wording_examples"]) <= 6
        assert len(context["top_ticket_clusters"]) <= 6
    else:
        assert service.calls[0]["filters"] == {
            "topic_type": "content_ops_support_ticket_faq",
        }
        assert service.calls[0]["kwargs"]["topic"] == (
            "Support-ticket questions customers keep asking"
        )
        data_context = service.calls[0]["kwargs"]["data_context"]
        assert data_context["source"] == "support_ticket_provider"
        assert data_context["source_row_count"] == 50_000
        assert data_context["included_ticket_row_count"] == 1_000
        assert data_context["question_like_ticket_count"] == 1_000
        assert data_context["has_dated_window"] is False
        assert data_context["support_ticket_resolution_evidence_present"] is False
        assert data_context["support_ticket_resolution_evidence_count"] == 0
        assert data_context["has_measured_outcomes"] is False
        assert data_context["measured_outcome_count"] == 0
        assert len(data_context["customer_wording_examples"]) <= 6
        assert len(data_context["top_ticket_clusters"]) <= 6


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ["landing_page", "blog_post"])
async def test_selected_faq_id_feeds_execute_context(output: str) -> None:
    service = (
        _LandingPageCaptureService()
        if output == "landing_page"
        else _BlogPostCaptureService()
    )
    pool = {
        "drafts": {FAQ_DRAFT_ID: _saved_faq_draft()},
        "calls": [],
    }
    router = _selected_faq_execute_router(
        service=service,
        output=output,
        pool=pool,
    )

    payload = await _route(router, "/content-ops/execute", "POST").endpoint({
        "outputs": [output],
        "require_quality_gates": False,
        "inputs": {"source_faq_ids": [FAQ_DRAFT_ID]},
    })

    assert pool["calls"] == [(FAQ_DRAFT_ID, "acct-support-ticket-content")]
    assert payload["status"] == "completed"
    assert payload["errors"] == []
    assert payload["input_provider"]["provider"] == "atlas_support_ticket_request"
    assert payload["input_provider"]["metadata"]["source_row_count"] == 1
    assert payload["input_provider"]["metadata"]["included_row_count"] == 1
    assert payload["input_provider"]["warnings"] == []
    assert len(service.calls) == 1

    expected_resolution = (
        "Open Billing, choose invoice history, and compare pending charges. "
        "Contact support with the invoice ID if both charges settled."
    )
    if output == "landing_page":
        context = service.calls[0]["campaign"].context
        assert "source_material" not in context
        assert context["faq_questions"] == ["Why was I charged twice?"]
        assert context["customer_wording_examples"][0]["source_id"] == FAQ_DRAFT_ID
        assert "Customers ask why duplicate-looking invoices appear." in (
            context["customer_wording_examples"][0]["text"]
        )
        assert context["support_ticket_resolution_evidence_present"] is True
        assert context["support_ticket_resolution_evidence_count"] == 1
        assert context["support_ticket_resolution_examples"] == [
            {
                "source_id": FAQ_DRAFT_ID,
                "source_title": "Why was I charged twice?",
                "text": expected_resolution,
            }
        ]
    else:
        data_context = service.calls[0]["kwargs"]["data_context"]
        assert "source_material" not in data_context
        assert data_context["faq_questions"] == ["Why was I charged twice?"]
        assert data_context["customer_wording_examples"][0]["source_id"] == FAQ_DRAFT_ID
        assert "Customers ask why duplicate-looking invoices appear." in (
            data_context["customer_wording_examples"][0]["text"]
        )
        assert data_context["support_ticket_resolution_evidence_present"] is True
        assert data_context["support_ticket_resolution_evidence_count"] == 1
        assert data_context["support_ticket_resolution_examples"] == [
            {
                "source_id": FAQ_DRAFT_ID,
                "source_title": "Why was I charged twice?",
                "text": expected_resolution,
            }
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ["landing_page", "blog_post"])
async def test_loader_backed_support_ticket_execute_is_stable_under_concurrency(
    output: str,
) -> None:
    rows = _generated_support_ticket_rows(10_000)
    service = (
        _LandingPageCaptureService()
        if output == "landing_page"
        else _BlogPostCaptureService()
    )
    services = (
        ContentOpsExecutionServices(landing_page=service)
        if output == "landing_page"
        else ContentOpsExecutionServices(blog_post=service)
    )
    router = _loader_backed_router(
        rows=rows,
        services=services,
        execute_max_concurrency=25,
    )
    route = _route(router, "/content-ops/execute", "POST")

    async def execute_request(index: int) -> dict[str, Any]:
        await asyncio.sleep(0)
        payload = {
            "outputs": [output],
            "limit": index + 1,
            "require_quality_gates": False,
        }
        if output == "landing_page":
            payload["inputs"] = {"cta_label": f"Request {index} CTA"}
        return await route.endpoint(payload)

    responses = await asyncio.gather(
        *(execute_request(index) for index in range(25)),
    )

    assert len(responses) == 25
    assert len(service.calls) == 25
    for response in responses:
        assert response["status"] == "completed"
        _assert_provider_counts(
            response,
            source_row_count=10_000,
            included_row_count=1_000,
        )

    if output == "landing_page":
        labels = {
            call["campaign"].context["cta_label"]
            for call in service.calls
        }
        assert labels == {f"Request {index} CTA" for index in range(25)}
    else:
        limits = {call["limit"] for call in service.calls}
        assert limits == set(range(1, 26))


@pytest.mark.asyncio
async def test_loader_backed_support_ticket_execute_rejects_overload_before_loading_source() -> None:
    rows = _generated_support_ticket_rows(50_000)
    loader_calls: list[dict[str, Any]] = []

    def loader(scope: TenantScope, request: Mapping[str, Any]) -> list[dict[str, str]]:
        loader_calls.append({"scope": scope, "request": dict(request)})
        return rows

    service = _BlockingLandingPageCaptureService()
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(execute_max_concurrency=1),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            landing_page=service,
        ),
        input_provider=SupportTicketInputProvider(source_material_loader=loader),
        scope_provider=lambda: TenantScope(
            account_id="acct-support-ticket-overload",
            user_id="user-support-ticket-overload",
        ),
    )
    route = _route(router, "/content-ops/execute", "POST")
    request = {
        "outputs": ["landing_page"],
        "require_quality_gates": False,
    }

    first = asyncio.create_task(route.endpoint(request))
    try:
        await asyncio.wait_for(service.started.wait(), timeout=1)
        assert len(loader_calls) == 1

        with pytest.raises(api_module.HTTPException) as exc:
            await route.endpoint(request)

        assert exc.value.status_code == 429
        assert exc.value.detail == {
            "reason": "content_ops_execute_at_capacity",
            "max_concurrency": 1,
        }
        assert len(loader_calls) == 1
        assert len(service.calls) == 1
    finally:
        service.release.set()

    first_payload = await first
    assert first_payload["status"] == "completed"

    second_payload = await route.endpoint(request)
    assert second_payload["status"] == "completed"
    assert len(loader_calls) == 2
    assert len(service.calls) == 2
