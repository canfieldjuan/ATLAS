from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from atlas_brain._content_ops_input_provider import build_content_ops_input_provider
import extracted_content_pipeline.api.control_surfaces as api_module
from extracted_content_pipeline.api.control_surfaces import (
    ContentOpsControlSurfaceApiConfig,
    create_content_ops_control_surface_router,
)
from extracted_content_pipeline.blog_generation import (
    BlogPostGenerationConfig,
    BlogPostGenerationService,
)
from extracted_content_pipeline.campaign_ports import LLMMessage, LLMResponse
from extracted_content_pipeline.content_ops_execution import (
    ContentOpsExecutionServices,
    execute_content_ops_from_mapping,
)
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.faq_deflection_report import FAQDeflectionReportService
from extracted_content_pipeline.landing_page_generation import (
    LandingPageGenerationService,
)
from extracted_content_pipeline.ticket_faq_markdown import TicketFAQMarkdownService
from extracted_content_pipeline.ticket_faq_ports import TicketFAQDraft
from extracted_quality_gate.types import QualityPolicy
from tests.content_ops_live_execute_harness import (
    MemoryDraftRepository,
    MemorySkillStore,
    build_content_ops_live_execute_harness,
    default_content_ops_execute_payload,
)


pytestmark = pytest.mark.skipif(
    api_module.APIRouter is None,
    reason="fastapi is not installed in this test environment",
)

FAQ_DRAFT_ID = "11111111-1111-4111-8111-111111111111"


def _route(router, path: str, method: str):
    for route in router.routes:
        methods = getattr(route, "methods", set())
        if getattr(route, "path", None) == path and method.upper() in methods:
            return route
    raise AssertionError(f"route not found: {method} {path}")


def _support_ticket_csv_rows() -> list[dict[str, str]]:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "extracted_content_pipeline"
        / "examples"
        / "support_ticket_sources.csv"
    )
    with source_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class _RecordingContentLLM:
    def __init__(self, *, blog_content_suffix: str = "") -> None:
        self.calls: list[dict[str, Any]] = []
        self.blog_content_suffix = blog_content_suffix

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int,
        temperature: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        del max_tokens, temperature
        meta = dict(metadata or {})
        self.calls.append({
            "messages": tuple(messages),
            "metadata": meta,
        })
        content = json.dumps(
            _real_service_response_for_metadata(
                meta,
                blog_content_suffix=self.blog_content_suffix,
            ),
            separators=(",", ":"),
        )
        return LLMResponse(
            content=content,
            model="support-ticket-provider-real-service-test",
            usage={"input_tokens": 11, "output_tokens": 17},
        )


class _RecordingBlueprintRepository:
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = [dict(row) for row in rows]
        self.calls: list[dict[str, Any]] = []

    async def read_blog_blueprints(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        limit: int,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[dict[str, Any]]:
        self.calls.append({
            "scope": scope,
            "target_mode": target_mode,
            "limit": limit,
            "filters": dict(filters or {}),
        })
        return self.rows[:limit]


class _SelectedFAQRepository:
    def __init__(self, pool: dict[str, Any]) -> None:
        self.pool = pool

    async def get_draft(
        self,
        faq_id: str,
        *,
        scope: TenantScope,
    ) -> TicketFAQDraft | None:
        self.pool["calls"].append((faq_id, scope.account_id))
        return self.pool["drafts"].get(faq_id)


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
        output_checks={
            "uses_user_vocabulary": True,
            "condensed": True,
            "has_action_items": True,
        },
        metadata={"source_period": "Uploaded support tickets"},
        status="draft",
    )


def _real_service_response_for_metadata(
    metadata: Mapping[str, Any],
    *,
    blog_content_suffix: str = "",
) -> Mapping[str, Any]:
    asset_type = str(metadata.get("asset_type") or "")
    if asset_type == "landing_page":
        return {
            "title": "FAQ Report From Support Tickets",
            "slug": "faq-report-from-support-tickets",
            "hero": {
                "headline": "Turn repeat support tickets into answers",
                "subheadline": (
                    "Use customer wording from old tickets to make answers "
                    "easier to find."
                ),
            },
            "sections": [{
                "id": "problem",
                "title": "Repeat Questions Mean Missing Answers",
                "body_markdown": (
                    "When customers keep asking the same question, the answer "
                    "is not where they are looking."
                ),
                "metadata": {"kind": "problem"},
            }],
            "cta": {
                "label": "Upload Ticket CSV -- Free Analysis",
                "url": "/systems/ai-content-ops/intake",
            },
            "meta": {
                "title_tag": "FAQ Report From Support Tickets",
                "description": (
                    "Turn your last 90 days of support tickets into FAQ "
                    "answers customers can use before they email support."
                ),
            },
            "reference_ids": ["ticket-acme-1", "ticket-northstar-1"],
        }
    if asset_type == "blog_post":
        return {
            "slug": "support-ticket-faq-report",
            "title": "Support Ticket FAQ Report",
            "content": (
                "## What does a support ticket FAQ report show?\n\n"
                "A support ticket FAQ report shows small teams the repeat "
                "questions customers keep asking across the uploaded support "
                "tickets. In the uploaded support tickets, 4 included rows show "
                "account and reporting questions that should become answers "
                "people can find before they email support.\n\n"
                "## Which support ticket answers should be written first?\n\n"
                "The first answers should cover the account and reporting "
                "questions customers already asked in the support tickets. "
                "Those support ticket patterns give the team a grounded FAQ "
                "list to write from customer wording."
                f"{blog_content_suffix}"
            ),
            "topic_type": "content_ops_support_ticket_faq",
            "description": "How repeat support tickets become FAQ articles.",
            "seo_title": "Support Ticket FAQ Report",
            "seo_description": (
                "Use support tickets to find repeat customer questions and "
                "write clearer FAQ answers."
            ),
            "target_keyword": "support ticket FAQ report",
            "secondary_keywords": [
                "customer support FAQ",
                "repeat support questions",
            ],
            "faq": [
                {
                    "question": "What does a support ticket FAQ report show?",
                    "answer": "It shows repeat customer questions from support tickets.",
                },
                {
                    "question": "Which support ticket answers should be written first?",
                    "answer": "Start with the highest-volume repeated questions.",
                },
                {
                    "question": "Why use customer wording?",
                    "answer": "Customers search with their own words.",
                },
            ],
        }
    raise AssertionError(f"unexpected LLM metadata: {metadata!r}")


def _message_texts(call: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(message.content) for message in call["messages"])


async def test_live_execute_harness_runs_all_outputs_with_reasoning() -> None:
    harness = build_content_ops_live_execute_harness()
    payload = await execute_content_ops_from_mapping(
        default_content_ops_execute_payload(),
        services=harness.services.with_reasoning_context(harness.reasoning),
        scope=harness.scope,
    )

    assert payload["status"] == "completed"
    steps = {step["output"]: step for step in payload["steps"]}
    assert set(steps) == {
        "email_campaign",
        "blog_post",
        "report",
        "landing_page",
        "sales_brief",
        "signal_extraction",
    }

    for output in ("email_campaign", "blog_post", "report", "landing_page", "sales_brief"):
        assert steps[output]["status"] == "completed"
        assert steps[output]["result"]["generated"] >= 1
        assert steps[output]["result"]["saved_ids"]
        assert steps[output]["reasoning"]["provider_configured"] is True
        assert steps[output]["reasoning"]["contexts_used"] >= 1
        assert steps[output]["reasoning"]["consumed_contexts"]

    assert steps["signal_extraction"]["status"] == "completed"
    assert steps["signal_extraction"]["result"]["generated"] == 1
    assert steps["signal_extraction"]["reasoning"]["requirement"] == "absent"

    repositories = {
        "campaigns": harness.campaigns,
        "blog_posts": harness.blog_posts,
        "reports": harness.reports,
        "landing_pages": harness.landing_pages,
        "sales_briefs": harness.sales_briefs,
    }
    assert {name: len(repo.saved) for name, repo in repositories.items()} == {
        "campaigns": 2,
        "blog_posts": 1,
        "reports": 1,
        "landing_pages": 1,
        "sales_briefs": 1,
    }
    for repository in repositories.values():
        for scope, _draft in repository.saved:
            assert scope.account_id == "acct-live"
            assert scope.user_id == "user-live"

    assert {call["skill_name"] for call in harness.llm.calls} == {
        "digest/b2b_campaign_generation",
        "digest/blog_post_generation",
        "digest/report_generation",
        "digest/landing_page_generation",
        "digest/sales_brief_generation",
    }


async def test_live_execute_route_persists_all_outputs_with_reasoning() -> None:
    harness = build_content_ops_live_execute_harness()
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(),
        execution_services_provider=lambda: harness.services,
        scope_provider=lambda: harness.scope,
        reasoning_context_provider=lambda: harness.reasoning,
    )

    route = _route(router, "/content-ops/execute", "POST")
    payload = await route.endpoint(default_content_ops_execute_payload())

    assert payload["status"] == "completed"
    steps = {step["output"]: step for step in payload["steps"]}
    assert set(steps) == {
        "email_campaign",
        "blog_post",
        "report",
        "landing_page",
        "sales_brief",
        "signal_extraction",
    }

    for output in ("email_campaign", "blog_post", "report", "landing_page", "sales_brief"):
        step = steps[output]
        assert step["status"] == "completed"
        assert step["result"]["generated"] >= 1
        assert step["result"]["saved_ids"]
        assert step["reasoning"]["provider_configured"] is True
        assert step["reasoning"]["contexts_used"] >= 1
        assert step["reasoning"]["consumed_contexts"]

    assert steps["signal_extraction"]["status"] == "completed"
    assert steps["signal_extraction"]["result"]["generated"] == 1
    assert steps["signal_extraction"]["reasoning"]["requirement"] == "absent"

    repositories = {
        "campaigns": harness.campaigns,
        "blog_posts": harness.blog_posts,
        "reports": harness.reports,
        "landing_pages": harness.landing_pages,
        "sales_briefs": harness.sales_briefs,
    }
    assert {name: len(repo.saved) for name, repo in repositories.items()} == {
        "campaigns": 2,
        "blog_posts": 1,
        "reports": 1,
        "landing_pages": 1,
        "sales_briefs": 1,
    }
    for repository in repositories.values():
        for scope, _draft in repository.saved:
            assert scope.account_id == "acct-live"
            assert scope.user_id == "user-live"


async def test_support_ticket_provider_feeds_real_landing_page_generation() -> None:
    llm = _RecordingContentLLM()
    landing_pages = MemoryDraftRepository("landing")
    service = LandingPageGenerationService(
        landing_pages=landing_pages,
        llm=llm,
        skills=MemorySkillStore(),
    )
    scope = TenantScope(
        account_id="acct-support-ticket-real",
        user_id="user-support-ticket-real",
    )
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            landing_page=service,
        ),
        scope_provider=lambda: scope,
        input_provider=build_content_ops_input_provider(),
    )

    route = _route(router, "/content-ops/execute", "POST")
    payload = await route.endpoint({
        "outputs": ["landing_page"],
        "limit": 1,
        "require_quality_gates": False,
        "inputs": {"source_material": _support_ticket_csv_rows()},
    })

    assert payload["status"] == "completed"
    step = payload["steps"][0]
    assert step["output"] == "landing_page"
    assert step["status"] == "completed"
    assert step["result"]["saved_ids"] == ["landing-1"]

    assert len(landing_pages.saved) == 1
    saved_scope, draft = landing_pages.saved[0]
    assert saved_scope == scope
    assert draft.campaign_name == "FAQ Report"
    assert draft.persona == "Small teams answering repeat support questions"
    assert (
        draft.value_prop
        == "Turn repeat support tickets into clear FAQ answers customers can "
        "use before they email support"
    )
    assert draft.cta == {
        "label": "Upload Ticket CSV -- Free Analysis",
        "url": "/systems/ai-content-ops/intake",
    }
    assert draft.reference_ids == ("ticket-acme-1", "ticket-northstar-1")

    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["metadata"]["skill_name"] == "digest/landing_page_generation"
    assert call["metadata"]["asset_type"] == "landing_page"
    system_prompt, user_prompt = _message_texts(call)
    assert "support ticket FAQ report" in system_prompt
    assert "Uploaded support tickets" in system_prompt
    assert "Last 90 days of support tickets" not in system_prompt
    assert "How do I change my login email?" in system_prompt
    assert "How do we export campaign attribution data before renewal?" in system_prompt
    assert "Generate one landing page" in user_prompt


async def test_support_ticket_provider_feeds_real_blog_post_generation() -> None:
    blueprints = _RecordingBlueprintRepository([{
        "id": "bp-support-ticket-faq",
        "slug": "support-ticket-faq-report",
        "topic": "Support-ticket questions customers keep asking",
        "topic_type": "content_ops_support_ticket_faq",
    }])
    llm = _RecordingContentLLM()
    blog_posts = MemoryDraftRepository("blog")
    service = BlogPostGenerationService(
        blueprints=blueprints,
        blog_posts=blog_posts,
        llm=llm,
        skills=MemorySkillStore(),
    )
    scope = TenantScope(
        account_id="acct-support-ticket-real",
        user_id="user-support-ticket-real",
    )
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            blog_post=service,
        ),
        scope_provider=lambda: scope,
        input_provider=build_content_ops_input_provider(),
    )

    route = _route(router, "/content-ops/execute", "POST")
    payload = await route.endpoint({
        "outputs": ["blog_post"],
        "limit": 1,
        "require_quality_gates": False,
        "inputs": {"source_material": _support_ticket_csv_rows()},
    })

    assert payload["status"] == "completed"
    step = payload["steps"][0]
    assert step["output"] == "blog_post"
    assert step["status"] == "completed"
    assert step["result"]["saved_ids"] == ["blog-1"]

    assert blueprints.calls == [{
        "scope": scope,
        "target_mode": "vendor_retention",
        "limit": 1,
        "filters": {"topic_type": "content_ops_support_ticket_faq"},
    }]
    assert len(blog_posts.saved) == 1
    saved_scope, draft = blog_posts.saved[0]
    assert saved_scope == scope
    assert draft.title == "Support Ticket FAQ Report"
    assert draft.topic_type == "content_ops_support_ticket_faq"
    assert draft.metadata["target_keyword"] == "support ticket FAQ report"
    assert draft.metadata["secondary_keywords"] == [
        "customer support FAQ",
        "repeat support questions",
    ]
    assert draft.data_context["source_period"] == "Uploaded support tickets"
    assert draft.data_context["source"] == "support_ticket_provider"
    assert draft.data_context["included_ticket_row_count"] == 4
    assert draft.data_context["top_clusters"]

    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["metadata"]["skill_name"] == "digest/blog_post_generation"
    assert call["metadata"]["asset_type"] == "blog_post"
    system_prompt, user_prompt = _message_texts(call)
    assert "Support-ticket questions customers keep asking" not in system_prompt
    assert "Support-ticket questions customers keep asking" in user_prompt
    assert "content_ops_support_ticket_faq" in user_prompt
    assert "support_ticket_provider" in user_prompt
    assert "included_ticket_row_count" in user_prompt
    assert "Generate one blog post" in user_prompt


async def test_selected_faq_id_feeds_real_landing_and_blog_generation() -> None:
    blueprints = _RecordingBlueprintRepository([{
        "id": "bp-selected-faq",
        "slug": "selected-faq-report",
        "topic": "Saved FAQ questions customers keep asking",
        "topic_type": "content_ops_support_ticket_faq",
    }])
    llm = _RecordingContentLLM()
    landing_pages = MemoryDraftRepository("landing")
    blog_posts = MemoryDraftRepository("blog")
    scope = TenantScope(
        account_id="acct-selected-faq-real",
        user_id="user-selected-faq-real",
    )
    pool = {
        "drafts": {FAQ_DRAFT_ID: _saved_faq_draft()},
        "calls": [],
    }
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            landing_page=LandingPageGenerationService(
                landing_pages=landing_pages,
                llm=llm,
                skills=MemorySkillStore(),
            ),
            blog_post=BlogPostGenerationService(
                blueprints=blueprints,
                blog_posts=blog_posts,
                llm=llm,
                skills=MemorySkillStore(),
            ),
        ),
        scope_provider=lambda: scope,
        input_provider=build_content_ops_input_provider(
            pool_provider=lambda: pool,
            faq_repository_factory=_SelectedFAQRepository,
        ),
    )

    route = _route(router, "/content-ops/execute", "POST")
    payload = await route.endpoint({
        "outputs": ["landing_page", "blog_post"],
        "limit": 1,
        "require_quality_gates": False,
        "inputs": {"source_faq_ids": [FAQ_DRAFT_ID]},
    })

    assert pool["calls"] == [(FAQ_DRAFT_ID, "acct-selected-faq-real")]
    assert payload["status"] == "completed"
    assert payload["input_provider"]["provider"] == "atlas_support_ticket_request"
    assert payload["input_provider"]["metadata"]["source_row_count"] == 1
    assert payload["input_provider"]["metadata"]["included_row_count"] == 1
    assert payload["input_provider"]["warnings"] == []
    assert [step["output"] for step in payload["steps"]] == [
        "landing_page",
        "blog_post",
    ]
    assert [step["status"] for step in payload["steps"]] == [
        "completed",
        "completed",
    ]

    assert len(landing_pages.saved) == 1
    landing_scope, landing_draft = landing_pages.saved[0]
    assert landing_scope == scope
    assert landing_draft.campaign_name == "FAQ Report"
    assert landing_draft.reference_ids == ("ticket-acme-1", "ticket-northstar-1")

    assert blueprints.calls == [{
        "scope": scope,
        "target_mode": "vendor_retention",
        "limit": 1,
        "filters": {"topic_type": "content_ops_support_ticket_faq"},
    }]
    assert len(blog_posts.saved) == 1
    blog_scope, blog_draft = blog_posts.saved[0]
    assert blog_scope == scope
    assert blog_draft.data_context["source"] == "support_ticket_provider"
    assert blog_draft.data_context["included_ticket_row_count"] == 1
    assert blog_draft.data_context["faq_questions"] == ["Why was I charged twice?"]
    assert blog_draft.data_context["support_ticket_resolution_evidence_present"] is True
    assert blog_draft.data_context["support_ticket_resolution_evidence_count"] == 1

    expected_resolution = (
        "Open Billing, choose invoice history, and compare pending charges. "
        "Contact support with the invoice ID if both charges settled."
    )
    assert blog_draft.data_context["support_ticket_resolution_examples"] == [{
        "source_id": FAQ_DRAFT_ID,
        "source_title": "Why was I charged twice?",
        "text": expected_resolution,
    }]

    assert len(llm.calls) == 2
    landing_call, blog_call = llm.calls
    landing_system_prompt, landing_user_prompt = _message_texts(landing_call)
    assert landing_call["metadata"]["asset_type"] == "landing_page"
    assert "Why was I charged twice?" in landing_system_prompt
    assert expected_resolution in landing_system_prompt
    assert "Generate one landing page" in landing_user_prompt

    blog_system_prompt, blog_user_prompt = _message_texts(blog_call)
    assert blog_call["metadata"]["asset_type"] == "blog_post"
    assert "Saved FAQ questions customers keep asking" not in blog_system_prompt
    assert "Why was I charged twice?" in blog_user_prompt
    assert "support_ticket_resolution_evidence_present" in blog_user_prompt
    assert "Generate one blog post" in blog_user_prompt


async def test_support_ticket_provider_triggers_blog_generated_content_gate() -> None:
    blueprints = _RecordingBlueprintRepository([{
        "id": "bp-support-ticket-faq",
        "slug": "support-ticket-faq-report",
        "topic": "Support-ticket questions customers keep asking",
        "topic_type": "content_ops_support_ticket_faq",
    }])
    llm = _RecordingContentLLM(
        blog_content_suffix=(
            "\n\nThese FAQ answers can reduce repeat tickets by 30-45%."
        )
    )
    blog_posts = MemoryDraftRepository("blog")
    service = BlogPostGenerationService(
        blueprints=blueprints,
        blog_posts=blog_posts,
        llm=llm,
        skills=MemorySkillStore(),
        config=BlogPostGenerationConfig(
            quality_repair_attempts=0,
            quality_policy=QualityPolicy(
                name="blog_post",
                thresholds={"min_words": 20, "target_words": 20, "pass_score": 0},
            ),
        ),
    )
    scope = TenantScope(
        account_id="acct-support-ticket-real",
        user_id="user-support-ticket-real",
    )
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            blog_post=service,
        ),
        scope_provider=lambda: scope,
        input_provider=build_content_ops_input_provider(),
    )

    route = _route(router, "/content-ops/execute", "POST")
    payload = await route.endpoint({
        "outputs": ["blog_post"],
        "limit": 1,
        "require_quality_gates": True,
        "inputs": {"source_material": _support_ticket_csv_rows()},
    })

    assert payload["status"] == "completed"
    step = payload["steps"][0]
    assert step["output"] == "blog_post"
    assert step["status"] == "completed"
    assert step["result"]["generated"] == 0
    assert step["result"]["saved_ids"] == []
    assert blog_posts.saved == []
    assert any(
        "support_ticket_generated_content:" in blocker
        and "percentage claims not backed" in blocker
        for blocker in step["result"]["errors"][0]["blockers"]
    )


async def test_live_execute_route_accepts_faq_vocabulary_gap_inputs() -> None:
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            faq_markdown=TicketFAQMarkdownService(),
        ),
        scope_provider=lambda: TenantScope(account_id="acct-faq", user_id="user-faq"),
    )

    route = _route(router, "/content-ops/execute", "POST")
    payload = await route.endpoint({
        "target_mode": "vendor_retention",
        "outputs": ["faq_markdown"],
        "limit": 2,
        "require_quality_gates": False,
        "inputs": {
            "faq_title": "Hosted FAQ Vocabulary Gap Smoke",
            "faq_documentation_terms": ["Single sign-on setup"],
            "faq_vocabulary_gap_rules": [["SSO", "single sign-on"]],
            "source_material": {
                "support_tickets": [
                    {
                        "ticket_id": "ticket-sso-1",
                        "source_type": "support_ticket",
                        "subject": "SSO setup",
                        "message": "How do I enable SSO for my team?",
                        "pain_category": "authentication",
                    }
                ]
            },
        },
    })

    assert payload["status"] == "completed"
    assert payload["plan"]["steps"][0]["config"]["documentation_terms"] == [
        "Single sign-on setup"
    ]
    assert payload["plan"]["steps"][0]["config"]["vocabulary_gap_rules"] == [
        ["SSO", "single sign-on"]
    ]

    step = payload["steps"][0]
    assert step["output"] == "faq_markdown"
    assert step["status"] == "completed"

    result = step["result"]
    assert result["generated"] == 1
    assert result["ticket_source_count"] == 1
    assert all(result["output_checks"].values())
    assert "Hosted FAQ Vocabulary Gap Smoke" in result["markdown"]
    assert "`ticket-sso-1` - SSO setup" in result["markdown"]

    item = result["items"][0]
    assert item["source_ids"] == ("ticket-sso-1",)
    assert item["term_mappings"][0]["customer_term"] == "SSO"
    # Resolves to the faq_documentation_terms entry through the rule alias,
    # proving documentation_terms and vocabulary_gap_rules combine.
    assert (
        item["term_mappings"][0]["documentation_term"]
        == "Single sign-on setup"
    )


async def test_live_execute_route_returns_faq_deflection_report_artifact() -> None:
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            faq_deflection_report=FAQDeflectionReportService(),
        ),
        scope_provider=lambda: TenantScope(account_id="acct-faq", user_id="user-faq"),
    )

    route = _route(router, "/content-ops/execute", "POST")
    payload = await route.endpoint({
        "target_mode": "vendor_retention",
        "outputs": ["faq_deflection_report"],
        "limit": 3,
        "require_quality_gates": False,
        "inputs": {
            "deflection_report_title": "Hosted FAQ Deflection Report",
            "faq_title": "Hosted FAQ Source",
            "source_material": {
                "support_tickets": [
                    {
                        "ticket_id": "ticket-export-1",
                        "source_type": "support_ticket",
                        "subject": "Export report",
                        "message": "How do I export attribution reports?",
                        "pain_category": "exports",
                        "resolution_text": (
                            "Open Analytics, choose Attribution, then select "
                            "Download report."
                        ),
                    },
                    {
                        "ticket_id": "ticket-billing-1",
                        "source_type": "support_ticket",
                        "subject": "Renewal invoice",
                        "message": "How do I confirm my renewal invoice before payment?",
                        "pain_category": "billing",
                    },
                ]
            },
        },
    })

    assert payload["status"] == "completed"
    assert payload["plan"]["steps"][0]["config"]["report_title"] == (
        "Hosted FAQ Deflection Report"
    )

    step = payload["steps"][0]
    assert step["output"] == "faq_deflection_report"
    assert step["status"] == "completed"
    assert step["result"]["summary"]["source_count"] == 2
    assert step["result"]["summary"]["drafted_answer_count"] == 1
    assert step["result"]["summary"]["no_proven_answer_count"] == 1
    assert step["result"]["markdown"].startswith("# Hosted FAQ Deflection Report")
    assert "## Ranked Question Opportunities" in step["result"]["markdown"]
    assert "## No Proven Answer Yet" in step["result"]["markdown"]
    assert step["result"]["faq_result"]["markdown"].startswith("# Hosted FAQ Source")


async def test_live_execute_route_handles_bulk_faq_source_material() -> None:
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            faq_markdown=TicketFAQMarkdownService(),
        ),
        scope_provider=lambda: TenantScope(account_id="acct-faq", user_id="user-faq"),
    )
    source_material = [
        {
            "ticket_id": f"ticket-bulk-{index}",
            "source_type": "support_ticket",
            "subject": "SSO setup",
            "message": "How do I enable SSO for my team?",
            "pain_category": "authentication",
        }
        for index in range(1000)
    ]

    route = _route(router, "/content-ops/execute", "POST")
    payload = await route.endpoint({
        "target_mode": "vendor_retention",
        "outputs": ["faq_markdown"],
        "limit": 5,
        "require_quality_gates": False,
        "inputs": {
            "faq_title": "Hosted FAQ Bulk Smoke",
            "source_material": source_material,
        },
    })

    assert payload["status"] == "completed"
    assert payload["plan"]["steps"][0]["config"]["max_items"] == 5

    step = payload["steps"][0]
    assert step["output"] == "faq_markdown"
    assert step["status"] == "completed"

    result = step["result"]
    assert result["generated"] == 1
    assert result["source_count"] == 1000
    assert result["ticket_source_count"] == 1000
    assert all(result["output_checks"].values())
    assert "Hosted FAQ Bulk Smoke" in result["markdown"]
    assert "`ticket-bulk-0` - SSO setup" in result["markdown"]

    item = result["items"][0]
    assert item["frequency"] == 1000
    assert item["evidence_count"] == 3
    assert len(item["source_ids"]) == 1000
    assert item["source_ids"][0] == "ticket-bulk-0"
    assert item["source_ids"][-1] == "ticket-bulk-999"


async def test_live_execute_route_handles_bulk_faq_source_material_bundle() -> None:
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            faq_markdown=TicketFAQMarkdownService(),
        ),
        scope_provider=lambda: TenantScope(account_id="acct-faq", user_id="user-faq"),
    )
    support_tickets = [
        {
            "ticket_id": f"ticket-bundle-{index}",
            "source_type": "support_ticket",
            "subject": "Billing renewal question",
            "message": "How do I confirm my renewal invoice before payment?",
            "pain_category": "billing",
        }
        for index in range(1000)
    ]

    route = _route(router, "/content-ops/execute", "POST")
    payload = await route.endpoint({
        "target_mode": "vendor_retention",
        "outputs": ["faq_markdown"],
        "limit": 5,
        "require_quality_gates": False,
        "inputs": {
            "faq_title": "Hosted FAQ Bundle Bulk Smoke",
            "source_material": {"support_tickets": support_tickets},
        },
    })

    assert payload["status"] == "completed"
    assert payload["plan"]["steps"][0]["config"]["max_items"] == 5

    step = payload["steps"][0]
    assert step["output"] == "faq_markdown"
    assert step["status"] == "completed"

    result = step["result"]
    assert result["generated"] == 1
    assert result["source_count"] == 1000
    assert result["ticket_source_count"] == 1000
    assert all(result["output_checks"].values())
    assert "Hosted FAQ Bundle Bulk Smoke" in result["markdown"]
    assert "`ticket-bundle-0` - Billing renewal question" in result["markdown"]

    item = result["items"][0]
    assert item["frequency"] == 1000
    assert item["evidence_count"] == 3
    assert len(item["source_ids"]) == 1000
    assert item["source_ids"][0] == "ticket-bundle-0"
    assert item["source_ids"][-1] == "ticket-bundle-999"
