from __future__ import annotations

import pytest

import extracted_content_pipeline.api.control_surfaces as api_module
from extracted_content_pipeline.api.control_surfaces import (
    ContentOpsControlSurfaceApiConfig,
    create_content_ops_control_surface_router,
)
from extracted_content_pipeline.content_ops_execution import (
    ContentOpsExecutionServices,
    execute_content_ops_from_mapping,
)
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.ticket_faq_markdown import TicketFAQMarkdownService
from tests.content_ops_live_execute_harness import (
    build_content_ops_live_execute_harness,
    default_content_ops_execute_payload,
)


pytestmark = pytest.mark.skipif(
    api_module.APIRouter is None,
    reason="fastapi is not installed in this test environment",
)


def _route(router, path: str, method: str):
    for route in router.routes:
        methods = getattr(route, "methods", set())
        if getattr(route, "path", None) == path and method.upper() in methods:
            return route
    raise AssertionError(f"route not found: {method} {path}")


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
