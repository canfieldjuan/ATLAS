from __future__ import annotations

import pytest

import extracted_content_pipeline.api.control_surfaces as api_module
from extracted_content_pipeline.api.control_surfaces import (
    ContentOpsControlSurfaceApiConfig,
    create_content_ops_control_surface_router,
)
from extracted_content_pipeline.content_ops_execution import (
    execute_content_ops_from_mapping,
)
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
