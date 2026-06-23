"""Atlas host mount tests for generated Content Ops asset routes.

The extracted router owns behavior-level tests. These tests only pin
the Atlas API aggregator wiring so operators can reach review/export
routes through the hosted API surface.
"""

from __future__ import annotations

from dataclasses import replace
import importlib
import inspect
from types import SimpleNamespace
import sys
from typing import Any

import pytest

from atlas_brain.auth.dependencies import AuthUser
from atlas_brain.auth.rate_limit import PLAN_RATE_LIMITS
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.landing_page_ports import (
    LandingPageDraft,
    LandingPageSection,
)

pytest.importorskip("asyncpg")
fastapi = pytest.importorskip("fastapi")
testclient_mod = pytest.importorskip("fastapi.testclient")
Depends = fastapi.Depends
FastAPI = fastapi.FastAPI
Request = fastapi.Request
TestClient = testclient_mod.TestClient


def _fresh_api_package():
    original = sys.modules.pop("atlas_brain.api", None)
    try:
        return importlib.import_module("atlas_brain.api")
    finally:
        if original is not None:
            sys.modules["atlas_brain.api"] = original


def _route(api_pkg, path: str):
    route = next((route for route in api_pkg.router.routes if route.path == path), None)
    assert route is not None, f"Route {path!r} not mounted"
    return route


def _router_route(router, path: str, method: str):
    route = next(
        (
            route
            for route in router.routes
            if route.path == path and method.upper() in getattr(route, "methods", ())
        ),
        None,
    )
    assert route is not None, f"Route {method.upper()} {path!r} not mounted"
    return route


def _content_ops_user(
    account_id: str = "account-1",
    *,
    plan: str = "b2b_growth",
) -> AuthUser:
    return AuthUser(
        user_id=f"user-{account_id}",
        account_id=account_id,
        plan=plan,
        plan_status="active",
        role="owner",
        product="b2b_retention",
        auth_method="api_key",
        api_key_scopes=("content_ops:deflection:*",),
    )


def _public_ready_landing_page(landing_page_id: str) -> LandingPageDraft:
    return LandingPageDraft(
        id=landing_page_id,
        status="draft",
        campaign_name="support-faq-report",
        persona="10-50 person SaaS team",
        value_prop="Turn repeat support tickets into customer-ready FAQs",
        title="Support FAQ Report for Small SaaS Teams",
        slug="support-faq-report",
        hero={
            "headline": "Turn repeat tickets into answers",
            "subheadline": (
                "10-50 person SaaS teams turn repeat support tickets into "
                "customer-ready FAQs before customers wait again."
            ),
            "cta_label": "Upload Ticket CSV -- Free Analysis",
            "cta_url": "/systems/ai-content-ops/intake",
        },
        sections=(
            LandingPageSection(
                id="repeat_support_problem",
                title="Repeat support questions become customer frustration",
                body_markdown=(
                    "10-50 person SaaS teams lose time when customers ask "
                    "the same support questions again and again. Repeat "
                    "questions create friction because the answer is not "
                    "where customers are looking."
                ),
                metadata={
                    "kind": "problem",
                    "primary_question": "Why do repeat support questions matter?",
                    "answer_summary": (
                        "10-50 person SaaS teams lose time when customers ask "
                        "the same support questions again and again."
                    ),
                },
            ),
            LandingPageSection(
                id="faq_report_solution",
                title="A FAQ Report turns tickets into findable answers",
                body_markdown=(
                    "10-50 person SaaS teams use the FAQ Report workflow to "
                    "turn old support tickets into clear answers customers "
                    "can find before they email support."
                ),
                metadata={
                    "kind": "solution",
                    "primary_question": "How does the FAQ Report help?",
                    "answer_summary": (
                        "10-50 person SaaS teams use the FAQ Report workflow "
                        "to turn old support tickets into clear answers."
                    ),
                },
            ),
            LandingPageSection(
                id="before_upload_questions",
                title="Questions before uploading tickets",
                body_markdown=(
                    "10-50 person SaaS teams can review privacy, publishing, "
                    "and setup questions before uploading tickets. That keeps "
                    "the process clear without giving up help-center control."
                ),
                metadata={
                    "kind": "objection",
                    "primary_question": "What should teams know before upload?",
                    "answer_summary": (
                        "10-50 person SaaS teams can review privacy, "
                        "publishing, and setup questions before uploading tickets."
                    ),
                },
            ),
        ),
        cta={
            "label": "Upload Ticket CSV -- Free Analysis",
            "url": "/systems/ai-content-ops/intake",
            "variant": "primary",
        },
        meta={
            "title_tag": "Support FAQ Report for Small SaaS Teams",
            "description": (
                "Turn repeat support tickets into customer-ready FAQ answers "
                "small SaaS teams can publish before customers wait again."
            ),
        },
        reference_ids=("support-ticket-sample",),
    )


class _MemoryLandingPageRepository:
    drafts: dict[tuple[str | None, str], LandingPageDraft] = {}
    status_calls: list[dict[str, Any]] = []

    def __init__(self, pool: Any) -> None:
        self.pool = pool

    async def update_status(
        self,
        draft_id: str,
        status: str,
        *,
        scope: TenantScope,
    ) -> bool:
        self.status_calls.append({
            "account_id": scope.account_id,
            "draft_id": draft_id,
            "status": status,
        })
        key = (scope.account_id, draft_id)
        draft = self.drafts.get(key)
        if draft is None:
            return False
        self.drafts[key] = replace(draft, status=status)
        return True

    async def get_public_approved_draft(
        self,
        landing_page_id: str,
    ) -> LandingPageDraft | None:
        for draft in self.drafts.values():
            if draft.id == landing_page_id and draft.status == "approved":
                return draft
        return None


class _CampaignReviewResult:
    def __init__(self, rows: tuple[dict[str, Any], ...]) -> None:
        self.rows = rows


def test_api_aggregator_mounts_generated_asset_routes() -> None:
    api_pkg = _fresh_api_package()

    paths = {getattr(route, "path", "") for route in api_pkg.router.routes}

    assert "/content-assets/{asset}/drafts" in paths
    assert "/content-assets/{asset}/drafts/export" in paths
    assert "/content-assets/{asset}/drafts/review" in paths
    assert "/content-assets/landing_page/public/sitemap.xml" in paths
    assert "/content-assets/landing_page/public/{landing_page_id}" in paths


@pytest.mark.asyncio
async def test_email_campaign_batch_review_delegates_to_campaign_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from extracted_content_pipeline.api import generated_assets

    campaign_ids = (
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    )
    calls: list[dict[str, Any]] = []

    async def _review_campaign_drafts(pool: Any, **kwargs: Any) -> _CampaignReviewResult:
        calls.append({"pool": pool, **kwargs})
        return _CampaignReviewResult((
            {"id": campaign_ids[0]},
            {"id": campaign_ids[1]},
        ))

    monkeypatch.setattr(
        generated_assets,
        "review_campaign_drafts",
        _review_campaign_drafts,
    )

    pool = object()
    scope = TenantScope(account_id="acct-campaign-review")
    updated = await generated_assets._update_asset_statuses(
        "email_campaign",
        pool,
        asset_ids=campaign_ids,
        status="approved",
        scope=scope,
    )

    assert updated == list(campaign_ids)
    assert calls == [{
        "pool": pool,
        "campaign_ids": campaign_ids,
        "status": "approved",
        "scope": scope,
    }]


@pytest.mark.asyncio
async def test_email_campaign_batch_review_empty_ids_does_not_delegate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from extracted_content_pipeline.api import generated_assets

    calls: list[dict[str, Any]] = []

    async def _review_campaign_drafts(pool: Any, **kwargs: Any) -> _CampaignReviewResult:
        calls.append({"pool": pool, **kwargs})
        return _CampaignReviewResult(())

    monkeypatch.setattr(
        generated_assets,
        "review_campaign_drafts",
        _review_campaign_drafts,
    )

    updated = await generated_assets._update_asset_statuses(
        "email_campaign",
        object(),
        asset_ids=(),
        status="approved",
        scope=TenantScope(account_id="acct-campaign-review"),
    )

    assert updated == ()
    assert calls == []


@pytest.mark.asyncio
async def test_email_campaign_review_rejects_unsupported_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from extracted_content_pipeline.api import generated_assets

    calls: list[dict[str, Any]] = []

    async def _review_campaign_drafts(pool: Any, **kwargs: Any) -> _CampaignReviewResult:
        calls.append({"pool": pool, **kwargs})
        return _CampaignReviewResult(())

    monkeypatch.setattr(
        generated_assets,
        "review_campaign_drafts",
        _review_campaign_drafts,
    )

    with pytest.raises(generated_assets.HTTPException) as exc_info:
        await generated_assets._update_asset_statuses(
            "email_campaign",
            object(),
            asset_ids=("11111111-1111-4111-8111-111111111111",),
            status="rejected",
            scope=TenantScope(account_id="acct-campaign-review"),
        )

    assert exc_info.value.status_code == 400
    assert "email_campaign review status" in exc_info.value.detail
    assert calls == []


def test_generated_asset_routes_use_shared_content_ops_auth_scope_and_pool() -> None:
    api_pkg = _fresh_api_package()
    route = _route(api_pkg, "/content-assets/{asset}/drafts")
    closure = dict(
        zip(
            route.endpoint.__code__.co_freevars,
            (cell.cell_contents for cell in route.endpoint.__closure__ or ()),
        )
    )
    dependency_names = [
        getattr(dependency.call, "__name__", "")
        for dependency in route.dependant.dependencies
    ]

    assert closure["pool_provider"].__name__ == "get_db_pool"
    assert closure["scope_provider"].__name__ == "build_content_ops_scope"
    assert "_capture_content_ops_auth_user" in dependency_names


def test_generated_asset_publish_macros_route_uses_host_provider_wiring() -> None:
    api_pkg = _fresh_api_package()
    route = _route(
        api_pkg,
        "/content-assets/{asset}/drafts/{draft_id}/publish-macros",
    )
    closure = dict(
        zip(
            route.endpoint.__code__.co_freevars,
            (cell.cell_contents for cell in route.endpoint.__closure__ or ()),
        )
    )
    provider_factory = closure["macro_publish_provider"]

    provider = provider_factory()

    assert provider_factory.__name__ == "<lambda>"
    assert inspect.iscoroutine(provider)
    assert provider.cr_code.co_name == "build_content_ops_macro_publish_provider"
    assert provider.cr_frame is not None
    provider.close()


def test_faq_deflection_search_route_uses_shared_content_ops_auth_scope_and_pool() -> None:
    api_pkg = _fresh_api_package()
    route = _route(api_pkg, "/content-ops/faq-deflection-search")
    closure = dict(
        zip(
            route.endpoint.__code__.co_freevars,
            (cell.cell_contents for cell in route.endpoint.__closure__ or ()),
        )
    )
    dependency_names = [
        getattr(dependency.call, "__name__", "")
        for dependency in route.dependant.dependencies
    ]

    assert closure["pool_provider"].__name__ == "get_db_pool"
    assert closure["scope_provider"].__name__ == "build_content_ops_scope"
    assert "_capture_content_ops_auth_user" in dependency_names


def test_content_ops_zendesk_credentials_route_uses_shared_auth_and_pool() -> None:
    api_pkg = _fresh_api_package()
    route = _route(api_pkg, "/content-ops/zendesk-credentials")
    closure = dict(
        zip(
            route.endpoint.__code__.co_freevars,
            (cell.cell_contents for cell in route.endpoint.__closure__ or ()),
        )
    )
    dependency_names = [
        getattr(dependency.call, "__name__", "")
        for dependency in route.dependant.dependencies
    ]

    assert closure["pool_provider"].__name__ == "get_db_pool"
    assert "_capture_content_ops_auth_user" in dependency_names


def test_content_ops_zendesk_export_route_uses_shared_auth_and_pool() -> None:
    api_pkg = _fresh_api_package()
    route = _route(api_pkg, "/content-ops/zendesk-export/full-thread")
    closure = dict(
        zip(
            route.endpoint.__code__.co_freevars,
            (cell.cell_contents for cell in route.endpoint.__closure__ or ()),
        )
    )
    dependency_names = [
        getattr(dependency.call, "__name__", "")
        for dependency in route.dependant.dependencies
    ]

    assert closure["pool_provider"].__name__ == "get_db_pool"
    assert "_capture_content_ops_auth_user" in dependency_names


def test_content_ops_brand_voice_profiles_route_uses_shared_auth_and_pool() -> None:
    api_pkg = _fresh_api_package()
    route = _route(api_pkg, "/content-ops/brand-voice-profiles")
    closure = dict(
        zip(
            route.endpoint.__code__.co_freevars,
            (cell.cell_contents for cell in route.endpoint.__closure__ or ()),
        )
    )
    dependency_names = [
        getattr(dependency.call, "__name__", "")
        for dependency in route.dependant.dependencies
    ]

    assert closure["pool_provider"].__name__ == "get_db_pool"
    assert "_capture_content_ops_auth_user" in dependency_names


def test_content_ops_preview_route_uses_host_brand_voice_profile_provider() -> None:
    api_pkg = _fresh_api_package()
    route = _route(api_pkg, "/content-ops/preview")
    closure = dict(
        zip(
            route.endpoint.__code__.co_freevars,
            (cell.cell_contents for cell in route.endpoint.__closure__ or ()),
        )
    )

    assert closure["brand_voice_profile_provider"].__name__ == "<lambda>"


def test_content_ops_usage_summary_route_uses_shared_auth_and_pool() -> None:
    api_pkg = _fresh_api_package()
    route = _route(api_pkg, "/content-ops/usage/summary")
    closure = dict(
        zip(
            route.endpoint.__code__.co_freevars,
            (cell.cell_contents for cell in route.endpoint.__closure__ or ()),
        )
    )
    dependency_names = [
        getattr(dependency.call, "__name__", "")
        for dependency in route.dependant.dependencies
    ]

    assert closure["usage_pool_provider"].__name__ == "get_db_pool"
    assert "_capture_content_ops_auth_user" in dependency_names
    assert "_require_content_ops_usage_operator" in dependency_names


def test_content_ops_deflection_paid_route_uses_operator_gate() -> None:
    api_pkg = _fresh_api_package()
    route = _route(api_pkg, "/content-ops/deflection-reports/{request_id}/paid")
    dependency_names = [
        getattr(dependency.call, "__name__", "")
        for dependency in route.dependant.dependencies
    ]

    assert "_capture_content_ops_auth_user" in dependency_names
    assert "_require_content_ops_usage_operator" in dependency_names


def test_content_ops_public_deflection_routes_use_rate_limit_gate() -> None:
    api_pkg = _fresh_api_package()
    public_routes = [
        ("/content-ops/deflection-reports/submit", "POST"),
        ("/content-ops/deflection-reports/pricing/standard", "GET"),
        ("/content-ops/deflection-reports/{request_id}/snapshot", "GET"),
        ("/content-ops/deflection-reports/{request_id}/artifact", "GET"),
        ("/content-ops/deflection-reports/{request_id}/report-model", "GET"),
        (
            "/content-ops/deflection-reports/{request_id}/checkout-authorization",
            "POST",
        ),
    ]

    for path, method in public_routes:
        route = _router_route(api_pkg.router, path, method)
        dependency_names = [
            getattr(dependency.call, "__name__", "")
            for dependency in route.dependant.dependencies
        ]
        assert "_capture_content_ops_auth_user" in dependency_names
        assert "_rate_limit_public_deflection_report" in dependency_names
        assert "_require_content_ops_usage_operator" not in dependency_names

    paid_route = _router_route(
        api_pkg.router,
        "/content-ops/deflection-reports/{request_id}/paid",
        "POST",
    )
    paid_dependency_names = [
        getattr(dependency.call, "__name__", "")
        for dependency in paid_route.dependant.dependencies
    ]
    assert "_rate_limit_public_deflection_report" not in paid_dependency_names
    assert "_require_content_ops_usage_operator" in paid_dependency_names


@pytest.mark.asyncio
async def test_content_ops_auth_bridge_sets_rate_limit_identity() -> None:
    api_pkg = _fresh_api_package()
    request = SimpleNamespace(state=SimpleNamespace())
    user = _content_ops_user("account-rl", plan="b2b_growth")

    assert (
        await api_pkg._capture_content_ops_auth_user(request=request, user=user)
    ) is user

    assert request.state.rate_limit_key == "account-rl"
    assert request.state.rate_limit_plan == "b2b_growth"


def test_public_deflection_rate_limit_shares_scope_and_keys_by_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded

    api_pkg = _fresh_api_package()
    app = FastAPI()
    app.state.limiter = api_pkg.limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    async def _fake_content_ops_auth_user(request: Request) -> AuthUser:
        account_id = request.headers.get("x-test-account", "account-a")
        plan = request.headers.get("x-test-plan", "b2b_growth")
        user = _content_ops_user(account_id, plan=plan)
        request.state.rate_limit_key = user.account_id
        request.state.rate_limit_plan = user.plan
        return user

    app.dependency_overrides[api_pkg._capture_content_ops_auth_user] = (
        _fake_content_ops_auth_user
    )

    @app.get(
        "/snapshot/{request_id}",
        dependencies=[Depends(api_pkg._rate_limit_public_deflection_report)],
    )
    async def _snapshot(request_id: str) -> dict[str, str]:
        return {"request_id": request_id}

    @app.get(
        "/artifact/{request_id}",
        dependencies=[Depends(api_pkg._rate_limit_public_deflection_report)],
    )
    async def _artifact(request_id: str) -> dict[str, str]:
        return {"request_id": request_id}

    limit_name = (
        f"{api_pkg._rate_limit_public_deflection_report.__module__}."
        f"{api_pkg._rate_limit_public_deflection_report.__name__}"
    )
    api_pkg.limiter._dynamic_route_limits[limit_name] = (
        api_pkg.limiter._dynamic_route_limits[limit_name][-1:]
    )
    monkeypatch.setitem(PLAN_RATE_LIMITS, "b2b_growth", "2/hour")
    api_pkg.limiter._limiter.storage.reset()
    try:
        client = TestClient(app)
        headers_a = {"x-test-account": "account-a"}
        headers_b = {"x-test-account": "account-b"}

        assert client.get("/snapshot/aaa", headers=headers_a).status_code == 200
        assert client.get("/artifact/bbb", headers=headers_a).status_code == 200
        assert client.get("/snapshot/ccc", headers=headers_a).status_code == 429
        assert client.get("/snapshot/ccc", headers=headers_b).status_code == 200
    finally:
        api_pkg.limiter._limiter.storage.reset()


def test_content_ops_preview_route_uses_host_cache_policy_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atlas_brain.config import settings

    monkeypatch.setattr(
        settings.b2b_campaign,
        "content_ops_cache_policy_default",
        "exact-cache",
    )
    api_pkg = _fresh_api_package()
    route = _route(api_pkg, "/content-ops/preview")
    closure = dict(
        zip(
            route.endpoint.__code__.co_freevars,
            (cell.cell_contents for cell in route.endpoint.__closure__ or ()),
        )
    )
    provider = closure["cache_policy_default_provider"]

    assert provider.__name__ == "_content_ops_cache_policy_default"
    assert provider(TenantScope(account_id="account-1")) == "exact-cache"


def test_content_ops_cache_policy_default_provider_keeps_blank_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atlas_brain.config import settings

    monkeypatch.setattr(settings.b2b_campaign, "content_ops_cache_policy_default", " ")
    api_pkg = _fresh_api_package()
    route = _route(api_pkg, "/content-ops/preview")
    closure = dict(
        zip(
            route.endpoint.__code__.co_freevars,
            (cell.cell_contents for cell in route.endpoint.__closure__ or ()),
        )
    )

    assert closure["cache_policy_default_provider"](TenantScope()) is None


def test_content_ops_tenant_usage_summary_route_uses_shared_auth_scope_and_pool() -> None:
    api_pkg = _fresh_api_package()
    route = _route(api_pkg, "/content-ops/usage/summary/tenant")
    closure = dict(
        zip(
            route.endpoint.__code__.co_freevars,
            (cell.cell_contents for cell in route.endpoint.__closure__ or ()),
        )
    )
    dependency_names = [
        getattr(dependency.call, "__name__", "")
        for dependency in route.dependant.dependencies
    ]

    assert closure["usage_pool_provider"].__name__ == "get_db_pool"
    assert closure["scope_provider"].__name__ == "build_content_ops_scope"
    assert "_capture_content_ops_auth_user" in dependency_names
    assert "_require_content_ops_usage_operator" not in dependency_names


@pytest.mark.asyncio
async def test_content_ops_usage_summary_operator_gate_rejects_account_admin() -> None:
    api_pkg = _fresh_api_package()
    user = AuthUser(
        user_id="user-1",
        account_id="account-1",
        plan="b2b_growth",
        plan_status="active",
        role="owner",
        product="b2b_retention",
        is_admin=True,
        is_platform_admin=False,
    )

    with pytest.raises(api_pkg.HTTPException) as exc:
        await api_pkg._require_content_ops_usage_operator(user=user)

    assert exc.value.status_code == 403
    assert exc.value.detail == "Platform admin access required"


@pytest.mark.asyncio
async def test_content_ops_usage_summary_operator_gate_allows_platform_admin() -> None:
    api_pkg = _fresh_api_package()
    user = AuthUser(
        user_id="user-1",
        account_id="account-1",
        plan="b2b_growth",
        plan_status="active",
        role="member",
        product="b2b_retention",
        is_admin=True,
        is_platform_admin=True,
    )

    assert await api_pkg._require_content_ops_usage_operator(user=user) is user


def test_public_landing_page_route_uses_pool_without_content_ops_auth_dependency() -> None:
    api_pkg = _fresh_api_package()
    route = _route(api_pkg, "/content-assets/landing_page/public/{landing_page_id}")
    closure = dict(
        zip(
            route.endpoint.__code__.co_freevars,
            (cell.cell_contents for cell in route.endpoint.__closure__ or ()),
        )
    )
    dependency_names = [
        getattr(dependency.call, "__name__", "")
        for dependency in route.dependant.dependencies
    ]

    assert closure["pool_provider"].__name__ == "get_db_pool"
    assert "_capture_content_ops_auth_user" not in dependency_names


@pytest.mark.asyncio
async def test_landing_page_review_approval_controls_public_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from extracted_content_pipeline.api import generated_assets

    landing_page_id = "11111111-1111-4111-8111-111111111111"
    pool = object()
    _MemoryLandingPageRepository.drafts = {
        ("acct-public-assets", landing_page_id): _public_ready_landing_page(
            landing_page_id
        )
    }
    _MemoryLandingPageRepository.status_calls = []
    monkeypatch.setattr(
        generated_assets,
        "PostgresLandingPageRepository",
        _MemoryLandingPageRepository,
    )

    public_router = generated_assets.create_public_landing_page_router(
        pool_provider=lambda: pool
    )
    review_router = generated_assets.create_generated_asset_router(
        pool_provider=lambda: pool,
        scope_provider=lambda: TenantScope(account_id="acct-public-assets"),
    )
    foreign_review_router = generated_assets.create_generated_asset_router(
        pool_provider=lambda: pool,
        scope_provider=lambda: TenantScope(account_id="acct-other"),
    )
    public_route = _router_route(
        public_router,
        "/content-assets/landing_page/public/{landing_page_id}",
        "GET",
    )
    review_route = _router_route(
        review_router,
        "/content-assets/{asset}/drafts/review",
        "POST",
    )
    foreign_review_route = _router_route(
        foreign_review_router,
        "/content-assets/{asset}/drafts/review",
        "POST",
    )

    with pytest.raises(generated_assets.HTTPException) as hidden_before_review:
        await public_route.endpoint(landing_page_id)
    assert hidden_before_review.value.status_code == 404

    foreign_response = await foreign_review_route.endpoint(
        "landing_page",
        {"id": landing_page_id, "status": "approved"},
    )

    assert foreign_response["updated"] is False
    with pytest.raises(generated_assets.HTTPException) as hidden_after_foreign_review:
        await public_route.endpoint(landing_page_id)
    assert hidden_after_foreign_review.value.status_code == 404

    approved_response = await review_route.endpoint(
        "landing_page",
        {"id": landing_page_id, "status": "approved"},
    )
    public_response = await public_route.endpoint(landing_page_id)

    assert approved_response == {
        "account_id": "acct-public-assets",
        "asset": "landing_page",
        "id": landing_page_id,
        "status": "approved",
        "updated": True,
    }
    assert public_response["id"] == landing_page_id
    assert public_response["slug"] == "support-faq-report"
    assert public_response["robots"] == "index,follow"

    rejected_response = await review_route.endpoint(
        "landing_page",
        {"asset_id": landing_page_id, "status": "rejected"},
    )

    assert rejected_response["updated"] is True
    assert rejected_response["status"] == "rejected"
    with pytest.raises(generated_assets.HTTPException) as hidden_after_reject:
        await public_route.endpoint(landing_page_id)
    assert hidden_after_reject.value.status_code == 404
    assert _MemoryLandingPageRepository.status_calls == [
        {
            "account_id": "acct-other",
            "draft_id": landing_page_id,
            "status": "approved",
        },
        {
            "account_id": "acct-public-assets",
            "draft_id": landing_page_id,
            "status": "approved",
        },
        {
            "account_id": "acct-public-assets",
            "draft_id": landing_page_id,
            "status": "rejected",
        },
    ]
