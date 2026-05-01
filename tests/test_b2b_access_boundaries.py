"""Access-boundary tests for high-risk B2B/operator routes.

These tests mount the aggregate API router under /api/v1, matching main.py,
so they catch both route-registration drift and endpoint-level auth drift.

The endpoints covered here are not public sales-flow routes. They mutate or
expose tenant/operator state and should reject unauthenticated requests before
touching the database or downstream services.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from atlas_brain.api import router as api_router
from atlas_brain.api import admin_costs
from atlas_brain.api import b2b_dashboard
from atlas_brain.api import b2b_campaigns
from atlas_brain.api import b2b_crm_events
from atlas_brain.api import b2b_tenant_dashboard
from atlas_brain.api import b2b_vendor_briefing
from atlas_brain.api import seller_campaigns
from atlas_brain.api import vendor_targets
from atlas_brain.auth.dependencies import AuthUser, require_auth
from atlas_brain.config import settings


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(api_router, prefix="/api/v1")
    return app


_TARGET_ID = str(uuid4())


@pytest.mark.parametrize(
    ("method", "path", "json_body"),
    [
        ("GET", "/api/v1/b2b/vendor-targets", None),
        (
            "POST",
            "/api/v1/b2b/vendor-targets",
            {
                "company_name": "ClickUp",
                "target_mode": "vendor_retention",
                "competitors_tracked": ["Monday.com"],
            },
        ),
        ("GET", f"/api/v1/b2b/vendor-targets/{_TARGET_ID}", None),
        ("PUT", f"/api/v1/b2b/vendor-targets/{_TARGET_ID}", {"notes": "boundary test"}),
        ("DELETE", f"/api/v1/b2b/vendor-targets/{_TARGET_ID}", None),
        ("POST", f"/api/v1/b2b/vendor-targets/{_TARGET_ID}/generate-report", None),
        ("POST", f"/api/v1/b2b/vendor-targets/{_TARGET_ID}/claim", None),
        ("POST", "/api/v1/b2b/briefings/preview", {"vendor_name": "ClickUp"}),
        ("POST", "/api/v1/b2b/briefings/generate", {"vendor_name": "ClickUp"}),
        ("POST", "/api/v1/b2b/briefings/send-batch", None),
        ("GET", "/api/v1/b2b/briefings", None),
        ("GET", "/api/v1/b2b/briefings/review-queue/summary", None),
        ("GET", "/api/v1/b2b/briefings/review-queue", None),
        ("POST", "/api/v1/b2b/briefings/bulk-approve", {"briefing_ids": [str(uuid4())]}),
        ("POST", "/api/v1/b2b/briefings/bulk-reject", {"briefing_ids": [str(uuid4())]}),
        ("GET", "/api/v1/b2b/briefings/export", None),
        ("GET", "/api/v1/admin/costs/summary", None),
        (
            "POST",
            "/api/v1/b2b/crm/events",
            {
                "crm_provider": "generic",
                "event_type": "deal_won",
                "company_name": "Boundary Corp",
            },
        ),
        (
            "POST",
            "/api/v1/b2b/crm/events/batch",
            {
                "events": [
                    {
                        "crm_provider": "generic",
                        "event_type": "deal_won",
                        "company_name": "Boundary Corp",
                    }
                ]
            },
        ),
        ("GET", "/api/v1/b2b/crm/events", None),
        ("GET", "/api/v1/b2b/crm/events/enrichment-stats", None),
        (
            "POST",
            f"/api/v1/b2b/tenant/company-signal-candidates/{_TARGET_ID}/approve",
            {"notes": "boundary"},
        ),
        (
            "POST",
            f"/api/v1/b2b/tenant/company-signal-candidates/{_TARGET_ID}/suppress",
            {"notes": "boundary"},
        ),
        (
            "POST",
            f"/api/v1/b2b/tenant/company-signal-candidate-groups/{_TARGET_ID}/approve",
            {"notes": "boundary"},
        ),
        (
            "POST",
            f"/api/v1/b2b/tenant/company-signal-candidate-groups/{_TARGET_ID}/suppress",
            {"notes": "boundary"},
        ),
        (
            "POST",
            "/api/v1/b2b/tenant/company-signal-candidate-groups/approve",
            {"group_ids": [str(uuid4())], "notes": "boundary"},
        ),
        (
            "POST",
            "/api/v1/b2b/tenant/company-signal-candidate-groups/suppress",
            {"group_ids": [str(uuid4())], "notes": "boundary"},
        ),
        ("GET", "/api/v1/b2b/tenant/company-signals", None),
        ("GET", "/api/v1/b2b/tenant/company-signal-candidates", None),
        ("GET", "/api/v1/b2b/tenant/company-signal-candidate-groups", None),
        ("GET", "/api/v1/b2b/tenant/company-signal-candidate-group-summary", None),
        ("GET", "/api/v1/b2b/tenant/company-signal-review-impact-summary", None),
        ("GET", "/api/v1/b2b/tenant/accounts-in-motion?vendor_name=ClickUp", None),
        ("GET", "/api/v1/b2b/tenant/accounts-in-motion/live?vendor_name=ClickUp", None),
        (
            "POST",
            "/api/v1/b2b/tenant/corrections",
            {
                "entity_type": "vendor",
                "entity_id": str(uuid4()),
                "correction_type": "merge_vendor",
                "old_value": "Salesforce",
                "new_value": "HubSpot",
                "reason": "boundary",
            },
        ),
        ("GET", "/api/v1/b2b/tenant/corrections", None),
        ("GET", "/api/v1/b2b/tenant/corrections/stats", None),
        ("GET", f"/api/v1/b2b/tenant/corrections/{_TARGET_ID}", None),
        (
            "POST",
            f"/api/v1/b2b/tenant/corrections/{_TARGET_ID}/revert",
            {"reason": "boundary"},
        ),
        ("GET", "/api/v1/b2b/tenant/source-corrections/impact", None),
        ("POST", "/api/v1/b2b/tenant/vendors/ClickUp/reason", None),
        (
            "POST",
            "/api/v1/b2b/tenant/vendors/compare-reasoning",
            {"vendors": ["ClickUp", "Monday.com"]},
        ),
        ("GET", "/api/v1/b2b/tenant/vendors", None),
        ("POST", "/api/v1/b2b/tenant/vendors", {"vendor_name": "ClickUp", "track_mode": "own"}),
        ("DELETE", "/api/v1/b2b/tenant/vendors/ClickUp", None),
        ("GET", "/api/v1/b2b/tenant/vendors/search?q=ClickUp", None),
        ("POST", "/api/v1/b2b/tenant/vendors/ClickUp/refresh", {}),
        ("GET", "/api/v1/b2b/tenant/export/signals", None),
        ("GET", "/api/v1/b2b/tenant/export/reviews", None),
        ("GET", "/api/v1/b2b/tenant/export/high-intent", None),
        ("GET", "/api/v1/b2b/tenant/export/source-health", None),
        ("GET", "/api/v1/b2b/tenant/overview", None),
        ("GET", "/api/v1/b2b/tenant/signals", None),
        ("GET", "/api/v1/b2b/tenant/signals/ClickUp", None),
        ("GET", "/api/v1/b2b/tenant/slow-burn-watchlist", None),
        ("GET", "/api/v1/b2b/tenant/accounts-in-motion-feed", None),
        ("GET", "/api/v1/b2b/tenant/vendor-history?vendor_name=ClickUp", None),
        ("GET", "/api/v1/b2b/tenant/compare-vendor-periods?vendor_name=ClickUp", None),
        ("GET", "/api/v1/b2b/tenant/pain-trends", None),
        ("GET", "/api/v1/b2b/tenant/displacement", None),
        ("GET", "/api/v1/b2b/tenant/pipeline", None),
        ("GET", "/api/v1/b2b/tenant/leads", None),
        ("GET", "/api/v1/b2b/tenant/high-intent", None),
        ("GET", "/api/v1/b2b/tenant/leads/Acme", None),
        ("GET", "/api/v1/b2b/tenant/reviews", None),
        ("GET", f"/api/v1/b2b/tenant/reviews/{_TARGET_ID}", None),
        (
            "POST",
            "/api/v1/b2b/tenant/push-to-crm",
            {"opportunities": [{"company": "Acme", "vendor": "ClickUp", "urgency": 8}]},
        ),
        ("GET", "/api/v1/b2b/tenant/opportunity-dispositions", None),
        (
            "POST",
            "/api/v1/b2b/tenant/opportunity-dispositions",
            {
                "opportunity_key": "Acme::ClickUp",
                "company": "Acme",
                "vendor": "ClickUp",
                "disposition": "saved",
            },
        ),
        (
            "POST",
            "/api/v1/b2b/tenant/opportunity-dispositions/bulk",
            {
                "items": [{"opportunity_key": "Acme::ClickUp", "company": "Acme", "vendor": "ClickUp"}],
                "disposition": "saved",
            },
        ),
        (
            "POST",
            "/api/v1/b2b/tenant/opportunity-dispositions/remove",
            {"opportunity_keys": ["Acme::ClickUp"]},
        ),
        ("POST", "/api/v1/b2b/tenant/calibration/trigger", None),
        ("GET", "/api/v1/b2b/tenant/reports", None),
        ("GET", f"/api/v1/b2b/tenant/reports/{_TARGET_ID}", None),
        (
            "POST",
            "/api/v1/b2b/tenant/reports/compare",
            {"primary_vendor": "ClickUp", "comparison_vendor": "Monday.com"},
        ),
        (
            "POST",
            "/api/v1/b2b/tenant/reports/compare-companies",
            {"primary_company": "Acme", "comparison_company": "Globex"},
        ),
        ("POST", "/api/v1/b2b/tenant/reports/company-deep-dive", {"company_name": "Acme"}),
        ("POST", "/api/v1/b2b/tenant/reports/battle-card", {"vendor_name": "ClickUp"}),
        ("GET", "/api/v1/b2b/tenant/report-subscriptions/library/default", None),
        (
            "PUT",
            "/api/v1/b2b/tenant/report-subscriptions/library/default",
            {"scope_label": "Boundary Library", "enabled": False},
        ),
        (
            "POST",
            "/api/v1/b2b/tenant/webhooks",
            {
                "url": "https://hooks.example.com/boundary",
                "secret": "boundary-secret-value",
                "event_types": ["signal_update"],
            },
        ),
        ("GET", "/api/v1/b2b/tenant/webhooks", None),
        ("GET", "/api/v1/b2b/tenant/webhooks/delivery-summary", None),
        ("GET", f"/api/v1/b2b/tenant/webhooks/{_TARGET_ID}", None),
        ("DELETE", f"/api/v1/b2b/tenant/webhooks/{_TARGET_ID}", None),
        ("PATCH", f"/api/v1/b2b/tenant/webhooks/{_TARGET_ID}", {"enabled": False}),
        ("GET", f"/api/v1/b2b/tenant/webhooks/{_TARGET_ID}/deliveries", None),
        ("POST", f"/api/v1/b2b/tenant/webhooks/{_TARGET_ID}/test", None),
        ("GET", f"/api/v1/b2b/tenant/webhooks/{_TARGET_ID}/crm-push-log", None),
        ("GET", f"/api/v1/b2b/tenant/reports/{_TARGET_ID}/pdf", None),
        ("GET", "/api/v1/b2b/tenant/vendors/ClickUp", None),
        ("GET", "/api/v1/b2b/tenant/vendor-pain-points", None),
        ("GET", "/api/v1/b2b/tenant/vendor-use-cases", None),
        ("GET", "/api/v1/b2b/tenant/vendor-integrations", None),
        ("GET", "/api/v1/b2b/tenant/vendor-buyer-profiles", None),
        ("GET", "/api/v1/b2b/tenant/watchlist-views", None),
        ("POST", "/api/v1/b2b/tenant/watchlist-views", {"name": "Boundary View", "vendor_names": ["ClickUp"]}),
        ("PUT", f"/api/v1/b2b/tenant/watchlist-views/{_TARGET_ID}", {"name": "Boundary View"}),
        ("DELETE", f"/api/v1/b2b/tenant/watchlist-views/{_TARGET_ID}", None),
        ("GET", f"/api/v1/b2b/tenant/watchlist-views/{_TARGET_ID}/alert-events", None),
        ("POST", f"/api/v1/b2b/tenant/watchlist-views/{_TARGET_ID}/alert-events/evaluate", None),
        ("GET", f"/api/v1/b2b/tenant/watchlist-views/{_TARGET_ID}/alert-email-log", None),
        (
            "POST",
            f"/api/v1/b2b/tenant/watchlist-views/{_TARGET_ID}/alert-events/deliver-email",
            {},
        ),
        ("GET", "/api/v1/b2b/tenant/product-profile?vendor_name=ClickUp", None),
        ("GET", "/api/v1/b2b/tenant/product-profile-history?vendor_name=ClickUp", None),
        ("GET", "/api/v1/b2b/tenant/change-events", None),
        ("GET", "/api/v1/b2b/tenant/change-events/summary", None),
        ("GET", "/api/v1/b2b/tenant/concurrent-events", None),
        ("GET", "/api/v1/b2b/tenant/displacement-edges", None),
        (
            "GET",
            "/api/v1/b2b/tenant/displacement-history?from_vendor=ClickUp&to_vendor=Monday.com",
            None,
        ),
        ("GET", "/api/v1/b2b/tenant/vendor-correlation?vendor_a=ClickUp&vendor_b=Monday.com", None),
        ("GET", "/api/v1/b2b/tenant/fuzzy-vendor-search?q=ClickUp", None),
        ("GET", "/api/v1/b2b/tenant/fuzzy-company-search?q=Acme", None),
        ("GET", "/api/v1/b2b/tenant/parser-version-status", None),
        ("GET", "/api/v1/b2b/tenant/parser-health", None),
        ("GET", "/api/v1/b2b/tenant/source-health", None),
        ("GET", "/api/v1/b2b/tenant/source-health/telemetry", None),
        ("GET", "/api/v1/b2b/tenant/source-health/telemetry-timeline", None),
        ("GET", "/api/v1/b2b/tenant/source-capabilities", None),
        ("GET", "/api/v1/b2b/tenant/operational-overview", None),
        ("GET", "/api/v1/b2b/tenant/calibration-weights", None),
        ("GET", "/api/v1/b2b/tenant/signal-to-outcome", None),
        ("GET", "/api/v1/b2b/tenant/competitive-sets", None),
        (
            "POST",
            "/api/v1/b2b/tenant/competitive-sets",
            {
                "name": "Boundary Set",
                "focal_vendor_name": "ClickUp",
                "competitor_vendor_names": ["Monday.com"],
            },
        ),
        (
            "PUT",
            f"/api/v1/b2b/tenant/competitive-sets/{_TARGET_ID}",
            {"name": "Boundary Set"},
        ),
        ("DELETE", f"/api/v1/b2b/tenant/competitive-sets/{_TARGET_ID}", None),
        ("GET", f"/api/v1/b2b/tenant/competitive-sets/{_TARGET_ID}/plan", None),
        ("POST", f"/api/v1/b2b/tenant/competitive-sets/{_TARGET_ID}/run", {}),
        ("GET", "/api/v1/b2b/campaigns/analytics/funnel", None),
        ("GET", "/api/v1/b2b/campaigns/analytics/by-vendor", None),
        ("GET", "/api/v1/b2b/campaigns/analytics/by-company", None),
        ("GET", "/api/v1/b2b/campaigns/analytics/timeline", None),
        ("GET", "/api/v1/b2b/campaigns", None),
        ("GET", "/api/v1/b2b/campaigns/stats", None),
        ("GET", "/api/v1/b2b/campaigns/quality-trends", None),
        ("GET", "/api/v1/b2b/campaigns/quality-diagnostics", None),
        ("GET", "/api/v1/b2b/campaigns/export", None),
        ("GET", f"/api/v1/b2b/campaigns/{_TARGET_ID}", None),
        ("GET", f"/api/v1/b2b/campaigns/{_TARGET_ID}/audit-log", None),
        ("GET", "/api/v1/b2b/campaigns/review-queue", None),
        ("GET", "/api/v1/b2b/campaigns/review-candidates", None),
        ("GET", "/api/v1/b2b/campaigns/review-candidates/summary", None),
        ("GET", "/api/v1/b2b/campaigns/review-queue/summary", None),
        (
            "POST",
            "/api/v1/b2b/campaigns/suppressions",
            {"email": "boundary@example.com", "reason": "manual"},
        ),
        ("DELETE", f"/api/v1/b2b/campaigns/suppressions/{_TARGET_ID}", None),
        ("GET", "/api/v1/b2b/campaigns/suppressions", None),
        ("GET", "/api/v1/b2b/campaigns/suppressions/check?email=boundary@example.com", None),
        ("GET", "/api/v1/b2b/campaigns/sequences", None),
        ("GET", f"/api/v1/b2b/campaigns/sequences/{_TARGET_ID}", None),
        ("GET", f"/api/v1/b2b/campaigns/sequences/{_TARGET_ID}/audit-log", None),
        (
            "POST",
            f"/api/v1/b2b/campaigns/sequences/{_TARGET_ID}/set-recipient",
            {"recipient_email": "boundary@example.com"},
        ),
        ("POST", f"/api/v1/b2b/campaigns/sequences/{_TARGET_ID}/pause", None),
        ("POST", f"/api/v1/b2b/campaigns/sequences/{_TARGET_ID}/resume", None),
        (
            "POST",
            f"/api/v1/b2b/campaigns/sequences/{_TARGET_ID}/outcome",
            {"outcome": "meeting_booked"},
        ),
        ("GET", f"/api/v1/b2b/campaigns/sequences/{_TARGET_ID}/outcome", None),
        ("GET", "/api/v1/seller/targets", None),
        (
            "POST",
            "/api/v1/seller/targets",
            {
                "seller_name": "Boundary Test Seller",
                "seller_type": "private_label",
                "categories": ["Home"],
                "source": "manual",
            },
        ),
        ("POST", "/api/v1/seller/campaigns/generate", {"limit": 1}),
    ],
)
def test_tenant_and_operator_routes_reject_unauthenticated_requests(
    monkeypatch,
    method: str,
    path: str,
    json_body: dict | None,
):
    monkeypatch.setattr(settings.saas_auth, "enabled", True, raising=False)

    def _fail_pool():
        raise AssertionError("db touched before auth")

    monkeypatch.setattr(vendor_targets, "get_db_pool", _fail_pool)
    monkeypatch.setattr(b2b_dashboard, "get_db_pool", _fail_pool)
    monkeypatch.setattr(b2b_vendor_briefing, "get_db_pool", _fail_pool)
    monkeypatch.setattr(b2b_campaigns, "get_db_pool", _fail_pool)
    monkeypatch.setattr(b2b_crm_events, "get_db_pool", _fail_pool)
    monkeypatch.setattr(b2b_tenant_dashboard, "get_db_pool", _fail_pool)
    monkeypatch.setattr(admin_costs, "get_db_pool", _fail_pool)
    monkeypatch.setattr(seller_campaigns, "get_db_pool", _fail_pool)

    def _fail_competitive_set_repo():
        raise AssertionError("competitive-set repo touched before auth")

    monkeypatch.setattr(b2b_tenant_dashboard, "get_competitive_set_repo", _fail_competitive_set_repo)

    app = _make_app()

    with TestClient(app) as client:
        response = client.request(method, path, json=json_body)

    assert response.status_code == 401, response.text
    assert response.json()["detail"] == "Authentication required"


@pytest.mark.parametrize(
    ("method", "path", "json_body"),
    [
        ("GET", "/api/v1/b2b/tenant/competitive-sets", None),
        (
            "POST",
            "/api/v1/b2b/tenant/competitive-sets",
            {
                "name": "Boundary Set",
                "focal_vendor_name": "ClickUp",
                "competitor_vendor_names": ["Monday.com"],
            },
        ),
        (
            "PUT",
            f"/api/v1/b2b/tenant/competitive-sets/{_TARGET_ID}",
            {"name": "Boundary Set"},
        ),
        ("DELETE", f"/api/v1/b2b/tenant/competitive-sets/{_TARGET_ID}", None),
        ("GET", f"/api/v1/b2b/tenant/competitive-sets/{_TARGET_ID}/plan", None),
        ("POST", f"/api/v1/b2b/tenant/competitive-sets/{_TARGET_ID}/run", {}),
        ("GET", "/api/v1/b2b/tenant/vendors", None),
        ("POST", "/api/v1/b2b/tenant/vendors", {"vendor_name": "ClickUp", "track_mode": "own"}),
        ("DELETE", "/api/v1/b2b/tenant/vendors/ClickUp", None),
        ("GET", "/api/v1/b2b/tenant/vendors/search?q=ClickUp", None),
        ("POST", "/api/v1/b2b/tenant/vendors/ClickUp/refresh", {}),
        ("GET", "/api/v1/b2b/tenant/export/signals", None),
        ("GET", "/api/v1/b2b/tenant/export/reviews", None),
        ("GET", "/api/v1/b2b/tenant/export/high-intent", None),
        ("GET", "/api/v1/b2b/tenant/export/source-health", None),
        ("GET", "/api/v1/b2b/tenant/overview", None),
        ("GET", "/api/v1/b2b/tenant/signals", None),
        ("GET", "/api/v1/b2b/tenant/signals/ClickUp", None),
        ("GET", "/api/v1/b2b/tenant/slow-burn-watchlist", None),
        ("GET", "/api/v1/b2b/tenant/accounts-in-motion-feed", None),
        ("GET", "/api/v1/b2b/tenant/vendor-history?vendor_name=ClickUp", None),
        ("GET", "/api/v1/b2b/tenant/compare-vendor-periods?vendor_name=ClickUp", None),
        ("GET", "/api/v1/b2b/tenant/pain-trends", None),
        ("GET", "/api/v1/b2b/tenant/displacement", None),
        ("GET", "/api/v1/b2b/tenant/pipeline", None),
        ("GET", "/api/v1/b2b/tenant/leads", None),
        ("GET", "/api/v1/b2b/tenant/high-intent", None),
        ("GET", "/api/v1/b2b/tenant/leads/Acme", None),
        ("GET", "/api/v1/b2b/tenant/reviews", None),
        ("GET", f"/api/v1/b2b/tenant/reviews/{_TARGET_ID}", None),
        (
            "POST",
            "/api/v1/b2b/tenant/push-to-crm",
            {"opportunities": [{"company": "Acme", "vendor": "ClickUp", "urgency": 8}]},
        ),
        ("GET", "/api/v1/b2b/tenant/opportunity-dispositions", None),
        (
            "POST",
            "/api/v1/b2b/tenant/opportunity-dispositions",
            {
                "opportunity_key": "Acme::ClickUp",
                "company": "Acme",
                "vendor": "ClickUp",
                "disposition": "saved",
            },
        ),
        (
            "POST",
            "/api/v1/b2b/tenant/opportunity-dispositions/bulk",
            {
                "items": [{"opportunity_key": "Acme::ClickUp", "company": "Acme", "vendor": "ClickUp"}],
                "disposition": "saved",
            },
        ),
        (
            "POST",
            "/api/v1/b2b/tenant/opportunity-dispositions/remove",
            {"opportunity_keys": ["Acme::ClickUp"]},
        ),
        ("GET", "/api/v1/b2b/tenant/watchlist-views", None),
        ("POST", "/api/v1/b2b/tenant/watchlist-views", {"name": "Boundary View", "vendor_names": ["ClickUp"]}),
        ("PUT", f"/api/v1/b2b/tenant/watchlist-views/{_TARGET_ID}", {"name": "Boundary View"}),
        ("DELETE", f"/api/v1/b2b/tenant/watchlist-views/{_TARGET_ID}", None),
        ("GET", f"/api/v1/b2b/tenant/watchlist-views/{_TARGET_ID}/alert-events", None),
        ("POST", f"/api/v1/b2b/tenant/watchlist-views/{_TARGET_ID}/alert-events/evaluate", None),
        ("GET", f"/api/v1/b2b/tenant/watchlist-views/{_TARGET_ID}/alert-email-log", None),
        (
            "POST",
            f"/api/v1/b2b/tenant/watchlist-views/{_TARGET_ID}/alert-events/deliver-email",
            {},
        ),
        ("GET", "/api/v1/b2b/tenant/reports", None),
        ("GET", f"/api/v1/b2b/tenant/reports/{_TARGET_ID}", None),
        (
            "POST",
            "/api/v1/b2b/tenant/reports/compare",
            {"primary_vendor": "ClickUp", "comparison_vendor": "Monday.com"},
        ),
        (
            "POST",
            "/api/v1/b2b/tenant/reports/compare-companies",
            {"primary_company": "Acme", "comparison_company": "Globex"},
        ),
        ("POST", "/api/v1/b2b/tenant/reports/company-deep-dive", {"company_name": "Acme"}),
        ("POST", "/api/v1/b2b/tenant/reports/battle-card", {"vendor_name": "ClickUp"}),
        ("GET", "/api/v1/b2b/tenant/report-subscriptions/library/default", None),
        (
            "PUT",
            "/api/v1/b2b/tenant/report-subscriptions/library/default",
            {"scope_label": "Boundary Library", "enabled": False},
        ),
    ],
)
def test_product_tenant_routes_reject_authenticated_underplan_before_service_touch(
    monkeypatch,
    method: str,
    path: str,
    json_body: dict | None,
):
    monkeypatch.setattr(settings.saas_auth, "enabled", True, raising=False)

    user = AuthUser(
        user_id=str(uuid4()),
        account_id=str(uuid4()),
        plan="b2b_trial",
        plan_status="active",
        role="owner",
        product="b2b_retention",
        is_admin=True,
    )

    def _fail_pool():
        raise AssertionError("db touched before plan gate")

    def _fail_competitive_set_repo():
        raise AssertionError("competitive-set repo touched before plan gate")

    monkeypatch.setattr(b2b_tenant_dashboard, "get_db_pool", _fail_pool)
    monkeypatch.setattr(b2b_tenant_dashboard, "get_competitive_set_repo", _fail_competitive_set_repo)

    app = _make_app()
    app.dependency_overrides[require_auth] = lambda: user

    with TestClient(app) as client:
        response = client.request(method, path, json=json_body)

    assert response.status_code == 403, response.text
    assert response.json()["detail"] == "Plan 'b2b_growth' or higher required (current: 'b2b_trial')"


@pytest.mark.parametrize(
    ("method", "path", "json_body"),
    [
        ("GET", "/api/v1/b2b/campaigns/analytics/funnel", None),
        ("GET", "/api/v1/b2b/campaigns/analytics/by-vendor", None),
        ("GET", "/api/v1/b2b/campaigns/analytics/by-company", None),
        ("GET", "/api/v1/b2b/campaigns/analytics/timeline", None),
        ("GET", "/api/v1/b2b/campaigns", None),
        ("GET", "/api/v1/b2b/campaigns/stats", None),
        ("GET", "/api/v1/b2b/campaigns/quality-trends", None),
        ("GET", "/api/v1/b2b/campaigns/quality-diagnostics", None),
        ("GET", "/api/v1/b2b/campaigns/export", None),
        ("GET", f"/api/v1/b2b/campaigns/{_TARGET_ID}", None),
        ("GET", f"/api/v1/b2b/campaigns/{_TARGET_ID}/audit-log", None),
        ("GET", "/api/v1/b2b/campaigns/review-queue", None),
        ("GET", "/api/v1/b2b/campaigns/review-candidates", None),
        ("GET", "/api/v1/b2b/campaigns/review-candidates/summary", None),
        ("GET", "/api/v1/b2b/campaigns/review-queue/summary", None),
        ("GET", "/api/v1/b2b/campaigns/suppressions", None),
        ("GET", "/api/v1/b2b/campaigns/suppressions/check?email=boundary@example.com", None),
        ("GET", "/api/v1/b2b/campaigns/sequences", None),
        ("GET", f"/api/v1/b2b/campaigns/sequences/{_TARGET_ID}", None),
        ("GET", f"/api/v1/b2b/campaigns/sequences/{_TARGET_ID}/audit-log", None),
        (
            "POST",
            f"/api/v1/b2b/campaigns/sequences/{_TARGET_ID}/set-recipient",
            {"recipient_email": "boundary@example.com"},
        ),
        ("POST", f"/api/v1/b2b/campaigns/sequences/{_TARGET_ID}/pause", None),
        ("POST", f"/api/v1/b2b/campaigns/sequences/{_TARGET_ID}/resume", None),
        (
            "POST",
            f"/api/v1/b2b/campaigns/sequences/{_TARGET_ID}/outcome",
            {"outcome": "meeting_booked"},
        ),
        ("GET", f"/api/v1/b2b/campaigns/sequences/{_TARGET_ID}/outcome", None),
    ],
)
def test_campaign_plan_gated_routes_reject_authenticated_underplan_before_db_touch(
    monkeypatch,
    method: str,
    path: str,
    json_body: dict | None,
):
    monkeypatch.setattr(settings.saas_auth, "enabled", True, raising=False)

    user = AuthUser(
        user_id=str(uuid4()),
        account_id=str(uuid4()),
        plan="b2b_trial",
        plan_status="active",
        role="owner",
        product="b2b_retention",
        is_admin=True,
    )

    def _fail_pool():
        raise AssertionError("db touched before plan gate")

    monkeypatch.setattr(b2b_campaigns, "get_db_pool", _fail_pool)

    app = _make_app()
    app.dependency_overrides[require_auth] = lambda: user

    with TestClient(app) as client:
        response = client.request(method, path, json=json_body)

    assert response.status_code == 403, response.text
    assert response.json()["detail"] == "Plan 'b2b_growth' or higher required (current: 'b2b_trial')"
