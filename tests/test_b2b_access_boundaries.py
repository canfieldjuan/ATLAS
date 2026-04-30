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
from atlas_brain.api import b2b_vendor_briefing
from atlas_brain.api import seller_campaigns
from atlas_brain.api import vendor_targets
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
        ("POST", "/api/v1/b2b/tenant/calibration/trigger", None),
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
        (
            "POST",
            "/api/v1/b2b/campaigns/suppressions",
            {"email": "boundary@example.com", "reason": "manual"},
        ),
        ("DELETE", f"/api/v1/b2b/campaigns/suppressions/{_TARGET_ID}", None),
        (
            "POST",
            f"/api/v1/b2b/campaigns/sequences/{_TARGET_ID}/set-recipient",
            {"recipient_email": "boundary@example.com"},
        ),
        ("POST", f"/api/v1/b2b/campaigns/sequences/{_TARGET_ID}/pause", None),
        ("POST", f"/api/v1/b2b/campaigns/sequences/{_TARGET_ID}/resume", None),
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
    monkeypatch.setattr(admin_costs, "get_db_pool", _fail_pool)
    monkeypatch.setattr(seller_campaigns, "get_db_pool", _fail_pool)

    app = _make_app()

    with TestClient(app) as client:
        response = client.request(method, path, json=json_body)

    assert response.status_code == 401, response.text
    assert response.json()["detail"] == "Authentication required"
