from __future__ import annotations

from datetime import datetime, timezone

import pytest

pytest.importorskip("fastapi")

from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from extracted_content_pipeline.api.b2b_campaigns import (
    B2BCampaignApiConfig,
    create_b2b_campaign_router,
)
import extracted_content_pipeline.api.b2b_campaigns as b2b_api
from extracted_content_pipeline.campaign_ports import TenantScope


CAMPAIGN_ID = "00000000-0000-0000-0000-000000000001"


class _Pool:
    def __init__(self, rows=None, *, initialized: bool = True) -> None:
        self.rows = list(rows or [])
        self.is_initialized = initialized
        self.fetch_calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetch(self, query, *args):
        self.fetch_calls.append((str(query), args))
        return self.rows


def _draft_row(**overrides):
    row = {
        "id": CAMPAIGN_ID,
        "company_name": "Acme",
        "vendor_name": "LegacyCRM",
        "target_mode": "vendor_retention",
        "channel": "email_cold",
        "status": "draft",
        "recipient_email": "buyer@example.com",
        "subject": "Acme renewal plan",
        "body": "Body",
        "cta": "Review plan",
        "llm_model": "offline",
        "created_at": datetime(2026, 5, 4, tzinfo=timezone.utc),
        "metadata": {"scope": {"account_id": "acct_1"}},
    }
    row.update(overrides)
    return row


def _review_row(**overrides):
    row = {
        "id": CAMPAIGN_ID,
        "previous_status": "draft",
        "status": "approved",
        "company_name": "Acme",
        "vendor_name": "LegacyCRM",
        "channel": "email_cold",
        "recipient_email": "buyer@example.com",
        "from_email": None,
        "metadata": {"scope": {"account_id": "acct_1"}},
    }
    row.update(overrides)
    return row


def _client(
    pool,
    *,
    scope=None,
    config: B2BCampaignApiConfig | None = None,
    dependencies=None,
) -> TestClient:
    app = FastAPI()

    async def pool_provider():
        return pool

    async def scope_provider():
        return scope

    app.include_router(
        create_b2b_campaign_router(
            pool_provider=pool_provider,
            scope_provider=scope_provider if scope is not None else None,
            config=config,
            dependencies=dependencies,
        )
    )
    return TestClient(app)


def test_b2b_campaign_router_lists_scoped_drafts() -> None:
    pool = _Pool(rows=[_draft_row()])

    response = _client(
        pool,
        scope=TenantScope(account_id="acct_1"),
    ).get(
        "/b2b/campaigns/drafts"
        "?statuses=draft,approved&target_mode=vendor_retention"
        "&channel=email_cold&vendor_name=LegacyCRM&company_name=Acme&limit=5"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["rows"][0]["company_name"] == "Acme"
    query, args = pool.fetch_calls[0]
    assert "FROM \"b2b_campaigns\"" in query
    assert "metadata -> 'scope' ->> 'account_id' = $2" in query
    assert args == (
        ["draft", "approved"],
        "acct_1",
        "vendor_retention",
        "email_cold",
        "LegacyCRM",
        "Acme",
        5,
    )


def test_b2b_campaign_router_caps_export_limit_from_config() -> None:
    pool = _Pool(rows=[_draft_row()])

    response = _client(
        pool,
        config=B2BCampaignApiConfig(default_limit=3, max_limit=3),
    ).get("/b2b/campaigns/drafts?limit=99")

    assert response.status_code == 200
    assert response.json()["limit"] == 3
    assert pool.fetch_calls[0][1] == (["draft"], 3)


def test_b2b_campaign_router_exports_csv() -> None:
    pool = _Pool(rows=[_draft_row()])

    response = _client(pool).get("/b2b/campaigns/drafts/export?format=csv")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment; filename=\"campaign_drafts.csv\"" in (
        response.headers["content-disposition"]
    )
    assert "company_name,vendor_name" in response.text
    assert "Acme" in response.text


def test_b2b_campaign_router_exports_json_when_requested() -> None:
    pool = _Pool(rows=[_draft_row()])

    response = _client(pool).get("/b2b/campaigns/drafts/export?format=json")

    assert response.status_code == 200
    assert response.json()["rows"][0]["vendor_name"] == "LegacyCRM"


def test_b2b_campaign_router_rejects_unknown_export_format() -> None:
    pool = _Pool(rows=[_draft_row()])

    response = _client(pool).get("/b2b/campaigns/drafts/export?format=xlsx")

    assert response.status_code == 400
    assert response.json()["detail"] == "format must be csv or json"


def test_b2b_campaign_router_reviews_selected_drafts() -> None:
    pool = _Pool(rows=[_review_row(status="queued", from_email="sales@example.com")])

    response = _client(
        pool,
        scope={"account_id": "acct_1"},
    ).post(
        "/b2b/campaigns/drafts/review",
        json={
            "campaign_ids": [CAMPAIGN_ID],
            "status": "queued",
            "from_statuses": "draft,approved",
            "from_email": "sales@example.com",
            "reason": "customer approved",
            "reviewed_by": "ops@example.com",
            "metadata": {"review_batch": "batch_1"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["updated"] == 1
    assert body["status"] == "queued"
    query, args = pool.fetch_calls[0]
    assert "UPDATE \"b2b_campaigns\" AS campaign" in query
    assert "metadata -> 'scope' ->> 'account_id' = $3" in query
    assert args[0] == [CAMPAIGN_ID]
    assert args[1] == ["draft", "approved"]
    assert args[2] == "acct_1"
    assert args[3] == "queued"
    assert args[5] == "sales@example.com"


def test_b2b_campaign_router_supports_dry_run_review() -> None:
    pool = _Pool(rows=[_review_row(status="draft")])

    response = _client(pool).post(
        "/b2b/campaigns/drafts/review",
        json={"campaign_ids": CAMPAIGN_ID, "status": "approved", "dry_run": True},
    )

    assert response.status_code == 200
    assert response.json()["dry_run"] is True
    query, args = pool.fetch_calls[0]
    assert "SELECT" in query
    assert "UPDATE" not in query
    assert args == ([CAMPAIGN_ID], ["draft"])


def test_b2b_campaign_router_rejects_invalid_review_status() -> None:
    pool = _Pool(rows=[_review_row()])

    response = _client(pool).post(
        "/b2b/campaigns/drafts/review",
        json={"campaign_ids": [CAMPAIGN_ID], "status": "sent"},
    )

    assert response.status_code == 400
    assert "unsupported review status" in response.json()["detail"]
    assert pool.fetch_calls == []


def test_b2b_campaign_router_requires_database() -> None:
    response = _client(_Pool(initialized=False)).get("/b2b/campaigns/drafts")

    assert response.status_code == 503
    assert response.json()["detail"] == "Database unavailable"


def test_b2b_campaign_router_honors_host_dependencies() -> None:
    pool = _Pool(rows=[_draft_row()])

    def require_auth():
        raise HTTPException(status_code=403, detail="forbidden")

    response = _client(pool, dependencies=[Depends(require_auth)]).get(
        "/b2b/campaigns/drafts"
    )

    assert response.status_code == 403
    assert pool.fetch_calls == []


def test_b2b_campaign_api_config_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="max_limit must be positive"):
        B2BCampaignApiConfig(max_limit=0)

    with pytest.raises(ValueError, match="default_limit must be less"):
        B2BCampaignApiConfig(default_limit=5, max_limit=4)


def test_b2b_campaign_router_requires_fastapi(monkeypatch) -> None:
    monkeypatch.setattr(b2b_api, "_FASTAPI_IMPORT_ERROR", ImportError("missing"))

    with pytest.raises(RuntimeError, match="FastAPI is required"):
        create_b2b_campaign_router(pool_provider=lambda: None)
