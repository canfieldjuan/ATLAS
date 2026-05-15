from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from extracted_content_pipeline.api.generated_assets import (
    GeneratedAssetApiConfig,
    create_generated_asset_router,
)
import extracted_content_pipeline.api.generated_assets as asset_api
from extracted_content_pipeline.campaign_ports import TenantScope


class _Pool:
    def __init__(
        self,
        rows=None,
        *,
        execute_result: str = "UPDATE 1",
        initialized: bool = True,
    ) -> None:
        self.rows = list(rows or [])
        self.execute_result = execute_result
        self.is_initialized = initialized
        self.fetch_calls: list[tuple[str, tuple[object, ...]]] = []
        self.execute_calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetch(self, query, *args):
        self.fetch_calls.append((str(query), args))
        return self.rows

    async def execute(self, query, *args):
        self.execute_calls.append((str(query), args))
        return self.execute_result


def _report_row():
    return {
        "id": "report-uuid-1",
        "status": "draft",
        "target_id": "vendor-acme",
        "target_mode": "vendor_retention",
        "report_type": "vendor_pressure",
        "title": "Acme report",
        "summary": "Pricing pressure dominates.",
        "sections": [{"id": "summary", "title": "Summary", "body_markdown": "Body"}],
        "reference_ids": ["r1"],
        "metadata": {
            "generation_usage": {"input_tokens": 10, "output_tokens": 5},
            "reasoning_context": {"wedge": "price_squeeze", "confidence": "high"},
        },
    }


def _blog_post_row():
    return {
        "id": "blog-post-uuid-1",
        "status": "draft",
        "slug": "acme-pricing-pressure",
        "title": "Acme Pricing Pressure",
        "description": "Pricing pressure dominates.",
        "topic_type": "vendor_alternative",
        "tags": ["pricing"],
        "content": "body",
        "charts": [],
        "data_context": {
            "_metadata": {
                "generation_usage": {"input_tokens": 9, "output_tokens": 4},
                "reasoning_context": {"wedge": "price_squeeze", "confidence": "high"},
            }
        },
        "llm_model": "fake-llm",
    }


def _landing_page_row():
    return {
        "id": "landing-page-uuid-1",
        "status": "draft",
        "campaign_name": "acme-launch",
        "persona": "VP Engineering",
        "value_prop": "Catch pressure early",
        "title": "Acme landing page",
        "slug": "acme-launch",
        "hero": {"headline": "Stop surprises"},
        "sections": [{"id": "problem", "title": "Problem", "body_markdown": "Body"}],
        "cta": {"label": "Book a demo"},
        "meta": {"title_tag": "Acme landing page"},
        "reference_ids": ["r1"],
        "metadata": {},
    }


def _sales_brief_row():
    return {
        "id": "brief-uuid-1",
        "status": "draft",
        "target_id": "vendor-acme",
        "target_mode": "vendor_retention",
        "brief_type": "pre_call",
        "title": "Acme brief",
        "headline": "Renewal pressure opens this week",
        "sections": [{"id": "context", "title": "Context", "body_markdown": "Body"}],
        "reference_ids": ["r1"],
        "metadata": {
            "generation_usage": {"input_tokens": 8, "output_tokens": 4},
            "reasoning_context": {"wedge": "support_erosion", "confidence": "medium"},
        },
    }


def _client(
    pool,
    *,
    scope=None,
    config: GeneratedAssetApiConfig | None = None,
    dependencies=None,
) -> TestClient:
    app = FastAPI()

    async def pool_provider():
        return pool

    async def scope_provider():
        return scope

    app.include_router(
        create_generated_asset_router(
            pool_provider=pool_provider,
            scope_provider=scope_provider if scope is not None else None,
            config=config,
            dependencies=dependencies,
        )
    )
    return TestClient(app)


def test_generated_asset_router_lists_report_drafts_with_filters() -> None:
    pool = _Pool(rows=[_report_row()])

    response = _client(
        pool,
        scope=TenantScope(account_id="acct_1"),
    ).get(
        "/content-assets/report/drafts"
        "?target_mode=vendor_retention&report_type=vendor_pressure&limit=5"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["rows"][0]["id"] == "report-uuid-1"
    assert body["rows"][0]["status"] == "draft"
    assert body["rows"][0]["title"] == "Acme report"
    assert body["rows"][0]["reasoning_wedge"] == "price_squeeze"
    query, args = pool.fetch_calls[0]
    assert "FROM reports" in query
    assert args == ("acct_1", "draft", "vendor_retention", "vendor_pressure", 5)


def test_generated_asset_router_lists_blog_post_drafts_with_filters() -> None:
    pool = _Pool(rows=[_blog_post_row()])

    response = _client(
        pool,
        scope=TenantScope(account_id="acct_1"),
    ).get(
        "/content-assets/blog_post/drafts"
        "?topic_type=vendor_alternative&limit=5"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["rows"][0]["id"] == "blog-post-uuid-1"
    assert body["rows"][0]["status"] == "draft"
    assert body["rows"][0]["slug"] == "acme-pricing-pressure"
    assert body["rows"][0]["reasoning_wedge"] == "price_squeeze"
    query, args = pool.fetch_calls[0]
    assert "FROM blog_posts" in query
    assert args == ("acct_1", "draft", "vendor_alternative", 5)


def test_generated_asset_router_exports_landing_page_csv() -> None:
    pool = _Pool(rows=[_landing_page_row()])

    response = _client(pool).get(
        "/content-assets/landing_page/drafts/export"
        "?format=csv&campaign_name=acme-launch&slug=acme-launch"
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "content_assets_landing_page.csv" in response.headers["content-disposition"]
    assert "campaign_name,persona,value_prop" in response.text
    assert "Acme landing page" in response.text
    query, args = pool.fetch_calls[0]
    assert "FROM landing_pages" in query
    assert args == ("", "draft", "acme-launch", "acme-launch", 20)


def test_generated_asset_router_exports_sales_brief_json_without_status_filter() -> None:
    pool = _Pool(rows=[_sales_brief_row()])

    response = _client(pool).get(
        "/content-assets/sales_brief/drafts/export"
        "?format=json&status=&target_mode=vendor_retention&brief_type=pre_call"
    )

    assert response.status_code == 200
    row = response.json()["rows"][0]
    assert row["brief_type"] == "pre_call"
    assert row["reasoning_wedge"] == "support_erosion"
    query, args = pool.fetch_calls[0]
    assert "FROM sales_briefs" in query
    assert "status = " not in query
    assert args == ("", "vendor_retention", "pre_call", 20)


def test_generated_asset_router_reviews_report_with_host_defined_status() -> None:
    pool = _Pool()

    response = _client(
        pool,
        scope={"account_id": "acct_1"},
    ).post(
        "/content-assets/report/drafts/review",
        json={"id": "report-uuid-1", "status": "published"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "account_id": "acct_1",
        "asset": "report",
        "id": "report-uuid-1",
        "status": "published",
        "updated": True,
    }
    query, args = pool.execute_calls[0]
    assert "UPDATE reports" in query
    assert args == ("report-uuid-1", "published", "acct_1")


def test_generated_asset_router_reviews_blog_post_with_host_defined_status() -> None:
    pool = _Pool()

    response = _client(
        pool,
        scope={"account_id": "acct_1"},
    ).post(
        "/content-assets/blog_post/drafts/review",
        json={"id": "blog-post-uuid-1", "status": "published"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "account_id": "acct_1",
        "asset": "blog_post",
        "id": "blog-post-uuid-1",
        "status": "published",
        "updated": True,
    }
    query, args = pool.execute_calls[0]
    assert "UPDATE blog_posts" in query
    assert args == ("blog-post-uuid-1", "published", "acct_1")


def test_generated_asset_router_returns_miss_without_hiding_result() -> None:
    pool = _Pool(execute_result="UPDATE 0")

    response = _client(pool).post(
        "/content-assets/sales_brief/drafts/review",
        json={"asset_id": "brief-uuid-1", "status": "ready_for_call"},
    )

    assert response.status_code == 200
    assert response.json()["updated"] is False
    query, args = pool.execute_calls[0]
    assert "UPDATE sales_briefs" in query
    assert args == ("brief-uuid-1", "ready_for_call", "")


def test_generated_asset_router_rejects_unknown_asset() -> None:
    response = _client(_Pool()).get("/content-assets/podcast_episode/drafts")

    assert response.status_code == 400
    assert "asset must be one of" in response.json()["detail"]


def test_generated_asset_router_rejects_unknown_asset_before_pool_resolution() -> None:
    app = FastAPI()
    calls = 0

    async def pool_provider():
        nonlocal calls
        calls += 1
        raise AssertionError("pool provider should not be touched")

    app.include_router(create_generated_asset_router(pool_provider=pool_provider))

    response = TestClient(app).get("/content-assets/podcast_episode/drafts")

    assert response.status_code == 400
    assert "asset must be one of" in response.json()["detail"]
    assert calls == 0


def test_generated_asset_router_rejects_empty_review_status() -> None:
    response = _client(_Pool()).post(
        "/content-assets/report/drafts/review",
        json={"id": "report-uuid-1", "status": ""},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "status is required"


def test_generated_asset_router_rejects_unknown_export_format() -> None:
    response = _client(_Pool(rows=[_report_row()])).get(
        "/content-assets/report/drafts/export?format=xlsx"
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "format must be csv or json"


def test_generated_asset_router_requires_database() -> None:
    response = _client(_Pool(initialized=False)).get("/content-assets/report/drafts")

    assert response.status_code == 503
    assert response.json()["detail"] == "Database unavailable"


def test_generated_asset_router_honors_host_dependencies() -> None:
    pool = _Pool(rows=[_report_row()])

    def require_auth():
        raise HTTPException(status_code=403, detail="forbidden")

    response = _client(pool, dependencies=[Depends(require_auth)]).get(
        "/content-assets/report/drafts"
    )

    assert response.status_code == 403
    assert pool.fetch_calls == []


def test_generated_asset_api_config_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="max_limit must be positive"):
        GeneratedAssetApiConfig(max_limit=0)

    with pytest.raises(ValueError, match="default_limit must be less"):
        GeneratedAssetApiConfig(default_limit=5, max_limit=4)


def test_generated_asset_router_requires_fastapi(monkeypatch) -> None:
    monkeypatch.setattr(asset_api, "_FASTAPI_IMPORT_ERROR", ImportError("missing"))

    with pytest.raises(RuntimeError, match="FastAPI is required"):
        create_generated_asset_router(pool_provider=lambda: None)
