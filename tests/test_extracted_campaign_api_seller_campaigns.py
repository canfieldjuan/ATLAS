from __future__ import annotations

from datetime import datetime, timezone

import pytest

pytest.importorskip("fastapi")

from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

import extracted_content_pipeline.api.seller_campaigns as seller_api
from extracted_content_pipeline.api.seller_campaigns import (
    SellerCampaignApiConfig,
    create_seller_campaign_router,
)
from extracted_content_pipeline.campaign_ports import TenantScope


TARGET_ID = "00000000-0000-0000-0000-000000000001"
CAMPAIGN_ID = "00000000-0000-0000-0000-000000000011"


class _Pool:
    def __init__(
        self,
        *,
        rows=None,
        row=None,
        total: int = 0,
        execute_result="UPDATE 1",
        initialized: bool = True,
    ) -> None:
        self.rows = list(rows or [])
        self.row = row
        self.total = total
        self.execute_result = execute_result
        self.is_initialized = initialized
        self.fetch_calls: list[tuple[str, tuple[object, ...]]] = []
        self.fetchrow_calls: list[tuple[str, tuple[object, ...]]] = []
        self.fetchval_calls: list[tuple[str, tuple[object, ...]]] = []
        self.execute_calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetch(self, query, *args):
        self.fetch_calls.append((str(query), args))
        return self.rows

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append((str(query), args))
        return self.row

    async def fetchval(self, query, *args):
        self.fetchval_calls.append((str(query), args))
        return self.total

    async def execute(self, query, *args):
        self.execute_calls.append((str(query), args))
        return self.execute_result


def _target_row(**overrides):
    row = {
        "id": TARGET_ID,
        "seller_name": "Acme Seller",
        "company_name": "Acme",
        "email": "owner@example.com",
        "seller_type": "private_label",
        "categories": ["supplements"],
        "storefront_url": None,
        "notes": None,
        "status": "active",
        "source": "manual",
        "created_at": datetime(2026, 5, 4, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 5, 4, tzinfo=timezone.utc),
    }
    row.update(overrides)
    return row


def _campaign_row(**overrides):
    row = {
        "id": CAMPAIGN_ID,
        "company_name": "Acme Seller",
        "vendor_name": None,
        "target_mode": "amazon_seller",
        "channel": "email_cold",
        "status": "draft",
        "recipient_email": "owner@example.com",
        "subject": "Category wins",
        "body": "Body",
        "cta": "Review category report",
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
        "status": "queued",
        "company_name": "Acme Seller",
        "vendor_name": None,
        "channel": "email_cold",
        "recipient_email": "owner@example.com",
        "from_email": "seller@example.com",
        "metadata": {"scope": {"account_id": "acct_1"}},
    }
    row.update(overrides)
    return row


class _Result:
    def __init__(self, **values) -> None:
        self.values = values

    def as_dict(self):
        return dict(self.values)


def _client(
    pool,
    *,
    scope=None,
    config: SellerCampaignApiConfig | None = None,
    dependencies=None,
) -> TestClient:
    app = FastAPI()

    async def pool_provider():
        return pool

    async def scope_provider():
        return scope

    app.include_router(
        create_seller_campaign_router(
            pool_provider=pool_provider,
            scope_provider=scope_provider if scope is not None else None,
            config=config,
            dependencies=dependencies,
        )
    )
    return TestClient(app)


def test_seller_campaign_router_lists_targets() -> None:
    pool = _Pool(rows=[_target_row()], total=1)

    response = _client(pool).get(
        "/seller/targets?status=active&seller_type=private_label"
        "&category=supplements&limit=5&offset=2"
    )

    assert response.status_code == 200
    assert response.json()["targets"][0]["seller_name"] == "Acme Seller"
    query, args = pool.fetch_calls[0]
    assert "FROM \"seller_targets\"" in query
    assert args == ("active", "private_label", "supplements", 5, 2)


def test_seller_campaign_router_creates_target() -> None:
    pool = _Pool(row=_target_row())

    response = _client(pool).post(
        "/seller/targets",
        json={
            "seller_name": "Acme Seller",
            "seller_type": "private_label",
            "categories": ["supplements"],
        },
    )

    assert response.status_code == 200
    assert response.json()["id"] == TARGET_ID
    query, args = pool.fetchrow_calls[0]
    assert "INSERT INTO \"seller_targets\"" in query
    assert args[0] == "Acme Seller"


def test_seller_campaign_router_rejects_target_without_name() -> None:
    response = _client(_Pool()).post(
        "/seller/targets",
        json={"seller_name": " ", "company_name": ""},
    )

    assert response.status_code == 400
    assert "seller_name or company_name" in response.json()["detail"]


def test_seller_campaign_router_updates_and_deletes_targets() -> None:
    pool = _Pool(row=_target_row(status="paused"), execute_result="UPDATE 1")
    client = _client(pool)

    patch = client.patch(f"/seller/targets/{TARGET_ID}", json={"status": "paused"})

    assert patch.status_code == 200
    assert patch.json()["status"] == "paused"
    assert pool.execute_calls[0][1] == ("paused", TARGET_ID)

    pool.execute_result = "DELETE 1"
    delete = client.delete(f"/seller/targets/{TARGET_ID}")

    assert delete.status_code == 200
    assert delete.json() == {"ok": True}


def test_seller_campaign_router_returns_404_for_missing_target() -> None:
    response = _client(_Pool(row=None)).get(f"/seller/targets/{TARGET_ID}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Seller target not found"


def test_seller_campaign_router_refreshes_category_intelligence(monkeypatch) -> None:
    pool = _Pool()
    calls = []

    async def _refresh(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(refreshed=1, failed=0, categories=["supplements"])

    monkeypatch.setattr(seller_api, "refresh_seller_category_intelligence", _refresh)

    response = _client(
        pool,
        config=SellerCampaignApiConfig(
            category_reviews_table="reviews",
            category_metadata_table="metadata",
            category_snapshots_table="snapshots",
        ),
    ).post(
        "/seller/intelligence/refresh",
        json={
            "category": "supplements",
            "categories": ["beauty", "supplements"],
            "min_reviews": 75,
            "limit": 7,
        },
    )

    assert response.status_code == 200
    assert response.json()["categories"] == ["supplements"]
    assert calls == [
        (
            pool,
            {
                "categories": ("beauty", "supplements"),
                "min_reviews": 75,
                "limit": 7,
                "reviews_table": "reviews",
                "metadata_table": "metadata",
                "snapshots_table": "snapshots",
            },
        )
    ]


def test_seller_campaign_router_prepares_opportunities_from_scope(monkeypatch) -> None:
    pool = _Pool()
    calls = []

    async def _prepare(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(prepared=2, target_mode="amazon_seller")

    monkeypatch.setattr(seller_api, "prepare_seller_campaign_opportunities", _prepare)

    response = _client(
        pool,
        scope=TenantScope(account_id="acct_1"),
        config=SellerCampaignApiConfig(
            seller_targets_table="targets",
            category_snapshots_table="snapshots",
            opportunities_table="opps",
        ),
    ).post(
        "/seller/opportunities/prepare",
        json={
            "category": "supplements",
            "seller_status": "paused",
            "limit": 9,
            "replace_existing": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["prepared"] == 2
    assert calls == [
        (
            pool,
            {
                "account_id": "acct_1",
                "category": "supplements",
                "seller_status": "paused",
                "limit": 9,
                "replace_existing": True,
                "target_mode": "amazon_seller",
                "seller_targets_table": "targets",
                "category_snapshots_table": "snapshots",
                "opportunities_table": "opps",
            },
        )
    ]


def test_seller_campaign_router_combined_operation_skips_prepare_on_refresh_failure(
    monkeypatch,
) -> None:
    prepare_calls = []

    async def _refresh(_pool, **_kwargs):
        return _Result(refreshed=0, failed=1, errors=["supplements: boom"])

    async def _prepare(received_pool, **kwargs):
        prepare_calls.append((received_pool, kwargs))
        return _Result(prepared=1)

    monkeypatch.setattr(seller_api, "refresh_seller_category_intelligence", _refresh)
    monkeypatch.setattr(seller_api, "prepare_seller_campaign_opportunities", _prepare)

    response = _client(_Pool()).post(
        "/seller/operations/refresh-and-prepare",
        json={"category": "supplements"},
    )

    assert response.status_code == 200
    assert response.json()["prepare"] is None
    assert response.json()["prepare_skipped"] is True
    assert prepare_calls == []


def test_seller_campaign_router_combined_operation_can_continue_after_refresh_failure(
    monkeypatch,
) -> None:
    calls = []

    async def _refresh(received_pool, **kwargs):
        calls.append(("refresh", received_pool, kwargs))
        return _Result(refreshed=0, failed=1, errors=["supplements: boom"])

    async def _prepare(received_pool, **kwargs):
        calls.append(("prepare", received_pool, kwargs))
        return _Result(prepared=1)

    monkeypatch.setattr(seller_api, "refresh_seller_category_intelligence", _refresh)
    monkeypatch.setattr(seller_api, "prepare_seller_campaign_opportunities", _prepare)

    response = _client(_Pool(), scope={"account_id": "acct_1"}).post(
        "/seller/operations/refresh-and-prepare",
        json={
            "category": "supplements",
            "continue_on_refresh_failure": True,
            "replace_existing": "yes",
        },
    )

    assert response.status_code == 200
    assert response.json()["prepare"] == {"prepared": 1}
    assert response.json()["prepare_skipped"] is False
    assert [item[0] for item in calls] == ["refresh", "prepare"]
    assert calls[1][2]["account_id"] == "acct_1"
    assert calls[1][2]["replace_existing"] is True


def test_seller_campaign_router_combined_operation_prepares_requested_categories(
    monkeypatch,
) -> None:
    prepare_categories = []

    async def _refresh(_pool, **_kwargs):
        return _Result(refreshed=2, failed=0, categories=["beauty", "supplements"])

    async def _prepare(_pool, **kwargs):
        prepare_categories.append(kwargs["category"])
        return _Result(
            prepared=1,
            skipped=0,
            replaced=0,
            target_mode="amazon_seller",
            target_ids=[f"target-{kwargs['category']}"],
            categories=[kwargs["category"]],
        )

    monkeypatch.setattr(seller_api, "refresh_seller_category_intelligence", _refresh)
    monkeypatch.setattr(seller_api, "prepare_seller_campaign_opportunities", _prepare)

    response = _client(_Pool()).post(
        "/seller/operations/refresh-and-prepare",
        json={"categories": ["beauty", "supplements"]},
    )

    assert response.status_code == 200
    assert prepare_categories == ["beauty", "supplements"]
    assert response.json()["prepare"]["prepared"] == 2
    assert response.json()["prepare"]["categories"] == ["beauty", "supplements"]


def test_seller_campaign_router_combined_operation_deduplicates_categories(
    monkeypatch,
) -> None:
    prepare_categories = []

    async def _refresh(_pool, **_kwargs):
        return _Result(refreshed=2, failed=0, categories=["beauty", "supplements"])

    async def _prepare(_pool, **kwargs):
        prepare_categories.append(kwargs["category"])
        return _Result(
            prepared=1,
            skipped=0,
            replaced=0,
            target_mode="amazon_seller",
            target_ids=[f"target-{kwargs['category']}"],
            categories=[kwargs["category"]],
        )

    monkeypatch.setattr(seller_api, "refresh_seller_category_intelligence", _refresh)
    monkeypatch.setattr(seller_api, "prepare_seller_campaign_opportunities", _prepare)

    response = _client(_Pool()).post(
        "/seller/operations/refresh-and-prepare",
        json={"categories": ["beauty", "beauty", "supplements"], "category": "beauty"},
    )

    assert response.status_code == 200
    assert prepare_categories == ["beauty", "supplements"]
    assert response.json()["prepare"]["prepared"] == 2
    assert response.json()["prepare"]["categories"] == ["beauty", "supplements"]


def test_seller_campaign_router_combined_operation_prepares_refreshed_categories(
    monkeypatch,
) -> None:
    prepare_categories = []

    async def _refresh(_pool, **_kwargs):
        return _Result(refreshed=2, failed=0, categories=["beauty", "supplements"])

    async def _prepare(_pool, **kwargs):
        prepare_categories.append(kwargs["category"])
        return _Result(
            prepared=1,
            skipped=0,
            replaced=0,
            target_mode="amazon_seller",
            target_ids=[f"target-{kwargs['category']}"],
            categories=[kwargs["category"]],
        )

    monkeypatch.setattr(seller_api, "refresh_seller_category_intelligence", _refresh)
    monkeypatch.setattr(seller_api, "prepare_seller_campaign_opportunities", _prepare)

    response = _client(_Pool()).post(
        "/seller/operations/refresh-and-prepare",
        json={"limit": 2},
    )

    assert response.status_code == 200
    assert prepare_categories == ["beauty", "supplements"]
    assert response.json()["prepare"]["prepared"] == 2
    assert response.json()["prepare"]["categories"] == ["beauty", "supplements"]


def test_seller_campaign_router_combined_operation_skips_without_refreshed_categories(
    monkeypatch,
) -> None:
    prepare_calls = []

    async def _refresh(_pool, **_kwargs):
        return _Result(refreshed=0, failed=0, categories=[])

    async def _prepare(received_pool, **kwargs):
        prepare_calls.append((received_pool, kwargs))
        return _Result(prepared=1)

    monkeypatch.setattr(seller_api, "refresh_seller_category_intelligence", _refresh)
    monkeypatch.setattr(seller_api, "prepare_seller_campaign_opportunities", _prepare)

    response = _client(_Pool()).post("/seller/operations/refresh-and-prepare")

    assert response.status_code == 200
    assert response.json()["prepare"] is None
    assert response.json()["prepare_skipped"] is True
    assert prepare_calls == []


def test_seller_campaign_router_rejects_invalid_continue_flag_before_refresh(
    monkeypatch,
) -> None:
    calls = []

    async def _refresh(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(refreshed=1, failed=0)

    monkeypatch.setattr(seller_api, "refresh_seller_category_intelligence", _refresh)

    response = _client(_Pool()).post(
        "/seller/operations/refresh-and-prepare",
        json={"category": "beauty", "continue_on_refresh_failure": "maybe"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "continue_on_refresh_failure must be a boolean"
    assert calls == []


def test_seller_campaign_router_rejects_malformed_refresh_categories(
    monkeypatch,
) -> None:
    calls = []

    async def _refresh(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(refreshed=1)

    monkeypatch.setattr(seller_api, "refresh_seller_category_intelligence", _refresh)

    response = _client(_Pool()).post(
        "/seller/intelligence/refresh",
        json={"categories": {"name": "beauty"}},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "categories must be a list or string"
    assert calls == []


def test_seller_campaign_router_rejects_malformed_combined_categories(
    monkeypatch,
) -> None:
    calls = []

    async def _refresh(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(refreshed=1)

    monkeypatch.setattr(seller_api, "refresh_seller_category_intelligence", _refresh)

    response = _client(_Pool()).post(
        "/seller/operations/refresh-and-prepare",
        json={"categories": 123},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "categories must be a list or string"
    assert calls == []


def test_seller_campaign_router_rejects_boolean_numeric_payload(monkeypatch) -> None:
    calls = []

    async def _refresh(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(refreshed=1)

    monkeypatch.setattr(seller_api, "refresh_seller_category_intelligence", _refresh)

    response = _client(_Pool()).post(
        "/seller/intelligence/refresh",
        json={"category": "supplements", "min_reviews": True},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "min_reviews must be an integer"
    assert calls == []


def test_seller_campaign_router_rejects_float_numeric_payload(monkeypatch) -> None:
    calls = []

    async def _refresh(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(refreshed=1)

    monkeypatch.setattr(seller_api, "refresh_seller_category_intelligence", _refresh)

    response = _client(_Pool()).post(
        "/seller/intelligence/refresh",
        json={"category": "supplements", "limit": 7.9},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "limit must be an integer"
    assert calls == []


def test_seller_campaign_router_rejects_refresh_limit_above_configured_cap(
    monkeypatch,
) -> None:
    calls = []

    async def _refresh(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(refreshed=1)

    monkeypatch.setattr(seller_api, "refresh_seller_category_intelligence", _refresh)

    response = _client(_Pool(), config=SellerCampaignApiConfig(max_limit=60)).post(
        "/seller/intelligence/refresh",
        json={"category": "supplements", "limit": 61},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "limit must be less than or equal to 60"
    assert calls == []


def test_seller_campaign_router_rejects_prepare_limit_above_configured_cap(
    monkeypatch,
) -> None:
    calls = []

    async def _prepare(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(prepared=1)

    monkeypatch.setattr(seller_api, "prepare_seller_campaign_opportunities", _prepare)

    response = _client(_Pool(), config=SellerCampaignApiConfig(max_limit=60)).post(
        "/seller/opportunities/prepare",
        json={"category": "supplements", "limit": 61},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "limit must be less than or equal to 60"
    assert calls == []


def test_seller_campaign_router_sanitizes_refresh_errors(monkeypatch) -> None:
    async def _refresh(_pool, **_kwargs):
        return _Result(
            refreshed=0,
            failed=1,
            errors=["supplements: SELECT * FROM private_table failed"],
        )

    monkeypatch.setattr(seller_api, "refresh_seller_category_intelligence", _refresh)

    response = _client(_Pool()).post(
        "/seller/intelligence/refresh",
        json={"category": "supplements"},
    )

    assert response.status_code == 200
    assert response.json()["failed"] == 1
    assert response.json()["errors"] == [
        seller_api._REFRESH_ERROR_SUMMARY,
    ]


def test_seller_campaign_router_rejects_unknown_boolean_payload(monkeypatch) -> None:
    calls = []

    async def _prepare(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(prepared=1)

    monkeypatch.setattr(seller_api, "prepare_seller_campaign_opportunities", _prepare)

    response = _client(_Pool()).post(
        "/seller/opportunities/prepare",
        json={"replace_existing": "maybe"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "replace_existing must be a boolean"
    assert calls == []


def test_seller_campaign_router_rejects_account_id_scope_mismatch(
    monkeypatch,
) -> None:
    calls = []

    async def _prepare(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(prepared=1)

    monkeypatch.setattr(seller_api, "prepare_seller_campaign_opportunities", _prepare)

    response = _client(_Pool(), scope={"account_id": "acct_1"}).post(
        "/seller/opportunities/prepare",
        json={"account_id": "acct_2"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "account_id does not match scope"
    assert calls == []


def test_seller_campaign_router_rejects_target_mode_override(monkeypatch) -> None:
    calls = []

    async def _prepare(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(prepared=1)

    monkeypatch.setattr(seller_api, "prepare_seller_campaign_opportunities", _prepare)

    response = _client(_Pool()).post(
        "/seller/opportunities/prepare",
        json={"target_mode": "vendor_retention"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "target_mode must match configured seller target mode"
    )
    assert calls == []


@pytest.mark.parametrize(
    ("payload", "scope", "status_code", "detail"),
    [
        (
            {"replace_existing": "maybe"},
            None,
            400,
            "replace_existing must be a boolean",
        ),
        (
            {"target_mode": "vendor_retention"},
            None,
            400,
            "target_mode must match configured seller target mode",
        ),
        (
            {"account_id": "acct_2"},
            {"account_id": "acct_1"},
            403,
            "account_id does not match scope",
        ),
    ],
)
def test_seller_campaign_router_combined_operation_preflights_prepare_inputs(
    monkeypatch,
    payload,
    scope,
    status_code,
    detail,
) -> None:
    refresh_calls = []
    prepare_calls = []

    async def _refresh(received_pool, **kwargs):
        refresh_calls.append((received_pool, kwargs))
        return _Result(refreshed=1)

    async def _prepare(received_pool, **kwargs):
        prepare_calls.append((received_pool, kwargs))
        return _Result(prepared=1)

    monkeypatch.setattr(seller_api, "refresh_seller_category_intelligence", _refresh)
    monkeypatch.setattr(seller_api, "prepare_seller_campaign_opportunities", _prepare)

    response = _client(_Pool(), scope=scope).post(
        "/seller/operations/refresh-and-prepare",
        json=payload,
    )

    assert response.status_code == status_code
    assert response.json()["detail"] == detail
    assert refresh_calls == []
    assert prepare_calls == []


def test_seller_campaign_router_combined_operation_resolves_dependencies_once(
    monkeypatch,
) -> None:
    pool = _Pool()
    pool_calls = 0
    scope_calls = 0

    async def _refresh(received_pool, **_kwargs):
        assert received_pool is pool
        return _Result(refreshed=1, failed=0)

    async def _prepare(received_pool, **kwargs):
        assert received_pool is pool
        assert kwargs["account_id"] == "acct_1"
        return _Result(prepared=1)

    async def pool_provider():
        nonlocal pool_calls
        pool_calls += 1
        return pool

    async def scope_provider():
        nonlocal scope_calls
        scope_calls += 1
        return {"account_id": "acct_1"}

    monkeypatch.setattr(seller_api, "refresh_seller_category_intelligence", _refresh)
    monkeypatch.setattr(seller_api, "prepare_seller_campaign_opportunities", _prepare)

    app = FastAPI()
    app.include_router(
        create_seller_campaign_router(
            pool_provider=pool_provider,
            scope_provider=scope_provider,
        )
    )

    response = TestClient(app).post(
        "/seller/operations/refresh-and-prepare",
        json={"category": "supplements"},
    )

    assert response.status_code == 200
    assert response.json()["prepare"] == {"prepared": 1}
    assert pool_calls == 1
    assert scope_calls == 1


def test_seller_campaign_router_lists_seller_drafts() -> None:
    pool = _Pool(rows=[_campaign_row()])

    response = _client(
        pool,
        scope=TenantScope(account_id="acct_1"),
    ).get("/seller/campaigns/drafts?statuses=draft,approved&limit=5")

    assert response.status_code == 200
    assert response.json()["rows"][0]["target_mode"] == "amazon_seller"
    query, args = pool.fetch_calls[0]
    assert "target_mode = $3" in query
    assert args == (["draft", "approved"], "acct_1", "amazon_seller", 5)


def test_seller_campaign_router_exports_seller_drafts_csv() -> None:
    pool = _Pool(rows=[_campaign_row()])

    response = _client(pool).get("/seller/campaigns/drafts/export")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "seller_campaign_drafts.csv" in response.headers["content-disposition"]
    assert "Acme Seller" in response.text


def test_seller_campaign_router_reviews_only_seller_drafts() -> None:
    pool = _Pool(rows=[_review_row()])

    response = _client(
        pool,
        scope={"account_id": "acct_1"},
    ).post(
        "/seller/campaigns/drafts/review",
        json={
            "campaign_ids": [CAMPAIGN_ID],
            "status": "queued",
            "from_statuses": "draft,approved",
            "from_email": "seller@example.com",
        },
    )

    assert response.status_code == 200
    query, args = pool.fetch_calls[0]
    assert "target_mode = $4" in query
    assert "campaign.target_mode = $4" in query
    assert args[0] == [CAMPAIGN_ID]
    assert args[1] == ["draft", "approved"]
    assert args[2] == "acct_1"
    assert args[3] == "amazon_seller"
    assert args[4] == "queued"


def test_seller_campaign_router_requires_database() -> None:
    response = _client(_Pool(initialized=False)).get("/seller/targets")

    assert response.status_code == 503
    assert response.json()["detail"] == "Database unavailable"


def test_seller_campaign_router_honors_host_dependencies() -> None:
    pool = _Pool(rows=[_target_row()])

    def require_auth():
        raise HTTPException(status_code=403, detail="forbidden")

    response = _client(pool, dependencies=[Depends(require_auth)]).get(
        "/seller/targets"
    )

    assert response.status_code == 403
    assert pool.fetch_calls == []


def test_seller_campaign_router_requires_fastapi(monkeypatch) -> None:
    monkeypatch.setattr(seller_api, "_FASTAPI_IMPORT_ERROR", ImportError("missing"))

    with pytest.raises(RuntimeError, match="FastAPI is required"):
        create_seller_campaign_router(pool_provider=lambda: None)
