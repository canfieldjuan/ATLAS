from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest


def test_api_router_exposes_only_tenant_b2b_paths():
    from atlas_brain.api import router

    paths = {route.path for route in router.routes}
    assert not any(path.startswith("/b2b/dashboard") for path in paths)
    assert "/b2b/tenant/reports/compare" in paths
    assert "/b2b/tenant/reports/compare-companies" in paths
    assert "/b2b/tenant/reports/company-deep-dive" in paths
    assert "/b2b/tenant/affiliates/opportunities" in paths


def test_tenant_router_covers_all_legacy_dashboard_routes():
    from fastapi.routing import APIRoute

    from atlas_brain.api.b2b_dashboard import router as legacy_router
    from atlas_brain.api.b2b_tenant_dashboard import router as tenant_router

    tenant_methods = set()
    for route in tenant_router.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in (route.methods or set()):
            if method in {"HEAD", "OPTIONS"}:
                continue
            tenant_methods.add((route.path, method))

    legacy_prefix = "/b2b/dashboard"
    tenant_prefix = "/b2b/tenant"
    missing = []
    for route in legacy_router.routes:
        if not isinstance(route, APIRoute):
            continue
        if not route.path.startswith(legacy_prefix):
            continue
        tenant_path = tenant_prefix + route.path[len(legacy_prefix):]
        for method in (route.methods or set()):
            if method in {"HEAD", "OPTIONS"}:
                continue
            if (tenant_path, method) not in tenant_methods:
                missing.append((method, route.path, tenant_path))

    assert not missing, f"Missing tenant aliases: {missing[:5]}"


def test_owner_role_gets_admin_scope_even_without_is_admin_flag(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    user = SimpleNamespace(
        account_id=str(uuid4()),
        product="b2b_retention",
        role="owner",
        is_admin=False,
    )
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    assert mod._vendor_scope_sql(1, user) == "TRUE"
    assert mod._tenant_params(user) == []


@pytest.mark.asyncio
async def test_dashboard_overview_owner_role_reads_global_vendor_count(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace(
        is_initialized=True,
        fetchval=AsyncMock(return_value=56),
        fetchrow=AsyncMock(return_value={"avg_urgency": 3.6, "total_churn_signals": 100, "total_reviews": 2000}),
        fetch=AsyncMock(return_value=[]),
    )
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention", role="owner", is_admin=False)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.dashboard_overview(user=user)

    assert result["tracked_vendors"] == 56
    fetchval_sql = pool.fetchval.await_args.args[0]
    assert "COUNT(DISTINCT vendor_name) FROM b2b_churn_signals" in fetchval_sql
    fetchrow_sql = pool.fetchrow.await_args.args[0]
    assert "WHERE TRUE" in fetchrow_sql


@pytest.mark.asyncio
async def test_dashboard_overview_member_role_reads_account_scoped_vendor_count(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace(
        is_initialized=True,
        fetchval=AsyncMock(return_value=15),
        fetchrow=AsyncMock(return_value={"avg_urgency": 4.2, "total_churn_signals": 22, "total_reviews": 330}),
        fetch=AsyncMock(return_value=[]),
    )
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention", role="member", is_admin=False)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.dashboard_overview(user=user)

    assert result["tracked_vendors"] == 15
    fetchval_sql, fetchval_acct = pool.fetchval.await_args.args
    assert "COUNT(*) FROM tracked_vendors WHERE account_id = $1" in fetchval_sql
    assert str(fetchval_acct) == user.account_id
    fetchrow_sql, fetchrow_acct = pool.fetchrow.await_args.args
    assert "vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = $1)" in fetchrow_sql
    assert str(fetchrow_acct) == user.account_id


@pytest.mark.asyncio
async def test_list_tenant_reports_excludes_stale_and_allows_global_rows(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace(is_initialized=True, fetch=AsyncMock(return_value=[]))
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.list_tenant_reports(
        report_type=None, include_stale=False, limit=10, user=user
    )

    assert result == {"reports": [], "count": 0}
    sql = pool.fetch.await_args.args[0]
    assert "vendor_filter IS NULL" in sql
    assert "account_id = $1" in sql
    assert "COALESCE((intelligence_data->>'data_stale')::boolean, false) = false" in sql


@pytest.mark.asyncio
async def test_list_leads_uses_event_recency_and_company_fallback(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(side_effect=[[{"vendor_name": "Acme"}], []]),
    )
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.list_leads(
        min_urgency=7, window_days=30, limit=20, user=user
    )

    assert result == {"leads": [], "count": 0}
    sql = pool.fetch.await_args_list[-1].args[0]
    assert "COALESCE(r.reviewed_at, r.imported_at, r.enriched_at)" in sql
    assert "ar.confidence_label IN ('high', 'medium')" in sql
    assert "ar.resolved_company_name" in sql
    assert "r.reviewer_company AS raw_reviewer_company" in sql


@pytest.mark.asyncio
async def test_list_tenant_reviews_uses_event_recency_and_company_fallback(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(side_effect=[[{"vendor_name": "Acme"}], []]),
    )
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.list_tenant_reviews(
        pain_category=None,
        min_urgency=None,
        company="Acme",
        has_churn_intent=None,
        window_days=90,
        limit=20,
        user=user,
    )

    assert result == {"reviews": [], "count": 0}
    sql = pool.fetch.await_args_list[-1].args[0]
    assert "COALESCE(r.reviewed_at, r.imported_at, r.enriched_at)" in sql
    assert "r.reviewer_company ILIKE '%' || $3 || '%'" in sql
    assert "ORDER BY (r.enrichment->>'urgency_score')::numeric DESC" in sql


@pytest.mark.asyncio
async def test_tenant_vendor_history_requires_tracked_vendor_and_reads_snapshots(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace(
        is_initialized=True,
        fetchval=AsyncMock(return_value=1),
        fetch=AsyncMock(return_value=[]),
    )
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.get_tenant_vendor_history(
        vendor_name="BigCommerce",
        days=90,
        limit=24,
        user=user,
    )

    assert result == {"vendor_name": "BigCommerce", "snapshots": [], "count": 0}
    assert "tracked_vendors" in pool.fetchval.await_args.args[0]
    assert "FROM b2b_vendor_snapshots" in pool.fetch.await_args.args[0]


@pytest.mark.asyncio
async def test_compare_tenant_vendor_periods_computes_deltas(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace(
        is_initialized=True,
        fetchval=AsyncMock(return_value=1),
        fetchrow=AsyncMock(
            side_effect=[
                {
                    "vendor_name": "BigCommerce",
                    "snapshot_date": date(2026, 2, 20),
                    "total_reviews": 120,
                    "churn_intent": 20,
                    "churn_density": 0.17,
                    "avg_urgency": 6.1,
                    "positive_review_pct": 0.42,
                    "recommend_ratio": 0.39,
                    "support_sentiment": 0.31,
                    "legacy_support_score": 0.22,
                    "new_feature_velocity": 0.41,
                    "employee_growth_rate": 0.08,
                    "top_pain": "reliability",
                    "top_competitor": "Shopify",
                    "pain_count": 44,
                    "competitor_count": 29,
                    "displacement_edge_count": 13,
                    "high_intent_company_count": 18,
                },
                {
                    "vendor_name": "BigCommerce",
                    "snapshot_date": date(2026, 3, 20),
                    "total_reviews": 140,
                    "churn_intent": 35,
                    "churn_density": 0.25,
                    "avg_urgency": 7.2,
                    "positive_review_pct": 0.35,
                    "recommend_ratio": 0.33,
                    "support_sentiment": 0.27,
                    "legacy_support_score": 0.19,
                    "new_feature_velocity": 0.37,
                    "employee_growth_rate": 0.05,
                    "top_pain": "security",
                    "top_competitor": "Shopify",
                    "pain_count": 61,
                    "competitor_count": 40,
                    "displacement_edge_count": 19,
                    "high_intent_company_count": 26,
                },
            ],
        ),
    )
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.compare_tenant_vendor_periods(
        vendor_name="BigCommerce",
        period_a_days_ago=30,
        period_b_days_ago=0,
        user=user,
    )

    assert result["vendor_name"] == "BigCommerce"
    assert result["deltas"]["avg_urgency"] == 1.1
    assert result["deltas"]["churn_intent"] == 15
    assert result["deltas"]["total_reviews"] == 20
    assert pool.fetchrow.await_count == 2


@pytest.mark.asyncio
async def test_get_vendor_target_prefers_new_report_types(monkeypatch):
    from atlas_brain.api import vendor_targets as mod

    target_id = uuid4()
    account_id = uuid4()
    target_row = {
        "id": target_id,
        "company_name": "Acme",
        "target_mode": "vendor_retention",
        "contact_name": None,
        "contact_email": None,
        "contact_role": None,
        "products_tracked": [],
        "competitors_tracked": [],
        "tier": "report",
        "status": "active",
        "notes": None,
        "account_id": account_id,
        "created_at": None,
        "updated_at": None,
    }
    stats_row = {
        "total_campaigns": 0,
        "drafts": 0,
        "sent": 0,
        "approved": 0,
        "last_campaign_at": None,
    }
    pool = SimpleNamespace(
        is_initialized=True,
        fetchrow=AsyncMock(side_effect=[target_row, stats_row]),
        fetch=AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)

    result = await mod.get_vendor_target(target_id=target_id, user=None)

    assert result["company_name"] == "Acme"
    sql = pool.fetch.await_args.args[0]
    report_types = pool.fetch.await_args.args[2]
    assert "report_type = ANY($2::text[])" in sql
    assert report_types[0] == "accounts_in_motion"
    assert "weekly_churn_feed" in report_types


@pytest.mark.asyncio
async def test_accounts_in_motion_feed_aggregates_tracked_vendor_reports(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(
            return_value=[
                {"vendor_name": "Zendesk", "track_mode": "competitor", "label": "Support", "added_at": None},
                {"vendor_name": "Intercom", "track_mode": "competitor", "label": None, "added_at": None},
            ]
        ),
    )
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention", role="member", is_admin=False)
    helper = AsyncMock(
        side_effect=[
            {
                "accounts": [
                    {
                        "company": "Acme Corp",
                        "vendor": "Zendesk",
                        "urgency": 8.5,
                        "opportunity_score": 82,
                        "pain_categories": [{"category": "pricing", "severity": ""}],
                        "evidence": ["We need to move fast."],
                    }
                ],
                "report_date": "2026-04-04",
                "stale_days": 0,
                "is_stale": False,
                "data_source": "persisted_report",
            },
            {
                "accounts": [
                    {
                        "company": "Bravo Ltd",
                        "vendor": "Intercom",
                        "urgency": 7.1,
                        "opportunity_score": 70,
                        "pain_categories": [{"category": "support", "severity": ""}],
                        "evidence": ["Support quality dropped."],
                    }
                ],
                "report_date": "2026-04-03",
                "stale_days": 1,
                "is_stale": True,
                "data_source": "persisted_report",
            },
        ]
    )
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "_list_accounts_in_motion_from_report", helper)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.list_tenant_accounts_in_motion_feed(
        min_urgency=7,
        per_vendor_limit=5,
        limit=10,
        user=user,
    )

    assert result["tracked_vendor_count"] == 2
    assert result["vendors_with_accounts"] == 2
    assert result["freshest_report_date"] == "2026-04-04"
    assert result["accounts"][0]["company"] == "Acme Corp"
    assert result["accounts"][0]["watch_vendor"] == "Zendesk"
    assert result["accounts"][0]["watchlist_label"] == "Support"
    assert result["accounts"][0]["is_stale"] is False
    assert result["accounts"][1]["company"] == "Bravo Ltd"
    helper.assert_awaited()


@pytest.mark.asyncio
async def test_accounts_in_motion_feed_handles_empty_watchlist(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(return_value=[]),
    )
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention", role="member", is_admin=False)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.list_tenant_accounts_in_motion_feed(
        min_urgency=mod.settings.b2b_churn.accounts_in_motion_min_urgency,
        per_vendor_limit=mod.settings.b2b_churn.accounts_in_motion_max_per_vendor,
        limit=mod.settings.b2b_churn.accounts_in_motion_feed_max_total,
        user=user,
    )

    assert result == {
        "accounts": [],
        "count": 0,
        "tracked_vendor_count": 0,
        "vendors_with_accounts": 0,
        "min_urgency": mod.settings.b2b_churn.accounts_in_motion_min_urgency,
        "per_vendor_limit": mod.settings.b2b_churn.accounts_in_motion_max_per_vendor,
        "freshest_report_date": None,
    }


@pytest.mark.asyncio
async def test_accounts_in_motion_feed_sorts_and_applies_total_limit(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(
            return_value=[
                {"vendor_name": "Zendesk", "track_mode": "competitor", "label": "Support", "added_at": None},
                {"vendor_name": "Intercom", "track_mode": "competitor", "label": "Chat", "added_at": None},
            ]
        ),
    )
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention", role="member", is_admin=False)
    helper = AsyncMock(
        side_effect=[
            {
                "accounts": [
                    {
                        "company": "Bravo Ltd",
                        "vendor": "Zendesk",
                        "urgency": 8.1,
                        "opportunity_score": 70,
                        "pain_categories": [{"category": "pricing", "severity": ""}],
                        "evidence": ["Budget pressure is rising."],
                    },
                    {
                        "company": "Acme Corp",
                        "vendor": "Zendesk",
                        "urgency": 9.0,
                        "opportunity_score": 85,
                        "pain_categories": [{"category": "support", "severity": ""}],
                        "evidence": ["We need a replacement this quarter."],
                    },
                ],
                "report_date": "2026-04-04",
                "stale_days": 0,
                "is_stale": False,
                "data_source": "persisted_report",
            },
            {
                "accounts": [
                    {
                        "company": "Charlie Inc",
                        "vendor": "Intercom",
                        "urgency": 9.4,
                        "opportunity_score": 90,
                        "pain_categories": [{"category": "reliability", "severity": ""}],
                        "evidence": ["We are actively evaluating alternatives."],
                    }
                ],
                "report_date": "2026-04-03",
                "stale_days": 2,
                "is_stale": True,
                "data_source": "persisted_report",
            },
        ]
    )
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "_list_accounts_in_motion_from_report", helper)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.list_tenant_accounts_in_motion_feed(
        min_urgency=7,
        per_vendor_limit=5,
        limit=2,
        user=user,
    )

    assert result["count"] == 2
    assert [row["company"] for row in result["accounts"]] == ["Acme Corp", "Bravo Ltd"]
    assert all(row["is_stale"] is False for row in result["accounts"])
