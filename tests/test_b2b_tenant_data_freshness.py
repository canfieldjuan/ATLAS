from datetime import date, datetime, timezone
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
async def test_slow_burn_watchlist_applies_threshold_flags(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(
            return_value=[
                {
                    "vendor_name": "Zendesk",
                    "product_category": "Helpdesk",
                    "total_reviews": 300,
                    "churn_intent_count": 22,
                    "avg_urgency_score": 8.2,
                    "avg_rating_normalized": 4.1,
                    "nps_proxy": 18.2,
                    "price_complaint_rate": 0.3,
                    "decision_maker_churn_rate": 0.2,
                    "support_sentiment": 2.4,
                    "legacy_support_score": 1.8,
                    "new_feature_velocity": 0.4,
                    "employee_growth_rate": 3.2,
                    "last_computed_at": datetime(2026, 4, 5, 15, 0, tzinfo=timezone.utc),
                },
            ]
        ),
    )
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention", role="member", is_admin=False)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "_load_reasoning_views_for_vendors", AsyncMock(return_value={}))
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.list_tenant_slow_burn_watchlist(
        vendor_alert_threshold=8,
        stale_days_threshold=1,
        user=user,
    )

    assert result["count"] == 1
    assert result["vendor_alert_hit_count"] == 1
    assert result["stale_threshold_hit_count"] == 1
    assert result["signals"][0]["vendor_alert_hit"] is True
    assert result["signals"][0]["stale_threshold_hit"] is True


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
                        "evidence_count": 2,
                        "source_review_ids": ["review-1"],
                        "source_reviews": [
                            {
                                "id": "review-1",
                                "source": "reddit",
                                "source_url": "https://reddit.example/review-1",
                                "vendor_name": "Zendesk",
                                "rating": 2.0,
                                "summary": "Support is slipping",
                                "review_excerpt": "We need to move fast.",
                                "reviewer_name": "Taylor",
                                "reviewer_title": "VP Support",
                                "reviewer_company": "Acme Corp",
                                "reviewed_at": "2026-04-03T00:00:00",
                            }
                        ],
                        "reasoning_reference_ids": {"witness_ids": ["witness:zendesk:1"]},
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
    assert result["accounts"][0]["evidence_count"] == 2
    assert result["accounts"][0]["source_reviews"][0]["id"] == "review-1"
    assert result["accounts"][0]["reasoning_reference_ids"]["witness_ids"] == ["witness:zendesk:1"]
    assert result["accounts"][1]["company"] == "Bravo Ltd"
    helper.assert_awaited()


@pytest.mark.asyncio
async def test_accounts_in_motion_feed_applies_threshold_flags(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(
            return_value=[
                {"vendor_name": "Zendesk", "track_mode": "competitor", "label": "Support", "added_at": None},
            ]
        ),
    )
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention", role="member", is_admin=False)
    helper = AsyncMock(
        return_value={
            "accounts": [
                {
                    "company": "Acme Corp",
                    "vendor": "Zendesk",
                    "urgency": 8.6,
                    "opportunity_score": 82,
                    "pain_categories": [{"category": "pricing", "severity": ""}],
                    "evidence": ["We need to move fast."],
                }
            ],
            "report_date": "2026-04-04",
            "stale_days": 2,
            "is_stale": True,
            "data_source": "persisted_report",
            "freshness_timestamp": "2026-04-04",
        }
    )
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "_list_accounts_in_motion_from_report", helper)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.list_tenant_accounts_in_motion_feed(
        account_alert_threshold=8,
        stale_days_threshold=1,
        user=user,
    )

    assert result["count"] == 1
    assert result["account_alert_hit_count"] == 1
    assert result["stale_threshold_hit_count"] == 1
    assert result["accounts"][0]["account_alert_hit"] is True
    assert result["accounts"][0]["stale_threshold_hit"] is True


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
        "account_alert_threshold": None,
        "account_alert_hit_count": 0,
        "stale_days_threshold": None,
        "stale_threshold_hit_count": 0,
        "per_vendor_limit": mod.settings.b2b_churn.accounts_in_motion_max_per_vendor,
        "freshest_report_date": None,
    }


@pytest.mark.asyncio
async def test_watchlist_views_list_returns_account_scoped_rows(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    view_id = uuid4()
    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(
            return_value=[
                {
                    "id": view_id,
                    "name": "Fresh named accounts",
                    "vendor_name": "Intercom",
                    "category": "Helpdesk",
                    "source": "reddit",
                    "min_urgency": 8.0,
                    "include_stale": False,
                    "named_accounts_only": True,
                    "changed_wedges_only": True,
                    "vendor_alert_threshold": 7.5,
                    "account_alert_threshold": 8.5,
                    "stale_days_threshold": 3,
                    "alert_email_enabled": True,
                    "alert_delivery_frequency": "weekly",
                    "next_alert_delivery_at": datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc),
                    "last_alert_delivery_at": datetime(2026, 4, 3, 12, 0, tzinfo=timezone.utc),
                    "last_alert_delivery_status": "sent",
                    "last_alert_delivery_summary": "1 alert delivered",
                    "created_at": None,
                    "updated_at": None,
                }
            ]
        ),
    )
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.list_watchlist_views(user=user)

    assert result["count"] == 1
    assert result["views"][0]["id"] == str(view_id)
    assert result["views"][0]["named_accounts_only"] is True
    assert result["views"][0]["changed_wedges_only"] is True
    assert result["views"][0]["vendor_alert_threshold"] == 7.5
    assert result["views"][0]["account_alert_threshold"] == 8.5
    assert result["views"][0]["stale_days_threshold"] == 3
    assert result["views"][0]["alert_email_enabled"] is True
    assert result["views"][0]["alert_delivery_frequency"] == "weekly"
    assert result["views"][0]["next_alert_delivery_at"] == "2026-04-10 12:00:00+00:00"
    assert result["views"][0]["last_alert_delivery_status"] == "sent"
    assert "is_default" not in result["views"][0]
    assert "FROM b2b_watchlist_views" in pool.fetch.await_args.args[0]


@pytest.mark.asyncio
async def test_list_tracked_vendors_includes_backend_freshness_fields(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(
            return_value=[
                {
                    "id": uuid4(),
                    "vendor_name": "Intercom",
                    "track_mode": "competitor",
                    "label": "Messaging",
                    "added_at": None,
                    "avg_urgency_score": 7.4,
                    "churn_intent_count": 14,
                    "total_reviews": 220,
                    "nps_proxy": 21.5,
                    "last_computed_at": datetime(2026, 4, 7, 15, 0, tzinfo=timezone.utc),
                    "latest_snapshot_date": date(2026, 4, 7),
                    "latest_accounts_report_date": date(2026, 4, 6),
                },
            ]
        ),
    )
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "_load_reasoning_views_for_vendors", AsyncMock(return_value={}))
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.list_tracked_vendors(user=user)

    assert result["count"] == 1
    assert result["vendors"][0]["latest_snapshot_date"] == "2026-04-07"
    assert result["vendors"][0]["latest_accounts_report_date"] == "2026-04-06"
    assert result["vendors"][0]["freshness_status"] == "synthesis_pending"
    assert result["vendors"][0]["freshness_timestamp"] == "2026-04-07 15:00:00+00:00"
    sql = pool.fetch.await_args.args[0]
    assert "FROM b2b_vendor_snapshots" in sql
    assert "FROM b2b_intelligence bi" in sql


@pytest.mark.asyncio
async def test_create_watchlist_view_persists_filters_and_validates_vendor(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    view_id = uuid4()
    account_id = uuid4()
    pool = SimpleNamespace(
        is_initialized=True,
        fetchval=AsyncMock(side_effect=[None, "Intercom"]),
        fetchrow=AsyncMock(
            return_value={
                "id": view_id,
                "name": "Intercom named only",
                "vendor_name": "Intercom",
                "category": "Helpdesk",
                "source": "reddit",
                "min_urgency": 8.0,
                "include_stale": False,
                "named_accounts_only": True,
                "changed_wedges_only": True,
                "vendor_alert_threshold": 7.0,
                "account_alert_threshold": 8.0,
                "stale_days_threshold": 2,
                "alert_email_enabled": True,
                "alert_delivery_frequency": "weekly",
                "next_alert_delivery_at": datetime(2026, 4, 14, 12, 0, tzinfo=timezone.utc),
                "last_alert_delivery_at": None,
                "last_alert_delivery_status": None,
                "last_alert_delivery_summary": None,
                "created_at": None,
                "updated_at": None,
            }
        ),
    )
    user = SimpleNamespace(account_id=str(account_id), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.create_watchlist_view(
        req=mod.WatchlistViewRequest(
            name="Intercom named only",
            vendor_name="intercom",
            category="Helpdesk",
            source="reddit",
            min_urgency=8,
            include_stale=False,
            named_accounts_only=True,
            changed_wedges_only=True,
            vendor_alert_threshold=7,
            account_alert_threshold=8,
            stale_days_threshold=2,
            alert_email_enabled=True,
            alert_delivery_frequency="weekly",
        ),
        user=user,
    )

    assert result["vendor_name"] == "Intercom"
    assert result["include_stale"] is False
    assert result["named_accounts_only"] is True
    assert result["changed_wedges_only"] is True
    assert result["vendor_alert_threshold"] == 7.0
    assert result["account_alert_threshold"] == 8.0
    assert result["stale_days_threshold"] == 2
    assert result["alert_email_enabled"] is True
    assert result["alert_delivery_frequency"] == "weekly"
    assert result["next_alert_delivery_at"] == "2026-04-14 12:00:00+00:00"
    vendor_lookup_sql = pool.fetchval.await_args_list[1].args[0]
    assert "FROM tracked_vendors" in vendor_lookup_sql
    insert_sql = pool.fetchrow.await_args.args[0]
    assert "INSERT INTO b2b_watchlist_views" in insert_sql
    assert "alert_email_enabled" in insert_sql
    assert pool.fetchrow.await_args.args[14] is True
    assert pool.fetchrow.await_args.args[15] == "weekly"
    assert pool.fetchrow.await_args.args[16] is not None


@pytest.mark.asyncio
async def test_update_and_delete_watchlist_view_are_account_scoped(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    view_id = uuid4()
    account_id = uuid4()
    existing_next_delivery_at = datetime(2026, 4, 12, 9, 0, tzinfo=timezone.utc)
    pool = SimpleNamespace(
        is_initialized=True,
        fetchrow=AsyncMock(
                side_effect=[
                    {
                        "id": view_id,
                        "name": "Changed wedges only",
                        "vendor_name": None,
                        "category": None,
                        "source": None,
                        "min_urgency": None,
                        "include_stale": True,
                        "named_accounts_only": False,
                        "changed_wedges_only": True,
                        "vendor_alert_threshold": 6.5,
                        "account_alert_threshold": 7.5,
                        "stale_days_threshold": 5,
                        "alert_email_enabled": True,
                        "alert_delivery_frequency": "weekly",
                        "next_alert_delivery_at": existing_next_delivery_at,
                    },
                {
                    "id": view_id,
                    "name": "Changed wedges only",
                    "vendor_name": None,
                    "category": None,
                    "source": None,
                    "min_urgency": None,
                    "include_stale": True,
                    "named_accounts_only": False,
                    "changed_wedges_only": True,
                    "vendor_alert_threshold": 6.5,
                    "account_alert_threshold": 7.5,
                    "stale_days_threshold": 5,
                    "alert_email_enabled": True,
                    "alert_delivery_frequency": "weekly",
                    "next_alert_delivery_at": existing_next_delivery_at,
                    "last_alert_delivery_at": None,
                    "last_alert_delivery_status": None,
                    "last_alert_delivery_summary": None,
                    "created_at": None,
                    "updated_at": None,
                },
                {"id": view_id},
            ]
        ),
        fetchval=AsyncMock(return_value=None),
    )
    user = SimpleNamespace(account_id=str(account_id), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    updated = await mod.update_watchlist_view(
        view_id=view_id,
        req=mod.WatchlistViewRequest(name="Changed wedges only", changed_wedges_only=True),
        user=user,
    )
    deleted = await mod.delete_watchlist_view(view_id=view_id, user=user)

    assert updated["changed_wedges_only"] is True
    assert updated["vendor_alert_threshold"] == 6.5
    assert updated["account_alert_threshold"] == 7.5
    assert updated["stale_days_threshold"] == 5
    assert updated["alert_email_enabled"] is True
    assert updated["alert_delivery_frequency"] == "weekly"
    assert updated["next_alert_delivery_at"] == "2026-04-12 09:00:00+00:00"
    assert "UPDATE b2b_watchlist_views" in pool.fetchrow.await_args_list[1].args[0]
    assert pool.fetchrow.await_args_list[1].args[15] == "weekly"
    assert pool.fetchrow.await_args_list[1].args[16] == existing_next_delivery_at
    assert deleted == {"deleted": True, "watchlist_view_id": str(view_id)}
    assert "DELETE FROM b2b_watchlist_views" in pool.fetchrow.await_args_list[-1].args[0]


@pytest.mark.asyncio
async def test_list_watchlist_alert_events_returns_view_scoped_rows(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    account_id = uuid4()
    view_id = uuid4()
    event_id = uuid4()
    pool = SimpleNamespace(
        is_initialized=True,
        fetchrow=AsyncMock(
            return_value={
                "id": view_id,
                "account_id": account_id,
                "name": "Hot CRM alerts",
                "vendor_name": "Salesforce",
                "category": "CRM",
                "source": None,
                "min_urgency": 8.0,
                "include_stale": False,
                "named_accounts_only": True,
                "changed_wedges_only": False,
                "vendor_alert_threshold": 7.5,
                "account_alert_threshold": 8.5,
                "stale_days_threshold": 2,
                "created_at": None,
                "updated_at": None,
            }
        ),
        fetch=AsyncMock(
            return_value=[
                {
                    "id": event_id,
                    "watchlist_view_id": view_id,
                    "event_type": "vendor_alert",
                    "threshold_field": "vendor_alert_threshold",
                    "entity_type": "vendor",
                    "entity_key": "vendor_alert:vendor:salesforce",
                    "vendor_name": "Salesforce",
                    "company_name": None,
                    "category": "CRM",
                    "source": None,
                    "threshold_value": 7.5,
                    "summary": "Salesforce crossed the vendor alert threshold at 8.2",
                    "payload": {"avg_urgency_score": 8.2},
                    "status": "open",
                    "first_seen_at": None,
                    "last_seen_at": None,
                    "resolved_at": None,
                    "created_at": None,
                    "updated_at": None,
                },
            ]
        ),
    )
    user = SimpleNamespace(account_id=str(account_id), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.list_watchlist_alert_events(view_id=view_id, status="open", limit=25, user=user)

    assert result["watchlist_view_id"] == str(view_id)
    assert result["watchlist_view_name"] == "Hot CRM alerts"
    assert result["count"] == 1
    assert result["events"][0]["event_type"] == "vendor_alert"
    sql = pool.fetch.await_args.args[0]
    assert "FROM b2b_watchlist_alert_events" in sql
    assert "status = $3" in sql


@pytest.mark.asyncio
async def test_evaluate_watchlist_alert_events_persists_and_resolves(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    account_id = uuid4()
    view_id = uuid4()
    existing_vendor_event_id = uuid4()
    resolved_event_id = uuid4()
    persisted_account_event_id = uuid4()
    view_row = {
        "id": view_id,
        "account_id": account_id,
        "name": "High urgency CRM",
        "vendor_name": "Salesforce",
        "category": "CRM",
        "source": "reddit",
        "min_urgency": 8.0,
        "include_stale": False,
        "named_accounts_only": True,
        "changed_wedges_only": False,
        "vendor_alert_threshold": 7.5,
        "account_alert_threshold": 8.5,
        "stale_days_threshold": 2,
        "created_at": None,
        "updated_at": None,
    }
    pool = SimpleNamespace(
        is_initialized=True,
        fetchrow=AsyncMock(
            side_effect=[
                view_row,
                {
                    "id": existing_vendor_event_id,
                    "watchlist_view_id": view_id,
                    "event_type": "vendor_alert",
                    "threshold_field": "vendor_alert_threshold",
                    "entity_type": "vendor",
                    "entity_key": "vendor_alert:vendor:salesforce",
                    "vendor_name": "Salesforce",
                    "company_name": None,
                    "category": "CRM",
                    "source": None,
                    "threshold_value": 7.5,
                    "summary": "Salesforce crossed the vendor alert threshold at 8.2",
                    "payload": {"avg_urgency_score": 8.2},
                    "status": "open",
                    "first_seen_at": None,
                    "last_seen_at": None,
                    "resolved_at": None,
                    "created_at": None,
                    "updated_at": None,
                },
                {
                    "id": persisted_account_event_id,
                    "watchlist_view_id": view_id,
                    "event_type": "account_alert",
                    "threshold_field": "account_alert_threshold",
                    "entity_type": "account",
                    "entity_key": "account_alert:account:salesforce:acme corp:crm:reddit:2026-04-05",
                    "vendor_name": "Salesforce",
                    "company_name": "Acme Corp",
                    "category": "CRM",
                    "source": "reddit",
                    "threshold_value": 8.5,
                    "summary": "Acme Corp crossed the account alert threshold at 9.0",
                    "payload": {"urgency": 9.0},
                    "status": "open",
                    "first_seen_at": None,
                    "last_seen_at": None,
                    "resolved_at": None,
                    "created_at": None,
                    "updated_at": None,
                },
            ]
        ),
        fetch=AsyncMock(
            return_value=[
                {
                    "id": existing_vendor_event_id,
                    "event_type": "vendor_alert",
                    "entity_key": "vendor_alert:vendor:salesforce",
                },
                {
                    "id": resolved_event_id,
                    "event_type": "stale_data",
                    "entity_key": "stale_data:vendor:salesforce",
                },
            ]
        ),
        execute=AsyncMock(),
    )
    user = SimpleNamespace(account_id=str(account_id), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)
    monkeypatch.setattr(
        mod,
        "list_tenant_slow_burn_watchlist",
        AsyncMock(
            return_value={
                "signals": [
                    {
                        "vendor_name": "Salesforce",
                        "product_category": "CRM",
                        "avg_urgency_score": 8.2,
                        "vendor_alert_hit": True,
                        "stale_threshold_hit": False,
                        "freshness_status": "fresh",
                        "freshness_timestamp": "2026-04-07T12:00:00Z",
                    }
                ]
            }
        ),
    )
    monkeypatch.setattr(
        mod,
        "list_tenant_accounts_in_motion_feed",
        AsyncMock(
            return_value={
                "accounts": [
                    {
                        "company": "Acme Corp",
                        "vendor": "Salesforce",
                        "category": "CRM",
                        "urgency": 9.0,
                        "account_alert_hit": True,
                        "stale_threshold_hit": False,
                        "report_date": "2026-04-05",
                        "source_distribution": {"reddit": 2},
                        "reasoning_reference_ids": {"witness_ids": ["w1"]},
                        "source_review_ids": ["r1"],
                    }
                ]
            }
        ),
    )

    result = await mod.evaluate_watchlist_alert_events(view_id=view_id, user=user)

    assert result["watchlist_view_id"] == str(view_id)
    assert result["count"] == 2
    assert result["new_open_event_count"] == 1
    assert result["resolved_event_count"] == 1
    assert pool.fetchrow.await_count == 3
    resolve_sql = pool.execute.await_args.args[0]
    assert "UPDATE b2b_watchlist_alert_events" in resolve_sql
    assert resolved_event_id in pool.execute.await_args.args[1]


@pytest.mark.asyncio
async def test_list_watchlist_alert_email_log_returns_view_scoped_rows(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    account_id = uuid4()
    view_id = uuid4()
    log_id = uuid4()
    pool = SimpleNamespace(
        is_initialized=True,
        fetchrow=AsyncMock(
            return_value={
                "id": view_id,
                "account_id": account_id,
                "name": "CRM pressure",
            }
        ),
        fetch=AsyncMock(
            return_value=[
                {
                    "id": log_id,
                    "recipient_emails": ["owner@example.com"],
                    "message_ids": ["msg-1"],
                    "event_count": 2,
                    "status": "sent",
                    "summary": "Delivered watchlist alert email to 1 of 1 recipient",
                    "error": None,
                    "delivered_at": datetime(2026, 4, 7, 18, 0, tzinfo=timezone.utc),
                    "created_at": datetime(2026, 4, 7, 18, 0, tzinfo=timezone.utc),
                    "updated_at": datetime(2026, 4, 7, 18, 0, tzinfo=timezone.utc),
                },
            ]
        ),
    )
    user = SimpleNamespace(account_id=str(account_id), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.list_watchlist_alert_email_log(view_id=view_id, limit=5, user=user)

    assert result["watchlist_view_id"] == str(view_id)
    assert result["watchlist_view_name"] == "CRM pressure"
    assert result["count"] == 1
    assert result["deliveries"][0]["status"] == "sent"
    assert result["deliveries"][0]["recipient_emails"] == ["owner@example.com"]


@pytest.mark.asyncio
async def test_deliver_watchlist_alert_email_logs_pre_send_failure(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    account_id = uuid4()
    view_id = uuid4()
    pool = SimpleNamespace(
        is_initialized=True,
        fetchrow=AsyncMock(
            return_value={
                "id": view_id,
                "account_id": account_id,
                "name": "CRM pressure",
            }
        ),
        execute=AsyncMock(),
        fetchval=AsyncMock(return_value="Acme"),
    )
    user = SimpleNamespace(account_id=str(account_id), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)
    monkeypatch.setattr(
        mod,
        "list_watchlist_alert_events",
        AsyncMock(
            return_value={
                "watchlist_view_id": str(view_id),
                "watchlist_view_name": "CRM pressure",
                "events": [
                    {
                        "id": str(uuid4()),
                        "summary": "Salesforce crossed the threshold",
                    }
                ],
            }
        ),
    )
    monkeypatch.setattr(mod, "_resolve_watchlist_alert_recipients", AsyncMock(return_value=[]))

    with pytest.raises(mod.HTTPException) as exc_info:
        await mod.deliver_watchlist_alert_email(
            view_id=view_id,
            body=mod.WatchlistAlertEmailRequest(evaluate_before_send=False),
            user=user,
        )

    assert exc_info.value.status_code == 422
    assert "No active owner email" in exc_info.value.detail
    insert_sql = pool.execute.await_args.args[0]
    assert "INSERT INTO b2b_watchlist_alert_email_log" in insert_sql
    assert pool.execute.await_args.args[8] == "failed"


@pytest.mark.asyncio
async def test_deliver_watchlist_alert_email_sends_to_owner_and_logs(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    account_id = uuid4()
    view_id = uuid4()
    event_id = uuid4()
    view_row = {
        "id": view_id,
        "account_id": account_id,
        "name": "CRM pressure",
        "vendor_name": "Salesforce",
        "category": "CRM",
        "source": "reddit",
        "min_urgency": 8.0,
        "include_stale": False,
        "named_accounts_only": True,
        "changed_wedges_only": False,
        "vendor_alert_threshold": 7.5,
        "account_alert_threshold": 8.5,
        "stale_days_threshold": 2,
        "created_at": None,
        "updated_at": None,
    }
    pool = SimpleNamespace(
        is_initialized=True,
        fetchrow=AsyncMock(return_value=view_row),
        fetch=AsyncMock(return_value=[
            {
                "email": "owner@example.com",
                "full_name": "Owner User",
            },
        ]),
        fetchval=AsyncMock(return_value="Acme Account"),
        execute=AsyncMock(),
    )
    sender = SimpleNamespace(send=AsyncMock(return_value={"id": "msg-1"}))
    user = SimpleNamespace(account_id=str(account_id), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)
    monkeypatch.setattr(mod.settings.b2b_alert, "email_enabled", True, raising=False)
    monkeypatch.setattr(mod.settings.campaign_sequence, "sender_type", "resend", raising=False)
    monkeypatch.setattr(mod.settings.campaign_sequence, "resend_api_key", "key", raising=False)
    monkeypatch.setattr(mod.settings.campaign_sequence, "resend_from_email", "alerts@example.com", raising=False)
    monkeypatch.setattr(mod.settings.b2b_alert, "dashboard_base_url", "https://churnsignals.co/watchlists", raising=False)
    monkeypatch.setattr(mod, "get_campaign_sender", lambda: sender)
    monkeypatch.setattr(mod, "is_suppressed", AsyncMock(return_value=None))
    monkeypatch.setattr(
        mod,
        "evaluate_watchlist_alert_events",
        AsyncMock(
            return_value={
                "watchlist_view_id": str(view_id),
                "watchlist_view_name": "CRM pressure",
                "events": [
                    {
                        "id": str(event_id),
                        "event_type": "vendor_alert",
                        "threshold_field": "vendor_alert_threshold",
                        "entity_type": "vendor",
                        "entity_key": "vendor_alert:vendor:salesforce",
                        "vendor_name": "Salesforce",
                        "company_name": None,
                        "category": "CRM",
                        "source": None,
                        "threshold_value": 7.5,
                        "summary": "Salesforce crossed the vendor alert threshold at 8.2",
                        "payload": {},
                        "status": "open",
                    },
                ],
            }
        ),
    )

    result = await mod.deliver_watchlist_alert_email(
        view_id=view_id,
        body=mod.WatchlistAlertEmailRequest(evaluate_before_send=True),
        user=user,
    )

    assert result["status"] == "sent"
    assert result["recipient_emails"] == ["owner@example.com"]
    assert result["message_ids"] == ["msg-1"]
    sender.send.assert_awaited()
    assert any(
        "INSERT INTO b2b_watchlist_alert_email_log" in call.args[0]
        for call in pool.execute.await_args_list
    )


@pytest.mark.asyncio
async def test_deliver_watchlist_alert_email_records_no_events(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    account_id = uuid4()
    view_id = uuid4()
    view_row = {
        "id": view_id,
        "account_id": account_id,
        "name": "Quiet view",
        "vendor_name": None,
        "category": None,
        "source": None,
        "min_urgency": None,
        "include_stale": True,
        "named_accounts_only": False,
        "changed_wedges_only": False,
        "vendor_alert_threshold": None,
        "account_alert_threshold": None,
        "stale_days_threshold": None,
        "created_at": None,
        "updated_at": None,
    }
    pool = SimpleNamespace(
        is_initialized=True,
        fetchrow=AsyncMock(return_value=view_row),
        fetch=AsyncMock(return_value=[]),
        fetchval=AsyncMock(return_value="Acme Account"),
        execute=AsyncMock(),
    )
    user = SimpleNamespace(account_id=str(account_id), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)
    monkeypatch.setattr(
        mod,
        "evaluate_watchlist_alert_events",
        AsyncMock(
            return_value={
                "watchlist_view_id": str(view_id),
                "watchlist_view_name": "Quiet view",
                "events": [],
            }
        ),
    )

    result = await mod.deliver_watchlist_alert_email(
        view_id=view_id,
        body=mod.WatchlistAlertEmailRequest(evaluate_before_send=True),
        user=user,
    )

    assert result["status"] == "no_events"
    assert result["event_count"] == 0
    assert "INSERT INTO b2b_watchlist_alert_email_log" in pool.execute.await_args.args[0]


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


@pytest.mark.asyncio
async def test_accounts_in_motion_feed_filters_by_vendor_category_source_and_stale(monkeypatch):
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
                        "company": "Acme Corp",
                        "vendor": "Zendesk",
                        "category": "Helpdesk",
                        "urgency": 8.5,
                        "opportunity_score": 82,
                        "pain_categories": [{"category": "pricing", "severity": ""}],
                        "evidence": ["We need to move fast."],
                        "source_distribution": {"reddit": 2},
                        "source_reviews": [{"id": "review-1", "source": "reddit"}],
                    },
                    {
                        "company": "Bravo Ltd",
                        "vendor": "Zendesk",
                        "category": "CRM",
                        "urgency": 7.2,
                        "opportunity_score": 61,
                        "pain_categories": [{"category": "support", "severity": ""}],
                        "evidence": ["Not a helpdesk issue."],
                        "source_distribution": {"g2": 1},
                        "source_reviews": [{"id": "review-2", "source": "g2"}],
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
                        "category": "Messaging",
                        "urgency": 9.4,
                        "opportunity_score": 90,
                        "pain_categories": [{"category": "reliability", "severity": ""}],
                        "evidence": ["Still stale."],
                        "source_distribution": {"reddit": 1},
                        "source_reviews": [{"id": "review-3", "source": "reddit"}],
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
        vendor_name="Zendesk",
        category="Helpdesk",
        source="reddit",
        min_urgency=7,
        include_stale=False,
        per_vendor_limit=5,
        limit=10,
        user=user,
    )

    assert result["count"] == 1
    assert result["vendors_with_accounts"] == 1
    assert result["freshest_report_date"] == "2026-04-04"
    assert result["accounts"][0]["company"] == "Acme Corp"
    assert result["accounts"][0]["watch_vendor"] == "Zendesk"
    helper.assert_awaited()
