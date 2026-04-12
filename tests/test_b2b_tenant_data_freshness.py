from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from atlas_brain.services.b2b import watchlist_alerts as watchlist_alert_service
from atlas_brain.services.b2b.report_trust import report_section_evidence_payload, report_trust_payload


def test_report_trust_payload_includes_artifact_state_when_status_present():
    created_at = datetime.now(timezone.utc) - timedelta(hours=8)

    trust = report_trust_payload(
        report_date=None,
        created_at=created_at,
        data_stale=False,
        blocker_count=0,
        warning_count=1,
        unresolved_issue_count=0,
        status="sales_ready",
    )

    assert trust == {
        "artifact_state": "ready",
        "artifact_label": "Ready",
        "freshness_state": "fresh",
        "freshness_label": "Fresh",
        "review_state": "warnings",
        "review_label": "Warnings",
    }


def test_report_trust_payload_omits_artifact_state_when_status_is_missing():
    created_at = datetime.now(timezone.utc) - timedelta(hours=36)

    trust = report_trust_payload(
        report_date=None,
        created_at=created_at,
        data_stale=False,
        blocker_count=0,
        warning_count=0,
        unresolved_issue_count=1,
        status=None,
    )

    assert trust == {
        "freshness_state": "monitor",
        "freshness_label": "Monitor",
        "review_state": "open_review",
        "review_label": "Open Review",
    }


def test_report_trust_payload_treats_date_only_anchor_as_monitor():
    created_at = datetime.now(timezone.utc) - timedelta(hours=8)

    trust = report_trust_payload(
        report_date=created_at.date(),
        created_at=created_at,
        data_stale=False,
        blocker_count=0,
        warning_count=1,
        unresolved_issue_count=0,
        status="published",
    )

    assert trust == {
        "artifact_state": "ready",
        "artifact_label": "Ready",
        "freshness_state": "monitor",
        "freshness_label": "Monitor",
        "review_state": "warnings",
        "review_label": "Warnings",
    }


def test_report_section_evidence_payload_marks_witness_backed_partial_and_thin_sections():
    payload = report_section_evidence_payload({
        "key_insights": [
            {"label": "Pricing", "summary": "Pricing churn risk"},
        ],
        "key_insights_reference_ids": {
            "witness_ids": ["w1", "w2"],
        },
        "objection_handlers": {
            "summary": "Procurement objections are rising",
            "reference_ids": {"metric_ids": ["m1"]},
        },
        "recommended_plays": {
            "summary": "Lead with migration support",
        },
        "reasoning_witness_highlights": [
            {"witness_id": "top-level-only", "excerpt_text": "Global highlight"},
        ],
    })

    assert payload["key_insights"] == {
        "state": "witness_backed",
        "label": "Witness-backed",
        "detail": "2 linked witness citations",
        "witness_count": 2,
        "metric_count": 0,
    }
    assert payload["objection_handlers"] == {
        "state": "partial",
        "label": "Partial evidence",
        "detail": "Section has evidence metadata, but no linked witness citations yet.",
        "witness_count": 0,
        "metric_count": 1,
    }
    assert payload["recommended_plays"] == {
        "state": "thin",
        "label": "Thin evidence",
        "detail": "No linked witness citations for this section yet.",
        "witness_count": 0,
        "metric_count": 0,
    }


def test_api_router_exposes_only_tenant_b2b_paths():
    from atlas_brain.api import router

    paths = {route.path for route in router.routes}
    assert not any(path.startswith("/b2b/dashboard") for path in paths)
    assert "/b2b/tenant/reports/compare" in paths
    assert "/b2b/tenant/reports/compare-companies" in paths
    assert "/b2b/tenant/reports/company-deep-dive" in paths
    assert "/b2b/tenant/reports/battle-card" in paths
    assert "/b2b/tenant/reports/{report_id}/pdf" in paths
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
        fetchval=AsyncMock(),
        fetchrow=AsyncMock(
            side_effect=[
                {"total_vendors": 56, "high_urgency_count": 12, "total_signal_reviews": 2000},
                {"avg_urgency": 3.6, "total_churn_signals": 100, "total_reviews": 2000},
            ]
        ),
        fetch=AsyncMock(return_value=[]),
    )
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention", role="owner", is_admin=False)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.dashboard_overview(user=user)

    assert result["tracked_vendors"] == 56
    assert pool.fetchval.await_count == 0
    summary_sql = pool.fetchrow.await_args_list[0].args[0]
    assert "COUNT(DISTINCT sig.vendor_name) AS total_vendors" in summary_sql
    overview_sql = pool.fetchrow.await_args_list[1].args[0]
    assert "COALESCE(AVG(sig.avg_urgency_score), 0) AS avg_urgency" in overview_sql


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
    assert "sig.vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = $1::uuid)" in fetchrow_sql
    assert str(fetchrow_acct) == user.account_id


@pytest.mark.asyncio
async def test_tenant_pipeline_excludes_cross_source_duplicates(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(return_value=[{"enrichment_status": "enriched", "cnt": 5}]),
        fetchrow=AsyncMock(
            side_effect=[
                {"recent_imports_24h": 2, "last_enrichment_at": None},
                {"active_scrape_targets": 1, "last_scrape_at": None},
            ]
        ),
    )
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention", role="owner", is_admin=False)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.get_tenant_pipeline_status(user=user)

    assert result["recent_imports_24h"] == 2
    status_sql = pool.fetch.await_args.args[0]
    stats_sql = pool.fetchrow.await_args_list[0].args[0]
    assert "duplicate_of_review_id IS NULL" in status_sql
    assert "duplicate_of_review_id IS NULL" in stats_sql


@pytest.mark.asyncio
async def test_tenant_pain_trends_excludes_cross_source_duplicates(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(return_value=[]),
    )
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention", role="owner", is_admin=False)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.pain_trends(window_days=90, user=user)

    assert result == {"trends": [], "count": 0}
    sql = pool.fetch.await_args.args[0]
    assert "duplicate_of_review_id IS NULL" in sql


@pytest.mark.asyncio
async def test_tenant_competitor_displacement_excludes_cross_source_duplicates(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(return_value=[]),
    )
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention", role="owner", is_admin=False)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.competitor_displacement(limit=20, user=user)

    assert result == {"displacement": [], "count": 0}
    sql = pool.fetch.await_args.args[0]
    assert "duplicate_of_review_id IS NULL" in sql


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
async def test_list_tenant_reports_normalizes_blank_and_trimmed_text_filters(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace(is_initialized=True, fetch=AsyncMock(return_value=[]))
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.list_tenant_reports(
        report_type="  challenger_intel  ",
        vendor_filter="   ",
        include_stale=True,
        limit=10,
        user=user,
    )

    assert result == {"reports": [], "count": 0}
    sql, *params = pool.fetch.await_args.args
    assert "vendor_filter ILIKE" not in sql
    assert any(param == "challenger_brief" for param in params)

    pool.fetch.reset_mock(return_value=True)
    pool.fetch.return_value = []

    await mod.list_tenant_reports(
        report_type=None,
        vendor_filter="  Zendesk  ",
        include_stale=True,
        limit=10,
        user=user,
    )

    sql, *params = pool.fetch.await_args.args
    assert "vendor_filter ILIKE" in sql
    assert any(param == "Zendesk" for param in params)
    assert not any(param == "  Zendesk  " for param in params)


@pytest.mark.asyncio
async def test_list_tenant_reports_exposes_normalized_trust_fields(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    created_at = datetime.now(timezone.utc) - timedelta(days=9)
    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(return_value=[{
            "id": uuid4(),
            "report_date": created_at.date(),
            "report_type": "battle_card",
            "executive_summary": "Summary",
            "vendor_filter": "Zendesk",
            "category_filter": None,
            "status": "published",
            "created_at": created_at,
            "intelligence_data": {
                "account_reasoning_preview_only": True,
                "account_reasoning_preview": {
                    "disclaimer": "Early account signal only.",
                    "account_pressure_summary": "A single named account is showing early evaluation pressure.",
                    "priority_account_names": ["Concentrix", "Concentrix"],
                },
            },
            "latest_failure_step": None,
            "latest_error_code": None,
            "latest_error_summary": None,
            "data_stale": True,
            "evidence_data_as_of_date": "2026-04-08",
            "evidence_as_of_date": None,
            "evidence_report_date": None,
            "evidence_analysis_window_days": None,
            "evidence_window_days": "60",
            "evidence_fallback_window_days": None,
            "blocker_count": 1,
            "warning_count": 0,
            "unresolved_issue_count": 0,
            "quality_status": "sales_ready",
            "quality_score": 92,
            "report_subscription_id": None,
            "report_subscription_scope_type": None,
            "report_subscription_scope_key": None,
            "report_subscription_scope_label": None,
            "report_subscription_enabled": None,
        }]),
    )
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.list_tenant_reports(
        report_type=None, include_stale=True, limit=10, user=user
    )

    assert result["count"] == 1
    report = result["reports"][0]
    assert report["has_pdf_export"] is True
    assert report["report_subscription"] is None
    assert report["artifact_state"] == "ready"
    assert report["artifact_label"] == "Ready"
    assert report["freshness_state"] == "stale"
    assert report["freshness_label"] == "Stale"
    assert report["review_state"] == "blocked"
    assert report["review_label"] == "Blocked"
    assert report["as_of_date"] == "2026-04-08"
    assert report["analysis_window_days"] == 60
    assert report["account_reasoning_preview_only"] is True
    assert report["account_pressure_summary"] == (
        "A single named account is showing early evaluation pressure."
    )
    assert report["priority_account_names"] == ["Concentrix"]
    assert report["account_pressure_disclaimer"] == "Early account signal only."
    assert report["trust"] == {
        "artifact_state": "ready",
        "artifact_label": "Ready",
        "freshness_state": "stale",
        "freshness_label": "Stale",
        "review_state": "blocked",
        "review_label": "Blocked",
    }


@pytest.mark.asyncio
async def test_list_tenant_reports_applies_quality_freshness_and_review_filters(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    created_at = datetime.now(timezone.utc) - timedelta(days=10)
    matching_id = uuid4()
    nonmatching_id = uuid4()
    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(return_value=[
            {
                "id": matching_id,
                "report_date": created_at.date(),
                "report_type": "battle_card",
                "executive_summary": "Keep",
                "vendor_filter": "Zendesk",
                "category_filter": None,
                "status": "published",
                "created_at": created_at,
                "latest_failure_step": None,
                "latest_error_code": None,
                "latest_error_summary": None,
                "data_stale": True,
                "blocker_count": 1,
                "warning_count": 0,
                "unresolved_issue_count": 0,
                "quality_status": "sales_ready",
                "quality_score": 92,
                "report_subscription_id": None,
                "report_subscription_scope_type": None,
                "report_subscription_scope_key": None,
                "report_subscription_scope_label": None,
                "report_subscription_enabled": None,
            },
            {
                "id": nonmatching_id,
                "report_date": created_at.date(),
                "report_type": "battle_card",
                "executive_summary": "Drop",
                "vendor_filter": "HubSpot",
                "category_filter": None,
                "status": "published",
                "created_at": created_at,
                "latest_failure_step": None,
                "latest_error_code": None,
                "latest_error_summary": None,
                "data_stale": False,
                "blocker_count": 0,
                "warning_count": 1,
                "unresolved_issue_count": 0,
                "quality_status": "sales_ready",
                "quality_score": 90,
                "report_subscription_id": None,
                "report_subscription_scope_type": None,
                "report_subscription_scope_key": None,
                "report_subscription_scope_label": None,
                "report_subscription_enabled": None,
            },
        ]),
    )
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.list_tenant_reports(
        report_type=None,
        vendor_filter=None,
        quality_status="sales_ready",
        freshness_state="stale",
        review_state="blocked",
        include_stale=True,
        limit=10,
        user=user,
    )

    assert result["count"] == 1
    assert [report["id"] for report in result["reports"]] == [str(matching_id)]
    _, *fetch_params = pool.fetch.await_args.args
    assert fetch_params[-2] == 50
    assert fetch_params[-1] == user.account_id


@pytest.mark.asyncio
async def test_list_tenant_reports_exposes_report_subscription_summary(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    report_id = uuid4()
    subscription_id = uuid4()
    created_at = datetime.now(timezone.utc) - timedelta(hours=4)
    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(return_value=[{
            "id": report_id,
            "report_date": created_at.date(),
            "report_type": "battle_card",
            "executive_summary": "Summary",
            "vendor_filter": "Zendesk",
            "category_filter": None,
            "status": "processing",
            "created_at": created_at,
            "latest_failure_step": None,
            "latest_error_code": None,
            "latest_error_summary": None,
            "data_stale": False,
            "blocker_count": 0,
            "warning_count": 0,
            "unresolved_issue_count": 0,
            "quality_status": "sales_ready",
            "quality_score": 90,
            "report_subscription_id": subscription_id,
            "report_subscription_scope_type": "report",
            "report_subscription_scope_key": str(report_id),
            "report_subscription_scope_label": "battle card - Zendesk",
            "report_subscription_enabled": True,
        }]),
    )
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.list_tenant_reports(
        report_type=None,
        include_stale=True,
        limit=10,
        user=user,
    )

    report = result["reports"][0]
    assert report["report_subscription"] == {
        "id": str(subscription_id),
        "scope_type": "report",
        "scope_key": str(report_id),
        "scope_label": "battle card - Zendesk",
        "enabled": True,
    }


@pytest.mark.asyncio
async def test_list_tenant_reports_rejects_invalid_quality_status_before_db_touch(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: (_ for _ in ()).throw(AssertionError("db should not be touched")))
    monkeypatch.setattr(mod, "_pool_or_503", lambda: (_ for _ in ()).throw(AssertionError("db should not be touched")))
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    with pytest.raises(HTTPException) as exc:
        await mod.list_tenant_reports(
            report_type=None,
            vendor_filter=None,
            quality_status="typo_status",
            freshness_state=None,
            review_state=None,
            include_stale=True,
            limit=10,
            user=user,
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == (
        "quality_status must be sales_ready, needs_review, thin_evidence, or deterministic_fallback"
    )


@pytest.mark.asyncio
async def test_get_tenant_report_exposes_normalized_trust_fields(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    report_id = uuid4()
    created_at = datetime.now(timezone.utc) - timedelta(hours=10)
    row = {
        "id": report_id,
        "report_date": None,
        "report_type": "vendor_scorecard",
        "vendor_filter": "Zendesk",
        "category_filter": None,
        "executive_summary": "Summary",
        "intelligence_data": {
            "data_stale": False,
            "data_as_of_date": "2026-04-09",
            "window_days": 60,
            "key_insights": [{"label": "Pricing", "summary": "Pricing churn risk"}],
            "key_insights_reference_ids": {"witness_ids": ["w1"]},
            "recommended_plays": {"summary": "Lead with migration support"},
        },
        "data_density": {},
        "status": "published",
        "latest_failure_step": None,
        "latest_error_code": None,
        "latest_error_summary": None,
        "blocker_count": 0,
        "warning_count": 2,
        "llm_model": "test-model",
        "created_at": created_at,
    }
    pool = SimpleNamespace(is_initialized=True, fetchval=AsyncMock(return_value=0))
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)
    monkeypatch.setattr(mod, "_load_accessible_tenant_report", AsyncMock(return_value=row))
    monkeypatch.setattr(mod, "_fetch_report_subscription_row", AsyncMock(return_value=None))

    result = await mod.get_tenant_report(str(report_id), user=user)

    assert result["artifact_state"] == "ready"
    assert result["has_pdf_export"] is True
    assert result["as_of_date"] == "2026-04-09"
    assert result["analysis_window_days"] == 60
    assert result["report_subscription"] is None
    assert result["artifact_label"] == "Ready"
    assert result["freshness_state"] == "fresh"
    assert result["freshness_label"] == "Fresh"
    assert result["review_state"] == "warnings"
    assert result["review_label"] == "Warnings"
    assert result["section_evidence"] == {
        "key_insights": {
            "state": "witness_backed",
            "label": "Witness-backed",
            "detail": "1 linked witness citation",
            "witness_count": 1,
            "metric_count": 0,
        },
        "recommended_plays": {
            "state": "thin",
            "label": "Thin evidence",
            "detail": "No linked witness citations for this section yet.",
            "witness_count": 0,
            "metric_count": 0,
        },
    }
    assert result["trust"] == {
        "artifact_state": "ready",
        "artifact_label": "Ready",
        "freshness_state": "fresh",
        "freshness_label": "Fresh",
        "review_state": "warnings",
        "review_label": "Warnings",
    }


@pytest.mark.asyncio
async def test_get_tenant_report_trims_uuid_path_before_lookup(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    report_id = uuid4()
    row = {
        "id": report_id,
        "report_date": None,
        "report_type": "vendor_scorecard",
        "vendor_filter": "Zendesk",
        "category_filter": None,
        "executive_summary": "Summary",
        "intelligence_data": {},
        "data_density": {},
        "status": "published",
        "latest_failure_step": None,
        "latest_error_code": None,
        "latest_error_summary": None,
        "blocker_count": 0,
        "warning_count": 0,
        "llm_model": "test-model",
        "created_at": datetime.now(timezone.utc),
    }
    pool = SimpleNamespace(is_initialized=True, fetchval=AsyncMock(return_value=0))
    load_mock = AsyncMock(return_value=row)
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)
    monkeypatch.setattr(mod, "_load_accessible_tenant_report", load_mock)
    monkeypatch.setattr(mod, "_fetch_report_subscription_row", AsyncMock(return_value=None))

    await mod.get_tenant_report(f"  {report_id}  ", user=user)

    load_mock.assert_awaited_once_with(pool, report_id, user)


@pytest.mark.asyncio
async def test_get_tenant_report_exposes_report_subscription_summary(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    report_id = uuid4()
    created_at = datetime.now(timezone.utc) - timedelta(hours=6)
    row = {
        "id": report_id,
        "report_date": None,
        "report_type": "vendor_scorecard",
        "vendor_filter": "Zendesk",
        "category_filter": None,
        "executive_summary": "Summary",
        "intelligence_data": {"data_stale": False},
        "data_density": {},
        "status": "published",
        "latest_failure_step": None,
        "latest_error_code": None,
        "latest_error_summary": None,
        "blocker_count": 0,
        "warning_count": 0,
        "llm_model": "test-model",
        "created_at": created_at,
    }
    subscription_id = uuid4()
    pool = SimpleNamespace(is_initialized=True, fetchval=AsyncMock(return_value=0))
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)
    monkeypatch.setattr(mod, "_load_accessible_tenant_report", AsyncMock(return_value=row))
    monkeypatch.setattr(
        mod,
        "_fetch_report_subscription_row",
        AsyncMock(return_value={
            "id": subscription_id,
            "scope_type": "report",
            "scope_key": str(report_id),
            "scope_label": "custom report label",
            "enabled": True,
        }),
    )

    result = await mod.get_tenant_report(str(report_id), user=user)

    assert result["report_subscription"] == {
        "id": str(subscription_id),
        "scope_type": "report",
        "scope_key": str(report_id),
        "scope_label": "custom report label",
        "enabled": True,
    }


@pytest.mark.asyncio
async def test_generate_tenant_battle_card_report_reuses_existing_persisted_report(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    report_id = uuid4()
    pool = SimpleNamespace(is_initialized=True)
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention", role="member", is_admin=False)

    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)
    monkeypatch.setattr(
        mod,
        "_resolve_tracked_vendor_for_view",
        AsyncMock(return_value="Zendesk"),
    )
    monkeypatch.setattr(
        mod,
        "_fetch_latest_tenant_vendor_report",
        AsyncMock(return_value={"id": report_id}),
    )

    result = await mod.generate_tenant_battle_card_report(
        mod.BattleCardRequest(vendor_name=" Zendesk "),
        user=user,
    )

    assert result["status"] == "ready"
    assert result["report_id"] == str(report_id)
    assert result["vendor_name"] == "Zendesk"
    assert result["reused"] is True
    mod._resolve_tracked_vendor_for_view.assert_awaited_once_with(pool, UUID(user.account_id), "Zendesk")


@pytest.mark.asyncio
async def test_generate_tenant_battle_card_report_runs_scoped_task_when_missing(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    report_id = uuid4()
    pool = SimpleNamespace(is_initialized=True)
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention", role="member", is_admin=False)
    task = SimpleNamespace(metadata={"seed": "value"})
    scheduler = SimpleNamespace(run_now=AsyncMock(return_value={"status": "ok"}))
    task_repo = SimpleNamespace(get_by_name=AsyncMock(return_value=task))

    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)
    monkeypatch.setattr(
        mod,
        "_resolve_tracked_vendor_for_view",
        AsyncMock(return_value="Zendesk"),
    )
    monkeypatch.setattr(
        mod,
        "_fetch_latest_tenant_vendor_report",
        AsyncMock(side_effect=[None, {"id": report_id}]),
    )
    monkeypatch.setattr(mod, "get_scheduled_task_repo", lambda: task_repo)
    monkeypatch.setattr(mod, "get_task_scheduler", lambda: scheduler)

    result = await mod.generate_tenant_battle_card_report(
        mod.BattleCardRequest(vendor_name=" Zendesk "),
        user=user,
    )

    assert result["status"] == "ready"
    assert result["report_id"] == str(report_id)
    assert result["vendor_name"] == "Zendesk"
    assert result["reused"] is False
    task_repo.get_by_name.assert_awaited_once_with("b2b_battle_cards")
    scheduler.run_now.assert_awaited_once()
    assert task.metadata["seed"] == "value"
    assert task.metadata["scope_name"] == "Zendesk"
    assert task.metadata["scope_trigger"] == "tenant_manual_request"
    assert task.metadata["test_vendors"] == ["Zendesk"]
    mod._resolve_tracked_vendor_for_view.assert_awaited_once_with(pool, UUID(user.account_id), "Zendesk")


@pytest.mark.asyncio
async def test_tenant_report_action_routes_reject_blank_required_body_text_without_db_touch(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention", role="member", is_admin=False)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    cases = [
        (
            lambda: mod.generate_tenant_comparison_report(
                mod.VendorComparisonRequest(primary_vendor="   ", comparison_vendor="Freshdesk"),
                user=user,
            ),
            "primary_vendor is required",
        ),
        (
            lambda: mod.generate_tenant_comparison_report(
                mod.VendorComparisonRequest(primary_vendor="Zendesk", comparison_vendor="   "),
                user=user,
            ),
            "comparison_vendor is required",
        ),
        (
            lambda: mod.generate_tenant_account_comparison_report(
                mod.AccountComparisonRequest(primary_company="   ", comparison_company="Acme"),
                user=user,
            ),
            "primary_company is required",
        ),
        (
            lambda: mod.generate_tenant_account_comparison_report(
                mod.AccountComparisonRequest(primary_company="Acme", comparison_company="   "),
                user=user,
            ),
            "comparison_company is required",
        ),
        (
            lambda: mod.generate_tenant_comparison_report(
                mod.VendorComparisonRequest(primary_vendor="Zendesk", comparison_vendor=" zendesk "),
                user=user,
            ),
            "Choose two different vendors",
        ),
        (
            lambda: mod.generate_tenant_account_comparison_report(
                mod.AccountComparisonRequest(primary_company="Acme", comparison_company=" acme "),
                user=user,
            ),
            "Choose two different companies",
        ),
        (
            lambda: mod.generate_tenant_account_deep_dive_report(
                mod.AccountDeepDiveRequest(company_name="   "),
                user=user,
            ),
            "company_name is required",
        ),
        (
            lambda: mod.generate_tenant_battle_card_report(
                mod.BattleCardRequest(vendor_name="   "),
                user=user,
            ),
            "vendor_name is required",
        ),
        (
            lambda: mod.upsert_report_subscription(
                "library",
                "library",
                mod.ReportSubscriptionUpsertRequest(scope_label="   ", enabled=False),
                user=user,
            ),
            "scope_label is required",
        ),
    ]

    for call, detail in cases:
        monkeypatch.setattr(mod, "get_db_pool", lambda: (_ for _ in ()).throw(AssertionError("db should not be touched")))
        monkeypatch.setattr(mod, "_pool_or_503", lambda: (_ for _ in ()).throw(AssertionError("db should not be touched")))
        with pytest.raises(mod.HTTPException) as exc:
            await call()
        assert exc.value.status_code == 400
        assert exc.value.detail == detail


@pytest.mark.asyncio
async def test_generate_tenant_comparison_report_trims_body_vendor_names(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod
    from atlas_brain.autonomous.tasks import b2b_churn_intelligence as intelligence

    pool = SimpleNamespace(is_initialized=True, fetchval=AsyncMock(return_value=1))
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention", role="member", is_admin=False)
    generate = AsyncMock(return_value={"status": "ready", "vendor_pair": ["Zendesk", "Freshdesk"]})

    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)
    monkeypatch.setattr(intelligence, "generate_vendor_comparison_report", generate)

    result = await mod.generate_tenant_comparison_report(
        mod.VendorComparisonRequest(primary_vendor=" Zendesk ", comparison_vendor=" Freshdesk ", window_days=30, persist=False),
        user=user,
    )

    assert result == {"status": "ready", "vendor_pair": ["Zendesk", "Freshdesk"]}
    assert pool.fetchval.await_args.args[1:] == (user.account_id, "Zendesk")
    generate.assert_awaited_once_with(pool, "Zendesk", "Freshdesk", window_days=30, persist=False)


@pytest.mark.asyncio
async def test_generate_tenant_company_reports_trim_body_company_names(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod
    from atlas_brain.autonomous.tasks import b2b_churn_intelligence as intelligence

    pool = SimpleNamespace(is_initialized=True)
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention", role="member", is_admin=False)
    compare = AsyncMock(return_value={"status": "ready", "companies": ["Acme", "Beta"]})
    deep_dive = AsyncMock(return_value={"status": "ready", "company_name": "Acme"})

    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(intelligence, "generate_company_comparison_report", compare)
    monkeypatch.setattr(intelligence, "generate_company_deep_dive_report", deep_dive)

    compare_result = await mod.generate_tenant_account_comparison_report(
        mod.AccountComparisonRequest(primary_company=" Acme ", comparison_company=" Beta ", window_days=14, persist=False),
        user=user,
    )
    deep_dive_result = await mod.generate_tenant_account_deep_dive_report(
        mod.AccountDeepDiveRequest(company_name=" Acme ", window_days=21, persist=False),
        user=user,
    )

    assert compare_result == {"status": "ready", "companies": ["Acme", "Beta"]}
    assert deep_dive_result == {"status": "ready", "company_name": "Acme"}
    compare.assert_awaited_once_with(pool, "Acme", "Beta", window_days=14, persist=False)
    deep_dive.assert_awaited_once_with(pool, "Acme", window_days=21, persist=False)


@pytest.mark.asyncio
async def test_report_subscription_routes_validate_scope_and_filters_before_db_touch(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    user = SimpleNamespace(account_id=str(uuid4()), user_id=str(uuid4()), product="b2b_retention")

    cases = [
        (
            lambda: mod.get_report_subscription("   ", "library", user=user),
            "scope_type must be library, library_view, or report",
        ),
        (
            lambda: mod.get_report_subscription("report", "   ", user=user),
            "scope_key is required",
        ),
        (
            lambda: mod.get_report_subscription("report", "not-a-uuid", user=user),
            "scope_key must be a report UUID",
        ),
        (
            lambda: mod.upsert_report_subscription(
                "report",
                "not-a-uuid",
                mod.ReportSubscriptionUpsertRequest(
                    scope_label="Pipeline",
                    enabled=False,
                ),
                user=user,
            ),
            "scope_key must be a report UUID",
        ),
        (
            lambda: mod.upsert_report_subscription(
                "library_view",
                "pipeline",
                mod.ReportSubscriptionUpsertRequest(
                    scope_label="Pipeline",
                    filter_payload={"vendor_filter": "   "},
                    enabled=False,
                ),
                user=user,
            ),
            "library_view subscriptions require at least one active filter",
        ),
    ]

    for call, detail in cases:
        monkeypatch.setattr(mod, "get_db_pool", lambda: (_ for _ in ()).throw(AssertionError("db should not be touched")))
        monkeypatch.setattr(mod, "_pool_or_503", lambda: (_ for _ in ()).throw(AssertionError("db should not be touched")))
        with pytest.raises(mod.HTTPException) as exc:
            await call()
        assert exc.value.status_code == 400
        assert exc.value.detail == detail


@pytest.mark.asyncio
async def test_upsert_report_subscription_trims_scope_label_and_delivery_note(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    account_id = uuid4()
    user_id = uuid4()
    subscription_id = uuid4()
    now = datetime(2026, 4, 12, 12, 0, tzinfo=timezone.utc)
    pool = SimpleNamespace(is_initialized=True, execute=AsyncMock())
    row = {
        "id": subscription_id,
        "scope_type": "library",
        "scope_key": "library",
        "scope_label": "Custom report label",
        "filter_payload": {},
        "report_id": None,
        "delivery_frequency": "weekly",
        "deliverable_focus": "all",
        "freshness_policy": "fresh_or_monitor",
        "recipient_emails": [],
        "delivery_note": "Internal note",
        "enabled": False,
        "next_delivery_at": None,
        "last_delivery_status": None,
        "last_delivery_at": None,
        "last_delivery_freshness_state": None,
        "last_delivery_summary": None,
        "last_delivery_error": None,
        "last_delivery_report_count": 0,
        "created_at": now,
        "updated_at": now,
    }

    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "_fetch_report_subscription_schedule_row", AsyncMock(return_value=None))
    monkeypatch.setattr(mod, "_fetch_report_subscription_row", AsyncMock(return_value=row))

    result = await mod.upsert_report_subscription(
        "library",
        "library",
        mod.ReportSubscriptionUpsertRequest(scope_label=" Custom report label ", delivery_note=" Internal note ", enabled=False),
        user=SimpleNamespace(account_id=str(account_id), user_id=str(user_id), product="b2b_retention"),
    )

    execute_args = pool.execute.await_args.args
    assert execute_args[5] == "Custom report label"
    assert execute_args[11] == "Internal note"
    assert result["subscription"]["scope_label"] == "Custom report label"
    assert result["subscription"]["delivery_note"] == "Internal note"


@pytest.mark.asyncio
async def test_generate_campaigns_rejects_blank_vendor_name_without_db_touch(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_growth", role="member", is_admin=False)

    monkeypatch.setattr(mod, "get_db_pool", lambda: (_ for _ in ()).throw(AssertionError("db should not be touched")))
    monkeypatch.setattr(mod, "_pool_or_503", lambda: (_ for _ in ()).throw(AssertionError("db should not be touched")))

    with pytest.raises(mod.HTTPException) as exc:
        await mod.generate_campaigns(
            mod.GenerateCampaignRequest(vendor_name="   ", company_filter="   "),
            user=user,
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "vendor_name is required"


@pytest.mark.asyncio
async def test_generate_campaigns_trims_vendor_and_normalizes_blank_company_filter(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace(is_initialized=True, fetchval=AsyncMock(return_value=1))
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_growth", role="member", is_admin=False)
    generate = AsyncMock(return_value={"generated": 2, "campaigns": []})
    cfg = mod.settings.b2b_campaign

    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "_generate_campaigns", generate)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.generate_campaigns(
        mod.GenerateCampaignRequest(vendor_name=" Zendesk ", company_filter="   "),
        user=user,
    )

    assert result["campaigns_created"] == 2
    assert pool.fetchval.await_args.args[1:] == (UUID(user.account_id), "Zendesk")
    generate.assert_awaited_once_with(
        pool,
        vendor_filter="Zendesk",
        company_filter=None,
        target_mode=cfg.target_mode,
        min_score=cfg.min_opportunity_score,
        limit=cfg.max_campaigns_per_run,
    )


@pytest.mark.asyncio
async def test_opportunity_disposition_routes_reject_blank_required_text_without_db_touch(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")

    cases = [
        (
            lambda: mod.set_opportunity_disposition(
                mod.SetDispositionRequest(
                    opportunity_key="   ",
                    company="Acme",
                    vendor="Zendesk",
                    disposition="saved",
                ),
                user=user,
            ),
            "opportunity_key is required",
        ),
        (
            lambda: mod.set_opportunity_disposition(
                mod.SetDispositionRequest(
                    opportunity_key="opp-1",
                    company="   ",
                    vendor="Zendesk",
                    disposition="saved",
                ),
                user=user,
            ),
            "company is required",
        ),
        (
            lambda: mod.bulk_set_opportunity_dispositions(
                mod.BulkSetDispositionRequest(
                    items=[
                        mod.BulkDispositionItem(
                            opportunity_key="opp-1",
                            company="Acme",
                            vendor="   ",
                            review_id="review-1",
                        )
                    ],
                    disposition="saved",
                ),
                user=user,
            ),
            "vendor is required",
        ),
    ]

    for call, detail in cases:
        monkeypatch.setattr(mod, "get_db_pool", lambda: (_ for _ in ()).throw(AssertionError("db should not be touched")))
        monkeypatch.setattr(mod, "_pool_or_503", lambda: (_ for _ in ()).throw(AssertionError("db should not be touched")))
        with pytest.raises(mod.HTTPException) as exc:
            await call()
        assert exc.value.status_code == 400
        assert exc.value.detail == detail


@pytest.mark.asyncio
async def test_opportunity_disposition_routes_trim_fields_before_persistence(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    account_id = uuid4()
    now = datetime(2026, 4, 12, 12, 0, tzinfo=timezone.utc)

    async def fetchrow(query, *args):
        assert args[1] == UUID(str(account_id))
        assert args[2:6] == ("opp-1", "Acme", "Zendesk", "review-1")
        return {
            "id": uuid4(),
            "opportunity_key": args[2],
            "company": args[3],
            "vendor": args[4],
            "review_id": args[5],
            "disposition": args[6],
            "snoozed_until": args[7],
            "created_at": now,
            "updated_at": now,
        }

    pool = SimpleNamespace(
        is_initialized=True,
        fetchrow=AsyncMock(side_effect=fetchrow),
        execute=AsyncMock(return_value="DELETE 2"),
        fetch=AsyncMock(return_value=[]),
    )
    user = SimpleNamespace(account_id=str(account_id), product="b2b_retention")

    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)

    single = await mod.set_opportunity_disposition(
        mod.SetDispositionRequest(
            opportunity_key=" opp-1 ",
            company=" Acme ",
            vendor=" Zendesk ",
            review_id=" review-1 ",
            disposition="saved",
        ),
        user=user,
    )
    bulk = await mod.bulk_set_opportunity_dispositions(
        mod.BulkSetDispositionRequest(
            items=[
                mod.BulkDispositionItem(
                    opportunity_key=" opp-1 ",
                    company=" Acme ",
                    vendor=" Zendesk ",
                    review_id=" review-1 ",
                )
            ],
            disposition="saved",
        ),
        user=user,
    )
    removed = await mod.remove_opportunity_dispositions(
        mod.RemoveDispositionsRequest(opportunity_keys=[" opp-1 ", "   ", "opp-2"]),
        user=user,
    )

    assert single["opportunity_key"] == "opp-1"
    assert single["company"] == "Acme"
    assert single["vendor"] == "Zendesk"
    assert single["review_id"] == "review-1"
    execute_calls = pool.execute.await_args_list
    bulk_args = execute_calls[0].args
    remove_args = execute_calls[1].args
    assert bulk_args[3:7] == ("opp-1", "Acme", "Zendesk", "review-1")
    assert remove_args[1] == UUID(str(account_id))
    assert remove_args[2] == ["opp-1", "opp-2"]
    assert bulk == {"updated": 1}
    assert removed == {"removed": 2}


@pytest.mark.asyncio
async def test_list_opportunity_dispositions_rejects_invalid_filter_before_db_touch(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")

    monkeypatch.setattr(mod, "get_db_pool", lambda: (_ for _ in ()).throw(AssertionError("db should not be touched")))
    monkeypatch.setattr(mod, "_pool_or_503", lambda: (_ for _ in ()).throw(AssertionError("db should not be touched")))

    with pytest.raises(mod.HTTPException) as exc:
        await mod.list_opportunity_dispositions(disposition="invalid", user=user)

    assert exc.value.status_code == 422
    assert exc.value.detail == "disposition must be one of: dismissed, saved, snoozed"


@pytest.mark.asyncio
async def test_list_opportunity_dispositions_normalizes_blank_filter(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace(
        is_initialized=True,
        execute=AsyncMock(),
        fetch=AsyncMock(return_value=[]),
    )
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")

    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)

    result = await mod.list_opportunity_dispositions(disposition="   ", user=user)

    assert result == {"dispositions": [], "count": 0}
    assert pool.fetch.await_args.args == (
        """
            SELECT id, opportunity_key, company, vendor, review_id,
                   disposition, snoozed_until, created_at, updated_at
            FROM b2b_opportunity_dispositions
            WHERE account_id = $1
            ORDER BY updated_at DESC
            """,
        UUID(user.account_id),
    )


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
async def test_list_leads_normalizes_blank_and_trimmed_vendor_name(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod
    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod

    pool = SimpleNamespace(is_initialized=True)
    read_mock = AsyncMock(return_value=[])
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "_tenant_params", lambda _user: [])
    monkeypatch.setattr(shared_mod, "read_high_intent_companies", read_mock)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.list_leads(
        vendor_name="   ",
        min_urgency=7,
        window_days=30,
        limit=20,
        user=user,
    )

    assert result == {"leads": [], "count": 0}
    read_mock.assert_awaited_once_with(
        pool,
        min_urgency=7,
        window_days=30,
        vendor_name=None,
        scoped_vendors=None,
        limit=20,
    )

    read_mock.reset_mock()

    await mod.list_leads(
        vendor_name="  Zendesk  ",
        min_urgency=7,
        window_days=30,
        limit=20,
        user=user,
    )

    read_mock.assert_awaited_once_with(
        pool,
        min_urgency=7,
        window_days=30,
        vendor_name="Zendesk",
        scoped_vendors=None,
        limit=20,
    )


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
async def test_list_tenant_reviews_normalizes_blank_and_trimmed_text_filters(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod
    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod

    pool = SimpleNamespace(is_initialized=True)
    read_mock = AsyncMock(return_value=[])
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "_tenant_params", lambda _user: [])
    monkeypatch.setattr(shared_mod, "read_review_details", read_mock)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.list_tenant_reviews(
        pain_category="   ",
        min_urgency=6,
        company="   ",
        has_churn_intent=False,
        window_days=90,
        limit=20,
        user=user,
    )

    assert result == {"reviews": [], "count": 0}
    read_mock.assert_awaited_once_with(
        pool,
        window_days=90,
        scoped_vendors=None,
        pain_category=None,
        min_urgency=6,
        company=None,
        has_churn_intent=False,
        recency_column="coalesce",
        limit=20,
    )

    read_mock.reset_mock()

    await mod.list_tenant_reviews(
        pain_category="  pricing  ",
        min_urgency=6,
        company="  Acme  ",
        has_churn_intent=False,
        window_days=90,
        limit=20,
        user=user,
    )

    read_mock.assert_awaited_once_with(
        pool,
        window_days=90,
        scoped_vendors=None,
        pain_category="pricing",
        min_urgency=6,
        company="Acme",
        has_churn_intent=False,
        recency_column="coalesce",
        limit=20,
    )


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
        vendor_name=" BigCommerce ",
        days=90,
        limit=24,
        user=user,
    )

    assert result == {"vendor_name": "BigCommerce", "snapshots": [], "count": 0}
    assert "tracked_vendors" in pool.fetchval.await_args.args[0]
    assert pool.fetchval.await_args.args[1:] == (UUID(user.account_id), "BigCommerce")
    assert "FROM b2b_vendor_snapshots" in pool.fetch.await_args.args[0]
    assert pool.fetch.await_args.args[1:] == ("BigCommerce", 90, 24)


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
        vendor_name=" BigCommerce ",
        period_a_days_ago=30,
        period_b_days_ago=0,
        user=user,
    )

    assert result["vendor_name"] == "BigCommerce"
    assert "tracked_vendors" in pool.fetchval.await_args.args[0]
    assert pool.fetchval.await_args.args[1:] == (UUID(user.account_id), "BigCommerce")
    assert result["deltas"]["avg_urgency"] == 1.1
    assert result["deltas"]["churn_intent"] == 15
    assert result["deltas"]["total_reviews"] == 20
    assert pool.fetchrow.await_count == 2


@pytest.mark.asyncio
async def test_tenant_vendor_routes_reject_blank_vendor_names_without_db_touch(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention", role="member", is_admin=False)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    cases = [
        lambda: mod.get_tenant_vendor_history(vendor_name="   ", days=90, limit=24, user=user),
        lambda: mod.compare_tenant_vendor_periods(vendor_name="	", period_a_days_ago=30, period_b_days_ago=0, user=user),
        lambda: mod.get_vendor_detail("  ", user=user),
    ]

    for call in cases:
        monkeypatch.setattr(mod, "get_db_pool", lambda: (_ for _ in ()).throw(AssertionError("db should not be touched")))
        monkeypatch.setattr(mod, "_pool_or_503", lambda: (_ for _ in ()).throw(AssertionError("db should not be touched")))
        with pytest.raises(mod.HTTPException) as exc:
            await call()
        assert exc.value.status_code == 400
        assert exc.value.detail == "vendor_name is required"


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
async def test_list_tenant_signals_normalizes_text_filters(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod
    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod

    pool = SimpleNamespace(is_initialized=True)
    rows_mock = AsyncMock(return_value=[])
    summary_mock = AsyncMock(
        return_value={
            "total_vendors": 0,
            "high_urgency_count": 0,
            "total_signal_reviews": 0,
        }
    )
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention", role="member", is_admin=False)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "_tenant_params", lambda _user: [])
    monkeypatch.setattr(mod, "_load_reasoning_views_for_vendors", AsyncMock(return_value={}))
    monkeypatch.setattr(shared_mod, "read_vendor_signal_rows", rows_mock)
    monkeypatch.setattr(shared_mod, "read_vendor_signal_summary", summary_mock)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.list_tenant_signals(
        vendor_name="   ",
        vendor_names=[" Zendesk ", " ", "HubSpot  "],
        min_urgency=6,
        category="  CRM  ",
        limit=10,
        user=user,
    )

    rows_mock.assert_awaited_once_with(
        pool,
        vendor_name_query=None,
        vendor_names=["Zendesk", "HubSpot"],
        min_urgency=6,
        product_category="CRM",
        tracked_account_id=None,
        include_snapshot_metrics=True,
        limit=10,
    )
    summary_mock.assert_awaited_once_with(
        pool,
        vendor_name_query=None,
        vendor_names=["Zendesk", "HubSpot"],
        min_urgency=6,
        product_category="CRM",
        tracked_account_id=None,
    )
    assert result == {
        "signals": [],
        "count": 0,
        "total_vendors": 0,
        "high_urgency_count": 0,
        "total_signal_reviews": 0,
    }


@pytest.mark.asyncio
async def test_slow_burn_watchlist_normalizes_text_filters(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod
    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod

    pool = SimpleNamespace(is_initialized=True)
    read_mock = AsyncMock(return_value=[])
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention", role="member", is_admin=False)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "_tenant_params", lambda _user: [])
    monkeypatch.setattr(mod, "_load_reasoning_views_for_vendors", AsyncMock(return_value={}))
    monkeypatch.setattr(shared_mod, "read_ranked_vendor_signal_rows", read_mock)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.list_tenant_slow_burn_watchlist(
        vendor_name="  Zendesk  ",
        vendor_names=None,
        category="  CRM  ",
        vendor_alert_threshold=None,
        stale_days_threshold=None,
        limit=5,
        user=user,
    )

    read_mock.assert_awaited_once_with(
        pool,
        vendor_name_query="Zendesk",
        vendor_names=None,
        product_category="CRM",
        tracked_account_id=None,
        require_snapshot_activity=True,
        limit=5,
    )
    assert result == {
        "signals": [],
        "count": 0,
        "vendor_alert_threshold": None,
        "vendor_alert_hit_count": 0,
        "stale_days_threshold": None,
        "stale_threshold_hit_count": 0,
    }


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
    assert helper.await_count > 0


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
                    "vendor_names": ["Intercom"],
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
            side_effect=[
                [
                    {
                        "id": uuid4(),
                        "vendor_name": "Intercom",
                        "track_mode": "competitor",
                        "label": "Messaging",
                        "added_at": None,
                        "latest_snapshot_date": date(2026, 4, 7),
                        "latest_accounts_report_date": date(2026, 4, 6),
                    },
                ],
                [
                    {
                        "vendor_name": "Intercom",
                        "product_category": "CRM",
                        "total_reviews": 220,
                        "churn_intent_count": 14,
                        "avg_urgency_score": 7.4,
                        "nps_proxy": 21.5,
                        "last_computed_at": datetime(2026, 4, 7, 15, 0, tzinfo=timezone.utc),
                    },
                ],
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
    tracked_sql = pool.fetch.await_args_list[0].args[0]
    signal_sql = pool.fetch.await_args_list[1].args[0]
    assert "FROM b2b_vendor_snapshots" in tracked_sql
    assert "FROM b2b_intelligence bi" in tracked_sql
    assert "WITH ranked_signals AS" in signal_sql
    assert "PARTITION BY sig.vendor_name" in signal_sql


@pytest.mark.asyncio
async def test_search_available_vendors_reads_best_signal_rows(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(
            return_value=[
                {
                    "vendor_name": "Zendesk",
                    "product_category": "Helpdesk",
                    "total_reviews": 320,
                    "churn_intent_count": 14,
                    "avg_urgency_score": 6.8,
                    "nps_proxy": 18.5,
                    "last_computed_at": None,
                },
            ]
        ),
    )
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.search_available_vendors(q=" Zen ", limit=10, user=user)

    assert result == {
        "vendors": [
            {
                "vendor_name": "Zendesk",
                "product_category": "Helpdesk",
                "total_reviews": 320,
                "avg_urgency": 6.8,
            },
        ],
        "count": 1,
    }
    search_sql, search_query, search_limit = pool.fetch.await_args.args
    assert "WITH ranked_signals AS" in search_sql
    assert "sig.vendor_name ILIKE '%' || $1 || '%'" in search_sql
    assert search_query == "Zen"
    assert search_limit == 10

@pytest.mark.asyncio
async def test_tracked_vendor_routes_reject_blank_required_text_without_db_touch(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")
    resolver = AsyncMock(side_effect=AssertionError("resolver should not be touched"))
    monkeypatch.setattr(mod, "resolve_vendor_name", resolver)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    cases = [
        (
            lambda: mod.add_tracked_vendor(
                req=mod.AddVendorRequest(vendor_name="   ", track_mode="own"),
                user=user,
            ),
            "vendor_name is required",
        ),
        (
            lambda: mod.add_tracked_vendor(
                req=mod.AddVendorRequest(vendor_name="Zendesk", track_mode="invalid"),
                user=user,
            ),
            "track_mode must be 'own' or 'competitor'",
        ),
        (lambda: mod.remove_tracked_vendor("   ", user=user), "vendor_name is required"),
        (lambda: mod.search_available_vendors(q="   ", limit=10, user=user), "q is required"),
    ]

    for call, detail in cases:
        monkeypatch.setattr(mod, "get_db_pool", lambda: (_ for _ in ()).throw(AssertionError("db should not be touched")))
        monkeypatch.setattr(mod, "_pool_or_503", lambda: (_ for _ in ()).throw(AssertionError("db should not be touched")))
        with pytest.raises(mod.HTTPException) as exc:
            await call()
        assert exc.value.status_code == 400
        assert exc.value.detail == detail

    assert resolver.await_count == 0


@pytest.mark.asyncio
async def test_remove_tracked_vendor_trims_vendor_before_resolution(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace(
        is_initialized=True,
        fetchval=AsyncMock(return_value=1),
    )
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")
    resolver = AsyncMock(return_value="Zendesk")
    cleanup = AsyncMock(return_value={"removed_sources": ["manual"], "still_tracked": False})

    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "resolve_vendor_name", resolver)
    monkeypatch.setattr(mod, "purge_tracked_vendor_sources", cleanup)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.remove_tracked_vendor(" Zendesk ", user=user)

    resolver.assert_awaited_once_with("Zendesk")
    assert pool.fetchval.await_args.args[1:] == (UUID(user.account_id), "Zendesk")
    cleanup.assert_awaited_once_with(pool, str(UUID(user.account_id)), "Zendesk")
    assert result == {
        "status": "ok",
        "vendor_name": "Zendesk",
        "removed_sources": ["manual"],
        "still_tracked": False,
    }



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
                "vendor_names": ["Intercom"],
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
            name="  Intercom named only  ",
            vendor_names=["intercom"],
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

    assert result["name"] == "Intercom named only"
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
    assert pool.fetchrow.await_args.args[3] == "Intercom named only"
    assert pool.fetchrow.await_args.args[14] is True
    assert pool.fetchrow.await_args.args[15] == "weekly"
    assert pool.fetchrow.await_args.args[16] is not None


@pytest.mark.asyncio
async def test_create_watchlist_view_falls_back_to_legacy_vendor_name_when_vendor_names_blank(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    view_id = uuid4()
    account_id = uuid4()
    pool = SimpleNamespace(
        is_initialized=True,
        fetchval=AsyncMock(side_effect=[None, "Intercom"]),
        fetchrow=AsyncMock(
            return_value={
                "id": view_id,
                "name": "Legacy vendor fallback",
                "vendor_names": ["Intercom"],
                "category": None,
                "source": None,
                "min_urgency": None,
                "include_stale": True,
                "named_accounts_only": False,
                "changed_wedges_only": False,
                "vendor_alert_threshold": None,
                "account_alert_threshold": None,
                "stale_days_threshold": None,
                "alert_email_enabled": False,
                "alert_delivery_frequency": "daily",
                "next_alert_delivery_at": None,
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
            name="Legacy vendor fallback",
            vendor_name="  Intercom  ",
            vendor_names=["   ", ""],
        ),
        user=user,
    )

    assert result["vendor_names"] == ["Intercom"]
    assert pool.fetchval.await_args_list[1].args[2] == "Intercom"
    assert pool.fetchrow.await_args.args[4] == ["Intercom"]


@pytest.mark.asyncio
async def test_create_watchlist_view_rejects_invalid_alert_frequency_without_db_touch(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace(is_initialized=True, fetchval=AsyncMock(), fetchrow=AsyncMock())
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    with pytest.raises(mod.HTTPException) as exc:
        await mod.create_watchlist_view(
            req=mod.WatchlistViewRequest(name="My alerts", alert_delivery_frequency="monthly"),
            user=user,
        )

    assert exc.value.status_code == 422
    assert exc.value.detail == "alert_delivery_frequency must be one of: daily, weekly"
    assert pool.fetchval.await_count == 0
    assert pool.fetchrow.await_count == 0


@pytest.mark.asyncio
async def test_create_watchlist_view_rejects_blank_name_without_db_touch(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace(
        is_initialized=True,
        fetchval=AsyncMock(),
        fetchrow=AsyncMock(),
    )
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    with pytest.raises(mod.HTTPException) as exc:
        await mod.create_watchlist_view(
            req=mod.WatchlistViewRequest(name="   "),
            user=user,
        )

    assert exc.value.status_code == 422
    assert exc.value.detail == "Saved view name is required"
    assert pool.fetchval.await_count == 0
    assert pool.fetchrow.await_count == 0


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
                        "vendor_names": [],
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
                    "vendor_names": [],
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
        req=mod.WatchlistViewRequest(name="  Changed wedges only  ", changed_wedges_only=True),
        user=user,
    )
    deleted = await mod.delete_watchlist_view(view_id=view_id, user=user)

    assert updated["name"] == "Changed wedges only"
    assert updated["changed_wedges_only"] is True
    assert updated["vendor_alert_threshold"] == 6.5
    assert updated["account_alert_threshold"] == 7.5
    assert updated["stale_days_threshold"] == 5
    assert updated["alert_email_enabled"] is True
    assert updated["alert_delivery_frequency"] == "weekly"
    assert updated["next_alert_delivery_at"] == "2026-04-12 09:00:00+00:00"
    assert "UPDATE b2b_watchlist_views" in pool.fetchrow.await_args_list[1].args[0]
    assert pool.fetchrow.await_args_list[1].args[3] == "Changed wedges only"
    assert pool.fetchrow.await_args_list[1].args[15] == "weekly"
    assert pool.fetchrow.await_args_list[1].args[16] == existing_next_delivery_at
    assert deleted == {"deleted": True, "watchlist_view_id": str(view_id)}
    assert "DELETE FROM b2b_watchlist_views" in pool.fetchrow.await_args_list[-1].args[0]


@pytest.mark.asyncio
async def test_update_watchlist_view_rejects_blank_name_without_duplicate_lookup(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    view_id = uuid4()
    account_id = uuid4()
    pool = SimpleNamespace(
        is_initialized=True,
        fetchrow=AsyncMock(
            return_value={
                "id": view_id,
                "name": "Changed wedges only",
                "vendor_names": [],
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
                "next_alert_delivery_at": datetime(2026, 4, 12, 9, 0, tzinfo=timezone.utc),
            }
        ),
        fetchval=AsyncMock(),
    )
    user = SimpleNamespace(account_id=str(account_id), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    with pytest.raises(mod.HTTPException) as exc:
        await mod.update_watchlist_view(
            view_id=view_id,
            req=mod.WatchlistViewRequest(name="   "),
            user=user,
        )

    assert exc.value.status_code == 422
    assert exc.value.detail == "Saved view name is required"
    assert pool.fetchrow.await_count == 0
    assert pool.fetchval.await_count == 0


@pytest.mark.asyncio
async def test_update_watchlist_view_rejects_invalid_alert_frequency_without_db_touch(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    view_id = uuid4()
    account_id = uuid4()
    pool = SimpleNamespace(is_initialized=True, fetchrow=AsyncMock(), fetchval=AsyncMock())
    user = SimpleNamespace(account_id=str(account_id), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    with pytest.raises(mod.HTTPException) as exc:
        await mod.update_watchlist_view(
            view_id=view_id,
            req=mod.WatchlistViewRequest(name="Changed wedges only", alert_delivery_frequency="monthly"),
            user=user,
        )

    assert exc.value.status_code == 422
    assert exc.value.detail == "alert_delivery_frequency must be one of: daily, weekly"
    assert pool.fetchrow.await_count == 0
    assert pool.fetchval.await_count == 0


@pytest.mark.asyncio
async def test_update_watchlist_view_falls_back_to_legacy_vendor_name_when_vendor_names_blank(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    view_id = uuid4()
    account_id = uuid4()
    pool = SimpleNamespace(
        is_initialized=True,
        fetchrow=AsyncMock(
            side_effect=[
                {
                    "id": view_id,
                    "name": "Changed wedges only",
                    "vendor_names": ["Zendesk"],
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
                    "next_alert_delivery_at": datetime(2026, 4, 12, 9, 0, tzinfo=timezone.utc),
                },
                {
                    "id": view_id,
                    "name": "Changed wedges only",
                    "vendor_names": ["Intercom"],
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
                    "next_alert_delivery_at": datetime(2026, 4, 12, 9, 0, tzinfo=timezone.utc),
                    "last_alert_delivery_at": None,
                    "last_alert_delivery_status": None,
                    "last_alert_delivery_summary": None,
                    "created_at": None,
                    "updated_at": None,
                },
            ]
        ),
        fetchval=AsyncMock(side_effect=[None, "Intercom"]),
    )
    user = SimpleNamespace(account_id=str(account_id), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.update_watchlist_view(
        view_id=view_id,
        req=mod.WatchlistViewRequest(
            name="Changed wedges only",
            vendor_name="  Intercom  ",
            vendor_names=["   "],
        ),
        user=user,
    )

    assert result["vendor_names"] == ["Intercom"]
    assert pool.fetchval.await_args_list[1].args[2] == "Intercom"
    assert pool.fetchrow.await_args_list[1].args[4] == ["Intercom"]


@pytest.mark.asyncio
async def test_get_lead_detail_rejects_blank_company_without_db_touch(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod
    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod

    pool = SimpleNamespace(is_initialized=True, fetch=AsyncMock())
    read_mock = AsyncMock()
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(shared_mod, "read_review_details", read_mock)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    with pytest.raises(mod.HTTPException) as exc:
        await mod.get_lead_detail("   ", user=user)

    assert exc.value.status_code == 400
    assert exc.value.detail == "company is required"
    assert pool.fetch.await_count == 0
    assert read_mock.await_count == 0


@pytest.mark.asyncio
async def test_get_lead_detail_trims_company_before_reader_call(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod
    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod

    account_id = uuid4()
    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(return_value=[]),
    )
    read_mock = AsyncMock(
        return_value=[
            {
                "id": "r1",
                "vendor_name": "Salesforce",
                "product_category": "CRM",
                "rating": 4.0,
                "urgency_score": 8.5,
                "pain_category": "support",
                "intent_to_leave": True,
                "decision_maker": "yes",
                "role_level": "director",
                "buying_stage": "active_evaluation",
                "competitors_mentioned": ["HubSpot"],
                "enriched_at": datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc),
            }
        ]
    )
    user = SimpleNamespace(account_id=str(account_id), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(shared_mod, "read_review_details", read_mock)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.get_lead_detail("  Acme Corp  ", user=user)

    assert read_mock.await_args.kwargs["company"] == "Acme Corp"
    assert result["company"] == "Acme Corp"
    assert result["count"] == 1


@pytest.mark.asyncio
async def test_get_tenant_review_trims_uuid_path_before_lookup(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    review_id = uuid4()
    pool = SimpleNamespace(
        is_initialized=True,
        fetchrow=AsyncMock(
            return_value={
                "id": review_id,
                "source": "g2",
                "source_url": None,
                "vendor_name": "Zendesk",
                "product_name": None,
                "product_category": "CRM",
                "rating": 4,
                "summary": None,
                "review_text": None,
                "pros": None,
                "cons": None,
                "reviewer_name": None,
                "reviewer_title": None,
                "reviewer_company": None,
                "company_size_raw": None,
                "reviewer_industry": None,
                "reviewed_at": None,
                "enrichment": {},
                "enrichment_status": "enriched",
                "enriched_at": None,
            }
        ),
        fetchval=AsyncMock(return_value=1),
    )
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention", role="member", is_admin=False)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    await mod.get_tenant_review(f"  {review_id}  ", user=user)

    assert pool.fetchrow.await_args.args[1] == review_id


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
                    "event_type": "account_alert",
                    "threshold_field": "account_alert_threshold",
                    "entity_type": "account",
                    "entity_key": "account_alert:account:salesforce:acme corp:crm:reddit:2026-04-05",
                    "vendor_name": "Salesforce",
                    "company_name": "Acme Corp",
                    "category": "CRM",
                    "source": "reddit",
                    "threshold_value": 8.5,
                    "summary": "Acme Corp crossed the account alert threshold at 8.2",
                    "payload": {
                        "urgency": 8.2,
                        "reasoning_reference_ids": {"witness_ids": ["w1"]},
                        "source_review_ids": ["r1", "r2"],
                        "account_review_focus": {
                            "vendor": "Salesforce",
                            "company": "Acme Corp",
                            "report_date": "2026-04-05",
                            "watch_vendor": "Salesforce",
                            "category": "CRM",
                            "track_mode": "competitor",
                        },
                    },
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

    result = await mod.list_watchlist_alert_events(view_id=view_id, status=" open ", limit=25, user=user)

    assert result["watchlist_view_id"] == str(view_id)
    assert result["watchlist_view_name"] == "Hot CRM alerts"
    assert result["count"] == 1
    assert result["events"][0]["event_type"] == "account_alert"
    assert result["events"][0]["source_review_ids"] == ["r1", "r2"]
    assert result["events"][0]["reasoning_reference_ids"] == {"witness_ids": ["w1"]}
    assert result["events"][0]["account_review_focus"] == {
        "vendor": "Salesforce",
        "company": "Acme Corp",
        "report_date": "2026-04-05",
        "watch_vendor": "Salesforce",
        "category": "CRM",
        "track_mode": "competitor",
    }
    sql = pool.fetch.await_args.args[0]
    assert "FROM b2b_watchlist_alert_events" in sql
    assert "status = $3" in sql


@pytest.mark.asyncio
async def test_list_tenant_campaigns_normalizes_blank_and_trimmed_status(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace(is_initialized=True, fetch=AsyncMock(return_value=[]))
    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_growth", role="member", is_admin=False)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.list_tenant_campaigns(status="   ", limit=20, user=user)

    assert result == {"campaigns": [], "count": 0}
    sql, *params = pool.fetch.await_args.args
    assert "bc.status =" not in sql

    pool.fetch.reset_mock(return_value=True)
    pool.fetch.return_value = []

    await mod.list_tenant_campaigns(status="  approved  ", limit=20, user=user)

    sql, *params = pool.fetch.await_args.args
    assert "bc.status =" in sql
    assert any(param == "approved" for param in params)
    assert not any(param == "  approved  " for param in params)


@pytest.mark.asyncio
async def test_list_watchlist_alert_events_rejects_invalid_status_before_db_touch(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    user = SimpleNamespace(account_id=str(uuid4()), product="b2b_retention")

    monkeypatch.setattr(mod, "get_db_pool", lambda: (_ for _ in ()).throw(AssertionError("db should not be touched")))
    monkeypatch.setattr(mod, "_pool_or_503", lambda: (_ for _ in ()).throw(AssertionError("db should not be touched")))

    with pytest.raises(mod.HTTPException) as exc:
        await mod.list_watchlist_alert_events(view_id=uuid4(), status="invalid", limit=25, user=user)

    assert exc.value.status_code == 422
    assert exc.value.detail == "status must be one of: open, resolved, all"


@pytest.mark.asyncio
async def test_list_watchlist_alert_events_normalizes_blank_status_to_open(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    account_id = uuid4()
    view_id = uuid4()
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
        fetch=AsyncMock(return_value=[]),
    )
    user = SimpleNamespace(account_id=str(account_id), product="b2b_retention")
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.list_watchlist_alert_events(view_id=view_id, status="   ", limit=25, user=user)

    assert result == {"watchlist_view_id": str(view_id), "watchlist_view_name": "Hot CRM alerts", "status": "open", "events": [], "count": 0}
    fetch_args = pool.fetch.await_args.args
    assert "AND status = $3" in fetch_args[0]
    assert fetch_args[1:] == (account_id, view_id, "open", 25)


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
                    "payload": {"avg_urgency_score": 8.2, "reasoning_reference_ids": {"witness_ids": ["vw1"]}},
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
                    "payload": {
                        "urgency": 9.0,
                        "reasoning_reference_ids": {"witness_ids": ["w1"]},
                        "source_review_ids": ["r1"],
                        "account_review_focus": {
                            "vendor": "Salesforce",
                            "company": "Acme Corp",
                            "report_date": "2026-04-05",
                            "watch_vendor": "Salesforce",
                            "category": "CRM",
                            "track_mode": "competitor",
                        },
                    },
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
                        "watch_vendor": "Salesforce",
                        "track_mode": "competitor",
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
    assert result["events"][0]["reasoning_reference_ids"] == {"witness_ids": ["vw1"]}
    assert result["events"][1]["source_review_ids"] == ["r1"]
    assert result["events"][1]["account_review_focus"] == {
        "vendor": "Salesforce",
        "company": "Acme Corp",
        "report_date": "2026-04-05",
        "watch_vendor": "Salesforce",
        "category": "CRM",
        "track_mode": "competitor",
    }
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
    monkeypatch.setattr(
        watchlist_alert_service,
        "resolve_watchlist_alert_recipients",
        AsyncMock(return_value=[]),
    )

    with pytest.raises(mod.HTTPException) as exc_info:
        await mod.deliver_watchlist_alert_email(
            view_id=view_id,
            body=mod.WatchlistAlertEmailRequest(evaluate_before_send=False),
            user=user,
        )

    assert exc_info.value.status_code == 422
    assert "No active owner email" in exc_info.value.detail
    statements = [call.args[0] for call in pool.execute.await_args_list]
    assert any("INSERT INTO b2b_watchlist_alert_email_log" in sql for sql in statements)
    assert any("UPDATE b2b_watchlist_views" in sql for sql in statements)
    update_call = next(call for call in pool.execute.await_args_list if "UPDATE b2b_watchlist_views" in call.args[0])
    assert update_call.args[3] == "failed"
    assert update_call.args[4] == "Watchlist alert email delivery failed before send"


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
    monkeypatch.setattr(watchlist_alert_service, "is_suppressed", AsyncMock(return_value=None))
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
    assert sender.send.await_count > 0
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
    statements = [call.args[0] for call in pool.execute.await_args_list]
    assert any("INSERT INTO b2b_watchlist_alert_email_log" in sql for sql in statements)
    assert any("UPDATE b2b_watchlist_views" in sql for sql in statements)
    update_call = next(call for call in pool.execute.await_args_list if "UPDATE b2b_watchlist_views" in call.args[0])
    assert update_call.args[3] == "no_events"
    assert update_call.args[4] == "No open alert events to deliver"


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
async def test_accounts_in_motion_feed_normalizes_blank_and_trimmed_text_filters(monkeypatch):
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
                    "category": "Helpdesk",
                    "urgency": 8.5,
                    "opportunity_score": 82,
                    "pain_categories": [{"category": "pricing", "severity": ""}],
                    "evidence": ["We need to move fast."],
                    "source_distribution": {"reddit": 2},
                    "source_reviews": [{"id": "review-1", "source": "reddit"}],
                },
            ],
            "report_date": "2026-04-04",
            "stale_days": 0,
            "is_stale": False,
            "data_source": "persisted_report",
        }
    )
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "_list_accounts_in_motion_from_report", helper)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)

    result = await mod.list_tenant_accounts_in_motion_feed(
        vendor_name="   ",
        vendor_names=[" Zendesk ", "  "],
        category="  Helpdesk  ",
        source="  reddit  ",
        min_urgency=7,
        include_stale=False,
        per_vendor_limit=5,
        limit=10,
        user=user,
    )

    assert result["count"] == 1
    assert result["vendors_with_accounts"] == 1
    assert result["accounts"][0]["company"] == "Acme Corp"
    assert result["accounts"][0]["watch_vendor"] == "Zendesk"
    assert helper.await_count == 1


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
    assert helper.await_count > 0


@pytest.mark.asyncio
async def test_get_vendor_detail_interpolates_canonical_review_predicate(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod
    from atlas_brain.autonomous.tasks import _b2b_shared as shared

    account_id = uuid4()
    captured_fetchrow_queries: list[str] = []
    captured_fetch_queries: list[str] = []

    async def fetchval(query, *args):
        assert "SELECT 1 FROM tracked_vendors" in query
        assert args == (account_id, "Salesforce")
        return 1

    async def fetchrow(query, *args):
        captured_fetchrow_queries.append(query)
        if "COUNT(*) AS total_reviews" in query:
            return {"total_reviews": 12, "enriched": 9}
        return None

    async def fetch(query, *args):
        captured_fetch_queries.append(query)
        if "GROUP BY enrichment->>'pain_category'" in query:
            return [{"pain": "support", "cnt": 4}]
        return []

    pool = SimpleNamespace(
        is_initialized=True,
        fetchval=AsyncMock(side_effect=fetchval),
        fetchrow=AsyncMock(side_effect=fetchrow),
        fetch=AsyncMock(side_effect=fetch),
    )
    user = SimpleNamespace(account_id=str(account_id), product="b2b_retention", role="member", is_admin=False)

    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", True, raising=False)
    monkeypatch.setattr(shared, "read_vendor_signal_detail", AsyncMock(return_value=None))
    monkeypatch.setattr(
        shared,
        "read_high_intent_companies",
        AsyncMock(
            return_value=[
                {
                    "company": "Acme Corp",
                    "urgency": 8.7,
                    "pain": "support",
                    "title": "VP IT",
                    "company_size": "201-500",
                    "industry": "SaaS",
                }
            ]
        ),
    )

    result = await mod.get_vendor_detail(" Salesforce ", user=user)

    assert result["vendor_name"] == "Salesforce"
    assert result["churn_signal"] is None
    assert result["review_counts"] == {"total": 12, "enriched": 9}
    assert result["pain_distribution"] == [{"pain_category": "support", "count": 4}]
    assert result["high_intent_companies"][0]["company"] == "Acme Corp"

    counts_query = next(query for query in captured_fetchrow_queries if "COUNT(*) AS total_reviews" in query)
    pain_query = next(query for query in captured_fetch_queries if "GROUP BY enrichment->>'pain_category'" in query)
    assert "{" not in counts_query and "}" not in counts_query
    assert "{" not in pain_query and "}" not in pain_query
