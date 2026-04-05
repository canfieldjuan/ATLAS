from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from atlas_brain.auth.dependencies import AuthUser
from atlas_brain.services.b2b_competitive_sets import (
    build_competitive_set_plan,
    estimate_competitive_set_plan,
    plan_to_synthesis_metadata,
)
from atlas_brain.storage.models import CompetitiveSet


def _competitive_set(**overrides) -> CompetitiveSet:
    payload = {
        "id": uuid4(),
        "account_id": uuid4(),
        "name": "Salesforce Core Competitors",
        "focal_vendor_name": "Salesforce",
        "competitor_vendor_names": ["HubSpot", "Microsoft Dynamics", "HubSpot"],
        "active": True,
        "refresh_mode": "scheduled",
        "refresh_interval_hours": 24,
        "vendor_synthesis_enabled": True,
        "pairwise_enabled": True,
        "category_council_enabled": True,
        "asymmetry_enabled": True,
    }
    payload.update(overrides)
    return CompetitiveSet(**payload)


def test_build_competitive_set_plan_scopes_to_focal_and_named_competitors():
    plan = build_competitive_set_plan(
        _competitive_set(),
        category_by_vendor={
            "salesforce": "CRM",
            "hubspot": "CRM",
            "microsoft dynamics": "CRM",
        },
    )

    assert plan.vendor_names == ["Salesforce", "HubSpot", "Microsoft Dynamics"]
    assert plan.pairwise_pairs == [
        ["Salesforce", "HubSpot"],
        ["Salesforce", "Microsoft Dynamics"],
    ]
    assert plan.category_names == ["CRM"]
    assert plan.asymmetry_pairs == [
        ["Salesforce", "HubSpot"],
        ["Salesforce", "Microsoft Dynamics"],
    ]


def test_build_competitive_set_plan_disables_optional_jobs_when_toggles_off():
    plan = build_competitive_set_plan(
        _competitive_set(
            pairwise_enabled=False,
            category_council_enabled=False,
            asymmetry_enabled=False,
        ),
        category_by_vendor={
            "salesforce": "CRM",
            "hubspot": "CRM",
            "microsoft dynamics": "CRM",
        },
    )

    assert plan.pairwise_pairs == []
    assert plan.category_names == []
    assert plan.asymmetry_pairs == []


def test_plan_to_synthesis_metadata_emits_explicit_scope_contract():
    plan = build_competitive_set_plan(
        _competitive_set(),
        category_by_vendor={
            "salesforce": "CRM",
            "hubspot": "CRM",
            "microsoft dynamics": "CRM",
        },
    )

    metadata = plan_to_synthesis_metadata(plan)

    assert metadata["scope_type"] == "competitive_set"
    assert metadata["scope_id"] == str(plan.competitive_set_id)
    assert metadata["scope_vendor_names"] == ["Salesforce", "HubSpot", "Microsoft Dynamics"]
    assert metadata["scope_pairwise_pairs"] == [
        ["Salesforce", "HubSpot"],
        ["Salesforce", "Microsoft Dynamics"],
    ]
    assert metadata["scope_category_names"] == ["CRM"]


@pytest.mark.asyncio
async def test_create_competitive_set_trims_name_before_duplicate_lookup(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    repo = SimpleNamespace(
        get_by_name_for_account=AsyncMock(return_value=_competitive_set(name="Sales Ops")),
    )
    user = AuthUser(
        user_id=str(uuid4()),
        account_id=str(uuid4()),
        plan="b2b_pro",
        plan_status="active",
        role="owner",
        product="b2b_retention",
        is_admin=True,
    )
    monkeypatch.setattr(mod, "get_competitive_set_repo", lambda: repo)
    monkeypatch.setattr(mod, "_pool_or_503", lambda: object())

    req = mod.CompetitiveSetRequest(
        name="  Sales Ops  ",
        focal_vendor_name="Salesforce",
        competitor_vendor_names=["HubSpot"],
    )

    with pytest.raises(mod.HTTPException) as exc:
        await mod.create_competitive_set(req, user=user)

    assert exc.value.status_code == 409
    assert repo.get_by_name_for_account.await_args.args[1] == "Sales Ops"


@pytest.mark.asyncio
async def test_list_competitive_sets_returns_backend_defaults(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    repo = SimpleNamespace(
        list_for_account=AsyncMock(return_value=[_competitive_set()]),
    )
    user = AuthUser(
        user_id=str(uuid4()),
        account_id=str(uuid4()),
        plan="b2b_pro",
        plan_status="active",
        role="owner",
        product="b2b_retention",
        is_admin=True,
    )
    monkeypatch.setattr(mod, "get_competitive_set_repo", lambda: repo)
    monkeypatch.setattr(
        mod.settings.b2b_churn,
        "competitive_set_refresh_interval_seconds",
        7200,
        raising=False,
    )
    monkeypatch.setattr(
        mod.settings.b2b_churn,
        "competitive_set_max_competitors",
        12,
        raising=False,
    )

    result = await mod.list_competitive_sets(include_inactive=False, user=user)

    assert result["count"] == 1
    assert result["defaults"] == {
        "default_refresh_interval_hours": 2,
        "max_competitors": 12,
    }


def test_competitive_scope_run_id_prefers_explicit_scope_run_id():
    from atlas_brain.autonomous.tasks import b2b_reasoning_synthesis as mod

    task = SimpleNamespace(
        metadata={
            "_execution_id": str(uuid4()),
            "run_id": "scope-run-123",
            "scope_type": "competitive_set",
            "scope_id": str(uuid4()),
        }
    )

    scope_meta = mod._competitive_scope_metadata(task)

    assert mod._competitive_scope_run_id(task, scope_meta) == "scope-run-123"


@pytest.mark.asyncio
async def test_estimate_competitive_set_plan_uses_history_and_fallback(monkeypatch):
    plan = build_competitive_set_plan(
        _competitive_set(),
        category_by_vendor={
            "salesforce": "CRM",
            "hubspot": "CRM",
            "microsoft dynamics": "CRM",
        },
    )

    class FakePool:
        async def fetch(self, query, *args):
            if "FROM b2b_reasoning_synthesis" in query:
                return [
                    {"vendor_name": "Salesforce", "tokens_used": 1000},
                    {"vendor_name": "HubSpot", "tokens_used": 1200},
                ]
            if "FROM llm_usage" in query:
                return [
                    {
                        "span_name": "task.b2b_reasoning_synthesis",
                        "avg_total_tokens": 900.0,
                        "avg_cost_usd": 0.09,
                        "sample_count": 20,
                    },
                    {
                        "span_name": "task.b2b_reasoning_synthesis.cross_vendor",
                        "avg_total_tokens": 200.0,
                        "avg_cost_usd": 0.02,
                        "sample_count": 10,
                    },
                ]
            if "FROM b2b_cross_vendor_reasoning_synthesis" in query:
                return [
                    {"analysis_type": "pairwise_battle", "avg_tokens_used": 250.0, "sample_count": 8},
                    {"analysis_type": "category_council", "avg_tokens_used": 400.0, "sample_count": 3},
                ]
            raise AssertionError(f"Unexpected query: {query}")

    monkeypatch.setattr(
        "atlas_brain.services.b2b_competitive_sets.settings.b2b_churn.competitive_set_preview_lookback_days",
        14,
        raising=False,
    )

    estimate = await estimate_competitive_set_plan(FakePool(), plan)

    assert estimate["estimated_vendor_tokens"] == 3100
    assert estimate["estimated_cross_vendor_tokens"] == 1300
    assert estimate["estimated_total_tokens"] == 4400
    assert estimate["estimated_total_cost_usd"] == 0.44
    assert estimate["vendor_jobs_with_history"] == 2
    assert estimate["vendor_jobs_using_fallback"] == 1
    assert estimate["cross_vendor_jobs_with_history"] == 3
    assert estimate["cross_vendor_jobs_using_fallback"] == 2


@pytest.mark.asyncio
async def test_preview_competitive_set_plan_returns_recent_runs(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    competitive_set = _competitive_set()
    repo = SimpleNamespace(
        get_by_id_for_account=AsyncMock(return_value=competitive_set),
        list_runs_for_account_set=AsyncMock(return_value=[
            SimpleNamespace(to_dict=lambda: {
                "id": str(uuid4()),
                "competitive_set_id": str(competitive_set.id),
                "account_id": str(competitive_set.account_id),
                "run_id": "scope-run-1",
                "trigger": "manual",
                "status": "succeeded",
                "execution_id": None,
                "summary": {"vendors_reasoned": 2, "vendors_skipped_hash_reuse": 1, "total_tokens": 12345},
                "started_at": "2026-04-05T12:00:00+00:00",
                "completed_at": "2026-04-05T12:01:00+00:00",
                "created_at": "2026-04-05T12:00:00+00:00",
            }),
        ]),
    )
    user = AuthUser(
        user_id=str(uuid4()),
        account_id=str(competitive_set.account_id),
        plan="b2b_pro",
        plan_status="active",
        role="owner",
        product="b2b_retention",
        is_admin=True,
    )
    monkeypatch.setattr(mod, "_pool_or_503", lambda: object())
    monkeypatch.setattr(mod, "get_competitive_set_repo", lambda: repo)
    monkeypatch.setattr(
        mod,
        "_competitive_set_plan_payload",
        AsyncMock(return_value={"estimated_total_jobs": 2, "estimate": {"estimated_total_cost_usd": 0.12}}),
    )

    result = await mod.preview_competitive_set_plan(competitive_set.id, user=user)

    assert result["competitive_set"]["id"] == str(competitive_set.id)
    assert result["plan"]["estimated_total_jobs"] == 2
    assert len(result["recent_runs"]) == 1
    assert result["recent_runs"][0]["summary"]["total_tokens"] == 12345
