from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from atlas_brain.auth.dependencies import AuthUser
from atlas_brain.services.b2b_competitive_sets import (
    build_competitive_set_plan,
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
