from uuid import uuid4

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
