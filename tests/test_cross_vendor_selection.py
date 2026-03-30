import pytest

from atlas_brain.reasoning.cross_vendor_selection import (
    select_asymmetry_pairs,
    select_battles,
    select_categories,
)


@pytest.mark.asyncio
async def test_select_battles_filters_pairs_without_shared_context():
    edges = [
        {
            "from_vendor": "Acme Analytics",
            "to_vendor": "Bravo BI",
            "signal_strength": "strong",
            "velocity_7d": 2.0,
            "mention_count": 15,
        },
        {
            "from_vendor": "Zendesk",
            "to_vendor": "Freshdesk",
            "signal_strength": "strong",
            "velocity_7d": 1.0,
            "mention_count": 12,
        },
    ]
    evidence = {
        "Acme Analytics": {"product_category": "Analytics", "top_use_cases": ["dashboards"]},
        "Bravo BI": {"product_category": "BI", "top_use_cases": ["financial reporting"]},
        "Zendesk": {
            "product_category": "Help Desk",
            "top_use_cases": ["ticketing", "support"],
            "buyer_authority": {"role_types": {"support": 4}},
            "competitors": [{"name": "Freshdesk", "mentions": 8}],
        },
        "Freshdesk": {
            "product_category": "Help Desk",
            "top_use_cases": ["ticketing"],
            "buyer_authority": {"role_types": {"support": 3}},
            "competitors": [{"name": "Zendesk", "mentions": 7}],
        },
    }
    profiles = {
        "Acme Analytics": {"typical_company_size": ["enterprise"]},
        "Bravo BI": {"typical_company_size": ["startup"]},
        "Zendesk": {"typical_company_size": ["enterprise"], "primary_use_cases": ["ticketing"]},
        "Freshdesk": {"typical_company_size": ["enterprise"], "primary_use_cases": ["ticketing"]},
    }

    result = await select_battles(
        None,
        edges,
        evidence,
        product_profiles=profiles,
        max_battles=5,
        min_context_score=2.0,
    )

    assert len(result) == 1
    assert result[0][0] == "Zendesk"
    assert result[0][1] == "Freshdesk"


def test_select_categories_requires_reasoned_and_context_rich_vendors():
    ecosystem = {
        "CRM": {"vendor_count": 10, "displacement_intensity": 2.2},
        "Analytics": {"vendor_count": 8, "displacement_intensity": 2.4},
    }
    reasoning_lookup = {
        "HubSpot": {},
        "Salesforce": {},
        "Pipedrive": {},
        "Looker": {},
        "Tableau": {},
        "Power BI": {},
    }
    evidence = {
        "HubSpot": {"product_category": "CRM", "top_use_cases": ["lead management"]},
        "Salesforce": {"product_category": "CRM", "buyer_authority": {"role_types": {"sales": 5}}},
        "Pipedrive": {"product_category": "CRM", "competitors": [{"name": "HubSpot"}]},
        "Looker": {"product_category": "Analytics"},
        "Tableau": {"product_category": "Analytics"},
        "Power BI": {"product_category": "Analytics"},
    }

    result = select_categories(
        ecosystem,
        reasoning_lookup,
        evidence,
        min_vendors=3,
        min_context_vendors=3,
        min_displacement_intensity=1.0,
        max_categories=5,
    )

    assert [category for category, _eco in result] == ["CRM"]


def test_select_categories_uses_vendor_membership_not_archetype_payload():
    ecosystem = {
        "CRM": {"vendor_count": 10, "displacement_intensity": 2.2},
    }
    # Values intentionally contain no archetype-era fields; only the keys matter.
    category_vendor_lookup = {
        "HubSpot": {},
        "Salesforce": {},
        "Pipedrive": {},
    }
    evidence = {
        "HubSpot": {"product_category": "CRM", "top_use_cases": ["lead management"]},
        "Salesforce": {"product_category": "CRM", "buyer_authority": {"role_types": {"sales": 5}}},
        "Pipedrive": {"product_category": "CRM", "competitors": [{"name": "HubSpot"}]},
    }

    result = select_categories(
        ecosystem,
        category_vendor_lookup,
        evidence,
        min_vendors=3,
        min_context_vendors=3,
        min_displacement_intensity=1.0,
        max_categories=5,
    )

    assert [category for category, _eco in result] == ["CRM"]


@pytest.mark.asyncio
async def test_select_asymmetry_pairs_require_overlap_and_resource_divergence():
    vendor_scores = [
        {"vendor_name": "HubSpot", "avg_urgency": 6.8, "total_reviews": 120},
        {"vendor_name": "Salesforce", "avg_urgency": 6.3, "total_reviews": 520},
        {"vendor_name": "Canva", "avg_urgency": 6.5, "total_reviews": 900},
    ]
    evidence = {
        "HubSpot": {
            "product_category": "CRM",
            "top_use_cases": ["lead management"],
            "buyer_authority": {"role_types": {"marketing": 4}},
        },
        "Salesforce": {
            "product_category": "CRM",
            "top_use_cases": ["lead management", "pipeline management"],
            "buyer_authority": {"role_types": {"marketing": 3}},
        },
        "Canva": {
            "product_category": "Design",
            "top_use_cases": ["graphics"],
            "buyer_authority": {"role_types": {"design": 6}},
        },
    }
    profiles = {
        "HubSpot": {"typical_company_size": ["smb"]},
        "Salesforce": {"typical_company_size": ["enterprise"]},
        "Canva": {"typical_company_size": ["smb"]},
    }

    result = await select_asymmetry_pairs(
        vendor_scores,
        evidence,
        profiles,
        max_pairs=5,
        pressure_delta_max=1.5,
        review_ratio_min=3.0,
        segment_divergence_bonus=5.0,
        min_divergence_score=2.0,
        min_context_score=2.0,
    )

    assert ("HubSpot", "Salesforce") in result
    assert all("Canva" not in pair for pair in result)
