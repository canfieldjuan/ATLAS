from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_ecosystem_analyze_all_categories_uses_shared_category_adapter(monkeypatch):
    from atlas_brain.reasoning.ecosystem import EcosystemAnalyzer
    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod

    analyzer = EcosystemAnalyzer(pool=SimpleNamespace())
    read_categories = AsyncMock(return_value=["CRM", "Helpdesk"])
    monkeypatch.setattr(shared_mod, "read_signal_product_categories", read_categories)
    monkeypatch.setattr(
        analyzer,
        "analyze_category",
        AsyncMock(side_effect=lambda category: f"eco:{category}"),
    )

    results = await analyzer.analyze_all_categories()

    read_categories.assert_awaited_once_with(analyzer._pool)
    assert results == {
        "CRM": "eco:CRM",
        "Helpdesk": "eco:Helpdesk",
    }


@pytest.mark.asyncio
async def test_ecosystem_load_category_vendors_uses_shared_category_signal_rows(monkeypatch):
    from atlas_brain.reasoning.ecosystem import EcosystemAnalyzer
    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod

    analyzer = EcosystemAnalyzer(pool=SimpleNamespace())
    read_rows = AsyncMock(
        return_value=[
            {"vendor_name": "Zendesk", "total_reviews": 120},
        ]
    )
    monkeypatch.setattr(shared_mod, "read_category_vendor_signal_rows", read_rows)

    rows = await analyzer._load_category_vendors("CRM")

    read_rows.assert_awaited_once_with(
        analyzer._pool,
        product_category="CRM",
    )
    assert rows == [{"vendor_name": "Zendesk", "total_reviews": 120}]


@pytest.mark.asyncio
async def test_ecosystem_pain_distribution_scopes_to_vendor_filters():
    from atlas_brain.reasoning.ecosystem import EcosystemAnalyzer

    pool = SimpleNamespace(
        fetch=AsyncMock(
            return_value=[
                {"pain_category": "support", "total_mentions": 12},
            ]
        ),
    )
    analyzer = EcosystemAnalyzer(pool=pool)

    rows = await analyzer._compute_pain_distribution(["Zendesk", "Freshdesk"])

    fetch_sql, fetch_vendors = pool.fetch.await_args.args
    assert "LOWER(pp.vendor_name) = ANY($1::text[])" in fetch_sql
    assert "JOIN b2b_churn_signals" not in fetch_sql
    assert fetch_vendors == ["freshdesk", "zendesk"]
    assert rows == {"support": 12}


@pytest.mark.asyncio
async def test_ecosystem_top_displacements_scopes_to_vendor_filters():
    from atlas_brain.reasoning.ecosystem import EcosystemAnalyzer

    pool = SimpleNamespace(
        fetch=AsyncMock(
            return_value=[
                {
                    "from_vendor": "Zendesk",
                    "to_vendor": "Freshdesk",
                    "mention_count": 9,
                    "primary_driver": "support",
                    "signal_strength": "strong",
                },
            ]
        ),
    )
    analyzer = EcosystemAnalyzer(pool=pool)

    rows = await analyzer._load_top_displacements(["Zendesk", "Freshdesk"])

    fetch_sql, fetch_vendors = pool.fetch.await_args.args
    assert "LOWER(de.from_vendor) = ANY($1::text[])" in fetch_sql
    assert "JOIN b2b_churn_signals" not in fetch_sql
    assert fetch_vendors == ["freshdesk", "zendesk"]
    assert rows == [
        {
            "from_vendor": "Zendesk",
            "to_vendor": "Freshdesk",
            "mention_count": 9,
            "primary_driver": "support",
            "signal_strength": "strong",
        },
    ]

