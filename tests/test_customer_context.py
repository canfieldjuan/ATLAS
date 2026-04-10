from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atlas_brain.services.customer_context import CustomerContextService


@pytest.mark.asyncio
async def test_get_b2b_churn_signals_uses_shared_company_context_adapter():
    service = CustomerContextService()
    pool = MagicMock()
    pool.is_initialized = True
    adapter = AsyncMock(
        return_value=[
            {
                "vendor_name": "HubSpot",
                "product_category": "CRM",
                "avg_urgency_score": 7.5,
                "top_pain_categories": ["pricing"],
                "top_competitors": [{"name": "Salesforce"}],
                "decision_maker_churn_rate": 0.4,
                "price_complaint_rate": 0.2,
            }
        ]
    )

    with patch(
        "atlas_brain.config.settings",
        SimpleNamespace(b2b_churn=SimpleNamespace(context_enrichment_enabled=True)),
    ):
        with patch(
            "atlas_brain.storage.database.get_db_pool",
            return_value=pool,
        ):
            with patch(
                "atlas_brain.autonomous.tasks._b2b_shared.read_company_churn_context",
                adapter,
            ):
                rows = await service._get_b2b_churn_signals({"email": "alex@acme.com"})

    adapter.assert_awaited_once_with(
        pool,
        company_hint="acme",
        limit=5,
    )
    assert rows[0]["vendor_name"] == "HubSpot"


@pytest.mark.asyncio
async def test_get_b2b_churn_signals_skips_free_email_domains():
    service = CustomerContextService()

    with patch(
        "atlas_brain.config.settings",
        SimpleNamespace(b2b_churn=SimpleNamespace(context_enrichment_enabled=True)),
    ):
        with patch(
            "atlas_brain.storage.database.get_db_pool",
            return_value=MagicMock(is_initialized=True),
        ):
            with patch(
                "atlas_brain.autonomous.tasks._b2b_shared.read_company_churn_context",
                new=AsyncMock(),
            ) as adapter:
                rows = await service._get_b2b_churn_signals({"email": "alex@gmail.com"})

    assert rows == []
    adapter.assert_not_awaited()
