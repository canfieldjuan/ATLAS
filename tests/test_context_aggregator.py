from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_fetch_b2b_churn_company_uses_shared_company_context_adapter(monkeypatch):
    from atlas_brain.reasoning import context_aggregator as mod
    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod
    from atlas_brain.storage import database as db_mod

    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(side_effect=AssertionError("unexpected direct fetch")),
        fetchval=AsyncMock(side_effect=AssertionError("unexpected contact lookup")),
    )
    read_context = AsyncMock(
        return_value=[
            {
                "vendor_name": "Zendesk",
                "product_category": "CRM",
                "avg_urgency_score": 6.4,
            },
        ]
    )

    monkeypatch.setattr(db_mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(shared_mod, "read_company_churn_context", read_context)

    rows = await mod._fetch_b2b_churn("company", "Acme")

    read_context.assert_awaited_once_with(
        pool,
        company_hint="Acme",
        limit=5,
    )
    assert rows == [
        {
            "vendor_name": "Zendesk",
            "product_category": "CRM",
            "avg_urgency_score": 6.4,
        },
    ]


@pytest.mark.asyncio
async def test_fetch_b2b_churn_contact_resolves_company_hint_before_shared_adapter(monkeypatch):
    from atlas_brain.reasoning import context_aggregator as mod
    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod
    from atlas_brain.storage import database as db_mod

    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(side_effect=AssertionError("unexpected direct fetch")),
        fetchval=AsyncMock(return_value="alex@acme.com"),
    )
    read_context = AsyncMock(return_value=[])

    monkeypatch.setattr(db_mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(shared_mod, "read_company_churn_context", read_context)

    rows = await mod._fetch_b2b_churn("contact", "contact-123")

    pool.fetchval.assert_awaited_once_with(
        "SELECT email FROM contacts WHERE id = $1",
        "contact-123",
    )
    read_context.assert_awaited_once_with(
        pool,
        company_hint="acme",
        limit=5,
    )
    assert rows == []

