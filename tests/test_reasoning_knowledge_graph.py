from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock
import sys

import pytest

from atlas_brain.reasoning.knowledge_graph import GROUP_ID, KnowledgeGraphSync


class FakeSession:
    def __init__(self):
        self.run = AsyncMock(return_value=None)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeDriver:
    def __init__(self, session):
        self._session = session

    def session(self):
        return self._session


@pytest.mark.asyncio
async def test_sync_vendors_uses_shared_graph_sync_adapter(monkeypatch):
    helper = AsyncMock(
        return_value=[
            {
                "canonical_name": "Zendesk",
                "aliases": '["Zen"]',
                "product_category": "CRM",
                "total_reviews": 120,
                "avg_urgency": 6.4,
                "confidence_score": 0.8,
                "churn_density": 28.0,
                "positive_review_pct": 42.0,
                "recommend_ratio": 0.63,
                "pain_count": 5,
                "competitor_count": 3,
            },
        ],
    )
    fake_shared = ModuleType("atlas_brain.autonomous.tasks._b2b_shared")
    fake_shared.read_vendor_graph_sync_rows = helper
    monkeypatch.setitem(sys.modules, "atlas_brain.autonomous.tasks._b2b_shared", fake_shared)

    pg_pool = SimpleNamespace(fetch=AsyncMock(side_effect=AssertionError("direct pool fetch should not run")))
    session = FakeSession()
    driver = FakeDriver(session)

    synced = await KnowledgeGraphSync(pg_pool, driver)._sync_vendors()

    helper.assert_awaited_once_with(pg_pool)
    assert synced == 1

    query = session.run.await_args.args[0]
    params = session.run.await_args.kwargs
    assert "MERGE (v:B2bVendor {canonical_name: $name})" in query
    assert params == {
        "name": "Zendesk",
        "gid": GROUP_ID,
        "aliases": ["Zen"],
        "category": "CRM",
        "reviews": 120,
        "churn": 28.0,
        "urgency": 6.4,
        "pos_pct": 42.0,
        "rec_ratio": 0.63,
        "pain_cnt": 5,
        "comp_cnt": 3,
        "conf": 0.8,
    }
