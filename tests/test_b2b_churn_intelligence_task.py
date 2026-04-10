from types import SimpleNamespace
from uuid import uuid4

import pytest
from unittest.mock import AsyncMock

from atlas_brain.autonomous.tasks import b2b_churn_intelligence as mod


class CapturePool:
    def __init__(self):
        self.execute_calls = []

    async def execute(self, query, *params):
        self.execute_calls.append((str(query), params))


@pytest.mark.asyncio
async def test_run_skips_when_b2b_churn_intelligence_disabled(monkeypatch):
    monkeypatch.setattr(mod.settings.b2b_churn, "enabled", False)
    monkeypatch.setattr(mod.settings.b2b_churn, "intelligence_enabled", False)

    task = SimpleNamespace(id=uuid4(), metadata={})

    result = await mod.run(task)

    assert result == {"_skip_synthesis": "B2B churn intelligence disabled"}


@pytest.mark.asyncio
async def test_run_allows_maintenance_override_when_disabled(monkeypatch):
    monkeypatch.setattr(mod.settings.b2b_churn, "enabled", False)
    monkeypatch.setattr(mod.settings.b2b_churn, "intelligence_enabled", False)

    class DummyPool:
        is_initialized = False

    monkeypatch.setattr(mod, "get_db_pool", lambda: DummyPool())

    task = SimpleNamespace(id=uuid4(), metadata={"maintenance_run": True})

    result = await mod.run(task)

    assert result == {"_skip_synthesis": "DB not ready"}


@pytest.mark.asyncio
async def test_load_category_council_lookup_uses_synthesis_first_loader(monkeypatch):
    fake_lookup = {
        "councils": {
            "CRM": {
                "confidence": 0.82,
                "conclusion": {
                    "winner": "HubSpot",
                    "loser": "Salesforce",
                    "conclusion": "HubSpot is gaining share in CRM.",
                    "market_regime": "fragmenting",
                    "durability_assessment": "structural",
                    "key_insights": ["signal-1", "signal-2", "signal-3", "signal-4"],
                },
            },
        },
    }
    loader = AsyncMock(return_value=fake_lookup)
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.load_best_cross_vendor_lookup",
        loader,
    )

    result = await mod._load_category_council_lookup(
        object(),
        [{"vendor_name": "Salesforce", "product_category": "CRM"}],
        as_of=mod.date(2026, 4, 7),
    )

    loader.assert_awaited_once()
    assert result[("Salesforce", "crm")]["winner"] == "HubSpot"
    assert result[("Salesforce", "crm")]["confidence"] == 0.82
    assert result[("Salesforce", "crm")]["key_insights"] == ["signal-1", "signal-2", "signal-3"]


@pytest.mark.asyncio
async def test_load_category_council_lookup_skips_generic_categories(monkeypatch):
    loader = AsyncMock(return_value={"councils": {"B2B Software": {"conclusion": {"winner": "A"}}}})
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.load_best_cross_vendor_lookup",
        loader,
    )

    result = await mod._load_category_council_lookup(
        object(),
        [{"vendor_name": "Salesforce", "product_category": "B2B Software"}],
        as_of=mod.date(2026, 4, 7),
    )

    assert result == {}


def test_task_run_id_prefers_scheduler_execution_id():
    execution_id = str(uuid4())
    task = SimpleNamespace(id=uuid4(), metadata={"_execution_id": execution_id, "run_id": "legacy-run"})

    assert mod._task_run_id(task) == execution_id


@pytest.mark.asyncio
async def test_upsert_churn_signals_persists_materialization_run_id():
    pool = CapturePool()

    failures = await mod._upsert_churn_signals(
        pool,
        vendor_scores=[
            {
                "vendor_name": "Zendesk",
                "total_reviews": 12,
                "churn_intent": 3,
                "avg_urgency": 7.5,
                "signal_reviews": 5,
            },
        ],
        neg_lookup={},
        pain_lookup={},
        competitor_lookup={},
        feature_gap_lookup={},
        price_lookup={},
        dm_lookup={},
        company_lookup={},
        quote_lookup={},
        materialization_run_id="run-123",
    )

    assert failures == 0
    assert len(pool.execute_calls) == 1
    query, params = pool.execute_calls[0]
    assert "materialization_run_id" in query
    assert params[-2] == "run-123"
