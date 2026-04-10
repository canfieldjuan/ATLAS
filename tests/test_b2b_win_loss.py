import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)


class FakePool:
    def __init__(self, *, fetchrow_map=None, fetch_map=None):
        self.fetchrow_map = fetchrow_map or {}
        self.fetch_map = fetch_map or {}
        self.calls = []

    async def fetchrow(self, query, *params):
        normalized = " ".join(str(query).split())
        self.calls.append((normalized, params))
        for needle, value in self.fetchrow_map.items():
            if needle in normalized:
                return value(*params) if callable(value) else value
        raise AssertionError(f"Unexpected fetchrow query: {normalized}")

    async def fetch(self, query, *params):
        normalized = " ".join(str(query).split())
        self.calls.append((normalized, params))
        for needle, value in self.fetch_map.items():
            if needle in normalized:
                return value(*params) if callable(value) else value
        raise AssertionError(f"Unexpected fetch query: {normalized}")


@pytest.mark.asyncio
async def test_compute_prediction_uses_exact_signal_adapter_for_review_gate(monkeypatch):
    from atlas_brain.api import b2b_win_loss as mod
    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod

    pool = FakePool(
        fetchrow_map={
            "AS edges,": {
                "edges": 3,
                "pain_points": 0,
                "buyer_profiles": 0,
                "product_profiles": 0,
                "outcome_sequences": 0,
            },
        },
        fetch_map={
            "FROM b2b_displacement_edges": [
                {
                    "to_vendor": "Freshdesk",
                    "mention_count": 8,
                    "primary_driver": "support",
                    "signal_strength": "strong",
                    "confidence_score": 0.8,
                    "key_quote": None,
                },
                {
                    "to_vendor": "Intercom",
                    "mention_count": 5,
                    "primary_driver": "pricing",
                    "signal_strength": "moderate",
                    "confidence_score": 0.6,
                    "key_quote": None,
                },
            ],
            "FROM b2b_vendor_witnesses": [],
        },
    )
    signal_row = {
        "vendor_name": "Zendesk",
        "product_category": "CRM",
        "total_reviews": 12,
        "negative_reviews": 4,
        "churn_intent_count": 3,
        "avg_urgency_score": 8.0,
        "decision_maker_churn_rate": 0.2,
        "nps_proxy": -0.1,
        "price_complaint_rate": 0.3,
        "archetype": "price_squeeze",
        "confidence_score": 0.8,
    }
    read_signal = AsyncMock(return_value=signal_row)

    monkeypatch.setattr(shared_mod, "read_vendor_signal_detail_exact", read_signal)
    monkeypatch.setattr(
        mod,
        "_load_calibrated_weights",
        AsyncMock(return_value=(dict(mod.WEIGHTS), "static", None)),
    )
    monkeypatch.setattr(mod, "call_llm_with_skill", lambda *args, **kwargs: None)

    result = await mod._compute_prediction(pool, "Zendesk")

    read_signal.assert_awaited_once_with(pool, vendor_name="Zendesk")
    assert result.is_gated is False
    assert result.data_coverage["reviews"] == 12
    churn_factor = next(f for f in result.factors if f.name == "Churn Severity")
    assert churn_factor.gated is False
    assert churn_factor.data_points == 12
    assert "3/12 reviews show churn intent" in churn_factor.evidence
    assert all("b2b_churn_signals" not in query for query, _ in pool.calls)

