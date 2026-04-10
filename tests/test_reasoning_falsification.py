from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_fetch_fresh_signals_uses_shared_exact_signal_adapter(monkeypatch):
    from atlas_brain.reasoning.falsification import FalsificationWatcher
    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod

    pool = SimpleNamespace(
        fetchrow=AsyncMock(
            side_effect=[
                {
                    "avg_urgency": 6.4,
                    "positive_review_pct": 42.0,
                    "churn_density": 28.0,
                    "pressure_score": 0.55,
                    "total_reviews": 120,
                    "recommend_ratio": 0.63,
                    "archetype": None,
                    "archetype_confidence": None,
                },
                {
                    "avg_urgency": 6.0,
                    "positive_review_pct": 45.0,
                    "churn_density": 25.0,
                    "pressure_score": 0.5,
                    "recommend_ratio": 0.66,
                },
            ]
        ),
        fetchval=AsyncMock(return_value=3),
        fetch=AsyncMock(return_value=[]),
    )
    read_signal = AsyncMock(
        return_value={
            "price_complaint_rate": 0.2,
            "decision_maker_churn_rate": 0.3,
            "archetype": "price_squeeze",
            "archetype_confidence": 0.8,
        }
    )
    monkeypatch.setattr(shared_mod, "read_vendor_signal_detail_exact", read_signal)

    watcher = FalsificationWatcher(pool, cache=object())
    signals = await watcher._fetch_fresh_signals("Zendesk")

    read_signal.assert_awaited_once_with(
        pool,
        vendor_name="Zendesk",
    )
    assert pool.fetchrow.await_count == 2
    assert signals["price_complaint_rate"] == 0.2
    assert signals["dm_churn_rate"] == 0.3
    assert signals["current_archetype"] == "price_squeeze"
    assert signals["archetype_confidence"] == 0.8
    assert signals["negative_review_count_7d"] == 3

