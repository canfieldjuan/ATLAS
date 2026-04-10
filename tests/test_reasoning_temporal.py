from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_temporal_engine_infer_category_uses_shared_exact_signal_adapter(monkeypatch):
    from atlas_brain.reasoning.temporal import TemporalEngine
    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod

    pool = SimpleNamespace(
        fetchrow=AsyncMock(side_effect=AssertionError("unexpected direct fetchrow")),
    )
    read_signal = AsyncMock(
        return_value={
            "vendor_name": "Zendesk",
            "product_category": "CRM",
        }
    )
    monkeypatch.setattr(shared_mod, "read_vendor_signal_detail_exact", read_signal)

    engine = TemporalEngine(pool)
    category = await engine._infer_category("Zendesk")

    read_signal.assert_awaited_once_with(
        pool,
        vendor_name="Zendesk",
    )
    assert category == "CRM"


@pytest.mark.asyncio
async def test_temporal_engine_compute_percentiles_uses_shared_category_rows(monkeypatch):
    from atlas_brain.reasoning.temporal import TemporalEngine
    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod

    pool = SimpleNamespace(
        fetch=AsyncMock(
            return_value=[
                {"vendor_name": "Zendesk", "churn_density": 28.0, "avg_urgency": 6.4},
                {"vendor_name": "Freshdesk", "churn_density": 22.0, "avg_urgency": 5.9},
                {"vendor_name": "Intercom", "churn_density": 25.0, "avg_urgency": 6.1},
            ]
        ),
    )
    read_rows = AsyncMock(
        return_value=[
            {"vendor_name": "Zendesk"},
            {"vendor_name": "Freshdesk"},
            {"vendor_name": "Intercom"},
        ]
    )
    monkeypatch.setattr(shared_mod, "read_category_vendor_signal_rows", read_rows)

    engine = TemporalEngine(pool)
    percentiles = await engine._compute_percentiles("CRM")

    read_rows.assert_awaited_once_with(
        pool,
        product_category="CRM",
    )
    fetch_sql, fetch_vendors = pool.fetch.await_args.args
    assert "LOWER(s.vendor_name) = ANY($1::text[])" in fetch_sql
    assert "JOIN b2b_churn_signals" not in fetch_sql
    assert fetch_vendors == ["freshdesk", "intercom", "zendesk"]
    assert any(item.metric == "churn_density" for item in percentiles)
