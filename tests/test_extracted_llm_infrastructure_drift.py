"""Tests for extracted_llm_infrastructure.services.cost.drift.

Uses an in-memory fake pool that emulates the FULL OUTER JOIN drift
query so we can verify the public API contract + chip classification
without a live database.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

import pytest

from extracted_llm_infrastructure.services.cost.drift import (
    DriftRow,
    compute_drift,
)


class _FakePool:
    """Fake pool that emulates the drift query against in-memory tables.

    Tests populate ``llm_usage_rows`` (each with model_provider,
    created_at, cost_usd) and ``daily_cost_rows`` (each with provider,
    cost_date, provider_cost_usd). ``fetch`` runs the equivalent of the
    drift module's CTE FULL OUTER JOIN against those lists.
    """

    def __init__(self) -> None:
        self.llm_usage_rows: list[dict[str, Any]] = []
        self.daily_cost_rows: list[dict[str, Any]] = []

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        provider, start, end = args
        # Local daily sums
        local_by_date: dict[date, Decimal] = {}
        for r in self.llm_usage_rows:
            if r["model_provider"] != provider:
                continue
            d = r["created_at"].astimezone(timezone.utc).date()
            if not (start <= d < end):
                continue
            local_by_date[d] = local_by_date.get(d, Decimal(0)) + Decimal(
                str(r["cost_usd"])
            )
        # Invoiced daily sums
        invoiced_by_date: dict[date, Decimal] = {}
        for r in self.daily_cost_rows:
            if r["provider"] != provider:
                continue
            d = r["cost_date"]
            if not (start <= d < end):
                continue
            invoiced_by_date[d] = invoiced_by_date.get(d, Decimal(0)) + Decimal(
                str(r["provider_cost_usd"])
            )
        # FULL OUTER JOIN
        all_dates = sorted(set(local_by_date) | set(invoiced_by_date))
        return [
            {
                "cost_date": d,
                "local_usd": local_by_date.get(d, Decimal(0)),
                "invoiced_usd": invoiced_by_date.get(d, Decimal(0)),
            }
            for d in all_dates
        ]


def _utc(year: int, month: int, day: int, hour: int = 12) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_compute_drift_empty_returns_empty_list():
    pool = _FakePool()
    result = await compute_drift(
        pool,
        provider="anthropic",
        date_range=(date(2026, 5, 1), date(2026, 5, 4)),
    )
    assert result == []


@pytest.mark.asyncio
async def test_compute_drift_single_day_clean_match():
    # local and invoiced match exactly -> delta 0, no chips
    pool = _FakePool()
    pool.llm_usage_rows = [
        {"model_provider": "anthropic", "created_at": _utc(2026, 5, 1), "cost_usd": Decimal("10.00")},
    ]
    pool.daily_cost_rows = [
        {"provider": "anthropic", "cost_date": date(2026, 5, 1), "provider_cost_usd": Decimal("10.00")},
    ]
    result = await compute_drift(
        pool,
        provider="anthropic",
        date_range=(date(2026, 5, 1), date(2026, 5, 4)),
    )
    assert len(result) == 1
    row = result[0]
    assert isinstance(row, DriftRow)
    assert row.provider == "anthropic"
    assert row.cost_date == date(2026, 5, 1)
    assert row.local_usd == Decimal("10.00")
    assert row.invoiced_usd == Decimal("10.00")
    assert row.delta_usd == Decimal("0")
    assert row.delta_pct == 0.0
    assert row.explained_by == []


@pytest.mark.asyncio
async def test_compute_drift_stale_pricing_chip_when_invoice_higher():
    # local 80, invoiced 100 -> 25% drift, "high_drift" + "stale_pricing"
    pool = _FakePool()
    pool.llm_usage_rows = [
        {"model_provider": "anthropic", "created_at": _utc(2026, 5, 1), "cost_usd": Decimal("80.00")},
    ]
    pool.daily_cost_rows = [
        {"provider": "anthropic", "cost_date": date(2026, 5, 1), "provider_cost_usd": Decimal("100.00")},
    ]
    result = await compute_drift(
        pool,
        provider="anthropic",
        date_range=(date(2026, 5, 1), date(2026, 5, 4)),
    )
    assert len(result) == 1
    row = result[0]
    assert row.delta_usd == Decimal("20.00")
    assert abs(row.delta_pct - 0.20) < 1e-9
    assert "high_drift" in row.explained_by
    assert "stale_pricing" in row.explained_by


@pytest.mark.asyncio
async def test_compute_drift_below_threshold_no_high_drift_chip():
    # local 90, invoiced 100 -> 10% drift, only "stale_pricing"
    pool = _FakePool()
    pool.llm_usage_rows = [
        {"model_provider": "anthropic", "created_at": _utc(2026, 5, 1), "cost_usd": Decimal("90.00")},
    ]
    pool.daily_cost_rows = [
        {"provider": "anthropic", "cost_date": date(2026, 5, 1), "provider_cost_usd": Decimal("100.00")},
    ]
    result = await compute_drift(
        pool,
        provider="anthropic",
        date_range=(date(2026, 5, 1), date(2026, 5, 4)),
    )
    row = result[0]
    assert "high_drift" not in row.explained_by
    assert "stale_pricing" in row.explained_by


@pytest.mark.asyncio
async def test_compute_drift_missing_local_rows_chip():
    # invoice has spend, local has none -> "missing_local_rows" + "high_drift"
    # (no "stale_pricing" since the trivial empty case is excluded)
    pool = _FakePool()
    pool.daily_cost_rows = [
        {"provider": "anthropic", "cost_date": date(2026, 5, 1), "provider_cost_usd": Decimal("50.00")},
    ]
    result = await compute_drift(
        pool,
        provider="anthropic",
        date_range=(date(2026, 5, 1), date(2026, 5, 4)),
    )
    row = result[0]
    assert row.local_usd == Decimal(0)
    assert row.invoiced_usd == Decimal("50.00")
    assert "missing_local_rows" in row.explained_by
    assert "high_drift" in row.explained_by
    assert "stale_pricing" not in row.explained_by


@pytest.mark.asyncio
async def test_compute_drift_missing_invoice_chip():
    # local has spend, invoice has none -> "missing_invoice"; no high_drift
    # (cannot compute % when invoiced is 0)
    pool = _FakePool()
    pool.llm_usage_rows = [
        {"model_provider": "anthropic", "created_at": _utc(2026, 5, 1), "cost_usd": Decimal("50.00")},
    ]
    result = await compute_drift(
        pool,
        provider="anthropic",
        date_range=(date(2026, 5, 1), date(2026, 5, 4)),
    )
    row = result[0]
    assert row.local_usd == Decimal("50.00")
    assert row.invoiced_usd == Decimal(0)
    assert row.delta_pct == 0.0
    assert "missing_invoice" in row.explained_by


@pytest.mark.asyncio
async def test_compute_drift_filters_by_provider():
    # Two providers in the pool; only the requested one shows up.
    pool = _FakePool()
    pool.llm_usage_rows = [
        {"model_provider": "anthropic", "created_at": _utc(2026, 5, 1), "cost_usd": Decimal("10.00")},
        {"model_provider": "openrouter", "created_at": _utc(2026, 5, 1), "cost_usd": Decimal("20.00")},
    ]
    pool.daily_cost_rows = [
        {"provider": "anthropic", "cost_date": date(2026, 5, 1), "provider_cost_usd": Decimal("10.00")},
        {"provider": "openrouter", "cost_date": date(2026, 5, 1), "provider_cost_usd": Decimal("20.00")},
    ]
    anthropic_only = await compute_drift(
        pool, provider="anthropic", date_range=(date(2026, 5, 1), date(2026, 5, 4)),
    )
    assert len(anthropic_only) == 1
    assert anthropic_only[0].provider == "anthropic"
    assert anthropic_only[0].local_usd == Decimal("10.00")


@pytest.mark.asyncio
async def test_compute_drift_filters_by_date_range():
    # Hits before, in, and after the range; only in-range survives.
    pool = _FakePool()
    pool.llm_usage_rows = [
        {"model_provider": "anthropic", "created_at": _utc(2026, 4, 30), "cost_usd": Decimal("9.99")},
        {"model_provider": "anthropic", "created_at": _utc(2026, 5, 2), "cost_usd": Decimal("5.00")},
        {"model_provider": "anthropic", "created_at": _utc(2026, 5, 4), "cost_usd": Decimal("9.99")},
    ]
    pool.daily_cost_rows = [
        {"provider": "anthropic", "cost_date": date(2026, 5, 2), "provider_cost_usd": Decimal("5.00")},
    ]
    result = await compute_drift(
        pool, provider="anthropic", date_range=(date(2026, 5, 1), date(2026, 5, 4)),
    )
    assert len(result) == 1
    assert result[0].cost_date == date(2026, 5, 2)


@pytest.mark.asyncio
async def test_compute_drift_full_outer_join_yields_all_dates():
    # Local has day 1, invoice has day 2. Both dates appear in the result.
    pool = _FakePool()
    pool.llm_usage_rows = [
        {"model_provider": "anthropic", "created_at": _utc(2026, 5, 1), "cost_usd": Decimal("10.00")},
    ]
    pool.daily_cost_rows = [
        {"provider": "anthropic", "cost_date": date(2026, 5, 2), "provider_cost_usd": Decimal("20.00")},
    ]
    result = await compute_drift(
        pool, provider="anthropic", date_range=(date(2026, 5, 1), date(2026, 5, 4)),
    )
    dates = [r.cost_date for r in result]
    assert dates == [date(2026, 5, 1), date(2026, 5, 2)]
    assert "missing_invoice" in result[0].explained_by
    assert "missing_local_rows" in result[1].explained_by


def test_drift_row_is_frozen_dataclass():
    row = DriftRow(
        provider="anthropic",
        cost_date=date(2026, 5, 1),
        local_usd=Decimal(0),
        invoiced_usd=Decimal(0),
        delta_usd=Decimal(0),
        delta_pct=0.0,
    )
    assert row.explained_by == []
    # Frozen
    with pytest.raises(Exception):
        row.provider = "openrouter"  # type: ignore[misc]
