from __future__ import annotations

from datetime import date, timedelta

import pytest

from extracted_content_pipeline.reasoning.temporal import (
    AnomalyScore,
    CategoryPercentile,
    LongTermTrend,
    TemporalEngine,
    TemporalEvidence,
    VendorVelocity,
)


class _Pool:
    def __init__(
        self,
        *,
        snapshots_by_vendor=None,
        category_rows=None,
        vendor_rows=None,
    ):
        self.snapshots_by_vendor = snapshots_by_vendor or {}
        self.category_rows = category_rows or {}
        self.vendor_rows = vendor_rows or []
        self.calls = []

    async def fetch(self, query, *args):
        self.calls.append((query, args))
        if "SELECT DISTINCT vendor_name" in query:
            return self.vendor_rows
        if "WHERE vendor_name = $1" in query:
            return self.snapshots_by_vendor.get(args[0], [])
        if "WHERE s.product_category = $1" in query:
            return self.category_rows.get(args[0], [])
        return []


def _snapshot(day: date, **values):
    row = {
        "snapshot_date": day,
        "vendor_name": values.pop("vendor_name", "Acme"),
        "product_category": values.pop("product_category", "CRM"),
    }
    row.update(values)
    return row


@pytest.mark.asyncio
async def test_analyze_vendor_returns_insufficient_data_for_missing_history():
    pool = _Pool(snapshots_by_vendor={"Acme": [_snapshot(date(2026, 1, 1), avg_urgency=5)]})

    result = await TemporalEngine(pool).analyze_vendor(" Acme ")

    assert result.vendor_name == "Acme"
    assert result.snapshot_days == 1
    assert result.insufficient_data is True


@pytest.mark.asyncio
async def test_analyze_vendor_computes_velocity_acceleration_and_anomaly():
    snapshots = [
        _snapshot(date(2026, 1, 1), avg_urgency=3, competitor_count=1),
        _snapshot(date(2026, 1, 8), avg_urgency=5, competitor_count=2),
        _snapshot(date(2026, 1, 15), avg_urgency=9, competitor_count=4),
    ]
    category_rows = [
        _snapshot(date(2026, 1, 15), vendor_name="A", avg_urgency=3),
        _snapshot(date(2026, 1, 15), vendor_name="B", avg_urgency=4),
        _snapshot(date(2026, 1, 15), vendor_name="C", avg_urgency=5),
        _snapshot(date(2026, 1, 15), vendor_name="D", avg_urgency=6),
        _snapshot(date(2026, 1, 15), vendor_name="Acme", avg_urgency=9),
    ]
    pool = _Pool(
        snapshots_by_vendor={"Acme": snapshots},
        category_rows={"CRM": category_rows},
    )

    result = await TemporalEngine(pool).analyze_vendor("Acme")
    evidence = TemporalEngine.to_evidence_dict(result)

    assert result.insufficient_data is False
    assert result.snapshot_days == 3
    assert evidence["velocity_avg_urgency"] == round((9 - 5) / 7, 4)
    assert evidence["accel_avg_urgency"] > 0
    assert result.category_baselines[0].product_category == "CRM"
    assert evidence["anomalies"][0]["metric"] == "avg_urgency"


def test_to_evidence_dict_serializes_velocity_trends_and_only_anomalies():
    temporal = TemporalEvidence(
        vendor_name="Acme",
        snapshot_days=21,
        velocities=[
            VendorVelocity(
                vendor_name="Acme",
                metric="avg_urgency",
                current_value=8,
                previous_value=5,
                velocity=0.5,
                days_between=6,
                acceleration=0.1,
            )
        ],
        trends=[LongTermTrend(metric="avg_urgency", slope_30d=0.2, slope_90d=0.1)],
        anomalies=[
            AnomalyScore("Acme", "avg_urgency", 8, 2.4, 0.01, True),
            AnomalyScore("Acme", "competitor_count", 4, 0.8, 0.2, False),
        ],
    )

    evidence = TemporalEngine.to_evidence_dict(temporal)

    assert evidence["velocity_avg_urgency"] == 0.5
    assert evidence["accel_avg_urgency"] == 0.1
    assert evidence["trend_30d_avg_urgency"] == 0.2
    assert evidence["trend_90d_avg_urgency"] == 0.1
    assert evidence["anomalies"] == [
        {"metric": "avg_urgency", "value": 8, "z_score": 2.4, "p_value": 0.01}
    ]


def test_to_evidence_dict_marks_insufficient_data():
    evidence = TemporalEngine.to_evidence_dict(
        TemporalEvidence("Acme", snapshot_days=1, insufficient_data=True)
    )

    assert evidence == {"temporal_status": "insufficient_data", "snapshot_days": 1}


def test_compute_long_term_trends_handles_date_strings_and_bad_values():
    start = date(2026, 1, 1)
    snapshots = [
        _snapshot(
            start + timedelta(days=i),
            avg_urgency=str(3 + i),
            churn_density="bad" if i == 5 else i,
        )
        for i in range(15)
    ]
    snapshots[-1]["snapshot_date"] = snapshots[-1]["snapshot_date"].isoformat()
    engine = TemporalEngine(None)

    trends = engine._compute_long_term_trends("Acme", snapshots)

    urgency = next(trend for trend in trends if trend.metric == "avg_urgency")
    assert urgency.slope_30d == pytest.approx(1.0)
    assert urgency.volatility is not None


@pytest.mark.asyncio
async def test_compute_percentiles_uses_local_snapshot_query():
    pool = _Pool(
        category_rows={
            "CRM": [
                {"avg_urgency": 3, "competitor_count": 1},
                {"avg_urgency": 4, "competitor_count": 2},
                {"avg_urgency": 5, "competitor_count": 3},
                {"avg_urgency": 6, "competitor_count": 4},
            ]
        }
    )

    percentiles = await TemporalEngine(pool)._compute_percentiles("CRM")

    query, args = pool.calls[0]
    assert "b2b_vendor_snapshots" in query
    assert "s.product_category = $1" in query
    assert args == ("CRM",)
    assert any(item.metric == "avg_urgency" for item in percentiles)


@pytest.mark.asyncio
async def test_analyze_all_vendors_uses_pool_distinct_vendor_query():
    pool = _Pool(
        vendor_rows=[{"vendor_name": "Acme"}, {"vendor_name": "Beta"}],
        snapshots_by_vendor={
            "Acme": [
                _snapshot(date(2026, 1, 1), vendor_name="Acme"),
                _snapshot(date(2026, 1, 2), vendor_name="Acme"),
            ],
            "Beta": [
                _snapshot(date(2026, 1, 1), vendor_name="Beta"),
                _snapshot(date(2026, 1, 2), vendor_name="Beta"),
            ],
        },
    )

    results = await TemporalEngine(pool).analyze_all_vendors()

    assert set(results) == {"Acme", "Beta"}
    assert pool.calls[0][0].startswith("SELECT DISTINCT vendor_name")


def test_recency_weight_uses_half_life_decay():
    assert TemporalEngine.recency_weight(0) == 1.0
    assert TemporalEngine.recency_weight(14) == pytest.approx(0.5)


def test_compute_anomalies_ignores_zero_iqr_baselines():
    engine = TemporalEngine(None)
    anomalies = engine._compute_anomalies(
        "Acme",
        {"avg_urgency": 9},
        [CategoryPercentile("CRM", "avg_urgency", p25=5, p50=5, p75=5, sample_count=3)],
    )

    assert anomalies == []
