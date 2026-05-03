"""Smoke tests for extracted_reasoning_core.temporal.

Locks the consolidation against regressions during the rest of PR-C1.
Atlas-side temporal tests will rename + redirect into this file's
neighborhood in PR-C1k; for now this carries the minimum coverage
needed to validate the consolidation:

  - 5 dataclasses are frozen (immutability decision per audit)
  - MIN_DAYS_FOR_PERCENTILES is the parameterizable default
  - TemporalEngine.__init__ accepts min_days_for_percentiles kwarg
  - _numeric_value defensively coerces messy inputs
  - _row_get handles dicts and attribute-access shapes
  - module loads cleanly without atlas_brain on path
    (the lazy `_b2b_shared` import paths degrade gracefully)
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from extracted_reasoning_core.temporal import (
    MIN_DAYS_FOR_ACCELERATION,
    MIN_DAYS_FOR_PERCENTILES,
    MIN_DAYS_FOR_TREND,
    MIN_DAYS_FOR_VELOCITY,
    RECENCY_HALF_LIFE_DAYS,
    AnomalyScore,
    CategoryPercentile,
    LongTermTrend,
    TemporalEngine,
    TemporalEvidence,
    VendorVelocity,
    _numeric_value,
    _row_get,
)


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------


def test_module_constants_have_expected_defaults() -> None:
    # MIN_DAYS_FOR_PERCENTILES canonicalized to 3 (atlas's hardcoded
    # gate value, which is what both forks actually used at runtime).
    assert MIN_DAYS_FOR_PERCENTILES == 3
    assert MIN_DAYS_FOR_VELOCITY == 2
    assert MIN_DAYS_FOR_ACCELERATION == 3
    assert MIN_DAYS_FOR_TREND == 14
    assert RECENCY_HALF_LIFE_DAYS == 14


# ------------------------------------------------------------------
# Defensive helpers
# ------------------------------------------------------------------


def test_numeric_value_handles_clean_numerics() -> None:
    assert _numeric_value(1) == 1.0
    assert _numeric_value(1.5) == 1.5
    assert _numeric_value(0) == 0.0
    assert _numeric_value(-3.14) == -3.14


def test_numeric_value_strips_separators_and_percent() -> None:
    assert _numeric_value("1,234") == 1234.0
    assert _numeric_value("42%") == 42.0
    assert _numeric_value("  3.14  ") == 3.14
    assert _numeric_value("1,234.5%") == 1234.5


def test_numeric_value_returns_none_on_garbage() -> None:
    assert _numeric_value(None) is None
    assert _numeric_value("") is None
    assert _numeric_value("not-a-number") is None
    # bool subclass of int -- explicitly excluded
    assert _numeric_value(True) is None
    assert _numeric_value(False) is None


def test_row_get_handles_dict_and_object_shapes() -> None:
    assert _row_get({"a": 1}, "a") == 1
    assert _row_get({"a": 1}, "missing") is None

    class _Row:
        a = 5

    assert _row_get(_Row(), "a") == 5
    assert _row_get(_Row(), "missing") is None


def test_row_get_handles_subscript_objects() -> None:
    class _SubscriptOnly:
        def __getitem__(self, key: str) -> int:
            return {"x": 7}[key]

    assert _row_get(_SubscriptOnly(), "x") == 7


# ------------------------------------------------------------------
# TemporalEngine constructor
# ------------------------------------------------------------------


def test_temporal_engine_default_min_days_for_percentiles() -> None:
    engine = TemporalEngine(pool=None)
    assert engine._min_days_for_percentiles == MIN_DAYS_FOR_PERCENTILES == 3


def test_temporal_engine_accepts_min_days_for_percentiles_override() -> None:
    engine = TemporalEngine(pool=None, min_days_for_percentiles=5)
    assert engine._min_days_for_percentiles == 5

    # Atlas's prior nominal "stricter" setting
    engine_strict = TemporalEngine(pool=None, min_days_for_percentiles=7)
    assert engine_strict._min_days_for_percentiles == 7


def test_temporal_engine_coerces_min_days_to_int() -> None:
    engine = TemporalEngine(pool=None, min_days_for_percentiles="4")  # type: ignore[arg-type]
    assert engine._min_days_for_percentiles == 4


# ------------------------------------------------------------------
# Frozen dataclasses
# ------------------------------------------------------------------


def _expect_frozen_rejects_reassignment(instance: object, field_name: str) -> None:
    """Assert that reassigning `field_name` on `instance` raises."""
    try:
        setattr(instance, field_name, "value-that-should-not-stick")
    except Exception as exc:
        assert exc.__class__.__name__ in {
            "FrozenInstanceError",
            "AttributeError",
        }
    else:
        raise AssertionError(
            f"frozen dataclass should reject reassignment to {field_name}"
        )


def test_vendor_velocity_is_frozen() -> None:
    v = VendorVelocity(
        vendor_name="acme",
        metric="avg_urgency",
        current_value=7.0,
        previous_value=5.0,
        velocity=0.2,
        days_between=10,
    )
    _expect_frozen_rejects_reassignment(v, "velocity")


def test_long_term_trend_is_frozen() -> None:
    t = LongTermTrend(metric="avg_urgency", slope_30d=0.1, data_points=14)
    _expect_frozen_rejects_reassignment(t, "slope_30d")


def test_category_percentile_is_frozen() -> None:
    p = CategoryPercentile(
        product_category="crm",
        metric="avg_urgency",
        p25=4.0,
        p50=5.0,
        p75=6.0,
        sample_count=10,
    )
    _expect_frozen_rejects_reassignment(p, "p50")


def test_anomaly_score_is_frozen() -> None:
    a = AnomalyScore(
        vendor_name="acme",
        metric="avg_urgency",
        value=8.0,
        z_score=2.5,
        p_value=0.01,
        is_anomaly=True,
    )
    _expect_frozen_rejects_reassignment(a, "z_score")


def test_temporal_evidence_is_frozen() -> None:
    te = TemporalEvidence(vendor_name="acme", snapshot_days=30)
    _expect_frozen_rejects_reassignment(te, "snapshot_days")


# ------------------------------------------------------------------
# Engine: pure-python paths (no DB)
# ------------------------------------------------------------------


def _make_snapshot(d: date, **metrics: float | int | None) -> dict:
    base = {"snapshot_date": d, "vendor_name": "acme", "product_category": "crm"}
    base.update(metrics)
    return base


def test_compute_velocities_returns_empty_with_one_snapshot() -> None:
    engine = TemporalEngine(pool=None)
    one = [_make_snapshot(date(2026, 5, 1), avg_urgency=5.0)]
    assert engine._compute_velocities("acme", one) == []


def test_compute_velocities_basic_two_snapshots() -> None:
    engine = TemporalEngine(pool=None)
    snaps = [
        _make_snapshot(date(2026, 5, 1), avg_urgency=5.0),
        _make_snapshot(date(2026, 5, 5), avg_urgency=7.0),
    ]
    velocities = engine._compute_velocities("acme", snaps)
    # Should produce one velocity for avg_urgency
    by_metric = {v.metric: v for v in velocities}
    assert "avg_urgency" in by_metric
    v = by_metric["avg_urgency"]
    assert v.current_value == 7.0
    assert v.previous_value == 5.0
    assert v.days_between == 4
    assert v.velocity == pytest.approx((7.0 - 5.0) / 4)


def test_recency_weight_decays_exponentially() -> None:
    # 0 days old -> weight 1.0
    assert TemporalEngine.recency_weight(0) == 1.0
    # half life = 14 days -> exactly 0.5 at 14
    assert TemporalEngine.recency_weight(14) == pytest.approx(0.5, abs=1e-9)
    # custom half life
    assert TemporalEngine.recency_weight(7, half_life=7) == pytest.approx(0.5, abs=1e-9)


def test_to_evidence_dict_marks_insufficient_data() -> None:
    te = TemporalEvidence(vendor_name="acme", snapshot_days=1, insufficient_data=True)
    out = TemporalEngine.to_evidence_dict(te)
    assert out["temporal_status"] == "insufficient_data"
    assert out["snapshot_days"] == 1
