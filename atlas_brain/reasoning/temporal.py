"""Temporal Reasoning Engine (WS1).

Computes time-series analytics over b2b_vendor_snapshots:
    - Velocity: rate of change per metric (needs 2+ days)
    - Acceleration: rate of velocity change (needs 3+ days)
    - Rolling percentiles by category (25th/50th/75th)
    - Z-score anomaly detection against category baselines
    - Recency-weighted review scoring

All outputs are pure data (no LLM) -- they feed into synthesis-first
vendor reasoning, archetype scoring, and downstream deterministic builders.

PR-D7b4 promoted this module's body into
:mod:`extracted_reasoning_core.temporal` (with the public dataclasses
in :mod:`extracted_reasoning_core.types` per PR-C1c). Atlas keeps the
import surface ``atlas_brain.reasoning.temporal`` as a thin re-export
so internal callers (b2b_churn_intelligence, _b2b_shared, market_pulse,
the reasoning_temporal / reasoning_market_pulse / reasoning_live test
suites) don't need to change import sites.

Drift notes for the conversion:

- The five public dataclasses (``TemporalEvidence`` and the four sub-
  types ``VendorVelocity`` / ``LongTermTrend`` / ``CategoryPercentile``
  / ``AnomalyScore``) live in core's ``types`` module, frozen+slots.
  Atlas's pre-PR-C1c definitions were mutable; the conversion makes
  them immutable. Verified pre-conversion: no atlas caller mutates
  these instances after construction (``TemporalEngine.analyze_vendor``
  and ``_compute_long_term_trends`` already rebuild the dataclass on
  every update).
- ``TemporalEngine.__init__`` gains a backward-compatible
  ``*, min_days_for_percentiles=MIN_DAYS_FOR_PERCENTILES`` kwarg in
  core. Atlas callers using the positional ``(pool)`` form keep
  working untouched.
- ``MIN_DAYS_FOR_PERCENTILES`` was canonicalized to ``3`` in core
  (atlas had a constant=7 alongside a hardcoded inline gate=3; core
  picked the actual runtime value, 3). Behaviorally identical to
  atlas's pre-conversion runtime.
- Logger namespace migrates ``atlas.reasoning.temporal`` ->
  ``extracted_reasoning_core.temporal`` (correct -- the code is core's).
"""

from __future__ import annotations

from extracted_reasoning_core.temporal import (
    MIN_DAYS_FOR_ACCELERATION,
    MIN_DAYS_FOR_PERCENTILES,
    MIN_DAYS_FOR_TREND,
    MIN_DAYS_FOR_VELOCITY,
    RECENCY_HALF_LIFE_DAYS,
    TemporalEngine,
)
from extracted_reasoning_core.types import (
    AnomalyScore,
    CategoryPercentile,
    LongTermTrend,
    TemporalEvidence,
    VendorVelocity,
)

__all__ = [
    "AnomalyScore",
    "CategoryPercentile",
    "LongTermTrend",
    "MIN_DAYS_FOR_ACCELERATION",
    "MIN_DAYS_FOR_PERCENTILES",
    "MIN_DAYS_FOR_TREND",
    "MIN_DAYS_FOR_VELOCITY",
    "RECENCY_HALF_LIFE_DAYS",
    "TemporalEngine",
    "TemporalEvidence",
    "VendorVelocity",
]
