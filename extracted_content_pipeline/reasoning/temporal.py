"""Compatibility wrapper for the shared extracted reasoning temporal module.

PR-C1j replaced the prior ~466-line local fork with this thin wrapper.
The canonical implementation lives in
`extracted_reasoning_core.temporal` (consolidated via PR-C1b / PR #100,
with the rich type promotion landing in PR-C1c / PR #102 -- both
merged before this PR's fork point).

This wrapper preserves the public-import path
(`from extracted_content_pipeline.reasoning.temporal import ...`) so
existing content_pipeline callers and the
`tests/test_extracted_reasoning_temporal.py` test suite keep working
unchanged. The public contract types (`TemporalEvidence`,
`VendorVelocity`, `LongTermTrend`, `CategoryPercentile`,
`AnomalyScore`) are re-exported from `extracted_reasoning_core.types`.
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
