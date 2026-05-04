"""Temporal Reasoning Engine.

Computes time-series analytics over `b2b_vendor_snapshots`:

    - Velocity: rate of change per metric (needs 2+ days)
    - Acceleration: rate of velocity change (needs 3+ days)
    - Rolling percentiles by category (25th / 50th / 75th)
    - Z-score anomaly detection against category baselines
    - Recency-weighted review scoring

All outputs are pure data (no LLM) -- they feed into synthesis-first
vendor reasoning, archetype scoring, and downstream deterministic
builders.

This module landed via PR-C1b as a consolidation of the atlas-canonical
temporal engine and the content_pipeline fork; PR-C1j (this commit's
predecessor) carried the rest of content_pipeline's defensive logic
across so the slim core is the single canonical implementation.
Documented decisions (per
`docs/extraction/evidence_temporal_archetypes_audit_2026-05-03.md`):

  - 5 dataclasses are `frozen=True` (carries content_pipeline's
    immutability decision forward).
  - `_numeric_value` / `_row_get` / `_coerce_date` / `_days_between` /
    `_volatility` / `_percentiles_from_rows` defensive helpers from
    content_pipeline are surfaced at module scope so callers handling
    messy snapshot payloads (strings with commas / percent signs, ISO
    date strings, asyncpg.Records vs plain dicts) have a single
    coercion point.
  - `TemporalEngine` takes a `min_days_for_percentiles` constructor
    argument (default = `MIN_DAYS_FOR_PERCENTILES` = 3). Atlas's prior
    module constant declared 7 but the percentile gate was hardcoded
    to `< 3` everywhere it was checked; the constant is the actual
    behavior, with a knob for products that want stricter gating.
  - `_compute_percentiles` is self-contained -- it queries
    `b2b_vendor_snapshots` directly with a single SELECT scoped to
    `s.product_category`. PR-C1j replaced the prior atlas-coupled
    implementation (which lazily imported `_b2b_shared` helpers) so
    reasoning core has zero `atlas_brain` dependency at runtime.
    Category inference now reads `latest.product_category` from the
    most recent snapshot rather than calling out to atlas; the dead
    `_infer_category` helper was dropped.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Any

from .types import (
    AnomalyScore,
    CategoryPercentile,
    LongTermTrend,
    TemporalEvidence,
    VendorVelocity,
)

MIN_DAYS_FOR_VELOCITY = 2
MIN_DAYS_FOR_ACCELERATION = 3
MIN_DAYS_FOR_PERCENTILES = 3  # Was atlas's constant=7 / hardcoded gate=3; canonicalized to 3 (atlas's actual behavior)
MIN_DAYS_FOR_TREND = 14  # Relaxed from 30 to allow shorter history
RECENCY_HALF_LIFE_DAYS = 14


# Metrics to track velocity on
_VELOCITY_METRICS = [
    "churn_density", "avg_urgency", "positive_review_pct",
    "recommend_ratio", "pain_count", "competitor_count",
    "displacement_edge_count", "high_intent_company_count",
    "support_sentiment", "employee_growth_rate",
    "new_feature_velocity", "legacy_support_score",
]


class TemporalEngine:
    """Computes temporal analytics from `b2b_vendor_snapshots`.

    `min_days_for_percentiles` controls the per-category sample-size gate
    used by `_compute_percentiles`. Default (`MIN_DAYS_FOR_PERCENTILES` = 3)
    matches both atlas's actual behavior and content_pipeline's wired
    constant. Products that want stricter gating can pass a larger value
    at construction.
    """

    def __init__(
        self,
        pool: Any,
        *,
        min_days_for_percentiles: int = MIN_DAYS_FOR_PERCENTILES,
    ) -> None:
        self._pool = pool
        self._min_days_for_percentiles = int(min_days_for_percentiles)

    async def analyze_vendor(self, vendor_name: str) -> TemporalEvidence:
        """Full temporal analysis for a single vendor.

        ``vendor_name`` is normalized via ``str(...).strip()`` so callers
        can pass loosely-shaped input (whitespace, ``None``) without
        breaking the per-vendor lookup. The returned ``TemporalEvidence``
        is constructed once with all fields -- the prior implementation
        mutated it after construction, which now fails because PR-C1c
        promoted ``TemporalEvidence`` to ``frozen=True``.
        """
        normalized_name = str(vendor_name or "").strip()
        snapshots = await self._load_snapshots(normalized_name)

        if len(snapshots) < MIN_DAYS_FOR_VELOCITY:
            return TemporalEvidence(
                vendor_name=normalized_name,
                snapshot_days=len(snapshots),
                insufficient_data=True,
            )

        velocities = self._compute_velocities(normalized_name, snapshots)
        trends: list[LongTermTrend] = []
        if len(snapshots) >= MIN_DAYS_FOR_TREND:
            trends = self._compute_long_term_trends(normalized_name, snapshots)

        baselines: list[CategoryPercentile] = []
        anomalies: list[AnomalyScore] = []
        latest = snapshots[-1]
        category = str(latest.get("product_category") or "").strip()
        if category:
            baselines = await self._compute_percentiles(category)
            anomalies = self._compute_anomalies(normalized_name, latest, baselines)

        return TemporalEvidence(
            vendor_name=normalized_name,
            snapshot_days=len(snapshots),
            velocities=velocities,
            trends=trends,
            anomalies=anomalies,
            category_baselines=baselines,
        )

    async def analyze_all_vendors(self) -> dict[str, TemporalEvidence]:
        """Temporal analysis for all vendors with snapshots."""
        if not self._pool or not hasattr(self._pool, "fetch"):
            return {}
        rows = await self._pool.fetch(
            "SELECT DISTINCT vendor_name FROM b2b_vendor_snapshots ORDER BY vendor_name"
        )
        results: dict[str, TemporalEvidence] = {}
        for row in rows:
            name = str(_row_get(row, "vendor_name") or "").strip()
            if not name:
                continue
            results[name] = await self.analyze_vendor(name)
        return results

    # ------------------------------------------------------------------
    # Velocity / Acceleration
    # ------------------------------------------------------------------

    def _compute_velocities(
        self,
        vendor_name: str,
        snapshots: list[dict[str, Any]],
    ) -> list[VendorVelocity]:
        """Compute rate-of-change for each metric between consecutive snapshots.

        Uses ``_days_between`` for date math (handles ISO date strings via
        ``_coerce_date``) and ``_numeric_value`` for metric values
        (handles "1,234"/"28%" strings) so messy inbound rows don't
        silently fall through.
        """
        if len(snapshots) < MIN_DAYS_FOR_VELOCITY:
            return []

        latest = snapshots[-1]
        previous = snapshots[-2]
        days = _days_between(previous.get("snapshot_date"), latest.get("snapshot_date"))
        if days <= 0:
            return []

        velocities: list[VendorVelocity] = []
        for metric in _VELOCITY_METRICS:
            current = _numeric_value(latest.get(metric))
            prior = _numeric_value(previous.get(metric))
            if current is None or prior is None:
                continue

            velocity = (current - prior) / days
            acceleration = None
            if len(snapshots) >= MIN_DAYS_FOR_ACCELERATION:
                prior2 = snapshots[-3]
                prior2_value = _numeric_value(prior2.get(metric))
                days2 = _days_between(prior2.get("snapshot_date"), previous.get("snapshot_date"))
                if prior2_value is not None and days2 > 0:
                    previous_velocity = (prior - prior2_value) / days2
                    acceleration = (velocity - previous_velocity) / days

            velocities.append(VendorVelocity(
                vendor_name=vendor_name,
                metric=metric,
                current_value=current,
                previous_value=prior,
                velocity=velocity,
                days_between=days,
                acceleration=acceleration,
            ))

        return velocities

    def _compute_long_term_trends(
        self,
        vendor_name: str,
        snapshots: list[dict[str, Any]],
    ) -> list[LongTermTrend]:
        """Compute 30-day and 90-day linear trends for key metrics.

        Uses ``_coerce_date`` so date math works on snapshot rows that
        carry ``snapshot_date`` as an ISO string. Builds each
        ``LongTermTrend`` with its final fields rather than mutating
        post-construction (the prior implementation triggered
        ``FrozenInstanceError`` after PR-C1c froze the dataclass).
        """
        if len(snapshots) < MIN_DAYS_FOR_TREND:
            return []

        latest_date = _coerce_date(snapshots[-1].get("snapshot_date"))
        if latest_date is None:
            return []

        def _window(days: int) -> list[dict[str, Any]]:
            cutoff = latest_date - timedelta(days=days)
            rows: list[dict[str, Any]] = []
            for snapshot in snapshots:
                snapshot_date = _coerce_date(snapshot.get("snapshot_date"))
                if snapshot_date is not None and snapshot_date >= cutoff:
                    rows.append(snapshot)
            return rows

        window_30 = _window(30)
        window_90 = _window(90)

        trends: list[LongTermTrend] = []
        for metric in _VELOCITY_METRICS:
            if _numeric_value(snapshots[-1].get(metric)) is None:
                continue
            slope_30 = self._calculate_slope(window_30, metric)
            slope_90 = (
                self._calculate_slope(window_90, metric)
                if len(window_90) > len(window_30)
                else None
            )
            if slope_30 is None and slope_90 is None:
                continue
            trends.append(
                LongTermTrend(
                    metric=metric,
                    slope_30d=slope_30,
                    slope_90d=slope_90,
                    volatility=_volatility(window_30, metric),
                    data_points=len(window_30),
                )
            )
        return trends

    def _calculate_slope(
        self,
        window: list[dict[str, Any]],
        metric: str,
    ) -> float | None:
        """Calculate linear regression slope (change per day)."""
        if len(window) < 2:
            return None

        start = _coerce_date(window[0].get("snapshot_date"))
        if start is None:
            return None

        x_values: list[int] = []
        y_values: list[float] = []
        for snapshot in window:
            snapshot_date = _coerce_date(snapshot.get("snapshot_date"))
            value = _numeric_value(snapshot.get(metric))
            if snapshot_date is None or value is None:
                continue
            x_values.append((snapshot_date - start).days)
            y_values.append(value)

        if len(x_values) < 2:
            return None

        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_xx = sum(x * x for x in x_values)
        denominator = n * sum_xx - sum_x * sum_x
        if denominator == 0:
            return None
        return (n * sum_xy - sum_x * sum_y) / denominator

    # ------------------------------------------------------------------
    # Category Percentiles
    # ------------------------------------------------------------------

    async def _compute_percentiles(self, category: str) -> list[CategoryPercentile]:
        """Compute rolling percentiles for a category from latest snapshots.

        Self-contained: queries ``b2b_vendor_snapshots`` directly (one
        SELECT) instead of layering atop the atlas-side
        ``read_category_vendor_signal_rows`` helper. This keeps reasoning
        core free of an ``atlas_brain`` import dependency, matching the
        standalone behavior content_pipeline shipped before PR-C1j
        consolidated onto core.
        """
        if not self._pool or not hasattr(self._pool, "fetch"):
            return []
        rows = await self._pool.fetch(
            """
            SELECT s.* FROM b2b_vendor_snapshots s
            JOIN (
                SELECT vendor_name, MAX(snapshot_date) AS max_date
                FROM b2b_vendor_snapshots
                GROUP BY vendor_name
            ) latest ON s.vendor_name = latest.vendor_name
                AND s.snapshot_date = latest.max_date
            WHERE s.product_category = $1
            """,
            category,
        )
        return _percentiles_from_rows(
            category,
            [dict(row) for row in rows],
            min_count=self._min_days_for_percentiles,
        )

    # ------------------------------------------------------------------
    # Z-Score Anomaly Detection
    # ------------------------------------------------------------------

    def _compute_anomalies(
        self,
        vendor_name: str,
        latest_snapshot: dict[str, Any],
        baselines: list[CategoryPercentile],
    ) -> list[AnomalyScore]:
        """Compute z-scores for vendor metrics against category baselines."""
        baseline_by_metric = {baseline.metric: baseline for baseline in baselines}
        anomalies: list[AnomalyScore] = []
        for metric in _VELOCITY_METRICS:
            value = _numeric_value(latest_snapshot.get(metric))
            baseline = baseline_by_metric.get(metric)
            if value is None or baseline is None:
                continue
            iqr = baseline.p75 - baseline.p25
            if iqr <= 0:
                continue
            std_approx = iqr / 1.35
            z_score = (value - baseline.p50) / std_approx
            p_value = _normal_sf(abs(z_score))
            anomalies.append(
                AnomalyScore(
                    vendor_name=vendor_name,
                    metric=metric,
                    value=value,
                    z_score=round(z_score, 3),
                    p_value=round(p_value, 4),
                    is_anomaly=abs(z_score) > 2.0,
                )
            )
        return anomalies

    # ------------------------------------------------------------------
    # Recency Weighting
    # ------------------------------------------------------------------

    @staticmethod
    def recency_weight(days_old: int, half_life: int = RECENCY_HALF_LIFE_DAYS) -> float:
        """Exponential decay weight for review recency."""
        if days_old <= 0:
            return 1.0
        return math.pow(2, -(days_old / half_life))

    # ------------------------------------------------------------------
    # Evidence serialization for synthesis/deterministic consumers
    # ------------------------------------------------------------------

    @staticmethod
    def to_evidence_dict(te: TemporalEvidence) -> dict[str, Any]:
        """Convert temporal evidence to a dict suitable for the reasoner."""
        if te.insufficient_data:
            return {
                "temporal_status": "insufficient_data",
                "snapshot_days": te.snapshot_days,
            }

        evidence: dict[str, Any] = {"snapshot_days": te.snapshot_days}

        # Velocities
        for v in te.velocities:
            evidence[f"velocity_{v.metric}"] = round(v.velocity, 4)
            if v.acceleration is not None:
                evidence[f"accel_{v.metric}"] = round(v.acceleration, 4)

        # Long-term trends
        for t in te.trends:
            if t.slope_30d is not None:
                evidence[f"trend_30d_{t.metric}"] = round(t.slope_30d, 4)
            if t.slope_90d is not None:
                evidence[f"trend_90d_{t.metric}"] = round(t.slope_90d, 4)

        # Anomalies
        anomaly_list = []
        for a in te.anomalies:
            if a.is_anomaly:
                anomaly_list.append({
                    "metric": a.metric,
                    "value": a.value,
                    "z_score": a.z_score,
                    "p_value": a.p_value,
                })
        if anomaly_list:
            evidence["anomalies"] = anomaly_list

        return evidence

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _load_snapshots(self, vendor_name: str) -> list[dict[str, Any]]:
        """Load snapshots for a vendor, ordered by date."""
        if not self._pool or not hasattr(self._pool, "fetch"):
            return []
        rows = await self._pool.fetch(
            """
            SELECT * FROM b2b_vendor_snapshots
            WHERE vendor_name = $1
            ORDER BY snapshot_date ASC
            """,
            vendor_name,
        )
        return [dict(row) for row in rows]


def _percentiles_from_rows(
    category: str,
    rows: list[dict[str, Any]],
    *,
    min_count: int = MIN_DAYS_FOR_PERCENTILES,
) -> list[CategoryPercentile]:
    """Compute per-metric p25/p50/p75 percentiles from a list of snapshot rows.

    ``min_count`` controls the per-metric sample-size gate -- callers
    that want stricter gating (atlas's prior constant=7 path) can pass
    a larger value via ``TemporalEngine(min_days_for_percentiles=...)``.
    """
    if len(rows) < min_count:
        return []

    percentiles: list[CategoryPercentile] = []
    for metric in _VELOCITY_METRICS:
        values = sorted(
            value
            for value in (_numeric_value(row.get(metric)) for row in rows)
            if value is not None
        )
        if len(values) < min_count:
            continue
        n = len(values)
        percentiles.append(
            CategoryPercentile(
                product_category=category,
                metric=metric,
                p25=values[int(n * 0.25)],
                p50=values[int(n * 0.50)],
                p75=values[int(n * 0.75)],
                sample_count=n,
            )
        )
    return percentiles


def _volatility(window: list[dict[str, Any]], metric: str) -> float | None:
    """Standard deviation of ``metric`` values in a snapshot window."""
    values = [
        value
        for value in (_numeric_value(snapshot.get(metric)) for snapshot in window)
        if value is not None
    ]
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _days_between(start: Any, end: Any) -> int:
    """Days between two date-shaped values, coerced via ``_coerce_date``.

    Returns 0 if either side fails to coerce -- callers gate on
    ``days > 0`` before doing arithmetic, so 0 is a safe sentinel.
    """
    start_date = _coerce_date(start)
    end_date = _coerce_date(end)
    if start_date is None or end_date is None:
        return 0
    return (end_date - start_date).days


def _coerce_date(value: Any) -> date | None:
    """Coerce a possibly-stringified date to a ``datetime.date``.

    Accepts ``date``, ``datetime``, and ISO-format strings (``"2026-05-04"``).
    Returns ``None`` for empty/None/unparseable input so callers can
    treat ingestion errors as missing data instead of raising.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return datetime.fromisoformat(stripped).date()
        except ValueError:
            return None
    return None


def _normal_sf(z: float) -> float:
    """Survival function (1 - CDF) for standard normal. Approximation."""
    # Abramowitz & Stegun approximation
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    sf = d * math.exp(-z * z / 2) * poly
    return sf if z >= 0 else 1.0 - sf


# ------------------------------------------------------------------
# Defensive value/row helpers (carried from content_pipeline fork)
# ------------------------------------------------------------------


def _numeric_value(value: Any) -> float | None:
    """Coerce a possibly-messy snapshot value to a float.

    Handles raw numerics, strings with thousands separators (`"1,234"`)
    and percent suffixes (`"42%"`), and the bool-is-not-a-number trap.
    Returns None on any conversion failure rather than raising; callers
    use the None to skip the metric without aborting the whole vendor.

    Surfaced at module scope so that any caller building snapshot
    payloads has one canonical coercion entry point.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().replace(",", "")
        if not stripped:
            return None
        if stripped.endswith("%"):
            stripped = stripped[:-1]
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _row_get(row: Any, key: str) -> Any:
    """Read a key from a row regardless of its concrete shape.

    Snapshot rows arrive as plain dicts in tests, asyncpg.Records in
    production, and occasionally as objects with attribute access.
    This helper papers over the shape difference so callers do not
    have to branch.
    """
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]
    except (KeyError, TypeError):
        return getattr(row, key, None)


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
