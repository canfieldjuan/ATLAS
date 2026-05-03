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
temporal engine and the content_pipeline fork. Documented decisions
(per `docs/extraction/evidence_temporal_archetypes_audit_2026-05-03.md`):

  - 5 dataclasses are `frozen=True` (carries content_pipeline's
    immutability decision forward).
  - `_numeric_value` / `_row_get` defensive helpers from
    content_pipeline are surfaced at module scope so callers handling
    messy snapshot payloads (strings with commas / percent signs,
    asyncpg.Records vs plain dicts) have a single coercion point.
  - `TemporalEngine` takes a `min_days_for_percentiles` constructor
    argument (default = `MIN_DAYS_FOR_PERCENTILES` = 3). Atlas's prior
    module constant declared 7 but the percentile gate was hardcoded
    to `< 3` everywhere it was checked; the constant is the actual
    behavior, with a knob for products that want stricter gating.
  - The `_b2b_shared` lookup helpers used by `_compute_percentiles` /
    `_infer_category` are imported from `atlas_brain` lazily so the
    module loads cleanly even when atlas_brain is not on the import
    path (degrades to empty percentiles + no inferred category in
    that case). A future PR will replace these with a port-based
    abstraction so reasoning core has zero atlas dependency.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
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
        """Full temporal analysis for a single vendor."""
        snapshots = await self._load_snapshots(vendor_name)

        if len(snapshots) < MIN_DAYS_FOR_VELOCITY:
            return TemporalEvidence(
                vendor_name=vendor_name,
                snapshot_days=len(snapshots),
                insufficient_data=True,
            )

        evidence = TemporalEvidence(
            vendor_name=vendor_name,
            snapshot_days=len(snapshots),
        )

        # Compute velocities
        evidence.velocities = self._compute_velocities(vendor_name, snapshots)

        # Compute long-term trends
        if len(snapshots) >= MIN_DAYS_FOR_TREND:
            evidence.trends = self._compute_long_term_trends(vendor_name, snapshots)

        # Compute category baselines + anomalies
        if len(snapshots) >= 2:
            latest = snapshots[-1]
            category = latest.get("product_category") or await self._infer_category(vendor_name)
            if category:
                evidence.category_baselines = await self._compute_percentiles(category)
                evidence.anomalies = self._compute_anomalies(
                    vendor_name, latest, evidence.category_baselines,
                )

        return evidence

    async def analyze_all_vendors(self) -> dict[str, TemporalEvidence]:
        """Temporal analysis for all vendors with snapshots."""
        vendors = await self._pool.fetch(
            "SELECT DISTINCT vendor_name FROM b2b_vendor_snapshots ORDER BY vendor_name"
        )
        results = {}
        for row in vendors:
            name = row["vendor_name"]
            results[name] = await self.analyze_vendor(name)
        return results

    # ------------------------------------------------------------------
    # Velocity / Acceleration
    # ------------------------------------------------------------------

    def _compute_velocities(
        self, vendor_name: str, snapshots: list[dict],
    ) -> list[VendorVelocity]:
        """Compute rate-of-change for each metric between consecutive snapshots."""
        if len(snapshots) < 2:
            return []

        velocities = []
        latest = snapshots[-1]
        previous = snapshots[-2]
        days = (latest["snapshot_date"] - previous["snapshot_date"]).days
        if days <= 0:
            return []

        for metric in _VELOCITY_METRICS:
            curr = latest.get(metric)
            prev = previous.get(metric)
            if curr is None or prev is None:
                continue
            try:
                curr_f = float(curr)
                prev_f = float(prev)
            except (ValueError, TypeError):
                continue

            velocity = (curr_f - prev_f) / days

            # Acceleration if 3+ snapshots
            accel = None
            if len(snapshots) >= 3:
                prev2 = snapshots[-3]
                prev2_val = prev2.get(metric)
                if prev2_val is not None:
                    days2 = (previous["snapshot_date"] - prev2["snapshot_date"]).days
                    if days2 > 0:
                        prev_velocity = (prev_f - float(prev2_val)) / days2
                        accel = (velocity - prev_velocity) / days

            velocities.append(VendorVelocity(
                vendor_name=vendor_name,
                metric=metric,
                current_value=curr_f,
                previous_value=prev_f,
                velocity=velocity,
                days_between=days,
                acceleration=accel,
            ))

        return velocities

    def _compute_long_term_trends(
        self, vendor_name: str, snapshots: list[dict],
    ) -> list[LongTermTrend]:
        """Compute 30-day and 90-day linear trends for key metrics."""
        trends = []
        if len(snapshots) < MIN_DAYS_FOR_TREND:
            return []

        latest_date = snapshots[-1]["snapshot_date"]
        
        # Helper to get subset of snapshots within last N days
        def _get_window(days: int) -> list[dict]:
            cutoff = latest_date - timedelta(days=days)
            return [s for s in snapshots if s["snapshot_date"] >= cutoff]

        window_30 = _get_window(30)
        window_90 = _get_window(90)

        for metric in _VELOCITY_METRICS:
            # Only compute trend if metric exists in latest snapshot
            if snapshots[-1].get(metric) is None:
                continue

            trend = LongTermTrend(metric=metric)
            
            # 30-day slope
            slope_30 = self._calculate_slope(window_30, metric)
            if slope_30 is not None:
                trend.slope_30d = slope_30
                trend.data_points = len(window_30)

            # 90-day slope (only if we have more history than 30 days)
            if len(window_90) > len(window_30):
                slope_90 = self._calculate_slope(window_90, metric)
                if slope_90 is not None:
                    trend.slope_90d = slope_90

            if trend.slope_30d is not None:
                trends.append(trend)

        return trends

    def _calculate_slope(self, window: list[dict], metric: str) -> float | None:
        """Calculate linear regression slope (change per day)."""
        if len(window) < 2:
            return None
            
        x_vals = [] # Days from start of window
        y_vals = []
        
        start_date = window[0]["snapshot_date"]
        
        for s in window:
            val = s.get(metric)
            if val is not None:
                try:
                    y = float(val)
                    days = (s["snapshot_date"] - start_date).days
                    x_vals.append(days)
                    y_vals.append(y)
                except (ValueError, TypeError):
                    continue
                    
        if len(x_vals) < 2:
            return None
            
        # Simple linear regression: slope = cov(x,y) / var(x)
        n = len(x_vals)
        sum_x = sum(x_vals)
        sum_y = sum(y_vals)
        sum_xy = sum(x*y for x, y in zip(x_vals, y_vals))
        sum_xx = sum(x*x for x in x_vals)
        
        denom = (n * sum_xx - sum_x * sum_x)
        if denom == 0:
            return None
            
        slope = (n * sum_xy - sum_x * sum_y) / denom
        return slope

    # ------------------------------------------------------------------
    # Category Percentiles
    # ------------------------------------------------------------------

    async def _compute_percentiles(self, category: str) -> list[CategoryPercentile]:
        """Compute rolling percentiles for a category from latest snapshots.

        Uses atlas's `read_category_vendor_signal_rows` helper when
        `atlas_brain` is on the import path; degrades to an empty list
        when it is not, so the module loads cleanly outside the host.
        A future PR will replace this lazy import with a port-based
        abstraction so reasoning core has zero atlas dependency.
        """
        try:
            from atlas_brain.autonomous.tasks._b2b_shared import (
                read_category_vendor_signal_rows,
            )
        except ImportError:
            return []

        vendor_rows = await read_category_vendor_signal_rows(
            self._pool,
            product_category=category,
        )
        normalized_vendor_names = sorted(
            {
                str(row.get("vendor_name") or "").strip().lower()
                for row in vendor_rows
                if str(row.get("vendor_name") or "").strip()
            }
        )
        if not normalized_vendor_names:
            return []

        rows = await self._pool.fetch(
            """
            SELECT s.* FROM b2b_vendor_snapshots s
            JOIN (
                SELECT vendor_name, MAX(snapshot_date) AS max_date
                FROM b2b_vendor_snapshots
                GROUP BY vendor_name
            ) latest ON s.vendor_name = latest.vendor_name AND s.snapshot_date = latest.max_date
            WHERE LOWER(s.vendor_name) = ANY($1::text[])
            """,
            normalized_vendor_names,
        )

        if len(rows) < self._min_days_for_percentiles:
            return []

        percentiles = []
        for metric in _VELOCITY_METRICS:
            values = []
            for row in rows:
                val = _row_get(row, metric)
                num = _numeric_value(val)
                if num is not None:
                    values.append(num)
            if len(values) < self._min_days_for_percentiles:
                continue

            values.sort()
            n = len(values)
            p25 = values[int(n * 0.25)]
            p50 = values[int(n * 0.50)]
            p75 = values[int(n * 0.75)]

            percentiles.append(CategoryPercentile(
                product_category=category,
                metric=metric,
                p25=p25,
                p50=p50,
                p75=p75,
                sample_count=n,
            ))

        return percentiles

    # ------------------------------------------------------------------
    # Z-Score Anomaly Detection
    # ------------------------------------------------------------------

    def _compute_anomalies(
        self,
        vendor_name: str,
        latest_snapshot: dict,
        baselines: list[CategoryPercentile],
    ) -> list[AnomalyScore]:
        """Compute z-scores for vendor metrics against category baselines."""
        anomalies = []
        baseline_map = {b.metric: b for b in baselines}

        for metric in _VELOCITY_METRICS:
            val = latest_snapshot.get(metric)
            baseline = baseline_map.get(metric)
            if val is None or baseline is None:
                continue

            try:
                val_f = float(val)
            except (ValueError, TypeError):
                continue

            # Approximate std from IQR: std ~= IQR / 1.35
            iqr = baseline.p75 - baseline.p25
            if iqr <= 0:
                continue
            std_approx = iqr / 1.35
            z = (val_f - baseline.p50) / std_approx

            # Approximate p-value from z-score (normal CDF)
            p_value = _normal_sf(abs(z))

            anomalies.append(AnomalyScore(
                vendor_name=vendor_name,
                metric=metric,
                value=val_f,
                z_score=round(z, 3),
                p_value=round(p_value, 4) if p_value else None,
                is_anomaly=abs(z) > 2.0,
            ))

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

    async def _load_snapshots(self, vendor_name: str) -> list[dict]:
        """Load snapshots for a vendor, ordered by date."""
        rows = await self._pool.fetch(
            """
            SELECT * FROM b2b_vendor_snapshots
            WHERE vendor_name = $1
            ORDER BY snapshot_date ASC
            """,
            vendor_name,
        )
        return [dict(r) for r in rows]

    async def _infer_category(self, vendor_name: str) -> str | None:
        """Try to infer product category from churn signals.

        Uses atlas's `read_vendor_signal_detail_exact` helper when
        `atlas_brain` is on the import path; returns None otherwise so
        callers degrade gracefully when reasoning core is loaded outside
        the host. A future PR will replace this lazy import with a
        port-based abstraction.
        """
        try:
            from atlas_brain.autonomous.tasks._b2b_shared import (
                read_vendor_signal_detail_exact,
            )
        except ImportError:
            return None

        row = await read_vendor_signal_detail_exact(
            self._pool,
            vendor_name=vendor_name,
        )
        return row["product_category"] if row else None


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
