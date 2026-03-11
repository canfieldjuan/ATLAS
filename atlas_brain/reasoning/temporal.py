"""Temporal Reasoning Engine (WS1).

Computes time-series analytics over b2b_vendor_snapshots:
    - Velocity: rate of change per metric (needs 2+ days)
    - Acceleration: rate of velocity change (needs 3+ days)
    - Rolling percentiles by category (25th/50th/75th)
    - Z-score anomaly detection against category baselines
    - Recency-weighted review scoring

All outputs are pure data (no LLM) -- they feed into the stratified reasoner
as evidence for archetype matching and full reasoning.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

logger = logging.getLogger("atlas.reasoning.temporal")

MIN_DAYS_FOR_VELOCITY = 2
MIN_DAYS_FOR_ACCELERATION = 3
MIN_DAYS_FOR_PERCENTILES = 7
RECENCY_HALF_LIFE_DAYS = 14


@dataclass
class VendorVelocity:
    """Rate-of-change metrics for a vendor."""

    vendor_name: str
    metric: str
    current_value: float
    previous_value: float
    velocity: float          # change per day
    days_between: int
    acceleration: float | None = None  # change in velocity (needs 3+ points)


@dataclass
class CategoryPercentile:
    """Rolling percentile baselines for a metric within a category."""

    product_category: str
    metric: str
    p25: float
    p50: float
    p75: float
    sample_count: int


@dataclass
class AnomalyScore:
    """Z-score anomaly detection for a vendor metric."""

    vendor_name: str
    metric: str
    value: float
    z_score: float
    p_value: float | None = None
    is_anomaly: bool = False    # |z| > 2.0


@dataclass
class TemporalEvidence:
    """Complete temporal analysis for a vendor, used as evidence in reasoning."""

    vendor_name: str
    snapshot_days: int
    velocities: list[VendorVelocity] = field(default_factory=list)
    anomalies: list[AnomalyScore] = field(default_factory=list)
    category_baselines: list[CategoryPercentile] = field(default_factory=list)
    insufficient_data: bool = False


# Metrics to track velocity on
_VELOCITY_METRICS = [
    "churn_density", "avg_urgency", "positive_review_pct",
    "recommend_ratio", "pain_count", "competitor_count",
    "displacement_edge_count", "high_intent_company_count",
]


class TemporalEngine:
    """Computes temporal analytics from b2b_vendor_snapshots."""

    def __init__(self, pool: Any):
        self._pool = pool

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

    # ------------------------------------------------------------------
    # Category Percentiles
    # ------------------------------------------------------------------

    async def _compute_percentiles(self, category: str) -> list[CategoryPercentile]:
        """Compute rolling percentiles for a category from latest snapshots."""
        rows = await self._pool.fetch(
            """
            SELECT s.* FROM b2b_vendor_snapshots s
            JOIN (
                SELECT vendor_name, MAX(snapshot_date) AS max_date
                FROM b2b_vendor_snapshots
                GROUP BY vendor_name
            ) latest ON s.vendor_name = latest.vendor_name AND s.snapshot_date = latest.max_date
            JOIN b2b_churn_signals cs ON LOWER(s.vendor_name) = LOWER(cs.vendor_name)
            WHERE LOWER(cs.product_category) = LOWER($1)
            """,
            category,
        )

        if len(rows) < 3:
            return []

        percentiles = []
        for metric in _VELOCITY_METRICS:
            values = []
            for row in rows:
                val = row.get(metric)
                if val is not None:
                    try:
                        values.append(float(val))
                    except (ValueError, TypeError):
                        pass
            if len(values) < 3:
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
    # Evidence serialization (for stratified reasoner)
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
        """Try to infer product category from churn signals."""
        row = await self._pool.fetchrow(
            "SELECT product_category FROM b2b_churn_signals WHERE vendor_name = $1 LIMIT 1",
            vendor_name,
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
