from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

MIN_DAYS_FOR_VELOCITY = 2
MIN_DAYS_FOR_ACCELERATION = 3
MIN_DAYS_FOR_PERCENTILES = 3
MIN_DAYS_FOR_TREND = 14
RECENCY_HALF_LIFE_DAYS = 14


@dataclass(frozen=True)
class VendorVelocity:
    vendor_name: str
    metric: str
    current_value: float
    previous_value: float
    velocity: float
    days_between: int
    acceleration: float | None = None


@dataclass(frozen=True)
class LongTermTrend:
    metric: str
    slope_30d: float | None = None
    slope_90d: float | None = None
    volatility: float | None = None
    data_points: int = 0


@dataclass(frozen=True)
class CategoryPercentile:
    product_category: str
    metric: str
    p25: float
    p50: float
    p75: float
    sample_count: int


@dataclass(frozen=True)
class AnomalyScore:
    vendor_name: str
    metric: str
    value: float
    z_score: float
    p_value: float | None = None
    is_anomaly: bool = False


@dataclass(frozen=True)
class TemporalEvidence:
    vendor_name: str
    snapshot_days: int
    velocities: list[VendorVelocity] = field(default_factory=list)
    trends: list[LongTermTrend] = field(default_factory=list)
    anomalies: list[AnomalyScore] = field(default_factory=list)
    category_baselines: list[CategoryPercentile] = field(default_factory=list)
    insufficient_data: bool = False


_VELOCITY_METRICS = [
    "churn_density",
    "avg_urgency",
    "positive_review_pct",
    "recommend_ratio",
    "pain_count",
    "competitor_count",
    "displacement_edge_count",
    "high_intent_company_count",
    "support_sentiment",
    "employee_growth_rate",
    "new_feature_velocity",
    "legacy_support_score",
]


class TemporalEngine:
    """Standalone temporal analyzer over vendor snapshot rows.

    The engine expects a pool/repository object with an async ``fetch`` method.
    It does not import Atlas helpers; hosts can satisfy the same contract with
    asyncpg, a repository adapter, or a test double.
    """

    def __init__(self, pool: Any):
        self._pool = pool

    async def analyze_vendor(self, vendor_name: str) -> TemporalEvidence:
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

    def _compute_velocities(
        self,
        vendor_name: str,
        snapshots: list[dict[str, Any]],
    ) -> list[VendorVelocity]:
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

            velocities.append(
                VendorVelocity(
                    vendor_name=vendor_name,
                    metric=metric,
                    current_value=current,
                    previous_value=prior,
                    velocity=velocity,
                    days_between=days,
                    acceleration=acceleration,
                )
            )
        return velocities

    def _compute_long_term_trends(
        self,
        vendor_name: str,
        snapshots: list[dict[str, Any]],
    ) -> list[LongTermTrend]:
        if len(snapshots) < MIN_DAYS_FOR_TREND:
            return []

        latest_date = _coerce_date(snapshots[-1].get("snapshot_date"))
        if latest_date is None:
            return []

        def _window(days: int) -> list[dict[str, Any]]:
            cutoff = latest_date - timedelta(days=days)
            rows = []
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

    def _calculate_slope(self, window: list[dict[str, Any]], metric: str) -> float | None:
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

    async def _compute_percentiles(self, category: str) -> list[CategoryPercentile]:
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
        return _percentiles_from_rows(category, [dict(row) for row in rows])

    def _compute_anomalies(
        self,
        vendor_name: str,
        latest_snapshot: dict[str, Any],
        baselines: list[CategoryPercentile],
    ) -> list[AnomalyScore]:
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

    @staticmethod
    def recency_weight(days_old: int, half_life: int = RECENCY_HALF_LIFE_DAYS) -> float:
        if days_old <= 0:
            return 1.0
        return math.pow(2, -(days_old / half_life))

    @staticmethod
    def to_evidence_dict(te: TemporalEvidence) -> dict[str, Any]:
        if te.insufficient_data:
            return {
                "temporal_status": "insufficient_data",
                "snapshot_days": te.snapshot_days,
            }

        evidence: dict[str, Any] = {"snapshot_days": te.snapshot_days}
        for velocity in te.velocities:
            evidence[f"velocity_{velocity.metric}"] = round(velocity.velocity, 4)
            if velocity.acceleration is not None:
                evidence[f"accel_{velocity.metric}"] = round(velocity.acceleration, 4)

        for trend in te.trends:
            if trend.slope_30d is not None:
                evidence[f"trend_30d_{trend.metric}"] = round(trend.slope_30d, 4)
            if trend.slope_90d is not None:
                evidence[f"trend_90d_{trend.metric}"] = round(trend.slope_90d, 4)

        anomalies = [
            {
                "metric": anomaly.metric,
                "value": anomaly.value,
                "z_score": anomaly.z_score,
                "p_value": anomaly.p_value,
            }
            for anomaly in te.anomalies
            if anomaly.is_anomaly
        ]
        if anomalies:
            evidence["anomalies"] = anomalies
        return evidence

    async def _load_snapshots(self, vendor_name: str) -> list[dict[str, Any]]:
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
) -> list[CategoryPercentile]:
    if len(rows) < MIN_DAYS_FOR_PERCENTILES:
        return []

    percentiles: list[CategoryPercentile] = []
    for metric in _VELOCITY_METRICS:
        values = sorted(
            value
            for value in (_numeric_value(row.get(metric)) for row in rows)
            if value is not None
        )
        if len(values) < MIN_DAYS_FOR_PERCENTILES:
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
    start_date = _coerce_date(start)
    end_date = _coerce_date(end)
    if start_date is None or end_date is None:
        return 0
    return (end_date - start_date).days


def _coerce_date(value: Any) -> date | None:
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


def _numeric_value(value: Any) -> float | None:
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
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]
    except (KeyError, TypeError):
        return getattr(row, key, None)


def _normal_sf(z: float) -> float:
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    d = 0.3989422804014327
    poly = t * (
        0.319381530
        + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    sf = d * math.exp(-z * z / 2) * poly
    return sf if z >= 0 else 1.0 - sf
