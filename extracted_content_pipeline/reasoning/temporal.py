from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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


class TemporalEngine:
    def __init__(self, pool: Any):
        self._pool = pool

    async def analyze_vendor(self, vendor_name: str) -> TemporalEvidence:
        return TemporalEvidence(
            vendor_name=str(vendor_name or ""),
            snapshot_days=0,
            insufficient_data=True,
        )

    async def analyze_all_vendors(self) -> dict[str, TemporalEvidence]:
        return {}

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
        return evidence
