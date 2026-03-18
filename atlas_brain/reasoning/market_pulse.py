"""Market Pulse Reasoner.

Analyzes aggregated signals across all vendors in a category to determine
the structural "Market Regime". This provides top-down context that differentiates
systemic churn (e.g., "The whole category is being disrupted") from
idiosyncratic churn (e.g., "Vendor A messed up").

Input: List of TemporalEvidence for all vendors in a category.
Output: MarketRegime (regime_type, confidence, outlier_vendors).
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any

from .temporal import TemporalEvidence

logger = logging.getLogger("atlas.reasoning.market_pulse")


@dataclass
class MarketRegime:
    category: str
    regime_type: str  # e.g., "stable", "high_churn", "entrenchment", "disruption"
    confidence: float
    avg_churn_velocity: float
    avg_price_pressure: float
    outlier_vendors: list[str]  # Vendors bucking the trend
    narrative: str


class MarketPulseReasoner:
    """Aggregates vendor signals to determine category-level dynamics."""

    def analyze_category(
        self,
        category: str,
        vendor_evidence: list[TemporalEvidence],
    ) -> MarketRegime:
        """Determine the market regime for a category."""
        if not vendor_evidence:
            return MarketRegime(
                category=category,
                regime_type="unknown",
                confidence=0.0,
                avg_churn_velocity=0.0,
                avg_price_pressure=0.0,
                outlier_vendors=[],
                narrative="Insufficient data.",
            )

        # 1. Aggregate Core Metrics
        churn_velocities = []
        price_pressures = []
        support_velocities = []
        feature_velocities = []

        for ve in vendor_evidence:
            for v in ve.velocities:
                if v.metric == "churn_density":
                    churn_velocities.append(v.velocity)
                elif v.metric == "avg_urgency":
                    price_pressures.append(v.velocity)
                elif v.metric == "support_sentiment":
                    support_velocities.append(v.velocity)
                elif v.metric == "new_feature_velocity":
                    feature_velocities.append(v.velocity)

        avg_churn_vel = statistics.mean(churn_velocities) if churn_velocities else 0.0
        avg_pressure = statistics.mean(price_pressures) if price_pressures else 0.0
        avg_support_vel = statistics.mean(support_velocities) if support_velocities else 0.0
        avg_feature_vel = statistics.mean(feature_velocities) if feature_velocities else 0.0
        vendor_count = len(vendor_evidence)

        # 2. Determine Regime
        regime = "stable"
        confidence = 0.5
        narrative = "The market is relatively stable."

        if avg_churn_vel > 0.1:
            regime = "high_churn"
            narrative = "Systemic churn across the category; customers are re-evaluating the entire stack."
            confidence = 0.8
        elif avg_churn_vel < -0.05:
            regime = "entrenchment"
            narrative = "Category is hardening; incumbents are locking in customers."
            confidence = 0.7

        if (
            regime == "stable"
            and avg_support_vel < -0.03
            and avg_feature_vel > 0.03
        ):
            regime = "disruption"
            narrative = (
                "Vendors are shipping quickly while support sentiment erodes, "
                "suggesting an unstable transition phase across the category."
            )
            confidence = 0.65

        if len(churn_velocities) > 3:
            stdev_churn = statistics.stdev(churn_velocities)
            if stdev_churn > 0.2 and regime == "stable":
                regime = "disruption"
                narrative = "High variance in vendor performance suggests active disruption."
                confidence = 0.6
            elif stdev_churn > 0.2 and regime == "high_churn" and vendor_count >= 6:
                narrative = (
                    "Systemic churn is rising, but unevenly enough to suggest "
                    "active share shifts rather than a uniform market shock."
                )
                confidence = 0.85

        if regime == "stable" and vendor_count >= 8 and avg_pressure > 0.05:
            narrative = "A crowded category with rising urgency suggests buyers are actively re-evaluating options."
            confidence = 0.55

        if avg_feature_vel > 0.05 and avg_support_vel < 0:
            narrative = f"{narrative} Product velocity is increasing while support sentiment trends softer."

        # 3. Identify Outliers
        outliers = []
        if churn_velocities and len(churn_velocities) > 2:
            mean_v = statistics.mean(churn_velocities)
            std_v = statistics.stdev(churn_velocities) if len(churn_velocities) > 1 else 0.1
            
            for ve in vendor_evidence:
                v_val = 0.0
                for v in ve.velocities:
                    if v.metric == "churn_density":
                        v_val = v.velocity
                        break
                
                z = (v_val - mean_v) / (std_v + 1e-6)
                if abs(z) > 1.5:
                    outliers.append(ve.vendor_name)

        return MarketRegime(
            category=category,
            regime_type=regime,
            confidence=confidence,
            avg_churn_velocity=avg_churn_vel,
            avg_price_pressure=avg_pressure,
            outlier_vendors=outliers,
            narrative=narrative,
        )
