"""Trigger Event Detection (WS4).

External signal detection that feeds causal reasoning. Defines event types,
correlates trigger events with review spike timestamps, and computes
composite risk scores.

Event taxonomy:
    funding_round      -- Competitor raises capital -> likely product investment
    leadership_change  -- VP/C-suite change -> strategy shift signal
    compliance_update  -- Vendor achieves/loses certification
    product_launch     -- Feature release or new product version
    pricing_change     -- Price increase/decrease or plan restructure
    acquisition        -- Vendor acquires or gets acquired
    outage_incident    -- Major downtime or security breach
    contract_cycle     -- Enterprise contract renewal windows

Events are stored in b2b_change_events (Postgres) and correlated with
review velocity spikes from the temporal engine. Composite triggers
combine multiple events for risk scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger("atlas.reasoning.trigger_events")


class TriggerType(str, Enum):
    """Canonical trigger event types."""

    FUNDING_ROUND = "funding_round"
    LEADERSHIP_CHANGE = "leadership_change"
    COMPLIANCE_UPDATE = "compliance_update"
    PRODUCT_LAUNCH = "product_launch"
    PRICING_CHANGE = "pricing_change"
    ACQUISITION = "acquisition"
    OUTAGE_INCIDENT = "outage_incident"
    CONTRACT_CYCLE = "contract_cycle"


@dataclass
class TriggerEventDef:
    """Definition of a trigger type with its expected impacts."""

    trigger_type: TriggerType
    description: str
    expected_review_lag_days: int  # how many days before reviews reflect this
    archetype_affinity: list[str]  # which archetypes this event amplifies
    urgency_boost: float  # how much this boosts urgency assessment (0-2)
    velocity_metrics: list[str]  # which temporal metrics to check for correlation


# ------------------------------------------------------------------
# Event taxonomy definitions (WS4A)
# ------------------------------------------------------------------

EVENT_TAXONOMY: dict[TriggerType, TriggerEventDef] = {
    TriggerType.FUNDING_ROUND: TriggerEventDef(
        trigger_type=TriggerType.FUNDING_ROUND,
        description="Competitor raises capital, likely product investment incoming",
        expected_review_lag_days=60,
        archetype_affinity=["category_disruption", "feature_gap"],
        urgency_boost=0.5,
        velocity_metrics=["competitor_count", "displacement_edge_count"],
    ),
    TriggerType.LEADERSHIP_CHANGE: TriggerEventDef(
        trigger_type=TriggerType.LEADERSHIP_CHANGE,
        description="VP/C-suite departure or hire signals strategy shift",
        expected_review_lag_days=90,
        archetype_affinity=["leadership_redesign", "acquisition_decay"],
        urgency_boost=0.3,
        velocity_metrics=["avg_urgency", "positive_review_pct"],
    ),
    TriggerType.COMPLIANCE_UPDATE: TriggerEventDef(
        trigger_type=TriggerType.COMPLIANCE_UPDATE,
        description="Vendor achieves or loses a compliance certification",
        expected_review_lag_days=30,
        archetype_affinity=["compliance_gap"],
        urgency_boost=1.0,
        velocity_metrics=["high_intent_company_count", "churn_density"],
    ),
    TriggerType.PRODUCT_LAUNCH: TriggerEventDef(
        trigger_type=TriggerType.PRODUCT_LAUNCH,
        description="Major feature release or new product version",
        expected_review_lag_days=14,
        archetype_affinity=["feature_gap", "leadership_redesign"],
        urgency_boost=0.5,
        velocity_metrics=["positive_review_pct", "pain_count"],
    ),
    TriggerType.PRICING_CHANGE: TriggerEventDef(
        trigger_type=TriggerType.PRICING_CHANGE,
        description="Price increase, plan restructure, or free tier removal",
        expected_review_lag_days=7,
        archetype_affinity=["pricing_shock"],
        urgency_boost=1.5,
        velocity_metrics=["avg_urgency", "competitor_count", "recommend_ratio"],
    ),
    TriggerType.ACQUISITION: TriggerEventDef(
        trigger_type=TriggerType.ACQUISITION,
        description="Vendor acquires or gets acquired by another company",
        expected_review_lag_days=30,
        archetype_affinity=["acquisition_decay"],
        urgency_boost=1.0,
        velocity_metrics=["positive_review_pct", "churn_density", "avg_urgency"],
    ),
    TriggerType.OUTAGE_INCIDENT: TriggerEventDef(
        trigger_type=TriggerType.OUTAGE_INCIDENT,
        description="Major downtime, data breach, or security incident",
        expected_review_lag_days=3,
        archetype_affinity=["support_collapse", "integration_break"],
        urgency_boost=2.0,
        velocity_metrics=["avg_urgency", "churn_density", "high_intent_company_count"],
    ),
    TriggerType.CONTRACT_CYCLE: TriggerEventDef(
        trigger_type=TriggerType.CONTRACT_CYCLE,
        description="Enterprise annual contract renewal window approaching",
        expected_review_lag_days=0,
        archetype_affinity=["pricing_shock", "feature_gap", "support_collapse"],
        urgency_boost=0.8,
        velocity_metrics=["high_intent_company_count"],
    ),
}


# ------------------------------------------------------------------
# Trigger event record
# ------------------------------------------------------------------


@dataclass
class TriggerEvent:
    """A detected trigger event for a vendor."""

    vendor_name: str
    trigger_type: TriggerType
    event_date: date
    source: str = ""  # where was this detected
    description: str = ""
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventCorrelation:
    """Correlation between a trigger event and review velocity changes."""

    event: TriggerEvent
    correlated_metric: str
    velocity_before: float | None
    velocity_after: float | None
    velocity_change: float | None
    lag_days: int = 0
    correlation_strength: str = ""  # "strong", "moderate", "weak", "none"


@dataclass
class CompositeRiskScore:
    """Risk score from multiple correlated trigger events."""

    vendor_name: str
    events: list[TriggerEvent]
    correlations: list[EventCorrelation]
    base_risk: float  # from archetype scoring
    event_boost: float  # accumulated urgency boosts
    composite_risk: float  # final 0-1 score
    risk_level: str  # low/medium/high/critical
    explanation: str = ""


# ------------------------------------------------------------------
# Event-signal correlation engine (WS4C)
# ------------------------------------------------------------------


class TriggerCorrelator:
    """Correlates trigger events with temporal velocity changes."""

    def __init__(self, pool: Any):
        self._pool = pool

    async def correlate_event(
        self,
        event: TriggerEvent,
        temporal_dict: dict[str, Any] | None = None,
    ) -> EventCorrelation | None:
        """Check if a trigger event correlates with a velocity spike.

        Looks at the expected metrics for this event type and checks
        if velocity changed in the expected direction within the lag window.
        """
        definition = EVENT_TAXONOMY.get(event.trigger_type)
        if not definition:
            return None

        if not temporal_dict:
            return None

        # Check each expected velocity metric
        best_correlation = None
        for metric in definition.velocity_metrics:
            vel_key = f"velocity_{metric}"
            vel = temporal_dict.get(vel_key)
            if vel is None:
                continue

            try:
                vel_f = float(vel)
            except (ValueError, TypeError):
                continue

            # Determine expected direction based on archetype affinity
            # Most trigger events cause urgency/churn to increase
            expected_positive = metric in (
                "avg_urgency", "churn_density", "competitor_count",
                "displacement_edge_count", "high_intent_company_count", "pain_count",
            )

            if expected_positive and vel_f > 0:
                strength = _classify_correlation_strength(vel_f, metric)
            elif not expected_positive and vel_f < 0:
                strength = _classify_correlation_strength(abs(vel_f), metric)
            else:
                strength = "none"

            if strength != "none":
                corr = EventCorrelation(
                    event=event,
                    correlated_metric=metric,
                    velocity_before=None,
                    velocity_after=vel_f,
                    velocity_change=vel_f,
                    correlation_strength=strength,
                )
                if best_correlation is None or _strength_rank(strength) > _strength_rank(
                    best_correlation.correlation_strength
                ):
                    best_correlation = corr

        return best_correlation

    async def compute_composite_risk(
        self,
        vendor_name: str,
        events: list[TriggerEvent],
        archetype_score: float = 0.0,
        temporal_dict: dict[str, Any] | None = None,
    ) -> CompositeRiskScore:
        """Compute composite risk from multiple trigger events + archetype score.

        Formula: composite = min(1.0, base_risk + sum(event_boosts) * correlation_factor)
        """
        correlations = []
        total_boost = 0.0

        for event in events:
            definition = EVENT_TAXONOMY.get(event.trigger_type)
            if not definition:
                continue

            corr = await self.correlate_event(event, temporal_dict)
            if corr:
                correlations.append(corr)
                # Boost is scaled by correlation strength
                strength_mult = {"strong": 1.0, "moderate": 0.6, "weak": 0.3, "none": 0.0}
                mult = strength_mult.get(corr.correlation_strength, 0.0)
                total_boost += definition.urgency_boost * mult * event.confidence
            else:
                # Event exists but no temporal correlation yet -- still counts for half
                total_boost += definition.urgency_boost * 0.5 * event.confidence

        # Composite formula
        base = archetype_score
        composite = min(1.0, base + total_boost * 0.15)  # scale boost to 0-1 range

        risk_level = _risk_level_from_score(composite)

        # Build explanation
        parts = []
        for event in events:
            parts.append(f"{event.trigger_type.value} ({event.event_date})")
        if correlations:
            strong = [c for c in correlations if c.correlation_strength == "strong"]
            if strong:
                parts.append(
                    f"{len(strong)} strong velocity correlation(s)"
                )
        explanation = "; ".join(parts)

        return CompositeRiskScore(
            vendor_name=vendor_name,
            events=events,
            correlations=correlations,
            base_risk=base,
            event_boost=total_boost,
            composite_risk=round(composite, 3),
            risk_level=risk_level,
            explanation=explanation,
        )

    async def load_recent_events(
        self, vendor_name: str, days: int = 90,
    ) -> list[TriggerEvent]:
        """Load trigger events from b2b_change_events for a vendor."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        rows = await self._pool.fetch(
            """
            SELECT event_type, event_date, direction, created_at,
                   vendor_name
            FROM b2b_change_events
            WHERE LOWER(vendor_name) = LOWER($1)
              AND created_at >= $2
            ORDER BY created_at DESC
            """,
            vendor_name,
            cutoff,
        )

        events = []
        for row in rows:
            event_type_str = row["event_type"] or ""
            trigger_type = _map_event_type(event_type_str)
            if trigger_type:
                events.append(TriggerEvent(
                    vendor_name=row["vendor_name"],
                    trigger_type=trigger_type,
                    event_date=row["event_date"] or row["created_at"].date()
                    if row["created_at"]
                    else date.today(),
                    source="b2b_change_events",
                    description=event_type_str,
                    confidence=0.7,
                ))

        return events


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def get_event_taxonomy() -> dict[str, TriggerEventDef]:
    """Return the full event taxonomy as a plain dict (for serialization)."""
    return {k.value: v for k, v in EVENT_TAXONOMY.items()}


def get_archetype_triggers(archetype_name: str) -> list[TriggerType]:
    """Return trigger types that amplify a given archetype."""
    triggers = []
    for trigger_type, definition in EVENT_TAXONOMY.items():
        if archetype_name in definition.archetype_affinity:
            triggers.append(trigger_type)
    return triggers


def _classify_correlation_strength(velocity: float, metric: str) -> str:
    """Classify velocity magnitude as correlation strength."""
    # Thresholds vary by metric type
    thresholds = {
        "avg_urgency": (0.05, 0.15, 0.3),
        "churn_density": (0.02, 0.05, 0.1),
        "positive_review_pct": (0.5, 1.5, 3.0),
        "recommend_ratio": (0.01, 0.03, 0.05),
        "competitor_count": (0.05, 0.15, 0.3),
        "displacement_edge_count": (0.05, 0.1, 0.2),
        "high_intent_company_count": (0.05, 0.15, 0.3),
        "pain_count": (0.1, 0.3, 0.5),
    }
    weak_t, mod_t, strong_t = thresholds.get(metric, (0.05, 0.15, 0.3))
    abs_vel = abs(velocity)

    if abs_vel >= strong_t:
        return "strong"
    elif abs_vel >= mod_t:
        return "moderate"
    elif abs_vel >= weak_t:
        return "weak"
    return "none"


def _strength_rank(strength: str) -> int:
    """Numeric rank for correlation strength comparison."""
    return {"strong": 3, "moderate": 2, "weak": 1, "none": 0}.get(strength, 0)


def _risk_level_from_score(score: float) -> str:
    """Map composite score to risk level."""
    if score >= 0.8:
        return "critical"
    if score >= 0.6:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def _map_event_type(event_type_str: str) -> TriggerType | None:
    """Map a free-form event_type string to a TriggerType enum."""
    et = event_type_str.lower()

    if any(kw in et for kw in ["funding", "raise", "series", "investment"]):
        return TriggerType.FUNDING_ROUND
    if any(kw in et for kw in ["leadership", "ceo", "cto", "vp", "hire", "depart"]):
        return TriggerType.LEADERSHIP_CHANGE
    if any(kw in et for kw in ["compliance", "soc2", "gdpr", "hipaa", "certif"]):
        return TriggerType.COMPLIANCE_UPDATE
    if any(kw in et for kw in ["launch", "release", "feature", "version", "update", "surge", "competitor"]):
        return TriggerType.PRODUCT_LAUNCH
    if any(kw in et for kw in ["price", "pricing", "plan change", "tier"]):
        return TriggerType.PRICING_CHANGE
    if any(kw in et for kw in ["acqui", "merger", "buyout", "bought"]):
        return TriggerType.ACQUISITION
    if any(kw in et for kw in ["outage", "downtime", "breach", "incident"]):
        return TriggerType.OUTAGE_INCIDENT
    if any(kw in et for kw in ["contract", "renewal", "annual"]):
        return TriggerType.CONTRACT_CYCLE

    return None
