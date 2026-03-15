"""Narrative Synthesis & Evidence Chain Engine (WS6).

Ties together all reasoning components into sellable intelligence output:
    - Evidence chains: every claim linked to source review/event/metric
    - Narrative synthesis: structured vendor intelligence with archetype context
    - Explainability: audit trail for why a vendor was scored/ranked
    - Rule engine: hard threshold triggers for auto-escalation

This module does NOT call the LLM directly for narrative generation --
that's handled by the existing b2b_churn_intelligence skill + autonomous
task. Instead, this module assembles the structured evidence payload that
feeds the LLM and post-processes results for evidence chain integrity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("atlas.reasoning.narrative")


# ------------------------------------------------------------------
# Evidence chain (WS6D)
# ------------------------------------------------------------------


@dataclass
class EvidenceLink:
    """A single piece of evidence supporting a claim."""

    source_type: str  # "review", "snapshot", "displacement", "change_event", "archetype", "temporal"
    source_id: str = ""  # review UUID, snapshot date, etc.
    metric: str = ""
    value: Any = None
    context: str = ""  # human-readable snippet


@dataclass
class EvidenceChain:
    """An ordered chain of evidence supporting a conclusion."""

    claim: str
    confidence: float = 0.0
    links: list[EvidenceLink] = field(default_factory=list)
    archetype: str = ""
    vendor_name: str = ""

    @property
    def source_count(self) -> int:
        """Number of distinct source types in the chain."""
        return len({link.source_type for link in self.links})

    @property
    def is_well_supported(self) -> bool:
        """True if evidence comes from 2+ source types."""
        return self.source_count >= 2


# ------------------------------------------------------------------
# Narrative payload (WS6C)
# ------------------------------------------------------------------


@dataclass
class VendorNarrative:
    """Structured intelligence narrative for a single vendor."""

    vendor_name: str
    archetype: str
    archetype_score: float
    risk_level: str
    executive_summary: str = ""
    evidence_chains: list[EvidenceChain] = field(default_factory=list)
    temporal_context: dict[str, Any] = field(default_factory=dict)
    competitive_context: dict[str, Any] = field(default_factory=dict)
    trigger_events: list[dict[str, Any]] = field(default_factory=list)
    ecosystem_context: dict[str, Any] = field(default_factory=dict)
    falsification_conditions: list[str] = field(default_factory=list)
    uncertainty_sources: list[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Rule engine (WS6B)
# ------------------------------------------------------------------


@dataclass
class ThresholdRule:
    """A hard rule for threshold-based triggers."""

    name: str
    description: str
    metric: str
    operator: str  # "gt", "lt", "gte", "lte", "eq"
    threshold: float
    action: str  # "escalate", "alert", "flag", "auto_intervention"
    priority: str = "medium"  # low, medium, high, critical
    archetype_filter: str | None = None  # only apply to this archetype


# Default threshold rules
THRESHOLD_RULES: list[ThresholdRule] = [
    ThresholdRule(
        name="critical_churn_density",
        description="Churn density exceeds 70% -- critical risk",
        metric="churn_density",
        operator="gte",
        threshold=70.0,
        action="escalate",
        priority="critical",
    ),
    ThresholdRule(
        name="urgency_spike",
        description="Average urgency exceeds 7.0 -- immediate attention",
        metric="avg_urgency",
        operator="gte",
        threshold=7.0,
        action="alert",
        priority="high",
    ),
    ThresholdRule(
        name="mass_displacement",
        description="Displacement edge count >= 5 -- active category disruption",
        metric="displacement_edge_count",
        operator="gte",
        threshold=5,
        action="flag",
        priority="high",
    ),
    ThresholdRule(
        name="positive_collapse",
        description="Positive review % drops below 25% -- trust erosion",
        metric="positive_review_pct",
        operator="lt",
        threshold=25.0,
        action="escalate",
        priority="high",
    ),
    ThresholdRule(
        name="high_intent_surge",
        description="High-intent companies exceed 5 -- sales opportunity",
        metric="high_intent_company_count",
        operator="gte",
        threshold=5,
        action="alert",
        priority="medium",
    ),
    ThresholdRule(
        name="velocity_urgency_acceleration",
        description="Urgency accelerating -- situation worsening",
        metric="accel_avg_urgency",
        operator="gt",
        threshold=0.0,
        action="flag",
        priority="medium",
        archetype_filter=None,  # applies to all
    ),
]


@dataclass
class RuleTriggered:
    """A threshold rule that was triggered by vendor evidence."""

    rule: ThresholdRule
    actual_value: float
    vendor_name: str


# ------------------------------------------------------------------
# Narrative assembly engine
# ------------------------------------------------------------------


class NarrativeEngine:
    """Assembles structured intelligence from all reasoning components."""

    def __init__(self, pool: Any = None):
        self._pool = pool

    def build_vendor_narrative(
        self,
        vendor_name: str,
        *,
        reasoning_result: dict[str, Any] | None = None,
        archetype_match: dict[str, Any] | None = None,
        temporal_dict: dict[str, Any] | None = None,
        competitive_landscape: dict[str, Any] | None = None,
        trigger_events: list[dict[str, Any]] | None = None,
        ecosystem_evidence: dict[str, Any] | None = None,
        snapshot: dict[str, Any] | None = None,
    ) -> VendorNarrative:
        """Assemble a complete vendor narrative from all available evidence.

        This is the main integration point -- it pulls together outputs
        from temporal, archetypes, knowledge graph, triggers, and ecosystem
        into a single structured intelligence payload.
        """
        archetype = ""
        archetype_score = 0.0
        risk_level = "low"
        exec_summary = ""
        falsification = []
        uncertainty = []

        # From stratified reasoner result
        if reasoning_result:
            archetype = reasoning_result.get("archetype", "")
            exec_summary = reasoning_result.get("executive_summary", "")
            risk_level = reasoning_result.get("risk_level", "low")
            falsification = reasoning_result.get("falsification_conditions", [])
            uncertainty = reasoning_result.get("uncertainty_sources", [])

        # From archetype pre-scoring
        if archetype_match:
            if not archetype:
                archetype = archetype_match.get("archetype", "")
            archetype_score = archetype_match.get("signal_score", 0.0)
            if not risk_level or risk_level == "low":
                risk_level = archetype_match.get("risk_level", "low")

        # Build evidence chains
        chains = self._build_evidence_chains(
            vendor_name, archetype, snapshot, temporal_dict,
            competitive_landscape, trigger_events,
        )

        return VendorNarrative(
            vendor_name=vendor_name,
            archetype=archetype,
            archetype_score=archetype_score,
            risk_level=risk_level,
            executive_summary=exec_summary,
            evidence_chains=chains,
            temporal_context=temporal_dict or {},
            competitive_context=competitive_landscape or {},
            trigger_events=trigger_events or [],
            ecosystem_context=ecosystem_evidence or {},
            falsification_conditions=falsification,
            uncertainty_sources=uncertainty,
        )

    def evaluate_rules(
        self,
        vendor_name: str,
        evidence: dict[str, Any],
        archetype: str = "",
    ) -> list[RuleTriggered]:
        """Evaluate threshold rules against vendor evidence."""
        triggered = []
        for rule in THRESHOLD_RULES:
            # Skip if rule is filtered to a specific archetype
            if rule.archetype_filter and rule.archetype_filter != archetype:
                continue

            val = evidence.get(rule.metric)
            if val is None:
                continue

            try:
                val_f = float(val)
            except (ValueError, TypeError):
                continue

            if _check_threshold(val_f, rule.operator, rule.threshold):
                triggered.append(RuleTriggered(
                    rule=rule,
                    actual_value=val_f,
                    vendor_name=vendor_name,
                ))

        return triggered

    def build_explainability(self, narrative: VendorNarrative) -> dict[str, Any]:
        """Generate an audit trail explaining the intelligence output.

        Returns a structured dict answering:
        - Why was this archetype assigned?
        - What evidence supports the risk level?
        - What would change the conclusion?
        - How confident are we and why?
        """
        explain: dict[str, Any] = {
            "vendor": narrative.vendor_name,
            "archetype": narrative.archetype,
            "archetype_score": narrative.archetype_score,
            "risk_level": narrative.risk_level,
        }

        # Evidence summary
        well_supported = [c for c in narrative.evidence_chains if c.is_well_supported]
        explain["evidence_summary"] = {
            "total_chains": len(narrative.evidence_chains),
            "well_supported": len(well_supported),
            "source_types": list({
                link.source_type
                for chain in narrative.evidence_chains
                for link in chain.links
            }),
        }

        # Confidence factors
        factors = []
        if narrative.temporal_context.get("snapshot_days", 0) < 7:
            factors.append("limited_temporal_data")
        if narrative.temporal_context.get("temporal_status") == "insufficient_data":
            factors.append("no_velocity_data")
        if not narrative.competitive_context.get("losing_to"):
            factors.append("no_displacement_data")
        if not narrative.trigger_events:
            factors.append("no_trigger_events")
        if len(well_supported) < 2:
            factors.append("thin_evidence")

        explain["confidence_factors"] = factors
        explain["confidence_assessment"] = (
            "high" if len(factors) == 0
            else "medium" if len(factors) <= 2
            else "low"
        )

        # Falsification
        explain["what_would_change_conclusion"] = narrative.falsification_conditions
        explain["uncertainty_sources"] = narrative.uncertainty_sources

        return explain

    # ------------------------------------------------------------------
    # Evidence chain construction
    # ------------------------------------------------------------------

    def _build_evidence_chains(
        self,
        vendor_name: str,
        archetype: str,
        snapshot: dict[str, Any] | None,
        temporal: dict[str, Any] | None,
        competitive: dict[str, Any] | None,
        triggers: list[dict[str, Any]] | None,
    ) -> list[EvidenceChain]:
        """Build evidence chains from all available data sources."""
        chains = []

        # Chain 1: Churn density evidence
        if snapshot:
            churn = snapshot.get("churn_density")
            if churn is not None:
                chain = EvidenceChain(
                    claim=f"{vendor_name} shows {churn}% churn signal density",
                    confidence=0.8,
                    archetype=archetype,
                    vendor_name=vendor_name,
                )
                chain.links.append(EvidenceLink(
                    source_type="snapshot",
                    metric="churn_density",
                    value=churn,
                    context=f"Latest snapshot: churn_density={churn}%",
                ))
                if temporal and temporal.get("velocity_churn_density"):
                    vel = temporal["velocity_churn_density"]
                    chain.links.append(EvidenceLink(
                        source_type="temporal",
                        metric="velocity_churn_density",
                        value=vel,
                        context=f"Churn density velocity: {vel}/day",
                    ))
                chains.append(chain)

        # Chain 2: Competitive pressure
        if competitive and competitive.get("losing_to"):
            losers = competitive["losing_to"]
            top = losers[0] if losers else {}
            chain = EvidenceChain(
                claim=f"{vendor_name} losing customers to {len(losers)} competitors",
                confidence=min(0.9, 0.5 + len(losers) * 0.1),
                archetype=archetype,
                vendor_name=vendor_name,
            )
            for l in losers[:5]:
                chain.links.append(EvidenceLink(
                    source_type="displacement",
                    metric="switched_to",
                    value=l.get("mentions", 0),
                    context=f"-> {l.get('name', '?')} ({l.get('mentions', 0)} mentions, driver: {l.get('driver', '?')})",
                ))
            chains.append(chain)

        # Chain 3: Urgency trend
        if snapshot and temporal:
            urgency = snapshot.get("avg_urgency")
            vel = temporal.get("velocity_avg_urgency")
            if urgency is not None:
                chain = EvidenceChain(
                    claim=f"Customer urgency at {urgency}" + (
                        f" and {'rising' if vel and vel > 0 else 'falling'}"
                        if vel is not None else ""
                    ),
                    confidence=0.7,
                    archetype=archetype,
                    vendor_name=vendor_name,
                )
                chain.links.append(EvidenceLink(
                    source_type="snapshot",
                    metric="avg_urgency",
                    value=urgency,
                    context=f"Average urgency score: {urgency}",
                ))
                if vel is not None:
                    chain.links.append(EvidenceLink(
                        source_type="temporal",
                        metric="velocity_avg_urgency",
                        value=vel,
                        context=f"Urgency velocity: {vel}/day",
                    ))
                chains.append(chain)

        # Chain 4: Trigger events
        if triggers:
            for trig in triggers[:3]:
                chain = EvidenceChain(
                    claim=f"Trigger event: {trig.get('trigger_type', trig.get('description', 'unknown'))}",
                    confidence=trig.get("confidence", 0.5),
                    archetype=archetype,
                    vendor_name=vendor_name,
                )
                chain.links.append(EvidenceLink(
                    source_type="change_event",
                    metric="trigger",
                    value=trig.get("trigger_type", ""),
                    context=trig.get("description", ""),
                ))
                chains.append(chain)

        return chains

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def to_intelligence_payload(narrative: VendorNarrative) -> dict[str, Any]:
        """Convert a VendorNarrative into the payload format expected by
        the b2b_churn_intelligence skill and report generation."""
        return {
            "vendor_name": narrative.vendor_name,
            "archetype": narrative.archetype,
            "archetype_score": narrative.archetype_score,
            "risk_level": narrative.risk_level,
            "executive_summary": narrative.executive_summary,
            "evidence_chain_count": len(narrative.evidence_chains),
            "well_supported_claims": sum(
                1 for c in narrative.evidence_chains if c.is_well_supported
            ),
            "temporal": {
                k: v for k, v in narrative.temporal_context.items()
                if k.startswith("velocity_") or k in ("snapshot_days", "temporal_status")
            },
            "competitive": {
                "losing_to_count": len(narrative.competitive_context.get("losing_to", [])),
                "winning_from_count": len(narrative.competitive_context.get("winning_from", [])),
            },
            "trigger_event_count": len(narrative.trigger_events),
            "ecosystem": {
                "market_structure": narrative.ecosystem_context.get("market_structure", ""),
                "category_hhi": narrative.ecosystem_context.get("hhi", 0),
            },
            "falsification_conditions": narrative.falsification_conditions[:5],
            "uncertainty_sources": narrative.uncertainty_sources[:5],
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _check_threshold(value: float, operator: str, threshold: float) -> bool:
    """Evaluate a threshold comparison."""
    if operator == "gt":
        return value > threshold
    if operator == "lt":
        return value < threshold
    if operator == "gte":
        return value >= threshold
    if operator == "lte":
        return value <= threshold
    if operator == "eq":
        return abs(value - threshold) < 0.001
    return False
