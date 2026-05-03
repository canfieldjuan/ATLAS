"""Churn Archetype Definitions and Signal Matching.

Defines 10 canonical churn archetypes with signal signatures. The scorer
evaluates vendor evidence (including temporal data) against each
archetype's expected signal profile and returns ranked matches.

This is a pure-data pre-filter -- it does NOT call the LLM. The ranked
scores feed into synthesis-first vendor reasoning as context, and enable
vendor-state updates without LLM calls.

Module layout:

  - ``SignalRule``, ``ArchetypeProfile`` -- catalog data structures
    (frozen dataclasses; immutable by contract).
  - ``ARCHETYPES`` -- the canonical 10-archetype catalog.
  - ``_ArchetypeMatchInternal`` -- rich internal scoring result. Atlas
    callers and the in-module scoring functions return this. The public
    contract type ``ArchetypeMatch`` lives in
    ``extracted_reasoning_core.types``; this module provides the
    ``_to_public_match`` adapter that produces it.
  - ``score_evidence``, ``best_match``, ``top_matches`` -- internal
    scoring helpers returning ``_ArchetypeMatchInternal``.
  - ``enrich_evidence_with_archetypes`` -- internal helper that
    returns the input evidence dict augmented with an
    ``archetype_scores`` key (top 3 matches, serialized).
  - The public boundary ``extracted_reasoning_core.api.score_archetypes``
    is currently a stub raising ``NotImplementedError``; it will be
    wired in a follow-up PR-C1 slice and will call ``score_evidence``
    + ``_to_public_match`` to return ``Sequence[ArchetypeMatch]``.

Archetypes:
    pricing_shock       -- Sudden price increase -> complaint spike -> competitor mentions
    feature_gap         -- Competitor launches key feature -> "missing X" reviews surge
    acquisition_decay   -- Post-acquisition quality decline -> support complaints -> churn
    leadership_redesign -- New VP Product -> UI overhaul -> "new interface" complaints
    integration_break   -- API change breaks workflows -> "integration" pain spike
    support_collapse    -- Support quality drop -> response time complaints -> trust erosion
    category_disruption -- New entrant class (AI-native) -> incumbents lose narrative
    compliance_gap      -- Regulatory requirement unmet -> enterprise segments leave
    scale_up_stumble    -- High growth -> support quality / culture degrades
    pivot_abandonment   -- Focus shifts to new segment -> legacy customer neglect
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import ArchetypeMatch

# Minimum score to consider an archetype a plausible match
MATCH_THRESHOLD = 0.25


@dataclass(frozen=True)
class SignalRule:
    """A single signal check within an archetype signature."""

    metric: str        # evidence key to check
    direction: str     # "high", "low", "increasing", "decreasing", "present"
    weight: float = 1.0
    threshold: float | None = None  # numeric threshold (meaning depends on direction)
    pain_keywords: list[str] = field(default_factory=list)  # keywords to match in text fields


@dataclass(frozen=True)
class ArchetypeProfile:
    """Definition of a churn archetype with its expected signal signature."""

    name: str
    description: str
    signals: list[SignalRule]
    typical_risk: str  # "low", "medium", "high", "critical"
    velocity_hints: dict[str, str] = field(default_factory=dict)
    falsification_templates: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _ArchetypeMatchInternal:
    """Rich internal result of scoring a vendor's evidence against an archetype.

    The public contract type ``ArchetypeMatch`` lives in
    ``extracted_reasoning_core.types``; converted via ``_to_public_match``.
    Atlas-side callers continue to consume this richer shape directly.
    """

    archetype: str
    score: float  # 0.0-1.0 weighted match score
    matched_signals: list[str]
    missing_signals: list[str]
    risk_level: str


# ------------------------------------------------------------------
# The 10 canonical archetypes
# ------------------------------------------------------------------

ARCHETYPES: dict[str, ArchetypeProfile] = {
    "pricing_shock": ArchetypeProfile(
        name="pricing_shock",
        description="Sudden price increase -> complaint spike -> competitor mentions",
        typical_risk="high",
        signals=[
            SignalRule("avg_urgency", "high", weight=1.5, threshold=6.0),
            SignalRule("top_pain", "present", weight=2.0,
                       pain_keywords=["price", "pricing", "cost", "expensive", "renewal", "invoice"]),
            SignalRule("competitor_count", "high", weight=1.0, threshold=3),
            SignalRule("recommend_ratio", "low", weight=1.0, threshold=0.4),
            SignalRule("displacement_edge_count", "high", weight=1.2, threshold=2),
            SignalRule("positive_review_pct", "low", weight=0.8, threshold=40.0),
        ],
        velocity_hints={
            "avg_urgency": "increasing",
            "competitor_count": "increasing",
            "recommend_ratio": "decreasing",
        },
        falsification_templates=[
            "Vendor reverts to previous pricing or introduces budget tier",
            "Positive review trend reversal (3+ consecutive weeks improving)",
            "Competitor count stabilizes or decreases",
        ],
    ),
    "feature_gap": ArchetypeProfile(
        name="feature_gap",
        description="Competitor launches key feature -> 'missing X' reviews surge",
        typical_risk="medium",
        signals=[
            SignalRule("top_pain", "present", weight=2.0,
                       pain_keywords=["missing", "lack", "need", "feature", "no support for",
                                      "doesn't have", "wish", "roadmap"]),
            SignalRule("competitor_count", "high", weight=1.5, threshold=2),
            SignalRule("displacement_edge_count", "high", weight=1.5, threshold=2),
            SignalRule("pain_count", "high", weight=1.0, threshold=4),
            SignalRule("recommend_ratio", "low", weight=0.8, threshold=0.5),
        ],
        velocity_hints={
            "competitor_count": "increasing",
            "displacement_edge_count": "increasing",
            "pain_count": "increasing",
        },
        falsification_templates=[
            "Vendor releases the missing feature",
            "Competitor mentions decrease by 50%+",
            "Pain count drops below 2",
        ],
    ),
    "acquisition_decay": ArchetypeProfile(
        name="acquisition_decay",
        description="Post-acquisition quality decline -> support complaints -> churn",
        typical_risk="high",
        signals=[
            SignalRule("positive_review_pct", "low", weight=1.5, threshold=35.0),
            SignalRule("avg_urgency", "high", weight=1.2, threshold=5.5),
            SignalRule("top_pain", "present", weight=2.0,
                       pain_keywords=["acquired", "acquisition", "buyout", "merged",
                                      "new ownership", "quality decline", "worse since"]),
            SignalRule("recommend_ratio", "low", weight=1.0, threshold=0.35),
            SignalRule("churn_density", "high", weight=1.0, threshold=0.6),
        ],
        velocity_hints={
            "positive_review_pct": "decreasing",
            "avg_urgency": "increasing",
            "churn_density": "increasing",
        },
        falsification_templates=[
            "Positive review percentage improves above 50%",
            "Support response time improves to < 4h average",
            "New leadership announces quality investment plan",
        ],
    ),
    "leadership_redesign": ArchetypeProfile(
        name="leadership_redesign",
        description="New VP Product -> UI overhaul -> 'new interface' complaints",
        typical_risk="medium",
        signals=[
            SignalRule("top_pain", "present", weight=2.5,
                       pain_keywords=["redesign", "new ui", "new interface", "update",
                                      "changed layout", "new version", "ui overhaul",
                                      "workflow changed", "moved features"]),
            SignalRule("avg_urgency", "high", weight=1.0, threshold=5.0),
            SignalRule("pain_count", "high", weight=0.8, threshold=3),
            SignalRule("positive_review_pct", "low", weight=1.0, threshold=45.0),
        ],
        velocity_hints={
            "avg_urgency": "increasing",
            "positive_review_pct": "decreasing",
            "pain_count": "increasing",
        },
        falsification_templates=[
            "Vendor reverts UI changes or provides classic mode",
            "Positive review pct recovers above 55%",
            "Pain count related to UI drops below 2",
        ],
    ),
    "integration_break": ArchetypeProfile(
        name="integration_break",
        description="API change breaks workflows -> 'integration' pain spike",
        typical_risk="high",
        signals=[
            SignalRule("top_pain", "present", weight=2.5,
                       pain_keywords=["api", "integration", "webhook", "connector",
                                      "broke", "breaking change", "deprecated",
                                      "incompatible", "migration", "sdk"]),
            SignalRule("avg_urgency", "high", weight=1.5, threshold=6.5),
            SignalRule("high_intent_company_count", "high", weight=1.2, threshold=3),
            SignalRule("churn_density", "high", weight=1.0, threshold=0.5),
        ],
        velocity_hints={
            "avg_urgency": "increasing",
            "churn_density": "increasing",
            "high_intent_company_count": "increasing",
        },
        falsification_templates=[
            "Vendor provides backward-compatible API or migration tooling",
            "Urgency drops below 4.0",
            "Integration-related pain mentions drop by 60%+",
        ],
    ),
    "support_collapse": ArchetypeProfile(
        name="support_collapse",
        description="Support quality drop -> response time complaints -> trust erosion",
        typical_risk="critical",
        signals=[
            SignalRule("top_pain", "present", weight=2.0,
                       pain_keywords=["support", "response time", "ticket", "help desk",
                                      "no response", "slow support", "waiting",
                                      "customer service", "escalation"]),
            SignalRule("avg_urgency", "high", weight=1.5, threshold=6.0),
            SignalRule("positive_review_pct", "low", weight=1.5, threshold=30.0),
            SignalRule("recommend_ratio", "low", weight=1.2, threshold=0.3),
            SignalRule("churn_density", "high", weight=1.0, threshold=0.7),
        ],
        velocity_hints={
            "positive_review_pct": "decreasing",
            "avg_urgency": "increasing",
            "churn_density": "increasing",
            "recommend_ratio": "decreasing",
        },
        falsification_templates=[
            "Support response time improves to < 4h average",
            "Positive review pct recovers above 45%",
            "Support-related complaints drop by 50%+",
        ],
    ),
    "category_disruption": ArchetypeProfile(
        name="category_disruption",
        description="New entrant class (e.g., AI-native) -> incumbents lose narrative",
        typical_risk="high",
        signals=[
            SignalRule("competitor_count", "high", weight=2.0, threshold=5),
            SignalRule("displacement_edge_count", "high", weight=2.0, threshold=4),
            SignalRule("top_pain", "present", weight=1.0,
                       pain_keywords=["alternative", "switch to", "moved to", "replaced by",
                                      "ai-native", "modern", "next-gen", "outdated"]),
            SignalRule("high_intent_company_count", "high", weight=1.5, threshold=5),
            SignalRule("recommend_ratio", "low", weight=0.8, threshold=0.4),
        ],
        velocity_hints={
            "competitor_count": "increasing",
            "displacement_edge_count": "increasing",
            "high_intent_company_count": "increasing",
        },
        falsification_templates=[
            "Vendor launches competitive AI/modern features",
            "Competitor count drops or stabilizes for 4+ weeks",
            "Displacement edge count decreases by 40%+",
        ],
    ),
    "compliance_gap": ArchetypeProfile(
        name="compliance_gap",
        description="Regulatory requirement unmet -> enterprise segments leave",
        typical_risk="critical",
        signals=[
            SignalRule("top_pain", "present", weight=3.0,
                       pain_keywords=["compliance", "gdpr", "hipaa", "soc2", "soc 2",
                                      "iso 27001", "fedramp", "pci", "audit",
                                      "regulation", "security", "data residency",
                                      "encryption", "certification"]),
            SignalRule("high_intent_company_count", "high", weight=1.5, threshold=3),
            SignalRule("avg_urgency", "high", weight=1.0, threshold=5.5),
            SignalRule("churn_density", "high", weight=1.0, threshold=0.5),
        ],
        velocity_hints={
            "high_intent_company_count": "increasing",
            "churn_density": "increasing",
        },
        falsification_templates=[
            "Vendor achieves the missing certification/compliance requirement",
            "Compliance-related complaints drop to zero",
            "Enterprise churn intent stabilizes",
        ],
    ),
    "scale_up_stumble": ArchetypeProfile(
        name="scale_up_stumble",
        description="High growth (hiring/reviews) -> support quality/culture degrades",
        typical_risk="medium",
        signals=[
            SignalRule("employee_growth_rate", "high", weight=2.0, threshold=0.20),  # >20% growth
            SignalRule("support_sentiment", "decreasing_30d", weight=1.5),
            SignalRule("recommend_ratio", "decreasing_30d", weight=1.0),
            SignalRule("pain_count", "high", weight=1.0, threshold=3),
            SignalRule("top_pain", "present", weight=1.0,
                        pain_keywords=["growing pains", "used to be better", "new reps", 
                                       "training", "knowledge gap", "slow response"]),
        ],
        velocity_hints={
            "pain_count": "increasing",
            "recommend_ratio": "decreasing",
        },
        falsification_templates=[
            "Support sentiment stabilizes for 30+ days",
            "Hiring pace normalizes (<5% quarterly)",
            "Training investment announced",
        ],
    ),
    "pivot_abandonment": ArchetypeProfile(
        name="pivot_abandonment",
        description="Focus shifts to new segment/product -> legacy customer neglect",
        typical_risk="high",
        signals=[
            SignalRule("new_feature_velocity", "high", weight=1.5, threshold=0.5),
            SignalRule("legacy_support_score", "low", weight=1.5, threshold=0.4),
            SignalRule("churn_density", "increasing_30d", weight=2.0),
            SignalRule("top_pain", "present", weight=1.5,
                        pain_keywords=["legacy", "ignored", "forgotten", "deprecated", 
                                       "forced migration", "sunset", "old version"]),
        ],
        velocity_hints={
            "churn_density": "increasing",
        },
        falsification_templates=[
            "Vendor recommits to legacy product roadmap",
            "Migration tools provided free of charge",
            "Churn density stabilizes",
        ],
    ),
}


# ------------------------------------------------------------------
# Scoring engine
# ------------------------------------------------------------------


def score_evidence(
    evidence: dict[str, Any],
    temporal: dict[str, Any] | None = None,
) -> list[_ArchetypeMatchInternal]:
    """Score vendor evidence against all archetypes.

    *evidence*: flat dict from b2b_vendor_snapshots (and enriched fields).
    *temporal*: output of TemporalEngine.to_evidence_dict() -- velocity/accel/anomaly data.

    Returns list of `_ArchetypeMatchInternal` sorted by score descending.
    Public callers should use ``extracted_reasoning_core.api.score_archetypes``,
    which returns the public ``ArchetypeMatch`` shape after conversion.
    """
    merged = {**evidence}
    if temporal:
        merged.update(temporal)

    results = []
    for name, profile in ARCHETYPES.items():
        match = _score_archetype(profile, merged)
        results.append(match)

    results.sort(key=lambda m: m.score, reverse=True)
    return results


def best_match(
    evidence: dict[str, Any],
    temporal: dict[str, Any] | None = None,
) -> _ArchetypeMatchInternal | None:
    """Return the highest-scoring archetype above threshold, or None."""
    matches = score_evidence(evidence, temporal)
    if matches and matches[0].score >= MATCH_THRESHOLD:
        return matches[0]
    return None


def top_matches(
    evidence: dict[str, Any],
    temporal: dict[str, Any] | None = None,
    *,
    limit: int = 3,
) -> list[_ArchetypeMatchInternal]:
    """Return the top N archetype matches above threshold."""
    matches = score_evidence(evidence, temporal)
    return [m for m in matches[:limit] if m.score >= MATCH_THRESHOLD]


def get_archetype(name: str) -> ArchetypeProfile | None:
    """Lookup an archetype by name."""
    return ARCHETYPES.get(name)


def get_falsification_conditions(archetype_name: str) -> list[str]:
    """Get falsification condition templates for a given archetype."""
    profile = ARCHETYPES.get(archetype_name)
    return list(profile.falsification_templates) if profile else []


# ------------------------------------------------------------------
# Internal scoring
# ------------------------------------------------------------------


def _score_archetype(profile: ArchetypeProfile, evidence: dict[str, Any]) -> _ArchetypeMatchInternal:
    """Score evidence against a single archetype profile."""
    total_weight = sum(s.weight for s in profile.signals)
    if total_weight == 0:
        return _ArchetypeMatchInternal(
            archetype=profile.name,
            score=0.0,
            matched_signals=[],
            missing_signals=[],
            risk_level=profile.typical_risk,
        )

    weighted_score = 0.0
    matched = []
    missing = []

    for rule in profile.signals:
        hit = _evaluate_rule(rule, evidence)
        if hit:
            weighted_score += rule.weight
            matched.append(rule.metric)
        else:
            missing.append(rule.metric)

    # Velocity bonus: if temporal data shows expected direction, boost score
    velocity_bonus = _velocity_bonus(profile, evidence)
    weighted_score += velocity_bonus

    # Anomaly bonus: if z-score anomalies overlap with archetype signals
    anomaly_bonus = _anomaly_bonus(profile, evidence)
    weighted_score += anomaly_bonus

    # Normalize to 0.0-1.0
    max_possible = total_weight + len(profile.velocity_hints) * 0.3 + 0.5
    score = min(weighted_score / max_possible, 1.0)

    # Determine risk level from score
    risk = _risk_from_score(score, profile.typical_risk)

    return _ArchetypeMatchInternal(
        archetype=profile.name,
        score=round(score, 3),
        matched_signals=matched,
        missing_signals=missing,
        risk_level=risk,
    )


def _evaluate_rule(rule: SignalRule, evidence: dict[str, Any]) -> bool:
    """Check if a single signal rule matches the evidence."""
    # Text-based keyword matching (for top_pain and similar text fields)
    if rule.pain_keywords:
        text_fields = ["top_pain", "top_competitor", "pain_summary"]
        combined = ""
        for f in text_fields:
            val = evidence.get(f)
            if val is not None:
                combined += " " + str(val).lower()
        # Also check list fields (pain_categories, etc.)
        for key in evidence:
            if "pain" in key or "complaint" in key:
                val = evidence[key]
                if isinstance(val, list):
                    combined += " " + " ".join(str(v).lower() for v in val)
        if combined.strip():
            if any(kw in combined for kw in rule.pain_keywords):
                return True
        return False

    if rule.direction == "present":
        return rule.metric in evidence

    # Value-based checks require the metric to be present and numeric
    if rule.direction in ("high", "low"):
        val = evidence.get(rule.metric)
        if val is None:
            return False
        try:
            val_f = float(val)
        except (ValueError, TypeError):
            return False

        if rule.direction == "high":
            return val_f >= (rule.threshold if rule.threshold is not None else 0)
        elif rule.direction == "low":
            return val_f <= (rule.threshold if rule.threshold is not None else float("inf"))

    # Velocity/Trend checks look for derived fields. Coercion is guarded
    # consistently with the high/low branch above: non-numeric values
    # treated as a non-match rather than aborting scoring for the vendor.
    elif rule.direction == "increasing":
        vel = evidence.get(f"velocity_{rule.metric}")
        if vel is None:
            return False
        try:
            return float(vel) > 0
        except (ValueError, TypeError):
            return False
    elif rule.direction == "decreasing":
        vel = evidence.get(f"velocity_{rule.metric}")
        if vel is None:
            return False
        try:
            return float(vel) < 0
        except (ValueError, TypeError):
            return False
    elif rule.direction == "increasing_30d":
        slope = evidence.get(f"trend_30d_{rule.metric}")
        if slope is None:
            return False
        try:
            return float(slope) > 0
        except (ValueError, TypeError):
            return False
    elif rule.direction == "decreasing_30d":
        slope = evidence.get(f"trend_30d_{rule.metric}")
        if slope is None:
            return False
        try:
            return float(slope) < 0
        except (ValueError, TypeError):
            return False

    return False


def _velocity_bonus(profile: ArchetypeProfile, evidence: dict[str, Any]) -> float:
    """Bonus score for velocity signals matching expected directions."""
    bonus = 0.0
    for metric, expected_dir in profile.velocity_hints.items():
        vel_key = f"velocity_{metric}"
        vel = evidence.get(vel_key)
        if vel is None:
            continue
        try:
            vel_f = float(vel)
        except (ValueError, TypeError):
            continue

        if expected_dir == "increasing" and vel_f > 0:
            bonus += 0.3
        elif expected_dir == "decreasing" and vel_f < 0:
            bonus += 0.3

        # Extra bonus for acceleration in the expected direction
        accel_key = f"accel_{metric}"
        accel = evidence.get(accel_key)
        if accel is not None:
            try:
                accel_f = float(accel)
                if expected_dir == "increasing" and accel_f > 0:
                    bonus += 0.15
                elif expected_dir == "decreasing" and accel_f < 0:
                    bonus += 0.15
            except (ValueError, TypeError):
                pass

    return bonus


def _anomaly_bonus(profile: ArchetypeProfile, evidence: dict[str, Any]) -> float:
    """Bonus if z-score anomalies align with the archetype's signal metrics."""
    anomalies = evidence.get("anomalies")
    if not anomalies or not isinstance(anomalies, list):
        return 0.0

    archetype_metrics = {s.metric for s in profile.signals if not s.pain_keywords}
    bonus = 0.0
    for anom in anomalies:
        if not isinstance(anom, dict):
            continue
        metric = anom.get("metric", "")
        if metric not in archetype_metrics:
            continue
        # z_score may arrive as a string from JSON payloads; coerce
        # defensively. Skip the anomaly on conversion failure rather
        # than aborting the bonus calculation for the vendor.
        try:
            z = abs(float(anom.get("z_score", 0)))
        except (ValueError, TypeError):
            continue
        if z > 2.0:
            bonus += 0.25
        elif z > 1.5:
            bonus += 0.1

    return min(bonus, 0.5)  # cap anomaly bonus


def _risk_from_score(score: float, base_risk: str) -> str:
    """Derive risk level from match score and archetype's typical risk."""
    if score < 0.25:
        return "low"
    if score < 0.45:
        return "medium" if base_risk in ("high", "critical") else "low"
    if score < 0.65:
        return "medium" if base_risk != "critical" else "high"
    if score < 0.80:
        return "high"
    return base_risk if base_risk == "critical" else "high"


# ------------------------------------------------------------------
# Utilities for integration
# ------------------------------------------------------------------


def enrich_evidence_with_archetypes(
    evidence: dict[str, Any],
    temporal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Enrich evidence dict with archetype scores for LLM context.

    Adds 'archetype_scores' key with top 3 matches, making the LLM's
    classification job easier and more consistent.
    """
    matches = top_matches(evidence, temporal, limit=3)
    if not matches:
        return evidence

    enriched = dict(evidence)
    enriched["archetype_scores"] = [
        {
            "archetype": m.archetype,
            "signal_score": m.score,
            "matched_signals": m.matched_signals,
            "missing_signals": m.missing_signals,
            "risk_level": m.risk_level,
        }
        for m in matches
    ]
    return enriched


# ------------------------------------------------------------------
# Public-shape adapter
# ------------------------------------------------------------------


def _label_for(archetype_name: str) -> str:
    """Derive a stable human-readable label from an archetype name.

    `pricing_shock` -> `Pricing Shock`. Used by `_to_public_match` so the
    public ArchetypeMatch.label field is populated without coupling to
    the description string (which content packs may override)."""
    return archetype_name.replace("_", " ").title()


def _to_public_match(internal: _ArchetypeMatchInternal) -> ArchetypeMatch:
    """Convert a rich internal match to the public contract type.

    Atlas-internal callers continue to consume `_ArchetypeMatchInternal`
    directly. The public boundary (extracted_reasoning_core.api.score_archetypes)
    converts via this function so consuming products see the stable
    public shape from `extracted_reasoning_core.types.ArchetypeMatch`.
    """
    return ArchetypeMatch(
        archetype_id=internal.archetype,
        label=_label_for(internal.archetype),
        score=internal.score,
        evidence_hits=tuple(internal.matched_signals),
        missing_evidence=tuple(internal.missing_signals),
        risk_label=internal.risk_level,
    )


__all__ = [
    "ARCHETYPES",
    "ArchetypeProfile",
    "MATCH_THRESHOLD",
    "SignalRule",
    "best_match",
    "enrich_evidence_with_archetypes",
    "get_archetype",
    "get_falsification_conditions",
    "score_evidence",
    "top_matches",
]
