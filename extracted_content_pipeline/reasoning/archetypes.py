from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MATCH_THRESHOLD = 0.25


@dataclass(frozen=True)
class SignalRule:
    metric: str
    direction: str
    weight: float = 1.0
    threshold: float | None = None
    pain_keywords: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ArchetypeProfile:
    name: str
    description: str = ""
    typical_risk: str = "medium"
    falsification_templates: list[str] = field(default_factory=list)
    signals: list[SignalRule] = field(default_factory=list)
    velocity_hints: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ArchetypeMatch:
    archetype: str
    score: float
    matched_signals: list[str]
    missing_signals: list[str]
    risk_level: str


ARCHETYPES: dict[str, ArchetypeProfile] = {
    "pricing_shock": ArchetypeProfile(
        name="pricing_shock",
        description="Sudden pricing pressure drives churn intent and competitor mentions",
        typical_risk="high",
        signals=[
            SignalRule("avg_urgency", "high", weight=1.5, threshold=6.0),
            SignalRule(
                "top_pain",
                "present",
                weight=2.0,
                pain_keywords=["price", "pricing", "cost", "expensive", "renewal", "invoice"],
            ),
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
            "Positive review trend reverses for multiple consecutive weeks",
            "Competitor count stabilizes or decreases",
        ],
    ),
    "feature_gap": ArchetypeProfile(
        name="feature_gap",
        description="Missing capabilities create switching pressure toward feature-complete alternatives",
        typical_risk="medium",
        signals=[
            SignalRule(
                "top_pain",
                "present",
                weight=2.0,
                pain_keywords=[
                    "missing",
                    "lack",
                    "need",
                    "feature",
                    "no support for",
                    "doesn't have",
                    "wish",
                    "roadmap",
                ],
            ),
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
            "Competitor mentions decrease materially",
            "Feature-related pain drops below sustained signal levels",
        ],
    ),
    "acquisition_decay": ArchetypeProfile(
        name="acquisition_decay",
        description="Post-acquisition quality decline creates support and trust erosion",
        typical_risk="high",
        signals=[
            SignalRule("positive_review_pct", "low", weight=1.5, threshold=35.0),
            SignalRule("avg_urgency", "high", weight=1.2, threshold=5.5),
            SignalRule(
                "top_pain",
                "present",
                weight=2.0,
                pain_keywords=[
                    "acquired",
                    "acquisition",
                    "buyout",
                    "merged",
                    "new ownership",
                    "quality decline",
                    "worse since",
                ],
            ),
            SignalRule("recommend_ratio", "low", weight=1.0, threshold=0.35),
            SignalRule("churn_density", "high", weight=1.0, threshold=0.6),
        ],
        velocity_hints={
            "positive_review_pct": "decreasing",
            "avg_urgency": "increasing",
            "churn_density": "increasing",
        },
        falsification_templates=[
            "Positive review percentage improves above neutral levels",
            "Support response quality improves materially",
            "New leadership announces and executes a quality recovery plan",
        ],
    ),
    "leadership_redesign": ArchetypeProfile(
        name="leadership_redesign",
        description="Product redesign or leadership change disrupts existing workflows",
        typical_risk="medium",
        signals=[
            SignalRule(
                "top_pain",
                "present",
                weight=2.5,
                pain_keywords=[
                    "redesign",
                    "new ui",
                    "new interface",
                    "update",
                    "changed layout",
                    "new version",
                    "ui overhaul",
                    "workflow changed",
                    "moved features",
                ],
            ),
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
            "Vendor reverts UI changes or provides a classic mode",
            "Positive review percentage recovers above neutral levels",
            "UI-related pain drops below sustained signal levels",
        ],
    ),
    "integration_break": ArchetypeProfile(
        name="integration_break",
        description="API, connector, or workflow breakage creates operational switching pressure",
        typical_risk="high",
        signals=[
            SignalRule(
                "top_pain",
                "present",
                weight=2.5,
                pain_keywords=[
                    "api",
                    "integration",
                    "webhook",
                    "connector",
                    "broke",
                    "breaking change",
                    "deprecated",
                    "incompatible",
                    "migration",
                    "sdk",
                ],
            ),
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
            "Integration-related urgency drops below sustained signal levels",
            "Integration-related pain mentions decline materially",
        ],
    ),
    "support_collapse": ArchetypeProfile(
        name="support_collapse",
        description="Support quality decline creates trust erosion and retention risk",
        typical_risk="critical",
        signals=[
            SignalRule(
                "top_pain",
                "present",
                weight=2.0,
                pain_keywords=[
                    "support",
                    "response time",
                    "ticket",
                    "help desk",
                    "no response",
                    "slow support",
                    "waiting",
                    "customer service",
                    "escalation",
                ],
            ),
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
            "Support response quality improves materially",
            "Positive review percentage recovers above warning levels",
            "Support-related complaints decline materially",
        ],
    ),
    "category_disruption": ArchetypeProfile(
        name="category_disruption",
        description="New entrant or category narrative shifts pull buyers away from incumbents",
        typical_risk="high",
        signals=[
            SignalRule("competitor_count", "high", weight=2.0, threshold=5),
            SignalRule("displacement_edge_count", "high", weight=2.0, threshold=4),
            SignalRule(
                "top_pain",
                "present",
                weight=1.0,
                pain_keywords=[
                    "alternative",
                    "switch to",
                    "moved to",
                    "replaced by",
                    "ai-native",
                    "modern",
                    "next-gen",
                    "outdated",
                ],
            ),
            SignalRule("high_intent_company_count", "high", weight=1.5, threshold=5),
            SignalRule("recommend_ratio", "low", weight=0.8, threshold=0.4),
        ],
        velocity_hints={
            "competitor_count": "increasing",
            "displacement_edge_count": "increasing",
            "high_intent_company_count": "increasing",
        },
        falsification_templates=[
            "Vendor launches features that answer the category shift",
            "Competitor count stabilizes for multiple periods",
            "Displacement edge count declines materially",
        ],
    ),
    "compliance_gap": ArchetypeProfile(
        name="compliance_gap",
        description="Unmet security or regulatory requirements create enterprise churn risk",
        typical_risk="critical",
        signals=[
            SignalRule(
                "top_pain",
                "present",
                weight=3.0,
                pain_keywords=[
                    "compliance",
                    "gdpr",
                    "hipaa",
                    "soc2",
                    "soc 2",
                    "iso 27001",
                    "fedramp",
                    "pci",
                    "audit",
                    "regulation",
                    "security",
                    "data residency",
                    "encryption",
                    "certification",
                ],
            ),
            SignalRule("high_intent_company_count", "high", weight=1.5, threshold=3),
            SignalRule("avg_urgency", "high", weight=1.0, threshold=5.5),
            SignalRule("churn_density", "high", weight=1.0, threshold=0.5),
        ],
        velocity_hints={
            "high_intent_company_count": "increasing",
            "churn_density": "increasing",
        },
        falsification_templates=[
            "Vendor achieves the missing certification or compliance requirement",
            "Compliance-related complaints drop to zero",
            "Enterprise churn intent stabilizes",
        ],
    ),
}


def score_evidence(
    evidence: dict[str, Any],
    temporal: dict[str, Any] | None = None,
) -> list[ArchetypeMatch]:
    merged: dict[str, Any] = dict(evidence or {})
    if isinstance(temporal, dict):
        merged.update(temporal)

    matches = [_score_archetype(profile, merged) for profile in ARCHETYPES.values()]
    matches.sort(key=lambda match: match.score, reverse=True)
    return matches


def best_match(
    evidence: dict[str, Any],
    temporal: dict[str, Any] | None = None,
) -> ArchetypeMatch | None:
    matches = score_evidence(evidence, temporal)
    if matches and matches[0].score >= MATCH_THRESHOLD:
        return matches[0]
    return None


def top_matches(
    evidence: dict[str, Any],
    temporal: dict[str, Any] | None = None,
    *,
    limit: int = 3,
) -> list[ArchetypeMatch]:
    if limit <= 0:
        return []
    matches = score_evidence(evidence, temporal)
    return [match for match in matches[:limit] if match.score >= MATCH_THRESHOLD]


def get_archetype(name: str) -> ArchetypeProfile | None:
    return ARCHETYPES.get(name)


def get_falsification_conditions(archetype_name: str) -> list[str]:
    profile = ARCHETYPES.get(archetype_name)
    return list(profile.falsification_templates) if profile else []


def enrich_evidence_with_archetypes(
    evidence: dict[str, Any],
    temporal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    enriched = dict(evidence or {})
    matches = top_matches(enriched, temporal, limit=3)
    if not matches:
        return enriched
    enriched["archetype_scores"] = [
        {
            "archetype": match.archetype,
            "signal_score": match.score,
            "matched_signals": list(match.matched_signals),
            "missing_signals": list(match.missing_signals),
            "risk_level": match.risk_level,
        }
        for match in matches
    ]
    return enriched


def _score_archetype(profile: ArchetypeProfile, evidence: dict[str, Any]) -> ArchetypeMatch:
    total_weight = sum(rule.weight for rule in profile.signals)
    if total_weight <= 0:
        return ArchetypeMatch(
            archetype=profile.name,
            score=0.0,
            matched_signals=[],
            missing_signals=[],
            risk_level=profile.typical_risk,
        )

    weighted_score = 0.0
    matched: list[str] = []
    missing: list[str] = []
    for rule in profile.signals:
        if _evaluate_rule(rule, evidence):
            weighted_score += rule.weight
            matched.append(rule.metric)
        else:
            missing.append(rule.metric)

    weighted_score += _velocity_bonus(profile, evidence)
    weighted_score += _anomaly_bonus(profile, evidence)

    max_possible = total_weight + len(profile.velocity_hints) * 0.45 + 0.5
    score = min(weighted_score / max_possible, 1.0)
    return ArchetypeMatch(
        archetype=profile.name,
        score=round(score, 3),
        matched_signals=matched,
        missing_signals=missing,
        risk_level=_risk_from_score(score, profile.typical_risk),
    )


def _evaluate_rule(rule: SignalRule, evidence: dict[str, Any]) -> bool:
    if rule.pain_keywords:
        text = _evidence_text(evidence)
        if not text:
            return False
        return any(keyword.lower() in text for keyword in rule.pain_keywords)

    if rule.direction == "present":
        return _has_value(evidence.get(rule.metric))

    if rule.direction in {"high", "low"}:
        value = _numeric_value(evidence.get(rule.metric))
        if value is None:
            return False
        threshold = rule.threshold if rule.threshold is not None else 0.0
        if rule.direction == "high":
            return value >= threshold
        return value <= threshold

    if rule.direction == "increasing":
        value = _numeric_value(evidence.get(f"velocity_{rule.metric}"))
        return value is not None and value > 0
    if rule.direction == "decreasing":
        value = _numeric_value(evidence.get(f"velocity_{rule.metric}"))
        return value is not None and value < 0
    if rule.direction == "increasing_30d":
        value = _numeric_value(evidence.get(f"trend_30d_{rule.metric}"))
        return value is not None and value > 0
    if rule.direction == "decreasing_30d":
        value = _numeric_value(evidence.get(f"trend_30d_{rule.metric}"))
        return value is not None and value < 0

    return False


def _velocity_bonus(profile: ArchetypeProfile, evidence: dict[str, Any]) -> float:
    bonus = 0.0
    for metric, expected in profile.velocity_hints.items():
        velocity = _numeric_value(evidence.get(f"velocity_{metric}"))
        if velocity is None:
            continue
        if expected == "increasing" and velocity > 0:
            bonus += 0.3
        elif expected == "decreasing" and velocity < 0:
            bonus += 0.3
        else:
            continue

        acceleration = _numeric_value(evidence.get(f"accel_{metric}"))
        if acceleration is None:
            continue
        if expected == "increasing" and acceleration > 0:
            bonus += 0.15
        elif expected == "decreasing" and acceleration < 0:
            bonus += 0.15
    return bonus


def _anomaly_bonus(profile: ArchetypeProfile, evidence: dict[str, Any]) -> float:
    anomalies = evidence.get("anomalies")
    if not isinstance(anomalies, list):
        return 0.0
    signal_metrics = {rule.metric for rule in profile.signals if not rule.pain_keywords}
    bonus = 0.0
    for anomaly in anomalies:
        if not isinstance(anomaly, dict) or anomaly.get("metric") not in signal_metrics:
            continue
        z_score = _numeric_value(anomaly.get("z_score"))
        if z_score is None:
            continue
        z_abs = abs(z_score)
        if z_abs > 2.0:
            bonus += 0.25
        elif z_abs > 1.5:
            bonus += 0.1
    return min(bonus, 0.5)


def _risk_from_score(score: float, base_risk: str) -> str:
    if score < 0.25:
        return "low"
    if score < 0.45:
        return "medium" if base_risk in {"high", "critical"} else "low"
    if score < 0.65:
        return "high" if base_risk == "critical" else "medium"
    if score < 0.80:
        return "high"
    return "critical" if base_risk == "critical" else "high"


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


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


_TEXT_KEY_HINTS = (
    "category",
    "complaint",
    "competitor",
    "driver",
    "evidence",
    "issue",
    "pain",
    "quote",
    "reason",
    "review",
    "signal",
    "summary",
    "theme",
    "weakness",
)


def _evidence_text(evidence: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in evidence.items():
        key_text = str(key).lower()
        if key in {"top_pain", "top_competitor", "pain_summary"} or any(
            hint in key_text for hint in _TEXT_KEY_HINTS
        ):
            parts.extend(_flatten_text(value))
    return " ".join(parts).lower()


def _flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        parts: list[str] = []
        for nested in value.values():
            parts.extend(_flatten_text(nested))
        return parts
    if isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            parts.extend(_flatten_text(item))
        return parts
    return []
