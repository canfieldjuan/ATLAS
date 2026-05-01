from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MATCH_THRESHOLD = 0.25


@dataclass(frozen=True)
class ArchetypeProfile:
    name: str
    description: str = ""
    typical_risk: str = "medium"
    falsification_templates: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ArchetypeMatch:
    archetype: str
    score: float
    matched_signals: list[str]
    missing_signals: list[str]
    risk_level: str


ARCHETYPES: dict[str, ArchetypeProfile] = {
    name: ArchetypeProfile(name=name)
    for name in (
        "pricing_shock",
        "feature_gap",
        "acquisition_decay",
        "leadership_redesign",
        "integration_break",
        "support_collapse",
        "category_disruption",
        "compliance_gap",
    )
}


def score_evidence(
    evidence: dict[str, Any],
    temporal: dict[str, Any] | None = None,
) -> list[ArchetypeMatch]:
    return []


def best_match(
    evidence: dict[str, Any],
    temporal: dict[str, Any] | None = None,
) -> ArchetypeMatch | None:
    return None


def top_matches(
    evidence: dict[str, Any],
    temporal: dict[str, Any] | None = None,
    *,
    limit: int = 3,
) -> list[ArchetypeMatch]:
    return []


def get_archetype(name: str) -> ArchetypeProfile | None:
    return ARCHETYPES.get(name)


def get_falsification_conditions(archetype_name: str) -> list[str]:
    profile = ARCHETYPES.get(archetype_name)
    return list(profile.falsification_templates) if profile else []


def enrich_evidence_with_archetypes(
    evidence: dict[str, Any],
    temporal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return dict(evidence or {})
