"""Smoke tests for extracted_reasoning_core.archetypes.

Locks the consolidation against regressions during the rest of PR-C1.
The full atlas-side archetype tests will rename + redirect into this
file's neighborhood in PR-C1k; for now this carries the minimum
coverage needed to validate the consolidation:

  - catalog loads (10 canonical archetypes)
  - score_evidence ranks matches by score descending
  - MATCH_THRESHOLD gate works (best_match returns None below)
  - top_matches respects the limit + threshold
  - _to_public_match translates field names + derives label
  - _evaluate_rule does not crash on non-numeric velocity strings
    (regression guard for the unguarded float() that PR #94 review
    flagged)
  - _anomaly_bonus tolerates string z_score values
"""

from __future__ import annotations

from extracted_reasoning_core.archetypes import (
    ARCHETYPES,
    MATCH_THRESHOLD,
    ArchetypeProfile,
    SignalRule,
    _ArchetypeMatchInternal,
    _to_public_match,
    best_match,
    enrich_evidence_with_archetypes,
    score_evidence,
    top_matches,
)
from extracted_reasoning_core.types import ArchetypeMatch


def test_catalog_has_ten_canonical_archetypes() -> None:
    assert len(ARCHETYPES) == 10
    expected = {
        "pricing_shock",
        "feature_gap",
        "acquisition_decay",
        "leadership_redesign",
        "integration_break",
        "support_collapse",
        "category_disruption",
        "compliance_gap",
        "scale_up_stumble",
        "pivot_abandonment",
    }
    assert set(ARCHETYPES.keys()) == expected
    for profile in ARCHETYPES.values():
        assert isinstance(profile, ArchetypeProfile)
        assert profile.signals, f"{profile.name} has no signals"


def test_score_evidence_returns_sorted_internal_matches() -> None:
    evidence = {
        "avg_urgency": 7.0,
        "top_pain": "pricing too expensive after renewal",
        "competitor_count": 4,
        "recommend_ratio": 0.3,
        "displacement_edge_count": 3,
        "positive_review_pct": 35,
    }
    matches = score_evidence(evidence)

    # one match per archetype, sorted descending
    assert len(matches) == len(ARCHETYPES)
    scores = [m.score for m in matches]
    assert scores == sorted(scores, reverse=True)
    # pricing_shock should top a pricing-flavored evidence dict
    assert matches[0].archetype == "pricing_shock"
    assert isinstance(matches[0], _ArchetypeMatchInternal)


def test_best_match_respects_threshold() -> None:
    # Empty evidence dict produces zero scores; best_match should be None.
    assert best_match({}) is None

    strong = {
        "avg_urgency": 7.0,
        "top_pain": "pricing too expensive",
        "competitor_count": 4,
        "recommend_ratio": 0.3,
        "displacement_edge_count": 3,
        "positive_review_pct": 35,
    }
    m = best_match(strong)
    assert m is not None
    assert m.score >= MATCH_THRESHOLD


def test_top_matches_caps_and_thresholds() -> None:
    strong = {
        "avg_urgency": 7.0,
        "top_pain": "pricing too expensive",
        "competitor_count": 4,
        "recommend_ratio": 0.3,
        "displacement_edge_count": 3,
        "positive_review_pct": 35,
    }
    top = top_matches(strong, limit=3)
    assert len(top) <= 3
    for m in top:
        assert m.score >= MATCH_THRESHOLD


def test_to_public_match_translates_fields() -> None:
    internal = _ArchetypeMatchInternal(
        archetype="pricing_shock",
        score=0.84,
        matched_signals=["avg_urgency", "top_pain"],
        missing_signals=["competitor_count"],
        risk_level="high",
    )
    public = _to_public_match(internal)
    assert isinstance(public, ArchetypeMatch)
    assert public.archetype_id == "pricing_shock"
    assert public.label == "Pricing Shock"  # title-case derivation
    assert public.score == 0.84
    assert public.evidence_hits == ("avg_urgency", "top_pain")
    assert public.missing_evidence == ("competitor_count",)
    assert public.risk_label == "high"


def test_evaluate_rule_tolerates_non_numeric_velocity() -> None:
    # Regression guard: PR #94 review flagged that velocity branches
    # of _evaluate_rule did float() without try/except. A non-numeric
    # velocity string used to raise; now it must be a non-match.
    evidence_with_string_velocity = {
        "velocity_avg_urgency": "not-a-number",
        "trend_30d_competitor_count": "n/a",
    }
    matches = score_evidence(evidence_with_string_velocity)
    # All scores should compute; nothing crashed.
    assert all(isinstance(m.score, float) for m in matches)


def test_anomaly_bonus_tolerates_string_z_score() -> None:
    # Regression guard: PR #94 review flagged that abs() on a string
    # z_score raised TypeError. Confirm scoring proceeds when anomaly
    # payloads carry stringified numerics.
    evidence_with_string_anomaly = {
        "avg_urgency": 7.0,
        "anomalies": [
            {"metric": "avg_urgency", "z_score": "2.5"},
            {"metric": "competitor_count", "z_score": "garbage"},
        ],
    }
    matches = score_evidence(evidence_with_string_anomaly)
    assert all(isinstance(m.score, float) for m in matches)


def test_enrich_evidence_with_archetypes_returns_dict() -> None:
    evidence = {
        "avg_urgency": 7.0,
        "top_pain": "pricing",
        "competitor_count": 4,
        "recommend_ratio": 0.3,
        "displacement_edge_count": 3,
        "positive_review_pct": 35,
    }
    enriched = enrich_evidence_with_archetypes(evidence)
    assert isinstance(enriched, dict)
    # Original keys preserved
    for k, v in evidence.items():
        assert enriched[k] == v
    # New key added
    assert "archetype_scores" in enriched
    assert isinstance(enriched["archetype_scores"], list)
    assert len(enriched["archetype_scores"]) <= 3


def test_signal_rule_is_immutable() -> None:
    # frozen=True means reassignment raises; verifying the audit-
    # documented immutability decision actually took.
    rule = SignalRule(metric="avg_urgency", direction="high", weight=1.0)
    try:
        rule.weight = 2.0  # type: ignore[misc]
    except Exception as exc:
        # FrozenInstanceError on Python 3.10+; AttributeError on older.
        assert exc.__class__.__name__ in {"FrozenInstanceError", "AttributeError"}
    else:
        raise AssertionError("frozen dataclass should reject reassignment")
