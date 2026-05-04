from __future__ import annotations

import json

from extracted_content_pipeline.reasoning.evidence_engine import (
    ConclusionResult,
    EvidenceEngine,
    SuppressionResult,
    get_evidence_engine,
)
from extracted_content_pipeline.reasoning.semantic_cache import compute_evidence_hash
from extracted_reasoning_core.semantic_cache_keys import (
    compute_evidence_hash as core_compute_evidence_hash,
)


def test_semantic_cache_hash_uses_reasoning_core_owner():
    assert compute_evidence_hash is core_compute_evidence_hash
    assert compute_evidence_hash({"a": 1}) == core_compute_evidence_hash({"a": 1})


def test_default_engine_has_stable_builtin_metadata():
    engine = EvidenceEngine()

    assert engine.map_path == "builtin"
    assert engine.map_hash != "standalone"
    assert len(engine.map_hash) == 16


def test_insufficient_data_suppresses_other_conclusions():
    results = EvidenceEngine().evaluate_conclusions({"total_reviews": 10})

    assert results == [
        ConclusionResult(
            conclusion_id="insufficient_data",
            met=True,
            confidence="insufficient",
            fallback_label="Insufficient evidence",
            fallback_action="Collect more reviews before making directional claims.",
        )
    ]


def test_pricing_crisis_met_and_amplified_to_high_confidence():
    evidence = {
        "total_reviews": 120,
        "pain_distribution": {"pricing": {"count": 18, "source_count": 4}},
        "pricing_phrases_total": "7",
    }

    results = EvidenceEngine().evaluate_conclusions(evidence)
    pricing = next(result for result in results if result.conclusion_id == "pricing_crisis")

    assert pricing.met is True
    assert pricing.confidence == "high"
    assert pricing.fallback_label is None


def test_unmet_conclusion_returns_fallback_guidance():
    evidence = {
        "total_reviews": 120,
        "pain_distribution": {"pricing": {"count": 2, "source_count": 1}},
        "pricing_phrases_total": 1,
    }

    results = EvidenceEngine().evaluate_conclusions(evidence)
    pricing = next(result for result in results if result.conclusion_id == "pricing_crisis")

    assert pricing.met is False
    assert pricing.confidence == "insufficient"
    assert pricing.fallback_label == "Pricing signal not established"


def test_losing_market_share_uses_nested_in_and_numeric_requirements():
    evidence = {
        "total_reviews": 200,
        "displacement_edge": {
            "mention_count": 8,
            "signal_strength": "strong",
            "net_flow": -8,
        },
    }

    results = EvidenceEngine().evaluate_conclusions(evidence)
    market_share = next(
        result for result in results if result.conclusion_id == "losing_market_share"
    )

    assert market_share.met is True
    assert market_share.confidence == "high"


def test_evaluate_suppression_suppresses_degrades_or_passes_sections():
    engine = EvidenceEngine()

    assert engine.evaluate_suppression("executive_summary", {"total_reviews": 10}) == (
        SuppressionResult(
            suppress=True,
            fallback_label="Not enough evidence for an executive summary",
        )
    )
    degraded = engine.evaluate_suppression("executive_summary", {"total_reviews": 35})
    assert degraded.degrade is True
    assert degraded.disclaimer == "Directional summary based on a limited evidence base."

    assert engine.evaluate_suppression("executive_summary", {"total_reviews": 200}) == (
        SuppressionResult()
    )
    assert engine.evaluate_suppression("unknown", {"total_reviews": 1}) == SuppressionResult()


def test_confidence_tier_and_label_are_data_driven():
    engine = EvidenceEngine()

    assert engine.get_confidence_tier(100) == "high"
    assert engine.get_confidence_tier(30) == "medium"
    assert engine.get_confidence_tier(5) == "low"
    assert engine.get_confidence_tier(0) == "insufficient"
    assert engine.get_confidence_label(100) == "High confidence"


def test_custom_json_map_path_overrides_builtin_rules(tmp_path):
    rules = {
        "confidence_tiers": {
            "high": {"min_reviews": 2, "label": "custom high"},
            "medium": {"min_reviews": 1, "label": "custom medium"},
            "low": {"min_reviews": 0, "label": "custom low"},
        },
        "conclusions": {
            "custom_claim": {
                "requires": [
                    {"field": "signals", "operator": "min_count", "value": 2},
                    {"field": "metadata.ready", "operator": "eq", "value": True},
                ],
                "confidence_when_met": "high",
                "fallback": {"label": "custom fallback"},
            }
        },
        "suppression": {
            "custom_section": {
                "suppress_when": [{"field": "score", "operator": "lt", "value": 3}],
                "fallback_label": "custom suppressed",
            }
        },
    }
    path = tmp_path / "evidence_map.json"
    path.write_text(json.dumps(rules))

    engine = EvidenceEngine(path)

    result = engine.evaluate_conclusions({
        "signals": ["a", "b"],
        "metadata": {"ready": True},
    })[0]
    assert result.conclusion_id == "custom_claim"
    assert result.met is True
    assert result.confidence == "high"

    suppressed = engine.evaluate_suppression("custom_section", {"score": "2"})
    assert suppressed.suppress is True
    assert suppressed.fallback_label == "custom suppressed"
    assert engine.get_confidence_label(2) == "custom high"


def test_get_evidence_engine_caches_by_resolved_path(tmp_path):
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps({"conclusions": {}, "suppression": {}}))
    second_path.write_text(json.dumps({"conclusions": {}, "suppression": {}}))

    first = get_evidence_engine(first_path)
    again = get_evidence_engine(first_path)
    second = get_evidence_engine(second_path)

    assert first is again
    assert second is not first
