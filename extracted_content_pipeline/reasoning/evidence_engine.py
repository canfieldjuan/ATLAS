"""Compatibility wrapper for the shared extracted reasoning evidence engine.

PR-C1i replaced the prior ~338-line local fork with this thin wrapper.
The canonical implementation lives in
`extracted_reasoning_core.evidence_engine` (slim core split per PR-C1d /
PR #104). The content-pipeline-specific rule catalog (consumer-review
pipeline conclusions like `pricing_crisis`, `losing_market_share`,
`active_churn_wave`, `support_quality_risk`) is shipped as a Python
dict (`_DEFAULT_RULES`) below and passed to the core engine via
``EvidenceEngine.from_rules(...)`` -- this keeps the default path
filesystem-free and yaml-free, matching the prior fork's behavior and
the ``EXTRACTED_PIPELINE_STANDALONE=1`` CI env (no PyYAML installed).

The exported `EvidenceEngine` subclass defaults to ``_DEFAULT_RULES``
when called with no arguments and exposes the same ``"builtin"``
sentinel for ``map_path`` that the prior fork did, so existing
content_pipeline callers and the
``tests/test_extracted_reasoning_evidence_engine.py`` test suite keep
working unchanged.

Public contract types `ConclusionResult` / `SuppressionResult` are
re-exported from `extracted_reasoning_core.types` (promoted in PR-C1c).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from extracted_reasoning_core.evidence_engine import (
    EvidenceEngine as _CoreEvidenceEngine,
)
from extracted_reasoning_core.types import ConclusionResult, SuppressionResult


_DEFAULT_RULES: dict[str, Any] = {
    "confidence_tiers": {
        "high": {"min_reviews": 80, "label": "High confidence"},
        "medium": {"min_reviews": 25, "label": "Medium confidence"},
        "low": {"min_reviews": 5, "label": "Low confidence"},
        "insufficient": {"min_reviews": 0, "label": "Insufficient evidence"},
    },
    "conclusions": {
        "insufficient_data": {
            "trigger": [{"field": "total_reviews", "operator": "lt", "value": 20}],
            "label": "Insufficient evidence",
            "action": "Collect more reviews before making directional claims.",
        },
        "pricing_crisis": {
            "requires": [
                {"field": "total_reviews", "operator": "gte", "value": 50},
                {"field": "pain_distribution.pricing.count", "operator": "gte", "value": 10},
                {"field": "pricing_phrases_total", "operator": "gte", "value": 3},
            ],
            "confidence_when_met": "medium",
            "amplifiers": [
                {
                    "field": "pain_distribution.pricing.source_count",
                    "operator": "gte",
                    "value": 3,
                    "boost_confidence": "high",
                }
            ],
            "fallback": {
                "label": "Pricing signal not established",
                "action": "Use pain wording without claiming pricing is the primary driver.",
            },
        },
        "losing_market_share": {
            "requires": [
                {"field": "total_reviews", "operator": "gte", "value": 50},
                {"field": "displacement_edge.mention_count", "operator": "gte", "value": 5},
                {
                    "field": "displacement_edge.signal_strength",
                    "operator": "in",
                    "values": ["moderate", "strong"],
                },
            ],
            "confidence_when_met": "medium",
            "amplifiers": [
                {
                    "field": "displacement_edge.net_flow",
                    "operator": "lte",
                    "value": -5,
                    "boost_confidence": "high",
                }
            ],
            "fallback": {
                "label": "Displacement signal not established",
                "action": "Avoid share-loss framing until displacement evidence improves.",
            },
        },
        "active_churn_wave": {
            "requires": [
                {"field": "total_reviews", "operator": "gte", "value": 40},
                {
                    "field": "indicator_counts.active_evaluation_language",
                    "operator": "gte",
                    "value": 4,
                },
                {
                    "field": "indicator_counts.explicit_cancel_language",
                    "operator": "gte",
                    "value": 2,
                },
            ],
            "confidence_when_met": "high",
            "fallback": {
                "label": "Active churn wave not established",
                "action": "Frame as retention risk, not an active churn wave.",
            },
        },
        "support_quality_risk": {
            "requires": [
                {"field": "total_reviews", "operator": "gte", "value": 40},
                {"field": "pain_distribution.support.count", "operator": "gte", "value": 8},
            ],
            "confidence_when_met": "medium",
            "fallback": {
                "label": "Support quality risk not established",
                "action": "Use generic dissatisfaction wording.",
            },
        },
    },
    "suppression": {
        "executive_summary": {
            "suppress_when": [{"field": "total_reviews", "operator": "lt", "value": 20}],
            "degrade_when": [{"field": "total_reviews", "operator": "lt", "value": 50}],
            "fallback_label": "Not enough evidence for an executive summary",
            "disclaimer": "Directional summary based on a limited evidence base.",
        },
        "target_accounts": {
            "suppress_when": [{"field": "named_company_count", "operator": "lte", "value": 0}],
            "fallback_label": "No named-account evidence available",
        },
        "recommend_ratio": {
            "suppress_when": [{"field": "recommend_denominator", "operator": "lt", "value": 5}],
            "fallback_label": "Recommendation sample too small",
        },
    },
}


class EvidenceEngine(_CoreEvidenceEngine):
    """Content-pipeline-tuned evidence engine.

    Same evaluation logic as the core engine; differs only in default
    rule catalog (the in-memory ``_DEFAULT_RULES`` dict) and in
    carrying the prior fork's ``"builtin"`` ``map_path`` sentinel
    when no path is passed explicitly.
    """

    def __init__(self, map_path: str | Path | None = None) -> None:
        if map_path is None:
            # No path -> use the in-memory default rules. Avoids a
            # filesystem dependency on the standalone CI path.
            tmp = _CoreEvidenceEngine.from_rules(_DEFAULT_RULES)
            self.__dict__.update(tmp.__dict__)
            self.map_path = "builtin"
        else:
            super().__init__(map_path)


_engine: EvidenceEngine | None = None
_engine_path: str | None = None


def get_evidence_engine(map_path: str | Path | None = None) -> EvidenceEngine:
    """Return a cached `EvidenceEngine` instance for the content pipeline.

    Defaults to the in-memory ``_DEFAULT_RULES`` dict shipped above.
    Pass an explicit ``map_path`` to load a different rule catalog
    (e.g. tests injecting a tmp_path JSON file).
    """
    global _engine, _engine_path

    resolved = str(map_path) if map_path else "builtin"
    if _engine is None or _engine_path != resolved:
        _engine = EvidenceEngine(map_path)
        _engine_path = resolved
    return _engine


def reload_evidence_engine() -> EvidenceEngine:
    """Force-reload the cached engine (e.g. after on-disk YAML edits)."""
    global _engine, _engine_path
    _engine = None
    _engine_path = None
    return get_evidence_engine()


__all__ = [
    "ConclusionResult",
    "EvidenceEngine",
    "SuppressionResult",
    "get_evidence_engine",
    "reload_evidence_engine",
]
