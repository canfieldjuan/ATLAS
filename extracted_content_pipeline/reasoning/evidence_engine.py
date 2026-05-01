from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ConclusionResult:
    conclusion_id: str
    met: bool
    confidence: str
    fallback_label: str | None = None
    fallback_action: str | None = None


@dataclass(frozen=True, slots=True)
class SuppressionResult:
    suppress: bool = False
    degrade: bool = False
    disclaimer: str | None = None
    fallback_label: str | None = None


class EvidenceEngine:
    def __init__(self, map_path: str | Path | None = None) -> None:
        self.map_path = str(map_path or "builtin")
        raw_rules = _load_rules(map_path)
        self.map_hash = hashlib.sha256(
            json.dumps(raw_rules, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        self._conclusions = raw_rules.get("conclusions", {})
        self._suppression = raw_rules.get("suppression", {})
        self._confidence_tiers = raw_rules.get("confidence_tiers", {})

    def evaluate_conclusions(
        self,
        vendor_evidence: dict[str, Any],
    ) -> list[ConclusionResult]:
        evidence = dict(vendor_evidence or {})
        insufficient = self._conclusions.get("insufficient_data")
        if insufficient and all(
            self._check_requirement(rule, evidence)
            for rule in insufficient.get("trigger", [])
        ):
            return [
                ConclusionResult(
                    conclusion_id="insufficient_data",
                    met=True,
                    confidence="insufficient",
                    fallback_label=insufficient.get("label"),
                    fallback_action=insufficient.get("action"),
                )
            ]

        results: list[ConclusionResult] = []
        for conclusion_id, spec in self._conclusions.items():
            if conclusion_id == "insufficient_data":
                continue
            requirements = spec.get("requires", [])
            met = all(self._check_requirement(rule, evidence) for rule in requirements)
            confidence = spec.get("confidence_when_met", "medium") if met else "insufficient"
            if met:
                for amplifier in spec.get("amplifiers", []):
                    if self._check_requirement(amplifier, evidence) and amplifier.get("boost_confidence"):
                        confidence = str(amplifier.get("boost_confidence"))
            fallback = spec.get("fallback", {}) if not met else {}
            results.append(
                ConclusionResult(
                    conclusion_id=conclusion_id,
                    met=met,
                    confidence=confidence,
                    fallback_label=fallback.get("label"),
                    fallback_action=fallback.get("action"),
                )
            )
        return results

    def evaluate_suppression(
        self,
        section: str,
        evidence: dict[str, Any],
    ) -> SuppressionResult:
        spec = self._suppression.get(section)
        if not spec:
            return SuppressionResult()

        evidence = dict(evidence or {})
        for rule in spec.get("suppress_when", []):
            if self._check_requirement(rule, evidence):
                return SuppressionResult(
                    suppress=True,
                    fallback_label=spec.get("fallback_label"),
                )

        for rule in spec.get("degrade_when", []):
            if self._check_requirement(rule, evidence):
                return SuppressionResult(
                    degrade=True,
                    disclaimer=spec.get("disclaimer"),
                )

        return SuppressionResult()

    def get_confidence_tier(self, total_reviews: int) -> str:
        review_count = _numeric_value(total_reviews) or 0.0
        for tier_name in ("high", "medium", "low"):
            tier = self._confidence_tiers.get(tier_name, {})
            if review_count >= (_numeric_value(tier.get("min_reviews")) or 0.0):
                return tier_name
        return "insufficient"

    def get_confidence_label(self, total_reviews: int) -> str:
        tier_name = self.get_confidence_tier(total_reviews)
        tier = self._confidence_tiers.get(tier_name, {})
        return str(tier.get("label") or tier_name)

    @staticmethod
    def _resolve_field(data: dict[str, Any], field_path: str) -> Any:
        current: Any = data
        for part in str(field_path or "").split("."):
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    def _check_requirement(self, rule: dict[str, Any], evidence: dict[str, Any]) -> bool:
        field = str(rule.get("field") or "")
        value = self._resolve_field(evidence, field)
        operator = rule.get("operator")

        if operator in {"gte", "gt", "lte", "lt"}:
            value_num = _numeric_value(value)
            rule_num = _numeric_value(rule.get("value"))
            if value_num is None or rule_num is None:
                return False
            if operator == "gte":
                return value_num >= rule_num
            if operator == "gt":
                return value_num > rule_num
            if operator == "lte":
                return value_num <= rule_num
            return value_num < rule_num

        if operator == "eq":
            return value == rule.get("value")
        if operator == "in":
            return value in rule.get("values", [])
        if operator == "exists":
            return value is not None and value != ""
        if operator == "min_count":
            expected = _numeric_value(rule.get("value"))
            return (
                isinstance(value, (list, tuple, set, dict))
                and expected is not None
                and len(value) >= expected
            )

        for op in ("gte", "gt", "lte", "lt", "eq"):
            if op in rule:
                return self._check_requirement(
                    {"field": field, "operator": op, "value": rule[op]},
                    evidence,
                )
        if "in" in rule:
            return self._check_requirement(
                {"field": field, "operator": "in", "values": rule["in"]},
                evidence,
            )
        return False


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


_engine: EvidenceEngine | None = None
_engine_path: str | None = None


def get_evidence_engine(map_path: str | Path | None = None) -> EvidenceEngine:
    global _engine, _engine_path
    resolved = str(map_path) if map_path else "builtin"
    if _engine is None or _engine_path != resolved:
        _engine = EvidenceEngine(map_path)
        _engine_path = resolved
    return _engine


def reload_evidence_engine() -> EvidenceEngine:
    global _engine, _engine_path
    _engine = None
    _engine_path = None
    return get_evidence_engine()


def _load_rules(map_path: str | Path | None) -> dict[str, Any]:
    if map_path is None:
        return dict(_DEFAULT_RULES)
    path = Path(map_path)
    raw = path.read_text()
    if path.suffix.lower() == ".json":
        loaded = json.loads(raw)
    else:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError:
            loaded = json.loads(raw)
        else:
            loaded = yaml.safe_load(raw)
    if not isinstance(loaded, dict):
        raise ValueError(f"Evidence map must decode to an object: {path}")
    return loaded


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
