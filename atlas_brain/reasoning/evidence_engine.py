"""Declarative evidence evaluation engine.

Loads rules from ``evidence_map.yaml`` and provides deterministic
per-review enrichment compute and per-vendor conclusion gating.

All business logic lives in the YAML -- this module is pure evaluation
machinery with zero hardcoded thresholds or domain knowledge.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("atlas.reasoning.evidence_engine")

_DEFAULT_MAP_PATH = Path(__file__).parent / "evidence_map.yaml"


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
    """Generic YAML-driven evidence evaluator."""

    def __init__(self, map_path: str | Path | None = None) -> None:
        path = Path(map_path) if map_path else _DEFAULT_MAP_PATH
        with open(path) as f:
            raw_bytes = f.read()
        self._rules: dict[str, Any] = yaml.safe_load(raw_bytes)
        self.map_hash: str = hashlib.sha256(raw_bytes.encode()).hexdigest()[:16]
        self.map_path: str = str(path)
        self._enrichment = self._rules.get("enrichment", {})
        self._conclusions = self._rules.get("conclusions", {})
        self._suppression = self._rules.get("suppression", {})
        self._confidence_tiers = self._rules.get("confidence_tiers", {})

        # Pre-compile regex patterns for recommend derivation
        rec = self._enrichment.get("recommend_derivation", {})
        self._rec_positive = [
            re.compile(p, re.IGNORECASE)
            for p in rec.get("positive_patterns", [])
        ]
        self._rec_negative = [
            re.compile(p, re.IGNORECASE)
            for p in rec.get("negative_patterns", [])
        ]

    # ------------------------------------------------------------------
    # Per-review: enrichment-time compute
    # ------------------------------------------------------------------

    def compute_urgency(
        self,
        indicators: dict[str, bool],
        rating: float | None,
        rating_max: float,
        content_type: str,
        source_weight: float,
    ) -> float:
        """Compute urgency_score (0-10) from boolean indicator flags."""
        cfg = self._enrichment.get("urgency_scoring", {})
        weights: dict[str, float] = cfg.get("weights", {})

        # Sum weighted indicators
        score = 0.0
        for key, weight in weights.items():
            if indicators.get(key):
                score += weight

        # Rating floors
        if rating is not None and rating_max > 0:
            normalized = rating / rating_max
            for floor in cfg.get("rating_floors", []):
                if normalized <= floor["max_normalized_rating"]:
                    score = max(score, floor["min_score"])
                    break

        # Adjustments
        for adj in cfg.get("adjustments", []):
            cond = adj.get("condition", {})
            if self._check_condition_simple(cond, {"content_type": content_type, "source_weight": source_weight}):
                score += adj.get("delta", 0.0)

        # Gates
        for gate in cfg.get("gates", []):
            cond = gate.get("condition", {})
            if self._check_condition_simple(cond, {"content_type": content_type, "source_weight": source_weight}):
                score = gate.get("force", score)

        # Clamp
        lo, hi = cfg.get("clamp", [0, 10])
        return round(min(max(score, lo), hi), 1)

    def override_pain(
        self,
        pain_category: str,
        specific_complaints: list[str],
        quotable_phrases: list[str] | None = None,
        pricing_phrases: list[str] | None = None,
        feature_gaps: list[str] | None = None,
        recommendation_language: list[str] | None = None,
    ) -> str:
        """Override 'other' pain_category using keyword scan."""
        cfg = self._enrichment.get("pain_override", {})
        trigger = cfg.get("trigger", {})
        if pain_category != trigger.get("eq", "other"):
            return pain_category

        keyword_map: dict[str, list[str]] = cfg.get("keyword_map", {})
        scan_fields = cfg.get("scan_fields", ["specific_complaints"])

        texts: list[str] = []
        if "specific_complaints" in scan_fields:
            texts.extend(specific_complaints or [])
        if "quotable_phrases" in scan_fields:
            texts.extend(quotable_phrases or [])
        if "pricing_phrases" in scan_fields:
            texts.extend(pricing_phrases or [])
        if "feature_gaps" in scan_fields:
            texts.extend(feature_gaps or [])
        if "recommendation_language" in scan_fields:
            texts.extend(recommendation_language or [])

        if not texts:
            return cfg.get("fallback", "other")

        combined = " ".join(texts).lower()
        scores: dict[str, int] = {}
        for category, keywords in keyword_map.items():
            count = sum(1 for kw in keywords if kw in combined)
            if count > 0:
                scores[category] = count

        if scores:
            return max(scores, key=scores.get)
        return cfg.get("fallback", "other")

    def derive_recommend(
        self,
        recommendation_language: list[str],
        rating: float | None,
        rating_max: float,
    ) -> bool | None:
        """Derive would_recommend from extracted language + rating."""
        cfg = self._enrichment.get("recommend_derivation", {})

        positive_hits = 0
        negative_hits = 0
        for phrase in (recommendation_language or []):
            # Check negative first -- longer patterns take priority
            neg_match = any(pat.search(phrase) for pat in self._rec_negative)
            if neg_match:
                negative_hits += 1
                continue
            pos_match = any(pat.search(phrase) for pat in self._rec_positive)
            if pos_match:
                positive_hits += 1

        if negative_hits > 0 and negative_hits >= positive_hits:
            return False
        # If rating is very low, a single positive phrase is likely sarcasm
        # -- require 2+ positive hits to override a clearly negative rating
        fallback = cfg.get("rating_fallback", {})
        if (
            positive_hits == 1
            and negative_hits == 0
            and rating is not None
            and rating_max > 0
            and (rating / rating_max) <= fallback.get("false_below", 0.3)
        ):
            return False
        if positive_hits > 0 and positive_hits > negative_hits:
            return True

        # Rating fallback
        fallback = cfg.get("rating_fallback", {})
        if rating is not None and rating_max > 0:
            normalized = rating / rating_max
            if normalized >= fallback.get("true_above", 0.7):
                return True
            if normalized <= fallback.get("false_below", 0.3):
                return False

        return cfg.get("default", None)

    def derive_price_complaint(
        self,
        enrichment: dict[str, Any],
    ) -> bool:
        """Derive price_complaint from Tier 1 flags + extracted phrases."""
        cfg = self._enrichment.get("price_complaint_derivation", {})
        for rule in cfg.get("true_if_any", []):
            if self._check_derivation_rule(rule, enrichment):
                return True
        return False

    def derive_budget_authority(
        self,
        enrichment: dict[str, Any],
    ) -> bool:
        """Derive has_budget_authority from role + language."""
        cfg = self._enrichment.get("budget_authority_derivation", {})
        for rule in cfg.get("true_if_any", []):
            if self._check_derivation_rule(rule, enrichment):
                return True
        return False

    # ------------------------------------------------------------------
    # Per-vendor: report-time conclusion gating
    # ------------------------------------------------------------------

    def evaluate_conclusions(
        self,
        vendor_evidence: dict[str, Any],
    ) -> list[ConclusionResult]:
        """Evaluate all conclusion gates against vendor-level evidence."""
        results: list[ConclusionResult] = []

        # Check insufficient_data first -- it can suppress everything
        insuf = self._conclusions.get("insufficient_data")
        if insuf:
            triggers = insuf.get("trigger", [])
            if all(self._check_requirement(t, vendor_evidence) for t in triggers):
                results.append(ConclusionResult(
                    conclusion_id="insufficient_data",
                    met=True,
                    confidence="insufficient",
                    fallback_label=insuf.get("label"),
                    fallback_action=insuf.get("action"),
                ))
                return results

        for cid, spec in self._conclusions.items():
            if cid == "insufficient_data":
                continue

            requires = spec.get("requires", [])
            all_met = all(
                self._check_requirement(r, vendor_evidence)
                for r in requires
            )

            confidence = spec.get("confidence_when_met", "medium") if all_met else "insufficient"

            # Amplifiers can boost confidence
            if all_met:
                for amp in spec.get("amplifiers", []):
                    if self._check_requirement(amp, vendor_evidence) and amp.get("boost_confidence"):
                        if confidence == "medium":
                            confidence = "high"

            fallback = spec.get("fallback", {})
            results.append(ConclusionResult(
                conclusion_id=cid,
                met=all_met,
                confidence=confidence,
                fallback_label=fallback.get("label") if not all_met else None,
                fallback_action=fallback.get("action") if not all_met else None,
            ))

        return results

    def evaluate_suppression(
        self,
        section: str,
        evidence: dict[str, Any],
    ) -> SuppressionResult:
        """Evaluate suppression rules for a report section."""
        spec = self._suppression.get(section)
        if not spec:
            return SuppressionResult()

        # Check suppress
        for rule in spec.get("suppress_when", []):
            if self._check_suppression_rule(rule, evidence):
                return SuppressionResult(
                    suppress=True,
                    fallback_label=spec.get("fallback_label"),
                )

        # Check degrade
        for rule in spec.get("degrade_when", []):
            if self._check_suppression_rule(rule, evidence):
                return SuppressionResult(
                    degrade=True,
                    disclaimer=spec.get("disclaimer"),
                )

        return SuppressionResult()

    def get_confidence_tier(self, total_reviews: int) -> str:
        """Return confidence tier label for a review count."""
        for tier_name in ("high", "medium", "low"):
            tier = self._confidence_tiers.get(tier_name, {})
            if total_reviews >= tier.get("min_reviews", 0):
                return tier_name
        return "insufficient"

    def get_confidence_label(self, total_reviews: int) -> str:
        """Return human-readable confidence label."""
        tier_name = self.get_confidence_tier(total_reviews)
        tier = self._confidence_tiers.get(tier_name, {})
        return tier.get("label", tier_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_field(data: dict[str, Any], field_path: str) -> Any:
        """Resolve a dotted field path against a nested dict."""
        parts = field_path.split(".")
        current: Any = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    def _check_condition_simple(
        self, cond: dict[str, Any], context: dict[str, Any],
    ) -> bool:
        """Check a simple {field, eq/lte/gte/in} condition."""
        field = cond.get("field", "")
        value = self._resolve_field(context, field)
        if "eq" in cond:
            return value == cond["eq"]
        if "lte" in cond:
            return value is not None and float(value) <= float(cond["lte"])
        if "gte" in cond:
            return value is not None and float(value) >= float(cond["gte"])
        if "in" in cond:
            return value in cond["in"]
        return False

    def _check_requirement(
        self, req: dict[str, Any], evidence: dict[str, Any],
    ) -> bool:
        """Check a conclusion requirement against vendor evidence."""
        field = req.get("field", "")
        value = self._resolve_field(evidence, field)
        op = req.get("operator", "eq")

        if op == "gte":
            return value is not None and float(value) >= float(req["value"])
        if op == "gt":
            return value is not None and float(value) > float(req["value"])
        if op == "lte":
            return value is not None and float(value) <= float(req["value"])
        if op == "lt":
            return value is not None and float(value) < float(req["value"])
        if op == "eq":
            return value == req.get("value")
        if op == "in":
            return value in req.get("values", [])
        return False

    def _check_derivation_rule(
        self, rule: dict[str, Any], enrichment: dict[str, Any],
    ) -> bool:
        """Check a derivation true_if_any rule."""
        field = rule.get("field", "")
        value = self._resolve_field(enrichment, field)

        if "eq" in rule:
            return value == rule["eq"]
        if "in" in rule:
            return value in rule["in"]
        if "not_null" in rule and rule["not_null"]:
            return value is not None and value != ""
        if "min_count" in rule:
            return isinstance(value, (list, tuple)) and len(value) >= rule["min_count"]
        if "contains_any" in rule:
            if not isinstance(value, (list, tuple)):
                return False
            combined = " ".join(str(v) for v in value).lower()
            return any(kw in combined for kw in rule["contains_any"])
        return False

    def _check_suppression_rule(
        self, rule: dict[str, Any], evidence: dict[str, Any],
    ) -> bool:
        """Check a suppression condition ({field, lt/eq/gt/lte/gte} format)."""
        field = rule.get("field", "")
        value = self._resolve_field(evidence, field)

        if "lt" in rule:
            return value is not None and float(value) < float(rule["lt"])
        if "lte" in rule:
            return value is not None and float(value) <= float(rule["lte"])
        if "gt" in rule:
            return value is not None and float(value) > float(rule["gt"])
        if "gte" in rule:
            return value is not None and float(value) >= float(rule["gte"])
        if "eq" in rule:
            return value == rule["eq"]

        return False


# Module-level singleton
_engine: EvidenceEngine | None = None
_engine_path: str | None = None


def get_evidence_engine(map_path: str | Path | None = None) -> EvidenceEngine:
    """Return a cached EvidenceEngine, respecting config path.

    If no map_path is provided, checks ``settings.b2b_churn.evidence_map_path``
    before falling back to the default YAML beside this module.

    The singleton is invalidated if the resolved path changes (e.g. config
    reload with a different path).  In-process YAML edits still require a
    ``reload_evidence_engine()`` call or process restart.
    """
    global _engine, _engine_path

    if map_path is None:
        try:
            from ..config import settings
            configured = (settings.b2b_churn.evidence_map_path or "").strip()
            if configured:
                map_path = configured
        except Exception:
            pass

    resolved = str(map_path) if map_path else str(_DEFAULT_MAP_PATH)

    if _engine is not None and _engine_path == resolved:
        return _engine

    try:
        _engine = EvidenceEngine(resolved)
    except FileNotFoundError:
        if resolved != str(_DEFAULT_MAP_PATH):
            logger.error(
                "Evidence map not found at configured path %s, falling back to default",
                resolved,
            )
            resolved = str(_DEFAULT_MAP_PATH)
            _engine = EvidenceEngine(resolved)
        else:
            raise
    _engine_path = resolved
    logger.info(
        "Evidence engine loaded: %s (hash=%s, %d enrichment rules, %d conclusions)",
        resolved,
        _engine.map_hash,
        len(_engine._enrichment),
        len(_engine._conclusions),
    )
    return _engine


def reload_evidence_engine() -> EvidenceEngine:
    """Force-reload the Evidence Map from disk (e.g. after YAML edits)."""
    global _engine, _engine_path
    _engine = None
    _engine_path = None
    return get_evidence_engine()
