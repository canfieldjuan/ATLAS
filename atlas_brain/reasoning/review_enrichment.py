"""Atlas-owned per-review enrichment pack for evidence rules.

The extracted reasoning core owns the slim conclusions and suppression
engine. Atlas still owns these review-level enrichment derivations because
they depend on Atlas review schema and phrase metadata helpers.
"""

from __future__ import annotations

import re
from typing import Any


class ReviewEnrichmentMixin:
    """Mixin that adds Atlas per-review enrichment methods to the slim engine.

    Requires a base class providing ``self._enrichment``,
    ``self._check_condition_simple(...)``, and ``self._resolve_field(...)``.
    Designed to be mixed with ``extracted_reasoning_core.evidence_engine.
    EvidenceEngine`` or a compatible subclass.
    """

    def _init_review_enrichment(self) -> None:
        """Precompile enrichment regexes after the core rules are loaded."""
        rec = self._enrichment.get("recommend_derivation", {})
        self._rec_positive = [
            re.compile(p, re.IGNORECASE)
            for p in rec.get("positive_patterns", [])
        ]
        self._rec_negative = [
            re.compile(p, re.IGNORECASE)
            for p in rec.get("negative_patterns", [])
        ]
        price = self._enrichment.get("price_complaint_derivation", {})
        self._price_positive = [
            re.compile(p, re.IGNORECASE)
            for p in price.get("positive_patterns", [])
        ]

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

        score = 0.0
        for key, weight in weights.items():
            if indicators.get(key):
                score += weight

        if rating is not None and rating_max > 0:
            normalized = rating / rating_max
            for floor in cfg.get("rating_floors", []):
                if normalized <= floor["max_normalized_rating"]:
                    score = max(score, floor["min_score"])
                    break

        context = {"content_type": content_type, "source_weight": source_weight}
        for adj in cfg.get("adjustments", []):
            cond = adj.get("condition", {})
            if self._check_condition_simple(cond, context):
                score += adj.get("delta", 0.0)

        for gate in cfg.get("gates", []):
            cond = gate.get("condition", {})
            if self._check_condition_simple(cond, context):
                score = gate.get("force", score)

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
        """Override generic pain_category using keyword scan."""
        cfg = self._enrichment.get("pain_override", {})
        trigger = cfg.get("trigger", {})
        trigger_values = trigger.get("in")
        if isinstance(trigger_values, list):
            normalized = {
                str(v).strip().lower()
                for v in trigger_values
                if str(v).strip()
            }
            if pain_category not in normalized:
                return pain_category
        elif pain_category != trigger.get("eq", "other"):
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
            return cfg.get("fallback", "overall_dissatisfaction")

        combined = " ".join(texts).lower()

        def _keyword_hits(keyword: str) -> bool:
            pattern = rf"(?<!\w){re.escape(str(keyword or '').lower())}(?!\w)"
            return bool(re.search(pattern, combined))

        scores: dict[str, int] = {}
        for category, keywords in keyword_map.items():
            count = sum(1 for kw in keywords if _keyword_hits(str(kw)))
            if count > 0:
                scores[category] = count

        if scores:
            return max(scores, key=scores.get)
        return cfg.get("fallback", "overall_dissatisfaction")

    def derive_recommend(
        self,
        recommendation_language: list[str],
        rating: float | None,
        rating_max: float,
    ) -> bool | None:
        """Derive would_recommend from extracted language + rating.

        Negative-bias: a phrase matching both negative and positive
        patterns counts only as negative. Conservative recommendation
        semantics should not let a mixed phrase lift the positive count.
        """
        cfg = self._enrichment.get("recommend_derivation", {})

        positive_hits = 0
        negative_hits = 0
        for phrase in (recommendation_language or []):
            neg_match = any(pat.search(phrase) for pat in self._rec_negative)
            if neg_match:
                negative_hits += 1
                continue
            pos_match = any(pat.search(phrase) for pat in self._rec_positive)
            if pos_match:
                positive_hits += 1

        if negative_hits > 0 and negative_hits >= positive_hits:
            return False
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
        # Lazy import keeps this atlas-owned mixin importable without loading
        # task-layer phrase metadata until the derivation path needs it.
        from ..autonomous.tasks._b2b_phrase_metadata import (
            is_v2_tagged,
            phrase_metadata_map,
        )

        cfg = self._enrichment.get("price_complaint_derivation", {})

        rule_input = enrichment
        if is_v2_tagged(enrichment):
            meta = phrase_metadata_map(enrichment)
            raw_pricing = enrichment.get("pricing_phrases") or []
            filtered: list[str] = []
            for index, phrase in enumerate(raw_pricing):
                if not str(phrase or "").strip():
                    continue
                row = meta.get(("pricing_phrases", index)) or {}
                if row.get("subject") != "subject_vendor":
                    continue
                if row.get("polarity") not in ("negative", "mixed"):
                    continue
                filtered.append(phrase)
            rule_input = {**enrichment, "pricing_phrases": filtered}

        pricing_phrases = [
            str(phrase or "").strip()
            for phrase in rule_input.get("pricing_phrases") or []
            if str(phrase or "").strip()
        ]
        all_pricing_phrases_positive = bool(pricing_phrases) and all(
            any(pattern.search(phrase) for pattern in self._price_positive)
            for phrase in pricing_phrases
        )
        for rule in cfg.get("true_if_any", []):
            if (
                rule.get("field") == "pricing_phrases"
                and "min_count" in rule
                and all_pricing_phrases_positive
            ):
                continue
            if self._check_derivation_rule(rule, rule_input):
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

    def _check_derivation_rule(
        self,
        rule: dict[str, Any],
        enrichment: dict[str, Any],
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


__all__ = ["ReviewEnrichmentMixin"]
