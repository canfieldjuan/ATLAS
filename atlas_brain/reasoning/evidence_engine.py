"""Declarative evidence evaluation engine (atlas-side wrapper).

PR-D7b3 promoted core's slim conclusions+suppression engine into
:mod:`extracted_reasoning_core.evidence_engine`. Atlas keeps the
import surface ``atlas_brain.reasoning.evidence_engine`` as a
subclass wrapper that adds the per-review enrichment methods that
stay atlas-side per PR-C1's slim-core split:

  - ``compute_urgency``
  - ``override_pain``
  - ``derive_recommend``
  - ``derive_price_complaint`` (depends on atlas-only
    ``_b2b_phrase_metadata``)
  - ``derive_budget_authority``

The subclass pattern keeps existing atlas callers
(``b2b_churn_intelligence``, ``_b2b_shared``,
``services/b2b/enrichment_derivation``, the test stubs in
``test_b2b_enrichment.py``) writing ``engine.compute_urgency(...)``
against a single object -- core's slim conclusions/suppression
methods (``evaluate_conclusions`` / ``evaluate_suppression`` /
``get_confidence_tier`` / ``get_confidence_label`` /
``evaluate_conclusion``) come from the core base class; the six
enrichment methods come from this subclass.

Re-exports ``ConclusionResult`` / ``SuppressionResult`` from
:mod:`extracted_reasoning_core.types` -- both shapes are byte-
identical to atlas's pre-PR-D7b3 local dataclasses (verified during
PR-D7b3 drift analysis), so atlas callers reading
``r.conclusion_id`` / ``r.met`` / ``r.confidence`` /
``r.fallback_label`` / ``r.fallback_action`` and ``s.suppress`` /
``s.degrade`` / ``s.disclaimer`` / ``s.fallback_label`` keep
working through the wrapper.

Atlas's factory ``get_evidence_engine`` continues to consult
``settings.b2b_churn.evidence_map_path`` before falling back to the
default YAML beside this module. Core's factory stays config-free
because external products don't ship atlas's settings module.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from extracted_reasoning_core.evidence_engine import (
    EvidenceEngine as _CoreEvidenceEngine,
)
from extracted_reasoning_core.types import ConclusionResult, SuppressionResult

logger = logging.getLogger("atlas.reasoning.evidence_engine")

_DEFAULT_MAP_PATH = Path(__file__).parent / "evidence_map.yaml"


class EvidenceEngine(_CoreEvidenceEngine):
    """Core slim engine extended with atlas-side per-review enrichment."""

    def _init_from_rules(
        self,
        rules: dict[str, Any],
        map_path_str: str,
        raw_bytes: bytes,
    ) -> None:
        super()._init_from_rules(rules, map_path_str, raw_bytes)
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

        for adj in cfg.get("adjustments", []):
            cond = adj.get("condition", {})
            if self._check_condition_simple(cond, {"content_type": content_type, "source_weight": source_weight}):
                score += adj.get("delta", 0.0)

        for gate in cfg.get("gates", []):
            cond = gate.get("condition", {})
            if self._check_condition_simple(cond, {"content_type": content_type, "source_weight": source_weight}):
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
            if pain_category not in {str(v).strip().lower() for v in trigger_values if str(v).strip()}:
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
        """Derive would_recommend from extracted language + rating."""
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
        # Single positive phrase against a clearly negative rating is likely
        # sarcasm -- require 2+ positive hits to override.
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
        """Derive price_complaint from Tier 1 flags + extracted phrases.

        Phase 2 (Layer 1 -- subject attribution gate): on v2-tagged
        enrichments, restrict the rule check to pricing_phrases the LLM
        marked as subject='subject_vendor'. A self-cost mention like
        "I pay $X for my own setup" must not flip the price_complaint
        flag on the vendor being reviewed. v1 enrichments fall through
        to the legacy code path.
        """
        from ..autonomous.tasks._b2b_phrase_metadata import (
            is_v2_tagged, phrase_metadata_map,
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
                # Phase 3 (Layer 2): only negative / mixed pricing phrases
                # should trip the price_complaint flag.
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
        self, rule: dict[str, Any], enrichment: dict[str, Any],
    ) -> bool:
        """Check a derivation true_if_any rule (atlas-only helper)."""
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


__all__ = [
    "ConclusionResult",
    "EvidenceEngine",
    "SuppressionResult",
    "get_evidence_engine",
    "reload_evidence_engine",
]
