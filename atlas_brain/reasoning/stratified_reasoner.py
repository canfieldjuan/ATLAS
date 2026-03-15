"""Stratified Reasoning Engine -- core 3-mode dispatch.

Three cognitive modes:
    Recall       - Semantic cache hit, fresh + same evidence hash  (~0 tokens)
    Reconstitute - Cache hit but evidence changed <30% delta       (~30% tokens)
    Reason       - Full LLM synthesis (no cache or large delta)    (~100% tokens)

Also integrates:
    - Metacognitive monitor (surprise detection, exploration budget)
    - Hierarchical tier context (inherited priors from higher tiers)
    - Differential engine (evidence diff classification)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .differential import EvidenceDiff, classify_evidence, reconstitute
from .episodic_store import (
    ConclusionNode,
    EpisodicStore,
    EvidenceNode,
    ReasoningTrace,
)
from .llm_utils import REASONING_CONCLUSION_JSON_SCHEMA, resolve_stratified_llm
from .semantic_cache import CacheEntry, SemanticCache, compute_evidence_hash

logger = logging.getLogger("atlas.reasoning.stratified")

# System prompt for Phase 1 (inline; replaced by skill file in WS6)
_REASON_SYSTEM_PROMPT = """\
You are a B2B churn intelligence analyst. Classify the vendor's churn pattern from the evidence provided.

ARCHETYPE DEFINITIONS -- choose the one that best matches the PRIMARY driver:
- pricing_shock: churn driven mainly by price increases/complaints. Requires price_complaint_rate to be the dominant signal.
- feature_gap: churn because competitors offer features this vendor lacks. Requires feature gaps or displacement data as primary driver.
- acquisition_decay: quality decline after M&A. Requires evidence of post-acquisition deterioration.
- leadership_redesign: UI/UX overhaul causing user frustration. Requires UX/usability as dominant pain category.
- integration_break: API or integration failures. Requires integration-related pain as dominant signal.
- support_collapse: support quality deterioration. Requires support-related pain as dominant signal.
- category_disruption: new entrant (often AI-native) disrupting the category. Requires displacement to newer/different category entrants.
- compliance_gap: unmet regulatory requirements. Requires compliance-related evidence.
- mixed: no single pattern dominates (use when top two signals are close and different archetypes).
- stable: vendor is healthy, no significant churn pattern detected.

CLASSIFICATION RULES:
1. Look at pain_categories FIRST. The category with the highest count is the primary driver.
2. archetype_scores are heuristic pre-scores -- treat them as hypotheses, NOT as the answer. Override them when the evidence disagrees.
3. If the top pain category is "ux" or "usability", the archetype should be leadership_redesign, NOT pricing_shock.
4. If the top pain category is "pricing" AND price_complaint_rate > 0.15, then pricing_shock is justified.
5. If the top pain is "other" or ambiguous, look at the SECOND pain category and displacement data.
6. Use "mixed" when the top two pain categories are within 20% of each other and map to different archetypes.

CONFIDENCE CALIBRATION:
- 0.85-1.0: Strong, unambiguous signal with 50+ reviews and clear dominant pattern.
- 0.65-0.84: Clear pattern but some noise or limited temporal data.
- 0.45-0.64: Pattern visible but evidence is thin, mixed, or contradictory.
- 0.20-0.44: Weak signal. Must use "mixed" or "stable".
- Do NOT default to 0.82. Confidence must vary based on actual evidence strength.

RISK LEVEL:
- critical: churn_density > 40% AND avg_urgency > 7
- high: churn_density > 25% OR (churn_density > 15% AND avg_urgency > 6)
- medium: churn_density > 10% OR avg_urgency > 4
- low: below all thresholds above

Output ONLY valid JSON:
{
  "archetype": "<one archetype from the list above>",
  "secondary_archetype": "<another archetype or null if gap > 0.15>",
  "confidence": <number 0.0-1.0>,
  "risk_level": "<low|medium|high|critical>",
  "executive_summary": "<3 sentences: (1) what pattern, (2) why now citing specific metrics, (3) what to watch>",
  "key_signals": ["<metric: value>", ...],
  "falsification_conditions": ["<what would prove this wrong>"],
  "uncertainty_sources": ["<what data is missing or weak>"]
}

GROUNDING RULES:
- Every key_signal MUST cite a specific metric and value from the evidence (e.g., "churn_density: 38.6%").
- executive_summary sentence 2 MUST reference at least 2 specific numbers from the evidence.
- If confidence < 0.6, include at least 2 uncertainty_sources.
- Do not invent data not present in the evidence.\
"""


@dataclass
class ReasoningResult:
    """Result of a stratified reasoning analysis."""

    mode: str           # "recall", "reconstitute", "reason"
    conclusion: dict[str, Any]
    confidence: float
    pattern_sig: str
    evidence_hash: str
    tokens_used: int
    cached: bool
    trace_id: str | None = None


class StratifiedReasoner:
    """Core dispatch engine: recall -> reconstitute -> reason."""

    _VALID_ARCHETYPES = frozenset({
        "pricing_shock", "feature_gap", "acquisition_decay",
        "leadership_redesign", "integration_break", "support_collapse",
        "category_disruption", "compliance_gap", "mixed", "stable",
    })
    _VALID_RISK_LEVELS = frozenset({"low", "medium", "high", "critical"})

    def __init__(
        self,
        cache: SemanticCache,
        episodic: EpisodicStore,
        metacognition: Any | None = None,
    ):
        self._cache = cache
        self._episodic = episodic
        self._meta = metacognition  # MetacognitiveMonitor (optional)

    @staticmethod
    def _normalize_conclusion(conclusion: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize LLM reasoning output.

        Clamps confidence to [0,1], whitelists archetype and risk_level,
        coerces list fields, and fills missing fields with safe defaults.
        """
        if not isinstance(conclusion, dict) or "error" in conclusion:
            return conclusion

        conclusion = dict(conclusion)

        # Clamp confidence
        try:
            conf = float(conclusion.get("confidence", 0.5))
            conclusion["confidence"] = max(0.0, min(1.0, conf))
        except (TypeError, ValueError):
            conclusion["confidence"] = 0.5

        # Whitelist archetype
        arch = conclusion.get("archetype", "")
        if arch not in StratifiedReasoner._VALID_ARCHETYPES:
            logger.warning("Normalized invalid archetype %r -> 'mixed'", arch)
            conclusion["archetype"] = "mixed"

        # Validate secondary_archetype
        sec_arch = conclusion.get("secondary_archetype")
        if sec_arch and sec_arch not in StratifiedReasoner._VALID_ARCHETYPES:
            conclusion["secondary_archetype"] = None

        # Enforce: if confidence < 0.4, archetype must be mixed or stable
        if conclusion["confidence"] < 0.4 and conclusion["archetype"] not in ("mixed", "stable"):
            logger.warning(
                "Low confidence %.2f with archetype %r -- forcing to mixed",
                conclusion["confidence"], conclusion["archetype"],
            )
            conclusion["archetype"] = "mixed"

        # Enforce: if confidence < 0.6, need at least 2 uncertainty_sources
        if conclusion["confidence"] < 0.6 and len(conclusion.get("uncertainty_sources", [])) < 2:
            conclusion.setdefault("uncertainty_sources", [])
            if len(conclusion["uncertainty_sources"]) < 2:
                conclusion["uncertainty_sources"].append("insufficient data coverage for confident classification")

        # Whitelist risk_level
        risk = conclusion.get("risk_level", "")
        if risk not in StratifiedReasoner._VALID_RISK_LEVELS:
            logger.warning("Normalized invalid risk_level %r -> 'medium'", risk)
            conclusion["risk_level"] = "medium"

        # Coerce list fields
        for fld in ("key_signals", "falsification_conditions", "uncertainty_sources"):
            val = conclusion.get(fld)
            if val is None:
                conclusion[fld] = []
            elif not isinstance(val, list):
                conclusion[fld] = [str(val)]
            else:
                conclusion[fld] = [str(s) for s in val if s]

        # Ground key_signals: require metric:value format, cap at 5
        _signals = conclusion.get("key_signals", [])
        _grounded = [s for s in _signals if ":" in s]
        _ungrounded = [s for s in _signals if ":" not in s]
        if _ungrounded:
            logger.debug("Dropped %d ungrounded key_signals", len(_ungrounded))
        conclusion["key_signals"] = _grounded[:5]

        # Ensure executive_summary is a string
        es = conclusion.get("executive_summary")
        if not isinstance(es, str):
            conclusion["executive_summary"] = str(es) if es else ""

        # Enforce: executive_summary must address what/why/watch
        es = conclusion.get("executive_summary", "")
        if es and len(es) < 20:
            logger.warning("Executive summary too short (%d chars), may lack structure", len(es))

        # Enforce: at least 1 falsification + 1 uncertainty when confidence < threshold
        if conclusion["confidence"] < 0.8:
            if not conclusion.get("falsification_conditions"):
                conclusion["falsification_conditions"] = [
                    "reversal of the primary metric trend would invalidate this classification"
                ]
            if not conclusion.get("uncertainty_sources"):
                conclusion["uncertainty_sources"] = [
                    "limited temporal depth or review volume"
                ]

        return conclusion

    @classmethod
    def _validate_conclusion(
        cls,
        conclusion: dict[str, Any],
        *,
        evidence_keys: set[str] | None = None,
    ) -> list[str]:
        """Return validation errors for a normalized reasoning conclusion."""
        if not isinstance(conclusion, dict):
            return ["conclusion is not an object"]
        if conclusion.get("_parse_fallback"):
            return ["llm output was not valid json"]
        if "error" in conclusion:
            return [str(conclusion["error"])]

        errors: list[str] = []
        required = (
            "archetype",
            "confidence",
            "risk_level",
            "executive_summary",
            "key_signals",
            "falsification_conditions",
            "uncertainty_sources",
        )
        for field in required:
            if field not in conclusion:
                errors.append(f"missing field: {field}")

        archetype = conclusion.get("archetype")
        if archetype not in cls._VALID_ARCHETYPES:
            errors.append(f"invalid archetype: {archetype!r}")

        secondary = conclusion.get("secondary_archetype")
        if secondary is not None and secondary not in cls._VALID_ARCHETYPES:
            errors.append(f"invalid secondary_archetype: {secondary!r}")

        confidence = conclusion.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
            errors.append("confidence must be a number between 0 and 1")

        risk_level = conclusion.get("risk_level")
        if risk_level not in cls._VALID_RISK_LEVELS:
            errors.append(f"invalid risk_level: {risk_level!r}")

        executive_summary = conclusion.get("executive_summary")
        if not isinstance(executive_summary, str) or len(executive_summary.strip()) < 20:
            errors.append("executive_summary is missing or too short")

        key_signals = conclusion.get("key_signals")
        if not isinstance(key_signals, list) or not key_signals:
            errors.append("key_signals must contain at least one grounded signal")
        else:
            grounded = [
                signal for signal in key_signals
                if cls._is_grounded_signal(signal, evidence_keys)
            ]
            if not grounded:
                errors.append("key_signals do not reference known evidence metrics")

        falsification_conditions = conclusion.get("falsification_conditions")
        if not isinstance(falsification_conditions, list) or not falsification_conditions:
            errors.append("at least one falsification_condition is required")

        uncertainty_sources = conclusion.get("uncertainty_sources")
        if not isinstance(uncertainty_sources, list) or not uncertainty_sources:
            errors.append("at least one uncertainty_source is required")
        elif float(confidence or 0.0) < 0.6 and len(uncertainty_sources) < 2:
            errors.append("confidence < 0.6 requires at least 2 uncertainty_sources")

        return errors

    @classmethod
    def _is_grounded_signal(cls, signal: Any, evidence_keys: set[str] | None = None) -> bool:
        if not isinstance(signal, str) or ":" not in signal:
            return False
        metric, value = signal.split(":", 1)
        metric = metric.strip().lower().replace(" ", "_")
        if not metric or not value.strip():
            return False
        if not evidence_keys:
            return True
        if metric in evidence_keys or metric in cls._CORE_SIGNAL_FIELDS:
            return True
        if metric.endswith("_score") and "archetype_scores" in evidence_keys:
            return True
        return any(key.startswith(metric) or metric.startswith(key) for key in evidence_keys)

    async def _invalidate_invalid_cache_entry(self, pattern_sig: str, reason: str) -> None:
        try:
            await self._cache.invalidate(pattern_sig, reason=reason)
        except Exception:
            logger.debug("Failed to invalidate cache entry %s", pattern_sig, exc_info=True)

    # Fields that define the core signal. Changes to these should always
    # trigger full reasoning in the differential engine.
    _CORE_SIGNAL_FIELDS = frozenset({
        "churn_density", "avg_urgency", "churn_intent", "total_reviews",
        "dm_churn_rate", "price_complaint_rate", "displacement_mention_count",
    })

    # Fields with low signal value -- remove to reduce noise
    _LOW_VALUE_FIELDS = frozenset({
        "recommend_yes", "recommend_no", "product_category",
    })

    @staticmethod
    def _prepare_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
        """Normalize, deduplicate, and rank evidence before LLM consumption.

        - Normalizes metric names to canonical forms with consistent units
        - Removes low-value and duplicate signals
        - Pre-ranks pain/competitor/feature lists by count/mentions descending
        - Keeps archetype pre-scores, temporal anomalies, and displacement compact
        """
        ev = {}

        # 1. Copy core metrics first (highest signal, LLM sees them first)
        for key in (
            "vendor_name", "churn_density", "avg_urgency", "total_reviews",
            "churn_intent", "dm_churn_rate", "price_complaint_rate",
            "displacement_mention_count", "insider_signal_count",
            "keyword_spike_count",
        ):
            if key in evidence:
                val = evidence[key]
                # Normalize numerics to consistent precision
                if isinstance(val, float):
                    ev[key] = round(val, 2)
                else:
                    ev[key] = val

        # 2. Pre-ranked lists (sorted by relevance, capped)
        for list_key, sort_field, cap in (
            ("pain_categories", "count", 5),
            ("competitors", "mentions", 5),
            ("feature_gaps", "mentions", 5),
            ("top_use_cases", None, 3),
        ):
            items = evidence.get(list_key)
            if items and isinstance(items, list):
                if sort_field:
                    items = sorted(
                        items, key=lambda x: -(x.get(sort_field, 0) if isinstance(x, dict) else 0),
                    )
                ev[list_key] = items[:cap]

        # 3. Archetype pre-scores (compact: top 3 only)
        arch_scores = evidence.get("archetype_scores")
        if arch_scores and isinstance(arch_scores, list):
            ev["archetype_scores"] = arch_scores[:3]

        # 4. Temporal anomalies (only anomalies that fired, plus velocities)
        anomalies = evidence.get("anomalies")
        if anomalies and isinstance(anomalies, list):
            ev["anomalies"] = [a for a in anomalies if isinstance(a, dict) and a.get("is_anomaly")]
        for key in evidence:
            if key.startswith("velocity_") or key.startswith("accel_"):
                val = evidence[key]
                if val is not None and val != 0:
                    ev[key] = round(val, 4) if isinstance(val, float) else val

        # 5. Compact context (budget, buyer authority -- single-value summaries)
        for ctx_key in ("budget_context", "buyer_authority"):
            val = evidence.get(ctx_key)
            if val and isinstance(val, dict):
                ev[ctx_key] = val

        # 6. Quote evidence (just count + best quote, not full list)
        if evidence.get("quote_count"):
            ev["quote_count"] = evidence["quote_count"]
        if evidence.get("top_quote"):
            ev["top_quote"] = str(evidence["top_quote"])[:200]

        # 7. Snapshot days (temporal depth indicator)
        if evidence.get("snapshot_days"):
            ev["snapshot_days"] = evidence["snapshot_days"]

        # Skip low-value fields entirely (recommend_yes/no, product_category already captured)

        return ev

    async def analyze(
        self,
        vendor_name: str,
        evidence: dict[str, Any],
        *,
        product_category: str = "",
        force_reason: bool = False,
        tier_context: dict[str, Any] | None = None,
    ) -> ReasoningResult:
        """Main entry point. Decides: recall, reconstitute, or reason.

        Flow:
            1. Check metacognition for forced exploration/surprise
            2. Try recall (semantic cache, same evidence hash)
            3. If evidence changed, try reconstitute (diff < 30%)
            4. Fall back to full reason
        """
        ev_hash = compute_evidence_hash(evidence)
        pattern_sig = self._build_pattern_sig(vendor_name, ev_hash)

        # 0. Metacognitive overrides
        if not force_reason and self._meta:
            if self._meta.should_force_exploration(ev_hash):
                logger.info("Exploration sample forced for %s", vendor_name)
                force_reason = True

        # 1. Try recall (unless forced)
        if not force_reason:
            cached = await self._recall(pattern_sig, ev_hash)
            if cached is not None:
                conclusion_type = cached.conclusion.get("archetype", "")
                # Surprise detection: rare conclusion type -> escalate to full reason
                if self._meta and conclusion_type:
                    try:
                        if await self._meta.is_surprise(conclusion_type):
                            logger.info(
                                "Surprise escalation for %s (type=%s)",
                                vendor_name, conclusion_type,
                            )
                            force_reason = True
                        else:
                            await self._log_metacognition("recall", 0, conclusion_type)
                            return cached
                    except Exception:
                        logger.debug("Surprise check failed, using cached", exc_info=True)
                        await self._log_metacognition("recall", 0, conclusion_type)
                        return cached
                else:
                    await self._log_metacognition("recall", 0, conclusion_type)
                    return cached

        # 2. Try reconstitute: cache hit on vendor but evidence changed
        if not force_reason:
            reconstituted = await self._try_reconstitute(
                vendor_name, evidence, ev_hash, pattern_sig, product_category,
            )
            if reconstituted is not None:
                await self._log_metacognition(
                    "reconstitute", reconstituted.tokens_used,
                    reconstituted.conclusion.get("archetype", ""),
                )
                return reconstituted

        # 3. Find similar traces for LLM context
        prior_traces = await self._find_prior_context(evidence, vendor_name)

        # 4. Full reason
        result = await self._reason(
            vendor_name, evidence, prior_traces,
            product_category=product_category,
            pattern_sig=pattern_sig,
            evidence_hash=ev_hash,
            tier_context=tier_context,
        )
        await self._log_metacognition("reason", result.tokens_used, result.conclusion.get("archetype", ""))
        return result

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    async def _recall(self, pattern_sig: str, evidence_hash: str) -> ReasoningResult | None:
        """Try semantic cache. Return None on miss or stale."""
        entry = await self._cache.lookup(pattern_sig)
        if entry is None:
            return None

        # If evidence changed since cache was stored, don't recall
        if entry.evidence_hash and entry.evidence_hash != evidence_hash:
            logger.info(
                "Cache hit for %s but evidence changed (old=%s new=%s)",
                pattern_sig, entry.evidence_hash, evidence_hash,
            )
            return None

        normalized = self._normalize_conclusion(entry.conclusion)
        validation_errors = self._validate_conclusion(normalized)
        if validation_errors:
            logger.warning(
                "Invalid cached reasoning for %s; treating as miss: %s",
                pattern_sig, "; ".join(validation_errors[:3]),
            )
            await self._invalidate_invalid_cache_entry(pattern_sig, "invalid reasoning payload")
            return None

        return ReasoningResult(
            mode="recall",
            conclusion=normalized,
            confidence=entry.effective_confidence or entry.confidence,
            pattern_sig=pattern_sig,
            evidence_hash=evidence_hash,
            tokens_used=0,
            cached=True,
        )

    # ------------------------------------------------------------------
    # Reconstitute (Phase 2)
    # ------------------------------------------------------------------

    async def _try_reconstitute(
        self,
        vendor_name: str,
        new_evidence: dict[str, Any],
        new_ev_hash: str,
        new_pattern_sig: str,
        product_category: str,
    ) -> ReasoningResult | None:
        """Try to patch an existing cached conclusion with the evidence delta.

        Looks for a prior cache entry for the same vendor (any evidence hash).
        If found and the diff ratio is < 30%, uses the LLM to patch only the
        delta instead of re-reasoning from scratch.
        """
        # Find any active cache entry for this vendor
        entries = await self._cache.lookup_by_class("", vendor_name=vendor_name, limit=1)
        if not entries:
            return None

        entry = entries[0]
        if entry.effective_confidence is not None and entry.effective_confidence < 0.3:
            return None  # too stale to reconstitute from

        cached_validation_errors = self._validate_conclusion(
            self._normalize_conclusion(entry.conclusion),
        )
        if cached_validation_errors:
            await self._invalidate_invalid_cache_entry(
                entry.pattern_sig, "invalid reconstitution source payload",
            )
            return None

        # Reconstruct old evidence from the stored evidence hash
        # We need the actual old evidence for diffing -- check episodic store
        old_evidence = await self._load_old_evidence(entry.pattern_sig)
        if not old_evidence:
            return None

        # Classify the diff
        diff = classify_evidence(old_evidence, new_evidence)
        logger.info("Diff for %s: %s", vendor_name, diff.summary())

        if not diff.should_reconstitute:
            return None  # delta too large, need full reason

        # Patch the conclusion with only the delta
        updated_conclusion, tokens = await reconstitute(
            vendor_name, entry.conclusion, diff, new_evidence,
        )
        updated_conclusion = self._normalize_conclusion(updated_conclusion)
        validation_errors = self._validate_conclusion(
            updated_conclusion,
            evidence_keys=set(self._prepare_evidence(new_evidence).keys()),
        )

        if not updated_conclusion or "error" in updated_conclusion or validation_errors:
            if validation_errors:
                logger.warning(
                    "Discarding weak reconstitution for %s; escalating to full reason: %s",
                    vendor_name, "; ".join(validation_errors[:3]),
                )
            return None

        new_confidence = updated_conclusion.get("confidence", entry.confidence)
        archetype = updated_conclusion.get("archetype", entry.conclusion_type)

        # Update semantic cache with new conclusion + evidence hash
        updated_entry = CacheEntry(
            pattern_sig=new_pattern_sig,
            pattern_class=archetype,
            vendor_name=vendor_name,
            product_category=product_category,
            conclusion=updated_conclusion,
            confidence=new_confidence,
            reasoning_steps=entry.reasoning_steps,
            boundary_conditions=entry.boundary_conditions,
            falsification_conditions=updated_conclusion.get("falsification_conditions", []),
            uncertainty_sources=updated_conclusion.get("uncertainty_sources", []),
            conclusion_type=archetype,
            evidence_hash=new_ev_hash,
        )
        try:
            await self._cache.store(updated_entry)
            # Invalidate the old entry if pattern_sig changed
            if entry.pattern_sig != new_pattern_sig:
                await self._cache.invalidate(entry.pattern_sig, reason="superseded by reconstitute")
        except Exception:
            logger.warning("Failed to update cache after reconstitute", exc_info=True)

        return ReasoningResult(
            mode="reconstitute",
            conclusion=updated_conclusion,
            confidence=new_confidence,
            pattern_sig=new_pattern_sig,
            evidence_hash=new_ev_hash,
            tokens_used=tokens,
            cached=False,
        )

    async def _load_old_evidence(self, pattern_sig: str) -> dict[str, Any] | None:
        """Load prior evidence for reconstitution diffing.

        Prefers the full raw_evidence JSON stored on the trace node.
        Falls back to reconstructing from truncated EvidenceNode values.
        """
        try:
            driver = await self._episodic._get_driver()
            async with driver.session() as session:
                # Try raw_evidence first (lossless)
                result = await session.run(
                    """
                    MATCH (t:ReasoningTrace {pattern_sig: $sig, group_id: $gid})
                    RETURN t.raw_evidence AS raw_evidence
                    """,
                    sig=pattern_sig,
                    gid="b2b-reasoning",
                )
                record = await result.single()
                if record and record["raw_evidence"]:
                    try:
                        return json.loads(record["raw_evidence"])
                    except (json.JSONDecodeError, TypeError):
                        pass

                # Fallback: reconstruct from truncated evidence nodes
                result = await session.run(
                    """
                    MATCH (t:ReasoningTrace {pattern_sig: $sig, group_id: $gid})
                          -[:SUPPORTED_BY]->(e:EvidenceNode)
                    RETURN e.type AS type, e.source AS source, e.value AS value
                    """,
                    sig=pattern_sig,
                    gid="b2b-reasoning",
                )
                evidence = {}
                async for rec in result:
                    key = rec["type"] or rec["source"]
                    raw = rec["value"]
                    try:
                        val = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        val = raw
                    if key in evidence:
                        if isinstance(evidence[key], list):
                            evidence[key].append(val)
                        else:
                            evidence[key] = [evidence[key], val]
                    else:
                        evidence[key] = val
                return evidence if evidence else None
        except Exception:
            logger.debug("Failed to load old evidence for %s", pattern_sig, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Find prior context
    # ------------------------------------------------------------------

    async def _find_prior_context(
        self, evidence: dict[str, Any], vendor_name: str
    ) -> list[ReasoningTrace]:
        """Find similar episodic traces to provide context to the LLM."""
        if getattr(self._episodic, "_degraded", False):
            return []
        try:
            summary = self._evidence_summary(evidence, vendor_name)
            embedding = self._episodic.embed_text(summary)
            traces = await self._episodic.find_similar(embedding, limit=3)
            if traces:
                logger.info(
                    "Found %d similar traces (best=%.3f)",
                    len(traces), traces[0].similarity_score or 0,
                )
            return traces
        except Exception:
            logger.warning("Episodic similarity search failed", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Reason
    # ------------------------------------------------------------------

    async def _reason(
        self,
        vendor_name: str,
        evidence: dict[str, Any],
        prior_traces: list[ReasoningTrace],
        *,
        product_category: str = "",
        pattern_sig: str = "",
        evidence_hash: str = "",
        tier_context: dict[str, Any] | None = None,
    ) -> ReasoningResult:
        """Full LLM synthesis. Store result in both memories."""
        from ..pipelines.llm import parse_json_response
        from ..services.protocols import Message

        # Build user payload
        prepared = self._prepare_evidence(evidence)
        payload = {
            "vendor_name": vendor_name,
            "product_category": product_category,
            "evidence": prepared,
        }
        if tier_context and tier_context.get("inherited_priors"):
            payload["tier_context"] = tier_context
        if prior_traces:
            payload["similar_patterns"] = [
                {
                    "vendor": t.vendor_name,
                    "archetype": t.conclusion_type,
                    "confidence": t.confidence,
                    "similarity": t.similarity_score,
                }
                for t in prior_traces
            ]

        from .config import ReasoningConfig
        _rcfg = ReasoningConfig()
        llm = resolve_stratified_llm(_rcfg)
        if llm is None:
            logger.error(
                "No LLM available for stratified reasoning (workload=%s)",
                _rcfg.stratified_llm_workload,
            )
            return ReasoningResult(
                mode="reason",
                conclusion={"error": "no_llm_available"},
                confidence=0.0,
                pattern_sig=pattern_sig,
                evidence_hash=evidence_hash,
                tokens_used=0,
                cached=False,
            )

        messages = [
            Message(role="system", content=_REASON_SYSTEM_PROMPT),
            Message(
                role="user",
                content=json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str),
            ),
        ]

        t0 = time.monotonic()
        try:
            result = llm.chat(
                messages=messages,
                max_tokens=_rcfg.max_tokens,
                temperature=_rcfg.temperature,
                guided_json=REASONING_CONCLUSION_JSON_SCHEMA,
                response_format={"type": "json_object"},
            )
            text = result.get("response", "").strip()
            usage = result.get("usage", {})
            tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

            # Clean think tags
            import re
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

            conclusion = parse_json_response(text, recover_truncated=True)
            conclusion = self._normalize_conclusion(conclusion)
        except Exception:
            logger.exception("LLM reasoning failed for %s", vendor_name)
            return ReasoningResult(
                mode="reason",
                conclusion={"error": "llm_failed"},
                confidence=0.0,
                pattern_sig=pattern_sig,
                evidence_hash=evidence_hash,
                tokens_used=0,
                cached=False,
            )

        confidence = conclusion.get("confidence", 0.5)
        archetype = conclusion.get("archetype", "unknown")
        duration_ms = (time.monotonic() - t0) * 1000

        validation_errors = self._validate_conclusion(
            conclusion,
            evidence_keys=set(prepared.keys()),
        )
        if validation_errors:
            logger.warning(
                "Rejected invalid reasoning output for %s: %s",
                vendor_name, "; ".join(validation_errors[:4]),
            )
            return ReasoningResult(
                mode="reason",
                conclusion={
                    "error": "invalid_reasoning_output",
                    "validation_errors": validation_errors,
                },
                confidence=0.0,
                pattern_sig=pattern_sig,
                evidence_hash=evidence_hash,
                tokens_used=tokens,
                cached=False,
            )

        logger.info(
            "Reasoned %s -> %s (conf=%.2f, %d tokens, %.0fms)",
            vendor_name, archetype, confidence, tokens, duration_ms,
        )

        # Merge LLM + archetype-specific falsification conditions
        llm_conds = conclusion.get("falsification_conditions", [])
        try:
            from .archetypes import get_falsification_conditions
            archetype_conds = get_falsification_conditions(archetype)
            all_conds = list(set(llm_conds + archetype_conds))
        except Exception:
            all_conds = llm_conds
        conclusion["falsification_conditions"] = all_conds

        # Store in semantic cache
        cache_entry = CacheEntry(
            pattern_sig=pattern_sig,
            pattern_class=archetype,
            vendor_name=vendor_name,
            product_category=product_category,
            conclusion=conclusion,
            confidence=confidence,
            reasoning_steps=[],
            boundary_conditions={},
            falsification_conditions=all_conds,
            uncertainty_sources=conclusion.get("uncertainty_sources", []),
            conclusion_type=archetype,
            evidence_hash=evidence_hash,
        )
        try:
            await self._cache.store(cache_entry)
        except Exception:
            logger.warning("Failed to store semantic cache for %s", pattern_sig, exc_info=True)

        # Store in episodic memory (skip when degraded)
        trace_id = None
        if getattr(self._episodic, "_degraded", False):
            logger.debug("Episodic store degraded, skipping trace storage for %s", vendor_name)
        else:
            trace_id = await self._store_episodic_trace(
                vendor_name, product_category, archetype, confidence,
                pattern_sig, evidence, evidence_nodes=None, conclusion=conclusion,
            )

        return ReasoningResult(
            mode="reason",
            conclusion=conclusion,
            confidence=confidence,
            pattern_sig=pattern_sig,
            evidence_hash=evidence_hash,
            tokens_used=tokens,
            cached=False,
            trace_id=trace_id,
        )

    async def _store_episodic_trace(
        self,
        vendor_name: str,
        product_category: str,
        archetype: str,
        confidence: float,
        pattern_sig: str,
        evidence: dict[str, Any],
        *,
        evidence_nodes: list[EvidenceNode] | None = None,
        conclusion: dict[str, Any] | None = None,
    ) -> str | None:
        """Store a reasoning trace in episodic memory. Returns trace_id or None."""
        trace_id = None
        try:
            evidence_summary = self._evidence_summary(evidence, vendor_name)
            try:
                embedding = self._episodic.embed_text(evidence_summary)
            except Exception:
                logger.debug("embed_text failed for %s, storing trace without embedding", vendor_name)
                embedding = [0.0] * 1024  # placeholder -- vector search won't match, but trace is preserved

            if evidence_nodes is None:
                evidence_nodes = self._build_evidence_nodes(evidence)
            _conc = conclusion or {}
            conclusion_node = ConclusionNode(
                claim=_conc.get("executive_summary", ""),
                confidence=confidence,
                evidence_chain=[e.id for e in evidence_nodes],
            )

            trace = ReasoningTrace(
                vendor_name=vendor_name,
                category=product_category,
                conclusion_type=archetype,
                confidence=confidence,
                pattern_sig=pattern_sig,
                trace_embedding=embedding,
                evidence=evidence_nodes,
                conclusions=[conclusion_node],
                raw_evidence=evidence,
            )
            trace_id = await self._episodic.store_trace(trace)
        except Exception:
            logger.warning("Failed to store episodic trace for %s", vendor_name, exc_info=True)
        return trace_id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_pattern_sig(self, vendor_name: str, evidence_hash: str) -> str:
        """Deterministic signature: vendor + evidence hash."""
        safe_name = vendor_name.lower().replace(" ", "_").replace(".", "")
        return f"{safe_name}:{evidence_hash}"

    @staticmethod
    def _evidence_summary(evidence: dict[str, Any], vendor_name: str) -> str:
        """Build a text summary of evidence for embedding."""
        parts = [f"Vendor: {vendor_name}"]
        for key, val in evidence.items():
            if isinstance(val, (list, dict)):
                parts.append(f"{key}: {json.dumps(val, default=str)[:200]}")
            else:
                parts.append(f"{key}: {val}")
        return " | ".join(parts)[:1000]

    @staticmethod
    def _build_evidence_nodes(evidence: dict[str, Any]) -> list[EvidenceNode]:
        """Convert evidence dict into EvidenceNode list.

        Values are JSON-serialized to preserve types (int, float, dict, list)
        so that reconstitute diffing can compare accurately.
        """
        import uuid

        def _serialize(v: Any) -> str:
            """JSON-encode non-string values to preserve types for diffing."""
            if isinstance(v, str):
                return v[:500]
            try:
                return json.dumps(v, separators=(",", ":"), default=str)[:500]
            except (TypeError, ValueError):
                return str(v)[:500]

        nodes = []
        for key, val in evidence.items():
            if isinstance(val, list):
                for i, item in enumerate(val[:10]):
                    nodes.append(EvidenceNode(
                        id=str(uuid.uuid4()),
                        type=key,
                        source=key,
                        value=_serialize(item),
                    ))
            else:
                nodes.append(EvidenceNode(
                    id=str(uuid.uuid4()),
                    type=key,
                    source=key,
                    value=_serialize(val),
                ))
        return nodes

    # ------------------------------------------------------------------
    # Metacognition
    # ------------------------------------------------------------------

    async def _log_metacognition(self, mode: str, tokens_used: int, conclusion_type: str = "") -> None:
        """Record reasoning outcome via the metacognitive monitor."""
        if self._meta:
            self._meta.record(mode, tokens_used, conclusion_type)
            return

        # Fallback: direct DB write (Phase 1 compat)
        try:
            pool = self._cache._pool
            today = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0,
            )
            tomorrow = today.replace(hour=23, minute=59, second=59)

            col_map = {"recall": "recall_hits", "reconstitute": "reconstitute_hits", "reason": "full_reasons"}
            col = col_map.get(mode, "full_reasons")
            saved = {"recall": 2000, "reconstitute": 1400, "reason": 0}.get(mode, 0)

            await pool.execute(
                f"""
                INSERT INTO reasoning_metacognition (
                    period_start, period_end, total_queries,
                    {col}, total_tokens_saved, total_tokens_spent
                ) VALUES ($1, $2, 1, 1, $3, $4)
                ON CONFLICT (period_start) DO UPDATE SET
                    total_queries = reasoning_metacognition.total_queries + 1,
                    {col} = reasoning_metacognition.{col} + 1,
                    total_tokens_saved = reasoning_metacognition.total_tokens_saved + $3,
                    total_tokens_spent = reasoning_metacognition.total_tokens_spent + $4
                """,
                today,
                tomorrow,
                saved,
                tokens_used,
            )
        except Exception:
            logger.debug("Metacognition logging failed", exc_info=True)
