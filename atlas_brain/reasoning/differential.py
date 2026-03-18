"""Differential Reasoning Engine (WS0D).

Classifies evidence deltas between a cached reasoning trace and new data,
then decides whether to Reconstitute (LLM patches the conclusion with the
delta only) or escalate to full Reason.

Evidence classification:
    CONFIRMED    - New data supports old evidence
    CONTRADICTED - New data conflicts with old evidence
    MISSING      - Old evidence no longer present in new data
    NOVEL        - New evidence not in original trace

Decision:
    weighted drift scores changes by business value, not just raw field count.
    Pain/theme shifts, buyer-role mix changes, and temporal changes weigh more
    than low-signal cosmetic fields. Missing evidence counts as change.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .llm_utils import REASONING_CONCLUSION_JSON_SCHEMA, resolve_stratified_llm

logger = logging.getLogger("atlas.reasoning.differential")


class EvidenceStatus(str, Enum):
    CONFIRMED = "confirmed"
    CONTRADICTED = "contradicted"
    MISSING = "missing"
    NOVEL = "novel"


@dataclass(frozen=True)
class DiffTuning:
    """Operational tuning for evidence drift scoring."""

    reconstitute_threshold: float = 0.3
    core_weight: float = 4.0
    thematic_weight: float = 3.0
    segment_weight: float = 3.0
    temporal_weight: float = 3.0
    quote_weight: float = 1.5
    minor_weight: float = 1.0
    weighted_sequence_match_threshold: float = 0.2
    strategic_component_threshold: float = 3.0
    contradiction_emergence_threshold: float = 4.0


@dataclass
class EvidenceDiff:
    """Result of diffing old vs new evidence."""

    confirmed: list[str] = field(default_factory=list)
    contradicted: list[tuple[str, str, str]] = field(default_factory=list)  # (key, old, new)
    missing: list[str] = field(default_factory=list)
    novel: list[tuple[str, str]] = field(default_factory=list)  # (key, value)
    tuning: DiffTuning = field(default_factory=DiffTuning)

    @property
    def total(self) -> int:
        return len(self.confirmed) + len(self.contradicted) + len(self.missing) + len(self.novel)

    @property
    def change_count(self) -> int:
        return len(self.contradicted) + len(self.novel) + len(self.missing)

    # Core signal fields where changes matter more
    _CORE_FIELDS = frozenset({
        "churn_density", "avg_urgency", "churn_intent", "total_reviews",
        "dm_churn_rate", "price_complaint_rate", "displacement_mention_count",
    })
    _THEMATIC_FIELDS = frozenset({
        "pain_categories", "competitors", "feature_gaps", "market_regime",
        "support_sentiment", "legacy_support_score", "new_feature_velocity",
        "employee_growth_rate", "keyword_spike_count", "keyword_spike_keywords",
        "insider_signal_count", "insider_talent_drain_rate",
    })
    _SEGMENT_FIELDS = frozenset({
        "buyer_authority", "budget_context", "top_use_cases",
    })
    _QUOTE_FIELDS = frozenset({
        "quote_count", "top_quote",
    })
    _TEMPORAL_FIELDS = frozenset({
        "anomalies", "velocity_trend", "snapshot_days",
        "displacement_velocity_7d", "displacement_velocity_30d",
    })
    _TEMPORAL_PREFIXES = ("velocity_", "accel_", "trend_30d_", "trend_90d_")
    _PAIN_COMPONENT_FIELDS = frozenset({
        "pain_categories", "feature_gaps", "support_sentiment",
        "legacy_support_score", "new_feature_velocity",
    })
    _ROLE_COMPONENT_FIELDS = frozenset({
        "buyer_authority", "top_use_cases", "budget_context",
    })
    _COMPETITIVE_COMPONENT_FIELDS = frozenset({
        "competitors", "displacement_mention_count", "market_regime",
        "displacement_velocity_7d", "displacement_velocity_30d", "velocity_trend",
    })
    _QUOTE_COMPONENT_FIELDS = frozenset({
        "quote_count", "top_quote",
    })

    @property
    def component_scores(self) -> dict[str, float]:
        """Explainable weighted drift by strategic component."""
        scores = {
            "pain_distribution": 0.0,
            "role_mix": 0.0,
            "competitive_shift": 0.0,
            "temporal_shift": 0.0,
            "quote_novelty": 0.0,
            "contradiction_emergence": 0.0,
        }
        for key, _, _ in self.contradicted:
            self._apply_component_weight(scores, key, contradiction=True)
        for key in self.missing:
            self._apply_component_weight(scores, key, contradiction=True)
        for key, _ in self.novel:
            self._apply_component_weight(scores, key, contradiction=False)
        return {name: round(score, 4) for name, score in scores.items() if score > 0}

    @property
    def has_strategic_shift(self) -> bool:
        """True when a single strategic component has materially shifted."""
        scores = self.component_scores
        for name in ("pain_distribution", "role_mix", "competitive_shift", "temporal_shift"):
            if scores.get(name, 0.0) >= self.tuning.strategic_component_threshold:
                return True
        return False

    @property
    def has_contradiction_emergence(self) -> bool:
        """True when contradictory or missing evidence has accumulated materially."""
        return (
            self.component_scores.get("contradiction_emergence", 0.0)
            >= self.tuning.contradiction_emergence_threshold
        )

    @property
    def diff_ratio(self) -> float:
        if self.total == 0:
            return 1.0
        return self.change_count / self.total

    @property
    def weighted_diff_ratio(self) -> float:
        """Importance-weighted diff ratio."""
        if self.total == 0:
            return 1.0
        weighted_changes = 0.0
        weighted_total = 0.0
        for key, _, _ in self.contradicted:
            w = self._weight_for_key(key)
            weighted_changes += w
            weighted_total += w
        for key, _ in self.novel:
            w = self._weight_for_key(key)
            weighted_changes += w
            weighted_total += w
        for key in self.missing:
            w = self._weight_for_key(key)
            weighted_changes += w
            weighted_total += w
        for key in self.confirmed:
            w = self._weight_for_key(key)
            weighted_total += w
        return weighted_changes / weighted_total if weighted_total > 0 else 1.0

    @property
    def has_core_contradiction(self) -> bool:
        """True if any core signal field is contradicted."""
        return any(key in self._CORE_FIELDS for key, _, _ in self.contradicted)

    @property
    def should_reconstitute(self) -> bool:
        """Use weighted ratio and block reconstitution on core contradictions."""
        if self.total == 0:
            return False
        if self.has_core_contradiction:
            return False  # core signal changed -> full reason always
        if self.has_strategic_shift:
            return False
        if self.has_contradiction_emergence:
            return False
        return self.weighted_diff_ratio < self.tuning.reconstitute_threshold

    def _weight_for_key(self, key: str) -> float:
        if key in self._CORE_FIELDS:
            return self.tuning.core_weight
        if key in self._THEMATIC_FIELDS:
            return self.tuning.thematic_weight
        if key in self._SEGMENT_FIELDS:
            return self.tuning.segment_weight
        if key in self._QUOTE_FIELDS:
            return self.tuning.quote_weight
        if key in self._TEMPORAL_FIELDS or key.startswith(self._TEMPORAL_PREFIXES):
            return self.tuning.temporal_weight
        return self.tuning.minor_weight

    def _apply_component_weight(
        self,
        scores: dict[str, float],
        key: str,
        *,
        contradiction: bool,
    ) -> None:
        weight = self._weight_for_key(key)
        if key in self._PAIN_COMPONENT_FIELDS:
            scores["pain_distribution"] += weight
        if key in self._ROLE_COMPONENT_FIELDS:
            scores["role_mix"] += weight
        if key in self._COMPETITIVE_COMPONENT_FIELDS:
            scores["competitive_shift"] += weight
        if key in self._QUOTE_COMPONENT_FIELDS:
            scores["quote_novelty"] += weight
        if key in self._TEMPORAL_FIELDS or key.startswith(self._TEMPORAL_PREFIXES):
            scores["temporal_shift"] += weight
        if contradiction:
            scores["contradiction_emergence"] += weight

    def summary(self) -> str:
        parts = []
        if self.contradicted:
            parts.append(f"{len(self.contradicted)} contradicted")
        if self.novel:
            parts.append(f"{len(self.novel)} novel")
        if self.missing:
            parts.append(f"{len(self.missing)} missing")
        if self.confirmed:
            parts.append(f"{len(self.confirmed)} confirmed")
        ratio_pct = self.weighted_diff_ratio * 100
        mode = "reconstitute" if self.should_reconstitute else "full_reason"
        core = " [core contradiction]" if self.has_core_contradiction else ""
        top_components = sorted(
            self.component_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:2]
        comp_text = ""
        if top_components:
            comp_text = " [" + ", ".join(f"{name}={score:.1f}" for name, score in top_components) + "]"
        return f"weighted_diff={ratio_pct:.0f}% ({', '.join(parts)}){core}{comp_text} -> {mode}"


def build_diff_tuning(cfg: Any | None = None) -> DiffTuning:
    """Build differential tuning from ReasoningConfig-like settings."""
    if cfg is None:
        return DiffTuning()
    return DiffTuning(
        reconstitute_threshold=float(getattr(cfg, "reconstitute_threshold", 0.3)),
        core_weight=float(getattr(cfg, "reconstitute_core_weight", 4.0)),
        thematic_weight=float(getattr(cfg, "reconstitute_thematic_weight", 3.0)),
        segment_weight=float(getattr(cfg, "reconstitute_segment_weight", 3.0)),
        temporal_weight=float(getattr(cfg, "reconstitute_temporal_weight", 3.0)),
        quote_weight=float(getattr(cfg, "reconstitute_quote_weight", 1.5)),
        minor_weight=float(getattr(cfg, "reconstitute_minor_weight", 1.0)),
        weighted_sequence_match_threshold=float(
            getattr(cfg, "reconstitute_weighted_sequence_match_threshold", 0.2),
        ),
        strategic_component_threshold=float(getattr(cfg, "reconstitute_strategic_component_threshold", 3.0)),
        contradiction_emergence_threshold=float(getattr(cfg, "reconstitute_contradiction_emergence_threshold", 4.0)),
    )


def classify_evidence(
    old_evidence: dict[str, Any],
    new_evidence: dict[str, Any],
    *,
    numeric_tolerance: float = 0.05,
    tuning: DiffTuning | None = None,
) -> EvidenceDiff:
    """Classify each evidence field as confirmed/contradicted/missing/novel.

    For numeric values, a relative change within *numeric_tolerance* (5%) is
    treated as confirmed.  For lists, element-level comparison (order-insensitive).
    For strings, exact match.
    """
    diff = EvidenceDiff(tuning=tuning or DiffTuning())
    old_keys = set(old_evidence.keys())
    new_keys = set(new_evidence.keys())

    # Missing: in old but not in new
    for key in old_keys - new_keys:
        diff.missing.append(key)

    # Novel: in new but not in old
    for key in new_keys - old_keys:
        diff.novel.append((key, str(new_evidence[key])[:200]))

    # Shared: compare values
    for key in old_keys & new_keys:
        old_val = old_evidence[key]
        new_val = new_evidence[key]

        if _values_match(
            old_val,
            new_val,
            numeric_tolerance,
            diff.tuning.weighted_sequence_match_threshold,
        ):
            diff.confirmed.append(key)
        else:
            diff.contradicted.append((key, str(old_val)[:200], str(new_val)[:200]))

    return diff


def _values_match(
    old: Any,
    new: Any,
    tolerance: float,
    weighted_sequence_match_threshold: float,
) -> bool:
    """Check if two evidence values are effectively the same."""
    # Both numeric
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        if old == 0 and new == 0:
            return True
        if old == 0:
            return abs(new) < tolerance
        return abs(new - old) / abs(old) <= tolerance

    normalized_old = _normalize_weighted_sequence(old)
    normalized_new = _normalize_weighted_sequence(new)
    if normalized_old is not None and normalized_new is not None:
        sub_diff = classify_evidence(
            normalized_old,
            normalized_new,
            numeric_tolerance=tolerance,
            tuning=DiffTuning(
                weighted_sequence_match_threshold=weighted_sequence_match_threshold,
            ),
        )
        return sub_diff.weighted_diff_ratio < weighted_sequence_match_threshold

    # Both lists (order-insensitive comparison)
    if isinstance(old, list) and isinstance(new, list):
        old_set = set(str(x) for x in old)
        new_set = set(str(x) for x in new)
        if not old_set and not new_set:
            return True
        union = old_set | new_set
        intersection = old_set & new_set
        # Jaccard similarity > 0.8 = confirmed
        return len(intersection) / len(union) > 0.8 if union else True

    # Both dicts
    if isinstance(old, dict) and isinstance(new, dict):
        sub_diff = classify_evidence(
            old,
            new,
            numeric_tolerance=tolerance,
            tuning=DiffTuning(
                weighted_sequence_match_threshold=weighted_sequence_match_threshold,
            ),
        )
        return sub_diff.diff_ratio < 0.1  # essentially unchanged

    # String or other: exact match
    return str(old) == str(new)


def _normalize_weighted_sequence(value: Any) -> dict[str, float] | None:
    """Normalize list-of-dict evidence into a weighted mapping for diffing."""
    if not isinstance(value, list) or not value:
        return None
    if not all(isinstance(item, dict) for item in value):
        return None

    label_keys = ("category", "name", "feature", "use_case", "label")
    count_keys = ("count", "mentions", "review_count")

    result: dict[str, float] = {}
    for item in value:
        label = ""
        for key in label_keys:
            raw = item.get(key)
            if raw:
                label = str(raw).strip().lower()
                break
        if not label:
            return None

        weight = 1.0
        for key in count_keys:
            raw_weight = item.get(key)
            if raw_weight is None:
                continue
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError):
                weight = 1.0
            break

        result[label] = weight

    return result or None


# ---------------------------------------------------------------------------
# Reconstitute: LLM patches conclusion with delta only
# ---------------------------------------------------------------------------

_RECONSTITUTE_SYSTEM_PROMPT = """\
You are updating a B2B churn intelligence conclusion with new evidence.

The original conclusion is provided along with the specific changes (delta).
Most evidence is unchanged -- only update the parts affected by the delta.

Output the UPDATED conclusion as JSON with the same fields:
{
  "archetype": "<same or changed>",
  "confidence": <adjusted 0.0-1.0>,
  "executive_summary": "<updated 2-3 sentences>",
  "key_signals": ["<signal1>", ...],
  "risk_level": "<low|medium|high|critical>",
  "falsification_conditions": ["<updated conditions>"],
  "uncertainty_sources": ["<updated sources>"]
}

Rules:
- Only change fields affected by the delta
- Adjust confidence based on whether delta strengthens or weakens the conclusion
- If delta contradicts the archetype classification, you may change it
- Keep executive_summary concise -- mention what changed\
"""


async def persist_evidence_diff(
    pool,
    vendor_name: str,
    diff: EvidenceDiff | None,
    decision: str,
) -> None:
    """Persist an evidence diff to reasoning_evidence_diffs table.

    When *diff* is None (cold full-reason or pure recall with no prior
    evidence), a zero-diff row is written so every vendor gets a row
    every run. Contradicted/novel field lists are truncated to 20 items.
    """
    compared = diff is not None
    if diff is None:
        confirmed = contradicted = novel = missing = total = 0
        ratio = w_ratio = 0.0
        core = False
        component_scores = "{}"
        c_fields = "[]"
        n_fields = "[]"
    else:
        confirmed = len(diff.confirmed)
        contradicted = len(diff.contradicted)
        novel = len(diff.novel)
        missing = len(diff.missing)
        total = diff.total
        ratio = round(diff.diff_ratio, 4)
        w_ratio = round(diff.weighted_diff_ratio, 4)
        core = diff.has_core_contradiction
        component_scores = json.dumps(diff.component_scores)
        c_fields = json.dumps([{"key": k, "old": o, "new": n} for k, o, n in diff.contradicted[:20]])
        n_fields = json.dumps([{"key": k, "value": v} for k, v in diff.novel[:20]])

    try:
        await pool.execute(
            """
            INSERT INTO reasoning_evidence_diffs (
                vendor_name, computed_date,
                confirmed_count, contradicted_count, novel_count, missing_count,
                total_fields, diff_ratio, weighted_diff_ratio,
                has_core_contradiction, component_scores, decision, compared,
                contradicted_fields, novel_fields
            ) VALUES ($1, CURRENT_DATE, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11, $12,
                      $13::jsonb, $14::jsonb)
            ON CONFLICT (vendor_name, computed_date) DO UPDATE SET
                confirmed_count = EXCLUDED.confirmed_count,
                contradicted_count = EXCLUDED.contradicted_count,
                novel_count = EXCLUDED.novel_count,
                missing_count = EXCLUDED.missing_count,
                total_fields = EXCLUDED.total_fields,
                diff_ratio = EXCLUDED.diff_ratio,
                weighted_diff_ratio = EXCLUDED.weighted_diff_ratio,
                has_core_contradiction = EXCLUDED.has_core_contradiction,
                component_scores = EXCLUDED.component_scores,
                decision = EXCLUDED.decision,
                compared = EXCLUDED.compared,
                contradicted_fields = EXCLUDED.contradicted_fields,
                novel_fields = EXCLUDED.novel_fields,
                created_at = NOW()
            """,
            vendor_name,
            confirmed, contradicted, novel, missing, total,
            ratio, w_ratio, core, component_scores, decision, compared,
            c_fields, n_fields,
        )
    except Exception:
        logger.debug("Failed to persist evidence diff for %s", vendor_name, exc_info=True)


async def reconstitute(
    vendor_name: str,
    old_conclusion: dict[str, Any],
    diff: EvidenceDiff,
    new_evidence: dict[str, Any],
    *,
    product_category: str = "",
) -> tuple[dict[str, Any], int]:
    """Patch an existing conclusion with only the evidence delta.

    Returns (updated_conclusion, tokens_used).
    """
    from ..pipelines.llm import parse_json_response, trace_llm_call
    from ..services.protocols import Message
    from ..services.tracing import build_business_trace_context

    # Build a compact delta payload (not the full evidence)
    delta = {
        "vendor_name": vendor_name,
        "original_conclusion": old_conclusion,
        "changes": {
            "contradicted": [
                {"field": k, "was": o, "now": n}
                for k, o, n in diff.contradicted
            ],
            "novel": [
                {"field": k, "value": v} for k, v in diff.novel
            ],
            "missing": diff.missing,
            "confirmed_count": len(diff.confirmed),
            "diff_ratio": round(diff.diff_ratio, 3),
        },
    }

    from .config import ReasoningConfig
    from .llm_utils import resolve_stratified_llm_light
    _rcfg = ReasoningConfig()
    llm = resolve_stratified_llm_light(_rcfg)  # reconstitute uses light model
    if llm is None:
        logger.error(
            "No LLM available for reconstitution (workload=%s)",
            _rcfg.stratified_llm_workload,
        )
        return old_conclusion, 0

    trace_metadata = {
        "workflow": "stratified_reasoning",
        "phase": "stratified_reconstitute",
        "reasoning_mode": "reconstitute",
        "vendor_name": vendor_name,
    }
    if product_category:
        trace_metadata["product_category"] = product_category
    business = build_business_trace_context(
        workflow="stratified_reasoning",
        vendor_name=vendor_name,
        product=product_category or None,
    )
    if business:
        trace_metadata["business"] = business

    if _rcfg.multi_pass_enabled:
        from .multi_pass import multi_pass_reason

        try:
            mp_result = await multi_pass_reason(
                llm=llm,
                system_prompt=_RECONSTITUTE_SYSTEM_PROMPT,
                evidence_payload=delta,
                json_schema=REASONING_CONCLUSION_JSON_SCHEMA,
                max_tokens=min(_rcfg.max_tokens, _rcfg.reconstitute_max_tokens),
                temperature=_rcfg.temperature,
                verify_enabled=_rcfg.multi_pass_verify_enabled,
                verify_min_reviews=_rcfg.multi_pass_verify_min_reviews,
                verify_min_snapshot_days=_rcfg.multi_pass_verify_min_snapshot_days,
                verify_min_grounded_signals=_rcfg.multi_pass_verify_min_grounded_signals,
                verify_confidence_cap=_rcfg.multi_pass_verify_confidence_cap,
                light_pass_max_tokens=_rcfg.reconstitute_max_tokens,
                ground_only=True,  # skip challenge, run classify -> ground
                span_prefix="reasoning.stratified.reconstitute",
                trace_metadata=trace_metadata,
            )
            logger.info(
                "Reconstituted %s (multi-pass, diff_ratio=%.0f%%, %d tokens, %.0fms, %d passes)",
                vendor_name, diff.diff_ratio * 100,
                mp_result.total_tokens, mp_result.total_duration_ms,
                mp_result.passes_executed,
            )
            return mp_result.final_conclusion, mp_result.total_tokens
        except Exception:
            logger.exception("Multi-pass reconstitution failed for %s", vendor_name)
            return old_conclusion, 0

    messages = [
        Message(role="system", content=_RECONSTITUTE_SYSTEM_PROMPT),
        Message(
            role="user",
            content=json.dumps(delta, separators=(",", ":"), sort_keys=True, default=str),
        ),
    ]

    t0 = time.monotonic()
    try:
        result = llm.chat(
            messages=messages,
            max_tokens=min(_rcfg.max_tokens, _rcfg.reconstitute_max_tokens),
            temperature=_rcfg.temperature,
            guided_json=REASONING_CONCLUSION_JSON_SCHEMA,
            response_format={"type": "json_object"},
        )
        text = result.get("response", "").strip()
        usage = result.get("usage", {})
        tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        trace_meta = result.get("_trace_meta", {})
        duration_ms = (time.monotonic() - t0) * 1000

        trace_llm_call(
            span_name="reasoning.stratified.reconstitute",
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            model=getattr(llm, "model", getattr(llm, "model_id", "")),
            provider=getattr(llm, "name", ""),
            duration_ms=duration_ms,
            metadata=trace_metadata,
            input_data={"messages": [{"role": m.role, "content": m.content[:500]} for m in messages]},
            output_data={"response": text[:2000]} if text else None,
            api_endpoint=trace_meta.get("api_endpoint"),
            provider_request_id=trace_meta.get("provider_request_id"),
            ttft_ms=trace_meta.get("ttft_ms"),
            inference_time_ms=trace_meta.get("inference_time_ms"),
            queue_time_ms=trace_meta.get("queue_time_ms"),
        )

        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        updated = parse_json_response(text, recover_truncated=True)

        logger.info(
            "Reconstituted %s (diff_ratio=%.0f%%, %d tokens, %.0fms)",
            vendor_name, diff.diff_ratio * 100, tokens, duration_ms,
        )
        return updated, tokens

    except Exception as exc:
        trace_llm_call(
            span_name="reasoning.stratified.reconstitute",
            model=getattr(llm, "model", getattr(llm, "model_id", "")),
            provider=getattr(llm, "name", ""),
            duration_ms=(time.monotonic() - t0) * 1000,
            status="failed",
            metadata=trace_metadata,
            error_message=str(exc)[:500],
            error_type=type(exc).__name__,
            input_data={"messages": [{"role": m.role, "content": m.content[:500]} for m in messages]},
        )
        logger.exception("Reconstitution LLM call failed for %s", vendor_name)
        return old_conclusion, 0
