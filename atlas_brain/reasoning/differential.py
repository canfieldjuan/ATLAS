"""Differential Reasoning Engine (WS0D).

Classifies evidence deltas between a cached reasoning trace and new data,
then decides whether to Reconstitute (LLM patches the conclusion with the
delta only, ~30% token cost) or escalate to full Reason (~100% cost).

Evidence classification:
    CONFIRMED   - New data supports old evidence (skip)
    CONTRADICTED - New data conflicts with old evidence (must reason)
    MISSING     - Old evidence no longer present in new data (must reason)
    NOVEL       - New evidence not in original trace (must reason)

Decision:
    diff_ratio = (contradicted + novel) / total
    if diff_ratio < RECONSTITUTE_THRESHOLD (0.3): Reconstitute
    else: full Reason
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

RECONSTITUTE_THRESHOLD = 0.3


class EvidenceStatus(str, Enum):
    CONFIRMED = "confirmed"
    CONTRADICTED = "contradicted"
    MISSING = "missing"
    NOVEL = "novel"


@dataclass
class EvidenceDiff:
    """Result of diffing old vs new evidence."""

    confirmed: list[str] = field(default_factory=list)
    contradicted: list[tuple[str, str, str]] = field(default_factory=list)  # (key, old, new)
    missing: list[str] = field(default_factory=list)
    novel: list[tuple[str, str]] = field(default_factory=list)  # (key, value)

    @property
    def total(self) -> int:
        return len(self.confirmed) + len(self.contradicted) + len(self.missing) + len(self.novel)

    @property
    def change_count(self) -> int:
        return len(self.contradicted) + len(self.novel)

    # Core signal fields where changes matter more
    _CORE_FIELDS = frozenset({
        "churn_density", "avg_urgency", "churn_intent", "total_reviews",
        "dm_churn_rate", "price_complaint_rate", "displacement_mention_count",
    })
    _CORE_WEIGHT = 3.0
    _MINOR_WEIGHT = 1.0

    @property
    def diff_ratio(self) -> float:
        if self.total == 0:
            return 1.0
        return self.change_count / self.total

    @property
    def weighted_diff_ratio(self) -> float:
        """Importance-weighted diff ratio. Core signal changes count 3x."""
        if self.total == 0:
            return 1.0
        weighted_changes = 0.0
        weighted_total = 0.0
        for key, _, _ in self.contradicted:
            w = self._CORE_WEIGHT if key in self._CORE_FIELDS else self._MINOR_WEIGHT
            weighted_changes += w
            weighted_total += w
        for key, _ in self.novel:
            w = self._CORE_WEIGHT if key in self._CORE_FIELDS else self._MINOR_WEIGHT
            weighted_changes += w
            weighted_total += w
        for key in self.confirmed:
            w = self._CORE_WEIGHT if key in self._CORE_FIELDS else self._MINOR_WEIGHT
            weighted_total += w
        for key in self.missing:
            w = self._CORE_WEIGHT if key in self._CORE_FIELDS else self._MINOR_WEIGHT
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
        return self.weighted_diff_ratio < RECONSTITUTE_THRESHOLD

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
        return f"weighted_diff={ratio_pct:.0f}% ({', '.join(parts)}){core} -> {mode}"


def classify_evidence(
    old_evidence: dict[str, Any],
    new_evidence: dict[str, Any],
    *,
    numeric_tolerance: float = 0.05,
) -> EvidenceDiff:
    """Classify each evidence field as confirmed/contradicted/missing/novel.

    For numeric values, a relative change within *numeric_tolerance* (5%) is
    treated as confirmed.  For lists, element-level comparison (order-insensitive).
    For strings, exact match.
    """
    diff = EvidenceDiff()
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

        if _values_match(old_val, new_val, numeric_tolerance):
            diff.confirmed.append(key)
        else:
            diff.contradicted.append((key, str(old_val)[:200], str(new_val)[:200]))

    return diff


def _values_match(old: Any, new: Any, tolerance: float) -> bool:
    """Check if two evidence values are effectively the same."""
    # Both numeric
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        if old == 0 and new == 0:
            return True
        if old == 0:
            return abs(new) < tolerance
        return abs(new - old) / abs(old) <= tolerance

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
        sub_diff = classify_evidence(old, new, numeric_tolerance=tolerance)
        return sub_diff.diff_ratio < 0.1  # essentially unchanged

    # String or other: exact match
    return str(old) == str(new)


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
    _rcfg = ReasoningConfig()
    llm = resolve_stratified_llm(_rcfg)
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
                max_tokens=min(_rcfg.max_tokens, 1024),
                temperature=_rcfg.temperature,
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
            max_tokens=min(_rcfg.max_tokens, 1024),
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
