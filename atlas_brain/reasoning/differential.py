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

    @property
    def diff_ratio(self) -> float:
        if self.total == 0:
            return 1.0  # no evidence at all -> full reason
        return self.change_count / self.total

    @property
    def should_reconstitute(self) -> bool:
        return self.diff_ratio < RECONSTITUTE_THRESHOLD and self.total > 0

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
        ratio_pct = self.diff_ratio * 100
        mode = "reconstitute" if self.should_reconstitute else "full_reason"
        return f"diff_ratio={ratio_pct:.0f}% ({', '.join(parts)}) -> {mode}"


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
) -> tuple[dict[str, Any], int]:
    """Patch an existing conclusion with only the evidence delta.

    Returns (updated_conclusion, tokens_used).
    """
    from ..pipelines.llm import get_pipeline_llm, parse_json_response
    from ..services.protocols import Message

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
    _workload = _rcfg.stratified_llm_workload
    llm = get_pipeline_llm(workload=_workload, auto_activate_ollama=(_workload == "vllm"))
    if llm is None:
        logger.error("No LLM available for reconstitution (workload=%s)", _workload)
        return old_conclusion, 0

    messages = [
        Message(role="system", content=_RECONSTITUTE_SYSTEM_PROMPT),
        Message(
            role="user",
            content=json.dumps(delta, separators=(",", ":"), default=str),
        ),
    ]

    t0 = time.monotonic()
    try:
        result = llm.chat(
            messages=messages,
            max_tokens=min(_rcfg.max_tokens, 1024),
            temperature=_rcfg.temperature,
            response_format={"type": "json_object"},
        )
        text = result.get("response", "").strip()
        usage = result.get("usage", {})
        tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        updated = parse_json_response(text)

        duration_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "Reconstituted %s (diff_ratio=%.0f%%, %d tokens, %.0fms)",
            vendor_name, diff.diff_ratio * 100, tokens, duration_ms,
        )
        return updated, tokens

    except Exception:
        logger.exception("Reconstitution LLM call failed for %s", vendor_name)
        return old_conclusion, 0
