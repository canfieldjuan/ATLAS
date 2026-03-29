"""Deterministic contract decomposition for reasoning synthesis output.

The current synthesis task still produces a battle-card-shaped response.
This module decomposes that response into reusable reasoning contracts so
downstream products can migrate to shared vendor, displacement, and category
objects without depending on one report-specific schema forever.
"""

from __future__ import annotations

import re
from typing import Any

from ...reasoning.wedge_registry import validate_wedge
from ...services.company_normalization import normalize_company_name


def _copy_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _copy_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _titleize_wedge(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.replace("_", " ").replace("-", " ").title()


_PLACEHOLDER_NAMED_EXAMPLE_PATTERNS = (
    re.compile(r"^unknown\b", re.I),
    re.compile(r"^anonymous\b", re.I),
    re.compile(r"^redacted\b", re.I),
    re.compile(r"\b(customer|prospect|company|account|buyer|reviewer|user|team|organization|organisation|business|department)\b", re.I),
    re.compile(r"\b(smb|mid[\s_-]?market|enterprise)\b", re.I),
)

_SEGMENT_ROLE_LABELS: dict[str, str] = {
    "decision maker": "decision-makers",
    "decision makers": "decision-makers",
    "economic buyer": "economic buyers",
    "economic buyers": "economic buyers",
    "champion": "internal champions",
    "champions": "internal champions",
    "internal champion": "internal champions",
    "internal champions": "internal champions",
    "evaluator": "evaluators",
    "evaluators": "evaluators",
    "end user": "end users",
    "end users": "end users",
}

_SEGMENT_OPENING_ANGLE_LOWERCASE_WORDS = frozenset((
    "address",
    "benchmark",
    "demonstrate",
    "emphasize",
    "highlight",
    "lead",
    "offer",
    "pitch",
    "position",
    "show",
))

_SEGMENT_TEAM_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("project management", "Project Management teams"),
    ("customer support", "Customer Support teams"),
    ("data & analytics", "Data & Analytics teams"),
    ("data and analytics", "Data & Analytics teams"),
    ("marketing", "Marketing teams"),
    ("sales", "Sales teams"),
    ("operations", "Operations teams"),
    ("support", "Support teams"),
    ("engineering", "Engineering teams"),
    ("finance", "Finance teams"),
    ("security", "Security teams"),
    ("product", "Product teams"),
    ("analytics", "Analytics teams"),
    ("data", "Data teams"),
    ("hr", "HR teams"),
    ("it", "IT teams"),
)


def _looks_like_tool_label(value: str) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    generic_modifiers = ("custom", "homegrown", "home-grown", "in-house", "internal")
    generic_artifacts = (
        "integration", "utility", "workflow", "tool", "stack",
        "system", "automation", "bot",
    )
    return any(token in text for token in generic_modifiers) and any(
        token in text for token in generic_artifacts
    )


def _contract_block(synthesis: dict[str, Any], name: str) -> dict[str, Any]:
    contracts = synthesis.get("reasoning_contracts")
    if isinstance(contracts, dict):
        block = contracts.get(name)
        if isinstance(block, dict) and block:
            return dict(block)
    return {}


def _aggregate_wrapper(packet: Any, source_id: str) -> dict[str, Any] | None:
    for agg in getattr(packet, "aggregates", []) or []:
        if getattr(agg, "source_id", "") == source_id:
            return {
                "value": getattr(agg, "value", None),
                "source_id": source_id,
            }
    return None


def _wrapper_numeric_value(wrapper: Any) -> float | None:
    if not isinstance(wrapper, dict):
        return None
    try:
        value = wrapper.get("value")
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_wrapper_value(wrapper: Any) -> Any:
    if not isinstance(wrapper, dict):
        return wrapper
    value = wrapper.get("value")
    if not isinstance(value, str):
        return wrapper
    text = value.strip()
    if not text:
        return wrapper
    try:
        numeric = float(text)
    except ValueError:
        return wrapper
    normalized = dict(wrapper)
    normalized["value"] = int(numeric) if numeric.is_integer() else numeric
    return normalized


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _normalize_contract_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    for codepoint in (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2212):
        text = text.replace(chr(codepoint), "-")
    text = text.replace(chr(0x202f), " ").replace(chr(0x00a0), " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_account_intent_score(value: Any) -> float:
    score = _safe_float(value, 0.0)
    if score <= 0:
        return 0.0
    if score > 1.0:
        score = score / 10.0
    return max(0.0, min(score, 1.0))


def _normalize_migration_semantics(migration_proof: dict[str, Any]) -> dict[str, Any]:
    """Make migration-proof evidence semantics deterministic from the wrappers."""
    switch_volume = _wrapper_numeric_value(migration_proof.get("switch_volume"))
    active_eval_volume = _wrapper_numeric_value(
        migration_proof.get("active_evaluation_volume"),
    )
    confidence = str(migration_proof.get("confidence") or "").strip().lower()

    has_switches = (switch_volume or 0.0) > 0.0
    has_active_eval = (active_eval_volume or 0.0) > 0.0

    migration_proof["switching_is_real"] = has_switches
    if has_switches:
        migration_proof["evidence_type"] = "explicit_switch"
    elif has_active_eval:
        migration_proof["evidence_type"] = "active_evaluation"
        if confidence in {"", "high"}:
            migration_proof["confidence"] = "medium"
    else:
        migration_proof["evidence_type"] = "insufficient_data"
        if confidence in {"", "high", "medium"}:
            migration_proof["confidence"] = "low"
    return migration_proof


def _ensure_migration_citations(migration_proof: dict[str, Any]) -> dict[str, Any]:
    """Backfill migration-proof citations from its deterministic wrappers."""
    citations = _copy_list(migration_proof.get("citations"))
    for key in (
        "switch_volume",
        "active_evaluation_volume",
        "displacement_mention_volume",
        "top_destination",
        "primary_switch_driver",
    ):
        wrapper = migration_proof.get(key)
        if not isinstance(wrapper, dict):
            continue
        sid = wrapper.get("source_id")
        if isinstance(sid, str) and sid and sid not in citations:
            citations.append(sid)
    migration_proof["citations"] = citations
    return migration_proof


def _canonicalize_trigger_type(value: Any) -> str:
    """Normalize temporal trigger type aliases to the validator contract."""
    raw = str(value or "").strip().lower()
    if raw in {"timeline", "timeline_signal", "turning_point"}:
        return "signal"
    if raw == "contract_end":
        return "deadline"
    return raw


def _segment_scale_label(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("small business", "smb", "sme")):
        return "SMB"
    if "mid market" in lowered or "mid-market" in lowered:
        return "mid-market"
    if "enterprise" in lowered:
        return "enterprise"
    if "startup" in lowered:
        return "startup"
    return ""


_SEGMENT_NAME_PRESERVATIONS: dict[str, str] = {
    "small business": "Small Business",
    "mid market": "Mid-Market",
    "mid-market": "Mid-Market",
}


def _normalize_priority_segment_label(value: Any) -> str:
    text = _normalize_contract_text(value)
    if not text:
        return ""
    lowered = text.lower()
    lowered = re.sub(r"\(\s*role\s*:\s*([^)]+)\)", r" \1 ", lowered)
    lowered = re.sub(r"\brole:\s*", "", lowered)
    lowered = re.sub(r"\([^)]*\b(use case|role|segment|department|size|contract)\b[^)]*\)", " ", lowered)
    # Strip parenthetical numeric ranges like "(1-10 employees)" or "(51-200)"
    lowered = re.sub(r"\([^)]*\d[^)]*\)", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip(" -")

    for key, label in _SEGMENT_ROLE_LABELS.items():
        if key in lowered or key.replace(" ", "_") in lowered:
            return label
    for key, label in _SEGMENT_TEAM_KEYWORDS:
        if key in lowered:
            return label
    # Preserve human-readable segment tier names before scale normalization
    preservation = _SEGMENT_NAME_PRESERVATIONS.get(lowered)
    if preservation:
        return preservation
    scale = _segment_scale_label(lowered)
    if scale:
        if "contract" in lowered:
            return f"{scale} contracts"
        return f"{scale} accounts"
    cleaned = re.sub(r"\b(role|segment|department|size|contract)\b", " ", lowered)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
    if not cleaned:
        return ""
    if cleaned == "mid market":
        return "mid-market"
    if cleaned == "smb":
        return "SMB"
    return cleaned[:1].upper() + cleaned[1:]


def _normalize_opening_angle_phrase(value: Any) -> str:
    text = _normalize_contract_text(value)
    if not text:
        return ""
    first = text.split(" ", 1)[0].lower()
    if first in _SEGMENT_OPENING_ANGLE_LOWERCASE_WORDS:
        return text[:1].lower() + text[1:]
    return text


def _normalize_best_timing_window(value: Any) -> str:
    text = _normalize_contract_text(value)
    if not text:
        return ""
    text = re.sub(r"\(\s*timeline[_ ]signal\s*:\s*[^)]+\)", "", text, flags=re.I)
    text = re.sub(r"\btimeline[_ ]signal\s*:\s*\w+\b", "", text, flags=re.I)
    replacements = (
        (r"\bactive[- ]evaluation signals? are present across multiple flows\b", "buyers are actively evaluating alternatives across multiple flows"),
        (r"\bactive[- ]evaluation signals? are present across all tracked accounts\b", "buyers are actively evaluating alternatives across tracked accounts"),
        (r"\bactive[- ]evaluation signals? are already present\b", "buyers are already evaluating alternatives"),
        (r"\ban active[- ]evaluation signal exists\b", "buyers are already evaluating alternatives"),
        (r"\bactive[- ]evaluation signal detected this week\b", "buyers are actively evaluating alternatives this week"),
        (r"\bsegment[- ]level active[- ]evaluation signals?\b", "buyer evaluation activity in this segment"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" -")
    return text[:1].upper() + text[1:] if text else ""


def _canonicalize_timing_intelligence(timing_intelligence: dict[str, Any]) -> dict[str, Any]:
    """Normalize immediate trigger types onto the persisted contract enum."""
    best_window = _normalize_best_timing_window(
        timing_intelligence.get("best_timing_window"),
    )
    if best_window:
        timing_intelligence["best_timing_window"] = best_window
    else:
        timing_intelligence.pop("best_timing_window", None)
    triggers = timing_intelligence.get("immediate_triggers")
    if not isinstance(triggers, list):
        return timing_intelligence
    normalized: list[Any] = []
    for trigger in triggers:
        if not isinstance(trigger, dict):
            normalized.append(trigger)
            continue
        item = dict(trigger)
        canonical = _canonicalize_trigger_type(
            item.get("type") or item.get("trigger_type"),
        )
        if canonical:
            item["type"] = canonical
            if "trigger_type" in item:
                item["trigger_type"] = canonical
        normalized.append(item)
    timing_intelligence["immediate_triggers"] = normalized
    return timing_intelligence


def _canonicalize_segment_playbook(segment_playbook: dict[str, Any]) -> dict[str, Any]:
    segments = segment_playbook.get("priority_segments")
    if not isinstance(segments, list):
        return segment_playbook
    normalized: list[Any] = []
    for segment in segments:
        if not isinstance(segment, dict):
            normalized.append(segment)
            continue
        item = dict(segment)
        label = _normalize_priority_segment_label(item.get("segment"))
        if label:
            item["segment"] = label
        else:
            item.pop("segment", None)
        opening = _normalize_opening_angle_phrase(item.get("best_opening_angle"))
        if opening:
            item["best_opening_angle"] = opening
        else:
            item.pop("best_opening_angle", None)
        normalized.append(item)
    segment_playbook["priority_segments"] = normalized
    return segment_playbook


def _preferred_active_eval_wrapper(
    packet: Any,
    current: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Prefer the strongest available active-evaluation signal wrapper."""
    candidates = [
        _normalize_wrapper_value(current) if isinstance(current, dict) else None,
        _aggregate_wrapper(packet, "accounts:summary:active_eval_signal_count"),
        _aggregate_wrapper(packet, "segment:aggregate:active_eval_signal_count"),
        _aggregate_wrapper(packet, "temporal:signal:evaluation_deadline_signals"),
    ]

    best: dict[str, Any] | None = None
    best_value = float("-inf")
    for wrapper in candidates:
        numeric = _wrapper_numeric_value(wrapper)
        if numeric is None:
            continue
        if numeric > best_value:
            best = wrapper
            best_value = numeric
    return best or current


def _preferred_migration_eval_wrapper(
    packet: Any,
    current: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Prefer the strongest migration-relevant evaluation signal wrapper."""
    candidates = [
        _normalize_wrapper_value(current) if isinstance(current, dict) else None,
        _aggregate_wrapper(packet, "displacement:aggregate:total_active_evaluations"),
        _aggregate_wrapper(packet, "accounts:summary:active_eval_signal_count"),
        _aggregate_wrapper(packet, "segment:aggregate:active_eval_signal_count"),
        _aggregate_wrapper(packet, "temporal:signal:evaluation_deadline_signals"),
    ]

    best: dict[str, Any] | None = None
    best_value = float("-inf")
    for wrapper in candidates:
        numeric = _wrapper_numeric_value(wrapper)
        if numeric is None:
            continue
        if numeric > best_value:
            best = wrapper
            best_value = numeric
    return best or current


def _preferred_migration_switch_wrapper(
    packet: Any,
    current: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Use the canonical explicit-switch aggregate when it exists."""
    aggregate = _aggregate_wrapper(
        packet,
        "displacement:aggregate:total_explicit_switches",
    )
    if aggregate is not None and aggregate.get("value") is not None:
        return aggregate
    if isinstance(current, dict):
        return _normalize_wrapper_value(current)
    return current


def _preferred_displacement_mention_wrapper(
    packet: Any,
    current: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Prefer the broadest non-switch mention/intensity wrapper."""
    candidates = [
        _normalize_wrapper_value(current) if isinstance(current, dict) else None,
        _aggregate_wrapper(packet, "vault:metric:displacement_mention_count"),
        _aggregate_wrapper(packet, "category:aggregate:displacement_flow_count"),
    ]

    best: dict[str, Any] | None = None
    best_value = float("-inf")
    for wrapper in candidates:
        numeric = _wrapper_numeric_value(wrapper)
        if numeric is None:
            continue
        if numeric > best_value:
            best = wrapper
            best_value = numeric
    return best or current


def _pool_items(packet: Any, pool_name: str) -> list[Any]:
    pools = getattr(packet, "pools", {}) or {}
    return list(pools.get(pool_name, []) or [])


def _segment_items_by_kind(
    packet: Any,
    kind: str,
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _pool_items(packet, "segment"):
        source_ref = getattr(item, "source_ref", None)
        if getattr(source_ref, "kind", "") != kind:
            continue
        data = _copy_dict(getattr(item, "data", None))
        if not data:
            continue
        source_id = getattr(source_ref, "source_id", "")
        if source_id:
            data["source_id"] = source_id
        rows.append(data)
        if len(rows) >= limit:
            break
    return rows


def _timing_items_by_kind(
    packet: Any,
    kinds: str | tuple[str, ...],
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    allowed = {kinds} if isinstance(kinds, str) else set(kinds)
    for item in _pool_items(packet, "temporal"):
        source_ref = getattr(item, "source_ref", None)
        if getattr(source_ref, "kind", "") not in allowed:
            continue
        data = _copy_dict(getattr(item, "data", None))
        if not data:
            continue
        canonical = _canonicalize_trigger_type(
            data.get("type") or data.get("trigger_type"),
        )
        if canonical:
            data["type"] = canonical
            data["trigger_type"] = canonical
        source_id = getattr(source_ref, "source_id", "")
        if source_id:
            data["source_id"] = source_id
        rows.append(data)
        if len(rows) >= limit:
            break
    return rows


def _timing_signal_item_from_trigger(trigger: dict[str, Any]) -> dict[str, Any]:
    item = _copy_dict(trigger)
    canonical = _canonicalize_trigger_type(
        item.get("type") or item.get("trigger_type"),
    )
    if canonical:
        item["type"] = canonical
        item["trigger_type"] = canonical
    source = item.get("source")
    source_id = str(item.get("source_id") or "").strip()
    if not source_id and isinstance(source, dict):
        source_id = str(source.get("source_id") or "").strip()
    if source_id:
        item["source_id"] = source_id
    return item


def _timing_signal_fallback_rows(
    timing_intelligence: dict[str, Any],
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for trigger in timing_intelligence.get("immediate_triggers") or []:
        if not isinstance(trigger, dict):
            continue
        item = _timing_signal_item_from_trigger(trigger)
        label = str(item.get("trigger") or item.get("label") or item.get("date") or "").strip()
        key = (str(item.get("type") or item.get("trigger_type") or "").strip(), label)
        if key in seen:
            continue
        seen.add(key)
        rows.append(item)
        if len(rows) >= limit:
            break
    return rows


def _timing_signal_summary_rows(
    supporting_evidence: dict[str, Any],
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    timeline = supporting_evidence.get("timeline_signal_summary")
    if not isinstance(timeline, dict):
        return rows
    field_map = (
        ("contract_end_signals", "deadline", "Contract end signals detected"),
        ("evaluation_deadline_signals", "deadline", "Evaluation deadline signals detected"),
        ("renewal_signals", "deadline", "Renewal signals detected"),
        ("budget_cycle_signals", "signal", "Budget cycle signals detected"),
    )
    for field, signal_type, trigger in field_map:
        wrapper = timeline.get(field)
        count = _safe_int((wrapper or {}).get("value") if isinstance(wrapper, dict) else None, 0)
        if count <= 0:
            continue
        rows.append({
            "type": signal_type,
            "trigger_type": signal_type,
            "trigger": trigger,
            "count": count,
            "source_id": str(wrapper.get("source_id") or "").strip() if isinstance(wrapper, dict) else "",
        })
        if len(rows) >= limit:
            break
    return rows


def _merge_timing_signal_rows(
    primary: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in list(primary) + list(fallback):
        if not isinstance(row, dict):
            continue
        key = (
            str(row.get("type") or row.get("trigger_type") or "").strip(),
            str(row.get("trigger") or row.get("label") or "").strip(),
            str(row.get("source_id") or "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
        if len(merged) >= limit:
            break
    return merged


_SEGMENT_STRATEGIC_ROLE_SCORES: dict[str, float] = {
    "decision_maker": 20.0,
    "economic_buyer": 15.0,
    "champion": 15.0,
    "evaluator": 10.0,
}


def _segment_role_sort_key(item: dict[str, Any]) -> tuple[float, float, float, str]:
    role_type = str(item.get("role_type") or "").strip() or "unknown"
    return (
        _safe_float(item.get("priority_score"), 0.0),
        _safe_float(item.get("review_count"), 0.0),
        _SEGMENT_STRATEGIC_ROLE_SCORES.get(role_type, 0.0),
        role_type,
    )


def _segment_strategic_roles(packet: Any, *, limit: int = 3) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _pool_items(packet, "segment"):
        source_ref = getattr(item, "source_ref", None)
        if getattr(source_ref, "kind", "") != "role":
            continue
        data = _copy_dict(getattr(item, "data", None))
        role_type = str(data.get("role_type") or "").strip() or "unknown"
        if role_type not in _SEGMENT_STRATEGIC_ROLE_SCORES:
            continue
        source_id = getattr(source_ref, "source_id", "")
        if source_id:
            data["source_id"] = source_id
        rows.append(data)
    rows.sort(key=_segment_role_sort_key, reverse=True)
    return rows[:limit]


def _segment_supporting_evidence(packet: Any) -> dict[str, Any]:
    supporting: dict[str, Any] = {}

    list_fields = (
        ("role", "top_roles"),
        ("department", "top_departments"),
        ("size", "top_company_sizes"),
        ("contract", "top_contract_segments"),
        ("duration", "top_usage_durations"),
        ("use_case", "top_use_cases"),
    )
    for kind, output_key in list_fields:
        rows = _segment_items_by_kind(packet, kind)
        if rows:
            supporting[output_key] = rows
    strategic_roles = _segment_strategic_roles(packet)
    if strategic_roles:
        supporting["top_strategic_roles"] = strategic_roles

    company_size: dict[str, Any] = {}
    for field in ("avg_seat_count", "median_seat_count", "max_seat_count"):
        wrapper = _aggregate_wrapper(packet, f"segment:size:{field}")
        if wrapper is not None and wrapper.get("value") is not None:
            company_size[field] = wrapper
    if company_size:
        supporting["company_size_context"] = company_size

    budget_context: dict[str, Any] = {}
    budget_fields = (
        "dm_churn_rate",
        "price_increase_rate",
        "price_increase_count",
        "annual_spend_signal_count",
        "price_per_seat_signal_count",
    )
    for field in budget_fields:
        wrapper = _aggregate_wrapper(packet, f"segment:budget:{field}")
        if wrapper is not None and wrapper.get("value") is not None:
            budget_context[field] = wrapper
    if budget_context:
        supporting["budget_context"] = budget_context

    active_eval = _aggregate_wrapper(packet, "segment:aggregate:active_eval_signal_count")
    if active_eval is not None and active_eval.get("value") is not None:
        supporting["active_eval_signals"] = active_eval

    return supporting


def _timing_supporting_evidence(packet: Any) -> dict[str, Any]:
    supporting: dict[str, Any] = {}

    list_fields = (
        (
            ("deadline", "contract_end", "signal", "timeline_signal", "spike"),
            "top_timing_signals",
        ),
        ("spike", "top_keyword_spikes"),
        ("turning_point", "top_turning_points"),
        ("tenure", "top_sentiment_tenure"),
    )
    for kinds, output_key in list_fields:
        rows = _timing_items_by_kind(packet, kinds)
        if rows:
            supporting[output_key] = rows

    timeline_summary: dict[str, Any] = {}
    for field in (
        "evaluation_deadline_signals",
        "contract_end_signals",
        "renewal_signals",
        "budget_cycle_signals",
    ):
        wrapper = _aggregate_wrapper(packet, f"temporal:signal:{field}")
        if wrapper is not None and wrapper.get("value") is not None:
            timeline_summary[field] = wrapper
    if timeline_summary:
        supporting["timeline_signal_summary"] = timeline_summary

    sentiment_snapshot: dict[str, Any] = {}
    for field in (
        "declining_count",
        "stable_count",
        "improving_count",
        "total_count",
        "declining_pct",
        "improving_pct",
    ):
        wrapper = _aggregate_wrapper(packet, f"temporal:sentiment:{field}")
        if wrapper is not None and wrapper.get("value") is not None:
            sentiment_snapshot[field] = wrapper
    if sentiment_snapshot:
        supporting["sentiment_snapshot"] = sentiment_snapshot

    spike_count = _aggregate_wrapper(packet, "temporal:spike:spike_count")
    if spike_count is not None and spike_count.get("value") is not None:
        supporting["spike_count"] = spike_count

    return supporting


def _timing_sentiment_direction(
    timing_intelligence: dict[str, Any],
    packet: Any,
) -> str:
    confidence = str(timing_intelligence.get("confidence") or "").strip().lower()
    if confidence in {"low", "insufficient"}:
        return "insufficient_data"

    counts = {
        "declining": _safe_int((_aggregate_wrapper(packet, "temporal:sentiment:declining_count") or {}).get("value"), 0),
        "stable": _safe_int((_aggregate_wrapper(packet, "temporal:sentiment:stable_count") or {}).get("value"), 0),
        "improving": _safe_int((_aggregate_wrapper(packet, "temporal:sentiment:improving_count") or {}).get("value"), 0),
    }
    total = _safe_int((_aggregate_wrapper(packet, "temporal:sentiment:total_count") or {}).get("value"), 0)
    if total <= 0:
        return ""

    top_direction = max(counts, key=counts.get)
    top_count = counts[top_direction]
    if top_count <= 0:
        return ""
    if sum(1 for value in counts.values() if value == top_count) > 1:
        return "stable"
    if top_direction != "stable" and top_count * 2 < total:
        return "stable"
    if top_direction:
        return top_direction

    existing = str(timing_intelligence.get("sentiment_direction") or "").strip().lower()
    if existing in {"declining", "stable", "improving", "insufficient_data"}:
        return existing
    return ""


def _segment_sample_size_from_reach(wrapper: Any) -> int | None:
    if not isinstance(wrapper, dict):
        return None
    source_id = str(wrapper.get("source_id") or "").strip()
    if not source_id.startswith("segment:reach:"):
        return None
    sample_size = _safe_int(wrapper.get("value"), 0)
    return sample_size if sample_size > 0 else None


def _attach_segment_sample_sizes(segment_playbook: dict[str, Any]) -> dict[str, Any]:
    segments = segment_playbook.get("priority_segments")
    if not isinstance(segments, list):
        return segment_playbook
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        sample_size = _segment_sample_size_from_reach(segment.get("estimated_reach"))
        if sample_size is not None:
            segment["sample_size"] = sample_size
            continue
        existing = _safe_int(segment.get("sample_size"), 0)
        if existing > 0:
            segment["sample_size"] = existing
        else:
            segment.pop("sample_size", None)
    return segment_playbook


def _top_displacement_flows(packet: Any, limit: int = 3) -> list[dict[str, Any]]:
    flows: list[dict[str, Any]] = []
    for item in _pool_items(packet, "displacement")[:limit]:
        data = getattr(item, "data", {}) or {}
        if not isinstance(data, dict):
            continue
        edge_metrics = data.get("edge_metrics") or {}
        switch_reasons = data.get("switch_reasons") or []
        evidence_breakdown = data.get("evidence_breakdown") or []
        summary = data.get("flow_summary") or {}
        primary_driver = (
            edge_metrics.get("primary_driver")
            or _infer_flow_driver(switch_reasons, evidence_breakdown)
        )
        flows.append({
            "to_vendor": data.get("to_vendor") or data.get("competitor") or "",
            "mention_count": (
                summary.get("mention_count")
                or summary.get("total_flow_mentions")
                or edge_metrics.get("mention_count")
            ),
            "explicit_switch_count": summary.get("explicit_switch_count"),
            "active_evaluation_count": summary.get("active_evaluation_count"),
            "primary_driver": primary_driver,
            "source_id": getattr(getattr(item, "source_ref", None), "source_id", ""),
        })
    return flows


def _infer_flow_driver(
    switch_reasons: list[dict[str, Any]],
    evidence_breakdown: list[dict[str, Any]],
) -> str | None:
    counts: dict[str, int] = {}
    for item in switch_reasons or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("reason_category") or item.get("reason") or "").strip().lower()
        if not key:
            continue
        counts[key] = counts.get(key, 0) + max(int(item.get("mention_count") or 0), 1)
    for item in evidence_breakdown or []:
        if not isinstance(item, dict):
            continue
        for key, count in (item.get("reason_categories") or {}).items():
            label = str(key or "").strip().lower()
            if not label:
                continue
            counts[label] = counts.get(label, 0) + int(count or 0)
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _flow_counts(data: dict[str, Any]) -> tuple[int, int, int]:
    summary = data.get("flow_summary") or {}
    edge_metrics = data.get("edge_metrics") or {}
    switches = int(_wrapper_numeric_value({"value": summary.get("explicit_switch_count")}) or 0)
    evals = int(_wrapper_numeric_value({"value": summary.get("active_evaluation_count")}) or 0)
    mentions = int(
        _wrapper_numeric_value(
            {
                "value": (
                    summary.get("mention_count")
                    or summary.get("total_flow_mentions")
                    or edge_metrics.get("mention_count")
                ),
            }
        ) or 0
    )
    return switches, evals, mentions


def _best_displacement_flow(packet: Any) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_score: tuple[int, int, int] | None = None
    for item in _pool_items(packet, "displacement"):
        data = getattr(item, "data", {}) or {}
        if not isinstance(data, dict):
            continue
        to_vendor = str(data.get("to_vendor") or data.get("competitor") or "").strip()
        if not to_vendor:
            continue
        switches, evals, mentions = _flow_counts(data)
        if max(switches, evals, mentions) <= 0:
            continue
        score = (switches, evals, mentions)
        if best_score is None or score > best_score:
            edge_metrics = data.get("edge_metrics") or {}
            switch_reasons = data.get("switch_reasons") or []
            evidence_breakdown = data.get("evidence_breakdown") or []
            best = {
                "to_vendor": to_vendor,
                "primary_driver": (
                    edge_metrics.get("primary_driver")
                    or _infer_flow_driver(switch_reasons, evidence_breakdown)
                ),
                "source_id": getattr(getattr(item, "source_ref", None), "source_id", ""),
                "switches": switches,
                "evals": evals,
                "mentions": mentions,
            }
            best_score = score
    return best


def _account_summary(packet: Any) -> dict[str, Any]:
    top_accounts: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    def _append_account_item(item: Any) -> None:
        data = getattr(item, "data", {}) or {}
        if not isinstance(data, dict):
            return
        name = data.get("company_name") or data.get("company") or data.get("name") or "unknown"
        normalized_name = normalize_company_name(name)
        dedupe_key = (normalized_name or str(name or "").strip()).lower()
        if not dedupe_key or dedupe_key in seen_names:
            return
        intent = _normalize_account_intent_score(
            data.get("urgency_score") or data.get("intent_score") or data.get("confidence_score", 0),
        )
        top_accounts.append({
            "name": name,
            "intent_score": intent,
            "source_id": getattr(getattr(item, "source_ref", None), "source_id", ""),
        })
        seen_names.add(dedupe_key)

    for item in _pool_items(packet, "accounts"):
        _append_account_item(item)

    if len(top_accounts) < 5:
        for item in _pool_items(packet, "evidence_vault"):
            source_ref = getattr(item, "source_ref", None)
            if getattr(source_ref, "kind", "") != "company":
                continue
            _append_account_item(item)
            if len(top_accounts) >= 5:
                break

    return {
        "schema_version": "v1",
        "total_accounts": _aggregate_wrapper(packet, "accounts:summary:total_accounts"),
        "high_intent_count": _aggregate_wrapper(packet, "accounts:summary:high_intent_count"),
        "active_eval_count": _aggregate_wrapper(packet, "accounts:summary:active_eval_signal_count"),
        "top_accounts": top_accounts[:5],
    }


def _category_summary(packet: Any) -> dict[str, Any]:
    regime_item = None
    council_item = None
    for item in _pool_items(packet, "category"):
        source_ref = getattr(item, "source_ref", None)
        kind = getattr(source_ref, "kind", "")
        if kind == "regime" and regime_item is None:
            regime_item = item
        elif kind == "council" and council_item is None:
            council_item = item
    regime_data = getattr(regime_item, "data", {}) or {}
    regime_source_id = getattr(getattr(regime_item, "source_ref", None), "source_id", "")
    council_data = getattr(council_item, "data", {}) or {}
    council_source_id = getattr(getattr(council_item, "source_ref", None), "source_id", "")

    citations: list[str] = []
    if regime_source_id:
        citations.append(regime_source_id)
    if council_source_id and council_source_id not in citations:
        citations.append(council_source_id)

    result: dict[str, Any] = {
        "schema_version": "v1",
        "market_regime": (
            council_data.get("market_regime")
            or regime_data.get("regime_type")
            or ""
        ),
        "narrative": (
            council_data.get("conclusion")
            or regime_data.get("narrative")
            or ""
        ),
        "confidence_score": (
            council_data.get("confidence")
            if council_data.get("confidence") is not None
            else regime_data.get("confidence")
        ),
        "vendor_count": _aggregate_wrapper(packet, "category:aggregate:vendor_count"),
        "displacement_flow_count": _aggregate_wrapper(
            packet,
            "category:aggregate:displacement_flow_count",
        ),
        "winner": council_data.get("winner") or None,
        "loser": council_data.get("loser") or None,
        "top_differentiator": council_data.get("top_differentiator") or None,
        "top_vulnerability": council_data.get("top_vulnerability") or None,
        "key_insights": council_data.get("key_insights") or [],
        "durability": council_data.get("durability_assessment") or None,
        "avg_churn_velocity": regime_data.get("avg_churn_velocity"),
        "avg_price_pressure": regime_data.get("avg_price_pressure"),
        "outlier_vendors": regime_data.get("outlier_vendors") or [],
        "segment_dynamics": council_data.get("segment_dynamics") or None,
        "category_default": council_data.get("category_default") or None,
        "citations": citations,
    }
    return result


def _is_placeholder_named_example_company(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    normalized = normalize_company_name(text)
    if not normalized:
        return True
    if _looks_like_tool_label(text):
        return True
    return any(pattern.search(text) for pattern in _PLACEHOLDER_NAMED_EXAMPLE_PATTERNS)


def _sanitize_named_examples(
    migration_proof: dict[str, Any],
) -> dict[str, Any]:
    examples = _copy_list(migration_proof.get("named_examples"))
    if not examples:
        return migration_proof

    cleaned: list[dict[str, Any]] = []
    dropped = False
    for item in examples:
        if not isinstance(item, dict):
            dropped = True
            continue
        if _is_placeholder_named_example_company(item.get("company")):
            dropped = True
            continue
        cleaned.append(item)

    migration_proof["named_examples"] = cleaned
    if dropped:
        data_gaps = _copy_list(migration_proof.get("data_gaps"))
        message = "No credible named migration examples"
        if message not in data_gaps:
            data_gaps.append(message)
        migration_proof["data_gaps"] = data_gaps
    return migration_proof


def _sanitize_text_wrapper(
    wrapper: Any,
) -> dict[str, Any]:
    if not isinstance(wrapper, dict):
        return {"value": None, "source_id": None}
    value = wrapper.get("value")
    source_id = wrapper.get("source_id")
    if not isinstance(value, str):
        return {"value": None, "source_id": None}
    text = value.strip()
    if not text or text.lower() in {"0", "none", "null", "unknown", "n/a", "na", "n.a."}:
        return {"value": None, "source_id": None}
    return {
        "value": text,
        "source_id": source_id if isinstance(source_id, str) and source_id.strip() else None,
    }


def _filter_valid_citations(value: Any, valid_source_ids: set[str]) -> Any:
    """Drop citations that do not exist in the current packet."""
    if isinstance(value, dict):
        citations = value.get("citations")
        if isinstance(citations, list):
            filtered: list[str] = []
            for sid in citations:
                if isinstance(sid, str) and sid in valid_source_ids and sid not in filtered:
                    filtered.append(sid)
            if filtered:
                value["citations"] = filtered
            else:
                value.pop("citations", None)
        for child in value.values():
            _filter_valid_citations(child, valid_source_ids)
    elif isinstance(value, list):
        for item in value:
            _filter_valid_citations(item, valid_source_ids)
    return value


# ---------------------------------------------------------------------------
# Phase 3 contract builders
# ---------------------------------------------------------------------------

def _build_why_they_stay(packet: Any) -> dict[str, Any] | None:
    """Build a why_they_stay block from packet retention_proof.

    Explains why customers remain despite churn pressure -- the counter-
    signal that prevents over-aggressive positioning.
    """
    retention_proof = getattr(packet, "retention_proof", None) or []
    if not retention_proof:
        return None
    strengths: list[dict[str, Any]] = []
    for proof in retention_proof:
        if not isinstance(proof, dict):
            continue
        area = proof.get("area", "")
        strength_text = proof.get("strength", "")
        if not area:
            continue
        entry: dict[str, Any] = {"area": area}
        if strength_text:
            entry["evidence"] = strength_text
        sid = proof.get("_sid")
        if sid:
            entry["_sid"] = sid
        mc = proof.get("mention_count")
        if mc is not None:
            entry["mention_count"] = mc
        strengths.append(entry)
    if not strengths:
        return None
    # Build summary from top strengths
    top_areas = [s["area"].replace("_", " ") for s in strengths[:3]]
    summary = (
        f"Retention anchored by {', '.join(top_areas)}"
        if top_areas else ""
    )
    return {
        "summary": summary,
        "strengths": strengths,
    }


def _build_confidence_posture(
    packet: Any,
    causal_narrative: dict[str, Any],
) -> dict[str, Any] | None:
    """Build a confidence_posture block from coverage gaps and section confidence.

    Surfaces the explicit limits on what the reasoning can claim, so
    downstream consumers can hedge appropriately.
    """
    coverage_gaps = getattr(packet, "coverage_gaps", None) or []
    section_confidence = causal_narrative.get("confidence", "medium")

    limits: list[str] = []
    for gap in coverage_gaps:
        if not isinstance(gap, dict):
            continue
        gap_type = gap.get("type", "")
        area = gap.get("area", "")
        sample = gap.get("sample_size", 0)
        if gap_type == "thin_segment_sample":
            limits.append(f"thin {area.replace('_', ' ')} sample (n={sample})")
        elif gap_type == "missing_pool":
            limits.append(f"no {area} evidence")
        elif gap_type == "thin_account_signals":
            limits.append(f"weak account signal density (n={sample})")
        elif gap_type == "shallow_evidence_window":
            limits.append(f"shallow evidence window ({sample} days)")

    if not limits and section_confidence in ("high", "medium"):
        return None

    return {
        "overall": section_confidence,
        "limits": limits,
    }


def _build_switch_triggers(
    timing_intelligence: dict[str, Any],
    packet: Any,
) -> list[dict[str, Any]]:
    """Extract switch triggers from timing intelligence and displacement data.

    Switch triggers are the specific events or conditions that push a
    customer from 'frustrated but staying' to 'actively switching'.
    """
    triggers: list[dict[str, Any]] = []
    seen_labels: set[str] = set()

    # Source 1: immediate_triggers from timing intelligence
    immediate = timing_intelligence.get("immediate_triggers") or []
    for t in immediate:
        if not isinstance(t, dict):
            continue
        trigger_type = t.get("type") or t.get("trigger_type") or ""
        description = t.get("description") or t.get("signal") or ""
        label = trigger_type or description[:40]
        if not label or label in seen_labels:
            continue
        seen_labels.add(label)
        entry: dict[str, Any] = {"type": trigger_type}
        if description:
            entry["description"] = description
        sid = t.get("_sid") or t.get("source_id")
        if sid:
            entry["_sid"] = sid
        triggers.append(entry)

    # Source 2: high-urgency temporal spikes from packet
    temporal_items = getattr(packet, "pools", {}).get("temporal", [])
    for si in temporal_items:
        data = getattr(si, "data", {})
        if not isinstance(data, dict):
            continue
        spike_type = data.get("spike_type") or data.get("type") or ""
        magnitude = data.get("magnitude") or data.get("spike_magnitude") or 0
        try:
            mag_val = float(magnitude)
        except (TypeError, ValueError):
            mag_val = 0.0
        if mag_val < 2.0:
            continue
        label = spike_type or "temporal_spike"
        if label in seen_labels:
            continue
        seen_labels.add(label)
        entry = {"type": "spike", "description": spike_type}
        ref = getattr(si, "source_ref", None)
        if ref:
            entry["_sid"] = ref.source_id
        triggers.append(entry)

    return triggers


def _build_evidence_governance(packet: Any) -> dict[str, Any] | None:
    """Build evidence_governance block from packet governance signals.

    Passthrough of metric_ledger, contradictions, and coverage_gaps so
    downstream consumers can read uncertainty/governance directly.
    """
    metric_ledger = getattr(packet, "metric_ledger", None) or []
    contradictions = getattr(packet, "contradiction_rows", None) or []
    coverage_gaps = getattr(packet, "coverage_gaps", None) or []
    minority_signals = getattr(packet, "minority_signals", None) or []

    if not any([metric_ledger, contradictions, coverage_gaps, minority_signals]):
        return None

    gov: dict[str, Any] = {}
    if metric_ledger:
        gov["metric_ledger"] = metric_ledger
    if contradictions:
        gov["contradictions"] = contradictions
    if coverage_gaps:
        gov["coverage_gaps"] = coverage_gaps
    if minority_signals:
        gov["minority_signals"] = minority_signals
    return gov


def build_reasoning_contracts(
    synthesis: dict[str, Any],
    packet: Any,
) -> dict[str, Any]:
    """Build reusable reasoning contracts from battle-card synthesis output."""
    explicit_contracts = synthesis.get("reasoning_contracts")
    has_explicit_contracts = (
        isinstance(explicit_contracts, dict) and bool(explicit_contracts)
    )
    valid_source_ids = set(packet.source_ids())
    explicit_vendor_core = _contract_block(synthesis, "vendor_core_reasoning")
    explicit_displacement = _contract_block(synthesis, "displacement_reasoning")
    explicit_category = _contract_block(synthesis, "category_reasoning")
    explicit_account = _contract_block(synthesis, "account_reasoning")

    causal_narrative = _copy_dict(
        explicit_vendor_core.get("causal_narrative")
        or ({} if has_explicit_contracts else synthesis.get("causal_narrative")),
    )
    segment_playbook = _copy_dict(
        explicit_vendor_core.get("segment_playbook")
        or ({} if has_explicit_contracts else synthesis.get("segment_playbook")),
    )
    if segment_playbook:
        segment_playbook = _attach_segment_sample_sizes(segment_playbook)
        segment_playbook = _canonicalize_segment_playbook(segment_playbook)
        supporting_evidence = _segment_supporting_evidence(packet)
        existing_support = _copy_dict(segment_playbook.get("supporting_evidence"))
        for key, value in existing_support.items():
            if key not in supporting_evidence:
                supporting_evidence[key] = value
        if supporting_evidence:
            segment_playbook["supporting_evidence"] = supporting_evidence
        else:
            segment_playbook.pop("supporting_evidence", None)
    timing_intelligence = _copy_dict(
        explicit_vendor_core.get("timing_intelligence")
        or ({} if has_explicit_contracts else synthesis.get("timing_intelligence")),
    )
    if timing_intelligence or not has_explicit_contracts:
        preferred_active_eval = _preferred_active_eval_wrapper(
            packet,
            timing_intelligence.get("active_eval_signals"),
        )
        if preferred_active_eval is not None:
            timing_intelligence["active_eval_signals"] = preferred_active_eval
        supporting_evidence = _timing_supporting_evidence(packet)
        timing_rows = list(supporting_evidence.get("top_timing_signals") or [])
        timing_rows = _merge_timing_signal_rows(
            timing_rows,
            _timing_signal_fallback_rows(timing_intelligence),
        )
        timing_rows = _merge_timing_signal_rows(
            timing_rows,
            _timing_signal_summary_rows(supporting_evidence),
        )
        if timing_rows:
            supporting_evidence["top_timing_signals"] = timing_rows
        existing_support = _copy_dict(timing_intelligence.get("supporting_evidence"))
        for key, value in existing_support.items():
            if key not in supporting_evidence:
                supporting_evidence[key] = value
        if supporting_evidence:
            timing_intelligence["supporting_evidence"] = supporting_evidence
        else:
            timing_intelligence.pop("supporting_evidence", None)
        sentiment_direction = _timing_sentiment_direction(timing_intelligence, packet)
        if sentiment_direction:
            timing_intelligence["sentiment_direction"] = sentiment_direction
        else:
            timing_intelligence.pop("sentiment_direction", None)
        timing_intelligence = _canonicalize_timing_intelligence(timing_intelligence)
    migration_proof = _copy_dict(
        explicit_displacement.get("migration_proof")
        or ({} if has_explicit_contracts else synthesis.get("migration_proof")),
    )
    migration_proof = _sanitize_named_examples(migration_proof)
    best_flow = _best_displacement_flow(packet)
    if migration_proof:
        fallback_destination = (
            {
                "value": best_flow.get("to_vendor"),
                "source_id": best_flow.get("source_id"),
            }
            if best_flow and best_flow.get("to_vendor")
            else None
        )
        fallback_driver = (
            {
                "value": best_flow.get("primary_driver"),
                "source_id": best_flow.get("source_id"),
            }
            if best_flow and best_flow.get("primary_driver")
            else None
        )
        migration_proof["top_destination"] = (
            fallback_destination
            or _sanitize_text_wrapper(migration_proof.get("top_destination"))
        )
        migration_proof["primary_switch_driver"] = (
            fallback_driver
            or _sanitize_text_wrapper(migration_proof.get("primary_switch_driver"))
        )
    if migration_proof or not has_explicit_contracts:
        preferred_switch_volume = _preferred_migration_switch_wrapper(
            packet,
            migration_proof.get("switch_volume"),
        )
        if preferred_switch_volume is not None:
            migration_proof["switch_volume"] = preferred_switch_volume
        preferred_migration_eval = _preferred_migration_eval_wrapper(
            packet,
            migration_proof.get("active_evaluation_volume"),
        )
        if preferred_migration_eval is not None:
            migration_proof["active_evaluation_volume"] = preferred_migration_eval
        preferred_mention_volume = _preferred_displacement_mention_wrapper(
            packet,
            migration_proof.get("displacement_mention_volume"),
        )
        if preferred_mention_volume is not None:
            migration_proof["displacement_mention_volume"] = preferred_mention_volume
        migration_proof = _normalize_migration_semantics(migration_proof)
        migration_proof = _ensure_migration_citations(migration_proof)
    competitive_reframes = _copy_dict(
        explicit_displacement.get("competitive_reframes")
        or ({} if has_explicit_contracts else synthesis.get("competitive_reframes")),
    )
    for section in (
        causal_narrative,
        segment_playbook,
        timing_intelligence,
        migration_proof,
        competitive_reframes,
    ):
        _filter_valid_citations(section, valid_source_ids)

    vendor_core_citations: list[str] = []
    for section in (causal_narrative, segment_playbook, timing_intelligence):
        vendor_core_citations.extend(_copy_list(section.get("citations")))
    deduped_vendor_core_citations = list(dict.fromkeys(vendor_core_citations))

    displacement_citations = _copy_list(migration_proof.get("citations"))
    top_flows = _top_displacement_flows(packet)
    for flow in top_flows:
        sid = flow.get("source_id")
        if sid and sid not in displacement_citations:
            displacement_citations.append(sid)

    vendor_core: dict[str, Any] = {}
    if explicit_vendor_core or not has_explicit_contracts:
        vendor_core = {
            **explicit_vendor_core,
            "schema_version": str(explicit_vendor_core.get("schema_version") or "v1"),
        }
        if causal_narrative:
            vendor_core["causal_narrative"] = causal_narrative
        if segment_playbook:
            vendor_core["segment_playbook"] = segment_playbook
        if timing_intelligence:
            vendor_core["timing_intelligence"] = timing_intelligence
        citations = deduped_vendor_core_citations or _copy_list(explicit_vendor_core.get("citations"))
        if citations:
            vendor_core["citations"] = citations

    displacement: dict[str, Any] = {}
    if explicit_displacement or not has_explicit_contracts:
        displacement = {
            **explicit_displacement,
            "schema_version": str(explicit_displacement.get("schema_version") or "v1"),
            "top_flows": _copy_list(explicit_displacement.get("top_flows")) or top_flows,
            "confirmed_switch_count": explicit_displacement.get("confirmed_switch_count") or _aggregate_wrapper(
                packet,
                "displacement:aggregate:total_explicit_switches",
            ),
            "active_evaluation_count": explicit_displacement.get("active_evaluation_count") or _aggregate_wrapper(
                packet,
                "displacement:aggregate:total_active_evaluations",
            ),
            "displacement_mention_volume": explicit_displacement.get("displacement_mention_volume") or _aggregate_wrapper(
                packet,
                "vault:metric:displacement_mention_count",
            ),
        }
        if migration_proof:
            displacement["migration_proof"] = migration_proof
        if competitive_reframes:
            displacement["competitive_reframes"] = competitive_reframes
        citations = displacement_citations or _copy_list(explicit_displacement.get("citations"))
        if citations:
            displacement["citations"] = citations

    account_summary = _account_summary(packet)
    account = {
        **explicit_account,
        **account_summary,
        "schema_version": str(explicit_account.get("schema_version") or "v1"),
        "market_summary": explicit_account.get("market_summary") or "",
        "confidence_score": (
            explicit_account.get("confidence_score")
            if explicit_account.get("confidence_score") is not None
            else account_summary.get("confidence_score")
        ),
    }
    citations = list(dict.fromkeys(
        _copy_list(explicit_account.get("citations"))
        + _copy_list(account_summary.get("citations"))
    ))
    if citations:
        account["citations"] = citations
    _filter_valid_citations(account, valid_source_ids)

    category_summary = _category_summary(packet)
    category = {
        **category_summary,
        **explicit_category,
        "schema_version": str(explicit_category.get("schema_version") or "v1"),
        "market_regime": explicit_category.get("market_regime") or category_summary.get("market_regime") or "",
        "narrative": explicit_category.get("narrative") or category_summary.get("narrative") or "",
        "confidence_score": (
            explicit_category.get("confidence_score")
            if explicit_category.get("confidence_score") is not None
            else category_summary.get("confidence_score")
        ),
        "vendor_count": explicit_category.get("vendor_count") or category_summary.get("vendor_count"),
        "displacement_flow_count": explicit_category.get("displacement_flow_count") or category_summary.get("displacement_flow_count"),
    }
    citations = list(dict.fromkeys(
        _copy_list(explicit_category.get("citations"))
        + _copy_list(category_summary.get("citations"))
    ))
    if citations:
        category["citations"] = citations
    _filter_valid_citations(category, valid_source_ids)

    # --- Phase 3: why_they_stay from retention_proof -----------------------
    # Merge: deterministic builder fills gaps, model-provided content preserved
    deterministic_wts = _build_why_they_stay(packet)
    if vendor_core:
        existing_wts = vendor_core.get("why_they_stay")
        if isinstance(existing_wts, dict) and existing_wts:
            # Model provided content -- merge deterministic strengths in
            if deterministic_wts:
                existing_strengths = existing_wts.get("strengths") or []
                existing_areas = {s.get("area") for s in existing_strengths if isinstance(s, dict)}
                for s in deterministic_wts.get("strengths") or []:
                    if isinstance(s, dict) and s.get("area") not in existing_areas:
                        existing_strengths.append(s)
                existing_wts["strengths"] = existing_strengths
            vendor_core["why_they_stay"] = existing_wts
        elif deterministic_wts:
            vendor_core["why_they_stay"] = deterministic_wts

    # --- Phase 3: confidence_posture from coverage_gaps -------------------
    deterministic_cp = _build_confidence_posture(packet, causal_narrative)
    if vendor_core:
        existing_cp = vendor_core.get("confidence_posture")
        if isinstance(existing_cp, dict) and existing_cp:
            # Model provided content -- merge deterministic limits in
            if deterministic_cp:
                existing_limits = list(existing_cp.get("limits") or [])
                existing_set = set(existing_limits)
                for lim in deterministic_cp.get("limits") or []:
                    if lim not in existing_set:
                        existing_limits.append(lim)
                existing_cp["limits"] = existing_limits
            vendor_core["confidence_posture"] = existing_cp
        elif deterministic_cp:
            vendor_core["confidence_posture"] = deterministic_cp

    # --- Phase 3: switch_triggers from timing/displacement ----------------
    deterministic_st = _build_switch_triggers(timing_intelligence, packet)
    if displacement:
        existing_st = displacement.get("switch_triggers")
        if isinstance(existing_st, list) and existing_st:
            # Model provided triggers -- merge deterministic ones in
            existing_types = {t.get("type") for t in existing_st if isinstance(t, dict)}
            for t in deterministic_st:
                if isinstance(t, dict) and t.get("type") not in existing_types:
                    existing_st.append(t)
            displacement["switch_triggers"] = existing_st
        elif deterministic_st:
            displacement["switch_triggers"] = deterministic_st

    # --- Phase 3: evidence_governance passthrough from packet -------------
    evidence_governance = _build_evidence_governance(packet)

    contracts = {"schema_version": "v1"}
    if vendor_core:
        contracts["vendor_core_reasoning"] = vendor_core
    if displacement:
        contracts["displacement_reasoning"] = displacement
    if category:
        contracts["category_reasoning"] = category
    if account:
        contracts["account_reasoning"] = account
    if evidence_governance:
        contracts["evidence_governance"] = evidence_governance
    return contracts


def build_persistable_synthesis(
    synthesis: dict[str, Any],
    packet: Any,
) -> dict[str, Any]:
    """Build the canonical persisted synthesis payload.

    Persist contracts as the source of truth. Flat battle-card-shaped sections are
    treated as transient prompt/validation intermediates and are not stored at the
    top level anymore.
    """
    contracts = build_reasoning_contracts(synthesis, packet)
    raw_schema = str(synthesis.get("schema_version") or "2.1")
    meta = _copy_dict(synthesis.get("meta"))
    warnings = _copy_list(synthesis.get("_validation_warnings"))

    persisted: dict[str, Any] = {
        "schema_version": raw_schema,
        "reasoning_shape": "contracts_first_v1",
        "reasoning_contracts": contracts,
    }
    if meta:
        persisted["meta"] = meta
    if warnings:
        persisted["_validation_warnings"] = warnings

    vendor_core = contracts.get("vendor_core_reasoning") or {}
    causal = vendor_core.get("causal_narrative") or {}
    wedge = str(causal.get("primary_wedge") or "").strip()
    valid_wedge = validate_wedge(wedge)
    if valid_wedge is not None:
        persisted["synthesis_wedge"] = valid_wedge.value
        persisted["synthesis_wedge_label"] = _titleize_wedge(valid_wedge.value)

    return persisted
