from __future__ import annotations

from typing import Any


def merge_tier1_tier2(tier1: dict[str, Any], tier2: dict[str, Any] | None) -> dict[str, Any]:
    result = dict(tier1)

    if tier2 is None:
        result.setdefault("pain_categories", [])
        result.setdefault("sentiment_trajectory", {})
        result.setdefault("buyer_authority", {"role_type": "unknown", "buying_stage": "unknown",
                                              "executive_sponsor_mentioned": False})
        result.setdefault("timeline", {"decision_timeline": "unknown"})
        result.setdefault("contract_context", {"contract_value_signal": "unknown"})
        result.setdefault("insider_signals", None)
        result.setdefault("positive_aspects", [])
        result.setdefault("feature_gaps", [])
        result.setdefault("recommendation_language", [])
        result.setdefault("pricing_phrases", [])
        result.setdefault("event_mentions", [])
        result.setdefault("urgency_indicators", {})
        for comp in result.get("competitors_mentioned", []):
            comp.setdefault("evidence_type", "neutral_mention")
            comp.setdefault("displacement_confidence", "low")
            comp.setdefault("reason_category", None)
        return result

    tier2_top_level_keys = {
        "pain_categories",
        "sentiment_trajectory", "buyer_authority", "timeline",
        "contract_context", "insider_signals",
        "positive_aspects", "feature_gaps",
        "recommendation_language", "pricing_phrases",
        "event_mentions", "urgency_indicators",
    }
    legacy_tier2_keys = {"urgency_score", "pain_category", "would_recommend"}
    for key in tier2_top_level_keys | legacy_tier2_keys:
        if key in tier2:
            result[key] = tier2[key]

    tier1_comps = {
        c["name"].lower(): c
        for c in result.get("competitors_mentioned", [])
        if isinstance(c, dict) and "name" in c
    }
    tier2_comps = tier2.get("competitors_mentioned", []) or []

    merged_comps = []
    seen = set()
    for t2_comp in tier2_comps:
        if not isinstance(t2_comp, dict) or "name" not in t2_comp:
            continue
        key = t2_comp["name"].lower()
        seen.add(key)
        base = dict(tier1_comps.get(key, {"name": t2_comp["name"]}))
        for field in ("evidence_type", "displacement_confidence", "reason_category"):
            if field in t2_comp:
                base[field] = t2_comp[field]
        if key in tier1_comps:
            base["name"] = tier1_comps[key]["name"]
        merged_comps.append(base)

    for key, t1_comp in tier1_comps.items():
        if key not in seen:
            t1_comp.setdefault("evidence_type", "neutral_mention")
            t1_comp.setdefault("displacement_confidence", "low")
            t1_comp.setdefault("reason_category", None)
            merged_comps.append(t1_comp)

    result["competitors_mentioned"] = merged_comps
    return result


def missing_witness_primitives(
    result: dict[str, Any],
    *,
    known_replacement_modes: set[str],
    known_operating_model_shifts: set[str],
    known_productivity_delta_claims: set[str],
    known_org_pressure_types: set[str],
) -> list[str]:
    missing: list[str] = []

    if str(result.get("replacement_mode") or "").strip() not in known_replacement_modes:
        missing.append("replacement_mode")
    if str(result.get("operating_model_shift") or "").strip() not in known_operating_model_shifts:
        missing.append("operating_model_shift")
    if str(result.get("productivity_delta_claim") or "").strip() not in known_productivity_delta_claims:
        missing.append("productivity_delta_claim")
    if str(result.get("org_pressure_type") or "").strip() not in known_org_pressure_types:
        missing.append("org_pressure_type")

    if not isinstance(result.get("salience_flags"), list):
        missing.append("salience_flags")
    if not isinstance(result.get("evidence_spans"), list):
        missing.append("evidence_spans")
    if not str(result.get("evidence_map_hash") or "").strip():
        missing.append("evidence_map_hash")

    return missing


def schema_version(result: dict[str, Any]) -> int:
    try:
        return int(result.get("enrichment_schema_version") or 0)
    except (TypeError, ValueError):
        return 0
