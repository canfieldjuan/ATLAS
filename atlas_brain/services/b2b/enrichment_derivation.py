from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnrichmentDerivationDeps:
    get_evidence_engine: Any
    coerce_legacy_phrase_arrays: Any
    apply_phrase_metadata_contract: Any
    derive_pain_categories: Any
    recover_competitor_mentions: Any
    derive_competitor_annotations: Any
    derive_budget_signals: Any
    derive_buyer_authority_fields: Any
    derive_concrete_timeline_fields: Any
    derive_decision_timeline: Any
    derive_contract_value_signal: Any
    derive_urgency_indicators: Any
    normalize_pain_category: Any
    subject_vendor_phrase_texts: Any
    compute_pain_confidence: Any
    demote_primary_pain: Any
    derive_replacement_mode: Any
    derive_operating_model_shift: Any
    derive_productivity_delta_claim: Any
    derive_org_pressure_type: Any
    derive_salience_flags: Any
    derive_evidence_spans: Any


def compute_derived_fields(
    result: dict[str, Any],
    source_row: dict[str, Any],
    *,
    deps: EnrichmentDerivationDeps,
) -> dict[str, Any]:
    engine = deps.get_evidence_engine()

    raw_meta = source_row.get("raw_metadata") or {}
    if isinstance(raw_meta, str):
        raw_meta = json.loads(raw_meta)
    source_weight = float(raw_meta.get("source_weight", 0.7))
    content_type = source_row.get("content_type") or result.get("content_classification") or "review"
    rating = float(source_row["rating"]) if source_row.get("rating") is not None else None
    rating_max = float(source_row.get("rating_max") or 5)

    deps.coerce_legacy_phrase_arrays(result)
    deps.apply_phrase_metadata_contract(result, source_row)

    pricing_phrases = result.get("pricing_phrases", [])
    rec_lang = result.get("recommendation_language", [])
    events = result.get("event_mentions", [])

    result["pain_categories"] = deps.derive_pain_categories(result)
    result["competitors_mentioned"] = deps.recover_competitor_mentions(result, source_row)
    result["competitors_mentioned"] = deps.derive_competitor_annotations(result, source_row)
    deps.derive_budget_signals(result, source_row)

    ba = result.get("buyer_authority")
    if not isinstance(ba, dict):
        ba = {}
        result["buyer_authority"] = ba
    role_type, executive_sponsor_mentioned, buying_stage = deps.derive_buyer_authority_fields(result, source_row)
    ba["role_type"] = role_type
    ba["executive_sponsor_mentioned"] = executive_sponsor_mentioned
    ba["buying_stage"] = buying_stage

    timeline = result.get("timeline")
    if not isinstance(timeline, dict):
        timeline = {}
        result["timeline"] = timeline
    contract_end, evaluation_deadline = deps.derive_concrete_timeline_fields(result, source_row)
    if contract_end and not str(timeline.get("contract_end") or "").strip():
        timeline["contract_end"] = contract_end
    if evaluation_deadline and not str(timeline.get("evaluation_deadline") or "").strip():
        timeline["evaluation_deadline"] = evaluation_deadline
    timeline["decision_timeline"] = deps.derive_decision_timeline(result, source_row)

    cc = result.get("contract_context")
    if not isinstance(cc, dict):
        cc = {}
        result["contract_context"] = cc
    cc["contract_value_signal"] = deps.derive_contract_value_signal(result)
    price_complaint = engine.derive_price_complaint(result)
    result["urgency_indicators"] = deps.derive_urgency_indicators(
        result,
        source_row,
        price_complaint=price_complaint,
    )

    indicators = result.get("urgency_indicators", {})
    pain_cats = result.get("pain_categories", [])
    result["urgency_score"] = engine.compute_urgency(
        indicators, rating, rating_max, content_type, source_weight,
    )

    primary_pain = "overall_dissatisfaction"
    if pain_cats:
        primary_list = [p for p in pain_cats if isinstance(p, dict) and p.get("severity") == "primary"]
        if primary_list:
            primary_pain = primary_list[0].get("category", "overall_dissatisfaction")
        elif isinstance(pain_cats[0], dict):
            primary_pain = pain_cats[0].get("category", "overall_dissatisfaction")
    result["pain_category"] = engine.override_pain(
        deps.normalize_pain_category(primary_pain),
        deps.subject_vendor_phrase_texts(result, "specific_complaints"),
        deps.subject_vendor_phrase_texts(result, "quotable_phrases"),
        deps.subject_vendor_phrase_texts(result, "pricing_phrases"),
        deps.subject_vendor_phrase_texts(result, "feature_gaps"),
        deps.subject_vendor_phrase_texts(result, "recommendation_language"),
    )

    result["would_recommend"] = engine.derive_recommend(rec_lang, rating, rating_max)

    st = result.get("sentiment_trajectory")
    if not isinstance(st, dict):
        st = {}
        result["sentiment_trajectory"] = st
    rating_norm = (rating / rating_max) if rating is not None and rating_max else None
    churn_signals_raw = result.get("churn_signals") or {}
    intent_to_leave = bool(churn_signals_raw.get("intent_to_leave")) if isinstance(churn_signals_raw, dict) else False
    would_rec = result.get("would_recommend")
    if rating_norm is not None:
        if rating_norm <= 0.4 or (rating_norm <= 0.6 and intent_to_leave):
            st["direction"] = "consistently_negative"
        elif rating_norm >= 0.8 and would_rec is True:
            st["direction"] = "stable_positive"
        elif rating_norm >= 0.7 and would_rec is not False:
            st["direction"] = "stable_positive"
        else:
            st["direction"] = "unknown"
    else:
        st["direction"] = "unknown"

    final_pain = deps.normalize_pain_category(result.get("pain_category"))
    confidence = deps.compute_pain_confidence(result, final_pain)
    if confidence == "none" and final_pain != "overall_dissatisfaction":
        deps.demote_primary_pain(result, final_pain)
        result["pain_category"] = "overall_dissatisfaction"
        confidence = deps.compute_pain_confidence(result, "overall_dissatisfaction")
    result["pain_confidence"] = confidence

    if events and isinstance(events, list) and len(events) > 0:
        first = events[0] if isinstance(events[0], dict) else {}
        event_text = str(first.get("event", "")).strip()
        timeframe = str(first.get("timeframe", "")).strip()
        if event_text and timeframe and timeframe.lower() != "null":
            st["turning_point"] = f"{event_text} ({timeframe})"
        elif event_text:
            st["turning_point"] = event_text
        else:
            st.setdefault("turning_point", None)
    else:
        st.setdefault("turning_point", None)

    ba["has_budget_authority"] = engine.derive_budget_authority(result)
    cc["price_complaint"] = price_complaint
    cc["price_context"] = pricing_phrases[0] if pricing_phrases else None

    result["replacement_mode"] = deps.derive_replacement_mode(result, source_row)
    result["operating_model_shift"] = deps.derive_operating_model_shift(result, source_row)
    result["productivity_delta_claim"] = deps.derive_productivity_delta_claim(source_row)
    result["org_pressure_type"] = deps.derive_org_pressure_type(source_row)
    result["salience_flags"] = deps.derive_salience_flags(result, source_row)
    result["evidence_spans"] = deps.derive_evidence_spans(result, source_row)
    result["evidence_map_hash"] = engine.map_hash
    return result
