from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("atlas.autonomous.tasks.b2b_enrichment")


@dataclass(frozen=True)
class EnrichmentValidationDeps:
    coerce_bool: Any
    normalize_pain_category: Any
    normalize_budget_value_text: Any
    normalize_budget_detail_text: Any
    canonical_role_type: Any
    canonical_role_level: Any
    infer_role_level_from_text: Any
    infer_decision_maker: Any
    infer_buyer_role_type: Any
    coerce_json_dict: Any
    schema_version: Any
    missing_witness_primitives: Any
    compute_derived_fields: Any
    trusted_reviewer_company_name: Any
    churn_signal_bool_fields: tuple[str, ...]
    known_severity_levels: set[str]
    known_lock_in_levels: set[str]
    known_sentiment_directions: set[str]
    known_buying_stages: set[str]
    known_decision_timelines: set[str]
    known_contract_value_signals: set[str]
    known_replacement_modes: set[str]
    known_operating_model_shifts: set[str]
    known_productivity_delta_claims: set[str]
    known_org_pressure_types: set[str]
    known_content_types: set[str]
    known_org_health_levels: set[str]
    known_leadership_qualities: set[str]
    known_innovation_climates: set[str]
    known_morale_levels: set[str]
    known_departure_types: set[str]
    known_pain_categories: set[str]


def validate_enrichment(
    result: dict[str, Any],
    source_row: dict[str, Any] | None = None,
    *,
    deps: EnrichmentValidationDeps,
) -> bool:
    if "churn_signals" not in result:
        return False
    if "urgency_score" not in result:
        return False
    if not isinstance(result.get("churn_signals"), dict):
        return False

    urgency = result.get("urgency_score")
    if isinstance(urgency, str):
        try:
            urgency = float(urgency)
            result["urgency_score"] = urgency
        except (ValueError, TypeError):
            logger.warning("urgency_score is non-numeric string: %r", urgency)
            return False
    if not isinstance(urgency, (int, float)):
        logger.warning("urgency_score has unexpected type: %s", type(urgency).__name__)
        return False
    if urgency < 0 or urgency > 10:
        logger.warning("urgency_score out of range [0,10]: %s", urgency)
        return False

    signals = result["churn_signals"]
    for field in deps.churn_signal_bool_fields:
        if field in signals:
            coerced = deps.coerce_bool(signals[field])
            if coerced is None:
                logger.warning("churn_signals.%s unrecognizable bool: %r -- rejecting", field, signals[field])
                return False
            signals[field] = coerced
    intent = signals.get("intent_to_leave")
    if urgency >= 9 and intent is False:
        logger.warning(
            "Contradictory: urgency=%s but intent_to_leave=false -- accepting with warning",
            urgency,
        )

    reviewer_ctx = result.get("reviewer_context")
    if isinstance(reviewer_ctx, dict) and "decision_maker" in reviewer_ctx:
        coerced = deps.coerce_bool(reviewer_ctx["decision_maker"])
        if coerced is None:
            logger.warning("reviewer_context.decision_maker unrecognizable bool: %r -- rejecting", reviewer_ctx["decision_maker"])
            return False
        reviewer_ctx["decision_maker"] = coerced

    if "would_recommend" in result:
        coerced = deps.coerce_bool(result["would_recommend"])
        result["would_recommend"] = None if coerced is None else coerced

    competitors = result.get("competitors_mentioned")
    if competitors is not None and not isinstance(competitors, list):
        logger.warning("competitors_mentioned is not a list: %s", type(competitors).__name__)
        result["competitors_mentioned"] = []
    elif isinstance(competitors, list):
        result["competitors_mentioned"] = [
            c for c in competitors if isinstance(c, dict) and "name" in c
        ]

    valid_evidence_types = {"explicit_switch", "active_evaluation", "implied_preference", "reverse_flow", "neutral_mention"}
    valid_disp_confidence = {"high", "medium", "low", "none"}
    valid_reason_categories = {"pricing", "features", "reliability", "ux", "support", "integration"}
    evidence_type_to_context = {
        "explicit_switch": "switched_to",
        "active_evaluation": "considering",
        "implied_preference": "compared",
        "reverse_flow": "switched_from",
        "neutral_mention": "compared",
    }
    context_to_evidence_type = {
        "switched_to": "explicit_switch",
        "considering": "active_evaluation",
        "compared": "implied_preference",
        "switched_from": "reverse_flow",
    }
    for comp in result.get("competitors_mentioned", []):
        et = comp.get("evidence_type")
        if et not in valid_evidence_types:
            comp["evidence_type"] = context_to_evidence_type.get(comp.get("context", ""), "neutral_mention")
        dc = comp.get("displacement_confidence")
        if dc not in valid_disp_confidence:
            comp["displacement_confidence"] = "low"
        if comp["evidence_type"] == "reverse_flow":
            comp["displacement_confidence"] = "none"
        if comp["evidence_type"] == "neutral_mention" and comp.get("displacement_confidence") in ("high", "medium"):
            comp["displacement_confidence"] = "low"
        rc = comp.get("reason_category")
        if rc and rc not in valid_reason_categories:
            comp["reason_category"] = None
        comp["context"] = evidence_type_to_context.get(comp["evidence_type"], "compared")
        rc = comp.get("reason_category")
        rd = comp.get("reason_detail")
        if rc and rd:
            comp["reason"] = f"{rc}: {rd}"
        elif rc:
            comp["reason"] = rc
        elif rd:
            comp["reason"] = rd

    qp = result.get("quotable_phrases")
    if qp is not None and not isinstance(qp, list):
        logger.warning("quotable_phrases is not a list: %s", type(qp).__name__)
        result["quotable_phrases"] = []
    fg = result.get("feature_gaps")
    if fg is not None and not isinstance(fg, list):
        logger.warning("feature_gaps is not a list: %s", type(fg).__name__)
        result["feature_gaps"] = []

    pain = result.get("pain_category")
    if pain is not None:
        normalized_pain = deps.normalize_pain_category(pain)
        if normalized_pain != str(pain).strip().lower():
            logger.warning("Normalizing pain_category %r -> %r", pain, normalized_pain)
        result["pain_category"] = normalized_pain

    pc = result.get("pain_categories")
    if pc is not None:
        if not isinstance(pc, list):
            result["pain_categories"] = []
        else:
            cleaned = []
            for item in pc:
                if not isinstance(item, dict):
                    continue
                cat = deps.normalize_pain_category(item.get("category", "overall_dissatisfaction"))
                sev = item.get("severity", "minor")
                if sev not in deps.known_severity_levels:
                    sev = "minor"
                cleaned.append({"category": cat, "severity": sev})
            result["pain_categories"] = cleaned

    bs = result.get("budget_signals")
    if bs is not None:
        if not isinstance(bs, dict):
            result["budget_signals"] = {}
        else:
            for field in ("annual_spend_estimate", "price_per_seat"):
                if field in bs and bs[field] is not None and not isinstance(bs[field], (int, float)):
                    text = deps.normalize_budget_value_text(bs[field])
                    bs[field] = text if text else None
            if "seat_count" in bs and bs["seat_count"] is not None:
                try:
                    seat = int(bs["seat_count"])
                    bs["seat_count"] = seat if 1 <= seat <= 1_000_000 else None
                except (ValueError, TypeError):
                    bs["seat_count"] = None
            if "price_increase_mentioned" in bs:
                coerced = deps.coerce_bool(bs["price_increase_mentioned"])
                bs["price_increase_mentioned"] = coerced if coerced is not None else False
            if "price_increase_detail" in bs and bs["price_increase_detail"] is not None:
                detail = deps.normalize_budget_detail_text(bs["price_increase_detail"])
                bs["price_increase_detail"] = detail if detail else None
                if bs["price_increase_detail"] and not deps.coerce_bool(bs.get("price_increase_mentioned")):
                    bs["price_increase_mentioned"] = True

    uc = result.get("use_case")
    if uc is not None:
        if not isinstance(uc, dict):
            result["use_case"] = {}
        else:
            if "modules_mentioned" in uc and not isinstance(uc["modules_mentioned"], list):
                uc["modules_mentioned"] = []
            if "integration_stack" in uc and not isinstance(uc["integration_stack"], list):
                uc["integration_stack"] = []
            lil = uc.get("lock_in_level")
            if lil and lil not in deps.known_lock_in_levels:
                uc["lock_in_level"] = "unknown"

    reviewer_ctx = result.get("reviewer_context")
    if reviewer_ctx is None or not isinstance(reviewer_ctx, dict):
        result["reviewer_context"] = {}
        reviewer_ctx = result["reviewer_context"]
    role_level = deps.canonical_role_level(reviewer_ctx.get("role_level"))
    if role_level == "unknown":
        role_level = deps.infer_role_level_from_text((source_row or {}).get("reviewer_title"), source_row)
    reviewer_ctx["role_level"] = role_level
    decision_maker = deps.coerce_bool(reviewer_ctx.get("decision_maker"))
    derived_decision_maker = deps.infer_decision_maker(result, source_row)
    if decision_maker is None:
        decision_maker = derived_decision_maker
    else:
        decision_maker = bool(decision_maker or derived_decision_maker)
    reviewer_ctx["decision_maker"] = decision_maker
    company_name = str(reviewer_ctx.get("company_name") or "").strip()
    if company_name:
        reviewer_ctx["company_name"] = company_name
    else:
        trusted_company = deps.trusted_reviewer_company_name(source_row)
        if trusted_company:
            reviewer_ctx["company_name"] = trusted_company

    st = result.get("sentiment_trajectory")
    if st is not None:
        if not isinstance(st, dict):
            result["sentiment_trajectory"] = {}
        else:
            d = st.get("direction")
            if d and d not in deps.known_sentiment_directions:
                st["direction"] = "unknown"

    ba = result.get("buyer_authority")
    if ba is not None:
        if not isinstance(ba, dict):
            result["buyer_authority"] = {}
            ba = result["buyer_authority"]
        reviewer_ctx = result.get("reviewer_context") if isinstance(result.get("reviewer_context"), dict) else {}
        for bool_field in ("has_budget_authority", "executive_sponsor_mentioned"):
            if bool_field in ba:
                coerced = deps.coerce_bool(ba[bool_field])
                ba[bool_field] = coerced if coerced is not None else False
        bstage = ba.get("buying_stage")
        if bstage and bstage not in deps.known_buying_stages:
            ba["buying_stage"] = "unknown"
        canonical_rt = deps.canonical_role_type(ba.get("role_type"))
        derived_role_type = deps.infer_buyer_role_type(
            ba, reviewer_ctx, (source_row or {}).get("reviewer_title"), source_row,
        )
        if canonical_rt == "unknown":
            ba["role_type"] = derived_role_type
        else:
            ba["role_type"] = canonical_rt
        if derived_role_type == "economic_buyer":
            ba["role_type"] = "economic_buyer"
        if ba["role_type"] == "economic_buyer":
            reviewer_ctx["decision_maker"] = True

    tl = result.get("timeline")
    if tl is not None:
        if not isinstance(tl, dict):
            result["timeline"] = {}
        else:
            dt = tl.get("decision_timeline")
            if dt and dt not in deps.known_decision_timelines:
                tl["decision_timeline"] = "unknown"

    cc = result.get("contract_context")
    if cc is not None:
        if not isinstance(cc, dict):
            result["contract_context"] = {}
        else:
            cvs = cc.get("contract_value_signal")
            if cvs and cvs not in deps.known_contract_value_signals:
                cc["contract_value_signal"] = "unknown"

    cc_val = result.get("content_classification")
    if cc_val and cc_val not in deps.known_content_types:
        result["content_classification"] = "review"

    if deps.schema_version(result) >= 3:
        missing_fields = deps.missing_witness_primitives(result)
        if missing_fields:
            if source_row is None:
                logger.warning(
                    "schema v3 enrichment missing witness primitives without source row: %s",
                    ", ".join(missing_fields),
                )
                return False
            try:
                recomputed = deps.compute_derived_fields(json.loads(json.dumps(result)), source_row)
            except Exception:
                logger.warning(
                    "schema v3 witness primitive recompute failed for %s",
                    source_row.get("id"),
                    exc_info=True,
                )
                return False
            result.clear()
            result.update(recomputed)

    replacement_mode = str(result.get("replacement_mode") or "").strip()
    if replacement_mode not in deps.known_replacement_modes:
        result["replacement_mode"] = "none"
    operating_model_shift = str(result.get("operating_model_shift") or "").strip()
    if operating_model_shift not in deps.known_operating_model_shifts:
        result["operating_model_shift"] = "none"
    productivity_delta_claim = str(result.get("productivity_delta_claim") or "").strip()
    if productivity_delta_claim not in deps.known_productivity_delta_claims:
        result["productivity_delta_claim"] = "unknown"
    org_pressure_type = str(result.get("org_pressure_type") or "").strip()
    if org_pressure_type not in deps.known_org_pressure_types:
        result["org_pressure_type"] = "none"

    salience_flags = result.get("salience_flags")
    if salience_flags is not None:
        if not isinstance(salience_flags, list):
            result["salience_flags"] = []
        else:
            result["salience_flags"] = [str(flag).strip() for flag in salience_flags if str(flag or "").strip()]

    evidence_spans = result.get("evidence_spans")
    if evidence_spans is not None:
        if not isinstance(evidence_spans, list):
            result["evidence_spans"] = []
        else:
            cleaned_spans: list[dict[str, Any]] = []
            for idx, span in enumerate(evidence_spans):
                if not isinstance(span, dict):
                    continue
                text = str(span.get("text") or "").strip()
                if not text:
                    continue
                pain_category = str(span.get("pain_category") or "").strip()
                replacement = str(span.get("replacement_mode") or "").strip()
                operating_shift = str(span.get("operating_model_shift") or "").strip()
                productivity = str(span.get("productivity_delta_claim") or "").strip()
                cleaned_spans.append({
                    "span_id": str(span.get("span_id") or f"derived:{idx}"),
                    "_sid": str(span.get("_sid") or span.get("span_id") or f"derived:{idx}"),
                    "text": text,
                    "start_char": span.get("start_char"),
                    "end_char": span.get("end_char"),
                    "signal_type": str(span.get("signal_type") or "review_context"),
                    "pain_category": pain_category if pain_category in deps.known_pain_categories else None,
                    "competitor": str(span.get("competitor") or "").strip() or None,
                    "company_name": str(span.get("company_name") or "").strip() or None,
                    "reviewer_title": str(span.get("reviewer_title") or "").strip() or None,
                    "time_anchor": str(span.get("time_anchor") or "").strip() or None,
                    "numeric_literals": span.get("numeric_literals") if isinstance(span.get("numeric_literals"), dict) else {},
                    "flags": [str(flag).strip() for flag in (span.get("flags") or []) if str(flag or "").strip()],
                    "replacement_mode": replacement if replacement in deps.known_replacement_modes else result.get("replacement_mode"),
                    "operating_model_shift": operating_shift if operating_shift in deps.known_operating_model_shifts else result.get("operating_model_shift"),
                    "productivity_delta_claim": productivity if productivity in deps.known_productivity_delta_claims else result.get("productivity_delta_claim"),
                })
            result["evidence_spans"] = cleaned_spans

    if deps.schema_version(result) >= 3:
        remaining_missing = deps.missing_witness_primitives(result)
        if remaining_missing:
            logger.warning(
                "schema v3 enrichment still missing witness primitives after normalization: %s",
                ", ".join(remaining_missing),
            )
            return False

    insider = result.get("insider_signals")
    if insider is not None:
        if not isinstance(insider, dict):
            result["insider_signals"] = None
        else:
            oh = insider.get("org_health")
            if oh is not None and not isinstance(oh, dict):
                insider["org_health"] = {}
            elif isinstance(oh, dict):
                ci = oh.get("culture_indicators")
                if ci is not None and not isinstance(ci, list):
                    oh["culture_indicators"] = []
                for field, allowed in (
                    ("bureaucracy_level", deps.known_org_health_levels),
                    ("leadership_quality", deps.known_leadership_qualities),
                    ("innovation_climate", deps.known_innovation_climates),
                ):
                    val = oh.get(field)
                    if val and val not in allowed:
                        oh[field] = "unknown"
            td = insider.get("talent_drain")
            if td is not None and not isinstance(td, dict):
                insider["talent_drain"] = {}
            elif isinstance(td, dict):
                for bool_field in ("departures_mentioned", "layoff_fear"):
                    if bool_field in td:
                        coerced = deps.coerce_bool(td[bool_field])
                        td[bool_field] = coerced if coerced is not None else False
                morale = td.get("morale")
                if morale and morale not in deps.known_morale_levels:
                    td["morale"] = "unknown"
            dt = insider.get("departure_type")
            if dt and dt not in deps.known_departure_types:
                insider["departure_type"] = "unknown"

    return True
