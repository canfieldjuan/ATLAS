from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnrichmentRepairDeps:
    normalize_text_list: Any
    normalize_pain_category: Any
    contains_any: Any
    coerce_json_dict: Any
    is_unknownish: Any
    trusted_repair_sources: Any
    normalize_company_name: Any
    repair_negative_patterns: tuple[str, ...]
    repair_competitor_patterns: tuple[str, ...]
    repair_pricing_patterns: tuple[str, ...]
    repair_recommend_patterns: tuple[str, ...]
    repair_feature_gap_patterns: tuple[str, ...]
    repair_timeline_patterns: tuple[str, ...]
    repair_category_shift_patterns: tuple[str, ...]
    repair_currency_re: Any


def repair_text_blob(source_row: dict[str, Any]) -> str:
    return " ".join(
        str(source_row.get(field) or "")
        for field in ("summary", "review_text", "pros", "cons")
    ).lower()


def repair_target_fields(
    result: dict[str, Any],
    source_row: dict[str, Any],
    *,
    deps: EnrichmentRepairDeps,
) -> list[str]:
    targets: list[str] = []

    def add_target(field: str) -> None:
        if field not in targets:
            targets.append(field)

    review_blob = repair_text_blob(source_row)
    source = str(source_row.get("source") or "").strip().lower()
    status = str(source_row.get("enrichment_status") or "").strip().lower()

    complaints = deps.normalize_text_list(result.get("specific_complaints"))
    pricing_phrases = deps.normalize_text_list(result.get("pricing_phrases"))
    recommendation_language = deps.normalize_text_list(result.get("recommendation_language"))
    feature_gaps = deps.normalize_text_list(result.get("feature_gaps"))
    event_mentions = result.get("event_mentions") or []
    competitors = result.get("competitors_mentioned") or []
    salience_flags = {
        str(flag or "").strip().lower()
        for flag in result.get("salience_flags") or []
        if str(flag or "").strip()
    }
    timeline = deps.coerce_json_dict(result.get("timeline"))

    if deps.normalize_pain_category(result.get("pain_category")) == "overall_dissatisfaction" and deps.contains_any(review_blob, deps.repair_negative_patterns):
        for field in ("specific_complaints", "pricing_phrases", "recommendation_language"):
            add_target(field)
    if not competitors and deps.contains_any(review_blob, deps.repair_competitor_patterns):
        add_target("competitors_mentioned")
    if not pricing_phrases and deps.contains_any(review_blob, deps.repair_pricing_patterns):
        add_target("pricing_phrases")
    if (
        str(result.get("pain_category") or "").strip().lower() not in {"pricing", "contract_lock_in"}
        and (deps.repair_currency_re.search(review_blob) or "explicit_dollar" in salience_flags)
    ):
        for field in ("specific_complaints", "pricing_phrases"):
            add_target(field)
    if not complaints and deps.contains_any(review_blob, deps.repair_negative_patterns):
        add_target("specific_complaints")
    if not recommendation_language and deps.contains_any(review_blob, deps.repair_recommend_patterns):
        add_target("recommendation_language")
    if not feature_gaps and deps.contains_any(review_blob, deps.repair_feature_gap_patterns):
        add_target("feature_gaps")
    if not event_mentions and deps.contains_any(review_blob, ("renewal", "migration", "switched", "price increase", "invoice")):
        add_target("event_mentions")
    if deps.contains_any(review_blob, deps.repair_timeline_patterns) and deps.is_unknownish(timeline.get("decision_timeline")) and not event_mentions:
        add_target("event_mentions")
    if competitors and all(
        not str(comp.get("reason_category") or "").strip()
        for comp in competitors if isinstance(comp, dict)
    ):
        add_target("specific_complaints")
    if deps.contains_any(review_blob, deps.repair_category_shift_patterns) and not feature_gaps and not complaints:
        add_target("specific_complaints")
    if status == "no_signal" and source in deps.trusted_repair_sources() and deps.contains_any(
        review_blob, deps.repair_negative_patterns + deps.repair_competitor_patterns,
    ):
        for field in ("specific_complaints", "pricing_phrases", "competitors_mentioned", "recommendation_language"):
            add_target(field)
    return targets


def needs_field_repair(
    result: dict[str, Any],
    source_row: dict[str, Any],
    *,
    deps: EnrichmentRepairDeps,
) -> bool:
    return bool(repair_target_fields(result, source_row, deps=deps))


def has_structural_gap(result: dict[str, Any], *, deps: EnrichmentRepairDeps) -> bool:
    buyer_authority = deps.coerce_json_dict(result.get("buyer_authority"))
    timeline = deps.coerce_json_dict(result.get("timeline"))
    contract = deps.coerce_json_dict(result.get("contract_context"))
    return any((
        deps.is_unknownish(buyer_authority.get("role_type")),
        deps.is_unknownish(timeline.get("decision_timeline")),
        deps.is_unknownish(contract.get("contract_value_signal")),
    ))


def apply_structural_repair(
    baseline: dict[str, Any],
    repair: dict[str, Any],
    *,
    deps: EnrichmentRepairDeps,
) -> tuple[dict[str, Any], list[str]]:
    merged = json.loads(json.dumps(baseline))
    applied: list[str] = []

    buyer_authority = deps.coerce_json_dict(merged.get("buyer_authority"))
    repair_authority = deps.coerce_json_dict(repair.get("buyer_authority"))
    if deps.is_unknownish(buyer_authority.get("role_type")) and not deps.is_unknownish(repair_authority.get("role_type")):
        buyer_authority["role_type"] = repair_authority.get("role_type")
        applied.append("buyer_authority.role_type")
    if deps.is_unknownish(buyer_authority.get("buying_stage")) and not deps.is_unknownish(repair_authority.get("buying_stage")):
        buyer_authority["buying_stage"] = repair_authority.get("buying_stage")
        applied.append("buyer_authority.buying_stage")
    if applied:
        merged["buyer_authority"] = buyer_authority

    timeline = deps.coerce_json_dict(merged.get("timeline"))
    repair_timeline = deps.coerce_json_dict(repair.get("timeline"))
    for field in ("decision_timeline", "contract_end", "evaluation_deadline"):
        if deps.is_unknownish(timeline.get(field)) and not deps.is_unknownish(repair_timeline.get(field)):
            timeline[field] = repair_timeline.get(field)
            applied.append(f"timeline.{field}")
    if any(field.startswith("timeline.") for field in applied):
        merged["timeline"] = timeline

    contract = deps.coerce_json_dict(merged.get("contract_context"))
    repair_contract = deps.coerce_json_dict(repair.get("contract_context"))
    for field in ("contract_value_signal", "usage_duration", "price_context"):
        if deps.is_unknownish(contract.get(field)) and not deps.is_unknownish(repair_contract.get(field)):
            contract[field] = repair_contract.get(field)
            applied.append(f"contract_context.{field}")
    if any(field.startswith("contract_context.") for field in applied):
        merged["contract_context"] = contract

    return merged, applied


def apply_field_repair(
    baseline: dict[str, Any],
    repair: dict[str, Any],
    *,
    deps: EnrichmentRepairDeps,
) -> tuple[dict[str, Any], list[str]]:
    merged = json.loads(json.dumps(baseline))
    applied: list[str] = []

    for field in ("specific_complaints", "pricing_phrases", "recommendation_language", "feature_gaps"):
        existing_items = deps.normalize_text_list(merged.get(field))
        repair_items = deps.normalize_text_list(repair.get(field))
        seen = {item.strip().lower() for item in existing_items if item.strip()}
        appended = False
        for item in repair_items:
            key = item.strip().lower()
            if key and key not in seen:
                existing_items.append(item)
                seen.add(key)
                appended = True
        if appended:
            merged[field] = existing_items
            applied.append(field)

    existing_events = []
    seen_events: set[tuple[str, str]] = set()
    for event in merged.get("event_mentions") or []:
        if not isinstance(event, dict):
            continue
        key = (
            str(event.get("event") or "").strip().lower(),
            str(event.get("timeframe") or "").strip().lower(),
        )
        if key not in seen_events:
            existing_events.append(dict(event))
            seen_events.add(key)
    event_added = False
    for event in repair.get("event_mentions") or []:
        if not isinstance(event, dict):
            continue
        key = (
            str(event.get("event") or "").strip().lower(),
            str(event.get("timeframe") or "").strip().lower(),
        )
        if key[0] and key not in seen_events:
            existing_events.append(dict(event))
            seen_events.add(key)
            event_added = True
    if event_added:
        merged["event_mentions"] = existing_events
        applied.append("event_mentions")

    existing_competitors = []
    seen_competitors: dict[str, dict[str, Any]] = {}
    for comp in merged.get("competitors_mentioned") or []:
        if not isinstance(comp, dict):
            continue
        name = str(comp.get("name") or "").strip()
        if not name:
            continue
        key = deps.normalize_company_name(name) or name.lower()
        clone = dict(comp)
        existing_competitors.append(clone)
        seen_competitors[key] = clone
    competitor_changed = False
    for comp in repair.get("competitors_mentioned") or []:
        if not isinstance(comp, dict):
            continue
        name = str(comp.get("name") or "").strip()
        if not name:
            continue
        key = deps.normalize_company_name(name) or name.lower()
        if key in seen_competitors:
            target = seen_competitors[key]
            for field in ("reason_detail",):
                if not str(target.get(field) or "").strip() and str(comp.get(field) or "").strip():
                    target[field] = comp.get(field)
                    competitor_changed = True
            existing_features = deps.normalize_text_list(target.get("features"))
            feature_seen = {item.strip().lower() for item in existing_features if item.strip()}
            for item in deps.normalize_text_list(comp.get("features")):
                key_feature = item.strip().lower()
                if key_feature and key_feature not in feature_seen:
                    existing_features.append(item)
                    feature_seen.add(key_feature)
                    competitor_changed = True
            if existing_features:
                target["features"] = existing_features
            continue
        clone = dict(comp)
        existing_competitors.append(clone)
        seen_competitors[key] = clone
        competitor_changed = True
    if competitor_changed:
        merged["competitors_mentioned"] = existing_competitors
        applied.append("competitors_mentioned")

    return merged, applied
