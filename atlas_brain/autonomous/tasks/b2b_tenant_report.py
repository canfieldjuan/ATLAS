"""Weekly per-tenant B2B intelligence report generation.

Runs before the global intelligence task (Sunday 8 PM). For each B2B
account with tracked vendors, gathers vendor-scoped intelligence data,
calls the LLM with the existing b2b_churn_intelligence skill, persists
to b2b_intelligence with account_id, and emails the report summary.
"""

import asyncio
import json
import logging
from datetime import date, timezone
from typing import Any

import httpx

from ...config import settings
from ...services.tracing import (
    build_business_trace_context,
    build_reasoning_trace_context,
    tracer,
)
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.tasks.b2b_tenant_report")


def _tenant_report_llm_model(llm_usage: dict[str, Any] | None) -> str:
    """Persist the actual model when tenant report synthesis used an LLM."""
    model = str((llm_usage or {}).get("model") or "").strip()
    return model or "pipeline_deterministic"


def _tenant_report_data_density(
    result: dict[str, Any],
    *,
    llm_usage: dict[str, Any] | None = None,
    narrative_evidence_count: int = 0,
    vendor_reasoning_count: int = 0,
    stratified_reasoning_count: int | None = None,
    synthesis_contract_vendor_count: int = 0,
) -> str:
    """Build persisted density/provenance metadata for tenant reports.

    `stratified_reasoning_count` and `stratified_reasoning_vendors` are retained
    as compatibility aliases for older consumers of tenant-report artifacts.
    """
    if stratified_reasoning_count is not None and not vendor_reasoning_count:
        vendor_reasoning_count = int(stratified_reasoning_count)
    density: dict[str, Any] = {
        "vendors_analyzed": result["vendors_analyzed"],
        "high_intent_companies": result["high_intent_companies"],
        "competitive_flows": result["competitive_flows"],
        "pain_categories": result.get("pain_categories", 0),
        "feature_gaps": result.get("feature_gaps", 0),
        "narrative_evidence_vendors": narrative_evidence_count,
        "vendor_reasoning_vendors": vendor_reasoning_count,
        "stratified_reasoning_vendors": vendor_reasoning_count,
        "synthesis_contract_vendors": synthesis_contract_vendor_count,
    }
    usage = llm_usage or {}
    if usage.get("input_tokens") is not None:
        density["llm_input_tokens"] = usage.get("input_tokens", 0)
    if usage.get("output_tokens") is not None:
        density["llm_output_tokens"] = usage.get("output_tokens", 0)
    if usage.get("chunk_count") is not None:
        density["llm_chunk_count"] = usage.get("chunk_count", 0)
    if usage.get("chunk_failures") is not None:
        density["llm_chunk_failures"] = usage.get("chunk_failures", 0)
    return json.dumps(density)


def _normalize_vendor_name(value: Any) -> str:
    return str(value or "").strip().lower()


def _tenant_report_chunk_size() -> int:
    """Return a safe per-chunk vendor cap for tenant synthesis."""
    model = str(getattr(settings.llm, "openrouter_reasoning_model", "") or "").lower()
    if "gpt-oss" in model:
        return 3
    if "deepseek" in model:
        return 4
    return 6


def _tenant_payload_vendor_chunks(payload: dict[str, Any]) -> list[list[str]]:
    """Build category-aware vendor chunks so cross-vendor context stays coherent."""
    scores = payload.get("vendor_churn_scores") or []
    if not isinstance(scores, list) or not scores:
        return [[]]

    max_vendors = max(1, _tenant_report_chunk_size())
    groups: list[list[str]] = []
    current_category = ""
    current_group: list[str] = []
    seen_vendors: set[str] = set()

    for row in scores:
        if not isinstance(row, dict):
            continue
        vendor = str(row.get("vendor_name") or row.get("vendor") or "").strip()
        if not vendor:
            continue
        vendor_key = _normalize_vendor_name(vendor)
        if vendor_key in seen_vendors:
            continue
        seen_vendors.add(vendor_key)
        category = str(row.get("product_category") or row.get("category") or "").strip().lower()
        if current_group and category != current_category:
            groups.append(current_group)
            current_group = []
        current_category = category
        current_group.append(vendor)
    if current_group:
        groups.append(current_group)

    chunks: list[list[str]] = []
    current_chunk: list[str] = []
    for group in groups:
        while len(group) > max_vendors:
            head = group[:max_vendors]
            group = group[max_vendors:]
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
            chunks.append(head)
        if current_chunk and len(current_chunk) + len(group) > max_vendors:
            chunks.append(current_chunk)
            current_chunk = []
        current_chunk.extend(group)
    if current_chunk:
        chunks.append(current_chunk)
    return chunks or [[]]


def _tenant_item_touches_vendors(item: Any, vendor_set: set[str]) -> bool:
    """Whether a payload item is relevant to the requested vendor subset."""
    if not vendor_set:
        return True
    if not isinstance(item, dict):
        return False
    for key in ("vendor", "vendor_name", "from_vendor", "to_vendor"):
        if _normalize_vendor_name(item.get(key)) in vendor_set:
            return True
    return False


def _compact_tenant_prior_reports(prior_reports: list[Any], vendor_set: set[str]) -> list[dict[str, Any]]:
    """Shrink prior-report blobs to vendor-relevant slices only."""
    compact: list[dict[str, Any]] = []
    for item in prior_reports[:2]:
        if not isinstance(item, dict):
            continue
        entry: dict[str, Any] = {
            "type": item.get("type") or item.get("report_type", ""),
            "date": item.get("date") or item.get("report_date", ""),
        }
        data = item.get("data")
        if isinstance(data, list):
            filtered = [row for row in data if _tenant_item_touches_vendors(row, vendor_set)]
            entry["data"] = filtered[:8]
        elif isinstance(data, dict):
            entry["data"] = {}
        else:
            entry["data"] = []
        compact.append(entry)
    return compact


def _filter_tenant_payload_for_vendors(payload: dict[str, Any], vendors: list[str]) -> dict[str, Any]:
    """Keep only chunk-relevant vendor rows while preserving global metadata."""
    vendor_set = {_normalize_vendor_name(v) for v in vendors if str(v or "").strip()}
    if not vendor_set:
        return dict(payload)

    filtered: dict[str, Any] = {}
    passthrough_keys = {"date", "data_context", "analysis_window_days"}

    for key, value in payload.items():
        if key == "prior_reports":
            filtered[key] = _compact_tenant_prior_reports(value if isinstance(value, list) else [], vendor_set)
            continue
        if key in passthrough_keys:
            filtered[key] = value
            continue
        if isinstance(value, list):
            filtered[key] = [item for item in value if _tenant_item_touches_vendors(item, vendor_set)]
        else:
            filtered[key] = value
    return filtered


def _tenant_vendor_context_lookup(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build canonical tenant vendor context from compact vendor score rows."""
    lookup: dict[str, dict[str, Any]] = {}
    for row in payload.get("vendor_churn_scores") or []:
        if not isinstance(row, dict):
            continue
        vendor = str(row.get("vendor_name") or row.get("vendor") or "").strip()
        if not vendor:
            continue
        vendor_key = _normalize_vendor_name(vendor)
        context: dict[str, Any] = {}
        category = str(row.get("category") or row.get("product_category") or "").strip()
        if category:
            context["category"] = category
        context["vendor"] = vendor
        context["reviews"] = int(row.get("reviews") or 0)
        context["churn"] = int(row.get("churn") or 0)
        try:
            context["urgency"] = round(float(row.get("urgency") or 0), 1)
        except (TypeError, ValueError):
            context["urgency"] = 0.0
        context["rating"] = row.get("rating")
        context["rec_yes"] = int(row.get("rec_yes") or 0)
        context["rec_no"] = int(row.get("rec_no") or 0)
        if isinstance(row.get("category_council"), dict) and row.get("category_council"):
            context["category_council"] = row.get("category_council")
        lookup[vendor_key] = context
    return lookup


def _tenant_vendor_churn_density(context: dict[str, Any]) -> float:
    reviews = int(context.get("reviews") or 0)
    churn = int(context.get("churn") or 0)
    return round((churn * 100.0 / reviews), 1) if reviews else 0.0


def _tenant_vendor_recommend_ratio(context: dict[str, Any]) -> float:
    reviews = int(context.get("reviews") or 0)
    if not reviews:
        return 0.0
    rec_yes = int(context.get("rec_yes") or 0)
    rec_no = int(context.get("rec_no") or 0)
    return round(((rec_yes - rec_no) / reviews) * 100.0, 2)


def _tenant_vendor_confidence(context: dict[str, Any]) -> str:
    reviews = int(context.get("reviews") or 0)
    if reviews >= 50:
        return "high"
    if reviews >= 20:
        return "medium"
    return "low"


def _tenant_vendor_pressure_score(context: dict[str, Any]) -> float:
    from ._b2b_shared import _compute_churn_pressure_score

    return _compute_churn_pressure_score(
        churn_density=_tenant_vendor_churn_density(context),
        avg_urgency=float(context.get("urgency") or 0),
        dm_churn_rate=0.0,
        displacement_mention_count=0,
        price_complaint_rate=0.0,
        total_reviews=int(context.get("reviews") or 0),
    )


def _tenant_weekly_feed_backfill_row(context: dict[str, Any]) -> dict[str, Any]:
    council = context.get("category_council") or {}
    winner = str(council.get("winner") or "").strip()
    vendor = str(context.get("vendor") or "").strip()
    action = "Review churn drivers and active evaluation pressure before renewal."
    if winner and winner != vendor:
        action = f"Pressure-test pricing and workflow fit against {winner} before renewal."
    return {
        "vendor": vendor,
        "category": context.get("category") or "",
        "total_reviews": int(context.get("reviews") or 0),
        "avg_urgency": float(context.get("urgency") or 0),
        "churn_signal_density": _tenant_vendor_churn_density(context),
        "churn_pressure_score": _tenant_vendor_pressure_score(context),
        "sample_size_confidence": _tenant_vendor_confidence(context),
        "trend": "stable",
        "sentiment_direction": "insufficient_history",
        "top_pain": "unknown",
        "pain_breakdown": [],
        "budget_context": {},
        "named_accounts": [],
        "evidence": [],
        "key_quote": None,
        "dominant_buyer_role": "unknown",
        "dm_churn_rate": 0.0,
        "price_complaint_rate": 0.0,
        "top_feature_gaps": [],
        "top_displacement_targets": [],
        "action_recommendation": action,
    }


def _tenant_scorecard_backfill_row(context: dict[str, Any]) -> dict[str, Any]:
    from ._b2b_shared import _fallback_scorecard_expert_take

    council = context.get("category_council") or {}
    winner = str(council.get("winner") or "").strip()
    vendor = str(context.get("vendor") or "").strip()
    competitor_overlap = []
    top_competitor_threat = "Insufficient displacement data"
    if winner and winner != vendor:
        competitor_overlap = [{"competitor": winner, "mentions": 0}]
        top_competitor_threat = winner

    scorecard = {
        "vendor": vendor,
        "category": context.get("category") or "",
        "total_reviews": int(context.get("reviews") or 0),
        "churn_signal_density": _tenant_vendor_churn_density(context),
        "positive_review_pct": None,
        "avg_urgency": float(context.get("urgency") or 0),
        "recommend_ratio": _tenant_vendor_recommend_ratio(context),
        "sample_size_confidence": _tenant_vendor_confidence(context),
        "churn_pressure_score": _tenant_vendor_pressure_score(context),
        "risk_level": "high" if _tenant_vendor_pressure_score(context) >= 70 else "medium" if _tenant_vendor_pressure_score(context) >= 40 else "low",
        "top_pain": "unknown",
        "budget_context": {},
        "sentiment_direction": "insufficient_history",
        "dm_churn_rate": 0.0,
        "price_complaint_rate": 0.0,
        "competitor_overlap": competitor_overlap,
        "top_competitor_threat": top_competitor_threat,
        "trend": "stable",
        "feature_analysis": {"loved": [], "hated": []},
        "churn_predictors": {
            "high_risk_industries": [],
            "high_risk_sizes": [],
            "dm_churn_rate": 0.0,
            "price_complaint_rate": 0.0,
        },
        "evidence": [],
        "named_accounts": [],
        "customer_profile": {
            "typical_industries": [],
            "typical_company_size": [],
            "primary_use_cases": [],
            "top_integrations": [],
            "industry_distribution": [],
            "company_size_distribution": [],
        },
        "dominant_buyer_role": "unknown",
        "industry_distribution": [],
        "company_size_distribution": [],
    }
    scorecard["expert_take"] = _fallback_scorecard_expert_take(scorecard)
    return scorecard


def _apply_tenant_vendor_context(parsed: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Reattach canonical vendor context and backfill omitted vendor rows."""
    lookup = _tenant_vendor_context_lookup(payload)
    if not lookup:
        return parsed

    for section_name in ("weekly_churn_feed", "vendor_scorecards"):
        rows = parsed.get(section_name)
        if not isinstance(rows, list):
            rows = []
            parsed[section_name] = rows
        rows[:] = [
            row
            for row in rows
            if isinstance(row, dict)
            and _normalize_vendor_name(row.get("vendor") or row.get("vendor_name")) in lookup
        ]
        existing: set[str] = set()
        for row in rows:
            vendor_key = _normalize_vendor_name(row.get("vendor") or row.get("vendor_name"))
            if vendor_key:
                existing.add(vendor_key)
            context = lookup.get(vendor_key)
            if not context:
                continue
            category = str(context.get("category") or "").strip()
            if category:
                row["category"] = category
            row.setdefault("total_reviews", int(context.get("reviews") or 0))
            row.setdefault("avg_urgency", float(context.get("urgency") or 0))
            row.setdefault("churn_signal_density", _tenant_vendor_churn_density(context))
            row.setdefault("sample_size_confidence", _tenant_vendor_confidence(context))
            row.setdefault("churn_pressure_score", _tenant_vendor_pressure_score(context))
            council = context.get("category_council")
            if isinstance(council, dict) and council:
                row["category_council"] = council
            else:
                row.pop("category_council", None)
        for vendor_key, context in lookup.items():
            if vendor_key in existing:
                continue
            if section_name == "weekly_churn_feed":
                row = _tenant_weekly_feed_backfill_row(context)
            else:
                row = _tenant_scorecard_backfill_row(context)
            council = context.get("category_council")
            if isinstance(council, dict) and council:
                row["category_council"] = council
            rows.append(row)
        rows.sort(
            key=lambda row: (
                float((row or {}).get("churn_pressure_score") or 0),
                float((row or {}).get("avg_urgency") or 0),
                str((row or {}).get("vendor") or (row or {}).get("vendor_name") or ""),
            ),
            reverse=True,
        )
    return parsed


def _tenant_category_contexts(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for context in _tenant_vendor_context_lookup(payload).values():
        category = str(context.get("category") or "").strip()
        if not category:
            continue
        grouped.setdefault(category, []).append(context)
    return grouped


def _tenant_category_overview_backfill_row(category: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        contexts,
        key=lambda context: (
            _tenant_vendor_pressure_score(context),
            _tenant_vendor_churn_density(context),
            str(context.get("vendor") or ""),
        ),
        reverse=True,
    )
    leader = ordered[0] if ordered else {}
    council = next(
        (
            context.get("category_council")
            for context in ordered
            if isinstance(context.get("category_council"), dict) and context.get("category_council")
        ),
        {},
    )
    winner = str(council.get("winner") or "").strip()
    leader_vendor = str(leader.get("vendor") or "").strip()
    regime = str(council.get("market_regime") or "").strip().lower()
    dominant_pain = "pricing" if "price" in regime else "unknown"
    if dominant_pain == "unknown":
        dominant_pain = "pricing" if any(
            str(item.get("winner") or "").strip()
            for item in [council]
            if "price" in str(item.get("conclusion") or "").lower()
        ) else "unknown"
    signal = str(council.get("conclusion") or "").strip()
    if not signal:
        signal = (
            f"{len(ordered)} tracked vendors are in scope for {category}; "
            f"{leader_vendor or 'the leading vendor'} shows the highest churn pressure."
        )
    return {
        "category": category,
        "dominant_pain": dominant_pain,
        "highest_churn_risk": leader_vendor,
        "emerging_challenger": winner if winner and winner != leader_vendor else "",
        "market_shift_signal": signal,
    }


def _tenant_payload_edge_lookup(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload.get("competitive_displacement") or []:
        if not isinstance(row, dict):
            continue
        from_vendor = str(row.get("vendor") or row.get("from_vendor") or "").strip()
        to_vendor = str(row.get("competitor") or row.get("to_vendor") or "").strip()
        if not from_vendor or not to_vendor:
            continue
        lookup[(from_vendor.casefold(), to_vendor.casefold())] = row
    return lookup


def _tenant_displacement_backfill_row(
    edge: dict[str, Any],
    *,
    reason_lookup: dict[tuple[str, str], list[str]],
) -> dict[str, Any]:
    from ._b2b_shared import _normalize_displacement_driver_label

    from_vendor = str(edge.get("vendor") or edge.get("from_vendor") or "").strip()
    to_vendor = str(edge.get("competitor") or edge.get("to_vendor") or "").strip()
    mention_count = int(edge.get("mention_count") or 0)
    primary_driver = ""
    reason_counts = edge.get("reason_categories") or {}
    if isinstance(reason_counts, dict):
        for label, _count in sorted(
            reason_counts.items(),
            key=lambda item: (-int(item[1] or 0), str(item[0] or "")),
        ):
            primary_driver = _normalize_displacement_driver_label(label)
            if primary_driver:
                break
    if not primary_driver:
        key = (from_vendor.casefold(), to_vendor.casefold())
        for reason in reason_lookup.get(key, []):
            primary_driver = _normalize_displacement_driver_label(reason)
            if primary_driver:
                break
    if mention_count >= 20 or int(edge.get("explicit_switches") or 0) > 0:
        strength = "strong"
    elif mention_count >= 5 or int(edge.get("active_evaluations") or 0) > 0:
        strength = "moderate"
    else:
        strength = "light"
    return {
        "from_vendor": from_vendor,
        "to_vendor": to_vendor,
        "mention_count": mention_count,
        "primary_driver": primary_driver or "overall_dissatisfaction",
        "signal_strength": strength,
        "key_quote": None,
    }


def _apply_tenant_category_context(parsed: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    category_lookup = _tenant_category_contexts(payload)
    if not category_lookup:
        return parsed
    rows = parsed.get("category_insights")
    if not isinstance(rows, list):
        rows = []
        parsed["category_insights"] = rows
    rows[:] = [
        row
        for row in rows
        if isinstance(row, dict)
        and str(row.get("category") or "").strip() in category_lookup
    ]
    existing: set[str] = set()
    for row in rows:
        category = str(row.get("category") or "").strip()
        if not category:
            continue
        existing.add(category)
        canonical = _tenant_category_overview_backfill_row(category, category_lookup.get(category, []))
        row.update(canonical)
    for category, contexts in category_lookup.items():
        if category not in existing:
            rows.append(_tenant_category_overview_backfill_row(category, contexts))
    rows.sort(
        key=lambda row: (
            len(category_lookup.get(str((row or {}).get("category") or "").strip(), [])),
            str((row or {}).get("category") or ""),
        ),
        reverse=True,
    )
    return parsed


def _apply_tenant_displacement_context(parsed: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    from ._b2b_shared import _build_reason_lookup

    edge_lookup = _tenant_payload_edge_lookup(payload)
    if not edge_lookup:
        return parsed
    reason_lookup = _build_reason_lookup(payload.get("competitor_reasons") or [])
    rows = parsed.get("displacement_map")
    if not isinstance(rows, list):
        rows = []
        parsed["displacement_map"] = rows
    rows[:] = [
        row
        for row in rows
        if isinstance(row, dict)
        and (
            str(row.get("from_vendor") or "").strip().casefold(),
            str(row.get("to_vendor") or "").strip().casefold(),
        ) in edge_lookup
    ]
    existing: set[tuple[str, str]] = set()
    for row in rows:
        key = (
            str(row.get("from_vendor") or "").strip().casefold(),
            str(row.get("to_vendor") or "").strip().casefold(),
        )
        if key not in edge_lookup:
            continue
        existing.add(key)
        row.update(
            _tenant_displacement_backfill_row(edge_lookup[key], reason_lookup=reason_lookup)
        )
    for key, edge in edge_lookup.items():
        if key not in existing:
            rows.append(_tenant_displacement_backfill_row(edge, reason_lookup=reason_lookup))
    rows.sort(
        key=lambda row: (
            int((row or {}).get("mention_count") or 0),
            str((row or {}).get("from_vendor") or ""),
            str((row or {}).get("to_vendor") or ""),
        ),
        reverse=True,
    )
    return parsed


def _apply_tenant_report_context(parsed: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    parsed = _apply_tenant_vendor_context(parsed, payload)
    parsed = _apply_tenant_category_context(parsed, payload)
    parsed = _apply_tenant_displacement_context(parsed, payload)
    return parsed


def _apply_tenant_synthesis_context(
    parsed: dict[str, Any],
    synthesis_views: dict[str, Any],
    *,
    requested_as_of: date | None,
) -> int:
    """Attach shared synthesis contracts to tenant vendor rows."""
    if not isinstance(parsed, dict) or not synthesis_views:
        return 0

    from .b2b_churn_reports import _attach_synthesis_contracts_to_report_entry

    attached_vendors: set[str] = set()
    sections = (
        ("weekly_churn_feed", "weekly_churn_feed", False),
        ("vendor_scorecards", "vendor_scorecard", True),
    )
    for section_name, consumer_name, include_displacement in sections:
        rows = parsed.get(section_name)
        if not isinstance(rows, list):
            continue
        for entry in rows:
            if not isinstance(entry, dict):
                continue
            vendor = str(entry.get("vendor") or entry.get("vendor_name") or "").strip()
            if not vendor:
                continue
            view = synthesis_views.get(vendor)
            if view is None:
                continue
            _attach_synthesis_contracts_to_report_entry(
                entry,
                view,
                consumer_name=consumer_name,
                requested_as_of=requested_as_of,
                include_displacement=include_displacement,
            )
            attached_vendors.add(vendor)
    return len(attached_vendors)


def _build_deterministic_tenant_report(
    payload: dict[str, Any],
    synthesis_views: dict[str, Any],
    *,
    requested_as_of: date | None,
) -> tuple[dict[str, Any], int]:
    """Build tenant report sections deterministically from scoped payload data."""
    from ._b2b_shared import _build_validated_executive_summary, _executive_source_list

    parsed: dict[str, Any] = {
        "weekly_churn_feed": [],
        "vendor_scorecards": [],
        "displacement_map": [],
        "category_insights": [],
        "timeline_hot_list": [],
    }
    parsed = _apply_tenant_report_context(parsed, payload)
    attached = _apply_tenant_synthesis_context(
        parsed,
        synthesis_views,
        requested_as_of=requested_as_of,
    )
    parsed["executive_summary"] = _build_validated_executive_summary(
        parsed,
        data_context=payload.get("data_context") or {},
        executive_sources=_executive_source_list(),
        report_type="weekly_churn_feed",
    )
    return parsed, attached


async def _build_deterministic_tenant_report_from_raw(
    pool,
    raw_artifacts: dict[str, Any],
    *,
    payload: dict[str, Any],
    synthesis_views: dict[str, Any],
    requested_as_of: date | None,
    analysis_window_days: int,
) -> tuple[dict[str, Any], int]:
    """Build tenant report sections from scoped raw artifacts."""
    from ._b2b_shared import (
        _build_validated_executive_summary,
        _executive_source_list,
        _fallback_scorecard_expert_take,
    )
    from .b2b_churn_reports import _build_deterministic_report_bundle

    requested = requested_as_of or date.today()
    bundle = await _build_deterministic_report_bundle(
        pool,
        as_of=requested,
        analysis_window_days=analysis_window_days,
        vendor_scores=list(
            raw_artifacts.get("vendor_scores_from_signals")
            or raw_artifacts.get("vendor_scores")
            or []
        ),
        competitive_disp=list(raw_artifacts.get("competitive_disp") or []),
        pain_dist=list(raw_artifacts.get("pain_dist") or []),
        feature_gaps=list(raw_artifacts.get("feature_gaps") or []),
        price_rates=list(raw_artifacts.get("price_rates") or []),
        dm_rates=list(raw_artifacts.get("dm_rates") or []),
        churning_companies=list(raw_artifacts.get("churning_companies") or []),
        quotable_evidence=list(raw_artifacts.get("quotable_evidence") or []),
        budget_signals=list(raw_artifacts.get("budget_signals") or []),
        use_case_dist=list(raw_artifacts.get("use_case_dist") or []),
        sentiment_traj=list(raw_artifacts.get("sentiment_traj") or []),
        sentiment_tenure=list(raw_artifacts.get("sentiment_tenure_raw") or []),
        buyer_auth=list(raw_artifacts.get("buyer_auth") or []),
        timeline_signals=list(raw_artifacts.get("timeline_signals") or []),
        competitor_reasons=list(raw_artifacts.get("competitor_reasons") or []),
        keyword_spikes=list(raw_artifacts.get("keyword_spikes") or []),
        product_profiles_raw=list(raw_artifacts.get("product_profiles_raw") or []),
        prior_reports=list(raw_artifacts.get("prior_reports") or []),
        displacement_provenance=dict(raw_artifacts.get("displacement_provenance") or {}),
        review_text_agg=raw_artifacts.get("review_text_aggs") or ([], []),
        department_dist=list(raw_artifacts.get("department_dist") or []),
        contract_ctx=raw_artifacts.get("contract_ctx_aggs") or ([], []),
        turning_points=list(raw_artifacts.get("turning_points_raw") or []),
        synthesis_views=synthesis_views,
    )
    for scorecard in bundle["vendor_scorecards"]:
        if not scorecard.get("expert_take"):
            scorecard["expert_take"] = _fallback_scorecard_expert_take(scorecard)

    parsed: dict[str, Any] = {
        "weekly_churn_feed": bundle["weekly_churn_feed"],
        "vendor_scorecards": bundle["vendor_scorecards"],
        "displacement_map": bundle["displacement_map"],
        "category_insights": bundle["category_insights"],
        "timeline_hot_list": [],
    }
    parsed["executive_summary"] = _build_validated_executive_summary(
        parsed,
        data_context=dict(raw_artifacts.get("data_context") or {}),
        executive_sources=_executive_source_list(),
        report_type="weekly_churn_feed",
    )
    return parsed, int(bundle.get("attached_contract_vendors") or 0)


def _merge_unique_rows(
    rows: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    seen: set[tuple[str, ...]] = set()
    merged: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = tuple(str(row.get(field) or "").strip().lower() for field in key_fields)
        if not any(key):
            continue
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
    return merged


def _merge_tenant_chunk_outputs(
    partials: list[dict[str, Any]],
    *,
    data_context: dict[str, Any],
) -> dict[str, Any]:
    """Merge chunk-level tenant synthesis into one canonical report payload."""
    from ._b2b_shared import _build_validated_executive_summary, _executive_source_list

    weekly_feed = _merge_unique_rows(
        [row for partial in partials for row in (partial.get("weekly_churn_feed") or [])],
        key_fields=("vendor",),
    )
    scorecards = _merge_unique_rows(
        [row for partial in partials for row in (partial.get("vendor_scorecards") or [])],
        key_fields=("vendor",),
    )
    displacement_map = _merge_unique_rows(
        [row for partial in partials for row in (partial.get("displacement_map") or [])],
        key_fields=("from_vendor", "to_vendor"),
    )
    category_insights = _merge_unique_rows(
        [row for partial in partials for row in (partial.get("category_insights") or [])],
        key_fields=("category",),
    )
    timeline_hot_list = _merge_unique_rows(
        [row for partial in partials for row in (partial.get("timeline_hot_list") or [])],
        key_fields=("company", "vendor"),
    )

    weekly_feed.sort(
        key=lambda row: (
            float(row.get("churn_pressure_score") or 0),
            float(row.get("avg_urgency") or 0),
        ),
        reverse=True,
    )
    scorecards.sort(
        key=lambda row: float(row.get("churn_pressure_score") or 0),
        reverse=True,
    )
    displacement_map.sort(
        key=lambda row: int(row.get("mention_count") or 0),
        reverse=True,
    )
    timeline_hot_list.sort(
        key=lambda row: float(row.get("urgency") or 0),
        reverse=True,
    )

    merged = {
        "weekly_churn_feed": weekly_feed,
        "vendor_scorecards": scorecards,
        "displacement_map": displacement_map,
        "category_insights": category_insights,
        "timeline_hot_list": timeline_hot_list,
    }
    merged["executive_summary"] = _build_validated_executive_summary(
        merged,
        data_context=data_context or {},
        executive_sources=_executive_source_list(),
        report_type="weekly_churn_feed",
    )
    return merged


async def _run_tenant_synthesis_llm(
    payload: dict[str, Any],
    *,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the existing tenant-report synthesis skill once."""
    from ...pipelines.llm import (
        call_llm_with_skill,
        get_pipeline_llm,
        parse_json_response,
    )
    from ...services.b2b.cache_runner import (
        lookup_b2b_exact_stage_text,
        prepare_b2b_exact_skill_stage_request,
        store_b2b_exact_stage_text,
    )
    from ...services.b2b.llm_exact_cache import CacheUnavailable, llm_identity

    llm_usage: dict[str, Any] = {}
    cache_stage_id = "b2b_tenant_report.synthesis_chunk"
    request: Any | None = None
    resolved_llm = get_pipeline_llm(workload="synthesis")
    provider, model = llm_identity(resolved_llm)
    if provider and model:
        try:
            request, _ = prepare_b2b_exact_skill_stage_request(
                cache_stage_id,
                skill_name="digest/b2b_churn_intelligence",
                payload=payload,
                provider=provider,
                model=model,
                max_tokens=max_tokens,
                temperature=0.4,
                response_format={"type": "json_object"},
            )
        except CacheUnavailable:
            request = None

    if request is not None:
        cached = await lookup_b2b_exact_stage_text(request)
        if cached is not None:
            parsed_cached = parse_json_response(
                cached["response_text"],
                recover_truncated=True,
            )
            if not parsed_cached.get("_parse_fallback"):
                llm_usage.update(
                    {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "model": cached["model"],
                        "provider": cached["provider"],
                        "cache_hit": True,
                    }
                )
                return parsed_cached, llm_usage

    analysis = await asyncio.wait_for(
        asyncio.to_thread(
            call_llm_with_skill,
            "digest/b2b_churn_intelligence",
            payload,
            max_tokens=max_tokens,
            temperature=0.4,
            workload="synthesis",
            response_format={"type": "json_object"},
            usage_out=llm_usage,
        ),
        timeout=300,
    )
    if not analysis:
        raise ValueError("tenant report llm returned no analysis")
    parsed = parse_json_response(analysis, recover_truncated=True)

    if request is None:
        provider = str(llm_usage.get("provider") or "")
        model = str(llm_usage.get("model") or "")
        if provider and model:
            try:
                request, _ = prepare_b2b_exact_skill_stage_request(
                    cache_stage_id,
                    skill_name="digest/b2b_churn_intelligence",
                    payload=payload,
                    provider=provider,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=0.4,
                    response_format={"type": "json_object"},
                )
            except CacheUnavailable:
                request = None

    if request is not None and not parsed.get("_parse_fallback"):
        await store_b2b_exact_stage_text(
            request,
            response_text=analysis,
            usage=llm_usage,
            metadata={"task": "tenant_report", "cache_stage": "synthesis_chunk"},
        )

    return parsed, llm_usage


async def _run_chunked_tenant_synthesis(
    payload: dict[str, Any],
    *,
    max_tokens: int,
    data_context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run tenant synthesis in vendor chunks and merge the outputs."""
    chunks = _tenant_payload_vendor_chunks(payload)
    if len(chunks) <= 1:
        return await _run_tenant_synthesis_llm(payload, max_tokens=max_tokens)

    partials: list[dict[str, Any]] = []
    total_input = 0
    total_output = 0
    model_name = ""
    chunk_failures = 0

    for chunk in chunks:
        chunk_payload = _filter_tenant_payload_for_vendors(payload, chunk)
        try:
            parsed, usage = await _run_tenant_synthesis_llm(chunk_payload, max_tokens=max_tokens)
            partials.append(parsed)
            total_input += int(usage.get("input_tokens") or 0)
            total_output += int(usage.get("output_tokens") or 0)
            if not model_name and usage.get("model"):
                model_name = str(usage.get("model") or "")
        except Exception:
            chunk_failures += 1
            logger.warning(
                "Tenant-report synthesis chunk failed for vendors=%s",
                ", ".join(chunk),
                exc_info=True,
            )

    if not partials:
        raise ValueError("tenant report llm returned no analysis")

    merged = _merge_tenant_chunk_outputs(partials, data_context=data_context)
    usage = {
        "model": model_name,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "chunk_count": len(chunks),
        "chunk_failures": chunk_failures,
    }
    return merged, usage


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Generate scoped intelligence reports for each B2B tenant."""
    cfg = settings.b2b_churn
    if not cfg.enabled or not cfg.intelligence_enabled:
        return {"_skip_synthesis": "B2B churn intelligence disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    # Only generate for accounts with b2b_starter+ plans (reports require b2b_starter)
    accounts = await pool.fetch(
        """
        SELECT sa.id AS account_id, sa.name AS account_name,
               sa.plan,
               su.email AS owner_email,
               array_agg(tv.vendor_name) AS vendors
        FROM saas_accounts sa
        JOIN saas_users su ON su.account_id = sa.id AND su.role = 'owner'
        JOIN tracked_vendors tv ON tv.account_id = sa.id
        WHERE sa.product IN ('b2b_retention', 'b2b_challenger')
          AND sa.plan IN ('b2b_starter', 'b2b_growth', 'b2b_pro')
        GROUP BY sa.id, sa.name, sa.plan, su.email
        """
    )

    if not accounts:
        return {"_skip_synthesis": "No eligible B2B accounts for reports"}

    from .b2b_churn_intelligence import gather_intelligence_data

    reports_generated = 0
    today = date.today()

    for acct in accounts:
        account_id = acct["account_id"]
        vendor_names = [v for v in (acct["vendors"] or []) if v]
        synthesis_views: dict[str, Any] = {}

        if not vendor_names:
            continue

        span = tracer.start_span(
            span_name="b2b.tenant_report",
            operation_type="intelligence",
            session_id=str(account_id),
            metadata={
                "business": build_business_trace_context(
                    account_id=str(account_id),
                    workflow="tenant_report",
                    report_type="weekly_b2b_intelligence",
                    vendor_name=", ".join(vendor_names[:5]),
                ),
            },
        )

        try:
            result = await gather_intelligence_data(
                pool,
                window_days=cfg.intelligence_window_days,
                min_reviews=cfg.intelligence_min_reviews,
                vendor_names=vendor_names,
                include_raw_artifacts=True,
            )
        except Exception as exc:
            tracer.end_span(
                span,
                status="failed",
                error_message=str(exc),
                error_type=type(exc).__name__,
            )
            logger.error("Data gathering failed for account %s: %s", account_id, exc)
            continue

        payload = result["payload"]
        narrative_evidence_count = 0
        vendor_reasoning_count = 0

        # Skip if no data
        if not payload.get("vendor_churn_scores") and not payload.get("high_intent_companies"):
            tracer.end_span(span, status="completed", output_data={"skipped": "no scoped intelligence data"})
            continue

        try:
            from .b2b_churn_reports import _fetch_latest_synthesis_views

            synthesis_views = await _fetch_latest_synthesis_views(
                pool,
                as_of=today,
                analysis_window_days=cfg.intelligence_window_days,
            )
        except Exception:
            logger.debug("Tenant report synthesis contract load skipped", exc_info=True)

        # --- Narrative engine: structured evidence chains + rule evaluation ---
        try:
            from atlas_brain.reasoning.narrative import NarrativeEngine

            narrative_engine = NarrativeEngine(pool)
            narrative_payloads = []
            all_triggered_rules = []

            for vs in (payload.get("vendor_churn_scores") or []):
                vname = vs.get("vendor_name") or vs.get("vendor") or ""
                if not vname:
                    continue

                # Build structured narrative from available evidence
                narrative = narrative_engine.build_vendor_narrative(
                    vendor_name=vname,
                    snapshot=vs,
                    archetype_match={"archetype": vs.get("archetype", ""),
                                     "signal_score": vs.get("signal_score", 0)},
                )

                # Evaluate threshold rules (P5-005)
                triggered = narrative_engine.evaluate_rules(vname, vs, narrative.archetype)
                all_triggered_rules.extend(triggered)

                # Build explainability audit trail
                explain = narrative_engine.build_explainability(narrative)

                # Add structured evidence to payload
                intel_payload = NarrativeEngine.to_intelligence_payload(narrative)
                intel_payload["explainability"] = explain
                narrative_payloads.append(intel_payload)

            if narrative_payloads:
                payload["narrative_evidence"] = narrative_payloads
                narrative_evidence_count = len(narrative_payloads)

            # P5-005: Send ntfy alerts for critical/high priority rule triggers
            critical_rules = [t for t in all_triggered_rules
                              if t.rule.priority in ("critical", "high")]
            if critical_rules:
                await _send_rule_alerts(critical_rules)

        except Exception:
            logger.debug("Narrative engine enrichment skipped", exc_info=True)

        # --- Vendor reasoning context: synthesis-first per vendor ---
        try:
            from ._b2b_synthesis_reader import (
                load_best_reasoning_views,
                load_prior_reasoning_snapshots,
                synthesis_view_to_reasoning_entry,
            )

            vendor_names = [
                (vs.get("vendor_name") or vs.get("vendor") or "").strip()
                for vs in (payload.get("vendor_churn_scores") or [])
                if (vs.get("vendor_name") or vs.get("vendor") or "").strip()
            ]
            reasoning_views = await load_best_reasoning_views(
                pool,
                vendor_names,
                as_of=today,
                analysis_window_days=cfg.intelligence_window_days,
            )
            if reasoning_views:
                prior_reasoning = await load_prior_reasoning_snapshots(
                    pool,
                    list(reasoning_views.keys()),
                    before_date=today,
                    analysis_window_days=cfg.intelligence_window_days,
                )
                vendor_reasoning = []
                for current_name, view in reasoning_views.items():
                    entry = synthesis_view_to_reasoning_entry(view)
                    prior = prior_reasoning.get(current_name, {})
                    vendor_reasoning.append({
                        "vendor_name": current_name,
                        "mode": entry.get("mode"),
                        "archetype": entry.get("archetype", ""),
                        "confidence": entry.get("confidence"),
                        "tokens_used": 0,
                        "conclusion": {
                            "archetype": entry.get("archetype", ""),
                            "risk_level": entry.get("risk_level", ""),
                            "executive_summary": entry.get("executive_summary", ""),
                            "key_signals": entry.get("key_signals", []),
                            "falsification_conditions": entry.get("falsification_conditions", []),
                            "uncertainty_sources": entry.get("uncertainty_sources", []),
                        },
                        "archetype_was": prior.get("archetype"),
                        "confidence_was": prior.get("confidence"),
                        "archetype_changed": (
                            prior.get("archetype") != entry.get("archetype")
                            if prior.get("archetype") and entry.get("archetype")
                            else None
                        ),
                    })

                payload["vendor_reasoning"] = vendor_reasoning
                # Legacy payload alias retained for downstream compatibility.
                payload["stratified_reasoning"] = vendor_reasoning
                vendor_reasoning_count = len(vendor_reasoning)
                logger.info(
                    "Vendor reasoning context: %d vendors (%s)",
                    len(vendor_reasoning),
                    ", ".join(f"{r['vendor_name']}={r['mode']}" for r in vendor_reasoning),
                )
        except Exception:
            logger.debug("Vendor reasoning context integration skipped", exc_info=True)

        raw_artifacts = result.get("raw_artifacts")
        try:
            if isinstance(raw_artifacts, dict) and raw_artifacts:
                parsed, synthesis_contract_vendor_count = await _build_deterministic_tenant_report_from_raw(
                    pool,
                    raw_artifacts,
                    payload=payload,
                    synthesis_views=synthesis_views,
                    requested_as_of=today,
                    analysis_window_days=cfg.intelligence_window_days,
                )
            else:
                parsed, synthesis_contract_vendor_count = _build_deterministic_tenant_report(
                    payload,
                    synthesis_views,
                    requested_as_of=today,
                )
        except Exception:
            logger.warning(
                "Tenant raw deterministic build failed for account=%s; falling back to payload build",
                account_id,
                exc_info=True,
            )
            parsed, synthesis_contract_vendor_count = _build_deterministic_tenant_report(
                payload,
                synthesis_views,
                requested_as_of=today,
            )
        llm_usage: dict[str, Any] = {}
        llm_summary_applied = False
        try:
            llm_parsed, llm_usage = await _run_chunked_tenant_synthesis(
                payload,
                max_tokens=cfg.intelligence_max_tokens,
                data_context=payload.get("data_context") or {},
            )
            llm_summary = str(llm_parsed.get("executive_summary") or "").strip()
            if llm_summary:
                parsed["executive_summary"] = llm_summary
                llm_summary_applied = True
        except asyncio.TimeoutError:
            logger.warning(
                "Tenant report LLM timed out for account=%s; persisting deterministic output",
                account_id,
            )
        except Exception as exc:
            logger.warning(
                "Tenant report LLM failed for account=%s; persisting deterministic output: %s",
                account_id,
                exc,
            )

        if llm_usage.get("input_tokens"):
            logger.info("b2b_tenant_report LLM tokens: in=%d out=%d model=%s account=%s chunks=%s failures=%s",
                         llm_usage["input_tokens"], llm_usage["output_tokens"],
                         llm_usage.get("model", ""), account_id,
                         llm_usage.get("chunk_count", 1), llm_usage.get("chunk_failures", 0))
        exec_summary = parsed.get("executive_summary", "")

        # Persist reports with account_id
        report_types = [
            ("weekly_churn_feed", parsed.get("weekly_churn_feed", [])),
            ("vendor_scorecard", parsed.get("vendor_scorecards", [])),
            ("displacement_report", parsed.get("displacement_map", [])),
            ("category_overview", parsed.get("category_insights", [])),
        ]

        data_density = _tenant_report_data_density(
            result,
            llm_usage=llm_usage,
            narrative_evidence_count=narrative_evidence_count,
            vendor_reasoning_count=vendor_reasoning_count,
            synthesis_contract_vendor_count=synthesis_contract_vendor_count,
        )
        llm_model = _tenant_report_llm_model(llm_usage)

        try:
            async with pool.transaction() as conn:
                for report_type, data in report_types:
                    report_status = "published" if data else "failed"
                    latest_failure_step = None if data else "no_data"
                    latest_error_code = None if data else "no_data"
                    latest_error_summary = None if data else "Report has no data"
                    blocker_count = 0 if data else 1
                    report_row = await conn.fetchrow(
                        """
                        INSERT INTO b2b_intelligence (
                            report_date, report_type, intelligence_data,
                            executive_summary, data_density, status,
                            llm_model, account_id,
                            latest_run_id, latest_attempt_no, latest_failure_step,
                            latest_error_code, latest_error_summary,
                            blocker_count, warning_count
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8,
                                  $9, $10, $11, $12, $13, $14, $15)
                        ON CONFLICT (report_date, report_type, LOWER(COALESCE(vendor_filter,'')),
                                     LOWER(COALESCE(category_filter,'')),
                                     COALESCE(account_id, '00000000-0000-0000-0000-000000000000'::uuid))
                        DO UPDATE SET intelligence_data = EXCLUDED.intelligence_data,
                                      executive_summary = EXCLUDED.executive_summary,
                                      data_density = EXCLUDED.data_density,
                                      status = EXCLUDED.status,
                                      llm_model = EXCLUDED.llm_model,
                                      latest_run_id = EXCLUDED.latest_run_id,
                                      latest_attempt_no = EXCLUDED.latest_attempt_no,
                                      latest_failure_step = EXCLUDED.latest_failure_step,
                                      latest_error_code = EXCLUDED.latest_error_code,
                                      latest_error_summary = EXCLUDED.latest_error_summary,
                                      blocker_count = EXCLUDED.blocker_count,
                                      warning_count = EXCLUDED.warning_count,
                                      created_at = now()
                        RETURNING id
                        """,
                        today,
                        report_type,
                        json.dumps(data, default=str),
                        exec_summary,
                        data_density,
                        report_status,
                        llm_model,
                        account_id,
                        str(task.id),
                        1,
                        latest_failure_step,
                        latest_error_code,
                        latest_error_summary,
                        blocker_count,
                        0,
                    )
                    from ..visibility import record_attempt
                    await record_attempt(
                        pool,
                        artifact_type="churn_report",
                        artifact_id=str(report_row["id"]),
                        run_id=str(task.id),
                        stage="persistence",
                        status="succeeded" if data else "failed",
                        failure_step=latest_failure_step,
                        error_message=latest_error_summary,
                    )
                    if not data:
                        from ..visibility import emit_event
                        await emit_event(
                            pool,
                            stage="reports",
                            event_type="report_failed",
                            entity_type="churn_report",
                            entity_id=str(report_row["id"]),
                            artifact_type="churn_report",
                            run_id=str(task.id),
                            severity="warning",
                            actionable=False,
                            reason_code="no_data",
                            summary=f"{report_type} produced no data for account {account_id}",
                            source_table="b2b_intelligence",
                            source_id=str(report_row["id"]),
                        )
            reports_generated += 1
        except Exception as exc:
            tracer.end_span(
                span,
                status="failed",
                error_message=str(exc),
                error_type=type(exc).__name__,
            )
            logger.error("Failed to persist tenant report for %s: %s", account_id, exc)
            continue

        # Send email summary via Resend if configured
        await _send_report_email(
            owner_email=acct["owner_email"],
            account_name=acct["account_name"],
            vendor_names=vendor_names,
            summary=exec_summary,
        )

        # ntfy notification
        await _send_ntfy(acct["account_name"], len(vendor_names))
        tracer.end_span(
            span,
            status="completed",
            input_data={"vendor_names": vendor_names, "window_days": cfg.intelligence_window_days},
            output_data={
                "reports_generated": len(report_types),
                "account_name": acct["account_name"],
                "synthesis_contract_vendors": synthesis_contract_vendor_count,
                "llm_summary_applied": llm_summary_applied,
            },
            input_tokens=llm_usage.get("input_tokens"),
            output_tokens=llm_usage.get("output_tokens"),
            metadata={
                "reasoning": build_reasoning_trace_context(
                    decision={"report_types": [name for name, _ in report_types]},
                    evidence={
                        "vendors_analyzed": result["vendors_analyzed"],
                        "high_intent_companies": result["high_intent_companies"],
                        "competitive_flows": result["competitive_flows"],
                    },
                    rationale=exec_summary,
                ),
            },
        )

    return {
        "_skip_synthesis": "B2B tenant reports complete",
        "accounts_processed": len(accounts),
        "reports_generated": reports_generated,
    }


async def _send_report_email(
    *,
    owner_email: str,
    account_name: str,
    vendor_names: list[str],
    summary: str,
) -> None:
    """Send weekly report email via Resend."""
    cfg = settings.campaign_sequence
    if not cfg.resend_api_key or not cfg.resend_from_email:
        return

    vendors_str = ", ".join(vendor_names[:5])
    subject = f"Weekly Intelligence Report: {vendors_str}"
    body = (
        f"<h2>Weekly B2B Intelligence Report</h2>"
        f"<p><strong>Account:</strong> {account_name}</p>"
        f"<p><strong>Tracked Vendors:</strong> {vendors_str}</p>"
        f"<hr>"
        f"<h3>Executive Summary</h3>"
        f"<p>{summary or 'No significant changes this week.'}</p>"
        f"<hr>"
        f"<p><em>View full details in your churn intelligence feed.</em></p>"
    )

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {cfg.resend_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": cfg.resend_from_email,
                    "to": [owner_email],
                    "subject": subject,
                    "html": body,
                },
            )
            resp.raise_for_status()
    except Exception as exc:
        logger.warning("Failed to send report email to %s: %s", owner_email, exc)


async def _send_ntfy(account_name: str, vendor_count: int) -> None:
    """Send ntfy notification about completed tenant report."""
    if not settings.alerts.ntfy_enabled:
        return

    ntfy_url = f"{settings.alerts.ntfy_url.rstrip('/')}/{settings.alerts.ntfy_topic}"
    message = f"Weekly report generated for {account_name} ({vendor_count} vendors)"
    headers = {
        "Title": f"Tenant Report: {account_name}",
        "Priority": "default",
        "Tags": "chart_with_upwards_trend,b2b,report",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(ntfy_url, content=message, headers=headers)
            resp.raise_for_status()
    except Exception as exc:
        logger.warning("ntfy failed for tenant report %s: %s", account_name, exc)


async def _send_rule_alerts(triggered_rules: list) -> None:
    """Send ntfy alerts for critical/high priority threshold rule triggers (P5-005)."""
    if not settings.alerts.ntfy_enabled:
        return

    # Group by vendor
    by_vendor: dict[str, list] = {}
    for t in triggered_rules:
        by_vendor.setdefault(t.vendor_name, []).append(t)

    ntfy_url = f"{settings.alerts.ntfy_url.rstrip('/')}/{settings.alerts.ntfy_topic}"

    for vendor, rules in by_vendor.items():
        lines = [f"- {t.rule.description} (actual: {t.actual_value:.1f})" for t in rules]
        message = f"{vendor}: {len(rules)} threshold alert(s)\n" + "\n".join(lines)
        headers = {
            "Title": f"B2B Alert: {vendor}",
            "Priority": "high",
            "Tags": "warning,b2b,threshold",
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(ntfy_url, content=message, headers=headers)
                resp.raise_for_status()
        except Exception as exc:
            logger.warning("ntfy rule alert failed for %s: %s", vendor, exc)
