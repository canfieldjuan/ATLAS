"""Follow-up task: build and persist churn intelligence reports.

Runs after b2b_churn_core (staggered cron). Reads persisted artifacts
from b2b_reasoning_synthesis, b2b_churn_signals, b2b_reviews, and
related tables. Builds deterministic reports, runs LLM enrichment,
and persists to b2b_intelligence.

Reasoning is synthesis-first: builds reasoning_lookup from
b2b_reasoning_synthesis views and treats missing synthesis coverage as a
real gap instead of silently filling from b2b_churn_signals.
"""

import asyncio
import json
import logging
from datetime import date, datetime, timezone
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from ._b2b_shared import _timing_summary_payload, _build_inbound_displacement_lookup, _reasoning_int

logger = logging.getLogger("atlas.tasks.b2b_churn_reports")


def _pair_opponent(pair_key: tuple[str, ...] | list[str] | Any, vendor: str) -> str:
    """Return the non-self vendor from a pair key, or empty string."""
    if not isinstance(pair_key, (tuple, list)):
        return ""
    vendor_text = str(vendor or "").strip()
    for value in pair_key:
        candidate = str(value or "").strip()
        if candidate and candidate != vendor_text:
            return candidate
    return ""


def _copy_reference_ids(reference_ids: dict[str, Any] | None) -> dict[str, list[str]]:
    from ._b2b_cross_vendor_synthesis import _copy_reference_ids as _canonical_copy

    result = _canonical_copy(reference_ids)
    # Strip empty lists so callers can use ``if reference_ids:`` truthiness
    return {k: v for k, v in result.items() if v}


def _build_category_council_context(council: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(council, dict):
        return {}
    conclusion = council.get("conclusion")
    if not isinstance(conclusion, dict):
        return {}
    context = {
        "winner": conclusion.get("winner") or "",
        "loser": conclusion.get("loser") or "",
        "conclusion": conclusion.get("conclusion") or "",
        "market_regime": conclusion.get("market_regime") or "",
        "durability": conclusion.get("durability_assessment") or "",
        "confidence": council.get("confidence"),
        "key_insights": conclusion.get("key_insights") or [],
    }
    if not any(
        [
            context.get("winner"),
            context.get("loser"),
            context.get("conclusion"),
            context.get("market_regime"),
            context.get("key_insights"),
        ]
    ):
        return {}
    return context


def _build_scorecard_narrative_payload(
    scorecard: dict[str, Any],
    *,
    reasoning_lookup: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a compact LLM payload for scorecard narrative generation."""
    from ._b2b_shared import _build_scorecard_locked_facts

    def _sort_text(value: Any) -> str:
        return str(value or "").strip().lower()

    def _sort_float(value: Any) -> float:
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0

    payload = {
        key: scorecard[key]
        for key in (
            "vendor",
            "churn_pressure_score",
            "risk_level",
            "churn_signal_density",
            "avg_urgency",
            "trend",
            "sentiment_direction",
        )
        if key in scorecard
    }

    feature_analysis = scorecard.get("feature_analysis") or {}
    if isinstance(feature_analysis, dict):
        loved = list(feature_analysis.get("loved") or [])
        hated = list(feature_analysis.get("hated") or [])
        loved.sort(
            key=lambda item: (
                -_sort_float(item.get("score") if isinstance(item, dict) else 0),
                _sort_text(item.get("feature") if isinstance(item, dict) else item),
            )
        )
        hated.sort(
            key=lambda item: (
                -_sort_float(item.get("mentions") if isinstance(item, dict) else 0),
                _sort_text(item.get("feature") if isinstance(item, dict) else item),
            )
        )
        payload["feature_analysis"] = {
            "loved": loved[:3],
            "hated": hated[:3],
        }

    churn_predictors = scorecard.get("churn_predictors") or {}
    if isinstance(churn_predictors, dict):
        high_risk_industries = list(
            churn_predictors.get("high_risk_industries") or []
        )
        high_risk_sizes = list(
            churn_predictors.get("high_risk_sizes") or []
        )
        high_risk_industries.sort(
            key=lambda item: (
                -_sort_float(item.get("count") if isinstance(item, dict) else 0),
                _sort_text(item.get("industry") if isinstance(item, dict) else item),
            )
        )
        high_risk_sizes.sort(
            key=lambda item: (
                -_sort_float(item.get("count") if isinstance(item, dict) else 0),
                _sort_text(item.get("size") if isinstance(item, dict) else item),
            )
        )
        payload["churn_predictors"] = {
            "high_risk_industries": high_risk_industries[:2],
            "high_risk_sizes": high_risk_sizes[:2],
            "dm_churn_rate": churn_predictors.get("dm_churn_rate"),
            "price_complaint_rate": churn_predictors.get("price_complaint_rate"),
        }

    competitor_overlap = scorecard.get("competitor_overlap") or []
    if competitor_overlap:
        competitor_rows = list(competitor_overlap)
        competitor_rows.sort(
            key=lambda item: (
                -_sort_float(item.get("mentions") if isinstance(item, dict) else 0),
                _sort_text(item.get("competitor") if isinstance(item, dict) else item),
            )
        )
        payload["competitor_overlap"] = competitor_rows[:3]

    cross_vendor_comparisons = scorecard.get("cross_vendor_comparisons") or []
    if cross_vendor_comparisons:
        comparison_rows = [
            item
            for item in list(cross_vendor_comparisons)
            if isinstance(item, dict)
        ]
        comparison_rows.sort(
            key=lambda item: (
                -_sort_float(item.get("confidence")),
                _sort_text(item.get("opponent")),
            )
        )
        payload["cross_vendor_comparisons"] = [
            {
                "opponent": item.get("opponent", ""),
                "conclusion": item.get("conclusion", ""),
                "confidence": item.get("confidence", 0),
                "resource_advantage": item.get("resource_advantage", ""),
            }
            for item in comparison_rows[:2]
        ]
        if not payload["cross_vendor_comparisons"]:
            payload.pop("cross_vendor_comparisons", None)

    category_council = scorecard.get("category_council") or {}
    if isinstance(category_council, dict) and any(
        [
            category_council.get("winner"),
            category_council.get("loser"),
            category_council.get("conclusion"),
            category_council.get("market_regime"),
            category_council.get("key_insights"),
        ]
    ):
        payload["category_council"] = {
            "winner": category_council.get("winner", ""),
            "loser": category_council.get("loser", ""),
            "conclusion": category_council.get("conclusion", ""),
            "market_regime": category_council.get("market_regime", ""),
            "durability": category_council.get("durability", ""),
            "confidence": category_council.get("confidence", 0),
            "key_insights": list(category_council.get("key_insights") or [])[:3],
        }

    payload["locked_facts"] = _build_scorecard_locked_facts(scorecard)

    if scorecard.get("archetype"):
        rc = (reasoning_lookup or {}).get(scorecard.get("vendor", ""), {})
        payload["reasoning_conclusion"] = {
            "archetype": scorecard["archetype"],
            "confidence": scorecard.get("archetype_confidence") or rc.get("confidence", 0),
            "executive_summary": scorecard.get("reasoning_summary", ""),
            "key_signals": list(rc.get("key_signals", []))[:4],
        }

    # Phase 6: include governance context for calibrated narrative
    wts = scorecard.get("why_they_stay")
    if isinstance(wts, dict) and wts:
        payload["why_they_stay"] = {
            "summary": wts.get("summary", ""),
            "top_strengths": [
                s.get("area", "") for s in wts.get("strengths", [])
                if isinstance(s, dict)
            ][:3],
        }
    cp = scorecard.get("confidence_posture")
    if isinstance(cp, dict) and cp.get("limits"):
        payload["confidence_limits"] = cp["limits"]
    cg = scorecard.get("coverage_gaps")
    if isinstance(cg, list) and cg:
        coverage_rows = [g for g in cg if isinstance(g, dict)]
        coverage_rows.sort(
            key=lambda item: (
                _sort_text(item.get("type")),
                _sort_text(item.get("area")),
            )
        )
        payload["coverage_gaps"] = [
            {"type": g.get("type", ""), "area": g.get("area", "")}
            for g in coverage_rows[:5]
        ]
    ml = scorecard.get("metric_ledger")
    if isinstance(ml, list) and ml:
        metric_rows = [m for m in ml if isinstance(m, dict)]
        metric_rows.sort(
            key=lambda item: (
                _sort_text(item.get("label")),
                _sort_text(item.get("scope")),
                _sort_text(item.get("value")),
            )
        )
        payload["metric_ledger"] = [
            {"label": m.get("label", ""), "value": m.get("value"), "scope": m.get("scope", "")}
            for m in metric_rows[:10]
        ]

    return payload


def _scorecard_narrative_max_tokens() -> int:
    """Return a configured token budget for scorecard narrative generation."""
    model = str(getattr(settings.llm, "openrouter_reasoning_model", "") or "").lower()
    cfg = settings.b2b_churn
    if "gpt-oss" in model:
        return int(cfg.scorecard_narrative_gpt_oss_max_tokens)
    if "deepseek" in model:
        return int(cfg.scorecard_narrative_deepseek_max_tokens)
    return int(cfg.scorecard_narrative_max_tokens)




def _account_reasoning_named_accounts(
    account_reasoning: dict[str, Any] | None,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Convert account_reasoning top_accounts into report named-account rows."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    if not isinstance(account_reasoning, dict):
        return rows
    for item in account_reasoning.get("top_accounts") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("company") or "").strip()
        if not name or name.casefold() in seen:
            continue
        seen.add(name.casefold())
        try:
            urgency = round(max(0.0, min(float(item.get("intent_score") or 0), 1.0)) * 10, 1)
        except (TypeError, ValueError):
            urgency = 0.0
        rows.append({
            "company": name,
            "urgency": urgency,
            "source": "account_reasoning",
            "confidence_score": item.get("intent_score"),
            "reasoning_backed": True,
            "source_id": item.get("source_id"),
        })
        if len(rows) >= limit:
            break
    return rows


def _apply_account_reasoning_to_report_entry(
    entry: dict[str, Any],
    account_reasoning: dict[str, Any] | None,
) -> None:
    """Promote account reasoning into visible report fields."""
    if not isinstance(entry, dict) or not isinstance(account_reasoning, dict) or not account_reasoning:
        return
    entry["account_reasoning"] = account_reasoning
    metrics: dict[str, int] = {}
    for key in ("total_accounts", "high_intent_count", "active_eval_count"):
        value = _reasoning_int(account_reasoning.get(key))
        if value is not None:
            metrics[key] = value
    summary = str(account_reasoning.get("market_summary") or "").strip()
    if not summary:
        active_eval = metrics.get("active_eval_count")
        high_intent = metrics.get("high_intent_count")
        total_accounts = metrics.get("total_accounts")
        if active_eval is not None and high_intent is not None:
            summary = (
                f"{active_eval} accounts are in active evaluation while "
                f"{high_intent} accounts show high-intent churn signals."
            )
        elif high_intent is not None:
            summary = f"{high_intent} accounts show high-intent churn signals."
        elif total_accounts is not None:
            summary = f"{total_accounts} accounts are currently in scope."
    if summary:
        entry["account_pressure_summary"] = summary
    if metrics:
        entry["account_pressure_metrics"] = metrics
    reasoning_accounts = _account_reasoning_named_accounts(account_reasoning)
    if reasoning_accounts:
        entry["priority_account_names"] = [row["company"] for row in reasoning_accounts]
    merged_accounts = list(entry.get("named_accounts") or [])
    seen_names = {
        str(item.get("company") or "").strip().casefold()
        for item in merged_accounts
        if isinstance(item, dict) and str(item.get("company") or "").strip()
    }
    for row in reasoning_accounts:
        key = str(row.get("company") or "").strip().casefold()
        if key and key not in seen_names:
            merged_accounts.append(row)
            seen_names.add(key)
    if merged_accounts:
        entry["named_accounts"] = merged_accounts[:5]


def _apply_timing_intelligence_to_report_entry(
    entry: dict[str, Any],
    timing_intelligence: dict[str, Any] | None,
) -> None:
    """Promote timing intelligence into visible report fields."""
    if not isinstance(entry, dict) or not isinstance(timing_intelligence, dict) or not timing_intelligence:
        return
    entry["timing_intelligence"] = timing_intelligence
    summary, metrics, priority_triggers = _timing_summary_payload(
        timing_intelligence
    )
    if summary:
        entry["timing_summary"] = summary
    if metrics:
        entry["timing_metrics"] = metrics
    if priority_triggers:
        entry["priority_timing_triggers"] = priority_triggers


async def _fetch_latest_synthesis_views(
    pool,
    *,
    as_of: date,
    analysis_window_days: int,
) -> dict[str, Any]:
    """Fetch best reasoning views for all vendors (synthesis-first, legacy fallback)."""
    from ._b2b_synthesis_reader import discover_reasoning_vendor_names, load_best_reasoning_views

    all_names = await discover_reasoning_vendor_names(
        pool,
        as_of=as_of,
        analysis_window_days=analysis_window_days,
    )
    if not all_names:
        return {}
    return await load_best_reasoning_views(
        pool,
        all_names,
        as_of=as_of,
        analysis_window_days=analysis_window_days,
    )


def _attach_synthesis_contracts_to_report_entry(
    entry: dict[str, Any],
    view: Any,
    *,
    consumer_name: str,
    requested_as_of: date | None,
    include_displacement: bool,
) -> None:
    """Attach contract-backed reasoning blocks to a report entry."""
    if not isinstance(entry, dict) or view is None:
        return
    from ._b2b_synthesis_reader import (
        inject_synthesis_freshness,
    )

    context_getter = getattr(view, "filtered_consumer_context", None)
    if callable(context_getter):
        context = context_getter(consumer_name)
    else:
        consumer_context = getattr(view, "consumer_context", None)
        context = consumer_context(consumer_name) if callable(consumer_context) else {}

    contracts: dict[str, Any] = {}
    materialized = context.get("reasoning_contracts") or view.materialized_contracts()
    vendor_core = materialized.get("vendor_core_reasoning")
    if vendor_core:
        contracts["vendor_core_reasoning"] = vendor_core
        timing_intelligence = vendor_core.get("timing_intelligence")
        if isinstance(timing_intelligence, dict) and timing_intelligence:
            _apply_timing_intelligence_to_report_entry(
                entry,
                timing_intelligence,
            )

    account_reasoning = materialized.get("account_reasoning")
    if account_reasoning:
        contracts["account_reasoning"] = account_reasoning
        _apply_account_reasoning_to_report_entry(entry, account_reasoning)

    if include_displacement:
        displacement = materialized.get("displacement_reasoning")
        if displacement:
            contracts["displacement_reasoning"] = displacement
        category = materialized.get("category_reasoning")
        if category:
            contracts["category_reasoning"] = category

    if contracts:
        contracts["schema_version"] = str(
            view.reasoning_contracts.get("schema_version") or "v1"
        )
        entry["reasoning_contracts"] = contracts
    reference_ids = context.get("reference_ids")
    if not isinstance(reference_ids, dict):
        reference_ids = getattr(view, "reference_ids", None)
    if isinstance(reference_ids, dict) and reference_ids:
        entry["reference_ids"] = reference_ids
    scope_manifest = context.get("scope_manifest")
    if isinstance(scope_manifest, dict) and scope_manifest:
        entry["scope_manifest"] = scope_manifest
    atom_summary = context.get("reasoning_atom_summary")
    if isinstance(atom_summary, dict) and atom_summary:
        entry["reasoning_atom_summary"] = atom_summary
    delta = context.get("reasoning_delta")
    if isinstance(delta, dict) and delta:
        entry["reasoning_delta"] = delta

    if view.primary_wedge:
        entry["synthesis_wedge"] = view.primary_wedge.value
        entry["synthesis_wedge_label"] = view.wedge_label

    meta = view.meta
    if meta:
        entry["evidence_window"] = meta
        ew_start = meta.get("evidence_window_start")
        ew_end = meta.get("evidence_window_end")
        if ew_start and ew_end:
            try:
                start = date.fromisoformat(str(ew_start)[:10])
                end = date.fromisoformat(str(ew_end)[:10])
                window_days = (end - start).days
                entry["evidence_window_days"] = window_days
            except (TypeError, ValueError):
                pass

    entry["synthesis_schema_version"] = view.schema_version
    entry["reasoning_source"] = "b2b_reasoning_synthesis"
    inject_synthesis_freshness(
        entry,
        view,
        requested_as_of=requested_as_of,
    )
    contract_gaps = context.get("reasoning_contract_gaps") or []
    if contract_gaps:
        entry["reasoning_contract_gaps"] = contract_gaps
    section_disclaimers = context.get("reasoning_section_disclaimers")
    if isinstance(section_disclaimers, dict) and section_disclaimers:
        entry["reasoning_section_disclaimers"] = section_disclaimers

    # Phase 6: surface governance fields for confidence/freshness transparency
    why_they_stay = getattr(view, "why_they_stay", None)
    if isinstance(why_they_stay, dict) and why_they_stay:
        entry["why_they_stay"] = why_they_stay

    confidence_posture = getattr(view, "confidence_posture", None)
    if isinstance(confidence_posture, dict) and confidence_posture:
        entry["confidence_posture"] = confidence_posture
        limits = confidence_posture.get("limits")
        if limits:
            entry["confidence_limits"] = limits

    coverage_gaps = getattr(view, "coverage_gaps", None)
    if isinstance(coverage_gaps, list) and coverage_gaps:
        entry["coverage_gaps"] = coverage_gaps

    evidence_governance = getattr(view, "evidence_governance", None)
    if isinstance(evidence_governance, dict) and evidence_governance:
        ml = evidence_governance.get("metric_ledger")
        if ml:
            entry["metric_ledger"] = ml


def _build_report_lookup_bundle(
    *,
    competitive_disp: list[dict[str, Any]],
    pain_dist: list[dict[str, Any]],
    feature_gaps: list[dict[str, Any]],
    price_rates: list[dict[str, Any]],
    dm_rates: list[dict[str, Any]],
    churning_companies: list[dict[str, Any]],
    quotable_evidence: list[dict[str, Any]],
    budget_signals: list[dict[str, Any]],
    use_case_dist: list[dict[str, Any]],
    sentiment_traj: list[dict[str, Any]],
    buyer_auth: list[dict[str, Any]],
    timeline_signals: list[dict[str, Any]],
    keyword_spikes: list[dict[str, Any]],
    product_profiles_raw: list[dict[str, Any]],
    review_text_agg: tuple[list[dict[str, Any]], list[dict[str, Any]]],
    department_dist: list[dict[str, Any]],
    contract_ctx: tuple[list[dict[str, Any]], list[dict[str, Any]]],
    turning_points: list[dict[str, Any]],
    sentiment_tenure: list[dict[str, Any]],
    evidence_vault_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from ._b2b_shared import (
        _build_buyer_auth_lookup,
        _build_complaint_lookup,
        _build_competitor_lookup,
        _build_contract_value_lookup,
        _build_department_lookup,
        _build_feature_gap_lookup,
        _build_keyword_spike_lookup,
        _build_pain_lookup,
        _build_positive_lookup,
        _build_sentiment_lookup,
        _build_tenure_lookup,
        _build_timeline_lookup,
        _build_turning_point_lookup,
        _build_use_case_lookup,
        _canonicalize_vendor,
        _merge_company_lookup_with_evidence_vault,
        _merge_feature_gap_lookup_with_evidence_vault,
        _merge_pain_lookup_with_evidence_vault,
    )

    raw_pain_lookup = _build_pain_lookup(pain_dist)
    raw_feature_gap_lookup = _build_feature_gap_lookup(feature_gaps)
    raw_company_lookup = {
        row["vendor"]: row["companies"]
        for row in churning_companies
        if isinstance(row, dict) and row.get("vendor")
    }
    product_profile_lookup: dict[str, dict[str, Any]] = {}
    for row in product_profiles_raw:
        if not isinstance(row, dict):
            continue
        vendor = _canonicalize_vendor(row.get("vendor_name") or "")
        if vendor and vendor not in product_profile_lookup:
            product_profile_lookup[vendor] = row

    complaints_raw, positives_raw = review_text_agg
    contract_values_raw, _durations_raw = contract_ctx
    return {
        "competitor_lookup": _build_competitor_lookup(competitive_disp),
        "pain_lookup": _merge_pain_lookup_with_evidence_vault(
            raw_pain_lookup,
            evidence_vault_lookup,
        ),
        "feature_gap_lookup": _merge_feature_gap_lookup_with_evidence_vault(
            raw_feature_gap_lookup,
            evidence_vault_lookup,
        ),
        "company_lookup": _merge_company_lookup_with_evidence_vault(
            raw_company_lookup,
            evidence_vault_lookup,
        ),
        "quote_lookup": {
            row["vendor"]: row["quotes"]
            for row in quotable_evidence
            if isinstance(row, dict) and row.get("vendor")
        },
        "budget_lookup": {
            row["vendor"]: {k: v for k, v in row.items() if k != "vendor"}
            for row in budget_signals
            if isinstance(row, dict) and row.get("vendor")
        },
        "use_case_lookup": _build_use_case_lookup(use_case_dist),
        "sentiment_lookup": _build_sentiment_lookup(sentiment_traj),
        "buyer_auth_lookup": _build_buyer_auth_lookup(buyer_auth),
        "timeline_lookup": _build_timeline_lookup(timeline_signals),
        "keyword_spike_lookup": _build_keyword_spike_lookup(keyword_spikes),
        "complaint_lookup": _build_complaint_lookup(complaints_raw),
        "positive_lookup": _build_positive_lookup(positives_raw),
        "department_lookup": _build_department_lookup(department_dist),
        "contract_value_lookup": _build_contract_value_lookup(contract_values_raw),
        "turning_point_lookup": _build_turning_point_lookup(turning_points),
        "tenure_lookup": _build_tenure_lookup(sentiment_tenure),
        "product_profile_lookup": product_profile_lookup,
        "price_lookup": {
            row["vendor"]: row["price_complaint_rate"]
            for row in price_rates
            if isinstance(row, dict) and row.get("vendor")
        },
        "dm_lookup": {
            row["vendor"]: row["dm_churn_rate"]
            for row in dm_rates
            if isinstance(row, dict) and row.get("vendor")
        },
    }


def _attach_context_to_deterministic_reports(
    *,
    pool,
    as_of: date,
    deterministic_weekly_feed: list[dict[str, Any]],
    deterministic_vendor_scorecards: list[dict[str, Any]],
    deterministic_displacement_map: list[dict[str, Any]],
    deterministic_category_overview: list[dict[str, Any]],
    deterministic_vendor_deep_dives: list[dict[str, Any]] | None = None,
    evidence_vault_lookup: dict[str, dict[str, Any]],
    synthesis_views: dict[str, Any],
    xv_lookup: dict[str, dict[str, Any]],
) -> int:
    attached_vendors: set[str] = set()

    for entry in deterministic_weekly_feed:
        vendor = entry.get("vendor", "")
        vault = (evidence_vault_lookup or {}).get(vendor, {})
        if isinstance(vault, dict) and vault.get("strength_evidence"):
            strengths: list[dict[str, Any]] = []
            for item in vault["strength_evidence"]:
                if not isinstance(item, dict):
                    continue
                area = str(item.get("label") or item.get("key") or "").strip()
                if not area:
                    continue
                strengths.append({
                    "area": area,
                    "mention_count": int(item.get("mention_count_total") or 0),
                })
            if strengths:
                strengths.sort(key=lambda row: -row["mention_count"])
                entry["retention_strengths"] = strengths[:3]
        view = synthesis_views.get(vendor)
        if view:
            _attach_synthesis_contracts_to_report_entry(
                entry,
                view,
                consumer_name="weekly_churn_feed",
                requested_as_of=as_of,
                include_displacement=False,
            )
            attached_vendors.add(vendor)
        category_name = str(entry.get("category") or "").strip()
        if category_name:
            council = xv_lookup.get("councils", {}).get(category_name)
            council_context = _build_category_council_context(council)
            if council_context:
                entry["category_council"] = council_context

    feed_category_lookup = {
        str(entry.get("vendor") or "").strip(): str(entry.get("category") or "").strip()
        for entry in deterministic_weekly_feed
        if str(entry.get("vendor") or "").strip()
    }
    for scorecard in deterministic_vendor_scorecards:
        vendor = scorecard.get("vendor", "")
        view = synthesis_views.get(vendor)
        if view:
            _attach_synthesis_contracts_to_report_entry(
                scorecard,
                view,
                consumer_name="vendor_scorecard",
                requested_as_of=as_of,
                include_displacement=True,
            )
            attached_vendors.add(vendor)
            # Synthesis narrative replaces legacy reasoning_summary to prevent
            # dual narratives.  Build from causal_narrative fields.
            causal = view.section("causal_narrative")
            if isinstance(causal, dict):
                summary = causal.get("summary") or causal.get("executive_summary") or ""
                if not summary:
                    trigger = causal.get("trigger", "")
                    why_now = causal.get("why_now", "")
                    summary = ". ".join(bit for bit in (trigger, why_now) if bit)
                if summary:
                    scorecard["reasoning_summary"] = summary
                    scorecard["reasoning_source"] = "b2b_reasoning_synthesis"
            # Override legacy archetype with synthesis wedge
            if view.primary_wedge:
                scorecard["archetype"] = view.primary_wedge.value
        category_name = str(feed_category_lookup.get(vendor) or scorecard.get("category") or "").strip()
        if category_name:
            scorecard.setdefault("category", category_name)
            council = xv_lookup.get("councils", {}).get(category_name)
            council_context = _build_category_council_context(council)
            if council_context:
                scorecard["category_council"] = council_context
        comparisons = []
        for pair_key, asym in xv_lookup.get("asymmetries", {}).items():
            if vendor in pair_key:
                ac = asym.get("conclusion", {})
                opponent = _pair_opponent(pair_key, vendor)
                if opponent:
                    comparisons.append({
                        "opponent": opponent,
                        "conclusion": ac.get("conclusion", ""),
                        "confidence": asym.get("confidence", 0),
                        "resource_advantage": ac.get("resource_advantage", ""),
                    })
        if comparisons:
            scorecard["cross_vendor_comparisons"] = comparisons

    for dd_entry in (deterministic_vendor_deep_dives or []):
        vendor = dd_entry.get("vendor", "")
        vault = (evidence_vault_lookup or {}).get(vendor, {})
        if isinstance(vault, dict) and vault.get("strength_evidence"):
            dd_strengths: list[dict[str, Any]] = []
            for item in vault["strength_evidence"]:
                if not isinstance(item, dict):
                    continue
                area = str(item.get("label") or item.get("key") or "").strip()
                if not area:
                    continue
                dd_strengths.append({
                    "area": area,
                    "mention_count": int(item.get("mention_count_total") or 0),
                })
            if dd_strengths:
                dd_strengths.sort(key=lambda row: -row["mention_count"])
                dd_entry["retention_strengths"] = dd_strengths[:5]
        dd_category = str(dd_entry.get("category") or "").strip()
        if dd_category:
            council = xv_lookup.get("councils", {}).get(dd_category)
            council_context = _build_category_council_context(council)
            if council_context:
                dd_entry["category_council"] = council_context
        view = synthesis_views.get(vendor)
        if view:
            _attach_synthesis_contracts_to_report_entry(
                dd_entry,
                view,
                consumer_name="vendor_deep_dive",
                requested_as_of=as_of,
                include_displacement=False,
            )

    for cat_entry in deterministic_category_overview:
        category_name = str(cat_entry.get("category") or "").strip()
        if not category_name:
            continue
        council = xv_lookup.get("councils", {}).get(category_name)
        council_context = _build_category_council_context(council)
        if not council_context:
            continue
        cat_entry["category_council"] = council_context
        cat_entry["cross_vendor_analysis"] = dict(council_context)
        cat_entry["reasoning_source"] = "b2b_cross_vendor_reasoning_synthesis"
        computed_date = council.get("computed_date") if isinstance(council, dict) else None
        if computed_date is not None and hasattr(computed_date, "isoformat"):
            cat_entry["data_as_of_date"] = computed_date.isoformat()
        reference_ids = _copy_reference_ids(
            council.get("reference_ids") if isinstance(council, dict) else None,
        )
        if reference_ids:
            cat_entry["reference_ids"] = reference_ids

    for edge in deterministic_displacement_map:
        pair = tuple(sorted([edge["from_vendor"], edge["to_vendor"]]))
        battle = xv_lookup.get("battles", {}).get(pair)
        if battle:
            bc = battle.get("conclusion", {})
            # Only attach when from_vendor is the battle loser: the conclusion prose
            # describes why the loser's customers are flowing to the winner, so it
            # is only coherent on the directed edge where the loser is the source.
            battle_loser = (bc.get("loser") or "").strip().lower()
            if not battle_loser or edge["from_vendor"].strip().lower() == battle_loser:
                edge["battle_conclusion"] = bc.get("conclusion", "")
                edge["durability"] = bc.get("durability_assessment", "")
                reference_ids = _copy_reference_ids(
                    battle.get("reference_ids") if isinstance(battle, dict) else None,
                )
                if reference_ids:
                    edge["reference_ids"] = reference_ids
                computed_date = battle.get("computed_date") if isinstance(battle, dict) else None
                if computed_date is not None and hasattr(computed_date, "isoformat"):
                    edge["data_as_of_date"] = computed_date.isoformat()
                battle_source = str(battle.get("source") or "").strip() if isinstance(battle, dict) else ""
                if battle_source:
                    edge["reasoning_source"] = (
                        "b2b_cross_vendor_reasoning_synthesis"
                        if battle_source == "synthesis"
                        else battle_source
                    )

    return len(attached_vendors)


async def _build_deterministic_report_bundle(
    pool,
    *,
    as_of: date,
    analysis_window_days: int,
    vendor_scores: list[dict[str, Any]],
    competitive_disp: list[dict[str, Any]],
    pain_dist: list[dict[str, Any]],
    feature_gaps: list[dict[str, Any]],
    price_rates: list[dict[str, Any]],
    dm_rates: list[dict[str, Any]],
    churning_companies: list[dict[str, Any]],
    quotable_evidence: list[dict[str, Any]],
    budget_signals: list[dict[str, Any]],
    use_case_dist: list[dict[str, Any]],
    sentiment_traj: list[dict[str, Any]],
    sentiment_tenure: list[dict[str, Any]],
    buyer_auth: list[dict[str, Any]],
    timeline_signals: list[dict[str, Any]],
    competitor_reasons: list[dict[str, Any]],
    keyword_spikes: list[dict[str, Any]],
    product_profiles_raw: list[dict[str, Any]],
    prior_reports: list[dict[str, Any]],
    displacement_provenance: dict[tuple[str, str], dict[str, Any]],
    review_text_agg: tuple[list[dict[str, Any]], list[dict[str, Any]]],
    department_dist: list[dict[str, Any]],
    contract_ctx: tuple[list[dict[str, Any]], list[dict[str, Any]]],
    turning_points: list[dict[str, Any]],
    vendor_scorecard_limit: int | None = 15,
    synthesis_views: dict[str, Any] | None = None,
    reasoning_lookup: dict[str, dict[str, Any]] | None = None,
    xv_lookup: dict[str, dict[str, Any]] | None = None,
    evidence_vault_lookup: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from ._b2b_shared import (
        _build_deterministic_category_overview,
        _build_deterministic_displacement_map,
        _build_deterministic_vendor_feed,
        _build_deterministic_vendor_scorecards,
        _build_vendor_deep_dives,
        _align_vendor_intelligence_records_to_scorecards,
        _compute_evidence_confidence,
        read_vendor_intelligence_records,
        _structure_displacement_report,
    )
    from ._b2b_cross_vendor_synthesis import load_best_cross_vendor_lookup
    from ._b2b_synthesis_reader import build_reasoning_lookup_from_views

    if xv_lookup is None:
        xv_lookup = await load_best_cross_vendor_lookup(
            pool,
            as_of=as_of,
            analysis_window_days=analysis_window_days,
        )
    if synthesis_views is None:
        synthesis_views = await _fetch_latest_synthesis_views(
            pool,
            as_of=as_of,
            analysis_window_days=analysis_window_days,
        )
    if reasoning_lookup is None:
        reasoning_lookup = build_reasoning_lookup_from_views(synthesis_views)
    if evidence_vault_lookup is None:
        if pool is None:
            evidence_vault_lookup = {}
        else:
            vendor_names = sorted(
                {
                    str(row.get("vendor_name") or row.get("vendor") or "").strip()
                    for row in vendor_scores
                    if str(row.get("vendor_name") or row.get("vendor") or "").strip()
                }
            )
            evidence_vault_records = await read_vendor_intelligence_records(
                pool,
                as_of=as_of,
                analysis_window_days=analysis_window_days,
                vendor_names=vendor_names or None,
            )
            evidence_vault_lookup, vault_alignment = (
                _align_vendor_intelligence_records_to_scorecards(
                    vendor_scores,
                    evidence_vault_records,
                )
            )
            if vault_alignment["mismatched_vendor_count"]:
                logger.info(
                    "Churn reports suppressed %d mismatched evidence-vault overlays: %s",
                    vault_alignment["mismatched_vendor_count"],
                    ", ".join(vault_alignment["mismatched_vendors"][:10]),
                )

    lookups = _build_report_lookup_bundle(
        competitive_disp=competitive_disp,
        pain_dist=pain_dist,
        feature_gaps=feature_gaps,
        price_rates=price_rates,
        dm_rates=dm_rates,
        churning_companies=churning_companies,
        quotable_evidence=quotable_evidence,
        budget_signals=budget_signals,
        use_case_dist=use_case_dist,
        sentiment_traj=sentiment_traj,
        buyer_auth=buyer_auth,
        timeline_signals=timeline_signals,
        keyword_spikes=keyword_spikes,
        product_profiles_raw=product_profiles_raw,
        review_text_agg=review_text_agg,
        department_dist=department_dist,
        contract_ctx=contract_ctx,
        turning_points=turning_points,
        sentiment_tenure=sentiment_tenure,
        evidence_vault_lookup=evidence_vault_lookup,
    )

    deterministic_weekly_feed = _build_deterministic_vendor_feed(
        vendor_scores,
        pain_lookup=lookups["pain_lookup"],
        competitor_lookup=lookups["competitor_lookup"],
        feature_gap_lookup=lookups["feature_gap_lookup"],
        quote_lookup=lookups["quote_lookup"],
        budget_lookup=lookups["budget_lookup"],
        sentiment_lookup=lookups["sentiment_lookup"],
        buyer_auth_lookup=lookups["buyer_auth_lookup"],
        dm_lookup=lookups["dm_lookup"],
        price_lookup=lookups["price_lookup"],
        company_lookup=lookups["company_lookup"],
        keyword_spike_lookup=lookups["keyword_spike_lookup"],
        prior_reports=prior_reports,
        synthesis_views=synthesis_views,
        reasoning_lookup=reasoning_lookup,
    )
    inbound_displacement_lookup = _build_inbound_displacement_lookup(competitive_disp)

    deterministic_vendor_scorecards = _build_deterministic_vendor_scorecards(
        vendor_scores,
        pain_lookup=lookups["pain_lookup"],
        competitor_lookup=lookups["competitor_lookup"],
        feature_gap_lookup=lookups["feature_gap_lookup"],
        quote_lookup=lookups["quote_lookup"],
        budget_lookup=lookups["budget_lookup"],
        sentiment_lookup=lookups["sentiment_lookup"],
        buyer_auth_lookup=lookups["buyer_auth_lookup"],
        dm_lookup=lookups["dm_lookup"],
        price_lookup=lookups["price_lookup"],
        company_lookup=lookups["company_lookup"],
        product_profile_lookup=lookups["product_profile_lookup"],
        prior_reports=prior_reports,
        inbound_displacement_lookup=inbound_displacement_lookup,
        synthesis_views=synthesis_views,
        reasoning_lookup=reasoning_lookup,
        timeline_lookup=lookups["timeline_lookup"],
        use_case_lookup=lookups["use_case_lookup"],
        complaint_lookup=lookups["complaint_lookup"],
        positive_lookup=lookups["positive_lookup"],
        department_lookup=lookups["department_lookup"],
        contract_value_lookup=lookups["contract_value_lookup"],
        turning_point_lookup=lookups["turning_point_lookup"],
        tenure_lookup=lookups["tenure_lookup"],
        evidence_vault_lookup=evidence_vault_lookup,
        limit=vendor_scorecard_limit,
    )
    deterministic_displacement_map = _build_deterministic_displacement_map(
        competitive_disp,
        competitor_reasons,
        lookups["quote_lookup"],
        synthesis_views=synthesis_views,
        reasoning_lookup=reasoning_lookup,
    )
    for edge in deterministic_displacement_map:
        prov = displacement_provenance.get((edge["from_vendor"], edge["to_vendor"]), {})
        src_dist = prov.get("source_distribution", {})
        edge["source_distribution"] = src_dist
        edge["sample_review_ids"] = prov.get("sample_review_ids", [])
        edge["confidence_score"] = _compute_evidence_confidence(
            edge["mention_count"], src_dist,
        )

    deterministic_category_overview = _build_deterministic_category_overview(
        vendor_scores,
        pain_lookup=lookups["pain_lookup"],
        competitive_disp=competitive_disp,
        company_lookup=lookups["company_lookup"],
        quote_lookup=lookups["quote_lookup"],
        feature_gap_lookup=lookups["feature_gap_lookup"],
        dm_lookup=lookups["dm_lookup"],
        price_lookup=lookups["price_lookup"],
        competitor_lookup=lookups["competitor_lookup"],
        synthesis_views=synthesis_views,
        reasoning_lookup=reasoning_lookup,
    )
    vendor_deep_dives = _build_vendor_deep_dives(
        vendor_scores,
        pain_lookup=lookups["pain_lookup"],
        competitor_lookup=lookups["competitor_lookup"],
        feature_gap_lookup=lookups["feature_gap_lookup"],
        quote_lookup=lookups["quote_lookup"],
        company_lookup=lookups["company_lookup"],
        dm_lookup=lookups["dm_lookup"],
        price_lookup=lookups["price_lookup"],
        sentiment_lookup=lookups["sentiment_lookup"],
        buyer_auth_lookup=lookups.get("buyer_auth_lookup"),
        synthesis_views=synthesis_views,
        reasoning_lookup=reasoning_lookup,
    )
    attached_contract_vendors = _attach_context_to_deterministic_reports(
        pool=pool,
        as_of=as_of,
        deterministic_weekly_feed=deterministic_weekly_feed,
        deterministic_vendor_scorecards=deterministic_vendor_scorecards,
        deterministic_displacement_map=deterministic_displacement_map,
        deterministic_category_overview=deterministic_category_overview,
        deterministic_vendor_deep_dives=vendor_deep_dives,
        evidence_vault_lookup=evidence_vault_lookup,
        synthesis_views=synthesis_views,
        xv_lookup=xv_lookup,
    )
    try:
        from atlas_brain.reasoning.ecosystem import EcosystemAnalyzer

        eco_analyzer = EcosystemAnalyzer(pool)
        ecosystem_evidence = await eco_analyzer.analyze_all_categories(reasoning_lookup)
        for cat_entry in deterministic_category_overview:
            cat_name = cat_entry.get("category", "")
            eco = ecosystem_evidence.get(cat_name)
            if eco:
                cat_entry["ecosystem"] = {
                    "hhi": eco.get("hhi"),
                    "market_structure": eco.get("market_structure"),
                    "displacement_intensity": eco.get("displacement_intensity"),
                    "dominant_archetype": eco.get("dominant_archetype"),
                }
    except Exception:
        logger.debug("Ecosystem analysis skipped", exc_info=True)
    for cat_entry in deterministic_category_overview:
        cat_name = cat_entry.get("category", "")
        council = xv_lookup.get("councils", {}).get(cat_name)
        if council:
            conclusion = council.get("conclusion", {})
            cat_entry["market_regime"] = conclusion.get("market_regime", "")
            cat_entry["market_conclusion"] = conclusion.get("conclusion", "")
            cat_entry["market_winner"] = conclusion.get("winner", "")
            cat_entry["market_loser"] = conclusion.get("loser", "")
            cat_entry["market_durability"] = conclusion.get("durability_assessment", "")
            cat_entry["market_confidence"] = council.get("confidence", 0)
            cat_entry["market_insights"] = conclusion.get("key_insights", [])
    return {
        "weekly_churn_feed": deterministic_weekly_feed,
        "vendor_scorecards": deterministic_vendor_scorecards,
        "displacement_map": deterministic_displacement_map,
        "category_insights": deterministic_category_overview,
        "vendor_deep_dives": vendor_deep_dives,
        "reasoning_lookup": reasoning_lookup,
        "xv_lookup": xv_lookup,
        "synthesis_views": synthesis_views,
        "evidence_vault_lookup": evidence_vault_lookup,
        "attached_contract_vendors": attached_contract_vendors,
    }


async def _check_freshness(pool) -> date | None:
    """Return today's date if the core run completed canonically, else None."""
    from ._b2b_shared import has_complete_core_run_marker

    today = date.today()
    if not await has_complete_core_run_marker(pool, today):
        logger.info("Core run not complete for %s, skipping", today)
        return None
    return today


def _scoped_report_vendor_filter(vendors: list[str] | None) -> str | None:
    """Return a stable vendor_filter token for scoped multi-vendor report rows."""
    if not vendors:
        return None
    deduped: dict[str, str] = {}
    for vendor in vendors:
        text = str(vendor or "").strip()
        if not text:
            continue
        deduped.setdefault(text.lower(), text)
    if not deduped:
        return None
    return ",".join(deduped[key] for key in sorted(deduped))


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Build deterministic reports + LLM enrichment from persisted artifacts."""
    cfg = settings.b2b_churn
    if not cfg.enabled or not cfg.intelligence_enabled:
        return {"_skip_synthesis": "B2B churn intelligence disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    today = await _check_freshness(pool)
    if today is None:
        from ._b2b_shared import describe_core_run_gap

        return {
            "_skip_synthesis": (
                await describe_core_run_gap(pool, date.today())
                or "Core signals not fresh for today"
            )
        }

    from ._b2b_shared import (
        _aggregate_competitive_disp,
        _build_deterministic_category_overview,
        _build_deterministic_displacement_map,
        _build_deterministic_vendor_feed,
        _build_deterministic_vendor_scorecards,
        _build_scorecard_locked_facts,
        _build_vendor_deep_dives,
        _structure_displacement_report,
        _build_pain_lookup,
        _build_competitor_lookup,
        _build_feature_gap_lookup,
        _build_use_case_lookup,
        _build_integration_lookup,
        _build_sentiment_lookup,
        _build_buyer_auth_lookup,
        _build_timeline_lookup,
        _build_keyword_spike_lookup,
        _build_complaint_lookup,
        _build_positive_lookup,
        _build_department_lookup,
        _build_contract_value_lookup,
        _build_turning_point_lookup,
        _build_tenure_lookup,
        _build_validated_executive_summary,
        _canonicalize_vendor,
        _compute_evidence_confidence,
        _executive_source_list,
        _fallback_scorecard_expert_take,
        _fetch_competitive_displacement_source_of_truth,
        _fetch_competitor_reasons,
        _fetch_data_context,
        read_vendor_scorecards,
        _fetch_pain_distribution,
        _fetch_feature_gaps,
        _fetch_negative_review_counts,
        _fetch_price_complaint_rates,
        _fetch_dm_churn_rates,
        _fetch_churning_companies,
        _fetch_quotable_evidence,
        _fetch_budget_signals,
        _fetch_use_case_distribution,
        _fetch_sentiment_trajectory,
        _fetch_sentiment_tenure,
        _fetch_buyer_authority_summary,
        _fetch_timeline_signals,
        _fetch_keyword_spikes,
        _fetch_product_profiles,
        _fetch_prior_reports,
        _fetch_review_text_aggregates,
        _fetch_department_distribution,
        _fetch_contract_context_distribution,
        _fetch_turning_points,
        _fetch_displacement_provenance,
        _merge_company_lookup_with_evidence_vault,
        _merge_feature_gap_lookup_with_evidence_vault,
        _merge_pain_lookup_with_evidence_vault,
        _validate_scorecard_expert_take,
    )
    from .b2b_churn_intelligence import (
        _apply_vendor_scope_to_churn_inputs,
        _normalize_test_vendors,
    )

    window_days = cfg.intelligence_window_days
    min_reviews = cfg.intelligence_min_reviews
    task_metadata = task.metadata or {}
    scoped_vendors = _normalize_test_vendors(task_metadata.get("test_vendors"))
    selected_report_type = str(task_metadata.get("test_report_type") or "").strip()
    persist_scoped_reports = bool(task_metadata.get("persist_scoped_reports"))
    valid_report_types = {
        "weekly_churn_feed",
        "vendor_scorecard",
        "displacement_report",
        "category_overview",
        "vendor_deep_dive",
    }
    if selected_report_type and selected_report_type not in valid_report_types:
        logger.warning(
            "Ignoring unsupported test_report_type=%s for churn reports",
            selected_report_type,
        )
        selected_report_type = ""
    selected_report_types = {selected_report_type} if selected_report_type else set()

    # --- Phase 1: Parallel data fetch ---
    try:
        (
            vendor_scores,
            competitive_disp, pain_dist, feature_gaps,
            negative_counts, price_rates, dm_rates,
            churning_companies, quotable_evidence,
            budget_signals, use_case_dist,
            sentiment_traj, sentiment_tenure, buyer_auth, timeline_signals,
            competitor_reasons, keyword_spikes,
            data_context, product_profiles_raw,
            prior_reports,
            displacement_provenance,
            review_text_agg, department_dist,
            contract_ctx, turning_points,
        ) = await asyncio.gather(
            read_vendor_scorecards(pool, window_days=window_days, min_reviews=min_reviews),
            _fetch_competitive_displacement_source_of_truth(
                pool,
                as_of=today,
                analysis_window_days=window_days,
            ),
            _fetch_pain_distribution(pool, window_days),
            _fetch_feature_gaps(pool, window_days, min_mentions=cfg.feature_gap_min_mentions),
            _fetch_negative_review_counts(pool, window_days),
            _fetch_price_complaint_rates(pool, window_days),
            _fetch_dm_churn_rates(pool, window_days),
            _fetch_churning_companies(pool, window_days),
            _fetch_quotable_evidence(pool, window_days, min_urgency=cfg.quotable_phrase_min_urgency),
            _fetch_budget_signals(pool, window_days),
            _fetch_use_case_distribution(pool, window_days),
            _fetch_sentiment_trajectory(pool, window_days),
            _fetch_sentiment_tenure(pool, window_days),
            _fetch_buyer_authority_summary(pool, window_days),
            _fetch_timeline_signals(pool, window_days),
            _fetch_competitor_reasons(pool, window_days),
            _fetch_keyword_spikes(pool),
            _fetch_data_context(pool, window_days),
            _fetch_product_profiles(pool),
            _fetch_prior_reports(pool),
            _fetch_displacement_provenance(pool, window_days),
            _fetch_review_text_aggregates(pool, window_days),
            _fetch_department_distribution(pool, window_days),
            _fetch_contract_context_distribution(pool, window_days),
            _fetch_turning_points(pool, window_days),
        )
    except Exception:
        logger.exception("Report data fetch failed")
        return {"_skip_synthesis": "Data fetch failed"}

    if not vendor_scores:
        return {"_skip_synthesis": "No vendor scores"}
    competitive_disp = _aggregate_competitive_disp(competitive_disp)

    if scoped_vendors:
        raw_vendor_count = len(vendor_scores)
        scoped_data, scoped_vendors = _apply_vendor_scope_to_churn_inputs(
            {
                "vendor_scores": vendor_scores,
                "competitive_disp": competitive_disp,
                "pain_dist": pain_dist,
                "feature_gaps": feature_gaps,
                "price_rates": price_rates,
                "dm_rates": dm_rates,
                "churning_companies": churning_companies,
                "quotable_evidence": quotable_evidence,
                "budget_signals": budget_signals,
                "use_case_dist": use_case_dist,
                "sentiment_traj": sentiment_traj,
                "buyer_auth": buyer_auth,
                "timeline_signals": timeline_signals,
                "competitor_reasons": competitor_reasons,
                "keyword_spikes": keyword_spikes,
                "product_profiles_raw": product_profiles_raw,
                "displacement_provenance": displacement_provenance,
                "review_text_aggs": review_text_agg,
                "department_dist": department_dist,
                "contract_ctx_aggs": contract_ctx,
            },
            scoped_vendors,
        )
        vendor_scores = scoped_data["vendor_scores"]
        competitive_disp = scoped_data["competitive_disp"]
        pain_dist = scoped_data["pain_dist"]
        feature_gaps = scoped_data["feature_gaps"]
        price_rates = scoped_data["price_rates"]
        dm_rates = scoped_data["dm_rates"]
        churning_companies = scoped_data["churning_companies"]
        quotable_evidence = scoped_data["quotable_evidence"]
        budget_signals = scoped_data["budget_signals"]
        use_case_dist = scoped_data["use_case_dist"]
        sentiment_traj = scoped_data["sentiment_traj"]
        buyer_auth = scoped_data["buyer_auth"]
        timeline_signals = scoped_data["timeline_signals"]
        competitor_reasons = scoped_data["competitor_reasons"]
        keyword_spikes = scoped_data["keyword_spikes"]
        product_profiles_raw = scoped_data["product_profiles_raw"]
        displacement_provenance = scoped_data["displacement_provenance"]
        review_text_agg = scoped_data["review_text_aggs"]
        department_dist = scoped_data["department_dist"]
        contract_ctx = scoped_data["contract_ctx_aggs"]
        logger.info(
            "Scoped churn reports to %d/%d vendors for test run: %s",
            len(vendor_scores),
            raw_vendor_count,
            sorted(scoped_vendors),
        )

    if not vendor_scores:
        return {"_skip_synthesis": "No vendor scores after vendor scope filter"}

    vendor_scorecard_limit: int | None = int(cfg.vendor_scorecard_limit)
    if scoped_vendors and (
        not selected_report_types or "vendor_scorecard" in selected_report_types
    ):
        vendor_scorecard_limit = None

    bundle = await _build_deterministic_report_bundle(
        pool,
        as_of=today,
        analysis_window_days=window_days,
        vendor_scores=vendor_scores,
        competitive_disp=competitive_disp,
        pain_dist=pain_dist,
        feature_gaps=feature_gaps,
        price_rates=price_rates,
        dm_rates=dm_rates,
        churning_companies=churning_companies,
        quotable_evidence=quotable_evidence,
        budget_signals=budget_signals,
        use_case_dist=use_case_dist,
        sentiment_traj=sentiment_traj,
        sentiment_tenure=sentiment_tenure,
        buyer_auth=buyer_auth,
        timeline_signals=timeline_signals,
        competitor_reasons=competitor_reasons,
        keyword_spikes=keyword_spikes,
        product_profiles_raw=product_profiles_raw,
        prior_reports=prior_reports,
        displacement_provenance=displacement_provenance,
        review_text_agg=review_text_agg,
        department_dist=department_dist,
        contract_ctx=contract_ctx,
        turning_points=turning_points,
        vendor_scorecard_limit=vendor_scorecard_limit,
    )
    reasoning_lookup = bundle["reasoning_lookup"]
    xv_lookup = bundle["xv_lookup"]
    deterministic_weekly_feed = bundle["weekly_churn_feed"]
    deterministic_vendor_scorecards = bundle["vendor_scorecards"]
    deterministic_displacement_map = bundle["displacement_map"]
    deterministic_category_overview = bundle["category_insights"]
    vendor_deep_dives = bundle["vendor_deep_dives"]
    if scoped_vendors:
        vendor_scope = {vendor.lower() for vendor in scoped_vendors}
        reasoning_lookup = {
            vendor: payload
            for vendor, payload in reasoning_lookup.items()
            if str(vendor or "").strip().lower() in vendor_scope
        }
    synth_count = sum(
        1 for v in reasoning_lookup.values()
        if v.get("mode") == "synthesis"
    )
    logger.info(
        "Reasoning for %d vendors (%d synthesis-first, %d legacy fallback)",
        len(reasoning_lookup), synth_count, len(reasoning_lookup) - synth_count,
    )
    logger.info(
        "Cross-vendor enrichment: %d battles, %d councils, %d asymmetries",
        len(xv_lookup.get("battles", {})),
        len(xv_lookup.get("councils", {})),
        len(xv_lookup.get("asymmetries", {})),
    )

    # --- Phase 5: Scorecard narrative LLM enrichment ---
    from ...pipelines.llm import (
        call_llm_with_skill,
        get_pipeline_llm,
        parse_json_response,
    )
    from ...services.b2b.anthropic_batch import (
        AnthropicBatchItem,
        mark_batch_fallback_result,
        run_anthropic_message_batch,
    )
    from ...services.b2b.cache_runner import (
        lookup_b2b_exact_stage_text,
        prepare_b2b_exact_skill_stage_request,
        store_b2b_exact_stage_text,
    )
    from ...services.b2b.llm_exact_cache import CacheUnavailable, llm_identity
    from ...services.protocols import Message
    from ._b2b_batch_utils import (
        anthropic_batch_min_items,
        anthropic_batch_requested,
        is_anthropic_llm,
        resolve_anthropic_batch_llm,
    )

    _llm_workload = "synthesis"
    _llm_max_tokens = _scorecard_narrative_max_tokens()
    _cache_stage_id = "b2b_churn_reports.scorecard_narrative"
    _resolved_llm = get_pipeline_llm(workload=_llm_workload)
    _provider, _model = llm_identity(_resolved_llm)
    _batch_requested = anthropic_batch_requested(
        task,
        global_default=bool(getattr(settings.b2b_churn, "anthropic_batch_enabled", False)),
        task_default=bool(getattr(settings.b2b_churn, "scorecard_anthropic_batch_enabled", True)),
        task_keys=("scorecard_anthropic_batch_enabled",),
    )
    _batch_llm = (
        resolve_anthropic_batch_llm(
            current_llm=_resolved_llm,
            target_model_candidates=(getattr(settings.llm, "openrouter_reasoning_model", ""),),
        )
        if _batch_requested
        else None
    )
    _batch_enabled = is_anthropic_llm(_batch_llm)
    _batch_provider, _batch_model = llm_identity(_batch_llm) if _batch_enabled else ("", "")
    scorecard_llm_failures = 0
    scorecard_cache_hits = 0
    scorecard_reasoning_reused = 0
    scorecard_guardrail_fallbacks = 0
    scorecard_batch_metrics = {
        "jobs": 0,
        "submitted_items": 0,
        "cache_prefiltered_items": 0,
        "fallback_single_call_items": 0,
        "completed_items": 0,
        "failed_items": 0,
    }
    should_generate_scorecard_narratives = (
        not selected_report_types or "vendor_scorecard" in selected_report_types
    )
    if should_generate_scorecard_narratives:
        from ._b2b_shared import _normalize_scorecard_expert_take

        async def _store_scorecard_exact_cache(
            request: Any | None,
            sc: dict[str, Any],
            parsed_narrative: dict[str, Any],
        ) -> None:
            if request is None:
                return
            await store_b2b_exact_stage_text(
                request,
                response_text=json.dumps(parsed_narrative, separators=(",", ":")),
                metadata={
                    "task": "b2b_churn_reports",
                    "vendor_name": sc.get("vendor_name") or sc.get("vendor"),
                    "cache_stage": "scorecard_narrative",
                },
            )

        async def _generate_scorecard_narrative_direct(
            sc: dict[str, Any],
            llm_input: dict[str, Any],
            *,
            request: Any | None,
            workload: str,
            usage_out: dict[str, Any] | None = None,
        ) -> str:
            try:
                llm_usage: dict[str, Any] = {}
                narrative = await asyncio.wait_for(
                    asyncio.to_thread(
                        call_llm_with_skill,
                        "digest/vendor_deep_dive_narrative",
                        json.dumps(llm_input, default=str),
                        max_tokens=_llm_max_tokens,
                        temperature=0.3,
                        response_format={"type": "json_object"},
                        workload=workload,
                        trace_metadata={
                            "vendor_name": sc.get("vendor_name") or sc.get("vendor"),
                            "report_type": "vendor_scorecard",
                        },
                        usage_out=llm_usage,
                    ),
                    timeout=45,
                )
                if usage_out is not None:
                    usage_out.clear()
                    usage_out.update(llm_usage)
                if not narrative:
                    sc["expert_take"] = _fallback_scorecard_expert_take(sc)
                    return "failed"
                parsed_narrative = parse_json_response(narrative)
                expert_take = _normalize_scorecard_expert_take(
                    parsed_narrative.get("expert_take", "")
                )
                narrative_errors = _validate_scorecard_expert_take(sc, expert_take)
                if narrative_errors:
                    sc["expert_take"] = _fallback_scorecard_expert_take(sc)
                    return "fallback"
                sc["expert_take"] = expert_take
                parsed_narrative["expert_take"] = expert_take
                await _store_scorecard_exact_cache(request, sc, parsed_narrative)
                return "generated"
            except Exception:
                sc["expert_take"] = _fallback_scorecard_expert_take(sc)
                return "failed"

        if _batch_enabled:
            batch_entries: list[dict[str, Any]] = []
            for index, sc in enumerate(deterministic_vendor_scorecards):
                reasoning_summary = sc.get("reasoning_summary", "")
                if reasoning_summary and not sc.get("cross_vendor_comparisons"):
                    sc["expert_take"] = reasoning_summary
                    scorecard_reasoning_reused += 1
                    continue

                llm_input = _build_scorecard_narrative_payload(
                    sc,
                    reasoning_lookup=reasoning_lookup,
                )
                request: Any | None = None
                request_messages: list[dict[str, str]] = []
                if _batch_provider and _batch_model:
                    try:
                        request, request_messages = prepare_b2b_exact_skill_stage_request(
                            _cache_stage_id,
                            skill_name="digest/vendor_deep_dive_narrative",
                            payload=json.dumps(llm_input, default=str),
                            provider=_batch_provider,
                            model=_batch_model,
                            max_tokens=_llm_max_tokens,
                            temperature=0.3,
                            response_format={"type": "json_object"},
                        )
                    except CacheUnavailable:
                        request = None
                        request_messages = []

                if request is not None:
                    cached = await lookup_b2b_exact_stage_text(request)
                    if cached is not None:
                        parsed_narrative = parse_json_response(cached["response_text"])
                        expert_take = _normalize_scorecard_expert_take(
                            parsed_narrative.get("expert_take", "")
                        )
                        narrative_errors = _validate_scorecard_expert_take(sc, expert_take)
                        if not narrative_errors:
                            sc["expert_take"] = expert_take
                            scorecard_cache_hits += 1
                            continue

                batch_entries.append(
                    {
                        "custom_id": f"scorecard:{index}:{str(sc.get('vendor_name') or sc.get('vendor') or '').strip().lower()}",
                        "artifact_id": str(sc.get("vendor_name") or sc.get("vendor") or f"scorecard-{index}"),
                        "scorecard": sc,
                        "llm_input": llm_input,
                        "request": request,
                        "request_messages": request_messages,
                    }
                )

            if batch_entries:
                execution = await run_anthropic_message_batch(
                    llm=_batch_llm,
                    stage_id=_cache_stage_id,
                    task_name="b2b_churn_reports",
                    items=[
                        AnthropicBatchItem(
                            custom_id=entry["custom_id"],
                            artifact_type="scorecard_narrative",
                            artifact_id=entry["artifact_id"],
                            vendor_name=entry["scorecard"].get("vendor_name") or entry["scorecard"].get("vendor"),
                            messages=[
                                Message(
                                    role=str(message.get("role") or ""),
                                    content=str(message.get("content") or ""),
                                )
                                for message in entry["request_messages"]
                            ],
                            max_tokens=_llm_max_tokens,
                            temperature=0.3,
                            trace_span_name="b2b.churn_intelligence.scorecard_narrative",
                            trace_metadata={
                                "vendor_name": entry["scorecard"].get("vendor_name") or entry["scorecard"].get("vendor"),
                                "report_type": "vendor_scorecard",
                                "workload": "anthropic_batch",
                            },
                            request_metadata={
                                "report_type": "vendor_scorecard",
                            },
                        )
                        for entry in batch_entries
                    ],
                    run_id=str(task.id),
                    min_batch_size=anthropic_batch_min_items(
                        task,
                        default=int(getattr(settings.b2b_churn, "scorecard_anthropic_batch_min_items", 2)),
                        keys=("scorecard_anthropic_batch_min_items",),
                    ),
                    batch_metadata={
                        "report_type": "vendor_scorecard",
                    },
                )
                scorecard_batch_metrics["jobs"] += 1 if execution.provider_batch_id else 0
                scorecard_batch_metrics["submitted_items"] += execution.submitted_items
                scorecard_batch_metrics["cache_prefiltered_items"] += execution.cache_prefiltered_items
                scorecard_batch_metrics["fallback_single_call_items"] += execution.fallback_single_call_items
                scorecard_batch_metrics["completed_items"] += execution.completed_items
                scorecard_batch_metrics["failed_items"] += execution.failed_items

                for entry in batch_entries:
                    sc = entry["scorecard"]
                    request = entry["request"]
                    outcome = execution.results_by_custom_id.get(entry["custom_id"])
                    if outcome is not None and outcome.response_text:
                        parsed_narrative = parse_json_response(outcome.response_text)
                        expert_take = _normalize_scorecard_expert_take(
                            parsed_narrative.get("expert_take", "")
                        )
                        narrative_errors = _validate_scorecard_expert_take(sc, expert_take)
                        if narrative_errors:
                            sc["expert_take"] = _fallback_scorecard_expert_take(sc)
                            scorecard_guardrail_fallbacks += 1
                            continue
                        sc["expert_take"] = expert_take
                        parsed_narrative["expert_take"] = expert_take
                        await _store_scorecard_exact_cache(request, sc, parsed_narrative)
                        continue

                    fallback_status = await _generate_scorecard_narrative_direct(
                        sc,
                        entry["llm_input"],
                        request=request,
                        workload=_llm_workload,
                        usage_out=(fallback_usage := {}),
                    )
                    await mark_batch_fallback_result(
                        batch_id=execution.local_batch_id,
                        custom_id=entry["custom_id"],
                        succeeded=fallback_status != "failed",
                        pool=pool,
                        error_text=(
                            outcome.error_text
                            if outcome is not None and outcome.error_text and fallback_status == "failed"
                            else "single_call_failed" if fallback_status == "failed" else None
                        ),
                        usage=fallback_usage,
                        provider=str(fallback_usage.get("provider") or "") or None,
                        model=str(fallback_usage.get("model") or "") or None,
                        provider_request_id=(
                            str(fallback_usage.get("provider_request_id") or "") or None
                        ),
                    )
                    if fallback_status == "fallback":
                        scorecard_guardrail_fallbacks += 1
                    elif fallback_status == "failed":
                        scorecard_llm_failures += 1
        else:
            _narrative_sem = asyncio.Semaphore(cfg.scorecard_narrative_concurrency)

            async def _generate_scorecard_narrative(sc: dict[str, Any]) -> str:
                """Generate one scorecard narrative. Returns status string."""
                reasoning_summary = sc.get("reasoning_summary", "")
                if reasoning_summary and not sc.get("cross_vendor_comparisons"):
                    sc["expert_take"] = reasoning_summary
                    return "reused"
                async with _narrative_sem:
                    llm_input = _build_scorecard_narrative_payload(
                        sc,
                        reasoning_lookup=reasoning_lookup,
                    )
                    request: Any | None = None
                    if _provider and _model:
                        try:
                            request, _ = prepare_b2b_exact_skill_stage_request(
                                _cache_stage_id,
                                skill_name="digest/vendor_deep_dive_narrative",
                                payload=json.dumps(llm_input, default=str),
                                provider=_provider,
                                model=_model,
                                max_tokens=_llm_max_tokens,
                                temperature=0.3,
                                response_format={"type": "json_object"},
                            )
                        except CacheUnavailable:
                            request = None

                    if request is not None:
                        cached = await lookup_b2b_exact_stage_text(request)
                        if cached is not None:
                            parsed_narrative = parse_json_response(cached["response_text"])
                            expert_take = _normalize_scorecard_expert_take(
                                parsed_narrative.get("expert_take", "")
                            )
                            narrative_errors = _validate_scorecard_expert_take(sc, expert_take)
                            if not narrative_errors:
                                sc["expert_take"] = expert_take
                                return "cached"

                    return await _generate_scorecard_narrative_direct(
                        sc,
                        llm_input,
                        request=request,
                        workload=_llm_workload,
                    )

            narrative_results = await asyncio.gather(*[
                _generate_scorecard_narrative(sc)
                for sc in deterministic_vendor_scorecards
            ])
            for status in narrative_results:
                if status == "reused":
                    scorecard_reasoning_reused += 1
                elif status == "cached":
                    scorecard_cache_hits += 1
                elif status == "fallback":
                    scorecard_guardrail_fallbacks += 1
                elif status == "failed":
                    scorecard_llm_failures += 1
    if (
        scorecard_llm_failures
        or scorecard_cache_hits
        or scorecard_reasoning_reused
        or scorecard_guardrail_fallbacks
    ):
        logger.info(
            "Scorecard LLM: %d cache hits, %d failed, %d reused reasoning, %d guardrail fallbacks, %d batch jobs, %d batch items",
            scorecard_cache_hits,
            scorecard_llm_failures,
            scorecard_reasoning_reused,
            scorecard_guardrail_fallbacks,
            scorecard_batch_metrics["jobs"],
            scorecard_batch_metrics["submitted_items"],
        )
    scorecard_llm_generated = 0
    if should_generate_scorecard_narratives:
        scorecard_llm_generated = max(
            0,
            len(deterministic_vendor_scorecards)
            - scorecard_cache_hits
            - scorecard_reasoning_reused
            - scorecard_llm_failures
            - scorecard_guardrail_fallbacks,
        )

    # --- Phase 6: Build executive summaries ---
    parsed: dict[str, Any] = {
        "weekly_churn_feed": deterministic_weekly_feed,
        "vendor_scorecards": deterministic_vendor_scorecards,
        "displacement_map": deterministic_displacement_map,
        "category_insights": deterministic_category_overview,
    }
    _exec_sources = _executive_source_list()
    _exec_summaries: dict[str, str] = {}
    for _rt in ("weekly_churn_feed", "vendor_scorecard", "displacement_report", "category_overview"):
        _exec_summaries[_rt] = _build_validated_executive_summary(
            parsed,
            data_context=data_context,
            executive_sources=_exec_sources,
            report_type=_rt,
        )
    _fallback_summary = _exec_summaries.get("weekly_churn_feed", "")

    # --- Phase 7: Persist reports ---
    report_types = [
        ("weekly_churn_feed", deterministic_weekly_feed),
        ("vendor_scorecard", deterministic_vendor_scorecards),
        ("displacement_report", _structure_displacement_report(deterministic_displacement_map)),
        ("category_overview", deterministic_category_overview),
        ("vendor_deep_dive", vendor_deep_dives),
    ]
    if selected_report_types:
        report_types = [
            (report_type, data)
            for report_type, data in report_types
            if report_type in selected_report_types
        ]

    base_data_density = {
        "vendors_analyzed": len(vendor_scores),
        "competitive_flows": len(competitive_disp),
        "pain_categories": len(pain_dist),
        "feature_gaps": len(feature_gaps),
    }

    report_source_review_count = data_context.get("reviews_in_analysis_window")
    report_source_dist = json.dumps(
        {src: info["reviews"] for src, info in data_context.get("source_distribution", {}).items()}
    )

    reports_persisted = 0
    should_persist_reports = not scoped_vendors or persist_scoped_reports
    report_vendor_filter = (
        _scoped_report_vendor_filter(scoped_vendors)
        if scoped_vendors and persist_scoped_reports
        else None
    )
    if should_persist_reports:
        try:
            async with pool.transaction() as conn:
                for report_type, data in report_types:
                    report_density = dict(base_data_density)
                    report_llm_model = "pipeline_deterministic"
                    if report_type == "vendor_scorecard":
                        report_density.update({
                            "scorecard_cache_hits": scorecard_cache_hits,
                            "scorecard_reasoning_reused": scorecard_reasoning_reused,
                            "scorecard_guardrail_fallbacks": scorecard_guardrail_fallbacks,
                            "scorecard_llm_generated": scorecard_llm_generated,
                            "scorecard_llm_failures": scorecard_llm_failures,
                            "scorecard_batch_jobs": scorecard_batch_metrics["jobs"],
                            "scorecard_batch_items_submitted": scorecard_batch_metrics["submitted_items"],
                            "scorecard_batch_cache_prefiltered": scorecard_batch_metrics["cache_prefiltered_items"],
                            "scorecard_batch_fallback_single_call": scorecard_batch_metrics["fallback_single_call_items"],
                            "scorecard_batch_completed_items": scorecard_batch_metrics["completed_items"],
                            "scorecard_batch_failed_items": scorecard_batch_metrics["failed_items"],
                        })
                        if scorecard_llm_generated > 0:
                            report_llm_model = str(
                                getattr(settings.llm, "openrouter_reasoning_model", "") or "pipeline_mixed"
                            )
                    report_status = "published" if data else "failed"
                    latest_failure_step = None if data else "no_data"
                    latest_error_code = None if data else "no_data"
                    latest_error_summary = None if data else "Report has no data"
                    blocker_count = 0 if data else 1
                    report_row = await conn.fetchrow(
                        """
                        INSERT INTO b2b_intelligence (
                            report_date, report_type, vendor_filter, intelligence_data,
                            executive_summary, data_density, status, llm_model,
                            source_review_count, source_distribution,
                            latest_run_id, latest_attempt_no, latest_failure_step,
                            latest_error_code, latest_error_summary,
                            blocker_count, warning_count, quality_score
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                                  $11, $12, $13, $14, $15, $16, $17, $18)
                        ON CONFLICT (report_date, report_type, LOWER(COALESCE(vendor_filter,'')),
                                     LOWER(COALESCE(category_filter,'')),
                                     COALESCE(account_id, '00000000-0000-0000-0000-000000000000'::uuid))
                        DO UPDATE SET intelligence_data = EXCLUDED.intelligence_data,
                                      executive_summary = EXCLUDED.executive_summary,
                                      data_density = EXCLUDED.data_density,
                                      status = EXCLUDED.status,
                                      llm_model = EXCLUDED.llm_model,
                                      source_review_count = EXCLUDED.source_review_count,
                                      source_distribution = EXCLUDED.source_distribution,
                                      latest_run_id = EXCLUDED.latest_run_id,
                                      latest_attempt_no = EXCLUDED.latest_attempt_no,
                                      latest_failure_step = EXCLUDED.latest_failure_step,
                                      latest_error_code = EXCLUDED.latest_error_code,
                                      latest_error_summary = EXCLUDED.latest_error_summary,
                                      blocker_count = EXCLUDED.blocker_count,
                                      warning_count = EXCLUDED.warning_count,
                                      quality_score = EXCLUDED.quality_score,
                                      created_at = now()
                        RETURNING id
                        """,
                        today,
                        report_type,
                        report_vendor_filter,
                        json.dumps(data, default=str),
                        _exec_summaries.get(report_type, _fallback_summary),
                        json.dumps(report_density),
                        report_status,
                        report_llm_model,
                        report_source_review_count,
                        report_source_dist,
                        str(task.id),
                        1,
                        latest_failure_step,
                        latest_error_code,
                        latest_error_summary,
                        blocker_count,
                        0,
                        ((data or {}).get("battle_card_quality") or {}).get("score") if isinstance(data, dict) else None,
                    )
                    reports_persisted += 1
                    from ..visibility import record_attempt
                    await record_attempt(
                        pool, artifact_type="churn_report",
                        artifact_id=str(report_row["id"]),
                        run_id=str(task.id), stage="persistence",
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
                            summary=f"{report_type} produced no data",
                            source_table="b2b_intelligence",
                            source_id=str(report_row["id"]),
                        )
        except Exception:
            logger.exception("Failed to persist intelligence reports")
            from ..visibility import emit_event
            await emit_event(
                pool, stage="reports", event_type="persistence_failure",
                entity_type="churn_report", entity_id="all",
                summary="Failed to persist intelligence reports",
                severity="critical", actionable=True,
                run_id=str(task.id),
                reason_code="persistence_exception",
            )
    else:
        logger.info(
            "Skipping report persistence for scoped churn-report test run: vendors=%s report_types=%s",
            sorted(scoped_vendors),
            sorted(selected_report_types) if selected_report_types else ["all"],
        )

    logger.info(
        "b2b_churn_reports: %d reports persisted, %d vendors, reasoning from %d vendors",
        reports_persisted, len(vendor_scores), len(reasoning_lookup),
    )
    if should_generate_scorecard_narratives:
        scorecard_llm_generated = max(
            0,
            len(deterministic_vendor_scorecards)
            - scorecard_cache_hits
            - scorecard_reasoning_reused
            - scorecard_llm_failures
            - scorecard_guardrail_fallbacks,
        )

    return {
        "_skip_synthesis": "B2B churn reports complete",
        "reports_persisted": reports_persisted,
        "vendors_analyzed": len(vendor_scores),
        "reasoning_vendors": len(reasoning_lookup),
        "scorecard_cache_hits": scorecard_cache_hits,
        "scorecard_reasoning_reused": scorecard_reasoning_reused,
        "scorecard_guardrail_fallbacks": scorecard_guardrail_fallbacks,
        "scorecard_llm_generated": scorecard_llm_generated,
        "scorecard_llm_failures": scorecard_llm_failures,
        "scorecard_batch_jobs": scorecard_batch_metrics["jobs"],
        "scorecard_batch_items_submitted": scorecard_batch_metrics["submitted_items"],
        "scorecard_batch_cache_prefiltered": scorecard_batch_metrics["cache_prefiltered_items"],
        "scorecard_batch_fallback_single_call": scorecard_batch_metrics["fallback_single_call_items"],
        "scorecard_batch_completed_items": scorecard_batch_metrics["completed_items"],
        "scorecard_batch_failed_items": scorecard_batch_metrics["failed_items"],
        "persistence_skipped": not should_persist_reports,
        "selected_report_types": sorted(selected_report_types),
        "scoped_vendors": sorted(scoped_vendors),
    }
