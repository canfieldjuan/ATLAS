"""Follow-up task: build and persist churn intelligence reports.

Runs after b2b_churn_core (staggered cron). Reads persisted artifacts
from b2b_churn_signals, b2b_reviews, and related tables. Builds
deterministic reports, runs LLM enrichment, and persists to
b2b_intelligence.

Does NOT re-run the stratified reasoner -- reads reasoning conclusions
from b2b_churn_signals via reconstruct_reasoning_lookup().
"""

import asyncio
import json
import logging
from datetime import date, datetime, timezone
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from ._b2b_shared import _timing_summary_payload

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


def _build_scorecard_narrative_payload(
    scorecard: dict[str, Any],
    *,
    reasoning_lookup: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a compact LLM payload for scorecard narrative generation."""
    from ._b2b_shared import _build_scorecard_locked_facts

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
        payload["feature_analysis"] = {
            "loved": list(feature_analysis.get("loved") or [])[:3],
            "hated": list(feature_analysis.get("hated") or [])[:3],
        }

    churn_predictors = scorecard.get("churn_predictors") or {}
    if isinstance(churn_predictors, dict):
        payload["churn_predictors"] = {
            "high_risk_industries": list(
                churn_predictors.get("high_risk_industries") or []
            )[:2],
            "high_risk_sizes": list(
                churn_predictors.get("high_risk_sizes") or []
            )[:2],
            "dm_churn_rate": churn_predictors.get("dm_churn_rate"),
            "price_complaint_rate": churn_predictors.get("price_complaint_rate"),
        }

    competitor_overlap = scorecard.get("competitor_overlap") or []
    if competitor_overlap:
        payload["competitor_overlap"] = list(competitor_overlap)[:3]

    cross_vendor_comparisons = scorecard.get("cross_vendor_comparisons") or []
    if cross_vendor_comparisons:
        payload["cross_vendor_comparisons"] = [
            {
                "opponent": item.get("opponent", ""),
                "conclusion": item.get("conclusion", ""),
                "confidence": item.get("confidence", 0),
                "resource_advantage": item.get("resource_advantage", ""),
            }
            for item in list(cross_vendor_comparisons)[:2]
            if isinstance(item, dict)
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
        payload["reasoning_conclusion"] = {
            "archetype": scorecard["archetype"],
            "confidence": scorecard.get("archetype_confidence", 0),
            "executive_summary": scorecard.get("reasoning_summary", ""),
            "key_signals": list(
                (reasoning_lookup or {}).get(scorecard.get("vendor", ""), {}).get("key_signals", [])
            )[:4],
        }

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


def _reasoning_int(value: Any) -> int | None:
    """Coerce traced-number wrappers into ints."""
    if isinstance(value, dict):
        value = value.get("value")
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


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
    """Fetch latest reasoning synthesis rows and wrap them in SynthesisView."""
    from ._b2b_synthesis_reader import load_synthesis_view

    rows = await pool.fetch(
        """
        SELECT DISTINCT ON (vendor_name)
               vendor_name, as_of_date, schema_version, synthesis
        FROM b2b_reasoning_synthesis
        WHERE as_of_date <= $1
          AND analysis_window_days = $2
        ORDER BY vendor_name, as_of_date DESC, created_at DESC
        """,
        as_of,
        analysis_window_days,
    )
    views: dict[str, Any] = {}
    for row in rows:
        raw = row["synthesis"]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                continue
        if isinstance(raw, dict):
            views[row["vendor_name"]] = load_synthesis_view(
                raw,
                row["vendor_name"],
                schema_version=row["schema_version"] or "",
                as_of_date=row["as_of_date"],
            )
    return views


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
        contract_gaps_for_consumer,
        inject_synthesis_freshness,
    )

    contracts: dict[str, Any] = {}
    materialized = view.materialized_contracts()
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
    contract_gaps = contract_gaps_for_consumer(view, consumer_name)
    if contract_gaps:
        entry["reasoning_contract_gaps"] = contract_gaps


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
            if council:
                c = council.get("conclusion", {})
                if any([c.get("winner"), c.get("loser"), c.get("conclusion"), c.get("market_regime"), c.get("key_insights")]):
                    entry["category_council"] = {
                        "winner": c.get("winner") or "",
                        "loser": c.get("loser") or "",
                        "conclusion": c.get("conclusion") or "",
                        "market_regime": c.get("market_regime") or "",
                        "durability": c.get("durability_assessment") or "",
                        "confidence": council.get("confidence"),
                        "key_insights": c.get("key_insights") or [],
                    }

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
            if not scorecard.get("reasoning_summary"):
                causal = view.section("causal_narrative")
                trigger = causal.get("trigger") if isinstance(causal, dict) else ""
                why_now = causal.get("why_now") if isinstance(causal, dict) else ""
                summary_bits = [bit for bit in (trigger, why_now) if bit]
                if summary_bits:
                    scorecard["reasoning_summary"] = ". ".join(summary_bits)
        category_name = str(feed_category_lookup.get(vendor) or scorecard.get("category") or "").strip()
        if category_name:
            scorecard.setdefault("category", category_name)
            council = xv_lookup.get("councils", {}).get(category_name)
            if council:
                c = council.get("conclusion", {})
                if any([c.get("winner"), c.get("loser"), c.get("conclusion"), c.get("market_regime"), c.get("key_insights")]):
                    scorecard["category_council"] = {
                        "winner": c.get("winner") or "",
                        "loser": c.get("loser") or "",
                        "conclusion": c.get("conclusion") or "",
                        "market_regime": c.get("market_regime") or "",
                        "durability": c.get("durability_assessment") or "",
                        "confidence": council.get("confidence"),
                        "key_insights": c.get("key_insights") or [],
                    }
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

    for edge in deterministic_displacement_map:
        pair = tuple(sorted([edge["from_vendor"], edge["to_vendor"]]))
        battle = xv_lookup.get("battles", {}).get(pair)
        if battle:
            bc = battle.get("conclusion", {})
            edge["battle_conclusion"] = bc.get("conclusion", "")
            edge["durability"] = bc.get("durability_assessment", "")

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
        _compute_evidence_confidence,
        _fetch_latest_evidence_vault,
    )
    from .b2b_churn_intelligence import (
        reconstruct_cross_vendor_lookup,
        reconstruct_reasoning_lookup,
    )

    if reasoning_lookup is None:
        reasoning_lookup = await reconstruct_reasoning_lookup(pool, as_of=as_of)
    if xv_lookup is None:
        xv_lookup = await reconstruct_cross_vendor_lookup(pool, as_of=as_of)
    if synthesis_views is None:
        synthesis_views = await _fetch_latest_synthesis_views(
            pool,
            as_of=as_of,
            analysis_window_days=analysis_window_days,
        )
    if evidence_vault_lookup is None:
        evidence_vault_lookup = await _fetch_latest_evidence_vault(
            pool,
            as_of=as_of,
            analysis_window_days=analysis_window_days,
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
        reasoning_lookup=reasoning_lookup,
        timeline_lookup=lookups["timeline_lookup"],
        use_case_lookup=lookups["use_case_lookup"],
        complaint_lookup=lookups["complaint_lookup"],
        positive_lookup=lookups["positive_lookup"],
        department_lookup=lookups["department_lookup"],
        contract_value_lookup=lookups["contract_value_lookup"],
        turning_point_lookup=lookups["turning_point_lookup"],
        tenure_lookup=lookups["tenure_lookup"],
    )
    deterministic_displacement_map = _build_deterministic_displacement_map(
        competitive_disp,
        competitor_reasons,
        lookups["quote_lookup"],
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
        reasoning_lookup=reasoning_lookup,
    )
    attached_contract_vendors = _attach_context_to_deterministic_reports(
        pool=pool,
        as_of=as_of,
        deterministic_weekly_feed=deterministic_weekly_feed,
        deterministic_vendor_scorecards=deterministic_vendor_scorecards,
        deterministic_displacement_map=deterministic_displacement_map,
        deterministic_category_overview=deterministic_category_overview,
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
            # Flatten cross_vendor_analysis to avoid [object Object] in UI GenericReportView
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
        "reasoning_lookup": reasoning_lookup,
        "xv_lookup": xv_lookup,
        "synthesis_views": synthesis_views,
        "evidence_vault_lookup": evidence_vault_lookup,
        "attached_contract_vendors": attached_contract_vendors,
    }


async def _check_freshness(pool) -> date | None:
    """Return today's date if core task wrote a completion marker, else None."""
    today = date.today()
    marker = await pool.fetchval(
        "SELECT 1 FROM b2b_intelligence "
        "WHERE report_type = 'core_run_complete' AND report_date = $1",
        today,
    )
    if not marker:
        logger.info("Core run not complete for %s, skipping", today)
        return None
    return today


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
        return {"_skip_synthesis": "Core signals not fresh for today"}

    from ._b2b_shared import (
        _aggregate_competitive_disp,
        _build_deterministic_category_overview,
        _build_deterministic_displacement_map,
        _build_deterministic_vendor_feed,
        _build_deterministic_vendor_scorecards,
        _build_scorecard_locked_facts,
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
        _fetch_latest_evidence_vault,
        _fetch_competitive_displacement_source_of_truth,
        _fetch_competitor_reasons,
        _fetch_data_context,
        _fetch_vendor_churn_scores_from_signals,
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
        reconstruct_reasoning_lookup,
        reconstruct_cross_vendor_lookup,
    )

    window_days = cfg.intelligence_window_days
    min_reviews = cfg.intelligence_min_reviews

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
            _fetch_vendor_churn_scores_from_signals(pool, window_days, min_reviews),
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
    )
    reasoning_lookup = bundle["reasoning_lookup"]
    xv_lookup = bundle["xv_lookup"]
    deterministic_weekly_feed = bundle["weekly_churn_feed"]
    deterministic_vendor_scorecards = bundle["vendor_scorecards"]
    deterministic_displacement_map = bundle["displacement_map"]
    deterministic_category_overview = bundle["category_insights"]
    logger.info(
        "Reconstructed reasoning for %d vendors from b2b_churn_signals",
        len(reasoning_lookup),
    )
    logger.info(
        "Cross-vendor enrichment: %d battles, %d councils, %d asymmetries",
        len(xv_lookup.get("battles", {})),
        len(xv_lookup.get("councils", {})),
        len(xv_lookup.get("asymmetries", {})),
    )

    # --- Phase 5: Scorecard narrative LLM enrichment ---
    from ...pipelines.llm import call_llm_with_skill, parse_json_response

    _llm_workload = "synthesis"
    _llm_max_tokens = _scorecard_narrative_max_tokens()
    scorecard_llm_failures = 0
    scorecard_reasoning_reused = 0
    scorecard_guardrail_fallbacks = 0
    for sc in deterministic_vendor_scorecards:
        reasoning_summary = sc.get("reasoning_summary", "")
        if reasoning_summary and cfg.stratified_reasoning_enabled and not sc.get("cross_vendor_comparisons"):
            sc["expert_take"] = reasoning_summary
            scorecard_reasoning_reused += 1
            continue
        try:
            llm_input = _build_scorecard_narrative_payload(
                sc,
                reasoning_lookup=reasoning_lookup,
            )
            narrative = await asyncio.wait_for(
                asyncio.to_thread(
                    call_llm_with_skill,
                    "digest/vendor_deep_dive_narrative",
                    json.dumps(llm_input, default=str),
                    max_tokens=_llm_max_tokens, temperature=0.3,
                    response_format={"type": "json_object"},
                    workload=_llm_workload,
                ),
                timeout=45,
            )
            parsed_narrative = parse_json_response(narrative)
            expert_take = parsed_narrative.get("expert_take", "")
            narrative_errors = _validate_scorecard_expert_take(sc, expert_take)
            if narrative_errors:
                scorecard_guardrail_fallbacks += 1
                sc["expert_take"] = _fallback_scorecard_expert_take(sc)
            else:
                sc["expert_take"] = expert_take
        except Exception:
            scorecard_llm_failures += 1
            sc["expert_take"] = _fallback_scorecard_expert_take(sc)
    if scorecard_llm_failures or scorecard_reasoning_reused or scorecard_guardrail_fallbacks:
        logger.info(
            "Scorecard LLM: %d failed, %d reused reasoning, %d guardrail fallbacks",
            scorecard_llm_failures, scorecard_reasoning_reused, scorecard_guardrail_fallbacks,
        )
    scorecard_llm_generated = max(
        0,
        len(deterministic_vendor_scorecards)
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
        ("displacement_report", deterministic_displacement_map),
        ("category_overview", deterministic_category_overview),
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
    try:
        async with pool.transaction() as conn:
            for report_type, data in report_types:
                report_density = dict(base_data_density)
                report_llm_model = "pipeline_deterministic"
                if report_type == "vendor_scorecard":
                    report_density.update({
                        "scorecard_reasoning_reused": scorecard_reasoning_reused,
                        "scorecard_guardrail_fallbacks": scorecard_guardrail_fallbacks,
                        "scorecard_llm_generated": scorecard_llm_generated,
                        "scorecard_llm_failures": scorecard_llm_failures,
                    })
                    if scorecard_llm_generated > 0:
                        report_llm_model = str(
                            getattr(settings.llm, "openrouter_reasoning_model", "") or "pipeline_mixed"
                        )
                await conn.execute(
                    """
                    INSERT INTO b2b_intelligence (
                        report_date, report_type, intelligence_data,
                        executive_summary, data_density, status, llm_model,
                        source_review_count, source_distribution
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (report_date, report_type, LOWER(COALESCE(vendor_filter,'')),
                                 LOWER(COALESCE(category_filter,'')),
                                 COALESCE(account_id, '00000000-0000-0000-0000-000000000000'::uuid))
                    DO UPDATE SET intelligence_data = EXCLUDED.intelligence_data,
                                  executive_summary = EXCLUDED.executive_summary,
                                  data_density = EXCLUDED.data_density,
                                  llm_model = EXCLUDED.llm_model,
                                  source_review_count = EXCLUDED.source_review_count,
                                  source_distribution = EXCLUDED.source_distribution,
                                  created_at = now()
                    """,
                    today,
                    report_type,
                    json.dumps(data, default=str),
                    _exec_summaries.get(report_type, _fallback_summary),
                    json.dumps(report_density),
                    "published",
                    report_llm_model,
                    report_source_review_count,
                    report_source_dist,
                )
                reports_persisted += 1
    except Exception:
        logger.exception("Failed to persist intelligence reports")

    logger.info(
        "b2b_churn_reports: %d reports persisted, %d vendors, reasoning from %d vendors",
        reports_persisted, len(vendor_scores), len(reasoning_lookup),
    )
    scorecard_llm_generated = max(
        0,
        len(deterministic_vendor_scorecards)
        - scorecard_reasoning_reused
        - scorecard_llm_failures
        - scorecard_guardrail_fallbacks,
    )

    return {
        "_skip_synthesis": "B2B churn reports complete",
        "reports_persisted": reports_persisted,
        "vendors_analyzed": len(vendor_scores),
        "reasoning_vendors": len(reasoning_lookup),
        "scorecard_reasoning_reused": scorecard_reasoning_reused,
        "scorecard_guardrail_fallbacks": scorecard_guardrail_fallbacks,
        "scorecard_llm_generated": scorecard_llm_generated,
        "scorecard_llm_failures": scorecard_llm_failures,
    }
