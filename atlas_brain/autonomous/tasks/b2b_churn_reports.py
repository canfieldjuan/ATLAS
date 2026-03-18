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

logger = logging.getLogger("atlas.tasks.b2b_churn_reports")


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
        _fetch_competitive_displacement,
        _fetch_competitor_reasons,
        _fetch_data_context,
        _fetch_vendor_churn_scores,
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
        _validate_scorecard_expert_take,
    )
    from .b2b_churn_intelligence import (
        reconstruct_reasoning_lookup,
        reconstruct_cross_vendor_lookup,
    )

    window_days = cfg.intelligence_window_days
    min_reviews = cfg.intelligence_min_reviews

    # --- Phase 1: Parallel data fetch (same queries as core, reads b2b_reviews) ---
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
            _fetch_vendor_churn_scores(pool, window_days, min_reviews),
            _fetch_competitive_displacement(pool, window_days),
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

    # --- Phase 2: Reconstruct reasoning + cross-vendor from DB ---
    reasoning_lookup = await reconstruct_reasoning_lookup(pool, as_of=today)
    xv_lookup = await reconstruct_cross_vendor_lookup(pool, as_of=today)
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

    # --- Phase 3: Build lookups ---
    pain_lookup = _build_pain_lookup(pain_dist)
    competitor_lookup = _build_competitor_lookup(competitive_disp)
    feature_gap_lookup = _build_feature_gap_lookup(feature_gaps)
    neg_lookup = {r["vendor"]: r["negative_count"] for r in negative_counts}
    price_lookup = {r["vendor"]: r["price_complaint_rate"] for r in price_rates}
    dm_lookup = {r["vendor"]: r["dm_churn_rate"] for r in dm_rates}
    company_lookup = {r["vendor"]: r["companies"] for r in churning_companies}
    quote_lookup = {r["vendor"]: r["quotes"] for r in quotable_evidence}
    budget_lookup = {r["vendor"]: {k: v for k, v in r.items() if k != "vendor"} for r in budget_signals}
    use_case_lookup = _build_use_case_lookup(use_case_dist)
    integration_lookup = _build_integration_lookup(use_case_dist)
    sentiment_lookup = _build_sentiment_lookup(sentiment_traj)
    buyer_auth_lookup = _build_buyer_auth_lookup(buyer_auth)
    timeline_lookup = _build_timeline_lookup(timeline_signals)
    keyword_spike_lookup = _build_keyword_spike_lookup(keyword_spikes)
    _complaints_raw, _positives_raw = review_text_agg
    complaint_lookup = _build_complaint_lookup(_complaints_raw)
    positive_lookup = _build_positive_lookup(_positives_raw)
    department_lookup = _build_department_lookup(department_dist)
    _contract_values_raw, _durations_raw = contract_ctx
    contract_value_lookup = _build_contract_value_lookup(_contract_values_raw)
    turning_point_lookup = _build_turning_point_lookup(turning_points)
    tenure_lookup = _build_tenure_lookup(sentiment_tenure)

    product_profile_lookup: dict[str, dict] = {}
    for pp in product_profiles_raw:
        vn = _canonicalize_vendor(pp.get("vendor_name", ""))
        if vn and vn not in product_profile_lookup:
            product_profile_lookup[vn] = pp

    # --- Phase 4: Build deterministic reports ---
    deterministic_weekly_feed = _build_deterministic_vendor_feed(
        vendor_scores,
        pain_lookup=pain_lookup,
        competitor_lookup=competitor_lookup,
        feature_gap_lookup=feature_gap_lookup,
        quote_lookup=quote_lookup,
        budget_lookup=budget_lookup,
        sentiment_lookup=sentiment_lookup,
        buyer_auth_lookup=buyer_auth_lookup,
        dm_lookup=dm_lookup,
        price_lookup=price_lookup,
        company_lookup=company_lookup,
        keyword_spike_lookup=keyword_spike_lookup,
        prior_reports=prior_reports,
        reasoning_lookup=reasoning_lookup,
    )
    deterministic_vendor_scorecards = _build_deterministic_vendor_scorecards(
        vendor_scores,
        pain_lookup=pain_lookup,
        competitor_lookup=competitor_lookup,
        feature_gap_lookup=feature_gap_lookup,
        quote_lookup=quote_lookup,
        budget_lookup=budget_lookup,
        sentiment_lookup=sentiment_lookup,
        buyer_auth_lookup=buyer_auth_lookup,
        dm_lookup=dm_lookup,
        price_lookup=price_lookup,
        company_lookup=company_lookup,
        product_profile_lookup=product_profile_lookup,
        prior_reports=prior_reports,
        reasoning_lookup=reasoning_lookup,
        timeline_lookup=timeline_lookup,
        use_case_lookup=use_case_lookup,
        complaint_lookup=complaint_lookup,
        positive_lookup=positive_lookup,
        department_lookup=department_lookup,
        contract_value_lookup=contract_value_lookup,
        turning_point_lookup=turning_point_lookup,
        tenure_lookup=tenure_lookup,
    )
    deterministic_displacement_map = _build_deterministic_displacement_map(
        competitive_disp,
        competitor_reasons,
        quote_lookup,
        reasoning_lookup=reasoning_lookup,
    )

    # Enrich displacement edges with provenance
    for edge in deterministic_displacement_map:
        prov_key = (edge["from_vendor"], edge["to_vendor"])
        prov = displacement_provenance.get(prov_key, {})
        src_dist = prov.get("source_distribution", {})
        edge["source_distribution"] = src_dist
        edge["sample_review_ids"] = prov.get("sample_review_ids", [])
        edge["confidence_score"] = _compute_evidence_confidence(
            edge["mention_count"], src_dist,
        )

    deterministic_category_overview = _build_deterministic_category_overview(
        vendor_scores,
        pain_lookup=pain_lookup,
        competitive_disp=competitive_disp,
        company_lookup=company_lookup,
        quote_lookup=quote_lookup,
        feature_gap_lookup=feature_gap_lookup,
        dm_lookup=dm_lookup,
        price_lookup=price_lookup,
        competitor_lookup=competitor_lookup,
        reasoning_lookup=reasoning_lookup,
    )

    # Enrich category overview with ecosystem evidence
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

    # Enrich category overview with cross-vendor conclusions
    for cat_entry in deterministic_category_overview:
        cat_name = cat_entry.get("category", "")
        council = xv_lookup.get("councils", {}).get(cat_name)
        if council:
            c = council.get("conclusion", {})
            cat_entry["cross_vendor_analysis"] = {
                "conclusion": c.get("conclusion", ""),
                "market_regime": c.get("market_regime", ""),
                "durability_assessment": c.get("durability_assessment", ""),
                "key_insights": c.get("key_insights", []),
                "winner": c.get("winner", ""),
                "loser": c.get("loser", ""),
                "confidence": council.get("confidence", 0),
            }

    # Enrich displacement edges with cross-vendor battle conclusions
    for edge in deterministic_displacement_map:
        pair = tuple(sorted([edge["from_vendor"], edge["to_vendor"]]))
        battle = xv_lookup.get("battles", {}).get(pair)
        if battle:
            bc = battle.get("conclusion", {})
            edge["battle_conclusion"] = bc.get("conclusion", "")
            edge["durability"] = bc.get("durability_assessment", "")

    # Enrich scorecards with cross-vendor comparisons
    for sc in deterministic_vendor_scorecards:
        vendor = sc.get("vendor", "")
        comparisons = []
        for pair_key, asym in xv_lookup.get("asymmetries", {}).items():
            if vendor in pair_key:
                ac = asym.get("conclusion", {})
                comparisons.append({
                    "opponent": [v for v in pair_key if v != vendor][0] if len(pair_key) > 1 else "",
                    "conclusion": ac.get("conclusion", ""),
                    "confidence": asym.get("confidence", 0),
                    "resource_advantage": ac.get("resource_advantage", ""),
                })
        if comparisons:
            sc["cross_vendor_comparisons"] = comparisons

    # --- Phase 5: Scorecard narrative LLM enrichment ---
    from ...pipelines.llm import call_llm_with_skill, parse_json_response

    _llm_workload = "synthesis"
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
            llm_input = {k: sc[k] for k in (
                "vendor", "churn_pressure_score", "risk_level", "churn_signal_density",
                "avg_urgency", "feature_analysis", "churn_predictors", "competitor_overlap",
                "trend", "sentiment_direction", "cross_vendor_comparisons",
            ) if k in sc}
            llm_input["locked_facts"] = _build_scorecard_locked_facts(sc)
            if sc.get("archetype"):
                llm_input["reasoning_conclusion"] = {
                    "archetype": sc["archetype"],
                    "confidence": sc.get("archetype_confidence", 0),
                    "executive_summary": reasoning_summary,
                    "key_signals": reasoning_lookup.get(sc.get("vendor", ""), {}).get("key_signals", []),
                }
            narrative = await asyncio.wait_for(
                asyncio.to_thread(
                    call_llm_with_skill,
                    "digest/vendor_deep_dive_narrative",
                    json.dumps(llm_input, default=str),
                    max_tokens=300, temperature=0.3,
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

    data_density = json.dumps({
        "vendors_analyzed": len(vendor_scores),
        "competitive_flows": len(competitive_disp),
        "pain_categories": len(pain_dist),
        "feature_gaps": len(feature_gaps),
    })

    report_source_review_count = data_context.get("reviews_in_analysis_window")
    report_source_dist = json.dumps(
        {src: info["reviews"] for src, info in data_context.get("source_distribution", {}).items()}
    )

    reports_persisted = 0
    try:
        async with pool.transaction() as conn:
            for report_type, data in report_types:
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
                                  source_review_count = EXCLUDED.source_review_count,
                                  source_distribution = EXCLUDED.source_distribution,
                                  created_at = now()
                    """,
                    today,
                    report_type,
                    json.dumps(data, default=str),
                    _exec_summaries.get(report_type, _fallback_summary),
                    data_density,
                    "published",
                    "pipeline_deterministic",
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

    return {
        "_skip_synthesis": "B2B churn reports complete",
        "reports_persisted": reports_persisted,
        "vendors_analyzed": len(vendor_scores),
        "reasoning_vendors": len(reasoning_lookup),
        "scorecard_reasoning_reused": scorecard_reasoning_reused,
        "scorecard_llm_failures": scorecard_llm_failures,
    }
