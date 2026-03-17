"""Follow-up task: build and LLM-enrich battle cards per vendor.

Runs after b2b_churn_core. Reads persisted artifacts from
b2b_churn_signals, b2b_reviews, and b2b_product_profiles. Builds
deterministic battle cards, runs LLM sales copy generation in parallel,
and persists to b2b_intelligence.
"""

import asyncio
import json
import logging
from datetime import date
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.tasks.b2b_battle_cards")


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
    """Build battle cards + LLM sales copy from persisted artifacts."""
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
        _build_deterministic_battle_cards,
        _build_pain_lookup,
        _build_competitor_lookup,
        _build_feature_gap_lookup,
        _build_use_case_lookup,
        _build_sentiment_lookup,
        _build_buyer_auth_lookup,
        _build_keyword_spike_lookup,
        _build_positive_lookup,
        _build_department_lookup,
        _build_usage_duration_lookup,
        _build_timeline_lookup,
        _canonicalize_vendor,
        _validate_battle_card_sales_copy,
        _fetch_competitive_displacement,
        _fetch_vendor_churn_scores,
        _fetch_pain_distribution,
        _fetch_feature_gaps,
        _fetch_price_complaint_rates,
        _fetch_dm_churn_rates,
        _fetch_churning_companies,
        _fetch_quotable_evidence,
        _fetch_budget_signals,
        _fetch_use_case_distribution,
        _fetch_sentiment_trajectory,
        _fetch_buyer_authority_summary,
        _fetch_timeline_signals,
        _fetch_keyword_spikes,
        _fetch_product_profiles,
        _fetch_competitor_reasons,
        _fetch_data_context,
        _fetch_review_text_aggregates,
        _fetch_department_distribution,
    )
    from .b2b_churn_intelligence import reconstruct_reasoning_lookup

    window_days = cfg.intelligence_window_days
    min_reviews = cfg.intelligence_min_reviews

    # --- Phase 1: Parallel data fetch ---
    try:
        (
            vendor_scores,
            competitive_disp, pain_dist, feature_gaps,
            price_rates, dm_rates,
            churning_companies, quotable_evidence,
            budget_signals, use_case_dist,
            sentiment_traj, buyer_auth, timeline_signals,
            competitor_reasons, keyword_spikes,
            data_context, product_profiles_raw,
            review_text_agg, department_dist,
        ) = await asyncio.gather(
            _fetch_vendor_churn_scores(pool, window_days, min_reviews),
            _fetch_competitive_displacement(pool, window_days),
            _fetch_pain_distribution(pool, window_days),
            _fetch_feature_gaps(pool, window_days, cfg.feature_gap_min_mentions),
            _fetch_price_complaint_rates(pool, window_days),
            _fetch_dm_churn_rates(pool, window_days),
            _fetch_churning_companies(pool, window_days),
            _fetch_quotable_evidence(pool, window_days, cfg.quotable_phrase_min_urgency),
            _fetch_budget_signals(pool, window_days),
            _fetch_use_case_distribution(pool, window_days),
            _fetch_sentiment_trajectory(pool, window_days),
            _fetch_buyer_authority_summary(pool, window_days),
            _fetch_timeline_signals(pool, window_days),
            _fetch_competitor_reasons(pool, window_days),
            _fetch_keyword_spikes(pool),
            _fetch_data_context(pool, window_days),
            _fetch_product_profiles(pool),
            _fetch_review_text_aggregates(pool, window_days),
            _fetch_department_distribution(pool, window_days),
        )
    except Exception:
        logger.exception("Battle card data fetch failed")
        return {"_skip_synthesis": "Data fetch failed"}

    if not vendor_scores:
        return {"_skip_synthesis": "No vendor scores"}

    # --- Phase 2: Reconstruct reasoning from DB ---
    reasoning_lookup = await reconstruct_reasoning_lookup(pool, as_of=today)

    # --- Phase 3: Build lookups ---
    pain_lookup = _build_pain_lookup(pain_dist)
    competitor_lookup = _build_competitor_lookup(competitive_disp)
    feature_gap_lookup = _build_feature_gap_lookup(feature_gaps)
    price_lookup = {r["vendor"]: r["price_complaint_rate"] for r in price_rates}
    dm_lookup = {r["vendor"]: r["dm_churn_rate"] for r in dm_rates}
    company_lookup = {r["vendor"]: r["companies"] for r in churning_companies}
    quote_lookup = {r["vendor"]: r["quotes"] for r in quotable_evidence}
    budget_lookup = {r["vendor"]: {k: v for k, v in r.items() if k != "vendor"} for r in budget_signals}
    sentiment_lookup = _build_sentiment_lookup(sentiment_traj)
    buyer_auth_lookup = _build_buyer_auth_lookup(buyer_auth)
    timeline_lookup = _build_timeline_lookup(timeline_signals)
    keyword_spike_lookup = _build_keyword_spike_lookup(keyword_spikes)
    positive_lookup = _build_positive_lookup(review_text_agg)
    department_lookup = _build_department_lookup(department_dist)
    usage_duration_lookup = _build_usage_duration_lookup(review_text_agg)

    product_profile_lookup: dict[str, dict] = {}
    for pp in product_profiles_raw:
        vn = _canonicalize_vendor(pp.get("vendor_name", ""))
        if vn and vn not in product_profile_lookup:
            product_profile_lookup[vn] = pp

    # --- Phase 4: Build deterministic battle cards ---
    deterministic_battle_cards = _build_deterministic_battle_cards(
        vendor_scores,
        pain_lookup=pain_lookup,
        competitor_lookup=competitor_lookup,
        feature_gap_lookup=feature_gap_lookup,
        quote_lookup=quote_lookup,
        price_lookup=price_lookup,
        budget_lookup=budget_lookup,
        sentiment_lookup=sentiment_lookup,
        dm_lookup=dm_lookup,
        company_lookup=company_lookup,
        product_profile_lookup=product_profile_lookup,
        competitive_disp=competitive_disp,
        competitor_reasons=competitor_reasons,
        reasoning_lookup=reasoning_lookup,
        timeline_lookup=timeline_lookup,
        use_case_lookup=_build_use_case_lookup(use_case_dist),
        positive_lookup=positive_lookup,
        department_lookup=department_lookup,
        usage_duration_lookup=usage_duration_lookup,
        buyer_auth_lookup=buyer_auth_lookup,
    )

    # Enrich with ecosystem context
    try:
        from atlas_brain.reasoning.ecosystem import EcosystemAnalyzer
        eco = EcosystemAnalyzer(pool)
        ecosystem_evidence = await eco.analyze_all_categories(reasoning_lookup)
        for card in deterministic_battle_cards:
            cat = card.get("category", "")
            eco_data = ecosystem_evidence.get(cat)
            if eco_data:
                card["ecosystem_context"] = {
                    "hhi": eco_data.get("hhi"),
                    "market_structure": eco_data.get("market_structure"),
                    "displacement_intensity": eco_data.get("displacement_intensity"),
                    "dominant_archetype": eco_data.get("dominant_archetype"),
                }
    except Exception:
        logger.debug("Ecosystem enrichment skipped", exc_info=True)

    logger.info("Built %d deterministic battle cards", len(deterministic_battle_cards))

    # --- Phase 5: LLM sales copy (parallel with semantic cache) ---
    from ...pipelines.llm import call_llm_with_skill, parse_json_response
    from ...reasoning.semantic_cache import SemanticCache, CacheEntry, compute_evidence_hash

    _bc_cache = SemanticCache(pool)
    bc_llm_failures = 0
    bc_cache_hits = 0
    _bc_llm_fields = (
        "executive_summary", "weakness_analysis", "discovery_questions",
        "landmine_questions", "objection_handlers", "competitive_landscape",
        "talk_track", "recommended_plays",
    )
    bc_sem = asyncio.Semaphore(cfg.battle_card_llm_concurrency)

    async def _enrich_one(card: dict[str, Any]) -> None:
        nonlocal bc_llm_failures, bc_cache_hits

        card_hash = compute_evidence_hash({
            "vendor": card.get("vendor"),
            "churn_pressure_score": card.get("churn_pressure_score"),
            "vendor_weaknesses": card.get("vendor_weaknesses"),
            "customer_pain_quotes": card.get("customer_pain_quotes"),
            "competitor_differentiators": card.get("competitor_differentiators"),
            "objection_data": card.get("objection_data"),
        })
        pattern_sig = f"battle_card:{card.get('vendor')}:{card_hash}"

        cached = await _bc_cache.lookup(pattern_sig)
        if cached:
            cached_errors = _validate_battle_card_sales_copy(card, cached.conclusion)
            if cached_errors:
                await _bc_cache.invalidate(pattern_sig, reason="invalid")
            else:
                for _cf in cached.conclusion:
                    card[_cf] = cached.conclusion[_cf]
                await _bc_cache.validate(pattern_sig)
                bc_cache_hits += 1
                return

        async with bc_sem:
            try:
                sales_copy = await asyncio.wait_for(
                    asyncio.to_thread(
                        call_llm_with_skill,
                        "digest/battle_card_sales_copy",
                        json.dumps(card, default=str),
                        max_tokens=3000, temperature=0.5,
                        response_format={"type": "json_object"},
                        workload="synthesis",
                    ),
                    timeout=90,
                )
                parsed_copy = parse_json_response(sales_copy)
                copy_errors = _validate_battle_card_sales_copy(card, parsed_copy)
                if copy_errors:
                    bc_llm_failures += 1
                    logger.warning("Battle card rejected for %s: %s",
                                   card.get("vendor"), "; ".join(copy_errors[:3]))
                    return
                for _f in _bc_llm_fields:
                    if _f in parsed_copy:
                        card[_f] = parsed_copy[_f]
            except Exception:
                bc_llm_failures += 1
                logger.warning("Battle card LLM failed for %s", card.get("vendor"))
                return

            try:
                await _bc_cache.store(CacheEntry(
                    pattern_sig=pattern_sig,
                    pattern_class="battle_card_sales_copy",
                    conclusion={_f: card[_f] for _f in _bc_llm_fields if _f in card},
                    confidence=0.95,
                    evidence_hash=card_hash,
                    vendor_name=card.get("vendor"),
                    conclusion_type="sales_copy",
                ))
            except Exception:
                logger.warning("Failed to cache battle card for %s", card.get("vendor"))

    await asyncio.gather(*[_enrich_one(c) for c in deterministic_battle_cards])

    logger.info(
        "Battle card LLM: %d cache hits, %d generated, %d failed (of %d)",
        bc_cache_hits,
        len(deterministic_battle_cards) - bc_cache_hits - bc_llm_failures,
        bc_llm_failures,
        len(deterministic_battle_cards),
    )

    # --- Phase 6: Persist battle cards ---
    data_density = json.dumps({"vendors_analyzed": len(vendor_scores)})
    report_source_review_count = data_context.get("reviews_in_analysis_window")
    report_source_dist = json.dumps(
        {src: info["reviews"] for src, info in data_context.get("source_distribution", {}).items()}
    )

    cards_persisted = 0
    for card in deterministic_battle_cards:
        vendor = card.get("vendor", "")
        if not vendor:
            continue
        try:
            await pool.execute(
                """
                INSERT INTO b2b_intelligence (
                    report_date, report_type, vendor_filter,
                    intelligence_data, executive_summary, data_density, status, llm_model,
                    source_review_count, source_distribution
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
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
                "battle_card",
                vendor,
                json.dumps(card, default=str),
                (
                    f"Battle card for {vendor}: "
                    f"score {card.get('churn_pressure_score', 0):.0f}, "
                    f"{len(card.get('vendor_weaknesses', []))} weaknesses, "
                    f"{len(card.get('competitor_differentiators', []))} competitors."
                ),
                data_density,
                "published",
                "pipeline_deterministic",
                report_source_review_count,
                report_source_dist,
            )
            cards_persisted += 1
        except Exception:
            logger.exception("Failed to persist battle card for %s", vendor)

    logger.info("Persisted %d/%d battle cards", cards_persisted, len(deterministic_battle_cards))

    return {
        "_skip_synthesis": "B2B battle cards complete",
        "cards_built": len(deterministic_battle_cards),
        "cards_persisted": cards_persisted,
        "cache_hits": bc_cache_hits,
        "llm_failures": bc_llm_failures,
        "reasoning_vendors": len(reasoning_lookup),
    }
