"""Follow-up task: build and LLM-enrich battle cards per vendor.

Runs after b2b_churn_core. Reads persisted artifacts from
b2b_churn_signals, b2b_reviews, and b2b_product_profiles. Builds
deterministic battle cards, runs LLM sales copy generation in parallel,
and persists to b2b_intelligence.
"""

import asyncio
import json
import logging
from datetime import date, datetime
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from ._execution_progress import _update_execution_progress

logger = logging.getLogger("atlas.tasks.b2b_battle_cards")

_STAGE_LOADING_INPUTS = "loading_inputs"
_STAGE_BUILDING = "building_deterministic_cards"
_STAGE_PERSISTING_DETERMINISTIC = "persisting_deterministic"
_STAGE_LLM_OVERLAY = "llm_overlay"
_STAGE_FINALIZING = "finalizing"

_BATTLE_CARD_LLM_FIELDS = (
    "executive_summary",
    "discovery_questions",
    "landmine_questions",
    "objection_handlers",
    "talk_track",
    "recommended_plays",
)

_BATTLE_CARD_SALES_COPY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "executive_summary": {"type": "string"},
        "discovery_questions": {"type": "array", "items": {"type": "string"}},
        "landmine_questions": {"type": "array", "items": {"type": "string"}},
        "objection_handlers": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "objection": {"type": "string"},
                    "acknowledge": {"type": "string"},
                    "pivot": {"type": "string"},
                    "proof_point": {"type": "string"},
                },
                "required": ["objection", "acknowledge", "pivot", "proof_point"],
            },
        },
        "talk_track": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "opening": {"type": "string"},
                "mid_call_pivot": {"type": "string"},
                "closing": {"type": "string"},
            },
            "required": ["opening", "mid_call_pivot", "closing"],
        },
        "recommended_plays": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "play": {"type": "string"},
                    "target_segment": {"type": "string"},
                    "key_message": {"type": "string"},
                    "timing": {"type": "string"},
                },
                "required": ["play", "target_segment", "key_message", "timing"],
            },
        },
    },
    "required": [
        "executive_summary",
        "discovery_questions",
        "landmine_questions",
        "objection_handlers",
        "talk_track",
        "recommended_plays",
    ],
}

def _parse_battle_card_sales_copy(text: str | None) -> dict[str, Any]:
    """Parse battle-card LLM output with truncation recovery enabled."""
    from ...pipelines.llm import parse_json_response

    parsed = parse_json_response(text or "", recover_truncated=True)
    if parsed.get("_parse_fallback"):
        return parsed
    if any(field in parsed for field in _BATTLE_CARD_LLM_FIELDS):
        return parsed
    return {"analysis_text": text or "", "_parse_fallback": True}


def _battle_card_prior_attempt(parsed_copy: dict[str, Any]) -> Any:
    """Convert invalid parse fallbacks into a raw draft for retry prompts."""
    if not isinstance(parsed_copy, dict):
        return parsed_copy
    if not parsed_copy.get("_parse_fallback"):
        return parsed_copy
    raw_text = str(parsed_copy.get("analysis_text") or "").strip()
    return raw_text or {}


def _battle_card_llm_options(cfg: Any) -> dict[str, Any]:
    """Resolve backend-specific call_llm_with_skill options for battle cards."""
    backend = str(getattr(cfg, "battle_card_llm_backend", "auto") or "auto").strip().lower()
    default_field = getattr(type(cfg), "model_fields", {}).get("battle_card_openrouter_model")
    default_model = getattr(default_field, "default", "") if default_field is not None else ""
    model = str(getattr(cfg, "battle_card_openrouter_model", "") or "").strip() or str(default_model or "").strip() or None
    if backend == "anthropic":
        return {
            "workload": "anthropic",
            "try_openrouter": False,
            "openrouter_model": None,
        }
    if backend == "openrouter":
        return {
            "workload": "synthesis",
            "try_openrouter": True,
            "openrouter_model": model,
        }
    return {
        "workload": "synthesis",
        "try_openrouter": True,
        "openrouter_model": model,
    }


def _build_category_council_fallback(card: dict[str, Any]) -> dict[str, Any] | None:
    """Create deterministic category context when no council conclusion exists."""
    eco = card.get("ecosystem_context") or {}
    regime = str(eco.get("market_structure") or "").strip()
    if not eco or not regime:
        return None
    insights: list[dict[str, str]] = [
        {
            "insight": f"Category market structure is {regime}.",
            "evidence": f"market_structure: {regime}",
        },
    ]
    if eco.get("hhi") is not None:
        insights.append({"insight": "Category concentration is visible in the current HHI.", "evidence": f"hhi: {eco['hhi']}"})
    if eco.get("displacement_intensity") is not None:
        insights.append({"insight": "Competitive displacement is active in this category.", "evidence": f"displacement_intensity: {eco['displacement_intensity']}"})
    if eco.get("dominant_archetype"):
        insights.append({"insight": f"{eco['dominant_archetype']} is the dominant churn archetype in this category.", "evidence": f"dominant_archetype: {eco['dominant_archetype']}"})
    return {
        "conclusion": f"{card.get('category') or 'This category'} currently shows {regime} dynamics, so reps should anchor positioning to category-wide pressure instead of a single isolated complaint stream.",
        "market_regime": regime,
        "durability": "uncertain",
        "confidence": 0.0,
        "winner": None,
        "loser": None,
        "key_insights": insights[:5],
    }


def _ecosystem_context_from_analysis(eco_data: Any) -> dict[str, Any] | None:
    """Normalize EcosystemEvidence or dict payloads into battle-card context."""
    if not eco_data:
        return None
    health = eco_data.get("health") if isinstance(eco_data, dict) else getattr(eco_data, "health", eco_data)
    hhi = health.get("hhi") if isinstance(health, dict) else getattr(health, "hhi", None)
    market_structure = health.get("market_structure") if isinstance(health, dict) else getattr(health, "market_structure", None)
    displacement = health.get("displacement_intensity") if isinstance(health, dict) else getattr(health, "displacement_intensity", None)
    archetype = health.get("dominant_archetype") if isinstance(health, dict) else getattr(health, "dominant_archetype", None)
    if hhi is None and displacement is None and not market_structure and not archetype:
        return None
    return {
        "hhi": hhi,
        "market_structure": market_structure,
        "displacement_intensity": displacement,
        "dominant_archetype": archetype,
    }


def _iso_dateish(value: Any) -> str | None:
    """Serialize date/datetime values for persisted card metadata."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _attach_battle_card_provenance(card: dict[str, Any], provenance: dict[str, Any]) -> None:
    """Attach vendor-specific source and review-window metadata to a card."""
    source_dist = provenance.get("source_distribution") or {}
    if source_dist:
        card["source_distribution"] = source_dist
    sample_ids = provenance.get("sample_review_ids") or []
    if sample_ids:
        card["sample_review_ids"] = [str(item) for item in sample_ids[:20]]
    window_start = _iso_dateish(provenance.get("review_window_start"))
    window_end = _iso_dateish(provenance.get("review_window_end"))
    if window_start:
        card["review_window_start"] = window_start
    if window_end:
        card["review_window_end"] = window_end


def _merge_battle_card_provenance(
    primary: dict[str, Any] | None,
    fallback: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge vendor provenance with vault provenance fallback."""
    merged: dict[str, Any] = {}
    for source in (fallback or {}, primary or {}):
        if not isinstance(source, dict):
            continue
        for key in ("source_distribution", "sample_review_ids", "review_window_start", "review_window_end"):
            value = source.get(key)
            if value in (None, "", [], {}):
                continue
            merged[key] = value
    return merged


def _battle_card_persist_summary(card: dict[str, Any]) -> str:
    """Build the persisted summary for deterministic or LLM-enriched cards."""
    vendor = str(card.get("vendor", "") or "")
    return str(card.get("executive_summary") or (
        f"Battle card for {vendor}: "
        f"score {card.get('churn_pressure_score', 0):.0f}, "
        f"{len(card.get('vendor_weaknesses', []))} weaknesses, "
        f"{len(card.get('competitor_differentiators', []))} competitors."
    ))


def _battle_card_source_metadata(
    card: dict[str, Any],
    report_source_review_count: int | None,
    report_source_dist: dict[str, int],
) -> tuple[int | None, dict[str, int]]:
    """Resolve row-level source metadata with vendor provenance fallback."""
    card_source_dist = card.get("source_distribution") or report_source_dist
    if not card_source_dist:
        return report_source_review_count, {}
    card_source_count = sum(int(v or 0) for v in card_source_dist.values())
    return card_source_count, card_source_dist


def _battle_card_llm_model_label(card: dict[str, Any], llm_options: dict[str, Any]) -> str:
    """Choose the persisted llm_model label for the current render state."""
    render_status = str(card.get("llm_render_status", "") or "").strip().lower()
    if render_status == "cached":
        return "pipeline_cached"
    if render_status == "succeeded":
        if llm_options.get("try_openrouter"):
            return str(llm_options.get("openrouter_model") or "openrouter")
        return str(llm_options.get("workload") or "anthropic")
    return "pipeline_deterministic"


async def _persist_battle_card(
    pool: Any,
    *,
    today: date,
    card: dict[str, Any],
    data_density: str,
    report_source_review_count: int | None,
    report_source_dist: dict[str, int],
    llm_model: str,
    status: str = "published",
) -> bool:
    """Persist a battle card row without dropping deterministic sections."""
    vendor = str(card.get("vendor", "") or "")
    if not vendor:
        return False
    persisted_summary = _battle_card_persist_summary(card)
    card["executive_summary"] = persisted_summary
    card_source_count, card_source_dist = _battle_card_source_metadata(
        card,
        report_source_review_count,
        report_source_dist,
    )
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
                      status = EXCLUDED.status,
                      llm_model = EXCLUDED.llm_model,
                      source_review_count = EXCLUDED.source_review_count,
                      source_distribution = EXCLUDED.source_distribution,
                      created_at = now()
        """,
        today,
        "battle_card",
        vendor,
        json.dumps(card, default=str),
        persisted_summary,
        data_density,
        status,
        llm_model,
        card_source_count,
        json.dumps(card_source_dist),
    )
    return True


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
        _aggregate_competitive_disp,
        _build_deterministic_battle_cards,
        _build_pain_lookup,
        _build_competitor_lookup,
        _build_feature_gap_lookup,
        _build_use_case_lookup,
        _build_sentiment_lookup,
        _build_buyer_auth_lookup,
        _build_keyword_spike_lookup,
        _battle_card_provenance_from_evidence_vault,
        _build_deterministic_battle_card_competitive_landscape,
        _build_deterministic_battle_card_weakness_analysis,
        _build_positive_lookup,
        _build_department_lookup,
        _build_usage_duration_lookup,
        _build_timeline_lookup,
        _build_battle_card_locked_facts,
        _canonicalize_vendor,
        _sanitize_battle_card_sales_copy,
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
        _fetch_vendor_provenance,
        _fetch_latest_evidence_vault,
        _fetch_review_text_aggregates,
        _fetch_department_distribution,
        _fetch_contract_context_distribution,
    )
    from .b2b_churn_intelligence import (
        reconstruct_reasoning_lookup,
        reconstruct_cross_vendor_lookup,
    )

    window_days = cfg.intelligence_window_days
    min_reviews = cfg.intelligence_min_reviews

    # --- Phase 1: Parallel data fetch ---
    await _update_execution_progress(
        task,
        stage=_STAGE_LOADING_INPUTS,
        progress_message="Loading battle-card source artifacts.",
    )
    try:
        (
            vendor_scores,
            competitive_disp, pain_dist, feature_gaps,
            price_rates, dm_rates,
            churning_companies, quotable_evidence,
            budget_signals, use_case_dist,
            sentiment_traj, buyer_auth, timeline_signals,
            competitor_reasons, keyword_spikes,
            data_context, vendor_provenance,
            evidence_vault_lookup,
            product_profiles_raw,
            review_text_agg, department_dist, contract_ctx,
        ) = await asyncio.gather(
            _fetch_vendor_churn_scores(pool, window_days, min_reviews),
            _fetch_competitive_displacement(pool, window_days),
            _fetch_pain_distribution(pool, window_days),
            _fetch_feature_gaps(pool, window_days, min_mentions=cfg.feature_gap_min_mentions),
            _fetch_price_complaint_rates(pool, window_days),
            _fetch_dm_churn_rates(pool, window_days),
            _fetch_churning_companies(pool, window_days),
            _fetch_quotable_evidence(pool, window_days, min_urgency=cfg.quotable_phrase_min_urgency),
            _fetch_budget_signals(pool, window_days),
            _fetch_use_case_distribution(pool, window_days),
            _fetch_sentiment_trajectory(pool, window_days),
            _fetch_buyer_authority_summary(pool, window_days),
            _fetch_timeline_signals(pool, window_days),
            _fetch_competitor_reasons(pool, window_days),
            _fetch_keyword_spikes(pool),
            _fetch_data_context(pool, window_days),
            _fetch_vendor_provenance(pool, window_days),
            _fetch_latest_evidence_vault(pool, as_of=today, analysis_window_days=window_days),
            _fetch_product_profiles(pool),
            _fetch_review_text_aggregates(pool, window_days),
            _fetch_department_distribution(pool, window_days),
            _fetch_contract_context_distribution(pool, window_days),
        )
    except Exception:
        logger.exception("Battle card data fetch failed")
        return {"_skip_synthesis": "Data fetch failed"}

    if not vendor_scores:
        return {"_skip_synthesis": "No vendor scores"}
    competitive_disp = _aggregate_competitive_disp(competitive_disp)

    # --- Load reasoning synthesis from pool ---
    reasoning_synthesis_lookup: dict[str, dict] = {}
    try:
        _synth_rows = await pool.fetch(
            """
            SELECT DISTINCT ON (vendor_name)
                   vendor_name, synthesis
            FROM b2b_reasoning_synthesis
            WHERE as_of_date <= $1
              AND analysis_window_days = $2
            ORDER BY vendor_name, as_of_date DESC, created_at DESC
            """,
            today, window_days,
        )
        for sr in _synth_rows:
            vn = _canonicalize_vendor(sr.get("vendor_name") or "")
            if vn:
                data = sr.get("synthesis")
                if isinstance(data, str):
                    import json as _json
                    data = _json.loads(data)
                if isinstance(data, dict):
                    reasoning_synthesis_lookup[vn] = data
        logger.info(
            "Loaded reasoning synthesis for %d vendors",
            len(reasoning_synthesis_lookup),
        )
    except Exception:
        logger.debug("Reasoning synthesis unavailable", exc_info=True)

    # --- Phase 2: Reconstruct reasoning + cross-vendor from DB ---
    reasoning_lookup = await reconstruct_reasoning_lookup(pool, as_of=today)
    xv_lookup = await reconstruct_cross_vendor_lookup(pool, as_of=today)
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
    price_lookup = {r["vendor"]: r["price_complaint_rate"] for r in price_rates}
    dm_lookup = {r["vendor"]: r["dm_churn_rate"] for r in dm_rates}
    company_lookup = {r["vendor"]: r["companies"] for r in churning_companies}
    quote_lookup = {r["vendor"]: r["quotes"] for r in quotable_evidence}
    budget_lookup = {r["vendor"]: {k: v for k, v in r.items() if k != "vendor"} for r in budget_signals}
    sentiment_lookup = _build_sentiment_lookup(sentiment_traj)
    buyer_auth_lookup = _build_buyer_auth_lookup(buyer_auth)
    timeline_lookup = _build_timeline_lookup(timeline_signals)
    keyword_spike_lookup = _build_keyword_spike_lookup(keyword_spikes)
    _complaints_raw, _positives_raw = review_text_agg
    positive_lookup = _build_positive_lookup(_positives_raw)
    department_lookup = _build_department_lookup(department_dist)
    _contract_values_raw, _durations_raw = contract_ctx
    usage_duration_lookup = _build_usage_duration_lookup(_durations_raw)

    product_profile_lookup: dict[str, dict] = {}
    for pp in product_profiles_raw:
        vn = _canonicalize_vendor(pp.get("vendor_name", ""))
        if vn and vn not in product_profile_lookup:
            product_profile_lookup[vn] = pp

    # --- Phase 4: Build deterministic battle cards ---
    await _update_execution_progress(
        task,
        stage=_STAGE_BUILDING,
        progress_message="Building deterministic battle cards.",
        vendors_total=len(vendor_scores),
    )
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
        keyword_spike_lookup=keyword_spike_lookup,
        evidence_vault_lookup=evidence_vault_lookup,
        reasoning_synthesis_lookup=reasoning_synthesis_lookup,
    )

    for card in deterministic_battle_cards:
        vendor = card.get("vendor", "")
        if not vendor:
            continue
        _attach_battle_card_provenance(
            card,
            _merge_battle_card_provenance(
                vendor_provenance.get(vendor, {}),
                _battle_card_provenance_from_evidence_vault(evidence_vault_lookup.get(vendor)),
            ),
        )

    # Enrich with ecosystem context
    try:
        from atlas_brain.reasoning.ecosystem import EcosystemAnalyzer
        eco = EcosystemAnalyzer(pool)
        ecosystem_evidence = await eco.analyze_all_categories()
        for card in deterministic_battle_cards:
            cat = card.get("category", "")
            eco_context = _ecosystem_context_from_analysis(ecosystem_evidence.get(cat))
            if eco_context:
                card["ecosystem_context"] = eco_context
    except Exception:
        logger.debug("Ecosystem enrichment skipped", exc_info=True)

    # Enrich with cross-vendor battle conclusions + resource asymmetry
    for card in deterministic_battle_cards:
        vendor = card.get("vendor", "")
        category = card.get("category", "")
        council = xv_lookup.get("councils", {}).get(category, {})
        if council:
            cc = council.get("conclusion", {})
            card["category_council"] = {
                "conclusion": cc.get("conclusion", ""),
                "market_regime": cc.get("market_regime", ""),
                "durability": cc.get("durability_assessment", ""),
                "confidence": council.get("confidence", 0),
                "winner": cc.get("winner", ""),
                "loser": cc.get("loser", ""),
                "key_insights": cc.get("key_insights", []),
            }
        elif card.get("ecosystem_context"):
            fallback_council = _build_category_council_fallback(card)
            if fallback_council:
                card["category_council"] = fallback_council
        # Battle conclusions involving this vendor
        battles = []
        for pair_key, battle in xv_lookup.get("battles", {}).items():
            if vendor in pair_key:
                bc = battle.get("conclusion", {})
                battles.append({
                    "opponent": [v for v in pair_key if v != vendor][0] if len(pair_key) > 1 else "",
                    "conclusion": bc.get("conclusion", ""),
                    "durability": bc.get("durability_assessment", ""),
                    "confidence": battle.get("confidence", 0),
                    "winner": bc.get("winner", ""),
                    "loser": bc.get("loser", ""),
                    "key_insights": bc.get("key_insights", []),
                })
        if battles:
            card["cross_vendor_battles"] = battles
        # Resource asymmetry involving this vendor
        for pair_key, asym in xv_lookup.get("asymmetries", {}).items():
            if vendor in pair_key:
                card["resource_asymmetry"] = {
                    "opponent": [v for v in pair_key if v != vendor][0] if len(pair_key) > 1 else "",
                    "conclusion": asym.get("conclusion", {}).get("conclusion", ""),
                    "resource_advantage": asym.get("conclusion", {}).get("resource_advantage", ""),
                    "confidence": asym.get("confidence", 0),
                }
                break  # first match is highest confidence (query ordered by confidence DESC)
        card["weakness_analysis"] = _build_deterministic_battle_card_weakness_analysis(card)
        card["competitive_landscape"] = _build_deterministic_battle_card_competitive_landscape(card)

    logger.info("Built %d deterministic battle cards", len(deterministic_battle_cards))
    total_cards = len(deterministic_battle_cards)
    await _update_execution_progress(
        task,
        stage=_STAGE_PERSISTING_DETERMINISTIC,
        progress_current=0,
        progress_total=total_cards,
        progress_message="Persisting deterministic battle cards.",
        cards_built=total_cards,
        cards_persisted=0,
    )

    data_density = json.dumps({"vendors_analyzed": len(vendor_scores)})
    report_source_review_count = data_context.get("reviews_in_analysis_window")
    report_source_dist = {
        src: info["reviews"] for src, info in data_context.get("source_distribution", {}).items()
    }

    cards_persisted = 0
    for card in deterministic_battle_cards:
        card["llm_render_status"] = "pending"
        card.pop("llm_render_error", None)
        try:
            persisted = await _persist_battle_card(
                pool,
                today=today,
                card=card,
                data_density=data_density,
                report_source_review_count=report_source_review_count,
                report_source_dist=report_source_dist,
                llm_model="pipeline_deterministic",
            )
        except Exception:
            logger.exception("Failed to persist deterministic battle card for %s", card.get("vendor"))
        else:
            cards_persisted += int(bool(persisted))
            await _update_execution_progress(
                task,
                stage=_STAGE_PERSISTING_DETERMINISTIC,
                progress_current=cards_persisted,
                progress_total=total_cards,
                progress_message="Persisting deterministic battle cards.",
                cards_built=total_cards,
                cards_persisted=cards_persisted,
            )

    logger.info(
        "Persisted %d/%d deterministic battle cards before LLM rendering",
        cards_persisted,
        len(deterministic_battle_cards),
    )
    await _update_execution_progress(
        task,
        stage=_STAGE_LLM_OVERLAY,
        progress_current=0,
        progress_total=total_cards,
        progress_message="Applying LLM overlay to battle cards.",
        cards_built=total_cards,
        cards_persisted=cards_persisted,
        cards_llm_updated=0,
        llm_failures=0,
        cache_hits=0,
    )

    # --- Phase 5: LLM sales copy (parallel with semantic cache) ---
    from ...pipelines.llm import call_llm_with_skill
    from ...reasoning.semantic_cache import SemanticCache, CacheEntry, compute_evidence_hash

    _bc_cache = SemanticCache(pool)
    bc_llm_failures = 0
    bc_cache_hits = 0
    bc_llm_updates = 0
    bc_overlay_completed = 0
    progress_lock = asyncio.Lock()
    bc_sem = asyncio.Semaphore(cfg.battle_card_llm_concurrency)
    max_attempts = cfg.battle_card_llm_attempts
    retry_delay = cfg.battle_card_llm_retry_delay_seconds
    feedback_limit = cfg.battle_card_llm_feedback_limit
    llm_max_tokens = cfg.battle_card_llm_max_tokens
    llm_temperature = cfg.battle_card_llm_temperature
    llm_timeout = cfg.battle_card_llm_timeout_seconds
    cache_confidence = cfg.battle_card_cache_confidence
    llm_options = _battle_card_llm_options(cfg)

    async def _request_sales_copy(payload: dict[str, Any]) -> dict[str, Any]:
        sales_copy = await asyncio.wait_for(
            asyncio.to_thread(
                call_llm_with_skill,
                "digest/battle_card_sales_copy",
                json.dumps(payload, default=str),
                max_tokens=llm_max_tokens,
                temperature=llm_temperature,
                guided_json=_BATTLE_CARD_SALES_COPY_JSON_SCHEMA,
                response_format={"type": "json_object"},
                workload=llm_options["workload"],
                try_openrouter=llm_options["try_openrouter"],
                openrouter_model=llm_options["openrouter_model"],
            ),
            timeout=llm_timeout,
        )
        return _parse_battle_card_sales_copy(sales_copy)

    async def _enrich_one(card: dict[str, Any]) -> None:
        nonlocal bc_llm_failures, bc_cache_hits, bc_llm_updates, bc_overlay_completed

        card_hash = compute_evidence_hash({
            key: value
            for key, value in card.items()
            if key not in _BATTLE_CARD_LLM_FIELDS and not key.startswith("_")
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
                card["llm_render_status"] = "cached"
                card.pop("llm_render_error", None)
                await _bc_cache.validate(pattern_sig)
                persisted_ok = False
                try:
                    persisted = await _persist_battle_card(
                        pool,
                        today=today,
                        card=card,
                        data_density=data_density,
                        report_source_review_count=report_source_review_count,
                        report_source_dist=report_source_dist,
                        llm_model=_battle_card_llm_model_label(card, llm_options),
                    )
                except Exception:
                    logger.exception("Failed to persist cached battle card for %s", card.get("vendor"))
                else:
                    persisted_ok = bool(persisted)
                async with progress_lock:
                    bc_cache_hits += 1
                    bc_llm_updates += int(persisted_ok)
                    bc_overlay_completed += 1
                    await _update_execution_progress(
                        task,
                        stage=_STAGE_LLM_OVERLAY,
                        progress_current=bc_overlay_completed,
                        progress_total=total_cards,
                        progress_message="Applying LLM overlay to battle cards.",
                        cards_built=total_cards,
                        cards_persisted=cards_persisted,
                        cards_llm_updated=bc_llm_updates,
                        llm_failures=bc_llm_failures,
                        cache_hits=bc_cache_hits,
                    )
                return

        async with bc_sem:
            payload = dict(card)
            payload["locked_facts"] = _build_battle_card_locked_facts(card)
            failure_reasons: list[str] = []
            render_succeeded = False
            for attempt in range(max_attempts):
                try:
                    parsed_copy = await _request_sales_copy(payload)
                except Exception as exc:
                    parsed_copy = {}
                    failure_reasons = [f"transport failure: {type(exc).__name__}"]
                else:
                    if parsed_copy.get("_parse_fallback"):
                        failure_reasons = ["LLM did not return valid JSON"]
                    else:
                        copy_errors = _validate_battle_card_sales_copy(card, parsed_copy)
                        if copy_errors:
                            sanitized_copy = _sanitize_battle_card_sales_copy(card, parsed_copy)
                            sanitized_errors = _validate_battle_card_sales_copy(card, sanitized_copy)
                            if not sanitized_errors:
                                parsed_copy = sanitized_copy
                                copy_errors = []
                        if not copy_errors:
                            for _f in _BATTLE_CARD_LLM_FIELDS:
                                if _f in parsed_copy:
                                    card[_f] = parsed_copy[_f]
                            card["llm_render_status"] = "succeeded"
                            card.pop("llm_render_error", None)
                            render_succeeded = True
                            break
                        failure_reasons = copy_errors
                if attempt + 1 >= max_attempts:
                    card["llm_render_status"] = "failed"
                    if failure_reasons:
                        card["llm_render_error"] = "; ".join(failure_reasons[:3])
                    logger.warning("Battle card rejected for %s: %s",
                                   card.get("vendor"), "; ".join(failure_reasons[:3]))
                    persisted_ok = False
                    try:
                        persisted = await _persist_battle_card(
                            pool,
                            today=today,
                            card=card,
                            data_density=data_density,
                            report_source_review_count=report_source_review_count,
                            report_source_dist=report_source_dist,
                            llm_model=_battle_card_llm_model_label(card, llm_options),
                        )
                    except Exception:
                        logger.exception("Failed to persist rejected battle card for %s", card.get("vendor"))
                    else:
                        persisted_ok = bool(persisted)
                    async with progress_lock:
                        bc_llm_failures += 1
                        bc_llm_updates += int(persisted_ok)
                        bc_overlay_completed += 1
                        await _update_execution_progress(
                            task,
                            stage=_STAGE_LLM_OVERLAY,
                            progress_current=bc_overlay_completed,
                            progress_total=total_cards,
                            progress_message="Applying LLM overlay to battle cards.",
                            cards_built=total_cards,
                            cards_persisted=cards_persisted,
                            cards_llm_updated=bc_llm_updates,
                            llm_failures=bc_llm_failures,
                            cache_hits=bc_cache_hits,
                        )
                    return
                payload = dict(card)
                payload["locked_facts"] = _build_battle_card_locked_facts(card)
                payload["prior_attempt"] = _battle_card_prior_attempt(parsed_copy)
                payload["validation_feedback"] = failure_reasons[:feedback_limit]
                if retry_delay > 0:
                    await asyncio.sleep(retry_delay)

            if not render_succeeded:
                return

            try:
                await _bc_cache.store(CacheEntry(
                    pattern_sig=pattern_sig,
                    pattern_class="battle_card_sales_copy",
                    conclusion={_f: card[_f] for _f in _BATTLE_CARD_LLM_FIELDS if _f in card},
                    confidence=cache_confidence,
                    evidence_hash=card_hash,
                    vendor_name=card.get("vendor"),
                    conclusion_type="sales_copy",
                ))
            except Exception:
                logger.warning("Failed to cache battle card for %s", card.get("vendor"))
            try:
                persisted = await _persist_battle_card(
                    pool,
                    today=today,
                    card=card,
                    data_density=data_density,
                    report_source_review_count=report_source_review_count,
                    report_source_dist=report_source_dist,
                    llm_model=_battle_card_llm_model_label(card, llm_options),
                )
            except Exception:
                logger.exception("Failed to persist enriched battle card for %s", card.get("vendor"))
                persisted_ok = False
            else:
                persisted_ok = bool(persisted)
            async with progress_lock:
                bc_llm_updates += int(persisted_ok)
                bc_overlay_completed += 1
                await _update_execution_progress(
                    task,
                    stage=_STAGE_LLM_OVERLAY,
                    progress_current=bc_overlay_completed,
                    progress_total=total_cards,
                    progress_message="Applying LLM overlay to battle cards.",
                    cards_built=total_cards,
                    cards_persisted=cards_persisted,
                    cards_llm_updated=bc_llm_updates,
                    llm_failures=bc_llm_failures,
                    cache_hits=bc_cache_hits,
                )

    await asyncio.gather(*[_enrich_one(c) for c in deterministic_battle_cards])

    logger.info(
        "Battle card LLM: %d cache hits, %d generated, %d failed (of %d)",
        bc_cache_hits,
        len(deterministic_battle_cards) - bc_cache_hits - bc_llm_failures,
        bc_llm_failures,
        len(deterministic_battle_cards),
    )
    logger.info(
        "Battle card overlay writes: %d/%d cards updated after LLM phase",
        bc_llm_updates,
        len(deterministic_battle_cards),
    )
    await _update_execution_progress(
        task,
        stage=_STAGE_FINALIZING,
        progress_current=total_cards,
        progress_total=total_cards,
        progress_message="Finalizing battle-card execution status.",
        cards_built=total_cards,
        cards_persisted=cards_persisted,
        cards_llm_updated=bc_llm_updates,
        llm_failures=bc_llm_failures,
        cache_hits=bc_cache_hits,
    )

    return {
        "_skip_synthesis": "B2B battle cards complete",
        "cards_built": len(deterministic_battle_cards),
        "cards_persisted": cards_persisted,
        "cards_llm_updated": bc_llm_updates,
        "cache_hits": bc_cache_hits,
        "llm_failures": bc_llm_failures,
        "reasoning_vendors": len(reasoning_lookup),
    }
