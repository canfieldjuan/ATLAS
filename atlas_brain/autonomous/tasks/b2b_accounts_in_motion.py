"""Follow-up task: build per-vendor accounts-in-motion prospecting lists.

Runs after b2b_churn_core. Reads persisted artifacts from b2b_churn_signals,
b2b_reviews, and b2b_company_signals. Merges company profiles from multiple
sources, scores each account, and persists one b2b_intelligence row per vendor
with report_type='accounts_in_motion'.
"""

import asyncio
import json
import logging
from datetime import date
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.tasks.b2b_accounts_in_motion")


# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

_ROLE_SCORES: dict[str, int] = {
    "executive": 20,
    "economic_buyer": 15,
    "champion": 15,
    "evaluator": 10,
}

_STAGE_SCORES: dict[str, int] = {
    "active_purchase": 25,
    "evaluation": 20,
    "renewal_decision": 15,
    "post_purchase": 5,
}


def _compute_account_opportunity_score(account: dict[str, Any]) -> tuple[int, dict[str, int]]:
    """Score an account profile on a 0-100 scale.

    Returns (total_score, component_breakdown).
    """
    # Urgency: (urgency - 5) * 6, clamped [0, 30]
    urgency_raw = float(account.get("urgency") or 0)
    urgency_pts = max(0, min(30, int((urgency_raw - 5) * 6)))

    # Role: DM=20, economic_buyer/champion=15, evaluator=10
    role_pts = 0
    if account.get("decision_maker"):
        role_pts = 20
    else:
        role_level = (account.get("role_level") or "").lower()
        role_pts = _ROLE_SCORES.get(role_level, 0)

    # Stage: active_purchase=25, evaluation=20, renewal_decision=15, post_purchase=5
    stage = (account.get("buying_stage") or "").lower()
    stage_pts = _STAGE_SCORES.get(stage, 0)

    # Seats: 500+=15, 100+=10, 20+=5
    seat_count = account.get("seat_count") or 0
    if seat_count >= 500:
        seat_pts = 15
    elif seat_count >= 100:
        seat_pts = 10
    elif seat_count >= 20:
        seat_pts = 5
    else:
        seat_pts = 0

    # Alternatives: 3+=10, 2=7, 1=4
    alts = account.get("alternatives_considering") or []
    alt_count = len(alts) if isinstance(alts, list) else 0
    if alt_count >= 3:
        alt_pts = 10
    elif alt_count == 2:
        alt_pts = 7
    elif alt_count == 1:
        alt_pts = 4
    else:
        alt_pts = 0

    total = urgency_pts + role_pts + stage_pts + seat_pts + alt_pts
    components = {
        "urgency": urgency_pts,
        "role": role_pts,
        "stage": stage_pts,
        "seats": seat_pts,
        "alternatives": alt_pts,
    }
    return total, components


def _normalize_company_key(company: str | None) -> str:
    """Normalize a company name for dedup key purposes."""
    return (company or "").strip().lower()


# ---------------------------------------------------------------------------
# New lightweight query: company signal metadata
# ---------------------------------------------------------------------------

async def _fetch_apollo_org_lookup(pool) -> dict[str, dict[str, Any]]:
    """Load Apollo org data keyed by normalized company name.

    Includes 'enriched' rows and 'pending' rows that already have a domain
    (org data arrived via person reveal but person search is incomplete).
    """
    try:
        rows = await pool.fetch(
            """
            SELECT company_name_norm, domain, industry,
                   employee_count, annual_revenue_range
            FROM prospect_org_cache
            WHERE status = 'enriched'
               OR (status = 'pending' AND domain IS NOT NULL)
            """
        )
    except Exception:
        logger.debug("prospect_org_cache not available, skipping Apollo lookup")
        return {}
    return {
        r["company_name_norm"]: {
            "domain": r["domain"],
            "industry": r["industry"],
            "employee_count": r["employee_count"],
            "annual_revenue_range": r["annual_revenue_range"],
        }
        for r in rows
    }


async def _fetch_company_signal_metadata(pool, window_days: int = 90) -> list[dict[str, Any]]:
    """Read first_seen_at, last_seen_at, confidence from b2b_company_signals."""
    rows = await pool.fetch(
        """
        SELECT company_name, vendor_name,
               first_seen_at, last_seen_at,
               urgency_score AS confidence
        FROM b2b_company_signals
        WHERE last_seen_at > NOW() - make_interval(days => $1)
        """,
        window_days,
    )
    return [
        {
            "company": r["company_name"],
            "vendor": r["vendor_name"],
            "first_seen": r["first_seen_at"].isoformat() if r["first_seen_at"] else None,
            "last_seen": r["last_seen_at"].isoformat() if r["last_seen_at"] else None,
            "confidence": float(r["confidence"]) if r["confidence"] is not None else None,
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def _merge_company_profiles(
    high_intent: list[dict[str, Any]],
    timeline_signals: list[dict[str, Any]],
    churning_companies: list[dict[str, Any]],
    quotable_evidence: list[dict[str, Any]],
    signal_metadata: list[dict[str, Any]],
    *,
    min_urgency: float = 5.0,
    apollo_org_lookup: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Merge company profiles from multiple data sources.

    Key: (normalize_company_key(company), canonicalized vendor)

    Returns: {key: merged_profile_dict, ...}
    """
    from ._b2b_shared import _canonicalize_vendor

    profiles: dict[tuple[str, str], dict[str, Any]] = {}

    # 1. Base from high_intent -- group by key, take highest-urgency
    for row in high_intent:
        company = row.get("company") or ""
        vendor = _canonicalize_vendor(row.get("vendor") or "")
        if not company or not vendor:
            continue
        key = (_normalize_company_key(company), vendor)

        existing = profiles.get(key)
        urgency = float(row.get("urgency") or 0)

        if urgency < min_urgency:
            continue

        if existing is None:
            alts = row.get("alternatives") or []
            if isinstance(alts, list):
                alts = [a.get("name") if isinstance(a, dict) else str(a) for a in alts if a]
            else:
                alts = []
            review_ids = []
            if row.get("review_id"):
                review_ids.append(row["review_id"])
            profiles[key] = {
                "company": company,
                "vendor": vendor,
                "category": row.get("category"),
                "urgency": urgency,
                "pain_category": row.get("pain"),
                "role_level": row.get("role_level"),
                "decision_maker": bool(row.get("decision_maker")),
                "buying_stage": row.get("buying_stage"),
                "seat_count": row.get("seat_count"),
                "contract_end": row.get("contract_end"),
                "alternatives_considering": alts,
                "source_reviews": review_ids,
                "evidence_count": 1,
                "title": None,
                "company_size": None,
                "industry": None,
                "evaluation_deadline": None,
                "decision_timeline": None,
                "top_quote": None,
                "domain": None,
                "annual_revenue_range": None,
                "first_seen": None,
                "last_seen": None,
                "confidence": None,
            }
        else:
            # Merge: union alternatives, collect review_ids, max urgency
            existing["urgency"] = max(existing["urgency"], urgency)
            existing["evidence_count"] = existing.get("evidence_count", 0) + 1
            if row.get("review_id") and row["review_id"] not in existing["source_reviews"]:
                existing["source_reviews"].append(row["review_id"])
            # Merge alternatives
            new_alts = row.get("alternatives") or []
            if isinstance(new_alts, list):
                for a in new_alts:
                    name = a.get("name") if isinstance(a, dict) else str(a) if a else ""
                    if name and name not in existing["alternatives_considering"]:
                        existing["alternatives_considering"].append(name)
            # Fill nulls from higher-urgency row
            if row.get("pain") and not existing.get("pain_category"):
                existing["pain_category"] = row["pain"]
            if row.get("decision_maker") and not existing.get("decision_maker"):
                existing["decision_maker"] = True
            if row.get("role_level") and not existing.get("role_level"):
                existing["role_level"] = row["role_level"]
            if row.get("buying_stage") and not existing.get("buying_stage"):
                existing["buying_stage"] = row["buying_stage"]
            if row.get("seat_count") and not existing.get("seat_count"):
                existing["seat_count"] = row["seat_count"]
            if row.get("contract_end") and not existing.get("contract_end"):
                existing["contract_end"] = row["contract_end"]

    # 2. Fill from timeline signals
    for row in timeline_signals:
        company = row.get("company") or ""
        vendor = _canonicalize_vendor(row.get("vendor") or "")
        key = (_normalize_company_key(company), vendor)
        prof = profiles.get(key)
        if prof is None:
            continue
        if row.get("contract_end") and not prof.get("contract_end"):
            prof["contract_end"] = row["contract_end"]
        if row.get("evaluation_deadline") and not prof.get("evaluation_deadline"):
            prof["evaluation_deadline"] = row["evaluation_deadline"]
        if row.get("decision_timeline") and not prof.get("decision_timeline"):
            prof["decision_timeline"] = row["decision_timeline"]
        if row.get("title") and not prof.get("title"):
            prof["title"] = row["title"]
        if row.get("company_size") and not prof.get("company_size"):
            prof["company_size"] = row["company_size"]
        if row.get("industry") and not prof.get("industry"):
            prof["industry"] = row["industry"]

    # 3. Fill from churning companies (vendor -> [{company, title, company_size, industry, ...}])
    for vendor_row in churning_companies:
        vendor = _canonicalize_vendor(vendor_row.get("vendor") or "")
        for c in (vendor_row.get("companies") or []):
            company = c.get("company") or ""
            key = (_normalize_company_key(company), vendor)
            prof = profiles.get(key)
            if prof is None:
                continue
            if c.get("title") and not prof.get("title"):
                prof["title"] = c["title"]
            if c.get("company_size") and not prof.get("company_size"):
                prof["company_size"] = c["company_size"]
            if c.get("industry") and not prof.get("industry"):
                prof["industry"] = c["industry"]

    # 4. Attach quote (highest-urgency per vendor, match by company)
    quote_by_vendor: dict[str, list[dict]] = {}
    for qr in quotable_evidence:
        vendor = _canonicalize_vendor(qr.get("vendor") or "")
        for q in (qr.get("quotes") or []):
            quote_by_vendor.setdefault(vendor, []).append(q)

    for key, prof in profiles.items():
        _, vendor = key
        company_key = key[0]
        quotes = quote_by_vendor.get(vendor, [])
        # Find quote matching this company
        best_quote = None
        best_urgency = -1
        for q in quotes:
            qc = _normalize_company_key(q.get("company") or "")
            if qc == company_key:
                qu = float(q.get("urgency") or 0)
                if qu > best_urgency:
                    best_urgency = qu
                    best_quote = q.get("quote")
        if best_quote:
            prof["top_quote"] = best_quote

    # 5. Attach metadata from b2b_company_signals
    meta_lookup: dict[tuple[str, str], dict] = {}
    for m in signal_metadata:
        mk = (_normalize_company_key(m.get("company") or ""), _canonicalize_vendor(m.get("vendor") or ""))
        meta_lookup[mk] = m

    for key, prof in profiles.items():
        meta = meta_lookup.get(key)
        if meta:
            prof["first_seen"] = meta.get("first_seen")
            prof["last_seen"] = meta.get("last_seen")
            if meta.get("confidence") is not None and prof.get("confidence") is None:
                prof["confidence"] = meta["confidence"]

    # 6. Fill from Apollo org cache (lowest priority -- review data wins)
    #    Apollo keys use normalize_company_name (strips legal suffixes),
    #    profile keys use _normalize_company_key (strip+lower only).
    #    Try both to maximize matches.
    if apollo_org_lookup:
        from ...services.company_normalization import normalize_company_name

        for key, prof in profiles.items():
            company_key = key[0]  # _normalize_company_key output
            apollo = apollo_org_lookup.get(company_key)
            if not apollo:
                apollo = apollo_org_lookup.get(normalize_company_name(prof.get("company") or ""))
            if not apollo:
                continue
            if not prof.get("industry") and apollo.get("industry"):
                prof["industry"] = apollo["industry"]
            if not prof.get("company_size") and apollo.get("employee_count"):
                prof["company_size"] = str(apollo["employee_count"])
            if apollo.get("domain"):
                prof["domain"] = apollo["domain"]
            if apollo.get("annual_revenue_range"):
                prof["annual_revenue_range"] = apollo["annual_revenue_range"]

    return profiles


# ---------------------------------------------------------------------------
# Aggregate builder
# ---------------------------------------------------------------------------

def _build_vendor_aggregate(
    vendor: str,
    accounts: list[dict[str, Any]],
    *,
    category: str | None = None,
    reasoning_lookup: dict[str, dict],
    xv_lookup: dict[str, dict],
    feature_gap_lookup: dict[str, list[dict]],
    price_lookup: dict[str, float],
    budget_lookup: dict[str, dict],
    competitor_lookup: dict[str, list[dict]],
) -> dict[str, Any]:
    """Build the full accounts_in_motion aggregate for one vendor."""
    # Archetype from reasoning
    r_info = reasoning_lookup.get(vendor, {})
    archetype = r_info.get("archetype")
    archetype_confidence = r_info.get("confidence", 0)

    # Feature gaps (top 10)
    vendor_gaps = feature_gap_lookup.get(vendor, [])[:10]
    feature_gaps = [
        {"feature": g.get("feature", ""), "mentions": g.get("mentions", 0)}
        for g in vendor_gaps
    ]

    # Pricing pressure
    price_rate = price_lookup.get(vendor, 0)
    budget = budget_lookup.get(vendor, {})
    pricing_pressure = {
        "price_complaint_rate": round(price_rate, 3),
        "price_increase_rate": round(budget.get("price_increase_rate", 0), 3),
        "avg_seat_count": round(budget.get("avg_seat_count") or 0),
    }

    # Cross-vendor context
    top_dest = None
    competitors = competitor_lookup.get(vendor, [])
    if competitors:
        top_dest = competitors[0].get("name")

    # Battle conclusion from xv_lookup
    battle_conclusion = None
    market_regime = None
    for pair_key, battle in xv_lookup.get("battles", {}).items():
        if vendor in pair_key:
            bc = battle.get("conclusion", {})
            battle_conclusion = bc.get("conclusion", "")
            break
    for cat_name, council in xv_lookup.get("councils", {}).items():
        cc = council.get("conclusion", {})
        if cc.get("market_regime"):
            market_regime = cc["market_regime"]
            break

    cross_vendor_context = {
        "top_destination": top_dest,
        "battle_conclusion": battle_conclusion,
        "market_regime": market_regime,
    }

    return {
        "vendor": vendor,
        "category": category,
        "archetype": archetype,
        "archetype_confidence": round(archetype_confidence, 2) if archetype_confidence else None,
        "total_accounts_in_motion": len(accounts),
        "accounts": accounts,
        "pricing_pressure": pricing_pressure,
        "feature_gaps": feature_gaps,
        "cross_vendor_context": cross_vendor_context,
    }


# ---------------------------------------------------------------------------
# Freshness check (same pattern as battle cards)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main task entry point
# ---------------------------------------------------------------------------

async def run(task: ScheduledTask) -> dict[str, Any]:
    """Build per-vendor accounts-in-motion prospecting lists."""
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
        _build_competitor_lookup,
        _build_feature_gap_lookup,
        _canonicalize_vendor,
        _fetch_budget_signals,
        _fetch_churning_companies,
        _fetch_competitive_displacement,
        _fetch_feature_gaps,
        _fetch_high_intent_companies,
        _fetch_price_complaint_rates,
        _fetch_quotable_evidence,
        _fetch_timeline_signals,
        _aggregate_competitive_disp,
    )
    from .b2b_churn_intelligence import (
        reconstruct_reasoning_lookup,
        reconstruct_cross_vendor_lookup,
    )

    window_days = cfg.intelligence_window_days
    min_urgency = cfg.accounts_in_motion_min_urgency
    max_per_vendor = cfg.accounts_in_motion_max_per_vendor

    # --- Phase 1: Parallel data fetch ---
    try:
        (
            high_intent,
            timeline_signals,
            churning_companies,
            quotable_evidence,
            feature_gaps,
            price_rates,
            budget_signals,
            competitive_disp,
            signal_metadata,
            apollo_org_lookup,
        ) = await asyncio.gather(
            _fetch_high_intent_companies(pool, int(min_urgency), window_days),
            _fetch_timeline_signals(pool, window_days),
            _fetch_churning_companies(pool, window_days),
            _fetch_quotable_evidence(pool, window_days, min_urgency=cfg.quotable_phrase_min_urgency),
            _fetch_feature_gaps(pool, window_days, min_mentions=cfg.feature_gap_min_mentions),
            _fetch_price_complaint_rates(pool, window_days),
            _fetch_budget_signals(pool, window_days),
            _fetch_competitive_displacement(pool, window_days),
            _fetch_company_signal_metadata(pool, window_days),
            _fetch_apollo_org_lookup(pool),
        )
    except Exception:
        logger.exception("Accounts in motion data fetch failed")
        return {"_skip_synthesis": "Data fetch failed"}

    if not high_intent:
        logger.info("No high-intent companies found, skipping")
        return {"_skip_synthesis": "No high-intent companies"}

    competitive_disp = _aggregate_competitive_disp(competitive_disp)

    # --- Phase 2: Reconstruct reasoning + cross-vendor ---
    reasoning_lookup = await reconstruct_reasoning_lookup(pool, as_of=today)
    xv_lookup = await reconstruct_cross_vendor_lookup(pool, as_of=today)

    # --- Phase 3: Merge company profiles ---
    merged = _merge_company_profiles(
        high_intent,
        timeline_signals,
        churning_companies,
        quotable_evidence,
        signal_metadata,
        min_urgency=min_urgency,
        apollo_org_lookup=apollo_org_lookup,
    )

    # --- Phase 4: Score each account ---
    for key, prof in merged.items():
        score, components = _compute_account_opportunity_score(prof)
        prof["opportunity_score"] = score
        prof["score_components"] = components

    # --- Phase 5: Build per-vendor aggregates ---
    # Group accounts by vendor
    vendor_accounts: dict[str, list[dict[str, Any]]] = {}
    vendor_categories: dict[str, str | None] = {}
    for key, prof in merged.items():
        vendor = prof["vendor"]
        vendor_accounts.setdefault(vendor, []).append(prof)
        if prof.get("category") and not vendor_categories.get(vendor):
            vendor_categories[vendor] = prof["category"]

    # Sort accounts per vendor by score DESC, limit to max_per_vendor
    for vendor in vendor_accounts:
        vendor_accounts[vendor].sort(key=lambda a: a.get("opportunity_score", 0), reverse=True)
        vendor_accounts[vendor] = vendor_accounts[vendor][:max_per_vendor]

    # Build lookups (canonicalize vendor keys to match merged profiles)
    feature_gap_lookup = _build_feature_gap_lookup(feature_gaps)
    price_lookup = {_canonicalize_vendor(r["vendor"]): r["price_complaint_rate"] for r in price_rates}
    budget_lookup = {
        _canonicalize_vendor(r["vendor"]): {k: v for k, v in r.items() if k != "vendor"}
        for r in budget_signals
    }
    competitor_lookup = _build_competitor_lookup(competitive_disp)

    aggregates: list[dict[str, Any]] = []
    for vendor, accounts in vendor_accounts.items():
        agg = _build_vendor_aggregate(
            vendor,
            accounts,
            category=vendor_categories.get(vendor),
            reasoning_lookup=reasoning_lookup,
            xv_lookup=xv_lookup,
            feature_gap_lookup=feature_gap_lookup,
            price_lookup=price_lookup,
            budget_lookup=budget_lookup,
            competitor_lookup=competitor_lookup,
        )
        aggregates.append(agg)

    # --- Phase 6: Persist ---
    total_accounts = sum(a["total_accounts_in_motion"] for a in aggregates)
    persisted = 0
    for agg in aggregates:
        vendor = agg["vendor"]
        if not vendor:
            continue
        n_accounts = agg["total_accounts_in_motion"]
        top_score = agg["accounts"][0]["opportunity_score"] if agg["accounts"] else 0
        exec_summary = (
            f"Accounts in motion for {vendor}: "
            f"{n_accounts} accounts, "
            f"top score {top_score}, "
            f"archetype {agg.get('archetype') or 'unknown'}."
        )
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
                "accounts_in_motion",
                vendor,
                json.dumps(agg, default=str),
                exec_summary,
                json.dumps({"vendors_analyzed": len(aggregates), "total_accounts": total_accounts}),
                "published",
                "pipeline_deterministic",
                total_accounts,
                json.dumps({}),
            )
            persisted += 1
        except Exception:
            logger.exception("Failed to persist accounts_in_motion for %s", vendor)

    logger.info(
        "Accounts in motion: %d vendors, %d total accounts",
        len(aggregates),
        total_accounts,
    )

    return {
        "_skip_synthesis": "Accounts in motion complete",
        "vendors": len(aggregates),
        "total_accounts": total_accounts,
        "persisted": persisted,
    }
