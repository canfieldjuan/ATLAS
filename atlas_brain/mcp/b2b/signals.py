"""B2B Churn MCP -- signal tools."""
import json
from typing import Optional

from ._shared import (
    _apply_field_overrides,
    _safe_json,
    _suppress_predicate,
    get_pool,
    logger,
)
from .server import mcp


@mcp.tool()
async def list_churn_signals(
    vendor_name: Optional[str] = None,
    min_urgency: float = 0,
    category: Optional[str] = None,
    limit: int = 20,
) -> str:
    """
    Query pre-aggregated weekly vendor churn metrics.

    vendor_name: Filter by vendor name (partial match, case-insensitive)
    min_urgency: Minimum avg_urgency_score threshold (default 0)
    category: Filter by product_category (exact match)
    limit: Maximum results (default 20, cap 100)
    """
    limit = max(1, min(limit, 100))
    min_urgency = max(0.0, min(min_urgency, 10.0))
    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})
        conditions = []
        params = []
        idx = 1

        if vendor_name:
            conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
            params.append(vendor_name)
            idx += 1

        if min_urgency > 0:
            conditions.append(f"avg_urgency_score >= ${idx}")
            params.append(min_urgency)
            idx += 1

        if category:
            conditions.append(f"product_category = ${idx}")
            params.append(category)
            idx += 1

        conditions.append(_suppress_predicate('churn_signal'))
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        capped = min(limit, 100)
        params.append(capped)

        rows = await pool.fetch(
            f"""
            SELECT vendor_name, product_category, total_reviews,
                   churn_intent_count, avg_urgency_score, avg_rating_normalized,
                   nps_proxy, price_complaint_rate, decision_maker_churn_rate,
                   archetype, archetype_confidence, reasoning_risk_level,
                   keyword_spike_count, insider_signal_count,
                   last_computed_at
            FROM b2b_churn_signals
            {where}
            ORDER BY avg_urgency_score DESC
            LIMIT ${idx}
            """,
            *params,
        )

        signals = [
            {
                "vendor_name": r["vendor_name"],
                "product_category": r["product_category"],
                "total_reviews": r["total_reviews"],
                "churn_intent_count": r["churn_intent_count"],
                "avg_urgency_score": float(r["avg_urgency_score"]) if r["avg_urgency_score"] is not None else 0.0,
                "avg_rating_normalized": float(r["avg_rating_normalized"]) if r["avg_rating_normalized"] is not None else None,
                "nps_proxy": float(r["nps_proxy"]) if r["nps_proxy"] is not None else None,
                "price_complaint_rate": float(r["price_complaint_rate"]) if r["price_complaint_rate"] is not None else None,
                "decision_maker_churn_rate": float(r["decision_maker_churn_rate"]) if r["decision_maker_churn_rate"] is not None else None,
                "archetype": r["archetype"],
                "archetype_confidence": float(r["archetype_confidence"]) if r["archetype_confidence"] is not None else None,
                "reasoning_risk_level": r["reasoning_risk_level"],
                "keyword_spike_count": r["keyword_spike_count"],
                "insider_signal_count": r["insider_signal_count"],
                "last_computed_at": r["last_computed_at"],
            }
            for r in rows
        ]

        return json.dumps({"signals": signals, "count": len(signals)}, default=str)
    except Exception as exc:
        logger.exception("list_churn_signals error")
        return json.dumps({"error": "Internal error", "signals": [], "count": 0})


@mcp.tool()
async def get_churn_signal(
    vendor_name: str,
    product_category: Optional[str] = None,
) -> str:
    """
    Get detailed churn signal for a specific vendor including all JSONB arrays.

    vendor_name: Vendor to look up (partial match, case-insensitive)
    product_category: Narrow to a specific product category (optional)
    """
    if not vendor_name or not vendor_name.strip():
        return json.dumps({"success": False, "error": "vendor_name is required"})

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})

        if product_category:
            row = await pool.fetchrow(
                f"""
                SELECT * FROM b2b_churn_signals
                WHERE vendor_name ILIKE '%' || $1 || '%'
                  AND product_category = $2
                  AND {_suppress_predicate('churn_signal')}
                ORDER BY avg_urgency_score DESC
                LIMIT 1
                """,
                vendor_name.strip(),
                product_category,
            )
        else:
            row = await pool.fetchrow(
                f"""
                SELECT * FROM b2b_churn_signals
                WHERE vendor_name ILIKE '%' || $1 || '%'
                  AND {_suppress_predicate('churn_signal')}
                ORDER BY avg_urgency_score DESC
                LIMIT 1
                """,
                vendor_name.strip(),
            )

        if not row:
            return json.dumps({"success": False, "error": "No churn signal found for that vendor"})

        signal = {
            "vendor_name": row["vendor_name"],
            "product_category": row["product_category"],
            "total_reviews": row["total_reviews"],
            "negative_reviews": row["negative_reviews"],
            "churn_intent_count": row["churn_intent_count"],
            "avg_urgency_score": float(row["avg_urgency_score"]) if row["avg_urgency_score"] is not None else 0.0,
            "avg_rating_normalized": float(row["avg_rating_normalized"]) if row["avg_rating_normalized"] is not None else None,
            "nps_proxy": float(row["nps_proxy"]) if row["nps_proxy"] is not None else None,
            "price_complaint_rate": float(row["price_complaint_rate"]) if row["price_complaint_rate"] is not None else None,
            "decision_maker_churn_rate": float(row["decision_maker_churn_rate"]) if row["decision_maker_churn_rate"] is not None else None,
            "top_pain_categories": _safe_json(row["top_pain_categories"]),
            "top_competitors": _safe_json(row["top_competitors"]),
            "top_feature_gaps": _safe_json(row["top_feature_gaps"]),
            "company_churn_list": _safe_json(row["company_churn_list"]),
            "quotable_evidence": _safe_json(row["quotable_evidence"]),
            "top_use_cases": _safe_json(row["top_use_cases"]),
            "top_integration_stacks": _safe_json(row["top_integration_stacks"]),
            "budget_signal_summary": _safe_json(row["budget_signal_summary"]),
            "sentiment_distribution": _safe_json(row["sentiment_distribution"]),
            "buyer_authority_summary": _safe_json(row["buyer_authority_summary"]),
            "timeline_summary": _safe_json(row["timeline_summary"]),
            "source_distribution": _safe_json(row["source_distribution"]),
            "sample_review_ids": [str(uid) for uid in (row["sample_review_ids"] or [])],
            "review_window_start": row["review_window_start"],
            "review_window_end": row["review_window_end"],
            "confidence_score": float(row["confidence_score"]) if row["confidence_score"] is not None else 0,
            "last_computed_at": row["last_computed_at"],
            "created_at": row["created_at"],
            # Reasoning
            "archetype": row["archetype"],
            "archetype_confidence": float(row["archetype_confidence"]) if row["archetype_confidence"] is not None else None,
            "reasoning_mode": row["reasoning_mode"],
            "reasoning_risk_level": row["reasoning_risk_level"],
            "reasoning_executive_summary": row["reasoning_executive_summary"],
            "reasoning_key_signals": _safe_json(row["reasoning_key_signals"]),
            "reasoning_uncertainty_sources": _safe_json(row["reasoning_uncertainty_sources"]),
            "falsification_conditions": _safe_json(row["falsification_conditions"]),
            # Insider signals
            "insider_signal_count": row["insider_signal_count"],
            "insider_org_health_summary": row["insider_org_health_summary"],
            "insider_talent_drain_rate": float(row["insider_talent_drain_rate"]) if row["insider_talent_drain_rate"] is not None else None,
            "insider_quotable_evidence": _safe_json(row["insider_quotable_evidence"]),
            # Keyword / temporal trends
            "keyword_spike_count": row["keyword_spike_count"],
            "keyword_spike_keywords": _safe_json(row["keyword_spike_keywords"]),
            "keyword_trend_summary": row["keyword_trend_summary"],
        }
        signal = await _apply_field_overrides(pool, "churn_signal", str(row["id"]), signal)

        return json.dumps({"success": True, "signal": signal}, default=str)
    except Exception as exc:
        logger.exception("get_churn_signal error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def list_high_intent_companies(
    vendor_name: Optional[str] = None,
    min_urgency: float = 7,
    window_days: int = 30,
    limit: int = 20,
) -> str:
    """
    Companies showing active churn intent from enriched reviews.

    vendor_name: Filter by vendor (partial match, case-insensitive)
    min_urgency: Minimum urgency score threshold (default 7)
    window_days: How far back to look in days (default 30)
    limit: Maximum results (default 20, cap 100)
    """
    limit = max(1, min(limit, 100))
    min_urgency = max(0.0, min(min_urgency, 10.0))
    window_days = max(1, min(window_days, 3650))
    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})
        # DEPRECATED-ENRICHMENT-READ: urgency_score, role_level, decision_maker, pain_category, competitors_mentioned, contract_value_signal, seat_count, lock_in_level, contract_end, buying_stage, industry
        # Migrate to: read_high_intent_companies() from _b2b_shared
        conditions = [
            "r.enrichment_status = 'enriched'",
            "(r.enrichment->>'urgency_score')::numeric >= $1",
            "r.reviewer_company IS NOT NULL AND r.reviewer_company != ''",
            "r.enriched_at > NOW() - make_interval(days => $2)",
        ]
        params: list = [min_urgency, window_days]
        idx = 3

        if vendor_name:
            conditions.append(f"r.vendor_name ILIKE '%' || ${idx} || '%'")
            params.append(vendor_name)
            idx += 1

        conditions.append(_suppress_predicate('review', id_expr='r.id', source_expr='r.source', vendor_expr='r.vendor_name'))
        capped = min(limit, 100)
        params.append(capped)
        where = " AND ".join(conditions)

        # DEPRECATED-ENRICHMENT-READ: urgency_score, pain_category, industry
        # Migrate to: read_high_intent_companies() from _b2b_shared
        rows = await pool.fetch(
            f"""
            SELECT
                COALESCE(
                    CASE
                        WHEN ar.confidence_label IN ('high', 'medium')
                        THEN ar.resolved_company_name
                        ELSE NULL
                    END,
                    r.reviewer_company
                ) AS company,
                r.reviewer_company AS raw_company,
                ar.confidence_label AS resolution_confidence,
                r.vendor_name, r.product_category,
                r.enrichment->'reviewer_context'->>'role_level' AS role_level,
                (r.enrichment->'reviewer_context'->>'decision_maker')::boolean AS is_dm,
                (r.enrichment->>'urgency_score')::numeric AS urgency,
                r.enrichment->>'pain_category' AS pain,
                r.enrichment->'competitors_mentioned' AS alternatives,
                r.enrichment->'contract_context'->>'contract_value_signal' AS value_signal,
                CASE WHEN r.enrichment->'budget_signals'->>'seat_count' ~ '^\\d+$'
                 THEN (r.enrichment->'budget_signals'->>'seat_count')::int END AS seat_count,
                r.enrichment->'use_case'->>'lock_in_level' AS lock_in_level,
                r.enrichment->'timeline'->>'contract_end' AS contract_end,
                r.enrichment->'buyer_authority'->>'buying_stage' AS buying_stage,
                r.reviewer_title, r.company_size_raw,
                COALESCE(poc.industry, r.reviewer_industry,
                         r.enrichment->'reviewer_context'->>'industry') AS industry,
                poc.employee_count AS verified_employee_count,
                poc.country AS company_country,
                poc.domain AS company_domain,
                poc.annual_revenue_range AS revenue_range,
                poc.founded_year,
                poc.total_funding,
                poc.latest_funding_stage,
                poc.headcount_growth_6m,
                poc.headcount_growth_12m,
                poc.headcount_growth_24m,
                poc.publicly_traded_exchange,
                poc.publicly_traded_symbol,
                poc.short_description AS company_description
            FROM b2b_reviews r
            LEFT JOIN b2b_account_resolution ar
                ON ar.review_id = r.id AND ar.resolution_status = 'resolved'
            LEFT JOIN prospect_org_cache poc
                ON poc.company_name_norm = CASE
                    WHEN ar.confidence_label IN ('high', 'medium')
                    THEN ar.normalized_company_name
                    ELSE NULL
                END
            WHERE {where}
            ORDER BY (r.enrichment->>'urgency_score')::numeric DESC
            LIMIT ${idx}
            """,
            *params,
        )

        companies = []
        for r in rows:
            try:
                urgency = float(r["urgency"]) if r["urgency"] is not None else 0
            except (ValueError, TypeError):
                urgency = 0
            companies.append({
                "company": r["company"],
                "raw_company": r["raw_company"],
                "resolution_confidence": r["resolution_confidence"],
                "vendor": r["vendor_name"],
                "category": r["product_category"],
                "role_level": r["role_level"],
                "decision_maker": r["is_dm"],
                "urgency": urgency,
                "pain": r["pain"],
                "alternatives": _safe_json(r["alternatives"]),
                "contract_signal": r["value_signal"],
                "seat_count": r["seat_count"],
                "lock_in_level": r["lock_in_level"],
                "contract_end": r["contract_end"],
                "buying_stage": r["buying_stage"],
                "reviewer_title": r["reviewer_title"],
                "company_size": r["company_size_raw"],
                "industry": r["industry"],
                "verified_employee_count": r["verified_employee_count"],
                "company_country": r["company_country"],
                "company_domain": r["company_domain"],
                "revenue_range": r["revenue_range"],
                "founded_year": r["founded_year"],
                "total_funding": r["total_funding"],
                "funding_stage": r["latest_funding_stage"],
                "headcount_growth_6m": float(r["headcount_growth_6m"]) if r["headcount_growth_6m"] is not None else None,
                "headcount_growth_12m": float(r["headcount_growth_12m"]) if r["headcount_growth_12m"] is not None else None,
                "headcount_growth_24m": float(r["headcount_growth_24m"]) if r["headcount_growth_24m"] is not None else None,
                "publicly_traded": r["publicly_traded_exchange"] or None,
                "ticker": r["publicly_traded_symbol"] or None,
                "company_description": r["company_description"],
            })

        return json.dumps({"companies": companies, "count": len(companies)}, default=str)
    except Exception as exc:
        logger.exception("list_high_intent_companies error")
        return json.dumps({"error": "Internal error", "companies": [], "count": 0})


@mcp.tool()
async def get_vendor_profile(vendor_name: str) -> str:
    """
    Comprehensive vendor risk profile -- joins churn signals with live review counts.

    Returns the churn signal row, live review counts (total, pending enrichment,
    enriched), top 5 recent high-intent companies, and pain distribution.

    vendor_name: Vendor to profile (required)
    """
    if not vendor_name or not vendor_name.strip():
        return json.dumps({"success": False, "error": "vendor_name is required"})

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})
        vname = vendor_name.strip()

        # Churn signal
        signal_row = await pool.fetchrow(
            f"""
            SELECT * FROM b2b_churn_signals
            WHERE vendor_name ILIKE '%' || $1 || '%'
              AND {_suppress_predicate('churn_signal')}
            ORDER BY avg_urgency_score DESC
            LIMIT 1
            """,
            vname,
        )

        # Live review counts
        counts = await pool.fetchrow(
            f"""
            SELECT
                COUNT(*) AS total_reviews,
                COUNT(*) FILTER (WHERE enrichment_status = 'pending') AS pending_enrichment,
                COUNT(*) FILTER (WHERE enrichment_status = 'enriched') AS enriched
            FROM b2b_reviews
            WHERE vendor_name ILIKE '%' || $1 || '%'
              AND {_suppress_predicate('review')}
            """,
            vname,
        )

        # DEPRECATED-ENRICHMENT-READ: urgency_score, pain_category, industry
        # Migrate to: read_high_intent_companies() from _b2b_shared
        # Top 5 high-intent companies
        hi_rows = await pool.fetch(
            f"""
            SELECT COALESCE(
                       CASE
                           WHEN ar.confidence_label IN ('high', 'medium')
                           THEN ar.resolved_company_name
                           ELSE NULL
                       END,
                       r.reviewer_company
                   ) AS reviewer_company,
                   (r.enrichment->>'urgency_score')::numeric AS urgency,
                   r.enrichment->>'pain_category' AS pain,
                   r.reviewer_title, r.company_size_raw,
                   COALESCE(poc.industry, r.reviewer_industry,
                            r.enrichment->'reviewer_context'->>'industry') AS industry,
                   poc.employee_count AS verified_employee_count,
                   poc.country AS company_country,
                   poc.domain AS company_domain,
                   poc.annual_revenue_range AS revenue_range,
                   poc.founded_year,
                   poc.total_funding,
                   poc.latest_funding_stage,
                   poc.headcount_growth_6m,
                   poc.headcount_growth_12m,
                   poc.headcount_growth_24m,
                   poc.publicly_traded_exchange,
                   poc.publicly_traded_symbol,
                   poc.short_description AS company_description
            FROM b2b_reviews r
            LEFT JOIN b2b_account_resolution ar
                ON ar.review_id = r.id AND ar.resolution_status = 'resolved'
            LEFT JOIN prospect_org_cache poc
                ON poc.company_name_norm = CASE
                    WHEN ar.confidence_label IN ('high', 'medium')
                    THEN ar.normalized_company_name
                    ELSE NULL
                END
            WHERE r.vendor_name ILIKE '%' || $1 || '%'
              AND r.enrichment_status = 'enriched'
              AND (r.enrichment->>'urgency_score')::numeric >= 7
              AND r.reviewer_company IS NOT NULL AND r.reviewer_company != ''
              AND {_suppress_predicate('review', id_expr='r.id', source_expr='r.source', vendor_expr='r.vendor_name')}
            ORDER BY (r.enrichment->>'urgency_score')::numeric DESC
            LIMIT 5
            """,
            vname,
        )

        # DEPRECATED-ENRICHMENT-READ: pain_category
        # Migrate to: read_vendor_evidence() from _b2b_shared
        # Pain distribution
        pain_rows = await pool.fetch(
            f"""
            SELECT enrichment->>'pain_category' AS pain, COUNT(*) AS cnt
            FROM b2b_reviews
            WHERE vendor_name ILIKE '%' || $1 || '%'
              AND enrichment_status = 'enriched'
              AND enrichment->>'pain_category' IS NOT NULL
              AND {_suppress_predicate('review')}
            GROUP BY enrichment->>'pain_category'
            ORDER BY cnt DESC
            """,
            vname,
        )

        profile: dict = {"vendor_name": vname}

        if signal_row:
            sig = {
                "avg_urgency_score": float(signal_row["avg_urgency_score"]) if signal_row["avg_urgency_score"] is not None else 0.0,
                "churn_intent_count": signal_row["churn_intent_count"],
                "total_reviews": signal_row["total_reviews"],
                "nps_proxy": float(signal_row["nps_proxy"]) if signal_row["nps_proxy"] is not None else None,
                "price_complaint_rate": float(signal_row["price_complaint_rate"]) if signal_row["price_complaint_rate"] is not None else None,
                "decision_maker_churn_rate": float(signal_row["decision_maker_churn_rate"]) if signal_row["decision_maker_churn_rate"] is not None else None,
                "top_pain_categories": _safe_json(signal_row["top_pain_categories"]),
                "top_competitors": _safe_json(signal_row["top_competitors"]),
                "top_feature_gaps": _safe_json(signal_row["top_feature_gaps"]),
                "top_use_cases": _safe_json(signal_row["top_use_cases"]),
                "top_integration_stacks": _safe_json(signal_row["top_integration_stacks"]),
                "budget_signal_summary": _safe_json(signal_row["budget_signal_summary"]),
                "sentiment_distribution": _safe_json(signal_row["sentiment_distribution"]),
                "buyer_authority_summary": _safe_json(signal_row["buyer_authority_summary"]),
                "timeline_summary": _safe_json(signal_row["timeline_summary"]),
                "confidence_score": float(signal_row["confidence_score"]) if signal_row["confidence_score"] is not None else 0,
                "last_computed_at": signal_row["last_computed_at"],
                # Reasoning
                "archetype": signal_row["archetype"],
                "archetype_confidence": float(signal_row["archetype_confidence"]) if signal_row["archetype_confidence"] is not None else None,
                "reasoning_mode": signal_row["reasoning_mode"],
                "reasoning_risk_level": signal_row["reasoning_risk_level"],
                "reasoning_executive_summary": signal_row["reasoning_executive_summary"],
                "reasoning_key_signals": _safe_json(signal_row["reasoning_key_signals"]),
                "reasoning_uncertainty_sources": _safe_json(signal_row["reasoning_uncertainty_sources"]),
                "falsification_conditions": _safe_json(signal_row["falsification_conditions"]),
                # Insider signals
                "insider_signal_count": signal_row["insider_signal_count"],
                "insider_org_health_summary": signal_row["insider_org_health_summary"],
                "insider_talent_drain_rate": float(signal_row["insider_talent_drain_rate"]) if signal_row["insider_talent_drain_rate"] is not None else None,
                "insider_quotable_evidence": _safe_json(signal_row["insider_quotable_evidence"]),
                # Keyword / temporal trends
                "keyword_spike_count": signal_row["keyword_spike_count"],
                "keyword_spike_keywords": _safe_json(signal_row["keyword_spike_keywords"]),
                "keyword_trend_summary": signal_row["keyword_trend_summary"],
            }
            sig = await _apply_field_overrides(pool, "churn_signal", str(signal_row["id"]), sig)
            profile["churn_signal"] = sig
        else:
            profile["churn_signal"] = None

        profile["review_counts"] = {
            "total": counts["total_reviews"] if counts else 0,
            "pending_enrichment": counts["pending_enrichment"] if counts else 0,
            "enriched": counts["enriched"] if counts else 0,
        }

        profile["high_intent_companies"] = [
            {
                "company": r["reviewer_company"],
                "urgency": float(r["urgency"]) if r["urgency"] is not None else 0,
                "pain": r["pain"],
                "title": r["reviewer_title"],
                "company_size": r["company_size_raw"],
                "industry": r["industry"],
                "verified_employee_count": r["verified_employee_count"],
                "company_country": r["company_country"],
                "company_domain": r["company_domain"],
                "revenue_range": r["revenue_range"],
                "founded_year": r["founded_year"],
                "total_funding": r["total_funding"],
                "funding_stage": r["latest_funding_stage"],
                "headcount_growth_6m": float(r["headcount_growth_6m"]) if r["headcount_growth_6m"] is not None else None,
                "headcount_growth_12m": float(r["headcount_growth_12m"]) if r["headcount_growth_12m"] is not None else None,
                "headcount_growth_24m": float(r["headcount_growth_24m"]) if r["headcount_growth_24m"] is not None else None,
                "publicly_traded": r["publicly_traded_exchange"] or None,
                "ticker": r["publicly_traded_symbol"] or None,
                "company_description": r["company_description"],
            }
            for r in hi_rows
        ]

        profile["pain_distribution"] = [
            {"pain_category": r["pain"], "count": r["cnt"]}
            for r in pain_rows
        ]

        return json.dumps({"success": True, "profile": profile}, default=str)
    except Exception as exc:
        logger.exception("get_vendor_profile error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def reason_vendor(
    vendor_name: str,
    force: bool = False,
) -> str:
    """Return the best available persisted reasoning for a specific vendor.

    vendor_name: Vendor to reason about (required)
    force: Compatibility no-op; returns the latest persisted reasoning
    """
    try:
        _ = force
        pool = get_pool()
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_best_reasoning_view,
            synthesis_view_to_reasoning_entry,
        )
        view = await load_best_reasoning_view(pool, vendor_name)
        if view is None:
            return json.dumps({"success": False, "error": f"No reasoning data for vendor: {vendor_name}"})
        entry = synthesis_view_to_reasoning_entry(view)
        return json.dumps({
            "success": True,
            "vendor_name": view.vendor_name,
            "mode": entry.get("mode"),
            "cached": True,
            "confidence": entry.get("confidence"),
            "tokens_used": 0,
            "archetype": entry.get("archetype"),
            "risk_level": entry.get("risk_level"),
            "executive_summary": entry.get("executive_summary"),
            "key_signals": entry.get("key_signals", []),
            "falsification_conditions": entry.get("falsification_conditions", []),
            "uncertainty_sources": entry.get("uncertainty_sources", []),
        }, default=str)
    except Exception as exc:
        logger.exception("reason_vendor error")
        return json.dumps({"success": False, "error": str(exc)[:200]})


@mcp.tool()
async def compare_vendors(
    vendors: list[str],
    force: bool = False,
) -> str:
    """Side-by-side persisted reasoning comparison for multiple vendors.

    vendors: List of vendor names to compare (2-5 required)
    force: Compatibility no-op; returns latest persisted reasoning
    """
    if len(vendors) < 2 or len(vendors) > 5:
        return json.dumps({"success": False, "error": "vendors must contain 2-5 vendor names"})

    try:
        _ = force
        pool = get_pool()
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_best_reasoning_views,
            synthesis_view_to_reasoning_entry,
        )
        views = await load_best_reasoning_views(pool, vendors)
        results = []
        for requested_name in vendors:
            matched_view = None
            for current_name, view in views.items():
                if str(current_name or "").strip().lower() == str(requested_name or "").strip().lower():
                    matched_view = view
                    break
            if matched_view is None:
                results.append({"vendor_name": requested_name, "error": "No reasoning data"})
                continue
            entry = synthesis_view_to_reasoning_entry(matched_view)
            results.append({
                "vendor_name": matched_view.vendor_name,
                "mode": entry.get("mode"),
                "cached": True,
                "confidence": entry.get("confidence"),
                "tokens_used": 0,
                "archetype": entry.get("archetype"),
                "risk_level": entry.get("risk_level"),
                "executive_summary": entry.get("executive_summary"),
                "key_signals": entry.get("key_signals", []),
                "falsification_conditions": entry.get("falsification_conditions", []),
            })
        return json.dumps({"success": True, "vendors": results, "count": len(results)}, default=str)
    except Exception as exc:
        logger.exception("compare_vendors error")
        return json.dumps({"success": False, "error": str(exc)[:200]})
