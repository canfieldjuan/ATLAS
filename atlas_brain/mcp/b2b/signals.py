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
        conditions = [
            "enrichment_status = 'enriched'",
            "(enrichment->>'urgency_score')::numeric >= $1",
            "reviewer_company IS NOT NULL AND reviewer_company != ''",
            "enriched_at > NOW() - make_interval(days => $2)",
        ]
        params: list = [min_urgency, window_days]
        idx = 3

        if vendor_name:
            conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
            params.append(vendor_name)
            idx += 1

        conditions.append(_suppress_predicate('review'))
        capped = min(limit, 100)
        params.append(capped)
        where = " AND ".join(conditions)

        rows = await pool.fetch(
            f"""
            SELECT reviewer_company, vendor_name, product_category,
                   enrichment->'reviewer_context'->>'role_level' AS role_level,
                   (enrichment->'reviewer_context'->>'decision_maker')::boolean AS is_dm,
                   (enrichment->>'urgency_score')::numeric AS urgency,
                   enrichment->>'pain_category' AS pain,
                   enrichment->'competitors_mentioned' AS alternatives,
                   enrichment->'contract_context'->>'contract_value_signal' AS value_signal,
                   CASE WHEN enrichment->'budget_signals'->>'seat_count' ~ '^\\d+$'
                    THEN (enrichment->'budget_signals'->>'seat_count')::int END AS seat_count,
                   enrichment->'use_case'->>'lock_in_level' AS lock_in_level,
                   enrichment->'timeline'->>'contract_end' AS contract_end,
                   enrichment->'buyer_authority'->>'buying_stage' AS buying_stage,
                   reviewer_title, company_size_raw,
                   COALESCE(reviewer_industry, enrichment->'reviewer_context'->>'industry') AS industry
            FROM b2b_reviews
            WHERE {where}
            ORDER BY (enrichment->>'urgency_score')::numeric DESC
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
                "company": r["reviewer_company"],
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

        # Top 5 high-intent companies
        hi_rows = await pool.fetch(
            f"""
            SELECT reviewer_company,
                   (enrichment->>'urgency_score')::numeric AS urgency,
                   enrichment->>'pain_category' AS pain,
                   reviewer_title, company_size_raw,
                   COALESCE(reviewer_industry, enrichment->'reviewer_context'->>'industry') AS industry
            FROM b2b_reviews
            WHERE vendor_name ILIKE '%' || $1 || '%'
              AND enrichment_status = 'enriched'
              AND (enrichment->>'urgency_score')::numeric >= 7
              AND reviewer_company IS NOT NULL AND reviewer_company != ''
              AND {_suppress_predicate('review')}
            ORDER BY (enrichment->>'urgency_score')::numeric DESC
            LIMIT 5
            """,
            vname,
        )

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
    """Run on-demand stratified reasoning for a specific vendor.

    Builds fresh evidence from current DB data, runs the reasoning engine
    (recall/reconstitute/reason), persists the archetype to churn_signals,
    and returns the full reasoning result.

    vendor_name: Vendor to reason about (required)
    force: Bypass reasoning cache and force fresh analysis (default false)
    """
    try:
        pool = get_pool()
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            fetch_vendor_evidence,
            persist_single_vendor_reasoning,
        )
        from atlas_brain.reasoning import get_stratified_reasoner, init_stratified_reasoner
        from atlas_brain.reasoning.tiers import Tier, gather_tier_context

        evidence = await fetch_vendor_evidence(pool, vendor_name)
        if evidence is None:
            return json.dumps({"success": False, "error": f"No churn signal data for vendor: {vendor_name}"})

        reasoner = get_stratified_reasoner()
        if reasoner is None:
            await init_stratified_reasoner(pool)
            reasoner = get_stratified_reasoner()
        if reasoner is None:
            return json.dumps({"success": False, "error": "Reasoning engine unavailable"})

        canonical = evidence.get("vendor_name", vendor_name)
        category = evidence.get("product_category", "")

        tier_ctx = await gather_tier_context(
            reasoner._cache, Tier.VENDOR_ARCHETYPE,
            vendor_name=canonical, product_category=category,
        )
        sr = await reasoner.analyze(
            vendor_name=canonical,
            evidence=evidence,
            product_category=category,
            force_reason=force,
            tier_context=tier_ctx,
        )

        try:
            await persist_single_vendor_reasoning(pool, canonical, sr)
        except Exception:
            pass

        conclusion = sr.conclusion or {}
        return json.dumps({
            "success": True,
            "vendor_name": canonical,
            "mode": sr.mode,
            "cached": sr.cached,
            "confidence": sr.confidence,
            "tokens_used": sr.tokens_used,
            "archetype": conclusion.get("archetype"),
            "risk_level": conclusion.get("risk_level"),
            "executive_summary": conclusion.get("executive_summary"),
            "key_signals": conclusion.get("key_signals", []),
            "falsification_conditions": conclusion.get("falsification_conditions", []),
            "uncertainty_sources": conclusion.get("uncertainty_sources", []),
        }, default=str)
    except Exception as exc:
        logger.exception("reason_vendor error")
        return json.dumps({"success": False, "error": str(exc)[:200]})


@mcp.tool()
async def compare_vendors(
    vendors: list[str],
    force: bool = False,
) -> str:
    """Side-by-side stratified reasoning comparison for multiple vendors.

    Runs reasoning for each vendor concurrently (2-5 vendors) and returns
    a comparison with archetypes, risk levels, and key signals.

    vendors: List of vendor names to compare (2-5 required)
    force: Bypass reasoning cache (default false)
    """
    import asyncio as _aio

    if len(vendors) < 2 or len(vendors) > 5:
        return json.dumps({"success": False, "error": "vendors must contain 2-5 vendor names"})

    try:
        pool = get_pool()
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            fetch_vendor_evidence,
            persist_single_vendor_reasoning,
        )
        from atlas_brain.reasoning import get_stratified_reasoner, init_stratified_reasoner
        from atlas_brain.reasoning.tiers import Tier, gather_tier_context

        reasoner = get_stratified_reasoner()
        if reasoner is None:
            await init_stratified_reasoner(pool)
            reasoner = get_stratified_reasoner()
        if reasoner is None:
            return json.dumps({"success": False, "error": "Reasoning engine unavailable"})

        async def _reason_one(vname: str) -> dict:
            evidence = await fetch_vendor_evidence(pool, vname)
            if evidence is None:
                return {"vendor_name": vname, "error": "No churn signal data"}

            canonical = evidence.get("vendor_name", vname)
            category = evidence.get("product_category", "")
            try:
                tier_ctx = await gather_tier_context(
                    reasoner._cache, Tier.VENDOR_ARCHETYPE,
                    vendor_name=canonical, product_category=category,
                )
                sr = await reasoner.analyze(
                    vendor_name=canonical,
                    evidence=evidence,
                    product_category=category,
                    force_reason=force,
                    tier_context=tier_ctx,
                )
            except Exception as exc:
                return {"vendor_name": canonical, "error": str(exc)[:200]}

            try:
                await persist_single_vendor_reasoning(pool, canonical, sr)
            except Exception:
                pass

            conclusion = sr.conclusion or {}
            return {
                "vendor_name": canonical,
                "mode": sr.mode,
                "cached": sr.cached,
                "confidence": sr.confidence,
                "tokens_used": sr.tokens_used,
                "archetype": conclusion.get("archetype"),
                "risk_level": conclusion.get("risk_level"),
                "executive_summary": conclusion.get("executive_summary"),
                "key_signals": conclusion.get("key_signals", []),
                "falsification_conditions": conclusion.get("falsification_conditions", []),
            }

        results = await _aio.gather(*[_reason_one(v) for v in vendors])
        return json.dumps({"success": True, "vendors": results, "count": len(results)}, default=str)
    except Exception as exc:
        logger.exception("compare_vendors error")
        return json.dumps({"success": False, "error": str(exc)[:200]})
