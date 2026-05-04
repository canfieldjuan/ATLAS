"""B2B Churn MCP -- signal tools."""
import json
from typing import Optional

from ._shared import (
    _apply_field_overrides,
    _canonical_review_predicate,
    _safe_json,
    _suppress_predicate,
    get_pool,
    logger,
)
from .server import mcp


def _clean_optional_text(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _clean_required_text(value: Optional[str]) -> str | None:
    return _clean_optional_text(value)


def _clean_vendor_names(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    for raw_value in values:
        vendor_name = _clean_optional_text(raw_value)
        if vendor_name is not None:
            cleaned.append(vendor_name)
    return cleaned


def _normalize_vendor_name(value: str | None) -> str:
    return str(value or "").strip().lower()


async def _load_reasoning_views_for_vendors(pool, vendor_names: list[str]) -> dict[str, object]:
    requested = [
        str(vendor_name).strip()
        for vendor_name in vendor_names
        if str(vendor_name or "").strip()
    ]
    if not requested:
        return {}
    try:
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import load_best_reasoning_views

        views = await load_best_reasoning_views(
            pool,
            requested,
        )
    except Exception:
        logger.debug("MCP reasoning view load failed", exc_info=True)
        return {}
    return {
        _normalize_vendor_name(vendor_name): view
        for vendor_name, view in views.items()
        if _normalize_vendor_name(vendor_name)
    }


def _overlay_reasoning_summary_from_view(target: dict, view: object) -> None:
    from atlas_brain.autonomous.tasks._b2b_reasoning_consumer_adapter import reasoning_summary_fields_from_view

    target.update(reasoning_summary_fields_from_view(view))


def _overlay_reasoning_detail_from_view(target: dict, view: object) -> None:
    from atlas_brain.autonomous.tasks._b2b_reasoning_consumer_adapter import reasoning_detail_fields_from_view

    target.update(reasoning_detail_fields_from_view(view))


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
    clean_vendor_name = _clean_optional_text(vendor_name)
    clean_category = _clean_optional_text(category)
    limit = max(1, min(limit, 100))
    min_urgency = max(0.0, min(min_urgency, 10.0))
    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})
        from atlas_brain.autonomous.tasks._b2b_shared import read_vendor_signal_rows

        rows = await read_vendor_signal_rows(
            pool,
            vendor_name_query=clean_vendor_name,
            min_urgency=min_urgency,
            product_category=clean_category,
            exclude_suppressed=True,
            limit=limit,
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
                "archetype": None,
                "archetype_confidence": None,
                "reasoning_mode": None,
                "reasoning_risk_level": None,
                "keyword_spike_count": r["keyword_spike_count"],
                "insider_signal_count": r["insider_signal_count"],
                "last_computed_at": r["last_computed_at"],
            }
            for r in rows
        ]
        reasoning_views = await _load_reasoning_views_for_vendors(
            pool,
            [signal.get("vendor_name", "") for signal in signals],
        )
        for signal in signals:
            view = reasoning_views.get(_normalize_vendor_name(signal.get("vendor_name")))
            if view is not None:
                _overlay_reasoning_summary_from_view(signal, view)

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
    clean_vendor_name = _clean_required_text(vendor_name)
    if clean_vendor_name is None:
        return json.dumps({"success": False, "error": "vendor_name is required"})
    clean_product_category = _clean_optional_text(product_category)

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})

        from atlas_brain.autonomous.tasks._b2b_shared import read_vendor_signal_detail

        row = await read_vendor_signal_detail(
            pool,
            vendor_name_query=clean_vendor_name,
            product_category=clean_product_category,
            exclude_suppressed=True,
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
            "archetype": None,
            "archetype_confidence": None,
            "reasoning_mode": None,
            "reasoning_risk_level": None,
            "reasoning_executive_summary": None,
            "reasoning_key_signals": [],
            "reasoning_uncertainty_sources": [],
            "falsification_conditions": [],
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
        reasoning_views = await _load_reasoning_views_for_vendors(pool, [row["vendor_name"]])
        view = reasoning_views.get(_normalize_vendor_name(row["vendor_name"]))
        if view is not None:
            _overlay_reasoning_detail_from_view(signal, view)
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
    clean_vendor_name = _clean_optional_text(vendor_name)
    limit = max(1, min(limit, 100))
    min_urgency = max(0.0, min(min_urgency, 10.0))
    window_days = max(1, min(window_days, 3650))
    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})
        from atlas_brain.autonomous.tasks._b2b_shared import read_high_intent_companies
        results = await read_high_intent_companies(
            pool,
            min_urgency=min_urgency,
            window_days=window_days,
            vendor_name=clean_vendor_name,
            limit=limit,
        )
        companies = []
        for r in results:
            companies.append({
                "company": r.get("company"),
                "raw_company": r.get("raw_company"),
                "resolution_confidence": r.get("resolution_confidence"),
                "vendor": r.get("vendor"),
                "category": r.get("category"),
                "role_level": r.get("role_level"),
                "decision_maker": r.get("decision_maker"),
                "urgency": r.get("urgency", 0),
                "pain": r.get("pain"),
                "alternatives": r.get("alternatives"),
                "contract_signal": r.get("contract_signal"),
                "seat_count": r.get("seat_count"),
                "lock_in_level": r.get("lock_in_level"),
                "contract_end": r.get("contract_end"),
                "buying_stage": r.get("buying_stage"),
                "reviewer_title": r.get("title"),
                "company_size": r.get("company_size"),
                "industry": r.get("industry"),
                "verified_employee_count": r.get("verified_employee_count"),
                "company_country": r.get("company_country"),
                "company_domain": r.get("company_domain"),
                "revenue_range": r.get("revenue_range"),
                "founded_year": r.get("founded_year"),
                "total_funding": r.get("total_funding"),
                "funding_stage": r.get("funding_stage"),
                "headcount_growth_6m": r.get("headcount_growth_6m"),
                "headcount_growth_12m": r.get("headcount_growth_12m"),
                "headcount_growth_24m": r.get("headcount_growth_24m"),
                "publicly_traded": r.get("publicly_traded"),
                "ticker": r.get("ticker"),
                "company_description": r.get("company_description"),
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
    vname = _clean_required_text(vendor_name)
    if vname is None:
        return json.dumps({"success": False, "error": "vendor_name is required"})

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})

        from atlas_brain.autonomous.tasks._b2b_shared import (
            read_vendor_signal_detail,
            read_high_intent_companies,
            _review_vendor_match_join,
        )

        signal_row = await read_vendor_signal_detail(
            pool,
            vendor_name_query=vname,
            exclude_suppressed=True,
        )

        # Live review counts
        vendor_join = _review_vendor_match_join(
            review_alias="r",
            vendor_param=1,
            output_alias="matched_vm",
        )

        counts = await pool.fetchrow(
            f"""
            SELECT
                COUNT(DISTINCT r.id) AS total_reviews,
                COUNT(DISTINCT r.id) FILTER (WHERE r.enrichment_status = 'pending') AS pending_enrichment,
                COUNT(DISTINCT r.id) FILTER (WHERE r.enrichment_status = 'enriched') AS enriched
            FROM b2b_reviews r
            {vendor_join}
            WHERE {_canonical_review_predicate('r')}
              AND {_suppress_predicate('review', id_expr='r.id', source_expr='r.source', vendor_expr='matched_vm.vendor_name')}
            """,
            vname,
        )

        hi_results = await read_high_intent_companies(
            pool,
            min_urgency=7.0,
            window_days=3650,
            vendor_name=vname,
            limit=5,
        )

        # APPROVED-ENRICHMENT-READ: pain_category
        # Reason: vendor profile pain distribution aggregation
        # Pain distribution
        pain_rows = await pool.fetch(
            f"""
            SELECT r.enrichment->>'pain_category' AS pain, COUNT(*) AS cnt
            FROM b2b_reviews r
            {vendor_join}
            WHERE {_canonical_review_predicate('r')}
              AND r.enrichment_status = 'enriched'
              AND r.enrichment->>'pain_category' IS NOT NULL
              AND {_suppress_predicate('review', id_expr='r.id', source_expr='r.source', vendor_expr='matched_vm.vendor_name')}
            GROUP BY r.enrichment->>'pain_category'
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
                "archetype": None,
                "archetype_confidence": None,
                "reasoning_mode": None,
                "reasoning_risk_level": None,
                "reasoning_executive_summary": None,
                "reasoning_key_signals": [],
                "reasoning_uncertainty_sources": [],
                "falsification_conditions": [],
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
            reasoning_views = await _load_reasoning_views_for_vendors(
                pool,
                [signal_row["vendor_name"]],
            )
            view = reasoning_views.get(_normalize_vendor_name(signal_row["vendor_name"]))
            if view is not None:
                _overlay_reasoning_detail_from_view(sig, view)
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
                "company": r.get("company"),
                "urgency": r.get("urgency", 0),
                "pain": r.get("pain"),
                "title": r.get("title"),
                "company_size": r.get("company_size"),
                "industry": r.get("industry"),
                "verified_employee_count": r.get("verified_employee_count"),
                "company_country": r.get("company_country"),
                "company_domain": r.get("company_domain"),
                "revenue_range": r.get("revenue_range"),
                "founded_year": r.get("founded_year"),
                "total_funding": r.get("total_funding"),
                "funding_stage": r.get("funding_stage"),
                "headcount_growth_6m": r.get("headcount_growth_6m"),
                "headcount_growth_12m": r.get("headcount_growth_12m"),
                "headcount_growth_24m": r.get("headcount_growth_24m"),
                "publicly_traded": r.get("publicly_traded"),
                "ticker": r.get("ticker"),
                "company_description": r.get("company_description"),
            }
            for r in hi_results
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
    clean_vendor_name = _clean_required_text(vendor_name)
    if clean_vendor_name is None:
        return json.dumps({"success": False, "error": "vendor_name is required"})

    try:
        _ = force
        pool = get_pool()
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_best_reasoning_view,
            synthesis_view_to_reasoning_entry,
        )
        view = await load_best_reasoning_view(
            pool,
            clean_vendor_name,
        )
        if view is None:
            return json.dumps({"success": False, "error": f"No reasoning data for vendor: {clean_vendor_name}"})
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
    clean_vendors = _clean_vendor_names(vendors)
    if len(clean_vendors) < 2 or len(clean_vendors) > 5:
        return json.dumps({"success": False, "error": "vendors must contain 2-5 vendor names"})

    try:
        _ = force
        pool = get_pool()
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_best_reasoning_views,
            synthesis_view_to_reasoning_entry,
        )
        views = await load_best_reasoning_views(
            pool,
            clean_vendors,
        )
        results = []
        for requested_name in clean_vendors:
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
