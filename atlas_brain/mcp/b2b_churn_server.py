"""
Atlas B2B Churn Intelligence MCP Server.

Exposes B2B churn intelligence data (reviews, vendor signals, reports,
pipeline health, blog posts, affiliate partners) to any MCP-compatible
client (Claude Desktop, Cursor, custom agents).

Tools:
    list_churn_signals       -- query pre-aggregated weekly vendor churn metrics
    get_churn_signal         -- detailed churn signal for a specific vendor
    list_high_intent_companies -- companies showing active churn intent
    get_vendor_profile       -- comprehensive vendor risk profile
    list_reports             -- list B2B intelligence reports
    get_report               -- fetch full intelligence report by UUID
    search_reviews           -- search enriched reviews with flexible filters
    get_review               -- fetch a single review with full enrichment
    get_pipeline_status      -- enrichment pipeline health snapshot
    get_source_health        -- per-source scrape reliability metrics with trend comparison
    get_source_capabilities  -- capability profiles (access patterns, anti-bot, proxy, quality)
    list_scrape_targets      -- view scrape target configuration and status
    get_product_profile      -- fetch pre-computed product knowledge card
    match_products_tool      -- find best alternatives for a churning company
    list_blog_posts          -- list/filter generated blog posts
    get_blog_post            -- fetch full blog post by ID or slug
    add_scrape_target        -- add a new vendor/source scrape target
    manage_scrape_target     -- update target settings (enabled, priority, pages, interval, metadata)
    delete_scrape_target     -- remove a scrape target and its logs
    list_affiliate_partners  -- list affiliate partner configurations
    list_vendors_registry    -- list all canonical vendors and their aliases
    add_vendor_to_registry   -- add a new vendor to the canonical registry
    add_vendor_alias         -- add an alias to an existing vendor
    list_displacement_edges  -- query persisted competitive displacement edges
    get_displacement_history -- time-series of edge strength for a vendor pair
    list_vendor_pain_points  -- query aggregated vendor pain points with confidence
    list_vendor_use_cases    -- query vendor use cases (modules mentioned in reviews)
    list_vendor_integrations -- query vendor integrations (tools mentioned in reviews)
    list_vendor_buyer_profiles -- query aggregated buyer authority profiles per vendor

Run:
    python -m atlas_brain.mcp.b2b_churn_server          # stdio
    python -m atlas_brain.mcp.b2b_churn_server --sse    # SSE HTTP transport
"""

import json
import logging
import sys
import uuid as _uuid
from contextlib import asynccontextmanager
from typing import Optional

from mcp.server.fastmcp import FastMCP

from ..config import settings
from ..services.scraping.target_validation import is_source_allowed, validate_target_input

logger = logging.getLogger("atlas.mcp.b2b_churn")

VALID_REPORT_TYPES = (
    "weekly_churn_feed",
    "vendor_scorecard",
    "displacement_report",
    "category_overview",
    "exploratory_overview",
    "vendor_comparison",
    "account_comparison",
    "account_deep_dive",
    "vendor_retention",
    "challenger_intel",
)

from atlas_brain.services.scraping.sources import ALL_SOURCES

VALID_SOURCES = ALL_SOURCES


def _is_uuid(value: str) -> bool:
    """Check if a string is a valid UUID."""
    try:
        _uuid.UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


def _safe_json(val):
    """Return val if it's already a list/dict, else try json.loads, else return val as-is."""
    if isinstance(val, (list, dict)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            pass
    return val


@asynccontextmanager
async def _lifespan(server):
    """Initialize DB pool on startup, close on shutdown."""
    from ..storage.database import init_database, close_database

    await init_database()
    logger.info("B2B Churn MCP: DB pool initialized")
    yield
    await close_database()


mcp = FastMCP(
    "atlas-b2b-churn",
    instructions=(
        "B2B churn intelligence server for Atlas. "
        "Query vendor churn signals, search enriched reviews, read intelligence "
        "reports, identify high-intent companies, monitor pipeline health, "
        "manage scrape targets, view blog posts, and list affiliate partners. "
        "Data sourced from 16 review sites: G2, Capterra, TrustRadius, Reddit, "
        "Gartner, GetApp, GitHub, HackerNews, PeerSpot, ProductHunt, Quora, "
        "RSS, StackOverflow, TrustPilot, Twitter/X, YouTube."
    ),
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Tool: list_churn_signals
# ---------------------------------------------------------------------------


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
        from ..storage.database import get_db_pool

        pool = get_db_pool()
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


# ---------------------------------------------------------------------------
# Tool: get_churn_signal
# ---------------------------------------------------------------------------


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
        from ..storage.database import get_db_pool

        pool = get_db_pool()

        if product_category:
            row = await pool.fetchrow(
                """
                SELECT * FROM b2b_churn_signals
                WHERE vendor_name ILIKE '%' || $1 || '%'
                  AND product_category = $2
                ORDER BY avg_urgency_score DESC
                LIMIT 1
                """,
                vendor_name.strip(),
                product_category,
            )
        else:
            row = await pool.fetchrow(
                """
                SELECT * FROM b2b_churn_signals
                WHERE vendor_name ILIKE '%' || $1 || '%'
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
            "last_computed_at": row["last_computed_at"],
            "created_at": row["created_at"],
        }

        return json.dumps({"success": True, "signal": signal}, default=str)
    except Exception as exc:
        logger.exception("get_churn_signal error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: list_high_intent_companies
# ---------------------------------------------------------------------------


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
        from ..storage.database import get_db_pool

        pool = get_db_pool()
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
                   enrichment->'buyer_authority'->>'buying_stage' AS buying_stage
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
            })

        return json.dumps({"companies": companies, "count": len(companies)}, default=str)
    except Exception as exc:
        logger.exception("list_high_intent_companies error")
        return json.dumps({"error": "Internal error", "companies": [], "count": 0})


# ---------------------------------------------------------------------------
# Tool: get_vendor_profile
# ---------------------------------------------------------------------------


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
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        vname = vendor_name.strip()

        # Churn signal
        signal_row = await pool.fetchrow(
            """
            SELECT * FROM b2b_churn_signals
            WHERE vendor_name ILIKE '%' || $1 || '%'
            ORDER BY avg_urgency_score DESC
            LIMIT 1
            """,
            vname,
        )

        # Live review counts
        counts = await pool.fetchrow(
            """
            SELECT
                COUNT(*) AS total_reviews,
                COUNT(*) FILTER (WHERE enrichment_status = 'pending') AS pending_enrichment,
                COUNT(*) FILTER (WHERE enrichment_status = 'enriched') AS enriched
            FROM b2b_reviews
            WHERE vendor_name ILIKE '%' || $1 || '%'
            """,
            vname,
        )

        # Top 5 high-intent companies
        hi_rows = await pool.fetch(
            """
            SELECT reviewer_company,
                   (enrichment->>'urgency_score')::numeric AS urgency,
                   enrichment->>'pain_category' AS pain
            FROM b2b_reviews
            WHERE vendor_name ILIKE '%' || $1 || '%'
              AND enrichment_status = 'enriched'
              AND (enrichment->>'urgency_score')::numeric >= 7
              AND reviewer_company IS NOT NULL AND reviewer_company != ''
            ORDER BY (enrichment->>'urgency_score')::numeric DESC
            LIMIT 5
            """,
            vname,
        )

        # Pain distribution
        pain_rows = await pool.fetch(
            """
            SELECT enrichment->>'pain_category' AS pain, COUNT(*) AS cnt
            FROM b2b_reviews
            WHERE vendor_name ILIKE '%' || $1 || '%'
              AND enrichment_status = 'enriched'
              AND enrichment->>'pain_category' IS NOT NULL
            GROUP BY enrichment->>'pain_category'
            ORDER BY cnt DESC
            """,
            vname,
        )

        profile: dict = {"vendor_name": vname}

        if signal_row:
            profile["churn_signal"] = {
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
                "last_computed_at": signal_row["last_computed_at"],
            }
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


# ---------------------------------------------------------------------------
# Tool: list_reports
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_reports(
    report_type: Optional[str] = None,
    vendor_filter: Optional[str] = None,
    limit: int = 10,
) -> str:
    """
    List B2B intelligence reports.

    report_type: Filter by type (weekly_churn_feed, vendor_scorecard,
                 displacement_report, category_overview, vendor_retention,
                 challenger_intel)
    vendor_filter: Filter by vendor name in report (partial match)
    limit: Maximum results (default 10, cap 50)
    """
    limit = max(1, min(limit, 50))
    if report_type and report_type not in VALID_REPORT_TYPES:
        return json.dumps({"error": f"report_type must be one of {VALID_REPORT_TYPES}", "reports": [], "count": 0})

    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        conditions = []
        params = []
        idx = 1

        if report_type:
            conditions.append(f"report_type = ${idx}")
            params.append(report_type)
            idx += 1

        if vendor_filter:
            conditions.append(f"vendor_filter ILIKE '%' || ${idx} || '%'")
            params.append(vendor_filter)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        capped = min(limit, 50)
        params.append(capped)

        rows = await pool.fetch(
            f"""
            SELECT id, report_date, report_type, executive_summary,
                   vendor_filter, status, created_at
            FROM b2b_intelligence
            {where}
            ORDER BY report_date DESC
            LIMIT ${idx}
            """,
            *params,
        )

        reports = [
            {
                "id": str(r["id"]),
                "report_date": r["report_date"],
                "report_type": r["report_type"],
                "executive_summary": r["executive_summary"],
                "vendor_filter": r["vendor_filter"],
                "status": r["status"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

        return json.dumps({"reports": reports, "count": len(reports)}, default=str)
    except Exception as exc:
        logger.exception("list_reports error")
        return json.dumps({"error": "Internal error", "reports": [], "count": 0})


# ---------------------------------------------------------------------------
# Tool: get_report
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_report(report_id: str) -> str:
    """
    Fetch a full B2B intelligence report by UUID.

    report_id: UUID of the report to retrieve
    """
    if not _is_uuid(report_id):
        return json.dumps({"success": False, "error": "Invalid report_id (must be UUID)"})

    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        row = await pool.fetchrow(
            "SELECT * FROM b2b_intelligence WHERE id = $1",
            _uuid.UUID(report_id),
        )

        if not row:
            return json.dumps({"success": False, "error": "Report not found"})

        report = {
            "id": str(row["id"]),
            "report_date": row["report_date"],
            "report_type": row["report_type"],
            "vendor_filter": row["vendor_filter"],
            "category_filter": row["category_filter"],
            "executive_summary": row["executive_summary"],
            "intelligence_data": _safe_json(row["intelligence_data"]),
            "data_density": _safe_json(row["data_density"]),
            "status": row["status"],
            "llm_model": row["llm_model"],
            "created_at": row["created_at"],
        }

        return json.dumps({"success": True, "report": report}, default=str)
    except Exception as exc:
        logger.exception("get_report error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: search_reviews
# ---------------------------------------------------------------------------


@mcp.tool()
async def search_reviews(
    vendor_name: Optional[str] = None,
    pain_category: Optional[str] = None,
    min_urgency: Optional[float] = None,
    company: Optional[str] = None,
    has_churn_intent: Optional[bool] = None,
    window_days: int = 30,
    limit: int = 20,
) -> str:
    """
    Search enriched reviews with flexible filters.

    vendor_name: Filter by vendor (partial match, case-insensitive)
    pain_category: Filter by pain category (exact match)
    min_urgency: Minimum urgency score
    company: Filter by reviewer company (partial match)
    has_churn_intent: Filter by churn intent flag
    window_days: How far back to look in days (default 30)
    limit: Maximum results (default 20, cap 100)
    """
    limit = max(1, min(limit, 100))
    window_days = max(1, min(window_days, 3650))
    if min_urgency is not None:
        min_urgency = max(0.0, min(min_urgency, 10.0))
    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        conditions = [
            "enrichment_status = 'enriched'",
            "enriched_at > NOW() - make_interval(days => $1)",
        ]
        params: list = [window_days]
        idx = 2

        if vendor_name:
            conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
            params.append(vendor_name)
            idx += 1

        if pain_category:
            conditions.append(f"enrichment->>'pain_category' = ${idx}")
            params.append(pain_category)
            idx += 1

        if min_urgency is not None:
            conditions.append(f"(enrichment->>'urgency_score')::numeric >= ${idx}")
            params.append(min_urgency)
            idx += 1

        if company:
            conditions.append(f"reviewer_company ILIKE '%' || ${idx} || '%'")
            params.append(company)
            idx += 1

        if has_churn_intent is not None:
            conditions.append(
                f"(enrichment->'churn_signals'->>'intent_to_leave')::boolean = ${idx}"
            )
            params.append(has_churn_intent)
            idx += 1

        capped = min(limit, 100)
        params.append(capped)
        where = " AND ".join(conditions)

        rows = await pool.fetch(
            f"""
            SELECT id, vendor_name, product_category, reviewer_company,
                   rating,
                   (enrichment->>'urgency_score')::numeric AS urgency_score,
                   enrichment->>'pain_category' AS pain_category,
                   (enrichment->'churn_signals'->>'intent_to_leave')::boolean AS intent_to_leave,
                   (enrichment->'reviewer_context'->>'decision_maker')::boolean AS decision_maker,
                   enriched_at
            FROM b2b_reviews
            WHERE {where}
            ORDER BY (enrichment->>'urgency_score')::numeric DESC
            LIMIT ${idx}
            """,
            *params,
        )

        reviews = [
            {
                "id": str(r["id"]),
                "vendor_name": r["vendor_name"],
                "product_category": r["product_category"],
                "reviewer_company": r["reviewer_company"],
                "rating": float(r["rating"]) if r["rating"] is not None else None,
                "urgency_score": float(r["urgency_score"]) if r["urgency_score"] is not None else None,
                "pain_category": r["pain_category"],
                "intent_to_leave": r["intent_to_leave"],
                "decision_maker": r["decision_maker"],
                "enriched_at": r["enriched_at"],
            }
            for r in rows
        ]

        return json.dumps({"reviews": reviews, "count": len(reviews)}, default=str)
    except Exception as exc:
        logger.exception("search_reviews error")
        return json.dumps({"error": "Internal error", "reviews": [], "count": 0})


# ---------------------------------------------------------------------------
# Tool: get_review
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_review(review_id: str) -> str:
    """
    Fetch a single review with full enrichment data.

    review_id: UUID of the review to retrieve
    """
    if not _is_uuid(review_id):
        return json.dumps({"success": False, "error": "Invalid review_id (must be UUID)"})

    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        row = await pool.fetchrow(
            "SELECT * FROM b2b_reviews WHERE id = $1",
            _uuid.UUID(review_id),
        )

        if not row:
            return json.dumps({"success": False, "error": "Review not found"})

        review = {
            "id": str(row["id"]),
            "source": row["source"],
            "source_url": row["source_url"],
            "vendor_name": row["vendor_name"],
            "product_name": row["product_name"],
            "product_category": row["product_category"],
            "rating": float(row["rating"]) if row["rating"] is not None else None,
            "summary": row["summary"],
            "review_text": row["review_text"],
            "pros": row["pros"],
            "cons": row["cons"],
            "reviewer_name": row["reviewer_name"],
            "reviewer_title": row["reviewer_title"],
            "reviewer_company": row["reviewer_company"],
            "company_size_raw": row["company_size_raw"],
            "reviewer_industry": row["reviewer_industry"],
            "reviewed_at": row["reviewed_at"],
            "imported_at": row["imported_at"],
            "enrichment": _safe_json(row["enrichment"]),
            "enrichment_status": row["enrichment_status"],
            "enriched_at": row["enriched_at"],
        }

        return json.dumps({"success": True, "review": review}, default=str)
    except Exception as exc:
        logger.exception("get_review error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: get_pipeline_status
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_pipeline_status() -> str:
    """
    Enrichment pipeline health snapshot.

    Returns enrichment counts by status, recent imports (last 24h),
    last enrichment timestamp, active scrape targets, and last scrape time.
    """
    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()

        # Enrichment counts by status
        status_rows = await pool.fetch(
            """
            SELECT enrichment_status, COUNT(*) AS cnt
            FROM b2b_reviews
            GROUP BY enrichment_status
            """
        )
        enrichment_counts = {r["enrichment_status"]: r["cnt"] for r in status_rows}

        # Recent imports + last enrichment
        stats = await pool.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (WHERE imported_at > NOW() - INTERVAL '24 hours') AS recent_imports_24h,
                MAX(enriched_at) AS last_enrichment_at
            FROM b2b_reviews
            """
        )

        # Scrape targets summary
        scrape_stats = await pool.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (WHERE enabled) AS active_scrape_targets,
                MAX(last_scraped_at) AS last_scrape_at
            FROM b2b_scrape_targets
            """
        )

        result = {
            "enrichment_counts": enrichment_counts,
            "recent_imports_24h": stats["recent_imports_24h"] if stats else 0,
            "last_enrichment_at": stats["last_enrichment_at"] if stats else None,
            "active_scrape_targets": scrape_stats["active_scrape_targets"] if scrape_stats else 0,
            "last_scrape_at": scrape_stats["last_scrape_at"] if scrape_stats else None,
        }

        return json.dumps({"success": True, **result}, default=str)
    except Exception as exc:
        logger.exception("get_pipeline_status error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: get_source_health
# ---------------------------------------------------------------------------

_SOURCE_HEALTH_SQL = """
WITH current_window AS (
    SELECT
        source,
        COUNT(*)                                            AS total_scrapes,
        COUNT(*) FILTER (WHERE status = 'success')          AS success_count,
        COUNT(*) FILTER (WHERE status = 'partial')          AS partial_count,
        COUNT(*) FILTER (WHERE status = 'failed')           AS failed_count,
        COUNT(*) FILTER (WHERE status = 'blocked')          AS blocked_count,
        AVG(reviews_found)                                  AS avg_reviews_found,
        AVG(reviews_inserted)                               AS avg_reviews_inserted,
        AVG(duration_ms)                                    AS avg_duration_ms,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95_duration_ms,
        MAX(started_at) FILTER (WHERE status = 'success')   AS last_success_at,
        MAX(started_at)                                     AS last_scrape_at
    FROM b2b_scrape_log
    WHERE started_at >= NOW() - make_interval(days => $1)
    {source_filter_current}
    GROUP BY source
),
prev_window AS (
    SELECT
        source,
        COUNT(*)                                            AS total_scrapes,
        COUNT(*) FILTER (WHERE status = 'success')          AS success_count,
        COUNT(*) FILTER (WHERE status = 'blocked')          AS blocked_count,
        AVG(reviews_found)                                  AS avg_reviews_found
    FROM b2b_scrape_log
    WHERE started_at >= NOW() - make_interval(days => $1 * 2)
      AND started_at <  NOW() - make_interval(days => $1)
    {source_filter_prev}
    GROUP BY source
),
target_counts AS (
    SELECT source, COUNT(*) FILTER (WHERE enabled) AS active_targets
    FROM b2b_scrape_targets
    {target_filter}
    GROUP BY source
)
SELECT
    c.source, c.total_scrapes, c.success_count, c.partial_count,
    c.failed_count, c.blocked_count, c.avg_reviews_found,
    c.avg_reviews_inserted, c.avg_duration_ms, c.p95_duration_ms,
    c.last_success_at, c.last_scrape_at,
    COALESCE(t.active_targets, 0)  AS active_targets,
    p.total_scrapes                AS prev_total_scrapes,
    p.success_count                AS prev_success_count,
    p.blocked_count                AS prev_blocked_count,
    p.avg_reviews_found            AS prev_avg_reviews_found
FROM current_window c
LEFT JOIN prev_window p USING (source)
LEFT JOIN target_counts t USING (source)
ORDER BY c.total_scrapes DESC
"""


@mcp.tool()
async def get_source_health(
    window_days: int = 7,
    source: Optional[str] = None,
) -> str:
    """
    Per-source scrape reliability metrics with trend comparison.

    Aggregates b2b_scrape_log over a configurable window and computes
    success/block rates, yield, duration, and recency per review source.
    Includes trend deltas vs the previous equivalent window.

    window_days: Aggregation window in days (1-30, default 7)
    source: Filter to a single source (g2, capterra, trustradius, etc.)
    """
    window_days = max(1, min(window_days, 30))
    if source:
        source = source.strip().lower()
        if source not in VALID_SOURCES:
            return json.dumps({
                "success": False,
                "error": f"source must be one of {sorted(s.value for s in VALID_SOURCES)}",
            })

    try:
        from ..storage.database import get_db_pool
        from ..services.scraping.sources import display_name as source_display_name

        pool = get_db_pool()

        if source:
            sql = _SOURCE_HEALTH_SQL.format(
                source_filter_current="AND source = $2",
                source_filter_prev="AND source = $2",
                target_filter="WHERE source = $2",
            )
            rows = await pool.fetch(sql, window_days, source)
        else:
            sql = _SOURCE_HEALTH_SQL.format(
                source_filter_current="",
                source_filter_prev="",
                target_filter="",
            )
            rows = await pool.fetch(sql, window_days)

        def _float(v):
            if v is None:
                return None
            try:
                return float(v)
            except (ValueError, TypeError):
                return None

        sources_list = []
        for r in rows:
            total = r["total_scrapes"] or 1
            success_rate = round(r["success_count"] / total, 3)
            block_rate = round(r["blocked_count"] / total, 3)

            prev_total = r["prev_total_scrapes"] or 0
            prev_success_rate = round(r["prev_success_count"] / max(prev_total, 1), 3) if prev_total else None
            prev_block_rate = round(r["prev_blocked_count"] / max(prev_total, 1), 3) if prev_total else None

            sources_list.append({
                "source": r["source"],
                "display_name": source_display_name(r["source"]),
                "total_scrapes": r["total_scrapes"],
                "success_count": r["success_count"],
                "partial_count": r["partial_count"],
                "failed_count": r["failed_count"],
                "blocked_count": r["blocked_count"],
                "success_rate": success_rate,
                "block_rate": block_rate,
                "avg_reviews_found": _float(r["avg_reviews_found"]),
                "avg_reviews_inserted": _float(r["avg_reviews_inserted"]),
                "avg_duration_ms": _float(r["avg_duration_ms"]),
                "p95_duration_ms": _float(r["p95_duration_ms"]),
                "last_success_at": r["last_success_at"],
                "last_scrape_at": r["last_scrape_at"],
                "active_targets": r["active_targets"],
                "trend": {
                    "prev_window_scrapes": prev_total,
                    "prev_success_rate": prev_success_rate,
                    "prev_block_rate": prev_block_rate,
                    "prev_avg_reviews_found": _float(r["prev_avg_reviews_found"]),
                    "success_rate_delta": round(success_rate - prev_success_rate, 3) if prev_success_rate is not None else None,
                    "block_rate_delta": round(block_rate - prev_block_rate, 3) if prev_block_rate is not None else None,
                },
            })

        total_scrapes = sum(s["total_scrapes"] for s in sources_list)
        total_success = sum(s["success_count"] for s in sources_list)
        total_blocked = sum(s["blocked_count"] for s in sources_list)

        result = {
            "success": True,
            "window_days": window_days,
            "sources": sources_list,
            "summary": {
                "total_sources": len(sources_list),
                "total_scrapes": total_scrapes,
                "overall_success_rate": round(total_success / max(total_scrapes, 1), 3),
                "overall_block_rate": round(total_blocked / max(total_scrapes, 1), 3),
                "worst_source": min(sources_list, key=lambda s: s["success_rate"])["source"] if sources_list else None,
                "best_source": max(sources_list, key=lambda s: s["success_rate"])["source"] if sources_list else None,
            },
        }

        return json.dumps(result, default=str)
    except Exception as exc:
        logger.exception("get_source_health error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: get_source_capabilities
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_source_capabilities(
    source: Optional[str] = None,
) -> str:
    """
    Get capability profiles for scrape sources.

    Returns access patterns, anti-bot protection, proxy requirements,
    data quality tier, and concurrency class for each source.

    source: Optional source name to filter (e.g. "g2", "reddit"). Returns all if omitted.
    """
    from ..services.scraping.capabilities import get_all_capabilities, get_capability

    if source:
        source = source.strip().lower()
        profile = get_capability(source)
        if not profile:
            return json.dumps({
                "success": False,
                "error": f"Unknown source '{source}'. Use without source param to list all.",
            })
        return json.dumps({"success": True, "profile": profile.to_dict()})

    all_profiles = get_all_capabilities()
    return json.dumps({
        "success": True,
        "total": len(all_profiles),
        "profiles": [p.to_dict() for p in all_profiles.values()],
    })


# ---------------------------------------------------------------------------
# Tool: list_scrape_targets
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_scrape_targets(
    source: Optional[str] = None,
    enabled_only: bool = True,
    limit: int = 20,
) -> str:
    """
    View scrape target configuration and last run status.

    source: Filter by source (g2, capterra, trustradius, reddit, gartner, getapp, github, hackernews, peerspot, producthunt, quora, rss, stackoverflow, trustpilot, youtube)
    enabled_only: Only show enabled targets (default true)
    limit: Maximum results (default 20, cap 100)
    """
    limit = max(1, min(limit, 100))
    if source and source not in VALID_SOURCES:
        return json.dumps({"error": f"source must be one of {sorted(s.value for s in VALID_SOURCES)}", "targets": [], "count": 0})

    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        conditions = []
        params = []
        idx = 1

        if enabled_only:
            conditions.append("enabled = true")

        if source:
            conditions.append(f"source = ${idx}")
            params.append(source)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        capped = min(limit, 100)
        params.append(capped)

        rows = await pool.fetch(
            f"""
            SELECT id, source, vendor_name, product_name, product_category,
                   enabled, priority, last_scraped_at, last_scrape_status,
                   last_scrape_reviews
            FROM b2b_scrape_targets
            {where}
            ORDER BY priority DESC, vendor_name ASC
            LIMIT ${idx}
            """,
            *params,
        )

        targets = [
            {
                "id": str(r["id"]),
                "source": r["source"],
                "vendor_name": r["vendor_name"],
                "product_name": r["product_name"],
                "product_category": r["product_category"],
                "enabled": r["enabled"],
                "priority": r["priority"],
                "last_scraped_at": r["last_scraped_at"],
                "last_scrape_status": r["last_scrape_status"],
                "last_scrape_reviews": r["last_scrape_reviews"],
            }
            for r in rows
        ]

        return json.dumps({"targets": targets, "count": len(targets)}, default=str)
    except Exception as exc:
        logger.exception("list_scrape_targets error")
        return json.dumps({"error": "Internal error", "targets": [], "count": 0})


# ---------------------------------------------------------------------------
# Tool: get_product_profile
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_product_profile(vendor_name: str) -> str:
    """
    Fetch a pre-computed product profile knowledge card for a vendor.

    vendor_name: Vendor name (fuzzy match, case-insensitive)

    Returns strengths, weaknesses, pain addressed scores, competitive
    positioning, use cases, company size fit, and LLM-generated summary.
    """
    if not vendor_name or not vendor_name.strip():
        return json.dumps({"success": False, "error": "vendor_name is required"})

    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        row = await pool.fetchrow(
            """
            SELECT id, vendor_name, product_category,
                   strengths, weaknesses, pain_addressed,
                   total_reviews_analyzed, avg_rating, recommend_rate, avg_urgency,
                   primary_use_cases, typical_company_size, typical_industries,
                   top_integrations, commonly_compared_to, commonly_switched_from,
                   profile_summary, last_computed_at, created_at
            FROM b2b_product_profiles
            WHERE vendor_name ILIKE '%' || $1 || '%'
            ORDER BY total_reviews_analyzed DESC
            LIMIT 1
            """,
            vendor_name.strip(),
        )

        if not row:
            return json.dumps({"success": False, "error": f"No product profile found for '{vendor_name}'"})

        profile = {
            "id": str(row["id"]),
            "vendor_name": row["vendor_name"],
            "product_category": row["product_category"],
            "strengths": _safe_json(row["strengths"]),
            "weaknesses": _safe_json(row["weaknesses"]),
            "pain_addressed": _safe_json(row["pain_addressed"]),
            "total_reviews_analyzed": row["total_reviews_analyzed"],
            "avg_rating": float(row["avg_rating"]) if row["avg_rating"] is not None else None,
            "recommend_rate": float(row["recommend_rate"]) if row["recommend_rate"] is not None else None,
            "avg_urgency": float(row["avg_urgency"]) if row["avg_urgency"] is not None else None,
            "primary_use_cases": _safe_json(row["primary_use_cases"]),
            "typical_company_size": _safe_json(row["typical_company_size"]),
            "typical_industries": _safe_json(row["typical_industries"]),
            "top_integrations": _safe_json(row["top_integrations"]),
            "commonly_compared_to": _safe_json(row["commonly_compared_to"]),
            "commonly_switched_from": _safe_json(row["commonly_switched_from"]),
            "profile_summary": row["profile_summary"],
            "last_computed_at": row["last_computed_at"],
            "created_at": row["created_at"],
        }

        return json.dumps({"success": True, "profile": profile}, default=str)
    except Exception:
        logger.exception("get_product_profile error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: match_products
# ---------------------------------------------------------------------------


@mcp.tool()
async def match_products_tool(
    churning_from: str,
    pain_categories: Optional[str] = None,
    company_size: Optional[int] = None,
    industry: Optional[str] = None,
    limit: int = 3,
) -> str:
    """
    Find the best product alternatives for a company churning from a vendor.

    Scores all product profiles against the company's pain points using a
    weighted algorithm (pain alignment, displacement evidence, company size
    fit, satisfaction delta, recommend rate).

    churning_from: Vendor name the company is leaving (required)
    pain_categories: JSON array of pain objects, e.g. [{"category": "pricing", "severity": "primary"}]
    company_size: Number of employees (optional)
    industry: Company industry (optional)
    limit: Max results (default 3, cap 10)
    """
    if not churning_from or not churning_from.strip():
        return json.dumps({"success": False, "error": "churning_from is required"})

    limit = max(1, min(limit, 10))

    # Parse pain_categories from JSON string
    pains: list[dict] = []
    if pain_categories:
        try:
            parsed = json.loads(pain_categories)
            if isinstance(parsed, list):
                pains = parsed
        except (json.JSONDecodeError, TypeError):
            return json.dumps({"success": False, "error": "pain_categories must be a valid JSON array"})

    try:
        from ..services.b2b.product_matching import match_products
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        matches = await match_products(
            churning_from=churning_from.strip(),
            pain_categories=pains,
            company_size=company_size,
            industry=industry,
            pool=pool,
            limit=limit,
        )

        return json.dumps({"success": True, "matches": matches, "count": len(matches)}, default=str)
    except Exception:
        logger.exception("match_products error")
        return json.dumps({"success": False, "error": "Internal error", "matches": [], "count": 0})


# ---------------------------------------------------------------------------
# Tool: list_blog_posts
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_blog_posts(
    status: Optional[str] = None,
    topic_type: Optional[str] = None,
    limit: int = 20,
) -> str:
    """
    List generated B2B blog posts.

    status: Filter by status (draft, published)
    topic_type: Filter by topic type (vendor_alternative, vendor_showdown,
                churn_report, migration_guide, vendor_deep_dive,
                market_landscape, pricing_reality_check, switching_story,
                pain_point_roundup, best_fit_guide)
    limit: Maximum results (default 20, cap 50)
    """
    limit = max(1, min(limit, 50))
    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        conditions = []
        params = []
        idx = 1

        if status:
            conditions.append(f"status = ${idx}")
            params.append(status)
            idx += 1

        if topic_type:
            conditions.append(f"topic_type = ${idx}")
            params.append(topic_type)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = await pool.fetch(
            f"""
            SELECT id, slug, title, description, topic_type, tags,
                   status, llm_model, created_at, published_at
            FROM blog_posts
            {where}
            ORDER BY created_at DESC
            LIMIT ${idx}
            """,
            *params,
        )

        posts = [
            {
                "id": str(r["id"]),
                "slug": r["slug"],
                "title": r["title"],
                "description": r["description"],
                "topic_type": r["topic_type"],
                "tags": _safe_json(r["tags"]),
                "status": r["status"],
                "llm_model": r["llm_model"],
                "created_at": r["created_at"],
                "published_at": r["published_at"],
            }
            for r in rows
        ]

        return json.dumps({"posts": posts, "count": len(posts)}, default=str)
    except Exception:
        logger.exception("list_blog_posts error")
        return json.dumps({"error": "Internal error", "posts": [], "count": 0})


# ---------------------------------------------------------------------------
# Tool: get_blog_post
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_blog_post(
    post_id: Optional[str] = None,
    slug: Optional[str] = None,
) -> str:
    """
    Fetch a full blog post by UUID or slug.

    post_id: UUID of the blog post (optional if slug provided)
    slug: URL slug of the blog post (optional if post_id provided)
    """
    if not post_id and not slug:
        return json.dumps({"success": False, "error": "Provide either post_id or slug"})

    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()

        if post_id and _is_uuid(post_id):
            row = await pool.fetchrow(
                "SELECT * FROM blog_posts WHERE id = $1",
                _uuid.UUID(post_id),
            )
        elif slug:
            row = await pool.fetchrow(
                "SELECT * FROM blog_posts WHERE slug = $1",
                slug.strip(),
            )
        else:
            return json.dumps({"success": False, "error": "Invalid post_id (must be UUID) or provide slug"})

        if not row:
            return json.dumps({"success": False, "error": "Blog post not found"})

        post = {
            "id": str(row["id"]),
            "slug": row["slug"],
            "title": row["title"],
            "description": row["description"],
            "topic_type": row["topic_type"],
            "tags": _safe_json(row["tags"]),
            "content": row["content"],
            "charts": _safe_json(row["charts"]),
            "data_context": _safe_json(row["data_context"]),
            "status": row["status"],
            "reviewer_notes": row["reviewer_notes"],
            "llm_model": row["llm_model"],
            "created_at": row["created_at"],
            "published_at": row["published_at"],
        }

        return json.dumps({"success": True, "post": post}, default=str)
    except Exception:
        logger.exception("get_blog_post error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: add_scrape_target
# ---------------------------------------------------------------------------


@mcp.tool()
async def add_scrape_target(
    source: str,
    vendor_name: str,
    product_slug: str,
    product_name: Optional[str] = None,
    product_category: Optional[str] = None,
    max_pages: int = 5,
    priority: int = 0,
    scrape_interval_hours: int = 168,
    metadata_json: Optional[str] = None,
) -> str:
    """
    Add a new scrape target to monitor a vendor on a review source.

    source: Review source -- one of: g2, capterra, trustradius, reddit, gartner,
            getapp, github, hackernews, peerspot, producthunt, quora, rss,
            stackoverflow, trustpilot, twitter, youtube
    vendor_name: Vendor/company name (e.g. "Salesforce")
    product_slug: Format depends on source type:
        URL-slug sources (required -- used in URL construction):
          g2: "salesforce-crm"
          capterra: "61368/Salesforce" (numeric-id/name)
          trustradius: "salesforce-crm"
          gartner: "market-slug/vendor-slug" (slash-separated)
          peerspot: "monday-com"
          getapp: "project-management-software/a/monday-com" (category/a/product)
          producthunt: "my-product" (GraphQL slug)
          trustpilot: "monday.com" (company domain)
        Search sources (informational -- vendor_name is used for search):
          reddit, hackernews, github, youtube, stackoverflow, quora, twitter:
          use vendor name as slug (e.g. "salesforce")
        Special:
          rss: full feed URL (e.g. "https://news.google.com/rss/search?q=salesforce")
    product_name: Optional product variant name
    product_category: Category (e.g. "CRM", "Project Management")
    max_pages: Pages to scrape per run (default 5, max 100)
    priority: Higher = scraped first (default 0, max 100)
    scrape_interval_hours: Re-scrape interval (default 168 = weekly, max 8760)
    metadata_json: Optional JSON string for source-specific config:
        reddit: '{"subreddits": ["sysadmin","projectmanagement"]}'
        twitter: '{"search_terms": ["salesforce down"], "min_likes": 2}'
        youtube: '{"search_terms": ["salesforce review"], "max_videos_per_query": 10}'
        hackernews: '{"min_points": 5, "include_comments": true}'
        github: '{"search_mode": "both", "min_stars": 10}'
        stackoverflow: '{"sites": "stackoverflow,softwarerecs", "min_score": 1}'
        rss: '{"feed_urls": ["https://..."], "keywords": ["migration","switching"]}'
    """
    source = source.strip().lower()
    if source not in VALID_SOURCES:
        return json.dumps({"success": False, "error": f"source must be one of {sorted(s.value for s in VALID_SOURCES)}"})
    if not vendor_name or not vendor_name.strip():
        return json.dumps({"success": False, "error": "vendor_name is required"})
    if not product_slug or not product_slug.strip():
        return json.dumps({"success": False, "error": "product_slug is required"})
    if not is_source_allowed(source, settings.b2b_scrape.source_allowlist):
        return json.dumps({
            "success": False,
            "error": (
                f"Source '{source}' is currently disabled by "
                "ATLAS_B2B_SCRAPE_SOURCE_ALLOWLIST"
            ),
        })

    try:
        source, product_slug = validate_target_input(source, product_slug)
    except ValueError as exc:
        return json.dumps({"success": False, "error": str(exc)})

    max_pages = max(1, min(max_pages, 100))
    priority = max(0, min(priority, 100))
    scrape_interval_hours = max(1, min(scrape_interval_hours, 8760))

    # Resolve to canonical vendor name
    from ..services.vendor_registry import resolve_vendor_name
    vendor_name = await resolve_vendor_name(vendor_name)

    meta = {}
    if metadata_json:
        try:
            meta = json.loads(metadata_json)
            if not isinstance(meta, dict):
                return json.dumps({"success": False, "error": "metadata_json must be a JSON object"})
        except (json.JSONDecodeError, TypeError):
            return json.dumps({"success": False, "error": "Invalid metadata_json"})

    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()

        # Check for duplicate
        existing = await pool.fetchrow(
            "SELECT id FROM b2b_scrape_targets WHERE source = $1 AND product_slug = $2",
            source, product_slug,
        )
        if existing:
            return json.dumps({
                "success": False,
                "error": f"Target already exists for {source}/{product_slug} (id: {existing['id']})",
            })

        row = await pool.fetchrow(
            """
            INSERT INTO b2b_scrape_targets
                (source, vendor_name, product_name, product_slug, product_category,
                 max_pages, priority, scrape_interval_hours, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
            RETURNING id, source, vendor_name, product_slug, enabled, priority
            """,
            source,
            vendor_name.strip(),
            product_name.strip() if product_name else None,
            product_slug,
            product_category.strip() if product_category else None,
            max_pages,
            priority,
            scrape_interval_hours,
            json.dumps(meta),
        )

        return json.dumps({
            "success": True,
            "target": {
                "id": str(row["id"]),
                "source": row["source"],
                "vendor_name": row["vendor_name"],
                "product_slug": row["product_slug"],
                "enabled": row["enabled"],
                "priority": row["priority"],
            },
        }, default=str)
    except Exception:
        logger.exception("add_scrape_target error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: manage_scrape_target
# ---------------------------------------------------------------------------


@mcp.tool()
async def manage_scrape_target(
    target_id: str,
    enabled: Optional[bool] = None,
    priority: Optional[int] = None,
    max_pages: Optional[int] = None,
    scrape_interval_hours: Optional[int] = None,
    metadata_json: Optional[str] = None,
) -> str:
    """
    Update a scrape target's settings.

    target_id: UUID of the scrape target (required)
    enabled: Set to true/false to enable/disable
    priority: Set priority (0-100, higher = scraped first)
    max_pages: Pages to scrape per run (1-100)
    scrape_interval_hours: Re-scrape interval in hours (1-8760)
    metadata_json: Replace source-specific config JSON (e.g. subreddits for reddit)
    """
    if not _is_uuid(target_id):
        return json.dumps({"success": False, "error": "Invalid target_id (must be UUID)"})

    if all(v is None for v in [enabled, priority, max_pages, scrape_interval_hours, metadata_json]):
        return json.dumps({"success": False, "error": "Provide at least one field to update"})

    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()

        sets = ["updated_at = NOW()"]
        params = []
        idx = 1

        if enabled is not None:
            sets.append(f"enabled = ${idx}")
            params.append(enabled)
            idx += 1

        if priority is not None:
            sets.append(f"priority = ${idx}")
            params.append(max(0, min(priority, 100)))
            idx += 1

        if max_pages is not None:
            sets.append(f"max_pages = ${idx}")
            params.append(max(1, min(max_pages, 100)))
            idx += 1

        if scrape_interval_hours is not None:
            sets.append(f"scrape_interval_hours = ${idx}")
            params.append(max(1, min(scrape_interval_hours, 8760)))
            idx += 1

        if metadata_json is not None:
            try:
                meta = json.loads(metadata_json)
                if not isinstance(meta, dict):
                    return json.dumps({"success": False, "error": "metadata_json must be a JSON object"})
            except (json.JSONDecodeError, TypeError):
                return json.dumps({"success": False, "error": "Invalid metadata_json"})
            sets.append(f"metadata = ${idx}::jsonb")
            params.append(json.dumps(meta))
            idx += 1

        params.append(_uuid.UUID(target_id))

        row = await pool.fetchrow(
            f"""
            UPDATE b2b_scrape_targets
            SET {', '.join(sets)}
            WHERE id = ${idx}
            RETURNING id, source, vendor_name, product_name, product_slug,
                      enabled, priority, max_pages, scrape_interval_hours
            """,
            *params,
        )

        if not row:
            return json.dumps({"success": False, "error": "Target not found"})

        return json.dumps({
            "success": True,
            "target": {
                "id": str(row["id"]),
                "source": row["source"],
                "vendor_name": row["vendor_name"],
                "product_name": row["product_name"],
                "product_slug": row["product_slug"],
                "enabled": row["enabled"],
                "priority": row["priority"],
                "max_pages": row["max_pages"],
                "scrape_interval_hours": row["scrape_interval_hours"],
            },
        }, default=str)
    except Exception:
        logger.exception("manage_scrape_target error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: delete_scrape_target
# ---------------------------------------------------------------------------


@mcp.tool()
async def delete_scrape_target(target_id: str) -> str:
    """
    Delete a scrape target and its associated scrape logs.

    target_id: UUID of the scrape target to delete (required)
    """
    if not _is_uuid(target_id):
        return json.dumps({"success": False, "error": "Invalid target_id (must be UUID)"})

    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()

        # Get target info before deleting
        row = await pool.fetchrow(
            "SELECT source, vendor_name, product_slug FROM b2b_scrape_targets WHERE id = $1",
            _uuid.UUID(target_id),
        )
        if not row:
            return json.dumps({"success": False, "error": "Target not found"})

        # Delete logs first (FK has no CASCADE), then target -- in a transaction
        async with pool.transaction() as conn:
            await conn.execute(
                "DELETE FROM b2b_scrape_log WHERE target_id = $1",
                _uuid.UUID(target_id),
            )
            await conn.execute(
                "DELETE FROM b2b_scrape_targets WHERE id = $1",
                _uuid.UUID(target_id),
            )

        return json.dumps({
            "success": True,
            "deleted": {
                "id": target_id,
                "source": row["source"],
                "vendor_name": row["vendor_name"],
                "product_slug": row["product_slug"],
            },
        })
    except Exception:
        logger.exception("delete_scrape_target error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: list_affiliate_partners
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_affiliate_partners(
    category: Optional[str] = None,
    enabled_only: bool = True,
    limit: int = 20,
) -> str:
    """
    List affiliate partner configurations.

    category: Filter by product category (partial match, case-insensitive)
    enabled_only: Only show enabled partners (default true)
    limit: Maximum results (default 20, cap 50)
    """
    limit = max(1, min(limit, 50))
    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        conditions = []
        params = []
        idx = 1

        if enabled_only:
            conditions.append("enabled = true")

        if category:
            conditions.append(f"category ILIKE '%' || ${idx} || '%'")
            params.append(category)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = await pool.fetch(
            f"""
            SELECT id, name, product_name, product_aliases, category,
                   affiliate_url, commission_type, commission_value,
                   enabled, created_at
            FROM affiliate_partners
            {where}
            ORDER BY name ASC
            LIMIT ${idx}
            """,
            *params,
        )

        partners = [
            {
                "id": str(r["id"]),
                "name": r["name"],
                "product_name": r["product_name"],
                "product_aliases": list(r["product_aliases"]) if r["product_aliases"] else [],
                "category": r["category"],
                "affiliate_url": r["affiliate_url"],
                "commission_type": r["commission_type"],
                "commission_value": r["commission_value"],
                "enabled": r["enabled"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

        return json.dumps({"partners": partners, "count": len(partners)}, default=str)
    except Exception:
        logger.exception("list_affiliate_partners error")
        return json.dumps({"error": "Internal error", "partners": [], "count": 0})


# ---------------------------------------------------------------------------
# Tool: list_vendors_registry
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_vendors_registry(limit: int = 100) -> str:
    """
    List all canonical vendors in the vendor registry with their aliases.

    limit: Maximum results (default 100, cap 500)
    """
    limit = max(1, min(limit, 500))
    try:
        from ..services.vendor_registry import list_vendors

        vendors = await list_vendors()
        result = [
            {
                "id": str(v["id"]),
                "canonical_name": v["canonical_name"],
                "aliases": list(v["aliases"]) if isinstance(v["aliases"], list) else [],
                "created_at": v["created_at"],
                "updated_at": v["updated_at"],
            }
            for v in vendors[:limit]
        ]
        return json.dumps({"vendors": result, "count": len(result)}, default=str)
    except Exception:
        logger.exception("list_vendors_registry error")
        return json.dumps({"error": "Internal error", "vendors": [], "count": 0})


# ---------------------------------------------------------------------------
# Tool: add_vendor_to_registry
# ---------------------------------------------------------------------------


@mcp.tool()
async def add_vendor_to_registry(
    canonical_name: str,
    aliases: Optional[str] = None,
) -> str:
    """
    Add or update a vendor in the canonical vendor registry.

    canonical_name: The official vendor name (e.g. "Salesforce")
    aliases: Comma-separated lowercase aliases (e.g. "sf,sfdc,salesforce.com")
    """
    if not canonical_name or not canonical_name.strip():
        return json.dumps({"success": False, "error": "canonical_name is required"})
    try:
        from ..services.vendor_registry import add_vendor

        alias_list = []
        if aliases:
            alias_list = [a.strip() for a in aliases.split(",") if a.strip()]

        row = await add_vendor(canonical_name.strip(), alias_list)
        return json.dumps({
            "success": True,
            "vendor": {
                "id": str(row["id"]),
                "canonical_name": row["canonical_name"],
                "aliases": list(row["aliases"]) if isinstance(row["aliases"], list) else [],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            },
        }, default=str)
    except Exception:
        logger.exception("add_vendor_to_registry error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: add_vendor_alias
# ---------------------------------------------------------------------------


@mcp.tool()
async def add_vendor_alias(
    canonical_name: str,
    alias: str,
) -> str:
    """
    Add an alias to an existing vendor in the registry.

    canonical_name: Existing canonical vendor name (e.g. "Salesforce")
    alias: New alias to add (e.g. "salesforce.com")
    """
    if not canonical_name or not canonical_name.strip():
        return json.dumps({"success": False, "error": "canonical_name is required"})
    if not alias or not alias.strip():
        return json.dumps({"success": False, "error": "alias is required"})
    try:
        from ..services.vendor_registry import add_alias

        row = await add_alias(canonical_name.strip(), alias.strip())
        if row is None:
            return json.dumps({
                "success": False,
                "error": f"Vendor '{canonical_name}' not found in registry",
            })
        return json.dumps({
            "success": True,
            "vendor": {
                "id": str(row["id"]),
                "canonical_name": row["canonical_name"],
                "aliases": list(row["aliases"]) if isinstance(row["aliases"], list) else [],
                "updated_at": row["updated_at"],
            },
        }, default=str)
    except Exception:
        logger.exception("add_vendor_alias error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Displacement edges + company signals
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_displacement_edges(
    from_vendor: Optional[str] = None,
    to_vendor: Optional[str] = None,
    min_strength: Optional[str] = None,
    min_confidence: Optional[float] = None,
    window_days: int = 90,
    limit: int = 50,
) -> str:
    """
    Query persisted competitive displacement edges (vendor A -> vendor B flows).

    from_vendor: Filter by source vendor losing customers (case-insensitive partial match)
    to_vendor: Filter by destination vendor gaining customers (case-insensitive partial match)
    min_strength: Minimum signal strength: 'strong', 'moderate', or 'emerging'
    min_confidence: Minimum confidence score (0.0-1.0)
    window_days: Only edges computed within this many days (default 90)
    limit: Max results (default 50, max 200)
    """
    limit = min(max(limit, 1), 200)
    strength_order = {"strong": 3, "moderate": 2, "emerging": 1}
    if min_strength and min_strength not in strength_order:
        return json.dumps({"error": f"Invalid min_strength: {min_strength}. Use strong/moderate/emerging"})

    try:
        from ..storage.database import get_db_pool
        pool = get_db_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        conditions = ["computed_date > NOW() - make_interval(days => $1)"]
        params: list = [window_days]
        idx = 2

        if from_vendor:
            conditions.append(f"from_vendor ILIKE '%' || ${idx} || '%'")
            params.append(from_vendor)
            idx += 1

        if to_vendor:
            conditions.append(f"to_vendor ILIKE '%' || ${idx} || '%'")
            params.append(to_vendor)
            idx += 1

        if min_strength:
            min_val = strength_order[min_strength]
            allowed = [k for k, v in strength_order.items() if v >= min_val]
            conditions.append(f"signal_strength = ANY(${idx}::text[])")
            params.append(allowed)
            idx += 1

        if min_confidence is not None:
            conditions.append(f"confidence_score >= ${idx}")
            params.append(min_confidence)
            idx += 1

        where = " AND ".join(conditions)

        rows = await pool.fetch(
            f"""
            SELECT id, from_vendor, to_vendor, mention_count,
                   primary_driver, signal_strength, key_quote,
                   source_distribution, sample_review_ids,
                   confidence_score, computed_date, report_id, created_at
            FROM b2b_displacement_edges
            WHERE {where}
            ORDER BY confidence_score DESC, mention_count DESC
            LIMIT ${idx}
            """,
            *params,
            limit,
        )

        edges = []
        for r in rows:
            edges.append({
                "id": str(r["id"]),
                "from_vendor": r["from_vendor"],
                "to_vendor": r["to_vendor"],
                "mention_count": r["mention_count"],
                "primary_driver": r["primary_driver"],
                "signal_strength": r["signal_strength"],
                "key_quote": r["key_quote"],
                "source_distribution": _safe_json(r["source_distribution"]),
                "sample_review_ids": [str(rid) for rid in (r["sample_review_ids"] or [])],
                "confidence_score": float(r["confidence_score"]) if r["confidence_score"] else 0,
                "computed_date": str(r["computed_date"]),
                "report_id": str(r["report_id"]) if r["report_id"] else None,
                "created_at": str(r["created_at"]),
            })

        return json.dumps({"edges": edges, "count": len(edges)}, default=str)
    except Exception:
        logger.exception("list_displacement_edges error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def get_displacement_history(
    from_vendor: str,
    to_vendor: str,
    window_days: int = 365,
) -> str:
    """
    Time-series of displacement edge strength for a specific vendor pair.

    from_vendor: Source vendor (exact match, case-insensitive)
    to_vendor: Destination vendor (exact match, case-insensitive)
    window_days: How far back to look (default 365)
    """
    if not from_vendor or not to_vendor:
        return json.dumps({"error": "Both from_vendor and to_vendor are required"})

    try:
        from ..storage.database import get_db_pool
        pool = get_db_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        rows = await pool.fetch(
            """
            SELECT computed_date, mention_count, signal_strength,
                   confidence_score, primary_driver, key_quote
            FROM b2b_displacement_edges
            WHERE LOWER(from_vendor) = LOWER($1)
              AND LOWER(to_vendor) = LOWER($2)
              AND computed_date > NOW() - make_interval(days => $3)
            ORDER BY computed_date ASC
            """,
            from_vendor,
            to_vendor,
            window_days,
        )

        history = []
        for r in rows:
            history.append({
                "computed_date": str(r["computed_date"]),
                "mention_count": r["mention_count"],
                "signal_strength": r["signal_strength"],
                "confidence_score": float(r["confidence_score"]) if r["confidence_score"] else 0,
                "primary_driver": r["primary_driver"],
                "key_quote": r["key_quote"],
            })

        return json.dumps({
            "from_vendor": from_vendor,
            "to_vendor": to_vendor,
            "window_days": window_days,
            "history": history,
            "data_points": len(history),
        }, default=str)
    except Exception:
        logger.exception("get_displacement_history error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def list_vendor_pain_points(
    vendor_name: Optional[str] = None,
    pain_category: Optional[str] = None,
    min_confidence: Optional[float] = None,
    min_mentions: Optional[int] = None,
    limit: int = 50,
) -> str:
    """
    Query vendor pain points aggregated from review intelligence.

    vendor_name: Filter by vendor (case-insensitive partial match)
    pain_category: Exact pain category (pricing, support, features, ux, reliability, performance, integration, security, onboarding, other)
    min_confidence: Minimum confidence score (0.0-1.0)
    min_mentions: Minimum mention count
    limit: Max results (default 50, max 200)
    """
    limit = min(max(limit, 1), 200)
    try:
        from ..storage.database import get_db_pool
        pool = get_db_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        conditions: list[str] = []
        params: list = []
        idx = 1

        if vendor_name:
            conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
            params.append(vendor_name)
            idx += 1

        if pain_category:
            conditions.append(f"pain_category = ${idx}")
            params.append(pain_category)
            idx += 1

        if min_confidence is not None:
            conditions.append(f"confidence_score >= ${idx}")
            params.append(min_confidence)
            idx += 1

        if min_mentions is not None:
            conditions.append(f"mention_count >= ${idx}")
            params.append(min_mentions)
            idx += 1

        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        rows = await pool.fetch(
            f"""
            SELECT id, vendor_name, pain_category, mention_count,
                   primary_count, secondary_count, minor_count,
                   avg_urgency, avg_rating,
                   source_distribution, sample_review_ids,
                   confidence_score, first_seen_at, last_seen_at
            FROM b2b_vendor_pain_points
            {where}
            ORDER BY mention_count DESC
            LIMIT ${idx}
            """,
            *params,
            limit,
        )

        items = []
        for r in rows:
            items.append({
                "id": str(r["id"]),
                "vendor_name": r["vendor_name"],
                "pain_category": r["pain_category"],
                "mention_count": r["mention_count"],
                "primary_count": r["primary_count"],
                "secondary_count": r["secondary_count"],
                "minor_count": r["minor_count"],
                "avg_urgency": float(r["avg_urgency"]) if r["avg_urgency"] else 0,
                "avg_rating": float(r["avg_rating"]) if r["avg_rating"] else 0,
                "source_distribution": _safe_json(r["source_distribution"]),
                "sample_review_ids": [str(rid) for rid in (r["sample_review_ids"] or [])],
                "confidence_score": float(r["confidence_score"]) if r["confidence_score"] else 0,
                "first_seen_at": str(r["first_seen_at"]) if r["first_seen_at"] else None,
                "last_seen_at": str(r["last_seen_at"]) if r["last_seen_at"] else None,
            })

        return json.dumps({"pain_points": items, "count": len(items)}, default=str)
    except Exception:
        logger.exception("list_vendor_pain_points error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def list_vendor_use_cases(
    vendor_name: Optional[str] = None,
    use_case_name: Optional[str] = None,
    min_confidence: Optional[float] = None,
    min_mentions: Optional[int] = None,
    limit: int = 50,
) -> str:
    """
    Query vendor use cases (modules/features mentioned in reviews).

    vendor_name: Filter by vendor (case-insensitive partial match)
    use_case_name: Filter by use case name (case-insensitive partial match)
    min_confidence: Minimum confidence score (0.0-1.0)
    min_mentions: Minimum mention count
    limit: Max results (default 50, max 200)
    """
    limit = min(max(limit, 1), 200)
    try:
        from ..storage.database import get_db_pool
        pool = get_db_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        conditions: list[str] = []
        params: list = []
        idx = 1

        if vendor_name:
            conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
            params.append(vendor_name)
            idx += 1

        if use_case_name:
            conditions.append(f"use_case_name ILIKE '%' || ${idx} || '%'")
            params.append(use_case_name)
            idx += 1

        if min_confidence is not None:
            conditions.append(f"confidence_score >= ${idx}")
            params.append(min_confidence)
            idx += 1

        if min_mentions is not None:
            conditions.append(f"mention_count >= ${idx}")
            params.append(min_mentions)
            idx += 1

        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        rows = await pool.fetch(
            f"""
            SELECT id, vendor_name, use_case_name, mention_count,
                   avg_urgency, lock_in_distribution,
                   source_distribution, sample_review_ids,
                   confidence_score, first_seen_at, last_seen_at
            FROM b2b_vendor_use_cases
            {where}
            ORDER BY mention_count DESC
            LIMIT ${idx}
            """,
            *params,
            limit,
        )

        items = []
        for r in rows:
            items.append({
                "id": str(r["id"]),
                "vendor_name": r["vendor_name"],
                "use_case_name": r["use_case_name"],
                "mention_count": r["mention_count"],
                "avg_urgency": float(r["avg_urgency"]) if r["avg_urgency"] else 0,
                "lock_in_distribution": _safe_json(r["lock_in_distribution"]),
                "source_distribution": _safe_json(r["source_distribution"]),
                "sample_review_ids": [str(rid) for rid in (r["sample_review_ids"] or [])],
                "confidence_score": float(r["confidence_score"]) if r["confidence_score"] else 0,
                "first_seen_at": str(r["first_seen_at"]) if r["first_seen_at"] else None,
                "last_seen_at": str(r["last_seen_at"]) if r["last_seen_at"] else None,
            })

        return json.dumps({"use_cases": items, "count": len(items)}, default=str)
    except Exception:
        logger.exception("list_vendor_use_cases error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def list_vendor_integrations(
    vendor_name: Optional[str] = None,
    integration_name: Optional[str] = None,
    min_confidence: Optional[float] = None,
    min_mentions: Optional[int] = None,
    limit: int = 50,
) -> str:
    """
    Query vendor integrations (tools/services mentioned in reviews).

    vendor_name: Filter by vendor (case-insensitive partial match)
    integration_name: Filter by integration name (case-insensitive partial match)
    min_confidence: Minimum confidence score (0.0-1.0)
    min_mentions: Minimum mention count
    limit: Max results (default 50, max 200)
    """
    limit = min(max(limit, 1), 200)
    try:
        from ..storage.database import get_db_pool
        pool = get_db_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        conditions: list[str] = []
        params: list = []
        idx = 1

        if vendor_name:
            conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
            params.append(vendor_name)
            idx += 1

        if integration_name:
            conditions.append(f"integration_name ILIKE '%' || ${idx} || '%'")
            params.append(integration_name)
            idx += 1

        if min_confidence is not None:
            conditions.append(f"confidence_score >= ${idx}")
            params.append(min_confidence)
            idx += 1

        if min_mentions is not None:
            conditions.append(f"mention_count >= ${idx}")
            params.append(min_mentions)
            idx += 1

        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        rows = await pool.fetch(
            f"""
            SELECT id, vendor_name, integration_name, mention_count,
                   source_distribution, sample_review_ids,
                   confidence_score, first_seen_at, last_seen_at
            FROM b2b_vendor_integrations
            {where}
            ORDER BY mention_count DESC
            LIMIT ${idx}
            """,
            *params,
            limit,
        )

        items = []
        for r in rows:
            items.append({
                "id": str(r["id"]),
                "vendor_name": r["vendor_name"],
                "integration_name": r["integration_name"],
                "mention_count": r["mention_count"],
                "source_distribution": _safe_json(r["source_distribution"]),
                "sample_review_ids": [str(rid) for rid in (r["sample_review_ids"] or [])],
                "confidence_score": float(r["confidence_score"]) if r["confidence_score"] else 0,
                "first_seen_at": str(r["first_seen_at"]) if r["first_seen_at"] else None,
                "last_seen_at": str(r["last_seen_at"]) if r["last_seen_at"] else None,
            })

        return json.dumps({"integrations": items, "count": len(items)}, default=str)
    except Exception:
        logger.exception("list_vendor_integrations error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def list_vendor_buyer_profiles(
    vendor_name: Optional[str] = None,
    role_type: Optional[str] = None,
    buying_stage: Optional[str] = None,
    min_confidence: Optional[float] = None,
    min_reviews: Optional[int] = None,
    limit: int = 50,
) -> str:
    """
    Query aggregated buyer authority profiles per vendor.

    vendor_name: Filter by vendor (case-insensitive partial match)
    role_type: Filter by role type (economic_buyer, champion, evaluator, end_user, unknown)
    buying_stage: Filter by buying stage (active_purchase, evaluation, renewal_decision, post_purchase, unknown)
    min_confidence: Minimum confidence score (0.0-1.0)
    min_reviews: Minimum review count
    limit: Max results (default 50, max 200)
    """
    limit = min(max(limit, 1), 200)
    try:
        from ..storage.database import get_db_pool
        pool = get_db_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        conditions: list[str] = []
        params: list = []
        idx = 1

        if vendor_name:
            conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
            params.append(vendor_name)
            idx += 1

        if role_type:
            conditions.append(f"role_type = ${idx}")
            params.append(role_type)
            idx += 1

        if buying_stage:
            conditions.append(f"buying_stage = ${idx}")
            params.append(buying_stage)
            idx += 1

        if min_confidence is not None:
            conditions.append(f"confidence_score >= ${idx}")
            params.append(min_confidence)
            idx += 1

        if min_reviews is not None:
            conditions.append(f"review_count >= ${idx}")
            params.append(min_reviews)
            idx += 1

        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        rows = await pool.fetch(
            f"""
            SELECT id, vendor_name, role_type, buying_stage,
                   review_count, dm_count, avg_urgency,
                   source_distribution, sample_review_ids,
                   confidence_score, first_seen_at, last_seen_at
            FROM b2b_vendor_buyer_profiles
            {where}
            ORDER BY review_count DESC
            LIMIT ${idx}
            """,
            *params,
            limit,
        )

        profiles = []
        for r in rows:
            profiles.append({
                "id": str(r["id"]),
                "vendor_name": r["vendor_name"],
                "role_type": r["role_type"],
                "buying_stage": r["buying_stage"],
                "review_count": r["review_count"],
                "dm_count": r["dm_count"],
                "avg_urgency": float(r["avg_urgency"]) if r["avg_urgency"] is not None else None,
                "source_distribution": _safe_json(r["source_distribution"]),
                "sample_review_ids": [str(rid) for rid in (r["sample_review_ids"] or [])],
                "confidence_score": float(r["confidence_score"]) if r["confidence_score"] else 0,
                "first_seen_at": str(r["first_seen_at"]) if r["first_seen_at"] else None,
                "last_seen_at": str(r["last_seen_at"]) if r["last_seen_at"] else None,
            })

        return json.dumps({"profiles": profiles, "count": len(profiles)}, default=str)
    except Exception:
        logger.exception("list_vendor_buyer_profiles error")
        return json.dumps({"error": "Internal error"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    transport = "sse" if "--sse" in sys.argv else "stdio"
    if transport == "sse":
        from ..config import settings
        from .auth import run_sse_with_auth

        mcp.settings.host = settings.mcp.host
        mcp.settings.port = settings.mcp.b2b_churn_port
        run_sse_with_auth(mcp, settings.mcp.host, settings.mcp.b2b_churn_port)
    else:
        mcp.run(transport="stdio")
