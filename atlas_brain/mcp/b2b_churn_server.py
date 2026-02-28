"""
Atlas B2B Churn Intelligence MCP Server.

Exposes B2B churn intelligence data (reviews, vendor signals, reports,
pipeline health) to any MCP-compatible client (Claude Desktop, Cursor,
custom agents).

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
    list_scrape_targets      -- view scrape target configuration and status

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

logger = logging.getLogger("atlas.mcp.b2b_churn")

VALID_REPORT_TYPES = (
    "weekly_churn_feed",
    "vendor_scorecard",
    "displacement_report",
    "category_overview",
)

VALID_SOURCES = ("g2", "capterra", "trustradius", "reddit")


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
        "reports, identify high-intent companies, and monitor pipeline health. "
        "Data sourced from G2, Capterra, TrustRadius, and Reddit reviews."
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
        return json.dumps({"error": str(exc), "signals": [], "count": 0})


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
        return json.dumps({"success": False, "error": str(exc)})


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
        return json.dumps({"error": str(exc), "companies": [], "count": 0})


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
        return json.dumps({"success": False, "error": str(exc)})


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
                 displacement_report, category_overview)
    vendor_filter: Filter by vendor name in report (partial match)
    limit: Maximum results (default 10, cap 50)
    """
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
        return json.dumps({"error": str(exc), "reports": [], "count": 0})


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
        return json.dumps({"success": False, "error": str(exc)})


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
        return json.dumps({"error": str(exc), "reviews": [], "count": 0})


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
        return json.dumps({"success": False, "error": str(exc)})


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
        return json.dumps({"success": False, "error": str(exc)})


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

    source: Filter by source (g2, capterra, trustradius, reddit)
    enabled_only: Only show enabled targets (default true)
    limit: Maximum results (default 20, cap 100)
    """
    if source and source not in VALID_SOURCES:
        return json.dumps({"error": f"source must be one of {VALID_SOURCES}", "targets": [], "count": 0})

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
        return json.dumps({"error": str(exc), "targets": [], "count": 0})


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
