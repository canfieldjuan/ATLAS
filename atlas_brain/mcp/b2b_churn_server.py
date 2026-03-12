"""
Atlas B2B Churn Intelligence MCP Server.

Continuously collected competitive intelligence for the B2B SaaS market.
Exposes the full displacement intelligence network -- vendor churn signals,
evidence-backed risk profiles, historical vendor memory, and action-ready
intelligence feeds -- to any MCP-compatible client (Claude Desktop, Cursor,
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
    get_parser_version_status -- per-source parser version status and re-extraction queue
    get_source_health        -- per-source scrape reliability metrics with trend comparison
    get_source_telemetry     -- CAPTCHA attempts, solve times, block types, proxy usage per source
    get_source_capabilities  -- capability profiles (access patterns, anti-bot, proxy, quality)
    get_operational_overview -- single snapshot: pipeline, source health, telemetry, recent events
    list_scrape_targets      -- view scrape target configuration and status
    get_product_profile      -- fetch pre-computed product knowledge card
    get_product_profile_history -- daily product profile snapshots over time
    match_products_tool      -- find best alternatives for a churning company
    list_blog_posts          -- list/filter generated blog posts
    get_blog_post            -- fetch full blog post by ID or slug
    add_scrape_target        -- add a new vendor/source scrape target
    manage_scrape_target     -- update target settings (enabled, priority, pages, interval, metadata)
    delete_scrape_target     -- remove a scrape target and its logs
    list_affiliate_partners  -- list affiliate partner configurations
    list_vendors_registry    -- list all canonical vendors and their aliases
    fuzzy_vendor_search      -- fuzzy search vendors by name (trigram similarity, handles typos)
    fuzzy_company_search     -- fuzzy search companies by name (trigram similarity, vendor-scoped)
    add_vendor_to_registry   -- add a new vendor to the canonical registry
    add_vendor_alias         -- add an alias to an existing vendor
    list_displacement_edges  -- query persisted competitive displacement edges
    get_displacement_history -- time-series of edge strength for a vendor pair
    list_vendor_pain_points  -- query aggregated vendor pain points with confidence
    list_vendor_use_cases    -- query vendor use cases (modules mentioned in reviews)
    list_vendor_integrations -- query vendor integrations (tools mentioned in reviews)
    list_vendor_buyer_profiles -- query aggregated buyer authority profiles per vendor
    get_vendor_history       -- daily health snapshots for a vendor over time
    list_change_events       -- structural change events with vendor/type filters
    compare_vendor_periods   -- compare a vendor's health between two dates
    record_campaign_outcome  -- record business outcome for a campaign sequence
    get_signal_effectiveness -- correlate signal dimensions with campaign outcomes
    create_data_correction   -- record an analyst correction (suppress, flag, override, merge, reclassify)
    list_data_corrections    -- list/filter recorded data corrections (corrected_by, date range)
    revert_data_correction   -- revert a previously applied correction
    get_data_correction      -- fetch single correction by ID with full details
    get_correction_stats     -- aggregate correction activity (by type, status, entity, top correctors)
    get_source_correction_impact -- show impact of active source suppressions on review counts
    get_parser_health        -- parser version distribution and stale review detection
    get_calibration_weights  -- view score calibration weights from outcome data
    trigger_score_calibration -- manually trigger score recalibration
    list_webhook_subscriptions -- list webhook subscriptions with delivery stats (generic/slack/teams/crm channels)
    send_test_webhook_tool   -- send a test payload to verify webhook connectivity (channel-aware formatting)
    update_webhook           -- update webhook subscription (toggle enabled, change event_types/url/description)
    get_webhook_delivery_summary -- aggregate delivery health across all subscriptions
    list_crm_pushes          -- list CRM push log entries showing intelligence data pushed to external CRMs
    export_report_pdf        -- export a B2B intelligence report as PDF
    get_outcome_distribution  -- system-wide campaign outcome funnel (counts, percentages, revenue)
    trigger_score_calibration -- on-demand score recalibration from campaign outcomes
    list_crm_events          -- list ingested CRM events with filters (date range, status validation)
    ingest_crm_event         -- manually ingest a CRM event for processing
    get_crm_enrichment_stats -- CRM event enrichment coverage and effectiveness stats
    list_concurrent_events   -- find dates where multiple vendors had the same change event
    get_vendor_correlation   -- pairwise vendor metric correlation with Pearson r

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

from atlas_brain.services.scraping.sources import ALL_SOURCES, ReviewSource

VALID_SOURCES = ALL_SOURCES


def _is_uuid(value: str) -> bool:
    """Check if a string is a valid UUID."""
    try:
        _uuid.UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


from ..services.b2b.corrections import suppress_predicate as _suppress_predicate
from ..services.b2b.corrections import apply_field_overrides as _apply_field_overrides


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
    content_type: Optional[str] = None,
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
    content_type: Filter by content type — one of: review, community_discussion, comment, insider_account
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

        if content_type:
            conditions.append(f"content_type = ${idx}")
            params.append(content_type)
            idx += 1
        else:
            conditions.append(_suppress_predicate('review'))

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
                   enriched_at, reviewer_title, company_size_raw,
                   COALESCE(reviewer_industry, enrichment->'reviewer_context'->>'industry') AS industry,
                   content_type, thread_id
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
                "reviewer_title": r["reviewer_title"],
                "company_size": r["company_size_raw"],
                "industry": r["industry"],
                "content_type": r["content_type"],
                "thread_id": r["thread_id"],
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
        rid = _uuid.UUID(review_id)
        row = await pool.fetchrow(
            "SELECT * FROM b2b_reviews WHERE id = $1",
            rid,
        )

        if not row:
            return json.dumps({"success": False, "error": "Review not found"})

        suppressed = await pool.fetchval(
            """SELECT 1 FROM data_corrections
               WHERE entity_type = 'review' AND entity_id = $1
                 AND correction_type = 'suppress' AND status = 'applied'""",
            row["id"],
        )
        if suppressed:
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
        review = await _apply_field_overrides(pool, "review", str(row["id"]), review)

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
        _rev_sup = _suppress_predicate('review')
        status_rows = await pool.fetch(
            f"""
            SELECT enrichment_status, COUNT(*) AS cnt
            FROM b2b_reviews
            WHERE {_rev_sup}
            GROUP BY enrichment_status
            """
        )
        enrichment_counts = {r["enrichment_status"]: r["cnt"] for r in status_rows}

        # Recent imports + last enrichment
        stats = await pool.fetchrow(
            f"""
            SELECT
                COUNT(*) FILTER (WHERE imported_at > NOW() - INTERVAL '24 hours') AS recent_imports_24h,
                MAX(enriched_at) AS last_enrichment_at
            FROM b2b_reviews
            WHERE {_rev_sup}
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
# Tool: get_parser_version_status
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_parser_version_status() -> str:
    """
    Show per-source parser version status and count of reviews needing re-extraction.

    Returns for each source: current parser version, total reviews,
    count at current version, count at outdated versions, and count with unknown version.
    Reviews with outdated parser versions are automatically re-queued for enrichment
    on the next enrichment run.
    """
    try:
        from ..services.scraping.parsers import get_all_parsers
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        parsers = get_all_parsers()
        sources = []
        for source_name, parser in sorted(parsers.items()):
            current_version = getattr(parser, "version", None)
            if not current_version:
                continue

            row = await pool.fetchrow(
                """
                SELECT COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE parser_version = $2) AS current_count,
                       COUNT(*) FILTER (WHERE parser_version IS NOT NULL AND parser_version != $2) AS outdated_count,
                       COUNT(*) FILTER (WHERE parser_version IS NULL) AS unknown_count
                FROM b2b_reviews
                WHERE source = $1
                """,
                source_name, current_version,
            )
            sources.append({
                "source": source_name,
                "current_version": current_version,
                "total_reviews": row["total"],
                "current_version_count": row["current_count"],
                "outdated_version_count": row["outdated_count"],
                "unknown_version_count": row["unknown_count"],
            })

        return json.dumps({"sources": sources, "count": len(sources)}, default=str)
    except Exception:
        logger.exception("get_parser_version_status error")
        return json.dumps({"error": "Internal error"})


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
# Tool: get_source_telemetry
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_source_telemetry(
    window_days: int = 7,
    source: Optional[str] = None,
) -> str:
    """
    Get CAPTCHA attempts, solve times, block type distribution, and proxy usage per source.

    window_days: How many days back to look (default 7, max 30)
    source: Filter to a single source (optional)
    """
    window_days = min(max(window_days, 1), 30)
    try:
        from ..storage.database import get_db_pool
        pool = get_db_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        conditions = ["started_at >= NOW() - make_interval(days => $1)"]
        params: list = [window_days]
        idx = 2
        if source:
            conditions.append(f"source = ${idx}")
            params.append(source.strip().lower())
            idx += 1

        where = " AND ".join(conditions)
        rows = await pool.fetch(
            f"""
            SELECT
                source,
                COUNT(*)                                                AS total_scrapes,
                SUM(COALESCE(captcha_attempts, 0))                      AS total_captcha_attempts,
                COUNT(*) FILTER (WHERE captcha_attempts > 0)            AS scrapes_with_captcha,
                AVG(captcha_solve_ms) FILTER (WHERE captcha_solve_ms > 0) AS avg_captcha_solve_ms,
                MAX(captcha_solve_ms)                                    AS max_captcha_solve_ms,
                COUNT(*) FILTER (WHERE block_type IS NOT NULL)           AS total_blocks,
                COUNT(*) FILTER (WHERE block_type = 'captcha')           AS blocks_captcha,
                COUNT(*) FILTER (WHERE block_type = 'ip_ban')            AS blocks_ip_ban,
                COUNT(*) FILTER (WHERE block_type = 'rate_limit')        AS blocks_rate_limit,
                COUNT(*) FILTER (WHERE block_type = 'waf')               AS blocks_waf,
                COUNT(*) FILTER (WHERE block_type = 'unknown')           AS blocks_unknown,
                COUNT(*) FILTER (WHERE proxy_type = 'datacenter')        AS proxy_datacenter,
                COUNT(*) FILTER (WHERE proxy_type = 'residential')       AS proxy_residential,
                COUNT(*) FILTER (WHERE proxy_type = 'none')              AS proxy_none
            FROM b2b_scrape_log
            WHERE {where}
            GROUP BY source
            ORDER BY total_captcha_attempts DESC
            """,
            *params,
        )

        sources_out = []
        for r in rows:
            total = r["total_scrapes"] or 1
            sources_out.append({
                "source": r["source"],
                "total_scrapes": r["total_scrapes"],
                "captcha": {
                    "total_attempts": r["total_captcha_attempts"] or 0,
                    "scrapes_with_captcha": r["scrapes_with_captcha"],
                    "captcha_rate": round(r["scrapes_with_captcha"] / total, 3),
                    "avg_solve_ms": round(float(r["avg_captcha_solve_ms"]), 0) if r["avg_captcha_solve_ms"] else None,
                    "max_solve_ms": r["max_captcha_solve_ms"],
                },
                "blocks": {
                    "total": r["total_blocks"],
                    "captcha": r["blocks_captcha"],
                    "ip_ban": r["blocks_ip_ban"],
                    "rate_limit": r["blocks_rate_limit"],
                    "waf": r["blocks_waf"],
                    "unknown": r["blocks_unknown"],
                },
                "proxy_usage": {
                    "datacenter": r["proxy_datacenter"],
                    "residential": r["proxy_residential"],
                    "none": r["proxy_none"],
                },
            })

        return json.dumps({
            "window_days": window_days,
            "sources": sources_out,
            "total_sources": len(sources_out),
        }, default=str)
    except Exception:
        logger.exception("get_source_telemetry error")
        return json.dumps({"error": "Internal error"})


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
# Tool: get_operational_overview
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_operational_overview() -> str:
    """
    Single snapshot combining pipeline status, source health, telemetry, and recent events.

    Returns data summary (total reviews, vendors tracked), enrichment pipeline counts,
    source health summary (7d), CAPTCHA/block telemetry (7d), and 10 most recent change events.
    """
    try:
        import asyncio as _aio
        from ..storage.database import get_db_pool
        pool = get_db_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        pipeline_row, health_rows, telemetry_row, event_rows, review_row = await _aio.gather(
            pool.fetchrow("""
                SELECT
                    COUNT(*) FILTER (WHERE enrichment_status = 'pending')   AS pending,
                    COUNT(*) FILTER (WHERE enrichment_status = 'enriched')  AS enriched,
                    COUNT(*) FILTER (WHERE enrichment_status = 'failed')    AS failed,
                    COUNT(*)                                                 AS total
                FROM b2b_reviews
            """),
            pool.fetch("""
                SELECT source, COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE status = 'success') AS success,
                       COUNT(*) FILTER (WHERE status = 'blocked') AS blocked
                FROM b2b_scrape_log
                WHERE started_at >= NOW() - INTERVAL '7 days'
                GROUP BY source ORDER BY total DESC
            """),
            pool.fetchrow("""
                SELECT
                    SUM(COALESCE(captcha_attempts, 0)) AS captcha_total,
                    COUNT(*) FILTER (WHERE block_type IS NOT NULL) AS blocks_total
                FROM b2b_scrape_log
                WHERE started_at >= NOW() - INTERVAL '7 days'
            """),
            pool.fetch("""
                SELECT vendor_name, event_type, event_date, description
                FROM b2b_change_events
                ORDER BY event_date DESC, created_at DESC
                LIMIT 10
            """),
            pool.fetchrow("""
                SELECT COUNT(*) AS total_reviews,
                       COUNT(DISTINCT vendor_name) AS vendors_tracked
                FROM b2b_reviews
            """),
        )

        total_scrapes = sum(r["total"] for r in health_rows)
        total_success = sum(r["success"] for r in health_rows)

        return json.dumps({
            "data_summary": {
                "total_reviews": review_row["total_reviews"],
                "vendors_tracked": review_row["vendors_tracked"],
            },
            "pipeline": {
                "pending": pipeline_row["pending"],
                "enriched": pipeline_row["enriched"],
                "failed": pipeline_row["failed"],
                "total": pipeline_row["total"],
            },
            "source_health_7d": {
                "total_scrapes": total_scrapes,
                "overall_success_rate": round(total_success / max(total_scrapes, 1), 3),
                "sources": [{"source": r["source"], "total": r["total"],
                             "success_rate": round(r["success"] / max(r["total"], 1), 3)}
                            for r in health_rows],
            },
            "telemetry_7d": {
                "captcha_attempts": telemetry_row["captcha_total"] or 0,
                "blocks": telemetry_row["blocks_total"] or 0,
            },
            "recent_events": [
                {"vendor": r["vendor_name"], "type": r["event_type"],
                 "date": str(r["event_date"]), "description": r["description"]}
                for r in event_rows
            ],
        }, default=str)
    except Exception:
        logger.exception("get_operational_overview error")
        return json.dumps({"error": "Internal error"})


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
                   profile_summary, confidence_score, last_computed_at, created_at
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
            "confidence_score": float(row["confidence_score"]) if row["confidence_score"] is not None else 0,
            "last_computed_at": row["last_computed_at"],
            "created_at": row["created_at"],
        }

        return json.dumps({"success": True, "profile": profile}, default=str)
    except Exception:
        logger.exception("get_product_profile error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: get_product_profile_history
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_product_profile_history(
    vendor_name: str,
    days: int = 90,
    limit: int = 90,
) -> str:
    """
    Get daily product profile snapshots for a vendor over time.

    vendor_name: Vendor name (case-insensitive partial match)
    days: How many days back to look (default 90)
    limit: Max snapshots to return (default 90, max 365)
    """
    limit = min(max(limit, 1), 365)
    days = min(max(days, 1), 365)
    try:
        from ..storage.database import get_db_pool
        pool = get_db_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        rows = await pool.fetch(
            """
            SELECT vendor_name, snapshot_date,
                   total_reviews_analyzed, avg_rating, recommend_rate, avg_urgency,
                   strength_count, weakness_count, top_strength, top_weakness,
                   top_use_case, top_integration,
                   compared_to_count, switched_from_count,
                   pain_categories_covered, profile_summary_len
            FROM b2b_product_profile_snapshots
            WHERE vendor_name ILIKE '%' || $1 || '%'
              AND snapshot_date >= CURRENT_DATE - $2::int
            ORDER BY snapshot_date DESC
            LIMIT $3
            """,
            vendor_name.strip(), days, limit,
        )

        if not rows:
            return json.dumps({
                "vendor_name": vendor_name,
                "snapshots": [],
                "count": 0,
                "message": f"No product profile snapshots found for '{vendor_name}'",
            })

        resolved = rows[0]["vendor_name"]
        snapshots = []
        for r in rows:
            snapshots.append({
                "snapshot_date": str(r["snapshot_date"]),
                "total_reviews_analyzed": r["total_reviews_analyzed"],
                "avg_rating": float(r["avg_rating"]) if r["avg_rating"] is not None else None,
                "recommend_rate": float(r["recommend_rate"]) if r["recommend_rate"] is not None else None,
                "avg_urgency": float(r["avg_urgency"]) if r["avg_urgency"] is not None else None,
                "strength_count": r["strength_count"],
                "weakness_count": r["weakness_count"],
                "top_strength": r["top_strength"],
                "top_weakness": r["top_weakness"],
                "top_use_case": r["top_use_case"],
                "top_integration": r["top_integration"],
                "compared_to_count": r["compared_to_count"],
                "switched_from_count": r["switched_from_count"],
                "pain_categories_covered": r["pain_categories_covered"],
                "profile_summary_len": r["profile_summary_len"],
            })

        return json.dumps({
            "vendor_name": resolved,
            "snapshots": snapshots,
            "count": len(snapshots),
        }, default=str)
    except Exception:
        logger.exception("get_product_profile_history error")
        return json.dumps({"error": "Internal error"})


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
# Tool: fuzzy_vendor_search
# ---------------------------------------------------------------------------


@mcp.tool()
async def fuzzy_vendor_search(
    query: str,
    limit: int = 10,
    min_similarity: float = 0.3,
) -> str:
    """
    Search vendors by name using fuzzy matching (trigram similarity).

    Finds vendors even with typos or partial names (e.g. "Salesfroce" -> "Salesforce").

    query: Vendor name to search for
    limit: Max results (default 10, max 100)
    min_similarity: Minimum similarity threshold 0.0-1.0 (default 0.3)
    """
    if not query or not query.strip():
        return json.dumps({"error": "query is required"})
    try:
        from ..services.vendor_registry import fuzzy_search_vendors

        results = await fuzzy_search_vendors(
            query.strip(), limit=limit, min_similarity=min_similarity,
        )
        return json.dumps({"query": query.strip(), "results": results, "count": len(results)}, default=str)
    except Exception:
        logger.exception("fuzzy_vendor_search error")
        return json.dumps({"error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: fuzzy_company_search
# ---------------------------------------------------------------------------


@mcp.tool()
async def fuzzy_company_search(
    query: str,
    vendor_name: Optional[str] = None,
    limit: int = 10,
    min_similarity: float = 0.3,
) -> str:
    """
    Search company names using fuzzy matching (trigram similarity).

    Finds companies even with typos or partial names. Optionally scoped to a vendor.

    query: Company name to search for
    vendor_name: Optional vendor name to scope the search
    limit: Max results (default 10, max 100)
    min_similarity: Minimum similarity threshold 0.0-1.0 (default 0.3)
    """
    if not query or not query.strip():
        return json.dumps({"error": "query is required"})
    try:
        from ..services.vendor_registry import fuzzy_search_companies

        results = await fuzzy_search_companies(
            query.strip(), vendor_name=vendor_name, limit=limit, min_similarity=min_similarity,
        )
        return json.dumps({
            "query": query.strip(),
            "vendor_filter": vendor_name,
            "results": results,
            "count": len(results),
        }, default=str)
    except Exception:
        logger.exception("fuzzy_company_search error")
        return json.dumps({"error": "Internal error"})


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

        conditions.append(_suppress_predicate('displacement_edge'))
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

        conditions.append(_suppress_predicate('pain_point'))
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

        conditions.append(_suppress_predicate('use_case'))
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

        conditions.append(_suppress_predicate('integration'))
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

        conditions.append(_suppress_predicate('buyer_profile'))
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
# Historical snapshots & change events
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_vendor_history(
    vendor_name: str,
    days: int = 90,
    limit: int = 90,
) -> str:
    """
    Get daily health snapshots for a vendor over time.

    vendor_name: Vendor name (case-insensitive partial match)
    days: How many days back to look (default 90)
    limit: Max snapshots to return (default 90, max 365)
    """
    limit = min(max(limit, 1), 365)
    days = min(max(days, 1), 365)
    try:
        from ..storage.database import get_db_pool
        pool = get_db_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        rows = await pool.fetch(
            """
            SELECT vendor_name, snapshot_date, total_reviews, churn_intent,
                   churn_density, avg_urgency, positive_review_pct, recommend_ratio,
                   top_pain, top_competitor, pain_count, competitor_count,
                   displacement_edge_count, high_intent_company_count
            FROM b2b_vendor_snapshots
            WHERE vendor_name ILIKE '%' || $1 || '%'
              AND snapshot_date >= CURRENT_DATE - $2::int
            ORDER BY snapshot_date DESC
            LIMIT $3
            """,
            vendor_name, days, limit,
        )

        resolved = rows[0]["vendor_name"] if rows else vendor_name
        snapshots = []
        for r in rows:
            snapshots.append({
                "snapshot_date": str(r["snapshot_date"]),
                "total_reviews": r["total_reviews"],
                "churn_intent": r["churn_intent"],
                "churn_density": float(r["churn_density"]),
                "avg_urgency": float(r["avg_urgency"]),
                "positive_review_pct": float(r["positive_review_pct"]) if r["positive_review_pct"] is not None else None,
                "recommend_ratio": float(r["recommend_ratio"]) if r["recommend_ratio"] is not None else None,
                "top_pain": r["top_pain"],
                "top_competitor": r["top_competitor"],
                "pain_count": r["pain_count"],
                "competitor_count": r["competitor_count"],
                "displacement_edge_count": r["displacement_edge_count"],
                "high_intent_company_count": r["high_intent_company_count"],
            })

        return json.dumps({"vendor_name": resolved, "snapshots": snapshots, "count": len(snapshots)}, default=str)
    except Exception:
        logger.exception("get_vendor_history error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def list_change_events(
    vendor_name: Optional[str] = None,
    event_type: Optional[str] = None,
    days: int = 30,
    limit: int = 50,
) -> str:
    """
    List structural change events detected across vendors.

    vendor_name: Filter by vendor (case-insensitive partial match)
    event_type: Filter by event type (urgency_spike, urgency_drop, churn_density_spike, nps_shift, new_pain_category, new_competitor, review_volume_spike)
    days: How many days back to look (default 30)
    limit: Max events to return (default 50, max 200)
    """
    limit = min(max(limit, 1), 200)
    days = min(max(days, 1), 365)
    try:
        from ..storage.database import get_db_pool
        pool = get_db_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        conditions: list[str] = ["event_date >= CURRENT_DATE - $1::int"]
        params: list = [days]
        idx = 2

        if vendor_name:
            conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
            params.append(vendor_name)
            idx += 1

        if event_type:
            conditions.append(f"event_type = ${idx}")
            params.append(event_type)
            idx += 1

        where = "WHERE " + " AND ".join(conditions)

        rows = await pool.fetch(
            f"""
            SELECT vendor_name, event_date, event_type, description,
                   old_value, new_value, delta, metadata
            FROM b2b_change_events
            {where}
            ORDER BY event_date DESC, created_at DESC
            LIMIT ${idx}
            """,
            *params, limit,
        )

        events = []
        for r in rows:
            events.append({
                "vendor_name": r["vendor_name"],
                "event_date": str(r["event_date"]),
                "event_type": r["event_type"],
                "description": r["description"],
                "old_value": float(r["old_value"]) if r["old_value"] is not None else None,
                "new_value": float(r["new_value"]) if r["new_value"] is not None else None,
                "delta": float(r["delta"]) if r["delta"] is not None else None,
                "metadata": _safe_json(r["metadata"]),
            })

        return json.dumps({"events": events, "count": len(events)}, default=str)
    except Exception:
        logger.exception("list_change_events error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def compare_vendor_periods(
    vendor_name: str,
    period_a_days_ago: int = 30,
    period_b_days_ago: int = 0,
) -> str:
    """
    Compare a vendor's health between two dates.

    vendor_name: Vendor name (case-insensitive partial match)
    period_a_days_ago: The older snapshot (days ago, default 30)
    period_b_days_ago: The newer snapshot (days ago, default 0 = today)
    """
    try:
        from ..storage.database import get_db_pool
        pool = get_db_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        async def _nearest_snapshot(target_days_ago: int):
            return await pool.fetchrow(
                """
                SELECT vendor_name, snapshot_date, total_reviews, churn_intent,
                       churn_density, avg_urgency, positive_review_pct, recommend_ratio,
                       top_pain, top_competitor, pain_count, competitor_count,
                       displacement_edge_count, high_intent_company_count
                FROM b2b_vendor_snapshots
                WHERE vendor_name ILIKE '%' || $1 || '%'
                  AND snapshot_date <= CURRENT_DATE - $2::int
                ORDER BY snapshot_date DESC
                LIMIT 1
                """,
                vendor_name, target_days_ago,
            )

        snap_a = await _nearest_snapshot(period_a_days_ago)
        snap_b = await _nearest_snapshot(period_b_days_ago)

        if not snap_a and not snap_b:
            return json.dumps({"error": f"No snapshots found for vendor matching '{vendor_name}'"})

        def _format(snap):
            if not snap:
                return None
            return {
                "snapshot_date": str(snap["snapshot_date"]),
                "total_reviews": snap["total_reviews"],
                "churn_intent": snap["churn_intent"],
                "churn_density": float(snap["churn_density"]),
                "avg_urgency": float(snap["avg_urgency"]),
                "positive_review_pct": float(snap["positive_review_pct"]) if snap["positive_review_pct"] is not None else None,
                "recommend_ratio": float(snap["recommend_ratio"]) if snap["recommend_ratio"] is not None else None,
                "top_pain": snap["top_pain"],
                "top_competitor": snap["top_competitor"],
                "pain_count": snap["pain_count"],
                "competitor_count": snap["competitor_count"],
                "displacement_edge_count": snap["displacement_edge_count"],
                "high_intent_company_count": snap["high_intent_company_count"],
            }

        a_fmt = _format(snap_a)
        b_fmt = _format(snap_b)

        deltas = {}
        if a_fmt and b_fmt:
            for key in ("churn_density", "avg_urgency", "recommend_ratio", "total_reviews",
                        "churn_intent", "pain_count", "competitor_count",
                        "displacement_edge_count", "high_intent_company_count"):
                a_val = a_fmt.get(key)
                b_val = b_fmt.get(key)
                if a_val is not None and b_val is not None:
                    deltas[key] = round(b_val - a_val, 2)

        resolved = (snap_a or snap_b)["vendor_name"]
        return json.dumps({
            "vendor_name": resolved,
            "period_a": a_fmt,
            "period_b": b_fmt,
            "deltas": deltas,
        }, default=str)
    except Exception:
        logger.exception("compare_vendor_periods error")
        return json.dumps({"error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: list_concurrent_events (Cross-vendor trend correlation)
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_concurrent_events(
    days: int = 30,
    event_type: Optional[str] = None,
    min_vendors: int = 2,
    limit: int = 50,
) -> str:
    """
    Find dates where multiple vendors had the same change event type.

    Surfaces cross-vendor correlations like 'urgency spiked at 4 vendors on
    the same day' -- may indicate market-level trends vs vendor-specific issues.

    days: Lookback period (default 30)
    event_type: Optional filter (urgency_spike, churn_density_spike, nps_shift, etc.)
    min_vendors: Minimum vendor count to qualify as concurrent (default 2)
    limit: Max results (default 50)
    """
    try:
        from ..storage.database import get_db_pool
        pool = get_db_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        type_filter = ""
        params: list = [days, min_vendors, limit]
        if event_type:
            type_filter = "AND event_type = $4"
            params.append(event_type)

        rows = await pool.fetch(
            f"""
            SELECT event_date, event_type,
                   COUNT(DISTINCT vendor_name) AS vendor_count,
                   ARRAY_AGG(DISTINCT vendor_name ORDER BY vendor_name) AS vendors,
                   AVG(delta) AS avg_delta,
                   MIN(delta) AS min_delta,
                   MAX(delta) AS max_delta
            FROM b2b_change_events
            WHERE event_date >= CURRENT_DATE - $1::int
              {type_filter}
            GROUP BY event_date, event_type
            HAVING COUNT(DISTINCT vendor_name) >= $2
            ORDER BY vendor_count DESC, event_date DESC
            LIMIT $3
            """,
            *params,
        )

        results = [
            {
                "event_date": str(r["event_date"]),
                "event_type": r["event_type"],
                "vendor_count": r["vendor_count"],
                "vendors": r["vendors"],
                "avg_delta": round(float(r["avg_delta"]), 2) if r["avg_delta"] is not None else None,
                "min_delta": round(float(r["min_delta"]), 2) if r["min_delta"] is not None else None,
                "max_delta": round(float(r["max_delta"]), 2) if r["max_delta"] is not None else None,
            }
            for r in rows
        ]
        return json.dumps({"concurrent_events": results, "total": len(results)}, default=str)
    except Exception:
        logger.exception("list_concurrent_events error")
        return json.dumps({"error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: get_vendor_correlation (Pairwise vendor trend analysis)
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_vendor_correlation(
    vendor_a: str,
    vendor_b: str,
    days: int = 90,
    metric: str = "churn_density",
) -> str:
    """
    Compare two vendors' metric trends and compute correlation coefficient.

    Returns aligned time-series and Pearson r. Negative correlation (r < -0.5)
    suggests one vendor gains when the other loses -- potential displacement.

    vendor_a: First vendor name (partial match)
    vendor_b: Second vendor name (partial match)
    days: Lookback period (default 90)
    metric: Metric to correlate (churn_density, avg_urgency, recommend_ratio,
            total_reviews, displacement_edge_count, high_intent_company_count)
    """
    try:
        from ..storage.database import get_db_pool
        pool = get_db_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        valid_metrics = {
            "churn_density", "avg_urgency", "recommend_ratio", "total_reviews",
            "displacement_edge_count", "high_intent_company_count", "pain_count",
            "competitor_count",
        }
        if metric not in valid_metrics:
            return json.dumps({"error": f"metric must be one of: {sorted(valid_metrics)}"})

        rows = await pool.fetch(
            f"""
            SELECT a.snapshot_date,
                   a.{metric} AS value_a,
                   b.{metric} AS value_b
            FROM b2b_vendor_snapshots a
            JOIN b2b_vendor_snapshots b
              ON a.snapshot_date = b.snapshot_date
            WHERE a.vendor_name ILIKE '%' || $1 || '%'
              AND b.vendor_name ILIKE '%' || $2 || '%'
              AND a.snapshot_date >= CURRENT_DATE - $3::int
            ORDER BY a.snapshot_date ASC
            """,
            vendor_a, vendor_b, days,
        )

        if not rows:
            return json.dumps({"error": "No overlapping snapshots found for these vendors"})

        vals_a = [float(r["value_a"] or 0) for r in rows]
        vals_b = [float(r["value_b"] or 0) for r in rows]

        # Pearson correlation
        n = len(vals_a)
        correlation = None
        if n >= 3:
            mean_a = sum(vals_a) / n
            mean_b = sum(vals_b) / n
            dx = [v - mean_a for v in vals_a]
            dy = [v - mean_b for v in vals_b]
            num = sum(a * b for a, b in zip(dx, dy))
            den_a = sum(a * a for a in dx) ** 0.5
            den_b = sum(b * b for b in dy) ** 0.5
            if den_a > 0 and den_b > 0:
                correlation = round(num / (den_a * den_b), 4)

        series = [
            {
                "date": str(r["snapshot_date"]),
                "value_a": float(r["value_a"] or 0),
                "value_b": float(r["value_b"] or 0),
            }
            for r in rows
        ]

        # Recent displacement edges between the pair
        edge_rows = await pool.fetch(
            """
            SELECT from_vendor, to_vendor, mention_count, signal_strength, primary_driver
            FROM b2b_displacement_edges
            WHERE (from_vendor ILIKE '%' || $1 || '%' AND to_vendor ILIKE '%' || $2 || '%')
               OR (from_vendor ILIKE '%' || $2 || '%' AND to_vendor ILIKE '%' || $1 || '%')
            ORDER BY computed_date DESC
            LIMIT 5
            """,
            vendor_a, vendor_b,
        )
        displacement = [
            {
                "from_vendor": r["from_vendor"],
                "to_vendor": r["to_vendor"],
                "mention_count": r["mention_count"],
                "signal_strength": r["signal_strength"],
                "primary_driver": r["primary_driver"],
            }
            for r in edge_rows
        ]

        return json.dumps({
            "vendor_a": vendor_a,
            "vendor_b": vendor_b,
            "metric": metric,
            "data_points": len(series),
            "correlation": correlation,
            "series": series,
            "displacement_edges": displacement,
        }, default=str)
    except Exception:
        logger.exception("get_vendor_correlation error")
        return json.dumps({"error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: record_campaign_outcome
# ---------------------------------------------------------------------------

_VALID_OUTCOMES = {
    "pending", "meeting_booked", "deal_opened",
    "deal_won", "deal_lost", "no_opportunity", "disqualified",
}


@mcp.tool()
async def record_campaign_outcome(
    sequence_id: str,
    outcome: str,
    notes: Optional[str] = None,
    revenue: Optional[float] = None,
    recorded_by: str = "mcp",
) -> str:
    """
    Record a business outcome for a campaign sequence.

    sequence_id: UUID of the campaign sequence
    outcome: One of: pending, meeting_booked, deal_opened, deal_won, deal_lost, no_opportunity, disqualified
    notes: Optional notes about the outcome
    revenue: Optional revenue amount (for deal_won)
    recorded_by: Who recorded this (default "mcp")
    """
    if not _is_uuid(sequence_id):
        return json.dumps({"success": False, "error": "Invalid sequence_id (must be UUID)"})

    if outcome not in _VALID_OUTCOMES:
        return json.dumps({"success": False, "error": f"Invalid outcome. Must be one of: {sorted(_VALID_OUTCOMES)}"})

    try:
        from datetime import datetime, timezone

        from ..autonomous.tasks.campaign_audit import log_campaign_event
        from ..storage.database import get_db_pool

        pool = get_db_pool()

        row = await pool.fetchrow(
            "SELECT id, outcome, outcome_history FROM campaign_sequences WHERE id = $1",
            _uuid.UUID(sequence_id),
        )
        if not row:
            return json.dumps({"success": False, "error": "Sequence not found"})

        previous = row["outcome"]
        now = datetime.now(timezone.utc)

        history_entry = {
            "stage": outcome,
            "recorded_at": now.isoformat(),
            "previous": previous,
            "notes": notes,
            "recorded_by": recorded_by,
        }
        existing_history = row["outcome_history"] if isinstance(row["outcome_history"], list) else []
        updated_history = existing_history + [history_entry]

        await pool.execute(
            """
            UPDATE campaign_sequences
            SET outcome = $1,
                outcome_recorded_at = $2,
                outcome_recorded_by = $3,
                outcome_notes = $4,
                outcome_revenue = $5,
                outcome_history = $6::jsonb,
                updated_at = NOW()
            WHERE id = $7
            """,
            outcome,
            now,
            recorded_by,
            notes,
            revenue,
            json.dumps(updated_history),
            _uuid.UUID(sequence_id),
        )

        await log_campaign_event(
            pool,
            event_type=f"outcome_{outcome}",
            source=recorded_by,
            sequence_id=_uuid.UUID(sequence_id),
            metadata={
                "outcome": outcome,
                "previous": previous,
                "revenue": revenue,
                "notes": notes,
            },
        )

        return json.dumps({
            "success": True,
            "sequence_id": sequence_id,
            "outcome": outcome,
            "previous_outcome": previous,
            "recorded_at": now.isoformat(),
        })
    except Exception:
        logger.exception("record_campaign_outcome error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: get_signal_effectiveness
# ---------------------------------------------------------------------------

_GROUP_BY_EXPRESSIONS = {
    "buying_stage": "bc.buying_stage",
    "role_type": "bc.role_type",
    "target_mode": "bc.target_mode",
    "opportunity_score_bucket": """CASE
        WHEN bc.opportunity_score >= 90 THEN '90-100'
        WHEN bc.opportunity_score >= 70 THEN '70-89'
        WHEN bc.opportunity_score >= 50 THEN '50-69'
        ELSE 'below_50'
    END""",
    "urgency_bucket": """CASE
        WHEN bc.urgency_score >= 8 THEN 'high_8+'
        WHEN bc.urgency_score >= 5 THEN 'medium_5-7'
        ELSE 'low_0-4'
    END""",
    "pain_category": "bc.pain_categories->0->>'category'",
}


@mcp.tool()
async def get_signal_effectiveness(
    vendor_name: Optional[str] = None,
    min_sequences: int = 5,
    group_by: str = "buying_stage",
) -> str:
    """
    Analyze which signal dimensions produce the best campaign outcomes.

    Groups completed campaign sequences by a signal dimension from the
    originating b2b_campaigns row (step 1 only) and computes outcome rates.

    vendor_name: Optional vendor filter (partial match, case-insensitive)
    min_sequences: Minimum sequences per group to include (default 5, 1-100)
    group_by: Signal dimension to group by. Options:
        buying_stage (default), role_type, target_mode,
        opportunity_score_bucket, urgency_bucket, pain_category
    """
    min_sequences = max(1, min(min_sequences, 100))

    if group_by not in _GROUP_BY_EXPRESSIONS:
        return json.dumps({
            "success": False,
            "error": f"Invalid group_by. Must be one of: {sorted(_GROUP_BY_EXPRESSIONS.keys())}",
        })

    group_expr = _GROUP_BY_EXPRESSIONS[group_by]

    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()

        vendor_filter = ""
        params: list = [min_sequences]
        idx = 2

        if vendor_name:
            vendor_filter = f"AND bc.vendor_name ILIKE '%' || ${idx} || '%'"
            params.append(vendor_name)
            idx += 1

        sql = f"""
        WITH seq_signals AS (
            SELECT DISTINCT ON (cs.id)
                cs.id, cs.outcome, cs.outcome_revenue,
                ({group_expr}) AS signal_group
            FROM campaign_sequences cs
            JOIN b2b_campaigns bc ON bc.sequence_id = cs.id
            WHERE bc.sequence_id IS NOT NULL
              AND cs.outcome != 'pending'
              {vendor_filter}
            ORDER BY cs.id, bc.step_number ASC
        )
        SELECT
            signal_group,
            COUNT(*) AS total_sequences,
            COUNT(*) FILTER (WHERE outcome = 'meeting_booked') AS meetings,
            COUNT(*) FILTER (WHERE outcome = 'deal_opened') AS deals_opened,
            COUNT(*) FILTER (WHERE outcome = 'deal_won') AS deals_won,
            COUNT(*) FILTER (WHERE outcome = 'deal_lost') AS deals_lost,
            COUNT(*) FILTER (WHERE outcome = 'no_opportunity') AS no_opportunity,
            COUNT(*) FILTER (WHERE outcome = 'disqualified') AS disqualified,
            ROUND(
                COUNT(*) FILTER (WHERE outcome IN ('meeting_booked', 'deal_opened', 'deal_won'))::numeric
                / NULLIF(COUNT(*), 0), 3
            ) AS positive_outcome_rate,
            COALESCE(SUM(outcome_revenue) FILTER (WHERE outcome = 'deal_won'), 0) AS total_revenue
        FROM seq_signals
        WHERE signal_group IS NOT NULL
        GROUP BY signal_group
        HAVING COUNT(*) >= $1
        ORDER BY positive_outcome_rate DESC
        """

        rows = await pool.fetch(sql, *params)

        groups = []
        for r in rows:
            groups.append({
                "signal_group": r["signal_group"],
                "total_sequences": r["total_sequences"],
                "meetings": r["meetings"],
                "deals_opened": r["deals_opened"],
                "deals_won": r["deals_won"],
                "deals_lost": r["deals_lost"],
                "no_opportunity": r["no_opportunity"],
                "disqualified": r["disqualified"],
                "positive_outcome_rate": float(r["positive_outcome_rate"]) if r["positive_outcome_rate"] is not None else 0.0,
                "total_revenue": float(r["total_revenue"]),
            })

        return json.dumps({
            "success": True,
            "group_by": group_by,
            "vendor_filter": vendor_name,
            "min_sequences": min_sequences,
            "groups": groups,
            "total_groups": len(groups),
        }, default=str)
    except Exception:
        logger.exception("get_signal_effectiveness error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def get_outcome_distribution(
    vendor_name: Optional[str] = None,
) -> str:
    """System-wide campaign outcome distribution (funnel view).

    Returns count and percentage for each outcome state, plus total revenue.

    vendor_name: Optional vendor filter (partial match, case-insensitive).
    """
    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        conditions: list[str] = []
        params: list = []
        idx = 1

        if vendor_name:
            conditions.append(f"bc.vendor_name ILIKE '%' || ${idx} || '%'")
            params.append(vendor_name)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        rows = await pool.fetch(
            f"""
            SELECT cs.outcome,
                   COUNT(*) AS count,
                   COALESCE(SUM(cs.outcome_revenue), 0) AS total_revenue,
                   MIN(cs.outcome_recorded_at) AS first_recorded,
                   MAX(cs.outcome_recorded_at) AS last_recorded
            FROM campaign_sequences cs
            LEFT JOIN b2b_campaigns bc ON bc.sequence_id = cs.id AND bc.step_number = 1
            {where}
            GROUP BY cs.outcome
            ORDER BY count DESC
            """,
            *params,
        )

        total = sum(r["count"] for r in rows)
        buckets = []
        for r in rows:
            buckets.append({
                "outcome": r["outcome"],
                "count": r["count"],
                "pct": round(r["count"] / total * 100, 1) if total else 0,
                "total_revenue": float(r["total_revenue"]) if r["total_revenue"] else 0.0,
                "first_recorded": str(r["first_recorded"]) if r["first_recorded"] else None,
                "last_recorded": str(r["last_recorded"]) if r["last_recorded"] else None,
            })

        return json.dumps({
            "success": True,
            "total_sequences": total,
            "vendor_filter": vendor_name,
            "buckets": buckets,
        }, default=str)
    except Exception:
        logger.exception("get_outcome_distribution error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def trigger_score_calibration(
    window_days: int = 180,
) -> str:
    """Trigger score calibration from campaign outcomes on-demand.

    Recalculates calibration weights based on observed outcome rates per
    signal dimension. Results are stored in score_calibration_weights.

    window_days: How many days of outcome data to consider (default 180, min 30, max 730).
    """
    try:
        from ..autonomous.tasks.b2b_score_calibration import calibrate
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        window_days = max(30, min(window_days, 730))
        result = await calibrate(pool, window_days=window_days)
        return json.dumps({
            "success": True,
            "triggered_by": "mcp",
            "window_days": window_days,
            **result,
        }, default=str)
    except Exception:
        logger.exception("trigger_score_calibration error")
        return json.dumps({"success": False, "error": "Calibration failed"})


# ---------------------------------------------------------------------------
# Tool: create_data_correction
# ---------------------------------------------------------------------------

_VALID_ENTITY_TYPES = {
    "review", "vendor", "displacement_edge", "pain_point",
    "churn_signal", "buyer_profile", "use_case", "integration",
    "source",
}
_VALID_CORRECTION_TYPES = {"suppress", "flag", "override_field", "merge_vendor", "reclassify", "suppress_source"}
_KNOWN_SOURCES = {s.value for s in ReviewSource}


@mcp.tool()
async def create_data_correction(
    entity_type: str,
    entity_id: str,
    correction_type: str,
    reason: str,
    field_name: Optional[str] = None,
    old_value: Optional[str] = None,
    new_value: Optional[str] = None,
    corrected_by: str = "mcp",
    metadata: Optional[str] = None,
) -> str:
    """
    Record an analyst correction for a data entity.

    entity_type: Type of entity being corrected (review, vendor, displacement_edge,
        pain_point, churn_signal, buyer_profile, use_case, integration, source)
    entity_id: UUID of the entity being corrected
    correction_type: Type of correction (suppress, flag, override_field, merge_vendor,
        reclassify, suppress_source)
    reason: Human explanation for the correction (required)
    field_name: Which field was changed (required for override_field; for suppress_source,
        optional vendor scope)
    old_value: Previous value (optional)
    new_value: New value (required for override_field)
    corrected_by: Who made the correction (default: mcp)
    metadata: Optional JSON string with extra data (e.g., '{"source_name": "reddit"}' for
        suppress_source)
    """
    if entity_type not in _VALID_ENTITY_TYPES:
        return json.dumps({
            "success": False,
            "error": f"Invalid entity_type. Must be one of: {sorted(_VALID_ENTITY_TYPES)}",
        })
    if correction_type not in _VALID_CORRECTION_TYPES:
        return json.dumps({
            "success": False,
            "error": f"Invalid correction_type. Must be one of: {sorted(_VALID_CORRECTION_TYPES)}",
        })
    if correction_type == "override_field":
        if not field_name:
            return json.dumps({"success": False, "error": "field_name required for override_field"})
        if new_value is None:
            return json.dumps({"success": False, "error": "new_value required for override_field"})
    if correction_type == "merge_vendor":
        if not old_value or not new_value:
            return json.dumps({
                "success": False,
                "error": "merge_vendor requires old_value (source vendor) and new_value (target vendor)",
            })
    if correction_type == "suppress_source":
        if entity_type != "source":
            return json.dumps({
                "success": False,
                "error": "suppress_source corrections must use entity_type='source'",
            })
        meta_dict = {}
        if metadata:
            try:
                meta_dict = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                return json.dumps({"success": False, "error": "metadata must be valid JSON"})
        if not meta_dict.get("source_name"):
            return json.dumps({
                "success": False,
                "error": "suppress_source requires metadata with source_name (e.g., '{\"source_name\": \"reddit\"}')",
            })
        if meta_dict["source_name"].lower() not in _KNOWN_SOURCES:
            return json.dumps({
                "success": False,
                "error": f"Unknown source '{meta_dict['source_name']}'. Known: {sorted(_KNOWN_SOURCES)}",
            })
    if not _is_uuid(entity_id):
        return json.dumps({"success": False, "error": "entity_id must be a valid UUID"})
    if not reason or not reason.strip():
        return json.dumps({"success": False, "error": "reason is required"})

    try:
        from ..storage.database import get_db_pool
        pool = get_db_pool()
        import uuid as _u
        entity_uuid = _u.UUID(entity_id)
        meta_json = metadata if metadata else "{}"

        row = await pool.fetchrow(
            """
            INSERT INTO data_corrections
                (entity_type, entity_id, correction_type, field_name,
                 old_value, new_value, reason, corrected_by, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
            RETURNING id, entity_type, entity_id, correction_type, status, created_at
            """,
            entity_type,
            entity_uuid,
            correction_type,
            field_name,
            old_value,
            new_value,
            reason,
            corrected_by,
            meta_json,
        )

        correction_id = row["id"]

        # Execute vendor merge if applicable
        merge_info = None
        if correction_type == "merge_vendor":
            from ..services.b2b.vendor_merge import execute_vendor_merge
            merge_result = await execute_vendor_merge(pool, old_value, new_value)
            await pool.execute(
                "UPDATE data_corrections SET affected_count = $1, metadata = $2::jsonb WHERE id = $3",
                merge_result["total_affected"], json.dumps(merge_result), correction_id,
            )
            merge_info = merge_result

        result = {
            "success": True,
            "correction": {
                "id": str(correction_id),
                "entity_type": row["entity_type"],
                "entity_id": str(row["entity_id"]),
                "correction_type": row["correction_type"],
                "status": row["status"],
                "created_at": str(row["created_at"]),
            },
        }
        if merge_info:
            result["merge"] = merge_info

        return json.dumps(result, default=str)
    except Exception:
        logger.exception("create_data_correction error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: list_data_corrections
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_data_corrections(
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    correction_type: Optional[str] = None,
    status: Optional[str] = None,
    corrected_by: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 50,
) -> str:
    """
    List recorded data corrections with optional filters.

    entity_type: Filter by entity type (review, vendor, etc.)
    entity_id: Filter by entity UUID
    correction_type: Filter by correction type (suppress, flag, override_field, etc.)
    status: Filter by status (applied, reverted, pending_review)
    corrected_by: Filter by who made the correction (substring match)
    start_date: Corrections created on or after (ISO 8601)
    end_date: Corrections created before (ISO 8601)
    limit: Max results (1-200, default 50)
    """
    limit = max(1, min(limit, 200))

    if entity_type and entity_type not in _VALID_ENTITY_TYPES:
        return json.dumps({
            "success": False,
            "error": f"Invalid entity_type. Must be one of: {sorted(_VALID_ENTITY_TYPES)}",
        })
    if correction_type and correction_type not in _VALID_CORRECTION_TYPES:
        return json.dumps({
            "success": False,
            "error": f"Invalid correction_type. Must be one of: {sorted(_VALID_CORRECTION_TYPES)}",
        })
    if entity_id and not _is_uuid(entity_id):
        return json.dumps({"success": False, "error": "entity_id must be a valid UUID"})

    try:
        from datetime import datetime as _dt

        from ..storage.database import get_db_pool
        pool = get_db_pool()

        conditions: list[str] = []
        params: list = []
        idx = 1

        if entity_type:
            conditions.append(f"entity_type = ${idx}")
            params.append(entity_type)
            idx += 1
        if entity_id:
            import uuid as _u
            conditions.append(f"entity_id = ${idx}")
            params.append(_u.UUID(entity_id))
            idx += 1
        if correction_type:
            conditions.append(f"correction_type = ${idx}")
            params.append(correction_type)
            idx += 1
        if status:
            conditions.append(f"status = ${idx}")
            params.append(status)
            idx += 1
        if corrected_by:
            conditions.append(f"corrected_by ILIKE '%' || ${idx} || '%'")
            params.append(corrected_by)
            idx += 1
        if start_date:
            try:
                sd = _dt.fromisoformat(start_date.replace("Z", "+00:00"))
                conditions.append(f"created_at >= ${idx}")
                params.append(sd)
                idx += 1
            except ValueError:
                return json.dumps({"error": "Invalid start_date (ISO 8601 expected)"})
        if end_date:
            try:
                ed = _dt.fromisoformat(end_date.replace("Z", "+00:00"))
                conditions.append(f"created_at < ${idx}")
                params.append(ed)
                idx += 1
            except ValueError:
                return json.dumps({"error": "Invalid end_date (ISO 8601 expected)"})

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = await pool.fetch(
            f"""
            SELECT id, entity_type, entity_id, correction_type, field_name,
                   old_value, new_value, reason, corrected_by, status,
                   affected_count, metadata, created_at, reverted_at, reverted_by
            FROM data_corrections
            {where}
            ORDER BY created_at DESC
            LIMIT ${idx}
            """,
            *params,
        )

        corrections = []
        for r in rows:
            corrections.append({
                "id": str(r["id"]),
                "entity_type": r["entity_type"],
                "entity_id": str(r["entity_id"]),
                "correction_type": r["correction_type"],
                "field_name": r["field_name"],
                "old_value": r["old_value"],
                "new_value": r["new_value"],
                "reason": r["reason"],
                "corrected_by": r["corrected_by"],
                "status": r["status"],
                "affected_count": r["affected_count"],
                "metadata": _safe_json(r["metadata"]),
                "created_at": str(r["created_at"]),
                "reverted_at": str(r["reverted_at"]) if r["reverted_at"] else None,
                "reverted_by": r["reverted_by"],
            })

        return json.dumps({
            "success": True,
            "corrections": corrections,
            "count": len(corrections),
        }, default=str)
    except Exception:
        logger.exception("list_data_corrections error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: revert_data_correction
# ---------------------------------------------------------------------------


@mcp.tool()
async def revert_data_correction(
    correction_id: str,
    reason: Optional[str] = None,
    reverted_by: str = "mcp",
) -> str:
    """
    Revert a previously applied data correction.

    correction_id: UUID of the correction to revert
    reason: Optional explanation for the revert
    reverted_by: Who reverted the correction (default: mcp)
    """
    if not _is_uuid(correction_id):
        return json.dumps({"success": False, "error": "correction_id must be a valid UUID"})

    try:
        from ..storage.database import get_db_pool
        pool = get_db_pool()
        import uuid as _u
        cid = _u.UUID(correction_id)

        row = await pool.fetchrow(
            "SELECT id, status FROM data_corrections WHERE id = $1",
            cid,
        )
        if not row:
            return json.dumps({"success": False, "error": "Correction not found"})
        if row["status"] != "applied":
            return json.dumps({
                "success": False,
                "error": f"Cannot revert correction with status '{row['status']}' (must be 'applied')",
            })

        updated = await pool.fetchrow(
            """
            UPDATE data_corrections
            SET status = 'reverted', reverted_at = NOW(), reverted_by = $2
            WHERE id = $1
            RETURNING id, status, reverted_at
            """,
            cid,
            reverted_by,
        )

        return json.dumps({
            "success": True,
            "id": str(updated["id"]),
            "status": updated["status"],
            "reverted_at": str(updated["reverted_at"]),
        }, default=str)
    except Exception:
        logger.exception("revert_data_correction error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def get_data_correction(
    correction_id: str,
) -> str:
    """Fetch a single data correction by ID with full details.

    correction_id: UUID of the correction to fetch
    """
    if not _is_uuid(correction_id):
        return json.dumps({"error": "correction_id must be a valid UUID"})

    try:
        import uuid as _u

        from ..storage.database import get_db_pool

        pool = get_db_pool()
        row = await pool.fetchrow(
            """
            SELECT id, entity_type, entity_id, correction_type, field_name,
                   old_value, new_value, reason, corrected_by, status,
                   affected_count, metadata, created_at, reverted_at, reverted_by
            FROM data_corrections
            WHERE id = $1
            """,
            _u.UUID(correction_id),
        )
        if not row:
            return json.dumps({"error": "Correction not found"})

        return json.dumps({
            "success": True,
            "correction": {
                "id": str(row["id"]),
                "entity_type": row["entity_type"],
                "entity_id": str(row["entity_id"]),
                "correction_type": row["correction_type"],
                "field_name": row["field_name"],
                "old_value": row["old_value"],
                "new_value": row["new_value"],
                "reason": row["reason"],
                "corrected_by": row["corrected_by"],
                "status": row["status"],
                "affected_count": row["affected_count"],
                "metadata": _safe_json(row["metadata"]),
                "created_at": str(row["created_at"]),
                "reverted_at": str(row["reverted_at"]) if row["reverted_at"] else None,
                "reverted_by": row["reverted_by"],
            },
        }, default=str)
    except Exception:
        logger.exception("get_data_correction error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def get_correction_stats(
    days: int = 30,
) -> str:
    """Aggregate correction activity: counts by type, status, and top correctors.

    days: Window in days (default 30, max 365)
    """
    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        days = max(1, min(days, 365))

        row = await pool.fetchrow(
            """
            SELECT
                COUNT(*) AS total_corrections,
                COUNT(*) FILTER (WHERE status = 'applied') AS applied,
                COUNT(*) FILTER (WHERE status = 'reverted') AS reverted,
                COUNT(*) FILTER (WHERE status = 'pending_review') AS pending_review,
                COUNT(*) FILTER (WHERE correction_type = 'suppress') AS suppress_count,
                COUNT(*) FILTER (WHERE correction_type = 'flag') AS flag_count,
                COUNT(*) FILTER (WHERE correction_type = 'override_field') AS override_count,
                COUNT(*) FILTER (WHERE correction_type = 'merge_vendor') AS merge_count,
                COUNT(*) FILTER (WHERE correction_type = 'reclassify') AS reclassify_count,
                COUNT(*) FILTER (WHERE correction_type = 'suppress_source') AS suppress_source_count,
                COALESCE(SUM(affected_count), 0) AS total_affected_records,
                MIN(created_at) AS first_correction_at,
                MAX(created_at) AS last_correction_at
            FROM data_corrections
            WHERE created_at > NOW() - ($1 || ' days')::interval
            """,
            str(days),
        )

        correctors = await pool.fetch(
            """
            SELECT corrected_by, COUNT(*) AS count
            FROM data_corrections
            WHERE created_at > NOW() - ($1 || ' days')::interval
            GROUP BY corrected_by
            ORDER BY count DESC
            LIMIT 10
            """,
            str(days),
        )

        by_entity = await pool.fetch(
            """
            SELECT entity_type, COUNT(*) AS count
            FROM data_corrections
            WHERE created_at > NOW() - ($1 || ' days')::interval
            GROUP BY entity_type
            ORDER BY count DESC
            """,
            str(days),
        )

        return json.dumps({
            "success": True,
            "window_days": days,
            "total_corrections": row["total_corrections"],
            "by_status": {
                "applied": row["applied"],
                "reverted": row["reverted"],
                "pending_review": row["pending_review"],
            },
            "by_type": {
                "suppress": row["suppress_count"],
                "flag": row["flag_count"],
                "override_field": row["override_count"],
                "merge_vendor": row["merge_count"],
                "reclassify": row["reclassify_count"],
                "suppress_source": row["suppress_source_count"],
            },
            "by_entity": [{"entity_type": r["entity_type"], "count": r["count"]} for r in by_entity],
            "total_affected_records": row["total_affected_records"],
            "top_correctors": [{"corrected_by": r["corrected_by"], "count": r["count"]} for r in correctors],
            "first_correction_at": str(row["first_correction_at"]) if row["first_correction_at"] else None,
            "last_correction_at": str(row["last_correction_at"]) if row["last_correction_at"] else None,
        }, default=str)
    except Exception:
        logger.exception("get_correction_stats error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: get_source_correction_impact
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_source_correction_impact() -> str:
    """
    Show impact of active source suppressions on review counts.

    Returns each active suppress_source correction with the number of enriched
    reviews that would be excluded by it (globally or vendor-scoped).
    """
    try:
        from ..storage.database import get_db_pool
        pool = get_db_pool()

        rows = await pool.fetch(
            """
            SELECT dc.metadata->>'source_name' AS source_name,
                   dc.field_name AS vendor_scope,
                   dc.reason,
                   dc.created_at,
                   (SELECT COUNT(*) FROM b2b_reviews r
                    WHERE LOWER(r.source) = LOWER(dc.metadata->>'source_name')
                      AND (dc.field_name IS NULL OR LOWER(r.vendor_name) = LOWER(dc.field_name))
                      AND r.enrichment_status = 'enriched'
                   ) AS affected_review_count
            FROM data_corrections dc
            WHERE dc.entity_type = 'source'
              AND dc.correction_type = 'suppress_source'
              AND dc.status = 'applied'
            ORDER BY dc.created_at DESC
            """,
        )

        return json.dumps({
            "success": True,
            "active_source_suppressions": [
                {
                    "source_name": r["source_name"],
                    "vendor_scope": r["vendor_scope"],
                    "reason": r["reason"],
                    "affected_review_count": r["affected_review_count"],
                    "created_at": r["created_at"],
                }
                for r in rows
            ],
            "total": len(rows),
        }, default=str)
    except Exception:
        logger.exception("get_source_correction_impact error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: get_parser_health
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_parser_health() -> str:
    """
    Show parser version distribution and identify stale reviews.

    Returns per-source counts of reviews by parser version, the latest version
    for each source, and a stale count where reviews were parsed by an older
    version than the latest for that source.
    """
    pool = _pool_or_fail()
    try:
        rows = await pool.fetch("""
            WITH version_counts AS (
                SELECT source,
                       COALESCE(parser_version, 'unknown') AS parser_version,
                       COUNT(*) AS review_count
                FROM b2b_reviews
                GROUP BY source, COALESCE(parser_version, 'unknown')
            ),
            latest AS (
                SELECT DISTINCT ON (source) source, parser_version AS latest_version
                FROM b2b_scrape_log
                WHERE parser_version IS NOT NULL
                ORDER BY source, started_at DESC
            )
            SELECT vc.source, vc.parser_version, vc.review_count,
                   l.latest_version,
                   (vc.parser_version != COALESCE(l.latest_version, vc.parser_version))
                       AS is_stale
            FROM version_counts vc
            LEFT JOIN latest l USING (source)
            ORDER BY vc.source, vc.review_count DESC
        """)

        sources: dict[str, dict] = {}
        for r in rows:
            src = r["source"]
            if src not in sources:
                sources[src] = {
                    "source": src,
                    "latest_version": r["latest_version"],
                    "versions": [],
                    "total_reviews": 0,
                    "stale_reviews": 0,
                }
            entry = sources[src]
            entry["versions"].append({
                "parser_version": r["parser_version"],
                "review_count": r["review_count"],
                "is_stale": r["is_stale"],
            })
            entry["total_reviews"] += r["review_count"]
            if r["is_stale"]:
                entry["stale_reviews"] += r["review_count"]

        result = sorted(sources.values(), key=lambda x: x["stale_reviews"], reverse=True)
        total_stale = sum(s["stale_reviews"] for s in result)
        return json.dumps({
            "success": True,
            "sources": result,
            "total_stale_reviews": total_stale,
            "total_sources": len(result),
        }, default=str)
    except Exception:
        logger.exception("get_parser_health error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: get_calibration_weights
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_calibration_weights(
    dimension: Optional[str] = None,
    model_version: Optional[int] = None,
) -> str:
    """
    Get current score calibration weights derived from campaign outcomes.

    Calibration weights adjust the static opportunity scoring formula based
    on observed conversion rates per signal dimension.

    dimension: Filter by dimension (role_type, buying_stage, urgency_bucket, seat_bucket, context_keyword)
    model_version: Specific version (default: latest)
    """
    valid_dims = {"role_type", "buying_stage", "urgency_bucket", "seat_bucket", "context_keyword"}
    if dimension and dimension not in valid_dims:
        return json.dumps({"success": False, "error": f"Invalid dimension. Must be one of: {sorted(valid_dims)}"})

    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()

        if model_version is None:
            model_version = await pool.fetchval(
                "SELECT MAX(model_version) FROM score_calibration_weights"
            )
            if model_version is None:
                return json.dumps({
                    "success": True,
                    "weights": [],
                    "count": 0,
                    "message": "No calibration data yet. Run calibration after recording campaign outcomes.",
                })

        conditions = ["model_version = $1"]
        params: list = [model_version]
        idx = 2

        if dimension:
            conditions.append(f"dimension = ${idx}")
            params.append(dimension)
            idx += 1

        where = " AND ".join(conditions)

        rows = await pool.fetch(
            f"""
            SELECT dimension, dimension_value, total_sequences, positive_outcomes,
                   deals_won, total_revenue, positive_rate, baseline_rate, lift,
                   weight_adjustment, static_default, calibrated_at,
                   sample_window_days, model_version
            FROM score_calibration_weights
            WHERE {where}
            ORDER BY dimension, lift DESC
            """,
            *params,
        )

        weights = []
        for r in rows:
            weights.append({
                "dimension": r["dimension"],
                "dimension_value": r["dimension_value"],
                "total_sequences": r["total_sequences"],
                "positive_outcomes": r["positive_outcomes"],
                "deals_won": r["deals_won"],
                "total_revenue": float(r["total_revenue"]),
                "positive_rate": float(r["positive_rate"]),
                "baseline_rate": float(r["baseline_rate"]),
                "lift": float(r["lift"]),
                "weight_adjustment": float(r["weight_adjustment"]),
                "static_default": float(r["static_default"]),
                "calibrated_at": r["calibrated_at"],
                "sample_window_days": r["sample_window_days"],
                "model_version": r["model_version"],
            })

        return json.dumps({
            "success": True,
            "model_version": model_version,
            "weights": weights,
            "count": len(weights),
        }, default=str)
    except Exception:
        logger.exception("get_calibration_weights error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: trigger_score_calibration
# ---------------------------------------------------------------------------


@mcp.tool()
async def trigger_score_calibration(
    window_days: int = 90,
) -> str:
    """
    Manually trigger score calibration from campaign outcome data.

    Computes conversion rates per signal dimension and derives weight
    adjustments. Requires at least 20 sequences with recorded outcomes.

    window_days: How far back to look for outcome data (default 90, max 365)
    """
    window_days = max(1, min(window_days, 365))
    try:
        from ..autonomous.tasks.b2b_score_calibration import calibrate
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        result = await calibrate(pool, window_days=window_days)
        return json.dumps({"success": True, **result}, default=str)
    except Exception:
        logger.exception("trigger_score_calibration error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: list_webhook_subscriptions
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_webhook_subscriptions(
    account_id: Optional[str] = None,
    enabled_only: bool = True,
) -> str:
    """
    List webhook subscriptions for debugging and monitoring.

    account_id: Optional UUID to filter by specific account
    enabled_only: If true, only show enabled subscriptions (default true)
    """
    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        conditions = []
        params: list = []
        idx = 1

        if account_id:
            if not _is_uuid(account_id):
                return json.dumps({"error": "account_id must be a valid UUID"})
            conditions.append(f"ws.account_id = ${idx}::uuid")
            params.append(account_id)
            idx += 1

        if enabled_only:
            conditions.append("ws.enabled = true")

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        rows = await pool.fetch(
            f"""
            SELECT ws.id, ws.account_id, ws.url, ws.event_types,
                   COALESCE(ws.channel, 'generic') AS channel,
                   ws.enabled, ws.description, ws.created_at,
                   sa.name AS account_name,
                   (SELECT COUNT(*) FROM b2b_webhook_delivery_log dl
                    WHERE dl.subscription_id = ws.id AND dl.delivered_at > NOW() - INTERVAL '7 days') AS recent_deliveries,
                   (SELECT COUNT(*) FILTER (WHERE dl2.success) FROM b2b_webhook_delivery_log dl2
                    WHERE dl2.subscription_id = ws.id AND dl2.delivered_at > NOW() - INTERVAL '7 days') AS recent_successes
            FROM b2b_webhook_subscriptions ws
            JOIN saas_accounts sa ON sa.id = ws.account_id
            {where}
            ORDER BY ws.created_at DESC
            """,
            *params,
        )

        subs = []
        for r in rows:
            recent_total = r["recent_deliveries"] or 0
            subs.append({
                "id": str(r["id"]),
                "account_id": str(r["account_id"]),
                "account_name": r["account_name"],
                "url": r["url"],
                "event_types": r["event_types"],
                "channel": r["channel"],
                "enabled": r["enabled"],
                "description": r["description"],
                "created_at": r["created_at"].isoformat(),
                "recent_deliveries_7d": recent_total,
                "recent_success_rate_7d": round(r["recent_successes"] / max(recent_total, 1), 3) if recent_total else None,
            })

        return json.dumps({"subscriptions": subs, "count": len(subs)}, default=str)
    except Exception:
        logger.exception("list_webhook_subscriptions error")
        return json.dumps({"error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: send_test_webhook
# ---------------------------------------------------------------------------


@mcp.tool()
async def send_test_webhook_tool(
    subscription_id: str,
) -> str:
    """
    Send a test payload to a webhook subscription to verify connectivity.

    subscription_id: UUID of the webhook subscription to test
    """
    if not _is_uuid(subscription_id):
        return json.dumps({"error": "subscription_id must be a valid UUID"})
    try:
        import uuid as _u

        from ..services.b2b.webhook_dispatcher import send_test_webhook
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        result = await send_test_webhook(pool, _u.UUID(subscription_id))
        return json.dumps(result, default=str)
    except Exception:
        logger.exception("send_test_webhook error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def update_webhook(
    subscription_id: str,
    enabled: Optional[bool] = None,
    event_types: Optional[str] = None,
    url: Optional[str] = None,
    description: Optional[str] = None,
) -> str:
    """Update a webhook subscription (toggle enabled, change event_types/url/description).

    subscription_id: UUID of the webhook to update
    enabled: Set to true/false to enable/disable the webhook
    event_types: Comma-separated list of event types (change_event, churn_alert, report_generated, signal_update)
    url: New webhook URL (must start with https:// or http://)
    description: New description
    """
    if not _is_uuid(subscription_id):
        return json.dumps({"error": "subscription_id must be a valid UUID"})

    _valid_events = {"change_event", "churn_alert", "report_generated", "signal_update"}

    try:
        import uuid as _u

        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        sets: list[str] = []
        params: list = []
        idx = 1

        if enabled is not None:
            sets.append(f"enabled = ${idx}")
            params.append(enabled)
            idx += 1
        if event_types is not None:
            et_list = [e.strip() for e in event_types.split(",") if e.strip()]
            invalid = set(et_list) - _valid_events
            if invalid:
                return json.dumps({"error": f"Invalid event_types: {sorted(invalid)}. Valid: {sorted(_valid_events)}"})
            if not et_list:
                return json.dumps({"error": "event_types must not be empty"})
            sets.append(f"event_types = ${idx}")
            params.append(et_list)
            idx += 1
        if url is not None:
            if not url.startswith(("https://", "http://")):
                return json.dumps({"error": "url must begin with https:// or http://"})
            sets.append(f"url = ${idx}")
            params.append(url)
            idx += 1
        if description is not None:
            sets.append(f"description = ${idx}")
            params.append(description)
            idx += 1

        if not sets:
            return json.dumps({"error": "No fields to update"})

        sets.append("updated_at = NOW()")
        params.append(_u.UUID(subscription_id))

        row = await pool.fetchrow(
            f"""
            UPDATE b2b_webhook_subscriptions
            SET {', '.join(sets)}
            WHERE id = ${idx}
            RETURNING id, url, event_types, COALESCE(channel, 'generic') AS channel,
                      enabled, description, updated_at
            """,
            *params,
        )
        if not row:
            return json.dumps({"error": "Webhook not found"})

        return json.dumps({
            "success": True,
            "webhook": {
                "id": str(row["id"]),
                "url": row["url"],
                "event_types": row["event_types"],
                "channel": row["channel"],
                "enabled": row["enabled"],
                "description": row["description"],
                "updated_at": row["updated_at"].isoformat(),
            },
        }, default=str)
    except Exception:
        logger.exception("update_webhook error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def get_webhook_delivery_summary(
    account_id: Optional[str] = None,
    days: int = 7,
) -> str:
    """Aggregate webhook delivery health across all subscriptions.

    account_id: Optional UUID to filter by account (shows all if omitted)
    days: Window in days for delivery stats (default 7, max 90)
    """
    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        days = max(1, min(days, 90))
        conditions = ["ws.enabled = true"]
        params: list = [str(days)]
        idx = 2

        if account_id:
            if not _is_uuid(account_id):
                return json.dumps({"error": "account_id must be a valid UUID"})
            conditions.append(f"ws.account_id = ${idx}::uuid")
            params.append(account_id)
            idx += 1

        where = " AND ".join(conditions)

        row = await pool.fetchrow(
            f"""
            SELECT
                COUNT(DISTINCT ws.id) AS active_subscriptions,
                COUNT(dl.id) AS total_deliveries,
                COUNT(dl.id) FILTER (WHERE dl.success) AS successful,
                COUNT(dl.id) FILTER (WHERE NOT dl.success) AS failed,
                AVG(dl.duration_ms) FILTER (WHERE dl.success) AS avg_success_duration_ms,
                MAX(dl.delivered_at) AS last_delivery_at
            FROM b2b_webhook_subscriptions ws
            LEFT JOIN b2b_webhook_delivery_log dl
                ON dl.subscription_id = ws.id
                AND dl.delivered_at > NOW() - ($1 || ' days')::interval
            WHERE {where}
            """,
            *params,
        )

        total = row["total_deliveries"] or 0
        return json.dumps({
            "success": True,
            "window_days": days,
            "active_subscriptions": row["active_subscriptions"] or 0,
            "total_deliveries": total,
            "successful": row["successful"] or 0,
            "failed": row["failed"] or 0,
            "success_rate": round((row["successful"] or 0) / max(total, 1), 3) if total else None,
            "avg_success_duration_ms": round(row["avg_success_duration_ms"], 1) if row["avg_success_duration_ms"] else None,
            "last_delivery_at": str(row["last_delivery_at"]) if row["last_delivery_at"] else None,
        }, default=str)
    except Exception:
        logger.exception("get_webhook_delivery_summary error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: list_crm_pushes
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_crm_pushes(
    subscription_id: Optional[str] = None,
    vendor_name: Optional[str] = None,
    limit: int = 50,
) -> str:
    """
    List CRM push log entries showing what intelligence data was pushed to external CRMs.

    subscription_id: Optional UUID to filter by specific webhook subscription
    vendor_name: Optional vendor name filter (case-insensitive)
    limit: Max results (default 50, max 200)
    """
    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        conditions = []
        params: list = []
        idx = 1

        if subscription_id:
            if not _is_uuid(subscription_id):
                return json.dumps({"error": "subscription_id must be a valid UUID"})
            conditions.append(f"pl.subscription_id = ${idx}::uuid")
            params.append(subscription_id)
            idx += 1

        if vendor_name:
            conditions.append(f"pl.vendor_name ILIKE ${idx}")
            params.append(f"%{vendor_name}%")
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit = max(1, min(limit, 200))

        rows = await pool.fetch(
            f"""
            SELECT pl.id, pl.subscription_id, pl.signal_type, pl.vendor_name,
                   pl.company_name, pl.crm_record_id, pl.crm_record_type,
                   pl.status, pl.error, pl.pushed_at,
                   COALESCE(ws.channel, 'generic') AS channel,
                   ws.url AS webhook_url
            FROM b2b_crm_push_log pl
            JOIN b2b_webhook_subscriptions ws ON ws.id = pl.subscription_id
            {where}
            ORDER BY pl.pushed_at DESC
            LIMIT {limit}
            """,
            *params,
        )

        pushes = []
        for r in rows:
            pushes.append({
                "id": str(r["id"]),
                "subscription_id": str(r["subscription_id"]),
                "channel": r["channel"],
                "webhook_url": r["webhook_url"],
                "signal_type": r["signal_type"],
                "vendor_name": r["vendor_name"],
                "company_name": r["company_name"],
                "crm_record_id": r["crm_record_id"],
                "crm_record_type": r["crm_record_type"],
                "status": r["status"],
                "error": r["error"],
                "pushed_at": r["pushed_at"].isoformat(),
            })

        return json.dumps({"pushes": pushes, "count": len(pushes)}, default=str)
    except Exception:
        logger.exception("list_crm_pushes error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def export_report_pdf(report_id: str) -> str:
    """Export a B2B intelligence report as a PDF download link.

    Returns base64-encoded PDF bytes so MCP clients can save the file locally.

    Args:
        report_id: UUID of the report from b2b_intelligence table.
    """
    if not _is_uuid(report_id):
        return json.dumps({"error": "report_id must be a valid UUID"})
    try:
        import base64
        import uuid as _u

        from ..services.b2b.pdf_renderer import render_report_pdf
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        row = await pool.fetchrow(
            "SELECT * FROM b2b_intelligence WHERE id = $1",
            _u.UUID(report_id),
        )
        if not row:
            return json.dumps({"error": "Report not found"})

        intel_data = row["intelligence_data"]
        if isinstance(intel_data, str):
            try:
                intel_data = json.loads(intel_data)
            except (json.JSONDecodeError, TypeError):
                intel_data = {}

        density = row["data_density"]
        if isinstance(density, str):
            try:
                density = json.loads(density)
            except (json.JSONDecodeError, TypeError):
                density = {}

        pdf_bytes = render_report_pdf(
            report_type=row["report_type"],
            vendor_filter=row["vendor_filter"],
            category_filter=row["category_filter"],
            report_date=row["report_date"],
            executive_summary=row["executive_summary"],
            intelligence_data=intel_data,
            data_density=density,
        )

        import re as _re

        vendor = row["vendor_filter"] or row["report_type"]
        filename = f"atlas-report-{vendor}-{row['report_date'] or 'latest'}.pdf"
        filename = _re.sub(r"[^a-z0-9._-]", "-", filename.lower())

        return json.dumps({
            "filename": filename,
            "size_bytes": len(pdf_bytes),
            "content_base64": base64.b64encode(pdf_bytes).decode(),
        })
    except Exception:
        logger.exception("export_report_pdf error")
        return json.dumps({"error": "Failed to generate PDF"})


@mcp.tool()
async def list_crm_events(
    status: Optional[str] = None,
    crm_provider: Optional[str] = None,
    company_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 50,
) -> str:
    """List ingested CRM events with optional filters.

    Args:
        status: Filter by processing status (pending, matched, unmatched, skipped).
        crm_provider: Filter by CRM provider (hubspot, salesforce, pipedrive, generic).
        company_name: Filter by company name (partial, case-insensitive).
        start_date: Filter events received on or after (ISO 8601, e.g. 2026-01-01).
        end_date: Filter events received before (ISO 8601).
        limit: Max events to return (default 50, max 200).
    """
    try:
        from datetime import datetime as _dt

        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        _valid_statuses = {"pending", "matched", "unmatched", "skipped"}
        if status and status not in _valid_statuses:
            return json.dumps({"error": f"Invalid status. Must be one of: {sorted(_valid_statuses)}"})

        limit = max(1, min(limit, 200))
        conditions = ["1=1"]
        params: list = []
        idx = 1

        if status:
            conditions.append(f"status = ${idx}")
            params.append(status)
            idx += 1
        if crm_provider:
            conditions.append(f"crm_provider = ${idx}")
            params.append(crm_provider)
            idx += 1
        if company_name:
            conditions.append(f"LOWER(company_name) LIKE '%' || LOWER(${idx}) || '%'")
            params.append(company_name)
            idx += 1
        if start_date:
            try:
                sd = _dt.fromisoformat(start_date.replace("Z", "+00:00"))
                conditions.append(f"received_at >= ${idx}")
                params.append(sd)
                idx += 1
            except ValueError:
                return json.dumps({"error": "Invalid start_date (ISO 8601 expected)"})
        if end_date:
            try:
                ed = _dt.fromisoformat(end_date.replace("Z", "+00:00"))
                conditions.append(f"received_at < ${idx}")
                params.append(ed)
                idx += 1
            except ValueError:
                return json.dumps({"error": "Invalid end_date (ISO 8601 expected)"})

        where = " AND ".join(conditions)
        params.append(limit)

        rows = await pool.fetch(
            f"""
            SELECT id, crm_provider, event_type, company_name, contact_email,
                   deal_stage, deal_amount, status, matched_sequence_id,
                   outcome_recorded, processing_notes,
                   event_timestamp, received_at, processed_at
            FROM b2b_crm_events
            WHERE {where}
            ORDER BY received_at DESC
            LIMIT ${idx}
            """,
            *params,
        )

        events = []
        for r in rows:
            events.append({
                "id": str(r["id"]),
                "crm_provider": r["crm_provider"],
                "event_type": r["event_type"],
                "company_name": r["company_name"],
                "contact_email": r["contact_email"],
                "deal_stage": r["deal_stage"],
                "deal_amount": float(r["deal_amount"]) if r["deal_amount"] else None,
                "status": r["status"],
                "matched_sequence_id": str(r["matched_sequence_id"]) if r["matched_sequence_id"] else None,
                "outcome_recorded": r["outcome_recorded"],
                "received_at": str(r["received_at"]),
            })

        return json.dumps({"events": events, "count": len(events)}, default=str)
    except Exception:
        logger.exception("list_crm_events error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def ingest_crm_event(
    crm_provider: str,
    event_type: str,
    company_name: Optional[str] = None,
    contact_email: Optional[str] = None,
    deal_stage: Optional[str] = None,
    deal_amount: Optional[float] = None,
    deal_id: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    """Manually ingest a CRM event for processing.

    The event will be stored and processed by the crm_event_processing task
    which matches it to campaign sequences and auto-records outcomes.

    Args:
        crm_provider: CRM source (hubspot, salesforce, pipedrive, generic).
        event_type: Event type (deal_won, deal_lost, meeting_booked, deal_stage_change, etc.).
        company_name: Company name for matching to campaign sequences.
        contact_email: Contact email for matching to campaign sequences.
        deal_stage: Current deal stage in the CRM.
        deal_amount: Deal value if available.
        deal_id: CRM deal/opportunity ID.
        notes: Optional notes about this event.
    """
    valid_providers = {"hubspot", "salesforce", "pipedrive", "generic"}
    valid_types = {
        "deal_stage_change", "deal_won", "deal_lost",
        "meeting_booked", "activity_logged", "contact_updated",
    }
    if crm_provider not in valid_providers:
        return json.dumps({"error": f"crm_provider must be one of {sorted(valid_providers)}"})
    if event_type not in valid_types:
        return json.dumps({"error": f"event_type must be one of {sorted(valid_types)}"})
    if not company_name and not contact_email:
        return json.dumps({"error": "At least one of company_name or contact_email is required"})

    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        event_data = {}
        if notes:
            event_data["notes"] = notes

        event_id = await pool.fetchval(
            """
            INSERT INTO b2b_crm_events (
                crm_provider, event_type, company_name, contact_email,
                deal_id, deal_stage, deal_amount, event_data
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
            RETURNING id
            """,
            crm_provider, event_type, company_name, contact_email,
            deal_id, deal_stage, deal_amount,
            json.dumps(event_data, default=str),
        )

        return json.dumps({
            "success": True,
            "event_id": str(event_id),
            "status": "pending",
            "message": "Event ingested. Will be processed by crm_event_processing task.",
        })
    except Exception:
        logger.exception("ingest_crm_event error")
        return json.dumps({"error": "Failed to ingest CRM event"})


# ---------------------------------------------------------------------------
# Tool: get_crm_enrichment_stats
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_crm_enrichment_stats() -> str:
    """
    Show enrichment coverage and effectiveness stats for CRM events.

    Returns total events, match rates, field coverage (company_name, contact_email),
    and enrichment counts (cross-event lookups, vendor normalization).
    """
    try:
        from ..storage.database import get_db_pool
        pool = get_db_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        row = await pool.fetchrow(
            """
            SELECT
                COUNT(*) AS total_events,
                COUNT(*) FILTER (WHERE status = 'matched') AS matched,
                COUNT(*) FILTER (WHERE status = 'unmatched') AS unmatched,
                COUNT(*) FILTER (WHERE status = 'skipped') AS skipped,
                COUNT(*) FILTER (WHERE status = 'pending') AS pending,
                COUNT(*) FILTER (WHERE status = 'error') AS errored,
                COUNT(*) FILTER (WHERE company_name IS NOT NULL) AS has_company,
                COUNT(*) FILTER (WHERE contact_email IS NOT NULL) AS has_email,
                COUNT(*) FILTER (WHERE company_name IS NULL AND contact_email IS NULL) AS missing_both,
                COUNT(*) FILTER (WHERE processing_notes LIKE '%[enriched]%') AS enriched_count,
                COUNT(*) FILTER (WHERE processing_notes LIKE '%[enriched]%' AND status = 'matched') AS enriched_matched
            FROM b2b_crm_events
            """
        )

        total = row["total_events"] or 0
        return json.dumps({
            "total_events": total,
            "matched": row["matched"],
            "unmatched": row["unmatched"],
            "skipped": row["skipped"],
            "pending": row["pending"],
            "errored": row["errored"],
            "field_coverage": {
                "has_company_name": row["has_company"],
                "has_contact_email": row["has_email"],
                "missing_both": row["missing_both"],
            },
            "enrichment": {
                "events_enriched": row["enriched_count"],
                "enriched_then_matched": row["enriched_matched"],
            },
            "match_rate": round(row["matched"] / total * 100, 1) if total else 0,
        })
    except Exception:
        logger.exception("get_crm_enrichment_stats error")
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
