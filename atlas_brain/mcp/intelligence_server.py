"""
Atlas Intelligence MCP Server.

Exposes intelligence report generation, consumer product review data,
brand intelligence, pain point analysis, and pressure baseline queries
to any MCP-compatible client (Claude Desktop, Cursor, custom agents).

Tools:
    -- Strategic Intelligence --
    generate_intelligence_report  -- generate a full or executive report for an entity
    list_intelligence_reports     -- list recent reports with optional entity filter
    get_intelligence_report       -- fetch a stored report by ID
    list_pressure_baselines       -- list entity pressure baselines (highest first)
    analyze_risk_sensors          -- run behavioral risk sensors on text
    run_intervention_pipeline     -- three-stage intervention analysis
    list_pending_approvals        -- list pending intervention approvals
    review_approval               -- approve or reject intervention requests

    -- Consumer Product Intelligence --
    search_product_reviews        -- search enriched Amazon product reviews
    get_product_review            -- fetch single review with full enrichment
    list_pain_points              -- top products by pain score
    get_brand_intelligence        -- brand health scorecard with competitive flows
    list_brands                   -- list brands sorted by health score
    list_market_reports           -- competitive intelligence reports
    get_market_report             -- fetch full market intelligence report
    get_consumer_pipeline_status  -- enrichment pipeline health snapshot
    list_complaint_content        -- generated content (articles, forum posts, email copy)

    -- Consumer History & Change Detection --
    get_brand_history             -- daily brand health snapshot time-series
    list_product_change_events    -- anomaly detection events (spikes, drops, emergences)

    -- Consumer Corrections --
    create_consumer_correction    -- suppress, flag, override, reclassify consumer entities
    list_consumer_corrections     -- list corrections with filters
    revert_consumer_correction    -- revert a previously applied correction

    -- Consumer Brand Registry --
    fuzzy_brand_search            -- trigram similarity search on brand names
    add_brand_to_registry         -- add/update canonical brand with aliases
    add_brand_alias               -- add alias to existing brand
    list_brand_registry           -- list all canonical brands

    -- Consumer Delivery Surfaces --
    export_market_report_pdf          -- export market intelligence report as PDF
    export_brand_report_pdf           -- export brand intelligence scorecard as PDF
    send_brand_health_digest          -- send email digest of brand health summaries

    -- Consumer Cross-Brand Correlation --
    list_concurrent_events            -- dates where 3+ brands had same event type
    get_brand_correlation             -- aligned snapshot time-series + Pearson r

    -- Consumer Displacement Edges --
    list_product_displacement_edges  -- query brand-to-brand competitive flows
    get_product_displacement_history -- time-series for a specific brand pair

Run:
    python -m atlas_brain.mcp.intelligence_server          # stdio
    python -m atlas_brain.mcp.intelligence_server --sse    # SSE HTTP transport
"""

import json
import logging
import sys
import uuid as _uuid
from contextlib import asynccontextmanager
from typing import Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("atlas.mcp.intelligence")

VALID_ENTITY_TYPES = ("company", "person", "sector")


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
    logger.info("Intelligence MCP: DB pool initialized")
    yield
    await close_database()


mcp = FastMCP(
    "atlas-intelligence",
    instructions=(
        "Intelligence and consumer product review server for Atlas. "
        "Generate behavioral intelligence reports, query Amazon product reviews, "
        "analyze brand health scores, view pain points and competitive flows, "
        "monitor enrichment pipeline health, and run intervention analysis. "
        "Consumer data sourced from Amazon product reviews with two-pass "
        "enrichment (complaint classification + deep extraction)."
    ),
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Tool: generate_intelligence_report
# ---------------------------------------------------------------------------

@mcp.tool()
async def generate_intelligence_report(
    entity_name: str,
    entity_type: str = "company",
    time_window_days: int = 7,
    report_type: str = "full",
    audience: str = "executive",
) -> str:
    """
    Generate an intelligence report for a specific entity.

    entity_name: The entity to analyze (company, person, sector name)
    entity_type: Classification -- "company", "person", or "sector"
    time_window_days: How far back to look for data (1-90 days, default 7)
    report_type: "full" (600-word report) or "executive" (200-word summary)
    audience: Target reader -- "executive", "ops lead", or "investor"
    """
    if not entity_name or not entity_name.strip():
        return json.dumps({"success": False, "error": "entity_name is required"})
    if entity_type not in VALID_ENTITY_TYPES:
        return json.dumps({"success": False, "error": f"entity_type must be one of {VALID_ENTITY_TYPES}"})

    try:
        from ..services.intelligence_report import generate_report

        result = await generate_report(
            entity_name=entity_name.strip(),
            entity_type=entity_type,
            time_window_days=max(1, min(90, time_window_days)),
            report_type=report_type if report_type in ("full", "executive") else "full",
            audience=audience,
            requested_by="mcp",
        )
        return json.dumps({"success": True, **result}, default=str)
    except Exception as exc:
        logger.exception("generate_intelligence_report error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: list_intelligence_reports
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_intelligence_reports(
    entity_name: Optional[str] = None,
    limit: int = 20,
) -> str:
    """
    List recent intelligence reports, optionally filtered by entity name.

    entity_name: Filter by entity name (partial match, case-insensitive)
    limit: Maximum reports to return (default 20)
    """
    try:
        from ..services.intelligence_report import list_reports

        reports = await list_reports(
            entity_name=entity_name,
            limit=min(limit, 100),
        )
        return json.dumps({"reports": reports, "count": len(reports)}, default=str)
    except Exception as exc:
        logger.exception("list_intelligence_reports error")
        return json.dumps({"error": "Internal error", "reports": []})


# ---------------------------------------------------------------------------
# Tool: get_intelligence_report
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_intelligence_report(report_id: str) -> str:
    """
    Fetch a full intelligence report by its UUID.

    report_id: UUID of the report to retrieve
    """
    if not _is_uuid(report_id):
        return json.dumps({"error": "Invalid report_id (must be UUID)", "found": False})

    try:
        from ..services.intelligence_report import get_report

        report = await get_report(report_id)
        if not report:
            return json.dumps({"error": "Report not found", "found": False})
        return json.dumps({"found": True, "report": report}, default=str)
    except Exception as exc:
        logger.exception("get_intelligence_report error")
        return json.dumps({"error": "Internal error", "found": False})


# ---------------------------------------------------------------------------
# Tool: list_pressure_baselines
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_pressure_baselines(
    entity_type: Optional[str] = None,
    limit: int = 20,
) -> str:
    """
    List entity pressure baselines sorted by pressure score (highest first).

    Pressure scores range 0-10:
      0-3 = background noise
      4-6 = elevated attention
      7-8 = significant accumulation
      9-10 = critical, event imminent

    entity_type: Filter by type ("company", "person", "sector")
    limit: Maximum results (default 20)
    """
    if entity_type and entity_type not in VALID_ENTITY_TYPES:
        return json.dumps({"error": f"entity_type must be one of {VALID_ENTITY_TYPES}", "baselines": []})

    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not available", "baselines": []})

        if entity_type:
            rows = await pool.fetch(
                """
                SELECT entity_name, entity_type, pressure_score, sentiment_drift,
                       narrative_frequency, soram_breakdown, linguistic_signals,
                       last_computed_at
                FROM entity_pressure_baselines
                WHERE entity_type = $1
                ORDER BY pressure_score DESC
                LIMIT $2
                """,
                entity_type,
                min(limit, 100),
            )
        else:
            rows = await pool.fetch(
                """
                SELECT entity_name, entity_type, pressure_score, sentiment_drift,
                       narrative_frequency, soram_breakdown, linguistic_signals,
                       last_computed_at
                FROM entity_pressure_baselines
                ORDER BY pressure_score DESC
                LIMIT $1
                """,
                min(limit, 100),
            )

        baselines = [
            {
                "entity_name": r["entity_name"],
                "entity_type": r["entity_type"],
                "pressure_score": float(r["pressure_score"]) if r["pressure_score"] is not None else 0.0,
                "sentiment_drift": float(r["sentiment_drift"]) if r["sentiment_drift"] is not None else 0.0,
                "narrative_frequency": r["narrative_frequency"] or 0,
                "soram_breakdown": r["soram_breakdown"] if isinstance(r["soram_breakdown"], dict) else {},
                "linguistic_signals": r["linguistic_signals"] if isinstance(r["linguistic_signals"], dict) else {},
                "last_computed_at": r["last_computed_at"].isoformat() if r["last_computed_at"] else None,
            }
            for r in rows
        ]

        return json.dumps({"baselines": baselines, "count": len(baselines)})
    except Exception as exc:
        logger.exception("list_pressure_baselines error")
        return json.dumps({"error": "Internal error", "baselines": []})


# ---------------------------------------------------------------------------
# Tool: analyze_risk_sensors
# ---------------------------------------------------------------------------

@mcp.tool()
async def analyze_risk_sensors(text: str) -> str:
    """
    Run all 3 behavioral risk sensors on text and return cross-correlation.

    Sensors:
      - Alignment: collaborative vs adversarial language
      - Operational Urgency: planning vs reactive/emergency language
      - Negotiation Rigidity: flexibility vs absolutist language

    Returns per-sensor results plus composite risk level (LOW/MEDIUM/HIGH/CRITICAL).
    """
    if not text or len(text.strip()) < 10:
        return json.dumps({"success": False, "error": "Text too short for analysis (minimum 10 characters)"})
    if len(text) > 50000:
        return json.dumps({"success": False, "error": "Text too long for analysis (maximum 50000 characters)"})

    try:
        from ..tools.risk_sensors import (
            alignment_sensor_tool,
            operational_urgency_tool,
            negotiation_rigidity_tool,
            correlate,
        )

        alignment = alignment_sensor_tool.analyze(text)
        urgency = operational_urgency_tool.analyze(text)
        rigidity = negotiation_rigidity_tool.analyze(text)
        cross = correlate(alignment, urgency, rigidity)

        return json.dumps({
            "success": True,
            "alignment": alignment,
            "operational_urgency": urgency,
            "negotiation_rigidity": rigidity,
            "cross_correlation": cross,
        })
    except Exception as exc:
        logger.exception("analyze_risk_sensors error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: run_intervention_pipeline
# ---------------------------------------------------------------------------

@mcp.tool()
async def run_intervention_pipeline(
    entity_name: str,
    entity_type: str = "company",
    time_window_days: int = 7,
    objectives: str = "de-escalate, stabilize",
    constraints: str = "",
    audience: str = "executive",
    risk_tolerance: str = "moderate",
    simulation_horizon: str = "7 days",
    hours_before_event: int = 48,
    channels: str = "internal comms",
    allow_narrative_architect: bool = False,
) -> str:
    """
    Run the three-stage intervention pipeline for an entity.

    Stage 1: Adaptive Intervention -- report findings to F.A.T.E. tactical playbook
    Stage 2: Simulated Evolution -- playbook + signals to scenario matrix and outcome trajectories
    Stage 3: Narrative Architect -- simulation to micro-intervention plan (safety-gated)

    entity_name: The entity to analyze
    entity_type: "company", "person", or "sector"
    time_window_days: How far back to look (1-90 days, default 7)
    objectives: Comma-separated goals (de-escalate, stabilize, capitalize)
    constraints: Comma-separated legal, ethical, or operational constraints
    audience: Target reader -- "executive", "ops lead", or "negotiator"
    risk_tolerance: "low", "moderate", or "high"
    simulation_horizon: Time window for projections (e.g. "7 days", "2 weeks")
    hours_before_event: T-minus hours for calibration checkpoints (default 48)
    channels: Comma-separated communication channels for interventions
    allow_narrative_architect: Enable stage 3 (blocked by default until safety layer exists)
    """
    if not entity_name or not entity_name.strip():
        return json.dumps({"success": False, "error": "entity_name is required"})
    if entity_type not in VALID_ENTITY_TYPES:
        return json.dumps({"success": False, "error": f"entity_type must be one of {VALID_ENTITY_TYPES}"})

    try:
        from ..services.intervention_pipeline import (
            run_intervention_pipeline as _run_pipeline,
        )

        result = await _run_pipeline(
            entity_name=entity_name.strip(),
            entity_type=entity_type,
            time_window_days=max(1, min(90, time_window_days)),
            objectives=[o.strip() for o in objectives.split(",")],
            constraints=[c.strip() for c in constraints.split(",") if c.strip()] if constraints else None,
            audience=audience if audience in ("executive", "ops lead", "negotiator") else "executive",
            risk_tolerance=risk_tolerance if risk_tolerance in ("low", "moderate", "high") else "moderate",
            simulation_horizon=simulation_horizon,
            hours_before_event=max(1, min(720, hours_before_event)),
            channels=[c.strip() for c in channels.split(",") if c.strip()] if channels else None,
            allow_narrative_architect=allow_narrative_architect,
            requested_by="mcp",
        )
        return json.dumps({"success": True, **result}, default=str)
    except Exception as exc:
        logger.exception("run_intervention_pipeline error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: list_pending_approvals
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_pending_approvals(
    entity_name: Optional[str] = None,
    limit: int = 20,
) -> str:
    """
    List pending intervention approval requests.

    High-risk intervention pipeline stages require human approval before
    proceeding. Use this tool to see what needs review.

    entity_name: Filter by entity name (optional, partial match)
    limit: Maximum results (default 20)
    """
    try:
        from ..services.safety_gate import get_safety_gate

        gate = get_safety_gate()
        approvals = await gate.list_pending(entity_name=entity_name, limit=min(limit, 100))
        return json.dumps({"approvals": approvals, "count": len(approvals)}, default=str)
    except Exception as exc:
        logger.exception("list_pending_approvals error")
        return json.dumps({"error": "Internal error", "approvals": []})


# ---------------------------------------------------------------------------
# Tool: review_approval
# ---------------------------------------------------------------------------

@mcp.tool()
async def review_approval(
    approval_id: str,
    action: str = "approve",
    reviewed_by: str = "mcp_user",
    notes: str = "",
) -> str:
    """
    Approve or reject a pending intervention approval request.

    approval_id: UUID of the approval request
    action: "approve" or "reject"
    reviewed_by: Identifier of the reviewer
    notes: Optional review notes
    """
    if not _is_uuid(approval_id):
        return json.dumps({"success": False, "error": "Invalid approval_id (must be UUID)"})
    if action not in ("approve", "reject"):
        return json.dumps({"success": False, "error": "action must be 'approve' or 'reject'"})

    try:
        from ..services.safety_gate import get_safety_gate

        gate = get_safety_gate()

        if action == "approve":
            ok = await gate.approve(approval_id, reviewed_by, notes)
        else:
            ok = await gate.reject(approval_id, reviewed_by, notes)

        if ok:
            return json.dumps({"success": True, "status": action + "d", "approval_id": approval_id})
        return json.dumps({"success": False, "error": "Approval not found or already reviewed"})
    except Exception as exc:
        logger.exception("review_approval error")
        return json.dumps({"success": False, "error": "Internal error"})


# ===========================================================================
# Consumer Product Intelligence Tools
# ===========================================================================


# ---------------------------------------------------------------------------
# Tool: search_product_reviews
# ---------------------------------------------------------------------------


@mcp.tool()
async def search_product_reviews(
    asin: Optional[str] = None,
    brand: Optional[str] = None,
    category: Optional[str] = None,
    min_pain_score: Optional[float] = None,
    has_safety_flag: Optional[bool] = None,
    enrichment_status: Optional[str] = None,
    window_days: int = 30,
    limit: int = 20,
) -> str:
    """
    Search enriched Amazon product reviews with flexible filters.

    asin: Filter by Amazon ASIN (exact match)
    brand: Filter by brand name (partial match from product_metadata)
    category: Filter by hardware_category or source_category (partial match)
    min_pain_score: Minimum pain_score threshold (0-10)
    has_safety_flag: Filter by safety_flagged in deep_extraction
    enrichment_status: Filter by status (pending, enriched, failed, not_applicable)
    window_days: How far back to look in days (default 30)
    limit: Maximum results (default 20, cap 100)
    """
    limit = max(1, min(limit, 100))
    window_days = max(1, min(window_days, 3650))
    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        conditions = [
            "r.imported_at > NOW() - make_interval(days => $1)",
        ]
        params: list = [window_days]
        idx = 2

        if asin:
            conditions.append(f"r.asin = ${idx}")
            params.append(asin.strip().upper())
            idx += 1

        if brand:
            conditions.append(f"m.brand ILIKE '%' || ${idx} || '%'")
            params.append(brand)
            idx += 1

        if category:
            conditions.append(
                f"(r.hardware_category::text ILIKE '%' || ${idx} || '%' "
                f"OR r.source_category ILIKE '%' || ${idx} || '%' "
                f"OR m.store ILIKE '%' || ${idx} || '%')"
            )
            params.append(category)
            idx += 1

        if min_pain_score is not None:
            conditions.append(f"r.pain_score >= ${idx}")
            params.append(max(0.0, min(float(min_pain_score), 10.0)))
            idx += 1

        if has_safety_flag is not None:
            conditions.append(
                f"(r.deep_extraction->>'safety_flagged')::boolean = ${idx}"
            )
            params.append(has_safety_flag)
            idx += 1

        if enrichment_status:
            conditions.append(f"r.enrichment_status = ${idx}")
            params.append(enrichment_status)
            idx += 1

        params.append(limit)
        where = " AND ".join(conditions)

        rows = await pool.fetch(
            f"""
            SELECT r.id, r.asin, r.rating, r.summary, r.pain_score,
                   r.root_cause, r.specific_complaint, r.severity,
                   r.enrichment_status, r.deep_enrichment_status,
                   r.imported_at, r.enriched_at,
                   m.brand, m.title AS product_title, m.store AS product_category
            FROM product_reviews r
            LEFT JOIN product_metadata m ON r.asin = m.asin
            WHERE {where}
            ORDER BY r.pain_score DESC NULLS LAST
            LIMIT ${idx}
            """,
            *params,
        )

        reviews = [
            {
                "id": str(r["id"]),
                "asin": r["asin"],
                "brand": r["brand"],
                "product_title": r["product_title"],
                "product_category": r["product_category"],
                "rating": r["rating"],
                "summary": r["summary"],
                "pain_score": float(r["pain_score"]) if r["pain_score"] is not None else None,
                "root_cause": r["root_cause"],
                "specific_complaint": r["specific_complaint"],
                "severity": r["severity"],
                "enrichment_status": r["enrichment_status"],
                "deep_enrichment_status": r["deep_enrichment_status"],
                "imported_at": r["imported_at"],
                "enriched_at": r["enriched_at"],
            }
            for r in rows
        ]

        return json.dumps({"reviews": reviews, "count": len(reviews)}, default=str)
    except Exception:
        logger.exception("search_product_reviews error")
        return json.dumps({"error": "Internal error", "reviews": [], "count": 0})


# ---------------------------------------------------------------------------
# Tool: get_product_review
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_product_review(review_id: str) -> str:
    """
    Fetch a single Amazon product review with full enrichment and deep extraction.

    review_id: UUID of the review to retrieve
    """
    if not _is_uuid(review_id):
        return json.dumps({"success": False, "error": "Invalid review_id (must be UUID)"})

    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        row = await pool.fetchrow(
            """
            SELECT r.*, m.brand, m.title AS product_title, m.store AS product_category
            FROM product_reviews r
            LEFT JOIN product_metadata m ON r.asin = m.asin
            WHERE r.id = $1
            """,
            _uuid.UUID(review_id),
        )

        if not row:
            return json.dumps({"success": False, "error": "Review not found"})

        review = {
            "id": str(row["id"]),
            "asin": row["asin"],
            "brand": row["brand"],
            "product_title": row["product_title"],
            "product_category": row["product_category"],
            "rating": row["rating"],
            "summary": row["summary"],
            "review_text": row["review_text"],
            "reviewer_id": row["reviewer_id"],
            "source": row["source"],
            "source_category": row["source_category"],
            "hardware_category": row["hardware_category"],
            "pain_score": float(row["pain_score"]) if row["pain_score"] is not None else None,
            "root_cause": row["root_cause"],
            "specific_complaint": row["specific_complaint"],
            "severity": row["severity"],
            "issue_types": _safe_json(row["issue_types"]),
            "time_to_failure": row["time_to_failure"],
            "workaround_found": row["workaround_found"],
            "alternative_mentioned": row["alternative_mentioned"],
            "enrichment_status": row["enrichment_status"],
            "enriched_at": row["enriched_at"],
            "deep_extraction": _safe_json(row["deep_extraction"]),
            "deep_enrichment_status": row["deep_enrichment_status"],
            "deep_enriched_at": row["deep_enriched_at"],
            "imported_at": row["imported_at"],
        }

        return json.dumps({"success": True, "review": review}, default=str)
    except Exception:
        logger.exception("get_product_review error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: list_pain_points
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_pain_points(
    category: Optional[str] = None,
    min_pain_score: float = 0,
    min_complaint_rate: Optional[float] = None,
    limit: int = 20,
) -> str:
    """
    List products ranked by pain score -- highest-pain products first.

    category: Filter by product category (partial match)
    min_pain_score: Minimum pain_score threshold (default 0)
    min_complaint_rate: Minimum complaint_rate (0-1, e.g. 0.5 = 50%)
    limit: Maximum results (default 20, cap 100)
    """
    limit = max(1, min(limit, 100))
    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        conditions = []
        params = []
        idx = 1

        if min_pain_score > 0:
            conditions.append(f"pain_score >= ${idx}")
            params.append(max(0.0, min(float(min_pain_score), 10.0)))
            idx += 1

        if min_complaint_rate is not None:
            conditions.append(f"complaint_rate >= ${idx}")
            params.append(max(0.0, min(float(min_complaint_rate), 1.0)))
            idx += 1

        if category:
            conditions.append(f"category ILIKE '%' || ${idx} || '%'")
            params.append(category)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = await pool.fetch(
            f"""
            SELECT asin, product_name, category, total_reviews,
                   complaint_reviews, complaint_rate, pain_score,
                   top_complaints, root_cause_distribution
            FROM product_pain_points
            {where}
            ORDER BY pain_score DESC
            LIMIT ${idx}
            """,
            *params,
        )

        pain_points = [
            {
                "asin": r["asin"],
                "product_name": r["product_name"],
                "category": r["category"],
                "total_reviews": r["total_reviews"],
                "complaint_reviews": r["complaint_reviews"],
                "complaint_rate": float(r["complaint_rate"]) if r["complaint_rate"] is not None else None,
                "pain_score": float(r["pain_score"]) if r["pain_score"] is not None else None,
                "top_complaints": _safe_json(r["top_complaints"]),
                "root_cause_distribution": _safe_json(r["root_cause_distribution"]),
            }
            for r in rows
        ]

        return json.dumps({"pain_points": pain_points, "count": len(pain_points)}, default=str)
    except Exception:
        logger.exception("list_pain_points error")
        return json.dumps({"error": "Internal error", "pain_points": [], "count": 0})


# ---------------------------------------------------------------------------
# Tool: list_brands
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_brands(
    min_health_score: Optional[float] = None,
    sort_by: str = "health_score",
    limit: int = 20,
) -> str:
    """
    List brands sorted by health score from brand_intelligence.

    min_health_score: Minimum health_score threshold
    sort_by: Sort field -- "health_score" (default), "total_reviews", "avg_rating",
             "avg_pain_score"
    limit: Maximum results (default 20, cap 100)
    """
    limit = max(1, min(limit, 100))
    valid_sorts = {
        "health_score": "health_score DESC NULLS LAST",
        "total_reviews": "total_reviews DESC",
        "avg_rating": "avg_rating DESC NULLS LAST",
        "avg_pain_score": "avg_pain_score DESC NULLS LAST",
    }
    order = valid_sorts.get(sort_by, valid_sorts["health_score"])

    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        conditions = []
        params = []
        idx = 1

        if min_health_score is not None:
            conditions.append(f"health_score >= ${idx}")
            params.append(float(min_health_score))
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = await pool.fetch(
            f"""
            SELECT brand, source, total_reviews, avg_rating, avg_pain_score,
                   repurchase_yes, repurchase_no, health_score,
                   top_complaints, top_feature_requests,
                   competitive_flows, positive_aspects,
                   last_computed_at
            FROM brand_intelligence
            {where}
            ORDER BY {order}
            LIMIT ${idx}
            """,
            *params,
        )

        brands = [
            {
                "brand": r["brand"],
                "source": r["source"],
                "total_reviews": r["total_reviews"],
                "avg_rating": float(r["avg_rating"]) if r["avg_rating"] is not None else None,
                "avg_pain_score": float(r["avg_pain_score"]) if r["avg_pain_score"] is not None else None,
                "repurchase_yes": r["repurchase_yes"],
                "repurchase_no": r["repurchase_no"],
                "health_score": float(r["health_score"]) if r["health_score"] is not None else None,
                "top_complaints": _safe_json(r["top_complaints"]),
                "top_feature_requests": _safe_json(r["top_feature_requests"]),
                "competitive_flows": _safe_json(r["competitive_flows"]),
                "positive_aspects": _safe_json(r["positive_aspects"]),
                "last_computed_at": r["last_computed_at"],
            }
            for r in rows
        ]

        return json.dumps({"brands": brands, "count": len(brands)}, default=str)
    except Exception:
        logger.exception("list_brands error")
        return json.dumps({"error": "Internal error", "brands": [], "count": 0})


# ---------------------------------------------------------------------------
# Tool: get_brand_intelligence
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_brand_intelligence(brand: str) -> str:
    """
    Detailed brand health scorecard with competitive flows, buyer profile,
    sentiment breakdown, and feature requests.

    brand: Brand name (partial match, case-insensitive)
    """
    if not brand or not brand.strip():
        return json.dumps({"success": False, "error": "brand is required"})

    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        row = await pool.fetchrow(
            """
            SELECT * FROM brand_intelligence
            WHERE brand ILIKE '%' || $1 || '%'
            ORDER BY total_reviews DESC
            LIMIT 1
            """,
            brand.strip(),
        )

        if not row:
            return json.dumps({"success": False, "error": f"No brand intelligence for '{brand}'"})

        bi = {
            "brand": row["brand"],
            "source": row["source"],
            "total_reviews": row["total_reviews"],
            "avg_rating": float(row["avg_rating"]) if row["avg_rating"] is not None else None,
            "avg_pain_score": float(row["avg_pain_score"]) if row["avg_pain_score"] is not None else None,
            "repurchase_yes": row["repurchase_yes"],
            "repurchase_no": row["repurchase_no"],
            "health_score": float(row["health_score"]) if row["health_score"] is not None else None,
            "sentiment_breakdown": _safe_json(row["sentiment_breakdown"]),
            "top_complaints": _safe_json(row["top_complaints"]),
            "top_feature_requests": _safe_json(row["top_feature_requests"]),
            "competitive_flows": _safe_json(row["competitive_flows"]),
            "buyer_profile": _safe_json(row["buyer_profile"]),
            "positive_aspects": _safe_json(row["positive_aspects"]),
            "last_computed_at": row["last_computed_at"],
        }

        return json.dumps({"success": True, "brand": bi}, default=str)
    except Exception:
        logger.exception("get_brand_intelligence error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: list_market_reports
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_market_reports(
    report_type: Optional[str] = None,
    limit: int = 10,
) -> str:
    """
    List competitive intelligence / market intelligence reports.

    report_type: Filter by type (e.g. "daily_competitive")
    limit: Maximum results (default 10, cap 50)
    """
    limit = max(1, min(limit, 50))
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

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = await pool.fetch(
            f"""
            SELECT id, report_date, report_type, analysis_text,
                   created_at
            FROM market_intelligence_reports
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
                "analysis_text": (r["analysis_text"] or "")[:500] + ("..." if r["analysis_text"] and len(r["analysis_text"]) > 500 else ""),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

        return json.dumps({"reports": reports, "count": len(reports)}, default=str)
    except Exception:
        logger.exception("list_market_reports error")
        return json.dumps({"error": "Internal error", "reports": [], "count": 0})


# ---------------------------------------------------------------------------
# Tool: get_market_report
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_market_report(report_id: str) -> str:
    """
    Fetch a full market intelligence report by UUID.

    report_id: UUID of the report
    """
    if not _is_uuid(report_id):
        return json.dumps({"success": False, "error": "Invalid report_id (must be UUID)"})

    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        row = await pool.fetchrow(
            "SELECT * FROM market_intelligence_reports WHERE id = $1",
            _uuid.UUID(report_id),
        )

        if not row:
            return json.dumps({"success": False, "error": "Report not found"})

        report = {
            "id": str(row["id"]),
            "report_date": row["report_date"],
            "report_type": row["report_type"],
            "analysis_text": row["analysis_text"],
            "competitive_flows": _safe_json(row["competitive_flows"]),
            "feature_gaps": _safe_json(row["feature_gaps"]),
            "buyer_personas": _safe_json(row["buyer_personas"]),
            "brand_scorecards": _safe_json(row["brand_scorecards"]),
            "insights": _safe_json(row["insights"]),
            "recommendations": _safe_json(row["recommendations"]),
            "created_at": row["created_at"],
        }

        return json.dumps({"success": True, "report": report}, default=str)
    except Exception:
        logger.exception("get_market_report error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: get_consumer_pipeline_status
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_consumer_pipeline_status() -> str:
    """
    Consumer review enrichment pipeline health snapshot.

    Returns enrichment counts by status (basic + deep), recent imports,
    brand intelligence count, pain point count, and content generation stats.
    """
    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()

        # Basic enrichment counts
        basic_rows = await pool.fetch(
            """
            SELECT enrichment_status, COUNT(*) AS cnt
            FROM product_reviews
            GROUP BY enrichment_status
            """
        )
        basic_counts = {r["enrichment_status"]: r["cnt"] for r in basic_rows}

        # Deep enrichment counts
        deep_rows = await pool.fetch(
            """
            SELECT deep_enrichment_status, COUNT(*) AS cnt
            FROM product_reviews
            WHERE deep_enrichment_status IS NOT NULL
            GROUP BY deep_enrichment_status
            """
        )
        deep_counts = {r["deep_enrichment_status"]: r["cnt"] for r in deep_rows}

        # Pipeline stats
        stats = await pool.fetchrow(
            """
            SELECT
                COUNT(*) AS total_reviews,
                COUNT(*) FILTER (WHERE imported_at > NOW() - INTERVAL '24 hours') AS imports_24h,
                MAX(enriched_at) AS last_enrichment_at,
                MAX(deep_enriched_at) AS last_deep_enrichment_at
            FROM product_reviews
            """
        )

        # Brand intelligence count
        brand_count = await pool.fetchrow(
            "SELECT COUNT(*) AS cnt FROM brand_intelligence"
        )

        # Pain points count
        pain_count = await pool.fetchrow(
            "SELECT COUNT(*) AS cnt FROM product_pain_points"
        )

        # Content stats
        content_stats = await pool.fetchrow(
            """
            SELECT
                COUNT(*) AS total_content,
                COUNT(*) FILTER (WHERE status = 'draft') AS drafts,
                COUNT(*) FILTER (WHERE status = 'published') AS published
            FROM complaint_content
            """
        )

        # Blog post stats
        blog_stats = await pool.fetchrow(
            """
            SELECT
                COUNT(*) AS total_posts,
                COUNT(*) FILTER (WHERE status = 'draft') AS drafts,
                COUNT(*) FILTER (WHERE status = 'published') AS published
            FROM blog_posts
            """
        )

        result = {
            "basic_enrichment_counts": basic_counts,
            "deep_enrichment_counts": deep_counts,
            "total_reviews": stats["total_reviews"] if stats else 0,
            "imports_24h": stats["imports_24h"] if stats else 0,
            "last_enrichment_at": stats["last_enrichment_at"] if stats else None,
            "last_deep_enrichment_at": stats["last_deep_enrichment_at"] if stats else None,
            "brand_intelligence_count": brand_count["cnt"] if brand_count else 0,
            "pain_point_products": pain_count["cnt"] if pain_count else 0,
            "content": {
                "total": content_stats["total_content"] if content_stats else 0,
                "drafts": content_stats["drafts"] if content_stats else 0,
                "published": content_stats["published"] if content_stats else 0,
            },
            "blog_posts": {
                "total": blog_stats["total_posts"] if blog_stats else 0,
                "drafts": blog_stats["drafts"] if blog_stats else 0,
                "published": blog_stats["published"] if blog_stats else 0,
            },
        }

        return json.dumps({"success": True, **result}, default=str)
    except Exception:
        logger.exception("get_consumer_pipeline_status error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: list_complaint_content
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_complaint_content(
    content_type: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 20,
) -> str:
    """
    List generated complaint-based content (articles, forum posts, email copy).

    content_type: Filter by type (comparison_article, forum_post, email_copy,
                  review_summary)
    category: Filter by product category (partial match)
    status: Filter by status (draft, published)
    limit: Maximum results (default 20, cap 50)
    """
    limit = max(1, min(limit, 50))
    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        conditions = []
        params = []
        idx = 1

        if content_type:
            conditions.append(f"content_type = ${idx}")
            params.append(content_type)
            idx += 1

        if category:
            conditions.append(f"category ILIKE '%' || ${idx} || '%'")
            params.append(category)
            idx += 1

        if status:
            conditions.append(f"status = ${idx}")
            params.append(status)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = await pool.fetch(
            f"""
            SELECT id, content_type, category, target_asin, competitor_asin,
                   title, pain_point_summary, status, llm_model, created_at
            FROM complaint_content
            {where}
            ORDER BY created_at DESC
            LIMIT ${idx}
            """,
            *params,
        )

        content = [
            {
                "id": str(r["id"]),
                "content_type": r["content_type"],
                "category": r["category"],
                "target_asin": r["target_asin"],
                "competitor_asin": r["competitor_asin"],
                "title": r["title"],
                "pain_point_summary": r["pain_point_summary"],
                "status": r["status"],
                "llm_model": r["llm_model"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

        return json.dumps({"content": content, "count": len(content)}, default=str)
    except Exception:
        logger.exception("list_complaint_content error")
        return json.dumps({"error": "Internal error", "content": [], "count": 0})


# ---------------------------------------------------------------------------
# Brand history, change events, corrections
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_brand_history(
    brand: str,
    days: int = 90,
) -> str:
    """Get daily brand health snapshot time-series.

    Returns historical snapshots of a brand's health metrics including
    pain score, vulnerability score, repurchase rates, safety counts,
    and more. Use this to identify trends and turning points.

    Args:
        brand: Brand name to query
        days: Number of days of history (default 90, max 730)
    """
    try:
        from ..storage.database import get_db_pool
        pool = get_db_pool()
        days = max(7, min(days, 730))
        rows = await pool.fetch(
            """
            SELECT snapshot_date, total_reviews, avg_rating, avg_pain_score,
                   health_score, repurchase_yes, repurchase_no,
                   complaint_count, safety_count, top_complaint,
                   top_feature_request, competitive_flow_count,
                   trajectory_positive, trajectory_negative
            FROM brand_intelligence_snapshots
            WHERE brand = $1
              AND snapshot_date >= CURRENT_DATE - $2::int
            ORDER BY snapshot_date ASC
            """,
            brand, days,
        )
        snapshots = [
            {
                "date": str(r["snapshot_date"]),
                "total_reviews": r["total_reviews"],
                "avg_rating": float(r["avg_rating"]) if r["avg_rating"] else None,
                "avg_pain_score": float(r["avg_pain_score"]) if r["avg_pain_score"] else None,
                "health_score": float(r["health_score"]) if r["health_score"] else None,
                "repurchase_yes": r["repurchase_yes"],
                "repurchase_no": r["repurchase_no"],
                "complaint_count": r["complaint_count"],
                "safety_count": r["safety_count"],
                "top_complaint": r["top_complaint"],
            }
            for r in rows
        ]
        return json.dumps({
            "brand": brand, "days": days,
            "snapshots": snapshots, "total": len(snapshots),
        }, default=str)
    except Exception:
        logger.exception("get_brand_history error")
        return json.dumps({"error": "Internal error", "snapshots": []})


@mcp.tool()
async def list_product_change_events(
    brand: Optional[str] = None,
    event_type: Optional[str] = None,
    days: int = 30,
    limit: int = 50,
) -> str:
    """List consumer product change events (anomalies, spikes, emerging signals).

    Detected event types:
    - pain_score_spike: Pain score increased >= 1.5 points
    - vulnerability_spike: Vulnerability score rose >= 10 points
    - safety_flag_emergence: New safety-flagged reviews appeared (was 0)
    - repurchase_decline: Repurchase rate dropped >= 15 percentage points
    - rating_drop: Average rating dropped >= 0.5 stars

    Args:
        brand: Filter by brand name (optional)
        event_type: Filter by event type (optional)
        days: Lookback window in days (default 30)
        limit: Max results (default 50)
    """
    try:
        from ..storage.database import get_db_pool
        pool = get_db_pool()
        clauses = ["event_date >= CURRENT_DATE - $1::int"]
        params: list = [days]
        idx = 2
        if brand:
            clauses.append(f"brand = ${idx}")
            params.append(brand)
            idx += 1
        if event_type:
            clauses.append(f"event_type = ${idx}")
            params.append(event_type)
            idx += 1
        where = " AND ".join(clauses)

        rows = await pool.fetch(
            f"""
            SELECT id, brand, asin, event_date, event_type, description,
                   old_value, new_value, delta
            FROM product_change_events
            WHERE {where}
            ORDER BY event_date DESC
            LIMIT ${idx}
            """,
            *params, limit,
        )
        events = [
            {
                "id": str(r["id"]),
                "brand": r["brand"],
                "event_date": str(r["event_date"]),
                "event_type": r["event_type"],
                "description": r["description"],
                "old_value": float(r["old_value"]) if r["old_value"] else None,
                "new_value": float(r["new_value"]) if r["new_value"] else None,
                "delta": float(r["delta"]) if r["delta"] else None,
            }
            for r in rows
        ]
        return json.dumps({"events": events, "total": len(events)}, default=str)
    except Exception:
        logger.exception("list_product_change_events error")
        return json.dumps({"error": "Internal error", "events": []})


@mcp.tool()
async def create_consumer_correction(
    entity_type: str,
    entity_id: str,
    correction_type: str,
    reason: str,
    field_name: Optional[str] = None,
    old_value: Optional[str] = None,
    new_value: Optional[str] = None,
) -> str:
    """Create a data correction for a consumer entity.

    Supported entity types: product_review, product_pain_point, brand,
    market_report, complaint_content.

    Supported correction types: suppress, flag, override_field, reclassify,
    merge_brand.

    For merge_brand: old_value = source brand name, new_value = target brand
    name. Renames across brand_intelligence, brand_intelligence_snapshots,
    product_change_events, product_displacement_edges, product_metadata.
    Adds source as alias on target in consumer_brand_registry.

    Args:
        entity_type: Type of entity to correct
        entity_id: UUID of the entity
        correction_type: Type of correction to apply
        reason: Human-readable reason for the correction
        field_name: Required for override_field corrections
        old_value: Original value (for override_field) or source brand (for merge_brand)
        new_value: New value (for override_field) or target brand (for merge_brand)
    """
    valid_types = {"product_review", "product_pain_point", "brand", "market_report", "complaint_content"}
    valid_corrections = {"suppress", "flag", "override_field", "reclassify", "merge_brand"}

    if entity_type not in valid_types:
        return json.dumps({"error": f"Invalid entity_type. Must be one of: {sorted(valid_types)}"})
    if correction_type not in valid_corrections:
        return json.dumps({"error": f"Invalid correction_type. Must be one of: {sorted(valid_corrections)}"})
    if not _is_uuid(entity_id):
        return json.dumps({"error": "entity_id must be a valid UUID"})
    if correction_type == "override_field" and not field_name:
        return json.dumps({"error": "field_name required for override_field"})
    if correction_type == "merge_brand":
        if not old_value or not new_value:
            return json.dumps({
                "error": "merge_brand requires old_value (source brand) and new_value (target brand)",
            })

    try:
        from ..storage.database import get_db_pool
        pool = get_db_pool()
        row = await pool.fetchrow(
            """
            INSERT INTO data_corrections
                (entity_type, entity_id, correction_type, field_name,
                 old_value, new_value, reason, corrected_by)
            VALUES ($1, $2::uuid, $3, $4, $5, $6, $7, 'mcp')
            RETURNING id, status, created_at
            """,
            entity_type, entity_id, correction_type,
            field_name, old_value, new_value, reason,
        )

        # Execute brand merge if applicable
        merge_info = None
        if correction_type == "merge_brand":
            from ..services.brand_merge import execute_brand_merge
            merge_result = await execute_brand_merge(pool, old_value, new_value)
            correction_id = row["id"]
            await pool.execute(
                "UPDATE data_corrections SET affected_count = $1, metadata = $2::jsonb WHERE id = $3",
                merge_result["total_affected"], json.dumps(merge_result), correction_id,
            )
            merge_info = merge_result

        result = {
            "success": True,
            "id": str(row["id"]),
            "status": row["status"],
            "created_at": row["created_at"],
        }
        if merge_info:
            result["merge"] = merge_info
        return json.dumps(result, default=str)
    except Exception:
        logger.exception("create_consumer_correction error")
        return json.dumps({"error": "Failed to create correction"})


@mcp.tool()
async def list_consumer_corrections(
    entity_type: Optional[str] = None,
    correction_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
) -> str:
    """List data corrections for consumer entities.

    Args:
        entity_type: Filter by entity type (optional)
        correction_type: Filter by correction type (optional)
        status: Filter by status: applied, reverted, pending_review (optional)
        limit: Max results (default 50)
    """
    try:
        from ..storage.database import get_db_pool
        pool = get_db_pool()
        clauses = [
            "entity_type IN ('product_review','product_pain_point','brand','market_report','complaint_content')",
        ]
        params: list = []
        idx = 1
        if entity_type:
            clauses.append(f"entity_type = ${idx}")
            params.append(entity_type)
            idx += 1
        if correction_type:
            clauses.append(f"correction_type = ${idx}")
            params.append(correction_type)
            idx += 1
        if status:
            clauses.append(f"status = ${idx}")
            params.append(status)
            idx += 1
        where = " AND ".join(clauses)

        rows = await pool.fetch(
            f"""
            SELECT id, entity_type, entity_id, correction_type,
                   field_name, reason, corrected_by, status, created_at
            FROM data_corrections
            WHERE {where}
            ORDER BY created_at DESC
            LIMIT ${idx}
            """,
            *params, limit,
        )
        corrections = [
            {
                "id": str(r["id"]),
                "entity_type": r["entity_type"],
                "entity_id": str(r["entity_id"]),
                "correction_type": r["correction_type"],
                "field_name": r["field_name"],
                "reason": r["reason"],
                "status": r["status"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]
        return json.dumps({"corrections": corrections, "total": len(corrections)}, default=str)
    except Exception:
        logger.exception("list_consumer_corrections error")
        return json.dumps({"error": "Internal error", "corrections": []})


@mcp.tool()
async def revert_consumer_correction(
    correction_id: str,
) -> str:
    """Revert a previously applied consumer correction.

    Args:
        correction_id: UUID of the correction to revert
    """
    if not _is_uuid(correction_id):
        return json.dumps({"error": "correction_id must be a valid UUID"})
    try:
        from ..storage.database import get_db_pool
        pool = get_db_pool()
        row = await pool.fetchrow(
            "SELECT status, entity_type FROM data_corrections WHERE id = $1",
            _uuid.UUID(correction_id),
        )
        if not row:
            return json.dumps({"error": "Correction not found"})
        if row["status"] != "applied":
            return json.dumps({"error": f"Cannot revert correction with status '{row['status']}'"})

        await pool.execute(
            """
            UPDATE data_corrections
            SET status = 'reverted', reverted_at = NOW(), reverted_by = 'mcp'
            WHERE id = $1
            """,
            _uuid.UUID(correction_id),
        )
        return json.dumps({"success": True, "id": correction_id, "status": "reverted"})
    except Exception:
        logger.exception("revert_consumer_correction error")
        return json.dumps({"error": "Failed to revert correction"})


# ---------------------------------------------------------------------------
# Brand registry tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def fuzzy_brand_search(
    query: str,
    limit: int = 10,
    min_similarity: float = 0.3,
) -> str:
    """Search consumer brands by name using fuzzy matching (trigram similarity).

    Finds brands even with typos or partial names (e.g. "Cuisinrt" finds "Cuisinart").

    Args:
        query: Brand name to search for
        limit: Max results (default 10, max 100)
        min_similarity: Minimum similarity threshold 0.0-1.0 (default 0.3)
    """
    if not query or not query.strip():
        return json.dumps({"error": "query is required"})
    try:
        from ..services.brand_registry import fuzzy_search_brands
        results = await fuzzy_search_brands(
            query.strip(), limit=limit, min_similarity=min_similarity,
        )
        return json.dumps({"query": query.strip(), "results": results, "count": len(results)}, default=str)
    except Exception:
        logger.exception("fuzzy_brand_search error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def add_brand_to_registry(
    canonical_name: str,
    aliases: Optional[str] = None,
) -> str:
    """Add or update a consumer brand in the canonical brand registry.

    On conflict with an existing brand, merges aliases into the existing set.

    Args:
        canonical_name: The official brand name (e.g. "KitchenAid")
        aliases: Comma-separated lowercase aliases (e.g. "kitchen aid,kitchenaid")
    """
    if not canonical_name or not canonical_name.strip():
        return json.dumps({"success": False, "error": "canonical_name is required"})
    try:
        from ..services.brand_registry import add_brand
        alias_list = []
        if aliases:
            alias_list = [a.strip() for a in aliases.split(",") if a.strip()]
        row = await add_brand(canonical_name.strip(), alias_list)
        return json.dumps({
            "success": True,
            "brand": {
                "id": str(row["id"]),
                "canonical_name": row["canonical_name"],
                "aliases": list(row["aliases"]) if isinstance(row["aliases"], list) else [],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            },
        }, default=str)
    except Exception:
        logger.exception("add_brand_to_registry error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def add_brand_alias(
    canonical_name: str,
    alias: str,
) -> str:
    """Add an alias to an existing consumer brand in the registry.

    Args:
        canonical_name: Existing canonical brand name (e.g. "KitchenAid")
        alias: New alias to add (e.g. "kitchen-aid")
    """
    if not canonical_name or not canonical_name.strip():
        return json.dumps({"success": False, "error": "canonical_name is required"})
    if not alias or not alias.strip():
        return json.dumps({"success": False, "error": "alias is required"})
    try:
        from ..services.brand_registry import add_alias
        row = await add_alias(canonical_name.strip(), alias.strip())
        if row is None:
            return json.dumps({
                "success": False,
                "error": f"Brand '{canonical_name}' not found in registry",
            })
        return json.dumps({
            "success": True,
            "brand": {
                "id": str(row["id"]),
                "canonical_name": row["canonical_name"],
                "aliases": list(row["aliases"]) if isinstance(row["aliases"], list) else [],
                "updated_at": row["updated_at"],
            },
        }, default=str)
    except Exception:
        logger.exception("add_brand_alias error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def list_brand_registry(limit: int = 100) -> str:
    """List all canonical consumer brands in the brand registry with their aliases.

    Args:
        limit: Maximum results (default 100, cap 500)
    """
    limit = max(1, min(limit, 500))
    try:
        from ..services.brand_registry import list_brands
        brands = await list_brands()
        result = [
            {
                "id": str(b["id"]),
                "canonical_name": b["canonical_name"],
                "aliases": list(b["aliases"]) if isinstance(b["aliases"], list) else [],
                "created_at": b["created_at"],
                "updated_at": b["updated_at"],
            }
            for b in brands[:limit]
        ]
        return json.dumps({"brands": result, "count": len(result)}, default=str)
    except Exception:
        logger.exception("list_brand_registry error")
        return json.dumps({"error": "Internal error", "brands": [], "count": 0})


# ---------------------------------------------------------------------------
# Displacement edge tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_product_displacement_edges(
    from_brand: Optional[str] = None,
    to_brand: Optional[str] = None,
    direction: Optional[str] = None,
    min_confidence: Optional[float] = None,
    days: int = 90,
    limit: int = 50,
) -> str:
    """Query product displacement edges (brand-to-brand competitive flows).

    Shows which brands customers compare, switch from/to, consider, or avoid.
    Edges have confidence scores based on review evidence strength.

    Args:
        from_brand: Filter by source brand (case-insensitive partial match)
        to_brand: Filter by destination brand (case-insensitive partial match)
        direction: Filter by flow direction (switched_from, switched_to, compared, considered, avoided)
        min_confidence: Minimum confidence score (0.0-1.0)
        days: Lookback window in days (default 90)
        limit: Max results (default 50, max 200)
    """
    try:
        from ..storage.database import get_db_pool
        pool = get_db_pool()
        clauses = ["computed_date >= CURRENT_DATE - $1::int"]
        params: list = [days]
        idx = 2

        if from_brand:
            clauses.append(f"from_brand ILIKE '%%' || ${idx} || '%%'")
            params.append(from_brand)
            idx += 1
        if to_brand:
            clauses.append(f"to_brand ILIKE '%%' || ${idx} || '%%'")
            params.append(to_brand)
            idx += 1
        if direction:
            clauses.append(f"direction = ${idx}")
            params.append(direction)
            idx += 1
        if min_confidence is not None:
            clauses.append(f"confidence_score >= ${idx}")
            params.append(min_confidence)
            idx += 1

        where = " AND ".join(clauses)
        rows = await pool.fetch(
            f"""
            SELECT id, from_brand, to_brand, direction, mention_count,
                   signal_strength, avg_rating, confidence_score, computed_date
            FROM product_displacement_edges
            WHERE {where}
            ORDER BY mention_count DESC, confidence_score DESC
            LIMIT ${idx}
            """,
            *params, min(limit, 200),
        )
        edges = [
            {
                "id": str(r["id"]),
                "from_brand": r["from_brand"],
                "to_brand": r["to_brand"],
                "direction": r["direction"],
                "mention_count": r["mention_count"],
                "signal_strength": r["signal_strength"],
                "avg_rating": float(r["avg_rating"]) if r["avg_rating"] else None,
                "confidence_score": float(r["confidence_score"]) if r["confidence_score"] else None,
                "computed_date": str(r["computed_date"]),
            }
            for r in rows
        ]
        return json.dumps({"edges": edges, "count": len(edges)}, default=str)
    except Exception:
        logger.exception("list_product_displacement_edges error")
        return json.dumps({"error": "Internal error", "edges": []})


@mcp.tool()
async def get_product_displacement_history(
    from_brand: str,
    to_brand: str,
    direction: Optional[str] = None,
    days: int = 365,
) -> str:
    """Time-series of displacement edge strength for a specific brand pair.

    Args:
        from_brand: Source brand (case-insensitive)
        to_brand: Destination brand (case-insensitive)
        direction: Optional direction filter
        days: How far back to look (default 365)
    """
    try:
        from ..storage.database import get_db_pool
        pool = get_db_pool()
        clauses = [
            "LOWER(from_brand) = LOWER($1)",
            "LOWER(to_brand) = LOWER($2)",
            "computed_date >= CURRENT_DATE - $3::int",
        ]
        params: list = [from_brand, to_brand, days]
        idx = 4
        if direction:
            clauses.append(f"direction = ${idx}")
            params.append(direction)

        where = " AND ".join(clauses)
        rows = await pool.fetch(
            f"""
            SELECT computed_date, direction, mention_count, signal_strength,
                   confidence_score, avg_rating
            FROM product_displacement_edges
            WHERE {where}
            ORDER BY computed_date ASC
            """,
            *params,
        )
        history = [
            {
                "date": str(r["computed_date"]),
                "direction": r["direction"],
                "mention_count": r["mention_count"],
                "signal_strength": r["signal_strength"],
                "confidence_score": float(r["confidence_score"]) if r["confidence_score"] else None,
                "avg_rating": float(r["avg_rating"]) if r["avg_rating"] else None,
            }
            for r in rows
        ]
        return json.dumps({
            "from_brand": from_brand,
            "to_brand": to_brand,
            "days": days,
            "history": history,
            "total": len(history),
        }, default=str)
    except Exception:
        logger.exception("get_product_displacement_history error")
        return json.dumps({"error": "Internal error", "history": []})


# ---------------------------------------------------------------------------
# Delivery surfaces (PDF export, email digest)
# ---------------------------------------------------------------------------


@mcp.tool()
async def export_market_report_pdf(
    report_id: str,
) -> str:
    """Export a market intelligence report as PDF.

    Returns base64-encoded PDF bytes with filename.

    report_id: UUID of the market_intelligence_reports row
    """
    import base64

    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool:
            return json.dumps({"error": "Database not ready"})

        if not _is_uuid(report_id):
            return json.dumps({"error": "report_id must be a valid UUID"})

        import uuid as _uuid
        row = await pool.fetchrow(
            """
            SELECT id, report_date, report_type, executive_summary,
                   analysis_text, report_data
            FROM market_intelligence_reports
            WHERE id = $1
            """,
            _uuid.UUID(report_id),
        )
        if not row:
            return json.dumps({"error": "Report not found"})

        report = dict(row)
        rd = report.get("report_data")
        if isinstance(rd, str):
            try:
                report["report_data"] = json.loads(rd)
            except (json.JSONDecodeError, TypeError):
                report["report_data"] = {}

        from ..services.consumer_pdf_renderer import render_market_report_pdf

        pdf_bytes, filename = render_market_report_pdf(report)
        return json.dumps({
            "success": True,
            "filename": filename,
            "size_bytes": len(pdf_bytes),
            "pdf_base64": base64.b64encode(pdf_bytes).decode(),
        })
    except Exception:
        logger.exception("export_market_report_pdf error")
        return json.dumps({"error": "Failed to generate PDF"})


@mcp.tool()
async def export_brand_report_pdf(
    brand_name: str,
    days: int = 90,
) -> str:
    """Export a brand intelligence scorecard as PDF.

    Returns base64-encoded PDF bytes with filename.

    brand_name: Brand name (case-insensitive match)
    days: Days of snapshot history to include (default 90)
    """
    import base64

    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool:
            return json.dumps({"error": "Database not ready"})

        row = await pool.fetchrow(
            """
            SELECT brand, total_reviews, avg_rating, avg_pain_score,
                   repurchase_yes, repurchase_no, health_score, confidence_score,
                   top_complaints, top_feature_requests, competitive_flows,
                   sentiment_breakdown, buyer_profile, positive_aspects,
                   last_computed_at
            FROM brand_intelligence
            WHERE brand ILIKE $1
            LIMIT 1
            """,
            brand_name,
        )
        if not row:
            return json.dumps({"error": f"Brand '{brand_name}' not found"})

        brand_data = dict(row)

        snapshot_rows = await pool.fetch(
            """
            SELECT snapshot_date, total_reviews, avg_rating, avg_pain_score,
                   health_score, safety_count
            FROM brand_intelligence_snapshots
            WHERE brand = $1
              AND snapshot_date >= CURRENT_DATE - $2::int
            ORDER BY snapshot_date ASC
            """,
            row["brand"], days,
        )
        snapshots = [dict(s) for s in snapshot_rows]

        from ..services.consumer_pdf_renderer import render_brand_report_pdf

        pdf_bytes, filename = render_brand_report_pdf(brand_data, snapshots=snapshots)
        return json.dumps({
            "success": True,
            "filename": filename,
            "size_bytes": len(pdf_bytes),
            "pdf_base64": base64.b64encode(pdf_bytes).decode(),
        })
    except Exception:
        logger.exception("export_brand_report_pdf error")
        return json.dumps({"error": "Failed to generate PDF"})


@mcp.tool()
async def send_brand_health_digest(
    recipient_email: str,
    top_n: int = 10,
    days: int = 7,
) -> str:
    """Send a brand health digest email summarizing recent changes.

    Uses Resend to send an HTML email with top brands by health score,
    recent change events, and displacement highlights.

    recipient_email: Email address to send the digest to
    top_n: Number of top brands to include (default 10)
    days: Lookback period for change events (default 7)
    """
    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool:
            return json.dumps({"error": "Database not ready"})

        # Top brands by health score
        brands = await pool.fetch(
            """
            SELECT brand, health_score, avg_rating, avg_pain_score,
                   total_reviews, confidence_score
            FROM brand_intelligence
            ORDER BY health_score DESC NULLS LAST
            LIMIT $1
            """,
            top_n,
        )

        # Recent change events
        events = await pool.fetch(
            """
            SELECT brand, event_type, description, delta, event_date
            FROM product_change_events
            WHERE event_date >= CURRENT_DATE - $1::int
              AND brand != '__market__'
            ORDER BY event_date DESC
            LIMIT 20
            """,
            days,
        )

        # Build HTML
        html_parts = [
            "<h2>Atlas Consumer Intelligence -- Weekly Digest</h2>",
            f"<p>Top {len(brands)} brands by health score, {days}-day change events.</p>",
            "<h3>Brand Health Leaderboard</h3>",
            "<table border='1' cellpadding='4' cellspacing='0'>",
            "<tr><th>Brand</th><th>Health</th><th>Rating</th><th>Pain</th><th>Reviews</th></tr>",
        ]
        for b in brands:
            health = float(b["health_score"] or 0)
            color = "#27ae60" if health >= 70 else "#f39c12" if health >= 40 else "#e74c3c"
            html_parts.append(
                f"<tr><td>{b['brand']}</td>"
                f"<td style='color:{color};font-weight:bold'>{health:.0f}</td>"
                f"<td>{float(b['avg_rating'] or 0):.2f}</td>"
                f"<td>{float(b['avg_pain_score'] or 0):.1f}</td>"
                f"<td>{b['total_reviews']}</td></tr>"
            )
        html_parts.append("</table>")

        if events:
            html_parts.append(f"<h3>Change Events (last {days} days)</h3>")
            html_parts.append("<ul>")
            for e in events:
                html_parts.append(
                    f"<li><strong>{e['brand']}</strong> -- {e['event_type']}: "
                    f"{e['description']} (delta: {float(e['delta'] or 0):.2f})</li>"
                )
            html_parts.append("</ul>")

        html_parts.append("<p><em>Generated by Atlas Intelligence</em></p>")
        html_body = "\n".join(html_parts)

        # Send via Resend
        from ..config import settings as app_settings

        api_key = getattr(getattr(app_settings, "email", None), "api_key", None)
        if not api_key:
            return json.dumps({"error": "Email not configured (no Resend API key)"})

        import httpx
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": getattr(
                        getattr(app_settings, "email", None),
                        "default_from", "atlas@notifications.local",
                    ),
                    "to": [recipient_email],
                    "subject": f"Atlas Brand Health Digest -- {days}-Day Summary",
                    "html": html_body,
                },
            )

        if 200 <= resp.status_code < 300:
            return json.dumps({
                "success": True,
                "recipient": recipient_email,
                "brands_included": len(brands),
                "events_included": len(events),
            })
        return json.dumps({
            "success": False,
            "error": f"Resend API returned {resp.status_code}: {resp.text[:200]}",
        })

    except Exception:
        logger.exception("send_brand_health_digest error")
        return json.dumps({"error": "Failed to send digest"})


# ---------------------------------------------------------------------------
# Cross-brand correlation
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_concurrent_events(
    days: int = 30,
    event_type: Optional[str] = None,
    min_brands: int = 2,
    limit: int = 50,
) -> str:
    """
    Find dates where multiple brands had the same change event type.

    Surfaces cross-brand correlations like 'pain score spiked at 4 brands on
    the same day' -- may indicate market-level trends vs brand-specific issues.

    days: Lookback period (default 30)
    event_type: Optional filter (pain_score_spike, vulnerability_spike, safety_flag_emergence, etc.)
    min_brands: Minimum brand count to qualify as concurrent (default 2)
    limit: Max results (default 50)
    """
    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool:
            return json.dumps({"error": "Database not ready"})

        type_filter = ""
        params: list = [days, min_brands, limit]
        if event_type:
            type_filter = "AND event_type = $4"
            params.append(event_type)

        rows = await pool.fetch(
            f"""
            SELECT event_date, event_type,
                   COUNT(DISTINCT brand) AS brand_count,
                   ARRAY_AGG(DISTINCT brand ORDER BY brand) AS brands,
                   AVG(delta) AS avg_delta,
                   MIN(delta) AS min_delta,
                   MAX(delta) AS max_delta
            FROM product_change_events
            WHERE event_date >= CURRENT_DATE - $1::int
              AND brand != '__market__'
              {type_filter}
            GROUP BY event_date, event_type
            HAVING COUNT(DISTINCT brand) >= $2
            ORDER BY brand_count DESC, event_date DESC
            LIMIT $3
            """,
            *params,
        )

        results = [
            {
                "event_date": str(r["event_date"]),
                "event_type": r["event_type"],
                "brand_count": r["brand_count"],
                "brands": r["brands"],
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


@mcp.tool()
async def get_brand_correlation(
    brand_a: str,
    brand_b: str,
    days: int = 90,
    metric: str = "avg_pain_score",
) -> str:
    """
    Compare two brands' metric trends and compute Pearson correlation coefficient.

    Returns aligned time-series from daily snapshots and Pearson r. Negative
    correlation (r < -0.5) suggests one brand gains when the other loses --
    potential displacement.

    brand_a: First brand name (partial match)
    brand_b: Second brand name (partial match)
    days: Lookback period (default 90)
    metric: Metric to correlate (health_score, avg_pain_score, avg_rating,
            total_reviews, repurchase_yes, safety_count, complaint_count,
            competitive_flow_count)
    """
    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool:
            return json.dumps({"error": "Database not ready"})

        valid_metrics = {
            "health_score", "avg_pain_score", "avg_rating", "total_reviews",
            "repurchase_yes", "safety_count", "complaint_count",
            "competitive_flow_count",
        }
        if metric not in valid_metrics:
            return json.dumps({"error": f"metric must be one of: {sorted(valid_metrics)}"})

        rows = await pool.fetch(
            f"""
            SELECT a.snapshot_date,
                   a.{metric} AS value_a,
                   b.{metric} AS value_b
            FROM brand_intelligence_snapshots a
            JOIN brand_intelligence_snapshots b
              ON a.snapshot_date = b.snapshot_date
            WHERE a.brand ILIKE '%' || $1 || '%'
              AND b.brand ILIKE '%' || $2 || '%'
              AND a.snapshot_date >= CURRENT_DATE - $3::int
            ORDER BY a.snapshot_date ASC
            """,
            brand_a, brand_b, days,
        )

        if not rows:
            return json.dumps({"error": "No overlapping snapshots found for these brands"})

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
            SELECT from_brand, to_brand, mention_count, signal_strength,
                   category_distribution
            FROM product_displacement_edges
            WHERE (from_brand ILIKE '%' || $1 || '%' AND to_brand ILIKE '%' || $2 || '%')
               OR (from_brand ILIKE '%' || $2 || '%' AND to_brand ILIKE '%' || $1 || '%')
            ORDER BY computed_date DESC
            LIMIT 5
            """,
            brand_a, brand_b,
        )
        displacement = []
        for r in edge_rows:
            cat_dist = r["category_distribution"]
            if isinstance(cat_dist, str):
                try:
                    cat_dist = json.loads(cat_dist)
                except Exception:
                    cat_dist = {}
            displacement.append({
                "from_brand": r["from_brand"],
                "to_brand": r["to_brand"],
                "mention_count": r["mention_count"],
                "signal_strength": r["signal_strength"],
                "top_categories": cat_dist or {},
            })

        return json.dumps({
            "brand_a": brand_a,
            "brand_b": brand_b,
            "metric": metric,
            "data_points": len(series),
            "correlation": correlation,
            "series": series,
            "displacement_edges": displacement,
        }, default=str)
    except Exception:
        logger.exception("get_brand_correlation error")
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
        mcp.settings.port = settings.mcp.intelligence_port
        run_sse_with_auth(mcp, settings.mcp.host, settings.mcp.intelligence_port)
    else:
        mcp.run(transport="stdio")
