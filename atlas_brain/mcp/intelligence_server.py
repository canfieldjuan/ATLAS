"""
Atlas Intelligence MCP Server.

Exposes intelligence report generation and pressure baseline queries
to any MCP-compatible client (Claude Desktop, Cursor, custom agents).

Tools:
    generate_intelligence_report  -- generate a full or executive report for an entity
    list_intelligence_reports     -- list recent reports with optional entity filter
    get_intelligence_report       -- fetch a stored report by ID
    list_pressure_baselines       -- list entity pressure baselines (highest first)
    analyze_risk_sensors          -- run behavioral risk sensors on text

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
        "Intelligence report server for Atlas. "
        "Generate behavioral intelligence reports from pressure baselines, "
        "SORAM-enriched news articles, and knowledge graph data. "
        "Use generate_intelligence_report for on-demand analysis of entities. "
        "Use list_pressure_baselines to see which entities have elevated pressure."
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
        return json.dumps({"success": False, "error": str(exc)})


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
        return json.dumps({"error": str(exc), "reports": []})


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
        return json.dumps({"error": str(exc), "found": False})


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
        return json.dumps({"error": str(exc), "baselines": []})


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
        return json.dumps({"success": False, "error": str(exc)})


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
        return json.dumps({"success": False, "error": str(exc)})


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
        return json.dumps({"error": str(exc), "approvals": []})


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
        return json.dumps({"success": False, "error": str(exc)})


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
