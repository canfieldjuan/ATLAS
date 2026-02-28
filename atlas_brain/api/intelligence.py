"""Intelligence report API endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.intelligence")

router = APIRouter(prefix="/intelligence", tags=["intelligence"])


class GenerateReportRequest(BaseModel):
    entity_name: str = Field(..., min_length=1, max_length=200)
    entity_type: str = Field(default="company", max_length=50)
    time_window_days: int = Field(default=7, ge=1, le=90)
    report_type: str = Field(default="full", pattern="^(full|executive)$")
    audience: str = Field(default="executive", max_length=50)


class RunInterventionRequest(BaseModel):
    entity_name: str = Field(..., min_length=1, max_length=200)
    entity_type: str = Field(default="company", max_length=50)
    time_window_days: int = Field(default=7, ge=1, le=90)
    objectives: list[str] = Field(default_factory=lambda: ["de-escalate", "stabilize"])
    constraints: list[str] = Field(default_factory=list)
    audience: str = Field(default="executive", max_length=50)
    risk_tolerance: str = Field(default="moderate", pattern="^(low|moderate|high)$")
    simulation_horizon: str = Field(default="7 days", max_length=50)
    hours_before_event: int = Field(default=48, ge=1, le=720)
    channels: list[str] = Field(default_factory=lambda: ["internal comms"])
    allow_narrative_architect: bool = Field(default=False)


@router.post("/report")
async def generate_report(req: GenerateReportRequest):
    """Generate an intelligence report for an entity on demand."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not available")

    from ..services.intelligence_report import generate_report as _generate

    result = await _generate(
        entity_name=req.entity_name,
        entity_type=req.entity_type,
        time_window_days=req.time_window_days,
        report_type=req.report_type,
        audience=req.audience,
        requested_by="api",
    )

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return result


@router.post("/intervention")
async def run_intervention(req: RunInterventionRequest):
    """Run the three-stage intervention pipeline for an entity."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not available")

    from ..services.intervention_pipeline import run_intervention_pipeline

    result = await run_intervention_pipeline(
        entity_name=req.entity_name,
        entity_type=req.entity_type,
        time_window_days=req.time_window_days,
        objectives=req.objectives,
        constraints=req.constraints,
        audience=req.audience,
        risk_tolerance=req.risk_tolerance,
        simulation_horizon=req.simulation_horizon,
        hours_before_event=req.hours_before_event,
        channels=req.channels,
        allow_narrative_architect=req.allow_narrative_architect,
        requested_by="api",
    )

    if "error" in result and "stages" not in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return result


@router.get("/reports")
async def list_reports(
    entity_name: Optional[str] = Query(default=None),
    limit: int = Query(default=20, le=100),
):
    """List recent intelligence reports."""
    from ..services.intelligence_report import list_reports as _list

    reports = await _list(entity_name=entity_name, limit=limit)
    return {
        "reports": [
            {
                **{k: (str(v) if k == "id" else v) for k, v in r.items()},
                "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
            }
            for r in reports
        ],
        "count": len(reports),
    }


@router.get("/reports/{report_id}")
async def get_report(report_id: str):
    """Get a full intelligence report by ID."""
    from ..services.intelligence_report import get_report as _get

    report = await _get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    result = dict(report)
    result["id"] = str(result["id"])
    if result.get("created_at"):
        result["created_at"] = result["created_at"].isoformat()
    return result


@router.get("/pressure")
async def list_pressure_baselines(
    limit: int = Query(default=20, le=100),
    entity_type: Optional[str] = Query(default=None),
):
    """List entity pressure baselines (highest pressure first)."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not available")

    conditions = []
    params = []
    idx = 1

    if entity_type:
        conditions.append(f"entity_type = ${idx}")
        params.append(entity_type)
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)

    rows = await pool.fetch(
        f"""
        SELECT entity_name, entity_type, pressure_score, sentiment_drift,
               narrative_frequency, soram_breakdown, linguistic_signals,
               last_computed_at
        FROM entity_pressure_baselines
        {where}
        ORDER BY pressure_score DESC
        LIMIT ${idx}
        """,
        *params,
    )

    return {
        "baselines": [
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
        ],
        "count": len(rows),
    }


# ---------------------------------------------------------------------------
# Approval workflow endpoints
# ---------------------------------------------------------------------------


class ApprovalReviewRequest(BaseModel):
    reviewed_by: str = Field(..., min_length=1, max_length=100)
    notes: str = Field(default="", max_length=1000)


@router.get("/approvals")
async def list_pending_approvals(
    entity_name: Optional[str] = Query(default=None),
    limit: int = Query(default=20, le=100),
):
    """List pending intervention approval requests."""
    from ..services.safety_gate import get_safety_gate

    gate = get_safety_gate()
    approvals = await gate.list_pending(entity_name=entity_name, limit=limit)
    return {"approvals": approvals, "count": len(approvals)}


@router.get("/approvals/{approval_id}")
async def get_approval(approval_id: str):
    """Check the status of an approval request."""
    from ..services.safety_gate import get_safety_gate

    gate = get_safety_gate()
    result = await gate.check_approval(approval_id)
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Approval not found")
    return result


@router.post("/approvals/{approval_id}/approve")
async def approve_intervention(approval_id: str, req: ApprovalReviewRequest):
    """Approve a pending intervention request."""
    from ..services.safety_gate import get_safety_gate

    gate = get_safety_gate()
    success = await gate.approve(approval_id, req.reviewed_by, req.notes)
    if not success:
        raise HTTPException(status_code=400, detail="Approval not found or already reviewed")
    return {"status": "approved", "approval_id": approval_id}


@router.post("/approvals/{approval_id}/reject")
async def reject_intervention(approval_id: str, req: ApprovalReviewRequest):
    """Reject a pending intervention request."""
    from ..services.safety_gate import get_safety_gate

    gate = get_safety_gate()
    success = await gate.reject(approval_id, req.reviewed_by, req.notes)
    if not success:
        raise HTTPException(status_code=400, detail="Approval not found or already reviewed")
    return {"status": "rejected", "approval_id": approval_id}
