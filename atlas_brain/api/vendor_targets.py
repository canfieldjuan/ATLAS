"""
REST endpoints for Vendor/Challenger target management.

Vendor targets represent our actual customers -- the companies we sell
churn intelligence (vendor_retention) or intent leads (challenger_intel) to.
"""

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.vendor_targets")

router = APIRouter(prefix="/b2b/vendor-targets", tags=["b2b-vendor-targets"])


def _pool_or_503():
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")
    return pool


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class VendorTargetCreate(BaseModel):
    company_name: str = Field(..., max_length=200)
    target_mode: str = Field(..., max_length=30)
    contact_name: str | None = Field(None, max_length=200)
    contact_email: str | None = Field(None, max_length=254)
    contact_role: str | None = Field(None, max_length=100)
    products_tracked: list[str] | None = Field(None, max_length=50)
    competitors_tracked: list[str] | None = Field(None, max_length=50)
    tier: str = Field("report", max_length=30)
    status: str = Field("active", max_length=30)
    notes: str | None = Field(None, max_length=5000)


class VendorTargetUpdate(BaseModel):
    company_name: str | None = Field(None, max_length=200)
    target_mode: str | None = Field(None, max_length=30)
    contact_name: str | None = Field(None, max_length=200)
    contact_email: str | None = Field(None, max_length=254)
    contact_role: str | None = Field(None, max_length=100)
    products_tracked: list[str] | None = Field(None, max_length=50)
    competitors_tracked: list[str] | None = Field(None, max_length=50)
    tier: str | None = Field(None, max_length=30)
    status: str | None = Field(None, max_length=30)
    notes: str | None = Field(None, max_length=5000)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("")
async def list_vendor_targets(
    target_mode: Optional[str] = Query(None, description="Filter by target mode"),
    status: Optional[str] = Query(None, description="Filter by status"),
    search: Optional[str] = Query(None, description="Search company name"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List all vendor/challenger targets."""
    pool = _pool_or_503()

    conditions = []
    params: list = []
    idx = 1

    if target_mode:
        conditions.append(f"target_mode = ${idx}")
        params.append(target_mode)
        idx += 1

    if status:
        conditions.append(f"status = ${idx}")
        params.append(status)
        idx += 1

    if search:
        conditions.append(f"company_name ILIKE '%' || ${idx} || '%'")
        params.append(search)
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    rows = await pool.fetch(
        f"""
        SELECT id, company_name, target_mode, contact_name, contact_email,
               contact_role, products_tracked, competitors_tracked, tier,
               status, notes, created_at, updated_at
        FROM vendor_targets
        {where}
        ORDER BY created_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
        """,
        *params, limit, offset,
    )

    count = await pool.fetchval(
        f"SELECT COUNT(*) FROM vendor_targets {where}",
        *params,
    )

    return {
        "targets": [dict(r) for r in rows],
        "count": count,
    }


@router.post("", status_code=201)
async def create_vendor_target(body: VendorTargetCreate):
    """Add a new vendor or challenger target."""
    pool = _pool_or_503()

    if body.target_mode not in ("vendor_retention", "challenger_intel"):
        raise HTTPException(status_code=400, detail="target_mode must be 'vendor_retention' or 'challenger_intel'")

    row = await pool.fetchrow(
        """
        INSERT INTO vendor_targets (
            company_name, target_mode, contact_name, contact_email,
            contact_role, products_tracked, competitors_tracked,
            tier, status, notes
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        RETURNING id, company_name, target_mode, contact_name, contact_email,
                  contact_role, products_tracked, competitors_tracked, tier,
                  status, notes, created_at, updated_at
        """,
        body.company_name,
        body.target_mode,
        body.contact_name,
        body.contact_email,
        body.contact_role,
        body.products_tracked,
        body.competitors_tracked,
        body.tier,
        body.status,
        body.notes,
    )

    return dict(row)


@router.get("/{target_id}")
async def get_vendor_target(target_id: UUID):
    """Get a single vendor/challenger target with intelligence summary."""
    pool = _pool_or_503()

    row = await pool.fetchrow(
        """
        SELECT id, company_name, target_mode, contact_name, contact_email,
               contact_role, products_tracked, competitors_tracked, tier,
               status, notes, created_at, updated_at
        FROM vendor_targets
        WHERE id = $1
        """,
        target_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Target not found")

    target = dict(row)

    # Fetch campaign stats for this target
    campaign_stats = await pool.fetchrow(
        """
        SELECT
            COUNT(*) AS total_campaigns,
            COUNT(*) FILTER (WHERE status = 'draft') AS drafts,
            COUNT(*) FILTER (WHERE status = 'sent') AS sent,
            COUNT(*) FILTER (WHERE status = 'approved') AS approved,
            MAX(created_at) AS last_campaign_at
        FROM b2b_campaigns
        WHERE LOWER(company_name) = LOWER($1)
          AND target_mode = $2
        """,
        target["company_name"],
        target["target_mode"],
    )
    target["campaign_stats"] = dict(campaign_stats) if campaign_stats else {}

    # Fetch recent reports for this target
    report_type = "vendor_retention" if target["target_mode"] == "vendor_retention" else "challenger_intel"
    reports = await pool.fetch(
        """
        SELECT id, report_date, report_type, executive_summary, created_at
        FROM b2b_intelligence
        WHERE vendor_filter ILIKE '%' || $1 || '%'
          AND report_type = $2
        ORDER BY report_date DESC
        LIMIT 5
        """,
        target["company_name"],
        report_type,
    )
    target["recent_reports"] = [dict(r) for r in reports]

    return target


@router.put("/{target_id}")
async def update_vendor_target(target_id: UUID, body: VendorTargetUpdate):
    """Update a vendor/challenger target."""
    pool = _pool_or_503()

    # Build dynamic SET clause -- model_fields_set tracks which fields were
    # explicitly provided in the request body (including explicit nulls).
    updates = []
    params: list = []
    idx = 1

    for field in [
        "company_name", "target_mode", "contact_name", "contact_email",
        "contact_role", "products_tracked", "competitors_tracked",
        "tier", "status", "notes",
    ]:
        if field in body.model_fields_set:
            updates.append(f"{field} = ${idx}")
            params.append(getattr(body, field))
            idx += 1

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    if "target_mode" in body.model_fields_set:
        if body.target_mode is None:
            raise HTTPException(status_code=400, detail="target_mode cannot be null")
        if body.target_mode not in ("vendor_retention", "challenger_intel"):
            raise HTTPException(status_code=400, detail="target_mode must be 'vendor_retention' or 'challenger_intel'")

    updates.append(f"updated_at = NOW()")

    row = await pool.fetchrow(
        f"""
        UPDATE vendor_targets
        SET {', '.join(updates)}
        WHERE id = ${idx}
        RETURNING id, company_name, target_mode, contact_name, contact_email,
                  contact_role, products_tracked, competitors_tracked, tier,
                  status, notes, created_at, updated_at
        """,
        *params, target_id,
    )

    if not row:
        raise HTTPException(status_code=404, detail="Target not found")

    return dict(row)


@router.delete("/{target_id}")
async def delete_vendor_target(target_id: UUID):
    """Remove a vendor/challenger target."""
    pool = _pool_or_503()

    result = await pool.execute(
        "DELETE FROM vendor_targets WHERE id = $1",
        target_id,
    )

    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Target not found")

    return {"deleted": True}


@router.post("/{target_id}/generate-report")
async def generate_target_report(target_id: UUID):
    """Generate an intelligence report for this target (vendor or challenger)."""
    pool = _pool_or_503()

    row = await pool.fetchrow(
        "SELECT company_name, target_mode FROM vendor_targets WHERE id = $1",
        target_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Target not found")

    if row["target_mode"] == "vendor_retention":
        from ..autonomous.tasks.b2b_churn_intelligence import generate_vendor_report

        report = await generate_vendor_report(pool, row["company_name"])
        if not report:
            raise HTTPException(status_code=404, detail="No churn signals found for this vendor")
    elif row["target_mode"] == "challenger_intel":
        from ..autonomous.tasks.b2b_churn_intelligence import generate_challenger_report

        report = await generate_challenger_report(pool, row["company_name"])
        if not report:
            raise HTTPException(status_code=404, detail="No competitor mentions found for this challenger")
    else:
        raise HTTPException(status_code=400, detail=f"Unknown target_mode: {row['target_mode']}")

    return report
