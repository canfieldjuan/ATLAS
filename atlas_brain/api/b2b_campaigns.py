"""
REST endpoints for B2B ABM Campaign management.

Provides CRUD, manual campaign generation trigger, approve/reject workflow,
and KPI stats for the campaign engine.
"""

import logging
import uuid as _uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.b2b_campaigns")

router = APIRouter(prefix="/b2b/campaigns", tags=["b2b-campaigns"])


def _pool_or_503():
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")
    return pool


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class CampaignUpdate(BaseModel):
    subject: str | None = None
    body: str | None = None
    cta: str | None = None
    status: str | None = None


class GenerateRequest(BaseModel):
    vendor_name: str | None = None
    company_name: str | None = None
    min_score: int = 70
    limit: int = 20


# ---------------------------------------------------------------------------
# LIST campaigns
# ---------------------------------------------------------------------------


@router.get("")
async def list_campaigns(
    status: Optional[str] = Query(None),
    company: Optional[str] = Query(None),
    vendor: Optional[str] = Query(None),
    channel: Optional[str] = Query(None),
    batch_id: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
):
    pool = _pool_or_503()

    conditions = []
    params: list[Any] = []
    idx = 1

    if status:
        conditions.append(f"status = ${idx}")
        params.append(status)
        idx += 1
    if company:
        conditions.append(f"company_name ILIKE '%' || ${idx} || '%'")
        params.append(company)
        idx += 1
    if vendor:
        conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
        params.append(vendor)
        idx += 1
    if channel:
        conditions.append(f"channel = ${idx}")
        params.append(channel)
        idx += 1
    if batch_id:
        conditions.append(f"batch_id = ${idx}")
        params.append(batch_id)
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(min(limit, 200))

    rows = await pool.fetch(
        f"""
        SELECT id, company_name, vendor_name, product_category,
               opportunity_score, urgency_score, channel, subject,
               body, cta,
               status, batch_id, llm_model, created_at, approved_at, sent_at
        FROM b2b_campaigns
        {where}
        ORDER BY created_at DESC
        LIMIT ${idx}
        """,
        *params,
    )

    campaigns = [
        {
            "id": str(r["id"]),
            "company_name": r["company_name"],
            "vendor_name": r["vendor_name"],
            "product_category": r["product_category"],
            "opportunity_score": r["opportunity_score"],
            "urgency_score": float(r["urgency_score"]) if r["urgency_score"] else None,
            "channel": r["channel"],
            "subject": r["subject"],
            "body": r["body"],
            "cta": r["cta"],
            "status": r["status"],
            "batch_id": r["batch_id"],
            "llm_model": r["llm_model"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "approved_at": r["approved_at"].isoformat() if r["approved_at"] else None,
            "sent_at": r["sent_at"].isoformat() if r["sent_at"] else None,
        }
        for r in rows
    ]

    return {"campaigns": campaigns, "count": len(campaigns)}


# ---------------------------------------------------------------------------
# GET single campaign
# ---------------------------------------------------------------------------


@router.get("/stats")
async def campaign_stats():
    """KPI summary: counts by status, top vendors, top channels."""
    pool = _pool_or_503()

    status_rows = await pool.fetch(
        "SELECT status, COUNT(*) AS cnt FROM b2b_campaigns GROUP BY status"
    )
    channel_rows = await pool.fetch(
        "SELECT channel, COUNT(*) AS cnt FROM b2b_campaigns GROUP BY channel ORDER BY cnt DESC"
    )
    vendor_rows = await pool.fetch(
        """
        SELECT vendor_name, COUNT(*) AS cnt
        FROM b2b_campaigns
        GROUP BY vendor_name
        ORDER BY cnt DESC
        LIMIT 10
        """
    )

    return {
        "by_status": {r["status"]: r["cnt"] for r in status_rows},
        "by_channel": {r["channel"]: r["cnt"] for r in channel_rows},
        "top_vendors": [
            {"vendor_name": r["vendor_name"], "count": r["cnt"]}
            for r in vendor_rows
        ],
        "total": sum(r["cnt"] for r in status_rows),
    }


@router.get("/{campaign_id}")
async def get_campaign(campaign_id: str):
    try:
        cid = _uuid.UUID(campaign_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid campaign_id")

    pool = _pool_or_503()
    row = await pool.fetchrow(
        "SELECT * FROM b2b_campaigns WHERE id = $1", cid
    )
    if not row:
        raise HTTPException(status_code=404, detail="Campaign not found")

    r = dict(row)
    # Serialize UUID and datetime fields
    r["id"] = str(r["id"])
    for ts_field in ("created_at", "approved_at", "sent_at", "opened_at", "clicked_at"):
        if r.get(ts_field):
            r[ts_field] = r[ts_field].isoformat()
    if r.get("source_review_ids"):
        r["source_review_ids"] = [str(u) for u in r["source_review_ids"]]
    # JSONB fields come back as dicts/lists already from asyncpg
    return r


# ---------------------------------------------------------------------------
# PATCH campaign (edit content or status)
# ---------------------------------------------------------------------------


@router.patch("/{campaign_id}")
async def update_campaign(campaign_id: str, body: CampaignUpdate):
    try:
        cid = _uuid.UUID(campaign_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid campaign_id")

    pool = _pool_or_503()

    updates: list[str] = []
    params: list[Any] = []
    idx = 1

    for field in ("subject", "body", "cta", "status"):
        val = getattr(body, field)
        if val is not None:
            updates.append(f"{field} = ${idx}")
            params.append(val)
            idx += 1

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    params.append(cid)
    await pool.execute(
        f"UPDATE b2b_campaigns SET {', '.join(updates)} WHERE id = ${idx}",
        *params,
    )
    return {"ok": True}


# ---------------------------------------------------------------------------
# Approve / Generate
# ---------------------------------------------------------------------------


@router.post("/{campaign_id}/approve")
async def approve_campaign(campaign_id: str):
    try:
        cid = _uuid.UUID(campaign_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid campaign_id")

    pool = _pool_or_503()

    row = await pool.fetchrow(
        "SELECT status FROM b2b_campaigns WHERE id = $1", cid
    )
    if not row:
        raise HTTPException(status_code=404, detail="Campaign not found")
    if row["status"] != "draft":
        raise HTTPException(
            status_code=409,
            detail=f"Campaign is '{row['status']}', can only approve drafts",
        )

    now = datetime.now(timezone.utc)
    await pool.execute(
        "UPDATE b2b_campaigns SET status = 'approved', approved_at = $1 WHERE id = $2",
        now, cid,
    )
    return {"ok": True, "approved_at": now.isoformat()}


@router.post("/generate")
async def generate_campaigns_endpoint(body: GenerateRequest):
    """Manual trigger: generate campaign content for top opportunities."""
    pool = _pool_or_503()

    # Import the generation function from the autonomous task
    from ..autonomous.tasks.b2b_campaign_generation import generate_campaigns as _generate

    result = await _generate(
        pool=pool,
        min_score=body.min_score,
        limit=body.limit,
        vendor_filter=body.vendor_name,
        company_filter=body.company_name,
    )

    # Send ntfy notification for manual triggers
    generated = result.get("generated", 0)
    if generated > 0 and result.get("batch_id"):
        await _send_campaign_notification(
            generated, result.get("companies", 0), result["batch_id"],
        )

    return result


# ---------------------------------------------------------------------------
# Notify helper
# ---------------------------------------------------------------------------


async def _send_campaign_notification(
    count: int,
    companies: int,
    batch_id: str,
) -> None:
    """Send ntfy notification for newly generated campaigns."""
    from ..config import settings

    if not settings.alerts.ntfy_enabled:
        return

    import httpx

    ntfy_url = f"{settings.alerts.ntfy_url.rstrip('/')}/{settings.alerts.ntfy_topic}"

    message = (
        f"Generated {count} campaign(s) for {companies} company/companies.\n"
        f"Batch: {batch_id}\n\n"
        "Review and approve in the Leads dashboard."
    )

    headers = {
        "Title": f"ABM Campaigns: {count} drafts ready",
        "Priority": "default",
        "Tags": "briefcase,campaign",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(ntfy_url, content=message, headers=headers)
            resp.raise_for_status()
        logger.info("Campaign notification sent (batch=%s)", batch_id)
    except Exception as e:
        logger.warning("Failed to send campaign notification: %s", e)
