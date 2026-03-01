"""
REST endpoints for B2B ABM Campaign management.

Provides CRUD, manual campaign generation trigger, approve/reject workflow,
and KPI stats for the campaign engine.
"""

import json
import logging
import uuid as _uuid
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..config import settings
from ..storage.database import get_db_pool
from ..autonomous.tasks.campaign_audit import log_campaign_event

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
               status, batch_id, llm_model, created_at, approved_at, sent_at,
               partner_id, industry
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
            "partner_id": str(r["partner_id"]) if r["partner_id"] else None,
            "industry": r["industry"],
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
    if r.get("partner_id"):
        r["partner_id"] = str(r["partner_id"])
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


# ---------------------------------------------------------------------------
# Sequence endpoints (stateful B2B campaign email sequences)
# ---------------------------------------------------------------------------


class SetRecipientBody(BaseModel):
    recipient_email: str


class ApproveQueueBody(BaseModel):
    recipient_email: str | None = None


def _row_to_dict(row) -> dict:
    d = {}
    for key in row.keys():
        val = row[key]
        if isinstance(val, UUID):
            d[key] = str(val)
        elif isinstance(val, datetime):
            d[key] = val.isoformat()
        else:
            d[key] = val
    return d


def _affected_rows(result: str) -> int:
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError, AttributeError):
        return -1


@router.get("/sequences")
async def list_sequences(
    status: str = Query(default="all", description="Filter: active, paused, completed, replied, bounced, unsubscribed, all"),
    company: str = Query(default="", description="Filter by company name (case-insensitive substring)"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List campaign sequences with optional filters."""
    pool = _pool_or_503()

    conditions = []
    params: list = []
    param_idx = 1

    if status != "all":
        conditions.append(f"cs.status = ${param_idx}")
        params.append(status)
        param_idx += 1

    if company:
        conditions.append(f"LOWER(cs.company_name) LIKE ${param_idx}")
        params.append(f"%{company.lower()}%")
        param_idx += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    params.extend([limit, offset])
    rows = await pool.fetch(
        f"""
        SELECT cs.*,
               (SELECT COUNT(*) FROM b2b_campaigns bc WHERE bc.sequence_id = cs.id) AS campaign_count
        FROM campaign_sequences cs
        {where}
        ORDER BY cs.created_at DESC
        LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """,
        *params,
    )

    return {
        "count": len(rows),
        "sequences": [_row_to_dict(r) for r in rows],
    }


@router.get("/sequences/{sequence_id}")
async def get_sequence(sequence_id: UUID):
    """Get a single sequence with full campaign history."""
    pool = _pool_or_503()

    seq = await pool.fetchrow(
        "SELECT * FROM campaign_sequences WHERE id = $1", sequence_id
    )
    if not seq:
        raise HTTPException(status_code=404, detail="Sequence not found")

    campaigns = await pool.fetch(
        """
        SELECT * FROM b2b_campaigns
        WHERE sequence_id = $1
        ORDER BY step_number ASC
        """,
        sequence_id,
    )

    recent_audit = await pool.fetch(
        """
        SELECT * FROM campaign_audit_log
        WHERE sequence_id = $1
        ORDER BY created_at DESC LIMIT 50
        """,
        sequence_id,
    )

    return {
        "sequence": _row_to_dict(seq),
        "campaigns": [_row_to_dict(c) for c in campaigns],
        "audit_log": [_row_to_dict(a) for a in recent_audit],
    }


@router.post("/sequences/{sequence_id}/set-recipient")
async def set_recipient(sequence_id: UUID, body: SetRecipientBody):
    """Set the recipient email for a sequence."""
    pool = _pool_or_503()

    result = await pool.execute(
        """
        UPDATE campaign_sequences
        SET recipient_email = $1, updated_at = NOW()
        WHERE id = $2
        """,
        body.recipient_email, sequence_id,
    )

    if _affected_rows(result) == 0:
        raise HTTPException(status_code=404, detail="Sequence not found")

    await pool.execute(
        """
        UPDATE b2b_campaigns
        SET recipient_email = $1
        WHERE sequence_id = $2 AND recipient_email IS NULL
        """,
        body.recipient_email, sequence_id,
    )

    return {"status": "ok", "recipient_email": body.recipient_email}


@router.post("/sequences/{sequence_id}/pause")
async def pause_sequence(sequence_id: UUID):
    """Pause an active sequence."""
    pool = _pool_or_503()

    result = await pool.execute(
        """
        UPDATE campaign_sequences
        SET status = 'paused', updated_at = NOW()
        WHERE id = $1 AND status = 'active'
        """,
        sequence_id,
    )

    if _affected_rows(result) == 0:
        raise HTTPException(status_code=404, detail="Sequence not found or not active")

    await log_campaign_event(
        pool, event_type="paused", source="api", sequence_id=sequence_id
    )
    return {"status": "paused"}


@router.post("/sequences/{sequence_id}/resume")
async def resume_sequence(sequence_id: UUID):
    """Resume a paused sequence."""
    pool = _pool_or_503()

    result = await pool.execute(
        """
        UPDATE campaign_sequences
        SET status = 'active', updated_at = NOW()
        WHERE id = $1 AND status = 'paused'
        """,
        sequence_id,
    )

    if _affected_rows(result) == 0:
        raise HTTPException(status_code=404, detail="Sequence not found or not paused")

    await log_campaign_event(
        pool, event_type="resumed", source="api", sequence_id=sequence_id
    )
    return {"status": "active"}


@router.post("/{campaign_id}/queue-send")
async def queue_campaign_for_send(campaign_id: str, body: ApproveQueueBody | None = None):
    """Queue a campaign for auto-send with a cancel window.

    Used for sequence campaigns. Sets status to 'queued' with approved_at
    for the cancel window. The campaign_send task picks it up after the
    window expires.
    """
    try:
        cid = _uuid.UUID(campaign_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid campaign_id")

    pool = _pool_or_503()

    campaign = await pool.fetchrow(
        "SELECT * FROM b2b_campaigns WHERE id = $1", cid
    )
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    if campaign["status"] not in ("draft",):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot queue campaign with status '{campaign['status']}'"
        )

    now = datetime.now(timezone.utc)
    sequence_id = campaign.get("sequence_id")

    recipient = campaign.get("recipient_email")
    if body and body.recipient_email:
        recipient = body.recipient_email
    elif not recipient and sequence_id:
        recipient = await pool.fetchval(
            "SELECT recipient_email FROM campaign_sequences WHERE id = $1",
            sequence_id,
        )

    if not recipient:
        raise HTTPException(
            status_code=400,
            detail="recipient_email is required (set on sequence or provide in body)"
        )

    await pool.execute(
        """
        UPDATE b2b_campaigns
        SET status = 'queued',
            approved_at = $1,
            recipient_email = $2,
            updated_at = $1
        WHERE id = $3
        """,
        now, recipient, cid,
    )

    await log_campaign_event(
        pool, event_type="queued", source="api",
        campaign_id=cid, sequence_id=sequence_id,
        step_number=campaign.get("step_number"),
        recipient_email=recipient,
        subject=campaign.get("subject"),
    )

    # ntfy notification with cancel button
    if settings.alerts.ntfy_enabled:
        import httpx

        api_url = settings.email_draft.atlas_api_url.rstrip("/")
        ntfy_url = f"{settings.alerts.ntfy_url.rstrip('/')}/{settings.alerts.ntfy_topic}"
        delay_min = settings.campaign_sequence.auto_send_delay_seconds // 60

        cancel_url = f"{api_url}/api/v1/b2b/campaigns/{campaign_id}/cancel"
        message = (
            f"To: {recipient}\n"
            f"Subject: {campaign['subject']}\n"
            f"Auto-sending in {delay_min} min."
        )
        headers = {
            "Title": f"Campaign Queued: {campaign['company_name']}",
            "Priority": "default",
            "Tags": "outbox,campaign",
            "Actions": f"http, Cancel Auto-Send, {cancel_url}, method=POST, clear=true",
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(ntfy_url, content=message, headers=headers)
        except Exception as exc:
            logger.warning("ntfy notification failed: %s", exc)

    return {
        "status": "queued",
        "campaign_id": str(cid),
        "recipient_email": recipient,
        "auto_send_in_seconds": settings.campaign_sequence.auto_send_delay_seconds,
    }


@router.post("/{campaign_id}/cancel")
async def cancel_campaign(campaign_id: str):
    """Cancel a queued campaign before it sends."""
    try:
        cid = _uuid.UUID(campaign_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid campaign_id")

    pool = _pool_or_503()

    campaign = await pool.fetchrow(
        "SELECT id, sequence_id, step_number, status FROM b2b_campaigns WHERE id = $1",
        cid,
    )
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    if campaign["status"] != "queued":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel campaign with status '{campaign['status']}'"
        )

    now = datetime.now(timezone.utc)
    await pool.execute(
        "UPDATE b2b_campaigns SET status = 'cancelled', updated_at = $1 WHERE id = $2",
        now, cid,
    )

    await log_campaign_event(
        pool, event_type="cancelled", source="api",
        campaign_id=cid,
        sequence_id=campaign.get("sequence_id"),
        step_number=campaign.get("step_number"),
    )

    return {"status": "cancelled", "campaign_id": str(cid)}


@router.get("/{campaign_id}/audit-log")
async def campaign_audit_log(
    campaign_id: str,
    limit: int = Query(default=100, ge=1, le=500),
):
    """Get the full audit trail for a campaign."""
    try:
        cid = _uuid.UUID(campaign_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid campaign_id")

    pool = _pool_or_503()

    rows = await pool.fetch(
        """
        SELECT * FROM campaign_audit_log
        WHERE campaign_id = $1
        ORDER BY created_at DESC
        LIMIT $2
        """,
        cid, limit,
    )

    return {"count": len(rows), "audit_log": [_row_to_dict(r) for r in rows]}


@router.get("/sequences/{sequence_id}/audit-log")
async def sequence_audit_log(
    sequence_id: UUID,
    limit: int = Query(default=100, ge=1, le=500),
):
    """Get the full audit trail for a sequence."""
    pool = _pool_or_503()

    rows = await pool.fetch(
        """
        SELECT * FROM campaign_audit_log
        WHERE sequence_id = $1
        ORDER BY created_at DESC
        LIMIT $2
        """,
        sequence_id, limit,
    )

    return {"count": len(rows), "audit_log": [_row_to_dict(r) for r in rows]}
