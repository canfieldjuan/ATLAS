"""
REST endpoints for B2B ABM Campaign management.

Provides CRUD, manual campaign generation trigger, approve/reject workflow,
and KPI stats for the campaign engine.
"""

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


class SetRecipientBody(BaseModel):
    recipient_email: str


class ApproveQueueBody(BaseModel):
    recipient_email: str | None = None


class SuppressionCreate(BaseModel):
    email: str | None = None
    domain: str | None = None
    reason: str = "manual"
    notes: str | None = None


class BulkApproveBody(BaseModel):
    campaign_ids: list[str]
    action: str  # "approve", "queue-send", "reject"


class BulkRejectBody(BaseModel):
    campaign_ids: list[str]
    reason: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Analytics endpoints (query campaign_funnel_stats materialized view)
# ---------------------------------------------------------------------------


def _handle_analytics_error(exc: Exception) -> None:
    """Raise appropriate HTTP error for analytics query failures."""
    err = str(exc)
    if "campaign_funnel_stats" in err or "UndefinedTable" in type(exc).__name__:
        raise HTTPException(
            status_code=503,
            detail="Analytics view not ready. Run migration 069 and refresh.",
        )
    if "invalid input syntax" in err or "invalid input for query argument" in err:
        raise HTTPException(status_code=400, detail=f"Invalid query parameter: {err}")
    raise


@router.get("/analytics/funnel")
async def analytics_funnel(
    since: Optional[str] = Query(None, description="ISO date filter (e.g. 2026-01-01)"),
    vendor: Optional[str] = Query(None),
    company: Optional[str] = Query(None),
    partner_id: Optional[str] = Query(None),
):
    """Overall funnel: sent -> opened -> clicked -> replied -> bounced, with rates."""
    pool = _pool_or_503()

    conditions: list[str] = []
    params: list[Any] = []
    idx = 1

    if since:
        conditions.append(f"week >= ${idx}::timestamptz")
        params.append(since)
        idx += 1
    if vendor:
        conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
        params.append(vendor)
        idx += 1
    if company:
        conditions.append(f"company_name ILIKE '%' || ${idx} || '%'")
        params.append(company)
        idx += 1
    if partner_id:
        conditions.append(f"partner_id = ${idx}::uuid")
        params.append(partner_id)
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    try:
        row = await pool.fetchrow(
            f"""
            SELECT
                COALESCE(SUM(total), 0)::int          AS total,
                COALESCE(SUM(sent), 0)::int            AS sent,
                COALESCE(SUM(opened), 0)::int          AS opened,
                COALESCE(SUM(clicked), 0)::int         AS clicked,
                COALESCE(SUM(replied), 0)::int         AS replied,
                COALESCE(SUM(bounced), 0)::int         AS bounced,
                COALESCE(SUM(unsubscribed), 0)::int    AS unsubscribed,
                COALESCE(SUM(completed), 0)::int       AS completed,
                AVG(avg_hours_to_open)                 AS avg_hours_to_open,
                AVG(avg_hours_to_click)                AS avg_hours_to_click
            FROM campaign_funnel_stats
            {where}
            """,
            *params,
        )
    except Exception as exc:
        _handle_analytics_error(exc)

    r = dict(row)
    sent = r["sent"] or 0
    r["open_rate"] = round(r["opened"] / sent, 4) if sent else 0
    r["click_rate"] = round(r["clicked"] / sent, 4) if sent else 0
    r["reply_rate"] = round(r["replied"] / sent, 4) if sent else 0
    r["bounce_rate"] = round(r["bounced"] / sent, 4) if sent else 0
    r["avg_hours_to_open"] = round(r["avg_hours_to_open"], 2) if r["avg_hours_to_open"] else None
    r["avg_hours_to_click"] = round(r["avg_hours_to_click"], 2) if r["avg_hours_to_click"] else None
    return r


@router.get("/analytics/by-vendor")
async def analytics_by_vendor(
    since: Optional[str] = Query(None),
    limit: int = Query(20, le=100),
):
    """Per-vendor funnel breakdown, ordered by sent DESC."""
    pool = _pool_or_503()

    params: list[Any] = []
    idx = 1
    where = ""
    if since:
        where = f"WHERE week >= ${idx}::timestamptz"
        params.append(since)
        idx += 1

    params.append(limit)

    try:
        rows = await pool.fetch(
            f"""
            SELECT
                vendor_name,
                SUM(total)::int AS total, SUM(sent)::int AS sent,
                SUM(opened)::int AS opened, SUM(clicked)::int AS clicked,
                SUM(replied)::int AS replied, SUM(bounced)::int AS bounced,
                SUM(completed)::int AS completed
            FROM campaign_funnel_stats
            {where}
            GROUP BY vendor_name
            ORDER BY SUM(sent) DESC
            LIMIT ${idx}
            """,
            *params,
        )
    except Exception as exc:
        _handle_analytics_error(exc)

    results = []
    for r in rows:
        d = dict(r)
        sent = d["sent"] or 0
        d["open_rate"] = round(d["opened"] / sent, 4) if sent else 0
        d["reply_rate"] = round(d["replied"] / sent, 4) if sent else 0
        results.append(d)

    return {"vendors": results}


@router.get("/analytics/by-company")
async def analytics_by_company(
    since: Optional[str] = Query(None),
    limit: int = Query(20, le=100),
):
    """Per-company engagement summary (best for identifying warm leads)."""
    pool = _pool_or_503()

    params: list[Any] = []
    idx = 1
    where = ""
    if since:
        where = f"WHERE week >= ${idx}::timestamptz"
        params.append(since)
        idx += 1

    params.append(limit)

    try:
        rows = await pool.fetch(
            f"""
            SELECT
                company_name,
                SUM(total)::int AS total, SUM(sent)::int AS sent,
                SUM(opened)::int AS opened, SUM(clicked)::int AS clicked,
                SUM(replied)::int AS replied, SUM(bounced)::int AS bounced
            FROM campaign_funnel_stats
            {where}
            GROUP BY company_name
            ORDER BY SUM(opened) DESC, SUM(clicked) DESC
            LIMIT ${idx}
            """,
            *params,
        )
    except Exception as exc:
        _handle_analytics_error(exc)

    results = []
    for r in rows:
        d = dict(r)
        sent = d["sent"] or 0
        d["open_rate"] = round(d["opened"] / sent, 4) if sent else 0
        d["click_rate"] = round(d["clicked"] / sent, 4) if sent else 0
        results.append(d)

    return {"companies": results}


@router.get("/analytics/timeline")
async def analytics_timeline(
    since: Optional[str] = Query(None),
    vendor: Optional[str] = Query(None),
    company: Optional[str] = Query(None),
):
    """Weekly time-series of sent/opened/clicked/replied for trend visualization."""
    pool = _pool_or_503()

    conditions: list[str] = []
    params: list[Any] = []
    idx = 1

    if since:
        conditions.append(f"week >= ${idx}::timestamptz")
        params.append(since)
        idx += 1
    if vendor:
        conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
        params.append(vendor)
        idx += 1
    if company:
        conditions.append(f"company_name ILIKE '%' || ${idx} || '%'")
        params.append(company)
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    try:
        rows = await pool.fetch(
            f"""
            SELECT
                week,
                SUM(sent)::int AS sent,
                SUM(opened)::int AS opened,
                SUM(clicked)::int AS clicked,
                SUM(replied)::int AS replied,
                SUM(bounced)::int AS bounced
            FROM campaign_funnel_stats
            {where}
            GROUP BY week
            ORDER BY week ASC
            """,
            *params,
        )
    except Exception as exc:
        _handle_analytics_error(exc)

    return {
        "timeline": [
            {
                "week": r["week"].isoformat() if r["week"] else None,
                "sent": r["sent"],
                "opened": r["opened"],
                "clicked": r["clicked"],
                "replied": r["replied"],
                "bounced": r["bounced"],
            }
            for r in rows
        ]
    }


# ---------------------------------------------------------------------------
# Suppression endpoints (do-not-contact / blocklist)
# ---------------------------------------------------------------------------


@router.get("/suppressions")
async def list_suppressions(
    reason: Optional[str] = Query(None),
    email: Optional[str] = Query(None),
    domain: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
):
    """List suppression records with optional filters."""
    pool = _pool_or_503()

    conditions: list[str] = []
    params: list[Any] = []
    idx = 1

    if reason:
        conditions.append(f"reason = ${idx}")
        params.append(reason)
        idx += 1
    if email:
        conditions.append(f"LOWER(email) LIKE '%' || ${idx} || '%'")
        params.append(email.lower())
        idx += 1
    if domain:
        conditions.append(f"LOWER(domain) LIKE '%' || ${idx} || '%'")
        params.append(domain.lower())
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)

    rows = await pool.fetch(
        f"""
        SELECT id, email, domain, reason, source, campaign_id,
               notes, created_at, expires_at
        FROM campaign_suppressions
        {where}
        ORDER BY created_at DESC
        LIMIT ${idx}
        """,
        *params,
    )

    return {
        "count": len(rows),
        "suppressions": [_row_to_dict(r) for r in rows],
    }


@router.post("/suppressions")
async def create_suppression(body: SuppressionCreate):
    """Manually add an email or domain suppression."""
    if not body.email and not body.domain:
        raise HTTPException(status_code=400, detail="email or domain required")

    pool = _pool_or_503()

    from ..autonomous.tasks.campaign_suppression import add_suppression

    sup_id = await add_suppression(
        pool,
        email=body.email,
        domain=body.domain,
        reason=body.reason,
        source="api",
        notes=body.notes,
    )

    if not sup_id:
        raise HTTPException(status_code=409, detail="Suppression already exists")

    return {"id": str(sup_id), "status": "created"}


@router.delete("/suppressions/{suppression_id}")
async def delete_suppression(suppression_id: str):
    """Remove a suppression (e.g., after fixing a bounce)."""
    try:
        sid = _uuid.UUID(suppression_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid suppression_id")

    pool = _pool_or_503()

    result = await pool.execute(
        "DELETE FROM campaign_suppressions WHERE id = $1", sid
    )
    if _affected_rows(result) == 0:
        raise HTTPException(status_code=404, detail="Suppression not found")

    return {"status": "deleted"}


@router.get("/suppressions/check")
async def check_suppression(email: str = Query(...)):
    """Check if an email is suppressed."""
    pool = _pool_or_503()

    from ..autonomous.tasks.campaign_suppression import is_suppressed

    sup = await is_suppressed(pool, email=email)
    if sup:
        return {
            "suppressed": True,
            "reason": sup["reason"],
            "source": sup["source"],
            "created_at": sup["created_at"].isoformat() if sup.get("created_at") else None,
        }
    return {"suppressed": False}


# ---------------------------------------------------------------------------
# Review queue endpoints
# ---------------------------------------------------------------------------


@router.get("/review-queue")
async def review_queue(
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
):
    """Purpose-built review endpoint: drafts enriched with sequence, partner, suppression info."""
    pool = _pool_or_503()

    rows = await pool.fetch(
        """
        SELECT bc.id, bc.company_name, bc.vendor_name, bc.channel,
               bc.subject, bc.body, bc.cta, bc.status, bc.step_number,
               bc.recipient_email, bc.partner_id, bc.created_at,
               cs.recipient_email AS seq_recipient, cs.open_count, cs.click_count,
               cs.status AS seq_status, cs.current_step, cs.max_steps,
               ap.name, ap.product_name,
               (SELECT COUNT(*) FROM campaign_suppressions sup
                WHERE (sup.expires_at IS NULL OR sup.expires_at > NOW())
                AND (
                    LOWER(sup.email) = LOWER(COALESCE(bc.recipient_email, cs.recipient_email))
                    OR LOWER(sup.domain) = LOWER(SPLIT_PART(COALESCE(bc.recipient_email, cs.recipient_email), '@', 2))
                )) AS is_suppressed
        FROM b2b_campaigns bc
        LEFT JOIN campaign_sequences cs ON cs.id = bc.sequence_id
        LEFT JOIN affiliate_partners ap ON ap.id = bc.partner_id
        WHERE bc.status = 'draft'
        ORDER BY bc.created_at DESC
        LIMIT $1 OFFSET $2
        """,
        limit, offset,
    )

    return {
        "count": len(rows),
        "drafts": [_row_to_dict(r) for r in rows],
    }


@router.post("/bulk-approve")
async def bulk_approve(body: BulkApproveBody):
    """Approve, queue-send, or reject multiple campaigns at once."""
    if body.action not in ("approve", "queue-send", "reject"):
        raise HTTPException(status_code=400, detail="action must be approve, queue-send, or reject")

    pool = _pool_or_503()

    from ..autonomous.tasks.campaign_suppression import is_suppressed

    approved = 0
    failed: list[dict] = []

    for cid_str in body.campaign_ids:
        try:
            cid = _uuid.UUID(cid_str)
        except (ValueError, AttributeError):
            failed.append({"id": cid_str, "reason": "invalid UUID"})
            continue

        campaign = await pool.fetchrow(
            """
            SELECT bc.id, bc.status, bc.recipient_email, bc.sequence_id,
                   bc.step_number, bc.subject,
                   cs.recipient_email AS seq_recipient
            FROM b2b_campaigns bc
            LEFT JOIN campaign_sequences cs ON cs.id = bc.sequence_id
            WHERE bc.id = $1
            """,
            cid,
        )
        if not campaign:
            failed.append({"id": cid_str, "reason": "not found"})
            continue
        if campaign["status"] != "draft":
            failed.append({"id": cid_str, "reason": f"status is '{campaign['status']}', expected 'draft'"})
            continue

        now = datetime.now(timezone.utc)

        if body.action == "reject":
            await pool.execute(
                "UPDATE b2b_campaigns SET status = 'cancelled' WHERE id = $1",
                cid,
            )
            await log_campaign_event(
                pool, event_type="cancelled", source="api",
                campaign_id=cid, sequence_id=campaign.get("sequence_id"),
                step_number=campaign.get("step_number"),
            )
            approved += 1
            continue

        if body.action == "approve":
            await pool.execute(
                "UPDATE b2b_campaigns SET status = 'approved', approved_at = $1 WHERE id = $2",
                now, cid,
            )
            approved += 1
            continue

        # queue-send: needs recipient + suppression check
        recipient = campaign["recipient_email"] or campaign["seq_recipient"]
        if not recipient:
            failed.append({"id": cid_str, "reason": "no recipient_email set"})
            continue

        sup = await is_suppressed(pool, email=recipient)
        if sup:
            failed.append({"id": cid_str, "reason": f"recipient suppressed ({sup['reason']})"})
            continue

        await pool.execute(
            """
            UPDATE b2b_campaigns
            SET status = 'queued', approved_at = $1, recipient_email = $2
            WHERE id = $3
            """,
            now, recipient, cid,
        )
        await log_campaign_event(
            pool, event_type="queued", source="api",
            campaign_id=cid, sequence_id=campaign.get("sequence_id"),
            step_number=campaign.get("step_number"),
            recipient_email=recipient,
            subject=campaign.get("subject"),
        )
        approved += 1

    return {"approved": approved, "failed": failed}


@router.post("/bulk-reject")
async def bulk_reject(body: BulkRejectBody):
    """Reject/cancel multiple campaigns."""
    pool = _pool_or_503()

    rejected = 0
    failed: list[dict] = []

    for cid_str in body.campaign_ids:
        try:
            cid = _uuid.UUID(cid_str)
        except (ValueError, AttributeError):
            failed.append({"id": cid_str, "reason": "invalid UUID"})
            continue

        campaign = await pool.fetchrow(
            "SELECT id, status, sequence_id, step_number FROM b2b_campaigns WHERE id = $1",
            cid,
        )
        if not campaign:
            failed.append({"id": cid_str, "reason": "not found"})
            continue
        if campaign["status"] not in ("draft", "approved"):
            failed.append({"id": cid_str, "reason": f"cannot reject '{campaign['status']}'"})
            continue

        now = datetime.now(timezone.utc)
        await pool.execute(
            "UPDATE b2b_campaigns SET status = 'cancelled' WHERE id = $1",
            cid,
        )
        await log_campaign_event(
            pool, event_type="cancelled", source="api",
            campaign_id=cid, sequence_id=campaign.get("sequence_id"),
            step_number=campaign.get("step_number"),
            metadata={"reject_reason": body.reason} if body.reason else None,
        )
        rejected += 1

    return {"rejected": rejected, "failed": failed}


@router.get("/review-queue/summary")
async def review_queue_summary():
    """Dashboard counts for the review queue."""
    pool = _pool_or_503()

    row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE bc.status = 'draft') AS pending_review,
            COUNT(*) FILTER (
                WHERE bc.status = 'draft'
                AND COALESCE(bc.recipient_email, cs.recipient_email) IS NULL
            ) AS pending_recipient,
            COUNT(*) FILTER (
                WHERE bc.status = 'draft'
                AND COALESCE(bc.recipient_email, cs.recipient_email) IS NOT NULL
            ) AS ready_to_send,
            COUNT(*) FILTER (
                WHERE bc.status = 'draft'
                AND EXISTS (
                    SELECT 1 FROM campaign_suppressions sup
                    WHERE (sup.expires_at IS NULL OR sup.expires_at > NOW())
                    AND (
                        LOWER(sup.email) = LOWER(COALESCE(bc.recipient_email, cs.recipient_email))
                        OR LOWER(sup.domain) = LOWER(SPLIT_PART(COALESCE(bc.recipient_email, cs.recipient_email), '@', 2))
                    )
                )
            ) AS suppressed,
            EXTRACT(EPOCH FROM (NOW() - MIN(bc.created_at) FILTER (WHERE bc.status = 'draft'))) / 3600
                AS oldest_draft_age_hours
        FROM b2b_campaigns bc
        LEFT JOIN campaign_sequences cs ON cs.id = bc.sequence_id
        """
    )

    by_partner = await pool.fetch(
        """
        SELECT ap.name, COUNT(*) AS count
        FROM b2b_campaigns bc
        LEFT JOIN affiliate_partners ap ON ap.id = bc.partner_id
        WHERE bc.status = 'draft' AND ap.name IS NOT NULL
        GROUP BY ap.name
        ORDER BY count DESC
        """
    )

    r = dict(row)
    return {
        "pending_review": r["pending_review"],
        "pending_recipient": r["pending_recipient"],
        "ready_to_send": r["ready_to_send"],
        "suppressed": r["suppressed"],
        "oldest_draft_age_hours": round(r["oldest_draft_age_hours"], 1) if r["oldest_draft_age_hours"] is not None else None,
        "by_partner": [{"partner_name": p["name"], "count": p["count"]} for p in by_partner],
    }


# ---------------------------------------------------------------------------
# Sequence endpoints (stateful B2B campaign email sequences)
# MUST be defined BEFORE /{campaign_id} catch-all to avoid route shadowing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Approve / Generate / Notify (static paths before /{campaign_id} catch-all)
# ---------------------------------------------------------------------------


@router.post("/generate")
async def generate_campaigns_endpoint(body: GenerateRequest):
    """Manual trigger: generate campaign content for top opportunities."""
    pool = _pool_or_503()

    from ..autonomous.tasks.b2b_campaign_generation import generate_campaigns as _generate

    result = await _generate(
        pool=pool,
        min_score=body.min_score,
        limit=body.limit,
        vendor_filter=body.vendor_name,
        company_filter=body.company_name,
    )

    generated = result.get("generated", 0)
    if generated > 0 and result.get("batch_id"):
        await _send_campaign_notification(
            generated, result.get("companies", 0), result["batch_id"],
        )

    return result


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
# Campaign CRUD (/{campaign_id} catch-all routes — MUST be last)
# ---------------------------------------------------------------------------


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
    r["id"] = str(r["id"])
    if r.get("partner_id"):
        r["partner_id"] = str(r["partner_id"])
    for ts_field in ("created_at", "approved_at", "sent_at", "opened_at", "clicked_at"):
        if r.get(ts_field):
            r[ts_field] = r[ts_field].isoformat()
    if r.get("source_review_ids"):
        r["source_review_ids"] = [str(u) for u in r["source_review_ids"]]
    return r


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
            recipient_email = $2
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
        "UPDATE b2b_campaigns SET status = 'cancelled' WHERE id = $1",
        cid,
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
