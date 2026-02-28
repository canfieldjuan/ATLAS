"""
REST endpoints for B2B Affiliate Opportunities.

Joins enriched review competitor mentions against registered affiliate partners
to surface monetization opportunities ranked by purchase-intent signals.
"""

import logging
import uuid as _uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.b2b_affiliates")

router = APIRouter(prefix="/b2b/dashboard/affiliates", tags=["b2b-affiliates"])


def _safe_float(val, default=None):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _pool_or_503():
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")
    return pool


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_ROLE_SCORES = {
    "decision_maker": 20,
    "economic_buyer": 15,
    "champion": 15,
    "evaluator": 10,
}

_STAGE_SCORES = {
    "active_purchase": 25,
    "evaluation": 20,
    "renewal_decision": 15,
    "post_purchase": 5,
}

_CONTEXT_SCORES = {
    "considering": 10,
    "switched_to": 8,
    "compared": 6,
    "switched_from": 2,
}


def _compute_score(row: dict) -> int:
    """Compute opportunity score (0-100) from enrichment signals."""
    score = 0.0

    # Urgency (max 30): (urgency - 5) * 6, clamped 0-30
    urgency = _safe_float(row.get("urgency"), 0)
    score += max(0, min(30, (urgency - 5) * 6))

    # Decision maker (max 20)
    is_dm = row.get("is_dm")
    role_type = row.get("role_type") or ""
    if is_dm:
        score += 20
    elif role_type in _ROLE_SCORES:
        score += _ROLE_SCORES[role_type]

    # Buying stage (max 25)
    buying_stage = row.get("buying_stage") or ""
    score += _STAGE_SCORES.get(buying_stage, 0)

    # Seat count (max 15)
    seat_count = row.get("seat_count")
    if seat_count is not None:
        if seat_count >= 500:
            score += 15
        elif seat_count >= 100:
            score += 10
        elif seat_count >= 20:
            score += 5

    # Mention context (max 10)
    mention_context = (row.get("mention_context") or "").lower()
    for keyword, pts in _CONTEXT_SCORES.items():
        if keyword in mention_context:
            score += pts
            break

    return int(min(100, max(0, score)))


# ---------------------------------------------------------------------------
# GET /opportunities
# ---------------------------------------------------------------------------


@router.get("/opportunities")
async def list_opportunities(
    min_urgency: float = Query(5),
    min_score: int = Query(0),
    window_days: int = Query(90),
    limit: int = Query(50, le=200),
    vendor_name: Optional[str] = Query(None),
    dm_only: bool = Query(False),
):
    pool = _pool_or_503()

    extra_conditions = ""
    params: list = [window_days, min_urgency, min(limit, 200)]
    idx = 4

    if vendor_name:
        extra_conditions += f" AND r.vendor_name ILIKE '%' || ${idx} || '%'"
        params.append(vendor_name)
        idx += 1

    if dm_only:
        extra_conditions += " AND (r.enrichment->'reviewer_context'->>'decision_maker')::boolean = true"

    rows = await pool.fetch(
        f"""
        WITH review_competitors AS (
            SELECT r.id AS review_id, r.vendor_name, r.reviewer_company, r.product_category,
                   (r.enrichment->>'urgency_score')::numeric AS urgency,
                   (r.enrichment->'reviewer_context'->>'decision_maker')::boolean AS is_dm,
                   r.enrichment->'buyer_authority'->>'role_type' AS role_type,
                   r.enrichment->'buyer_authority'->>'buying_stage' AS buying_stage,
                   CASE WHEN r.enrichment->'budget_signals'->>'seat_count' ~ '^\\d+$'
                        THEN (r.enrichment->'budget_signals'->>'seat_count')::int END AS seat_count,
                   r.enrichment->'timeline'->>'contract_end' AS contract_end,
                   r.enrichment->'timeline'->>'decision_timeline' AS decision_timeline,
                   comp.value->>'name' AS competitor_name,
                   comp.value->>'context' AS mention_context,
                   comp.value->>'reason' AS mention_reason
            FROM b2b_reviews r
            CROSS JOIN LATERAL jsonb_array_elements(
                CASE WHEN jsonb_typeof(r.enrichment->'competitors_mentioned') = 'array'
                     THEN r.enrichment->'competitors_mentioned'
                     ELSE '[]'::jsonb END
            ) AS comp(value)
            WHERE r.enrichment_status = 'enriched'
              AND r.enriched_at > NOW() - make_interval(days => $1)
              AND (r.enrichment->>'urgency_score')::numeric >= $2
              {extra_conditions}
        )
        SELECT rc.*, ap.id AS partner_id, ap.name AS partner_name,
               ap.affiliate_url, ap.commission_type, ap.commission_value,
               ap.category AS partner_category
        FROM review_competitors rc
        JOIN affiliate_partners ap ON ap.enabled = true
            AND (LOWER(rc.competitor_name) = LOWER(ap.product_name)
                 OR LOWER(rc.competitor_name) = ANY(SELECT LOWER(unnest(ap.product_aliases))))
        ORDER BY rc.urgency DESC, rc.is_dm DESC NULLS LAST
        LIMIT $3
        """,
        *params,
    )

    opportunities = []
    for r in rows:
        row_dict = dict(r)
        opp_score = _compute_score(row_dict)
        if opp_score < min_score:
            continue
        opportunities.append({
            "review_id": str(r["review_id"]),
            "vendor_name": r["vendor_name"],
            "reviewer_company": r["reviewer_company"],
            "product_category": r["product_category"],
            "urgency": _safe_float(r["urgency"], 0),
            "is_dm": r["is_dm"],
            "role_type": r["role_type"],
            "buying_stage": r["buying_stage"],
            "seat_count": r["seat_count"],
            "contract_end": r["contract_end"],
            "decision_timeline": r["decision_timeline"],
            "competitor_name": r["competitor_name"],
            "mention_context": r["mention_context"],
            "mention_reason": r["mention_reason"],
            "partner_id": str(r["partner_id"]),
            "partner_name": r["partner_name"],
            "affiliate_url": r["affiliate_url"],
            "commission_type": r["commission_type"],
            "commission_value": r["commission_value"],
            "partner_category": r["partner_category"],
            "opportunity_score": opp_score,
        })

    # Re-sort by score descending after filtering
    opportunities.sort(key=lambda o: o["opportunity_score"], reverse=True)

    return {"opportunities": opportunities, "count": len(opportunities)}


# ---------------------------------------------------------------------------
# Partner CRUD
# ---------------------------------------------------------------------------


class PartnerCreate(BaseModel):
    name: str
    product_name: str
    product_aliases: list[str] = []
    category: str | None = None
    affiliate_url: str
    commission_type: str = "unknown"
    commission_value: str | None = None
    notes: str | None = None
    enabled: bool = True


class PartnerUpdate(BaseModel):
    name: str | None = None
    product_name: str | None = None
    product_aliases: list[str] | None = None
    category: str | None = None
    affiliate_url: str | None = None
    commission_type: str | None = None
    commission_value: str | None = None
    notes: str | None = None
    enabled: bool | None = None


@router.get("/partners")
async def list_partners():
    pool = _pool_or_503()
    rows = await pool.fetch(
        """
        SELECT id, name, product_name, product_aliases, category,
               affiliate_url, commission_type, commission_value, notes,
               enabled, created_at, updated_at
        FROM affiliate_partners
        ORDER BY name
        """
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
            "notes": r["notes"],
            "enabled": r["enabled"],
            "created_at": str(r["created_at"]) if r["created_at"] else None,
            "updated_at": str(r["updated_at"]) if r["updated_at"] else None,
        }
        for r in rows
    ]
    return {"partners": partners, "count": len(partners)}


@router.post("/partners", status_code=201)
async def create_partner(body: PartnerCreate):
    pool = _pool_or_503()
    try:
        row = await pool.fetchrow(
            """
            INSERT INTO affiliate_partners
                (name, product_name, product_aliases, category, affiliate_url,
                 commission_type, commission_value, notes, enabled)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id, created_at
            """,
            body.name,
            body.product_name,
            body.product_aliases,
            body.category,
            body.affiliate_url,
            body.commission_type,
            body.commission_value,
            body.notes,
            body.enabled,
        )
    except Exception as e:
        if "unique" in str(e).lower() or "duplicate" in str(e).lower():
            raise HTTPException(
                status_code=409,
                detail=f"Partner with product_name '{body.product_name}' already exists",
            )
        raise
    return {"id": str(row["id"]), "created_at": str(row["created_at"])}


@router.patch("/partners/{partner_id}")
async def update_partner(partner_id: str, body: PartnerUpdate):
    try:
        pid = _uuid.UUID(partner_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid partner_id")

    pool = _pool_or_503()

    updates: list[str] = []
    params: list = []
    idx = 1

    for field in (
        "name", "product_name", "product_aliases", "category",
        "affiliate_url", "commission_type", "commission_value",
        "notes", "enabled",
    ):
        val = getattr(body, field)
        if val is not None:
            updates.append(f"{field} = ${idx}")
            params.append(val)
            idx += 1

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    updates.append(f"updated_at = NOW()")
    params.append(pid)

    try:
        await pool.execute(
            f"""
            UPDATE affiliate_partners
            SET {', '.join(updates)}
            WHERE id = ${idx}
            """,
            *params,
        )
    except Exception as e:
        if "unique" in str(e).lower() or "duplicate" in str(e).lower():
            raise HTTPException(
                status_code=409,
                detail="Another partner already uses that product_name",
            )
        raise
    return {"ok": True}


@router.delete("/partners/{partner_id}")
async def delete_partner(partner_id: str):
    try:
        pid = _uuid.UUID(partner_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid partner_id")

    pool = _pool_or_503()
    result = await pool.execute("DELETE FROM affiliate_partners WHERE id = $1", pid)
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Partner not found")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Clicks
# ---------------------------------------------------------------------------


class ClickRecord(BaseModel):
    partner_id: str
    review_id: str | None = None
    referrer: str | None = "dashboard"


@router.post("/clicks", status_code=201)
async def record_click(body: ClickRecord):
    try:
        pid = _uuid.UUID(body.partner_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid partner_id")

    rid = None
    if body.review_id:
        try:
            rid = _uuid.UUID(body.review_id)
        except (ValueError, AttributeError):
            pass

    pool = _pool_or_503()
    await pool.execute(
        """
        INSERT INTO affiliate_clicks (partner_id, review_id, referrer)
        VALUES ($1, $2, $3)
        """,
        pid,
        rid,
        body.referrer,
    )
    return {"ok": True}


@router.get("/clicks/summary")
async def click_summary():
    pool = _pool_or_503()
    rows = await pool.fetch(
        """
        SELECT ap.id, ap.name, ap.product_name,
               COUNT(ac.id) AS click_count
        FROM affiliate_partners ap
        LEFT JOIN affiliate_clicks ac
            ON ac.partner_id = ap.id
            AND ac.clicked_at > NOW() - INTERVAL '30 days'
        GROUP BY ap.id, ap.name, ap.product_name
        ORDER BY click_count DESC
        """
    )
    return {
        "clicks": [
            {
                "id": str(r["id"]),
                "name": r["name"],
                "product_name": r["product_name"],
                "click_count": r["click_count"],
            }
            for r in rows
        ]
    }
