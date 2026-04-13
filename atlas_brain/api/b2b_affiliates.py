"""
REST endpoints for B2B Affiliate Opportunities.

Joins enriched review competitor mentions against registered affiliate partners
to identify monetization opportunities ranked by purchase-intent signals.
"""

import logging
import uuid as _uuid
from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.params import Param
from pydantic import BaseModel, Field

from ..auth.dependencies import AuthUser, require_auth
from ..config import settings
from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.b2b_affiliates")
_REVIEW_BASIS_CANONICAL = "canonical_reviews"

tenant_router = APIRouter(
    prefix="/b2b/tenant/affiliates",
    tags=["b2b-affiliates"],
    dependencies=[Depends(require_auth)],
)

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


def _as_iso_text(value):
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)

def _unwrap_param_default(value):
    if isinstance(value, Param):
        return value.default
    return value


def _clean_required_text(value, field_name: str) -> str:
    value = _unwrap_param_default(value)
    text = str(value or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail=f"{field_name} is required")
    return text


def _clean_optional_text(value):
    value = _unwrap_param_default(value)
    text = str(value or "").strip()
    return text or None


def _clean_int_query(value, *, default: int) -> int:
    value = _unwrap_param_default(value)
    if value is None:
        return default
    return int(value)


def _clean_float_query(value, *, default: float) -> float:
    value = _unwrap_param_default(value)
    if value is None:
        return default
    return float(value)


def _clean_bool_query(value, *, default: bool) -> bool:
    value = _unwrap_param_default(value)
    if value is None:
        return default
    return bool(value)


def _clean_optional_text_list(values):
    if values is None:
        return None
    cleaned: list[str] = []
    for value in values:
        text = _clean_optional_text(value)
        if text:
            cleaned.append(text)
    return cleaned



def _canonical_review_predicate(alias: str = "") -> str:
    prefix = f"{alias}." if alias else ""
    return f"{prefix}duplicate_of_review_id IS NULL"


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


@tenant_router.get("/opportunities")
async def list_opportunities(
    min_urgency: float = Query(5),
    min_score: int = Query(0),
    window_days: int = Query(90),
    limit: int = Query(50, ge=1, le=200),
    vendor_name: Optional[str] = Query(None),
    dm_only: bool = Query(False),
    user: AuthUser = Depends(require_auth),
):
    min_urgency = _clean_float_query(min_urgency, default=5.0)
    min_score = _clean_int_query(min_score, default=0)
    window_days = _clean_int_query(window_days, default=90)
    limit = min(_clean_int_query(limit, default=50), 200)
    clean_vendor_name = _clean_optional_text(vendor_name)
    dm_only = _clean_bool_query(dm_only, default=False)
    pool = _pool_or_503()

    extra_conditions = ""
    params: list = [window_days, min_urgency, limit]
    idx = 4

    if clean_vendor_name:
        extra_conditions += f" AND vm.vendor_name ILIKE '%' || ${idx} || '%'"
        params.append(clean_vendor_name)
        idx += 1

    # APPROVED-ENRICHMENT-READ: reviewer_context.decision_maker
    # (CTE with competitor expansion + affiliate partner JOIN, structurally coupled)
    if dm_only:
        extra_conditions += " AND (r.enrichment->'reviewer_context'->>'decision_maker')::boolean = true"

    account_id = getattr(user, "account_id", None)
    if settings.saas_auth.enabled and account_id:
        extra_conditions += (
            f" AND vm.vendor_name IN ("
            f"SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid"
            f")"
        )
        params.append(account_id)
        idx += 1

    # APPROVED-ENRICHMENT-READ: urgency_score, reviewer_context.decision_maker, buyer_authority.role_type, buyer_authority.buying_stage, budget_signals.seat_count, timeline.contract_end, timeline.decision_timeline, competitors_mentioned
    # (CTE with competitor expansion + affiliate partner JOIN, structurally coupled)
    rows = await pool.fetch(
        f"""
        WITH review_competitors AS (
            SELECT r.id AS review_id,
                   vm.vendor_name AS vendor_name,
                   COALESCE(
                       NULLIF(BTRIM(r.reviewer_company_norm), ''),
                       NULLIF(BTRIM(r.reviewer_company), '')
                   ) AS reviewer_company,
                   NULLIF(BTRIM(r.reviewer_name), '') AS reviewer_name,
                   r.product_category,
                   r.source,
                   r.reviewed_at,
                   (r.enrichment->>'urgency_score')::numeric AS urgency,
                   (r.enrichment->'reviewer_context'->>'decision_maker')::boolean AS is_dm,
                   r.enrichment->'buyer_authority'->>'role_type' AS role_type,
                   r.enrichment->'buyer_authority'->>'buying_stage' AS buying_stage,
                   CASE WHEN r.enrichment->'budget_signals'->>'seat_count' ~ '^\\d+$'
                        THEN (r.enrichment->'budget_signals'->>'seat_count')::int END AS seat_count,
                   r.enrichment->'timeline'->>'contract_end' AS contract_end,
                   r.enrichment->'timeline'->>'decision_timeline' AS decision_timeline,
                   NULLIF(BTRIM(comp.value->>'name'), '') AS competitor_name,
                   comp.value->>'context' AS mention_context,
                   comp.value->>'reason' AS mention_reason
            FROM b2b_reviews r
            JOIN b2b_review_vendor_mentions vm
              ON vm.review_id = r.id
            CROSS JOIN LATERAL jsonb_array_elements(
                CASE WHEN jsonb_typeof(r.enrichment->'competitors_mentioned') = 'array'
                     THEN r.enrichment->'competitors_mentioned'
                     ELSE '[]'::jsonb END
            ) AS comp(value)
            WHERE r.enrichment_status = 'enriched'
              AND {_canonical_review_predicate('r')}
              AND COALESCE(r.reviewed_at, r.imported_at, r.enriched_at) > NOW() - make_interval(days => $1)
              AND (r.enrichment->>'urgency_score')::numeric >= $2
              AND NULLIF(BTRIM(comp.value->>'name'), '') IS NOT NULL
              {extra_conditions}
        )
        SELECT rc.*, ap.id AS partner_id, ap.name AS partner_name,
               ap.affiliate_url, ap.commission_type, ap.commission_value,
               ap.category AS partner_category
        FROM review_competitors rc
        JOIN affiliate_partners ap ON ap.enabled = true
            AND (LOWER(rc.competitor_name) = LOWER(ap.product_name)
                 OR LOWER(rc.competitor_name) = ANY(SELECT LOWER(unnest(ap.product_aliases))))
            AND rc.reviewer_company IS NOT NULL
            AND LOWER(rc.competitor_name) <> LOWER(rc.vendor_name)
            AND LOWER(rc.reviewer_company) <> LOWER(rc.vendor_name)
            AND LOWER(rc.reviewer_company) <> LOWER(rc.competitor_name)
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
        reviewer_company = r["reviewer_company"]
        if not reviewer_company:
            # Defensive guard: opportunities must map to a real company.
            continue
        reviewer_company_display = reviewer_company
        opportunities.append({
            "review_id": str(r["review_id"]),
            "vendor_name": r["vendor_name"],
            "reviewer_company": reviewer_company,
            "reviewer_company_display": reviewer_company_display,
            "reviewer_company_inferred": False,
            "product_category": r["product_category"],
            "urgency": _safe_float(r["urgency"], 0),
            "is_dm": r["is_dm"],
            "role_type": r["role_type"],
            "buying_stage": r["buying_stage"],
            "seat_count": r["seat_count"],
            "contract_end": r["contract_end"],
            "decision_timeline": r["decision_timeline"],
            "source": r["source"],
            "reviewed_at": _as_iso_text(r["reviewed_at"]),
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

    return {
        "basis": _REVIEW_BASIS_CANONICAL,
        "opportunities": opportunities,
        "count": len(opportunities),
    }


# ---------------------------------------------------------------------------
# Partner CRUD
# ---------------------------------------------------------------------------


class PartnerCreate(BaseModel):
    name: str = Field(..., max_length=200)
    product_name: str = Field(..., max_length=200)
    product_aliases: list[str] = Field(default=[], max_length=50)
    category: str | None = Field(None, max_length=100)
    affiliate_url: str = Field(..., max_length=2000)
    commission_type: str = Field("unknown", max_length=50)
    commission_value: str | None = Field(None, max_length=50)
    notes: str | None = Field(None, max_length=2000)
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


@tenant_router.get("/partners")
async def list_partners():
    pool = _pool_or_503()
    rows = await pool.fetch(
        """
        SELECT id, name, product_name, product_aliases, category,
               affiliate_url, commission_type, commission_value, notes,
               enabled, created_at, updated_at
        FROM affiliate_partners
        ORDER BY name
        LIMIT 500
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


@tenant_router.post("/partners", status_code=201)
async def create_partner(body: PartnerCreate):
    name = _clean_required_text(body.name, "name")
    product_name = _clean_required_text(body.product_name, "product_name")
    product_aliases = _clean_optional_text_list(body.product_aliases) or []
    category = _clean_optional_text(body.category)
    affiliate_url = _clean_required_text(body.affiliate_url, "affiliate_url")
    commission_type = _clean_required_text(body.commission_type, "commission_type")
    commission_value = _clean_optional_text(body.commission_value)
    notes = _clean_optional_text(body.notes)

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
            name,
            product_name,
            product_aliases,
            category,
            affiliate_url,
            commission_type,
            commission_value,
            notes,
            body.enabled,
        )
    except Exception as e:
        if "unique" in str(e).lower() or "duplicate" in str(e).lower():
            raise HTTPException(
                status_code=409,
                detail=f"Partner with product_name '{product_name}' already exists",
            )
        raise
    return {"id": str(row["id"]), "created_at": str(row["created_at"])}


@tenant_router.patch("/partners/{partner_id}")
async def update_partner(partner_id: str, body: PartnerUpdate):
    try:
        pid = _uuid.UUID(partner_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid partner_id")

    normalized: dict[str, object] = {}
    if body.name is not None:
        normalized["name"] = _clean_required_text(body.name, "name")
    if body.product_name is not None:
        normalized["product_name"] = _clean_required_text(body.product_name, "product_name")
    if body.product_aliases is not None:
        normalized["product_aliases"] = _clean_optional_text_list(body.product_aliases) or []
    if body.category is not None:
        normalized["category"] = _clean_optional_text(body.category)
    if body.affiliate_url is not None:
        normalized["affiliate_url"] = _clean_required_text(body.affiliate_url, "affiliate_url")
    if body.commission_type is not None:
        normalized["commission_type"] = _clean_required_text(body.commission_type, "commission_type")
    if body.commission_value is not None:
        normalized["commission_value"] = _clean_optional_text(body.commission_value)
    if body.notes is not None:
        normalized["notes"] = _clean_optional_text(body.notes)
    if body.enabled is not None:
        normalized["enabled"] = body.enabled

    if not normalized:
        raise HTTPException(status_code=400, detail="No fields to update")

    pool = _pool_or_503()

    updates: list[str] = []
    params: list = []
    idx = 1

    for field in (
        "name", "product_name", "product_aliases", "category",
        "affiliate_url", "commission_type", "commission_value",
        "notes", "enabled",
    ):
        if field in normalized:
            updates.append(f"{field} = ${idx}")
            params.append(normalized[field])
            idx += 1

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


@tenant_router.delete("/partners/{partner_id}")
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


@tenant_router.post("/clicks", status_code=201)
async def record_click(body: ClickRecord):
    try:
        pid = _uuid.UUID(body.partner_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid partner_id")

    review_id = _clean_optional_text(body.review_id)
    rid = None
    if review_id:
        try:
            rid = _uuid.UUID(review_id)
        except (ValueError, AttributeError):
            pass

    referrer = _clean_optional_text(body.referrer) or "dashboard"

    pool = _pool_or_503()
    await pool.execute(
        """
        INSERT INTO affiliate_clicks (partner_id, review_id, referrer)
        VALUES ($1, $2, $3)
        """,
        pid,
        rid,
        referrer,
    )
    return {"ok": True}


@tenant_router.get("/clicks/summary")
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
