"""
REST endpoints for Amazon Seller Intelligence campaign management.

Provides:
- Seller target CRUD (manage outreach recipient list)
- Manual campaign generation trigger
- List/filter campaigns with target_mode='amazon_seller'
- Category intelligence snapshot viewer
"""

import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..config import settings
from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.seller_campaigns")

router = APIRouter(prefix="/seller", tags=["seller-campaigns"])


def _pool_or_503():
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")
    return pool


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


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SellerTargetCreate(BaseModel):
    seller_name: str | None = None
    company_name: str | None = None
    email: str | None = None
    seller_type: str = "private_label"
    categories: list[str] = []
    storefront_url: str | None = None
    notes: str | None = None
    source: str = "manual"


class SellerTargetUpdate(BaseModel):
    seller_name: str | None = None
    company_name: str | None = None
    email: str | None = None
    seller_type: str | None = None
    categories: list[str] | None = None
    storefront_url: str | None = None
    notes: str | None = None
    status: str | None = None


class GenerateRequest(BaseModel):
    category: str | None = None
    limit: int = 10


# ---------------------------------------------------------------------------
# Seller Targets CRUD
# ---------------------------------------------------------------------------


@router.get("/targets")
async def list_seller_targets(
    status: Optional[str] = Query(None),
    seller_type: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List seller targets with optional filters."""
    pool = _pool_or_503()

    conditions = []
    params = []
    idx = 1

    if status:
        conditions.append(f"status = ${idx}")
        params.append(status)
        idx += 1

    if seller_type:
        conditions.append(f"seller_type = ${idx}")
        params.append(seller_type)
        idx += 1

    if category:
        conditions.append(f"${idx} = ANY(categories)")
        params.append(category)
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.extend([limit, offset])

    rows = await pool.fetch(
        f"""
        SELECT * FROM seller_targets
        {where}
        ORDER BY created_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
        """,
        *params,
    )

    count = await pool.fetchval(
        f"SELECT COUNT(*) FROM seller_targets {where}",
        *params[:-2],
    )

    return {
        "targets": [_row_to_dict(r) for r in rows],
        "total": count,
    }


@router.post("/targets")
async def create_seller_target(body: SellerTargetCreate):
    """Create a new seller target."""
    pool = _pool_or_503()

    row = await pool.fetchrow(
        """
        INSERT INTO seller_targets (
            seller_name, company_name, email, seller_type,
            categories, storefront_url, notes, source
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING *
        """,
        body.seller_name,
        body.company_name,
        body.email,
        body.seller_type,
        body.categories,
        body.storefront_url,
        body.notes,
        body.source,
    )
    return _row_to_dict(row)


@router.get("/targets/{target_id}")
async def get_seller_target(target_id: str):
    """Get a single seller target."""
    pool = _pool_or_503()
    row = await pool.fetchrow(
        "SELECT * FROM seller_targets WHERE id = $1",
        UUID(target_id),
    )
    if not row:
        raise HTTPException(404, "Seller target not found")
    return _row_to_dict(row)


@router.patch("/targets/{target_id}")
async def update_seller_target(target_id: str, body: SellerTargetUpdate):
    """Update a seller target."""
    pool = _pool_or_503()

    updates = []
    params = []
    idx = 1

    for field in ["seller_name", "company_name", "email", "seller_type",
                   "storefront_url", "notes", "status"]:
        val = getattr(body, field, None)
        if val is not None:
            updates.append(f"{field} = ${idx}")
            params.append(val)
            idx += 1

    if body.categories is not None:
        updates.append(f"categories = ${idx}")
        params.append(body.categories)
        idx += 1

    if not updates:
        raise HTTPException(400, "No fields to update")

    updates.append(f"updated_at = NOW()")
    params.append(UUID(target_id))

    result = await pool.execute(
        f"UPDATE seller_targets SET {', '.join(updates)} WHERE id = ${idx}",
        *params,
    )

    if result.split()[-1] == "0":
        raise HTTPException(404, "Seller target not found")

    row = await pool.fetchrow(
        "SELECT * FROM seller_targets WHERE id = $1",
        UUID(target_id),
    )
    return _row_to_dict(row)


@router.delete("/targets/{target_id}")
async def delete_seller_target(target_id: str):
    """Delete a seller target."""
    pool = _pool_or_503()
    result = await pool.execute(
        "DELETE FROM seller_targets WHERE id = $1",
        UUID(target_id),
    )
    if result.split()[-1] == "0":
        raise HTTPException(404, "Seller target not found")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Campaign Generation
# ---------------------------------------------------------------------------


@router.post("/campaigns/generate")
async def trigger_generation(body: GenerateRequest):
    """Manually trigger Amazon seller campaign generation."""
    if not settings.seller_campaign.enabled:
        raise HTTPException(400, "Amazon seller campaign engine is disabled")

    pool = _pool_or_503()

    from ..autonomous.tasks.amazon_seller_campaign_generation import generate_campaigns

    result = await generate_campaigns(
        pool=pool,
        category_filter=body.category,
        limit=body.limit,
    )
    return result


# ---------------------------------------------------------------------------
# Campaign Listing (filtered view of b2b_campaigns)
# ---------------------------------------------------------------------------


@router.get("/campaigns")
async def list_seller_campaigns(
    status: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    channel: Optional[str] = Query(None),
    batch_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List campaigns with target_mode='amazon_seller'."""
    pool = _pool_or_503()

    conditions = ["target_mode = 'amazon_seller'"]
    params = []
    idx = 1

    if status:
        conditions.append(f"status = ${idx}")
        params.append(status)
        idx += 1

    if category:
        conditions.append(f"product_category = ${idx}")
        params.append(category)
        idx += 1

    if channel:
        conditions.append(f"channel = ${idx}")
        params.append(channel)
        idx += 1

    if batch_id:
        conditions.append(f"batch_id = ${idx}")
        params.append(batch_id)
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}"
    params.extend([limit, offset])

    rows = await pool.fetch(
        f"""
        SELECT id, company_name, product_category, channel,
               subject, body, cta, status, batch_id,
               recipient_email, created_at, approved_at, sent_at,
               metadata
        FROM b2b_campaigns
        {where}
        ORDER BY created_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
        """,
        *params,
    )

    count = await pool.fetchval(
        f"SELECT COUNT(*) FROM b2b_campaigns {where}",
        *params[:-2],
    )

    return {
        "campaigns": [_row_to_dict(r) for r in rows],
        "total": count,
    }


# ---------------------------------------------------------------------------
# Category Intelligence
# ---------------------------------------------------------------------------


@router.get("/intelligence")
async def list_category_intelligence(
    category: Optional[str] = Query(None),
):
    """List latest category intelligence snapshots."""
    pool = _pool_or_503()

    if category:
        rows = await pool.fetch(
            """
            SELECT * FROM category_intelligence_snapshots
            WHERE category = $1
            ORDER BY snapshot_date DESC
            LIMIT 10
            """,
            category,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT DISTINCT ON (category) *
            FROM category_intelligence_snapshots
            ORDER BY category, snapshot_date DESC
            """
        )

    return {
        "snapshots": [_row_to_dict(r) for r in rows],
    }


@router.post("/intelligence/refresh")
async def refresh_category_intelligence(
    category: Optional[str] = Query(None),
):
    """Manually refresh category intelligence snapshot(s)."""
    pool = _pool_or_503()

    from ..autonomous.tasks.amazon_seller_campaign_generation import (
        _aggregate_category_intelligence,
        _save_intelligence_snapshot,
    )

    if category:
        categories = [category]
    else:
        cat_rows = await pool.fetch(
            """
            SELECT DISTINCT source_category
            FROM product_reviews
            WHERE source_category IS NOT NULL AND source_category != ''
            """
        )
        categories = [r["source_category"] for r in cat_rows]

    refreshed = 0
    for cat in categories:
        intel = await _aggregate_category_intelligence(pool, cat)
        if intel:
            await _save_intelligence_snapshot(pool, intel)
            refreshed += 1

    return {"refreshed": refreshed, "categories": len(categories)}
