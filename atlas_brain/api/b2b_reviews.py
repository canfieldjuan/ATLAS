"""
REST API for B2B review import.

Accepts JSON array of reviews from any source and inserts into b2b_reviews
table with dedup via SHA-256 keys.
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.b2b_reviews")

router = APIRouter(prefix="/b2b/reviews", tags=["b2b-reviews"])


class B2BReviewInput(BaseModel):
    source: str = Field(description="Review source: g2, capterra, trustradius, reddit, manual")
    vendor_name: str = Field(description="Vendor/company name")
    review_text: str = Field(description="Main review body text")

    source_url: Optional[str] = None
    source_review_id: Optional[str] = None
    product_name: Optional[str] = None
    product_category: Optional[str] = None
    rating: Optional[float] = None
    rating_max: int = 5
    summary: Optional[str] = None
    pros: Optional[str] = None
    cons: Optional[str] = None
    reviewer_name: Optional[str] = None
    reviewer_title: Optional[str] = None
    reviewer_company: Optional[str] = None
    company_size_raw: Optional[str] = None
    reviewer_industry: Optional[str] = None
    reviewed_at: Optional[str] = None
    metadata: Optional[dict] = Field(default_factory=dict)


_INSERT_SQL = """
INSERT INTO b2b_reviews (
    dedup_key, source, source_url, source_review_id,
    vendor_name, product_name, product_category,
    rating, rating_max, summary, review_text, pros, cons,
    reviewer_name, reviewer_title, reviewer_company,
    company_size_raw, reviewer_industry, reviewed_at,
    import_batch_id, raw_metadata
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21
)
ON CONFLICT (dedup_key) DO NOTHING
"""


def _make_dedup_key(source: str, vendor_name: str, source_review_id: str | None,
                    reviewer_name: str | None, reviewed_at: str | None) -> str:
    if source_review_id:
        raw = f"{source}:{vendor_name}:{source_review_id}"
    else:
        raw = f"{source}:{vendor_name}:{reviewer_name or ''}:{reviewed_at or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()


@router.post("/import")
async def import_b2b_reviews(reviews: list[B2BReviewInput]) -> dict:
    """Import B2B reviews from any source. Accepts JSON array."""
    if not reviews:
        raise HTTPException(status_code=400, detail="Empty review list")

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    batch_id = f"api_{int(time.time())}"
    rows = []
    for r in reviews:
        reviewed_at_ts = None
        if r.reviewed_at:
            try:
                reviewed_at_ts = datetime.fromisoformat(r.reviewed_at.replace("Z", "+00:00"))
            except ValueError:
                pass

        dedup_key = _make_dedup_key(
            r.source, r.vendor_name, r.source_review_id,
            r.reviewer_name, r.reviewed_at,
        )

        rows.append((
            dedup_key,
            r.source,
            r.source_url,
            r.source_review_id,
            r.vendor_name,
            r.product_name,
            r.product_category,
            r.rating,
            r.rating_max,
            r.summary,
            r.review_text,
            r.pros,
            r.cons,
            r.reviewer_name,
            r.reviewer_title,
            r.reviewer_company,
            r.company_size_raw,
            r.reviewer_industry,
            reviewed_at_ts,
            batch_id,
            json.dumps(r.metadata or {}),
        ))

    try:
        async with pool.transaction() as conn:
            await conn.executemany(_INSERT_SQL, rows)
    except Exception:
        logger.exception("Failed to import B2B reviews")
        raise HTTPException(status_code=500, detail="Database insert failed")

    # Count how many actually inserted (not duplicates)
    count_row = await pool.fetchrow(
        "SELECT count(*) as cnt FROM b2b_reviews WHERE import_batch_id = $1",
        batch_id,
    )
    imported = count_row["cnt"] if count_row else 0
    duplicates = len(reviews) - imported

    logger.info("B2B review import: %d imported, %d duplicates (batch %s)", imported, duplicates, batch_id)

    return {
        "imported": imported,
        "duplicates": duplicates,
        "total": len(reviews),
        "batch_id": batch_id,
    }
