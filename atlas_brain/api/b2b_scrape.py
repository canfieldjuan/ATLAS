"""
REST API for B2B scrape target management.

CRUD for scrape targets, manual trigger, and log viewing.
"""

import json
import logging
import time as _time
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.b2b_scrape")

router = APIRouter(prefix="/b2b/scrape", tags=["b2b-scrape"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ScrapeTargetCreate(BaseModel):
    source: str = Field(description="Source: g2, capterra, trustradius, reddit")
    vendor_name: str
    product_name: Optional[str] = None
    product_slug: str
    product_category: Optional[str] = None
    max_pages: int = 5
    enabled: bool = True
    priority: int = 0
    scrape_interval_hours: int = 168
    metadata: dict = Field(default_factory=dict)


class ScrapeTargetUpdate(BaseModel):
    enabled: Optional[bool] = None
    priority: Optional[int] = None
    max_pages: Optional[int] = None
    scrape_interval_hours: Optional[int] = None
    metadata: Optional[dict] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/targets")
async def list_targets(
    source: Optional[str] = None,
    enabled_only: bool = True,
) -> list[dict]:
    """List scrape targets."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    conditions = []
    args = []
    idx = 1

    if enabled_only:
        conditions.append("enabled = true")
    if source:
        conditions.append(f"source = ${idx}")
        args.append(source)
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    rows = await pool.fetch(
        f"""
        SELECT id, source, vendor_name, product_name, product_slug,
               product_category, max_pages, enabled, priority,
               last_scraped_at, last_scrape_status, last_scrape_reviews,
               scrape_interval_hours, metadata, created_at
        FROM b2b_scrape_targets
        {where}
        ORDER BY priority DESC, vendor_name
        """,
        *args,
    )

    return [dict(r) for r in rows]


@router.post("/targets")
async def create_target(body: ScrapeTargetCreate) -> dict:
    """Create a new scrape target."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    valid_sources = {"g2", "capterra", "trustradius", "reddit"}
    if body.source not in valid_sources:
        raise HTTPException(status_code=400, detail=f"Invalid source. Must be one of: {valid_sources}")

    try:
        row = await pool.fetchrow(
            """
            INSERT INTO b2b_scrape_targets
                (source, vendor_name, product_name, product_slug,
                 product_category, max_pages, enabled, priority,
                 scrape_interval_hours, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
            RETURNING id, source, vendor_name, product_slug, enabled, created_at
            """,
            body.source, body.vendor_name, body.product_name, body.product_slug,
            body.product_category, body.max_pages, body.enabled, body.priority,
            body.scrape_interval_hours, json.dumps(body.metadata),
        )
    except Exception as exc:
        if "idx_b2b_scrape_targets_dedup" in str(exc):
            raise HTTPException(
                status_code=409,
                detail=f"Target already exists for {body.source}/{body.product_slug}",
            )
        raise HTTPException(status_code=500, detail=str(exc))

    return dict(row)


@router.patch("/targets/{target_id}")
async def update_target(target_id: UUID, body: ScrapeTargetUpdate) -> dict:
    """Update a scrape target."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    updates = []
    args = []
    idx = 2  # $1 is target_id

    if body.enabled is not None:
        updates.append(f"enabled = ${idx}")
        args.append(body.enabled)
        idx += 1
    if body.priority is not None:
        updates.append(f"priority = ${idx}")
        args.append(body.priority)
        idx += 1
    if body.max_pages is not None:
        updates.append(f"max_pages = ${idx}")
        args.append(body.max_pages)
        idx += 1
    if body.scrape_interval_hours is not None:
        updates.append(f"scrape_interval_hours = ${idx}")
        args.append(body.scrape_interval_hours)
        idx += 1
    if body.metadata is not None:
        updates.append(f"metadata = ${idx}::jsonb")
        args.append(json.dumps(body.metadata))
        idx += 1

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    updates.append("updated_at = NOW()")

    row = await pool.fetchrow(
        f"""
        UPDATE b2b_scrape_targets
        SET {', '.join(updates)}
        WHERE id = $1
        RETURNING id, source, vendor_name, product_slug, enabled, priority, updated_at
        """,
        target_id, *args,
    )

    if not row:
        raise HTTPException(status_code=404, detail="Target not found")
    return dict(row)


@router.delete("/targets/{target_id}")
async def delete_target(target_id: UUID) -> dict:
    """Delete a scrape target and its logs."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    async with pool.transaction() as conn:
        await conn.execute("DELETE FROM b2b_scrape_log WHERE target_id = $1", target_id)
        result = await conn.execute("DELETE FROM b2b_scrape_targets WHERE id = $1", target_id)

    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Target not found")
    return {"deleted": str(target_id)}


@router.post("/targets/{target_id}/run")
async def trigger_scrape(target_id: UUID) -> dict:
    """Manually trigger a scrape for a specific target."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    row = await pool.fetchrow(
        """
        SELECT id, source, vendor_name, product_name, product_slug,
               product_category, max_pages, metadata
        FROM b2b_scrape_targets
        WHERE id = $1
        """,
        target_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Target not found")

    # Import and run inline
    from ..services.scraping.client import get_scrape_client
    from ..services.scraping.parsers import ScrapeTarget, get_parser

    raw_meta = row["metadata"] or "{}"
    target = ScrapeTarget(
        id=str(row["id"]),
        source=row["source"],
        vendor_name=row["vendor_name"],
        product_name=row["product_name"],
        product_slug=row["product_slug"],
        product_category=row["product_category"],
        max_pages=row["max_pages"],
        metadata=json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta,
    )

    parser = get_parser(target.source)
    if not parser:
        raise HTTPException(status_code=400, detail=f"No parser for source: {target.source}")

    client = get_scrape_client()
    started_at = _time.monotonic()

    try:
        result = await parser.scrape(target, client)
    except Exception as exc:
        duration_ms = int((_time.monotonic() - started_at) * 1000)
        # Log failure
        await _write_scrape_log(pool, target_id, target.source, "failed", 0, 0, 0,
                                [str(exc)], duration_ms, parser)
        await pool.execute(
            """
            UPDATE b2b_scrape_targets
            SET last_scraped_at = NOW(), last_scrape_status = 'failed',
                last_scrape_reviews = 0, updated_at = NOW()
            WHERE id = $1
            """,
            target_id,
        )
        raise HTTPException(status_code=502, detail=f"Scrape failed: {exc}")

    # Insert reviews
    inserted = 0
    if result.reviews:
        from ..autonomous.tasks.b2b_scrape_intake import _insert_reviews
        batch_id = f"manual_{target.source}_{target.product_slug}_{int(_time.time())}"
        inserted = await _insert_reviews(pool, result.reviews, batch_id)

    duration_ms = int((_time.monotonic() - started_at) * 1000)

    # Log to scrape log
    await _write_scrape_log(pool, target_id, target.source, result.status,
                            len(result.reviews), inserted, result.pages_scraped,
                            result.errors, duration_ms, parser)

    # Update target status
    await pool.execute(
        """
        UPDATE b2b_scrape_targets
        SET last_scraped_at = NOW(), last_scrape_status = $2,
            last_scrape_reviews = $3, updated_at = NOW()
        WHERE id = $1
        """,
        target_id, result.status, inserted,
    )

    return {
        "target_id": str(target_id),
        "source": target.source,
        "vendor": target.vendor_name,
        "status": result.status,
        "reviews_found": len(result.reviews),
        "reviews_inserted": inserted,
        "pages_scraped": result.pages_scraped,
        "duration_ms": duration_ms,
        "errors": result.errors,
    }


async def _write_scrape_log(
    pool, target_id: UUID, source: str, status: str,
    reviews_found: int, reviews_inserted: int, pages_scraped: int,
    errors: list[str], duration_ms: int, parser,
) -> None:
    """Write a record to b2b_scrape_log for observability."""
    proxy_type = "residential" if parser.prefer_residential else "none"
    try:
        await pool.execute(
            """
            INSERT INTO b2b_scrape_log
                (target_id, source, status, reviews_found, reviews_inserted,
                 pages_scraped, errors, duration_ms, proxy_type)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9)
            """,
            target_id, source, status, reviews_found, reviews_inserted,
            pages_scraped, json.dumps(errors), duration_ms, proxy_type,
        )
    except Exception:
        logger.debug("Failed to write scrape log", exc_info=True)


@router.get("/logs")
async def list_logs(
    target_id: Optional[UUID] = None,
    limit: int = 50,
) -> list[dict]:
    """View scrape execution logs."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    if target_id:
        rows = await pool.fetch(
            """
            SELECT l.*, t.vendor_name, t.source as target_source
            FROM b2b_scrape_log l
            JOIN b2b_scrape_targets t ON t.id = l.target_id
            WHERE l.target_id = $1
            ORDER BY l.started_at DESC
            LIMIT $2
            """,
            target_id, limit,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT l.*, t.vendor_name, t.source as target_source
            FROM b2b_scrape_log l
            JOIN b2b_scrape_targets t ON t.id = l.target_id
            ORDER BY l.started_at DESC
            LIMIT $1
            """,
            limit,
        )

    return [dict(r) for r in rows]
