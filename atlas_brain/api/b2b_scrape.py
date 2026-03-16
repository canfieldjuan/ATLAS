"""
REST API for B2B scrape target management.

CRUD for scrape targets, manual trigger, and log viewing.
"""

import asyncio
import json
import logging
import time as _time
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..config import settings
from ..services.scraping.target_validation import is_source_allowed, validate_target_input
from ..services.vendor_registry import resolve_vendor_name
from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.b2b_scrape")

router = APIRouter(prefix="/b2b/scrape", tags=["b2b-scrape"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ScrapeTargetCreate(BaseModel):
    source: str = Field(description="Source: g2, capterra, trustradius, reddit, hackernews, github, rss")
    vendor_name: str
    product_name: Optional[str] = None
    product_slug: str
    product_category: Optional[str] = None
    max_pages: int = Field(default=5, ge=1, le=100)
    enabled: bool = True
    priority: int = Field(default=0, ge=0, le=100)
    scrape_interval_hours: int = Field(default=168, ge=1, le=8760)
    scrape_mode: str = Field(default="incremental", pattern="^(incremental|exhaustive)$")
    metadata: dict = Field(default_factory=dict)


class ScrapeTargetUpdate(BaseModel):
    enabled: Optional[bool] = None
    priority: Optional[int] = Field(default=None, ge=0, le=100)
    max_pages: Optional[int] = Field(default=None, ge=1, le=100)
    scrape_interval_hours: Optional[int] = Field(default=None, ge=1, le=8760)
    scrape_mode: Optional[str] = Field(default=None, pattern="^(incremental|exhaustive)$")
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
               product_category, max_pages, enabled, priority, scrape_mode,
               last_scraped_at, last_scrape_status, last_scrape_reviews,
               scrape_interval_hours, metadata, created_at
        FROM b2b_scrape_targets
        {where}
        ORDER BY priority DESC, vendor_name
        LIMIT 500
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

    from ..services.scraping.parsers import get_all_parsers

    source = body.source.strip().lower()
    valid_sources = set(get_all_parsers().keys())
    if source not in valid_sources:
        raise HTTPException(status_code=400, detail=f"Invalid source. Must be one of: {valid_sources}")
    if not is_source_allowed(source, settings.b2b_scrape.source_allowlist):
        raise HTTPException(
            status_code=400,
            detail=f"Source '{source}' is currently disabled by ATLAS_B2B_SCRAPE_SOURCE_ALLOWLIST",
        )

    try:
        source, product_slug = validate_target_input(source, body.product_slug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    vendor_name = await resolve_vendor_name(body.vendor_name)
    product_name = body.product_name.strip() if body.product_name else None
    product_category = body.product_category.strip() if body.product_category else None

    try:
        row = await pool.fetchrow(
            """
            INSERT INTO b2b_scrape_targets
                (source, vendor_name, product_name, product_slug,
                 product_category, max_pages, enabled, priority,
                 scrape_interval_hours, scrape_mode, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb)
            RETURNING id, source, vendor_name, product_slug, enabled, scrape_mode, created_at
            """,
            source, vendor_name, product_name, product_slug,
            product_category, body.max_pages, body.enabled, body.priority,
            body.scrape_interval_hours, body.scrape_mode, json.dumps(body.metadata),
        )
    except Exception as exc:
        if "idx_b2b_scrape_targets_dedup" in str(exc):
            raise HTTPException(
                status_code=409,
                detail=f"Target already exists for {source}/{product_slug}/{body.scrape_mode}",
            )
        logger.error("Failed to create scrape target: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create target")

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
    if body.scrape_mode is not None:
        updates.append(f"scrape_mode = ${idx}")
        args.append(body.scrape_mode)
        idx += 1
    if body.metadata is not None:
        updates.append(f"metadata = ${idx}::jsonb")
        args.append(json.dumps(body.metadata))
        idx += 1

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    updates.append("updated_at = NOW()")

    try:
        row = await pool.fetchrow(
            f"""
            UPDATE b2b_scrape_targets
            SET {', '.join(updates)}
            WHERE id = $1
            RETURNING id, source, vendor_name, product_slug, enabled, priority, scrape_mode, updated_at
            """,
            target_id, *args,
        )
    except Exception as exc:
        if "idx_b2b_scrape_targets_dedup" in str(exc):
            raise HTTPException(
                status_code=409,
                detail="A target with that source/product_slug/scrape_mode already exists",
            )
        raise

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
               product_category, max_pages, metadata, scrape_mode
        FROM b2b_scrape_targets
        WHERE id = $1
        """,
        target_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Target not found")
    if not is_source_allowed(row["source"], settings.b2b_scrape.source_allowlist):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Source '{row['source']}' is currently disabled by "
                "ATLAS_B2B_SCRAPE_SOURCE_ALLOWLIST"
            ),
        )

    # Import and run inline
    from ..services.scraping.client import get_scrape_client
    from ..services.scraping.parsers import ScrapeTarget, get_parser

    raw_meta = row["metadata"] or "{}"
    if isinstance(raw_meta, str):
        try:
            raw_meta = json.loads(raw_meta)
        except (json.JSONDecodeError, TypeError):
            raw_meta = {}
    target = ScrapeTarget(
        id=str(row["id"]),
        source=row["source"],
        vendor_name=row["vendor_name"],
        product_name=row["product_name"],
        product_slug=row["product_slug"],
        product_category=row["product_category"],
        max_pages=row["max_pages"],
        metadata=raw_meta if isinstance(raw_meta, dict) else {},
    )

    # Apply exhaustive mode config (date cutoff + raised max_pages cap)
    scrape_mode = row.get("scrape_mode", "incremental")
    if scrape_mode == "exhaustive":
        from datetime import date, timedelta
        cfg = settings.b2b_scrape
        lookback_days = raw_meta.get("lookback_days", cfg.exhaustive_lookback_days)
        target.date_cutoff = str(date.today() - timedelta(days=lookback_days))
        if target.max_pages <= 5:
            target.max_pages = raw_meta.get("exhaustive_max_pages", cfg.exhaustive_max_pages_default)

    parser = get_parser(target.source)
    if not parser:
        raise HTTPException(status_code=400, detail=f"No parser for source: {target.source}")

    client = get_scrape_client()
    started_at = _time.monotonic()

    try:
        result = await asyncio.wait_for(parser.scrape(target, client), timeout=300)
    except asyncio.TimeoutError:
        duration_ms = int((_time.monotonic() - started_at) * 1000)
        await _write_scrape_log(pool, target_id, target.source, "failed", 0, 0, 0,
                                ["scrape timed out after 300s"], duration_ms, parser)
        await pool.execute(
            """
            UPDATE b2b_scrape_targets
            SET last_scraped_at = NOW(), last_scrape_status = 'failed',
                last_scrape_reviews = 0, updated_at = NOW()
            WHERE id = $1
            """,
            target_id,
        )
        raise HTTPException(status_code=504, detail="Scrape timed out after 300s")
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
        logger.error("Scrape failed for target %s: %s", target_id, exc)
        raise HTTPException(status_code=502, detail="Scrape failed")

    # Relevance filter: drop noise from social media sources
    filtered_count = 0
    if result.reviews:
        from ..services.scraping.relevance import STRUCTURED_SOURCES, filter_reviews
        if target.source not in STRUCTURED_SOURCES:
            original_count = len(result.reviews)
            result.reviews, filtered_count = filter_reviews(
                result.reviews, target.vendor_name, 0.55,
            )

    # Exhaustive mode: date filtering
    date_dropped = 0
    if scrape_mode == "exhaustive" and result.reviews and target.date_cutoff:
        from ..autonomous.tasks.b2b_scrape_intake import _filter_by_date
        cutoff = date.fromisoformat(target.date_cutoff)
        result.reviews, date_dropped = _filter_by_date(result.reviews, cutoff)

    # Insert reviews
    inserted = 0
    pv = getattr(parser, 'version', None)
    if result.reviews:
        from ..autonomous.tasks.b2b_scrape_intake import _insert_reviews
        batch_id = f"manual_{target.source}_{target.product_slug}_{int(_time.time())}"
        inserted = await _insert_reviews(pool, result.reviews, batch_id, parser_version=pv)

    duration_ms = int((_time.monotonic() - started_at) * 1000)

    # Log to scrape log
    if scrape_mode == "exhaustive":
        from ..autonomous.tasks.b2b_scrape_intake import (
            _determine_stop_reason, _log_scrape_exhaustive, _review_date_stats,
        )
        date_info = _review_date_stats(result.reviews) if result.reviews else {"oldest": None, "newest": None}
        stop_reason = _determine_stop_reason(result, target, date_dropped)
        await _log_scrape_exhaustive(
            pool, target, result.status,
            {
                "found": len(result.reviews) + filtered_count + date_dropped,
                "inserted": inserted,
                "date_dropped": date_dropped,
                "stop_reason": stop_reason,
                "oldest_review": date_info["oldest"],
                "newest_review": date_info["newest"],
                "status": result.status,
            },
            result, parser, duration_ms,
        )
    else:
        await _write_scrape_log(pool, target_id, target.source, result.status,
                                len(result.reviews) + filtered_count, inserted, result.pages_scraped,
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
        "scrape_mode": scrape_mode,
        "reviews_found": len(result.reviews) + filtered_count + date_dropped,
        "reviews_inserted": inserted,
        "reviews_filtered": filtered_count,
        "date_dropped": date_dropped,
        "pages_scraped": result.pages_scraped,
        "duration_ms": duration_ms,
        "errors": result.errors,
    }


@router.post("/run-all")
async def trigger_scrape_all(
    source: Optional[str] = None,
) -> dict:
    """Trigger scrape for all enabled targets (optionally filtered by source).

    Runs sequentially with a 1s delay between targets. Returns summary.
    """
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    query = """
        SELECT id, source, vendor_name, product_name, product_slug,
               product_category, max_pages, metadata, scrape_mode
        FROM b2b_scrape_targets
        WHERE enabled = true
    """
    params: list = []
    if source:
        query += " AND source = $1"
        params.append(source)
    query += " ORDER BY priority DESC, source, vendor_name"

    rows = await pool.fetch(query, *params)
    if not rows:
        return {"targets": 0, "message": "No enabled targets found"}

    from datetime import date, timedelta

    from ..autonomous.tasks.b2b_scrape_intake import (
        _filter_by_date, _determine_stop_reason, _insert_reviews,
        _log_scrape_exhaustive, _review_date_stats,
    )
    from ..services.scraping.client import get_scrape_client
    from ..services.scraping.parsers import ScrapeTarget, get_parser
    from ..services.scraping.relevance import STRUCTURED_SOURCES, filter_reviews

    import asyncio

    client = get_scrape_client()
    results = []
    total_inserted = 0
    total_filtered = 0

    for row in rows:
        raw_meta = row["metadata"] or "{}"
        if isinstance(raw_meta, str):
            try:
                raw_meta = json.loads(raw_meta)
            except (json.JSONDecodeError, TypeError):
                raw_meta = {}
        target = ScrapeTarget(
            id=str(row["id"]),
            source=row["source"],
            vendor_name=row["vendor_name"],
            product_name=row["product_name"],
            product_slug=row["product_slug"],
            product_category=row["product_category"],
            max_pages=row["max_pages"],
            metadata=raw_meta if isinstance(raw_meta, dict) else {},
        )

        # Apply exhaustive mode config
        row_mode = row.get("scrape_mode", "incremental")
        if row_mode == "exhaustive":
            cfg = settings.b2b_scrape
            lookback_days = raw_meta.get("lookback_days", cfg.exhaustive_lookback_days)
            target.date_cutoff = str(date.today() - timedelta(days=lookback_days))
            if target.max_pages <= 5:
                target.max_pages = raw_meta.get("exhaustive_max_pages", cfg.exhaustive_max_pages_default)

        parser = get_parser(target.source)
        if not parser:
            results.append({"source": target.source, "vendor": target.vendor_name, "status": "no_parser"})
            continue

        started_at = _time.monotonic()
        try:
            result = await parser.scrape(target, client)
        except Exception as exc:
            duration_ms = int((_time.monotonic() - started_at) * 1000)
            await _write_scrape_log(pool, row["id"], target.source, "failed", 0, 0, 0,
                                    [str(exc)], duration_ms, parser)
            await pool.execute(
                "UPDATE b2b_scrape_targets SET last_scraped_at = NOW(), last_scrape_status = 'failed', last_scrape_reviews = 0, updated_at = NOW() WHERE id = $1",
                row["id"],
            )
            results.append({"source": target.source, "vendor": target.vendor_name, "status": "failed", "error": str(exc)[:200]})
            logger.warning("Scrape failed for %s/%s: %s", target.source, target.vendor_name, exc)
            await asyncio.sleep(1)
            continue

        # Relevance filter
        filtered_count = 0
        if result.reviews and target.source not in STRUCTURED_SOURCES:
            result.reviews, filtered_count = filter_reviews(result.reviews, target.vendor_name, 0.55)
            total_filtered += filtered_count

        # Exhaustive mode: date filtering
        date_dropped = 0
        if row_mode == "exhaustive" and result.reviews and target.date_cutoff:
            cutoff = date.fromisoformat(target.date_cutoff)
            result.reviews, date_dropped = _filter_by_date(result.reviews, cutoff)

        inserted = 0
        pv = getattr(parser, 'version', None)
        if result.reviews:
            batch_id = f"bulk_{target.source}_{target.product_slug}_{int(_time.time())}"
            inserted = await _insert_reviews(pool, result.reviews, batch_id, parser_version=pv)
            total_inserted += inserted

        duration_ms = int((_time.monotonic() - started_at) * 1000)

        if row_mode == "exhaustive":
            date_info = _review_date_stats(result.reviews) if result.reviews else {"oldest": None, "newest": None}
            stop_reason = _determine_stop_reason(result, target, date_dropped)
            await _log_scrape_exhaustive(
                pool, target, result.status,
                {
                    "found": len(result.reviews) + filtered_count + date_dropped,
                    "inserted": inserted,
                    "date_dropped": date_dropped,
                    "stop_reason": stop_reason,
                    "oldest_review": date_info["oldest"],
                    "newest_review": date_info["newest"],
                    "status": result.status,
                },
                result, parser, duration_ms,
            )
        else:
            scrape_errors = list(result.errors)
            if filtered_count:
                scrape_errors.append(f"relevance_filtered={filtered_count}")
            await _write_scrape_log(pool, row["id"], target.source, result.status,
                                    len(result.reviews) + filtered_count, inserted, result.pages_scraped,
                                    scrape_errors, duration_ms, parser)

        await pool.execute(
            "UPDATE b2b_scrape_targets SET last_scraped_at = NOW(), last_scrape_status = $2, last_scrape_reviews = $3, updated_at = NOW() WHERE id = $1",
            row["id"], result.status, inserted,
        )

        results.append({
            "source": target.source,
            "vendor": target.vendor_name,
            "status": result.status,
            "mode": row_mode,
            "found": len(result.reviews) + filtered_count + date_dropped,
            "inserted": inserted,
            "filtered": filtered_count,
            "date_dropped": date_dropped,
        })
        logger.info("Scraped %s/%s [%s]: found=%d inserted=%d filtered=%d date_dropped=%d",
                     target.source, target.vendor_name, row_mode,
                     len(result.reviews) + filtered_count + date_dropped, inserted, filtered_count, date_dropped)

        await asyncio.sleep(1)

    return {
        "targets_scraped": len(results),
        "total_inserted": total_inserted,
        "total_filtered": total_filtered,
        "results": results,
    }


async def _write_scrape_log(
    pool, target_id: UUID, source: str, status: str,
    reviews_found: int, reviews_inserted: int, pages_scraped: int,
    errors: list[str], duration_ms: int, parser,
    *, captcha_attempts: int = 0, captcha_types: list[str] | None = None,
    captcha_solve_ms: int = 0,
) -> None:
    """Write a record to b2b_scrape_log for observability."""
    proxy_type = "residential" if parser.prefer_residential else "none"
    pv = getattr(parser, 'version', None)
    # Classify block type from errors
    block_type = None
    if status in ("blocked", "failed"):
        error_text = " ".join(errors).lower()
        if "captcha" in error_text or "challenge" in error_text:
            block_type = "captcha"
        elif "403" in error_text and ("ban" in error_text or "forbidden" in error_text):
            block_type = "ip_ban"
        elif "429" in error_text or "rate" in error_text:
            block_type = "rate_limit"
        elif "403" in error_text or "blocked" in error_text:
            block_type = "waf"
        elif status == "blocked":
            block_type = "unknown"
    try:
        await pool.execute(
            """
            INSERT INTO b2b_scrape_log
                (target_id, source, status, reviews_found, reviews_inserted,
                 pages_scraped, errors, duration_ms, proxy_type, parser_version,
                 captcha_attempts, captcha_types, captcha_solve_ms, block_type)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11, $12, $13, $14)
            """,
            target_id, source, status, reviews_found, reviews_inserted,
            pages_scraped, json.dumps(errors), duration_ms, proxy_type, pv,
            captcha_attempts,
            captcha_types or [],
            captcha_solve_ms if captcha_solve_ms > 0 else None,
            block_type,
        )
    except Exception:
        logger.warning("Failed to write scrape log", exc_info=True)


@router.get("/logs")
async def list_logs(
    target_id: Optional[UUID] = None,
    limit: int = Query(default=50, ge=1, le=500),
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
