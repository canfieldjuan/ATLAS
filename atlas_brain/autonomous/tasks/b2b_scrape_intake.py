"""
B2B review scrape intake: poll configured scrape targets, fetch reviews
from G2, Capterra, TrustRadius, and Reddit, and insert into b2b_reviews
for automatic enrichment pickup.

Runs as an autonomous task on a configurable interval (default 1 hour).
"""

import hashlib
import json
import logging
import time
import uuid as _uuid
from datetime import datetime, timezone
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.b2b_scrape_intake")


def _make_dedup_key(
    source: str,
    vendor_name: str,
    source_review_id: str | None,
    reviewer_name: str | None,
    reviewed_at: str | None,
) -> str:
    """Generate deterministic dedup key for a review.

    Identical logic to api/b2b_reviews.py and scripts/import_b2b_reviews.py.
    """
    if source_review_id:
        raw = f"{source}:{vendor_name}:{source_review_id}"
    else:
        raw = f"{source}:{vendor_name}:{reviewer_name or ''}:{reviewed_at or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()


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

_TARGET_QUERY = """
SELECT id, source, vendor_name, product_name, product_slug,
       product_category, max_pages, metadata
FROM b2b_scrape_targets
WHERE enabled = true
  AND (last_scraped_at IS NULL
       OR last_scraped_at < NOW() - make_interval(hours => scrape_interval_hours))
  AND (last_scrape_status IS NULL
       OR last_scrape_status != 'blocked'
       OR last_scraped_at < NOW() - make_interval(hours => $1))
ORDER BY priority DESC, last_scraped_at ASC NULLS FIRST
LIMIT $2
"""


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: scrape B2B review sites per configured targets."""
    cfg = settings.b2b_scrape
    if not cfg.enabled:
        return {"_skip_synthesis": True, "skipped": "b2b_scrape disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": True, "skipped": "db not ready"}

    # Import here to avoid circular imports and lazy-load curl_cffi
    from ...services.scraping.client import get_scrape_client
    from ...services.scraping.parsers import ScrapeTarget, get_parser

    client = get_scrape_client()

    # Fetch due targets
    targets = await pool.fetch(
        _TARGET_QUERY,
        cfg.blocked_cooldown_hours,
        cfg.max_targets_per_run,
    )

    if not targets:
        return {"_skip_synthesis": True, "targets_due": 0}

    total_reviews = 0
    total_inserted = 0
    results_summary: list[dict] = []

    for row in targets:
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
            logger.warning("No parser for source %r, skipping target %s", target.source, target.id)
            continue

        started_at = time.monotonic()
        batch_id = f"scrape_{target.source}_{target.product_slug}_{int(time.time())}"

        try:
            result = await parser.scrape(target, client)
        except Exception as exc:
            logger.error("Scrape failed for %s/%s: %s", target.source, target.vendor_name, exc)
            duration_ms = int((time.monotonic() - started_at) * 1000)

            # Log failure
            await _log_scrape(pool, target, "failed", 0, 0, 0, [str(exc)], duration_ms, parser)

            # Update target status
            await pool.execute(
                """
                UPDATE b2b_scrape_targets
                SET last_scraped_at = NOW(), last_scrape_status = 'failed',
                    last_scrape_reviews = 0, updated_at = NOW()
                WHERE id = $1
                """,
                row["id"],
            )
            results_summary.append({
                "source": target.source,
                "vendor": target.vendor_name,
                "status": "failed",
                "error": str(exc),
            })
            continue

        # Insert reviews
        inserted = 0
        if result.reviews:
            inserted = await _insert_reviews(pool, result.reviews, batch_id)

        duration_ms = int((time.monotonic() - started_at) * 1000)
        total_reviews += len(result.reviews)
        total_inserted += inserted

        # Log to b2b_scrape_log
        await _log_scrape(
            pool, target, result.status,
            len(result.reviews), inserted, result.pages_scraped,
            result.errors, duration_ms, parser,
        )

        # Update target status
        await pool.execute(
            """
            UPDATE b2b_scrape_targets
            SET last_scraped_at = NOW(), last_scrape_status = $2,
                last_scrape_reviews = $3, updated_at = NOW()
            WHERE id = $1
            """,
            row["id"], result.status, inserted,
        )

        results_summary.append({
            "source": target.source,
            "vendor": target.vendor_name,
            "status": result.status,
            "found": len(result.reviews),
            "inserted": inserted,
            "pages": result.pages_scraped,
        })

        logger.info(
            "Scraped %s/%s: %d found, %d inserted (%s) in %dms",
            target.source, target.vendor_name,
            len(result.reviews), inserted, result.status, duration_ms,
        )

    return {
        "_skip_synthesis": True,
        "targets_scraped": len(results_summary),
        "total_reviews_found": total_reviews,
        "total_reviews_inserted": total_inserted,
        "results": results_summary,
    }


async def _insert_reviews(pool, reviews: list[dict], batch_id: str) -> int:
    """Insert reviews into b2b_reviews with dedup. Returns count of new inserts."""
    rows = []
    for r in reviews:
        reviewed_at_ts = None
        if r.get("reviewed_at"):
            try:
                reviewed_at_ts = datetime.fromisoformat(
                    str(r["reviewed_at"]).replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        dedup_key = _make_dedup_key(
            r["source"], r["vendor_name"],
            r.get("source_review_id"),
            r.get("reviewer_name"),
            r.get("reviewed_at"),
        )

        rows.append((
            dedup_key,
            r["source"],
            r.get("source_url"),
            r.get("source_review_id"),
            r["vendor_name"],
            r.get("product_name"),
            r.get("product_category"),
            r.get("rating"),
            r.get("rating_max") or 5,
            r.get("summary"),
            r["review_text"],
            r.get("pros"),
            r.get("cons"),
            r.get("reviewer_name"),
            r.get("reviewer_title"),
            r.get("reviewer_company"),
            r.get("company_size_raw"),
            r.get("reviewer_industry"),
            reviewed_at_ts,
            batch_id,
            json.dumps(r.get("raw_metadata", {})),
        ))

    if not rows:
        return 0

    try:
        async with pool.transaction() as conn:
            await conn.executemany(_INSERT_SQL, rows)
    except Exception:
        logger.exception("Failed to insert scraped reviews (batch %s)", batch_id)
        return 0

    # Count actual inserts
    count_row = await pool.fetchrow(
        "SELECT count(*) as cnt FROM b2b_reviews WHERE import_batch_id = $1",
        batch_id,
    )
    return count_row["cnt"] if count_row else 0


async def _log_scrape(
    pool, target, status: str, reviews_found: int, reviews_inserted: int,
    pages_scraped: int, errors: list[str], duration_ms: int, parser,
) -> None:
    """Insert a record into b2b_scrape_log."""
    proxy_type = "residential" if parser.prefer_residential else "none"
    try:
        await pool.execute(
            """
            INSERT INTO b2b_scrape_log
                (target_id, source, status, reviews_found, reviews_inserted,
                 pages_scraped, errors, duration_ms, proxy_type)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9)
            """,
            _uuid.UUID(target.id),
            target.source,
            status,
            reviews_found,
            reviews_inserted,
            pages_scraped,
            json.dumps(errors),
            duration_ms,
            proxy_type,
        )
    except Exception:
        logger.debug("Failed to log scrape result", exc_info=True)
