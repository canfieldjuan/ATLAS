"""
B2B review scrape intake: poll configured scrape targets, fetch reviews
from G2, Capterra, TrustRadius, Reddit, HackerNews, GitHub, and RSS feeds,
and insert into b2b_reviews for automatic enrichment pickup.

Runs as an autonomous task on a configurable interval (default 1 hour).
"""

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid as _uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from ...services.scraping.sources import API_SOURCES, parse_source_allowlist
from ...services.vendor_registry import resolve_vendor_name

logger = logging.getLogger("atlas.autonomous.tasks.b2b_scrape_intake")


# Common date formats returned by review site parsers
_DATE_FORMATS = [
    "%Y-%m-%dT%H:%M:%S%z",         # ISO 8601 with tz
    "%Y-%m-%dT%H:%M:%S",           # ISO 8601 no tz
    "%Y-%m-%d",                      # ISO date only
    "%b %d, %Y",                     # "Feb 15, 2024"
    "%B %d, %Y",                     # "February 15, 2024"
    "%d %b %Y",                      # "15 Feb 2024"
    "%d %B %Y",                      # "15 February 2024"
    "%m/%d/%Y",                      # "02/15/2024"
    "%d/%m/%Y",                      # "15/02/2024" (EU)
]


def _parse_date(raw: Any) -> datetime | None:
    """Parse a date string in various formats.  Returns None on failure."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    # Fast path: ISO 8601 (most common from APIs)
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        pass

    # Try common formats
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue

    # Last resort: extract "Month Day, Year" or similar from longer text
    m = re.search(
        r"(\w+ \d{1,2},?\s+\d{4})",
        s,
    )
    if m:
        for fmt in ("%b %d, %Y", "%B %d, %Y", "%b %d %Y", "%B %d %Y"):
            try:
                return datetime.strptime(m.group(1), fmt).replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

    return None


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
    import_batch_id, raw_metadata, parser_version
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22
)
ON CONFLICT (dedup_key) DO NOTHING
"""

_TARGET_QUERY = """
SELECT id, source, vendor_name, product_name, product_slug,
       product_category, max_pages, metadata
FROM b2b_scrape_targets
WHERE enabled = true
    AND source = ANY($3::text[])
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
    from ...services.scraping.parsers import ScrapeTarget, get_all_parsers, get_parser
    from ...services.scraping.relevance import STRUCTURED_SOURCES, filter_reviews

    client = get_scrape_client()
    allowed_sources = parse_source_allowlist(cfg.source_allowlist)
    if not allowed_sources:
        allowed_sources = list(get_all_parsers().keys())

    # Fetch due targets
    targets = await pool.fetch(
        _TARGET_QUERY,
        cfg.blocked_cooldown_hours,
        cfg.max_targets_per_run,
        allowed_sources,
    )

    if not targets:
        return {"_skip_synthesis": True, "targets_due": 0}

    total_reviews = 0
    total_inserted = 0
    results_summary: list[dict] = []
    results_lock = asyncio.Lock()

    # Group targets by source for concurrent scraping.
    # Each source gets its own semaphore to respect rate limits:
    #   - API sources (youtube, stackoverflow, producthunt, hackernews, github, rss):
    #     high concurrency — APIs handle it, rate limiter throttles per-domain
    #   - Web scrape sources (g2, capterra, trustradius, getapp, gartner, peerspot,
    #     trustpilot, quora, reddit): lower concurrency to avoid proxy overload
    _API_SOURCES = API_SOURCES
    _WEB_CONCURRENCY = 4   # Concurrent web scrape targets
    _API_CONCURRENCY = 10  # Concurrent API targets

    source_sems: dict[str, asyncio.Semaphore] = {}
    for row in targets:
        src = row["source"]
        if src not in source_sems:
            limit = _API_CONCURRENCY if src in _API_SOURCES else _WEB_CONCURRENCY
            source_sems[src] = asyncio.Semaphore(limit)

    async def _scrape_one(row):
        """Scrape a single target with per-source concurrency control."""
        nonlocal total_reviews, total_inserted

        raw_meta = row["metadata"] or "{}"
        if isinstance(raw_meta, str):
            try:
                raw_meta = json.loads(raw_meta)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Malformed metadata JSON for target %s, defaulting to empty", row["id"])
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

        parser = get_parser(target.source)
        if not parser:
            logger.warning("No parser for source %r, skipping target %s", target.source, target.id)
            return

        sem = source_sems.get(target.source, asyncio.Semaphore(2))
        async with sem:
            started_at = time.monotonic()
            batch_id = f"scrape_{target.source}_{target.product_slug}_{int(time.time())}"

            try:
                result = await parser.scrape(target, client)
            except Exception as exc:
                logger.error("Scrape failed for %s/%s: %s", target.source, target.vendor_name, exc)
                duration_ms = int((time.monotonic() - started_at) * 1000)
                await _log_scrape(pool, target, "failed", 0, 0, 0, [str(exc)], duration_ms, parser)
                await pool.execute(
                    """
                    UPDATE b2b_scrape_targets
                    SET last_scraped_at = NOW(), last_scrape_status = 'failed',
                        last_scrape_reviews = 0, updated_at = NOW()
                    WHERE id = $1
                    """,
                    row["id"],
                )
                async with results_lock:
                    results_summary.append({
                        "source": target.source,
                        "vendor": target.vendor_name,
                        "status": "failed",
                        "error": str(exc),
                    })
                return

            # Relevance filter
            filtered_count = 0
            if (cfg.relevance_filter_enabled
                    and target.source not in STRUCTURED_SOURCES
                    and result.reviews):
                original_count = len(result.reviews)
                result.reviews, filtered_count = filter_reviews(
                    result.reviews, target.vendor_name, cfg.relevance_threshold,
                )
                if filtered_count:
                    logger.info(
                        "Relevance filter: kept %d/%d for %s/%s",
                        len(result.reviews), original_count,
                        target.source, target.vendor_name,
                    )

            # Insert reviews + fire enrichment immediately (background task)
            inserted = 0
            pv = getattr(parser, 'version', None)
            if result.reviews:
                inserted = await _insert_reviews(pool, result.reviews, batch_id, parser_version=pv)

                # Fire enrichment NOW — don't wait for it, let vLLM chew
                if inserted > 0:
                    asyncio.create_task(
                        _fire_enrichment(batch_id, target.source, target.vendor_name),
                        name=f"enrich_{batch_id}",
                    )

                # Mark synthetic aggregate reviews as not_applicable
                synthetic_keys = [
                    _make_dedup_key(
                        r["source"], r["vendor_name"],
                        r.get("source_review_id"),
                        r.get("reviewer_name"),
                        r.get("reviewed_at"),
                    )
                    for r in result.reviews
                    if r.get("raw_metadata", {}).get("extraction_method") == "jsonld_aggregate"
                ]
                if synthetic_keys:
                    await pool.execute(
                        """
                        UPDATE b2b_reviews
                        SET enrichment_status = 'not_applicable'
                        WHERE dedup_key = ANY($1::text[])
                          AND enrichment_status = 'pending'
                        """,
                        synthetic_keys,
                    )

            duration_ms = int((time.monotonic() - started_at) * 1000)

            # Log + update target
            scrape_errors = list(result.errors)
            if filtered_count:
                scrape_errors.append(f"relevance_filtered={filtered_count}")
            await _log_scrape(
                pool, target, result.status,
                len(result.reviews) + filtered_count, inserted, result.pages_scraped,
                scrape_errors, duration_ms, parser,
            )
            await pool.execute(
                """
                UPDATE b2b_scrape_targets
                SET last_scraped_at = NOW(), last_scrape_status = $2,
                    last_scrape_reviews = $3, updated_at = NOW()
                WHERE id = $1
                """,
                row["id"], result.status, inserted,
            )

            async with results_lock:
                total_reviews += len(result.reviews) + filtered_count
                total_inserted += inserted
                results_summary.append({
                    "source": target.source,
                    "vendor": target.vendor_name,
                    "status": result.status,
                    "found": len(result.reviews) + filtered_count,
                    "inserted": inserted,
                    "filtered": filtered_count,
                    "pages": result.pages_scraped,
                })

            logger.info(
                "Scraped %s/%s: %d found, %d inserted (%s) in %dms",
                target.source, target.vendor_name,
                len(result.reviews), inserted, result.status, duration_ms,
            )

    # Fire all targets concurrently (per-source semaphores handle throttling)
    logger.info(
        "Scraping %d targets concurrently (API: %d concurrent, Web: %d concurrent)",
        len(targets), _API_CONCURRENCY, _WEB_CONCURRENCY,
    )
    await asyncio.gather(
        *[_scrape_one(row) for row in targets],
        return_exceptions=True,
    )

    # Enrichment fires as background tasks per-target (see _fire_enrichment).
    # No need to wait — vLLM handles concurrent requests natively.
    # The b2b_enrichment scheduler task catches any stragglers every 30s.

    return {
        "_skip_synthesis": True,
        "targets_scraped": len(results_summary),
        "total_reviews_found": total_reviews,
        "total_reviews_inserted": total_inserted,
        "results": results_summary,
    }


async def _fire_enrichment(batch_id: str, source: str, vendor: str) -> None:
    """Fire-and-forget enrichment for a single scrape batch.

    Runs as a background asyncio task so scraping continues unblocked.
    vLLM with PagedAttention handles concurrent requests natively.
    """
    try:
        from .b2b_enrichment import enrich_batch
        result = await enrich_batch(batch_id)
        logger.info(
            "Enrichment done for %s/%s: %s",
            source, vendor, result,
        )
    except Exception as exc:
        logger.warning(
            "Background enrichment failed for %s/%s (scheduler retries): %s",
            source, vendor, exc,
        )


_MIN_ENRICHABLE_TEXT_LEN = 80  # Reviews shorter than this can't produce useful enrichment


async def _insert_reviews(pool, reviews: list[dict], batch_id: str, parser_version: str | None = None) -> int:
    """Insert reviews into b2b_reviews with dedup. Returns count of new inserts."""
    rows = []
    skipped_short = 0
    for r in reviews:
        # Gate: don't insert reviews with no meaningful text body
        # Combine review_text + pros + cons for length check (some sources
        # put the substance in pros/cons rather than the main body)
        review_text = r.get("review_text") or ""
        pros = r.get("pros") or ""
        cons = r.get("cons") or ""
        combined_len = len(review_text) + len(pros) + len(cons)
        if combined_len < _MIN_ENRICHABLE_TEXT_LEN:
            skipped_short += 1
            continue

        reviewed_at_ts = _parse_date(r.get("reviewed_at"))

        dedup_key = _make_dedup_key(
            r["source"], r["vendor_name"],
            r.get("source_review_id"),
            r.get("reviewer_name"),
            r.get("reviewed_at"),
        )

        # Resolve to canonical vendor name (dedup key uses raw value above)
        canonical_vendor = await resolve_vendor_name(r["vendor_name"])

        rows.append((
            dedup_key,
            r["source"],
            r.get("source_url"),
            r.get("source_review_id"),
            canonical_vendor,
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
            parser_version,
        ))

    if skipped_short:
        logger.info("Skipped %d reviews with text < %d chars", skipped_short, _MIN_ENRICHABLE_TEXT_LEN)

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
    pv = getattr(parser, 'version', None)
    try:
        await pool.execute(
            """
            INSERT INTO b2b_scrape_log
                (target_id, source, status, reviews_found, reviews_inserted,
                 pages_scraped, errors, duration_ms, proxy_type, parser_version)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10)
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
            pv,
        )
    except Exception:
        logger.warning("Failed to log scrape result", exc_info=True)
