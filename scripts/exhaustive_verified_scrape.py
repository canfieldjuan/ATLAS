#!/usr/bin/env python3
"""Exhaustive scrape of all verified review platforms (BrightData Web Unlocker).

Scrapes G2, Capterra, Trustpilot, TrustRadius, Gartner, PeerSpot,
Software Advice, and GetApp across all enabled targets with deep pagination
to pull historical review data.

Stopping rules (in priority order):
  1. Date cutoff: stop when oldest review on page < lookback window
  2. Max reviews: stop after collecting N reviews per vendor (optional)
  3. Max pages: safety cap to prevent runaway scraping
  4. Empty pages: parser's own 2-consecutive-empty-pages logic

Rate limiting:
  - Per-source inter-vendor delay (configurable, default 3-5s)
  - Sequential vendors per source to respect proxy rate limits
  - Sources run in parallel (independent proxy sessions)

Dedup:
  - DB-level: ON CONFLICT (dedup_key) DO NOTHING

Resume:
  - Writes progress to /tmp/exhaustive_verified_progress.json after each vendor
  - On restart, skips already-completed (source, vendor) pairs

Usage:
  cd /home/juan-canfield/Desktop/Atlas
  source .venv/bin/activate
  python scripts/exhaustive_verified_scrape.py --lookback-days 365
  python scripts/exhaustive_verified_scrape.py --source g2,capterra --lookback-days 180
  python scripts/exhaustive_verified_scrape.py --resume  # pick up where last run stopped
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from atlas_brain.services.scraping.parsers import ScrapeTarget
from atlas_brain.services.scraping.relevance import filter_reviews

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/exhaustive_verified.log"),
    ],
)
logger = logging.getLogger("exhaustive_verified")
logging.getLogger("httpx").setLevel(logging.WARNING)

PROGRESS_FILE = Path("/tmp/exhaustive_verified_progress.json")

# Source configs: parser class path, default max_pages (safety cap), inter-vendor delay
# max_pages are now SAFETY CAPS -- the real stopping rule is the date cutoff.
SOURCE_CONFIGS = {
    "g2": {
        "parser": "atlas_brain.services.scraping.parsers.g2.G2Parser",
        "max_pages": 100,
        "delay": 5.0,
        "relevance_threshold": 0.0,
    },
    "capterra": {
        "parser": "atlas_brain.services.scraping.parsers.capterra.CapterraParser",
        "max_pages": 100,
        "delay": 4.0,
        "relevance_threshold": 0.0,
    },
    "trustpilot": {
        "parser": "atlas_brain.services.scraping.parsers.trustpilot.TrustpilotParser",
        "max_pages": 100,
        "delay": 4.0,
        "relevance_threshold": 0.0,
    },
    "trustradius": {
        "parser": "atlas_brain.services.scraping.parsers.trustradius.TrustRadiusParser",
        "max_pages": 80,
        "delay": 4.0,
        "relevance_threshold": 0.0,
    },
    "gartner": {
        "parser": "atlas_brain.services.scraping.parsers.gartner.GartnerParser",
        "max_pages": 80,
        "delay": 5.0,
        "relevance_threshold": 0.0,
    },
    "peerspot": {
        "parser": "atlas_brain.services.scraping.parsers.peerspot.PeerSpotParser",
        "max_pages": 60,
        "delay": 4.0,
        "relevance_threshold": 0.0,
    },
    "software_advice": {
        "parser": "atlas_brain.services.scraping.parsers.software_advice.SoftwareAdviceParser",
        "max_pages": 80,
        "delay": 4.0,
        "relevance_threshold": 0.0,
    },
    "getapp": {
        "parser": "atlas_brain.services.scraping.parsers.getapp.GetAppParser",
        "max_pages": 60,
        "delay": 4.0,
        "relevance_threshold": 0.0,
    },
}


def _import_parser(dotted_path: str):
    """Dynamically import a parser class."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _parse_review_date(raw: str | None) -> date | None:
    """Best-effort parse of a review date string to a date object."""
    if not raw:
        return None
    raw = raw.strip()
    # Try ISO format first (most common from JSON-LD datePublished)
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d",
                "%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y",
                "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(raw[:30], fmt).date()
        except (ValueError, TypeError):
            continue
    # Last resort: look for YYYY-MM-DD anywhere in string
    import re
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


def _review_date_stats(reviews: list[dict]) -> dict:
    """Compute date diagnostics for a batch of reviews."""
    dates = []
    null_count = 0
    parse_fail_count = 0
    for r in reviews:
        raw = r.get("reviewed_at")
        if not raw:
            null_count += 1
            continue
        d = _parse_review_date(str(raw))
        if d:
            dates.append(d)
        else:
            parse_fail_count += 1
    if not dates:
        return {
            "oldest": None, "newest": None, "total": len(reviews),
            "with_date": 0, "null_dates": null_count,
            "parse_failures": parse_fail_count,
        }
    return {
        "oldest": min(dates),
        "newest": max(dates),
        "total": len(reviews),
        "with_date": len(dates),
        "null_dates": null_count,
        "parse_failures": parse_fail_count,
    }


def _filter_by_date(reviews: list[dict], cutoff: date) -> tuple[list[dict], int]:
    """Keep only reviews with reviewed_at >= cutoff. Returns (kept, dropped)."""
    kept = []
    dropped = 0
    for r in reviews:
        d = _parse_review_date(str(r.get("reviewed_at", "")))
        if d is None or d >= cutoff:
            # Keep reviews with unparseable dates (let DB dedup handle them)
            kept.append(r)
        else:
            dropped += 1
    return kept, dropped


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"completed": {}, "source_stats": {}, "started_at": datetime.now(timezone.utc).isoformat()}


def save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2, default=str))


async def _log_scrape(pool, target_id, source, vendor, result, stats, duration_ms, parser_version):
    """Write a b2b_scrape_log entry with run-level telemetry. Returns run_id or None."""
    import uuid as _uuid
    oldest = stats.get("oldest_review")
    newest = stats.get("newest_review")
    try:
        oldest_date = date.fromisoformat(oldest) if oldest else None
    except (ValueError, TypeError):
        oldest_date = None
    try:
        newest_date = date.fromisoformat(newest) if newest else None
    except (ValueError, TypeError):
        newest_date = None

    # Determine if page logs should be persisted
    page_logs = getattr(result, "page_logs", []) if result else []
    should_persist_pages = bool(page_logs) and _should_persist_page_logs(stats, page_logs)

    try:
        run_id = await pool.fetchval(
            """
            INSERT INTO b2b_scrape_log
                (target_id, source, status, reviews_found, reviews_inserted,
                 pages_scraped, errors, duration_ms, proxy_type, parser_version,
                 stop_reason, oldest_review, newest_review, date_dropped, duplicate_pages,
                 has_page_logs)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10,
                    $11, $12, $13, $14, $15, $16)
            RETURNING id
            """,
            _uuid.UUID(target_id) if isinstance(target_id, str) else target_id,
            source,
            "success" if stats.get("status") == "ok" else "failed",
            stats.get("found", 0),
            stats.get("inserted", 0),
            result.pages_scraped if result else 0,
            json.dumps(result.errors[:10] if result and result.errors else []),
            duration_ms,
            "web_unlocker",
            parser_version,
            stats.get("stop_reason"),
            oldest_date,
            newest_date,
            stats.get("date_dropped", 0),
            stats.get("duplicate_pages", 0),
            should_persist_pages,
        )
    except Exception:
        logger.debug("Failed to write scrape log for %s/%s (non-fatal)", source, vendor, exc_info=True)
        return None

    # Conditionally persist page-level telemetry
    if should_persist_pages and run_id:
        await _persist_page_logs(pool, run_id, page_logs)

    return run_id


def _should_persist_page_logs(stats: dict, page_logs: list) -> bool:
    """Decide whether page logs warrant DB persistence.

    Persist when:
    - Run failed or was blocked
    - Duplicate pages detected
    - High parse failure rate (>20% missing dates)
    - Any page had a stop_reason indicating trouble
    """
    status = stats.get("status", "")
    if status in ("error", "blocked"):
        return True
    stop = stats.get("stop_reason", "")
    if stop in ("blocked_or_error", "exception"):
        return True
    # Duplicate pages
    dup_pages = sum(1 for pl in page_logs if pl.stop_reason == "duplicate_page")
    if dup_pages > 0:
        return True
    # High missing-date rate
    total_parsed = sum(pl.reviews_parsed for pl in page_logs)
    total_missing = sum(pl.missing_date for pl in page_logs)
    if total_parsed > 0 and total_missing / total_parsed > 0.2:
        return True
    # Any blocked/throttled page
    if any(pl.stop_reason in ("blocked_or_throttled", "http_error") for pl in page_logs):
        return True
    return False


async def _persist_page_logs(pool, run_id, page_logs: list) -> None:
    """Write page-level telemetry rows to b2b_scrape_page_logs."""
    for pl in page_logs:
        try:
            oldest_d = None
            newest_d = None
            if pl.oldest_review:
                try:
                    oldest_d = date.fromisoformat(pl.oldest_review)
                except (ValueError, TypeError):
                    pass
            if pl.newest_review:
                try:
                    newest_d = date.fromisoformat(pl.newest_review)
                except (ValueError, TypeError):
                    pass
            await pool.execute(
                """
                INSERT INTO b2b_scrape_page_logs
                    (run_id, page, url, requested_at, status_code, final_url,
                     response_bytes, duration_ms,
                     review_nodes_found, reviews_parsed,
                     missing_date, missing_rating, missing_body, missing_author,
                     oldest_review, newest_review,
                     next_page_found, next_page_url, content_hash,
                     duplicate_reviews, stop_reason, errors)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22::jsonb)
                """,
                run_id,
                pl.page,
                pl.url,
                datetime.fromisoformat(pl.timestamp) if pl.timestamp else datetime.now(timezone.utc),
                pl.status_code,
                pl.final_url or pl.url,
                pl.response_bytes,
                pl.duration_ms,
                pl.review_nodes_found,
                pl.reviews_parsed,
                pl.missing_date,
                pl.missing_rating,
                pl.missing_body,
                pl.missing_author,
                oldest_d,
                newest_d,
                pl.next_page_found,
                pl.next_page_url or None,
                pl.content_hash or None,
                pl.duplicate_reviews,
                pl.stop_reason or None,
                json.dumps(pl.errors[:5] if pl.errors else []),
            )
        except Exception:
            logger.debug("Failed to persist page log page=%d (non-fatal)", pl.page, exc_info=True)


async def scrape_vendor(parser, target, source, pool, relevance_threshold,
                        date_cutoff=None, max_reviews=None):
    """Scrape one vendor on one source. Returns stats dict with diagnostics."""
    vendor = target.vendor_name
    started = time.monotonic()

    try:
        result = await parser.scrape(target, None)
    except Exception as exc:
        logger.error("FAIL %s/%s: %s", source, vendor, exc)
        err_stats = {
            "vendor": vendor, "source": source, "status": "error",
            "error": str(exc)[:200], "found": 0, "inserted": 0, "duration_s": 0,
            "stop_reason": "exception",
        }
        await _log_scrape(pool, target.id, source, vendor, None, err_stats, 0,
                          f"{source}:exhaust")
        return err_stats

    duration_s = round(time.monotonic() - started, 1)
    reviews = result.reviews
    pages = result.pages_scraped
    stop_reason = "pages_exhausted"

    # Date diagnostics (before any filtering)
    date_info = _review_date_stats(reviews)

    # Relevance filter (threshold=0.0 for verified sources -- keeps everything)
    filtered_count = 0
    if reviews and relevance_threshold > 0:
        reviews, filtered_count = filter_reviews(reviews, vendor, relevance_threshold)

    # Date cutoff filter: drop reviews older than the lookback window
    date_dropped = 0
    if reviews and date_cutoff:
        reviews, date_dropped = _filter_by_date(reviews, date_cutoff)
        if date_dropped > 0:
            stop_reason = "date_cutoff"

    # Max reviews cap
    if max_reviews and len(reviews) > max_reviews:
        reviews = reviews[:max_reviews]
        stop_reason = "max_reviews"

    # Determine actual stop reason from parser behavior
    if pages >= target.max_pages:
        stop_reason = "page_cap"
    elif not result.reviews and result.errors:
        stop_reason = "blocked_or_error"
    elif not result.reviews:
        stop_reason = "no_reviews"

    # Insert into DB
    inserted = 0
    if reviews and pool:
        from atlas_brain.autonomous.tasks.b2b_scrape_intake import _insert_reviews
        batch_id = f"exhaust_{source}_{vendor.lower().replace(' ', '-')}_{int(time.time())}"
        pv = getattr(parser, "version", None)
        inserted = await _insert_reviews(pool, reviews, batch_id, parser_version=pv)

        if inserted > 0:
            from atlas_brain.autonomous.tasks.b2b_scrape_intake import _fire_enrichment
            try:
                await _fire_enrichment(batch_id, source, vendor)
            except Exception:
                logger.warning("Enrichment fire failed for %s/%s (non-fatal)", source, vendor)

    stats = {
        "vendor": vendor, "source": source, "status": "ok",
        "found": len(reviews) + filtered_count + date_dropped,
        "relevant": len(reviews),
        "inserted": inserted, "filtered": filtered_count,
        "date_dropped": date_dropped,
        "pages": pages,
        "duration_s": duration_s,
        "stop_reason": stop_reason,
        "oldest_review": str(date_info["oldest"]) if date_info["oldest"] else None,
        "newest_review": str(date_info["newest"]) if date_info["newest"] else None,
        "null_dates": date_info["null_dates"],
        "parse_failures": date_info["parse_failures"],
    }

    await _log_scrape(pool, target.id, source, vendor, result, stats,
                      int(duration_s * 1000), f"{source}:exhaust")

    logger.info(
        "%-15s %-20s found=%-4d inserted=%-4d pages=%-3d oldest=%-10s newest=%-10s stop=%s (%.1fs)",
        source, vendor,
        stats["found"], inserted, pages,
        stats["oldest_review"] or "n/a",
        stats["newest_review"] or "n/a",
        stop_reason,
        duration_s,
    )
    if date_dropped:
        logger.info("  -> %d reviews dropped (before %s)", date_dropped, date_cutoff)
    if stats["parse_failures"]:
        logger.warning("  -> %d date parse failures for %s/%s", stats["parse_failures"], source, vendor)

    return stats


async def run_source(
    source: str,
    config: dict,
    targets: list,
    pool,
    progress: dict,
    max_pages_override: int | None,
    date_cutoff: date | None,
    max_reviews: int | None,
):
    """Run exhaustive scrape for one source across all its targets."""
    ParserClass = _import_parser(config["parser"])
    parser = ParserClass()
    delay = config["delay"]
    max_pages = max_pages_override or config["max_pages"]
    threshold = config["relevance_threshold"]

    cutoff_str = str(date_cutoff) if date_cutoff else "none"
    logger.info("=" * 80)
    logger.info("SOURCE: %s -- %d vendors, max_pages=%d, date_cutoff=%s, delay=%.1fs",
                source, len(targets), max_pages, cutoff_str, delay)
    logger.info("=" * 80)

    source_results = []
    for i, target_row in enumerate(targets, 1):
        vendor = target_row["vendor_name"]
        vkey = f"{source}:{vendor}"

        if vkey in progress["completed"]:
            cached = progress["completed"][vkey]
            source_results.append(cached)
            logger.info("SKIP %s/%s (already completed)", source, vendor)
            continue

        # Build ScrapeTarget with high safety cap + date_cutoff hint
        target = ScrapeTarget(
            id=str(target_row["id"]),
            source=source,
            vendor_name=vendor,
            product_name=target_row["product_name"] or vendor,
            product_slug=target_row["product_slug"],
            product_category=target_row["product_category"] or "B2B Software",
            max_pages=max_pages,
            metadata=target_row.get("metadata") or {},
            date_cutoff=str(date_cutoff) if date_cutoff else None,
        )

        logger.info("[%d/%d] %s/%s (slug=%s, pages=%d, cutoff=%s)...",
                     i, len(targets), source, vendor,
                     target.product_slug[:40], max_pages, cutoff_str)
        stats = await scrape_vendor(
            parser, target, source, pool, threshold,
            date_cutoff=date_cutoff, max_reviews=max_reviews,
        )
        source_results.append(stats)

        progress["completed"][vkey] = stats
        save_progress(progress)

        if i < len(targets):
            await asyncio.sleep(delay)

    # Source summary
    total_found = sum(r.get("found", 0) for r in source_results)
    total_inserted = sum(r.get("inserted", 0) for r in source_results)
    total_date_dropped = sum(r.get("date_dropped", 0) for r in source_results)
    errors = sum(1 for r in source_results if r.get("status") == "error")

    # Stop reason distribution
    stop_reasons = {}
    for r in source_results:
        sr = r.get("stop_reason", "unknown")
        stop_reasons[sr] = stop_reasons.get(sr, 0) + 1

    summary = {
        "source": source, "vendors": len(targets),
        "total_found": total_found, "total_inserted": total_inserted,
        "total_date_dropped": total_date_dropped,
        "errors": errors, "stop_reasons": stop_reasons,
    }
    progress["source_stats"][source] = summary
    save_progress(progress)

    logger.info("-" * 80)
    logger.info(
        "%s COMPLETE: %d vendors, %d found, %d inserted, %d date-dropped, %d errors",
        source.upper(), len(targets), total_found, total_inserted, total_date_dropped, errors,
    )
    logger.info("  Stop reasons: %s", json.dumps(stop_reasons))
    logger.info("-" * 80)

    return summary


async def main():
    ap = argparse.ArgumentParser(description="Exhaustive verified platform scrape")
    ap.add_argument("--source", default="all",
                    help="Comma-separated sources or 'all' (default: all)")
    ap.add_argument("--max-pages", type=int, default=None,
                    help="Override max pages safety cap for all sources")
    ap.add_argument("--lookback-days", type=int, default=365,
                    help="Only keep reviews from last N days (default: 365)")
    ap.add_argument("--max-reviews", type=int, default=None,
                    help="Max reviews to keep per vendor (optional)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from progress file")
    ap.add_argument("--delay", type=float, default=None,
                    help="Override inter-vendor delay (seconds)")
    ap.add_argument("--concurrency", type=int, default=3,
                    help="How many sources to scrape in parallel (default: 3)")
    args = ap.parse_args()

    # Compute date cutoff
    date_cutoff = date.today() - timedelta(days=args.lookback_days)

    # Init DB
    from atlas_brain.storage.database import init_database, get_db_pool, close_database
    await init_database()
    pool = get_db_pool()
    logger.info("DB pool initialized")
    logger.info("Date cutoff: %s (lookback_days=%d)", date_cutoff, args.lookback_days)

    # Determine which sources to scrape
    if args.source == "all":
        sources = list(SOURCE_CONFIGS.keys())
    else:
        sources = [s.strip() for s in args.source.split(",")]
        for s in sources:
            if s not in SOURCE_CONFIGS:
                logger.error("Unknown source: %s (available: %s)", s, list(SOURCE_CONFIGS.keys()))
                await close_database()
                return

    # Load targets from DB
    source_targets = {}
    for source in sources:
        rows = await pool.fetch(
            """
            SELECT id, vendor_name, product_name, product_slug, product_category, metadata
            FROM b2b_scrape_targets
            WHERE source = $1 AND enabled = true
            ORDER BY vendor_name
            """,
            source,
        )
        if rows:
            source_targets[source] = rows
            logger.info("  %s: %d enabled targets", source, len(rows))
        else:
            logger.warning("  %s: no enabled targets -- skipping", source)

    if not source_targets:
        logger.error("No targets found for any source")
        await close_database()
        return

    # Load or reset progress
    if args.resume and PROGRESS_FILE.exists():
        progress = load_progress()
        n_done = len(progress.get("completed", {}))
        logger.info("Resuming: %d vendor/source pairs already done", n_done)
    else:
        progress = load_progress()
        if not args.resume:
            progress = {"completed": {}, "source_stats": {},
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "lookback_days": args.lookback_days,
                        "date_cutoff": str(date_cutoff)}
            save_progress(progress)

    # Apply delay override
    if args.delay is not None:
        for cfg in SOURCE_CONFIGS.values():
            cfg["delay"] = args.delay

    # Run sources with controlled concurrency
    sem = asyncio.Semaphore(args.concurrency)

    async def _run_one(source):
        async with sem:
            return await run_source(
                source, SOURCE_CONFIGS[source], source_targets[source],
                pool, progress, args.max_pages,
                date_cutoff=date_cutoff, max_reviews=args.max_reviews,
            )

    logger.info("Launching %d sources (concurrency=%d)...", len(source_targets), args.concurrency)
    await asyncio.gather(*[_run_one(s) for s in source_targets])

    # Final summary
    logger.info("=" * 80)
    logger.info("EXHAUSTIVE VERIFIED SCRAPE COMPLETE")
    total_inserted = sum(
        s.get("total_inserted", 0) for s in progress.get("source_stats", {}).values()
    )
    total_found = sum(
        s.get("total_found", 0) for s in progress.get("source_stats", {}).values()
    )
    total_dropped = sum(
        s.get("total_date_dropped", 0) for s in progress.get("source_stats", {}).values()
    )
    logger.info("Total found: %d  Total inserted: %d  Total date-dropped: %d",
                total_found, total_inserted, total_dropped)
    for source, stats in sorted(progress.get("source_stats", {}).items()):
        logger.info(
            "  %-20s vendors=%d found=%d inserted=%d dropped=%d errors=%d stops=%s",
            source, stats["vendors"], stats["total_found"],
            stats["total_inserted"], stats.get("total_date_dropped", 0),
            stats["errors"], json.dumps(stats.get("stop_reasons", {})),
        )
    logger.info("Date cutoff: %s (%d days)", date_cutoff, args.lookback_days)
    logger.info("Progress file: %s", PROGRESS_FILE)
    logger.info("Full log: /tmp/exhaustive_verified.log")
    logger.info("=" * 80)

    await close_database()


if __name__ == "__main__":
    asyncio.run(main())
