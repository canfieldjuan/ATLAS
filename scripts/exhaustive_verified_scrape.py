#!/usr/bin/env python3
"""Exhaustive scrape of all verified review platforms (BrightData Web Unlocker).

Scrapes G2, Capterra, Trustpilot, TrustRadius, Gartner, PeerSpot,
Software Advice, and GetApp across all enabled targets with deep pagination
to pull a year+ of review history.

Rate limiting:
  - Per-source inter-vendor delay (configurable, default 3s)
  - Sequential vendors per source to respect proxy rate limits
  - All 8 sources run in parallel (independent proxy sessions)

Dedup:
  - DB-level: ON CONFLICT (dedup_key) DO NOTHING

Resume:
  - Writes progress to /tmp/exhaustive_verified_progress.json after each vendor
  - On restart, skips already-completed (source, vendor) pairs

Usage:
  cd /home/juan-canfield/Desktop/Atlas
  source .venv/bin/activate
  python scripts/exhaustive_verified_scrape.py [--source g2,capterra,...] [--max-pages 20] [--resume]
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
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

# Source configs: parser class path, default max_pages, inter-vendor delay, relevance threshold
SOURCE_CONFIGS = {
    "g2": {
        "parser": "atlas_brain.services.scraping.parsers.g2.G2Parser",
        "max_pages": 20,
        "delay": 5.0,
        "relevance_threshold": 0.0,  # Verified source -- keep everything
    },
    "capterra": {
        "parser": "atlas_brain.services.scraping.parsers.capterra.CapterraParser",
        "max_pages": 30,
        "delay": 4.0,
        "relevance_threshold": 0.0,
    },
    "trustpilot": {
        "parser": "atlas_brain.services.scraping.parsers.trustpilot.TrustpilotParser",
        "max_pages": 20,
        "delay": 4.0,
        "relevance_threshold": 0.0,
    },
    "trustradius": {
        "parser": "atlas_brain.services.scraping.parsers.trustradius.TrustRadiusParser",
        "max_pages": 15,
        "delay": 4.0,
        "relevance_threshold": 0.0,
    },
    "gartner": {
        "parser": "atlas_brain.services.scraping.parsers.gartner.GartnerParser",
        "max_pages": 15,
        "delay": 5.0,
        "relevance_threshold": 0.0,
    },
    "peerspot": {
        "parser": "atlas_brain.services.scraping.parsers.peerspot.PeerSpotParser",
        "max_pages": 10,
        "delay": 4.0,
        "relevance_threshold": 0.0,
    },
    "software_advice": {
        "parser": "atlas_brain.services.scraping.parsers.software_advice.SoftwareAdviceParser",
        "max_pages": 15,
        "delay": 4.0,
        "relevance_threshold": 0.0,
    },
    "getapp": {
        "parser": "atlas_brain.services.scraping.parsers.getapp.GetAppParser",
        "max_pages": 15,
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
    """Write a b2b_scrape_log entry."""
    import uuid as _uuid
    try:
        await pool.execute(
            """
            INSERT INTO b2b_scrape_log
                (target_id, source, status, reviews_found, reviews_inserted,
                 pages_scraped, errors, duration_ms, proxy_type, parser_version)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10)
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
        )
    except Exception:
        logger.debug("Failed to write scrape log for %s/%s (non-fatal)", source, vendor, exc_info=True)


async def scrape_vendor(parser, target, source, pool, relevance_threshold):
    """Scrape one vendor on one source. Returns stats dict."""
    vendor = target.vendor_name
    started = time.monotonic()

    try:
        result = await parser.scrape(target, None)
    except Exception as exc:
        logger.error("FAIL %s/%s: %s", source, vendor, exc)
        err_stats = {
            "vendor": vendor, "source": source, "status": "error",
            "error": str(exc)[:200], "found": 0, "inserted": 0, "duration_s": 0,
        }
        await _log_scrape(pool, target.id, source, vendor, None, err_stats, 0,
                          f"{source}:exhaust")
        return err_stats

    duration_s = round(time.monotonic() - started, 1)
    reviews = result.reviews

    # Relevance filter (for verified sources threshold=0.0, so keeps everything)
    filtered_count = 0
    if reviews and relevance_threshold > 0:
        reviews, filtered_count = filter_reviews(reviews, vendor, relevance_threshold)

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
        "found": len(reviews) + filtered_count, "relevant": len(reviews),
        "inserted": inserted, "filtered": filtered_count,
        "pages": result.pages_scraped if result else 0,
        "duration_s": duration_s,
    }

    await _log_scrape(pool, target.id, source, vendor, result, stats,
                      int(duration_s * 1000), f"{source}:exhaust")

    logger.info(
        "%-20s %-20s found=%d inserted=%d pages=%d (%.1fs)",
        source, vendor, stats["found"], inserted, stats["pages"], duration_s,
    )
    return stats


async def run_source(
    source: str,
    config: dict,
    targets: list,
    pool,
    progress: dict,
    max_pages_override: int | None,
):
    """Run exhaustive scrape for one source across all its targets."""
    ParserClass = _import_parser(config["parser"])
    parser = ParserClass()
    delay = config["delay"]
    max_pages = max_pages_override or config["max_pages"]
    threshold = config["relevance_threshold"]

    logger.info("=" * 80)
    logger.info("SOURCE: %s -- %d vendors, max_pages=%d, delay=%.1fs",
                source, len(targets), max_pages, delay)
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

        # Build ScrapeTarget with deep pagination
        target = ScrapeTarget(
            id=str(target_row["id"]),
            source=source,
            vendor_name=vendor,
            product_name=target_row["product_name"] or vendor,
            product_slug=target_row["product_slug"],
            product_category=target_row["product_category"] or "B2B Software",
            max_pages=max_pages,
            metadata=target_row.get("metadata") or {},
        )

        logger.info("[%d/%d] %s/%s (slug=%s, pages=%d)...",
                     i, len(targets), source, vendor, target.product_slug[:40], max_pages)
        stats = await scrape_vendor(parser, target, source, pool, threshold)
        source_results.append(stats)

        progress["completed"][vkey] = stats
        save_progress(progress)

        if i < len(targets):
            await asyncio.sleep(delay)

    # Source summary
    total_found = sum(r.get("found", 0) for r in source_results)
    total_inserted = sum(r.get("inserted", 0) for r in source_results)
    errors = sum(1 for r in source_results if r.get("status") == "error")

    summary = {
        "source": source, "vendors": len(targets),
        "total_found": total_found, "total_inserted": total_inserted,
        "errors": errors,
    }
    progress["source_stats"][source] = summary
    save_progress(progress)

    logger.info("-" * 80)
    logger.info(
        "%s COMPLETE: %d vendors, %d found, %d inserted, %d errors",
        source.upper(), len(targets), total_found, total_inserted, errors,
    )
    logger.info("-" * 80)

    return summary


async def main():
    ap = argparse.ArgumentParser(description="Exhaustive verified platform scrape")
    ap.add_argument("--source", default="all",
                    help="Comma-separated sources or 'all' (default: all)")
    ap.add_argument("--max-pages", type=int, default=None,
                    help="Override max pages for all sources")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from progress file")
    ap.add_argument("--delay", type=float, default=None,
                    help="Override inter-vendor delay (seconds)")
    ap.add_argument("--concurrency", type=int, default=3,
                    help="How many sources to scrape in parallel (default: 3)")
    args = ap.parse_args()

    # Init DB
    from atlas_brain.storage.database import init_database, get_db_pool, close_database
    await init_database()
    pool = get_db_pool()
    logger.info("DB pool initialized")

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
                        "started_at": datetime.now(timezone.utc).isoformat()}
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
    logger.info("Total found: %d  Total inserted: %d", total_found, total_inserted)
    for source, stats in sorted(progress.get("source_stats", {}).items()):
        logger.info(
            "  %-20s vendors=%d found=%d inserted=%d errors=%d",
            source, stats["vendors"], stats["total_found"],
            stats["total_inserted"], stats["errors"],
        )
    logger.info("Progress file: %s", PROGRESS_FILE)
    logger.info("Full log: /tmp/exhaustive_verified.log")
    logger.info("=" * 80)

    await close_database()


if __name__ == "__main__":
    asyncio.run(main())
