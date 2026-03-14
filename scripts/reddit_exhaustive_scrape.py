#!/usr/bin/env python3
"""Exhaustive tiered Reddit scrape -- runs all 3 search profiles across all vendors.

Strategy:
  Phase 1: churn  -- all 55 vendors (baseline churn/switching signals)
  Phase 2: deep   -- top vendors by Phase 1 yield (pain, pricing, comparison)
  Phase 3: insider -- top vendors by Phase 1 yield (employee accounts, org health)

Rate limiting:
  - Reddit OAuth2: 600 req / 10 min = 1 req/sec sustained
  - Built-in: parser has 0.3s sleep between requests + 120s pause every 480 reqs
  - This script adds 2s sleep between vendors to avoid bursts
  - Semaphore of 1 (sequential vendors) to respect rate limits

Dedup:
  - DB-level: ON CONFLICT (dedup_key) DO NOTHING -- same Reddit post ID = same dedup key
  - Cross-profile safe: churn and deep may find the same post, DB silently skips dupes

Resume:
  - Writes progress to /tmp/reddit_exhaustive_progress.json after each vendor
  - On restart, skips already-completed (vendor, profile) pairs

Usage:
  cd /home/juan-canfield/Desktop/Atlas
  source .venv/bin/activate
  python scripts/reddit_exhaustive_scrape.py [--phase 1|2|3|all] [--min-yield 20] [--resume]
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from atlas_brain.services.scraping.parsers.reddit import RedditParser
from atlas_brain.services.scraping.parsers import ScrapeTarget
from atlas_brain.services.scraping.relevance import filter_reviews

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/reddit_exhaustive.log"),
    ],
)
logger = logging.getLogger("reddit_exhaustive")

# Suppress noisy httpx request logs
logging.getLogger("httpx").setLevel(logging.WARNING)

PROGRESS_FILE = Path("/tmp/reddit_exhaustive_progress.json")
RELEVANCE_THRESHOLD = 0.55

# Cache target_id lookups so we only query once per vendor
_target_id_cache: dict[str, str | None] = {}


async def _resolve_target_id(pool, vendor_name: str) -> str | None:
    """Look up the real b2b_scrape_targets UUID for a vendor (reddit source)."""
    if vendor_name in _target_id_cache:
        return _target_id_cache[vendor_name]
    row = await pool.fetchrow(
        "SELECT id FROM b2b_scrape_targets WHERE vendor_name = $1 AND source = 'reddit' LIMIT 1",
        vendor_name,
    )
    tid = str(row["id"]) if row else None
    _target_id_cache[vendor_name] = tid
    return tid


async def _log_scrape(pool, vendor: str, profile: str, result, stats: dict, duration_ms: int) -> None:
    """Write a b2b_scrape_log entry so the admin UI can see this run."""
    import uuid as _uuid
    target_id = await _resolve_target_id(pool, vendor)
    if not target_id:
        return  # no matching target in DB, skip logging
    try:
        await pool.execute(
            """
            INSERT INTO b2b_scrape_log
                (target_id, source, status, reviews_found, reviews_inserted,
                 pages_scraped, errors, duration_ms, proxy_type, parser_version)
            VALUES ($1, 'reddit', $2, $3, $4, $5, $6::jsonb, $7, $8, $9)
            """,
            _uuid.UUID(target_id),
            "success" if stats.get("status") == "ok" else "failed",
            stats.get("found", 0),
            stats.get("inserted", 0),
            result.pages_scraped if result else 0,
            json.dumps(result.errors[:10] if result and result.errors else []),
            duration_ms,
            "none",
            f"reddit:3:exhaust:{profile}",
        )
    except Exception:
        logger.debug("Failed to write scrape log for %s (non-fatal)", vendor, exc_info=True)

# All 55 vendors from the database
ALL_VENDORS = [
    "ActiveCampaign", "Amazon Web Services", "Asana", "Azure", "BambooHR",
    "Basecamp", "BigCommerce", "Brevo", "ClickUp", "Close",
    "Copper", "CrowdStrike", "DigitalOcean", "Fortinet", "Freshdesk",
    "Freshsales", "GetResponse", "Google Cloud Platform", "Gusto", "HappyFox",
    "Help Scout", "HubSpot", "Insightly", "Intercom", "Jira",
    "Klaviyo", "Linode", "Looker", "Magento", "Mailchimp",
    "Metabase", "Microsoft Teams", "Monday.com", "Notion", "Nutshell",
    "Palo Alto Networks", "Pipedrive", "Power BI", "RingCentral", "Rippling",
    "Salesforce", "SentinelOne", "Shopify", "Slack", "Smartsheet",
    "Tableau", "Teamwork", "Trello", "WooCommerce", "Workday",
    "Wrike", "Zendesk", "Zoho CRM", "Zoho Desk", "Zoom",
]


def make_target(vendor_name: str, profile: str) -> ScrapeTarget:
    return ScrapeTarget(
        id=f"exhaust-{vendor_name.lower().replace(' ', '-')}-{profile}",
        source="reddit",
        vendor_name=vendor_name,
        product_name=vendor_name,
        product_slug=vendor_name.lower().replace(" ", "-"),
        product_category="B2B Software",
        max_pages=3,
        metadata={"search_profile": profile, "scrape_mode": "initial"},
    )


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"completed": {}, "phase_stats": {}, "started_at": datetime.now(timezone.utc).isoformat()}


def save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2, default=str))


async def scrape_vendor(
    parser: RedditParser,
    vendor: str,
    profile: str,
    pool,
) -> dict:
    """Scrape one vendor with one profile. Returns stats dict."""
    target = make_target(vendor, profile)
    started = time.monotonic()

    try:
        result = await parser.scrape(target, None)
    except Exception as exc:
        logger.error("FAIL %s [%s]: %s", vendor, profile, exc)
        err_stats = {
            "vendor": vendor, "profile": profile, "status": "error",
            "error": str(exc)[:200], "found": 0, "inserted": 0,
            "filtered": 0, "insiders": 0, "duration_s": 0,
        }
        if pool:
            await _log_scrape(pool, vendor, profile, None, err_stats, 0)
        return err_stats

    duration_s = round(time.monotonic() - started, 1)
    reviews = result.reviews

    # Relevance filter
    filtered_count = 0
    if reviews:
        reviews, filtered_count = filter_reviews(reviews, vendor, RELEVANCE_THRESHOLD)

    # Insert into DB
    inserted = 0
    if reviews and pool:
        from atlas_brain.autonomous.tasks.b2b_scrape_intake import _insert_reviews
        batch_id = f"exhaust_{profile}_{vendor.lower().replace(' ', '-')}_{int(time.time())}"
        pv = getattr(parser, "version", None)
        inserted = await _insert_reviews(pool, reviews, batch_id, parser_version=pv)

        # Fire enrichment for new inserts
        if inserted > 0:
            from atlas_brain.autonomous.tasks.b2b_scrape_intake import _fire_enrichment
            try:
                await _fire_enrichment(batch_id, "reddit", vendor)
            except Exception:
                logger.warning("Enrichment fire failed for %s (non-fatal)", batch_id)

    insiders = sum(1 for r in reviews if r.get("content_type") == "insider_account")

    stats = {
        "vendor": vendor, "profile": profile, "status": "ok",
        "found": len(reviews) + filtered_count, "relevant": len(reviews),
        "inserted": inserted, "filtered": filtered_count,
        "insiders": insiders, "duration_s": duration_s,
    }

    # Log to b2b_scrape_log so admin UI can track this run
    if pool:
        await _log_scrape(pool, vendor, profile, result, stats, int(duration_s * 1000))

    logger.info(
        "%-25s [%-7s] found=%d relevant=%d inserted=%d filtered=%d insiders=%d (%.1fs)",
        vendor, profile, stats["found"], stats["relevant"],
        inserted, filtered_count, insiders, duration_s,
    )
    return stats


async def run_phase(
    phase: int,
    vendors: list[str],
    profile: str,
    parser: RedditParser,
    pool,
    progress: dict,
    inter_vendor_delay: float = 2.0,
) -> dict:
    """Run one phase (all vendors for a given profile). Returns phase summary."""
    phase_key = f"phase_{phase}_{profile}"
    completed_key = progress["completed"]
    phase_results = []

    logger.info("=" * 80)
    logger.info("PHASE %d: %s profile -- %d vendors", phase, profile, len(vendors))
    logger.info("=" * 80)

    for i, vendor in enumerate(vendors, 1):
        vkey = f"{vendor}:{profile}"
        if vkey in completed_key:
            logger.info("SKIP %s [%s] (already completed)", vendor, profile)
            # Re-add cached stats for yield calculation
            cached = completed_key[vkey]
            phase_results.append(cached)
            continue

        logger.info("[%d/%d] Scraping %s [%s]...", i, len(vendors), vendor, profile)
        stats = await scrape_vendor(parser, vendor, profile, pool)
        phase_results.append(stats)

        # Save progress
        completed_key[vkey] = stats
        save_progress(progress)

        # Inter-vendor delay
        if i < len(vendors):
            await asyncio.sleep(inter_vendor_delay)

    # Phase summary
    total_found = sum(r.get("found", 0) for r in phase_results)
    total_relevant = sum(r.get("relevant", 0) for r in phase_results)
    total_inserted = sum(r.get("inserted", 0) for r in phase_results)
    total_filtered = sum(r.get("filtered", 0) for r in phase_results)
    total_insiders = sum(r.get("insiders", 0) for r in phase_results)
    errors = [r for r in phase_results if r.get("status") == "error"]

    summary = {
        "phase": phase, "profile": profile, "vendors": len(vendors),
        "total_found": total_found, "total_relevant": total_relevant,
        "total_inserted": total_inserted, "total_filtered": total_filtered,
        "total_insiders": total_insiders, "errors": len(errors),
    }
    progress["phase_stats"][phase_key] = summary
    save_progress(progress)

    logger.info("-" * 80)
    logger.info(
        "PHASE %d COMPLETE: %d vendors, %d found, %d relevant, %d inserted, %d filtered, %d insiders, %d errors",
        phase, len(vendors), total_found, total_relevant,
        total_inserted, total_filtered, total_insiders, len(errors),
    )
    logger.info("-" * 80)

    return summary


def select_top_vendors(progress: dict, min_yield: int) -> list[str]:
    """Select vendors with >= min_yield relevant reviews from phase 1 for deeper scraping."""
    completed = progress.get("completed", {})
    vendor_yields = {}
    for vkey, stats in completed.items():
        vendor, profile = vkey.rsplit(":", 1)
        if profile == "churn":
            vendor_yields[vendor] = stats.get("relevant", 0)

    top = [v for v, y in vendor_yields.items() if y >= min_yield]
    top.sort(key=lambda v: vendor_yields[v], reverse=True)

    logger.info(
        "Vendor yield filter: %d / %d vendors with >= %d relevant reviews",
        len(top), len(vendor_yields), min_yield,
    )
    return top


async def main():
    ap = argparse.ArgumentParser(description="Exhaustive tiered Reddit scrape")
    ap.add_argument("--phase", choices=["1", "2", "3", "all"], default="all",
                    help="Which phase(s) to run (default: all)")
    ap.add_argument("--min-yield", type=int, default=20,
                    help="Min relevant reviews from phase 1 to qualify for phases 2-3 (default: 20)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from progress file (default: start fresh)")
    ap.add_argument("--delay", type=float, default=2.0,
                    help="Seconds between vendors (default: 2.0)")
    args = ap.parse_args()

    # Init Atlas DB pool singleton (needed by resolve_vendor_name, _insert_reviews)
    from atlas_brain.storage.database import init_database, get_db_pool
    await init_database()
    pool = get_db_pool()
    logger.info("DB pool initialized")

    # Init parser
    parser = RedditParser()

    # Load or reset progress
    if args.resume and PROGRESS_FILE.exists():
        progress = load_progress()
        n_done = len(progress.get("completed", {}))
        logger.info("Resuming from progress file: %d vendor/profile pairs already done", n_done)
    else:
        progress = load_progress()
        if not args.resume:
            progress = {"completed": {}, "phase_stats": {}, "started_at": datetime.now(timezone.utc).isoformat()}
            save_progress(progress)

    phases_to_run = [1, 2, 3] if args.phase == "all" else [int(args.phase)]

    # PHASE 1: churn across all vendors
    if 1 in phases_to_run:
        await run_phase(1, ALL_VENDORS, "churn", parser, pool, progress, args.delay)

    # Select top vendors for phases 2-3
    top_vendors = select_top_vendors(progress, args.min_yield)
    if not top_vendors:
        top_vendors = ALL_VENDORS  # fallback if phase 1 wasn't run

    # PHASE 2: deep profile on top vendors
    if 2 in phases_to_run:
        await run_phase(2, top_vendors, "deep", parser, pool, progress, args.delay)

    # PHASE 3: insider profile on top vendors
    if 3 in phases_to_run:
        await run_phase(3, top_vendors, "insider", parser, pool, progress, args.delay)

    # Final summary
    logger.info("=" * 80)
    logger.info("EXHAUSTIVE SCRAPE COMPLETE")
    total_inserted = sum(
        s.get("total_inserted", 0) for s in progress.get("phase_stats", {}).values()
    )
    total_insiders = sum(
        s.get("total_insiders", 0) for s in progress.get("phase_stats", {}).values()
    )
    logger.info("Total new reviews inserted: %d", total_inserted)
    logger.info("Total insider signals: %d", total_insiders)
    logger.info("Progress file: %s", PROGRESS_FILE)
    logger.info("Full log: /tmp/reddit_exhaustive.log")
    logger.info("=" * 80)

    from atlas_brain.storage.database import close_database
    await close_database()


if __name__ == "__main__":
    asyncio.run(main())
