"""
Retroactively score existing social media B2B reviews and mark noise as
enrichment_status = 'filtered'.

Uses the same relevance scorer as the live intake pipeline (single source
of truth).  Idempotent -- safe to run multiple times.

Usage:
    # Preview what would be filtered (no DB changes)
    python scripts/cleanup_b2b_noise.py --dry-run --verbose

    # Run cleanup with default threshold (0.55)
    python scripts/cleanup_b2b_noise.py

    # Custom threshold
    python scripts/cleanup_b2b_noise.py --threshold 0.40

    # Verbose: print each review's score and reason
    python scripts/cleanup_b2b_noise.py --verbose
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("cleanup_b2b_noise")

# Ensure atlas_brain is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from atlas_brain.services.scraping.relevance import score_relevance  # noqa: E402
from atlas_brain.services.scraping.parsers import ScrapeTarget  # noqa: E402
from atlas_brain.services.scraping.parsers.reddit import (  # noqa: E402
    _build_alias_pattern,
    _build_strong_alias_pattern,
    _build_vendor_aliases,
    _has_ambiguous_vendor_context,
    _resolve_subreddit_context,
)

_SOCIAL_SOURCES = ("reddit", "hackernews", "github", "rss")
_DEFAULT_THRESHOLD = 0.55
_REDDIT_HARD_NOISE_MARKERS = (
    "reddit_aggregator_noise(",
    "reddit_builder_self_promo(",
    "reddit_career_subreddit_noise(",
    "reddit_investor_news(",
    "reddit_low_signal_subreddit(",
    "reddit_match_thread_noise(",
    "reddit_user_profile_subreddit(",
)

def _build_fetch_sql(include_vendor_filter: bool) -> str:
    where = [
        "source = ANY($1::text[])",
        "enrichment_status != 'filtered'",
    ]
    if include_vendor_filter:
        where.append("vendor_name = ANY($2::text[])")

    return f"""
    SELECT id, source, vendor_name, product_name, product_category,
           summary, review_text, raw_metadata, enrichment,
           enrichment_status, content_type, reddit_subreddit
    FROM b2b_reviews
    WHERE {' AND '.join(where)}
    """


def _is_protected(enrichment: dict | None) -> bool:
    """Return True if enrichment found genuine signal we should keep."""
    if not enrichment:
        return False

    # High urgency score
    urgency = enrichment.get("urgency_score")
    if isinstance(urgency, (int, float)) and urgency >= 5:
        return True

    # Explicit intent to leave
    churn = enrichment.get("churn_signals") or {}
    if churn.get("intent_to_leave"):
        return True

    # Competitors mentioned
    competitors = enrichment.get("competitors_mentioned")
    if isinstance(competitors, list) and len(competitors) > 0:
        return True

    return False


def _coerce_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _should_honor_protection(source: str, reason: str) -> bool:
    if source != "reddit":
        return True
    return not any(marker in reason for marker in _REDDIT_HARD_NOISE_MARKERS)


def _build_cleanup_target(row: dict[str, Any], target_meta: dict[str, Any]) -> ScrapeTarget:
    metadata = _coerce_json_dict(target_meta.get("metadata"))
    return ScrapeTarget(
        id=str(row["id"]),
        source="reddit",
        vendor_name=row["vendor_name"],
        product_name=target_meta.get("product_name") or row["product_name"],
        product_slug=target_meta.get("product_slug") or str(row["id"]),
        product_category=target_meta.get("product_category") or row["product_category"],
        max_pages=1,
        metadata=metadata,
    )


async def _load_reddit_targets(pool, vendor_names: list[str]) -> dict[str, dict[str, Any]]:
    if not vendor_names:
        return {}

    rows = await pool.fetch(
        """
        SELECT DISTINCT ON (vendor_name)
               vendor_name, product_name, product_slug, product_category, metadata
        FROM b2b_scrape_targets
        WHERE source = 'reddit'
          AND vendor_name = ANY($1::text[])
        ORDER BY vendor_name, enabled DESC, updated_at DESC NULLS LAST, created_at DESC NULLS LAST
        """,
        vendor_names,
    )
    return {r["vendor_name"]: dict(r) for r in rows}


def _review_dict_from_row(row: dict[str, Any], raw_meta: dict[str, Any]) -> dict[str, Any]:
    normalized_meta = dict(raw_meta)
    if row["source"] == "reddit" and row.get("reddit_subreddit") and not normalized_meta.get("subreddit"):
        normalized_meta["subreddit"] = row["reddit_subreddit"]

    return {
        "source": row["source"],
        "summary": row["summary"],
        "review_text": row["review_text"],
        "content_type": row["content_type"],
        "raw_metadata": normalized_meta,
    }


def _reddit_context_reject_reason(
    row: dict[str, Any],
    raw_meta: dict[str, Any],
    target_meta: dict[str, Any],
) -> str | None:
    target = _build_cleanup_target(row, target_meta)
    profile = str(raw_meta.get("search_profile") or "churn").lower()
    aliases = _build_vendor_aliases(
        target.vendor_name,
        target.metadata.get("vendor_aliases"),
        target.product_name,
    )
    alias_pattern = _build_alias_pattern(aliases)
    strong_alias_pattern = _build_strong_alias_pattern(target.vendor_name, aliases)
    title = row.get("summary") or ""
    body = row.get("review_text") or ""
    subreddit = row.get("reddit_subreddit") or raw_meta.get("subreddit") or ""

    if not _has_ambiguous_vendor_context(
        vendor_name=target.vendor_name,
        product_name=target.product_name,
        product_category=target.product_category,
        title=title,
        selftext=body,
        subreddit=subreddit,
        alias_pattern=alias_pattern,
        strong_alias_pattern=strong_alias_pattern,
    ):
        return "ambiguous_vendor_context"

    allowed, reason, _ = _resolve_subreddit_context(
        target=target,
        profile=profile,
        subreddit=subreddit,
        title=title,
        selftext=body,
        alias_pattern=alias_pattern,
        strong_alias_pattern=strong_alias_pattern,
    )
    return None if allowed else reason


async def run_cleanup(
    threshold: float,
    dry_run: bool,
    verbose: bool,
    sources: list[str],
    vendors: list[str],
) -> dict:
    from atlas_brain.storage.database import get_db_pool

    pool = get_db_pool()
    await pool.initialize()
    logger.info("Database pool initialized")

    fetch_sql = _build_fetch_sql(bool(vendors))
    if vendors:
        rows = await pool.fetch(fetch_sql, list(sources), vendors)
    else:
        rows = await pool.fetch(fetch_sql, list(sources))
    logger.info("Found %d social media reviews to evaluate", len(rows))

    reddit_targets = await _load_reddit_targets(
        pool,
        sorted({r["vendor_name"] for r in rows if r["source"] == "reddit"}),
    )

    to_filter: list[str] = []
    kept = 0
    protected = 0
    context_filtered = 0

    for row in rows:
        raw_meta = _coerce_json_dict(row["raw_metadata"])
        review_dict = _review_dict_from_row(row, raw_meta)

        if row["source"] == "reddit":
            target_meta = reddit_targets.get(row["vendor_name"], {})
            context_reason = _reddit_context_reject_reason(row, raw_meta, target_meta)
            if context_reason:
                to_filter.append(str(row["id"]))
                context_filtered += 1
                if verbose:
                    logger.info(
                        "FILTER_CONTEXT %-12s %-20s %s",
                        row["source"], row["vendor_name"], context_reason,
                    )
                continue

        rel_score, reason = score_relevance(
            review_dict,
            row["vendor_name"],
            row["content_type"],
        )

        if rel_score >= threshold:
            kept += 1
            if verbose:
                logger.info(
                    "KEEP  %.3f  %-12s %-15s %s",
                    rel_score, row["source"], row["vendor_name"], reason,
                )
            continue

        # Check enrichment protection
        enrichment = _coerce_json_dict(row["enrichment"])
        if not enrichment:
            enrichment = None

        if _is_protected(enrichment) and _should_honor_protection(row["source"], reason):
            protected += 1
            if verbose:
                logger.info(
                    "PROTECT %.3f  %-12s %-15s enrichment has real signal",
                    rel_score, row["source"], row["vendor_name"],
                )
            continue

        to_filter.append(str(row["id"]))
        if verbose:
            logger.info(
                "FILTER %.3f  %-12s %-15s %s",
                rel_score, row["source"], row["vendor_name"], reason,
            )

    logger.info(
        "Results: %d to filter, %d kept (score >= %.2f), %d protected, %d filtered by Reddit context gate",
        len(to_filter), kept, threshold, protected, context_filtered,
    )

    if to_filter and not dry_run:
        updated = await pool.execute(
            """
            UPDATE b2b_reviews
            SET enrichment_status = 'filtered'
            WHERE id = ANY($1::uuid[])
              AND enrichment_status != 'filtered'
            """,
            to_filter,
        )
        count = updated.split()[-1] if isinstance(updated, str) else updated
        logger.info("Updated %s rows to enrichment_status='filtered'", count)
    elif dry_run:
        logger.info("DRY RUN: would filter %d reviews", len(to_filter))

    # Final counts
    status_counts = await pool.fetch(
        "SELECT enrichment_status, COUNT(*) as cnt FROM b2b_reviews GROUP BY enrichment_status ORDER BY cnt DESC"
    )
    logger.info("Current enrichment_status distribution:")
    for sc in status_counts:
        logger.info("  %-20s %6d", sc["enrichment_status"], sc["cnt"])

    return {
        "evaluated": len(rows),
        "filtered": len(to_filter),
        "kept": kept,
        "protected": protected,
        "context_filtered": context_filtered,
        "dry_run": dry_run,
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Clean up noise from social media B2B reviews"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be filtered without changing DB",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=_DEFAULT_THRESHOLD,
        help=f"Relevance score threshold (default: {_DEFAULT_THRESHOLD:.2f})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each review's score and reason",
    )
    parser.add_argument(
        "--source",
        action="append",
        choices=sorted(_SOCIAL_SOURCES),
        dest="sources",
        help="Limit cleanup to a specific source; repeatable",
    )
    parser.add_argument(
        "--vendor",
        action="append",
        dest="vendors",
        help="Limit cleanup to a specific vendor name; repeatable",
    )
    args = parser.parse_args()

    result = await run_cleanup(
        threshold=args.threshold,
        dry_run=args.dry_run,
        verbose=args.verbose,
        sources=args.sources or list(_SOCIAL_SOURCES),
        vendors=args.vendors or [],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
