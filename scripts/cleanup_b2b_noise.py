"""
Retroactively score existing social media B2B reviews and mark noise as
enrichment_status = 'filtered'.

Uses the same relevance scorer as the live intake pipeline (single source
of truth).  Idempotent -- safe to run multiple times.

Usage:
    # Preview what would be filtered (no DB changes)
    python scripts/cleanup_b2b_noise.py --dry-run --verbose

    # Run cleanup with default threshold (0.35)
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("cleanup_b2b_noise")

# Ensure atlas_brain is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from atlas_brain.services.scraping.relevance import score_relevance  # noqa: E402

_SOCIAL_SOURCES = ("reddit", "hackernews", "github", "rss")

_FETCH_SQL = """
SELECT id, source, vendor_name, summary, review_text,
       raw_metadata, enrichment, enrichment_status
FROM b2b_reviews
WHERE source = ANY($1::text[])
  AND enrichment_status != 'filtered'
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


async def run_cleanup(
    threshold: float,
    dry_run: bool,
    verbose: bool,
) -> dict:
    from atlas_brain.storage.database import get_db_pool

    pool = get_db_pool()
    await pool.initialize()
    logger.info("Database pool initialized")

    rows = await pool.fetch(_FETCH_SQL, list(_SOCIAL_SOURCES))
    logger.info("Found %d social media reviews to evaluate", len(rows))

    to_filter: list[str] = []
    kept = 0
    protected = 0

    for row in rows:
        review_dict = {
            "source": row["source"],
            "summary": row["summary"],
            "review_text": row["review_text"],
            "raw_metadata": (
                json.loads(row["raw_metadata"])
                if isinstance(row["raw_metadata"], str)
                else (row["raw_metadata"] or {})
            ),
        }

        rel_score, reason = score_relevance(review_dict, row["vendor_name"])

        if rel_score >= threshold:
            kept += 1
            if verbose:
                logger.info(
                    "KEEP  %.3f  %-12s %-15s %s",
                    rel_score, row["source"], row["vendor_name"], reason,
                )
            continue

        # Check enrichment protection
        enrichment = row["enrichment"]
        if isinstance(enrichment, str):
            try:
                enrichment = json.loads(enrichment)
            except (json.JSONDecodeError, TypeError):
                enrichment = None

        if _is_protected(enrichment):
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
        "Results: %d to filter, %d kept (score >= %.2f), %d protected (enrichment signal)",
        len(to_filter), kept, threshold, protected,
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
        default=0.35,
        help="Relevance score threshold (default: 0.35)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each review's score and reason",
    )
    args = parser.parse_args()

    result = await run_cleanup(
        threshold=args.threshold,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
