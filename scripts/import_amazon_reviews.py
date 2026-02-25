"""
Import pre-filtered Amazon Electronics reviews into the product_reviews table.

Reads categorized_reviews.json (dict of category -> list of reviews),
generates dedup_key = sha256(asin + review_id), and batch-inserts via
asyncpg with ON CONFLICT DO NOTHING. Idempotent.

Usage:
    # Preview counts (no DB writes)
    python scripts/import_amazon_reviews.py --dry-run

    # Live import (default source file)
    python scripts/import_amazon_reviews.py

    # Custom source file
    python scripts/import_amazon_reviews.py --file /path/to/reviews.json

    # Import a single category
    python scripts/import_amazon_reviews.py --category gpu
"""

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("import_amazon_reviews")

DEFAULT_FILE = Path("/home/juan-canfield/Desktop/web-ui/output/categorized_reviews.json")
BATCH_SIZE = 5000

INSERT_SQL = """
INSERT INTO product_reviews (
    dedup_key, asin, rating, summary, review_text, reviewer_id,
    source, source_category, matched_keywords, hardware_category, issue_types
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
ON CONFLICT (dedup_key) DO NOTHING
"""


def make_dedup_key(asin: str, review_id: str) -> str:
    """SHA-256 of asin + review_id for deterministic dedup."""
    return hashlib.sha256(f"{asin}:{review_id}".encode()).hexdigest()


def load_reviews(file_path: Path, category_filter: str | None = None) -> list[tuple[str, dict]]:
    """Load reviews from JSON. Returns list of (category, review_dict) tuples."""
    logger.info("Loading reviews from %s", file_path)
    with open(file_path) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict with category keys, got {type(data)}")

    reviews = []
    for category, items in data.items():
        if category_filter and category != category_filter:
            continue
        for item in items:
            reviews.append((category, item))

    return reviews


async def import_reviews(reviews: list[tuple[str, dict]], dry_run: bool = False) -> dict:
    """Insert reviews into product_reviews table."""
    if dry_run:
        logger.info("DRY RUN: would import %d reviews", len(reviews))
        return {"total": len(reviews), "inserted": 0, "skipped": 0}

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from atlas_brain.storage.database import get_db_pool

    pool = get_db_pool()
    await pool.initialize()
    logger.info("Database pool initialized")

    inserted = 0
    skipped = 0
    start = time.monotonic()

    for batch_start in range(0, len(reviews), BATCH_SIZE):
        batch = reviews[batch_start:batch_start + BATCH_SIZE]
        rows = []
        for category, review in batch:
            asin = review.get("asin", "")
            review_id = review.get("review_id", "")
            if not asin or not review_id:
                skipped += 1
                continue

            dedup_key = make_dedup_key(asin, review_id)
            hw_cats = review.get("hardware_category", [])
            if isinstance(hw_cats, str):
                hw_cats = [hw_cats]
            issue = review.get("issue_types", [])
            if isinstance(issue, str):
                issue = [issue]
            keywords = review.get("matched_keywords", [])
            if isinstance(keywords, str):
                keywords = [keywords]

            rows.append((
                dedup_key,
                asin,
                float(review.get("rating", 0)),
                review.get("summary", ""),
                review.get("review_text", ""),
                review_id,
                "amazon",
                category,
                keywords,
                hw_cats,
                issue,
            ))

        if rows:
            async with pool.transaction() as conn:
                await conn.executemany(INSERT_SQL, rows)
            inserted += len(rows)

        elapsed = time.monotonic() - start
        logger.info(
            "Progress: %d / %d (%.1fs elapsed)",
            min(batch_start + BATCH_SIZE, len(reviews)),
            len(reviews),
            elapsed,
        )

    elapsed = time.monotonic() - start
    logger.info(
        "Import complete: %d rows sent, %d skipped in %.1fs",
        inserted, skipped, elapsed,
    )

    # Verify
    row = await pool.fetchrow("SELECT count(*) as cnt FROM product_reviews")
    logger.info("Total rows in product_reviews: %d", row["cnt"])

    return {"total": len(reviews), "inserted": inserted, "skipped": skipped}


async def main():
    parser = argparse.ArgumentParser(description="Import Amazon reviews into product_reviews")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE, help="Path to categorized_reviews.json")
    parser.add_argument("--category", type=str, default=None, help="Import only this category")
    parser.add_argument("--dry-run", action="store_true", help="Preview counts without writing")
    args = parser.parse_args()

    if not args.file.exists():
        logger.error("File not found: %s", args.file)
        sys.exit(1)

    reviews = load_reviews(args.file, args.category)

    # Print category breakdown
    cats: dict[str, int] = {}
    for cat, _ in reviews:
        cats[cat] = cats.get(cat, 0) + 1
    logger.info("Category breakdown:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        logger.info("  %-15s %6d", cat, count)
    logger.info("  %-15s %6d", "TOTAL", len(reviews))

    result = await import_reviews(reviews, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
