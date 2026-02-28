"""
Import Amazon product reviews into the product_reviews table.

Supports two input formats:
  - JSON: categorized_reviews.json (dict of category -> list of reviews)
  - JSONL: McAuley Amazon-Reviews-2023 format (one JSON object per line)

Format is auto-detected from file extension (.json vs .jsonl).

Usage:
    # Preview counts (no DB writes)
    python scripts/import_amazon_reviews.py --dry-run

    # Import categorized JSON (default)
    python scripts/import_amazon_reviews.py --file /path/to/reviews.json

    # Import a single category from JSON
    python scripts/import_amazon_reviews.py --file reviews.json --category gpu

    # Import JSONL dataset (e.g. beauty reviews)
    python scripts/import_amazon_reviews.py \
        --file /path/to/All_Beauty.jsonl \
        --source-category beauty \
        --min-text-length 50
"""

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
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


def make_dedup_key_jsonl(asin: str, user_id: str, timestamp: int) -> str:
    """SHA-256 of asin:user_id:timestamp for JSONL records (no review_id field)."""
    return hashlib.sha256(f"{asin}:{user_id}:{timestamp}".encode()).hexdigest()


def load_reviews(file_path: Path, category_filter: str | None = None) -> list[tuple[str, dict]]:
    """Load reviews from categorized JSON. Returns list of (category, review_dict) tuples."""
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


def load_reviews_jsonl(
    file_path: Path,
    source_category: str,
    min_text_length: int = 50,
) -> list[tuple[str, dict]]:
    """Stream reviews from JSONL (McAuley Amazon-Reviews-2023 format).

    Maps JSONL fields to our internal schema:
        title         -> summary
        text          -> review_text
        parent_asin   -> asin (falls back to asin)
        user_id       -> reviewer_id
        helpful_vote  -> helpful_votes
        timestamp(ms) -> reviewed_at
        rating        -> rating

    Filters out reviews with text shorter than min_text_length.
    """
    logger.info("Streaming JSONL reviews from %s (min_text_length=%d)", file_path, min_text_length)
    reviews = []
    total_lines = 0
    filtered_short = 0

    with open(file_path) as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = record.get("text", "")
            if len(text) < min_text_length:
                filtered_short += 1
                continue

            asin = record.get("parent_asin") or record.get("asin", "")
            user_id = record.get("user_id", "")
            timestamp = record.get("timestamp", 0)

            if not asin or not user_id:
                filtered_short += 1
                continue

            # Convert ms epoch to ISO datetime string
            reviewed_at = None
            if timestamp:
                try:
                    reviewed_at = datetime.fromtimestamp(
                        timestamp / 1000, tz=timezone.utc
                    ).isoformat()
                except (ValueError, OSError, OverflowError):
                    pass

            review = {
                "asin": asin,
                "reviewer_id": user_id,
                "rating": float(record.get("rating", 0)),
                "summary": record.get("title", ""),
                "review_text": text,
                "dedup_key": make_dedup_key_jsonl(asin, user_id, timestamp),
                "reviewed_at": reviewed_at,
                "matched_keywords": [],
                "hardware_category": [],
                "issue_types": [],
            }
            reviews.append((source_category, review))

            if total_lines % 100000 == 0:
                logger.info("  Read %dk lines, %dk kept so far...",
                            total_lines // 1000, len(reviews) // 1000)

    logger.info(
        "JSONL load complete: %d total lines, %d kept, %d filtered (short/invalid)",
        total_lines, len(reviews), filtered_short,
    )
    return reviews


async def import_reviews(reviews: list[tuple[str, dict]], dry_run: bool = False, is_jsonl: bool = False) -> dict:
    """Insert reviews into product_reviews table."""
    if dry_run:
        logger.info("DRY RUN: would import %d reviews", len(reviews))
        # Show rating breakdown for JSONL
        if is_jsonl:
            neg = sum(1 for _, r in reviews if r.get("rating", 0) <= 3.0)
            pos = len(reviews) - neg
            logger.info("  1-3 star (complaint classification): %d", neg)
            logger.info("  4-5 star (praise classification): %d", pos)
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
            if is_jsonl:
                # JSONL records already have fields mapped
                asin = review.get("asin", "")
                reviewer_id = review.get("reviewer_id", "")
                if not asin or not reviewer_id:
                    skipped += 1
                    continue
                dedup_key = review["dedup_key"]
                rows.append((
                    dedup_key,
                    asin,
                    float(review.get("rating", 0)),
                    review.get("summary", ""),
                    review.get("review_text", ""),
                    reviewer_id,
                    "amazon",
                    category,
                    review.get("matched_keywords", []),
                    review.get("hardware_category", []),
                    review.get("issue_types", []),
                ))
            else:
                # Original categorized JSON format
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


def detect_format(file_path: Path) -> str:
    """Auto-detect file format from extension."""
    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    return "json"


async def main():
    parser = argparse.ArgumentParser(description="Import Amazon reviews into product_reviews")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE, help="Path to reviews file")
    parser.add_argument("--category", type=str, default=None, help="Import only this category (JSON mode)")
    parser.add_argument("--source-category", type=str, default=None,
                        help="Source category label for JSONL imports (e.g. 'beauty')")
    parser.add_argument("--min-text-length", type=int, default=50,
                        help="Minimum review text length to import (JSONL mode, default 50)")
    parser.add_argument("--dry-run", action="store_true", help="Preview counts without writing")
    args = parser.parse_args()

    if not args.file.exists():
        logger.error("File not found: %s", args.file)
        sys.exit(1)

    fmt = detect_format(args.file)
    logger.info("Detected format: %s", fmt)

    if fmt == "jsonl":
        source_category = args.source_category
        if not source_category:
            # Derive from filename: All_Beauty.jsonl -> beauty
            stem = args.file.stem.lower()
            source_category = stem.replace("all_", "").replace("_", " ").strip()
            logger.info("Auto-derived source_category: '%s'", source_category)
        reviews = load_reviews_jsonl(args.file, source_category, args.min_text_length)
    else:
        reviews = load_reviews(args.file, args.category)

    # Print category breakdown
    cats: dict[str, int] = {}
    for cat, _ in reviews:
        cats[cat] = cats.get(cat, 0) + 1
    logger.info("Category breakdown:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        logger.info("  %-15s %6d", cat, count)
    logger.info("  %-15s %6d", "TOTAL", len(reviews))

    # Rating breakdown for JSONL
    if fmt == "jsonl":
        neg = sum(1 for _, r in reviews if r.get("rating", 0) <= 3.0)
        pos = len(reviews) - neg
        logger.info("Rating split:")
        logger.info("  1-3 star (complaint classification): %d", neg)
        logger.info("  4-5 star (praise classification): %d", pos)

    result = await import_reviews(reviews, dry_run=args.dry_run, is_jsonl=(fmt == "jsonl"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
