"""
Import B2B software reviews into the b2b_reviews table.

Reads a JSON array of reviews (source-agnostic format), generates
dedup_key = sha256(source:vendor_name:source_review_id) or
sha256(source:vendor_name:reviewer_name:reviewed_at), and batch-inserts
via asyncpg with ON CONFLICT DO NOTHING.  Idempotent.

Usage:
    # Preview counts (no DB writes)
    python scripts/import_b2b_reviews.py --file reviews.json --dry-run

    # Live import
    python scripts/import_b2b_reviews.py --file reviews.json

    # Custom batch size
    python scripts/import_b2b_reviews.py --file reviews.json --batch-size 1000
"""

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("import_b2b_reviews")

BATCH_SIZE = 1000

INSERT_SQL = """
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


def make_dedup_key(source: str, vendor_name: str, source_review_id: str | None,
                   reviewer_name: str | None, reviewed_at: str | None) -> str:
    """Generate deterministic dedup key for a review."""
    if source_review_id:
        raw = f"{source}:{vendor_name}:{source_review_id}"
    else:
        raw = f"{source}:{vendor_name}:{reviewer_name or ''}:{reviewed_at or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()


def load_reviews(file_path: Path) -> list[dict]:
    """Load reviews from JSON file. Expects a JSON array."""
    logger.info("Loading reviews from %s", file_path)
    with open(file_path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "reviews" in data:
        return data["reviews"]
    else:
        raise ValueError(f"Expected JSON array or object with 'reviews' key, got {type(data)}")


async def import_reviews(reviews: list[dict], dry_run: bool = False,
                         batch_size: int = BATCH_SIZE,
                         allow_unknown_sources: bool = False) -> dict:
    """Insert reviews into b2b_reviews table."""
    if dry_run:
        logger.info("DRY RUN: would import %d reviews", len(reviews))
        vendors: dict[str, int] = {}
        for r in reviews:
            v = r.get("vendor_name", "unknown")
            vendors[v] = vendors.get(v, 0) + 1
        for v, c in sorted(vendors.items(), key=lambda x: -x[1]):
            logger.info("  %-30s %6d", v, c)
        return {"total": len(reviews), "imported": 0, "duplicates": 0}

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from atlas_brain.storage.database import get_db_pool
    from atlas_brain.services.b2b.reviewer_identity import sanitize_reviewer_title
    from atlas_brain.services.vendor_registry import resolve_vendor_name
    from atlas_brain.services.scraping.sources import ReviewSource

    valid_sources = {s.value for s in ReviewSource}

    pool = get_db_pool()
    await pool.initialize()
    logger.info("Database pool initialized")

    imported = 0
    skipped = 0
    batch_id = f"import_{int(time.time())}"
    start = time.monotonic()

    for batch_start in range(0, len(reviews), batch_size):
        batch = reviews[batch_start:batch_start + batch_size]
        rows = []
        for review in batch:
            source = review.get("source", "manual")
            vendor_name = review.get("vendor_name", "")
            review_text = review.get("review_text", "")

            if not vendor_name or not review_text:
                skipped += 1
                continue

            if source not in valid_sources:
                if allow_unknown_sources:
                    logger.warning("Unknown source '%s' for vendor '%s', importing anyway", source, vendor_name)
                else:
                    logger.warning("Unknown source '%s' for vendor '%s', skipping (use --allow-unknown-sources to override)", source, vendor_name)
                    skipped += 1
                    continue

            # Resolve to canonical vendor name BEFORE dedup key so both use canonical form
            canonical_vendor = await resolve_vendor_name(vendor_name)

            dedup_key = make_dedup_key(
                source, canonical_vendor,
                review.get("source_review_id"),
                review.get("reviewer_name"),
                review.get("reviewed_at"),
            )

            reviewed_at = review.get("reviewed_at")
            if reviewed_at and isinstance(reviewed_at, str):
                try:
                    reviewed_at = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
                except ValueError:
                    reviewed_at = None

            metadata = review.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            rows.append((
                dedup_key,
                source,
                review.get("source_url"),
                review.get("source_review_id"),
                canonical_vendor,
                review.get("product_name"),
                review.get("product_category"),
                float(review["rating"]) if review.get("rating") is not None else None,
                int(review.get("rating_max", 5)),
                review.get("summary"),
                review_text,
                review.get("pros"),
                review.get("cons"),
                review.get("reviewer_name"),
                sanitize_reviewer_title(review.get("reviewer_title")),
                review.get("reviewer_company"),
                review.get("company_size_raw"),
                review.get("reviewer_industry"),
                reviewed_at,
                batch_id,
                json.dumps(metadata),
                None,  # parser_version: N/A for file imports
            ))

        if rows:
            async with pool.transaction() as conn:
                await conn.executemany(INSERT_SQL, rows)
            imported += len(rows)

        elapsed = time.monotonic() - start
        logger.info(
            "Progress: %d / %d (%.1fs elapsed)",
            min(batch_start + batch_size, len(reviews)),
            len(reviews),
            elapsed,
        )

    elapsed = time.monotonic() - start
    logger.info(
        "Import complete: %d rows sent, %d skipped in %.1fs",
        imported, skipped, elapsed,
    )

    row = await pool.fetchrow("SELECT count(*) as cnt FROM b2b_reviews")
    logger.info("Total rows in b2b_reviews: %d", row["cnt"])

    return {"total": len(reviews), "imported": imported, "duplicates": skipped, "batch_id": batch_id}


async def main():
    parser = argparse.ArgumentParser(description="Import B2B software reviews into b2b_reviews")
    parser.add_argument("--file", type=Path, required=True, help="Path to reviews JSON file")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Insert batch size")
    parser.add_argument("--dry-run", action="store_true", help="Preview counts without writing")
    parser.add_argument("--allow-unknown-sources", action="store_true",
                        help="Import reviews with non-canonical source values instead of skipping")
    args = parser.parse_args()

    if not args.file.exists():
        logger.error("File not found: %s", args.file)
        sys.exit(1)

    reviews = load_reviews(args.file)

    # Print vendor breakdown
    vendors: dict[str, int] = {}
    for r in reviews:
        v = r.get("vendor_name", "unknown")
        vendors[v] = vendors.get(v, 0) + 1
    logger.info("Vendor breakdown:")
    for v, c in sorted(vendors.items(), key=lambda x: -x[1]):
        logger.info("  %-30s %6d", v, c)
    logger.info("  %-30s %6d", "TOTAL", len(reviews))

    result = await import_reviews(
        reviews, dry_run=args.dry_run, batch_size=args.batch_size,
        allow_unknown_sources=args.allow_unknown_sources,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
