"""
Match our product_reviews ASINs against the McAuley Amazon-Reviews-2023
Electronics metadata (Parquet shards on Hugging Face).

Downloads all 10 shards, scans for matching parent_asin values, and writes
matches to the product_metadata table in PostgreSQL.

Usage:
    python scripts/match_product_metadata.py
    python scripts/match_product_metadata.py --dry-run   # preview only
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("match_product_metadata")

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
SHARD_PATTERN = "raw_meta_Electronics/full-{:05d}-of-00010.parquet"
NUM_SHARDS = 10
CACHE_DIR = Path("/home/juan-canfield/Desktop/Atlas/data/hf_cache")

# Columns we need from the parquet files
COLUMNS = [
    "parent_asin", "title", "average_rating", "rating_number",
    "price", "store", "features", "description", "categories", "details",
]

UPSERT_SQL = """
INSERT INTO product_metadata (
    asin, title, brand, average_rating, rating_number,
    price, store, features, description, categories, details
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
ON CONFLICT (asin) DO UPDATE SET
    title = EXCLUDED.title,
    brand = EXCLUDED.brand,
    average_rating = EXCLUDED.average_rating,
    rating_number = EXCLUDED.rating_number,
    price = EXCLUDED.price,
    store = EXCLUDED.store,
    features = EXCLUDED.features,
    description = EXCLUDED.description,
    categories = EXCLUDED.categories,
    details = EXCLUDED.details
"""


def extract_brand(details) -> str:
    """Pull brand/manufacturer from the details dict."""
    if not details:
        return ""
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except (json.JSONDecodeError, TypeError):
            return ""
    if not isinstance(details, dict):
        return ""
    return (
        details.get("Brand", "")
        or details.get("brand", "")
        or details.get("Manufacturer", "")
        or details.get("manufacturer", "")
        or ""
    )


def download_shard(shard_idx: int) -> Path:
    """Download a single parquet shard, return local path."""
    filename = SHARD_PATTERN.format(shard_idx)
    path = hf_hub_download(
        REPO_ID, filename, repo_type="dataset", cache_dir=str(CACHE_DIR)
    )
    return Path(path)


def scan_shard(shard_path: Path, target_asins: set[str]) -> list[dict]:
    """Scan a parquet shard for matching ASINs. Returns list of metadata dicts."""
    table = pq.read_table(shard_path, columns=COLUMNS)
    matches = []

    parent_asins = table.column("parent_asin").to_pylist()

    # Fast set intersection to find row indices
    for i, asin in enumerate(parent_asins):
        if asin in target_asins:
            row = {}
            for col in COLUMNS:
                row[col] = table.column(col)[i].as_py()
            matches.append(row)

    return matches


async def write_to_db(matches: list[dict]) -> int:
    """Write matched metadata to product_metadata table."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from atlas_brain.storage.database import get_db_pool

    pool = get_db_pool()
    await pool.initialize()

    # Create table if not exists
    await pool.execute("""
        CREATE TABLE IF NOT EXISTS product_metadata (
            asin            TEXT PRIMARY KEY,
            title           TEXT,
            brand           TEXT,
            average_rating  REAL,
            rating_number   INT,
            price           REAL,
            store           TEXT,
            features        JSONB,
            description     JSONB,
            categories      JSONB,
            details         JSONB,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    rows = []
    for m in matches:
        features = m.get("features")
        if isinstance(features, list):
            features = json.dumps(features)
        elif features is None:
            features = "[]"
        else:
            features = json.dumps(features)

        description = m.get("description")
        if isinstance(description, list):
            description = json.dumps(description)
        elif description is None:
            description = "[]"
        else:
            description = json.dumps(description)

        categories = m.get("categories")
        if isinstance(categories, list):
            categories = json.dumps(categories)
        elif categories is None:
            categories = "[]"
        else:
            categories = json.dumps(categories)

        details = m.get("details")
        if isinstance(details, dict):
            details_json = json.dumps(details)
        elif isinstance(details, str):
            details_json = details
        elif details is None:
            details_json = "{}"
        else:
            details_json = json.dumps(details)

        brand = extract_brand(m.get("details"))

        # Coerce price to float (comes as string from parquet sometimes)
        raw_price = m.get("price")
        try:
            price = float(raw_price) if raw_price is not None else None
        except (ValueError, TypeError):
            price = None

        rows.append((
            m["parent_asin"],
            m.get("title", ""),
            brand,
            m.get("average_rating"),
            m.get("rating_number"),
            price,
            m.get("store", ""),
            features,
            description,
            categories,
            details_json,
        ))

    if rows:
        async with pool.transaction() as conn:
            await conn.executemany(UPSERT_SQL, rows)

    return len(rows)


async def get_our_asins() -> set[str]:
    """Fetch distinct ASINs from product_reviews."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from atlas_brain.storage.database import get_db_pool

    pool = get_db_pool()
    await pool.initialize()

    rows = await pool.fetch("SELECT DISTINCT asin FROM product_reviews")
    return {r["asin"] for r in rows}


async def main():
    parser = argparse.ArgumentParser(description="Match product ASINs to Amazon metadata")
    parser.add_argument("--dry-run", action="store_true", help="Preview matches without writing to DB")
    args = parser.parse_args()

    start = time.monotonic()

    # Step 1: Get our ASINs
    logger.info("Fetching distinct ASINs from product_reviews...")
    our_asins = await get_our_asins()
    logger.info("Found %d distinct ASINs to match", len(our_asins))

    # Step 2: Download and scan all shards
    all_matches = []
    matched_asins = set()

    for shard_idx in range(NUM_SHARDS):
        logger.info("Downloading shard %d/%d...", shard_idx + 1, NUM_SHARDS)
        shard_path = download_shard(shard_idx)
        shard_size = os.path.getsize(shard_path) / 1024 / 1024

        remaining = our_asins - matched_asins
        if not remaining:
            logger.info("All ASINs matched! Skipping remaining shards.")
            break

        logger.info("Scanning shard %d (%.1f MB) for %d remaining ASINs...",
                     shard_idx, shard_size, len(remaining))
        matches = scan_shard(shard_path, remaining)

        for m in matches:
            matched_asins.add(m["parent_asin"])
        all_matches.extend(matches)

        logger.info("  Found %d matches (total: %d/%d = %.1f%%)",
                     len(matches), len(matched_asins), len(our_asins),
                     100 * len(matched_asins) / len(our_asins))

    elapsed = time.monotonic() - start

    # Step 3: Report
    missing = our_asins - matched_asins
    logger.info("=" * 60)
    logger.info("RESULTS: %d matched, %d missing (%.1f%% hit rate) in %.1fs",
                len(matched_asins), len(missing),
                100 * len(matched_asins) / len(our_asins), elapsed)

    if all_matches:
        # Show some sample matches
        logger.info("Sample matches:")
        for m in all_matches[:10]:
            brand = extract_brand(m.get("details"))
            logger.info("  %s: %s [%s] (%.1f stars, %s ratings, $%s)",
                        m["parent_asin"],
                        m.get("title", "?")[:60],
                        brand or "?",
                        m.get("average_rating", 0),
                        m.get("rating_number", "?"),
                        m.get("price", "?"))

    if missing:
        logger.info("Sample missing ASINs: %s", list(missing)[:20])

    # Step 4: Write to DB
    if args.dry_run:
        logger.info("DRY RUN: would write %d rows to product_metadata", len(all_matches))
    else:
        logger.info("Writing %d matches to product_metadata table...", len(all_matches))
        written = await write_to_db(all_matches)
        logger.info("Wrote %d rows to product_metadata", written)

    total_elapsed = time.monotonic() - start
    logger.info("Total time: %.1fs", total_elapsed)


if __name__ == "__main__":
    asyncio.run(main())
