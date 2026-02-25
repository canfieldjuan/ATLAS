"""
Second-pass metadata matching: scan additional McAuley categories (2023 JSONL
+ 2018 gzipped JSON) for ASINs not found in the Electronics parquet.

Usage:
    python scripts/match_metadata_extra.py
"""

import asyncio
import gzip
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("match_metadata_extra")

CACHE = Path("/home/juan-canfield/Desktop/Atlas/data/hf_cache")
DATA_DIR = Path("/home/juan-canfield/Desktop/Atlas/data")

# 2023 JSONL categories to scan
CATEGORIES_2023 = [
    "Office_Products", "Software", "Video_Games",
    "Tools_and_Home_Improvement", "Unknown",
    "Cell_Phones_and_Accessories", "Industrial_and_Scientific",
]

# 2018 gzipped JSON files (different format: single-line JSON per line)
FILES_2018 = [
    DATA_DIR / "meta_Electronics_2018.json.gz",
]

UPSERT_SQL = """
INSERT INTO product_metadata (
    asin, title, brand, average_rating, rating_number,
    price, store, features, description, categories, details
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
ON CONFLICT (asin) DO NOTHING
"""


def extract_brand(details) -> str:
    if not details or not isinstance(details, dict):
        return ""
    return (
        details.get("Brand", "")
        or details.get("brand", "")
        or details.get("Manufacturer", "")
        or details.get("manufacturer", "")
        or ""
    )


def safe_float(val):
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def safe_int(val):
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def parse_2023_item(item: dict, asin: str) -> tuple:
    """Parse a 2023 format item into a DB row tuple."""
    details = item.get("details", {})
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except (json.JSONDecodeError, TypeError):
            details = {}

    brand = extract_brand(details) or item.get("brand", "")
    features = item.get("features") or []
    description = item.get("description") or []
    categories = item.get("categories") or []

    return (
        asin,
        item.get("title", ""),
        brand,
        safe_float(item.get("average_rating")),
        safe_int(item.get("rating_number")),
        safe_float(item.get("price")),
        item.get("store", ""),
        json.dumps(features if isinstance(features, list) else []),
        json.dumps(description if isinstance(description, list) else []),
        json.dumps(categories if isinstance(categories, list) else []),
        json.dumps(details if isinstance(details, dict) else {}),
    )


def parse_2018_item(item: dict, asin: str) -> tuple:
    """Parse a 2018 format item into a DB row tuple.

    2018 format differences:
    - 'asin' field (not parent_asin)
    - 'brand' as top-level field
    - 'price' as string like '$29.99'
    - 'category' list instead of 'categories'
    - 'feature' list instead of 'features'
    """
    brand = item.get("brand", "")
    price_str = item.get("price", "")
    price = None
    if isinstance(price_str, str):
        price_str = price_str.replace("$", "").replace(",", "").strip()
        if " - " in price_str:
            price_str = price_str.split(" - ")[0]  # take lower bound of range
        try:
            price = float(price_str) if price_str else None
        except ValueError:
            price = None
    elif isinstance(price_str, (int, float)):
        price = float(price_str)

    features = item.get("feature") or item.get("features") or []
    description = item.get("description") or []
    categories = item.get("category") or item.get("categories") or []
    details = item.get("tech1", {}) or item.get("details", {})
    if not isinstance(details, dict):
        details = {}

    # 2018 doesn't have rating aggregates at product level
    return (
        asin,
        item.get("title", ""),
        brand,
        None,  # no average_rating in 2018 metadata
        None,  # no rating_number in 2018 metadata
        price,
        "",  # no store in 2018
        json.dumps(features if isinstance(features, list) else []),
        json.dumps(description if isinstance(description, list) else []),
        json.dumps(categories if isinstance(categories, list) else []),
        json.dumps(details if isinstance(details, dict) else {}),
    )


async def main():
    start = time.monotonic()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from atlas_brain.storage.database import get_db_pool

    pool = get_db_pool()
    await pool.initialize()

    # Get missing ASINs
    rows = await pool.fetch("""
        SELECT DISTINCT pr.asin
        FROM product_reviews pr
        LEFT JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pm.asin IS NULL
    """)
    missing = {r["asin"] for r in rows}
    logger.info("Missing ASINs to search: %d", len(missing))

    all_rows = []

    # Scan 2023 JSONL categories
    from huggingface_hub import hf_hub_download

    for cat in CATEGORIES_2023:
        filename = f"raw/meta_categories/meta_{cat}.jsonl"
        try:
            path = hf_hub_download(
                "McAuley-Lab/Amazon-Reviews-2023", filename,
                repo_type="dataset", cache_dir=str(CACHE),
            )
        except Exception as e:
            logger.warning("Skip %s: %s", cat, e)
            continue

        found = 0
        with open(path) as f:
            for line in f:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                asin = item.get("parent_asin", "")
                if asin in missing:
                    all_rows.append(parse_2023_item(item, asin))
                    missing.discard(asin)
                    found += 1

        logger.info("  %s: %d matches (remaining: %d)", cat, found, len(missing))

    # Scan 2018 gzipped JSON files
    for filepath in FILES_2018:
        if not filepath.exists():
            logger.warning("Skip %s: not downloaded yet", filepath.name)
            continue

        logger.info("Scanning %s...", filepath.name)
        found = 0
        opener = gzip.open if filepath.suffix == ".gz" else open

        with opener(filepath, "rt", errors="replace") as f:
            for line in f:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                asin = item.get("asin", "")
                if asin in missing:
                    all_rows.append(parse_2018_item(item, asin))
                    missing.discard(asin)
                    found += 1

        logger.info("  %s: %d matches (remaining: %d)", filepath.name, found, len(missing))

    # Write to DB
    if all_rows:
        logger.info("Writing %d matches to product_metadata...", len(all_rows))
        async with pool.transaction() as conn:
            await conn.executemany(UPSERT_SQL, all_rows)
        logger.info("Done writing")

    elapsed = time.monotonic() - start

    # Final stats
    total_meta = await pool.fetchval("SELECT count(*) FROM product_metadata")
    total_asins = await pool.fetchval("SELECT count(DISTINCT asin) FROM product_reviews")
    covered_reviews = await pool.fetchval("""
        SELECT count(*) FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
    """)
    total_reviews = await pool.fetchval("SELECT count(*) FROM product_reviews")

    logger.info("=" * 60)
    logger.info("ASIN coverage: %d / %d (%.1f%%)", total_meta, total_asins, 100 * total_meta / total_asins)
    logger.info("Review coverage: %d / %d (%.1f%%)", covered_reviews, total_reviews, 100 * covered_reviews / total_reviews)
    logger.info("Still missing: %d ASINs", len(missing))
    logger.info("Total time: %.1fs", elapsed)


if __name__ == "__main__":
    asyncio.run(main())
