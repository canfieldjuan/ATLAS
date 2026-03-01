#!/usr/bin/env python3
"""
Brand-first sampling from McAuley Amazon-Reviews-2023 JSONL files.

Strategy: spread coverage across top brands, then go deep per brand.
  Step 1: Scan metadata  -> build {ASIN: brand} mapping
  Step 2: Scan reviews   -> count reviews per ASIN
  Step 3: Join & select  -> top N brands by volume, top M ASINs per brand
  Step 4: Import         -> stream reviews for selected ASINs into DB
  Step 5: Metadata       -> upsert product_metadata for selected ASINs

Usage:
    # Top 20 brands, 10 ASINs each from Cell Phones (dry run)
    python scripts/sample_and_import.py --category cell_phones --dry-run

    # Top 20 brands, 10 ASINs each from all downloaded categories
    python scripts/sample_and_import.py --all

    # Wider: top 30 brands, 20 ASINs each
    python scripts/sample_and_import.py --category electronics \
        --top-brands 30 --asins-per-brand 20

    # Deeper pass: 50 ASINs per brand (run after initial pass)
    python scripts/sample_and_import.py --all --asins-per-brand 50
"""

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("sample_and_import")

DATA_DIR = Path("/home/juan-canfield/Desktop/Atlas/data/amazon_2023")
REVIEWS_DIR = DATA_DIR / "reviews" / "raw" / "review_categories"
META_DIR = DATA_DIR / "metadata" / "raw" / "meta_categories"

BATCH_SIZE = 5000
MIN_TEXT_LENGTH = 50

CATEGORIES = {
    "cell_phones": {
        "review_file": "Cell_Phones_and_Accessories.jsonl",
        "meta_file": "meta_Cell_Phones_and_Accessories.jsonl",
        "source_category": "cell phones",
    },
    "sports": {
        "review_file": "Sports_and_Outdoors.jsonl",
        "meta_file": "meta_Sports_and_Outdoors.jsonl",
        "source_category": "sports outdoors",
    },
    "tools": {
        "review_file": "Tools_and_Home_Improvement.jsonl",
        "meta_file": "meta_Tools_and_Home_Improvement.jsonl",
        "source_category": "tools home improvement",
    },
    "electronics": {
        "review_file": "Electronics.jsonl",
        "meta_file": "meta_Electronics.jsonl",
        "source_category": "electronics",
    },
    "home_kitchen": {
        "review_file": "Home_and_Kitchen.jsonl",
        "meta_file": "meta_Home_and_Kitchen.jsonl",
        "source_category": "home kitchen",
    },
}

INSERT_SQL = """
INSERT INTO product_reviews (
    dedup_key, asin, rating, summary, review_text, reviewer_id,
    source, source_category, matched_keywords, hardware_category,
    issue_types, reviewed_at
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
ON CONFLICT (dedup_key) DO NOTHING
"""

META_UPSERT_SQL = """
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


def make_dedup_key(asin: str, user_id: str, timestamp: int) -> str:
    return hashlib.sha256(f"{asin}:{user_id}:{timestamp}".encode()).hexdigest()


def extract_brand(details) -> str:
    if not details:
        return ""
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except (json.JSONDecodeError, TypeError):
            return ""
    return (
        details.get("Brand", "")
        or details.get("brand", "")
        or details.get("Manufacturer", "")
        or details.get("manufacturer", "")
        or ""
    ).strip()


def normalize_brand(brand: str) -> str:
    """Normalize brand names for grouping (lowercase, strip whitespace)."""
    return brand.strip().lower()


# ---------------------------------------------------------------------------
# Step 1: Scan metadata -> {ASIN: brand}
# ---------------------------------------------------------------------------

def build_asin_brand_map(meta_path: Path) -> dict[str, str]:
    """Stream metadata JSONL, return {asin: brand_name} for ASINs with a brand."""
    logger.info("Step 1: Building ASIN->brand map from %s", meta_path.name)
    asin_brand: dict[str, str] = {}
    total = 0
    no_brand = 0
    t0 = time.monotonic()

    with open(meta_path) as f:
        for line in f:
            total += 1
            if total % 500_000 == 0:
                elapsed = time.monotonic() - t0
                logger.info(
                    "  %dk metadata lines, %d with brand (%.0fs)",
                    total // 1000, len(asin_brand), elapsed,
                )
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            asin = record.get("parent_asin") or record.get("asin", "")
            if not asin:
                continue

            brand = extract_brand(record.get("details"))
            if not brand:
                no_brand += 1
                continue

            asin_brand[asin] = brand

    elapsed = time.monotonic() - t0
    logger.info(
        "Step 1 done: %d products scanned, %d have brand, %d missing brand (%.0fs)",
        total, len(asin_brand), no_brand, elapsed,
    )
    return asin_brand


# ---------------------------------------------------------------------------
# Step 2: Count reviews per ASIN
# ---------------------------------------------------------------------------

def count_reviews_per_asin(review_path: Path) -> Counter:
    """Stream review JSONL, count reviews per parent_asin."""
    logger.info("Step 2: Counting reviews per ASIN in %s", review_path.name)
    counts: Counter = Counter()
    total = 0
    t0 = time.monotonic()

    with open(review_path) as f:
        for line in f:
            total += 1
            if total % 1_000_000 == 0:
                elapsed = time.monotonic() - t0
                logger.info(
                    "  %dM lines, %d unique ASINs (%.0fs)",
                    total // 1_000_000, len(counts), elapsed,
                )
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            asin = record.get("parent_asin") or record.get("asin", "")
            if asin:
                counts[asin] += 1

    elapsed = time.monotonic() - t0
    logger.info(
        "Step 2 done: %d lines, %d unique ASINs in %.0fs",
        total, len(counts), elapsed,
    )
    return counts


# ---------------------------------------------------------------------------
# Step 3: Brand-first selection
# ---------------------------------------------------------------------------

def select_by_brand(
    asin_counts: Counter,
    asin_brand: dict[str, str],
    top_brands: int,
    asins_per_brand: int,
    min_reviews: int,
) -> tuple[set[str], dict]:
    """
    Select ASINs using brand-first strategy:
      1. Group ASINs by brand (using metadata)
      2. Rank brands by total review volume
      3. Pick top N brands
      4. Within each brand, pick top M ASINs by review count
    """
    # Group ASINs by normalized brand
    brand_asins: dict[str, list[tuple[str, int]]] = defaultdict(list)
    unbranded = 0

    for asin, count in asin_counts.items():
        if count < min_reviews:
            continue
        brand = asin_brand.get(asin)
        if not brand:
            unbranded += 1
            continue
        brand_asins[normalize_brand(brand)].append((asin, count))

    # Rank brands by total review volume
    brand_totals = {
        brand: sum(c for _, c in asins)
        for brand, asins in brand_asins.items()
    }
    ranked_brands = sorted(brand_totals.items(), key=lambda x: -x[1])[:top_brands]

    # For display, get the original-case brand name from the first ASIN
    brand_display: dict[str, str] = {}
    for norm_brand, _ in ranked_brands:
        for asin, _ in brand_asins[norm_brand]:
            original = asin_brand.get(asin, norm_brand)
            brand_display[norm_brand] = original
            break

    # Select top M ASINs per brand
    selected: set[str] = set()
    brand_details: list[dict] = []

    for norm_brand, total_vol in ranked_brands:
        asins_sorted = sorted(brand_asins[norm_brand], key=lambda x: -x[1])
        picked = asins_sorted[:asins_per_brand]
        for asin, _ in picked:
            selected.add(asin)

        brand_details.append({
            "brand": brand_display[norm_brand],
            "total_reviews": total_vol,
            "asins_available": len(asins_sorted),
            "asins_selected": len(picked),
            "reviews_selected": sum(c for _, c in picked),
            "top_asin": picked[0] if picked else None,
        })

    total_reviews = sum(bd["reviews_selected"] for bd in brand_details)

    stats = {
        "total_unique_asins": len(asin_counts),
        "asins_with_brand": len(asin_counts) - unbranded,
        "unbranded_asins": unbranded,
        "total_brands": len(brand_asins),
        "selected_brands": len(ranked_brands),
        "selected_asins": len(selected),
        "estimated_reviews": total_reviews,
        "brand_details": brand_details,
    }
    return selected, stats


# ---------------------------------------------------------------------------
# Step 4: Stream reviews for selected ASINs, batch insert to DB
# ---------------------------------------------------------------------------

async def import_sampled_reviews(
    review_path: Path,
    selected_asins: set[str],
    source_category: str,
    dry_run: bool = False,
) -> dict:
    """Stream JSONL, insert only reviews for selected ASINs."""
    logger.info(
        "Step 4: Importing reviews for %d ASINs from %s",
        len(selected_asins), review_path.name,
    )

    pool = None
    if not dry_run:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from atlas_brain.storage.database import get_db_pool
        pool = get_db_pool()
        await pool.initialize()

    total_lines = 0
    kept = 0
    inserted = 0
    filtered_short = 0
    batch: list[tuple] = []
    t0 = time.monotonic()

    with open(review_path) as f:
        for line in f:
            total_lines += 1
            if total_lines % 1_000_000 == 0:
                elapsed = time.monotonic() - t0
                rate = kept / elapsed if elapsed > 0 else 0
                logger.info(
                    "  %dM lines | %dk kept | %dk inserted | %.0f/s | %.0fs",
                    total_lines // 1_000_000,
                    kept // 1000,
                    inserted // 1000,
                    rate,
                    elapsed,
                )

            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            asin = record.get("parent_asin") or record.get("asin", "")
            if asin not in selected_asins:
                continue

            text = record.get("text", "")
            if len(text) < MIN_TEXT_LENGTH:
                filtered_short += 1
                continue

            user_id = record.get("user_id", "")
            if not user_id:
                filtered_short += 1
                continue

            timestamp = record.get("timestamp", 0)
            reviewed_at = None
            if timestamp:
                try:
                    reviewed_at = datetime.fromtimestamp(
                        timestamp / 1000, tz=timezone.utc
                    )
                except (ValueError, OSError, OverflowError):
                    pass

            kept += 1
            dedup_key = make_dedup_key(asin, user_id, timestamp)
            batch.append((
                dedup_key,
                asin,
                float(record.get("rating", 0)),
                record.get("title", ""),
                text,
                user_id,
                "amazon",
                source_category,
                [],    # matched_keywords
                [],    # hardware_category
                [],    # issue_types
                reviewed_at,
            ))

            if len(batch) >= BATCH_SIZE:
                if pool:
                    async with pool.transaction() as conn:
                        await conn.executemany(INSERT_SQL, batch)
                    inserted += len(batch)
                batch.clear()

    # Flush remaining
    if batch and pool:
        async with pool.transaction() as conn:
            await conn.executemany(INSERT_SQL, batch)
        inserted += len(batch)
    elif batch and dry_run:
        kept = kept  # already counted

    elapsed = time.monotonic() - t0
    logger.info(
        "Step 4 done: %d lines, %d kept, %d inserted, %d filtered in %.0fs",
        total_lines, kept, inserted, filtered_short, elapsed,
    )

    if pool:
        row = await pool.fetchrow("SELECT count(*) as cnt FROM product_reviews")
        logger.info("Total rows in product_reviews: %d", row["cnt"])

    return {
        "total_lines": total_lines,
        "kept": kept,
        "inserted": inserted,
        "filtered_short": filtered_short,
        "source_category": source_category,
    }


# ---------------------------------------------------------------------------
# Step 5: Metadata upsert for selected ASINs
# ---------------------------------------------------------------------------

async def upsert_metadata(
    meta_path: Path,
    target_asins: set[str],
    dry_run: bool = False,
) -> dict:
    """Stream meta JSONL, upsert product_metadata for target ASINs."""
    if not meta_path.exists():
        logger.warning("Metadata file not found: %s", meta_path)
        return {"matched": 0, "scanned": 0}

    logger.info("Step 5: Upserting metadata for %d ASINs from %s",
                len(target_asins), meta_path.name)

    pool = None
    if not dry_run:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from atlas_brain.storage.database import get_db_pool
        pool = get_db_pool()
        await pool.initialize()

    scanned = 0
    matched = 0
    batch: list[tuple] = []
    t0 = time.monotonic()

    with open(meta_path) as f:
        for line in f:
            scanned += 1
            if scanned % 500_000 == 0:
                logger.info(
                    "  Metadata: %dk scanned, %d matched (%.0fs)",
                    scanned // 1000, matched, time.monotonic() - t0,
                )

            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            asin = record.get("parent_asin") or record.get("asin", "")
            if asin not in target_asins:
                continue

            matched += 1
            brand = extract_brand(record.get("details"))

            # Parse price
            price_raw = record.get("price")
            price = None
            if price_raw is not None:
                if isinstance(price_raw, (int, float)):
                    price = float(price_raw)
                elif isinstance(price_raw, str):
                    cleaned = price_raw.replace("$", "").replace(",", "").strip()
                    try:
                        price = float(cleaned)
                    except ValueError:
                        pass

            features = record.get("features")
            description = record.get("description")
            categories = record.get("categories")
            details = record.get("details")

            batch.append((
                asin,
                record.get("title", ""),
                brand,
                record.get("average_rating"),
                record.get("rating_number"),
                price,
                record.get("store", ""),
                json.dumps(features) if features else None,
                json.dumps(description) if description else None,
                json.dumps(categories) if categories else None,
                json.dumps(details) if details else None,
            ))

            if len(batch) >= BATCH_SIZE:
                if pool:
                    async with pool.transaction() as conn:
                        await conn.executemany(META_UPSERT_SQL, batch)
                batch.clear()

    if batch and pool:
        async with pool.transaction() as conn:
            await conn.executemany(META_UPSERT_SQL, batch)

    elapsed = time.monotonic() - t0
    logger.info(
        "Step 5 done: %d scanned, %d matched in %.0fs",
        scanned, matched, elapsed,
    )
    return {"scanned": scanned, "matched": matched}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def process_category(
    cat_key: str,
    top_brands: int,
    asins_per_brand: int,
    min_reviews: int,
    dry_run: bool,
    skip_metadata: bool,
) -> dict:
    """Full brand-first pipeline for one category."""
    cat = CATEGORIES[cat_key]
    review_path = REVIEWS_DIR / cat["review_file"]
    meta_path = META_DIR / cat["meta_file"]

    if not review_path.exists():
        logger.error("Review file not found: %s (download may still be in progress)",
                     review_path)
        return {"category": cat_key, "error": f"File not found: {review_path}"}

    if not meta_path.exists():
        logger.error("Metadata file not found: %s (download may still be in progress)",
                     meta_path)
        return {"category": cat_key, "error": f"Metadata not found: {meta_path}"}

    logger.info("")
    logger.info("=" * 70)
    logger.info("CATEGORY: %s", cat_key.upper())
    logger.info("  Strategy: top %d brands x %d ASINs/brand, min %d reviews/ASIN",
                top_brands, asins_per_brand, min_reviews)
    logger.info("=" * 70)

    # Step 1: ASIN -> brand from metadata
    asin_brand = build_asin_brand_map(meta_path)

    # Step 2: Review counts per ASIN
    asin_counts = count_reviews_per_asin(review_path)

    # Step 3: Brand-first selection
    selected, stats = select_by_brand(
        asin_counts, asin_brand, top_brands, asins_per_brand, min_reviews,
    )

    # Display brand breakdown
    logger.info("")
    logger.info("Brand selection breakdown:")
    logger.info("  %-30s %8s %8s %10s", "Brand", "ASINs", "Selected", "Reviews")
    logger.info("  " + "-" * 60)
    for bd in stats["brand_details"]:
        logger.info(
            "  %-30s %8d %8d %10d",
            bd["brand"][:30], bd["asins_available"],
            bd["asins_selected"], bd["reviews_selected"],
        )
    logger.info("  " + "-" * 60)
    logger.info(
        "  %-30s %8s %8d %10d",
        "TOTAL", "", stats["selected_asins"], stats["estimated_reviews"],
    )
    logger.info("")
    logger.info("  Total unique ASINs in file: %d", stats["total_unique_asins"])
    logger.info("  ASINs with brand metadata: %d", stats["asins_with_brand"])
    logger.info("  ASINs without brand (skipped): %d", stats["unbranded_asins"])
    logger.info("  Total brands found: %d", stats["total_brands"])

    if dry_run:
        logger.info("")
        logger.info("[DRY RUN] Would import ~%d reviews across %d ASINs from %d brands",
                     stats["estimated_reviews"], stats["selected_asins"],
                     stats["selected_brands"])
        return {"category": cat_key, "dry_run": True, **stats}

    # Step 4: Import reviews
    import_result = await import_sampled_reviews(
        review_path, selected, cat["source_category"], dry_run,
    )

    # Step 5: Upsert metadata
    meta_result = {"skipped": True}
    if not skip_metadata:
        meta_result = await upsert_metadata(meta_path, selected, dry_run)

    return {
        "category": cat_key,
        "selection": stats,
        "import": import_result,
        "metadata": meta_result,
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Brand-first sampling from McAuley Amazon-Reviews-2023",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        choices=list(CATEGORIES.keys()),
        help="Single category to process",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all categories with downloaded files",
    )
    parser.add_argument(
        "--top-brands", type=int, default=20,
        help="Number of top brands per category (default: 20)",
    )
    parser.add_argument(
        "--asins-per-brand", type=int, default=10,
        help="Top ASINs to select per brand (default: 10)",
    )
    parser.add_argument(
        "--min-reviews", type=int, default=50,
        help="Minimum reviews per ASIN to be eligible (default: 50)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview brand/ASIN selection without importing",
    )
    parser.add_argument(
        "--skip-metadata", action="store_true",
        help="Skip metadata upsert step",
    )
    args = parser.parse_args()

    if not args.category and not args.all:
        parser.error("Specify --category <name> or --all")

    if args.all:
        cats = list(CATEGORIES.keys())
    else:
        cats = [args.category]

    results = []
    for cat_key in cats:
        result = await process_category(
            cat_key, args.top_brands, args.asins_per_brand,
            args.min_reviews, args.dry_run, args.skip_metadata,
        )
        results.append(result)

    # Grand summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("GRAND SUMMARY")
    logger.info("=" * 70)
    logger.info("  %-20s %6s %6s %10s", "Category", "Brands", "ASINs", "Reviews")
    logger.info("  " + "-" * 50)

    grand_brands = 0
    grand_asins = 0
    grand_reviews = 0
    for r in results:
        if "error" in r:
            logger.info("  %-20s SKIPPED: %s", r.get("category", "?"), r["error"])
            continue
        sel = r.get("selection", r)
        imp = r.get("import", {})
        cat = r.get("category", "?")
        n_brands = sel.get("selected_brands", 0)
        n_asins = sel.get("selected_asins", 0)
        n_reviews = imp.get("inserted", sel.get("estimated_reviews", 0))
        grand_brands += n_brands
        grand_asins += n_asins
        grand_reviews += n_reviews
        logger.info("  %-20s %6d %6d %10d", cat, n_brands, n_asins, n_reviews)

    logger.info("  " + "-" * 50)
    logger.info("  %-20s %6d %6d %10d", "TOTAL", grand_brands, grand_asins, grand_reviews)

    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
