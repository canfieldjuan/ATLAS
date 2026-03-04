"""Autonomous task: sync product_metadata for ASINs missing metadata.

Imports core functions from scripts/match_product_metadata.py and runs
incrementally -- only processes ASINs in product_reviews that are NOT
already in product_metadata.

Schedule: weekly (Sunday 4 AM).
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger("atlas.autonomous.tasks.product_metadata_sync")

# Add project root so we can import the script module
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _import_script():
    """Lazy-import functions from scripts/match_product_metadata.py."""
    import importlib.util

    script_path = Path(_PROJECT_ROOT) / "scripts" / "match_product_metadata.py"
    spec = importlib.util.spec_from_file_location("match_product_metadata", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def run(task) -> dict:
    """Find ASINs missing from product_metadata and match them against HF metadata."""
    from atlas_brain.storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "Database pool not initialized"}

    # Check if product_metadata table exists yet (created by match_product_metadata script)
    has_table = await pool.fetchval(
        "SELECT EXISTS ("
        "  SELECT 1 FROM information_schema.tables"
        "  WHERE table_schema = 'public' AND table_name = 'product_metadata'"
        ")"
    )

    if has_table:
        # Find ASINs in product_reviews but NOT in product_metadata
        missing_rows = await pool.fetch("""
            SELECT DISTINCT pr.asin
            FROM product_reviews pr
            LEFT JOIN product_metadata pm ON pr.asin = pm.asin
            WHERE pm.asin IS NULL
        """)
    else:
        # Table doesn't exist yet -- all ASINs need metadata
        logger.info("product_metadata table does not exist, treating all ASINs as missing")
        missing_rows = await pool.fetch(
            "SELECT DISTINCT asin FROM product_reviews"
        )

    missing_asins = {r["asin"] for r in missing_rows}

    if not missing_asins:
        return {"_skip_synthesis": "All ASINs already have metadata"}

    logger.info("Found %d ASINs missing metadata, starting scan...", len(missing_asins))
    start = time.monotonic()

    # Import script functions
    script = await asyncio.to_thread(_import_script)

    # Scan HF shards for missing ASINs (blocking I/O in thread)
    config = script.CATEGORY_CONFIG["electronics"]
    num_shards = config["num_shards"]
    shard_pattern = config["shard_pattern"].format(num=num_shards)

    all_matches = []
    matched_asins = set()

    for shard_idx in range(num_shards):
        remaining = missing_asins - matched_asins
        if not remaining:
            logger.info("All missing ASINs matched after %d shards", shard_idx)
            break

        logger.info(
            "Downloading + scanning shard %d/%d for %d remaining ASINs...",
            shard_idx + 1, num_shards, len(remaining),
        )

        shard_path = await asyncio.to_thread(script.download_shard, shard_idx, shard_pattern)
        matches = await asyncio.to_thread(script.scan_shard, shard_path, remaining)

        for m in matches:
            matched_asins.add(m["parent_asin"])
        all_matches.extend(matches)

        logger.info(
            "  Shard %d: %d matches (total %d/%d)",
            shard_idx, len(matches), len(matched_asins), len(missing_asins),
        )

    # Write matches to DB
    written = 0
    if all_matches:
        written = await script.write_to_db(all_matches)

    elapsed = time.monotonic() - start
    still_missing = len(missing_asins) - len(matched_asins)

    msg = (
        f"Matched {len(matched_asins)} of {len(missing_asins)} missing ASINs "
        f"({still_missing} still unmatched) in {elapsed:.0f}s"
    )
    logger.info(msg)

    return {
        "_skip_synthesis": msg,
        "matched": len(matched_asins),
        "written": written,
        "still_missing": still_missing,
        "elapsed_seconds": round(elapsed, 1),
    }
