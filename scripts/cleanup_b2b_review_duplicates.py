"""
Clean legacy duplicate b2b_reviews rows caused by historical dedup-key drift.

Duplicate groups are identified by:
    (source, vendor_name, source_review_id)

Keep policy:
1. Prefer rows with a parser_version over null parser_version.
2. Prefer newer parser versions.
3. Prefer richer rows.
4. Prefer newer imported rows.
5. Tie-break by id.

Before deleting duplicate review rows, child foreign keys are remapped to the
chosen keep row so downstream artifacts are preserved.

Usage:
    python scripts/cleanup_b2b_review_duplicates.py
    python scripts/cleanup_b2b_review_duplicates.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from pathlib import Path

import asyncpg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("cleanup_b2b_review_duplicates")

sys.path.insert(0, str(Path(__file__).parent.parent))

from atlas_brain.storage.config import db_settings  # noqa: E402


_RANKED_DUPES_SQL = """
WITH ranked AS (
    SELECT
        id,
        source,
        vendor_name,
        source_review_id,
        row_number() OVER (
            PARTITION BY source, vendor_name, source_review_id
            ORDER BY
                CASE WHEN parser_version IS NOT NULL AND parser_version <> '' THEN 1 ELSE 0 END DESC,
                parser_version DESC NULLS LAST,
                (
                    CASE WHEN summary IS NOT NULL AND summary <> '' THEN 1 ELSE 0 END +
                    CASE WHEN pros IS NOT NULL AND pros <> '' THEN 1 ELSE 0 END +
                    CASE WHEN cons IS NOT NULL AND cons <> '' THEN 1 ELSE 0 END +
                    CASE WHEN reviewer_company_norm IS NOT NULL THEN 1 ELSE 0 END +
                    CASE WHEN raw_metadata IS NOT NULL AND raw_metadata::text <> '{}' THEN 1 ELSE 0 END +
                    CASE WHEN reviewed_at IS NOT NULL THEN 1 ELSE 0 END
                ) DESC,
                imported_at DESC NULLS LAST,
                id DESC
        ) AS rn,
        first_value(id) OVER (
            PARTITION BY source, vendor_name, source_review_id
            ORDER BY
                CASE WHEN parser_version IS NOT NULL AND parser_version <> '' THEN 1 ELSE 0 END DESC,
                parser_version DESC NULLS LAST,
                (
                    CASE WHEN summary IS NOT NULL AND summary <> '' THEN 1 ELSE 0 END +
                    CASE WHEN pros IS NOT NULL AND pros <> '' THEN 1 ELSE 0 END +
                    CASE WHEN cons IS NOT NULL AND cons <> '' THEN 1 ELSE 0 END +
                    CASE WHEN reviewer_company_norm IS NOT NULL THEN 1 ELSE 0 END +
                    CASE WHEN raw_metadata IS NOT NULL AND raw_metadata::text <> '{}' THEN 1 ELSE 0 END +
                    CASE WHEN reviewed_at IS NOT NULL THEN 1 ELSE 0 END
                ) DESC,
                imported_at DESC NULLS LAST,
                id DESC
        ) AS keep_id
    FROM b2b_reviews
    WHERE source_review_id IS NOT NULL
      AND source_review_id <> ''
)
SELECT keep_id, id AS drop_id, source
FROM ranked
WHERE rn > 1
ORDER BY source, keep_id, drop_id
"""


def _parse_status_count(status: str) -> int:
    match = re.search(r"(\d+)$", status or "")
    return int(match.group(1)) if match else 0


async def _fetch_duplicate_mappings(conn: asyncpg.Connection) -> list[asyncpg.Record]:
    return await conn.fetch(_RANKED_DUPES_SQL)


async def _summarize_duplicates(conn: asyncpg.Connection) -> tuple[int, dict[str, int]]:
    rows = await _fetch_duplicate_mappings(conn)
    by_source: dict[str, int] = {}
    for row in rows:
        by_source[row["source"]] = by_source.get(row["source"], 0) + 1
    return len(rows), by_source


async def _apply_cleanup(conn: asyncpg.Connection, mappings: list[asyncpg.Record]) -> dict[str, int]:
    await conn.execute(
        """
        CREATE TEMP TABLE _review_dupe_cleanup (
            keep_id uuid NOT NULL,
            drop_id uuid PRIMARY KEY,
            source text NOT NULL
        ) ON COMMIT DROP
        """
    )
    await conn.executemany(
        "INSERT INTO _review_dupe_cleanup (keep_id, drop_id, source) VALUES ($1, $2, $3)",
        [(row["keep_id"], row["drop_id"], row["source"]) for row in mappings],
    )

    remapped_parent = _parse_status_count(
        await conn.execute(
            """
            UPDATE b2b_reviews child
            SET parent_review_id = dupes.keep_id
            FROM _review_dupe_cleanup dupes
            WHERE child.parent_review_id = dupes.drop_id
            """
        )
    )
    remapped_affiliate = _parse_status_count(
        await conn.execute(
            """
            UPDATE affiliate_clicks clicks
            SET review_id = dupes.keep_id
            FROM _review_dupe_cleanup dupes
            WHERE clicks.review_id = dupes.drop_id
            """
        )
    )
    remapped_company_signals = _parse_status_count(
        await conn.execute(
            """
            UPDATE b2b_company_signals signals
            SET review_id = dupes.keep_id
            FROM _review_dupe_cleanup dupes
            WHERE signals.review_id = dupes.drop_id
            """
        )
    )
    deleted_reviews = _parse_status_count(
        await conn.execute(
            """
            DELETE FROM b2b_reviews reviews
            USING _review_dupe_cleanup dupes
            WHERE reviews.id = dupes.drop_id
            """
        )
    )
    return {
        "remapped_parent_review_id": remapped_parent,
        "remapped_affiliate_clicks": remapped_affiliate,
        "remapped_company_signals": remapped_company_signals,
        "deleted_reviews": deleted_reviews,
    }


async def main(apply: bool) -> int:
    conn = await asyncpg.connect(db_settings.dsn)
    try:
        before_total, before_by_source = await _summarize_duplicates(conn)
        logger.info("Duplicate review rows detected: %d", before_total)
        for source, count in sorted(before_by_source.items(), key=lambda item: (-item[1], item[0])):
            logger.info("  %s: %d", source, count)

        if not apply or before_total == 0:
            logger.info("Dry run only; no changes applied")
            return 0

        async with conn.transaction():
            mappings = await _fetch_duplicate_mappings(conn)
            stats = await _apply_cleanup(conn, mappings)

        after_total, after_by_source = await _summarize_duplicates(conn)
        logger.info("Cleanup applied")
        for key, value in stats.items():
            logger.info("  %s=%d", key, value)
        logger.info("Duplicates remaining: %d", after_total)
        for source, count in sorted(after_by_source.items(), key=lambda item: (-item[1], item[0])):
            logger.info("  remaining %s: %d", source, count)
        return 0 if after_total == 0 else 1
    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean legacy duplicate b2b_reviews rows")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the cleanup. Without this flag the script only prints a dry-run summary.",
    )
    raise SystemExit(asyncio.run(main(apply=parser.parse_args().apply)))
