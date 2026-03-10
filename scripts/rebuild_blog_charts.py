#!/usr/bin/env python3
"""Rebuild charts and data_context for stale blog posts without re-running LLM.

Re-runs _gather_data() + _build_blueprint() to get vendor-scoped review counts
and fixed chart data, then updates just charts + data_context columns in the DB.
Preserves existing LLM-generated content (title, description, body).

Usage:
    python scripts/rebuild_blog_charts.py [--limit N] [--dry-run] [--type TYPE]
"""
import argparse
import asyncio
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main(limit: int, dry_run: bool, topic_type_filter: str | None):
    from atlas_brain.storage.config import db_settings

    import asyncpg

    pool = await asyncpg.create_pool(
        host=db_settings.host,
        port=db_settings.port,
        database=db_settings.database,
        user=db_settings.user,
        password=db_settings.password,
        min_size=2,
        max_size=4,
    )

    from atlas_brain.autonomous.tasks.b2b_blog_post_generation import (
        _gather_data,
        _build_blueprint,
    )

    where_clause = """
        status = 'draft'
        AND data_context->'topic_ctx' IS NOT NULL
        AND data_context->'topic_ctx' != 'null'::jsonb
        AND topic_type IN (
            'vendor_showdown', 'vendor_deep_dive', 'churn_report',
            'pricing_reality_check', 'vendor_alternative', 'switching_story',
            'migration_guide', 'market_landscape', 'pain_point_roundup',
            'best_fit_guide'
        )
    """
    params: list = []
    if topic_type_filter:
        where_clause += " AND topic_type = $1"
        params.append(topic_type_filter)

    rows = await pool.fetch(
        f"""
        SELECT id, slug, topic_type, data_context
        FROM blog_posts
        WHERE {where_clause}
        ORDER BY topic_type, created_at ASC
        LIMIT {limit}
        """,
        *params,
    )

    logger.info("Found %d posts to rebuild", len(rows))
    updated = 0
    failed = 0

    for row in rows:
        slug = row["slug"]
        topic_type = row["topic_type"]
        old_ctx = row["data_context"] or {}
        if isinstance(old_ctx, str):
            try:
                old_ctx = json.loads(old_ctx)
            except (json.JSONDecodeError, TypeError):
                old_ctx = {}

        stored_ctx = old_ctx.get("topic_ctx")
        if not stored_ctx or not isinstance(stored_ctx, dict):
            logger.warning("No topic_ctx for %s, skipping", slug)
            continue

        topic_ctx = {**stored_ctx, "slug": slug}

        try:
            data = await _gather_data(pool, topic_type, topic_ctx)
            blueprint = _build_blueprint(topic_type, topic_ctx, data)
            charts_json = json.dumps([asdict(c) for c in blueprint.charts], default=str)
            data_context_json = json.dumps(blueprint.data_context, default=str)

            old_enriched = old_ctx.get("enriched_count", "?")
            old_total = old_ctx.get("total_reviews_analyzed", "?")
            new_enriched = blueprint.data_context.get("enriched_count", "?")
            new_total = blueprint.data_context.get("total_reviews_analyzed", "?")
            n_charts = len(blueprint.charts)

            logger.info(
                "%-40s  reviews: %s->%s  enriched: %s->%s  charts: %d",
                slug[:40], old_total, new_total, old_enriched, new_enriched, n_charts,
            )

            if not dry_run:
                await pool.execute(
                    """
                    UPDATE blog_posts
                    SET charts = $1::jsonb,
                        data_context = $2::jsonb
                    WHERE id = $3
                    """,
                    charts_json,
                    data_context_json,
                    row["id"],
                )
            updated += 1
        except Exception:
            logger.exception("Failed to rebuild %s", slug)
            failed += 1

    logger.info(
        "Done. Updated: %d, Failed: %d, Skipped: %d",
        updated, failed, len(rows) - updated - failed,
    )
    await pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild blog post charts and data_context")
    parser.add_argument("--limit", type=int, default=200, help="Max posts to process")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without saving")
    parser.add_argument("--type", dest="topic_type", help="Filter by topic_type")
    args = parser.parse_args()
    asyncio.run(main(args.limit, args.dry_run, args.topic_type))
