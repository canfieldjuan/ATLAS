#!/usr/bin/env python3
"""Backfill truthful visibility fields on legacy blog_posts rows.

Usage:
  python scripts/backfill_blog_visibility_truth.py
  python scripts/backfill_blog_visibility_truth.py --apply
  python scripts/backfill_blog_visibility_truth.py --apply --limit 200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_brain.config import settings
from atlas_brain.services.blog_visibility_backfill import (
    apply_blog_visibility_backfill,
    plan_blog_visibility_backfill,
)
from atlas_brain.storage.database import close_database, get_db_pool, init_database


async def _main(*, apply: bool, limit: int | None) -> None:
    await init_database()
    pool = get_db_pool()
    try:
        if apply:
            result = await apply_blog_visibility_backfill(
                pool,
                limit=limit,
                default_threshold=settings.b2b_churn.blog_quality_pass_score,
            )
        else:
            result = await plan_blog_visibility_backfill(
                pool,
                limit=limit,
                default_threshold=settings.b2b_churn.blog_quality_pass_score,
            )
    finally:
        await close_database()

    preview = result["items"][:10]
    print(
        json.dumps(
            {
                "apply": apply,
                "limit": limit,
                "default_threshold": settings.b2b_churn.blog_quality_pass_score,
                "scanned": result["scanned"],
                "changed": result["changed"],
                "applied": result.get("applied", 0),
                "preview": preview,
            },
            ensure_ascii=True,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write updates to the database.")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to scan.")
    args = parser.parse_args()
    asyncio.run(_main(apply=args.apply, limit=args.limit))
