#!/usr/bin/env python3
"""Backfill canonical blog quality audits on recent blog_posts rows.

Usage:
  python scripts/backfill_blog_quality_truth.py
  python scripts/backfill_blog_quality_truth.py --apply
  python scripts/backfill_blog_quality_truth.py --apply --limit 200
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
from atlas_brain.services.blog_quality_backfill import (
    apply_blog_quality_backfill,
    plan_blog_quality_backfill,
)
from atlas_brain.storage.database import close_database, get_db_pool, init_database


async def _main(*, apply: bool, limit: int | None, days: int) -> None:
    await init_database()
    pool = get_db_pool()
    try:
        kwargs = {
            "days": int(days),
            "limit": limit,
        }
        if apply:
            result = await apply_blog_quality_backfill(pool, **kwargs)
        else:
            result = await plan_blog_quality_backfill(pool, **kwargs)
    finally:
        await close_database()

    print(
        json.dumps(
            {
                "apply": apply,
                "limit": limit,
                "days": int(days),
                "scanned": result["scanned"],
                "changed": result["changed"],
                "applied": result.get("applied", 0),
                "preview": result["items"][:10],
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
    parser.add_argument(
        "--days",
        type=int,
        default=int(settings.b2b_churn.blog_quality_backfill_days),
        help="Lookback window in days.",
    )
    args = parser.parse_args()
    asyncio.run(_main(apply=args.apply, limit=args.limit, days=args.days))
