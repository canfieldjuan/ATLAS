#!/usr/bin/env python3
"""Backfill campaign specificity audits on legacy b2b_campaigns rows.

Usage:
  python scripts/backfill_campaign_specificity_truth.py
  python scripts/backfill_campaign_specificity_truth.py --apply
  python scripts/backfill_campaign_specificity_truth.py --apply --limit 200
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
from atlas_brain.services.campaign_specificity_backfill import (
    apply_campaign_specificity_backfill,
    plan_campaign_specificity_backfill,
)
from atlas_brain.storage.database import close_database, get_db_pool, init_database


async def _main(*, apply: bool, limit: int | None) -> None:
    await init_database()
    pool = get_db_pool()
    try:
        kwargs = {
            "limit": limit,
            "min_anchor_hits": int(settings.b2b_campaign.specificity_min_anchor_hits),
            "require_anchor_support": bool(settings.b2b_campaign.specificity_require_anchor_support),
            "require_timing_or_numeric_when_available": bool(
                settings.b2b_campaign.specificity_require_timing_or_numeric_when_available
            ),
        }
        if apply:
            result = await apply_campaign_specificity_backfill(pool, **kwargs)
        else:
            result = await plan_campaign_specificity_backfill(pool, **kwargs)
    finally:
        await close_database()

    preview = result["items"][:10]
    print(
        json.dumps(
            {
                "apply": apply,
                "limit": limit,
                "min_anchor_hits": int(settings.b2b_campaign.specificity_min_anchor_hits),
                "require_anchor_support": bool(settings.b2b_campaign.specificity_require_anchor_support),
                "require_timing_or_numeric_when_available": bool(
                    settings.b2b_campaign.specificity_require_timing_or_numeric_when_available
                ),
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
