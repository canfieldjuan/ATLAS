#!/usr/bin/env python3
"""Backfill witness-quality fields into persisted reasoning/report JSON.

Read-only by default. The script decorates quote-bearing witness objects by
joining their witness_id/_sid/source_id to the latest b2b_vendor_witnesses row.
It does not rerun LLMs or regenerate reports.

Usage:
  python scripts/backfill_witness_quality_fields.py --days 30
  python scripts/backfill_witness_quality_fields.py --days 30 --apply
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

from atlas_brain.services.witness_quality_maintenance import run_backfill
from atlas_brain.storage.database import close_database, get_db_pool, init_database


async def _main(args: argparse.Namespace) -> None:
    await init_database()
    pool = get_db_pool()
    try:
        result = await run_backfill(
            pool,
            days=args.days,
            apply=args.apply,
            overwrite=args.overwrite,
            limit=args.limit,
        )
    finally:
        await close_database()

    print(json.dumps(result, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write updates.")
    parser.add_argument("--days", type=int, default=30, help="Lookback window.")
    parser.add_argument("--limit", type=int, default=None, help="Max rows per table.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing quality fields.")
    asyncio.run(_main(parser.parse_args()))
