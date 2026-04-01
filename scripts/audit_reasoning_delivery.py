#!/usr/bin/env python3
"""Audit reasoning delivery health and downstream provenance coverage.

Usage:
  python scripts/audit_reasoning_delivery.py
  python scripts/audit_reasoning_delivery.py --days 14 --top 15
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

from atlas_brain.services.reasoning_delivery_audit import summarize_reasoning_delivery_health
from atlas_brain.storage.database import close_database, get_db_pool, init_database


async def _main(*, days: int, top: int) -> None:
    await init_database()
    pool = get_db_pool()
    try:
        result = await summarize_reasoning_delivery_health(
            pool,
            days=days,
            top_n=top,
        )
    finally:
        await close_database()

    print(json.dumps(result, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days.")
    parser.add_argument("--top", type=int, default=10, help="Top N validation findings to return.")
    args = parser.parse_args()
    asyncio.run(_main(days=args.days, top=args.top))
