#!/usr/bin/env python3
"""Audit recovered reasoning retry churn and escalation pressure.

Usage:
  python scripts/audit_reasoning_retry_churn.py
  python scripts/audit_reasoning_retry_churn.py --hours 72 --top 15
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

from atlas_brain.services.reasoning_retry_audit import summarize_reasoning_retry_churn
from atlas_brain.storage.database import close_database, get_db_pool, init_database


async def _main(*, hours: int, top: int, queue_limit: int) -> None:
    await init_database()
    pool = get_db_pool()
    try:
        result = await summarize_reasoning_retry_churn(
            pool,
            hours=hours,
            top_n=top,
            queue_limit=queue_limit,
        )
    finally:
        await close_database()

    print(json.dumps(result, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=24, help="Lookback window in hours.")
    parser.add_argument("--top", type=int, default=10, help="Top N rules/vendors to return.")
    parser.add_argument(
        "--queue-limit",
        type=int,
        default=20,
        help="Max open escalated retry items to return.",
    )
    args = parser.parse_args()
    asyncio.run(_main(hours=args.hours, top=args.top, queue_limit=args.queue_limit))
