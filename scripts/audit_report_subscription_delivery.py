#!/usr/bin/env python3
"""Audit recurring report-subscription delivery health.

Usage:
  python scripts/audit_report_subscription_delivery.py
  python scripts/audit_report_subscription_delivery.py --days 30 --top 20
  python scripts/audit_report_subscription_delivery.py --account-id <uuid>
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

from atlas_brain.services.report_subscription_delivery_audit import (
    summarize_report_subscription_delivery_health,
)
from atlas_brain.storage.database import close_database, get_db_pool, init_database


async def _main(*, days: int, top: int, account_ids: list[str]) -> None:
    await init_database()
    pool = get_db_pool()
    try:
        result = await summarize_report_subscription_delivery_health(
            pool,
            days=days,
            top_n=top,
            account_ids=account_ids or None,
        )
    finally:
        await close_database()

    print(json.dumps(result, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days.")
    parser.add_argument("--top", type=int, default=10, help="Top N recent attempts/accounts to return.")
    parser.add_argument(
        "--account-id",
        action="append",
        default=[],
        help="Optional account UUID filter. Repeat to audit multiple accounts.",
    )
    args = parser.parse_args()
    asyncio.run(_main(days=args.days, top=args.top, account_ids=args.account_id))
