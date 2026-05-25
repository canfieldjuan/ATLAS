#!/usr/bin/env python3
"""Backfill approved ticket FAQ drafts into the search projection table.

Usage:
  python scripts/backfill_ticket_faq_search_documents.py
  python scripts/backfill_ticket_faq_search_documents.py --apply
  python scripts/backfill_ticket_faq_search_documents.py --account-id acct-1 --limit 100 --apply
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

from atlas_brain.storage.database import close_database, get_db_pool, init_database
from extracted_content_pipeline.ticket_faq_postgres import backfill_ticket_faq_search_documents


async def _main(
    *,
    apply: bool,
    status: str,
    account_id: str | None,
    limit: int | None,
) -> None:
    await init_database()
    pool = get_db_pool()
    try:
        result = await backfill_ticket_faq_search_documents(
            pool,
            apply=apply,
            status=status,
            account_id=account_id,
            limit=limit,
        )
    finally:
        await close_database()

    print(json.dumps(result.as_dict(), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write search projection rows.")
    parser.add_argument("--status", default="approved", help="FAQ draft status to backfill.")
    parser.add_argument("--account-id", default=None, help="Restrict backfill to one account.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum FAQ drafts to scan.")
    args = parser.parse_args()
    asyncio.run(
        _main(
            apply=args.apply,
            status=args.status,
            account_id=args.account_id,
            limit=args.limit,
        )
    )
