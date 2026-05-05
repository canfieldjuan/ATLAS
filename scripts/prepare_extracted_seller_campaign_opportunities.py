#!/usr/bin/env python3
"""Prepare Amazon seller campaign opportunities from seller targets."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.campaign_postgres_seller_opportunities import (  # noqa: E402
    prepare_seller_campaign_opportunities,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare campaign_opportunities rows for Amazon seller campaigns "
            "from active seller_targets and cached category intelligence."
        )
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL"),
        help="Postgres DSN. Defaults to EXTRACTED_DATABASE_URL or DATABASE_URL.",
    )
    parser.add_argument("--account-id", help="Tenant/account scope for generated rows.")
    parser.add_argument("--category", help="Only prepare sellers in one category.")
    parser.add_argument("--seller-status", default="active")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Delete matching seller opportunity rows before inserting fresh rows.",
    )
    parser.add_argument("--target-mode", default="amazon_seller")
    parser.add_argument("--seller-targets-table", default="seller_targets")
    parser.add_argument(
        "--category-snapshots-table",
        default="category_intelligence_snapshots",
    )
    parser.add_argument("--opportunities-table", default="campaign_opportunities")
    parser.add_argument(
        "--output",
        type=Path,
        help="Write result JSON to this path instead of stdout.",
    )
    return parser.parse_args(argv)


async def _create_pool(database_url: str):
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on host environment
        raise RuntimeError(
            "asyncpg is required for seller opportunity preparation; "
            "install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def _main() -> int:
    args = _parse_args()
    if not args.database_url:
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")
    pool = await _create_pool(args.database_url)
    try:
        result = await prepare_seller_campaign_opportunities(
            pool,
            account_id=args.account_id,
            category=args.category,
            seller_status=args.seller_status,
            limit=args.limit,
            replace_existing=args.replace_existing,
            target_mode=args.target_mode,
            seller_targets_table=args.seller_targets_table,
            category_snapshots_table=args.category_snapshots_table,
            opportunities_table=args.opportunities_table,
        )
    finally:
        close = getattr(pool, "close", None)
        if close is not None:
            maybe_awaitable = close()
            if hasattr(maybe_awaitable, "__await__"):
                await maybe_awaitable
    output = json.dumps(result.as_dict(), indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(f"{output}\n", encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
