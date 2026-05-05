#!/usr/bin/env python3
"""Refresh Amazon seller category intelligence snapshots."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.campaign_postgres_seller_category_intelligence import (  # noqa: E402
    refresh_seller_category_intelligence,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh category_intelligence_snapshots from product_reviews "
            "for Amazon seller campaign generation."
        )
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL"),
        help="Postgres DSN. Defaults to EXTRACTED_DATABASE_URL or DATABASE_URL.",
    )
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="Refresh one category. Repeat for multiple categories.",
    )
    parser.add_argument("--min-reviews", type=int, default=50)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--reviews-table", default="product_reviews")
    parser.add_argument("--metadata-table", default="product_metadata")
    parser.add_argument("--snapshots-table", default="category_intelligence_snapshots")
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
            "asyncpg is required for seller category intelligence refresh; "
            "install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def _main() -> int:
    args = _parse_args()
    if not args.database_url:
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")
    pool = await _create_pool(args.database_url)
    try:
        result = await refresh_seller_category_intelligence(
            pool,
            categories=tuple(args.category or ()),
            min_reviews=args.min_reviews,
            limit=args.limit,
            reviews_table=args.reviews_table,
            metadata_table=args.metadata_table,
            snapshots_table=args.snapshots_table,
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
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
