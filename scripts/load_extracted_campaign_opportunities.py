#!/usr/bin/env python3
"""Load customer campaign opportunities into the extracted product database."""

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

from extracted_content_pipeline.campaign_customer_data import (  # noqa: E402
    load_campaign_opportunities_from_file,
)
from extracted_content_pipeline.campaign_ports import TenantScope  # noqa: E402
from extracted_content_pipeline.campaign_postgres_import import (  # noqa: E402
    import_campaign_opportunities,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load JSON or CSV campaign opportunities into Postgres."
    )
    parser.add_argument("path", type=Path, help="Customer JSON or CSV opportunity file.")
    parser.add_argument(
        "--database-url",
        default=os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL"),
        help="Postgres DSN. Defaults to EXTRACTED_DATABASE_URL or DATABASE_URL.",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "json", "csv"),
        default="auto",
        help="Customer data file format.",
    )
    parser.add_argument(
        "--target-mode",
        default="vendor_retention",
        help="Campaign target mode to stamp on imported rows.",
    )
    parser.add_argument(
        "--account-id",
        default=None,
        help="Optional tenant/account id to stamp on imported rows.",
    )
    parser.add_argument(
        "--opportunity-table",
        default="campaign_opportunities",
        help="Target table for opportunity rows.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Delete matching target ids for this account/mode before inserting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and count rows without connecting to Postgres.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON summary instead of a concise text summary.",
    )
    return parser.parse_args(argv)


async def _create_pool(database_url: str):
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "asyncpg is required to load opportunities; install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def _main() -> int:
    args = _parse_args()
    loaded = load_campaign_opportunities_from_file(
        args.path,
        file_format=args.format,
        target_mode=args.target_mode,
    )
    pool = None
    if not args.dry_run:
        if not args.database_url:
            raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")
        pool = await _create_pool(args.database_url)
    try:
        result = await import_campaign_opportunities(
            pool or object(),
            loaded.opportunities,
            scope=TenantScope(account_id=args.account_id),
            target_mode=args.target_mode,
            opportunity_table=args.opportunity_table,
            replace_existing=args.replace_existing,
            dry_run=args.dry_run,
            normalize=False,
            warnings=loaded.warnings,
            source=loaded.source,
        )
    finally:
        close = getattr(pool, "close", None)
        if close is not None:
            maybe_awaitable = close()
            if hasattr(maybe_awaitable, "__await__"):
                await maybe_awaitable
    if args.json:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    else:
        mode = "would insert" if result.dry_run else "inserted"
        print(
            f"{mode} {result.inserted} opportunity row(s); "
            f"skipped {result.skipped}; warnings {len(result.warnings)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
