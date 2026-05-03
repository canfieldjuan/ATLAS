#!/usr/bin/env python3
"""Export generated campaign drafts from the extracted product database."""

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

from extracted_content_pipeline.campaign_ports import TenantScope  # noqa: E402
from extracted_content_pipeline.campaign_postgres_export import (  # noqa: E402
    list_campaign_drafts,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export generated campaign drafts from Postgres."
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL"),
        help="Postgres DSN. Defaults to EXTRACTED_DATABASE_URL or DATABASE_URL.",
    )
    parser.add_argument("--account-id", default=None, help="Optional tenant/account id.")
    parser.add_argument(
        "--status",
        default="draft",
        help="Comma-separated statuses to export. Use empty string for all statuses.",
    )
    parser.add_argument("--target-mode", default=None, help="Optional target_mode filter.")
    parser.add_argument("--channel", default=None, help="Optional channel filter.")
    parser.add_argument("--vendor-name", default=None, help="Optional vendor filter.")
    parser.add_argument("--company-name", default=None, help="Optional company filter.")
    parser.add_argument("--campaign-table", default="b2b_campaigns", help="Campaign table.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum rows to export.")
    parser.add_argument(
        "--format",
        choices=("json", "csv"),
        default="json",
        help="Output format.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output path.")
    return parser.parse_args(argv)


async def _create_pool(database_url: str):
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "asyncpg is required to export drafts; install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def _main() -> int:
    args = _parse_args()
    if not args.database_url:
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")
    pool = await _create_pool(args.database_url)
    try:
        result = await list_campaign_drafts(
            pool,
            scope=TenantScope(account_id=args.account_id),
            campaign_table=args.campaign_table,
            statuses=_parse_statuses(args.status),
            target_mode=args.target_mode,
            channel=args.channel,
            vendor_name=args.vendor_name,
            company_name=args.company_name,
            limit=args.limit,
        )
    finally:
        close = getattr(pool, "close", None)
        if close is not None:
            maybe_awaitable = close()
            if hasattr(maybe_awaitable, "__await__"):
                await maybe_awaitable
    output = (
        result.as_csv()
        if args.format == "csv"
        else json.dumps(result.as_dict(), indent=2, sort_keys=True)
    )
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="" if output.endswith("\n") else "\n")
    return 0


def _parse_statuses(raw: str | None) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in str(raw or "").split(",")
        if item.strip()
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
