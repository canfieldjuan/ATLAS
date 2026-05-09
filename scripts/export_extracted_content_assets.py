#!/usr/bin/env python3
"""Export generated Content Ops assets from the extracted product database."""

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

from extracted_content_pipeline.campaign_ports import TenantScope  # noqa: E402
from extracted_content_pipeline.landing_page_export import (  # noqa: E402
    export_landing_page_drafts,
)
from extracted_content_pipeline.landing_page_postgres import (  # noqa: E402
    PostgresLandingPageRepository,
)
from extracted_content_pipeline.report_export import export_report_drafts  # noqa: E402
from extracted_content_pipeline.report_postgres import PostgresReportRepository  # noqa: E402
from extracted_content_pipeline.sales_brief_export import (  # noqa: E402
    export_sales_brief_drafts,
)
from extracted_content_pipeline.sales_brief_postgres import (  # noqa: E402
    PostgresSalesBriefRepository,
)


ASSET_CHOICES = ("report", "landing_page", "sales_brief")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export generated AI Content Ops assets from Postgres."
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL"),
        help="Postgres DSN. Defaults to EXTRACTED_DATABASE_URL or DATABASE_URL.",
    )
    parser.add_argument("--asset", choices=ASSET_CHOICES, required=True)
    parser.add_argument("--account-id", default=None, help="Optional tenant/account id.")
    parser.add_argument(
        "--status",
        default="draft",
        help="Status to export. Use empty string for all statuses.",
    )
    parser.add_argument("--target-mode", default=None, help="Report/sales brief target_mode filter.")
    parser.add_argument("--report-type", default=None, help="Report type filter.")
    parser.add_argument("--campaign-name", default=None, help="Landing page campaign_name filter.")
    parser.add_argument("--slug", default=None, help="Landing page slug filter.")
    parser.add_argument("--brief-type", default=None, help="Sales brief type filter.")
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
            "asyncpg is required to export assets; install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def _main() -> int:
    args = _parse_args()
    if not args.database_url:
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")
    pool = await _create_pool(args.database_url)
    try:
        result = await _export_for_asset(args, pool)
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


async def _export_for_asset(args: argparse.Namespace, pool: Any):
    scope = TenantScope(account_id=args.account_id)
    status = _status_arg(args.status)
    if args.asset == "report":
        return await export_report_drafts(
            PostgresReportRepository(pool),
            scope=scope,
            status=status,
            target_mode=args.target_mode,
            report_type=args.report_type,
            limit=args.limit,
        )
    if args.asset == "landing_page":
        return await export_landing_page_drafts(
            PostgresLandingPageRepository(pool),
            scope=scope,
            status=status,
            campaign_name=args.campaign_name,
            slug=args.slug,
            limit=args.limit,
        )
    if args.asset == "sales_brief":
        return await export_sales_brief_drafts(
            PostgresSalesBriefRepository(pool),
            scope=scope,
            status=status,
            target_mode=args.target_mode,
            brief_type=args.brief_type,
            limit=args.limit,
        )
    raise ValueError(f"Unsupported asset: {args.asset}")


def _status_arg(raw: str | None) -> str | None:
    status = str(raw or "").strip()
    return status or None


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
