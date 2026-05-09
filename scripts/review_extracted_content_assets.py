#!/usr/bin/env python3
"""Update generated Content Ops asset statuses in the extracted database."""

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
from extracted_content_pipeline.landing_page_postgres import (  # noqa: E402
    PostgresLandingPageRepository,
)
from extracted_content_pipeline.report_postgres import PostgresReportRepository  # noqa: E402
from extracted_content_pipeline.sales_brief_postgres import (  # noqa: E402
    PostgresSalesBriefRepository,
)


ASSET_CHOICES = ("report", "landing_page", "sales_brief")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review generated AI Content Ops assets in Postgres."
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL"),
        help="Postgres DSN. Defaults to EXTRACTED_DATABASE_URL or DATABASE_URL.",
    )
    parser.add_argument("--asset", choices=ASSET_CHOICES, required=True)
    parser.add_argument("--id", dest="asset_id", required=True, help="Asset UUID to update.")
    parser.add_argument(
        "--status",
        required=True,
        help="Status to apply. Host-defined lifecycle statuses are accepted.",
    )
    parser.add_argument("--account-id", default=None, help="Optional tenant/account id.")
    return parser.parse_args(argv)


async def _create_pool(database_url: str):
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "asyncpg is required to review assets; install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def _main() -> int:
    args = _parse_args()
    if not args.database_url:
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")
    args.status = _status_arg(args.status)
    pool = await _create_pool(args.database_url)
    try:
        updated = await _review_asset(args, pool)
    finally:
        close = getattr(pool, "close", None)
        if close is not None:
            maybe_awaitable = close()
            if hasattr(maybe_awaitable, "__await__"):
                await maybe_awaitable
    print(json.dumps(_result_payload(args, updated), indent=2, sort_keys=True))
    return 0 if updated else 1


async def _review_asset(args: argparse.Namespace, pool: Any) -> bool:
    scope = TenantScope(account_id=args.account_id)
    if args.asset == "report":
        return await PostgresReportRepository(pool).update_status(
            args.asset_id,
            args.status,
            scope=scope,
        )
    if args.asset == "landing_page":
        return await PostgresLandingPageRepository(pool).update_status(
            args.asset_id,
            args.status,
            scope=scope,
        )
    if args.asset == "sales_brief":
        return await PostgresSalesBriefRepository(pool).update_status(
            args.asset_id,
            args.status,
            scope=scope,
        )
    raise ValueError(f"Unsupported asset: {args.asset}")


def _result_payload(args: argparse.Namespace, updated: bool) -> dict[str, Any]:
    return {
        "account_id": args.account_id,
        "asset": args.asset,
        "id": args.asset_id,
        "status": args.status,
        "updated": bool(updated),
    }


def _status_arg(raw: str | None) -> str:
    status = str(raw or "").strip()
    if not status:
        raise SystemExit("--status must be a non-empty string")
    return status


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
