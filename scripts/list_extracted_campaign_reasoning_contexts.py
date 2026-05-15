#!/usr/bin/env python3
"""List/export Content Ops campaign reasoning contexts from Postgres."""

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
from extracted_content_pipeline.campaign_reasoning_postgres import (  # noqa: E402
    PostgresCampaignReasoningContextRepository,
)


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List/export campaign_reasoning_contexts rows."
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL"),
        help="Postgres DSN. Defaults to EXTRACTED_DATABASE_URL or DATABASE_URL.",
    )
    parser.add_argument("--account-id", default=None, help="Optional tenant/account id.")
    parser.add_argument("--target-mode", default=None, help="Optional target_mode filter.")
    parser.add_argument(
        "--selector",
        action="append",
        default=[],
        help="Only rows matching this selector; repeatable.",
    )
    parser.add_argument(
        "--table",
        default="campaign_reasoning_contexts",
        help="Reasoning context table name.",
    )
    parser.add_argument(
        "--limit",
        type=_non_negative_int,
        default=20,
        help="Maximum rows to return. Default: 20.",
    )
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path.")
    return parser.parse_args(argv)


async def _create_pool(database_url: str) -> Any:
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "asyncpg is required to list campaign reasoning contexts; "
            "install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def _main_from_args(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.database_url:
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")
    pool = await _create_pool(args.database_url)
    try:
        repository = PostgresCampaignReasoningContextRepository(
            pool=pool,
            table=args.table,
        )
        result = await repository.list_contexts(
            scope=TenantScope(account_id=args.account_id)
            if args.account_id is not None
            else None,
            target_mode=args.target_mode,
            selectors=tuple(args.selector or ()),
            limit=args.limit,
        )
    finally:
        await pool.close()
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


async def _main() -> int:
    return await _main_from_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
