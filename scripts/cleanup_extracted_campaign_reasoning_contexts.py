#!/usr/bin/env python3
"""Dry-run or delete stale Content Ops campaign reasoning contexts."""

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


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count or delete stale campaign_reasoning_contexts rows. "
            "Runs as a dry-run unless --apply is passed."
        )
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL"),
        help="Postgres DSN. Defaults to EXTRACTED_DATABASE_URL or DATABASE_URL.",
    )
    parser.add_argument(
        "--older-than-days",
        type=_positive_int,
        required=True,
        help="Only rows older than this many days are counted or deleted.",
    )
    parser.add_argument("--account-id", default=None, help="Optional tenant/account filter.")
    parser.add_argument("--target-mode", default=None, help="Optional target_mode filter.")
    parser.add_argument(
        "--table",
        default="campaign_reasoning_contexts",
        help="Reasoning context table name.",
    )
    parser.add_argument("--apply", action="store_true", help="Delete matching rows.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args(argv)


async def _create_pool(database_url: str) -> Any:
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "asyncpg is required to clean campaign reasoning contexts; "
            "install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def _cleanup(
    repository: PostgresCampaignReasoningContextRepository,
    *,
    account_id: str | None,
    target_mode: str | None,
    older_than_days: int,
    apply: bool,
) -> dict[str, Any]:
    affected = await repository.delete_stale_contexts(
        older_than_days=older_than_days,
        scope=TenantScope(account_id=account_id) if account_id is not None else None,
        target_mode=target_mode,
        dry_run=not apply,
    )
    return {
        "status": "deleted" if apply else "dry_run",
        "affected": affected,
        "older_than_days": older_than_days,
        "account_id": account_id,
        "target_mode": target_mode,
    }


async def _main() -> int:
    args = _parse_args()
    if not args.database_url:
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")
    pool = await _create_pool(args.database_url)
    try:
        repository = PostgresCampaignReasoningContextRepository(
            pool=pool,
            table=args.table,
        )
        result = await _cleanup(
            repository,
            account_id=args.account_id,
            target_mode=args.target_mode,
            older_than_days=args.older_than_days,
            apply=bool(args.apply),
        )
    finally:
        await pool.close()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        action = "deleted" if args.apply else "would delete"
        print(
            f"{action} {result['affected']} campaign reasoning context rows "
            f"older_than_days={result['older_than_days']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
