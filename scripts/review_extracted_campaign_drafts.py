#!/usr/bin/env python3
"""Approve, queue, cancel, or expire generated campaign drafts."""

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
from extracted_content_pipeline.campaign_postgres_review import (  # noqa: E402
    review_campaign_drafts,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review generated campaign drafts in Postgres."
    )
    parser.add_argument(
        "campaign_ids",
        nargs="+",
        help="Campaign UUIDs to update. Comma-separated ids are accepted.",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL"),
        help="Postgres DSN. Defaults to EXTRACTED_DATABASE_URL or DATABASE_URL.",
    )
    parser.add_argument("--account-id", default=None, help="Optional tenant/account id.")
    parser.add_argument(
        "--status",
        choices=("approved", "queued", "cancelled", "expired"),
        default="approved",
        help="Review status to apply.",
    )
    parser.add_argument(
        "--from-status",
        default="draft",
        help="Comma-separated source statuses allowed to update. Empty string disables the guard.",
    )
    parser.add_argument(
        "--from-email",
        default=None,
        help="Optional sender email to stamp when queueing drafts.",
    )
    parser.add_argument("--reason", default=None, help="Optional review reason.")
    parser.add_argument("--reviewed-by", default=None, help="Optional reviewer id/email.")
    parser.add_argument("--campaign-table", default="b2b_campaigns", help="Campaign table.")
    parser.add_argument("--dry-run", action="store_true", help="Preview matching rows without updating.")
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
            "asyncpg is required to review drafts; install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def _main() -> int:
    args = _parse_args()
    if not args.database_url:
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")
    pool = await _create_pool(args.database_url)
    try:
        result = await review_campaign_drafts(
            pool,
            campaign_ids=_parse_campaign_ids(args.campaign_ids),
            status=args.status,
            scope=TenantScope(account_id=args.account_id),
            campaign_table=args.campaign_table,
            from_statuses=_parse_statuses(args.from_status),
            from_email=args.from_email,
            reason=args.reason,
            reviewed_by=args.reviewed_by,
            dry_run=args.dry_run,
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
        verb = "would update" if result.dry_run else "updated"
        print(
            f"{verb} {result.updated} of {len(result.requested_ids)} campaign draft(s) "
            f"to status={result.status}"
        )
    return 0


def _parse_campaign_ids(values: list[str]) -> tuple[str, ...]:
    ids: list[str] = []
    for value in values:
        ids.extend(item.strip() for item in str(value or "").split(",") if item.strip())
    return tuple(ids)


def _parse_statuses(raw: str | None) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in str(raw or "").split(",")
        if item.strip()
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
