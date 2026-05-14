#!/usr/bin/env python3
"""Load host-supplied blog blueprints into the extracted product database."""

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

from extracted_content_pipeline.blog_blueprint_ingest import (  # noqa: E402
    load_blog_blueprints_from_file,
)
from extracted_content_pipeline.blog_blueprint_postgres import (  # noqa: E402
    PostgresBlogBlueprintRepository,
)
from extracted_content_pipeline.campaign_ports import TenantScope  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load JSON blog blueprints into Postgres."
    )
    parser.add_argument("path", type=Path, help="Blog blueprint JSON file.")
    parser.add_argument(
        "--database-url",
        default=os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL"),
        help="Postgres DSN. Defaults to EXTRACTED_DATABASE_URL or DATABASE_URL.",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "json"),
        default="auto",
        help="Blueprint file format.",
    )
    parser.add_argument(
        "--account-id",
        required=True,
        help="Tenant/account id that owns the saved blueprints.",
    )
    parser.add_argument(
        "--target-mode",
        default=None,
        help="Default target_mode for rows that omit target_mode.",
    )
    parser.add_argument(
        "--topic-type",
        default=None,
        help="Default topic_type for rows that omit topic_type.",
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
            "asyncpg is required to load blog blueprints; install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def _main() -> int:
    args = _parse_args()
    loaded = load_blog_blueprints_from_file(
        args.path,
        file_format=args.format,
        target_mode=args.target_mode,
        topic_type=args.topic_type,
    )
    saved_ids: tuple[str, ...] = ()
    pool = None
    if not args.dry_run:
        if not args.database_url:
            raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")
        pool = await _create_pool(args.database_url)
    try:
        if not args.dry_run:
            repo = PostgresBlogBlueprintRepository(pool=pool)
            saved_ids = tuple(
                await repo.save_blueprints(
                    loaded.blueprints,
                    scope=TenantScope(account_id=args.account_id),
                )
            )
    finally:
        close = getattr(pool, "close", None)
        if close is not None:
            maybe_awaitable = close()
            if hasattr(maybe_awaitable, "__await__"):
                await maybe_awaitable

    summary = {
        **loaded.as_dict(),
        "dry_run": bool(args.dry_run),
        "account_id": args.account_id,
        "saved": len(saved_ids),
        "saved_ids": list(saved_ids),
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        mode = "would save" if args.dry_run else "saved"
        print(
            f"{mode} {summary['loaded']} blog blueprint row(s); "
            f"skipped {summary['skipped']}; warnings {len(summary['warnings'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
