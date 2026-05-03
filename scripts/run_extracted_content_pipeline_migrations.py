#!/usr/bin/env python3
"""Apply extracted AI Content Ops SQL migrations."""

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

from extracted_content_pipeline.storage.migration_runner import (  # noqa: E402
    DEFAULT_MIGRATION_TABLE,
    MIGRATIONS_DIR,
    apply_content_pipeline_migrations,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply packaged extracted_content_pipeline SQL migrations."
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL"),
        help="Postgres DSN. Defaults to EXTRACTED_DATABASE_URL or DATABASE_URL.",
    )
    parser.add_argument(
        "--migrations-dir",
        type=Path,
        default=MIGRATIONS_DIR,
        help="Directory containing .sql migrations.",
    )
    parser.add_argument(
        "--migration-table",
        default=DEFAULT_MIGRATION_TABLE,
        help="Metadata table used to track applied migrations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List pending migrations without executing SQL.",
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
            "asyncpg is required to apply migrations; install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def _main() -> int:
    args = _parse_args()
    if not args.database_url:
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")
    pool = await _create_pool(args.database_url)
    try:
        result = await apply_content_pipeline_migrations(
            pool,
            migrations_dir=args.migrations_dir,
            migration_table=args.migration_table,
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
        mode = "would apply" if result.dry_run else "applied"
        print(
            f"{mode} {len(result.applied)} migration(s); "
            f"skipped {len(result.skipped)} already-applied migration(s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
