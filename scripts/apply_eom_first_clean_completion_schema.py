#!/usr/bin/env python3
"""Run the DBA-only EOM first-clean completion schema safely.

The normal Atlas runtime must never gain the authority needed to create a
foreign key to the guard-owned handoff table or to own immutable completion
evidence. This command defaults to a read-only preflight and applies only
migration 394 after an explicit ``--apply`` using a protected typed DBA DSN
configuration.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_brain.config import (  # noqa: E402
    EOM_FIRST_CLEAN_COMPLETION_DBA_DATABASE_URL_ENV,
    EOMFirstCleanCompletionDBAConfig,
)
from atlas_brain.storage.migrations import run_migrations  # noqa: E402


DBA_DSN_ENV = EOM_FIRST_CLEAN_COMPLETION_DBA_DATABASE_URL_ENV
MIGRATION_NAME = "394_eom_first_clean_completion_receipts"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or apply the DBA-only EOM first-clean completion schema."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Run migration 394 after the read-only DBA preflight.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the redacted preflight/result payload as JSON.",
    )
    return parser.parse_args(argv)


def _safe_target_label(database_url: str) -> str:
    """Return a target label that never includes credentials or query values."""

    try:
        parsed = urlsplit(database_url)
        host = parsed.hostname or "connection-string"
        port = f":{parsed.port}" if parsed.port else ""
        database = parsed.path.lstrip("/") or "<database>"
        return f"dsn={host}{port}/{database}"
    except ValueError:
        return "dsn=<configured>"


async def _create_pool(database_url: str) -> Any:
    try:
        import asyncpg
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "asyncpg is required to run the DBA migration preflight"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=1)


async def _migration_state(pool: Any) -> tuple[bool, bool]:
    """Return (executor_is_superuser, migration_recorded) without writes."""

    async with pool.acquire() as connection:
        executor_is_superuser = bool(
            await connection.fetchval(
                """
                SELECT COALESCE((
                    SELECT role.rolsuper
                    FROM pg_roles AS role
                    WHERE role.rolname = current_user
                ), FALSE)
                """
            )
        )
        migrations_table_exists = bool(
            await connection.fetchval(
                "SELECT to_regclass('schema_migrations') IS NOT NULL"
            )
        )
        migration_recorded = False
        if migrations_table_exists:
            migration_recorded = bool(
                await connection.fetchval(
                    "SELECT EXISTS (SELECT 1 FROM schema_migrations WHERE name = $1)",
                    MIGRATION_NAME,
                )
            )
    return executor_is_superuser, migration_recorded


async def _run(
    args: argparse.Namespace,
    *,
    create_pool: Callable[[str], Awaitable[Any]] = _create_pool,
    run_migrations_fn: Callable[..., Awaitable[None]] = run_migrations,
    config_factory: Callable[[], EOMFirstCleanCompletionDBAConfig] = (
        EOMFirstCleanCompletionDBAConfig
    ),
) -> dict[str, object]:
    database_url = config_factory().database_url.get_secret_value().strip()
    if not database_url:
        raise RuntimeError(
            f"Missing protected DBA DSN configuration {DBA_DSN_ENV}"
        )

    pool = await create_pool(database_url)
    try:
        executor_is_superuser, migration_recorded = await _migration_state(pool)
        result: dict[str, object] = {
            "target": _safe_target_label(database_url),
            "executor_is_superuser": executor_is_superuser,
            "migration": MIGRATION_NAME,
            "migration_recorded": migration_recorded,
            "applied": False,
        }
        if not executor_is_superuser:
            raise RuntimeError(
                "Configured DBA connection is not a PostgreSQL superuser; "
                "refusing to run the first-clean completion schema migration"
            )
        if args.apply and not migration_recorded:
            await run_migrations_fn(pool, only=(MIGRATION_NAME,))
            _executor_is_superuser, migration_recorded = await _migration_state(pool)
            if not migration_recorded:
                raise RuntimeError(
                    "Migration runner returned without recording the EOM "
                    "first-clean completion schema"
                )
            result["migration_recorded"] = True
            result["applied"] = True
        return result
    finally:
        await pool.close()


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = await _run(args)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        action = "applied" if result["applied"] else "checked"
        print(
            f"{action} {result['migration']} on {result['target']}; "
            f"recorded={result['migration_recorded']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
