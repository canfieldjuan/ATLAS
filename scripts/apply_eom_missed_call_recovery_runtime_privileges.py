#!/usr/bin/env python3
"""Run the DBA-only EOM missed-call recovery privilege repair safely.

The normal Atlas runtime must never be given the database authority required by
migration 393. This command defaults to a read-only preflight and applies only
the selected migration after an explicit ``--apply`` using a protected DBA DSN
environment variable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_brain.storage.migrations import run_migrations  # noqa: E402


DEFAULT_DSN_ENV = "ATLAS_EOM_MISSED_CALL_RECOVERY_DBA_DATABASE_URL"
PREREQUISITE_MIGRATION_NAME = "389_eom_missed_call_recovery"
MIGRATION_NAME = "393_eom_missed_call_recovery_runtime_privileges"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or apply the DBA-only EOM missed-call recovery "
            "privilege repair."
        )
    )
    parser.add_argument(
        "--database-url-env",
        default=DEFAULT_DSN_ENV,
        help=(
            "Environment variable holding the protected DBA PostgreSQL DSN "
            f"(default: {DEFAULT_DSN_ENV})."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Run migration 393 after the read-only DBA preflight.",
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


async def _migration_state(pool: Any) -> tuple[bool, bool, bool]:
    """Return executor, prerequisite, and repair ledger state without writes."""

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
        prerequisite_migration_recorded = False
        migration_recorded = False
        if migrations_table_exists:
            prerequisite_migration_recorded = bool(
                await connection.fetchval(
                    "SELECT EXISTS (SELECT 1 FROM schema_migrations WHERE name = $1)",
                    PREREQUISITE_MIGRATION_NAME,
                )
            )
            migration_recorded = bool(
                await connection.fetchval(
                    "SELECT EXISTS (SELECT 1 FROM schema_migrations WHERE name = $1)",
                    MIGRATION_NAME,
                )
            )
    return (
        executor_is_superuser,
        prerequisite_migration_recorded,
        migration_recorded,
    )


async def _run(
    args: argparse.Namespace,
    *,
    create_pool: Callable[[str], Awaitable[Any]] = _create_pool,
    run_migrations_fn: Callable[..., Awaitable[None]] = run_migrations,
) -> dict[str, object]:
    database_url = os.environ.get(args.database_url_env, "").strip()
    if not database_url:
        raise RuntimeError(
            "Missing protected DBA DSN environment variable "
            f"{args.database_url_env}"
        )

    pool = await create_pool(database_url)
    try:
        (
            executor_is_superuser,
            prerequisite_migration_recorded,
            migration_recorded,
        ) = await _migration_state(pool)
        result: dict[str, object] = {
            "target": _safe_target_label(database_url),
            "executor_is_superuser": executor_is_superuser,
            "prerequisite_migration": PREREQUISITE_MIGRATION_NAME,
            "prerequisite_migration_recorded": prerequisite_migration_recorded,
            "migration": MIGRATION_NAME,
            "migration_recorded": migration_recorded,
            "applied": False,
        }
        if not executor_is_superuser:
            raise RuntimeError(
                "Configured DBA connection is not a PostgreSQL superuser; "
                "refusing to run the privilege repair"
            )
        if args.apply and not prerequisite_migration_recorded:
            raise RuntimeError(
                f"Migration {PREREQUISITE_MIGRATION_NAME} is not recorded; "
                "run the slim EOM bootstrap before the privilege repair"
            )
        if args.apply and not migration_recorded:
            await run_migrations_fn(pool, only=(MIGRATION_NAME,))
            _, _, migration_recorded = await _migration_state(pool)
            if not migration_recorded:
                raise RuntimeError(
                    "Migration runner returned without recording the EOM "
                    "missed-call recovery privilege repair"
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
            f"recorded={result['migration_recorded']}; "
            f"prerequisite_recorded={result['prerequisite_migration_recorded']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
