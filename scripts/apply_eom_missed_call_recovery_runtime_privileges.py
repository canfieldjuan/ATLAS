#!/usr/bin/env python3
"""Run the DBA-only EOM missed-call recovery privilege repair safely.

The normal Atlas runtime must never be given the database authority required by
migration 393 or its exact historical migration prerequisites. This command
defaults to a read-only preflight. With ``--apply``, it first lets the existing
historical selector apply only an attested EOM recovery prelude, then applies
migration 393 after migration 389 is recorded, using a protected DBA DSN.
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
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_brain.storage.migrations import (  # noqa: E402
    PendingMigrationContentIntegrityError,
    run_migrations,
)


DEFAULT_DSN_ENV = "ATLAS_EOM_MISSED_CALL_RECOVERY_DBA_DATABASE_URL"
DEFAULT_RUNTIME_DATABASE_URL_ENV = "ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING"
RUNTIME_ROLE_SETTING = "atlas.eom_missed_call_recovery_runtime_role"
PREREQUISITE_MIGRATION_NAME = "389_eom_missed_call_recovery"
MIGRATION_NAME = "393_eom_missed_call_recovery_runtime_privileges"
HISTORICAL_PRELUDE_MIGRATION_NAMES: tuple[str, ...] = (
    "390_eom_won_loss_direct_sql_fence_recovery",
    "391_eom_commercial_billing_run_fence_recovery",
    "392_eom_commercial_billing_run_fence_schema_binding",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or apply the DBA-only EOM missed-call recovery privilege repair."
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
        help=(
            "Apply an exact historical EOM recovery prelude if required, then "
            "run migration 393 after the read-only DBA preflight."
        ),
    )
    parser.add_argument(
        "--runtime-database-url-env",
        default=DEFAULT_RUNTIME_DATABASE_URL_ENV,
        help=(
            "Environment variable holding the configured EOM funnel PostgreSQL "
            "DSN; its username selects the runtime role granted by migration 393 "
            f"(default: {DEFAULT_RUNTIME_DATABASE_URL_ENV})."
        ),
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


def _runtime_role_from_database_url(database_url: str) -> str:
    """Extract the configured EOM runtime login without retaining its DSN."""

    try:
        runtime_role = unquote(urlsplit(database_url).username or "").strip()
    except ValueError as exc:
        raise RuntimeError("Configured EOM runtime DSN is invalid") from exc
    if not runtime_role:
        raise RuntimeError("Configured EOM runtime DSN must include a username")
    return runtime_role


async def _create_pool(database_url: str) -> Any:
    try:
        import asyncpg
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "asyncpg is required to run the DBA migration preflight"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=1)


class _RuntimeRoleMigrationPool:
    """Bind the configured runtime role to every controlled migration session."""

    def __init__(self, pool: Any, runtime_role: str) -> None:
        self._pool = pool
        self._runtime_role = runtime_role

    async def acquire(self) -> Any:
        connection = await self._pool.acquire()
        try:
            await connection.execute(
                "SELECT pg_catalog.set_config($1, $2, FALSE)",
                RUNTIME_ROLE_SETTING,
                self._runtime_role,
            )
        except Exception:
            await self._pool.release(connection)
            raise
        return connection

    async def release(self, connection: Any) -> None:
        await self._pool.release(connection)


async def _ensure_pgcrypto(pool: Any) -> None:
    """Establish the historical-prelude SHA-256 dependency under the DBA DSN."""

    async with pool.acquire() as connection:
        await connection.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")


async def _migration_state(
    pool: Any,
) -> tuple[bool, bool, bool, bool, dict[str, bool]]:
    """Return executor, ledger, and exact EOM migration state without writes."""

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
        historical_prelude_migration_records = {
            name: False for name in HISTORICAL_PRELUDE_MIGRATION_NAMES
        }
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
            for historical_migration_name in HISTORICAL_PRELUDE_MIGRATION_NAMES:
                historical_prelude_migration_records[historical_migration_name] = bool(
                    await connection.fetchval(
                        "SELECT EXISTS (SELECT 1 FROM schema_migrations WHERE name = $1)",
                        historical_migration_name,
                    )
                )
    return (
        executor_is_superuser,
        migrations_table_exists,
        prerequisite_migration_recorded,
        migration_recorded,
        historical_prelude_migration_records,
    )


async def _apply_required_historical_prelude(
    pool: Any,
    *,
    migration_pool: Any,
    run_migrations_fn: Callable[..., Awaitable[None]],
) -> None:
    """Apply at most the three explicitly authorized historical EOM preludes.

    ``run_migrations`` deliberately commits one selected historical recovery per
    invocation. Re-read the ledger after each invocation and stop once the
    selector has no exact prelude to apply. This command never includes 389 in
    its requested set, so normal bootstrap ownership remains with the slim EOM
    runtime path.
    """

    for _ in HISTORICAL_PRELUDE_MIGRATION_NAMES:
        _, _, _, _, before = await _migration_state(pool)
        try:
            await run_migrations_fn(
                migration_pool,
                only=HISTORICAL_PRELUDE_MIGRATION_NAMES,
            )
        except PendingMigrationContentIntegrityError:
            # The generic runner intentionally commits one selected historical
            # recovery, then raises while another evidence gap remains. Admit
            # only that exact committed-progress stop; every no-progress
            # integrity failure remains fail-closed.
            _, _, _, _, after = await _migration_state(pool)
            if not any(
                not before[name] and after[name]
                for name in HISTORICAL_PRELUDE_MIGRATION_NAMES
            ):
                raise
            continue
        _, _, _, _, after = await _migration_state(pool)
        if all(
            after[name] == before[name] for name in HISTORICAL_PRELUDE_MIGRATION_NAMES
        ):
            return


async def _run(
    args: argparse.Namespace,
    *,
    create_pool: Callable[[str], Awaitable[Any]] = _create_pool,
    run_migrations_fn: Callable[..., Awaitable[None]] = run_migrations,
) -> dict[str, object]:
    database_url = os.environ.get(args.database_url_env, "").strip()
    if not database_url:
        raise RuntimeError(
            f"Missing protected DBA DSN environment variable {args.database_url_env}"
        )
    runtime_database_url = os.environ.get(args.runtime_database_url_env, "").strip()
    if not runtime_database_url:
        raise RuntimeError(
            "Missing configured EOM runtime DSN environment variable "
            f"{args.runtime_database_url_env}"
        )
    runtime_role = _runtime_role_from_database_url(runtime_database_url)

    pool = await create_pool(database_url)
    migration_pool = _RuntimeRoleMigrationPool(pool, runtime_role)
    try:
        (
            executor_is_superuser,
            migrations_table_exists,
            prerequisite_migration_recorded,
            migration_recorded,
            historical_prelude_migration_records,
        ) = await _migration_state(pool)
        result: dict[str, object] = {
            "target": _safe_target_label(database_url),
            "executor_is_superuser": executor_is_superuser,
            "runtime_role": runtime_role,
            "historical_prelude_migrations": historical_prelude_migration_records,
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
        if args.apply and migrations_table_exists and not migration_recorded:
            await _ensure_pgcrypto(pool)
            await _apply_required_historical_prelude(
                pool,
                migration_pool=migration_pool,
                run_migrations_fn=run_migrations_fn,
            )
            (
                _,
                _,
                prerequisite_migration_recorded,
                migration_recorded,
                historical_prelude_migration_records,
            ) = await _migration_state(pool)
            result["historical_prelude_migrations"] = (
                historical_prelude_migration_records
            )
        if args.apply and not prerequisite_migration_recorded:
            raise RuntimeError(
                f"Migration {PREREQUISITE_MIGRATION_NAME} is not recorded; "
                "run the slim EOM bootstrap before the privilege repair"
            )
        if args.apply and not migration_recorded:
            await run_migrations_fn(migration_pool, only=(MIGRATION_NAME,))
            (
                _,
                _,
                _,
                migration_recorded,
                historical_prelude_migration_records,
            ) = await _migration_state(pool)
            if not migration_recorded:
                raise RuntimeError(
                    "Migration runner returned without recording the EOM "
                    "missed-call recovery privilege repair"
                )
            result["migration_recorded"] = True
            result["historical_prelude_migrations"] = (
                historical_prelude_migration_records
            )
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
            f"prerequisite_recorded={result['prerequisite_migration_recorded']}; "
            f"historical_preludes={result['historical_prelude_migrations']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
