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
from dataclasses import dataclass
import json
import secrets
from collections.abc import Awaitable, Callable
from pathlib import Path
import sys
from typing import Any
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_brain.config import (  # noqa: E402
    EOM_MISSED_CALL_RECOVERY_DBA_DATABASE_URL_ENV,
    EOMMissedCallRecoveryDBAConfig,
)
from atlas_brain.eom_api.config import EOMFunnelConfig  # noqa: E402
from atlas_brain.storage.migrations import (  # noqa: E402
    PendingMigrationContentIntegrityError,
    run_migrations,
)


DBA_DSN_ENV = EOM_MISSED_CALL_RECOVERY_DBA_DATABASE_URL_ENV
FUNNEL_DSN_ENV = "ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING"
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
        "--apply",
        action="store_true",
        help=(
            "Apply an exact historical EOM recovery prelude if required, then "
            "run migration 393 after the read-only DBA preflight."
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


def _require_schema_name(value: object, *, source: str) -> str:
    """Accept one canonical PostgreSQL identifier from a trusted query result."""

    if not isinstance(value, str):
        raise RuntimeError(f"Missing or invalid schema from {source}")
    schema_name = value.strip()
    if not schema_name or not schema_name.isascii() or not schema_name.isidentifier():
        raise RuntimeError(f"Missing or invalid schema from {source}")
    return schema_name


def _schema_search_path(schema_name: str) -> str:
    """Render the validated schema for an asyncpg startup setting."""

    return f'"{schema_name}", pg_catalog'


def _require_database_name(value: object, *, source: str) -> str:
    """Reject a missing database name rather than comparing an ambiguous target."""

    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Missing or invalid database identity from {source}")
    return value


def _require_database_oid(value: object, *, source: str) -> int:
    """Accept only one positive PostgreSQL database OID."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"Missing or invalid database identity from {source}")
    return value


def _require_role_name(value: object, *, source: str) -> str:
    """Require a non-empty PostgreSQL role name from a catalog identity row."""

    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Missing or invalid session identity from {source}")
    return value


@dataclass(frozen=True)
class _TargetIdentity:
    """One connection's canonical EOM database and schema target."""

    schema_name: str
    database_name: str
    database_oid: int
    current_user: str
    session_user: str

    @property
    def database_identity(self) -> tuple[str, int]:
        """Return only fields that must match across runtime and DBA pools."""

        return (self.database_name, self.database_oid)


async def _create_pool(
    database_url: str,
    *,
    schema_name: str | None = None,
) -> Any:
    try:
        import asyncpg
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "asyncpg is required to run the DBA migration preflight"
        ) from exc
    pool_kwargs: dict[str, Any] = {
        "dsn": database_url,
        "min_size": 1,
        "max_size": 1,
        "statement_cache_size": 0,
    }
    if schema_name is not None:
        pool_kwargs["server_settings"] = {
            "search_path": _schema_search_path(schema_name)
        }
    return await asyncpg.create_pool(**pool_kwargs)


async def _target_identity(pool: Any, *, source: str) -> _TargetIdentity:
    """Read the current schema and database identity without DDL."""

    async with pool.acquire() as connection:
        row = await connection.fetchrow(
            """
            SELECT pg_catalog.current_schema() AS schema_name,
                   pg_catalog.current_database() AS database_name,
                   CURRENT_USER AS current_user,
                   SESSION_USER AS session_user,
                   (
                       SELECT oid
                         FROM pg_catalog.pg_database
                        WHERE datname = pg_catalog.current_database()
                   ) AS database_oid
            """
        )
    if row is None:
        raise RuntimeError(f"Missing or invalid database identity from {source}")
    return _TargetIdentity(
        schema_name=_require_schema_name(
            row["schema_name"],
            source=f"{source} current_schema()",
        ),
        database_name=_require_database_name(row["database_name"], source=source),
        database_oid=_require_database_oid(row["database_oid"], source=source),
        current_user=_require_role_name(row["current_user"], source=source),
        session_user=_require_role_name(row["session_user"], source=source),
    )


async def _attest_shared_database_lock(runtime_pool: Any, dba_pool: Any) -> None:
    """Require both pools to contend on one unpredictable advisory lock."""

    for _attempt in range(3):
        lock_key = secrets.randbits(63)
        async with runtime_pool.acquire() as runtime_connection:
            async with runtime_connection.transaction():
                runtime_acquired = bool(
                    await runtime_connection.fetchval(
                        "SELECT pg_catalog.pg_try_advisory_xact_lock($1)",
                        lock_key,
                    )
                )
                if not runtime_acquired:
                    continue
                async with dba_pool.acquire() as dba_connection:
                    dba_acquired = bool(
                        await dba_connection.fetchval(
                            "SELECT pg_catalog.pg_try_advisory_xact_lock($1)",
                            lock_key,
                        )
                    )
                    if dba_acquired:
                        raise RuntimeError(
                            "Controlled DBA pool does not share the EOM "
                            "funnel runtime database"
                        )
                return
    raise RuntimeError("Could not reserve a fresh EOM funnel target-attestation lock")


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


async def _require_role_admission_before_mutation(pool: Any, runtime_role: str) -> None:
    """Fail closed before extension or prelude writes on an invalid role boundary.

    Migration 393 repeats these admissions authoritatively inside its own
    transaction. The runner needs the same read-only boundary before it creates
    ``pgcrypto`` for a historical prelude that runs before 393.
    """

    async with pool.acquire() as connection:
        admissions = await connection.fetchrow(
            """
            -- atlas:eom-missed-call-recovery-role-admission
            SELECT
                EXISTS (
                    SELECT 1
                      FROM pg_catalog.pg_roles AS guard_role
                     WHERE guard_role.rolname = 'atlas_eom_handoff_owner'
                       AND NOT guard_role.rolcanlogin
                       AND NOT guard_role.rolinherit
                       AND NOT guard_role.rolsuper
                       AND NOT guard_role.rolcreaterole
                       AND NOT guard_role.rolcreatedb
                       AND NOT guard_role.rolreplication
                       AND NOT guard_role.rolbypassrls
                       AND NOT EXISTS (
                           SELECT 1
                             FROM pg_catalog.pg_roles AS member_role
                            WHERE member_role.rolcanlogin
                              AND NOT member_role.rolsuper
                              AND pg_catalog.pg_has_role(
                                  member_role.oid, guard_role.oid, 'MEMBER'
                              )
                       )
                ) AS guard_role_ready,
                EXISTS (
                    SELECT 1
                      FROM pg_catalog.pg_roles AS runtime_role_state
                     WHERE runtime_role_state.rolname = $1
                       AND runtime_role_state.rolcanlogin
                       AND runtime_role_state.rolname <> 'atlas_nocodb'
                       AND NOT runtime_role_state.rolsuper
                       AND NOT runtime_role_state.rolcreaterole
                       AND NOT runtime_role_state.rolcreatedb
                       AND NOT runtime_role_state.rolreplication
                       AND NOT runtime_role_state.rolbypassrls
                       AND NOT EXISTS (
                           SELECT 1
                             FROM pg_catalog.pg_auth_members AS membership
                             JOIN pg_catalog.pg_roles AS guard_role
                               ON guard_role.oid = membership.roleid
                            WHERE membership.member = runtime_role_state.oid
                              AND guard_role.rolname = 'atlas_eom_handoff_owner'
                       )
                       AND NOT EXISTS (
                           SELECT 1
                             FROM pg_catalog.pg_roles AS delegating_login
                            WHERE delegating_login.rolcanlogin
                              AND NOT delegating_login.rolsuper
                              AND delegating_login.oid <> runtime_role_state.oid
                              AND pg_catalog.pg_has_role(
                                  delegating_login.oid, runtime_role_state.oid, 'MEMBER'
                              )
                       )
                ) AS runtime_role_ready,
                EXISTS (
                    SELECT 1
                      FROM pg_catalog.pg_roles AS nocodb_role
                     WHERE nocodb_role.rolname = 'atlas_nocodb'
                       AND nocodb_role.rolcanlogin
                       AND NOT nocodb_role.rolsuper
                       AND NOT nocodb_role.rolcreaterole
                       AND NOT nocodb_role.rolcreatedb
                       AND NOT nocodb_role.rolreplication
                       AND NOT nocodb_role.rolbypassrls
                       AND NOT nocodb_role.rolinherit
                       AND NOT EXISTS (
                           SELECT 1
                             FROM pg_catalog.pg_auth_members AS membership
                            WHERE membership.member = nocodb_role.oid
                       )
                       AND NOT EXISTS (
                           SELECT 1
                             FROM pg_catalog.pg_roles AS delegating_login
                            WHERE delegating_login.rolcanlogin
                              AND NOT delegating_login.rolsuper
                              AND delegating_login.oid <> nocodb_role.oid
                              AND pg_catalog.pg_has_role(
                                  delegating_login.oid, nocodb_role.oid, 'MEMBER'
                              )
                       )
                ) AS nocodb_role_ready
            """,
            runtime_role,
        )
    if admissions is None or not all(
        bool(admissions[name])
        for name in (
            "guard_role_ready",
            "runtime_role_ready",
            "nocodb_role_ready",
        )
    ):
        raise RuntimeError(
            "Guard/runtime/NocoDB role admission failed; refusing to create "
            "pgcrypto or run historical recovery"
        )


async def _migration_state(
    pool: Any,
    *,
    expected_target: _TargetIdentity,
) -> tuple[bool, bool, bool, bool, dict[str, bool]]:
    """Return executor and ledger state after re-attesting the DBA target."""

    observed_target = await _target_identity(
        pool,
        source="controlled DBA pool",
    )
    if observed_target.schema_name != expected_target.schema_name:
        raise RuntimeError(
            "Controlled DBA pool did not resolve to the EOM funnel runtime schema"
        )
    if observed_target.database_identity != expected_target.database_identity:
        raise RuntimeError(
            "Controlled DBA pool does not target the EOM funnel runtime database"
        )

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
    expected_target: _TargetIdentity,
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
        _, _, _, _, before = await _migration_state(
            pool,
            expected_target=expected_target,
        )
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
            _, _, _, _, after = await _migration_state(
                pool,
                expected_target=expected_target,
            )
            if not any(
                not before[name] and after[name]
                for name in HISTORICAL_PRELUDE_MIGRATION_NAMES
            ):
                raise
            continue
        _, _, _, _, after = await _migration_state(
            pool,
            expected_target=expected_target,
        )
        if all(
            after[name] == before[name] for name in HISTORICAL_PRELUDE_MIGRATION_NAMES
        ):
            return


async def _run(
    args: argparse.Namespace,
    *,
    create_pool: Callable[..., Awaitable[Any]] = _create_pool,
    run_migrations_fn: Callable[..., Awaitable[None]] = run_migrations,
    config_factory: Callable[[], EOMMissedCallRecoveryDBAConfig] = (
        EOMMissedCallRecoveryDBAConfig
    ),
    funnel_config_factory: Callable[[], EOMFunnelConfig] = EOMFunnelConfig,
) -> dict[str, object]:
    config = config_factory()
    database_url = config.database_url.get_secret_value().strip()
    if not database_url:
        raise RuntimeError(f"Missing protected DBA DSN configuration {DBA_DSN_ENV}")
    runtime_database_url = funnel_config_factory().db_connection_string.strip()
    if not runtime_database_url:
        raise RuntimeError(
            f"Missing EOM funnel runtime DSN configuration {FUNNEL_DSN_ENV}"
        )
    runtime_role = _runtime_role_from_database_url(runtime_database_url)

    runtime_pool = await create_pool(runtime_database_url)
    try:
        runtime_target = await _target_identity(
            runtime_pool,
            source="EOM funnel runtime",
        )
        if (
            runtime_target.current_user != runtime_role
            or runtime_target.session_user != runtime_role
        ):
            raise RuntimeError(
                "EOM funnel runtime connection must use its direct configured login"
            )
        pool = await create_pool(database_url, schema_name=runtime_target.schema_name)
        migration_pool = _RuntimeRoleMigrationPool(pool, runtime_role)
        try:
            await _attest_shared_database_lock(runtime_pool, pool)
            (
                executor_is_superuser,
                migrations_table_exists,
                prerequisite_migration_recorded,
                migration_recorded,
                historical_prelude_migration_records,
            ) = await _migration_state(pool, expected_target=runtime_target)
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
                await _require_role_admission_before_mutation(pool, runtime_role)
                await _ensure_pgcrypto(pool)
                await _apply_required_historical_prelude(
                    pool,
                    expected_target=runtime_target,
                    migration_pool=migration_pool,
                    run_migrations_fn=run_migrations_fn,
                )
                await _attest_shared_database_lock(runtime_pool, pool)
                (
                    _,
                    _,
                    prerequisite_migration_recorded,
                    migration_recorded,
                    historical_prelude_migration_records,
                ) = await _migration_state(pool, expected_target=runtime_target)
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
                await _attest_shared_database_lock(runtime_pool, pool)
                (
                    _,
                    _,
                    _,
                    migration_recorded,
                    historical_prelude_migration_records,
                ) = await _migration_state(pool, expected_target=runtime_target)
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
    finally:
        await runtime_pool.close()


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
