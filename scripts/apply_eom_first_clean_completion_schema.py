#!/usr/bin/env python3
"""Run one approved DBA-only EOM guarded schema migration safely.

The normal Atlas runtime must never gain the authority needed to create a
foreign key to the guard-owned handoff table or to own immutable completion
or Terms evidence. This command defaults to a read-only preflight and applies
only the selected allowlisted migration after an explicit ``--apply`` using a
protected typed DBA DSN configuration.
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
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_brain.config import (  # noqa: E402
    EOM_FIRST_CLEAN_COMPLETION_DBA_DATABASE_URL_ENV,
    EOM_FIRST_CLEAN_COMPLETION_DBA_SCHEMA_ENV,
    EOMFirstCleanCompletionDBAConfig,
)
from atlas_brain.eom_api.config import EOMFunnelConfig  # noqa: E402
from atlas_brain.storage.migrations import (  # noqa: E402
    _MIGRATIONS_ADVISORY_LOCK_KEY,
    run_migrations,
)


DBA_DSN_ENV = EOM_FIRST_CLEAN_COMPLETION_DBA_DATABASE_URL_ENV
DBA_SCHEMA_ENV = EOM_FIRST_CLEAN_COMPLETION_DBA_SCHEMA_ENV
FUNNEL_DSN_ENV = "ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING"
MIGRATION_NAME = "394_eom_first_clean_completion_receipts"
TERMS_AUTHORITY_MIGRATION_NAME = "396_eom_terms_authority"
TERMS_ACCEPTANCE_MIGRATION_NAME = "397_eom_terms_acceptance"
CARD_VAULT_MIGRATION_NAME = "398_eom_card_vault"
CONTROLLED_MIGRATION_NAMES = frozenset(
    {
        MIGRATION_NAME,
        TERMS_AUTHORITY_MIGRATION_NAME,
        TERMS_ACCEPTANCE_MIGRATION_NAME,
        CARD_VAULT_MIGRATION_NAME,
    }
)
CONTROLLED_MIGRATION_PREDECESSORS = {
    TERMS_ACCEPTANCE_MIGRATION_NAME: (TERMS_AUTHORITY_MIGRATION_NAME,),
    CARD_VAULT_MIGRATION_NAME: (
        "395_eom_post_clean_onboarding_candidates",
        TERMS_ACCEPTANCE_MIGRATION_NAME,
    ),
}


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

        return (
            self.database_name,
            self.database_oid,
        )

    @property
    def uses_direct_atlas_login(self) -> bool:
        """Return whether this is the exact runtime session the route requires."""

        return self.current_user == "atlas" and self.session_user == "atlas"


class _PinnedMigrationPool:
    """Expose one transaction-pinned connection to the canonical runner."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    async def acquire(self) -> Any:
        return self._connection

    async def release(self, connection: Any) -> None:
        if connection is not self._connection:
            raise RuntimeError("Controlled migration attempted to release another connection")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight or apply one approved DBA-only EOM schema migration."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Run the selected migration after the read-only DBA preflight.",
    )
    parser.add_argument(
        "--migration",
        choices=sorted(CONTROLLED_MIGRATION_NAMES),
        default=MIGRATION_NAME,
        help="Allowlisted controlled migration to preflight or apply.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the redacted preflight/result payload as JSON.",
    )
    return parser.parse_args(argv)


def _require_schema_name(value: object, *, source: str) -> str:
    """Accept one canonical PostgreSQL identifier from a named trusted source."""

    if not isinstance(value, str):
        raise RuntimeError(f"Missing or invalid schema from {source}")
    schema_name = value.strip()
    if not schema_name or not schema_name.isascii() or not schema_name.isidentifier():
        raise RuntimeError(f"Missing or invalid schema from {source}")
    return schema_name


def _schema_search_path(schema_name: str) -> str:
    """Render the validated schema for an asyncpg startup setting."""

    return f'"{schema_name}", pg_catalog'


def _safe_target_label(database_url: str, *, schema_name: str) -> str:
    """Return a target label that never includes credentials or query values."""

    try:
        parsed = urlsplit(database_url)
        host = parsed.hostname or "connection-string"
        port = f":{parsed.port}" if parsed.port else ""
        database = parsed.path.lstrip("/") or "<database>"
        return f"dsn={host}{port}/{database};schema={schema_name}"
    except ValueError:
        return f"dsn=<configured>;schema={schema_name}"


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
        # The runner deliberately supports PgBouncer transaction pooling. A
        # prepared-statement cache could reuse a statement name on a different
        # backend after an acquire/release boundary, including after DDL has
        # committed but before the final target attestation.
        "statement_cache_size": 0,
    }
    if schema_name is not None:
        pool_kwargs["server_settings"] = {
            "search_path": _schema_search_path(schema_name)
        }
    return await asyncpg.create_pool(**pool_kwargs)


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
    schema_name = _require_schema_name(
        row["schema_name"],
        source=f"{source} current_schema()",
    )
    database_name = _require_database_name(row["database_name"], source=source)
    database_oid = _require_database_oid(row["database_oid"], source=source)
    current_user = _require_role_name(row["current_user"], source=source)
    session_user = _require_role_name(row["session_user"], source=source)
    return _TargetIdentity(
        schema_name=schema_name,
        database_name=database_name,
        database_oid=database_oid,
        current_user=current_user,
        session_user=session_user,
    )


async def _attest_shared_database_lock(runtime_pool: Any, dba_pool: Any) -> None:
    """Require both pools to contend on one unpredictable advisory lock.

    A database name and OID identify a database inside one cluster but can be
    duplicated in a clone. A transaction-scoped advisory lock is held in the
    shared memory of exactly one live PostgreSQL cluster, so a DBA connection
    can only see the runtime connection's randomly selected key as busy when
    both pools reach that same database. Holding the explicit transaction keeps
    the proof valid through a transaction-pooling proxy and releases the key
    automatically before this preflight returns or raises.
    """

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
                    # An unrelated process could have selected this key. Pick a
                    # fresh one rather than treating that rare collision as proof.
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
    raise RuntimeError(
        "Could not reserve a fresh EOM funnel target-attestation lock"
    )


async def _migration_state(
    pool: Any,
    *,
    expected_target: _TargetIdentity,
    migration_name: str,
) -> tuple[bool, bool]:
    """Return executor/migration state after re-attesting its full target."""

    observed_target = await _target_identity(
        pool,
        source="controlled DBA pool",
    )
    if observed_target.schema_name != expected_target.schema_name:
        raise RuntimeError(
            "Controlled DBA pool did not resolve to the configured EOM funnel schema"
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
                    FROM pg_catalog.pg_roles AS role
                    WHERE role.rolname = current_user
                ), FALSE)
                """
            )
        )
        migrations_table_exists = bool(
            await connection.fetchval(
                "SELECT pg_catalog.to_regclass('schema_migrations') IS NOT NULL"
            )
        )
        migration_recorded = False
        if migrations_table_exists:
            migration_recorded = bool(
                await connection.fetchval(
                    "SELECT EXISTS (SELECT 1 FROM schema_migrations WHERE name = $1)",
                    migration_name,
                )
            )
    return executor_is_superuser, migration_recorded


async def _run_pinned_controlled_migration(
    pool: Any,
    *,
    migration_name: str,
    run_migrations_fn: Callable[..., Awaitable[None]],
) -> None:
    """Run one allowlisted migration in the canonical pinned transaction.

    ``run_migrations`` uses a session advisory lock because ordinary catalog
    runs can contain ``CREATE INDEX CONCURRENTLY``. Every migration admitted by
    this controlled runner is atomic and cannot contain concurrent DDL, so the
    runner can first acquire that same lock inside an explicit transaction,
    then pin the canonical runner to that connection. A transaction-pooling
    proxy consequently keeps the lock, migration SQL, and bookkeeping on one
    backend; the outer release removes the extra re-entrant session lock before
    the transaction ends.
    """

    async with pool.acquire() as connection:
        while True:
            lock_acquired = False
            async with connection.transaction():
                lock_acquired = bool(
                    await connection.fetchval(
                        "SELECT pg_catalog.pg_try_advisory_lock($1)",
                        _MIGRATIONS_ADVISORY_LOCK_KEY,
                    )
                )
                if lock_acquired:
                    try:
                        await run_migrations_fn(
                            _PinnedMigrationPool(connection),
                            only=(migration_name,),
                        )
                    finally:
                        released = bool(
                            await connection.fetchval(
                                "SELECT pg_catalog.pg_advisory_unlock($1)",
                                _MIGRATIONS_ADVISORY_LOCK_KEY,
                            )
                        )
                        if not released:
                            raise RuntimeError(
                                "Controlled migration did not release its "
                                "database serialization lock"
                            )
                    return
            # Do not wait while holding an open transaction: another ordinary
            # migration may need that absence to finish CONCURRENTLY work.
            await asyncio.sleep(0.2)


async def _card_vault_schema_ready(pool: Any) -> bool:
    """Load the card-vault attestation only for migration 398 operations."""

    from atlas_brain.services.eom_card_vault import eom_card_vault_schema_ready

    return await eom_card_vault_schema_ready(pool)


async def _run(
    args: argparse.Namespace,
    *,
    create_pool: Callable[..., Awaitable[Any]] = _create_pool,
    run_migrations_fn: Callable[..., Awaitable[None]] = run_migrations,
    config_factory: Callable[[], EOMFirstCleanCompletionDBAConfig] = (
        EOMFirstCleanCompletionDBAConfig
    ),
    funnel_config_factory: Callable[[], EOMFunnelConfig] = EOMFunnelConfig,
    card_vault_schema_ready_fn: Callable[[Any], Awaitable[bool]] = (
        _card_vault_schema_ready
    ),
) -> dict[str, object]:
    migration_name = str(args.migration)
    if migration_name not in CONTROLLED_MIGRATION_NAMES:
        raise RuntimeError("Refusing a non-allowlisted controlled EOM migration")
    config = config_factory()
    database_url = config.database_url.get_secret_value().strip()
    if not database_url:
        raise RuntimeError(f"Missing protected DBA DSN configuration {DBA_DSN_ENV}")
    schema_name = _require_schema_name(config.schema_name, source=DBA_SCHEMA_ENV)
    funnel_database_url = funnel_config_factory().db_connection_string.strip()
    if not funnel_database_url:
        raise RuntimeError(
            f"Missing EOM funnel runtime DSN configuration {FUNNEL_DSN_ENV}"
        )

    runtime_pool = await create_pool(funnel_database_url)
    try:
        runtime_target = await _target_identity(
            runtime_pool,
            source="EOM funnel runtime",
        )
        if not runtime_target.uses_direct_atlas_login:
            raise RuntimeError(
                "EOM funnel runtime connection must use the direct atlas login"
            )
        if runtime_target.schema_name != schema_name:
            raise RuntimeError(
                "Configured controlled DBA schema does not match the EOM funnel "
                "runtime schema"
            )
        pool = await create_pool(database_url, schema_name=schema_name)
        try:
            await _attest_shared_database_lock(runtime_pool, pool)
            executor_is_superuser, migration_recorded = await _migration_state(
                pool,
                expected_target=runtime_target,
                migration_name=migration_name,
            )
            predecessors = CONTROLLED_MIGRATION_PREDECESSORS.get(migration_name, ())
            for predecessor in predecessors:
                _executor_is_superuser, predecessor_recorded = await _migration_state(
                    pool,
                    expected_target=runtime_target,
                    migration_name=predecessor,
                )
                if not predecessor_recorded:
                    raise RuntimeError(
                        f"Controlled predecessor {predecessor} must be recorded "
                        f"before {migration_name}"
                    )
            result: dict[str, object] = {
                "target": _safe_target_label(database_url, schema_name=schema_name),
                "executor_is_superuser": executor_is_superuser,
                "migration": migration_name,
                "migration_recorded": migration_recorded,
                "applied": False,
            }
            if not executor_is_superuser:
                raise RuntimeError(
                    "Configured DBA connection is not a PostgreSQL superuser; "
                    "refusing to run the controlled EOM schema migration"
                )
            if args.apply and not migration_recorded:
                await _run_pinned_controlled_migration(
                    pool,
                    migration_name=migration_name,
                    run_migrations_fn=run_migrations_fn,
                )
                await _attest_shared_database_lock(runtime_pool, pool)
                _executor_is_superuser, migration_recorded = await _migration_state(
                    pool,
                    expected_target=runtime_target,
                    migration_name=migration_name,
                )
                if not migration_recorded:
                    raise RuntimeError(
                        "Migration runner returned without recording the EOM "
                        "controlled schema"
                    )
                result["migration_recorded"] = True
                result["applied"] = True
            if migration_name == CARD_VAULT_MIGRATION_NAME:
                schema_ready = bool(await card_vault_schema_ready_fn(runtime_pool))
                result["schema_ready"] = schema_ready
                if migration_recorded != schema_ready:
                    raise RuntimeError(
                        "Card-vault migration bookkeeping does not match the "
                        "runtime-role schema attestation"
                    )
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
            f"recorded={result['migration_recorded']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
