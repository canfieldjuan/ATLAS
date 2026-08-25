"""Focused proof for EOM first-clean completion receipts.

All fixtures are synthetic.  The tests use an isolated PostgreSQL schema and
never contact a calendar, email provider, Stripe, or real EOM customer.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from itertools import count
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

asyncpg = pytest.importorskip("asyncpg")

from atlas_brain.eom_api import funnel as funnel_mod  # noqa: E402
from atlas_brain.eom_api import funnel_auth as funnel_auth_mod  # noqa: E402
from atlas_brain.services.eom_first_clean_completion import (  # noqa: E402
    EOMFirstCleanCompletionConflictError,
    EOMFirstCleanCompletionService,
    EOMFirstCleanCompletionUnavailableError,
    EOMFirstCleanCompletionValidationError,
    first_clean_completion_schema_ready,
)


ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS = ROOT / "atlas_brain" / "storage" / "migrations"
DATABASE_URL_ENV = "ATLAS_EOM_FIRST_CLEAN_TEST_DATABASE_URL"
DBA_DATABASE_URL_ENV = "ATLAS_EOM_FIRST_CLEAN_DBA_DATABASE_URL"
_NOW = datetime(2026, 8, 24, 20, 0, tzinfo=timezone.utc)
_EOM_CONTEXT = "effingham_maids"
_TRACKER_IDS = count(10_000)
_COMPLETION_SCHEMA_MIGRATIONS = (
    "035_contacts",
    "256_contact_interaction_dedupe",
    "346_contact_lead_pipeline",
    "351_eom_lead_lifecycle_events",
    "353_eom_customer_handoffs",
    "354_eom_customer_handoff_privileges",
    "363_eom_lead_lifecycle_sequence",
    "366_contacts_customer_type",
    "394_eom_first_clean_completion_receipts",
)
_PREREQUISITE_INTEGRITY_TRIGGERS = (
    ("eom_customer_handoffs", "trg_require_eom_customer_handoff_finalization"),
    ("eom_customer_handoffs", "trg_prevent_eom_customer_handoff_mutation"),
    ("eom_customer_handoffs", "trg_prevent_eom_customer_handoff_truncate"),
    ("eom_lead_lifecycle_events", "trg_prevent_eom_lead_lifecycle_event_mutation"),
    ("eom_lead_lifecycle_events", "trg_prevent_eom_lead_lifecycle_event_truncate"),
)
_PREREQUISITE_HANDOFF_GUARD_FUNCTIONS = (
    "require_eom_customer_handoff_finalization",
    "prevent_eom_customer_handoff_mutation",
)
_PREREQUISITE_HANDOFF_ADMISSION_FUNCTIONS = (
    "require_eom_customer_handoff_finalization",
)
_LIFECYCLE_GUARD_FUNCTION = "prevent_eom_lead_lifecycle_event_mutation"
_LIFECYCLE_GUARD_TRIGGERS = (
    "trg_prevent_eom_lead_lifecycle_event_mutation",
    "trg_prevent_eom_lead_lifecycle_event_truncate",
)
_RECEIPT_GUARD_FUNCTIONS = (
    "prevent_eom_first_clean_completion_mutation",
    "require_eom_first_clean_completion_operation_scope",
    "require_eom_first_clean_completion_receipt",
)
_RECEIPT_ADMISSION_FUNCTIONS = (
    "require_eom_first_clean_completion_operation_scope",
    "require_eom_first_clean_completion_receipt",
)
_ACTOR_ID_LENGTH_BOUNDARIES = (
    1,
    10,
    100,
    10_000,
    1_000_000,
    2_147_483_647,
    9_223_372_036_854_775_807,
)
_EFFECTIVE_RUNTIME_PRIVILEGE_OPTIONS = (
    "SUPERUSER",
    "CREATEROLE",
    "CREATEDB",
    "REPLICATION",
    "BYPASSRLS",
)


class _ConnectionPool:
    """Expose the small async pool protocol the service needs in tests."""

    is_initialized = True

    def __init__(
        self, runtime_connection: Any, dba_connection: Any | None = None
    ) -> None:
        self._runtime_connection = runtime_connection
        # Direct fixture setup and deliberate catalog corruption need a DBA;
        # service calls below always execute as the unprivileged atlas runtime.
        self._connection = dba_connection or runtime_connection

    @asynccontextmanager
    async def transaction(self):
        async with self._runtime_connection.transaction():
            yield self._runtime_connection

    async def fetchval(self, *args: Any, **kwargs: Any) -> Any:
        return await self._runtime_connection.fetchval(*args, **kwargs)


def _database_url_or_skip() -> str:
    database_url = os.environ.get(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")
    return database_url


def _dba_database_url_or_skip() -> str:
    database_url = os.environ.get(DBA_DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DBA_DATABASE_URL_ENV} is not configured")
    return database_url


async def _provision_nocodb_login(connection: Any) -> None:
    """Supply migration 354's no-privilege login prerequisite in test only."""

    exists = await connection.fetchval(
        "SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'atlas_nocodb')"
    )
    if exists:
        await connection.execute(
            "ALTER ROLE atlas_nocodb LOGIN NOINHERIT NOSUPERUSER "
            "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
        )
    else:
        await connection.execute(
            "CREATE ROLE atlas_nocodb LOGIN NOINHERIT NOSUPERUSER "
            "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
        )
    assert not await connection.fetchval(
        """
        SELECT EXISTS (
            SELECT 1
            FROM pg_auth_members AS membership
            JOIN pg_roles AS role ON role.oid = membership.member
            WHERE role.rolname = 'atlas_nocodb'
        )
        """
    )


async def _provision_guard_role(connection: Any) -> None:
    """Supply the isolated guard-owner prerequisite from the test DBA."""

    await connection.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM pg_roles
                WHERE rolname = 'atlas_eom_handoff_owner'
            ) THEN
                ALTER ROLE atlas_eom_handoff_owner
                    NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE
                    NOREPLICATION NOBYPASSRLS;
            ELSE
                CREATE ROLE atlas_eom_handoff_owner
                    NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE
                    NOREPLICATION NOBYPASSRLS;
            END IF;
        END;
        $$;
        """
    )


async def _grant_completion_runtime_dml_as_guard(connection: Any) -> None:
    """Restore the guarded tables' durable runtime grants without ownership."""

    await connection.execute("SET ROLE atlas_eom_handoff_owner")
    try:
        await connection.execute(
            "GRANT SELECT, INSERT, UPDATE "
            "ON TABLE eom_first_clean_completion_operation_receipts TO atlas"
        )
        await connection.execute(
            "GRANT SELECT, INSERT, UPDATE "
            "ON TABLE eom_first_clean_completion_receipts TO atlas"
        )
        await connection.execute(
            "GRANT SELECT, INSERT, UPDATE ON TABLE eom_lead_lifecycle_events TO atlas"
        )
        await connection.execute(
            "GRANT USAGE ON SEQUENCE eom_lead_lifecycle_events_sequence_seq "
            "TO atlas"
        )
    finally:
        await connection.execute("RESET ROLE")


async def _apply_schema(
    runtime_connection: Any,
    dba_connection: Any,
    schema: str,
    *,
    migrations: tuple[str, ...] = _COMPLETION_SCHEMA_MIGRATIONS,
) -> None:
    await runtime_connection.execute(f'CREATE SCHEMA "{schema}"')
    await runtime_connection.execute(f'SET search_path TO "{schema}", public')
    await dba_connection.execute(f'SET search_path TO "{schema}", public')
    # Migration 035 remains additive to the production appointments relation;
    # the focused empty schema supplies that dependency explicitly.
    await runtime_connection.execute("CREATE TABLE appointments (id UUID PRIMARY KEY)")
    # The completion schema is DBA-only because it references and protects the
    # guard-owned handoff table. Keep this disposable proof limited to its
    # actual canonical prerequisites rather than a developer's broader schema.
    await _provision_nocodb_login(dba_connection)
    await _provision_guard_role(dba_connection)
    await dba_connection.execute(
        "GRANT atlas_eom_handoff_owner TO atlas WITH ADMIN OPTION"
    )
    try:
        for migration in migrations:
            migration_sql = (MIGRATIONS / f"{migration}.sql").read_text()
            if migration == "394_eom_first_clean_completion_receipts":
                # Migration 354 needs this temporary membership to transfer its
                # protected objects; production removes it before the DBA-only
                # completion migration records its final guard boundary.
                await dba_connection.execute(
                    "REVOKE atlas_eom_handoff_owner FROM atlas"
                )
                await dba_connection.execute(migration_sql)
            else:
                await runtime_connection.execute(migration_sql)
    finally:
        await dba_connection.execute("REVOKE atlas_eom_handoff_owner FROM atlas")


@asynccontextmanager
async def _test_store(
    *,
    migrations: tuple[str, ...] = _COMPLETION_SCHEMA_MIGRATIONS,
):
    database_url = _database_url_or_skip()
    dba_database_url = _dba_database_url_or_skip()
    schema = f"atlas_eom_first_clean_{uuid4().hex}"
    runtime_connection = await asyncpg.connect(database_url)
    dba_connection = await asyncpg.connect(dba_database_url)
    try:
        await _apply_schema(
            runtime_connection,
            dba_connection,
            schema,
            migrations=migrations,
        )
        yield _ConnectionPool(runtime_connection, dba_connection), schema
    finally:
        try:
            await dba_connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await dba_connection.close()
            await runtime_connection.close()


async def _schema_connection(database_url: str, schema: str) -> _ConnectionPool:
    connection = await asyncpg.connect(database_url)
    await connection.execute(f'SET search_path TO "{schema}", public')
    return _ConnectionPool(connection)


async def _close_schema_connection(pool: _ConnectionPool) -> None:
    await pool._runtime_connection.close()


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


async def _has_guard_membership(pool: _ConnectionPool, role_name: str) -> bool:
    return bool(
        await pool.fetchval(
            """
            WITH RECURSIVE role_chain(roleid) AS (
                SELECT membership.roleid
                FROM pg_auth_members AS membership
                WHERE membership.member = (
                    SELECT oid FROM pg_roles WHERE rolname = $1
                )
                UNION
                SELECT membership.roleid
                FROM pg_auth_members AS membership
                JOIN role_chain ON membership.member = role_chain.roleid
            )
            SELECT EXISTS (
                SELECT 1
                FROM role_chain
                WHERE roleid = (
                    SELECT oid
                    FROM pg_roles
                    WHERE rolname = 'atlas_eom_handoff_owner'
                )
            )
            """,
            role_name,
        )
    )


def _operation_key(prefix: str = "first-clean") -> str:
    return f"{prefix}-{uuid4().hex}"


def _service(pool: _ConnectionPool) -> EOMFirstCleanCompletionService:
    return EOMFirstCleanCompletionService(pool=pool, now=lambda: _NOW)


async def _insert_customer(
    pool: _ConnectionPool,
    *,
    customer_type: str = "residential",
    status: str = "active",
    with_handoff: bool = True,
) -> tuple[UUID, int | None, int | None]:
    contact_id = uuid4()
    tracker_customer_id = next(_TRACKER_IDS)
    tracker_site_id = next(_TRACKER_IDS)
    await pool._connection.execute(
        """
        INSERT INTO contacts (
            id, full_name, business_context_id, contact_type, status,
            customer_type, created_at
        ) VALUES ($1, 'First Completion Test Customer', $2, 'customer', $3, $4, $5)
        """,
        contact_id,
        _EOM_CONTEXT,
        status,
        customer_type,
        _NOW - timedelta(days=1),
    )
    if not with_handoff:
        return contact_id, None, None

    approval_key = _operation_key("approved")
    await pool._connection.execute(
        """
        INSERT INTO eom_lead_lifecycle_events (
            contact_id, event_type, actor, source, operation_key, metadata,
            occurred_at
        ) VALUES (
            $1, 'customer_approved', 'employee:7:Juan', 'eom_office', $2,
            jsonb_build_object(
                'tracker_customer_id', $3::bigint,
                'tracker_site_id', $4::bigint,
                'approved_by_employee_id', 7::bigint
            ),
            $5
        )
        """,
        contact_id,
        approval_key,
        tracker_customer_id,
        tracker_site_id,
        _NOW - timedelta(days=1),
    )
    await pool._connection.execute(
        """
        INSERT INTO eom_customer_handoffs (
            contact_id, approval_key, tracker_customer_id, tracker_site_id,
            approved_by_employee_id, approved_by_name
        ) VALUES ($1, $2, $3, $4, 7, 'Juan')
        """,
        contact_id,
        approval_key,
        tracker_customer_id,
        tracker_site_id,
    )
    return contact_id, tracker_customer_id, tracker_site_id


def _completion_kwargs(
    *,
    contact_id: UUID,
    tracker_customer_id: int,
    tracker_site_id: int,
    operation_key: str,
    tracker_service_id: int = 6001,
    completed_at: datetime = _NOW - timedelta(hours=1),
) -> dict[str, Any]:
    return {
        "contact_id": contact_id,
        "tracker_customer_id": tracker_customer_id,
        "tracker_site_id": tracker_site_id,
        "tracker_service_kind": "job",
        "tracker_service_id": tracker_service_id,
        "completed_at": completed_at,
        "operation_key": operation_key,
        "actor_id": 7,
        "actor_name": "Juan",
    }


def _app(service: EOMFirstCleanCompletionService) -> FastAPI:
    app = FastAPI()
    app.include_router(funnel_mod.router, prefix="/api/v1")
    app.dependency_overrides[funnel_auth_mod.require_eom_funnel_api] = lambda: None
    app.dependency_overrides[funnel_auth_mod.require_eom_funnel_actor] = lambda: {
        "id": 7,
        "name": "Juan",
    }
    app.dependency_overrides[funnel_mod._first_clean_completion_dependency] = lambda: (
        service
    )
    return app


def _payload(
    tracker_customer_id: int,
    tracker_site_id: int,
    *,
    tracker_service_id: int = 6001,
    completed_at: str = "2026-08-24T19:00:00Z",
) -> dict[str, object]:
    return {
        "tracker_customer_id": tracker_customer_id,
        "tracker_site_id": tracker_site_id,
        "tracker_service_kind": "job",
        "tracker_service_id": tracker_service_id,
        "completed_at": completed_at,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("actor_id", _ACTOR_ID_LENGTH_BOUNDARIES)
async def test_completion_rejects_serialized_lifecycle_actor_overflow_before_db_access(
    actor_id: int,
) -> None:
    """The stored lifecycle actor, not only the raw name, fits VARCHAR(128)."""

    actor_name = "a" * (129 - len(f"employee:{actor_id}:"))
    service = EOMFirstCleanCompletionService(pool=object(), now=lambda: _NOW)

    with pytest.raises(EOMFirstCleanCompletionValidationError):
        await service.record_completion(
            contact_id=uuid4(),
            tracker_customer_id=1,
            tracker_site_id=2,
            tracker_service_kind="job",
            tracker_service_id=3,
            completed_at=_NOW - timedelta(hours=1),
            operation_key=_operation_key("actor-overflow"),
            actor_id=actor_id,
            actor_name=actor_name,
        )


@pytest.mark.asyncio
async def test_completion_persists_lifecycle_actor_at_serialized_column_limit() -> None:
    async with _test_store() as (pool, _schema):
        for index, actor_id in enumerate(_ACTOR_ID_LENGTH_BOUNDARIES):
            contact_id, tracker_customer_id, tracker_site_id = await _insert_customer(
                pool
            )
            assert tracker_customer_id is not None and tracker_site_id is not None
            actor_name = "a" * (128 - len(f"employee:{actor_id}:"))

            completion = _completion_kwargs(
                contact_id=contact_id,
                tracker_customer_id=tracker_customer_id,
                tracker_site_id=tracker_site_id,
                operation_key=_operation_key("actor-boundary"),
                tracker_service_id=6_001 + index,
            )
            completion["actor_id"] = actor_id
            completion["actor_name"] = actor_name
            await _service(pool).record_completion(**completion)

            lifecycle_actor = await pool.fetchval(
                """
                SELECT actor
                FROM eom_lead_lifecycle_events
                WHERE contact_id = $1 AND event_type = 'first_clean_completed'
                """,
                contact_id,
            )
            assert lifecycle_actor == f"employee:{actor_id}:{actor_name}"
            assert len(lifecycle_actor) == 128


def test_slim_eom_profile_binds_completion_to_canonical_funnel_pool() -> None:
    from atlas_brain import main_eom

    assert (
        main_eom.app.state.eom_funnel_first_clean_completion_pool
        is main_eom.get_eom_funnel_db_pool
    )


def test_slim_eom_profile_does_not_run_dba_completion_migrations() -> None:
    from atlas_brain import main_eom

    assert not hasattr(main_eom, "_run_eom_first_clean_completion_startup_migrations")


@pytest.mark.asyncio
async def test_dba_migration_uses_guard_ownership_and_minimal_runtime_acl() -> None:
    async with _test_store() as (pool, schema):
        assert await first_clean_completion_schema_ready(pool) is True
        namespace_access = await pool._connection.fetchrow(
            """
            SELECT owner.rolname AS owner,
                   has_schema_privilege('atlas', current_schema(), 'USAGE')
                       AS runtime_usage,
                   has_schema_privilege('atlas', current_schema(), 'CREATE')
                       AS runtime_create
            FROM pg_namespace AS namespace
            JOIN pg_roles AS owner ON owner.oid = namespace.nspowner
            WHERE namespace.nspname = current_schema()
            """
        )
        assert dict(namespace_access) == {
            "owner": "atlas_eom_handoff_owner",
            "runtime_usage": True,
            "runtime_create": True,
        }
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await pool._runtime_connection.execute(
                f"DROP SCHEMA {_quote_ident(schema)} CASCADE"
            )
        ownership = await pool._connection.fetch(
            """
            SELECT relation.relname, role.rolname AS owner
            FROM pg_class AS relation
            JOIN pg_roles AS role ON role.oid = relation.relowner
            JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = current_schema()
              AND relation.relname = ANY($1::text[])
            ORDER BY relation.relname
            """,
            [
                "eom_customer_handoffs",
                "eom_lead_lifecycle_events",
                "eom_first_clean_completion_operation_receipts",
                "eom_first_clean_completion_receipts",
            ],
        )
        assert {row["relname"]: row["owner"] for row in ownership} == {
            "eom_customer_handoffs": "atlas_eom_handoff_owner",
            "eom_lead_lifecycle_events": "atlas_eom_handoff_owner",
            "eom_first_clean_completion_operation_receipts": (
                "atlas_eom_handoff_owner"
            ),
            "eom_first_clean_completion_receipts": "atlas_eom_handoff_owner",
        }
        lifecycle_sequence = await pool._connection.fetchrow(
            """
            SELECT owner.rolname AS owner,
                   has_sequence_privilege(
                       'atlas',
                       'eom_lead_lifecycle_events_sequence_seq',
                       'USAGE'
                   ) AS runtime_usage,
                   has_sequence_privilege(
                       'atlas',
                       'eom_lead_lifecycle_events_sequence_seq',
                       'SELECT'
                   ) AS runtime_select
            FROM pg_class AS relation
            JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            JOIN pg_roles AS owner ON owner.oid = relation.relowner
            WHERE namespace.nspname = current_schema()
              AND relation.relkind = 'S'
              AND relation.relname = 'eom_lead_lifecycle_events_sequence_seq'
            """
        )
        assert dict(lifecycle_sequence) == {
            "owner": "atlas_eom_handoff_owner",
            "runtime_usage": True,
            "runtime_select": False,
        }
        assert await pool._connection.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_attrdef AS attribute_default
                JOIN pg_attribute AS attribute
                  ON attribute.attrelid = attribute_default.adrelid
                 AND attribute.attnum = attribute_default.adnum
                JOIN pg_depend AS dependency
                  ON dependency.classid = 'pg_attrdef'::regclass
                 AND dependency.objid = attribute_default.oid
                 AND dependency.refclassid = 'pg_class'::regclass
                 AND dependency.refobjid = relation.oid
                WHERE attribute_default.adrelid = 'eom_lead_lifecycle_events'::regclass
                  AND attribute.attname = 'lifecycle_sequence'
            )
            FROM pg_class AS relation
            JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = current_schema()
              AND relation.relkind = 'S'
              AND relation.relname = 'eom_lead_lifecycle_events_sequence_seq'
            """
        )
        lifecycle_sequence_acl = await pool._connection.fetch(
            """
            SELECT acl.privilege_type, acl.is_grantable
            FROM pg_class AS relation
            JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            CROSS JOIN LATERAL aclexplode(
                COALESCE(relation.relacl, ARRAY[]::aclitem[])
            ) AS acl
            WHERE namespace.nspname = current_schema()
              AND relation.relkind = 'S'
              AND relation.relname = 'eom_lead_lifecycle_events_sequence_seq'
              AND acl.grantee = (SELECT oid FROM pg_roles WHERE rolname = 'atlas')
            """
        )
        assert [dict(row) for row in lifecycle_sequence_acl] == [
            {"privilege_type": "USAGE", "is_grantable": False}
        ]
        protected_function_ownership = await pool._connection.fetch(
            """
            SELECT protected_function.proname, role.rolname AS owner
            FROM pg_proc AS protected_function
            JOIN pg_namespace AS namespace
              ON namespace.oid = protected_function.pronamespace
            JOIN pg_roles AS role ON role.oid = protected_function.proowner
            WHERE namespace.nspname = current_schema()
              AND protected_function.proname = ANY($1::text[])
              AND protected_function.pronargs = 0
            ORDER BY protected_function.proname
            """,
            [*_PREREQUISITE_HANDOFF_GUARD_FUNCTIONS, _LIFECYCLE_GUARD_FUNCTION],
        )
        assert {
            row["proname"]: row["owner"] for row in protected_function_ownership
        } == {
            function_name: "atlas_eom_handoff_owner"
            for function_name in (
                *_PREREQUISITE_HANDOFF_GUARD_FUNCTIONS,
                _LIFECYCLE_GUARD_FUNCTION,
            )
        }
        lifecycle_trigger_bindings = await pool._connection.fetch(
            """
            SELECT trigger.tgname, trigger_function.proname,
                   owner.rolname AS function_owner
            FROM pg_trigger AS trigger
            JOIN pg_proc AS trigger_function ON trigger_function.oid = trigger.tgfoid
            JOIN pg_roles AS owner ON owner.oid = trigger_function.proowner
            WHERE trigger.tgrelid = 'eom_lead_lifecycle_events'::regclass
              AND trigger.tgname = ANY($1::text[])
              AND NOT trigger.tgisinternal
            ORDER BY trigger.tgname
            """,
            list(_LIFECYCLE_GUARD_TRIGGERS),
        )
        assert [dict(row) for row in lifecycle_trigger_bindings] == [
            {
                "tgname": trigger_name,
                "proname": _LIFECYCLE_GUARD_FUNCTION,
                "function_owner": "atlas_eom_handoff_owner",
            }
            for trigger_name in sorted(_LIFECYCLE_GUARD_TRIGGERS)
        ]
        runtime_acl_rows = await pool._connection.fetch(
            """
            SELECT relation.relname, acl.privilege_type
            FROM pg_class AS relation
            JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            CROSS JOIN LATERAL aclexplode(
                COALESCE(relation.relacl, ARRAY[]::aclitem[])
            ) AS acl
            WHERE namespace.nspname = current_schema()
              AND relation.relname = ANY($1::text[])
              AND acl.grantee = (SELECT oid FROM pg_roles WHERE rolname = 'atlas')
            ORDER BY relation.relname, acl.privilege_type
            """,
            [
                "eom_first_clean_completion_operation_receipts",
                "eom_first_clean_completion_receipts",
                "eom_lead_lifecycle_events",
            ],
        )
        runtime_acl: dict[str, set[str]] = {}
        for row in runtime_acl_rows:
            runtime_acl.setdefault(row["relname"], set()).add(row["privilege_type"])
        assert runtime_acl == {
            "eom_first_clean_completion_operation_receipts": {
                "INSERT",
                "SELECT",
                "UPDATE",
            },
            "eom_first_clean_completion_receipts": {
                "INSERT",
                "SELECT",
                "UPDATE",
            },
            "eom_lead_lifecycle_events": {"INSERT", "SELECT", "UPDATE"},
        }
        guard_acl_rows = await pool._connection.fetch(
            """
            SELECT relation.relname, acl.privilege_type
            FROM pg_class AS relation
            JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            CROSS JOIN LATERAL aclexplode(
                COALESCE(relation.relacl, ARRAY[]::aclitem[])
            ) AS acl
            WHERE namespace.nspname = current_schema()
              AND relation.relname = ANY($1::text[])
              AND acl.grantee = (
                  SELECT oid FROM pg_roles WHERE rolname = 'atlas_eom_handoff_owner'
              )
            ORDER BY relation.relname, acl.privilege_type
            """,
            [
                "eom_first_clean_completion_operation_receipts",
                "eom_first_clean_completion_receipts",
            ],
        )
        guard_acl: dict[str, set[str]] = {}
        for row in guard_acl_rows:
            guard_acl.setdefault(row["relname"], set()).add(row["privilege_type"])
        assert guard_acl == {
            "eom_first_clean_completion_operation_receipts": {
                "DELETE",
                "INSERT",
                "REFERENCES",
                "SELECT",
                "TRIGGER",
                "TRUNCATE",
                "UPDATE",
            },
            "eom_first_clean_completion_receipts": {
                "DELETE",
                "INSERT",
                "REFERENCES",
                "SELECT",
                "TRIGGER",
                "TRUNCATE",
                "UPDATE",
            },
        }
        unexpected_acl_rows = await pool._connection.fetch(
            """
            SELECT relation.relname, grantee.rolname, acl.privilege_type,
                   acl.is_grantable
            FROM pg_class AS relation
            JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            CROSS JOIN LATERAL aclexplode(
                COALESCE(relation.relacl, ARRAY[]::aclitem[])
            ) AS acl
            JOIN pg_roles AS grantee ON grantee.oid = acl.grantee
            WHERE namespace.nspname = current_schema()
              AND relation.relname = ANY($1::text[])
              AND (
                  grantee.rolname NOT IN ('atlas', 'atlas_eom_handoff_owner')
                  OR acl.is_grantable
              )
            """,
            [
                "eom_first_clean_completion_operation_receipts",
                "eom_first_clean_completion_receipts",
            ],
        )
        assert unexpected_acl_rows == []
        guard_fk_access = await pool._connection.fetchrow(
            """
            SELECT has_schema_privilege(
                       'atlas_eom_handoff_owner', current_schema(), 'USAGE'
                   ) AS schema_usage,
                   has_table_privilege(
                       'atlas_eom_handoff_owner',
                       'eom_first_clean_completion_operation_receipts',
                       'SELECT'
                   ) AS operation_select,
                   has_table_privilege(
                       'atlas_eom_handoff_owner',
                       'eom_first_clean_completion_operation_receipts',
                       'UPDATE'
                   ) AS operation_update
            """
        )
        assert dict(guard_fk_access) == {
            "schema_usage": True,
            "operation_select": True,
            "operation_update": True,
        }
        assert not await _has_guard_membership(pool, "atlas")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("relation_name", "other_relation"),
    (
        (
            "eom_first_clean_completion_operation_receipts",
            "eom_first_clean_completion_receipts",
        ),
        (
            "eom_first_clean_completion_receipts",
            "eom_first_clean_completion_operation_receipts",
        ),
    ),
)
async def test_completion_migration_refuses_preexisting_runtime_receipt_relation(
    relation_name: str,
    other_relation: str,
) -> None:
    """Migration 394 never adopts a runtime-created receipt lookalike."""

    async with _test_store(
        migrations=_COMPLETION_SCHEMA_MIGRATIONS[:-1]
    ) as (pool, _schema):
        await pool._runtime_connection.execute(
            f"CREATE TABLE {_quote_ident(relation_name)} (id BIGINT PRIMARY KEY)"
        )
        completion_sql = (
            MIGRATIONS / "394_eom_first_clean_completion_receipts.sql"
        ).read_text()

        with pytest.raises(asyncpg.DuplicateTableError):
            async with pool._connection.transaction():
                await pool._connection.execute(completion_sql)

        assert await pool._connection.fetchval(
            "SELECT pg_catalog.to_regclass($1) IS NULL",
            other_relation,
        )


@pytest.mark.asyncio
async def test_completion_migration_rebuilds_lifecycle_guard_before_transfer() -> None:
    """A pre-394 permissive lifecycle guard cannot survive the DBA migration."""

    async with _test_store(
        migrations=_COMPLETION_SCHEMA_MIGRATIONS[:-1]
    ) as (pool, _schema):
        await pool._connection.execute(
            """
            CREATE OR REPLACE FUNCTION prevent_eom_lead_lifecycle_event_mutation()
            RETURNS TRIGGER
            LANGUAGE plpgsql
            SECURITY DEFINER
            SET search_path TO pg_temp
            AS $$
            BEGIN
                RETURN NEW;
            END;
            $$;

            DROP TRIGGER IF EXISTS trg_prevent_eom_lead_lifecycle_event_mutation
                ON eom_lead_lifecycle_events;
            CREATE TRIGGER trg_prevent_eom_lead_lifecycle_event_mutation
                BEFORE INSERT ON eom_lead_lifecycle_events
                FOR EACH ROW
                EXECUTE FUNCTION prevent_eom_lead_lifecycle_event_mutation();

            DROP TRIGGER IF EXISTS trg_prevent_eom_lead_lifecycle_event_truncate
                ON eom_lead_lifecycle_events;
            CREATE TRIGGER trg_prevent_eom_lead_lifecycle_event_truncate
                BEFORE DELETE ON eom_lead_lifecycle_events
                FOR EACH ROW
                EXECUTE FUNCTION prevent_eom_lead_lifecycle_event_mutation();
            """
        )
        completion_sql = (
            MIGRATIONS / "394_eom_first_clean_completion_receipts.sql"
        ).read_text()
        async with pool._connection.transaction():
            await pool._connection.execute(completion_sql)

        assert await first_clean_completion_schema_ready(pool) is True
        assert await pool._connection.fetchval(
            """
            SELECT NOT protected_function.prosecdef
               AND protected_function.proconfig IS NULL
            FROM pg_proc AS protected_function
            WHERE protected_function.oid =
                'prevent_eom_lead_lifecycle_event_mutation()'::regprocedure
            """
        )
        contact_id, _customer_id, _site_id = await _insert_customer(pool)
        with pytest.raises(
            asyncpg.RaiseError,
            match="eom_lead_lifecycle_events is append-only",
        ):
            await pool._runtime_connection.execute(
                "UPDATE eom_lead_lifecycle_events SET actor = actor "
                "WHERE contact_id = $1",
                contact_id,
            )
        with pytest.raises(
            asyncpg.RaiseError,
            match="eom_lead_lifecycle_events is append-only",
        ):
            await pool._connection.execute("TRUNCATE eom_lead_lifecycle_events")


@pytest.mark.asyncio
async def test_completion_migration_rebuilds_handoff_guards_before_trusting_them() -> None:
    """A permissive handoff boundary cannot survive the DBA completion migration."""

    async with _test_store(
        migrations=_COMPLETION_SCHEMA_MIGRATIONS[:-1]
    ) as (pool, schema):
        await pool._connection.execute(
            """
            CREATE OR REPLACE FUNCTION require_eom_customer_handoff_finalization()
            RETURNS TRIGGER
            LANGUAGE plpgsql
            SECURITY DEFINER
            SET search_path TO pg_temp
            AS $$
            BEGIN
                RETURN NEW;
            END;
            $$;

            CREATE OR REPLACE FUNCTION prevent_eom_customer_handoff_mutation()
            RETURNS TRIGGER
            LANGUAGE plpgsql
            SECURITY DEFINER
            SET search_path TO pg_temp
            AS $$
            BEGIN
                RETURN NEW;
            END;
            $$;

            DROP TRIGGER IF EXISTS trg_require_eom_customer_handoff_finalization
                ON eom_customer_handoffs;
            CREATE TRIGGER trg_require_eom_customer_handoff_finalization
                BEFORE UPDATE ON eom_customer_handoffs
                FOR EACH ROW
                EXECUTE FUNCTION require_eom_customer_handoff_finalization();

            DROP TRIGGER IF EXISTS trg_prevent_eom_customer_handoff_mutation
                ON eom_customer_handoffs;
            CREATE TRIGGER trg_prevent_eom_customer_handoff_mutation
                BEFORE INSERT ON eom_customer_handoffs
                FOR EACH ROW
                EXECUTE FUNCTION prevent_eom_customer_handoff_mutation();

            DROP TRIGGER IF EXISTS trg_prevent_eom_customer_handoff_truncate
                ON eom_customer_handoffs;
            CREATE TRIGGER trg_prevent_eom_customer_handoff_truncate
                BEFORE DELETE ON eom_customer_handoffs
                FOR EACH ROW
                EXECUTE FUNCTION prevent_eom_customer_handoff_mutation();
            """
        )
        completion_sql = (
            MIGRATIONS / "394_eom_first_clean_completion_receipts.sql"
        ).read_text()
        async with pool._connection.transaction():
            await pool._connection.execute(completion_sql)

        assert await first_clean_completion_schema_ready(pool) is True
        handoff_guard_properties = await pool._connection.fetch(
            """
            SELECT protected_function.proname,
                   protected_function.prosecdef,
                   protected_function.proconfig
            FROM pg_proc AS protected_function
            WHERE protected_function.oid = ANY(
                ARRAY[
                    'require_eom_customer_handoff_finalization()'::regprocedure,
                    'prevent_eom_customer_handoff_mutation()'::regprocedure
                ]
            )
            ORDER BY protected_function.proname
            """
        )
        assert [dict(row) for row in handoff_guard_properties] == [
            {
                "proname": "prevent_eom_customer_handoff_mutation",
                "prosecdef": False,
                "proconfig": None,
            },
            {
                "proname": "require_eom_customer_handoff_finalization",
                "prosecdef": False,
                "proconfig": [
                    f"search_path=pg_catalog, {schema}, pg_temp"
                ],
            },
        ]

        contact_id, _customer_id, _site_id = await _insert_customer(
            pool,
            with_handoff=False,
        )
        with pytest.raises(
            asyncpg.RaiseError,
            match="matching customer transition and lifecycle evidence",
        ):
            await pool._runtime_connection.execute(
                """
                INSERT INTO eom_customer_handoffs (
                    contact_id, approval_key, tracker_customer_id, tracker_site_id,
                    approved_by_employee_id, approved_by_name
                ) VALUES ($1, $2, $3, $4, 7, 'Juan')
                """,
                contact_id,
                _operation_key("forged-handoff"),
                next(_TRACKER_IDS),
                next(_TRACKER_IDS),
            )

        valid_contact_id, _valid_customer_id, _valid_site_id = await _insert_customer(
            pool
        )
        with pytest.raises(
            asyncpg.RaiseError,
            match="eom_customer_handoffs is immutable",
        ):
            await pool._runtime_connection.execute(
                "UPDATE eom_customer_handoffs SET approved_by_name = approved_by_name "
                "WHERE contact_id = $1",
                valid_contact_id,
            )
        # PostgreSQL checks the receipt FK before it can dispatch the handoff
        # table's BEFORE TRUNCATE trigger, and the runtime intentionally lacks
        # TRUNCATE on the receipt table. The isolated DBA fixture removes only
        # that unrelated FK admission barrier so this assertion reaches the
        # actual handoff trigger under the normal runtime role.
        await pool._connection.execute(
            "ALTER TABLE eom_first_clean_completion_receipts "
            "DROP CONSTRAINT eom_first_clean_completion_receipts_handoff_id_fkey"
        )
        with pytest.raises(
            asyncpg.RaiseError,
            match="eom_customer_handoffs is immutable",
        ):
            await pool._runtime_connection.execute("TRUNCATE eom_customer_handoffs")


@pytest.mark.asyncio
async def test_runtime_and_dba_connections_share_advisory_lock_namespace() -> None:
    """The controlled runner's live target proof holds across real connections."""

    async with _test_store() as (pool, _schema):
        runtime_connection = pool._runtime_connection
        dba_connection = pool._connection
        lock_key = uuid4().int & ((1 << 63) - 1)
        async with runtime_connection.transaction():
            assert await runtime_connection.fetchval(
                "SELECT pg_catalog.pg_try_advisory_xact_lock($1)",
                lock_key,
            )
            dba_acquired = bool(
                await dba_connection.fetchval(
                    "SELECT pg_catalog.pg_try_advisory_xact_lock($1)",
                    lock_key,
                )
            )
            assert dba_acquired is False


@pytest.mark.asyncio
async def test_handoff_finalization_trigger_rejects_runtime_temp_shadowing() -> None:
    """Runtime TEMP relations cannot fabricate an approved customer handoff."""

    async with _test_store() as (pool, schema):
        contact_id, customer_id, site_id = await _insert_customer(
            pool,
            with_handoff=False,
        )
        assert customer_id is None and site_id is None
        runtime_connection = pool._runtime_connection
        approval_key = _operation_key("handoff-temp-shadow")
        tracker_customer_id = next(_TRACKER_IDS)
        tracker_site_id = next(_TRACKER_IDS)
        await runtime_connection.execute(
            """
            CREATE TEMP TABLE contacts (
                id UUID,
                business_context_id TEXT,
                contact_type TEXT,
                lead_stage TEXT,
                status TEXT
            );
            CREATE TEMP TABLE eom_lead_lifecycle_events (
                contact_id UUID,
                event_type VARCHAR(64),
                source VARCHAR(32),
                operation_key VARCHAR(128),
                actor VARCHAR(128),
                metadata JSONB
            )
            """
        )
        await runtime_connection.execute(
            """
            INSERT INTO contacts VALUES (
                $1, 'effingham_maids', 'customer', NULL, 'active'
            )
            """,
            contact_id,
        )
        await runtime_connection.execute(
            """
            INSERT INTO eom_lead_lifecycle_events VALUES (
                $1, 'customer_approved', 'eom_office', $2, 'employee:7:Juan',
                jsonb_build_object(
                    'tracker_customer_id', $3::bigint,
                    'tracker_site_id', $4::bigint,
                    'approved_by_employee_id', 7::bigint
                )
            )
            """,
            contact_id,
            approval_key,
            tracker_customer_id,
            tracker_site_id,
        )
        handoff_insert = """
            INSERT INTO eom_customer_handoffs (
                contact_id, approval_key, tracker_customer_id, tracker_site_id,
                approved_by_employee_id, approved_by_name
            ) VALUES ($1, $2, $3, $4, 7, 'Juan')
        """
        handoff_args = (
            contact_id,
            approval_key,
            tracker_customer_id,
            tracker_site_id,
        )
        function_ident = _quote_ident("require_eom_customer_handoff_finalization")
        schema_ident = _quote_ident(schema)
        try:
            with pytest.raises(
                asyncpg.RaiseError,
                match="matching customer transition and lifecycle evidence",
            ):
                await runtime_connection.execute(handoff_insert, *handoff_args)

            # The negative control proves the test exercises relation lookup:
            # a temp-first trigger path admits the deliberately fabricated
            # lifecycle evidence, while the deployed catalog-first path does not.
            await pool._connection.execute(
                f"ALTER FUNCTION {function_ident}() "
                f"SET search_path TO pg_temp, {schema_ident}, pg_catalog"
            )
            assert await first_clean_completion_schema_ready(pool) is False
            await runtime_connection.execute(handoff_insert, *handoff_args)
            assert await pool.fetchval(
                "SELECT COUNT(*) FROM eom_customer_handoffs WHERE contact_id = $1",
                contact_id,
            ) == 1
        finally:
            await pool._connection.execute(
                f"ALTER FUNCTION {function_ident}() "
                f"SET search_path TO pg_catalog, {schema_ident}, pg_temp"
            )
        assert await first_clean_completion_schema_ready(pool) is True


@pytest.mark.asyncio
async def test_receipt_trigger_rejects_temp_relation_shadowing() -> None:
    """Runtime TEMP objects cannot fabricate guarded handoff/lifecycle evidence."""

    async with _test_store() as (pool, _schema):
        contact_id, customer_id, site_id = await _insert_customer(pool)
        assert customer_id is not None and site_id is not None
        handoff_id = await pool._connection.fetchval(
            "SELECT id FROM eom_customer_handoffs WHERE contact_id = $1",
            contact_id,
        )
        assert handoff_id is not None
        runtime_connection = pool._runtime_connection
        operation_key = _operation_key("temp-shadow")
        fingerprint = "c" * 64
        completed_at = _NOW - timedelta(hours=1)
        await runtime_connection.execute(
            """
            INSERT INTO eom_first_clean_completion_operation_receipts (
                operation_key, contact_id, request_fingerprint
            ) VALUES ($1, $2, $3)
            """,
            operation_key,
            contact_id,
            fingerprint,
        )

        fake_customer_id = next(_TRACKER_IDS)
        fake_site_id = next(_TRACKER_IDS)
        receipt_id = uuid4()
        await runtime_connection.execute(
            """
            CREATE TEMP TABLE contacts (
                id UUID,
                business_context_id TEXT,
                contact_type TEXT,
                status TEXT,
                customer_type TEXT
            );
            CREATE TEMP TABLE eom_customer_handoffs (
                id UUID,
                contact_id UUID,
                tracker_customer_id BIGINT,
                tracker_site_id BIGINT
            );
            CREATE TEMP TABLE eom_first_clean_completion_operation_receipts (
                operation_key VARCHAR(128),
                contact_id UUID,
                request_fingerprint VARCHAR(64)
            );
            CREATE TEMP TABLE eom_lead_lifecycle_events (
                contact_id UUID,
                event_type VARCHAR(64),
                actor VARCHAR(128),
                source VARCHAR(32),
                operation_key VARCHAR(128),
                metadata JSONB,
                occurred_at TIMESTAMPTZ
            )
            """
        )
        await runtime_connection.execute(
            """
            INSERT INTO contacts VALUES (
                $1, 'effingham_maids', 'customer', 'active', 'residential'
            )
            """,
            contact_id,
        )
        await runtime_connection.execute(
            """
            INSERT INTO eom_customer_handoffs VALUES ($1, $2, $3, $4)
            """,
            handoff_id,
            contact_id,
            fake_customer_id,
            fake_site_id,
        )
        await runtime_connection.execute(
            """
            INSERT INTO eom_first_clean_completion_operation_receipts
                VALUES ($1, $2, $3)
            """,
            operation_key,
            contact_id,
            fingerprint,
        )
        await runtime_connection.execute(
            """
            INSERT INTO eom_lead_lifecycle_events VALUES (
                $1, 'first_clean_completed', 'employee:7:Juan', 'time_tracker',
                $2,
                jsonb_build_object(
                    'completion_receipt_id', $3::text,
                    'handoff_id', $4::text,
                    'tracker_customer_id', $5::bigint,
                    'tracker_site_id', $6::bigint,
                    'tracker_service_kind', 'job',
                    'tracker_service_id', 9001::bigint
                ),
                $7
            )
            """,
            contact_id,
            operation_key,
            str(receipt_id),
            str(handoff_id),
            fake_customer_id,
            fake_site_id,
            completed_at,
        )

        with pytest.raises(
            asyncpg.RaiseError,
            match="matching active residential customer handoff",
        ):
            await runtime_connection.execute(
                """
                INSERT INTO eom_first_clean_completion_receipts (
                    id, contact_id, handoff_id, tracker_customer_id,
                    tracker_site_id, tracker_service_kind, tracker_service_id,
                    completed_at, operation_key, request_fingerprint, actor_id,
                    actor_name
                ) VALUES (
                    $1, $2, $3, $4, $5, 'job', 9001, $6, $7, $8, 7, 'Juan'
                )
                """,
                receipt_id,
                contact_id,
                handoff_id,
                fake_customer_id,
                fake_site_id,
                completed_at,
                operation_key,
                fingerprint,
            )


@pytest.mark.asyncio
async def test_receipt_trigger_resists_runtime_operator_shadowing() -> None:
    """Catalog-first admission predicates ignore an operator created by atlas."""

    async with _test_store() as (pool, schema):
        contact_id, customer_id, site_id = await _insert_customer(pool)
        assert customer_id is not None and site_id is not None
        handoff_id = await pool._connection.fetchval(
            "SELECT id FROM eom_customer_handoffs WHERE contact_id = $1",
            contact_id,
        )
        assert handoff_id is not None
        runtime_connection = pool._runtime_connection
        operation_key = _operation_key("operator-shadow")
        fingerprint = "d" * 64
        completed_at = _NOW - timedelta(hours=1)
        fake_customer_id = next(_TRACKER_IDS)
        fake_site_id = next(_TRACKER_IDS)
        receipt_id = uuid4()
        await runtime_connection.execute(
            """
            INSERT INTO eom_first_clean_completion_operation_receipts (
                operation_key, contact_id, request_fingerprint
            ) VALUES ($1, $2, $3)
            """,
            operation_key,
            contact_id,
            fingerprint,
        )
        await runtime_connection.execute(
            """
            INSERT INTO eom_lead_lifecycle_events (
                contact_id, event_type, actor, source, operation_key, metadata,
                occurred_at
            ) VALUES (
                $1, 'first_clean_completed', 'employee:7:Juan', 'time_tracker',
                $2,
                jsonb_build_object(
                    'completion_receipt_id', $3::text,
                    'handoff_id', $4::text,
                    'tracker_customer_id', $5::bigint,
                    'tracker_site_id', $6::bigint,
                    'tracker_service_kind', 'job',
                    'tracker_service_id', 9002::bigint
                ),
                $7
            )
            """,
            contact_id,
            operation_key,
            str(receipt_id),
            str(handoff_id),
            fake_customer_id,
            fake_site_id,
            completed_at,
        )

        schema_ident = _quote_ident(schema)
        rogue_function_ident = _quote_ident("eom_completion_unsafe_bigint_equality")
        await runtime_connection.execute(
            f"""
            CREATE FUNCTION {schema_ident}.{rogue_function_ident}(BIGINT, BIGINT)
            RETURNS BOOLEAN
            LANGUAGE sql
            IMMUTABLE
            AS 'SELECT TRUE';
            CREATE OPERATOR {schema_ident}.= (
                LEFTARG = BIGINT,
                RIGHTARG = BIGINT,
                PROCEDURE = {schema_ident}.{rogue_function_ident}
            )
            """
        )

        receipt_insert = """
            INSERT INTO eom_first_clean_completion_receipts (
                id, contact_id, handoff_id, tracker_customer_id,
                tracker_site_id, tracker_service_kind, tracker_service_id,
                completed_at, operation_key, request_fingerprint, actor_id,
                actor_name
            ) VALUES (
                $1, $2, $3, $4, $5, 'job', 9002, $6, $7, $8, 7, 'Juan'
            )
        """
        receipt_args = (
            receipt_id,
            contact_id,
            handoff_id,
            fake_customer_id,
            fake_site_id,
            completed_at,
            operation_key,
            fingerprint,
        )
        function_ident = _quote_ident("require_eom_first_clean_completion_receipt")
        try:
            with pytest.raises(
                asyncpg.RaiseError,
                match="matching active residential customer handoff",
            ):
                await runtime_connection.execute(receipt_insert, *receipt_args)

            # The negative control proves the assertion exercises operator
            # resolution: schema-first would admit the deliberately mismatched
            # tracker identity. This disposable schema is torn down after test.
            await pool._connection.execute(
                f"ALTER FUNCTION {function_ident}() "
                f"SET search_path TO {schema_ident}, pg_catalog, pg_temp"
            )
            await runtime_connection.execute(receipt_insert, *receipt_args)
            assert (
                await pool.fetchval(
                    "SELECT COUNT(*) FROM eom_first_clean_completion_receipts"
                )
                == 1
            )
        finally:
            await pool._connection.execute(
                f"ALTER FUNCTION {function_ident}() "
                f"SET search_path TO pg_catalog, {schema_ident}, pg_temp"
            )
            await pool._connection.execute(
                f"DROP OPERATOR IF EXISTS {schema_ident}.= (BIGINT, BIGINT)"
            )
            await pool._connection.execute(
                f"DROP FUNCTION IF EXISTS {schema_ident}.{rogue_function_ident}"
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("privileged_option", "unprivileged_option"),
    (
        ("SUPERUSER", "NOSUPERUSER"),
        ("CREATEROLE", "NOCREATEROLE"),
        ("CREATEDB", "NOCREATEDB"),
        ("REPLICATION", "NOREPLICATION"),
        ("BYPASSRLS", "NOBYPASSRLS"),
    ),
)
async def test_schema_readiness_rejects_privileged_or_reassumed_runtime(
    privileged_option: str,
    unprivileged_option: str,
) -> None:
    """Readiness must attest both the atlas role and its actual pool session."""

    async with _test_store() as (pool, _schema):
        assert await first_clean_completion_schema_ready(pool) is True
        try:
            await pool._connection.execute(f"ALTER ROLE atlas {privileged_option}")
            assert await first_clean_completion_schema_ready(pool) is False
        finally:
            await pool._connection.execute(f"ALTER ROLE atlas {unprivileged_option}")
        assert await first_clean_completion_schema_ready(pool) is True
        assert (
            await first_clean_completion_schema_ready(_ConnectionPool(pool._connection))
            is False
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("privileged_option", _EFFECTIVE_RUNTIME_PRIVILEGE_OPTIONS)
async def test_schema_readiness_rejects_effective_runtime_privilege_membership(
    privileged_option: str,
) -> None:
    """A NOLOGIN administrator is still reachable through SET ROLE."""

    async with _test_store() as (pool, _schema):
        assert await first_clean_completion_schema_ready(pool) is True
        role_name = f"eom_completion_effective_{uuid4().hex[:16]}"
        role_ident = _quote_ident(role_name)
        role_created = False
        try:
            can_administer_roles = await pool._connection.fetchval(
                """
                SELECT rolsuper OR rolcreaterole
                FROM pg_roles
                WHERE rolname = current_user
                """
            )
            if not can_administer_roles:
                pytest.skip(
                    "effective-runtime membership proof requires role administration"
                )
            await pool._connection.execute(
                f"CREATE ROLE {role_ident} NOLOGIN NOINHERIT {privileged_option}"
            )
            role_created = True
            await pool._connection.execute(f"GRANT {role_ident} TO atlas")
            assert await first_clean_completion_schema_ready(pool) is False
        finally:
            if role_created:
                await pool._connection.execute(f"REVOKE {role_ident} FROM atlas")
                await pool._connection.execute(f"DROP ROLE {role_ident}")
        assert await first_clean_completion_schema_ready(pool) is True


@pytest.mark.asyncio
async def test_schema_readiness_rejects_atlas_database_ownership() -> None:
    """Owning the database is an effective administrative path."""

    async with _test_store() as (pool, _schema):
        assert await first_clean_completion_schema_ready(pool) is True
        database = await pool._connection.fetchrow(
            """
            SELECT database.datname, owner.rolname AS owner
            FROM pg_database AS database
            JOIN pg_roles AS owner ON owner.oid = database.datdba
            WHERE database.datname = current_database()
            """
        )
        assert database is not None
        database_ident = _quote_ident(str(database["datname"]))
        owner_ident = _quote_ident(str(database["owner"]))
        can_alter_database = await pool._connection.fetchval(
            """
            SELECT rolsuper OR rolcreatedb
            FROM pg_roles
            WHERE rolname = current_user
            """
        )
        if not can_alter_database:
            pytest.skip("database-owner proof requires database administration")
        try:
            await pool._connection.execute(
                f"ALTER DATABASE {database_ident} OWNER TO atlas"
            )
            assert await first_clean_completion_schema_ready(pool) is False
        finally:
            await pool._connection.execute(
                f"ALTER DATABASE {database_ident} OWNER TO {owner_ident}"
            )
            # A temporary ownership transfer can remove the test runtime's
            # explicit database CREATE baseline. Restore it before this fixture
            # releases so later disposable schemas remain isolated.
            await pool._connection.execute(
                f"GRANT CREATE ON DATABASE {database_ident} TO atlas"
            )
        assert await pool._connection.fetchval(
            "SELECT has_database_privilege('atlas', current_database(), 'CREATE')"
        )
        assert await first_clean_completion_schema_ready(pool) is True


@pytest.mark.asyncio
async def test_schema_readiness_rejects_live_former_guard_session() -> None:
    """A former guard session keeps serving fail-closed until it disconnects."""

    guard_connection: Any | None = None
    async with _test_store() as (pool, _schema):
        assert await first_clean_completion_schema_ready(pool) is True
        try:
            await pool._connection.execute(
                "ALTER ROLE atlas_eom_handoff_owner LOGIN PASSWORD 'test-guard-session'"
            )
            guard_connection = await asyncpg.connect(
                _dba_database_url_or_skip(),
                user="atlas_eom_handoff_owner",
                password="test-guard-session",
            )
            await pool._connection.execute("ALTER ROLE atlas_eom_handoff_owner NOLOGIN")

            assert await first_clean_completion_schema_ready(pool) is False
        finally:
            if guard_connection is not None:
                await guard_connection.close()
            await pool._connection.execute(
                "ALTER ROLE atlas_eom_handoff_owner NOLOGIN PASSWORD NULL"
            )

        assert await first_clean_completion_schema_ready(pool) is True


@pytest.mark.asyncio
async def test_schema_readiness_reattests_receipt_guard_owners_and_paths() -> None:
    """A runtime cannot retain a stale ready state after guard drift."""

    async with _test_store() as (pool, schema):
        assert await first_clean_completion_schema_ready(pool) is True
        for function_name in _RECEIPT_GUARD_FUNCTIONS:
            function_ident = _quote_ident(function_name)
            try:
                await pool._connection.execute(
                    f"ALTER FUNCTION {function_ident}() OWNER TO atlas"
                )
                assert await first_clean_completion_schema_ready(pool) is False
            finally:
                await pool._connection.execute(
                    "ALTER FUNCTION "
                    f"{function_ident}() OWNER TO atlas_eom_handoff_owner"
                )
            assert await first_clean_completion_schema_ready(pool) is True

        schema_ident = _quote_ident(schema)
        for function_name in (
            *_PREREQUISITE_HANDOFF_ADMISSION_FUNCTIONS,
            *_RECEIPT_ADMISSION_FUNCTIONS,
        ):
            function_ident = _quote_ident(function_name)
            try:
                await pool._connection.execute(
                    f"ALTER FUNCTION {function_ident}() RESET search_path"
                )
                assert await first_clean_completion_schema_ready(pool) is False
            finally:
                await pool._connection.execute(
                    f"ALTER FUNCTION {function_ident}() "
                    f"SET search_path TO pg_catalog, {schema_ident}, pg_temp"
                )
            assert await first_clean_completion_schema_ready(pool) is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("privileged_option", "unprivileged_option"),
    (
        ("SUPERUSER", "NOSUPERUSER"),
        ("CREATEROLE", "NOCREATEROLE"),
        ("CREATEDB", "NOCREATEDB"),
        ("REPLICATION", "NOREPLICATION"),
        ("BYPASSRLS", "NOBYPASSRLS"),
    ),
)
async def test_completion_migration_rejects_privileged_atlas_runtime(
    privileged_option: str,
    unprivileged_option: str,
) -> None:
    """The DBA-only migration refuses to bless an elevated runtime login."""

    async with _test_store(
        migrations=tuple(
            migration
            for migration in _COMPLETION_SCHEMA_MIGRATIONS
            if migration != "394_eom_first_clean_completion_receipts"
        )
    ) as (pool, _schema):
        try:
            await pool._connection.execute(f"ALTER ROLE atlas {privileged_option}")
            with pytest.raises(
                asyncpg.RaiseError,
                match="atlas must be an unprivileged login runtime role",
            ):
                await pool._connection.execute(
                    (
                        MIGRATIONS / "394_eom_first_clean_completion_receipts.sql"
                    ).read_text()
                )
        finally:
            await pool._connection.execute(f"ALTER ROLE atlas {unprivileged_option}")


@pytest.mark.asyncio
@pytest.mark.parametrize("privileged_option", _EFFECTIVE_RUNTIME_PRIVILEGE_OPTIONS)
async def test_completion_migration_rejects_effective_runtime_privilege_membership(
    privileged_option: str,
) -> None:
    """The DBA path rejects every indirect elevated runtime role."""

    migrations = tuple(
        migration
        for migration in _COMPLETION_SCHEMA_MIGRATIONS
        if migration != "394_eom_first_clean_completion_receipts"
    )
    async with _test_store(migrations=migrations) as (pool, _schema):
        role_name = f"eom_completion_preflight_effective_{uuid4().hex[:16]}"
        role_ident = _quote_ident(role_name)
        role_created = False
        try:
            can_administer_roles = await pool._connection.fetchval(
                """
                SELECT rolsuper OR rolcreaterole
                FROM pg_roles
                WHERE rolname = current_user
                """
            )
            if not can_administer_roles:
                pytest.skip(
                    "effective-runtime migration proof requires role administration"
                )
            await pool._connection.execute(
                f"CREATE ROLE {role_ident} NOLOGIN NOINHERIT {privileged_option}"
            )
            role_created = True
            await pool._connection.execute(f"GRANT {role_ident} TO atlas")
            with pytest.raises(
                asyncpg.RaiseError,
                match="atlas must be an unprivileged login runtime role",
            ):
                await pool._connection.execute(
                    (
                        MIGRATIONS / "394_eom_first_clean_completion_receipts.sql"
                    ).read_text()
                )
            assert await pool._connection.fetchval(
                "SELECT to_regclass('eom_first_clean_completion_receipts') IS NULL"
            )
        finally:
            if role_created:
                await pool._connection.execute(f"REVOKE {role_ident} FROM atlas")
                await pool._connection.execute(f"DROP ROLE {role_ident}")


@pytest.mark.asyncio
async def test_completion_migration_rejects_atlas_database_ownership() -> None:
    """The DBA path rejects database ownership before it creates receipts."""

    migrations = tuple(
        migration
        for migration in _COMPLETION_SCHEMA_MIGRATIONS
        if migration != "394_eom_first_clean_completion_receipts"
    )
    async with _test_store(migrations=migrations) as (pool, _schema):
        database = await pool._connection.fetchrow(
            """
            SELECT database.datname, owner.rolname AS owner
            FROM pg_database AS database
            JOIN pg_roles AS owner ON owner.oid = database.datdba
            WHERE database.datname = current_database()
            """
        )
        assert database is not None
        database_ident = _quote_ident(str(database["datname"]))
        owner_ident = _quote_ident(str(database["owner"]))
        can_alter_database = await pool._connection.fetchval(
            """
            SELECT rolsuper OR rolcreatedb
            FROM pg_roles
            WHERE rolname = current_user
            """
        )
        if not can_alter_database:
            pytest.skip("database-owner migration proof requires database administration")
        try:
            await pool._connection.execute(
                f"ALTER DATABASE {database_ident} OWNER TO atlas"
            )
            with pytest.raises(
                asyncpg.RaiseError,
                match="atlas must be an unprivileged login runtime role",
            ):
                await pool._connection.execute(
                    (
                        MIGRATIONS / "394_eom_first_clean_completion_receipts.sql"
                    ).read_text()
                )
            assert await pool._connection.fetchval(
                "SELECT to_regclass('eom_first_clean_completion_receipts') IS NULL"
            )
        finally:
            await pool._connection.execute(
                f"ALTER DATABASE {database_ident} OWNER TO {owner_ident}"
            )
            # Keep the disposable runtime's explicit CREATE baseline after the
            # owner reversal so following isolated fixtures can create schemas.
            await pool._connection.execute(
                f"GRANT CREATE ON DATABASE {database_ident} TO atlas"
            )
        assert await pool._connection.fetchval(
            "SELECT has_database_privilege('atlas', current_database(), 'CREATE')"
        )


@pytest.mark.asyncio
async def test_completion_migration_rejects_elevated_runtime_before_guard_ddl() -> None:
    """A rejected runtime cannot silently normalize the protected guard role."""

    async with _test_store(
        migrations=tuple(
            migration
            for migration in _COMPLETION_SCHEMA_MIGRATIONS
            if migration != "394_eom_first_clean_completion_receipts"
        )
    ) as (pool, _schema):
        try:
            await pool._connection.execute("ALTER ROLE atlas_eom_handoff_owner LOGIN")
            await pool._connection.execute("ALTER ROLE atlas SUPERUSER")
            with pytest.raises(
                asyncpg.RaiseError,
                match="atlas must be an unprivileged login runtime role",
            ):
                await pool._connection.execute(
                    (
                        MIGRATIONS / "394_eom_first_clean_completion_receipts.sql"
                    ).read_text()
                )
            assert await pool._connection.fetchval(
                "SELECT rolcanlogin FROM pg_roles "
                "WHERE rolname = 'atlas_eom_handoff_owner'"
            )
        finally:
            await pool._connection.execute("ALTER ROLE atlas NOSUPERUSER")
            await pool._connection.execute("ALTER ROLE atlas_eom_handoff_owner NOLOGIN")


@pytest.mark.asyncio
async def test_completion_migration_rejects_preexisting_login_guard_before_ddl() -> None:
    """A guard with an authenticated-session path cannot be normalized in place."""

    async with _test_store(
        migrations=tuple(
            migration
            for migration in _COMPLETION_SCHEMA_MIGRATIONS
            if migration != "394_eom_first_clean_completion_receipts"
        )
    ) as (pool, _schema):
        try:
            await pool._connection.execute("ALTER ROLE atlas_eom_handoff_owner LOGIN")
            with pytest.raises(
                asyncpg.RaiseError,
                match="pre-existing atlas_eom_handoff_owner must be NOLOGIN",
            ):
                await pool._connection.execute(
                    (
                        MIGRATIONS / "394_eom_first_clean_completion_receipts.sql"
                    ).read_text()
                )
            assert await pool._connection.fetchval(
                "SELECT rolcanlogin FROM pg_roles "
                "WHERE rolname = 'atlas_eom_handoff_owner'"
            )
            assert await pool._connection.fetchval(
                "SELECT to_regclass('eom_first_clean_completion_receipts') IS NULL"
            )
        finally:
            await pool._connection.execute("ALTER ROLE atlas_eom_handoff_owner NOLOGIN")


@pytest.mark.asyncio
async def test_completion_migration_rejects_live_former_guard_session_before_ddl() -> (
    None
):
    """A historical guard session must not gain a newly transferred boundary."""

    migrations = tuple(
        migration
        for migration in _COMPLETION_SCHEMA_MIGRATIONS
        if migration != "394_eom_first_clean_completion_receipts"
    )
    guard_connection: Any | None = None
    async with _test_store(migrations=migrations) as (pool, _schema):
        try:
            await pool._connection.execute(
                "ALTER ROLE atlas_eom_handoff_owner LOGIN PASSWORD 'test-guard-session'"
            )
            guard_connection = await asyncpg.connect(
                _dba_database_url_or_skip(),
                user="atlas_eom_handoff_owner",
                password="test-guard-session",
            )
            await pool._connection.execute("ALTER ROLE atlas_eom_handoff_owner NOLOGIN")

            with pytest.raises(
                asyncpg.RaiseError,
                match="atlas_eom_handoff_owner must have no live sessions",
            ):
                await pool._connection.execute(
                    (
                        MIGRATIONS / "394_eom_first_clean_completion_receipts.sql"
                    ).read_text()
                )
            assert await pool._connection.fetchval(
                "SELECT to_regclass('eom_first_clean_completion_receipts') IS NULL"
            )
        finally:
            if guard_connection is not None:
                await guard_connection.close()
            await pool._connection.execute(
                "ALTER ROLE atlas_eom_handoff_owner NOLOGIN PASSWORD NULL"
            )


@pytest.mark.asyncio
async def test_completion_migration_requires_canonical_lifecycle_sequence_before_ddl() -> (
    None
):
    """A partial migration chain cannot create receipts with a broken default."""

    async with _test_store(
        migrations=tuple(
            migration
            for migration in _COMPLETION_SCHEMA_MIGRATIONS
            if migration
            not in {
                "363_eom_lead_lifecycle_sequence",
                "394_eom_first_clean_completion_receipts",
            }
        )
    ) as (pool, _schema):
        with pytest.raises(
            asyncpg.RaiseError,
            match="its ordering sequence",
        ):
            await pool._connection.execute(
                (MIGRATIONS / "394_eom_first_clean_completion_receipts.sql").read_text()
            )
        assert await pool._connection.fetchval(
            "SELECT to_regclass('eom_first_clean_completion_receipts') IS NULL"
        )


@pytest.mark.asyncio
async def test_completion_migration_requires_lifecycle_default_before_ddl() -> None:
    """An owned sequence alone cannot substitute for the canonical nextval default."""

    async with _test_store(
        migrations=tuple(
            migration
            for migration in _COMPLETION_SCHEMA_MIGRATIONS
            if migration != "394_eom_first_clean_completion_receipts"
        )
    ) as (pool, _schema):
        await pool._connection.execute(
            "ALTER TABLE eom_lead_lifecycle_events "
            "ALTER COLUMN lifecycle_sequence DROP DEFAULT"
        )
        with pytest.raises(
            asyncpg.RaiseError,
            match="canonical nextval default",
        ):
            await pool._connection.execute(
                (MIGRATIONS / "394_eom_first_clean_completion_receipts.sql").read_text()
            )
        assert await pool._connection.fetchval(
            "SELECT to_regclass('eom_first_clean_completion_receipts') IS NULL"
        )


@pytest.mark.asyncio
async def test_completion_migration_requires_guard_owned_handoff_prerequisites() -> (
    None
):
    async with _test_store(
        migrations=tuple(
            migration
            for migration in _COMPLETION_SCHEMA_MIGRATIONS
            if migration
            not in {
                "354_eom_customer_handoff_privileges",
                "394_eom_first_clean_completion_receipts",
            }
        )
    ) as (pool, _schema):
        with pytest.raises(
            asyncpg.RaiseError,
            match=(
                "eom_customer_handoffs and its protected functions must be guard-owned"
            ),
        ):
            await pool._connection.execute(
                (MIGRATIONS / "394_eom_first_clean_completion_receipts.sql").read_text()
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("relation_name", "constraint_name"),
    (
        ("eom_customer_handoffs", "eom_customer_handoffs_contact_id_fkey"),
        ("eom_lead_lifecycle_events", "eom_lead_lifecycle_events_contact_id_fkey"),
    ),
)
async def test_completion_migration_requires_canonical_contact_foreign_keys(
    relation_name: str,
    constraint_name: str,
) -> None:
    """A compromised prerequisite cannot be blessed by the DBA migration."""

    migrations = tuple(
        migration
        for migration in _COMPLETION_SCHEMA_MIGRATIONS
        if migration != "394_eom_first_clean_completion_receipts"
    )
    async with _test_store(migrations=migrations) as (pool, _schema):
        await pool._connection.execute(
            "ALTER TABLE "
            f"{_quote_ident(relation_name)} DROP CONSTRAINT "
            f"{_quote_ident(constraint_name)}"
        )
        with pytest.raises(
            asyncpg.RaiseError,
            match="must retain validated contact foreign keys",
        ):
            await pool._connection.execute(
                (MIGRATIONS / "394_eom_first_clean_completion_receipts.sql").read_text()
            )
        assert await pool._connection.fetchval(
            "SELECT to_regclass('eom_first_clean_completion_receipts') IS NULL"
        )


@pytest.mark.asyncio
async def test_completion_migration_rejects_non_superuser_login_guard_membership() -> None:
    """A foreign login cannot become the guard owner during the DBA migration."""

    migrations = tuple(
        migration
        for migration in _COMPLETION_SCHEMA_MIGRATIONS
        if migration != "394_eom_first_clean_completion_receipts"
    )
    async with _test_store(migrations=migrations) as (pool, _schema):
        login_name = f"eom_completion_preflight_login_{uuid4().hex[:16]}"
        login_ident = _quote_ident(login_name)
        login_created = False
        try:
            can_administer_roles = await pool._connection.fetchval(
                """
                SELECT rolsuper OR rolcreaterole
                FROM pg_roles
                WHERE rolname = current_user
                """
            )
            if not can_administer_roles:
                pytest.skip(
                    "guard-membership preflight proof requires role administration"
                )
            await pool._connection.execute(
                f"CREATE ROLE {login_ident} LOGIN NOINHERIT NOSUPERUSER"
            )
            login_created = True
            await pool._connection.execute(
                f"GRANT atlas_eom_handoff_owner TO {login_ident}"
            )
            with pytest.raises(
                asyncpg.RaiseError,
                match="no non-superuser login may retain direct or inherited membership",
            ):
                await pool._connection.execute(
                    (MIGRATIONS / "394_eom_first_clean_completion_receipts.sql").read_text()
                )
            assert await pool._connection.fetchval(
                "SELECT to_regclass('eom_first_clean_completion_receipts') IS NULL"
            )
        finally:
            if login_created:
                await pool._connection.execute(
                    f"REVOKE atlas_eom_handoff_owner FROM {login_ident}"
                )
                await pool._connection.execute(f"DROP ROLE {login_ident}")


@pytest.mark.asyncio
async def test_schema_readiness_rejects_non_superuser_login_guard_membership() -> None:
    async with _test_store() as (pool, _schema):
        assert await first_clean_completion_schema_ready(pool) is True
        login_name = f"eom_completion_login_{uuid4().hex[:16]}"
        login_ident = _quote_ident(login_name)
        intermediary_name = f"eom_completion_membership_{uuid4().hex[:16]}"
        intermediary_ident = _quote_ident(intermediary_name)
        login_created = False
        intermediary_created = False
        try:
            can_administer_roles = await pool._connection.fetchval(
                """
                SELECT rolsuper OR rolcreaterole
                FROM pg_roles
                WHERE rolname = current_user
                """
            )
            if not can_administer_roles:
                pytest.skip(
                    "guard-membership proof requires disposable role administration"
                )
            await pool._connection.execute(
                f"CREATE ROLE {login_ident} LOGIN NOINHERIT NOSUPERUSER"
            )
            login_created = True
            await pool._connection.execute(
                f"GRANT atlas_eom_handoff_owner TO {login_ident}"
            )
            assert await first_clean_completion_schema_ready(pool) is False
            await pool._connection.execute(
                f"REVOKE atlas_eom_handoff_owner FROM {login_ident}"
            )
            assert await first_clean_completion_schema_ready(pool) is True
            await pool._connection.execute(
                f"CREATE ROLE {intermediary_ident} NOLOGIN NOINHERIT"
            )
            intermediary_created = True
            await pool._connection.execute(
                f"GRANT atlas_eom_handoff_owner TO {intermediary_ident}"
            )
            await pool._connection.execute(
                f"GRANT {intermediary_ident} TO {login_ident}"
            )
            assert await first_clean_completion_schema_ready(pool) is False
        finally:
            if intermediary_created:
                await pool._connection.execute(
                    f"REVOKE {intermediary_ident} FROM {login_ident}"
                )
                await pool._connection.execute(
                    f"REVOKE atlas_eom_handoff_owner FROM {intermediary_ident}"
                )
                await pool._connection.execute(f"DROP ROLE {intermediary_ident}")
            if login_created:
                await pool._connection.execute(
                    f"REVOKE atlas_eom_handoff_owner FROM {login_ident}"
                )
                await pool._connection.execute(f"DROP ROLE {login_ident}")
        assert await first_clean_completion_schema_ready(pool) is True


@pytest.mark.asyncio
async def test_schema_readiness_rejects_non_guard_table_ownership() -> None:
    async with _test_store() as (pool, _schema):
        assert await first_clean_completion_schema_ready(pool) is True
        try:
            await pool._connection.execute(
                """
                ALTER TABLE eom_first_clean_completion_receipts
                OWNER TO atlas
                """
            )
            assert await first_clean_completion_schema_ready(pool) is False
        finally:
            await pool._connection.execute(
                """
                ALTER TABLE eom_first_clean_completion_receipts
                OWNER TO atlas_eom_handoff_owner
                """
            )
            await _grant_completion_runtime_dml_as_guard(pool._connection)
        assert await first_clean_completion_schema_ready(pool) is True


@pytest.mark.asyncio
async def test_schema_readiness_requires_guard_owned_namespace() -> None:
    """Relation ownership is insufficient while Atlas owns their schema."""

    async with _test_store() as (pool, schema):
        schema_ident = _quote_ident(schema)
        assert await first_clean_completion_schema_ready(pool) is True
        try:
            await pool._connection.execute(
                f"ALTER SCHEMA {schema_ident} OWNER TO atlas"
            )
            assert await first_clean_completion_schema_ready(pool) is False
        finally:
            await pool._connection.execute(
                f"ALTER SCHEMA {schema_ident} OWNER TO atlas_eom_handoff_owner"
            )
            # Re-owning the schema does not restore the explicit runtime schema
            # grants that migration 394 establishes. Restore the complete
            # guarded baseline before asserting readiness again.
            assert await first_clean_completion_schema_ready(pool) is False
            await pool._connection.execute(
                f"GRANT USAGE, CREATE ON SCHEMA {schema_ident} TO atlas"
            )
        assert await first_clean_completion_schema_ready(pool) is True


@pytest.mark.asyncio
async def test_schema_readiness_requires_guard_owned_lifecycle_boundary() -> None:
    async with _test_store() as (pool, _schema):
        assert await first_clean_completion_schema_ready(pool) is True
        try:
            await pool._connection.execute(
                "ALTER TABLE eom_lead_lifecycle_events OWNER TO atlas"
            )
            assert await first_clean_completion_schema_ready(pool) is False
        finally:
            await pool._connection.execute(
                "ALTER TABLE eom_lead_lifecycle_events OWNER TO atlas_eom_handoff_owner"
            )
            await _grant_completion_runtime_dml_as_guard(pool._connection)
        assert await first_clean_completion_schema_ready(pool) is True

        try:
            await pool._connection.execute(
                "ALTER FUNCTION prevent_eom_lead_lifecycle_event_mutation() "
                "OWNER TO atlas"
            )
            assert await first_clean_completion_schema_ready(pool) is False
        finally:
            await pool._connection.execute(
                "ALTER FUNCTION prevent_eom_lead_lifecycle_event_mutation() "
                "OWNER TO atlas_eom_handoff_owner"
            )
        assert await first_clean_completion_schema_ready(pool) is True


@pytest.mark.asyncio
async def test_schema_readiness_rejects_lifecycle_trigger_bound_to_other_function() -> (
    None
):
    async with _test_store() as (pool, _schema):
        assert await first_clean_completion_schema_ready(pool) is True
        definitions: list[str] = []
        try:
            await pool._connection.execute(
                """
                CREATE FUNCTION eom_lifecycle_guard_probe()
                RETURNS TRIGGER
                LANGUAGE plpgsql
                AS $$
                BEGIN
                    RETURN NEW;
                END;
                $$
                """
            )
            await pool._connection.execute(
                "ALTER FUNCTION eom_lifecycle_guard_probe() "
                "OWNER TO atlas_eom_handoff_owner"
            )
            for trigger_name in _LIFECYCLE_GUARD_TRIGGERS:
                definition = await pool._connection.fetchval(
                    """
                    SELECT pg_get_triggerdef(trigger.oid)
                    FROM pg_trigger AS trigger
                    WHERE trigger.tgrelid = 'eom_lead_lifecycle_events'::regclass
                      AND trigger.tgname = $1
                    """,
                    trigger_name,
                )
                assert isinstance(definition, str)
                definitions.append(definition)
                trigger_ident = _quote_ident(trigger_name)
                await pool._connection.execute(
                    f"DROP TRIGGER {trigger_ident} ON eom_lead_lifecycle_events"
                )
                if trigger_name.endswith("mutation"):
                    await pool._connection.execute(
                        "CREATE TRIGGER "
                        f"{trigger_ident} BEFORE UPDATE OR DELETE "
                        "ON eom_lead_lifecycle_events FOR EACH ROW "
                        "EXECUTE FUNCTION eom_lifecycle_guard_probe()"
                    )
                else:
                    await pool._connection.execute(
                        "CREATE TRIGGER "
                        f"{trigger_ident} BEFORE TRUNCATE "
                        "ON eom_lead_lifecycle_events FOR EACH STATEMENT "
                        "EXECUTE FUNCTION eom_lifecycle_guard_probe()"
                    )

            assert await first_clean_completion_schema_ready(pool) is False
        finally:
            for trigger_name in _LIFECYCLE_GUARD_TRIGGERS:
                trigger_ident = _quote_ident(trigger_name)
                await pool._connection.execute(
                    "DROP TRIGGER IF EXISTS "
                    f"{trigger_ident} ON eom_lead_lifecycle_events"
                )
            for definition in definitions:
                await pool._connection.execute(definition)
            await pool._connection.execute(
                "DROP FUNCTION IF EXISTS eom_lifecycle_guard_probe()"
            )
        assert await first_clean_completion_schema_ready(pool) is True


@pytest.mark.asyncio
async def test_schema_readiness_rejects_guard_fk_access_stripped_by_acl_cleanup() -> (
    None
):
    async with _test_store() as (pool, _schema):
        await pool._connection.execute("SET ROLE atlas_eom_handoff_owner")
        try:
            await pool._connection.execute(
                "REVOKE SELECT, UPDATE ON TABLE "
                "eom_first_clean_completion_operation_receipts "
                "FROM atlas_eom_handoff_owner"
            )
        finally:
            await pool._connection.execute("RESET ROLE")

        assert await first_clean_completion_schema_ready(pool) is False


@pytest.mark.asyncio
async def test_schema_readiness_requires_guard_owned_handoff_and_functions() -> None:
    async with _test_store() as (pool, _schema):
        assert await first_clean_completion_schema_ready(pool) is True
        try:
            await pool._connection.execute(
                "ALTER TABLE eom_customer_handoffs OWNER TO atlas"
            )
            assert await first_clean_completion_schema_ready(pool) is False
        finally:
            await pool._connection.execute(
                "ALTER TABLE eom_customer_handoffs OWNER TO atlas_eom_handoff_owner"
            )
            await pool._connection.execute(
                """
                GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE
                ON TABLE eom_customer_handoffs TO atlas
                """
            )
        assert await first_clean_completion_schema_ready(pool) is True

        for function_name in _PREREQUISITE_HANDOFF_GUARD_FUNCTIONS:
            function_ident = _quote_ident(function_name)
            try:
                await pool._connection.execute(
                    f"ALTER FUNCTION {function_ident}() OWNER TO atlas"
                )
                assert await first_clean_completion_schema_ready(pool) is False
            finally:
                await pool._connection.execute(
                    "ALTER FUNCTION "
                    f"{function_ident}() OWNER TO atlas_eom_handoff_owner"
                )
            assert await first_clean_completion_schema_ready(pool) is True


@pytest.mark.asyncio
async def test_schema_readiness_rejects_broadened_runtime_or_external_acl() -> None:
    async with _test_store() as (pool, _schema):
        assert await first_clean_completion_schema_ready(pool) is True
        role_name = f"eom_completion_acl_{uuid4().hex[:16]}"
        role_ident = _quote_ident(role_name)
        role_created = False
        try:
            can_administer_roles = await pool._connection.fetchval(
                """
                SELECT rolsuper OR rolcreaterole
                FROM pg_roles
                WHERE rolname = current_user
                """
            )
            if not can_administer_roles:
                pytest.skip("ACL proof requires disposable role administration")
            await pool._connection.execute(
                f"CREATE ROLE {role_ident} NOLOGIN NOINHERIT"
            )
            role_created = True
            await pool._connection.execute(
                "REVOKE INSERT ON TABLE eom_lead_lifecycle_events FROM atlas"
            )
            assert await first_clean_completion_schema_ready(pool) is False
            await _grant_completion_runtime_dml_as_guard(pool._connection)
            assert await first_clean_completion_schema_ready(pool) is True
            await pool._connection.execute(
                """
                GRANT DELETE ON TABLE eom_first_clean_completion_receipts TO atlas
                """
            )
            assert await first_clean_completion_schema_ready(pool) is False
            await pool._connection.execute(
                """
                REVOKE DELETE ON TABLE eom_first_clean_completion_receipts FROM atlas
                """
            )
            await pool._connection.execute(
                "REVOKE UPDATE ON TABLE eom_lead_lifecycle_events FROM atlas"
            )
            assert await first_clean_completion_schema_ready(pool) is False
            await _grant_completion_runtime_dml_as_guard(pool._connection)
            assert await first_clean_completion_schema_ready(pool) is True
            await pool._connection.execute(
                "GRANT SELECT ON TABLE eom_first_clean_completion_receipts "
                f"TO {role_ident}"
            )
            assert await first_clean_completion_schema_ready(pool) is False
        finally:
            await pool._connection.execute(
                """
                REVOKE DELETE ON TABLE eom_first_clean_completion_receipts FROM atlas
                """
            )
            if role_created:
                await pool._connection.execute(
                    "REVOKE ALL PRIVILEGES ON TABLE "
                    f"eom_first_clean_completion_receipts FROM {role_ident}"
                )
                await pool._connection.execute(f"DROP ROLE {role_ident}")
        assert await first_clean_completion_schema_ready(pool) is True


@pytest.mark.asyncio
async def test_runtime_can_lock_existing_lifecycle_evidence_but_cannot_mutate() -> None:
    """Existing operator/booking row locks work without weakening append-only evidence."""

    async with _test_store() as (pool, _schema):
        contact_id, customer_id, site_id = await _insert_customer(pool)
        assert customer_id is not None and site_id is not None
        operation_key = _operation_key("runtime-lifecycle-lock")
        await _service(pool).record_completion(
            **_completion_kwargs(
                contact_id=contact_id,
                tracker_customer_id=customer_id,
                tracker_site_id=site_id,
                operation_key=operation_key,
            )
        )

        runtime_connection = pool._runtime_connection
        async with runtime_connection.transaction():
            lifecycle = await runtime_connection.fetchrow(
                """
                SELECT contact_id, event_type, operation_key
                FROM eom_lead_lifecycle_events
                WHERE operation_key = $1
                  AND event_type = 'first_clean_completed'
                FOR UPDATE
                """,
                operation_key,
            )
            assert lifecycle is not None
            assert lifecycle["contact_id"] == contact_id

        with pytest.raises(
            asyncpg.RaiseError,
            match="eom_lead_lifecycle_events is append-only",
        ):
            await runtime_connection.execute(
                """
                UPDATE eom_lead_lifecycle_events
                SET actor = actor
                WHERE operation_key = $1
                """,
                operation_key,
            )


@pytest.mark.asyncio
async def test_schema_readiness_requires_guarded_lifecycle_sequence_usage() -> None:
    """The lifecycle default is usable by runtime, but no broader sequence ACL."""

    async with _test_store() as (pool, _schema):
        sequence_name = "eom_lead_lifecycle_events_sequence_seq"
        assert await first_clean_completion_schema_ready(pool) is True
        await pool._connection.execute(
            f"REVOKE USAGE ON SEQUENCE {sequence_name} FROM atlas"
        )
        assert await first_clean_completion_schema_ready(pool) is False
        await pool._connection.execute(
            f"GRANT USAGE ON SEQUENCE {sequence_name} TO atlas"
        )
        assert await first_clean_completion_schema_ready(pool) is True
        await pool._connection.execute(
            f"GRANT SELECT ON SEQUENCE {sequence_name} TO atlas"
        )
        assert await first_clean_completion_schema_ready(pool) is False
        await pool._connection.execute(
            f"REVOKE SELECT ON SEQUENCE {sequence_name} FROM atlas"
        )
        assert await first_clean_completion_schema_ready(pool) is True
        await pool._connection.execute(
            f"GRANT USAGE ON SEQUENCE {sequence_name} TO atlas WITH GRANT OPTION"
        )
        assert await first_clean_completion_schema_ready(pool) is False
        await pool._connection.execute(
            f"REVOKE GRANT OPTION FOR USAGE ON SEQUENCE {sequence_name} FROM atlas"
        )
        assert await first_clean_completion_schema_ready(pool) is True
        external_role_name = f"eom_sequence_acl_{uuid4().hex[:16]}"
        external_role_ident = _quote_ident(external_role_name)
        await pool._connection.execute(
            f"CREATE ROLE {external_role_ident} NOLOGIN NOINHERIT"
        )
        try:
            await pool._connection.execute(
                f"GRANT USAGE ON SEQUENCE {sequence_name} TO {external_role_ident}"
            )
            assert await first_clean_completion_schema_ready(pool) is False
        finally:
            await pool._connection.execute(
                f"REVOKE ALL PRIVILEGES ON SEQUENCE {sequence_name} "
                f"FROM {external_role_ident}"
            )
            await pool._connection.execute(f"DROP ROLE {external_role_ident}")
        assert await first_clean_completion_schema_ready(pool) is True
        await pool._connection.execute(
            f"GRANT USAGE ON SEQUENCE {sequence_name} TO PUBLIC"
        )
        assert await first_clean_completion_schema_ready(pool) is False
        await pool._connection.execute(
            f"REVOKE ALL PRIVILEGES ON SEQUENCE {sequence_name} FROM PUBLIC"
        )
        assert await first_clean_completion_schema_ready(pool) is True
        await pool._connection.execute(
            f"ALTER SEQUENCE {sequence_name} OWNED BY NONE"
        )
        await pool._connection.execute(f"ALTER SEQUENCE {sequence_name} OWNER TO atlas")
        assert await first_clean_completion_schema_ready(pool) is False
        await pool._connection.execute(
            f"ALTER SEQUENCE {sequence_name} OWNER TO atlas_eom_handoff_owner"
        )
        await pool._connection.execute(
            f"GRANT USAGE ON SEQUENCE {sequence_name} TO atlas"
        )
        assert await first_clean_completion_schema_ready(pool) is False
        await pool._connection.execute(
            f"ALTER SEQUENCE {sequence_name} OWNED BY "
            "eom_lead_lifecycle_events.lifecycle_sequence"
        )
        assert await first_clean_completion_schema_ready(pool) is True
        await pool._connection.execute(
            "ALTER TABLE eom_lead_lifecycle_events "
            "ALTER COLUMN lifecycle_sequence DROP DEFAULT"
        )
        assert await first_clean_completion_schema_ready(pool) is False
        await pool._connection.execute(
            "ALTER TABLE eom_lead_lifecycle_events "
            "ALTER COLUMN lifecycle_sequence SET DEFAULT "
            "nextval('eom_lead_lifecycle_events_sequence_seq'::regclass)"
        )
        assert await first_clean_completion_schema_ready(pool) is True


@pytest.mark.asyncio
async def test_schema_readiness_rejects_missing_or_disabled_prerequisite_guards() -> (
    None
):
    """Receipt source evidence is valid only while its existing guards are live."""

    async with _test_store() as (pool, _schema):
        assert await first_clean_completion_schema_ready(pool) is True
        for relation_name, trigger_name in _PREREQUISITE_INTEGRITY_TRIGGERS:
            relation_ident = _quote_ident(relation_name)
            trigger_ident = _quote_ident(trigger_name)
            trigger_definition = await pool._connection.fetchval(
                """
                SELECT pg_get_triggerdef(trigger.oid)
                FROM pg_trigger AS trigger
                WHERE trigger.tgrelid = $1::regclass
                  AND trigger.tgname = $2
                """,
                relation_name,
                trigger_name,
            )
            assert isinstance(trigger_definition, str)

            await pool._connection.execute(
                f"ALTER TABLE {relation_ident} DISABLE TRIGGER {trigger_ident}"
            )
            assert await first_clean_completion_schema_ready(pool) is False
            await pool._connection.execute(
                f"ALTER TABLE {relation_ident} ENABLE TRIGGER {trigger_ident}"
            )
            assert await first_clean_completion_schema_ready(pool) is True

            await pool._connection.execute(
                f"DROP TRIGGER {trigger_ident} ON {relation_ident}"
            )
            assert await first_clean_completion_schema_ready(pool) is False
            await pool._connection.execute(trigger_definition)
            assert await first_clean_completion_schema_ready(pool) is True


@pytest.mark.asyncio
async def test_completion_schema_rejects_a_non_superuser_executor() -> None:
    database_url = _dba_database_url_or_skip()
    role_name = f"eom_completion_runtime_{uuid4().hex[:16]}"
    role_ident = _quote_ident(role_name)
    role_created = False
    connection = await asyncpg.connect(database_url)
    try:
        await connection.execute(f"CREATE ROLE {role_ident} NOLOGIN NOINHERIT")
        role_created = True
        await connection.execute(f"SET ROLE {role_ident}")
        with pytest.raises(
            asyncpg.RaiseError,
            match="database administrator must run 394_eom_first_clean_completion_receipts",
        ):
            await connection.execute(
                (MIGRATIONS / "394_eom_first_clean_completion_receipts.sql").read_text()
            )
        await connection.execute("RESET ROLE")
    finally:
        await connection.execute("RESET ROLE")
        if role_created:
            await connection.execute(f"DROP ROLE {role_ident}")
        await connection.close()


@pytest.fixture(autouse=True)
def _reset_capability_cache() -> None:
    funnel_mod._served_capabilities_cache = None
    yield
    funnel_mod._served_capabilities_cache = None


@pytest.mark.asyncio
async def test_record_completion_persists_one_receipt_and_lifecycle_evidence() -> None:
    async with _test_store() as (pool, _schema):
        contact_id, customer_id, site_id = await _insert_customer(pool)
        assert customer_id is not None and site_id is not None
        service = _service(pool)

        result = await service.record_completion(
            **_completion_kwargs(
                contact_id=contact_id,
                tracker_customer_id=customer_id,
                tracker_site_id=site_id,
                operation_key=_operation_key(),
            )
        )

        assert result["idempotent"] is False
        assert result["contactId"] == str(contact_id)
        assert result["trackerServiceKind"] == "job"
        assert result["completedAt"] == "2026-08-24T19:00:00Z"
        assert (
            await pool.fetchval(
                "SELECT COUNT(*) FROM eom_first_clean_completion_receipts"
            )
            == 1
        )
        lifecycle = await pool._connection.fetchrow(
            """
            SELECT event_type, source, operation_key, metadata
            FROM eom_lead_lifecycle_events
            WHERE contact_id = $1 AND event_type = 'first_clean_completed'
            """,
            contact_id,
        )
        assert lifecycle is not None
        assert lifecycle["source"] == "time_tracker"
        assert lifecycle["operation_key"]
        metadata = lifecycle["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        assert isinstance(metadata, dict)
        assert metadata["completion_receipt_id"] == result["receiptId"]
        assert metadata["handoff_id"] == result["handoffId"]
        assert metadata["tracker_service_id"] == 6001


@pytest.mark.asyncio
async def test_unchanged_retry_returns_original_receipt_without_second_write() -> None:
    async with _test_store() as (pool, _schema):
        contact_id, customer_id, site_id = await _insert_customer(pool)
        assert customer_id is not None and site_id is not None
        service = _service(pool)
        kwargs = _completion_kwargs(
            contact_id=contact_id,
            tracker_customer_id=customer_id,
            tracker_site_id=site_id,
            operation_key=_operation_key(),
        )

        first = await service.record_completion(**kwargs)
        replay = await service.record_completion(**kwargs)

        assert first["receiptId"] == replay["receiptId"]
        assert first["idempotent"] is False
        assert replay["idempotent"] is True
        assert (
            await pool.fetchval(
                "SELECT COUNT(*) FROM eom_first_clean_completion_receipts"
            )
            == 1
        )
        assert (
            await pool.fetchval(
                "SELECT COUNT(*) FROM eom_first_clean_completion_operation_receipts"
            )
            == 1
        )


@pytest.mark.asyncio
async def test_changed_retry_or_new_source_fails_without_rewriting_history() -> None:
    async with _test_store() as (pool, _schema):
        contact_id, customer_id, site_id = await _insert_customer(pool)
        assert customer_id is not None and site_id is not None
        service = _service(pool)
        key = _operation_key()
        kwargs = _completion_kwargs(
            contact_id=contact_id,
            tracker_customer_id=customer_id,
            tracker_site_id=site_id,
            operation_key=key,
        )
        first = await service.record_completion(**kwargs)

        with pytest.raises(EOMFirstCleanCompletionConflictError):
            await service.record_completion(
                **{
                    **kwargs,
                    "completed_at": _NOW - timedelta(minutes=30),
                }
            )
        with pytest.raises(EOMFirstCleanCompletionConflictError):
            await service.record_completion(
                **{
                    **kwargs,
                    "actor_id": 8,
                    "actor_name": "Other operator",
                }
            )
        with pytest.raises(EOMFirstCleanCompletionConflictError):
            await service.record_completion(
                **{
                    **kwargs,
                    "actor_name": "Renamed operator",
                }
            )
        with pytest.raises(EOMFirstCleanCompletionConflictError):
            await service.record_completion(
                **{
                    **kwargs,
                    "operation_key": _operation_key("new-key"),
                    "tracker_service_id": 6002,
                }
            )

        row = await pool._connection.fetchrow(
            """
            SELECT id, tracker_service_id, completed_at
            FROM eom_first_clean_completion_receipts
            WHERE contact_id = $1
            """,
            contact_id,
        )
        assert str(row["id"]) == first["receiptId"]
        assert row["tracker_service_id"] == 6001
        assert row["completed_at"] == _NOW - timedelta(hours=1)
        assert (
            await pool.fetchval(
                "SELECT COUNT(*) FROM eom_first_clean_completion_operation_receipts"
            )
            == 1
        )


@pytest.mark.asyncio
async def test_cross_contact_key_and_handoff_scope_conflicts_leave_no_receipt() -> None:
    async with _test_store() as (pool, _schema):
        first_contact, first_customer, first_site = await _insert_customer(pool)
        second_contact, second_customer, second_site = await _insert_customer(pool)
        assert all(
            value is not None
            for value in (first_customer, first_site, second_customer, second_site)
        )
        service = _service(pool)
        key = _operation_key()
        await service.record_completion(
            **_completion_kwargs(
                contact_id=first_contact,
                tracker_customer_id=first_customer,
                tracker_site_id=first_site,
                operation_key=key,
            )
        )

        with pytest.raises(EOMFirstCleanCompletionConflictError):
            await service.record_completion(
                **_completion_kwargs(
                    contact_id=second_contact,
                    tracker_customer_id=second_customer,
                    tracker_site_id=second_site,
                    operation_key=key,
                    tracker_service_id=7001,
                )
            )
        with pytest.raises(EOMFirstCleanCompletionConflictError):
            await service.record_completion(
                **_completion_kwargs(
                    contact_id=second_contact,
                    tracker_customer_id=second_customer,
                    tracker_site_id=second_site + 1,
                    operation_key=_operation_key("wrong-handoff"),
                    tracker_service_id=7002,
                )
            )

        assert (
            await pool.fetchval(
                "SELECT COUNT(*) FROM eom_first_clean_completion_receipts"
            )
            == 1
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("customer_type", "status", "with_handoff"),
    [
        ("commercial", "active", True),
        ("unknown", "active", True),
        # Migration 353 admits canonical handoffs only after an active customer
        # transition; eligibility rejects this contact before handoff lookup.
        ("residential", "inactive", False),
        ("residential", "active", False),
    ],
)
async def test_ineligible_or_unlinked_customer_cannot_record_completion(
    customer_type: str,
    status: str,
    with_handoff: bool,
) -> None:
    async with _test_store() as (pool, _schema):
        contact_id, customer_id, site_id = await _insert_customer(
            pool,
            customer_type=customer_type,
            status=status,
            with_handoff=with_handoff,
        )
        service = _service(pool)
        with pytest.raises(EOMFirstCleanCompletionConflictError):
            await service.record_completion(
                **_completion_kwargs(
                    contact_id=contact_id,
                    tracker_customer_id=customer_id or 7001,
                    tracker_site_id=site_id or 7002,
                    operation_key=_operation_key(),
                )
            )
        assert (
            await pool.fetchval(
                "SELECT COUNT(*) FROM eom_first_clean_completion_receipts"
            )
            == 0
        )


@pytest.mark.asyncio
async def test_concurrent_same_operation_creates_one_receipt() -> None:
    database_url = _database_url_or_skip()
    async with _test_store() as (pool, schema):
        contact_id, customer_id, site_id = await _insert_customer(pool)
        assert customer_id is not None and site_id is not None
        first_pool = await _schema_connection(database_url, schema)
        second_pool = await _schema_connection(database_url, schema)
        kwargs = _completion_kwargs(
            contact_id=contact_id,
            tracker_customer_id=customer_id,
            tracker_site_id=site_id,
            operation_key=_operation_key("concurrent"),
        )
        try:
            first, second = await asyncio.gather(
                _service(first_pool).record_completion(**kwargs),
                _service(second_pool).record_completion(**kwargs),
            )
        finally:
            await _close_schema_connection(first_pool)
            await _close_schema_connection(second_pool)

        assert sorted((first["idempotent"], second["idempotent"])) == [False, True]
        assert first["receiptId"] == second["receiptId"]
        assert (
            await pool.fetchval(
                "SELECT COUNT(*) FROM eom_first_clean_completion_receipts"
            )
            == 1
        )


@pytest.mark.asyncio
async def test_concurrent_distinct_operations_for_one_customer_fail_closed() -> None:
    database_url = _database_url_or_skip()
    async with _test_store() as (pool, schema):
        contact_id, customer_id, site_id = await _insert_customer(pool)
        assert customer_id is not None and site_id is not None
        first_pool = await _schema_connection(database_url, schema)
        second_pool = await _schema_connection(database_url, schema)
        first_kwargs = _completion_kwargs(
            contact_id=contact_id,
            tracker_customer_id=customer_id,
            tracker_site_id=site_id,
            operation_key=_operation_key("concurrent-first"),
            tracker_service_id=6001,
        )
        second_kwargs = _completion_kwargs(
            contact_id=contact_id,
            tracker_customer_id=customer_id,
            tracker_site_id=site_id,
            operation_key=_operation_key("concurrent-second"),
            tracker_service_id=6002,
        )
        try:
            first, second = await asyncio.gather(
                _service(first_pool).record_completion(**first_kwargs),
                _service(second_pool).record_completion(**second_kwargs),
                return_exceptions=True,
            )
        finally:
            await _close_schema_connection(first_pool)
            await _close_schema_connection(second_pool)

        outcomes = (first, second)
        assert sum(isinstance(item, dict) for item in outcomes) == 1
        assert (
            sum(
                isinstance(item, EOMFirstCleanCompletionConflictError)
                for item in outcomes
            )
            == 1
        )
        assert (
            await pool.fetchval(
                "SELECT COUNT(*) FROM eom_first_clean_completion_receipts"
            )
            == 1
        )


@pytest.mark.asyncio
async def test_private_route_forwards_only_the_closed_completion_contract() -> None:
    captured: dict[str, Any] = {}

    class _CompletionSpy:
        async def require_schema_ready(self) -> None:
            captured["schema_checked"] = True

        async def record_completion(self, **kwargs: Any) -> dict[str, Any]:
            captured["request"] = kwargs
            return {
                "receiptId": str(uuid4()),
                "contactId": str(kwargs["contact_id"]),
                "handoffId": str(uuid4()),
                "trackerCustomerId": kwargs["tracker_customer_id"],
                "trackerSiteId": kwargs["tracker_site_id"],
                "trackerServiceKind": kwargs["tracker_service_kind"],
                "trackerServiceId": kwargs["tracker_service_id"],
                "completedAt": "2026-08-24T19:00:00Z",
                "recordedAt": "2026-08-24T20:00:00Z",
                "idempotent": False,
            }

    contact_id = uuid4()
    operation_key = _operation_key("route-projection")
    app = _app(_CompletionSpy())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            f"/api/v1/eom-funnel/customer-handoffs/{contact_id}/first-clean-completions",
            headers={"Idempotency-Key": operation_key},
            json=_payload(101, 202, tracker_service_id=303),
        )

    assert response.status_code == 201
    assert response.json()["success"] is True
    assert captured["schema_checked"] is True
    assert captured["request"] == {
        "contact_id": contact_id,
        "tracker_customer_id": 101,
        "tracker_site_id": 202,
        "tracker_service_kind": "job",
        "tracker_service_id": 303,
        "completed_at": datetime(2026, 8, 24, 19, 0, tzinfo=timezone.utc),
        "operation_key": operation_key,
        "actor_id": 7,
        "actor_name": "Juan",
    }


@pytest.mark.asyncio
async def test_private_route_reaches_persisted_completion_without_email_or_stripe() -> (
    None
):
    async with _test_store() as (pool, _schema):
        contact_id, customer_id, site_id = await _insert_customer(pool)
        assert customer_id is not None and site_id is not None
        app = _app(_service(pool))
        headers = {"Idempotency-Key": _operation_key("route")}
        path = (
            f"/api/v1/eom-funnel/customer-handoffs/{contact_id}/first-clean-completions"
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            created = await client.post(
                path,
                headers=headers,
                json=_payload(customer_id, site_id),
            )
            replay = await client.post(
                path,
                headers=headers,
                json=_payload(customer_id, site_id),
            )

        assert created.status_code == 201
        assert created.json()["success"] is True
        assert replay.status_code == 200
        assert replay.json()["idempotent"] is True
        assert created.json()["receiptId"] == replay.json()["receiptId"]
        assert (
            await pool.fetchval(
                "SELECT COUNT(*) FROM eom_first_clean_completion_receipts"
            )
            == 1
        )


@pytest.mark.asyncio
async def test_route_rejects_invalid_or_future_payload_before_receipt_write() -> None:
    async with _test_store() as (pool, _schema):
        contact_id, customer_id, site_id = await _insert_customer(pool)
        assert customer_id is not None and site_id is not None
        app = _app(_service(pool))
        path = (
            f"/api/v1/eom-funnel/customer-handoffs/{contact_id}/first-clean-completions"
        )
        invalid_payloads = [
            {**_payload(customer_id, site_id), "completed_at": "2026-08-24T19:00:00"},
            {**_payload(customer_id, site_id), "completed_at": "not-a-date"},
            {**_payload(customer_id, site_id), "unexpected": True},
            _payload(customer_id, site_id, completed_at="2026-08-24T20:00:01Z"),
        ]
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            responses = [
                await client.post(
                    path,
                    headers={"Idempotency-Key": _operation_key("invalid")},
                    json=payload,
                )
                for payload in invalid_payloads
            ]

        assert [response.status_code for response in responses] == [422, 422, 422, 422]
        assert (
            await pool.fetchval(
                "SELECT COUNT(*) FROM eom_first_clean_completion_receipts"
            )
            == 0
        )


@pytest.mark.asyncio
async def test_route_fails_closed_when_the_additive_schema_is_not_ready() -> None:
    class _NoSchemaPool:
        is_initialized = True

        async def fetchval(self, *_args: Any, **_kwargs: Any) -> bool:
            return False

    app = _app(EOMFirstCleanCompletionService(pool=_NoSchemaPool()))
    contact_id = uuid4()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            f"/api/v1/eom-funnel/customer-handoffs/{contact_id}/first-clean-completions",
            headers={"Idempotency-Key": _operation_key("missing-schema")},
            json=_payload(1, 2),
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "First-clean completion schema is unavailable"


@pytest.mark.asyncio
async def test_route_translates_database_command_timeout_to_unavailable() -> None:
    """A stalled advisory lock must preserve the route's safe error contract."""

    class _TimeoutConnection:
        async def execute(self, *_args: Any, **_kwargs: Any) -> None:
            raise TimeoutError("command timed out")

    class _TimeoutPool:
        is_initialized = True

        @asynccontextmanager
        async def transaction(self):
            yield _TimeoutConnection()

        async def fetchval(self, *_args: Any, **_kwargs: Any) -> bool:
            return True

    contact_id = uuid4()
    app = _app(EOMFirstCleanCompletionService(pool=_TimeoutPool(), now=lambda: _NOW))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            f"/api/v1/eom-funnel/customer-handoffs/{contact_id}/first-clean-completions",
            headers={"Idempotency-Key": _operation_key("command-timeout")},
            json=_payload(1, 2),
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "First-clean completion could not be recorded"


@pytest.mark.asyncio
async def test_completion_rechecks_contact_foreign_keys_inside_its_transaction() -> None:
    """A cascaded runtime-owned contact relation cannot race past readiness."""

    async with _test_store() as (pool, _schema):
        contact_id, customer_id, site_id = await _insert_customer(pool)
        assert customer_id is not None and site_id is not None
        assert await first_clean_completion_schema_ready(pool) is True

        await pool._runtime_connection.execute("DROP TABLE contacts CASCADE")
        await pool._runtime_connection.execute(
            """
            CREATE TABLE contacts (
                id UUID PRIMARY KEY,
                business_context_id VARCHAR(64) NOT NULL,
                contact_type VARCHAR(32) NOT NULL,
                customer_type VARCHAR(32),
                status VARCHAR(32) NOT NULL
            )
            """
        )

        assert await first_clean_completion_schema_ready(pool) is False
        with pytest.raises(
            EOMFirstCleanCompletionUnavailableError,
            match="First-clean completion schema is unavailable",
        ):
            await _service(pool).record_completion(
                **_completion_kwargs(
                    contact_id=contact_id,
                    tracker_customer_id=customer_id,
                    tracker_site_id=site_id,
                    operation_key=_operation_key("cascaded-contact-fk"),
                )
            )
        assert (
            await pool.fetchval(
                "SELECT COUNT(*) FROM eom_first_clean_completion_receipts"
            )
            == 0
        )


@pytest.mark.asyncio
async def test_contact_dependency_lock_blocks_cascaded_ddl_during_completion() -> None:
    """The transaction fence prevents a post-attestation contact cascade."""

    async with _test_store() as (pool, _schema):
        async with pool._runtime_connection.transaction():
            await pool._runtime_connection.execute(
                "LOCK TABLE contacts IN ACCESS SHARE MODE"
            )
            async with pool._connection.transaction():
                await pool._connection.execute("SET LOCAL lock_timeout = '100ms'")
                with pytest.raises(asyncpg.LockNotAvailableError):
                    await pool._connection.execute("DROP TABLE contacts CASCADE")


@pytest.mark.asyncio
async def test_schema_is_ready_and_direct_receipt_rewrite_is_refused() -> None:
    async with _test_store() as (pool, _schema):
        assert await first_clean_completion_schema_ready(pool) is True
        contact_id, customer_id, site_id = await _insert_customer(pool)
        assert customer_id is not None and site_id is not None
        result = await _service(pool).record_completion(
            **_completion_kwargs(
                contact_id=contact_id,
                tracker_customer_id=customer_id,
                tracker_site_id=site_id,
                operation_key=_operation_key("immutable"),
            )
        )

        with pytest.raises(asyncpg.RaiseError, match="append-only"):
            await pool._connection.execute(
                """
                UPDATE eom_first_clean_completion_receipts
                SET actor_name = 'Different Actor'
                WHERE id = $1::uuid
                """,
                result["receiptId"],
            )
        second_contact_id, second_customer_id, second_site_id = await _insert_customer(
            pool
        )
        assert second_customer_id is not None and second_site_id is not None
        second_handoff_id = await pool.fetchval(
            "SELECT id FROM eom_customer_handoffs WHERE contact_id = $1",
            second_contact_id,
        )
        second_receipt_id = uuid4()
        operation_key = _operation_key("lifecycle-time")
        fingerprint = "a" * 64
        lifecycle_time = _NOW - timedelta(hours=2)
        receipt_time = _NOW - timedelta(hours=1)
        await pool._connection.execute(
            """
            INSERT INTO eom_first_clean_completion_operation_receipts (
                operation_key, contact_id, request_fingerprint
            ) VALUES ($1, $2, $3)
            """,
            operation_key,
            second_contact_id,
            fingerprint,
        )
        await pool._connection.execute(
            """
            INSERT INTO eom_lead_lifecycle_events (
                contact_id, event_type, actor, source, operation_key,
                metadata, occurred_at
            ) VALUES (
                $1, 'first_clean_completed', 'employee:7:Juan', 'time_tracker',
                $2,
                jsonb_build_object(
                    'completion_receipt_id', $3::text,
                    'handoff_id', $4::text,
                    'tracker_customer_id', $5::bigint,
                    'tracker_site_id', $6::bigint,
                    'tracker_service_kind', 'planned_visit',
                    'tracker_service_id', 8001::bigint
                ),
                $7
            )
            """,
            second_contact_id,
            operation_key,
            str(second_receipt_id),
            str(second_handoff_id),
            second_customer_id,
            second_site_id,
            lifecycle_time,
        )
        with pytest.raises(asyncpg.RaiseError, match="matching lifecycle evidence"):
            await pool._connection.execute(
                """
                INSERT INTO eom_first_clean_completion_receipts (
                    id, contact_id, handoff_id, tracker_customer_id,
                    tracker_site_id, tracker_service_kind, tracker_service_id,
                    completed_at, operation_key, request_fingerprint,
                    actor_id, actor_name
                ) VALUES (
                    $1, $2, $3, $4, $5, 'planned_visit', 8001, $6, $7, $8,
                    7, 'Juan'
                )
                """,
                second_receipt_id,
                second_contact_id,
                second_handoff_id,
                second_customer_id,
                second_site_id,
                receipt_time,
                operation_key,
                fingerprint,
            )

        future_receipt_id = uuid4()
        future_operation_key = _operation_key("future-trigger")
        future_time = await pool.fetchval(
            "SELECT CURRENT_TIMESTAMP + INTERVAL '1 hour'"
        )
        await pool._connection.execute(
            """
            INSERT INTO eom_first_clean_completion_operation_receipts (
                operation_key, contact_id, request_fingerprint
            ) VALUES ($1, $2, $3)
            """,
            future_operation_key,
            second_contact_id,
            fingerprint,
        )
        await pool._connection.execute(
            """
            INSERT INTO eom_lead_lifecycle_events (
                contact_id, event_type, actor, source, operation_key,
                metadata, occurred_at
            ) VALUES (
                $1, 'first_clean_completed', 'employee:7:Juan', 'time_tracker',
                $2,
                jsonb_build_object(
                    'completion_receipt_id', $3::text,
                    'handoff_id', $4::text,
                    'tracker_customer_id', $5::bigint,
                    'tracker_site_id', $6::bigint,
                    'tracker_service_kind', 'planned_visit',
                    'tracker_service_id', 8002::bigint
                ),
                $7
            )
            """,
            second_contact_id,
            future_operation_key,
            str(future_receipt_id),
            str(second_handoff_id),
            second_customer_id,
            second_site_id,
            future_time,
        )
        with pytest.raises(
            asyncpg.RaiseError, match="cannot be recorded in the future"
        ):
            await pool._connection.execute(
                """
                INSERT INTO eom_first_clean_completion_receipts (
                    id, contact_id, handoff_id, tracker_customer_id,
                    tracker_site_id, tracker_service_kind, tracker_service_id,
                    completed_at, operation_key, request_fingerprint,
                    actor_id, actor_name
                ) VALUES (
                    $1, $2, $3, $4, $5, 'planned_visit', 8002, $6, $7, $8,
                    7, 'Juan'
                )
                """,
                future_receipt_id,
                second_contact_id,
                second_handoff_id,
                second_customer_id,
                second_site_id,
                future_time,
                future_operation_key,
                fingerprint,
            )


@pytest.mark.asyncio
async def test_schema_readiness_fails_closed_when_an_append_only_trigger_is_missing() -> (
    None
):
    async with _test_store() as (pool, _schema):
        assert await first_clean_completion_schema_ready(pool) is True

        await pool._connection.execute(
            """
            DROP TRIGGER trg_prevent_eom_first_clean_completion_operation_truncate
            ON eom_first_clean_completion_operation_receipts
            """
        )

        assert await first_clean_completion_schema_ready(pool) is False


@pytest.mark.asyncio
async def test_schema_readiness_fails_closed_when_an_append_only_trigger_is_disabled() -> (
    None
):
    async with _test_store() as (pool, _schema):
        assert await first_clean_completion_schema_ready(pool) is True

        await pool._connection.execute(
            """
            ALTER TABLE eom_first_clean_completion_receipts
            DISABLE TRIGGER trg_prevent_eom_first_clean_completion_receipt_mutation
            """
        )

        assert await first_clean_completion_schema_ready(pool) is False


def test_funnel_advertises_completion_only_with_its_registered_route() -> None:
    capability = "customer.first_clean_completion.record"
    route = (
        "POST",
        "/eom-funnel/customer-handoffs/{contact_id}/first-clean-completions",
    )
    registered = {
        (method, item.path)
        for item in funnel_mod.router.routes
        for method in (item.methods or ())
    }
    assert funnel_mod._CAPABILITY_ROUTES[capability] == route
    assert route in registered
    assert capability in funnel_mod.served_capabilities()
