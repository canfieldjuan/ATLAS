"""Focused proof for EOM first-clean completion receipts.

All fixtures are synthetic.  The tests use an isolated PostgreSQL schema and
never contact a calendar, email provider, Stripe, or real EOM customer.
"""

from __future__ import annotations

import asyncio
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
    EOMFirstCleanCompletionValidationError,
    first_clean_completion_schema_ready,
)


ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS = ROOT / "atlas_brain" / "storage" / "migrations"
DATABASE_URL_ENV = "ATLAS_MIGRATION_TEST_DATABASE_URL"
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
_ACTOR_ID_LENGTH_BOUNDARIES = (
    1,
    10,
    100,
    10_000,
    1_000_000,
    2_147_483_647,
    9_223_372_036_854_775_807,
)


class _ConnectionPool:
    """Expose the small async pool protocol the service needs in tests."""

    is_initialized = True

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    @asynccontextmanager
    async def transaction(self):
        async with self._connection.transaction():
            yield self._connection

    async def fetchval(self, *args: Any, **kwargs: Any) -> Any:
        return await self._connection.fetchval(*args, **kwargs)


def _database_url_or_skip() -> str:
    database_url = os.environ.get(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")
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


async def _apply_schema(
    connection: Any,
    schema: str,
    *,
    migrations: tuple[str, ...] = _COMPLETION_SCHEMA_MIGRATIONS,
) -> None:
    await connection.execute(f'CREATE SCHEMA "{schema}"')
    await connection.execute(f'SET search_path TO "{schema}", public')
    # Migration 035 remains additive to the production appointments relation;
    # the focused empty schema supplies that dependency explicitly.
    await connection.execute("CREATE TABLE appointments (id UUID PRIMARY KEY)")
    # The completion schema is DBA-only because it references and protects the
    # guard-owned handoff table. Keep this disposable proof limited to its
    # actual canonical prerequisites rather than a developer's broader schema.
    await _provision_nocodb_login(connection)
    for migration in migrations:
        await connection.execute((MIGRATIONS / f"{migration}.sql").read_text())


@asynccontextmanager
async def _test_store():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_first_clean_{uuid4().hex}"
    connection = await asyncpg.connect(database_url)
    try:
        await _apply_schema(connection, schema)
        yield _ConnectionPool(connection), schema
    finally:
        try:
            await connection.execute("RESET search_path")
            await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await connection.close()


async def _schema_connection(database_url: str, schema: str) -> _ConnectionPool:
    connection = await asyncpg.connect(database_url)
    await connection.execute(f'SET search_path TO "{schema}", public')
    return _ConnectionPool(connection)


async def _close_schema_connection(pool: _ConnectionPool) -> None:
    await pool._connection.close()


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
    async with _test_store() as (pool, _schema):
        assert await first_clean_completion_schema_ready(pool) is True
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
                "eom_first_clean_completion_operation_receipts",
                "eom_first_clean_completion_receipts",
            ],
        )
        assert {row["relname"]: row["owner"] for row in ownership} == {
            "eom_customer_handoffs": "atlas_eom_handoff_owner",
            "eom_first_clean_completion_operation_receipts": (
                "atlas_eom_handoff_owner"
            ),
            "eom_first_clean_completion_receipts": "atlas_eom_handoff_owner",
        }
        handoff_function_ownership = await pool._connection.fetch(
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
            list(_PREREQUISITE_HANDOFF_GUARD_FUNCTIONS),
        )
        assert {row["proname"]: row["owner"] for row in handoff_function_ownership} == {
            function_name: "atlas_eom_handoff_owner"
            for function_name in _PREREQUISITE_HANDOFF_GUARD_FUNCTIONS
        }
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
            "eom_first_clean_completion_receipts": {"INSERT", "SELECT", "UPDATE"},
        }
        unexpected_acl_rows = await pool._connection.fetch(
            """
            SELECT relation.relname, acl.grantee, acl.privilege_type
            FROM pg_class AS relation
            JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            CROSS JOIN LATERAL aclexplode(
                COALESCE(relation.relacl, ARRAY[]::aclitem[])
            ) AS acl
            WHERE namespace.nspname = current_schema()
              AND relation.relname = ANY($1::text[])
              AND acl.grantee <> (SELECT oid FROM pg_roles WHERE rolname = 'atlas')
            """,
            [
                "eom_first_clean_completion_operation_receipts",
                "eom_first_clean_completion_receipts",
            ],
        )
        assert unexpected_acl_rows == []
        assert not await _has_guard_membership(pool, "atlas")


@pytest.mark.asyncio
async def test_completion_migration_requires_guard_owned_handoff_prerequisites() -> (
    None
):
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_first_clean_unguarded_{uuid4().hex}"
    connection = await asyncpg.connect(database_url)
    try:
        await _apply_schema(
            connection,
            schema,
            migrations=tuple(
                migration
                for migration in _COMPLETION_SCHEMA_MIGRATIONS
                if migration
                not in {
                    "354_eom_customer_handoff_privileges",
                    "394_eom_first_clean_completion_receipts",
                }
            ),
        )
        with pytest.raises(
            asyncpg.RaiseError,
            match=(
                "eom_customer_handoffs and its protected functions must be guard-owned"
            ),
        ):
            await connection.execute(
                (MIGRATIONS / "394_eom_first_clean_completion_receipts.sql").read_text()
            )
    finally:
        try:
            await connection.execute("RESET search_path")
            await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await connection.close()


@pytest.mark.asyncio
async def test_schema_readiness_rejects_indirect_runtime_guard_membership() -> None:
    async with _test_store() as (pool, _schema):
        assert await first_clean_completion_schema_ready(pool) is True
        role_name = f"eom_completion_membership_{uuid4().hex[:16]}"
        role_ident = _quote_ident(role_name)
        role_created = False
        try:
            can_administer_roles = await pool.fetchval(
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
                f"CREATE ROLE {role_ident} NOLOGIN NOINHERIT"
            )
            role_created = True
            await pool._connection.execute(
                f"GRANT atlas_eom_handoff_owner TO {role_ident}"
            )
            await pool._connection.execute(f"GRANT {role_ident} TO atlas")
            assert await first_clean_completion_schema_ready(pool) is False
        finally:
            if role_created:
                await pool._connection.execute(f"REVOKE {role_ident} FROM atlas")
                await pool._connection.execute(
                    f"REVOKE atlas_eom_handoff_owner FROM {role_ident}"
                )
                await pool._connection.execute(f"DROP ROLE {role_ident}")
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
            await pool._connection.execute(
                """
                GRANT SELECT, INSERT, UPDATE
                ON TABLE eom_first_clean_completion_receipts TO atlas
                """
            )
        assert await first_clean_completion_schema_ready(pool) is True


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
            can_administer_roles = await pool.fetchval(
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
    database_url = _database_url_or_skip()
    role_name = f"eom_completion_runtime_{uuid4().hex[:16]}"
    role_ident = _quote_ident(role_name)
    role_created = False
    connection = await asyncpg.connect(database_url)
    try:
        can_administer_roles = await connection.fetchval(
            """
            SELECT rolsuper OR rolcreaterole
            FROM pg_roles
            WHERE rolname = current_user
            """
        )
        if not can_administer_roles:
            pytest.skip("DBA migration proof requires disposable role administration")
        await connection.execute(f"CREATE ROLE {role_ident} NOLOGIN NOINHERIT")
        role_created = True
        await connection.execute(f"GRANT {role_ident} TO atlas")
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
            await connection.execute(f"REVOKE {role_ident} FROM atlas")
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
        assert lifecycle["metadata"]["completion_receipt_id"] == result["receiptId"]
        assert lifecycle["metadata"]["handoff_id"] == result["handoffId"]
        assert lifecycle["metadata"]["tracker_service_id"] == 6001


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
        ("residential", "inactive", True),
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
            second_receipt_id,
            second_handoff_id,
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
            future_receipt_id,
            second_handoff_id,
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
