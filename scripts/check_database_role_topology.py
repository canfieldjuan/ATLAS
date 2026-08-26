#!/usr/bin/env python3
"""Produce a fixed, read-only PostgreSQL role-topology evidence receipt.

This command is a preflight for a later least-privilege role cutover. It never
creates roles, changes grants or ownership, runs migrations, or exposes a
generic SQL interface. It compares the normal Atlas runtime target with a
separately configured DBA target before reading a fixed catalog projection.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import secrets
import sys
from typing import Any
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_brain.config import (  # noqa: E402
    DATABASE_ROLE_TOPOLOGY_DBA_DATABASE_URL_ENV,
    DatabaseRoleTopologyDBAConfig,
)
from atlas_brain.storage.config import DatabaseConfig  # noqa: E402


DBA_DSN_ENV = DATABASE_ROLE_TOPOLOGY_DBA_DATABASE_URL_ENV
RECEIPT_VERSION = "database-role-topology-preflight.v1"
_READ_ONLY_SERVER_SETTINGS = {"default_transaction_read_only": "on"}


class PreflightError(RuntimeError):
    """A safe, operator-actionable admission failure before a receipt exists."""


@dataclass(frozen=True)
class _TargetIdentity:
    """One session's target and authenticated PostgreSQL identity."""

    schema_name: str
    database_name: str
    database_oid: int
    current_user: str
    session_user: str
    current_user_is_superuser: bool

    @property
    def database_identity(self) -> tuple[str, int]:
        """Return the database identity that must match before catalog access."""

        return (self.database_name, self.database_oid)

    @property
    def uses_direct_superuser_login(self) -> bool:
        """Return whether the active session is a direct superuser login."""

        return (
            self.current_user == self.session_user
            and self.current_user_is_superuser
        )


_TARGET_IDENTITY_QUERY = """
    SELECT pg_catalog.current_schema() AS schema_name,
           pg_catalog.current_database() AS database_name,
           CURRENT_USER AS current_user,
           SESSION_USER AS session_user,
           COALESCE((
               SELECT role.rolsuper
                 FROM pg_catalog.pg_roles AS role
                WHERE role.rolname = CURRENT_USER
           ), FALSE) AS current_user_is_superuser,
           (
               SELECT database_catalog.oid
                 FROM pg_catalog.pg_database AS database_catalog
                WHERE database_catalog.datname = pg_catalog.current_database()
           ) AS database_oid
"""

_DATABASE_OWNER_QUERY = """
    SELECT database_catalog.datname AS database_name,
           pg_catalog.pg_get_userbyid(database_catalog.datdba) AS owner_role
      FROM pg_catalog.pg_database AS database_catalog
     WHERE database_catalog.oid = (
         SELECT oid
           FROM pg_catalog.pg_database
          WHERE datname = pg_catalog.current_database()
     )
"""

_ROLES_QUERY = """
    SELECT role.rolname AS role_name,
           role.rolcanlogin AS can_login,
           role.rolsuper AS is_superuser,
           role.rolinherit AS inherits_privileges,
           role.rolcreaterole AS can_create_roles,
           role.rolcreatedb AS can_create_databases,
           role.rolreplication AS can_replicate,
           role.rolbypassrls AS bypasses_row_security
      FROM pg_catalog.pg_roles AS role
     WHERE role.rolname !~ '^pg_'
        OR EXISTS (
            SELECT 1
              FROM pg_catalog.pg_auth_members AS membership
              JOIN pg_catalog.pg_roles AS member_role
                ON member_role.oid = membership.member
             WHERE membership.roleid = role.oid
               AND member_role.rolname !~ '^pg_'
        )
     ORDER BY role.rolname
"""

_ROLE_MEMBERSHIPS_QUERY = """
    SELECT granted_role.rolname AS granted_role,
           member_role.rolname AS member_role,
           membership.admin_option AS admin_option
      FROM pg_catalog.pg_auth_members AS membership
      JOIN pg_catalog.pg_roles AS granted_role
        ON granted_role.oid = membership.roleid
      JOIN pg_catalog.pg_roles AS member_role
        ON member_role.oid = membership.member
     WHERE granted_role.rolname !~ '^pg_'
        OR member_role.rolname !~ '^pg_'
     ORDER BY granted_role.rolname, member_role.rolname
"""

_SCHEMA_OWNERS_QUERY = """
    SELECT namespace.nspname AS schema_name,
           owner_role.rolname AS owner_role
      FROM pg_catalog.pg_namespace AS namespace
      JOIN pg_catalog.pg_roles AS owner_role
        ON owner_role.oid = namespace.nspowner
     WHERE namespace.nspname !~ '^pg_'
       AND namespace.nspname <> 'information_schema'
     ORDER BY namespace.nspname
"""

_RELATION_OWNER_SUMMARY_QUERY = """
    SELECT namespace.nspname AS schema_name,
           relation.relkind::text AS relation_kind,
           owner_role.rolname AS owner_role,
           relation.relrowsecurity AS row_security_enabled,
           relation.relforcerowsecurity AS row_security_forced,
           count(*)::bigint AS object_count
      FROM pg_catalog.pg_class AS relation
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = relation.relnamespace
      JOIN pg_catalog.pg_roles AS owner_role
        ON owner_role.oid = relation.relowner
     WHERE namespace.nspname !~ '^pg_'
       AND namespace.nspname <> 'information_schema'
       AND relation.relkind IN ('r', 'p', 'v', 'm', 'S', 'f')
     GROUP BY namespace.nspname, relation.relkind, owner_role.rolname,
              relation.relrowsecurity, relation.relforcerowsecurity
     ORDER BY namespace.nspname, relation.relkind, owner_role.rolname,
              relation.relrowsecurity, relation.relforcerowsecurity
"""

_FUNCTION_OWNER_SUMMARY_QUERY = """
    SELECT namespace.nspname AS schema_name,
           owner_role.rolname AS owner_role,
           procedure.prosecdef AS is_security_definer,
           count(*)::bigint AS function_count
      FROM pg_catalog.pg_proc AS procedure
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = procedure.pronamespace
      JOIN pg_catalog.pg_roles AS owner_role
        ON owner_role.oid = procedure.proowner
     WHERE namespace.nspname !~ '^pg_'
       AND namespace.nspname <> 'information_schema'
     GROUP BY namespace.nspname, owner_role.rolname, procedure.prosecdef
     ORDER BY namespace.nspname, owner_role.rolname, procedure.prosecdef
"""

_DATABASE_ACL_QUERY = """
    SELECT CASE
               WHEN database_catalog.datacl IS NULL THEN 'default'
               ELSE 'explicit'
           END AS acl_source,
           COALESCE(grantee_role.rolname, 'PUBLIC') AS grantee_role,
           acl.privilege_type AS privilege_type,
           acl.is_grantable AS is_grantable
      FROM pg_catalog.pg_database AS database_catalog
      CROSS JOIN LATERAL pg_catalog.aclexplode(
          COALESCE(
              database_catalog.datacl,
              pg_catalog.acldefault('d', database_catalog.datdba)
          )
      ) AS acl
      LEFT JOIN pg_catalog.pg_roles AS grantee_role
        ON grantee_role.oid = acl.grantee
     WHERE database_catalog.oid = (
         SELECT oid
           FROM pg_catalog.pg_database
          WHERE datname = pg_catalog.current_database()
     )
     ORDER BY acl_source, grantee_role, acl.privilege_type, acl.is_grantable
"""

_SCHEMA_ACL_QUERY = """
    SELECT CASE
               WHEN namespace.nspacl IS NULL THEN 'default'
               ELSE 'explicit'
           END AS acl_source,
           namespace.nspname AS schema_name,
           COALESCE(grantee_role.rolname, 'PUBLIC') AS grantee_role,
           acl.privilege_type AS privilege_type,
           acl.is_grantable AS is_grantable
      FROM pg_catalog.pg_namespace AS namespace
      CROSS JOIN LATERAL pg_catalog.aclexplode(
          COALESCE(
              namespace.nspacl,
              pg_catalog.acldefault('n', namespace.nspowner)
          )
      ) AS acl
      LEFT JOIN pg_catalog.pg_roles AS grantee_role
        ON grantee_role.oid = acl.grantee
     WHERE namespace.nspname !~ '^pg_'
       AND namespace.nspname <> 'information_schema'
     ORDER BY acl_source, namespace.nspname, grantee_role, acl.privilege_type,
              acl.is_grantable
"""

_RELATION_ACL_SUMMARY_QUERY = """
    SELECT CASE
               WHEN relation.relacl IS NULL THEN 'default'
               ELSE 'explicit'
           END AS acl_source,
           namespace.nspname AS schema_name,
           relation.relkind::text AS relation_kind,
           relation.relrowsecurity AS row_security_enabled,
           relation.relforcerowsecurity AS row_security_forced,
           COALESCE(grantee_role.rolname, 'PUBLIC') AS grantee_role,
           acl.privilege_type AS privilege_type,
           acl.is_grantable AS is_grantable,
           count(*)::bigint AS object_count
      FROM pg_catalog.pg_class AS relation
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = relation.relnamespace
      CROSS JOIN LATERAL pg_catalog.aclexplode(
          COALESCE(
              relation.relacl,
              pg_catalog.acldefault(
                  CASE
                      WHEN relation.relkind = 'S' THEN 'S'::"char"
                      ELSE 'r'::"char"
                  END,
                  relation.relowner
              )
          )
      ) AS acl
      LEFT JOIN pg_catalog.pg_roles AS grantee_role
        ON grantee_role.oid = acl.grantee
     WHERE namespace.nspname !~ '^pg_'
       AND namespace.nspname <> 'information_schema'
       AND relation.relkind IN ('r', 'p', 'v', 'm', 'S', 'f')
     GROUP BY relation.relacl IS NULL, namespace.nspname, relation.relkind,
              relation.relrowsecurity, relation.relforcerowsecurity,
              grantee_role, acl.privilege_type, acl.is_grantable
     ORDER BY acl_source, namespace.nspname, relation.relkind,
              relation.relrowsecurity, relation.relforcerowsecurity,
              grantee_role, acl.privilege_type, acl.is_grantable
"""

_FUNCTION_ACL_SUMMARY_QUERY = """
    SELECT CASE
               WHEN procedure.proacl IS NULL THEN 'default'
               ELSE 'explicit'
           END AS acl_source,
           namespace.nspname AS schema_name,
           procedure.prosecdef AS is_security_definer,
           COALESCE(grantee_role.rolname, 'PUBLIC') AS grantee_role,
           acl.privilege_type AS privilege_type,
           acl.is_grantable AS is_grantable,
           count(*)::bigint AS function_count
      FROM pg_catalog.pg_proc AS procedure
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = procedure.pronamespace
      CROSS JOIN LATERAL pg_catalog.aclexplode(
          COALESCE(
              procedure.proacl,
              pg_catalog.acldefault('f', procedure.proowner)
          )
      ) AS acl
      LEFT JOIN pg_catalog.pg_roles AS grantee_role
        ON grantee_role.oid = acl.grantee
     WHERE namespace.nspname !~ '^pg_'
       AND namespace.nspname <> 'information_schema'
     GROUP BY procedure.proacl IS NULL, namespace.nspname,
              procedure.prosecdef, grantee_role, acl.privilege_type,
              acl.is_grantable
     ORDER BY acl_source, namespace.nspname, procedure.prosecdef, grantee_role,
              acl.privilege_type, acl.is_grantable
"""

_COLUMN_ACL_SUMMARY_QUERY = """
    SELECT 'explicit_column' AS acl_source,
           namespace.nspname AS schema_name,
           relation.relkind::text AS relation_kind,
           COALESCE(grantee_role.rolname, 'PUBLIC') AS grantee_role,
           acl.privilege_type AS privilege_type,
           acl.is_grantable AS is_grantable,
           count(*)::bigint AS column_count
      FROM pg_catalog.pg_attribute AS column_attribute
      JOIN pg_catalog.pg_class AS relation
        ON relation.oid = column_attribute.attrelid
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = relation.relnamespace
      CROSS JOIN LATERAL pg_catalog.aclexplode(column_attribute.attacl) AS acl
      LEFT JOIN pg_catalog.pg_roles AS grantee_role
        ON grantee_role.oid = acl.grantee
     WHERE namespace.nspname !~ '^pg_'
       AND namespace.nspname <> 'information_schema'
       AND relation.relkind IN ('r', 'p', 'v', 'm', 'f')
       AND column_attribute.attnum > 0
       AND NOT column_attribute.attisdropped
     GROUP BY namespace.nspname, relation.relkind, grantee_role,
              acl.privilege_type, acl.is_grantable
     ORDER BY namespace.nspname, relation.relkind, grantee_role,
              acl.privilege_type, acl.is_grantable
"""

_ROW_SECURITY_POLICY_SUMMARY_QUERY = """
    SELECT namespace.nspname AS schema_name,
           relation.relkind::text AS relation_kind,
           policy.polcmd::text AS command,
           policy.polpermissive AS is_permissive,
           COALESCE(role.rolname, 'PUBLIC') AS role_name,
           count(*)::bigint AS policy_count
      FROM pg_catalog.pg_policy AS policy
      JOIN pg_catalog.pg_class AS relation
        ON relation.oid = policy.polrelid
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = relation.relnamespace
      CROSS JOIN LATERAL unnest(
          COALESCE(
              NULLIF(policy.polroles, ARRAY[]::oid[]),
              ARRAY[0::oid]
          )
      ) AS policy_role(role_oid)
      LEFT JOIN pg_catalog.pg_roles AS role
        ON role.oid = policy_role.role_oid
     WHERE namespace.nspname !~ '^pg_'
       AND namespace.nspname <> 'information_schema'
     GROUP BY namespace.nspname, relation.relkind, policy.polcmd,
              policy.polpermissive, role.rolname
     ORDER BY namespace.nspname, relation.relkind, policy.polcmd,
              policy.polpermissive, role_name
"""

_DEFAULT_ACL_SUMMARY_QUERY = """
    SELECT COALESCE(namespace.nspname, '<database>') AS schema_name,
           owner_role.rolname AS owner_role,
           default_acl.defaclobjtype::text AS object_type,
           COALESCE(grantee_role.rolname, 'PUBLIC') AS grantee_role,
           acl.privilege_type AS privilege_type,
           acl.is_grantable AS is_grantable
      FROM pg_catalog.pg_default_acl AS default_acl
      JOIN pg_catalog.pg_roles AS owner_role
        ON owner_role.oid = default_acl.defaclrole
      LEFT JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = default_acl.defaclnamespace
      CROSS JOIN LATERAL pg_catalog.aclexplode(default_acl.defaclacl) AS acl
      LEFT JOIN pg_catalog.pg_roles AS grantee_role
        ON grantee_role.oid = acl.grantee
     WHERE namespace.nspname IS NULL
        OR (
            namespace.nspname !~ '^pg_'
            AND namespace.nspname <> 'information_schema'
        )
     ORDER BY schema_name, owner_role, object_type, grantee_role,
              acl.privilege_type, acl.is_grantable
"""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the intentionally argument-free, evidence-only command surface."""

    parser = argparse.ArgumentParser(
        description=(
            "Print a fixed, read-only PostgreSQL database role-topology receipt."
        )
    )
    return parser.parse_args(argv)


def _require_text(value: object, *, source: str) -> str:
    """Require a non-empty catalog text field without interpolating it into SQL."""

    if not isinstance(value, str) or not value.strip():
        raise PreflightError(f"Missing or invalid {source}")
    return value


def _require_database_oid(value: object, *, source: str) -> int:
    """Require one positive PostgreSQL database OID."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PreflightError(f"Missing or invalid {source}")
    return value


def _require_bool(value: object, *, source: str) -> bool:
    """Require a catalog boolean instead of coercing an ambiguous value."""

    if not isinstance(value, bool):
        raise PreflightError(f"Missing or invalid {source}")
    return value


def _safe_target_label(database_url: str) -> str:
    """Render a target label without credentials or connection query values."""

    try:
        parsed = urlsplit(database_url)
        host = parsed.hostname or "connection-string"
        port = f":{parsed.port}" if parsed.port else ""
        database = parsed.path.lstrip("/") or "<database>"
        return f"dsn={host}{port}/{database}"
    except ValueError:
        return "dsn=<configured>"


def _runtime_target_label(config: object) -> str:
    """Use DatabaseConfig's safe label, never reconstructing a raw DSN."""

    target_label = getattr(config, "target_label", "")
    if isinstance(target_label, str) and target_label.strip():
        return target_label
    return "configured-runtime-target"


def _runtime_connection_kwargs(config: object) -> dict[str, object]:
    """Get normal runtime kwargs without changing normal runtime configuration."""

    connection_kwargs = getattr(config, "connection_kwargs", None)
    if not callable(connection_kwargs):
        raise PreflightError("Runtime database configuration is unavailable")
    values = connection_kwargs()
    if not isinstance(values, Mapping):
        raise PreflightError("Runtime database configuration is invalid")
    return dict(values)


async def _create_pool(connection_kwargs: Mapping[str, object]) -> Any:
    """Open one bounded PostgreSQL pool with read-only defaults enforced."""

    try:
        import asyncpg
    except ImportError as exc:  # pragma: no cover - host dependency
        raise PreflightError(
            "asyncpg is required to run the database role-topology preflight"
        ) from exc

    pool_kwargs = dict(connection_kwargs)
    configured_settings = pool_kwargs.pop("server_settings", {})
    if not isinstance(configured_settings, Mapping):
        raise PreflightError("Runtime database server settings are invalid")
    pool_kwargs.update(
        {
            "min_size": 1,
            "max_size": 1,
            "statement_cache_size": 0,
            "server_settings": {
                **dict(configured_settings),
                **_READ_ONLY_SERVER_SETTINGS,
            },
        }
    )
    return await asyncpg.create_pool(**pool_kwargs)


async def _target_identity(pool: Any, *, source: str) -> _TargetIdentity:
    """Read one session's database/schema/role identity in a read-only tx."""

    async with pool.acquire() as connection:
        async with connection.transaction(readonly=True):
            row = await connection.fetchrow(_TARGET_IDENTITY_QUERY)
    if row is None:
        raise PreflightError(f"Missing or invalid database identity from {source}")
    return _TargetIdentity(
        schema_name=_require_text(
            row["schema_name"], source=f"current schema from {source}"
        ),
        database_name=_require_text(
            row["database_name"], source=f"database name from {source}"
        ),
        database_oid=_require_database_oid(
            row["database_oid"], source=f"database OID from {source}"
        ),
        current_user=_require_text(
            row["current_user"], source=f"current user from {source}"
        ),
        session_user=_require_text(
            row["session_user"], source=f"session user from {source}"
        ),
        current_user_is_superuser=_require_bool(
            row["current_user_is_superuser"],
            source=f"superuser status from {source}",
        ),
    )


def _require_matching_target(
    runtime_target: _TargetIdentity,
    dba_target: _TargetIdentity,
) -> None:
    """Reject a DBA pool that is not the exact configured runtime target."""

    if dba_target.database_identity != runtime_target.database_identity:
        raise PreflightError(
            "Configured DBA connection does not target the Atlas runtime database"
        )
    if dba_target.schema_name != runtime_target.schema_name:
        raise PreflightError(
            "Configured DBA connection does not resolve to the Atlas runtime schema"
        )


def _require_direct_dba_superuser(target: _TargetIdentity) -> None:
    """Require a direct superuser session, not a weaker or switched role."""

    if target.uses_direct_superuser_login:
        return
    if target.current_user != target.session_user:
        raise PreflightError(
            "Configured DBA connection must use a direct DBA login, not SET ROLE"
        )
    if not target.current_user_is_superuser:
        raise PreflightError(
            "Configured DBA connection is not a PostgreSQL superuser"
        )


async def _attest_shared_database_lock(runtime_pool: Any, dba_pool: Any) -> None:
    """Prove both pools contend on one live database-cluster lock namespace."""

    for _attempt in range(3):
        lock_key = secrets.randbits(63)
        async with runtime_pool.acquire() as runtime_connection:
            async with runtime_connection.transaction(readonly=True):
                runtime_acquired = bool(
                    await runtime_connection.fetchval(
                        "SELECT pg_catalog.pg_try_advisory_xact_lock($1)",
                        lock_key,
                    )
                )
                if not runtime_acquired:
                    continue
                async with dba_pool.acquire() as dba_connection:
                    async with dba_connection.transaction(readonly=True):
                        dba_acquired = bool(
                            await dba_connection.fetchval(
                                "SELECT pg_catalog.pg_try_advisory_xact_lock($1)",
                                lock_key,
                            )
                        )
                if dba_acquired:
                    raise PreflightError(
                        "Configured DBA connection does not share the Atlas "
                        "runtime database cluster"
                    )
                return
    raise PreflightError(
        "Could not reserve a fresh runtime target-attestation advisory lock"
    )


def _json_catalog_value(value: object, *, source: str) -> object:
    """Keep the receipt strictly JSON scalar data from fixed catalog queries."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise PreflightError(f"Catalog receipt contained an invalid value from {source}")


def _json_catalog_row(row: object, *, source: str) -> dict[str, object]:
    """Convert one static catalog result to a safe JSON object."""

    try:
        values = dict(row)
    except (TypeError, ValueError) as exc:
        raise PreflightError(
            f"Catalog receipt contained an invalid row from {source}"
        ) from exc
    if any(not isinstance(key, str) for key in values):
        raise PreflightError(
            f"Catalog receipt contained an invalid row from {source}"
        )
    return {
        key: _json_catalog_value(value, source=source)
        for key, value in values.items()
    }


async def _catalog_receipt(pool: Any) -> dict[str, object]:
    """Return the fixed catalog facts required for later role design only."""

    async with pool.acquire() as connection:
        async with connection.transaction(readonly=True):
            database_owner = await connection.fetchrow(_DATABASE_OWNER_QUERY)
            roles = await connection.fetch(_ROLES_QUERY)
            memberships = await connection.fetch(_ROLE_MEMBERSHIPS_QUERY)
            schemas = await connection.fetch(_SCHEMA_OWNERS_QUERY)
            relation_owners = await connection.fetch(_RELATION_OWNER_SUMMARY_QUERY)
            function_owners = await connection.fetch(_FUNCTION_OWNER_SUMMARY_QUERY)
            database_acl = await connection.fetch(_DATABASE_ACL_QUERY)
            schema_acl = await connection.fetch(_SCHEMA_ACL_QUERY)
            relation_acl = await connection.fetch(_RELATION_ACL_SUMMARY_QUERY)
            function_acl = await connection.fetch(_FUNCTION_ACL_SUMMARY_QUERY)
            column_acl = await connection.fetch(_COLUMN_ACL_SUMMARY_QUERY)
            row_security_policies = await connection.fetch(
                _ROW_SECURITY_POLICY_SUMMARY_QUERY
            )
            default_acl = await connection.fetch(_DEFAULT_ACL_SUMMARY_QUERY)

    if database_owner is None:
        raise PreflightError("Catalog receipt is missing the current database owner")
    return {
        "database_owner": _json_catalog_row(
            database_owner,
            source="database owner",
        ),
        "roles": [
            _json_catalog_row(row, source="roles")
            for row in roles
        ],
        "memberships": [
            _json_catalog_row(row, source="role memberships")
            for row in memberships
        ],
        "schema_owners": [
            _json_catalog_row(row, source="schema owners")
            for row in schemas
        ],
        "relation_owner_summary": [
            _json_catalog_row(row, source="relation owners")
            for row in relation_owners
        ],
        "function_owner_summary": [
            _json_catalog_row(row, source="function owners")
            for row in function_owners
        ],
        "database_acl": [
            _json_catalog_row(row, source="database ACL")
            for row in database_acl
        ],
        "schema_acl": [
            _json_catalog_row(row, source="schema ACL")
            for row in schema_acl
        ],
        "relation_acl_summary": [
            _json_catalog_row(row, source="relation ACL")
            for row in relation_acl
        ],
        "function_acl_summary": [
            _json_catalog_row(row, source="function ACL")
            for row in function_acl
        ],
        "column_acl_summary": [
            _json_catalog_row(row, source="column ACL")
            for row in column_acl
        ],
        "row_security_policy_summary": [
            _json_catalog_row(row, source="row security policy")
            for row in row_security_policies
        ],
        "default_acl_summary": [
            _json_catalog_row(row, source="default ACL")
            for row in default_acl
        ],
    }


def _identity_receipt(target: _TargetIdentity) -> dict[str, object]:
    """Render only non-secret session facts from a target identity."""

    return {
        "schema_name": target.schema_name,
        "database_name": target.database_name,
        "database_oid": target.database_oid,
        "current_user": target.current_user,
        "session_user": target.session_user,
        "current_user_is_superuser": target.current_user_is_superuser,
    }


async def _run(
    *,
    create_pool: Callable[[Mapping[str, object]], Awaitable[Any]] = _create_pool,
    dba_config_factory: Callable[[], DatabaseRoleTopologyDBAConfig] = (
        DatabaseRoleTopologyDBAConfig
    ),
    runtime_config_factory: Callable[[], DatabaseConfig] = DatabaseConfig,
) -> dict[str, object]:
    """Attest both targets and return a role-topology receipt or fail closed."""

    dba_config = dba_config_factory()
    dba_database_url = dba_config.database_url.get_secret_value().strip()
    if not dba_database_url:
        raise PreflightError(f"Missing protected DBA DSN configuration {DBA_DSN_ENV}")

    runtime_config = runtime_config_factory()
    runtime_pool = await create_pool(_runtime_connection_kwargs(runtime_config))
    try:
        runtime_target = await _target_identity(runtime_pool, source="Atlas runtime")
        dba_pool = await create_pool({"dsn": dba_database_url})
        try:
            dba_target = await _target_identity(
                dba_pool,
                source="configured DBA connection",
            )
            _require_matching_target(runtime_target, dba_target)
            _require_direct_dba_superuser(dba_target)
            await _attest_shared_database_lock(runtime_pool, dba_pool)
            catalog = await _catalog_receipt(dba_pool)
            return {
                "receipt_version": RECEIPT_VERSION,
                "mode": "read-only",
                "runtime_target": _runtime_target_label(runtime_config),
                "dba_target": _safe_target_label(dba_database_url),
                "target_attested": True,
                "runtime_session": _identity_receipt(runtime_target),
                "dba_session": _identity_receipt(dba_target),
                "catalog": catalog,
            }
        finally:
            await dba_pool.close()
    finally:
        await runtime_pool.close()


async def _main(argv: list[str] | None = None) -> int:
    """Run the fixed preflight and print either a receipt or a safe failure."""

    _parse_args(argv)
    try:
        receipt = await _run()
    except PreflightError as exc:
        print(f"Database role-topology preflight failed: {exc}", file=sys.stderr)
        return 2
    except Exception:  # pragma: no cover - protects credential-bearing drivers
        print(
            "Database role-topology preflight failed before producing a receipt.",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
