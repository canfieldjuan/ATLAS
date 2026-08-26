"""Unit proof for the fixed, read-only database role-topology preflight."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from urllib.parse import quote, urlsplit, urlunsplit
from uuid import uuid4

import pytest


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_database_role_topology.py"


def _load_preflight_module() -> ModuleType:
    module_name = "test_database_role_topology_preflight_script"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _Acquire:
    def __init__(self, connection: "_Connection") -> None:
        self._connection = connection

    async def __aenter__(self) -> "_Connection":
        return self._connection

    async def __aexit__(self, *_args: object) -> None:
        return None


class _Transaction:
    def __init__(
        self,
        state: SimpleNamespace,
        *,
        readonly: bool,
        isolation: str | None,
    ) -> None:
        self._state = state
        self._readonly = readonly
        self._isolation = isolation

    async def __aenter__(self) -> None:
        self._state.readonly_transactions.append(self._readonly)
        self._state.transaction_isolations.append(self._isolation)
        self._state.transaction_depth += 1
        return None

    async def __aexit__(self, *_args: object) -> None:
        self._state.transaction_depth -= 1
        return None


class _Connection:
    def __init__(self, state: SimpleNamespace) -> None:
        self._state = state

    def transaction(
        self,
        *,
        readonly: bool = False,
        isolation: str | None = None,
    ) -> _Transaction:
        return _Transaction(self._state, readonly=readonly, isolation=isolation)

    def _require_readonly_transaction(self) -> None:
        if self._state.transaction_depth < 1:
            raise AssertionError("preflight query escaped its transaction")
        if not self._state.readonly_transactions[-1]:
            raise AssertionError("preflight query escaped read-only mode")

    async def fetchrow(self, query: str) -> object:
        self._require_readonly_transaction()
        if "current_schema() AS schema_name" in query:
            return {
                "schema_name": self._state.schema_name,
                "database_name": self._state.database_name,
                "database_oid": self._state.database_oid,
                "current_user": self._state.current_user,
                "session_user": self._state.session_user,
                "current_user_is_superuser": self._state.current_user_is_superuser,
            }
        if "database_catalog.datname AS database_name" in query:
            self._state.catalog_fetches += 1
            return {
                "database_name": self._state.database_name,
                "owner_role": self._state.database_owner,
            }
        raise AssertionError(f"unexpected fetchrow query: {query}")

    async def fetchval(self, query: str, *_args: object) -> object:
        self._require_readonly_transaction()
        if "pg_try_advisory_xact_lock" in query:
            self._state.advisory_lock_attempts += 1
            return self._state.advisory_lock_available
        raise AssertionError(f"unexpected fetchval query: {query}")

    async def fetch(self, query: str) -> list[dict[str, object]]:
        self._require_readonly_transaction()
        self._state.catalog_fetches += 1
        if "procedure.proacl" in query:
            return self._state.catalog["function_acl"]
        if "pg_catalog.pg_attribute AS column_attribute" in query:
            return self._state.catalog["column_acl"]
        if "pg_catalog.pg_policy AS policy" in query:
            return self._state.catalog["row_security_policies"]
        if "pg_catalog.pg_roles AS role" in query:
            return self._state.catalog["roles"]
        if "pg_catalog.pg_auth_members AS membership" in query:
            return self._state.catalog["memberships"]
        if "namespace.nspowner" in query:
            return self._state.catalog["schema_owners"]
        if "relation.relowner" in query:
            return self._state.catalog["relation_owners"]
        if "pg_catalog.pg_proc AS procedure" in query:
            return self._state.catalog["function_owners"]
        if "database_catalog.datacl" in query:
            return self._state.catalog["database_acl"]
        if "namespace.nspacl" in query:
            return self._state.catalog["schema_acl"]
        if "relation.relacl" in query:
            return self._state.catalog["relation_acl"]
        if "pg_catalog.pg_default_acl" in query:
            return self._state.catalog["default_acl"]
        raise AssertionError(f"unexpected fetch query: {query}")

    async def execute(self, *_args: object) -> None:
        raise AssertionError("read-only preflight must not execute a mutation")


class _Pool:
    def __init__(self, state: SimpleNamespace) -> None:
        self._connection = _Connection(state)
        self.acquire_calls = 0
        self.closed = False

    def acquire(self) -> _Acquire:
        self.acquire_calls += 1
        return _Acquire(self._connection)

    async def close(self) -> None:
        self.closed = True


def _catalog() -> dict[str, list[dict[str, object]]]:
    return {
        "roles": [
            {
                "role_name": "atlas",
                "can_login": True,
                "is_superuser": False,
                "inherits_privileges": True,
                "can_create_roles": False,
                "can_create_databases": False,
                "can_replicate": False,
                "bypasses_row_security": False,
            }
        ],
        "memberships": [
            {
                "membership_oid": 20_001,
                "granted_role": "atlas_eom_handoff_owner",
                "member_role": "atlas",
                "grantor_role": "postgres",
                "admin_option": False,
                "inherit_option": False,
                "set_option": True,
            }
        ],
        "schema_owners": [
            {"schema_name": "public", "owner_role": "atlas_eom_handoff_owner"}
        ],
        "relation_owners": [
            {
                "schema_name": "public",
                "relation_oid": 20_002,
                "relation_name": "work_log",
                "relation_kind": "r",
                "owner_role": "atlas_eom_handoff_owner",
                "row_security_enabled": True,
                "row_security_forced": False,
            }
        ],
        "function_owners": [
            {
                "schema_name": "public",
                "function_oid": 20_003,
                "function_name": "write_log",
                "function_kind": "f",
                "identity_arguments": "employee_id bigint",
                "owner_role": "atlas_eom_handoff_owner",
                "is_security_definer": True,
            }
        ],
        "database_acl": [
            {
                "acl_source": "explicit",
                "grantee_role": "atlas",
                "privilege_type": "CONNECT",
                "is_grantable": False,
            }
        ],
        "schema_acl": [
            {
                "acl_source": "explicit",
                "schema_name": "public",
                "grantee_role": "atlas",
                "privilege_type": "USAGE",
                "is_grantable": False,
            }
        ],
        "relation_acl": [
            {
                "acl_source": "explicit",
                "schema_name": "public",
                "relation_oid": 20_002,
                "relation_name": "work_log",
                "relation_kind": "r",
                "row_security_enabled": True,
                "row_security_forced": False,
                "grantee_role": "atlas",
                "privilege_type": "SELECT",
                "is_grantable": False,
            }
        ],
        "function_acl": [
            {
                "acl_source": "default",
                "schema_name": "public",
                "function_oid": 20_003,
                "function_name": "write_log",
                "function_kind": "f",
                "identity_arguments": "employee_id bigint",
                "is_security_definer": True,
                "grantee_role": "PUBLIC",
                "privilege_type": "EXECUTE",
                "is_grantable": False,
            }
        ],
        "column_acl": [
            {
                "acl_source": "explicit_column",
                "schema_name": "public",
                "relation_oid": 20_002,
                "relation_name": "work_log",
                "relation_kind": "r",
                "column_number": 2,
                "column_name": "hours_worked",
                "grantee_role": "atlas",
                "privilege_type": "UPDATE",
                "is_grantable": False,
            }
        ],
        "row_security_policies": [
            {
                "schema_name": "public",
                "relation_oid": 20_002,
                "relation_name": "work_log",
                "relation_kind": "r",
                "command": "r",
                "is_permissive": True,
                "policy_oid": 20_004,
                "policy_name": "atlas_work_log_read",
                "role_name": "atlas",
                "role_oid": 20_005,
            }
        ],
        "default_acl": [],
    }


def _state(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "schema_name": "public",
        "database_name": "atlas",
        "database_oid": 16_384,
        "current_user": "atlas",
        "session_user": "atlas",
        "current_user_is_superuser": False,
        "advisory_lock_available": True,
        "advisory_lock_attempts": 0,
        "readonly_transactions": [],
        "transaction_isolations": [],
        "transaction_depth": 0,
        "catalog_fetches": 0,
        "database_owner": "atlas_eom_handoff_owner",
        "catalog": _catalog(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _RuntimeConfig:
    target_label = "dsn=runtime.example.test:5432/atlas"

    def connection_kwargs(self) -> dict[str, object]:
        return {
            "host": "runtime.example.test",
            "port": 5432,
            "database": "atlas",
            "user": "atlas",
            "password": "runtime-secret",
        }


def _dba_config(runner: ModuleType, monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.setenv(
        runner.DBA_DSN_ENV,
        "postgresql://operator:dba-secret@example.test:5432/atlas?sslmode=require",
    )
    return runner.DatabaseRoleTopologyDBAConfig(_env_file=None)


def test_create_pool_enforces_readonly_defaults_and_disables_statement_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_preflight_module()
    calls: list[dict[str, object]] = []
    created_pool = object()

    async def create_pool(**kwargs: object) -> object:
        calls.append(dict(kwargs))
        return created_pool

    monkeypatch.setitem(sys.modules, "asyncpg", SimpleNamespace(create_pool=create_pool))

    pool = asyncio.run(
        runner._create_pool(
            {
                "dsn": "postgresql://operator:secret@example.test/atlas",
                "server_settings": {
                    "search_path": "public",
                    "default_transaction_read_only": "off",
                },
            }
        )
    )

    assert pool is created_pool
    assert calls == [
        {
            "dsn": "postgresql://operator:secret@example.test/atlas",
            "min_size": 1,
            "max_size": 1,
            "statement_cache_size": 0,
            "server_settings": {
                "search_path": "public",
                "default_transaction_read_only": "on",
            },
        }
    ]


def test_dba_config_uses_redacted_secret_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_preflight_module()
    config = _dba_config(runner, monkeypatch)

    assert config.database_url.get_secret_value().endswith("sslmode=require")
    assert "dba-secret" not in repr(config)


def test_preflight_returns_redacted_snapshot_from_pinned_readonly_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_preflight_module()
    runtime_state = _state()
    dba_state = _state(
        current_user="postgres",
        session_user="postgres",
        current_user_is_superuser=True,
        advisory_lock_available=False,
    )
    runtime_pool = _Pool(runtime_state)
    dba_pool = _Pool(dba_state)
    pool_kwargs: list[dict[str, object]] = []

    async def create_pool(connection_kwargs: object) -> _Pool:
        pool_kwargs.append(dict(connection_kwargs))
        return runtime_pool if len(pool_kwargs) == 1 else dba_pool

    receipt = asyncio.run(
        runner._run(
            create_pool=create_pool,
            dba_config_factory=lambda: _dba_config(runner, monkeypatch),
            runtime_config_factory=_RuntimeConfig,
        )
    )

    rendered = json.dumps(receipt, sort_keys=True)
    assert receipt["receipt_version"] == runner.RECEIPT_VERSION
    assert receipt["mode"] == "read-only"
    assert receipt["target_attested"] is True
    assert receipt["runtime_target"] == "dsn=runtime.example.test:5432/atlas"
    assert receipt["dba_target"] == "dsn=example.test:5432/atlas"
    assert receipt["runtime_session"]["current_user"] == "atlas"
    assert receipt["dba_session"]["current_user"] == "postgres"
    assert receipt["catalog"]["database_owner"] == {
        "database_name": "atlas",
        "owner_role": "atlas_eom_handoff_owner",
    }
    assert receipt["catalog"]["roles"] == dba_state.catalog["roles"]
    assert receipt["catalog"]["memberships"] == dba_state.catalog["memberships"]
    assert receipt["catalog"]["relation_owners"] == dba_state.catalog[
        "relation_owners"
    ]
    assert receipt["catalog"]["function_acl"] == dba_state.catalog[
        "function_acl"
    ]
    assert receipt["catalog"]["column_acl"] == dba_state.catalog[
        "column_acl"
    ]
    assert receipt["catalog"]["row_security_policies"] == dba_state.catalog[
        "row_security_policies"
    ]
    assert "dba-secret" not in rendered
    assert "runtime-secret" not in rendered
    assert pool_kwargs == [
        {
            "host": "runtime.example.test",
            "port": 5432,
            "database": "atlas",
            "user": "atlas",
            "password": "runtime-secret",
        },
        {
            "dsn": "postgresql://operator:dba-secret@example.test:5432/atlas?sslmode=require"
        },
    ]
    assert runtime_pool.closed is True
    assert dba_pool.closed is True
    assert runtime_pool.acquire_calls == 1
    assert dba_pool.acquire_calls == 1
    assert runtime_state.readonly_transactions
    assert dba_state.readonly_transactions
    assert all(runtime_state.readonly_transactions)
    assert all(dba_state.readonly_transactions)
    assert runtime_state.transaction_isolations == ["repeatable_read"]
    assert dba_state.transaction_isolations == ["repeatable_read"]


@pytest.mark.parametrize("database_url", ("", "   "))
def test_preflight_rejects_missing_or_blank_dba_dsn_before_opening_a_pool(
    database_url: str,
) -> None:
    runner = _load_preflight_module()

    async def create_pool(_connection_kwargs: object) -> _Pool:
        raise AssertionError("missing DBA config must prevent pool creation")

    with pytest.raises(
        runner.PreflightError,
        match=f"Missing protected DBA DSN configuration {runner.DBA_DSN_ENV}",
    ):
        asyncio.run(
            runner._run(
                create_pool=create_pool,
                dba_config_factory=lambda: runner.DatabaseRoleTopologyDBAConfig(
                    database_url=database_url,
                    _env_file=None,
                ),
                runtime_config_factory=_RuntimeConfig,
            )
        )


def test_preflight_rejects_database_mismatch_before_catalog_reporting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_preflight_module()
    runtime_pool = _Pool(_state())
    dba_state = _state(
        database_name="atlas_clone",
        current_user="postgres",
        session_user="postgres",
        current_user_is_superuser=True,
    )
    dba_pool = _Pool(dba_state)
    calls = 0

    async def create_pool(_connection_kwargs: object) -> _Pool:
        nonlocal calls
        calls += 1
        return runtime_pool if calls == 1 else dba_pool

    with pytest.raises(
        runner.PreflightError,
        match="does not target the Atlas runtime database",
    ):
        asyncio.run(
            runner._run(
                create_pool=create_pool,
                dba_config_factory=lambda: _dba_config(runner, monkeypatch),
                runtime_config_factory=_RuntimeConfig,
            )
        )

    assert runtime_pool.closed is True
    assert dba_pool.closed is True
    assert dba_state.catalog_fetches == 0
    assert all(dba_state.readonly_transactions)


@pytest.mark.parametrize(
    ("dba_overrides", "message"),
    (
        ({"database_oid": 16_385}, "does not target the Atlas runtime database"),
        ({"schema_name": "operations"}, "does not resolve to the Atlas runtime schema"),
    ),
)
def test_preflight_rejects_every_target_identity_mismatch_before_report(
    monkeypatch: pytest.MonkeyPatch,
    dba_overrides: dict[str, object],
    message: str,
) -> None:
    runner = _load_preflight_module()
    runtime_pool = _Pool(_state())
    dba_state = _state(
        current_user="postgres",
        session_user="postgres",
        current_user_is_superuser=True,
        **dba_overrides,
    )
    dba_pool = _Pool(dba_state)
    calls = 0

    async def create_pool(_connection_kwargs: object) -> _Pool:
        nonlocal calls
        calls += 1
        return runtime_pool if calls == 1 else dba_pool

    with pytest.raises(runner.PreflightError, match=message):
        asyncio.run(
            runner._run(
                create_pool=create_pool,
                dba_config_factory=lambda: _dba_config(runner, monkeypatch),
                runtime_config_factory=_RuntimeConfig,
            )
        )

    assert dba_state.catalog_fetches == 0
    assert dba_state.advisory_lock_attempts == 0
    assert all(dba_state.readonly_transactions)


def test_target_identity_rejects_an_ambiguous_catalog_superuser_flag() -> None:
    runner = _load_preflight_module()
    state = _state(current_user_is_superuser="true")
    pool = _Pool(state)

    async def read_identity() -> object:
        async with pool.acquire() as connection:
            async with connection.transaction(
                isolation="repeatable_read",
                readonly=True,
            ):
                return await runner._target_identity(connection, source="test target")

    with pytest.raises(
        runner.PreflightError,
        match="Missing or invalid superuser status from test target",
    ):
        asyncio.run(read_identity())

    assert state.readonly_transactions == [True]
    assert state.transaction_isolations == ["repeatable_read"]


def test_catalog_row_rejects_an_unexpected_column_key() -> None:
    runner = _load_preflight_module()

    with pytest.raises(
        runner.PreflightError,
        match="Catalog receipt contained an invalid row from test catalog",
    ):
        runner._json_catalog_row({1: "value"}, source="test catalog")


def test_preflight_rejects_a_switched_or_non_superuser_dba_session_before_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_preflight_module()
    runtime_pool = _Pool(_state())
    dba_state = _state(
        current_user="atlas",
        session_user="postgres",
        current_user_is_superuser=True,
    )
    dba_pool = _Pool(dba_state)
    calls = 0

    async def create_pool(_connection_kwargs: object) -> _Pool:
        nonlocal calls
        calls += 1
        return runtime_pool if calls == 1 else dba_pool

    with pytest.raises(
        runner.PreflightError,
        match="must use a direct DBA login, not SET ROLE",
    ):
        asyncio.run(
            runner._run(
                create_pool=create_pool,
                dba_config_factory=lambda: _dba_config(runner, monkeypatch),
                runtime_config_factory=_RuntimeConfig,
            )
        )

    assert dba_state.catalog_fetches == 0
    assert dba_state.advisory_lock_attempts == 0
    assert all(dba_state.readonly_transactions)


def test_preflight_rejects_non_superuser_dba_session_before_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_preflight_module()
    runtime_pool = _Pool(_state())
    dba_state = _state(current_user="operator", session_user="operator")
    dba_pool = _Pool(dba_state)
    calls = 0

    async def create_pool(_connection_kwargs: object) -> _Pool:
        nonlocal calls
        calls += 1
        return runtime_pool if calls == 1 else dba_pool

    with pytest.raises(
        runner.PreflightError,
        match="Configured DBA connection is not a PostgreSQL superuser",
    ):
        asyncio.run(
            runner._run(
                create_pool=create_pool,
                dba_config_factory=lambda: _dba_config(runner, monkeypatch),
                runtime_config_factory=_RuntimeConfig,
            )
        )

    assert dba_state.catalog_fetches == 0
    assert dba_state.advisory_lock_attempts == 0
    assert all(dba_state.readonly_transactions)


def test_preflight_rejects_a_target_that_does_not_share_the_lock_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_preflight_module()
    runtime_pool = _Pool(_state())
    dba_state = _state(
        current_user="postgres",
        session_user="postgres",
        current_user_is_superuser=True,
        advisory_lock_available=True,
    )
    dba_pool = _Pool(dba_state)
    calls = 0

    async def create_pool(_connection_kwargs: object) -> _Pool:
        nonlocal calls
        calls += 1
        return runtime_pool if calls == 1 else dba_pool

    with pytest.raises(
        runner.PreflightError,
        match="does not share the Atlas runtime database cluster",
    ):
        asyncio.run(
            runner._run(
                create_pool=create_pool,
                dba_config_factory=lambda: _dba_config(runner, monkeypatch),
                runtime_config_factory=_RuntimeConfig,
            )
    )

    assert dba_state.catalog_fetches == 0
    assert all(runtime_pool._connection._state.readonly_transactions)
    assert all(dba_state.readonly_transactions)


def test_preflight_has_no_apply_or_mutation_command_surface() -> None:
    runner = _load_preflight_module()
    source = SCRIPT.read_text(encoding="utf-8")

    with pytest.raises(SystemExit):
        runner._parse_args(["--apply"])
    assert "--apply" not in source
    assert ".execute(" not in source
    assert "readonly=True" in source
    assert 'isolation="repeatable_read"' in source
    assert "acldefault('d'" in source
    assert "acldefault('n'" in source
    assert "acldefault('f'" in source
    assert "NULLIF(policy.polroles, ARRAY[]::oid[])" in source
    assert "membership.inherit_option" in source
    assert "membership.set_option" in source
    assert "relation.relname AS relation_name" in source
    assert "policy.polname AS policy_name" in source
    assert "count(*)::bigint" not in source


def test_main_does_not_render_an_unexpected_driver_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _load_preflight_module()

    async def fail_with_sensitive_driver_message() -> dict[str, object]:
        raise ValueError("postgresql://operator:driver-secret@example.test/atlas")

    monkeypatch.setattr(runner, "_run", fail_with_sensitive_driver_message)

    assert asyncio.run(runner._main([])) == 2
    captured = capsys.readouterr()
    assert "driver-secret" not in captured.err
    assert "failed before producing a receipt" in captured.err


_ROLE_TOPOLOGY_TEST_DBA_DSN_ENV = (
    "ATLAS_DATABASE_ROLE_TOPOLOGY_TEST_DBA_DATABASE_URL"
)


def _disposable_role_topology_dba_dsn() -> str:
    """Return the explicitly provisioned PostgreSQL 16 test target only."""

    database_url = os.environ.get(_ROLE_TOPOLOGY_TEST_DBA_DSN_ENV, "").strip()
    if not database_url:
        pytest.skip(
            "set ATLAS_DATABASE_ROLE_TOPOLOGY_TEST_DBA_DATABASE_URL to run "
            "the disposable PostgreSQL 16 role-topology integration test"
        )
    try:
        return _require_disposable_role_topology_dba_dsn(database_url)
    except ValueError as exc:
        pytest.fail(str(exc))


def _require_disposable_role_topology_dba_dsn(database_url: str) -> str:
    """Reject a non-test or remote database before the fixture can mutate it."""

    parsed = urlsplit(database_url)
    database_name = parsed.path.lstrip("/")
    if not database_name.endswith(("_test", "_tests")):
        raise ValueError("role-topology integration requires a disposable test database")
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("role-topology integration requires a loopback database host")
    if not parsed.hostname or not parsed.username:
        raise ValueError("role-topology integration requires a complete disposable DBA DSN")
    return database_url


@pytest.mark.parametrize(
    ("database_url", "message"),
    (
        ("postgresql://atlas:atlas@localhost:5432/atlas", "disposable test database"),
        (
            "postgresql://atlas:atlas@example.test:5432/atlas_migration_tests",
            "loopback database host",
        ),
        ("postgresql://localhost:5432/atlas_migration_tests", "complete disposable DBA DSN"),
    ),
)
def test_disposable_role_topology_dsn_rejects_non_test_targets(
    database_url: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _require_disposable_role_topology_dba_dsn(database_url)


def _quoted_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("work_log", '"work_log"'),
        ('work"log', '"work""log"'),
    ),
)
def test_quoted_identifier_uses_postgresql_double_quote_escaping(
    value: str,
    expected: str,
) -> None:
    assert _quoted_identifier(value) == expected


def _runtime_dsn_from_dba_dsn(
    dba_database_url: str,
    *,
    runtime_role: str,
    runtime_password: str,
) -> str:
    """Build a test-only runtime login URL without changing the DBA target."""

    parsed = urlsplit(dba_database_url)
    assert parsed.hostname is not None
    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port else ""
    netloc = (
        f"{quote(runtime_role, safe='')}:{quote(runtime_password, safe='')}"
        f"@{host}{port}"
    )
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, ""))


@pytest.mark.integration
def test_role_topology_preflight_executes_fixed_queries_against_postgresql_16() -> None:
    """Exercise the real command against a sealed disposable PostgreSQL target."""

    asyncpg = pytest.importorskip("asyncpg")
    dba_database_url = _disposable_role_topology_dba_dsn()
    suffix = uuid4().hex
    schema_name = f"role_topology_{suffix}"
    runtime_role = f"role_topology_runtime_{suffix}"
    granted_role = f"role_topology_granted_{suffix}"
    relation_name = "work_log"
    function_name = "write_log"
    policy_name = "read_work_log"
    runtime_password = f"runtime-{suffix}"
    schema = _quoted_identifier(schema_name)
    runtime = _quoted_identifier(runtime_role)
    granted = _quoted_identifier(granted_role)
    relation = f"{schema}.{_quoted_identifier(relation_name)}"
    function = f"{schema}.{_quoted_identifier(function_name)}(employee_id bigint)"
    policy = _quoted_identifier(policy_name)

    async def exercise() -> dict[str, object]:
        connection = await asyncpg.connect(dba_database_url)
        runtime_created = False
        granted_created = False
        try:
            await connection.execute(
                f"CREATE ROLE {runtime} LOGIN PASSWORD "
                f"'{runtime_password}' INHERIT NOSUPERUSER NOCREATEDB "
                "NOCREATEROLE NOREPLICATION NOBYPASSRLS"
            )
            runtime_created = True
            await connection.execute(f"CREATE ROLE {granted} NOLOGIN")
            granted_created = True
            await connection.execute(
                f"GRANT {granted} TO {runtime} WITH ADMIN FALSE, "
                "INHERIT FALSE, SET TRUE"
            )
            await connection.execute(f"GRANT USAGE ON SCHEMA public TO {runtime}")
            await connection.execute(f"CREATE SCHEMA {schema}")
            await connection.execute(
                f"CREATE TABLE {relation} (row_id bigint, hours_worked integer)"
            )
            await connection.execute(f"ALTER TABLE {relation} ENABLE ROW LEVEL SECURITY")
            await connection.execute(
                f"CREATE POLICY {policy} ON {relation} FOR SELECT TO {runtime} "
                "USING (true)"
            )
            await connection.execute(f"GRANT USAGE ON SCHEMA {schema} TO {runtime}")
            await connection.execute(f"GRANT SELECT ON {relation} TO {runtime}")
            await connection.execute(
                f"GRANT UPDATE (hours_worked) ON {relation} TO {runtime}"
            )
            await connection.execute(
                f"CREATE FUNCTION {function} RETURNS bigint LANGUAGE sql "
                "SECURITY DEFINER AS $$ SELECT employee_id $$"
            )
            await connection.execute(f"GRANT EXECUTE ON FUNCTION {function} TO {runtime}")

            environment = dict(os.environ)
            environment["ATLAS_DB_CONNECTION_STRING"] = _runtime_dsn_from_dba_dsn(
                dba_database_url,
                runtime_role=runtime_role,
                runtime_password=runtime_password,
            )
            environment["ATLAS_DATABASE_ROLE_TOPOLOGY_DBA_DATABASE_URL"] = (
                dba_database_url
            )
            result = subprocess.run(
                [sys.executable, str(SCRIPT)],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            assert result.returncode == 0, result.stderr
            return json.loads(result.stdout)
        finally:
            await connection.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            if runtime_created and granted_created:
                await connection.execute(f"REVOKE {granted} FROM {runtime}")
            if runtime_created:
                await connection.execute(f"DROP ROLE {runtime}")
            if granted_created:
                await connection.execute(f"DROP ROLE {granted}")
            await connection.close()

    receipt = asyncio.run(exercise())
    catalog = receipt["catalog"]
    membership = next(
        row
        for row in catalog["memberships"]
        if row["granted_role"] == granted_role and row["member_role"] == runtime_role
    )
    assert isinstance(membership["membership_oid"], int)
    assert isinstance(membership["grantor_role"], str)
    assert membership["admin_option"] is False
    assert membership["inherit_option"] is False
    assert membership["set_option"] is True

    relation_owner = next(
        row
        for row in catalog["relation_owners"]
        if row["schema_name"] == schema_name
        and row["relation_name"] == relation_name
    )
    assert isinstance(relation_owner["relation_oid"], int)

    relation_acl = next(
        row
        for row in catalog["relation_acl"]
        if row["schema_name"] == schema_name
        and row["relation_name"] == relation_name
        and row["grantee_role"] == runtime_role
        and row["privilege_type"] == "SELECT"
    )
    assert isinstance(relation_acl["relation_oid"], int)

    column_acl = next(
        row
        for row in catalog["column_acl"]
        if row["schema_name"] == schema_name
        and row["relation_name"] == relation_name
        and row["column_name"] == "hours_worked"
        and row["grantee_role"] == runtime_role
        and row["privilege_type"] == "UPDATE"
    )
    assert isinstance(column_acl["relation_oid"], int)
    assert isinstance(column_acl["column_number"], int)

    function_acl = next(
        row
        for row in catalog["function_acl"]
        if row["schema_name"] == schema_name
        and row["function_name"] == function_name
        and row["grantee_role"] == runtime_role
        and row["privilege_type"] == "EXECUTE"
    )
    assert isinstance(function_acl["function_oid"], int)
    assert function_acl["identity_arguments"] == "employee_id bigint"

    function_owner = next(
        row
        for row in catalog["function_owners"]
        if row["schema_name"] == schema_name
        and row["function_name"] == function_name
    )
    assert isinstance(function_owner["function_oid"], int)
    assert function_owner["identity_arguments"] == "employee_id bigint"

    policy_row = next(
        row
        for row in catalog["row_security_policies"]
        if row["schema_name"] == schema_name
        and row["relation_name"] == relation_name
        and row["policy_name"] == policy_name
        and row["role_name"] == runtime_role
    )
    assert isinstance(policy_row["policy_oid"], int)
    assert isinstance(policy_row["relation_oid"], int)
