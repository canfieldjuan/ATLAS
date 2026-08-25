"""Unit proof for the controlled DBA migration entrypoint."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "apply_eom_missed_call_recovery_runtime_privileges.py"
TEST_DBA_DSN = "postgresql://operator:test-only@example.test:5432/atlas"
TEST_RUNTIME_DSN = "postgresql://atlas_funnel:test-only@example.test:5432/atlas"


def _load_runner_module() -> ModuleType:
    module_name = "test_eom_missed_call_privilege_runner_script"
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

    def __await__(self):
        async def _connection() -> "_Connection":
            return self._connection

        return _connection().__await__()


class _Transaction:
    def __init__(self, connection: "_Connection") -> None:
        self._connection = connection

    async def __aenter__(self) -> "_Connection":
        state = self._connection._state
        state.transaction_depth = getattr(state, "transaction_depth", 0) + 1
        return self._connection

    async def __aexit__(self, *_args: object) -> None:
        state = self._connection._state
        state.transaction_depth -= 1
        if state.transaction_depth == 0:
            lock_state = getattr(state, "advisory_lock_state", state)
            lock_state.advisory_lock_held = False
        return None


class _Connection:
    def __init__(self, state: SimpleNamespace) -> None:
        self._state = state

    def transaction(self) -> _Transaction:
        return _Transaction(self)

    async def fetchval(self, query: str, *args: object) -> object:
        if "pg_try_advisory_xact_lock" in query:
            self._state.advisory_lock_attempts = (
                getattr(self._state, "advisory_lock_attempts", 0) + 1
            )
            lock_state = getattr(self._state, "advisory_lock_state", self._state)
            if getattr(lock_state, "advisory_lock_held", False):
                return False
            lock_state.advisory_lock_held = True
            return True
        if "pg_try_advisory_lock" in query:
            self._state.controlled_migration_lock_attempts = (
                getattr(self._state, "controlled_migration_lock_attempts", 0) + 1
            )
            lock_results = getattr(
                self._state,
                "controlled_migration_lock_results",
                [],
            )
            if lock_results:
                return lock_results.pop(0)
            return True
        if "pg_advisory_unlock" in query:
            self._state.controlled_migration_unlocks = (
                getattr(self._state, "controlled_migration_unlocks", 0) + 1
            )
            return True
        if "rolsuper" in query:
            return self._state.executor_is_superuser
        if "to_regclass" in query:
            return self._state.migrations_table_exists
        if "schema_migrations" in query:
            assert len(args) == 1
            return args[0] in self._state.recorded_migrations
        raise AssertionError(f"unexpected query: {query}")

    async def fetchrow(self, query: str, *args: object) -> object:
        if "current_schema() AS schema_name" in query:
            return {
                "schema_name": getattr(self._state, "schema_name", "public"),
                "database_name": getattr(self._state, "database_name", "atlas"),
                "database_oid": getattr(self._state, "database_oid", 16_384),
                "current_user": getattr(self._state, "current_user", "atlas_funnel"),
                "session_user": getattr(self._state, "session_user", "atlas_funnel"),
            }
        assert "atlas:eom-missed-call-recovery-role-admission" in query
        assert args == ("atlas_funnel",)
        self._state.role_admission_calls = getattr(
            self._state, "role_admission_calls", []
        )
        self._state.role_admission_calls.append(args)
        return {
            "guard_role_ready": getattr(self._state, "guard_role_ready", True),
            "runtime_role_ready": getattr(self._state, "runtime_role_ready", True),
            "nocodb_role_ready": getattr(self._state, "nocodb_role_ready", True),
        }

    async def execute(self, query: str, *args: object) -> None:
        self._state.executed = getattr(self._state, "executed", [])
        self._state.executed.append((query, args))
        self._state.events = getattr(self._state, "events", [])
        self._state.events.append(("execute", query, args))


class _Pool:
    def __init__(self, state: SimpleNamespace) -> None:
        self._connection = _Connection(state)
        self.closed = False

    def acquire(self) -> _Acquire:
        return _Acquire(self._connection)

    async def release(self, connection: _Connection) -> None:
        assert connection is self._connection

    async def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _configured_runtime_dsn(monkeypatch) -> None:
    monkeypatch.setenv(
        "ATLAS_EOM_MISSED_CALL_RECOVERY_DBA_DATABASE_URL",
        TEST_DBA_DSN,
    )
    monkeypatch.setenv(
        "ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING",
        TEST_RUNTIME_DSN,
    )


def test_privilege_runner_defaults_to_read_only_and_redacts_dsn(monkeypatch) -> None:
    runner = _load_runner_module()
    monkeypatch.setenv(
        runner.DBA_DSN_ENV,
        "postgresql://operator:secret@example.test:5432/atlas?sslmode=require",
    )
    state = SimpleNamespace(
        executor_is_superuser=True,
        migrations_table_exists=True,
        recorded_migrations=set(),
    )
    pool = _Pool(state)
    calls: list[object] = []

    async def create_pool(_database_url: str, **_kwargs: object) -> _Pool:
        return pool

    async def run_migrations(*_args: object, **_kwargs: object) -> None:
        calls.append(object())

    args = runner._parse_args([])
    result = asyncio.run(
        runner._run(
            args,
            create_pool=create_pool,
            run_migrations_fn=run_migrations,
        )
    )

    assert result == {
        "target": "dsn=example.test:5432/atlas",
        "executor_is_superuser": True,
        "runtime_role": "atlas_funnel",
        "historical_prelude_migrations": {
            name: False for name in runner.HISTORICAL_PRELUDE_MIGRATION_NAMES
        },
        "prerequisite_migration": runner.PREREQUISITE_MIGRATION_NAME,
        "prerequisite_migration_recorded": False,
        "migration": runner.MIGRATION_NAME,
        "migration_recorded": False,
        "applied": False,
    }
    assert calls == []
    assert pool.closed is True


def test_privilege_runner_requires_superuser_before_apply(monkeypatch) -> None:
    runner = _load_runner_module()
    monkeypatch.setenv(runner.DBA_DSN_ENV, "postgresql://example.test/atlas")
    state = SimpleNamespace(
        executor_is_superuser=False,
        migrations_table_exists=True,
        recorded_migrations={runner.PREREQUISITE_MIGRATION_NAME},
    )
    pool = _Pool(state)

    async def create_pool(_database_url: str, **_kwargs: object) -> _Pool:
        return pool

    args = runner._parse_args(["--apply"])
    with pytest.raises(RuntimeError, match="not a PostgreSQL superuser"):
        asyncio.run(runner._run(args, create_pool=create_pool))
    assert pool.closed is True


def test_privilege_runner_rejects_missing_runtime_dsn(monkeypatch) -> None:
    runner = _load_runner_module()
    monkeypatch.setenv(runner.DBA_DSN_ENV, "postgresql://example.test/atlas")
    monkeypatch.delenv("ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING")

    args = runner._parse_args([])
    with pytest.raises(
        RuntimeError, match="Missing EOM funnel runtime DSN configuration"
    ):
        asyncio.run(
            runner._run(
                args,
                funnel_config_factory=lambda: SimpleNamespace(db_connection_string=""),
            )
        )


def test_runtime_role_migration_pool_binds_the_configured_role() -> None:
    runner = _load_runner_module()
    state = SimpleNamespace(
        executor_is_superuser=True,
        migrations_table_exists=True,
        recorded_migrations=set(),
    )
    pool = _Pool(state)
    migration_pool = runner._RuntimeRoleMigrationPool(pool, "atlas_funnel")

    connection = asyncio.run(migration_pool.acquire())
    asyncio.run(migration_pool.release(connection))

    assert state.executed == [
        (
            "SELECT pg_catalog.set_config($1, $2, FALSE)",
            (runner.RUNTIME_ROLE_SETTING, "atlas_funnel"),
        )
    ]


def test_privilege_runner_applies_only_the_recorded_repair(monkeypatch) -> None:
    runner = _load_runner_module()
    monkeypatch.setenv(runner.DBA_DSN_ENV, "postgresql://example.test/atlas")
    state = SimpleNamespace(
        executor_is_superuser=True,
        migrations_table_exists=True,
        recorded_migrations={runner.PREREQUISITE_MIGRATION_NAME},
    )
    pool = _Pool(state)
    calls: list[tuple[object, tuple[str, ...]]] = []

    async def create_pool(_database_url: str, **_kwargs: object) -> _Pool:
        return pool

    async def run_migrations(observed_pool: object, *, only: tuple[str, ...]) -> None:
        calls.append((observed_pool, only))
        if only == (runner.MIGRATION_NAME,):
            state.recorded_migrations.add(runner.MIGRATION_NAME)
        else:
            assert only == runner.HISTORICAL_PRELUDE_MIGRATION_NAMES

    args = runner._parse_args(["--apply"])
    result = asyncio.run(
        runner._run(
            args,
            create_pool=create_pool,
            run_migrations_fn=run_migrations,
        )
    )

    assert [only for _pool, only in calls] == [
        runner.HISTORICAL_PRELUDE_MIGRATION_NAMES,
        (runner.MIGRATION_NAME,),
    ]
    historical_pool, _historical_only = calls[0]
    assert isinstance(historical_pool, runner._RuntimeRoleMigrationPool)
    assert historical_pool._pool is pool
    assert historical_pool._runtime_role == "atlas_funnel"
    controlled_pool, _controlled_only = calls[1]
    assert isinstance(controlled_pool, runner._RuntimeRoleMigrationPool)
    assert isinstance(controlled_pool._pool, runner._PinnedMigrationPool)
    assert controlled_pool._pool._connection is pool._connection
    assert controlled_pool._runtime_role == "atlas_funnel"
    assert result["prerequisite_migration_recorded"] is True
    assert result["migration_recorded"] is True
    assert result["applied"] is True
    assert pool.closed is True


@pytest.mark.parametrize(
    ("migrations_table_exists", "recorded_migrations"),
    [
        (False, frozenset()),
        (True, frozenset()),
        (True, frozenset({"388_unrelated_migration"})),
        (True, frozenset({"393_eom_missed_call_recovery_runtime_privileges"})),
        (
            True,
            frozenset(
                {
                    "388_unrelated_migration",
                    "393_eom_missed_call_recovery_runtime_privileges",
                }
            ),
        ),
    ],
    ids=(
        "no-ledger-table",
        "empty-ledger",
        "unrelated-receipt",
        "repair-without-prerequisite",
        "mixed-receipts-without-prerequisite",
    ),
)
def test_privilege_runner_refuses_apply_without_prerequisite_receipt(
    monkeypatch,
    migrations_table_exists: bool,
    recorded_migrations: frozenset[str],
) -> None:
    runner = _load_runner_module()
    monkeypatch.setenv(runner.DBA_DSN_ENV, "postgresql://example.test/atlas")
    state = SimpleNamespace(
        executor_is_superuser=True,
        migrations_table_exists=migrations_table_exists,
        recorded_migrations=set(recorded_migrations),
    )
    pool = _Pool(state)
    calls: list[tuple[str, ...]] = []

    async def create_pool(_database_url: str, **_kwargs: object) -> _Pool:
        return pool

    async def run_migrations(*_args: object, **kwargs: object) -> None:
        only = kwargs["only"]
        assert isinstance(only, tuple)
        calls.append(only)

    args = runner._parse_args(["--apply"])
    with pytest.raises(
        RuntimeError,
        match="389_eom_missed_call_recovery is not recorded",
    ):
        asyncio.run(
            runner._run(
                args,
                create_pool=create_pool,
                run_migrations_fn=run_migrations,
            )
        )

    expected_calls = (
        []
        if (not migrations_table_exists or runner.MIGRATION_NAME in recorded_migrations)
        else [runner.HISTORICAL_PRELUDE_MIGRATION_NAMES]
    )
    assert calls == expected_calls
    assert (runner.MIGRATION_NAME,) not in calls
    assert pool.closed is True


def test_privilege_runner_applies_historical_preludes_before_privilege_repair(
    monkeypatch,
) -> None:
    runner = _load_runner_module()
    monkeypatch.setenv(runner.DBA_DSN_ENV, "postgresql://example.test/atlas")
    state = SimpleNamespace(
        executor_is_superuser=True,
        migrations_table_exists=True,
        recorded_migrations={runner.PREREQUISITE_MIGRATION_NAME},
    )
    pool = _Pool(state)
    calls: list[tuple[str, ...]] = []
    prelude_names = iter(runner.HISTORICAL_PRELUDE_MIGRATION_NAMES)

    async def create_pool(_database_url: str, **_kwargs: object) -> _Pool:
        return pool

    async def run_migrations(_pool: object, *, only: tuple[str, ...]) -> None:
        calls.append(only)
        if only == runner.HISTORICAL_PRELUDE_MIGRATION_NAMES:
            state.recorded_migrations.add(next(prelude_names))
        elif only == (runner.MIGRATION_NAME,):
            state.recorded_migrations.add(runner.MIGRATION_NAME)
        else:  # pragma: no cover - regression guard for this controlled runner
            raise AssertionError(f"unexpected migration selection: {only}")

    args = runner._parse_args(["--apply"])
    result = asyncio.run(
        runner._run(
            args,
            create_pool=create_pool,
            run_migrations_fn=run_migrations,
        )
    )

    assert calls == [
        runner.HISTORICAL_PRELUDE_MIGRATION_NAMES,
        runner.HISTORICAL_PRELUDE_MIGRATION_NAMES,
        runner.HISTORICAL_PRELUDE_MIGRATION_NAMES,
        (runner.MIGRATION_NAME,),
    ]
    assert result["historical_prelude_migrations"] == {
        name: True for name in runner.HISTORICAL_PRELUDE_MIGRATION_NAMES
    }
    assert result["prerequisite_migration_recorded"] is True
    assert result["migration_recorded"] is True
    assert result["applied"] is True
    assert pool.closed is True


def test_privilege_runner_pins_393_role_setting_and_runner_connection(
    monkeypatch,
) -> None:
    """A transaction-pooling proxy cannot split 393 across backends."""

    runner = _load_runner_module()
    lock_state = SimpleNamespace()
    runtime_pool = _Pool(
        SimpleNamespace(
            schema_name="eom_canonical",
            advisory_lock_state=lock_state,
        )
    )
    dba_state = SimpleNamespace(
        schema_name="eom_canonical",
        executor_is_superuser=True,
        migrations_table_exists=True,
        recorded_migrations={
            runner.PREREQUISITE_MIGRATION_NAME,
            *runner.HISTORICAL_PRELUDE_MIGRATION_NAMES,
        },
        advisory_lock_state=lock_state,
        controlled_migration_lock_results=[False, True],
    )
    dba_pool = _Pool(dba_state)
    sleep_transaction_depths: list[int] = []
    observed_migration_connections: list[object] = []

    async def create_pool(
        database_url: str,
        *,
        schema_name: str | None = None,
    ) -> _Pool:
        if database_url == TEST_RUNTIME_DSN:
            assert schema_name is None
            return runtime_pool
        assert database_url == TEST_DBA_DSN
        assert schema_name == "eom_canonical"
        return dba_pool

    async def sleep(_delay: float) -> None:
        sleep_transaction_depths.append(dba_state.transaction_depth)

    async def run_migrations(observed_pool: object, *, only: tuple[str, ...]) -> None:
        if only == runner.HISTORICAL_PRELUDE_MIGRATION_NAMES:
            return
        assert only == (runner.MIGRATION_NAME,)
        assert isinstance(observed_pool, runner._RuntimeRoleMigrationPool)
        connection = await observed_pool.acquire()
        observed_migration_connections.append(connection)
        assert connection is dba_pool._connection
        assert dba_state.transaction_depth == 1
        await observed_pool.release(connection)
        dba_state.recorded_migrations.add(runner.MIGRATION_NAME)

    monkeypatch.setattr(runner.asyncio, "sleep", sleep)
    result = asyncio.run(
        runner._run(
            runner._parse_args(["--apply"]),
            create_pool=create_pool,
            run_migrations_fn=run_migrations,
            funnel_config_factory=lambda: SimpleNamespace(
                db_connection_string=TEST_RUNTIME_DSN
            ),
        )
    )

    assert sleep_transaction_depths == [0]
    assert observed_migration_connections == [dba_pool._connection]
    assert dba_state.controlled_migration_lock_attempts == 2
    assert dba_state.controlled_migration_unlocks == 1
    assert (
        "SELECT pg_catalog.set_config($1, $2, FALSE)",
        (runner.RUNTIME_ROLE_SETTING, "atlas_funnel"),
    ) in dba_state.executed
    assert result["migration_recorded"] is True
    assert result["applied"] is True
    assert runtime_pool.closed is True
    assert dba_pool.closed is True


def test_privilege_runner_provisions_pgcrypto_before_historical_preludes(
    monkeypatch,
) -> None:
    runner = _load_runner_module()
    monkeypatch.setenv(runner.DBA_DSN_ENV, "postgresql://example.test/atlas")
    state = SimpleNamespace(
        executor_is_superuser=True,
        migrations_table_exists=True,
        recorded_migrations={runner.PREREQUISITE_MIGRATION_NAME},
        events=[],
    )
    pool = _Pool(state)

    async def create_pool(_database_url: str, **_kwargs: object) -> _Pool:
        return pool

    async def run_migrations(_pool: object, *, only: tuple[str, ...]) -> None:
        state.events.append(("migration", only))
        if only == runner.HISTORICAL_PRELUDE_MIGRATION_NAMES:
            state.recorded_migrations.add(runner.HISTORICAL_PRELUDE_MIGRATION_NAMES[0])
        elif only == (runner.MIGRATION_NAME,):
            state.recorded_migrations.add(runner.MIGRATION_NAME)

    args = runner._parse_args(["--apply"])
    asyncio.run(
        runner._run(
            args,
            create_pool=create_pool,
            run_migrations_fn=run_migrations,
        )
    )

    assert state.events[0] == (
        "execute",
        "CREATE EXTENSION IF NOT EXISTS pgcrypto",
        (),
    )
    assert state.events[1] == (
        "migration",
        runner.HISTORICAL_PRELUDE_MIGRATION_NAMES,
    )
    assert state.role_admission_calls == [("atlas_funnel",)]


@pytest.mark.parametrize(
    "invalid_admission",
    ("guard_role_ready", "runtime_role_ready", "nocodb_role_ready"),
)
def test_privilege_runner_rejects_invalid_role_admission_before_pgcrypto(
    monkeypatch,
    invalid_admission: str,
) -> None:
    runner = _load_runner_module()
    monkeypatch.setenv(runner.DBA_DSN_ENV, "postgresql://example.test/atlas")
    state = SimpleNamespace(
        executor_is_superuser=True,
        migrations_table_exists=True,
        recorded_migrations={runner.PREREQUISITE_MIGRATION_NAME},
    )
    setattr(state, invalid_admission, False)
    pool = _Pool(state)
    calls: list[tuple[str, ...]] = []

    async def create_pool(_database_url: str, **_kwargs: object) -> _Pool:
        return pool

    async def run_migrations(_pool: object, *, only: tuple[str, ...]) -> None:
        calls.append(only)

    args = runner._parse_args(["--apply"])
    with pytest.raises(RuntimeError, match="role admission failed"):
        asyncio.run(
            runner._run(
                args,
                create_pool=create_pool,
                run_migrations_fn=run_migrations,
            )
        )

    assert state.role_admission_calls == [("atlas_funnel",)]
    assert getattr(state, "executed", []) == []
    assert calls == []
    assert pool.closed is True


def test_privilege_runner_continues_after_committed_historical_progress_stop(
    monkeypatch,
) -> None:
    """A committed prelude receipt authorizes retrying the exact selector."""

    runner = _load_runner_module()
    monkeypatch.setenv(runner.DBA_DSN_ENV, "postgresql://example.test/atlas")
    state = SimpleNamespace(
        executor_is_superuser=True,
        migrations_table_exists=True,
        recorded_migrations={runner.PREREQUISITE_MIGRATION_NAME},
    )
    pool = _Pool(state)
    calls: list[tuple[str, ...]] = []
    prelude_names = iter(runner.HISTORICAL_PRELUDE_MIGRATION_NAMES)

    async def create_pool(_database_url: str, **_kwargs: object) -> _Pool:
        return pool

    async def run_migrations(_pool: object, *, only: tuple[str, ...]) -> None:
        calls.append(only)
        if only == runner.HISTORICAL_PRELUDE_MIGRATION_NAMES:
            state.recorded_migrations.add(next(prelude_names))
            if len(calls) < len(runner.HISTORICAL_PRELUDE_MIGRATION_NAMES):
                raise runner.PendingMigrationContentIntegrityError(
                    "next exact historical prelude remains"
                )
        elif only == (runner.MIGRATION_NAME,):
            state.recorded_migrations.add(runner.MIGRATION_NAME)
        else:  # pragma: no cover - regression guard for this controlled runner
            raise AssertionError(f"unexpected migration selection: {only}")

    args = runner._parse_args(["--apply"])
    result = asyncio.run(
        runner._run(
            args,
            create_pool=create_pool,
            run_migrations_fn=run_migrations,
        )
    )

    assert calls == [
        runner.HISTORICAL_PRELUDE_MIGRATION_NAMES,
        runner.HISTORICAL_PRELUDE_MIGRATION_NAMES,
        runner.HISTORICAL_PRELUDE_MIGRATION_NAMES,
        (runner.MIGRATION_NAME,),
    ]
    assert result["historical_prelude_migrations"] == {
        name: True for name in runner.HISTORICAL_PRELUDE_MIGRATION_NAMES
    }
    assert result["migration_recorded"] is True
    assert pool.closed is True


def test_privilege_runner_reraises_non_prelude_progress_stop(
    monkeypatch,
) -> None:
    """Only a newly recorded authorized prelude may absorb the stop."""

    runner = _load_runner_module()
    monkeypatch.setenv(runner.DBA_DSN_ENV, "postgresql://example.test/atlas")
    state = SimpleNamespace(
        executor_is_superuser=True,
        migrations_table_exists=True,
        recorded_migrations={runner.PREREQUISITE_MIGRATION_NAME},
    )
    pool = _Pool(state)
    calls: list[tuple[str, ...]] = []

    async def create_pool(_database_url: str, **_kwargs: object) -> _Pool:
        return pool

    async def run_migrations(_pool: object, *, only: tuple[str, ...]) -> None:
        calls.append(only)
        assert only == runner.HISTORICAL_PRELUDE_MIGRATION_NAMES
        # A concurrent/incorrectly selected non-prelude receipt is not the
        # committed exact prelude that this controlled retry boundary admits.
        state.recorded_migrations.add(runner.MIGRATION_NAME)
        raise runner.PendingMigrationContentIntegrityError(
            "historical prelude did not advance"
        )

    args = runner._parse_args(["--apply"])
    with pytest.raises(
        runner.PendingMigrationContentIntegrityError,
        match="historical prelude did not advance",
    ):
        asyncio.run(
            runner._run(
                args,
                create_pool=create_pool,
                run_migrations_fn=run_migrations,
            )
        )

    assert calls == [runner.HISTORICAL_PRELUDE_MIGRATION_NAMES]
    assert runner.MIGRATION_NAME in state.recorded_migrations
    assert pool.closed is True


def test_create_pool_binds_the_attested_runtime_schema(monkeypatch) -> None:
    """The controlled DBA pool must not depend on its DSN default search path."""

    runner = _load_runner_module()
    calls: list[dict[str, object]] = []
    created_pool = object()

    async def create_pool(**kwargs: object) -> object:
        calls.append(dict(kwargs))
        return created_pool

    monkeypatch.setitem(
        sys.modules, "asyncpg", SimpleNamespace(create_pool=create_pool)
    )

    runtime_pool = asyncio.run(runner._create_pool(TEST_RUNTIME_DSN))
    dba_pool = asyncio.run(
        runner._create_pool(TEST_DBA_DSN, schema_name="eom_canonical")
    )

    assert runtime_pool is created_pool
    assert dba_pool is created_pool
    assert calls == [
        {
            "dsn": TEST_RUNTIME_DSN,
            "min_size": 1,
            "max_size": 1,
            "statement_cache_size": 0,
        },
        {
            "dsn": TEST_DBA_DSN,
            "min_size": 1,
            "max_size": 1,
            "statement_cache_size": 0,
            "server_settings": {"search_path": '"eom_canonical", pg_catalog'},
        },
    ]


@pytest.mark.parametrize("env_filename", (".env", ".env.local"))
def test_typed_dba_config_loads_the_canonical_dsn_from_supported_env_file(
    monkeypatch,
    tmp_path: Path,
    env_filename: str,
) -> None:
    runner = _load_runner_module()
    monkeypatch.delenv(runner.DBA_DSN_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / env_filename
    env_file.write_text(f"{runner.DBA_DSN_ENV}={TEST_DBA_DSN}\n", encoding="utf-8")

    config = runner.EOMMissedCallRecoveryDBAConfig()

    assert config.database_url.get_secret_value() == TEST_DBA_DSN
    assert TEST_DBA_DSN not in repr(config)


def test_privilege_runner_rejects_missing_typed_dba_dsn_before_pool(
    monkeypatch,
) -> None:
    runner = _load_runner_module()
    monkeypatch.delenv(runner.DBA_DSN_ENV, raising=False)
    pool_calls: list[str] = []

    async def create_pool(database_url: str, **_kwargs: object) -> _Pool:
        pool_calls.append(database_url)
        raise AssertionError("missing DBA DSN must fail before opening a pool")

    with pytest.raises(
        RuntimeError,
        match=f"Missing protected DBA DSN configuration {runner.DBA_DSN_ENV}",
    ):
        asyncio.run(
            runner._run(
                runner._parse_args([]),
                create_pool=create_pool,
                config_factory=lambda: runner.EOMMissedCallRecoveryDBAConfig(
                    _env_file=None
                ),
            )
        )
    assert pool_calls == []


def test_privilege_runner_attests_matching_runtime_target_before_apply() -> None:
    """A matching live runtime target reaches the existing controlled apply path."""

    runner = _load_runner_module()
    lock_state = SimpleNamespace()
    runtime_pool = _Pool(
        SimpleNamespace(
            schema_name="eom_canonical",
            advisory_lock_state=lock_state,
        )
    )
    dba_state = SimpleNamespace(
        schema_name="eom_canonical",
        executor_is_superuser=True,
        migrations_table_exists=True,
        recorded_migrations={runner.PREREQUISITE_MIGRATION_NAME},
        advisory_lock_state=lock_state,
    )
    dba_pool = _Pool(dba_state)
    pool_calls: list[tuple[str, str | None]] = []
    migration_calls: list[tuple[str, ...]] = []

    async def create_pool(
        database_url: str,
        *,
        schema_name: str | None = None,
    ) -> _Pool:
        pool_calls.append((database_url, schema_name))
        if database_url == TEST_RUNTIME_DSN:
            assert schema_name is None
            return runtime_pool
        assert database_url == TEST_DBA_DSN
        assert schema_name == "eom_canonical"
        return dba_pool

    async def run_migrations(_pool: object, *, only: tuple[str, ...]) -> None:
        migration_calls.append(only)
        if only == (runner.MIGRATION_NAME,):
            dba_state.recorded_migrations.add(runner.MIGRATION_NAME)
        else:
            assert only == runner.HISTORICAL_PRELUDE_MIGRATION_NAMES

    result = asyncio.run(
        runner._run(
            runner._parse_args(["--apply"]),
            create_pool=create_pool,
            run_migrations_fn=run_migrations,
            funnel_config_factory=lambda: SimpleNamespace(
                db_connection_string=TEST_RUNTIME_DSN
            ),
        )
    )

    assert pool_calls == [
        (TEST_RUNTIME_DSN, None),
        (TEST_DBA_DSN, "eom_canonical"),
    ]
    assert migration_calls == [
        runner.HISTORICAL_PRELUDE_MIGRATION_NAMES,
        (runner.MIGRATION_NAME,),
    ]
    assert result["applied"] is True
    assert runtime_pool.closed is True
    assert dba_pool.closed is True


def test_privilege_runner_rejects_non_direct_runtime_login_before_dba_pool() -> None:
    runner = _load_runner_module()
    runtime_pool = _Pool(
        SimpleNamespace(
            schema_name="eom_canonical",
            current_user="other_runtime",
            session_user="atlas_funnel",
        )
    )
    pool_calls: list[tuple[str, str | None]] = []

    async def create_pool(
        database_url: str,
        *,
        schema_name: str | None = None,
    ) -> _Pool:
        pool_calls.append((database_url, schema_name))
        assert database_url == TEST_RUNTIME_DSN
        assert schema_name is None
        return runtime_pool

    with pytest.raises(
        RuntimeError,
        match="runtime connection must use its direct configured login",
    ):
        asyncio.run(
            runner._run(
                runner._parse_args(["--apply"]),
                create_pool=create_pool,
                funnel_config_factory=lambda: SimpleNamespace(
                    db_connection_string=TEST_RUNTIME_DSN
                ),
            )
        )

    assert pool_calls == [(TEST_RUNTIME_DSN, None)]
    assert runtime_pool.closed is True


@pytest.mark.parametrize(
    ("dba_schema_name", "dba_database_name", "dba_database_oid", "error"),
    (
        (
            "other_schema",
            "atlas",
            16_384,
            "did not resolve to the EOM funnel runtime schema",
        ),
        (
            "eom_canonical",
            "other_atlas",
            16_384,
            "does not target the EOM funnel runtime database",
        ),
        (
            "eom_canonical",
            "atlas",
            16_385,
            "does not target the EOM funnel runtime database",
        ),
    ),
)
def test_privilege_runner_rejects_dba_target_mismatch_before_mutation(
    dba_schema_name: str,
    dba_database_name: str,
    dba_database_oid: int,
    error: str,
) -> None:
    runner = _load_runner_module()
    lock_state = SimpleNamespace()
    runtime_pool = _Pool(
        SimpleNamespace(
            schema_name="eom_canonical",
            advisory_lock_state=lock_state,
        )
    )
    dba_state = SimpleNamespace(
        schema_name=dba_schema_name,
        database_name=dba_database_name,
        database_oid=dba_database_oid,
        advisory_lock_state=lock_state,
    )
    dba_pool = _Pool(dba_state)
    migration_calls: list[object] = []

    async def create_pool(
        database_url: str,
        *,
        schema_name: str | None = None,
    ) -> _Pool:
        if database_url == TEST_RUNTIME_DSN:
            assert schema_name is None
            return runtime_pool
        assert database_url == TEST_DBA_DSN
        assert schema_name == "eom_canonical"
        return dba_pool

    async def run_migrations(*args: object, **_kwargs: object) -> None:
        migration_calls.append(args)

    with pytest.raises(RuntimeError, match=error):
        asyncio.run(
            runner._run(
                runner._parse_args(["--apply"]),
                create_pool=create_pool,
                run_migrations_fn=run_migrations,
                funnel_config_factory=lambda: SimpleNamespace(
                    db_connection_string=TEST_RUNTIME_DSN
                ),
            )
        )

    assert getattr(dba_state, "executed", []) == []
    assert migration_calls == []
    assert runtime_pool.closed is True
    assert dba_pool.closed is True


def test_privilege_runner_rejects_same_named_clone_before_mutation() -> None:
    """Name/OID equality alone is not enough when clusters do not share a lock."""

    runner = _load_runner_module()
    runtime_pool = _Pool(SimpleNamespace(schema_name="eom_canonical"))
    dba_state = SimpleNamespace(
        schema_name="eom_canonical",
        executor_is_superuser=True,
        migrations_table_exists=True,
        recorded_migrations={runner.PREREQUISITE_MIGRATION_NAME},
    )
    dba_pool = _Pool(dba_state)
    migration_calls: list[object] = []

    async def create_pool(
        database_url: str,
        *,
        schema_name: str | None = None,
    ) -> _Pool:
        if database_url == TEST_RUNTIME_DSN:
            assert schema_name is None
            return runtime_pool
        assert database_url == TEST_DBA_DSN
        assert schema_name == "eom_canonical"
        return dba_pool

    async def run_migrations(*args: object, **_kwargs: object) -> None:
        migration_calls.append(args)

    with pytest.raises(
        RuntimeError,
        match="does not share the EOM funnel runtime database",
    ):
        asyncio.run(
            runner._run(
                runner._parse_args(["--apply"]),
                create_pool=create_pool,
                run_migrations_fn=run_migrations,
                funnel_config_factory=lambda: SimpleNamespace(
                    db_connection_string=TEST_RUNTIME_DSN
                ),
            )
        )

    assert getattr(dba_state, "executed", []) == []
    assert migration_calls == []
    assert runtime_pool.closed is True
    assert dba_pool.closed is True


def test_privilege_runner_rejects_caller_selected_dsn_environment_name() -> None:
    runner = _load_runner_module()

    with pytest.raises(SystemExit):
        runner._parse_args(["--database-url-env", "TEST_EOM_DBA_DSN"])
