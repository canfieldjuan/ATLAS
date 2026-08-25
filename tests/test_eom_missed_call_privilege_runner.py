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


class _Connection:
    def __init__(self, state: SimpleNamespace) -> None:
        self._state = state

    async def fetchval(self, query: str, *args: object) -> bool:
        if "rolsuper" in query:
            return self._state.executor_is_superuser
        if "to_regclass" in query:
            return self._state.migrations_table_exists
        if "schema_migrations" in query:
            assert len(args) == 1
            return args[0] in self._state.recorded_migrations
        raise AssertionError(f"unexpected query: {query}")


class _Pool:
    def __init__(self, state: SimpleNamespace) -> None:
        self._connection = _Connection(state)
        self.closed = False

    def acquire(self) -> _Acquire:
        return _Acquire(self._connection)

    async def close(self) -> None:
        self.closed = True


def test_privilege_runner_defaults_to_read_only_and_redacts_dsn(monkeypatch) -> None:
    runner = _load_runner_module()
    monkeypatch.setenv(
        "TEST_EOM_DBA_DSN",
        "postgresql://operator:secret@example.test:5432/atlas?sslmode=require",
    )
    state = SimpleNamespace(
        executor_is_superuser=True,
        migrations_table_exists=True,
        recorded_migrations=set(),
    )
    pool = _Pool(state)
    calls: list[object] = []

    async def create_pool(_database_url: str) -> _Pool:
        return pool

    async def run_migrations(*_args: object, **_kwargs: object) -> None:
        calls.append(object())

    args = runner._parse_args(["--database-url-env", "TEST_EOM_DBA_DSN"])
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
    monkeypatch.setenv("TEST_EOM_DBA_DSN", "postgresql://example.test/atlas")
    state = SimpleNamespace(
        executor_is_superuser=False,
        migrations_table_exists=True,
        recorded_migrations={runner.PREREQUISITE_MIGRATION_NAME},
    )
    pool = _Pool(state)

    async def create_pool(_database_url: str) -> _Pool:
        return pool

    args = runner._parse_args(
        ["--database-url-env", "TEST_EOM_DBA_DSN", "--apply"]
    )
    with pytest.raises(RuntimeError, match="not a PostgreSQL superuser"):
        asyncio.run(runner._run(args, create_pool=create_pool))
    assert pool.closed is True


def test_privilege_runner_applies_only_the_recorded_repair(monkeypatch) -> None:
    runner = _load_runner_module()
    monkeypatch.setenv("TEST_EOM_DBA_DSN", "postgresql://example.test/atlas")
    state = SimpleNamespace(
        executor_is_superuser=True,
        migrations_table_exists=True,
        recorded_migrations={runner.PREREQUISITE_MIGRATION_NAME},
    )
    pool = _Pool(state)
    calls: list[tuple[object, tuple[str, ...]]] = []

    async def create_pool(_database_url: str) -> _Pool:
        return pool

    async def run_migrations(observed_pool: object, *, only: tuple[str, ...]) -> None:
        calls.append((observed_pool, only))
        if only == (runner.MIGRATION_NAME,):
            state.recorded_migrations.add(runner.MIGRATION_NAME)
        else:
            assert only == runner.HISTORICAL_PRELUDE_MIGRATION_NAMES

    args = runner._parse_args(
        ["--database-url-env", "TEST_EOM_DBA_DSN", "--apply"]
    )
    result = asyncio.run(
        runner._run(
            args,
            create_pool=create_pool,
            run_migrations_fn=run_migrations,
        )
    )

    assert calls == [
        (pool, runner.HISTORICAL_PRELUDE_MIGRATION_NAMES),
        (pool, (runner.MIGRATION_NAME,)),
    ]
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
    monkeypatch.setenv("TEST_EOM_DBA_DSN", "postgresql://example.test/atlas")
    state = SimpleNamespace(
        executor_is_superuser=True,
        migrations_table_exists=migrations_table_exists,
        recorded_migrations=set(recorded_migrations),
    )
    pool = _Pool(state)
    calls: list[tuple[str, ...]] = []

    async def create_pool(_database_url: str) -> _Pool:
        return pool

    async def run_migrations(*_args: object, **kwargs: object) -> None:
        only = kwargs["only"]
        assert isinstance(only, tuple)
        calls.append(only)

    args = runner._parse_args(
        ["--database-url-env", "TEST_EOM_DBA_DSN", "--apply"]
    )
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
        if (
            not migrations_table_exists
            or runner.MIGRATION_NAME in recorded_migrations
        )
        else [runner.HISTORICAL_PRELUDE_MIGRATION_NAMES]
    )
    assert calls == expected_calls
    assert (runner.MIGRATION_NAME,) not in calls
    assert pool.closed is True


def test_privilege_runner_applies_historical_preludes_before_privilege_repair(
    monkeypatch,
) -> None:
    runner = _load_runner_module()
    monkeypatch.setenv("TEST_EOM_DBA_DSN", "postgresql://example.test/atlas")
    state = SimpleNamespace(
        executor_is_superuser=True,
        migrations_table_exists=True,
        recorded_migrations={runner.PREREQUISITE_MIGRATION_NAME},
    )
    pool = _Pool(state)
    calls: list[tuple[str, ...]] = []
    prelude_names = iter(runner.HISTORICAL_PRELUDE_MIGRATION_NAMES)

    async def create_pool(_database_url: str) -> _Pool:
        return pool

    async def run_migrations(_pool: object, *, only: tuple[str, ...]) -> None:
        calls.append(only)
        if only == runner.HISTORICAL_PRELUDE_MIGRATION_NAMES:
            state.recorded_migrations.add(next(prelude_names))
        elif only == (runner.MIGRATION_NAME,):
            state.recorded_migrations.add(runner.MIGRATION_NAME)
        else:  # pragma: no cover - regression guard for this controlled runner
            raise AssertionError(f"unexpected migration selection: {only}")

    args = runner._parse_args(
        ["--database-url-env", "TEST_EOM_DBA_DSN", "--apply"]
    )
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


def test_privilege_runner_continues_after_committed_historical_progress_stop(
    monkeypatch,
) -> None:
    """A committed prelude receipt authorizes retrying the exact selector."""

    runner = _load_runner_module()
    monkeypatch.setenv("TEST_EOM_DBA_DSN", "postgresql://example.test/atlas")
    state = SimpleNamespace(
        executor_is_superuser=True,
        migrations_table_exists=True,
        recorded_migrations={runner.PREREQUISITE_MIGRATION_NAME},
    )
    pool = _Pool(state)
    calls: list[tuple[str, ...]] = []
    prelude_names = iter(runner.HISTORICAL_PRELUDE_MIGRATION_NAMES)

    async def create_pool(_database_url: str) -> _Pool:
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

    args = runner._parse_args(
        ["--database-url-env", "TEST_EOM_DBA_DSN", "--apply"]
    )
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
    monkeypatch.setenv("TEST_EOM_DBA_DSN", "postgresql://example.test/atlas")
    state = SimpleNamespace(
        executor_is_superuser=True,
        migrations_table_exists=True,
        recorded_migrations={runner.PREREQUISITE_MIGRATION_NAME},
    )
    pool = _Pool(state)
    calls: list[tuple[str, ...]] = []

    async def create_pool(_database_url: str) -> _Pool:
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

    args = runner._parse_args(
        ["--database-url-env", "TEST_EOM_DBA_DSN", "--apply"]
    )
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
