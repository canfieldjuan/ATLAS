"""Unit proof for the controlled first-clean completion DBA entrypoint."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "apply_eom_first_clean_completion_schema.py"


def _load_runner_module() -> ModuleType:
    module_name = "test_eom_first_clean_completion_dba_runner_script"
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

    async def fetchval(self, query: str, *_args: object) -> bool:
        if "rolsuper" in query:
            return self._state.executor_is_superuser
        if "to_regclass" in query:
            return self._state.migrations_table_exists
        if "schema_migrations" in query:
            return self._state.migration_recorded
        raise AssertionError(f"unexpected query: {query}")


class _Pool:
    def __init__(self, state: SimpleNamespace) -> None:
        self._connection = _Connection(state)
        self.closed = False

    def acquire(self) -> _Acquire:
        return _Acquire(self._connection)

    async def close(self) -> None:
        self.closed = True


def test_runner_defaults_to_read_only_and_redacts_dsn(monkeypatch) -> None:
    runner = _load_runner_module()
    monkeypatch.setenv(
        runner.DBA_DSN_ENV,
        "postgresql://operator:secret@example.test:5432/atlas?sslmode=require",
    )
    state = SimpleNamespace(
        executor_is_superuser=True,
        migrations_table_exists=True,
        migration_recorded=False,
    )
    pool = _Pool(state)
    calls: list[object] = []

    async def create_pool(_database_url: str) -> _Pool:
        return pool

    async def run_migrations(*_args: object, **_kwargs: object) -> None:
        calls.append(object())

    args = runner._parse_args([])
    result = asyncio.run(
        runner._run(
            args,
            create_pool=create_pool,
            run_migrations_fn=run_migrations,
            config_factory=lambda: runner.EOMFirstCleanCompletionDBAConfig(
                _env_file=None
            ),
        )
    )

    assert result == {
        "target": "dsn=example.test:5432/atlas",
        "executor_is_superuser": True,
        "migration": runner.MIGRATION_NAME,
        "migration_recorded": False,
        "applied": False,
    }
    assert calls == []
    assert pool.closed is True


def test_runner_rejects_non_superuser_before_apply(monkeypatch) -> None:
    runner = _load_runner_module()
    monkeypatch.setenv(runner.DBA_DSN_ENV, "postgresql://example.test/atlas")
    state = SimpleNamespace(
        executor_is_superuser=False,
        migrations_table_exists=True,
        migration_recorded=False,
    )
    pool = _Pool(state)

    async def create_pool(_database_url: str) -> _Pool:
        return pool

    args = runner._parse_args(["--apply"])
    with pytest.raises(RuntimeError, match="not a PostgreSQL superuser"):
        asyncio.run(
            runner._run(
                args,
                create_pool=create_pool,
                config_factory=lambda: runner.EOMFirstCleanCompletionDBAConfig(
                    _env_file=None
                ),
            )
        )
    assert pool.closed is True


def test_runner_applies_only_its_named_migration(monkeypatch) -> None:
    runner = _load_runner_module()
    monkeypatch.setenv(runner.DBA_DSN_ENV, "postgresql://example.test/atlas")
    state = SimpleNamespace(
        executor_is_superuser=True,
        migrations_table_exists=True,
        migration_recorded=False,
    )
    pool = _Pool(state)
    calls: list[tuple[object, tuple[str, ...]]] = []

    async def create_pool(_database_url: str) -> _Pool:
        return pool

    async def run_migrations(observed_pool: object, *, only: tuple[str, ...]) -> None:
        calls.append((observed_pool, only))
        state.migration_recorded = True

    args = runner._parse_args(["--apply"])
    result = asyncio.run(
        runner._run(
            args,
            create_pool=create_pool,
            run_migrations_fn=run_migrations,
            config_factory=lambda: runner.EOMFirstCleanCompletionDBAConfig(
                _env_file=None
            ),
        )
    )

    assert calls == [(pool, (runner.MIGRATION_NAME,))]
    assert result["migration_recorded"] is True
    assert result["applied"] is True
    assert pool.closed is True


def test_runner_rejects_missing_typed_dsn_before_pool(monkeypatch) -> None:
    runner = _load_runner_module()
    monkeypatch.delenv(runner.DBA_DSN_ENV, raising=False)
    pool_calls: list[str] = []

    async def create_pool(database_url: str) -> _Pool:
        pool_calls.append(database_url)
        return _Pool(
            SimpleNamespace(
                executor_is_superuser=True,
                migrations_table_exists=True,
                migration_recorded=False,
            )
        )

    with pytest.raises(
        RuntimeError, match=f"Missing protected DBA DSN configuration {runner.DBA_DSN_ENV}"
    ):
        asyncio.run(
            runner._run(
                runner._parse_args([]),
                create_pool=create_pool,
                config_factory=lambda: runner.EOMFirstCleanCompletionDBAConfig(
                    _env_file=None
                ),
            )
        )
    assert pool_calls == []


def test_runner_rejects_caller_selected_dsn_environment_name() -> None:
    runner = _load_runner_module()

    with pytest.raises(SystemExit):
        runner._parse_args(["--database-url-env", "TEST_EOM_DBA_DSN"])
