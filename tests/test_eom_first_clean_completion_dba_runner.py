"""Unit proof for the controlled first-clean completion DBA entrypoint."""

from __future__ import annotations

import asyncio
import importlib.util
from itertools import product
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
from pydantic import ValidationError


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

    async def fetchval(self, query: str, *_args: object) -> object:
        if "current_schema()" in query:
            return self._state.schema_name
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
    runtime_database_url = "postgresql://runtime@example.test/atlas"
    schema_name = "eom_canonical"
    monkeypatch.setenv(
        runner.DBA_DSN_ENV,
        "postgresql://operator:secret@example.test:5432/atlas?sslmode=require",
    )
    monkeypatch.setenv(runner.DBA_SCHEMA_ENV, schema_name)
    state = SimpleNamespace(
        executor_is_superuser=True,
        migrations_table_exists=True,
        migration_recorded=False,
        schema_name=schema_name,
    )
    runtime_pool = _Pool(SimpleNamespace(schema_name=schema_name))
    dba_pool = _Pool(state)
    calls: list[object] = []

    async def create_pool(
        database_url: str, *, schema_name: str | None = None
    ) -> _Pool:
        calls.append((database_url, schema_name))
        if database_url == runtime_database_url:
            assert schema_name is None
            return runtime_pool
        assert database_url.startswith("postgresql://operator:secret@example.test")
        assert schema_name == "eom_canonical"
        return dba_pool

    async def run_migrations(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("read-only preflight must not apply migration 394")

    args = runner._parse_args([])
    result = asyncio.run(
        runner._run(
            args,
            create_pool=create_pool,
            run_migrations_fn=run_migrations,
            config_factory=lambda: runner.EOMFirstCleanCompletionDBAConfig(
                _env_file=None
            ),
            funnel_config_factory=lambda: SimpleNamespace(
                db_connection_string=runtime_database_url
            ),
        )
    )

    assert result == {
        "target": "dsn=example.test:5432/atlas;schema=eom_canonical",
        "executor_is_superuser": True,
        "migration": runner.MIGRATION_NAME,
        "migration_recorded": False,
        "applied": False,
    }
    assert calls == [
        (runtime_database_url, None),
        (
            "postgresql://operator:secret@example.test:5432/atlas?sslmode=require",
            "eom_canonical",
        ),
    ]
    assert runtime_pool.closed is True
    assert dba_pool.closed is True


def test_runner_rejects_non_superuser_before_apply(monkeypatch) -> None:
    runner = _load_runner_module()
    runtime_database_url = "postgresql://runtime@example.test/atlas"
    schema_name = "eom_canonical"
    monkeypatch.setenv(runner.DBA_DSN_ENV, "postgresql://example.test/atlas")
    monkeypatch.setenv(runner.DBA_SCHEMA_ENV, schema_name)
    state = SimpleNamespace(
        executor_is_superuser=False,
        migrations_table_exists=True,
        migration_recorded=False,
        schema_name=schema_name,
    )
    runtime_pool = _Pool(SimpleNamespace(schema_name=schema_name))
    dba_pool = _Pool(state)

    async def create_pool(
        database_url: str, *, schema_name: str | None = None
    ) -> _Pool:
        if database_url == runtime_database_url:
            assert schema_name is None
            return runtime_pool
        assert schema_name == "eom_canonical"
        return dba_pool

    args = runner._parse_args(["--apply"])
    with pytest.raises(RuntimeError, match="not a PostgreSQL superuser"):
        asyncio.run(
            runner._run(
                args,
                create_pool=create_pool,
                config_factory=lambda: runner.EOMFirstCleanCompletionDBAConfig(
                    _env_file=None
                ),
                funnel_config_factory=lambda: SimpleNamespace(
                    db_connection_string=runtime_database_url
                ),
            )
        )
    assert runtime_pool.closed is True
    assert dba_pool.closed is True


def test_runner_applies_only_its_named_migration(monkeypatch) -> None:
    runner = _load_runner_module()
    runtime_database_url = "postgresql://runtime@example.test/atlas"
    schema_name = "eom_canonical"
    monkeypatch.setenv(runner.DBA_DSN_ENV, "postgresql://example.test/atlas")
    monkeypatch.setenv(runner.DBA_SCHEMA_ENV, schema_name)
    state = SimpleNamespace(
        executor_is_superuser=True,
        migrations_table_exists=True,
        migration_recorded=False,
        schema_name=schema_name,
    )
    runtime_pool = _Pool(SimpleNamespace(schema_name=schema_name))
    dba_pool = _Pool(state)
    calls: list[tuple[object, tuple[str, ...]]] = []

    async def create_pool(
        database_url: str, *, schema_name: str | None = None
    ) -> _Pool:
        if database_url == runtime_database_url:
            assert schema_name is None
            return runtime_pool
        assert schema_name == "eom_canonical"
        return dba_pool

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
            funnel_config_factory=lambda: SimpleNamespace(
                db_connection_string=runtime_database_url
            ),
        )
    )

    assert calls == [(dba_pool, (runner.MIGRATION_NAME,))]
    assert result["migration_recorded"] is True
    assert result["applied"] is True
    assert runtime_pool.closed is True
    assert dba_pool.closed is True


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
        RuntimeError,
        match=f"Missing protected DBA DSN configuration {runner.DBA_DSN_ENV}",
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


def test_runner_rejects_missing_typed_schema_before_pool(monkeypatch) -> None:
    runner = _load_runner_module()
    monkeypatch.setenv(runner.DBA_DSN_ENV, "postgresql://example.test/atlas")
    monkeypatch.delenv(runner.DBA_SCHEMA_ENV, raising=False)
    pool_calls: list[str] = []

    async def create_pool(database_url: str, **_kwargs: object) -> _Pool:
        pool_calls.append(database_url)
        raise AssertionError("missing schema must fail before opening a pool")

    with pytest.raises(
        RuntimeError,
        match=f"Missing or invalid schema from {runner.DBA_SCHEMA_ENV}",
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


def test_typed_dba_schema_rejects_non_identifier_configuration(monkeypatch) -> None:
    runner = _load_runner_module()
    monkeypatch.setenv(runner.DBA_SCHEMA_ENV, "eom;drop schema public")

    with pytest.raises(ValidationError, match="one ASCII PostgreSQL identifier"):
        runner.EOMFirstCleanCompletionDBAConfig(_env_file=None)


def test_eom_first_clean_completion_dba_config_schema_grammar_property() -> None:
    """The typed schema config admits exactly the identifier grammar invariant."""

    from atlas_brain.config import EOMFirstCleanCompletionDBAConfig

    valid_tokens = ("public", "tenant", "_private")
    valid_wrappers = ("", " ", "\t")
    valid_families = ("", "_1", "9")
    for token, wrapper, family in product(
        valid_tokens,
        valid_wrappers,
        valid_families,
    ):
        expected_schema = f"{token}{family}"
        assert (
            EOMFirstCleanCompletionDBAConfig.validate_schema_name(
                f"{wrapper}{expected_schema}{wrapper}"
            )
            == expected_schema
        )

    invalid_tokens = ("-", ".", ";", '"', "\n", "λ")
    invalid_wrappers = ("", " ", "\t")
    invalid_families = ("public", "tenant", "_private")
    for token, wrapper, family in product(
        invalid_tokens,
        invalid_wrappers,
        invalid_families,
    ):
        with pytest.raises(ValueError):
            EOMFirstCleanCompletionDBAConfig.validate_schema_name(
                f"{wrapper}{family}{token}x{wrapper}"
            )


def test_runner_rejects_runtime_schema_mismatch_before_opening_dba_pool(
    monkeypatch,
) -> None:
    runner = _load_runner_module()
    runtime_database_url = "postgresql://runtime@example.test/atlas"
    monkeypatch.setenv(runner.DBA_DSN_ENV, "postgresql://example.test/atlas")
    monkeypatch.setenv(runner.DBA_SCHEMA_ENV, "eom_canonical")
    runtime_pool = _Pool(SimpleNamespace(schema_name="other_schema"))
    pool_calls: list[tuple[str, str | None]] = []

    async def create_pool(
        database_url: str, *, schema_name: str | None = None
    ) -> _Pool:
        pool_calls.append((database_url, schema_name))
        assert database_url == runtime_database_url
        assert schema_name is None
        return runtime_pool

    with pytest.raises(
        RuntimeError,
        match="does not match the EOM funnel runtime schema",
    ):
        asyncio.run(
            runner._run(
                runner._parse_args([]),
                create_pool=create_pool,
                config_factory=lambda: runner.EOMFirstCleanCompletionDBAConfig(
                    _env_file=None
                ),
                funnel_config_factory=lambda: SimpleNamespace(
                    db_connection_string=runtime_database_url
                ),
            )
        )
    assert pool_calls == [(runtime_database_url, None)]
    assert runtime_pool.closed is True


def test_runner_rejects_absent_runtime_schema_before_opening_dba_pool(
    monkeypatch,
) -> None:
    runner = _load_runner_module()
    runtime_database_url = "postgresql://runtime@example.test/atlas"
    monkeypatch.setenv(runner.DBA_DSN_ENV, "postgresql://example.test/atlas")
    monkeypatch.setenv(runner.DBA_SCHEMA_ENV, "eom_canonical")
    runtime_pool = _Pool(SimpleNamespace(schema_name=None))
    pool_calls: list[tuple[str, str | None]] = []

    async def create_pool(
        database_url: str, *, schema_name: str | None = None
    ) -> _Pool:
        pool_calls.append((database_url, schema_name))
        assert database_url == runtime_database_url
        assert schema_name is None
        return runtime_pool

    with pytest.raises(
        RuntimeError,
        match="Missing or invalid schema from EOM funnel runtime current_schema",
    ):
        asyncio.run(
            runner._run(
                runner._parse_args([]),
                create_pool=create_pool,
                config_factory=lambda: runner.EOMFirstCleanCompletionDBAConfig(
                    _env_file=None
                ),
                funnel_config_factory=lambda: SimpleNamespace(
                    db_connection_string=runtime_database_url
                ),
            )
        )
    assert pool_calls == [(runtime_database_url, None)]
    assert runtime_pool.closed is True


def test_runner_rejects_missing_funnel_runtime_dsn_before_pool(monkeypatch) -> None:
    runner = _load_runner_module()
    monkeypatch.setenv(runner.DBA_DSN_ENV, "postgresql://example.test/atlas")
    monkeypatch.setenv(runner.DBA_SCHEMA_ENV, "eom_canonical")
    pool_calls: list[str] = []

    async def create_pool(database_url: str, **_kwargs: object) -> _Pool:
        pool_calls.append(database_url)
        raise AssertionError("missing runtime DSN must fail before opening a pool")

    with pytest.raises(
        RuntimeError,
        match=f"Missing EOM funnel runtime DSN configuration {runner.FUNNEL_DSN_ENV}",
    ):
        asyncio.run(
            runner._run(
                runner._parse_args([]),
                create_pool=create_pool,
                config_factory=lambda: runner.EOMFirstCleanCompletionDBAConfig(
                    _env_file=None
                ),
                funnel_config_factory=lambda: SimpleNamespace(db_connection_string=""),
            )
        )
    assert pool_calls == []


def test_runner_rejects_dba_pool_that_does_not_hold_the_pinned_schema(
    monkeypatch,
) -> None:
    runner = _load_runner_module()
    runtime_database_url = "postgresql://runtime@example.test/atlas"
    monkeypatch.setenv(runner.DBA_DSN_ENV, "postgresql://example.test/atlas")
    monkeypatch.setenv(runner.DBA_SCHEMA_ENV, "eom_canonical")
    runtime_pool = _Pool(SimpleNamespace(schema_name="eom_canonical"))
    dba_pool = _Pool(
        SimpleNamespace(
            executor_is_superuser=True,
            migrations_table_exists=True,
            migration_recorded=False,
            schema_name="other_schema",
        )
    )

    async def create_pool(
        database_url: str, *, schema_name: str | None = None
    ) -> _Pool:
        if database_url == runtime_database_url:
            assert schema_name is None
            return runtime_pool
        assert schema_name == "eom_canonical"
        return dba_pool

    with pytest.raises(
        RuntimeError,
        match="Controlled DBA pool did not resolve to the configured EOM funnel schema",
    ):
        asyncio.run(
            runner._run(
                runner._parse_args([]),
                create_pool=create_pool,
                config_factory=lambda: runner.EOMFirstCleanCompletionDBAConfig(
                    _env_file=None
                ),
                funnel_config_factory=lambda: SimpleNamespace(
                    db_connection_string=runtime_database_url
                ),
            )
        )
    assert runtime_pool.closed is True
    assert dba_pool.closed is True


def test_runner_rejects_caller_selected_dsn_environment_name() -> None:
    runner = _load_runner_module()

    with pytest.raises(SystemExit):
        runner._parse_args(["--database-url-env", "TEST_EOM_DBA_DSN"])
