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
TERMS_SCRIPT = ROOT / "scripts" / "apply_eom_terms_authority_schema.py"
TERMS_ACCEPTANCE_SCRIPT = ROOT / "scripts" / "apply_eom_terms_acceptance_schema.py"
CARD_VAULT_SCRIPT = ROOT / "scripts" / "apply_eom_card_vault_schema.py"


def _load_runner_module() -> ModuleType:
    module_name = "test_eom_first_clean_completion_dba_runner_script"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_terms_runner_module() -> ModuleType:
    module_name = "test_eom_terms_authority_dba_runner_script"
    spec = importlib.util.spec_from_file_location(module_name, TERMS_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(TERMS_SCRIPT.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _load_terms_acceptance_runner_module() -> ModuleType:
    module_name = "test_eom_terms_acceptance_dba_runner_script"
    spec = importlib.util.spec_from_file_location(module_name, TERMS_ACCEPTANCE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(TERMS_ACCEPTANCE_SCRIPT.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _load_card_vault_runner_module() -> ModuleType:
    module_name = "test_eom_card_vault_dba_runner_script"
    spec = importlib.util.spec_from_file_location(module_name, CARD_VAULT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(CARD_VAULT_SCRIPT.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class _Acquire:
    def __init__(self, connection: "_Connection") -> None:
        self._connection = connection

    async def __aenter__(self) -> "_Connection":
        return self._connection

    async def __aexit__(self, *_args: object) -> None:
        return None


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
            return getattr(
                self._state,
                "advisory_lock_available",
                not hasattr(self._state, "executor_is_superuser"),
            )
        if "pg_try_advisory_lock" in query:
            self._state.migration_lock_attempts = (
                getattr(self._state, "migration_lock_attempts", 0) + 1
            )
            results = getattr(self._state, "migration_lock_results", None)
            if results:
                acquired = bool(results.pop(0))
            else:
                acquired = True
            if acquired:
                self._state.migration_lock_depth = (
                    getattr(self._state, "migration_lock_depth", 0) + 1
                )
            return acquired
        if "pg_advisory_unlock" in query:
            self._state.migration_lock_unlocks = (
                getattr(self._state, "migration_lock_unlocks", 0) + 1
            )
            depth = getattr(self._state, "migration_lock_depth", 0)
            if depth <= 0:
                return False
            self._state.migration_lock_depth = depth - 1
            return True
        if "rolsuper" in query:
            return self._state.executor_is_superuser
        if "to_regclass" in query:
            return self._state.migrations_table_exists
        if "schema_migrations" in query:
            migration_records = getattr(self._state, "migration_records", None)
            if migration_records is not None:
                assert len(args) == 1
                return bool(migration_records.get(str(args[0]), False))
            return self._state.migration_recorded
        raise AssertionError(f"unexpected query: {query}")

    async def fetchrow(self, query: str, *_args: object) -> object:
        if "current_schema() AS schema_name" in query:
            return {
                "schema_name": self._state.schema_name,
                "database_name": getattr(self._state, "database_name", "atlas"),
                "database_oid": getattr(self._state, "database_oid", 16_384),
                "current_user": getattr(self._state, "current_user", "atlas"),
                "session_user": getattr(self._state, "session_user", "atlas"),
            }
        raise AssertionError(f"unexpected query: {query}")


class _Pool:
    def __init__(self, state: SimpleNamespace) -> None:
        self._connection = _Connection(state)
        self.closed = False

    def acquire(self) -> _Acquire:
        return _Acquire(self._connection)

    async def close(self) -> None:
        self.closed = True


def test_create_pool_disables_statement_cache_for_transaction_pooling(
    monkeypatch,
) -> None:
    """Every controlled pool must avoid backend-specific prepared statements."""

    runner = _load_runner_module()
    calls: list[dict[str, object]] = []
    created_pool = object()

    async def create_pool(**kwargs: object) -> object:
        calls.append(dict(kwargs))
        return created_pool

    monkeypatch.setitem(sys.modules, "asyncpg", SimpleNamespace(create_pool=create_pool))

    runtime_pool = asyncio.run(runner._create_pool("postgresql://runtime/test"))
    dba_pool = asyncio.run(
        runner._create_pool("postgresql://dba/test", schema_name="eom_canonical")
    )

    assert runtime_pool is created_pool
    assert dba_pool is created_pool
    assert calls == [
        {
            "dsn": "postgresql://runtime/test",
            "min_size": 1,
            "max_size": 1,
            "statement_cache_size": 0,
        },
        {
            "dsn": "postgresql://dba/test",
            "min_size": 1,
            "max_size": 1,
            "statement_cache_size": 0,
            "server_settings": {"search_path": '"eom_canonical", pg_catalog'},
        },
    ]


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
    assert runtime_pool._connection._state.advisory_lock_attempts == 1
    assert dba_pool._connection._state.advisory_lock_attempts == 1


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


@pytest.mark.parametrize(
    "migration_name",
    [
        "394_eom_first_clean_completion_receipts",
        "396_eom_terms_authority",
        "397_eom_terms_acceptance",
        "398_eom_card_vault",
    ],
)
def test_runner_applies_only_its_named_migration_in_pinned_transaction(
    monkeypatch,
    migration_name: str,
) -> None:
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
        migration_lock_results=[False, True],
    )
    if migration_name in {"397_eom_terms_acceptance", "398_eom_card_vault"}:
        state.migration_records = {
            "395_eom_post_clean_onboarding_candidates": True,
            "396_eom_terms_authority": True,
            "397_eom_terms_acceptance": migration_name == "398_eom_card_vault",
            "398_eom_card_vault": False,
        }
    runtime_pool = _Pool(SimpleNamespace(schema_name=schema_name))
    dba_pool = _Pool(state)
    calls: list[tuple[object, tuple[str, ...]]] = []
    sleep_delays: list[float] = []
    schema_attestation_calls: list[object] = []

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
        assert await observed_pool.acquire() is dba_pool._connection
        assert state.transaction_depth == 1
        await observed_pool.release(dba_pool._connection)
        state.migration_recorded = True
        if getattr(state, "migration_records", None) is not None:
            state.migration_records[migration_name] = True

    async def sleep(delay: float) -> None:
        sleep_delays.append(delay)
        assert state.transaction_depth == 0

    async def card_vault_schema_ready(pool: object) -> bool:
        schema_attestation_calls.append(pool)
        return bool(
            getattr(state, "migration_records", {}).get(
                runner.CARD_VAULT_MIGRATION_NAME,
                False,
            )
        )

    monkeypatch.setattr(runner.asyncio, "sleep", sleep)

    args = runner._parse_args(["--apply", "--migration", migration_name])
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
            card_vault_schema_ready_fn=card_vault_schema_ready,
        )
    )

    assert [only for _pool, only in calls] == [(migration_name,)]
    assert result["migration_recorded"] is True
    assert result["applied"] is True
    assert runtime_pool.closed is True
    assert dba_pool.closed is True
    assert runtime_pool._connection._state.advisory_lock_attempts == 2
    assert dba_pool._connection._state.advisory_lock_attempts == 2
    assert state.migration_lock_attempts == 2
    assert state.migration_lock_unlocks == 1
    assert state.migration_lock_depth == 0
    assert sleep_delays == [0.2]
    if migration_name == runner.CARD_VAULT_MIGRATION_NAME:
        assert result["schema_ready"] is True
        assert schema_attestation_calls == [runtime_pool]
    else:
        assert "schema_ready" not in result
        assert schema_attestation_calls == []


def test_terms_entrypoint_pins_migration_396() -> None:
    terms_runner = _load_terms_runner_module()

    argv = terms_runner._terms_argv(["--apply", "--json"])
    assert argv == [
        "--migration",
        "396_eom_terms_authority",
        "--apply",
        "--json",
    ]
    shared_runner = sys.modules[terms_runner._main.__module__]
    assert shared_runner._parse_args(argv).migration == "396_eom_terms_authority"
    with pytest.raises(SystemExit):
        terms_runner._terms_argv(
            ["--migration=394_eom_first_clean_completion_receipts"]
        )


def test_terms_acceptance_entrypoint_pins_migration_397() -> None:
    terms_runner = _load_terms_acceptance_runner_module()

    argv = terms_runner._terms_acceptance_argv(["--apply", "--json"])
    assert argv == [
        "--migration",
        "397_eom_terms_acceptance",
        "--apply",
        "--json",
    ]
    shared_runner = sys.modules[terms_runner._main.__module__]
    assert shared_runner._parse_args(argv).migration == "397_eom_terms_acceptance"
    with pytest.raises(SystemExit):
        terms_runner._terms_acceptance_argv(["--migration=396_eom_terms_authority"])


def test_card_vault_entrypoint_pins_migration_398() -> None:
    card_vault_runner = _load_card_vault_runner_module()

    argv = card_vault_runner._card_vault_argv(["--apply", "--json"])
    assert argv == ["--migration", "398_eom_card_vault", "--apply", "--json"]
    shared_runner = sys.modules[card_vault_runner._main.__module__]
    assert shared_runner._parse_args(argv).migration == "398_eom_card_vault"
    with pytest.raises(SystemExit):
        card_vault_runner._card_vault_argv(["--migration=397_eom_terms_acceptance"])


@pytest.mark.parametrize(
    ("migration_recorded", "schema_ready"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_card_vault_migration_bookkeeping_matches_runtime_schema_attestation(
    monkeypatch,
    migration_recorded: bool,
    schema_ready: bool,
) -> None:
    runner = _load_runner_module()
    runtime_database_url = "postgresql://runtime@example.test/atlas"
    schema_name = "eom_canonical"
    monkeypatch.setenv(runner.DBA_DSN_ENV, "postgresql://example.test/atlas")
    monkeypatch.setenv(runner.DBA_SCHEMA_ENV, schema_name)
    runtime_pool = _Pool(SimpleNamespace(schema_name=schema_name))
    dba_pool = _Pool(
        SimpleNamespace(
            executor_is_superuser=True,
            migrations_table_exists=True,
            migration_recorded=migration_recorded,
            migration_records={
                "395_eom_post_clean_onboarding_candidates": True,
                "397_eom_terms_acceptance": True,
                "398_eom_card_vault": migration_recorded,
            },
            schema_name=schema_name,
        )
    )
    attested_pools: list[object] = []

    async def create_pool(
        database_url: str, *, schema_name: str | None = None
    ) -> _Pool:
        if database_url == runtime_database_url:
            return runtime_pool
        return dba_pool

    async def attest_schema(pool: object) -> bool:
        attested_pools.append(pool)
        return schema_ready

    args = runner._parse_args(["--migration", runner.CARD_VAULT_MIGRATION_NAME])
    if migration_recorded != schema_ready:
        with pytest.raises(
            RuntimeError,
            match="bookkeeping does not match the runtime-role schema attestation",
        ):
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
                    card_vault_schema_ready_fn=attest_schema,
                )
            )
    else:
        result = asyncio.run(
            runner._run(
                args,
                create_pool=create_pool,
                config_factory=lambda: runner.EOMFirstCleanCompletionDBAConfig(
                    _env_file=None
                ),
                funnel_config_factory=lambda: SimpleNamespace(
                    db_connection_string=runtime_database_url
                ),
                card_vault_schema_ready_fn=attest_schema,
            )
        )
        assert result["migration_recorded"] is migration_recorded
        assert result["schema_ready"] is schema_ready

    assert attested_pools == [runtime_pool]
    assert runtime_pool.closed is True
    assert dba_pool.closed is True


@pytest.mark.parametrize(
    "missing_predecessor",
    [
        "395_eom_post_clean_onboarding_candidates",
        "397_eom_terms_acceptance",
    ],
)
def test_card_vault_refuses_any_unrecorded_predecessor(
    monkeypatch,
    missing_predecessor: str,
) -> None:
    runner = _load_runner_module()
    runtime_database_url = "postgresql://runtime@example.test/atlas"
    schema_name = "eom_canonical"
    monkeypatch.setenv(runner.DBA_DSN_ENV, "postgresql://example.test/atlas")
    monkeypatch.setenv(runner.DBA_SCHEMA_ENV, schema_name)
    state = SimpleNamespace(
        executor_is_superuser=True,
        migrations_table_exists=True,
        migration_recorded=False,
        migration_records={
            "395_eom_post_clean_onboarding_candidates": True,
            "397_eom_terms_acceptance": True,
            "398_eom_card_vault": False,
        },
        schema_name=schema_name,
    )
    runtime_pool = _Pool(SimpleNamespace(schema_name=schema_name))
    dba_pool = _Pool(state)
    state.migration_records[missing_predecessor] = False

    async def create_pool(
        database_url: str, *, schema_name: str | None = None
    ) -> _Pool:
        if database_url == runtime_database_url:
            return runtime_pool
        return dba_pool

    args = runner._parse_args(["--migration", "398_eom_card_vault"])
    with pytest.raises(
        RuntimeError,
        match=f"Controlled predecessor {missing_predecessor} must be recorded",
    ):
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


def test_terms_acceptance_refuses_an_unrecorded_authority_predecessor(
    monkeypatch,
) -> None:
    runner = _load_runner_module()
    runtime_database_url = "postgresql://runtime@example.test/atlas"
    schema_name = "eom_canonical"
    monkeypatch.setenv(runner.DBA_DSN_ENV, "postgresql://example.test/atlas")
    monkeypatch.setenv(runner.DBA_SCHEMA_ENV, schema_name)
    state = SimpleNamespace(
        executor_is_superuser=True,
        migrations_table_exists=True,
        migration_recorded=False,
        migration_records={
            "396_eom_terms_authority": False,
            "397_eom_terms_acceptance": False,
        },
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

    args = runner._parse_args(["--migration", "397_eom_terms_acceptance"])
    with pytest.raises(
        RuntimeError,
        match="Controlled predecessor 396_eom_terms_authority must be recorded",
    ):
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


@pytest.mark.parametrize(
    ("current_user", "session_user"),
    (("postgres", "postgres"), ("atlas", "postgres")),
)
def test_runner_rejects_non_direct_atlas_runtime_before_opening_dba_pool(
    monkeypatch,
    current_user: str,
    session_user: str,
) -> None:
    """A matching schema/database is not enough without the route's login."""

    runner = _load_runner_module()
    runtime_database_url = "postgresql://runtime@example.test/atlas"
    monkeypatch.setenv(runner.DBA_DSN_ENV, "postgresql://example.test/atlas")
    monkeypatch.setenv(runner.DBA_SCHEMA_ENV, "eom_canonical")
    runtime_pool = _Pool(
        SimpleNamespace(
            schema_name="eom_canonical",
            current_user=current_user,
            session_user=session_user,
        )
    )
    pool_calls: list[tuple[str, str | None]] = []
    migration_calls: list[object] = []

    async def create_pool(
        database_url: str, *, schema_name: str | None = None
    ) -> _Pool:
        pool_calls.append((database_url, schema_name))
        if database_url == runtime_database_url:
            assert schema_name is None
            return runtime_pool
        raise AssertionError("invalid runtime identity must not open the DBA pool")

    async def run_migrations(*args: object, **_kwargs: object) -> None:
        migration_calls.append(args)

    with pytest.raises(
        RuntimeError,
        match="runtime connection must use the direct atlas login",
    ):
        asyncio.run(
            runner._run(
                runner._parse_args(["--apply"]),
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

    assert pool_calls == [(runtime_database_url, None)]
    assert migration_calls == []
    assert runtime_pool.closed is True


@pytest.mark.parametrize(
    ("identity_field", "dba_value"),
    (
        ("database_name", "other_atlas"),
        ("database_oid", 16_385),
    ),
)
def test_runner_rejects_dba_database_identity_mismatch_before_migration(
    monkeypatch,
    identity_field: str,
    dba_value: object,
) -> None:
    """Schema equality alone cannot admit a similarly named clone target."""

    runner = _load_runner_module()
    runtime_database_url = "postgresql://runtime@example.test/atlas"
    monkeypatch.setenv(runner.DBA_DSN_ENV, "postgresql://example.test/atlas")
    monkeypatch.setenv(runner.DBA_SCHEMA_ENV, "eom_canonical")
    runtime_pool = _Pool(
        SimpleNamespace(
            schema_name="eom_canonical",
            database_name="atlas",
            database_oid=16_384,
        )
    )
    dba_target = {
        "schema_name": "eom_canonical",
        "database_name": "atlas",
        "database_oid": 16_384,
        "executor_is_superuser": True,
        "migrations_table_exists": True,
        "migration_recorded": False,
    }
    dba_target[identity_field] = dba_value
    dba_pool = _Pool(SimpleNamespace(**dba_target))
    calls: list[tuple[str, str | None]] = []
    migration_calls: list[object] = []

    async def create_pool(
        database_url: str, *, schema_name: str | None = None
    ) -> _Pool:
        calls.append((database_url, schema_name))
        if database_url == runtime_database_url:
            assert schema_name is None
            return runtime_pool
        assert schema_name == "eom_canonical"
        return dba_pool

    async def run_migrations(*args: object, **_kwargs: object) -> None:
        migration_calls.append(args)

    with pytest.raises(
        RuntimeError,
        match="does not target the EOM funnel runtime database",
    ):
        asyncio.run(
            runner._run(
                runner._parse_args(["--apply"]),
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

    assert calls == [
        (runtime_database_url, None),
        ("postgresql://example.test/atlas", "eom_canonical"),
    ]
    assert migration_calls == []
    assert runtime_pool.closed is True
    assert dba_pool.closed is True


def test_runner_rejects_same_named_socket_clone_without_shared_lock(
    monkeypatch,
) -> None:
    """A Unix-socket target needs live cross-connection, not endpoint, proof."""

    runner = _load_runner_module()
    runtime_database_url = "postgresql://runtime@example.test/atlas"
    monkeypatch.setenv(runner.DBA_DSN_ENV, "postgresql://example.test/atlas")
    monkeypatch.setenv(runner.DBA_SCHEMA_ENV, "eom_canonical")
    runtime_state = SimpleNamespace(
        schema_name="eom_canonical",
        database_name="atlas",
        database_oid=16_384,
        advisory_lock_available=True,
    )
    dba_state = SimpleNamespace(
        schema_name="eom_canonical",
        database_name="atlas",
        database_oid=16_384,
        executor_is_superuser=True,
        migrations_table_exists=True,
        migration_recorded=False,
        advisory_lock_available=True,
    )
    runtime_pool = _Pool(runtime_state)
    dba_pool = _Pool(dba_state)
    migration_calls: list[object] = []

    async def create_pool(
        database_url: str, *, schema_name: str | None = None
    ) -> _Pool:
        if database_url == runtime_database_url:
            assert schema_name is None
            return runtime_pool
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
                config_factory=lambda: runner.EOMFirstCleanCompletionDBAConfig(
                    _env_file=None
                ),
                funnel_config_factory=lambda: SimpleNamespace(
                    db_connection_string=runtime_database_url
                ),
            )
        )

    assert migration_calls == []
    assert runtime_state.advisory_lock_attempts == 1
    assert dba_state.advisory_lock_attempts == 1
    assert runtime_pool.closed is True
    assert dba_pool.closed is True


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


def test_runner_rejects_missing_runtime_database_identity_before_opening_dba_pool(
    monkeypatch,
) -> None:
    """A missing catalog identity must not degrade into schema-only matching."""

    runner = _load_runner_module()
    runtime_database_url = "postgresql://runtime@example.test/atlas"
    monkeypatch.setenv(runner.DBA_DSN_ENV, "postgresql://example.test/atlas")
    monkeypatch.setenv(runner.DBA_SCHEMA_ENV, "eom_canonical")
    runtime_pool = _Pool(
        SimpleNamespace(
            schema_name="eom_canonical",
            database_name="atlas",
            database_oid=None,
            server_address="127.0.0.1",
            server_port=5432,
        )
    )
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
        match="Missing or invalid database identity from EOM funnel runtime",
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
