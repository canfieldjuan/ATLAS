from __future__ import annotations

import builtins
import os
import re
import runpy
import subprocess
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).parents[1]
OPS = ROOT / "ops"
OPS_MODULE = runpy.run_path(str(OPS), run_name="atlas_ops_contract_test")
DATABASE_INSPECTIONS = OPS_MODULE["DATABASE_INSPECTIONS"]
DATABASE_CONFIG_KEYS = OPS_MODULE["DATABASE_CONFIG_KEYS"]
TEST_DATABASE_URL_KEYS = OPS_MODULE["TEST_DATABASE_URL_KEYS"]
OpsError = OPS_MODULE["OpsError"]
database_inspection_command = OPS_MODULE["database_inspection_command"]
database_env_files = OPS_MODULE["database_env_files"]
database_runtime_environment = OPS_MODULE["database_runtime_environment"]


@pytest.mark.parametrize(
    ("returncode", "common_dir", "expected"),
    (
        (0, "/srv/atlas/.git", Path("/srv/atlas")),
        (0, "/srv/Atlas", Path("/srv/Atlas")),
        (1, "", Path("/workspace/atlas-worktree")),
    ),
)
def test_shared_root_resolves_conventional_bare_and_failure_layouts(
    returncode: int,
    common_dir: str,
    expected: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = Path("/workspace/atlas-worktree")

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, returncode, f"{common_dir}\n", "")

    function_globals = OPS_MODULE["shared_root"].__globals__
    monkeypatch.setitem(function_globals, "repo_root", lambda: repo)
    monkeypatch.setitem(function_globals, "run", fake_run)

    assert OPS_MODULE["shared_root"]() == expected


def test_database_surface_contains_only_fixed_inspections() -> None:
    assert DATABASE_INSPECTIONS == {
        "connectivity": "SELECT 1 AS ok",
        "migrations": (
            "SELECT count(*) AS applied_count, max(version) AS latest_version "
            "FROM public.schema_migrations"
        ),
    }
    for name in DATABASE_INSPECTIONS:
        command = database_inspection_command(name)
        assert command[-2:] == ["__db-inspect", name]
        assert DATABASE_INSPECTIONS[name] not in command


@pytest.mark.parametrize(
    "args",
    (
        ("db", "query", "SELECT pg_terminate_backend(1)"),
        ("db", "query", "SELECT 1"),
        ("db", "inspect", "pg_terminate_backend"),
        ("__db-inspect", "connectivity"),
    ),
)
def test_database_surface_rejects_arbitrary_sql(args: tuple[str, ...]) -> None:
    result = subprocess.run(
        [str(OPS), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert any(
        reason in result.stderr
        for reason in (
            "arbitrary SQL",
            "unknown database inspection",
            "internal database inspection entrypoint is not public",
        )
    )


def test_database_runtime_preserves_exact_dsn_without_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dsn = (
        "postgresql://agent:canary-password@db.example/atlas"
        "?sslmode=require&host=%2Fvar%2Frun%2Fpostgresql"
    )
    env_file = tmp_path / ".env"
    env_file.write_text(f"ATLAS_DB_CONNECTION_STRING={dsn}\n", encoding="utf-8")
    monkeypatch.setenv("ATLAS_OPS_ENV_FILES", str(env_file))
    for key in TEST_DATABASE_URL_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("ATLAS_DB_CONNECTION_STRING", raising=False)

    child_env = database_runtime_environment()
    command = database_inspection_command("connectivity")

    assert child_env["ATLAS_DB_CONNECTION_STRING"] == dsn
    assert all(dsn not in argument for argument in command)


def test_database_runtime_uses_project_dotenv_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atlas_brain.storage.config import DatabaseConfig

    env_file = tmp_path / ".env"
    env_file.write_text(
        "DB_USER=agent\n"
        "DB_HOST=db.example\n"
        'ATLAS_DB_CONNECTION_STRING="postgresql://${DB_USER}:pa\\\"ss@'
        '${DB_HOST}/atlas?sslmode=require" # local database\n'
        'ATLAS_DB_SOCKET_PATH="/var/run/postgresql\\tcluster"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("ATLAS_OPS_ENV_FILES", str(env_file))
    for key in DATABASE_CONFIG_KEYS:
        monkeypatch.delenv(key, raising=False)

    child_env = database_runtime_environment()
    application_config = DatabaseConfig(_env_file=env_file)

    assert child_env["ATLAS_DB_CONNECTION_STRING"] == (
        'postgresql://agent:pa"ss@db.example/atlas?sslmode=require'
    )
    assert child_env["ATLAS_DB_SOCKET_PATH"] == "/var/run/postgresql\tcluster"
    assert (
        child_env["ATLAS_DB_CONNECTION_STRING"]
        == application_config.connection_string
    )
    assert child_env["ATLAS_DB_SOCKET_PATH"] == application_config.socket_path


def test_database_runtime_environment_overrides_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ATLAS_DB_HOST=file.example\n", encoding="utf-8")
    monkeypatch.setenv("ATLAS_OPS_ENV_FILES", str(env_file))
    monkeypatch.setenv("ATLAS_DB_HOST", "runtime.example")

    child_env = database_runtime_environment()

    assert child_env["ATLAS_DB_HOST"] == "runtime.example"


def test_database_runtime_environment_ignores_case_variant_database_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atlas_brain.storage.config import DatabaseConfig

    file_dsn = "postgresql://file.example:5433/atlas"
    shadow_dsn = "postgresql://shadow.example:5433/shadow"
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"ATLAS_DB_CONNECTION_STRING={file_dsn}\n"
        "ATLAS_DB_HOST=file.example\n"
        "ATLAS_DB_CONNECT_TIMEOUT=7.5\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ATLAS_OPS_ENV_FILES", str(env_file))
    monkeypatch.setenv("atlas_db_connection_string", shadow_dsn)
    monkeypatch.setenv("AtLaS_Db_HoSt", "shadow.example")
    monkeypatch.setenv("AtLaS_Db_Connect_Timeout", "0.01")
    for key in DATABASE_CONFIG_KEYS:
        monkeypatch.delenv(key, raising=False)

    child_env = database_runtime_environment()

    assert child_env["ATLAS_DB_CONNECTION_STRING"] == file_dsn
    assert child_env["ATLAS_DB_HOST"] == "file.example"
    assert child_env["ATLAS_DB_CONNECT_TIMEOUT"] == "7.5"
    assert "atlas_db_connection_string" not in child_env
    assert "AtLaS_Db_HoSt" not in child_env
    assert "AtLaS_Db_Connect_Timeout" not in child_env
    with monkeypatch.context() as child_process:
        for key in tuple(os.environ):
            child_process.delenv(key, raising=False)
        for key, value in child_env.items():
            child_process.setenv(key, value)
        application_config = DatabaseConfig(_env_file=None)

    assert application_config.connection_string == file_dsn
    assert application_config.host == "file.example"
    assert application_config.connect_timeout == 7.5


def test_database_file_context_prefers_worktree_over_shared_and_systemd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree = tmp_path / "worktree"
    shared = tmp_path / "shared"
    systemd = tmp_path / "systemd.env"
    for path in (worktree, shared):
        path.mkdir()
    (worktree / ".env").write_text("ATLAS_DB_HOST=worktree.example\n", encoding="utf-8")
    (shared / ".env").write_text("ATLAS_DB_HOST=shared.example\n", encoding="utf-8")
    systemd.write_text("ATLAS_DB_HOST=systemd.example\n", encoding="utf-8")
    monkeypatch.delenv("ATLAS_OPS_ENV_FILES", raising=False)
    monkeypatch.delenv("ATLAS_DB_HOST", raising=False)
    function_globals = database_env_files.__globals__
    monkeypatch.setitem(function_globals, "worktree_root", lambda: worktree)
    monkeypatch.setitem(function_globals, "shared_root", lambda: shared)
    monkeypatch.setitem(function_globals, "systemd_env_files", lambda: [systemd])

    selected = database_env_files()
    child_env = database_runtime_environment()

    assert selected == [worktree / ".env", worktree / ".env.local"]
    assert child_env["ATLAS_DB_HOST"] == "worktree.example"


def test_database_file_context_fallbacks_are_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree = tmp_path / "worktree"
    shared = tmp_path / "shared"
    systemd = tmp_path / "systemd.env"
    for path in (worktree, shared):
        path.mkdir()
    (shared / ".env.local").write_text("ATLAS_DB_HOST=shared.example\n", encoding="utf-8")
    systemd.write_text("ATLAS_DB_HOST=systemd.example\n", encoding="utf-8")
    monkeypatch.delenv("ATLAS_OPS_ENV_FILES", raising=False)
    function_globals = database_env_files.__globals__
    monkeypatch.setitem(function_globals, "worktree_root", lambda: worktree)
    monkeypatch.setitem(function_globals, "shared_root", lambda: shared)
    monkeypatch.setitem(function_globals, "systemd_env_files", lambda: [systemd])

    assert database_env_files() == [shared / ".env", shared / ".env.local"]

    (shared / ".env.local").unlink()
    assert database_env_files() == [systemd]


def test_database_file_context_honors_explicit_override_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.env"
    second = tmp_path / "second.env"
    first.write_text("ATLAS_DB_HOST=first.example\n", encoding="utf-8")
    second.write_text("ATLAS_DB_HOST=second.example\n", encoding="utf-8")
    monkeypatch.setenv("ATLAS_OPS_ENV_FILES", os.pathsep.join((str(first), str(second))))
    monkeypatch.delenv("ATLAS_DB_HOST", raising=False)

    assert database_env_files() == [first, second]
    assert database_runtime_environment()["ATLAS_DB_HOST"] == "second.example"


def test_database_probe_degrades_when_project_dotenv_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ATLAS_DB_HOST=localhost\n", encoding="utf-8")
    monkeypatch.setenv("ATLAS_OPS_ENV_FILES", str(env_file))
    for key in DATABASE_CONFIG_KEYS:
        monkeypatch.delenv(key, raising=False)
    real_import = builtins.__import__

    def import_without_dotenv(name: str, *args: object, **kwargs: object) -> object:
        if name == "dotenv":
            raise ImportError("dependency intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_dotenv)

    assert OPS_MODULE["database_probe"]() == (
        "UNAVAILABLE",
        "project Python is missing python-dotenv",
    )
    with pytest.raises(OpsError, match="missing python-dotenv"):
        database_runtime_environment()


def test_brain_health_never_returns_the_configured_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_url = (
        "https://operator:canary-password@brain.example.invalid/api/v1/ping"
        "?token=canary-secret#credential-fragment"
    )
    requested: list[str] = []

    class FakeResponse:
        status = 204

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def fake_urlopen(url: str, **_kwargs: object) -> FakeResponse:
        requested.append(url)
        return FakeResponse()

    function_globals = OPS_MODULE["brain_health"].__globals__
    monkeypatch.setenv("ATLAS_OPS_BRAIN_URL", configured_url)
    monkeypatch.delenv("ATLAS_OPS_OFFLINE", raising=False)
    monkeypatch.setitem(function_globals, "urlopen", fake_urlopen)

    state, detail = OPS_MODULE["brain_health"]()

    assert state == "OK"
    assert requested == [configured_url]
    assert configured_url not in detail
    assert "canary-password" not in detail
    assert "canary-secret" not in detail


def test_git_snapshot_never_reads_or_prints_raw_origin(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    commands: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        output = "main\n" if "--show-current" in args else ""
        return subprocess.CompletedProcess(args, 0, output, "")

    function_globals = OPS_MODULE["print_git_snapshot"].__globals__
    monkeypatch.setitem(function_globals, "worktree_root", lambda: ROOT)
    monkeypatch.setitem(function_globals, "run", fake_run)
    OPS_MODULE["print_git_snapshot"]()

    output = capsys.readouterr().out
    assert "canfieldjuan/ATLAS" in output
    assert "canary-token" not in output
    assert all("get-url" not in command for command in commands)


@pytest.mark.parametrize("database_key", TEST_DATABASE_URL_KEYS)
def test_integration_guard_admits_each_canonical_database_variable(
    database_key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in TEST_DATABASE_URL_KEYS:
        monkeypatch.delenv(key, raising=False)
    confirmed_url = f"postgresql://confirmed.invalid/{database_key.lower()}"
    unconfirmed_url = "postgresql://unconfirmed.invalid/must-not-reach-pytest"
    monkeypatch.setenv(database_key, confirmed_url)
    for key in ("DATABASE_URL", "EXTRACTED_DATABASE_URL", "FUTURE_DATABASE_URL"):
        monkeypatch.setenv(key, unconfirmed_url)
    libpq_keys = (
        "PGHOST",
        "PGDATABASE",
        "PGUSER",
        "PGPASSWORD",
        "PGPASSFILE",
        "PGSERVICE",
        "PGSERVICEFILE",
        "PGSSLKEY",
        "PGSSLCERT",
        "PGSSLROOTCERT",
        "PGCHANNELBINDING",
        "PGTARGETSESSIONATTRS",
        "PGFUTURE_CREDENTIAL",
    )
    for key in libpq_keys:
        monkeypatch.setenv(key, unconfirmed_url)
    for key in DATABASE_CONFIG_KEYS:
        monkeypatch.setenv(key, unconfirmed_url)
    monkeypatch.setenv("ATLAS_CONFIRM_DISPOSABLE_TEST_DB", "1")
    monkeypatch.setenv("ATLAS_INTEGRATION_CANARY", "preserved")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_exec(args: list[str], **kwargs: object) -> int:
        calls.append((args, kwargs))
        return 0

    function_globals = OPS_MODULE["test_command"].__globals__
    monkeypatch.setitem(function_globals, "worktree_root", lambda: ROOT)
    monkeypatch.setitem(function_globals, "exec_command", fake_exec)

    target = "tests/test_agent_operations_contract.py::test_doctor_runs_without_live_provider_access"
    assert OPS_MODULE["test_command"](["integration", target, "-q"]) == 0
    assert calls and calls[0][0][-2:] == [target, "-q"]
    child_env = calls[0][1]["env"]
    assert isinstance(child_env, dict)
    assert child_env[database_key] == confirmed_url
    assert child_env["DATABASE_URL"] == confirmed_url
    assert child_env["EXTRACTED_DATABASE_URL"] == confirmed_url
    assert child_env["ATLAS_DB_CONNECTION_STRING"] == confirmed_url
    assert child_env["ATLAS_INTEGRATION_CANARY"] == "preserved"
    assert "FUTURE_DATABASE_URL" not in child_env
    assert all(key not in child_env for key in libpq_keys)
    assert not any(key.startswith("PG") for key in child_env)
    for key in DATABASE_CONFIG_KEYS - {"ATLAS_DB_CONNECTION_STRING"}:
        assert key not in child_env
    assert unconfirmed_url not in child_env.values()
    assert all(confirmed_url not in argument for argument in calls[0][0])
    assert all(unconfirmed_url not in argument for argument in calls[0][0])


@pytest.mark.parametrize("active_count", (0, 2, 3))
def test_integration_guard_requires_exactly_one_database_variable(
    active_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in TEST_DATABASE_URL_KEYS:
        monkeypatch.delenv(key, raising=False)
    active_keys = TEST_DATABASE_URL_KEYS[:active_count]
    for key in active_keys:
        monkeypatch.setenv(key, f"postgresql://canary.invalid/{key.lower()}")
    monkeypatch.setenv("ATLAS_CONFIRM_DISPOSABLE_TEST_DB", "1")
    calls: list[list[str]] = []
    function_globals = OPS_MODULE["test_command"].__globals__
    monkeypatch.setitem(function_globals, "worktree_root", lambda: ROOT)
    monkeypatch.setitem(
        function_globals,
        "exec_command",
        lambda args, **_kwargs: calls.append(args) or 0,
    )

    with pytest.raises(OpsError, match="exactly one") as exc_info:
        OPS_MODULE["test_command"](
            ["integration", "tests/test_agent_operations_contract.py"]
        )
    assert calls == []
    assert all(key in str(exc_info.value) for key in active_keys)
    assert "postgresql://" not in str(exc_info.value)


@pytest.mark.parametrize(
    "target",
    (
        "-q",
        ".",
        "tests",
        "tests/",
        "README.md",
        "tests/does_not_exist.py",
        "../tests/test_agent_operations_contract.py",
    ),
)
def test_integration_guard_rejects_unbounded_or_invalid_targets(
    target: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        TEST_DATABASE_URL_KEYS[0],
        "postgresql://disposable.invalid/atlas_test",
    )
    monkeypatch.setenv("ATLAS_CONFIRM_DISPOSABLE_TEST_DB", "1")
    calls: list[list[str]] = []

    function_globals = OPS_MODULE["test_command"].__globals__
    monkeypatch.setitem(function_globals, "worktree_root", lambda: ROOT)
    monkeypatch.setitem(
        function_globals,
        "exec_command",
        lambda args, **_kwargs: calls.append(args) or 0,
    )

    with pytest.raises(OpsError, match="existing Python file under tests"):
        OPS_MODULE["test_command"](["integration", target])
    assert calls == []


def test_integration_guard_rejects_additional_positional_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        TEST_DATABASE_URL_KEYS[0],
        "postgresql://disposable.invalid/atlas_test",
    )
    monkeypatch.setenv("ATLAS_CONFIRM_DISPOSABLE_TEST_DB", "1")
    calls: list[list[str]] = []
    target = "tests/test_agent_operations_contract.py"

    function_globals = OPS_MODULE["test_command"].__globals__
    monkeypatch.setitem(function_globals, "worktree_root", lambda: ROOT)
    monkeypatch.setitem(
        function_globals,
        "exec_command",
        lambda args, **_kwargs: calls.append(args) or 0,
    )

    with pytest.raises(OpsError, match="pytest option tokens"):
        OPS_MODULE["test_command"](["integration", target, "tests/"])
    assert calls == []


def test_unit_mode_is_github_only_and_never_launches_pytest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    function_globals = OPS_MODULE["test_command"].__globals__
    monkeypatch.setitem(function_globals, "worktree_root", lambda: ROOT)
    monkeypatch.setitem(
        function_globals,
        "exec_command",
        lambda args, **_kwargs: calls.append(args) or 0,
    )

    with pytest.raises(OpsError, match="full unit gate is GitHub-only"):
        OPS_MODULE["test_command"](["unit"])
    assert calls == []


def test_focused_mode_removes_database_authority_from_child_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unconfirmed = "postgresql://unconfirmed.invalid/must-not-reach-focused-pytest"
    database_keys = {
        *TEST_DATABASE_URL_KEYS,
        *DATABASE_CONFIG_KEYS,
        "DATABASE_URL",
        "EXTRACTED_DATABASE_URL",
        "FUTURE_DATABASE_URL",
        "PGHOST",
        "PGPASSWORD",
        "PGSERVICEFILE",
        "PGFUTURE_CREDENTIAL",
    }
    for key in database_keys:
        monkeypatch.setenv(key, unconfirmed)
    monkeypatch.setenv("ATLAS_CONFIRM_DISPOSABLE_TEST_DB", "1")
    monkeypatch.setenv("ATLAS_FOCUSED_CANARY", "preserved")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_exec(args: list[str], **kwargs: object) -> int:
        calls.append((args, kwargs))
        return 0

    function_globals = OPS_MODULE["test_command"].__globals__
    monkeypatch.setitem(function_globals, "worktree_root", lambda: ROOT)
    monkeypatch.setitem(function_globals, "exec_command", fake_exec)

    target = "tests/test_agent_operations_contract.py::test_doctor_runs_without_live_provider_access"
    assert OPS_MODULE["test_command"](["focused", target, "-q"]) == 0
    assert calls and calls[0][0][-2:] == [target, "-q"]
    child_env = calls[0][1]["env"]
    assert isinstance(child_env, dict)
    assert all(key not in child_env for key in database_keys)
    assert not any(
        key == "DATABASE_URL"
        or key.endswith("_DATABASE_URL")
        or key.startswith("PG")
        or key in DATABASE_CONFIG_KEYS
        for key in child_env
    )
    assert "ATLAS_CONFIRM_DISPOSABLE_TEST_DB" not in child_env
    assert child_env["ATLAS_FOCUSED_CANARY"] == "preserved"
    assert unconfirmed not in child_env.values()
    assert all(unconfirmed not in argument for argument in calls[0][0])


def test_env_keys_never_emit_values_or_evaluate_file(tmp_path: Path) -> None:
    canary = "never-print-this-secret-value"
    side_effect = tmp_path / "must-not-exist"
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        f"ATLAS_SECRET={canary}\n"
        "export SAFE_NAME=present\n"
        f"EVIL=$(touch {side_effect})\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["ATLAS_OPS_ENV_FILES"] = str(env_file)
    env["ATLAS_OPS_OFFLINE"] = "1"
    result = subprocess.run(
        [str(OPS), "env", "keys"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ATLAS_SECRET" in result.stdout
    assert "SAFE_NAME" in result.stdout
    assert "EVIL" in result.stdout
    assert canary not in result.stdout
    assert not side_effect.exists()


def test_doctor_runs_without_live_provider_access() -> None:
    env = os.environ.copy()
    env["ATLAS_OPS_OFFLINE"] = "1"
    env["ATLAS_OPS_ENV_FILES"] = ""
    result = subprocess.run(
        [str(OPS), "doctor"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for heading in (
        "PROJECT",
        "GIT",
        "RUNTIME",
        "TESTING",
        "DEPLOYMENT",
        "DATABASE",
        "CI",
        "USEFUL COMMANDS",
    ):
        assert heading in result.stdout
    assert "UNKNOWN" in result.stdout


def test_container_status_stops_at_unavailable_daemon(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    commands: list[list[str]] = []
    function_globals = OPS_MODULE["container_status"].__globals__
    monkeypatch.setattr(function_globals["shutil"], "which", lambda _name: None)
    OPS_MODULE["container_status"]()
    assert "containers: UNAVAILABLE" in capsys.readouterr().out

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        return subprocess.CompletedProcess(args, 1, "", "daemon unavailable")

    monkeypatch.delenv("ATLAS_OPS_OFFLINE", raising=False)
    monkeypatch.setitem(function_globals, "run", fake_run)
    monkeypatch.setattr(function_globals["shutil"], "which", lambda _name: "/usr/bin/docker")

    OPS_MODULE["container_status"]()

    output = capsys.readouterr().out
    assert commands == [["docker", "info"]]
    assert "containers: UNAVAILABLE" in output
    assert "absent" not in output


def test_container_status_classifies_offline_and_object_states(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    function_globals = OPS_MODULE["container_status"].__globals__
    monkeypatch.setattr(function_globals["shutil"], "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setenv("ATLAS_OPS_OFFLINE", "1")
    monkeypatch.setitem(
        function_globals,
        "run",
        lambda *_args, **_kwargs: pytest.fail("offline status must not call Docker"),
    )
    OPS_MODULE["container_status"]()
    assert "containers: UNKNOWN (offline test mode)" in capsys.readouterr().out

    monkeypatch.delenv("ATLAS_OPS_OFFLINE", raising=False)
    monkeypatch.setitem(function_globals, "KNOWN_CONTAINERS", ("running", "missing", "empty"))

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if args == ["docker", "info"]:
            return subprocess.CompletedProcess(args, 0, "ok", "")
        name = args[-1]
        if name == "running":
            return subprocess.CompletedProcess(args, 0, "running\n", "")
        if name == "empty":
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, 1, "", "not found")

    monkeypatch.setitem(function_globals, "run", fake_run)
    OPS_MODULE["container_status"]()
    output = capsys.readouterr().out
    assert "running: running" in output
    assert "missing: absent" in output
    assert "empty: UNKNOWN" in output


def test_mutating_database_and_deployment_commands_are_not_exposed() -> None:
    for args in (("db", "write"), ("db", "migrate"), ("deploy", "production")):
        result = subprocess.run(
            [str(OPS), *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 2
        assert "intentionally" in result.stderr or "unknown command" in result.stderr


def test_capability_map_is_machine_readable_and_has_required_surfaces() -> None:
    capability_path = ROOT / ".agent/capabilities.yaml"
    capabilities = yaml.safe_load(capability_path.read_text(encoding="utf-8"))
    assert capabilities["schema_version"] == 1
    for key in (
        "runtime",
        "testing",
        "ci",
        "deployment",
        "database",
        "logs",
        "environment",
        "external_services",
        "tools",
        "repository_operations",
    ):
        assert key in capabilities
    assert capabilities["database"]["safe_inspection"] == "./ops db inspect connectivity"
    assert capabilities["database"]["arbitrary_query"].startswith("unavailable")
    assert capabilities["deployment"]["brain"]["provider"] == "local systemd user service"
    assert capabilities["deployment"]["frontend"]["provider"] == "Vercel"
    assert "./ops doctor" in capabilities["project"]["source_of_truth"]
    assert "./ops runtime inspection" not in capabilities["project"]["source_of_truth"]
    assert capabilities["workspace"]["shared_root"]["path_hint"] == (
        "the parent for a conventional .git common directory; otherwise the "
        "direct bare common directory"
    )


def test_fresh_agent_documentation_contract_is_complete() -> None:
    runbooks = ROOT / ".agent/runbooks"
    required = (
        "environment.md",
        "deployment.md",
        "database.md",
        "testing.md",
        "logs.md",
        "ci.md",
        "discovery-ledger.md",
    )
    for name in required:
        text = (runbooks / name).read_text(encoding="utf-8")
        assert "Failure" in text or "failure" in text

    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert ".agent/capabilities.yaml" in agents
    assert "./ops doctor" in agents
    assert "do not leave that knowledge only in the session" in " ".join(agents.split())

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "cp .env.example .env" not in readme
    assert "docker compose up -d postgres" not in readme
    assert "docker compose up -d postgres" not in claude


def test_operational_contract_contains_no_credential_bearing_urls() -> None:
    files = [OPS, ROOT / ".agent/capabilities.yaml"]
    files.extend((ROOT / ".agent/runbooks").glob("*.md"))
    credential_url = re.compile(r"(?:postgres(?:ql)?|https?)://[^\s/:]+:[^\s/@]+@")
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert credential_url.search(text) is None, path
        assert "gho_" not in text, path
