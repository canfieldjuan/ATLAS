from __future__ import annotations

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
TEST_DATABASE_URL_KEYS = OPS_MODULE["TEST_DATABASE_URL_KEYS"]
database_inspection_command = OPS_MODULE["database_inspection_command"]
database_runtime_environment = OPS_MODULE["database_runtime_environment"]


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
    monkeypatch.setenv(database_key, "postgresql://disposable.invalid/atlas_test")
    monkeypatch.setenv("ATLAS_CONFIRM_DISPOSABLE_TEST_DB", "1")
    calls: list[list[str]] = []

    def fake_exec(args: list[str], **_kwargs: object) -> int:
        calls.append(args)
        return 0

    function_globals = OPS_MODULE["test_command"].__globals__
    monkeypatch.setitem(function_globals, "worktree_root", lambda: ROOT)
    monkeypatch.setitem(function_globals, "exec_command", fake_exec)

    assert OPS_MODULE["test_command"](["integration", "tests/example.py", "-q"]) == 0
    assert calls and calls[0][-2:] == ["tests/example.py", "-q"]


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
