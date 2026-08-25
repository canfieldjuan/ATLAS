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
validate_read_only_sql = OPS_MODULE["validate_read_only_sql"]
psql_readonly_command = OPS_MODULE["psql_readonly_command"]


@pytest.mark.parametrize(
    "sql",
    (
        "SELECT 1",
        "-- harmless comment\nSELECT ';' AS value;",
        "SELECT $$semicolon ; in dollar quote$$",
        "SELECT 1 /* ignored ; nested /* comment */ safely */;",
        "SHOW transaction_read_only",
        "TABLE schema_migrations",
        "VALUES (1), (2)",
    ),
)
def test_read_only_sql_admits_single_safe_statement(sql: str) -> None:
    assert validate_read_only_sql(sql) == (True, "ok")


@pytest.mark.parametrize(
    "sql,reason_fragment",
    (
        ("", "empty"),
        ("INSERT INTO audit_log VALUES (1)", "only SELECT"),
        ("WITH changed AS (DELETE FROM x RETURNING *) SELECT * FROM changed", "only SELECT"),
        ("SELECT 1; DELETE FROM contacts", "exactly one"),
        ("SELECT * INTO copied_contacts FROM contacts", "SELECT INTO"),
        ("SELECT * FROM contacts FOR UPDATE", "row-locking"),
        ("SELECT 'unterminated", "unterminated"),
        ("/* unterminated", "unterminated"),
        ("\\copy contacts TO '/tmp/contacts'", "only SELECT"),
    ),
)
def test_read_only_sql_rejects_write_and_ambiguous_boundaries(
    sql: str,
    reason_fragment: str,
) -> None:
    allowed, reason = validate_read_only_sql(sql)
    assert allowed is False
    assert reason_fragment in reason


def test_psql_query_has_database_enforced_read_only_boundary() -> None:
    command = psql_readonly_command("SELECT 1")
    assert "BEGIN TRANSACTION READ ONLY;" in command
    assert "ROLLBACK;" in command
    assert command.index("BEGIN TRANSACTION READ ONLY;") < command.index("SELECT 1")
    assert command.index("SELECT 1") < command.index("ROLLBACK;")


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
    assert capabilities["database"]["safe_query"].startswith("./ops db query")
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
