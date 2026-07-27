import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "codex_review_scope_policy.py"

SPEC = importlib.util.spec_from_file_location("codex_review_scope_policy", SCRIPT)
policy = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = policy
SPEC.loader.exec_module(policy)


def test_builtin_fixtures_match_expected_dispositions():
    assert policy.validate_fixtures() == []


def test_fixture_scenarios_cover_operator_requested_review_shapes():
    names = {fixture.name for fixture in policy.FIXTURES}
    assert names == {
        "docs_process_noise",
        "duplicate_instance",
        "out_of_scope_hardening",
        "missing_regression_test",
        "concrete_security_data_failure",
        "speculative_risk_no_failure_path",
        "nit_suppression",
    }


def test_cli_self_test_reports_pass_count():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--self-test"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK: 7 Codex review scope fixtures passed" in proc.stdout


def test_active_docs_remove_second_reviewer_gate():
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    rules = (ROOT / "docs" / "REVIEWER_RULES.md").read_text(encoding="utf-8")
    watcher = (ROOT / "scripts" / "watch_owned_pr.sh").read_text(encoding="utf-8")

    assert "Codex connector review" in agents
    assert "live-reconciliation" in agents
    assert "claude-review" not in agents
    assert "claude-review" not in rules
    assert "claude-review" not in watcher
    assert "Suppress NITs by default" in agents
    assert "WAIVE_DUPLICATE" in agents
    assert "WAIVE_OUT_OF_SCOPE" in agents
