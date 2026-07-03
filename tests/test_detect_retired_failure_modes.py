from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "detect_retired_failure_modes.py"

SPEC = importlib.util.spec_from_file_location("detect_retired_failure_modes", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
sys.modules["detect_retired_failure_modes"] = MOD
SPEC.loader.exec_module(MOD)


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test User")
    return repo


def commit_all(repo: Path, message: str) -> None:
    git(repo, "add", ".")
    git(repo, "commit", "-m", message)


def write(repo: Path, path: str, text: str) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(text), encoding="utf-8")


def modes(report: dict[str, object]) -> set[str]:
    return {signal["mode"] for signal in report["signals"]}


def signals_for(report: dict[str, object], mode: str) -> list[dict[str, object]]:
    return [signal for signal in report["signals"] if signal["mode"] == mode]


def test_plan_weakening_signal_when_obligation_removed_with_code_change(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    write(
        repo,
        "plans/PR-Example.md",
        """
        # PR-Example

        ## Scope (this PR)

        ### Files touched

        - `plans/PR-Example.md`
        - `src/app.py`

        ## Mechanism

        The parser must reject malformed input and keep the negative fixture.
        """,
    )
    write(repo, "src/app.py", "def parse(value):\n    return value.strip()\n")
    commit_all(repo, "base")

    write(
        repo,
        "plans/PR-Example.md",
        """
        # PR-Example

        ## Scope (this PR)

        ### Files touched

        - `plans/PR-Example.md`
        - `src/app.py`

        ## Mechanism

        The parser handles common input.
        """,
    )
    write(repo, "src/app.py", "def parse(value):\n    return value.strip().lower()\n")
    commit_all(repo, "feature")

    report = MOD.build_report("HEAD~1", cwd=repo)

    assert MOD.MODE_PLAN_WEAKENING in modes(report)
    signal = signals_for(report, MOD.MODE_PLAN_WEAKENING)[0]
    assert signal["signature"] == "plan_obligation_removed_with_code_change"
    assert any("must reject malformed" in item for item in signal["evidence"])


def test_test_weakening_signal_when_assertion_removed_with_code_change(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    write(repo, "src/calculator.py", "def total(values):\n    return sum(values)\n")
    write(
        repo,
        "tests/test_calculator.py",
        """
        from src.calculator import total

        def test_total_rejects_empty():
            assert total([1, 2]) == 3
        """,
    )
    commit_all(repo, "base")

    write(repo, "src/calculator.py", "def total(values):\n    return sum(values or [])\n")
    write(
        repo,
        "tests/test_calculator.py",
        """
        from src.calculator import total

        def test_total_rejects_empty():
            result = total([1, 2])
        """,
    )
    commit_all(repo, "feature")

    report = MOD.build_report("HEAD~1", cwd=repo)

    assert MOD.MODE_TEST_WEAKENING in modes(report)
    signal = signals_for(report, MOD.MODE_TEST_WEAKENING)[0]
    assert signal["signature"] == "test_assertion_removed_with_code_change"
    assert any("assert total" in item for item in signal["evidence"])


def test_scope_drift_signal_when_diff_exceeds_plan_files_touched(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    write(repo, "README.md", "base\n")
    commit_all(repo, "base")

    write(
        repo,
        "plans/PR-Example.md",
        """
        # PR-Example

        ## Scope (this PR)

        ### Files touched

        - `plans/PR-Example.md`

        ## Mechanism

        Adds a tiny helper.
        """,
    )
    write(repo, "src/helper.py", "def helper():\n    return 'ok'\n")
    commit_all(repo, "feature")

    report = MOD.build_report("HEAD~1", cwd=repo)

    assert MOD.MODE_SCOPE_DRIFT in modes(report)
    signal = signals_for(report, MOD.MODE_SCOPE_DRIFT)[0]
    assert signal["signature"] == "changed_files_outside_plan_files_touched"
    assert "src/helper.py" in signal["paths"]


def test_symptom_patching_signal_for_fix_plan_without_root_cause(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    write(repo, "README.md", "base\n")
    commit_all(repo, "base")

    write(
        repo,
        "plans/PR-Fix-Thing.md",
        """
        # PR-Fix-Thing

        ## Why this slice exists

        This fixes a bug in the output.

        ## Scope (this PR)

        ### Files touched

        - `plans/PR-Fix-Thing.md`
        - `src/api/view.py`
        """,
    )
    write(repo, "src/api/view.py", "def render(value):\n    return str(value)\n")
    commit_all(repo, "feature")

    report = MOD.build_report("HEAD~1", cwd=repo)

    assert MOD.MODE_SYMPTOM_PATCHING in modes(report)
    signal = signals_for(report, MOD.MODE_SYMPTOM_PATCHING)[0]
    assert signal["signature"] == "fix_plan_missing_root_cause_language"
    assert signal["confidence"] == "low"


def test_clean_diff_emits_stable_empty_json_report(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    write(repo, "src/app.py", "def parse(value):\n    return value.strip()\n")
    commit_all(repo, "base")

    write(
        repo,
        "plans/PR-Clean.md",
        """
        # PR-Clean

        ## Scope (this PR)

        ### Files touched

        - `plans/PR-Clean.md`
        - `src/app.py`

        ## Mechanism

        Adds a non-fix normalization helper.
        """,
    )
    write(repo, "src/app.py", "def parse(value):\n    return value.strip().lower()\n")
    commit_all(repo, "feature")

    report = MOD.build_report("HEAD~1", cwd=repo)

    assert report["schema_version"] == 1
    assert report["signal_type"] == "retired_failure_recurrence"
    assert report["detector_version"] == MOD.DETECTOR_VERSION
    assert report["signals"] == []


def test_cli_writes_json_and_exits_zero_when_signal_found(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    write(repo, "src/app.py", "def value():\n    return 1\n")
    write(repo, "tests/test_app.py", "def test_value():\n    assert 1 == 1\n")
    commit_all(repo, "base")

    write(repo, "src/app.py", "def value():\n    return 2\n")
    write(repo, "tests/test_app.py", "def test_value():\n    result = 1\n")
    commit_all(repo, "feature")
    out = repo / "signals.json"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--base", "HEAD~1", "--json-out", str(out)],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert MOD.MODE_TEST_WEAKENING in modes(payload)
    assert "advisory signal" in result.stdout


def test_git_error_path_raises_runtime_error(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    write(repo, "README.md", "base\n")
    commit_all(repo, "base")

    with pytest.raises(RuntimeError, match="missing-ref"):
        MOD.merge_base("missing-ref", cwd=repo)
