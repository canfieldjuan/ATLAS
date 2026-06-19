from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "audit_plan_doc_files_touched.py"


def load_auditor():
    name = "audit_plan_doc_files_touched"
    if name in sys.modules:
        return sys.modules[name]

    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _plan(files_section: str, max_files: object | None = None) -> str:
    budget_line = f"\n        Max files: {max_files}\n" if max_files is not None else ""
    return textwrap.dedent(
        f"""\
        # Plan

        ## Scope
        {budget_line}
        Backticks here, like `not-a-file.py`, are not file claims.

        ### Files touched

        {files_section}

        ## Verification
        """
    )


def test_claimed_files_only_come_from_files_touched_subsection():
    auditor = load_auditor()
    text = _plan(
        """\
        - `scripts/audit_plan_doc_files_touched.py`
        - `tests/test_audit_plan_doc_files_touched.py`
        """
    )

    assert auditor.claimed_files_touched(text) == {
        "scripts/audit_plan_doc_files_touched.py",
        "tests/test_audit_plan_doc_files_touched.py",
    }


def test_missing_actual_file_is_reported_as_missing_in_plan():
    auditor = load_auditor()
    audit = auditor.audit_files_touched(
        _plan("- `scripts/audit_plan_doc_files_touched.py`"),
        {
            "scripts/audit_plan_doc_files_touched.py",
            "tests/test_audit_plan_doc_files_touched.py",
        },
    )

    assert audit.missing_in_plan == {"tests/test_audit_plan_doc_files_touched.py"}
    assert audit.extra_in_plan == set()
    assert not audit.ok


def test_extra_claimed_file_is_reported_as_extra_in_plan():
    auditor = load_auditor()
    audit = auditor.audit_files_touched(
        _plan(
            """\
            - `scripts/audit_plan_doc_files_touched.py`
            - `docs/not-touched.md`
            """
        ),
        {"scripts/audit_plan_doc_files_touched.py"},
    )

    assert audit.missing_in_plan == set()
    assert audit.extra_in_plan == {"docs/not-touched.md"}
    assert not audit.ok


def test_declared_max_files_parses_budget():
    auditor = load_auditor()
    assert auditor.declared_max_files(_plan("- `a.py`", max_files=3)) == 3


def test_declared_max_files_absent_is_none():
    auditor = load_auditor()
    assert auditor.declared_max_files(_plan("- `a.py`")) is None


def test_declared_max_files_ignores_mentions_outside_scope():
    auditor = load_auditor()
    # A digit-only mention in a later section must not arm the budget.
    text = (
        "# Plan\n\n## Scope\n\n### Files touched\n\n- `a.py`\n\n"
        "## Deferred\n\nMax files: 9\n"
    )
    assert auditor.declared_max_files(text) is None


def test_declared_max_files_malformed_raises():
    auditor = load_auditor()
    for bad in ("four", "4.5", "-1"):
        with pytest.raises(auditor.PlanBudgetError):
            auditor.declared_max_files(_plan("- `a.py`", max_files=bad))


def test_cli_accepts_matching_plan_against_real_git_diff(tmp_path):
    repo = _git_repo_with_base_commit(tmp_path)
    plan = repo / "plans" / "slice.md"
    plan.parent.mkdir()
    plan.write_text(
        _plan(
            """\
            - `plans/slice.md`
            - `src/app.py`
            """
        ),
        encoding="utf-8",
    )
    source = repo / "src" / "app.py"
    source.parent.mkdir()
    source.write_text("print('ok')\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "change")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(plan), "HEAD~1"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "OK" in result.stdout


def test_cli_rejects_plan_that_omits_changed_file(tmp_path):
    repo = _git_repo_with_base_commit(tmp_path)
    plan = repo / "plans" / "slice.md"
    plan.parent.mkdir()
    plan.write_text(_plan("- `plans/slice.md`"), encoding="utf-8")
    source = repo / "src" / "app.py"
    source.parent.mkdir()
    source.write_text("print('ok')\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "change")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(plan), "HEAD~1"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "MISSING" in result.stdout
    assert "src/app.py" in result.stdout


def _two_file_repo(tmp_path: Path, max_files: object | None) -> tuple[Path, Path]:
    """Repo whose diff vs HEAD~1 touches exactly two files, both listed in the
    plan, so MISSING/EXTRA is clean and only the budget can fail."""
    repo = _git_repo_with_base_commit(tmp_path)
    plan = repo / "plans" / "slice.md"
    plan.parent.mkdir()
    plan.write_text(
        _plan(
            """\
            - `plans/slice.md`
            - `src/app.py`
            """,
            max_files=max_files,
        ),
        encoding="utf-8",
    )
    source = repo / "src" / "app.py"
    source.parent.mkdir()
    source.write_text("print('ok')\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "change")
    return repo, plan


def test_cli_rejects_over_budget(tmp_path):
    repo, plan = _two_file_repo(tmp_path, max_files=1)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(plan), "HEAD~1"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "OVER BUDGET" in result.stdout
    # Budget is the only failing dimension: both files are listed in the plan.
    assert "MISSING" not in result.stdout.replace("OVER BUDGET", "")


def test_cli_flag_overrides_plan_budget(tmp_path):
    repo, plan = _two_file_repo(tmp_path, max_files=None)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(plan), "HEAD~1", "--max-files", "1"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "OVER BUDGET" in result.stdout


def test_cli_malformed_budget_fails_closed(tmp_path):
    repo, plan = _two_file_repo(tmp_path, max_files="four")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(plan), "HEAD~1"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "malformed" in result.stderr.lower()


def test_cli_no_budget_is_unbounded(tmp_path):
    repo, plan = _two_file_repo(tmp_path, max_files=None)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(plan), "HEAD~1"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "OK" in result.stdout


def _git_repo_with_base_commit(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    return repo


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
