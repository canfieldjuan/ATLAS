from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

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


def _plan(files_section: str) -> str:
    return textwrap.dedent(
        f"""\
        # Plan

        ## Scope

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
