from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "audit_plan_doc_diff_size.py"


def load_auditor():
    name = "audit_plan_doc_diff_size"
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


def _plan(total: str, earlier_example_total: str = "") -> str:
    return textwrap.dedent(
        f"""\
        # Plan

        ## Mechanism

        {earlier_example_total}

        ## Estimated diff size

        | File | LOC |
        |---|---:|
        | `script.py` | 10 |
        | **Total** | **{total}** |
        """
    )


def test_estimated_total_loc_reads_only_estimated_diff_size_section():
    auditor = load_auditor()
    text = _plan("100", "| **Total** | **999** |")

    assert auditor.estimated_total_loc(text) == 100


def test_estimated_total_loc_accepts_approximate_and_comma_values():
    auditor = load_auditor()

    assert auditor.estimated_total_loc(_plan("~1,250")) == 1250


def test_audit_diff_size_statuses():
    auditor = load_auditor()

    assert auditor.audit_diff_size(_plan("100"), 124).status == "OK"
    assert auditor.audit_diff_size(_plan("100"), 126).status == "WARN"
    assert auditor.audit_diff_size(_plan("100"), 151).status == "FAIL"


def test_missing_total_returns_none():
    auditor = load_auditor()
    text = textwrap.dedent(
        """\
        # Plan

        ## Estimated diff size

        | File | LOC |
        |---|---:|
        | `script.py` | 10 |
        """
    )

    assert auditor.audit_diff_size(text, 10) is None


def test_cli_treats_missing_total_as_audit_failure(tmp_path):
    repo = _git_repo_with_base_commit(tmp_path)
    plan = repo / "plans" / "slice.md"
    plan.parent.mkdir()
    plan.write_text(
        textwrap.dedent(
            """\
            # Plan

            ## Estimated diff size

            | File | LOC |
            |---|---:|
            | `script.py` | 10 |
            """
        ),
        encoding="utf-8",
    )
    (repo / "script.py").write_text("a = 1\n", encoding="utf-8")
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
    assert "status: FAIL" in result.stdout
    assert "estimated diff size total not found" in result.stdout
    assert result.stderr == ""


def test_cli_accepts_diff_within_soft_threshold(tmp_path):
    repo = _git_repo_with_base_commit(tmp_path)
    plan = repo / "plans" / "slice.md"
    plan.parent.mkdir()
    plan.write_text(_plan("14"), encoding="utf-8")
    (repo / "script.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
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
    assert "status: OK" in result.stdout


def test_cli_fails_diff_beyond_hard_threshold(tmp_path):
    repo = _git_repo_with_base_commit(tmp_path)
    plan = repo / "plans" / "slice.md"
    plan.parent.mkdir()
    plan.write_text(_plan("2"), encoding="utf-8")
    (repo / "script.py").write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
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
    assert "status: FAIL" in result.stdout


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
