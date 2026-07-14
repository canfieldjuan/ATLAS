from __future__ import annotations

import importlib.util
import subprocess
import sys
import os
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_pr_plan_presence.py"
POLICY_SCRIPT = SCRIPT.parent / "_pr_change_policy.py"
POLICY_SPEC = importlib.util.spec_from_file_location("pr_change_policy", POLICY_SCRIPT)
assert POLICY_SPEC is not None and POLICY_SPEC.loader is not None
policy_module = importlib.util.module_from_spec(POLICY_SPEC)
sys.modules[POLICY_SPEC.name] = policy_module
POLICY_SPEC.loader.exec_module(policy_module)

ChangePolicyError = policy_module.ChangePolicyError
ChangeKind = policy_module.ChangeKind
classify_changes = policy_module.classify_changes


def test_human_non_markdown_diff_without_plan_fails(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write(repo, "scripts/example.py", "print('changed')\n")
    _commit(repo, "code change")

    result = _run(repo)

    assert result.returncode == 1
    assert "must add exactly one" in result.stdout
    assert "branch-added plans: none" in result.stdout


def test_human_non_markdown_diff_with_one_added_plan_passes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write(repo, "scripts/example.py", "print('changed')\n")
    _write(repo, "plans/PR-Example.md", "# Example\n")
    _commit(repo, "planned code change")

    result = _run(repo)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "required branch-added plan: plans/PR-Example.md" in result.stdout


def test_human_non_markdown_diff_with_multiple_plans_fails(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write(repo, "scripts/example.py", "print('changed')\n")
    _write(repo, "plans/PR-First.md", "# First\n")
    _write(repo, "plans/PR-Second.md", "# Second\n")
    _commit(repo, "ambiguous plans")

    result = _run(repo)

    assert result.returncode == 1
    assert "plans/PR-First.md" in result.stdout
    assert "plans/PR-Second.md" in result.stdout


def test_markdown_only_human_diff_is_explicitly_exempt(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write(repo, "docs/example.md", "# changed\n")
    _commit(repo, "docs change")

    result = _run(repo)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Markdown-only diff is explicitly exempt" in result.stdout


def test_markdown_symlink_requires_a_plan(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    docs = repo / "docs"
    docs.mkdir()
    (docs / "sneaky.md").symlink_to("../README.md")
    _commit(repo, "symlinked docs change")

    result = _run(repo)

    assert result.returncode == 1
    assert "classification: plan-required" in result.stdout


def test_compound_markdown_suffix_requires_a_plan(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write(repo, "docs/install.sh.md", "#!/usr/bin/env bash\necho changed\n")
    _commit(repo, "compound-suffix executable")

    result = _run(repo)

    assert result.returncode == 1
    assert "classification: plan-required" in result.stdout


def test_symlinked_branch_plan_does_not_count_toward_admission(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write(repo, "scripts/example.py", "print('changed')\n")
    plans = repo / "plans"
    plans.mkdir()
    (plans / "PR-Example.md").symlink_to("../README.md")
    _commit(repo, "symlinked plan")

    result = _run(repo)

    assert result.returncode == 1
    assert "branch-added plans: none" in result.stdout


def test_source_renamed_to_markdown_is_not_docs_only(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write(repo, "scripts/example.py", "print('base')\n")
    _commit(repo, "add source")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    (repo / "docs").mkdir()
    _git(repo, "mv", "scripts/example.py", "docs/example.md")
    _commit(repo, "rename source into docs")

    result = _run(repo)

    assert result.returncode == 1
    assert "classification: plan-required" in result.stdout
    assert "must add exactly one" in result.stdout


def test_dependabot_keeps_explicit_exemption_for_non_markdown_diff(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write(repo, "requirements.txt", "example==1\n")
    _commit(repo, "dependency change")

    result = _run(repo, "--pr-author", "app/dependabot")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Dependabot PR is explicitly exempt" in result.stdout


def test_missing_base_ref_is_infrastructure_failure(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    result = _run(repo, "refs/remotes/origin/missing")

    assert result.returncode == 2
    assert "base ref not found" in result.stderr


def test_missing_base_ref_raises_change_policy_error(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    with pytest.raises(ChangePolicyError, match="base ref not found"):
        classify_changes(
            author=None,
            base_ref="refs/remotes/origin/missing",
            repo_root=repo,
        )


def test_change_kind_keeps_its_string_contract_without_strenum() -> None:
    assert isinstance(ChangeKind.PLAN_REQUIRED, str)
    assert ChangeKind.PLAN_REQUIRED == "plan-required"


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(repo, "README.md", "base\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "branch", "-M", "main")
    _git(repo, "remote", "add", "origin", str(repo))
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    return repo


def _write(repo: Path, relative: str, text: str) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _commit(repo: Path, message: str) -> None:
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", message)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "ATLAS_AUDIT_REPO_ROOT": str(repo)},
    )
