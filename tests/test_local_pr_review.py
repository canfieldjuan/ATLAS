from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_local_pr_review_fails_on_dirty_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    (repo / "dirty.txt").write_text("not committed\n", encoding="utf-8")

    result = _run(repo, ["bash", "scripts/local_pr_review.sh"])

    assert result.returncode == 1
    assert "worktree has uncommitted changes" in result.stderr
    assert "dirty.txt" in result.stderr
    assert "Pre-push audit wrapper" not in result.stdout


def test_local_pr_review_allow_dirty_runs_checks(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    (repo / "dirty.txt").write_text("not committed\n", encoding="utf-8")

    result = _run(repo, ["bash", "scripts/local_pr_review.sh", "--allow-dirty"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Pre-push audit wrapper" in result.stdout
    assert "local PR review passed" in result.stdout


def test_local_pr_review_allow_dirty_preserves_base_ref_arg(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    (repo / "dirty.txt").write_text("not committed\n", encoding="utf-8")

    result = _run(
        repo,
        ["bash", "scripts/local_pr_review.sh", "--allow-dirty", "origin/main"],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "base ref: origin/main" in result.stdout


def test_local_pr_review_help_exits_cleanly(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)

    result = _run(repo, ["bash", "scripts/local_pr_review.sh", "--help"])

    assert result.returncode == 0
    assert "Usage: bash scripts/local_pr_review.sh" in result.stdout


def test_local_pr_review_rejects_unknown_option(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)

    result = _run(repo, ["bash", "scripts/local_pr_review.sh", "--unknown"])

    assert result.returncode == 2
    assert "unknown option: --unknown" in result.stderr


def test_local_pr_review_rejects_multiple_base_refs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)

    result = _run(
        repo,
        ["bash", "scripts/local_pr_review.sh", "origin/main", "origin/main"],
    )

    assert result.returncode == 2
    assert "multiple base refs supplied" in result.stderr


def _write_fixture_repo(repo: Path) -> None:
    (repo / "scripts").mkdir(parents=True)
    _write_executable(
        repo / "scripts" / "local_pr_review.sh",
        (REPO_ROOT / "scripts" / "local_pr_review.sh").read_text(encoding="utf-8"),
    )
    _write_executable(
        repo / "scripts" / "pre_push_audit.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\necho pre-push ok\n",
    )
    _write_executable(
        repo / "scripts" / "audit_plan_code_consistency.py",
        "#!/usr/bin/env python3\nprint('plan ok')\n",
    )
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    _git(repo, "branch", "-M", "main")
    _git(repo, "remote", "add", "origin", str(repo))
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(repo, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main")


def _run(repo: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(repo)},
    )


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
