from __future__ import annotations

import os
import subprocess
from pathlib import Path
from shutil import copy2


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "push_pr.sh"


def test_push_pr_dry_run_without_managed_hook_runs_wrapper_review(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    env = {**os.environ, "ATLAS_PUSH_PR_DRY_RUN": "1"}

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "-u", "origin", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert f"ATLAS_CURRENT_PR_BODY_FILE={body}" in result.stdout
    assert f"--current-pr-body-file {body}" in result.stdout
    assert "git push -u origin HEAD" in result.stdout


def test_push_pr_dry_run_with_managed_hook_uses_hook_as_single_review_runner(
    tmp_path: Path,
) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    _write_managed_hook(repo)
    env = {**os.environ, "ATLAS_PUSH_PR_DRY_RUN": "1"}

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "-u", "origin", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "managed pre-push hook will run local PR review once" in result.stdout
    assert "bash scripts/local_pr_review.sh" not in result.stdout
    assert f"ATLAS_CURRENT_PR_BODY_FILE={body}" in result.stdout
    assert "git push -u origin HEAD" in result.stdout


def test_push_pr_dry_run_with_skip_env_keeps_wrapper_review(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    _write_managed_hook(repo)
    env = {
        **os.environ,
        "ATLAS_PUSH_PR_DRY_RUN": "1",
        "ATLAS_SKIP_LOCAL_PR_REVIEW": "1",
    }

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "-u", "origin", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert f"ATLAS_CURRENT_PR_BODY_FILE={body}" in result.stdout
    assert f"--current-pr-body-file {body}" in result.stdout
    assert "managed pre-push hook will run local PR review once" not in result.stdout


def test_push_pr_rejects_no_verify(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "--no-verify", "-u", "origin", "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "refusing to forward --no-verify" in result.stderr


def test_push_pr_missing_body_file_fails_clearly(tmp_path: Path) -> None:
    missing = tmp_path / "missing.md"

    result = subprocess.run(
        ["bash", str(SCRIPT), str(missing), "-u", "origin", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "PR body file not found" in result.stderr
    assert str(missing) in result.stderr


def _write_fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    copy2(SCRIPT, repo / "scripts" / "push_pr.sh")
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    return repo


def _write_body(repo: Path) -> Path:
    body = repo / "body.md"
    body.write_text("Plan: plans/PR-Test.md\nSlice phase: Workflow/process\n", encoding="utf-8")
    return body


def _write_managed_hook(repo: Path) -> None:
    hook = repo / ".git" / "hooks" / "pre-push"
    hook.write_text(
        "#!/usr/bin/env bash\n# ATLAS_LOCAL_PR_REVIEW_HOOK\n",
        encoding="utf-8",
    )
