from __future__ import annotations

import os
import subprocess
from pathlib import Path
from shutil import copy2


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "open_pr.sh"


def test_open_pr_create_passes_body_via_stdin_not_path(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=1)

    result = subprocess.run(
        [
            "bash",
            "scripts/open_pr.sh",
            str(body),
            "--title",
            "Workflow wrapper",
            "--base",
            "main",
        ],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert log.read_text(encoding="utf-8").strip() == (
        "pr create --title Workflow wrapper --base main --body-file -"
    )
    assert str(body) not in log.read_text(encoding="utf-8")
    assert stdin_capture.read_text(encoding="utf-8") == body.read_text(encoding="utf-8")


def test_open_pr_edit_passes_body_via_stdin_not_path(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=0)

    result = subprocess.run(
        ["bash", "scripts/open_pr.sh", str(body)],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert log.read_text(encoding="utf-8").strip() == "pr edit main --body-file -"
    assert str(body) not in log.read_text(encoding="utf-8")
    assert stdin_capture.read_text(encoding="utf-8") == body.read_text(encoding="utf-8")


def test_open_pr_existing_pr_rejects_create_only_args(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    env, _, _ = _fake_gh_env(tmp_path, view_exit=0)

    result = subprocess.run(
        ["bash", "scripts/open_pr.sh", str(body), "--title", "New title"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "PR already exists" in result.stderr


def test_open_pr_rejects_direct_body_args(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    env, _, _ = _fake_gh_env(tmp_path, view_exit=1)

    result = subprocess.run(
        ["bash", "scripts/open_pr.sh", str(body), "--body-file", str(body)],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "pass the PR body as BODY_FILE" in result.stderr


def test_open_pr_missing_body_file_fails_clearly(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    missing = repo / "missing.md"

    result = subprocess.run(
        ["bash", "scripts/open_pr.sh", str(missing)],
        cwd=repo,
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
    copy2(SCRIPT, repo / "scripts" / "open_pr.sh")
    subprocess.run(
        ["git", "init", "--initial-branch", "main"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return repo


def _write_body(repo: Path) -> Path:
    body = repo / "body.md"
    body.write_text(
        "Plan: plans/PR-Test.md\nSlice phase: Workflow/process\n",
        encoding="utf-8",
    )
    return body


def _fake_gh_env(
    tmp_path: Path,
    *,
    view_exit: int,
) -> tuple[dict[str, str], Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "gh-argv.txt"
    stdin_capture = tmp_path / "gh-stdin.txt"
    gh = bin_dir / "gh"
    gh.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [ "$1" = "pr" ] && [ "$2" = "view" ]; then
    exit "${GH_VIEW_EXIT}"
fi
printf '%s\\n' "$*" > "${GH_ARGV_LOG}"
cat > "${GH_STDIN_CAPTURE}"
""",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "GH_VIEW_EXIT": str(view_exit),
        "GH_ARGV_LOG": str(log),
        "GH_STDIN_CAPTURE": str(stdin_capture),
    }
    return env, log, stdin_capture
