from __future__ import annotations

import os
import subprocess
from pathlib import Path
from shutil import copy2


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "open_pr.sh"
AUDIT_SCRIPT = REPO_ROOT / "scripts" / "audit_pr_body.py"
CHANGE_POLICY_SCRIPT = REPO_ROOT / "scripts" / "_pr_change_policy.py"


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


def test_open_pr_rejects_invalid_body_before_gh_create(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_invalid_body(repo)
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=1)

    result = subprocess.run(
        ["bash", "scripts/open_pr.sh", str(body), "--title", "Workflow wrapper"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "missing required section: ## Cold diff reconstruction" in result.stdout
    assert not log.exists()
    assert not stdin_capture.exists()


def test_open_pr_rejects_invalid_body_before_gh_edit(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_invalid_body(repo)
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=0)

    result = subprocess.run(
        ["bash", "scripts/open_pr.sh", str(body)],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "missing required section: ## Cold diff reconstruction" in result.stdout
    assert not log.exists()
    assert not stdin_capture.exists()


def test_open_pr_accepts_explicit_docs_only_body(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_docs_only_body(repo)
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=1)

    result = subprocess.run(
        ["bash", "scripts/open_pr.sh", str(body), "--title", "Docs only"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "explicit Markdown-only body exemption" in result.stdout
    assert log.read_text(encoding="utf-8").strip() == "pr create --title Docs only --body-file -"
    assert stdin_capture.read_text(encoding="utf-8") == body.read_text(encoding="utf-8")


def test_open_pr_refreshes_base_before_docs_only_audit(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_docs_only_body(repo)
    _git(repo, "update-ref", "-d", "refs/remotes/origin/main")
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=1)

    result = subprocess.run(
        ["bash", "scripts/open_pr.sh", str(body), "--title", "Docs only"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Refreshing origin/main before PR body audit" in result.stdout
    assert log.read_text(encoding="utf-8").strip() == "pr create --title Docs only --body-file -"
    assert stdin_capture.read_text(encoding="utf-8") == body.read_text(encoding="utf-8")


def _write_fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    copy2(SCRIPT, repo / "scripts" / "open_pr.sh")
    copy2(AUDIT_SCRIPT, repo / "scripts" / "audit_pr_body.py")
    copy2(CHANGE_POLICY_SCRIPT, repo / "scripts" / "_pr_change_policy.py")
    subprocess.run(
        ["git", "init", "--initial-branch", "main"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    remote = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "--bare", str(remote)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "-u", "origin", "main")
    return repo


def _write_body(repo: Path) -> Path:
    body = repo / "body.md"
    _write_plan(repo)
    (repo / "scripts" / "example.py").write_text("print('changed')\n", encoding="utf-8")
    _git(repo, "add", "plans/PR-Test.md", "scripts/example.py")
    _git(repo, "commit", "-qm", "planned change")
    body.write_text(_valid_body(), encoding="utf-8")
    return body


def _write_invalid_body(repo: Path) -> Path:
    _write_body(repo)
    body = repo / "body-invalid.md"
    body.write_text(
        _valid_body().replace(
            "## Cold diff reconstruction\n"
            "- Changed: scripts/example.sh:1 updates the wrapper.\n"
            "- Contract match: traces to the body contract.\n"
            "- Gaps: none.\n\n",
            "",
        ),
        encoding="utf-8",
    )
    return body


def _write_docs_only_body(repo: Path) -> Path:
    doc = repo / "docs" / "example.md"
    doc.parent.mkdir()
    doc.write_text("# docs only\n", encoding="utf-8")
    _git(repo, "add", "docs/example.md")
    _git(repo, "commit", "-qm", "docs only")
    body = repo / "body-docs-only.md"
    body.write_text("Docs-only: true\n\nCorrect a documentation typo.\n", encoding="utf-8")
    return body


def _write_plan(repo: Path) -> None:
    plan = repo / "plans" / "PR-Test.md"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("# Test plan\n", encoding="utf-8")


def _valid_body() -> str:
    return "\n".join([
        "Plan: plans/PR-Test.md",
        "Slice phase: Workflow/process",
        "",
        "One-paragraph why.",
        "",
        "## Intentional",
        "- a trade-off",
        "",
        "## Deferred",
        "- a follow-up",
        "",
        "## Parked hardening",
        "None.",
        "",
        "## Cold diff reconstruction",
        "- Changed: scripts/example.sh:1 updates the wrapper.",
        "- Contract match: traces to the body contract.",
        "- Gaps: none.",
        "",
        "## Verification",
        "- pytest passed",
        "",
        "## Diff size",
        "2 files, +10 / -2",
    ])


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


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
