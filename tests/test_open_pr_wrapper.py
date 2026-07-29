from __future__ import annotations

import os
import subprocess
import hashlib
from pathlib import Path
from shutil import copy2


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "open_pr.sh"
AUDIT_SCRIPT = REPO_ROOT / "scripts" / "audit_pr_body.py"
CHANGE_POLICY_SCRIPT = REPO_ROOT / "scripts" / "_pr_change_policy.py"


def test_open_pr_create_passes_body_via_stdin_not_path(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    _write_review_proof(repo, body)
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
        "pr create --head claude/pr-test --title Workflow wrapper --base main --body-file -"
    )
    assert str(body) not in log.read_text(encoding="utf-8")
    assert stdin_capture.read_text(encoding="utf-8") == body.read_text(encoding="utf-8")


def test_open_pr_uses_snapshot_when_body_mutates_during_gh_create(
    tmp_path: Path,
) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    reviewed_body = body.read_text(encoding="utf-8")
    _write_review_proof(repo, body)
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=1)
    env["GH_MUTATE_BODY"] = str(body)

    result = subprocess.run(
        ["bash", "scripts/open_pr.sh", str(body), "--title", "Workflow wrapper"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        log.read_text(encoding="utf-8").strip()
        == "pr create --head claude/pr-test --title Workflow wrapper --body-file -"
    )
    assert stdin_capture.read_text(encoding="utf-8") == reviewed_body
    assert body.read_text(encoding="utf-8") != reviewed_body


def test_open_pr_edit_passes_body_via_stdin_not_path(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    _write_review_proof(repo, body)
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
    assert log.read_text(encoding="utf-8").strip() == "pr edit 123 --body-file -"
    assert str(body) not in log.read_text(encoding="utf-8")
    assert stdin_capture.read_text(encoding="utf-8") == body.read_text(encoding="utf-8")


def test_open_pr_existing_pr_rejects_create_only_args(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    _write_review_proof(repo, body)
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


def test_open_pr_numeric_branch_edits_matched_head_pr_not_numeric_selector(
    tmp_path: Path,
) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body_for_branch(repo, "123")
    _write_review_proof(repo, body)
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=0)
    env["GH_PR_NUMBER"] = "456"

    result = subprocess.run(
        ["bash", "scripts/open_pr.sh", str(body)],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert log.read_text(encoding="utf-8").strip() == "pr edit 456 --body-file -"
    assert stdin_capture.read_text(encoding="utf-8") == body.read_text(encoding="utf-8")


def test_open_pr_ignores_same_branch_fork_pr_before_create(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    _write_review_proof(repo, body)
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=0)
    env["GH_PR_CROSS_REPOSITORY"] = "true"
    env["GH_PR_HEAD_OWNER"] = "fork-owner"

    result = subprocess.run(
        ["bash", "scripts/open_pr.sh", str(body), "--title", "Workflow wrapper"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        log.read_text(encoding="utf-8").strip()
        == "pr create --head claude/pr-test --title Workflow wrapper --body-file -"
    )
    assert stdin_capture.read_text(encoding="utf-8") == body.read_text(encoding="utf-8")


def test_open_pr_create_binds_captured_head_branch(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    _write_review_proof(repo, body)
    env, log, _ = _fake_gh_env(tmp_path, view_exit=1)

    result = subprocess.run(
        ["bash", "scripts/open_pr.sh", str(body), "--title", "Workflow wrapper"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert log.read_text(encoding="utf-8").startswith("pr create --head claude/pr-test ")


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


def test_open_pr_rejects_head_target_override_before_gh(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    _write_review_proof(repo, body)
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=1)

    result = subprocess.run(
        ["bash", "scripts/open_pr.sh", str(body), "--head", "other-branch"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "refusing target-changing create arg: --head" in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


def test_open_pr_rejects_repo_target_override_before_gh(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    _write_review_proof(repo, body)
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=1)

    result = subprocess.run(
        ["bash", "scripts/open_pr.sh", str(body), "--repo", "canfieldjuan/OTHER"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "refusing target-changing create arg: --repo" in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


def test_open_pr_rejects_gh_repo_environment_target_override_before_gh(
    tmp_path: Path,
) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    _write_review_proof(repo, body)
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=1)
    env["GH_REPO"] = "canfieldjuan/OTHER"

    result = subprocess.run(
        ["bash", "scripts/open_pr.sh", str(body), "--title", "Workflow wrapper"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "refusing GH_REPO target override" in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


def test_open_pr_rejects_non_main_base_before_gh(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    _write_review_proof(repo, body)
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=1)

    result = subprocess.run(
        ["bash", "scripts/open_pr.sh", str(body), "--base", "release"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "refusing non-main base: release" in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


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
    _write_review_proof(repo, body)
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
    assert (
        log.read_text(encoding="utf-8").strip()
        == "pr create --head claude/pr-test --title Docs only --body-file -"
    )
    assert stdin_capture.read_text(encoding="utf-8") == body.read_text(encoding="utf-8")


def test_open_pr_refreshes_base_before_docs_only_audit(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_docs_only_body(repo)
    _write_review_proof(repo, body)
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
    assert (
        log.read_text(encoding="utf-8").strip()
        == "pr create --head claude/pr-test --title Docs only --body-file -"
    )
    assert stdin_capture.read_text(encoding="utf-8") == body.read_text(encoding="utf-8")


def test_open_pr_requires_local_review_proof_before_gh(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=1)

    result = subprocess.run(
        ["bash", "scripts/open_pr.sh", str(body), "--title", "Workflow wrapper"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "missing local review proof" in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


def test_open_pr_rejects_stale_head_proof_before_gh(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    _write_review_proof(repo, body)
    (repo / "scripts" / "another.py").write_text("print('later')\n", encoding="utf-8")
    _git(repo, "add", "scripts/another.py")
    _git(repo, "commit", "-qm", "later change")
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=1)

    result = subprocess.run(
        ["bash", "scripts/open_pr.sh", str(body), "--title", "Workflow wrapper"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "stale local review proof" in result.stderr
    assert "Run scripts/push_pr.sh again" in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


def test_open_pr_rejects_stale_body_proof_before_gh(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    _write_review_proof(repo, body)
    body.write_text(body.read_text(encoding="utf-8") + "\nRegenerated body.\n", encoding="utf-8")
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=1)

    result = subprocess.run(
        ["bash", "scripts/open_pr.sh", str(body), "--title", "Workflow wrapper"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "PR body changed after local review" in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


def test_open_pr_rejects_stale_base_proof_before_gh(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    _write_review_proof(repo, body, base_sha="0" * 40)
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=1)

    result = subprocess.run(
        ["bash", "scripts/open_pr.sh", str(body), "--title", "Workflow wrapper"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "expected origin/main" in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


def test_open_pr_rejects_same_head_different_branch_proof_before_gh(
    tmp_path: Path,
) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    _write_review_proof(repo, body)
    _git(repo, "checkout", "-b", "claude/pr-other")
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=1)

    result = subprocess.run(
        ["bash", "scripts/open_pr.sh", str(body), "--title", "Workflow wrapper"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "expected branch claude/pr-other" in result.stderr
    assert "found claude/pr-test" in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


def test_open_pr_rejects_stale_remote_head_before_gh(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    _write_review_proof(repo, body)
    _git(repo, "push", "-q", "--force", "origin", "HEAD^:refs/heads/claude/pr-test")
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=1)

    result = subprocess.run(
        ["bash", "scripts/open_pr.sh", str(body), "--title", "Workflow wrapper"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "stale local review proof" in result.stderr
    assert "origin/claude/pr-test" in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


def test_open_pr_rechecks_remote_head_after_pr_view_before_create(
    tmp_path: Path,
) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    _write_review_proof(repo, body)
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=1)
    env["GH_FORCE_PUSH_DURING_VIEW"] = "1"
    env["GH_FAKE_REPO"] = str(repo)

    result = subprocess.run(
        ["bash", "scripts/open_pr.sh", str(body), "--title", "Workflow wrapper"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "stale local review proof" in result.stderr
    assert "origin/claude/pr-test" in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


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
    return _write_body_for_branch(repo, "claude/pr-test")


def _write_body_for_branch(repo: Path, branch: str) -> Path:
    body = repo / "body.md"
    _checkout_pr_branch(repo, branch)
    _write_plan(repo)
    (repo / "scripts" / "example.py").write_text("print('changed')\n", encoding="utf-8")
    _git(repo, "add", "plans/PR-Test.md", "scripts/example.py")
    _git(repo, "commit", "-qm", "planned change")
    _git(repo, "push", "-q", "-u", "origin", "HEAD")
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
    _checkout_pr_branch(repo, "claude/pr-test")
    doc = repo / "docs" / "example.md"
    doc.parent.mkdir()
    doc.write_text("# docs only\n", encoding="utf-8")
    _git(repo, "add", "docs/example.md")
    _git(repo, "commit", "-qm", "docs only")
    _git(repo, "push", "-q", "-u", "origin", "HEAD")
    body = repo / "body-docs-only.md"
    body.write_text("Docs-only: true\n\nCorrect a documentation typo.\n", encoding="utf-8")
    return body


def _write_plan(repo: Path) -> None:
    plan = repo / "plans" / "PR-Test.md"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("# Test plan\n", encoding="utf-8")


def _checkout_pr_branch(repo: Path, target: str) -> None:
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if branch != target:
        _git(repo, "checkout", "-b", target)


def _write_review_proof(repo: Path, body: Path, *, base_sha: str | None = None) -> Path:
    proof = Path(
        subprocess.run(
            ["git", "rev-parse", "--git-path", "atlas-local-pr-review-proof"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if not proof.is_absolute():
        proof = repo / proof
    proof.parent.mkdir(parents=True, exist_ok=True)
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if base_sha is None:
        base_sha = subprocess.run(
            ["git", "rev-parse", "origin/main"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    body_hash = hashlib.sha256(body.read_bytes()).hexdigest()
    proof.write_text(
        (
            f"branch={branch}\n"
            f"head_sha={head}\n"
            f"base_sha={base_sha}\n"
            f"body_sha256={body_hash}\n"
        ),
        encoding="utf-8",
    )
    return proof


def _valid_body() -> str:
    return "\n".join([
        "Plan: plans/PR-Test.md",
        "Slice phase: Workflow/process",
        "Ownership lane: dev-workflow/process-gate-enrollment",
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
if [ "$1" = "pr" ] && [ "$2" = "list" ]; then
    head=""
    while [ "$#" -gt 0 ]; do
        if [ "${1:-}" = "--head" ]; then
            head="${2:-}"
            shift 2
            continue
        fi
        shift
    done
    if [ "${GH_FORCE_PUSH_DURING_VIEW:-}" = "1" ]; then
        git -C "${GH_FAKE_REPO}" push -q --force origin HEAD^:refs/heads/claude/pr-test
    fi
    if [ "${GH_VIEW_EXIT}" = "0" ]; then
        printf '[{"number":%s,"headRefName":"%s","headRepository":{"name":"%s"},"headRepositoryOwner":{"login":"%s"},"isCrossRepository":%s}]\\n' "${GH_PR_NUMBER:-123}" "$head" "${GH_PR_HEAD_REPO:-ATLAS}" "${GH_PR_HEAD_OWNER:-canfieldjuan}" "${GH_PR_CROSS_REPOSITORY:-false}"
    else
        printf '[]\\n'
    fi
    exit 0
fi
if [ "$1" = "repo" ] && [ "$2" = "view" ]; then
    printf '{"name":"ATLAS","owner":{"login":"canfieldjuan"}}\\n'
    exit 0
fi
printf '%s\\n' "$*" > "${GH_ARGV_LOG}"
if [ -n "${GH_MUTATE_BODY:-}" ]; then
    printf '\\nmutated during gh\\n' >> "${GH_MUTATE_BODY}"
fi
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
