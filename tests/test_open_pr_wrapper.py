from __future__ import annotations

import os
import fcntl
import subprocess
from pathlib import Path
from shutil import copy2, copytree, ignore_patterns

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "open_pr.sh"
AUDIT_SCRIPT = REPO_ROOT / "scripts" / "audit_pr_body.py"
CHANGE_POLICY_SCRIPT = REPO_ROOT / "scripts" / "_pr_change_policy.py"
LOCAL_REVIEW_SCRIPT = REPO_ROOT / "scripts" / "local_pr_review.sh"


def test_open_pr_create_passes_body_via_stdin_not_path(tmp_path: Path) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=1)

    result = _run(repo, env, body, "--title", "Workflow wrapper", "--base", "main")

    assert result.returncode == 0
    assert log.read_text(encoding="utf-8").strip() == (
        "pr create --title Workflow wrapper --repo canfieldjuan/ATLAS --base main --body-file -"
    )
    assert (repo / "local-review.log").read_text(encoding="utf-8").strip() == f"local_pr_review {body}"
    assert str(body) not in log.read_text(encoding="utf-8")
    assert stdin_capture.read_text(encoding="utf-8") == body.read_text(encoding="utf-8")


def test_open_pr_edit_passes_body_via_stdin_not_path(tmp_path: Path) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=0)

    result = _run(repo, env, body)

    assert result.returncode == 0
    assert log.read_text(encoding="utf-8").strip() == "pr edit 17 --repo canfieldjuan/ATLAS --body-file -"
    assert str(body) not in log.read_text(encoding="utf-8")
    assert stdin_capture.read_text(encoding="utf-8") == body.read_text(encoding="utf-8")


def test_open_pr_existing_pr_rejects_create_only_args(tmp_path: Path) -> None:
    repo, body, env, _, _ = _ready(tmp_path, view_exit=0)

    result = _run(repo, env, body, "--title", "New title")

    assert result.returncode == 2
    assert "PR already exists" in result.stderr


@pytest.mark.parametrize(
    ("args", "extra_env", "expected"),
    [
        (["--body-file", "body.md"], {}, "pass the PR body as BODY_FILE"),
        (["--head", "other"], {}, "refusing target-changing create arg: --head"),
        (["--repo", "other/repo"], {}, "refusing target-changing create arg: --repo"),
        (["--base", "release"], {}, "refusing non-main base: release"),
        (["-Brelease"], {}, "refusing non-main base: release"),
        ([], {"GH_REPO": "other/repo"}, "refusing GH_REPO target override"),
    ],
)
def test_open_pr_rejects_unsafe_inputs_before_gh(
    tmp_path: Path,
    args: list[str],
    extra_env: dict[str, str],
    expected: str,
) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=1)
    env.update(extra_env)

    result = _run(repo, env, body, *args)

    assert result.returncode == 2
    assert expected in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


def test_open_pr_rejects_invalid_body_before_gh(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    _write_body(repo)
    body = repo.parent / "body-invalid.md"
    body.write_text(_valid_body().replace("## Cold diff reconstruction\n- Changed: scripts/example.sh:1 updates the wrapper.\n- Contract match: traces to the body contract.\n- Gaps: none.\n\n", ""), encoding="utf-8")
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=1)

    result = _run(repo, env, body, "--title", "Workflow wrapper")

    assert result.returncode == 1
    assert "missing required section: ## Cold diff reconstruction" in result.stdout
    assert not log.exists()
    assert not stdin_capture.exists()


def test_open_pr_accepts_docs_only_body(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_docs_only_body(repo)
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=1)

    result = _run(repo, env, body, "--title", "Docs only")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "explicit Markdown-only body exemption" in result.stdout
    assert log.read_text(encoding="utf-8").strip() == "pr create --title Docs only --repo canfieldjuan/ATLAS --base main --body-file -"
    assert stdin_capture.read_text(encoding="utf-8") == body.read_text(encoding="utf-8")


def test_open_pr_accepts_normal_ssh_origin_url(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path, origin_url="ssh://git@github.com/canfieldjuan/ATLAS.git")
    body = _write_body(repo)
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=1)

    result = _run(repo, env, body, "--title", "Workflow wrapper")

    assert result.returncode == 0, result.stdout + result.stderr
    assert log.read_text(encoding="utf-8").strip() == (
        "pr create --title Workflow wrapper --repo canfieldjuan/ATLAS --base main --body-file -"
    )
    assert stdin_capture.read_text(encoding="utf-8") == body.read_text(encoding="utf-8")


def test_open_pr_accepts_case_insensitive_repo_identity_after_create(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path, origin_url="git@github.com:canfieldjuan/atlas.git")
    body = _write_body(repo)
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=1)

    result = _run(repo, env, body, "--title", "Workflow wrapper")

    assert result.returncode == 0, result.stdout + result.stderr
    assert log.read_text(encoding="utf-8").strip() == (
        "pr create --title Workflow wrapper --repo canfieldjuan/atlas --base main --body-file -"
    )
    assert stdin_capture.read_text(encoding="utf-8") == body.read_text(encoding="utf-8")


def test_open_pr_rejects_unpublished_current_head_before_gh(tmp_path: Path) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=1)
    _git(repo, "commit", "--allow-empty", "-qm", "unpushed")

    result = _run(repo, env, body, "--title", "Workflow wrapper")

    assert result.returncode == 2
    assert "current HEAD is" in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


def test_open_pr_rejects_missing_remote_branch_before_gh(tmp_path: Path) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=1)
    _git(repo, "push", "-q", "origin", "--delete", "claude/pr-test")

    result = _run(repo, env, body, "--title", "Workflow wrapper")

    assert result.returncode == 1
    assert "failed to refresh origin/claude/pr-test" in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


def test_open_pr_rejects_parallel_local_writer_before_review(tmp_path: Path) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=1)
    lock_path = repo / ".git" / "open_pr_wrapper.lock"

    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = _run(repo, env, body, "--title", "Workflow wrapper")

    assert result.returncode == 2
    assert "another open_pr.sh mutation is already running" in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


def test_open_pr_reads_fetched_head_without_tracking_ref(tmp_path: Path) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=1)
    _git(repo, "config", "remote.origin.fetch", "+refs/heads/main:refs/remotes/origin/main")
    _git(repo, "update-ref", "-d", "refs/remotes/origin/claude/pr-test")

    result = _run(repo, env, body, "--title", "Workflow wrapper")

    assert result.returncode == 0, result.stdout + result.stderr
    assert log.read_text(encoding="utf-8").strip() == (
        "pr create --title Workflow wrapper --repo canfieldjuan/ATLAS --base main --body-file -"
    )
    assert stdin_capture.read_text(encoding="utf-8") == body.read_text(encoding="utf-8")


def test_open_pr_rejects_local_review_failure_before_gh(tmp_path: Path) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=1)
    env["LOCAL_REVIEW_EXIT"] = "42"

    result = _run(repo, env, body, "--title", "Workflow wrapper")

    assert result.returncode == 42
    assert "Running final local PR review before GitHub mutation" in result.stdout
    assert not log.exists()
    assert not stdin_capture.exists()


@pytest.mark.parametrize(
    ("env_flag", "expected"),
    [
        ("LOCAL_REVIEW_ADVANCE_REMOTE", "origin/claude/pr-test changed after review"),
        ("LOCAL_REVIEW_ADVANCE_BOTH", "current HEAD changed after review"),
        ("LOCAL_REVIEW_MUTATE_BODY", "PR body changed after review"),
    ],
)
def test_open_pr_rejects_snapshot_changes_after_local_review(
    tmp_path: Path,
    env_flag: str,
    expected: str,
) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=1)
    env[env_flag] = "1"

    result = _run(repo, env, body, "--title", "Workflow wrapper")

    assert result.returncode == 2
    assert expected in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


@pytest.mark.parametrize(("view_exit", "args"), [(1, ["--title", "Workflow wrapper"]), (0, [])])
def test_open_pr_real_local_review_failure_blocks_mutation_branches(
    tmp_path: Path,
    view_exit: int,
    args: list[str],
) -> None:
    repo = _write_fixture_repo(tmp_path, real_local_review=True)
    body = _write_body(repo)
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=view_exit)

    result = _run(repo, env, body, *args)

    assert result.returncode != 0
    assert "Running final local PR review before GitHub mutation" in result.stdout
    assert "worktree has uncommitted changes" not in result.stderr
    assert "==> Pre-push audit wrapper" in result.stdout
    assert "==> Plan shape: plans/PR-Test.md" in result.stdout
    assert "plans/PR-Test.md: missing Ownership lane" in result.stdout
    assert "plans/PR-Test.md: missing Slice phase" in result.stdout
    assert "No such file or directory" not in result.stderr
    gh_log = log.read_text(encoding="utf-8") if log.exists() else ""
    assert "pr create" not in gh_log
    assert "pr edit" not in gh_log
    captured_stdin = stdin_capture.read_text(encoding="utf-8") if stdin_capture.exists() else ""
    assert captured_stdin == ""


def test_open_pr_rejects_mismatched_existing_pr_identity_before_edit(tmp_path: Path) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=1)
    env["GH_PR_LIST_JSON"] = (
        '[{"number":17,"headRefName":"claude/pr-test","baseRefName":"release",'
        '"headRepository":{"nameWithOwner":"canfieldjuan/ATLAS"},"isCrossRepository":false}]'
    )

    result = _run(repo, env, body)

    assert result.returncode == 1
    assert "outside canfieldjuan/ATLAS->main" in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


def test_open_pr_rejects_stale_existing_pr_head_before_edit(tmp_path: Path) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=0)
    stale_head = subprocess.check_output(["git", "rev-parse", "HEAD^"], cwd=repo, text=True).strip()
    env["GH_PR_LIST_JSON"] = (
        f'[{{"number":17,"headRefName":"claude/pr-test","headRefOid":"{stale_head}",'
        '"baseRefName":"main","headRepository":{"nameWithOwner":"canfieldjuan/ATLAS"},'
        '"isCrossRepository":false}]'
    )

    result = _run(repo, env, body)

    assert result.returncode == 2
    assert "existing PR head does not match reviewed head" in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


def _ready(tmp_path: Path, *, view_exit: int) -> tuple[Path, Path, dict[str, str], Path, Path]:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=view_exit)
    return repo, body, env, log, stdin_capture


def _run(repo: Path, env: dict[str, str], body: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "scripts/open_pr.sh", str(body), *args],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _write_fixture_repo(
    tmp_path: Path,
    *,
    real_local_review: bool = False,
    origin_url: str = "git@github.com:canfieldjuan/ATLAS.git",
) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    if real_local_review:
        copytree(REPO_ROOT / "scripts", repo / "scripts", dirs_exist_ok=True, ignore=ignore_patterns("__pycache__"))
        copytree(REPO_ROOT / "docs", repo / "docs", dirs_exist_ok=True, ignore=ignore_patterns("__pycache__"))
        copytree(REPO_ROOT / ".github", repo / ".github", dirs_exist_ok=True, ignore=ignore_patterns("__pycache__"))
        copytree(REPO_ROOT / "extracted", repo / "extracted", dirs_exist_ok=True, ignore=ignore_patterns("__pycache__"))
        for package_root in REPO_ROOT.glob("extracted_*"):
            if package_root.is_dir():
                copytree(package_root, repo / package_root.name, dirs_exist_ok=True, ignore=ignore_patterns("__pycache__"))
        copytree(REPO_ROOT / "atlas_brain", repo / "atlas_brain", dirs_exist_ok=True, ignore=ignore_patterns("__pycache__"))
        copy2(REPO_ROOT / "AGENTS.md", repo / "AGENTS.md")
        copy2(REPO_ROOT / "CLAUDE.md", repo / "CLAUDE.md")
    else:
        copy2(SCRIPT, repo / "scripts" / "open_pr.sh")
        copy2(AUDIT_SCRIPT, repo / "scripts" / "audit_pr_body.py")
        copy2(CHANGE_POLICY_SCRIPT, repo / "scripts" / "_pr_change_policy.py")
        (repo / "scripts" / "local_pr_review.sh").write_text(
            """#!/usr/bin/env bash
set -euo pipefail
printf 'local_pr_review %s\\n' "${ATLAS_CURRENT_PR_BODY_FILE:-}" >> local-review.log
if [ "${LOCAL_REVIEW_ADVANCE_REMOTE:-}" = "1" ]; then
    git --git-dir="$(git config --get atlas.testRemoteGitDir)" update-ref refs/heads/claude/pr-test "$(git rev-parse HEAD^)"
fi
if [ "${LOCAL_REVIEW_ADVANCE_BOTH:-}" = "1" ]; then
    printf 'post-review\\n' >> scripts/example.py
    git add scripts/example.py
    git -c user.email=t@example.com -c user.name=t commit -qm post-review
    git push -q origin HEAD:claude/pr-test
fi
if [ "${LOCAL_REVIEW_MUTATE_BODY:-}" = "1" ]; then
    printf '\\npost-review body mutation\\n' >> "${ATLAS_CURRENT_PR_BODY_FILE}"
fi
exit "${LOCAL_REVIEW_EXIT:-0}"
""",
            encoding="utf-8",
        )
    (repo / "scripts" / "local_pr_review.sh").chmod(0o755)
    subprocess.run(["git", "init", "--initial-branch", "main"], cwd=repo, check=True, capture_output=True, text=True)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    remote = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True, text=True)
    _git(repo, "config", f"url.{remote}.insteadOf", origin_url)
    _git(repo, "config", "atlas.testRemoteGitDir", str(remote))
    _git(repo, "remote", "add", "origin", origin_url)
    _git(repo, "push", "-q", "-u", "origin", "main")
    _git(repo, "switch", "-c", "claude/pr-test")
    return repo


def _write_body(repo: Path) -> Path:
    (repo / "plans").mkdir(exist_ok=True)
    (repo / "plans" / "PR-Test.md").write_text("# Test plan\n", encoding="utf-8")
    (repo / "scripts" / "example.py").write_text("print('changed')\n", encoding="utf-8")
    _git(repo, "add", "plans/PR-Test.md", "scripts/example.py")
    _git(repo, "commit", "-qm", "planned change")
    _git(repo, "push", "-q", "-u", "origin", "HEAD")
    body = repo.parent / "body.md"
    body.write_text(_valid_body(), encoding="utf-8")
    return body


def _write_docs_only_body(repo: Path) -> Path:
    doc = repo / "docs" / "example.md"
    doc.parent.mkdir()
    doc.write_text("# docs only\n", encoding="utf-8")
    _git(repo, "add", "docs/example.md")
    _git(repo, "commit", "-qm", "docs only")
    _git(repo, "push", "-q", "-u", "origin", "HEAD")
    body = repo.parent / "body-docs-only.md"
    body.write_text("Docs-only: true\n\nCorrect a documentation typo.\n", encoding="utf-8")
    return body


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


def _fake_gh_env(tmp_path: Path, *, view_exit: int) -> tuple[dict[str, str], Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "gh-argv.txt"
    stdin_capture = tmp_path / "gh-stdin.txt"
    created_flag = tmp_path / "gh-created-pr"
    gh = bin_dir / "gh"
    gh.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [ "$1" = "repo" ] && [ "$2" = "view" ]; then
    printf 'canfieldjuan/ATLAS\\n'
    exit 0
fi
if [ "$1" = "pr" ] && [ "$2" = "list" ]; then
    if [ -n "${GH_PR_LIST_JSON:-}" ]; then
        printf '%s\\n' "${GH_PR_LIST_JSON}"
        exit 0
    fi
    if [ "${GH_VIEW_EXIT}" = "0" ] || [ -f "${GH_CREATED_PR_FLAG}" ]; then
        printf '[{"number":17,"headRefName":"claude/pr-test","headRefOid":"%s","baseRefName":"main","headRepository":{"nameWithOwner":"canfieldjuan/ATLAS"},"isCrossRepository":false}]\\n' "$(git rev-parse HEAD)"
    else
        printf '[]\\n'
    fi
    exit 0
fi
printf '%s\\n' "$*" > "${GH_ARGV_LOG}"
cat > "${GH_STDIN_CAPTURE}"
if [ "$1" = "pr" ] && [ "$2" = "create" ]; then
    : > "${GH_CREATED_PR_FLAG}"
fi
""",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "GH_VIEW_EXIT": str(view_exit),
        "GH_ARGV_LOG": str(log),
        "GH_STDIN_CAPTURE": str(stdin_capture),
        "GH_CREATED_PR_FLAG": str(created_flag),
        "PYTHONDONTWRITEBYTECODE": "1",
    }, log, stdin_capture


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
