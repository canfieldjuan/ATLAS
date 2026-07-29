from __future__ import annotations

import os
import subprocess
import hashlib
from pathlib import Path
from shutil import copy2, which


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "push_pr.sh"
AUDIT_SCRIPT = REPO_ROOT / "scripts" / "audit_pr_body.py"
CHANGE_POLICY_SCRIPT = REPO_ROOT / "scripts" / "_pr_change_policy.py"


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
    assert "DRY RUN: git fetch --quiet origin main" in result.stdout
    assert f"ATLAS_CURRENT_PR_BODY_FILE={body}" in result.stdout
    assert f"--current-pr-body-file {body}" in result.stdout
    assert "git push -u origin HEAD" in result.stdout


def test_push_pr_dry_run_with_managed_hook_uses_immutable_wrapper_review(
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
    assert "DRY RUN: git fetch --quiet origin main" in result.stdout
    assert "immutable captured-head worktree" in result.stdout
    assert "managed pre-push hook will run local PR review once" not in result.stdout
    assert f"ATLAS_CURRENT_PR_BODY_FILE={body}" in result.stdout
    assert "ATLAS_SKIP_LOCAL_PR_REVIEW=1" in result.stdout
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
    assert "DRY RUN: git fetch --quiet origin main" in result.stdout
    assert f"ATLAS_CURRENT_PR_BODY_FILE={body}" in result.stdout
    assert f"--current-pr-body-file {body}" in result.stdout
    assert "managed pre-push hook will run local PR review once" not in result.stdout


def test_push_pr_dry_run_with_non_executable_managed_hook_keeps_wrapper_review(
    tmp_path: Path,
) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    _write_managed_hook(repo, executable=False)
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
    assert "DRY RUN: git fetch --quiet origin main" in result.stdout
    assert f"ATLAS_CURRENT_PR_BODY_FILE={body}" in result.stdout
    assert "immutable captured-head worktree" in result.stdout
    assert "managed pre-push hook will run local PR review once" not in result.stdout


def test_push_pr_docs_only_dry_run_does_not_fetch_or_audit(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_docs_only_body(repo)

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "-u", "origin", "HEAD"],
        cwd=repo,
        env={**os.environ, "ATLAS_PUSH_PR_DRY_RUN": "1"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "DRY RUN: git fetch --quiet origin main" in result.stdout


def test_push_pr_refreshes_before_docs_only_body_audit(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_docs_only_body(repo)
    order_log = _order_log(repo)
    _write_body_audit_recorder(repo)
    _write_local_review(repo)
    fake_bin = _write_fake_git(repo, order_log)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GIT_TOPLEVEL": str(repo),
        "FAKE_GIT_LOG": str(order_log),
        "ORDER_LOG": str(order_log),
    }

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "-u", "origin", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    lines = order_log.read_text(encoding="utf-8").splitlines()
    assert lines.index("git fetch --quiet origin main") < lines.index("body-audit")


def test_push_pr_refreshes_base_before_wrapper_review_and_push(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    order_log = _order_log(repo)
    _write_local_review(repo)
    fake_bin = _write_fake_git(repo, order_log)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GIT_TOPLEVEL": str(repo),
        "FAKE_GIT_LOG": str(order_log),
        "ORDER_LOG": str(order_log),
    }

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "-u", "origin", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    lines = order_log.read_text(encoding="utf-8").splitlines()
    local_review_index = next(
        index for index, line in enumerate(lines) if line.startswith("local-review ")
    )
    assert lines.index("git fetch --quiet origin main") < local_review_index
    assert local_review_index < lines.index("git push -u origin HEAD")


def test_push_pr_sets_github_head_ref_for_detached_immutable_review(
    tmp_path: Path,
) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    order_log = _order_log(repo)
    _write_local_review(repo)
    fake_bin = _write_fake_git(repo, order_log)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GIT_TOPLEVEL": str(repo),
        "FAKE_GIT_LOG": str(order_log),
        "ORDER_LOG": str(order_log),
    }

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "-u", "origin", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    lines = order_log.read_text(encoding="utf-8").splitlines()
    assert "github-head-ref claude/pr-test" in lines


def test_push_pr_refreshes_base_before_immutable_review_and_managed_hook_push(
    tmp_path: Path,
) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    order_log = _order_log(repo)
    _write_local_review(repo)
    _write_managed_hook(repo)
    fake_bin = _write_fake_git(repo, order_log)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GIT_TOPLEVEL": str(repo),
        "FAKE_GIT_LOG": str(order_log),
        "ORDER_LOG": str(order_log),
    }

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "-u", "origin", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    lines = order_log.read_text(encoding="utf-8").splitlines()
    local_review_index = next(
        index for index, line in enumerate(lines) if line.startswith("local-review ")
    )
    assert lines.index("git fetch --quiet origin main") < lines.index(
        "git push -u origin HEAD"
    )
    assert lines.index("git fetch --quiet origin main") < local_review_index
    assert local_review_index < lines.index("git push -u origin HEAD")


def test_push_pr_writes_review_proof_after_successful_push(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    order_log = _order_log(repo)
    _write_local_review(repo)
    fake_bin = _write_fake_git(repo, order_log)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GIT_TOPLEVEL": str(repo),
        "FAKE_GIT_LOG": str(order_log),
        "ORDER_LOG": str(order_log),
    }

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "-u", "origin", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    proof = _proof_path(repo)
    assert proof.is_file()
    proof_text = proof.read_text(encoding="utf-8")
    assert "branch=claude/pr-test" in proof_text
    assert f"head_sha={_head(repo)}" in proof_text
    assert f"base_sha={_base(repo)}" in proof_text
    assert f"body_sha256={hashlib.sha256(body.read_bytes()).hexdigest()}" in proof_text
    assert "Wrote local review proof" in result.stdout


def test_push_pr_uses_reviewed_body_snapshot_when_body_mutates_during_push(
    tmp_path: Path,
) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    reviewed_body = body.read_bytes()
    reviewed_hash = hashlib.sha256(reviewed_body).hexdigest()
    order_log = _order_log(repo)
    _write_local_review(repo)
    fake_bin = _write_fake_git(repo, order_log)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GIT_TOPLEVEL": str(repo),
        "FAKE_GIT_LOG": str(order_log),
        "FAKE_GIT_MUTATE_BODY": str(body),
        "ORDER_LOG": str(order_log),
    }

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "-u", "origin", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert body.read_bytes() != reviewed_body
    proof_text = _proof_path(repo).read_text(encoding="utf-8")
    assert f"body_sha256={reviewed_hash}" in proof_text
    assert hashlib.sha256(body.read_bytes()).hexdigest() not in proof_text


def test_push_pr_reviews_immutable_captured_head_despite_aba_checkout(
    tmp_path: Path,
) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (repo / "scripts" / "aba.py").write_text("print('aba')\n", encoding="utf-8")
    _git(repo, "add", "scripts/aba.py")
    _git(repo, "commit", "-qm", "aba other commit")
    other_head = _head(repo)
    _git(repo, "reset", "--hard", "HEAD^")
    order_log = _order_log(repo)
    _write_aba_local_review(repo)
    reviewed_head = _head(repo)
    fake_bin = _write_fake_git(repo, order_log)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GIT_TOPLEVEL": str(repo),
        "FAKE_GIT_LOG": str(order_log),
        "ABA_REPO": str(repo),
        "ABA_BRANCH": branch,
        "ABA_OTHER_SHA": other_head,
        "ORDER_LOG": str(order_log),
    }

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "-u", "origin", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    lines = order_log.read_text(encoding="utf-8").splitlines()
    assert f"local-review-head {reviewed_head}" in lines
    assert f"local-review-original-after {reviewed_head}" in lines
    assert f"head_sha={reviewed_head}" in _proof_path(repo).read_text(encoding="utf-8")


def test_push_pr_rejects_staged_source_worktree_before_immutable_review(
    tmp_path: Path,
) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    order_log = _order_log(repo)
    _write_local_review(repo)
    fake_bin = _write_fake_git(repo, order_log)
    (repo / "scripts" / "example.py").write_text("print('staged')\n", encoding="utf-8")
    _git(repo, "add", "scripts/example.py")
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GIT_TOPLEVEL": str(repo),
        "FAKE_GIT_LOG": str(order_log),
        "ORDER_LOG": str(order_log),
    }

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "-u", "origin", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "source worktree has uncommitted changes" in result.stderr
    assert "M  scripts/example.py" in result.stderr
    lines = order_log.read_text(encoding="utf-8").splitlines()
    assert not any(line.startswith("local-review ") for line in lines)
    assert "git push -u origin HEAD" not in lines
    assert not _proof_path(repo).exists()


def test_push_pr_rejects_unstaged_source_worktree_before_immutable_review(
    tmp_path: Path,
) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    order_log = _order_log(repo)
    _write_local_review(repo)
    fake_bin = _write_fake_git(repo, order_log)
    (repo / "scripts" / "example.py").write_text("print('unstaged')\n", encoding="utf-8")
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GIT_TOPLEVEL": str(repo),
        "FAKE_GIT_LOG": str(order_log),
        "ORDER_LOG": str(order_log),
    }

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "-u", "origin", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "source worktree has uncommitted changes" in result.stderr
    assert " M scripts/example.py" in result.stderr
    lines = order_log.read_text(encoding="utf-8").splitlines()
    assert not any(line.startswith("local-review ") for line in lines)
    assert "git push -u origin HEAD" not in lines
    assert not _proof_path(repo).exists()


def test_push_pr_rejects_untracked_source_worktree_before_immutable_review(
    tmp_path: Path,
) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    order_log = _order_log(repo)
    _write_local_review(repo)
    fake_bin = _write_fake_git(repo, order_log)
    (repo / "scratch.txt").write_text("not reviewed\n", encoding="utf-8")
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GIT_TOPLEVEL": str(repo),
        "FAKE_GIT_LOG": str(order_log),
        "ORDER_LOG": str(order_log),
    }

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "-u", "origin", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "source worktree has uncommitted changes" in result.stderr
    assert "?? scratch.txt" in result.stderr
    lines = order_log.read_text(encoding="utf-8").splitlines()
    assert not any(line.startswith("local-review ") for line in lines)
    assert "git push -u origin HEAD" not in lines
    assert not _proof_path(repo).exists()


def test_push_pr_rejects_source_worktree_dirtied_during_push_before_proof(
    tmp_path: Path,
) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    order_log = _order_log(repo)
    _write_local_review(repo)
    fake_bin = _write_fake_git(repo, order_log)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GIT_TOPLEVEL": str(repo),
        "FAKE_GIT_LOG": str(order_log),
        "FAKE_GIT_DIRTY_SOURCE_DURING_PUSH": "1",
        "ORDER_LOG": str(order_log),
    }

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "-u", "origin", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "source worktree has uncommitted changes" in result.stderr
    assert "?? pushed-dirty.txt" in result.stderr
    lines = order_log.read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("local-review ") for line in lines)
    assert "git push -u origin HEAD" in lines
    assert not _proof_path(repo).exists()


def test_push_pr_rejects_git_dry_run_before_review_or_push(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    order_log = _order_log(repo)
    _write_local_review(repo)
    fake_bin = _write_fake_git(repo, order_log)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GIT_TOPLEVEL": str(repo),
        "FAKE_GIT_LOG": str(order_log),
        "ORDER_LOG": str(order_log),
    }

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "--dry-run", "origin", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "refusing git push dry-run" in result.stderr
    assert "git push" not in order_log.read_text(encoding="utf-8")
    assert not _proof_path(repo).exists()


def test_push_pr_rejects_abbreviated_dry_run_before_review_or_push(
    tmp_path: Path,
) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    order_log = _order_log(repo)
    _write_local_review(repo)
    fake_bin = _write_fake_git(repo, order_log)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GIT_TOPLEVEL": str(repo),
        "FAKE_GIT_LOG": str(order_log),
        "ORDER_LOG": str(order_log),
    }

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "--dry", "origin", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "refusing git push dry-run" in result.stderr
    assert "git push" not in order_log.read_text(encoding="utf-8")
    assert not _proof_path(repo).exists()


def test_push_pr_rejects_bundled_short_dry_run_before_review_or_push(
    tmp_path: Path,
) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    order_log = _order_log(repo)
    _write_local_review(repo)
    fake_bin = _write_fake_git(repo, order_log)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GIT_TOPLEVEL": str(repo),
        "FAKE_GIT_LOG": str(order_log),
        "ORDER_LOG": str(order_log),
    }

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "-un", "origin", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "refusing git push dry-run" in result.stderr
    assert "git push" not in order_log.read_text(encoding="utf-8")
    assert not _proof_path(repo).exists()


def test_push_pr_rejects_unrelated_refspec_before_review_or_push(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    order_log = _order_log(repo)
    _write_local_review(repo)
    fake_bin = _write_fake_git(repo, order_log)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GIT_TOPLEVEL": str(repo),
        "FAKE_GIT_LOG": str(order_log),
        "ORDER_LOG": str(order_log),
    }

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "origin", "HEAD:other-branch"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "does not publish the current branch HEAD" in result.stderr
    assert "git push" not in order_log.read_text(encoding="utf-8")
    assert not _proof_path(repo).exists()


def test_push_pr_rejects_non_origin_positional_remote_before_review_or_push(
    tmp_path: Path,
) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    order_log = _order_log(repo)
    _write_local_review(repo)
    fake_bin = _write_fake_git(repo, order_log)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GIT_TOPLEVEL": str(repo),
        "FAKE_GIT_LOG": str(order_log),
        "ORDER_LOG": str(order_log),
    }

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "upstream", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "refusing git push remote upstream" in result.stderr
    lines = order_log.read_text(encoding="utf-8").splitlines()
    assert not any(line.startswith("local-review ") for line in lines)
    assert "git push upstream HEAD" not in lines
    assert not _proof_path(repo).exists()


def test_push_pr_does_not_write_review_proof_when_remote_head_is_stale(
    tmp_path: Path,
) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    order_log = _order_log(repo)
    _write_local_review(repo)
    fake_bin = _write_fake_git(repo, order_log)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GIT_TOPLEVEL": str(repo),
        "FAKE_GIT_LOG": str(order_log),
        "FAKE_GIT_SKIP_REMOTE_UPDATE": "1",
        "ORDER_LOG": str(order_log),
    }

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "-u", "origin", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "pushed branch proof missing refs/remotes/origin/claude/pr-test" in result.stderr
    assert "git push -u origin HEAD" in order_log.read_text(encoding="utf-8")
    assert not _proof_path(repo).exists()


def test_push_pr_does_not_write_review_proof_when_base_moves_during_push(
    tmp_path: Path,
) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    order_log = _order_log(repo)
    _write_local_review(repo)
    fake_bin = _write_fake_git(repo, order_log)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GIT_TOPLEVEL": str(repo),
        "FAKE_GIT_LOG": str(order_log),
        "FAKE_GIT_ADVANCE_BASE_DURING_PUSH": "1",
        "ORDER_LOG": str(order_log),
    }

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "-u", "origin", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "origin/main changed" in result.stderr
    assert "git push -u origin HEAD" in order_log.read_text(encoding="utf-8")
    assert not _proof_path(repo).exists()


def test_push_pr_does_not_write_review_proof_when_push_fails(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    order_log = _order_log(repo)
    _write_local_review(repo)
    fake_bin = _write_fake_git(repo, order_log)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GIT_TOPLEVEL": str(repo),
        "FAKE_GIT_LOG": str(order_log),
        "FAKE_GIT_PUSH_FAIL": "1",
        "ORDER_LOG": str(order_log),
    }

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "-u", "origin", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 42
    assert "git push -u origin HEAD" in order_log.read_text(encoding="utf-8")
    assert not _proof_path(repo).exists()


def test_push_pr_fetch_failure_aborts_before_review_or_push(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    order_log = _order_log(repo)
    _write_local_review(repo)
    fake_bin = _write_fake_git(repo, order_log)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GIT_TOPLEVEL": str(repo),
        "FAKE_GIT_LOG": str(order_log),
        "FAKE_GIT_FETCH_FAIL": "1",
        "ORDER_LOG": str(order_log),
    }

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "-u", "origin", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "failed to refresh origin/main" in result.stderr
    lines = order_log.read_text(encoding="utf-8").splitlines()
    assert "git fetch --quiet origin main" in lines
    assert "git push -u origin HEAD" not in lines
    assert not any(line.startswith("local-review ") for line in lines)


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


def test_push_pr_rejects_abbreviated_no_verify_before_managed_hook_push(
    tmp_path: Path,
) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    order_log = _order_log(repo)
    _write_local_review(repo)
    _write_managed_hook(repo)
    fake_bin = _write_fake_git(repo, order_log)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GIT_TOPLEVEL": str(repo),
        "FAKE_GIT_LOG": str(order_log),
        "ORDER_LOG": str(order_log),
    }

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "--no-veri", "-u", "origin", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Git-abbreviated spellings" in result.stderr
    assert "git push" not in order_log.read_text(encoding="utf-8")
    assert not _proof_path(repo).exists()


def test_push_pr_allows_no_verbose_without_bypassing_review(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    order_log = _order_log(repo)
    _write_local_review(repo)
    fake_bin = _write_fake_git(repo, order_log)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GIT_TOPLEVEL": str(repo),
        "FAKE_GIT_LOG": str(order_log),
        "ORDER_LOG": str(order_log),
    }

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "--no-verbose", "-u", "origin", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "git push --no-verbose -u origin HEAD" in order_log.read_text(encoding="utf-8")
    assert _proof_path(repo).is_file()


def test_push_pr_consumes_long_option_operands_before_remote_and_refspec(
    tmp_path: Path,
) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    order_log = _order_log(repo)
    _write_local_review(repo)
    fake_bin = _write_fake_git(repo, order_log)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GIT_TOPLEVEL": str(repo),
        "FAKE_GIT_LOG": str(order_log),
        "ORDER_LOG": str(order_log),
    }

    result = subprocess.run(
        [
            "bash",
            "scripts/push_pr.sh",
            str(body),
            "--receive-pack",
            "git-receive-pack",
            "--recurse-submodules",
            "check",
            "origin",
            "HEAD",
        ],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "git push --receive-pack git-receive-pack --recurse-submodules check origin HEAD"
        in order_log.read_text(encoding="utf-8")
    )
    assert _proof_path(repo).is_file()


def test_push_pr_consumes_short_push_option_operand_before_remote_and_refspec(
    tmp_path: Path,
) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    order_log = _order_log(repo)
    _write_local_review(repo)
    fake_bin = _write_fake_git(repo, order_log)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GIT_TOPLEVEL": str(repo),
        "FAKE_GIT_LOG": str(order_log),
        "ORDER_LOG": str(order_log),
    }

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "-o", "ci.skip", "origin", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "git push -o ci.skip origin HEAD" in order_log.read_text(encoding="utf-8")
    assert _proof_path(repo).is_file()


def test_push_pr_rejects_repo_target_override_before_review_or_push(
    tmp_path: Path,
) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    order_log = _order_log(repo)
    _write_local_review(repo)
    fake_bin = _write_fake_git(repo, order_log)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GIT_TOPLEVEL": str(repo),
        "FAKE_GIT_LOG": str(order_log),
        "ORDER_LOG": str(order_log),
    }

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "--repo", "origin", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "refusing git push target override" in result.stderr
    assert "git push" not in order_log.read_text(encoding="utf-8")
    assert not _proof_path(repo).exists()


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


def test_push_pr_rejects_invalid_body_before_fetch_review_or_push(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_invalid_body(repo)
    order_log = _order_log(repo)
    _write_local_review(repo)
    fake_bin = _write_fake_git(repo, order_log)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GIT_TOPLEVEL": str(repo),
        "FAKE_GIT_LOG": str(order_log),
        "ORDER_LOG": str(order_log),
    }

    result = subprocess.run(
        ["bash", "scripts/push_pr.sh", str(body), "-u", "origin", "HEAD"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "missing required section: ## Cold diff reconstruction" in result.stdout
    lines = order_log.read_text(encoding="utf-8").splitlines()
    assert "git fetch --quiet origin main" in lines
    assert "git push -u origin HEAD" not in lines
    assert not any(line.startswith("local-review ") for line in lines)


def _write_fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    copy2(SCRIPT, repo / "scripts" / "push_pr.sh")
    copy2(AUDIT_SCRIPT, repo / "scripts" / "audit_pr_body.py")
    copy2(CHANGE_POLICY_SCRIPT, repo / "scripts" / "_pr_change_policy.py")
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "branch", "-M", "main")
    _git(repo, "remote", "add", "origin", str(repo))
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    return repo


def _write_body(repo: Path) -> Path:
    body = repo.parent / "body.md"
    _checkout_pr_branch(repo)
    _write_plan(repo)
    (repo / "scripts" / "example.py").write_text("print('changed')\n", encoding="utf-8")
    _git(repo, "add", "plans/PR-Test.md", "scripts/example.py")
    _git(repo, "commit", "-qm", "planned change")
    body.write_text(_valid_body(), encoding="utf-8")
    return body


def _write_invalid_body(repo: Path) -> Path:
    _write_body(repo)
    body = repo.parent / "body-invalid.md"
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
    _checkout_pr_branch(repo)
    doc = repo / "docs" / "example.md"
    doc.parent.mkdir()
    doc.write_text("# docs only\n", encoding="utf-8")
    _git(repo, "add", "docs/example.md")
    _git(repo, "commit", "-qm", "docs only")
    body = repo.parent / "body-docs-only.md"
    body.write_text("Docs-only: true\n\nCorrect a documentation typo.\n", encoding="utf-8")
    return body


def _write_plan(repo: Path) -> None:
    plan = repo / "plans" / "PR-Test.md"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("# Test plan\n", encoding="utf-8")


def _checkout_pr_branch(repo: Path) -> None:
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if branch != "claude/pr-test":
        _git(repo, "checkout", "-b", "claude/pr-test")


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


def _write_local_review(repo: Path) -> None:
    review = repo / "scripts" / "local_pr_review.sh"
    review.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf 'local-review %s\\n' \"$*\" >> \"$ORDER_LOG\"\n"
        "printf 'github-head-ref %s\\n' \"${GITHUB_HEAD_REF:-}\" >> \"$ORDER_LOG\"\n",
        encoding="utf-8",
    )
    review.chmod(0o755)
    _git(repo, "add", "scripts/local_pr_review.sh")
    _git(repo, "commit", "--amend", "--no-edit", "-q")


def _write_aba_local_review(repo: Path) -> None:
    review = repo / "scripts" / "local_pr_review.sh"
    review.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf 'local-review-head %s\\n' \"$(git rev-parse HEAD)\" >> \"$ORDER_LOG\"\n"
        "git -C \"$ABA_REPO\" checkout -q \"$ABA_OTHER_SHA\"\n"
        "git -C \"$ABA_REPO\" checkout -q \"$ABA_BRANCH\"\n"
        "printf 'local-review-original-after %s\\n' \"$(git -C \"$ABA_REPO\" rev-parse HEAD)\" >> \"$ORDER_LOG\"\n",
        encoding="utf-8",
    )
    review.chmod(0o755)
    _git(repo, "add", "scripts/local_pr_review.sh")
    _git(repo, "commit", "--amend", "--no-edit", "-q")


def _write_body_audit_recorder(repo: Path) -> None:
    (repo / "scripts" / "audit_pr_body.py").write_text(
        "from os import environ\n"
        "from pathlib import Path\n"
        "with Path(environ['ORDER_LOG']).open('a', encoding='utf-8') as handle:\n"
        "    handle.write('body-audit\\n')\n",
        encoding="utf-8",
    )
    _git(repo, "add", "scripts/audit_pr_body.py")
    _git(repo, "commit", "--amend", "--no-edit", "-q")


def _write_managed_hook(repo: Path, *, executable: bool = True) -> None:
    hook = repo / ".git" / "hooks" / "pre-push"
    hook.write_text(
        "#!/usr/bin/env bash\n# ATLAS_LOCAL_PR_REVIEW_HOOK\n",
        encoding="utf-8",
    )
    if executable:
        hook.chmod(0o755)


def _order_log(repo: Path) -> Path:
    return repo.parent / "order.log"


def _write_fake_git(repo: Path, order_log: Path) -> Path:
    fake_bin = repo.parent / "fake-bin"
    fake_bin.mkdir(exist_ok=True)
    git = fake_bin / "git"
    real_git = which("git")
    assert real_git is not None
    git.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "case \"${1:-}\" in\n"
        "  rev-parse)\n"
        "    if [ \"${2:-}\" = \"--show-toplevel\" ]; then\n"
        "      printf '%s\\n' \"$FAKE_GIT_TOPLEVEL\"\n"
        "      exit 0\n"
        "    fi\n"
        "    if [ \"${2:-}\" = \"--git-path\" ]; then\n"
        "      printf '%s/.git/%s\\n' \"$FAKE_GIT_TOPLEVEL\" \"${3:-}\"\n"
        "      exit 0\n"
        "    fi\n"
        "    ;;\n"
        "  fetch)\n"
        "    printf 'git %s\\n' \"$*\" >> \"$FAKE_GIT_LOG\"\n"
        "    if [ \"${FAKE_GIT_FETCH_FAIL:-}\" = \"1\" ]; then\n"
        "      exit 42\n"
        "    fi\n"
        "    exit 0\n"
        "    ;;\n"
        "  push)\n"
        "    printf 'git %s\\n' \"$*\" >> \"$FAKE_GIT_LOG\"\n"
        "    if [ \"${FAKE_GIT_PUSH_FAIL:-}\" = \"1\" ]; then\n"
        "      exit 42\n"
        "    fi\n"
        "    if [ -n \"${FAKE_GIT_MUTATE_BODY:-}\" ]; then\n"
        "      printf '\\nmutated during push\\n' >> \"$FAKE_GIT_MUTATE_BODY\"\n"
        "    fi\n"
        "    if [ \"${FAKE_GIT_DIRTY_SOURCE_DURING_PUSH:-}\" = \"1\" ]; then\n"
        "      printf 'not reviewed\\n' > \"$FAKE_GIT_TOPLEVEL/pushed-dirty.txt\"\n"
        "    fi\n"
        "    if [ \"${FAKE_GIT_SKIP_REMOTE_UPDATE:-}\" != \"1\" ]; then\n"
        f"      branch=\"$({real_git!r} branch --show-current)\"\n"
        f"      {real_git!r} update-ref \"refs/remotes/origin/$branch\" HEAD\n"
        "    fi\n"
        "    if [ \"${FAKE_GIT_ADVANCE_BASE_DURING_PUSH:-}\" = \"1\" ]; then\n"
        f"      {real_git!r} update-ref refs/remotes/origin/main HEAD\n"
        "    fi\n"
        "    exit 0\n"
        "    ;;\n"
        "esac\n"
        f"exec {real_git!r} \"$@\"\n",
        encoding="utf-8",
    )
    git.chmod(0o755)
    order_log.write_text("", encoding="utf-8")
    return fake_bin


def _proof_path(repo: Path) -> Path:
    proof = Path(
        subprocess.run(
            ["git", "rev-parse", "--git-path", "atlas-local-pr-review-proof"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return proof if proof.is_absolute() else repo / proof


def _head(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _base(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "origin/main"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
