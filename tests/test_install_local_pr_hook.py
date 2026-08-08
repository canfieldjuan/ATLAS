from __future__ import annotations

import os
import pty
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_installs_managed_pre_push_hook(tmp_path):
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)

    result = _run(repo, ["bash", "scripts/install_local_pr_hook.sh"])

    assert result.returncode == 0, result.stdout + result.stderr
    hook = repo / ".git" / "hooks" / "pre-push"
    text = hook.read_text(encoding="utf-8")
    assert "ATLAS_LOCAL_PR_REVIEW_HOOK" in text
    assert "exec bash scripts/local_pr_review.sh" in text
    assert _is_executable(hook)


def test_refuses_to_overwrite_unmanaged_pre_push_hook(tmp_path):
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    hook = repo / ".git" / "hooks" / "pre-push"
    hook.write_text("#!/usr/bin/env bash\necho custom\n", encoding="utf-8")

    result = _run(repo, ["bash", "scripts/install_local_pr_hook.sh"])

    assert result.returncode == 1
    assert "refusing to overwrite unmanaged hook" in result.stderr
    assert hook.read_text(encoding="utf-8") == "#!/usr/bin/env bash\necho custom\n"


def test_force_overwrites_unmanaged_pre_push_hook(tmp_path):
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    hook = repo / ".git" / "hooks" / "pre-push"
    hook.write_text("#!/usr/bin/env bash\necho custom\n", encoding="utf-8")

    result = _run(repo, ["bash", "scripts/install_local_pr_hook.sh", "--force"])

    assert result.returncode == 0, result.stdout + result.stderr
    text = hook.read_text(encoding="utf-8")
    assert "ATLAS_LOCAL_PR_REVIEW_HOOK" in text
    assert "echo custom" not in text


def test_installed_hook_invokes_local_review_bundle(tmp_path):
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    _run(repo, ["bash", "scripts/install_local_pr_hook.sh"])
    _write_executable(repo / "scripts" / "local_pr_review.sh", "echo ran local review\n")

    result = _run(repo, [".git/hooks/pre-push"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "ran local review" in result.stdout


def test_installed_hook_invokes_local_review_bundle_with_tty_stdin(tmp_path):
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    _run(repo, ["bash", "scripts/install_local_pr_hook.sh"])
    _write_executable(repo / "scripts" / "local_pr_review.sh", "echo ran local review\n")

    result = _run_with_tty_stdin(repo, [".git/hooks/pre-push"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "ran local review" in result.stdout


def test_installed_hook_supports_explicit_skip(tmp_path):
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    _run(repo, ["bash", "scripts/install_local_pr_hook.sh"])
    _write_executable(
        repo / "scripts" / "local_pr_review.sh",
        "echo should not run\nexit 42\n",
    )

    result = _run(
        repo,
        [".git/hooks/pre-push"],
        env={"ATLAS_SKIP_LOCAL_PR_REVIEW": "1"},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "ATLAS local PR review hook skipped" in result.stdout
    assert "should not run" not in result.stdout


def test_installed_hook_skips_delete_only_push(tmp_path):
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    _run(repo, ["bash", "scripts/install_local_pr_hook.sh"])
    _write_executable(
        repo / "scripts" / "local_pr_review.sh",
        "echo should not run\nexit 42\n",
    )
    zeros = "0" * 40
    old_sha = "1" * 40

    result = _run(
        repo,
        [".git/hooks/pre-push"],
        input=f"(delete) {zeros} refs/heads/claude/pr-old {old_sha}\n",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "ATLAS local PR review hook skipped (delete-only push)." in result.stdout
    assert "should not run" not in result.stdout


def test_installed_hook_runs_review_for_malformed_delete_record(tmp_path):
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    _run(repo, ["bash", "scripts/install_local_pr_hook.sh"])
    _write_executable(repo / "scripts" / "local_pr_review.sh", "echo ran local review\n")

    result = _run(repo, [".git/hooks/pre-push"], input="(delete)\n")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "ran local review" in result.stdout
    assert "delete-only push" not in result.stdout


def test_installed_hook_runs_review_for_whitespace_record_before_delete(tmp_path):
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    _run(repo, ["bash", "scripts/install_local_pr_hook.sh"])
    _write_executable(repo / "scripts" / "local_pr_review.sh", "echo ran local review\n")
    zeros = "0" * 40
    old_sha = "1" * 40

    result = _run(
        repo,
        [".git/hooks/pre-push"],
        input=f"   \t  \n(delete) {zeros} refs/heads/claude/pr-old {old_sha}\n",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "ran local review" in result.stdout
    assert "delete-only push" not in result.stdout


def test_installed_hook_runs_review_for_malformed_complete_delete_record(tmp_path):
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    _run(repo, ["bash", "scripts/install_local_pr_hook.sh"])
    _write_executable(repo / "scripts" / "local_pr_review.sh", "echo ran local review\n")
    zeros = "0" * 40

    for ref_update in (
        f"(delete) {zeros} refs/heads/claude/pr-old not-an-object-id\n",
        f"(delete) {zeros} claude/pr-old {'1' * 40}\n",
        f"(delete) {zeros} refs/heads/x..y {'1' * 40}\n",
        f"(delete) {zeros} refs/heads/claude/pr-old {zeros}\n",
    ):
        result = _run(repo, [".git/hooks/pre-push"], input=ref_update)

        assert result.returncode == 0, result.stdout + result.stderr
        assert "ran local review" in result.stdout
        assert "delete-only push" not in result.stdout


def test_installed_hook_runs_review_for_conflicting_delete_markers(tmp_path):
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    _run(repo, ["bash", "scripts/install_local_pr_hook.sh"])
    _write_executable(repo / "scripts" / "local_pr_review.sh", "echo ran local review\n")
    zeros = "0" * 40
    old_sha = "1" * 40
    new_sha = "2" * 40

    for ref_update in (
        f"(delete) {new_sha} refs/heads/claude/pr-old {old_sha}\n",
        f"refs/heads/claude/pr-old {zeros} refs/heads/claude/pr-old {old_sha}\n",
    ):
        result = _run(repo, [".git/hooks/pre-push"], input=ref_update)

        assert result.returncode == 0, result.stdout + result.stderr
        assert "ran local review" in result.stdout
        assert "delete-only push" not in result.stdout


def test_installed_hook_runs_review_for_mixed_delete_and_update_push(tmp_path):
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    _run(repo, ["bash", "scripts/install_local_pr_hook.sh"])
    _write_executable(repo / "scripts" / "local_pr_review.sh", "echo ran local review\n")
    zeros = "0" * 40
    old_sha = "1" * 40
    new_sha = "2" * 40

    for ref_update in (
        (
            f"(delete) {zeros} refs/heads/claude/pr-old {old_sha}\n"
            f"refs/heads/claude/pr-new {new_sha} refs/heads/claude/pr-new {old_sha}\n"
        ),
        (
            f"(delete) {zeros} refs/heads/claude/pr-old {old_sha}\n"
            f"refs/heads/claude/pr-new {new_sha} refs/heads/claude/pr-new {old_sha}"
        ),
    ):
        result = _run(repo, [".git/hooks/pre-push"], input=ref_update)

        assert result.returncode == 0, result.stdout + result.stderr
        assert "ran local review" in result.stdout
        assert "delete-only push" not in result.stdout


def _write_fixture_repo(repo: Path) -> None:
    (repo / "scripts").mkdir(parents=True)
    for name in ("install_local_pr_hook.sh", "local_pr_review.sh"):
        target = repo / "scripts" / name
        _write_executable(
            target,
            (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8"),
        )

    _git(repo, "init")


def _run(
    repo: Path,
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    input: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        input=input,
        env={**os.environ, "PYTHONPATH": str(repo), **(env or {})},
    )


def _run_with_tty_stdin(repo: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    master_fd, slave_fd = pty.openpty()
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            args,
            cwd=repo,
            stdin=slave_fd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONPATH": str(repo)},
        )
        os.close(slave_fd)
        slave_fd = -1
        try:
            stdout, stderr = proc.communicate(timeout=2)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            stdout, stderr = proc.communicate(timeout=2)
            raise AssertionError("pre-push hook blocked on TTY stdin") from exc
        return subprocess.CompletedProcess(
            args=args,
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
        )
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
        if slave_fd >= 0:
            os.close(slave_fd)
        os.close(master_fd)


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _is_executable(path: Path) -> bool:
    return bool(path.stat().st_mode & stat.S_IXUSR)
