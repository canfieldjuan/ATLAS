from __future__ import annotations

import os
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
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(repo), **(env or {})},
    )


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _is_executable(path: Path) -> bool:
    return bool(path.stat().st_mode & stat.S_IXUSR)
