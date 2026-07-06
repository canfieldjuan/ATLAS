"""Tests for the claude-review merge-gate status setter."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/set_claude_review_status.py"
SPEC = importlib.util.spec_from_file_location("set_claude_review_status", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

# The GitHub statuses API requires a full 40-char SHA; the tool enforces it.
VALID_SHA = "0123456789abcdef0123456789abcdef01234567"


def test_build_gh_args_always_uses_claude_review_context() -> None:
    args = module.build_gh_args(
        repo="canfieldjuan/ATLAS",
        sha=VALID_SHA,
        state="success",
        description="ok",
        target_url=None,
    )
    assert f"repos/canfieldjuan/ATLAS/statuses/{VALID_SHA}" in args
    assert "context=claude-review" in args
    assert "state=success" in args
    # No path lets the caller set a different context.
    assert sum(1 for a in args if a.startswith("context=")) == 1


def test_build_gh_args_includes_target_url_when_pr_given() -> None:
    args = module.build_gh_args(
        repo="canfieldjuan/ATLAS",
        sha=VALID_SHA,
        state="failure",
        description="blocker",
        target_url="https://github.com/canfieldjuan/ATLAS/pull/42",
    )
    assert "target_url=https://github.com/canfieldjuan/ATLAS/pull/42" in args
    assert "state=failure" in args


@pytest.mark.parametrize("state", ["approve", "green", "", "SUCCESS"])
def test_rejects_non_allowed_state(state: str) -> None:
    with pytest.raises(module.UsageError):
        module.build_gh_args(
            repo="canfieldjuan/ATLAS", sha=VALID_SHA, state=state, description="x", target_url=None
        )


@pytest.mark.parametrize("repo", ["ATLAS", "a/b/c", "canfieldjuan/", "/ATLAS"])
def test_rejects_malformed_repo(repo: str) -> None:
    with pytest.raises(module.UsageError):
        module.build_gh_args(
            repo=repo, sha=VALID_SHA, state="success", description="x", target_url=None
        )


@pytest.mark.parametrize(
    "sha",
    [
        "",
        "xyz",
        "12345",
        "nothex!!",
        "abcdef1",  # 7-char abbreviation: valid hex but the statuses API rejects it
        "0123456789abcdef0123456789abcdef0123456",  # 39 chars
        "0123456789abcdef0123456789abcdef012345678",  # 41 chars
    ],
)
def test_rejects_non_full_sha(sha: str) -> None:
    with pytest.raises(module.UsageError):
        module.build_gh_args(
            repo="canfieldjuan/ATLAS", sha=sha, state="success", description="x", target_url=None
        )


def test_cli_dry_run_prints_argv_and_exits_zero() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo",
            "canfieldjuan/ATLAS",
            "--sha",
            VALID_SHA,
            "--state",
            "success",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "context=claude-review" in proc.stdout
    assert "state=success" in proc.stdout
    # Dry run must not touch the network.
    assert f"gh api -X POST repos/canfieldjuan/ATLAS/statuses/{VALID_SHA}" in proc.stdout


def test_cli_bad_state_exits_two() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo",
            "canfieldjuan/ATLAS",
            "--sha",
            VALID_SHA,
            "--state",
            "lgtm",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "state must be one of" in proc.stderr


def test_cli_abbreviated_sha_exits_two() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo",
            "canfieldjuan/ATLAS",
            "--sha",
            "719855dd921f",  # the exact abbreviation that the statuses API rejected in practice
            "--state",
            "success",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "full 40-char hex commit SHA" in proc.stderr
