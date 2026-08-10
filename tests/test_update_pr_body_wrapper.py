from __future__ import annotations

import os
import subprocess
from pathlib import Path
from shutil import copy2


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "update_pr_body.sh"
OPEN_PR_WRAPPER_MARKER = "<!-- atlas-open-pr-wrapper: v1 -->"


def test_update_wrapper_stamps_full_body_before_publish(tmp_path: Path) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path)

    result = _run(repo, env, body)

    assert result.returncode == 0, result.stdout + result.stderr
    assert log.read_text(encoding="utf-8").strip() == "pr edit 17 --repo canfieldjuan/ATLAS --body-file -"
    assert stdin_capture.read_text(encoding="utf-8") == _stamped_body(body)
    assert OPEN_PR_WRAPPER_MARKER not in body.read_text(encoding="utf-8")


def test_update_wrapper_does_not_run_full_local_review(tmp_path: Path) -> None:
    repo, body, env, log, _ = _ready(tmp_path)

    result = _run(repo, env, body)

    assert result.returncode == 0, result.stdout + result.stderr
    assert log.exists()
    assert not (repo / "local-review.log").exists()


def test_update_wrapper_runs_live_reconciliation_with_publish_body(tmp_path: Path) -> None:
    repo, body, env, _, _ = _ready(tmp_path)

    result = _run(repo, env, body)

    assert result.returncode == 0, result.stdout + result.stderr
    live_log = Path(env["LIVE_RECONCILIATION_LOG"]).read_text(encoding="utf-8")
    assert "--repo canfieldjuan/ATLAS --pr 17 --body-file" in live_log
    assert Path(env["LIVE_RECONCILIATION_BODY_CAPTURE"]).read_text(encoding="utf-8") == _stamped_body(body)


def test_update_wrapper_rejects_target_changing_args(tmp_path: Path) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path)

    result = _run(repo, env, body, "--title", "New title")

    assert result.returncode == 2
    assert "body-only updates do not accept PR create/edit args" in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


def test_update_wrapper_rejects_head_drift_before_edit(tmp_path: Path) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path)
    stale_head = _git_output(repo, "rev-parse", "HEAD^")
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


def test_update_wrapper_ownership_guard_failure_blocks_before_edit(tmp_path: Path) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path)
    env["OWNERSHIP_GUARD_EXIT"] = "23"

    result = _run(repo, env, body)

    assert result.returncode == 23
    assert "fake ownership guard failed" in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


def _ready(tmp_path: Path) -> tuple[Path, Path, dict[str, str], Path, Path]:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    env, log, stdin_capture = _fake_gh_env(tmp_path)
    return repo, body, env, log, stdin_capture


def _run(repo: Path, env: dict[str, str], body: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "scripts/update_pr_body.sh", str(body), *args],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _stamped_body(body: Path) -> str:
    return body.read_text(encoding="utf-8") + f"\n{OPEN_PR_WRAPPER_MARKER}\n"


def _write_fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    for name in (
        "update_pr_body.sh",
        "audit_pr_body.py",
        "audit_ai_reconciliation.py",
        "audit_fix_loop_disposition.py",
        "_pr_change_policy.py",
        "check_pr_branch_name.py",
        "fix_loop_trace_contract.py",
    ):
        copy2(REPO_ROOT / "scripts" / name, scripts / name)
    (scripts / "check_session_pr_ownership.py").write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

log = os.environ.get("OWNERSHIP_GUARD_LOG")
if log:
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(" ".join(sys.argv[1:]) + "\\n")
exit_code = int(os.environ.get("OWNERSHIP_GUARD_EXIT", "0"))
if exit_code:
    print("fake ownership guard failed", file=sys.stderr)
raise SystemExit(exit_code)
""",
        encoding="utf-8",
    )
    (scripts / "check_session_pr_ownership.py").chmod(0o755)
    (scripts / "check_ai_reconciliation_live.py").write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

log = os.environ["LIVE_RECONCILIATION_LOG"]
with open(log, "a", encoding="utf-8") as handle:
    handle.write(" ".join(sys.argv[1:]) + "\\n")
if "--body-file" in sys.argv:
    body_path = sys.argv[sys.argv.index("--body-file") + 1]
    with open(body_path, "r", encoding="utf-8") as source:
        body = source.read()
    with open(os.environ["LIVE_RECONCILIATION_BODY_CAPTURE"], "w", encoding="utf-8") as target:
        target.write(body)
raise SystemExit(int(os.environ.get("LIVE_RECONCILIATION_EXIT", "0")))
""",
        encoding="utf-8",
    )
    (scripts / "check_ai_reconciliation_live.py").chmod(0o755)
    (scripts / "local_pr_review.sh").write_text(
        """#!/usr/bin/env bash
printf 'local_pr_review called\\n' >> local-review.log
exit 99
""",
        encoding="utf-8",
    )
    (scripts / "local_pr_review.sh").chmod(0o755)
    subprocess.run(["git", "init", "--initial-branch", "main"], cwd=repo, check=True, capture_output=True, text=True)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    remote = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True, text=True)
    _git(repo, "config", "url.git@github.com:canfieldjuan/ATLAS.git.insteadOf", str(remote))
    _git(repo, "config", f"url.{remote}.insteadOf", "git@github.com:canfieldjuan/ATLAS.git")
    _git(repo, "remote", "add", "origin", "git@github.com:canfieldjuan/ATLAS.git")
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


def _valid_body() -> str:
    return "\n".join(
        [
            "Plan: plans/PR-Test.md",
            "Slice phase: Workflow/process",
            "Ownership lane: dev-workflow/process-gate-enrollment",
            "",
            "One-paragraph why.",
            "",
            "## Intentional",
            "- a trade-off",
            "",
            "## AI reconciliation",
            "- no-findings",
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
            "## Mechanical verification",
            "- Command: pytest tests/test_update_pr_body_wrapper.py - Result: passed - Environment: local",
            "",
            "## Diff size",
            "2 files, +10 / -2",
        ]
    )


def _fake_gh_env(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "gh-argv.txt"
    stdin_capture = tmp_path / "gh-stdin.txt"
    ownership_guard_log = tmp_path / "ownership-guard-argv.txt"
    live_log = tmp_path / "live-reconciliation-argv.txt"
    live_body_capture = tmp_path / "live-reconciliation-body.md"
    gh = bin_dir / "gh"
    gh.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [ "$1" = "pr" ] && [ "$2" = "list" ]; then
    if [ -n "${GH_PR_LIST_JSON:-}" ]; then
        printf '%s\\n' "${GH_PR_LIST_JSON}"
        exit 0
    fi
    current_branch="$(git branch --show-current)"
    printf '[{"number":17,"headRefName":"%s","headRefOid":"%s","baseRefName":"main","headRepository":{"nameWithOwner":"canfieldjuan/ATLAS"},"isCrossRepository":false}]\\n' "$current_branch" "$(git rev-parse HEAD)"
    exit 0
fi
printf '%s\\n' "$*" > "${GH_ARGV_LOG}"
cat > "${GH_STDIN_CAPTURE}"
""",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "GH_ARGV_LOG": str(log),
        "GH_STDIN_CAPTURE": str(stdin_capture),
        "OWNERSHIP_GUARD_LOG": str(ownership_guard_log),
        "LIVE_RECONCILIATION_LOG": str(live_log),
        "LIVE_RECONCILIATION_BODY_CAPTURE": str(live_body_capture),
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


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()
