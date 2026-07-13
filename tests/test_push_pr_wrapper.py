from __future__ import annotations

import os
import subprocess
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
    assert "DRY RUN: git fetch --quiet origin main" in result.stdout
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
    assert f"--current-pr-body-file {body}" in result.stdout
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
    order_log = repo / "order.log"
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
    order_log = repo / "order.log"
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
    assert lines.index("git fetch --quiet origin main") < lines.index(
        f"local-review --current-pr-body-file {body}"
    )
    assert lines.index(f"local-review --current-pr-body-file {body}") < lines.index(
        "git push -u origin HEAD"
    )


def test_push_pr_refreshes_base_before_managed_hook_push(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    order_log = repo / "order.log"
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
    assert lines.index("git fetch --quiet origin main") < lines.index(
        "git push -u origin HEAD"
    )
    assert not any(line.startswith("local-review ") for line in lines)


def test_push_pr_fetch_failure_aborts_before_review_or_push(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    order_log = repo / "order.log"
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
    order_log = repo / "order.log"
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
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "branch", "-M", "main")
    _git(repo, "remote", "add", "origin", str(repo))
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
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


def _write_local_review(repo: Path) -> None:
    review = repo / "scripts" / "local_pr_review.sh"
    review.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf 'local-review %s\\n' \"$*\" >> \"$ORDER_LOG\"\n",
        encoding="utf-8",
    )
    review.chmod(0o755)


def _write_body_audit_recorder(repo: Path) -> None:
    (repo / "scripts" / "audit_pr_body.py").write_text(
        "from os import environ\n"
        "from pathlib import Path\n"
        "with Path(environ['ORDER_LOG']).open('a', encoding='utf-8') as handle:\n"
        "    handle.write('body-audit\\n')\n",
        encoding="utf-8",
    )


def _write_managed_hook(repo: Path, *, executable: bool = True) -> None:
    hook = repo / ".git" / "hooks" / "pre-push"
    hook.write_text(
        "#!/usr/bin/env bash\n# ATLAS_LOCAL_PR_REVIEW_HOOK\n",
        encoding="utf-8",
    )
    if executable:
        hook.chmod(0o755)


def _write_fake_git(repo: Path, order_log: Path) -> Path:
    fake_bin = repo / "fake-bin"
    fake_bin.mkdir()
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
        "    exit 0\n"
        "    ;;\n"
        "esac\n"
        f"exec {real_git!r} \"$@\"\n",
        encoding="utf-8",
    )
    git.chmod(0o755)
    order_log.write_text("", encoding="utf-8")
    return fake_bin


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
