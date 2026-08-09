from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_local_pr_review_fails_on_dirty_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    (repo / "dirty.txt").write_text("not committed\n", encoding="utf-8")

    result = _run(repo, ["bash", "scripts/local_pr_review.sh"])

    assert result.returncode == 1
    assert "worktree has uncommitted changes" in result.stderr
    assert "dirty.txt" in result.stderr
    assert "Pre-push audit wrapper" not in result.stdout


def test_local_pr_review_allow_dirty_runs_checks(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    (repo / "dirty.txt").write_text("not committed\n", encoding="utf-8")

    result = _run(repo, ["bash", "scripts/local_pr_review.sh", "--allow-dirty"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Pre-push audit wrapper" in result.stdout
    assert "local PR review passed" in result.stdout


def test_local_pr_review_allow_dirty_preserves_base_ref_arg(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    (repo / "dirty.txt").write_text("not committed\n", encoding="utf-8")

    result = _run(
        repo,
        ["bash", "scripts/local_pr_review.sh", "--allow-dirty", "origin/main"],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "base ref: origin/main" in result.stdout


def test_local_pr_review_help_exits_cleanly(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)

    result = _run(repo, ["bash", "scripts/local_pr_review.sh", "--help"])

    assert result.returncode == 0
    assert "Usage: bash scripts/local_pr_review.sh" in result.stdout
    assert "--pr-author LOGIN" in result.stdout


def test_local_pr_review_rejects_unknown_option(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)

    result = _run(repo, ["bash", "scripts/local_pr_review.sh", "--unknown"])

    assert result.returncode == 2
    assert "unknown option: --unknown" in result.stderr


def test_local_pr_review_rejects_multiple_base_refs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)

    result = _run(
        repo,
        ["bash", "scripts/local_pr_review.sh", "origin/main", "origin/main"],
    )

    assert result.returncode == 2
    assert "multiple base refs supplied" in result.stderr


def test_local_pr_review_runs_cross_session_drift_audit_when_present(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    _write_executable(
        repo / "scripts" / "audit_pr_session_drift.py",
        "#!/usr/bin/env python3\nprint('drift guard ran')\n",
    )
    _git(repo, "add", "scripts/audit_pr_session_drift.py")
    _git(repo, "commit", "-m", "add drift guard")

    result = _run(repo, ["bash", "scripts/local_pr_review.sh"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Cross-session PR drift" in result.stdout
    assert "drift guard ran" in result.stdout


def test_local_pr_review_passes_base_ref_to_plan_code_consistency(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    plan = repo / "plans" / "PR-BaseRef.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# PR-BaseRef\n", encoding="utf-8")
    _write_executable(
        repo / "scripts" / "audit_plan_code_consistency.py",
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "print('plan-code args=' + ' '.join(sys.argv[1:]))\n",
    )
    _git(repo, "add", "plans/PR-BaseRef.md", "scripts/audit_plan_code_consistency.py")
    _git(repo, "commit", "-m", "add plan and capture plan-code args")
    _git(repo, "update-ref", "refs/remotes/origin/release", "HEAD^")

    result = _run(repo, ["bash", "scripts/local_pr_review.sh", "origin/release"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Plan/code consistency: plans/PR-BaseRef.md" in result.stdout
    assert "--base-ref origin/release plans/PR-BaseRef.md" in result.stdout


def test_local_pr_review_runs_pr_body_contract_when_body_supplied(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    body = tmp_path / "body.md"
    body.write_text("PR body\n", encoding="utf-8")
    _write_executable(
        repo / "scripts" / "audit_pr_body.py",
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "print('body audit args=' + ' '.join(sys.argv[1:]))\n",
    )
    _git(repo, "add", "scripts/audit_pr_body.py")
    _git(repo, "commit", "-m", "add body audit")

    result = _run(repo, ["bash", "scripts/local_pr_review.sh", "--current-pr-body-file", str(body)])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PR body contract" in result.stdout
    assert f"--repo-root {repo}" in result.stdout
    assert "--base-ref origin/main" in result.stdout
    assert f" {body}" in result.stdout
    assert "local PR review passed" in result.stdout


def test_local_pr_review_runs_fix_loop_disposition_when_body_supplied(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    body = tmp_path / "body.md"
    body.write_text("PR body\n", encoding="utf-8")
    _write_executable(
        repo / "scripts" / "audit_fix_loop_disposition.py",
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "print('fix-loop args=' + ' '.join(sys.argv[1:]))\n",
    )
    _git(repo, "add", "scripts/audit_fix_loop_disposition.py")
    _git(repo, "commit", "-m", "add fix-loop disposition audit")

    result = _run(repo, ["bash", "scripts/local_pr_review.sh", "--current-pr-body-file", str(body)])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Fix-loop disposition preflight" in result.stdout
    assert f"--current-pr-body-file {body}" in result.stdout
    assert f"--repo-root {repo}" in result.stdout
    assert "--base-ref origin/main" in result.stdout
    assert "local PR review passed" in result.stdout


def test_local_pr_review_blocks_unclosed_guard_class(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    _add_guard_shaped_change(repo)

    result = _run(repo, ["bash", "scripts/local_pr_review.sh"])

    assert result.returncode == 1
    assert "Guard class-closure lint" in result.stdout
    assert "guard-shaped change over open input with no co-changed property/generative test" in result.stdout
    assert "1 local review check(s) failed" in result.stdout


def test_local_pr_review_allows_guard_class_waiver_from_pr_body(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    _add_guard_shaped_change(repo)
    body = tmp_path / "body.md"
    body.write_text(
        "guard-class-closure: waived\n\n"
        "Reason: false-positive synthetic guard fixture for local review wiring.\n",
        encoding="utf-8",
    )

    result = _run(repo, ["bash", "scripts/local_pr_review.sh", "--current-pr-body-file", str(body)])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Guard class-closure lint" in result.stdout
    assert "WAIVED: 'guard-class-closure: waived' present" in result.stdout
    assert "local PR review passed" in result.stdout


def test_local_pr_review_forwards_pr_author_to_body_contract(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    body = tmp_path / "body.md"
    body.write_text("generated body\n", encoding="utf-8")
    _write_executable(
        repo / "scripts" / "audit_pr_body.py",
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "print('body audit args=' + ' '.join(sys.argv[1:]))\n",
    )
    _git(repo, "add", "scripts/audit_pr_body.py")
    _git(repo, "commit", "-m", "add body audit")

    result = _run(
        repo,
        [
            "bash",
            "scripts/local_pr_review.sh",
            "--current-pr-body-file",
            str(body),
            "--pr-author",
            "dependabot[bot]",
        ],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "--pr-author dependabot[bot]" in result.stdout
    assert "--base-ref origin/main" in result.stdout


def test_local_pr_review_env_pr_author_reaches_body_contract(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    body = tmp_path / "body.md"
    body.write_text("generated body\n", encoding="utf-8")
    _write_executable(
        repo / "scripts" / "audit_pr_body.py",
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "print('body audit args=' + ' '.join(sys.argv[1:]))\n",
    )
    _git(repo, "add", "scripts/audit_pr_body.py")
    _git(repo, "commit", "-m", "add body audit")

    result = _run(
        repo,
        ["bash", "scripts/local_pr_review.sh"],
        env={
            "ATLAS_CURRENT_PR_BODY_FILE": str(body),
            "ATLAS_CURRENT_PR_AUTHOR": "app/dependabot",
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "--pr-author app/dependabot" in result.stdout
    assert "--base-ref origin/main" in result.stdout


def test_local_pr_review_forwards_pr_author_to_pre_push_audit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    _write_executable(
        repo / "scripts" / "pre_push_audit.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf 'pre-push args=%s\\n' \"$*\"\n",
    )
    _git(repo, "add", "scripts/pre_push_audit.sh")
    _git(repo, "commit", "-m", "capture pre-push args")

    result = _run(
        repo,
        ["bash", "scripts/local_pr_review.sh", "--pr-author", "dependabot[bot]"],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "pre-push args=--repo-root" in result.stdout
    assert "--pr-author dependabot[bot]" in result.stdout


def test_local_pr_review_runs_local_unit_gate_mirror(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    _write_unit_gate_fixture(
        repo,
        selector_script="#!/usr/bin/env python3\nprint('tests/test_example.py')\n",
        checker_script=(
            "#!/usr/bin/env python3\n"
            "import os\n"
            "import sys\n"
            "print('body env=' + str(os.environ.get('ATLAS_CURRENT_PR_BODY_FILE')))\n"
            "print('git prefix env=' + str(os.environ.get('GIT_PREFIX')))\n"
            "print('pytest addopts env=' + str(os.environ.get('PYTEST_ADDOPTS')))\n"
            "print('pytest disable plugins env=' + str(os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD')))\n"
            "print('db connection env=' + str(os.environ.get('ATLAS_DB_CONNECTION_STRING')))\n"
            "print('db host env=' + str(os.environ.get('ATLAS_DB_HOST')))\n"
            "print('db port env=' + str(os.environ.get('ATLAS_DB_PORT')))\n"
            "print('db database env=' + str(os.environ.get('ATLAS_DB_DATABASE')))\n"
            "print('db user env=' + str(os.environ.get('ATLAS_DB_USER')))\n"
            "print('db password env=' + str(os.environ.get('ATLAS_DB_PASSWORD')))\n"
            "print('db socket env=' + str(os.environ.get('ATLAS_DB_SOCKET_PATH')))\n"
            "print('unit gate args=' + ' '.join(sys.argv[1:]))\n"
        ),
    )
    (repo / "requirements.unit_gate.txt").write_text("scapy==2.7.0\n", encoding="utf-8")
    _git(repo, "add", "requirements.unit_gate.txt")
    _git(repo, "commit", "-m", "add unit gate test requirements")
    fake_python = tmp_path / "fake-python"
    _write_executable(
        fake_python,
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [ \"${1:-}\" = \"-m\" ] && [ \"${2:-}\" = \"pip\" ]; then\n"
        "  shift 2\n"
        "  printf 'unit gate deps install=%s\\n' \"$*\"\n"
        "  exit 0\n"
        "fi\n"
        f"exec {sys.executable!r} \"$@\"\n",
    )

    result = _run(
        repo,
        ["bash", "scripts/local_pr_review.sh"],
        env={
            "ATLAS_CURRENT_PR_BODY_FILE": "/tmp/body-from-wrapper.md",
            "GIT_PREFIX": "from-hook/",
            "GITHUB_ACTIONS": "false",
            "PYTEST_ADDOPTS": "-k passing",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "ATLAS_DB_SOCKET_PATH": "/tmp/dev-postgres.sock",
            "PYTHON": str(fake_python),
        },
    )

    stdout_lines = set(result.stdout.splitlines())
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Local unit gate mirror" in result.stdout
    assert "unit gate deps install=install -r " + str(repo / "requirements.unit_gate.txt") in stdout_lines
    assert "body env=None" in stdout_lines
    assert "git prefix env=None" in stdout_lines
    assert "pytest addopts env=None" in stdout_lines
    assert "pytest disable plugins env=None" in stdout_lines
    assert "db connection env=" in stdout_lines
    assert "db host env=127.0.0.1" in stdout_lines
    assert "db port env=1" in stdout_lines
    assert "db database env=atlas" in stdout_lines
    assert "db user env=atlas" in stdout_lines
    assert "db password env=atlas_dev_password" in stdout_lines
    assert "db socket env=" in stdout_lines
    assert "running 1 impacted test file(s)" in result.stdout
    assert "--selected-files" in result.stdout
    assert "tests/test_example.py -m not integration and not e2e" in result.stdout
    assert "local PR review passed" in result.stdout


def test_local_pr_review_unit_gate_mirror_runs_growth_only_for_empty_selection(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    _write_unit_gate_fixture(repo, selector_script="#!/usr/bin/env python3\n")

    result = _run(repo, ["bash", "scripts/local_pr_review.sh"], env={"GITHUB_ACTIONS": "false"})

    assert result.returncode == 0, result.stdout + result.stderr
    assert "no test is reachable from the changed files; growth guard only" in result.stdout
    assert "--growth-only" in result.stdout
    assert "--selected-files" not in result.stdout
    assert "local PR review passed" in result.stdout


def test_local_pr_review_unit_gate_mirror_runs_full_selection(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    _write_unit_gate_fixture(repo, selector_script="#!/usr/bin/env python3\nprint('FULL')\n")

    result = _run(repo, ["bash", "scripts/local_pr_review.sh"], env={"GITHUB_ACTIONS": "false"})

    assert result.returncode == 0, result.stdout + result.stderr
    assert "running the FULL suite (selection escalated)" in result.stdout
    assert "--growth-only" not in result.stdout
    assert "--selected-files" not in result.stdout
    assert "local PR review passed" in result.stdout


def test_local_pr_review_unit_gate_mirror_checker_failures_block_every_mode(tmp_path: Path) -> None:
    cases = {
        "selected": (
            "#!/usr/bin/env python3\nprint('tests/test_example.py')\n",
            "running 1 impacted test file(s)",
            "--selected-files",
        ),
        "growth_only": (
            "#!/usr/bin/env python3\n",
            "no test is reachable from the changed files; growth guard only",
            "--growth-only",
        ),
        "full": (
            "#!/usr/bin/env python3\nprint('FULL')\n",
            "running the FULL suite (selection escalated)",
            "--baseline tests/unit_gate_baseline.txt",
        ),
    }

    for name, (selector_script, mode_message, args_fragment) in cases.items():
        repo = tmp_path / name
        _write_fixture_repo(repo)
        _write_unit_gate_fixture(
            repo,
            selector_script=selector_script,
            checker_script="#!/usr/bin/env python3\nimport sys\nprint('unit gate rejected: ' + ' '.join(sys.argv[1:]))\nsys.exit(9)\n",
        )

        result = _run(repo, ["bash", "scripts/local_pr_review.sh"], env={"GITHUB_ACTIONS": "false"})

        assert result.returncode == 1, result.stdout + result.stderr
        assert "Local unit gate mirror" in result.stdout
        assert mode_message in result.stdout
        assert args_fragment in result.stdout
        assert "unit gate rejected:" in result.stdout
        assert "1 local review check(s) failed" in result.stdout


def test_local_pr_review_unit_gate_mirror_propagates_selector_failure(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    _write_unit_gate_fixture(
        repo,
        selector_script="#!/usr/bin/env python3\nimport sys\nprint('selector failed')\nsys.exit(7)\n",
        checker_script="#!/usr/bin/env python3\nprint('unit gate should not run')\n",
    )

    result = _run(repo, ["bash", "scripts/local_pr_review.sh"], env={"GITHUB_ACTIONS": "false"})

    assert result.returncode == 1
    assert "Local unit gate mirror" in result.stdout
    assert "selector failed while choosing local unit-gate tests" in result.stdout
    assert "unit gate should not run" not in result.stdout
    assert "1 local review check(s) failed" in result.stdout


def test_local_pr_review_unit_gate_mirror_fails_when_checker_was_removed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    _write_unit_gate_fixture(repo, selector_script="#!/usr/bin/env python3\nprint('FULL')\n")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(repo, "rm", "scripts/check_unit_gate.py")
    _git(repo, "commit", "-m", "remove unit gate checker")

    result = _run(repo, ["bash", "scripts/local_pr_review.sh"], env={"GITHUB_ACTIONS": "false"})

    assert result.returncode == 1
    assert "Local unit gate mirror" in result.stdout
    assert "scripts/check_unit_gate.py is absent from this PR head" in result.stdout
    assert "SKIP (scripts/check_unit_gate.py not found)" not in result.stdout
    assert "1 local review check(s) failed" in result.stdout


def test_local_pr_review_skips_unit_gate_mirror_in_github_actions(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    _write_executable(
        repo / "scripts" / "check_unit_gate.py",
        "#!/usr/bin/env python3\nimport sys\nprint('should not run')\nsys.exit(1)\n",
    )
    _git(repo, "add", "scripts/check_unit_gate.py")
    _git(repo, "commit", "-m", "add unit gate stub")

    result = _run(repo, ["bash", "scripts/local_pr_review.sh"], env={"GITHUB_ACTIONS": "true"})

    assert result.returncode == 0, result.stdout + result.stderr
    assert "SKIP (GitHub Actions runs .github/workflows/unit_gate.yml as its own required check)" in result.stdout
    assert "should not run" not in result.stdout
    assert "local PR review passed" in result.stdout


def test_local_pr_review_env_pr_body_contract_fails_closed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    body = tmp_path / "body.md"
    body.write_text("missing receipt\n", encoding="utf-8")
    _write_executable(
        repo / "scripts" / "audit_pr_body.py",
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "print('body audit rejected ' + sys.argv[-1])\n"
        "sys.exit(1)\n",
    )
    _git(repo, "add", "scripts/audit_pr_body.py")
    _git(repo, "commit", "-m", "add failing body audit")

    result = _run(repo, ["bash", "scripts/local_pr_review.sh"], env={"ATLAS_CURRENT_PR_BODY_FILE": str(body)})

    assert result.returncode == 1
    assert "PR body contract" in result.stdout
    assert f"body audit rejected {body}" in result.stdout
    assert "1 local review check(s) failed" in result.stdout


def test_local_pr_review_real_body_audit_honors_dependabot_exemption(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    body = tmp_path / "body.md"
    body.write_text(
        "Dependabot generated body without the Atlas plan contract.\n"
        "guard-class-closure: waived\n",
        encoding="utf-8",
    )
    _write_executable(
        repo / "scripts" / "audit_pr_body.py",
        (REPO_ROOT / "scripts" / "audit_pr_body.py").read_text(encoding="utf-8"),
    )
    _write_executable(
        repo / "scripts" / "_pr_change_policy.py",
        (REPO_ROOT / "scripts" / "_pr_change_policy.py").read_text(encoding="utf-8"),
    )
    _write_executable(
        repo / "scripts" / "audit_ai_reconciliation.py",
        (REPO_ROOT / "scripts" / "audit_ai_reconciliation.py").read_text(encoding="utf-8"),
    )
    _git(
        repo,
        "add",
        "scripts/audit_pr_body.py",
        "scripts/_pr_change_policy.py",
        "scripts/audit_ai_reconciliation.py",
    )
    _git(repo, "commit", "-m", "add real body audit")

    result = _run(
        repo,
        [
            "bash",
            "scripts/local_pr_review.sh",
            "--current-pr-body-file",
            str(body),
            "--pr-author",
            "app/dependabot",
        ],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Dependabot PR body exempt" in result.stdout


def test_local_pr_review_ai_reconciliation_honors_plain_dependabot_author(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    body = tmp_path / "body.md"
    body.write_text(
        "Dependabot generated body without Atlas reconciliation.\n"
        "guard-class-closure: waived\n",
        encoding="utf-8",
    )
    _write_executable(
        repo / "scripts" / "audit_pr_body.py",
        (REPO_ROOT / "scripts" / "audit_pr_body.py").read_text(encoding="utf-8"),
    )
    _write_executable(
        repo / "scripts" / "_pr_change_policy.py",
        (REPO_ROOT / "scripts" / "_pr_change_policy.py").read_text(encoding="utf-8"),
    )
    _write_executable(
        repo / "scripts" / "audit_ai_reconciliation.py",
        (REPO_ROOT / "scripts" / "audit_ai_reconciliation.py").read_text(encoding="utf-8"),
    )
    _git(
        repo,
        "add",
        "scripts/audit_pr_body.py",
        "scripts/_pr_change_policy.py",
        "scripts/audit_ai_reconciliation.py",
    )
    _git(repo, "commit", "-m", "add real body audit")

    result = _run(
        repo,
        [
            "bash",
            "scripts/local_pr_review.sh",
            "--current-pr-body-file",
            str(body),
            "--pr-author",
            "dependabot",
        ],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Dependabot PR body exempt" in result.stdout
    assert "no 'AI reconciliation' section" not in result.stdout


def test_local_pr_review_real_body_audit_honors_docs_only_exemption(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    body = tmp_path / "body.md"
    body.write_text("Docs-only: true\n\nCorrect docs.\n", encoding="utf-8")
    _write_executable(
        repo / "scripts" / "audit_pr_body.py",
        (REPO_ROOT / "scripts" / "audit_pr_body.py").read_text(encoding="utf-8"),
    )
    _write_executable(
        repo / "scripts" / "_pr_change_policy.py",
        (REPO_ROOT / "scripts" / "_pr_change_policy.py").read_text(encoding="utf-8"),
    )
    _write_executable(
        repo / "scripts" / "audit_ai_reconciliation.py",
        (REPO_ROOT / "scripts" / "audit_ai_reconciliation.py").read_text(encoding="utf-8"),
    )
    _git(
        repo,
        "add",
        "scripts/audit_pr_body.py",
        "scripts/_pr_change_policy.py",
        "scripts/audit_ai_reconciliation.py",
    )
    _git(repo, "commit", "-m", "add docs-only audit fixture")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    doc = repo / "docs" / "docs-only.md"
    doc.parent.mkdir()
    doc.write_text("# docs only\n", encoding="utf-8")
    _git(repo, "add", "docs/docs-only.md")
    _git(repo, "commit", "-m", "docs only")

    result = _run(
        repo,
        ["bash", "scripts/local_pr_review.sh", "--current-pr-body-file", str(body)],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "explicit Markdown-only body exemption" in result.stdout
    assert "no 'AI reconciliation' section" not in result.stdout


def test_local_pr_review_runs_plans_archive_advisory_when_present(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    # A stub that deliberately exits non-zero. The real archive_plans.py check
    # always exits 0; this simulates an *unexpected* failure to prove the advisory
    # stays non-blocking regardless, because it is wrapped in `|| true`.
    _write_executable(
        repo / "scripts" / "archive_plans.py",
        "#!/usr/bin/env python3\nimport sys\nprint('WARNING: plans backlog advisory ran')\nsys.exit(1)\n",
    )
    _git(repo, "add", "scripts/archive_plans.py")
    _git(repo, "commit", "-m", "add archive_plans stub")

    result = _run(repo, ["bash", "scripts/local_pr_review.sh"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Plans archive backlog (advisory, non-blocking)" in result.stdout
    assert "plans backlog advisory ran" in result.stdout
    assert "local main synced to origin/main" in result.stdout
    assert "move only that plan by name" in result.stdout
    assert "python scripts/archive_plans.py archive" not in result.stdout
    assert "local PR review passed" in result.stdout


def test_local_pr_review_trusted_script_root_does_not_execute_repo_scripts(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    _write_executable(
        repo / "scripts" / "pre_push_audit.sh",
        "#!/usr/bin/env bash\necho repo pre-push should not run >&2\nexit 99\n",
    )
    _git(repo, "add", "scripts/pre_push_audit.sh")
    _git(repo, "commit", "-m", "hostile repo pre-push")

    trusted = tmp_path / "trusted"
    (trusted / "scripts").mkdir(parents=True)
    _write_executable(
        trusted / "scripts" / "local_pr_review.sh",
        (REPO_ROOT / "scripts" / "local_pr_review.sh").read_text(encoding="utf-8"),
    )
    _write_executable(
        trusted / "scripts" / "pre_push_audit.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf 'trusted pre-push cwd=%s\\n' \"$PWD\"\n",
    )

    result = _run(
        tmp_path,
        [
            "bash",
            str(trusted / "scripts" / "local_pr_review.sh"),
            "--repo-root",
            str(repo),
            "--script-root",
            str(trusted),
        ],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"trusted pre-push cwd={repo}" in result.stdout
    assert "repo pre-push should not run" not in result.stderr


def test_local_pr_review_trusted_guard_checker_inspects_repo_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    _add_guard_shaped_change(repo)

    trusted = tmp_path / "trusted"
    (trusted / "scripts").mkdir(parents=True)
    _write_executable(
        trusted / "scripts" / "local_pr_review.sh",
        (REPO_ROOT / "scripts" / "local_pr_review.sh").read_text(encoding="utf-8"),
    )
    _write_executable(
        trusted / "scripts" / "check_guard_class_closure.py",
        (REPO_ROOT / "scripts" / "check_guard_class_closure.py").read_text(encoding="utf-8"),
    )
    _write_executable(
        trusted / "scripts" / "pre_push_audit.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\necho trusted pre-push ok\n",
    )

    result = _run(
        tmp_path,
        [
            "bash",
            str(trusted / "scripts" / "local_pr_review.sh"),
            "--repo-root",
            str(repo),
            "--script-root",
            str(trusted),
        ],
    )

    assert result.returncode == 1
    assert "Guard class-closure lint" in result.stdout
    assert "extracted_content_pipeline/support_ticket_privacy.py" in result.stdout
    assert "guard-shaped change over open input" in result.stdout


def test_local_pr_review_body_audit_inspects_repo_root_plan_with_trusted_scripts(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    plan = repo / "plans" / "PR-TrustedSplit.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# PR-TrustedSplit\n", encoding="utf-8")
    _git(repo, "add", "plans/PR-TrustedSplit.md")
    _git(repo, "commit", "-m", "add PR-only plan")

    body = tmp_path / "body.md"
    body.write_text(_valid_pr_body("plans/PR-TrustedSplit.md"), encoding="utf-8")

    trusted = tmp_path / "trusted"
    (trusted / "scripts").mkdir(parents=True)
    _write_executable(
        trusted / "scripts" / "local_pr_review.sh",
        (REPO_ROOT / "scripts" / "local_pr_review.sh").read_text(encoding="utf-8"),
    )
    _write_executable(
        trusted / "scripts" / "audit_pr_body.py",
        (REPO_ROOT / "scripts" / "audit_pr_body.py").read_text(encoding="utf-8"),
    )
    _write_executable(
        trusted / "scripts" / "_pr_change_policy.py",
        (REPO_ROOT / "scripts" / "_pr_change_policy.py").read_text(encoding="utf-8"),
    )
    _write_executable(
        trusted / "scripts" / "audit_ai_reconciliation.py",
        (REPO_ROOT / "scripts" / "audit_ai_reconciliation.py").read_text(encoding="utf-8"),
    )
    _write_executable(
        trusted / "scripts" / "pre_push_audit.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\necho trusted pre-push ok\n",
    )

    result = _run(
        tmp_path,
        [
            "bash",
            str(trusted / "scripts" / "local_pr_review.sh"),
            "--repo-root",
            str(repo),
            "--script-root",
            str(trusted),
            "--current-pr-body-file",
            str(body),
        ],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "pr body audit: PASS" in result.stdout
    assert "trusted pre-push ok" in result.stdout


def test_local_pr_review_skips_plans_advisory_when_absent(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)

    result = _run(repo, ["bash", "scripts/local_pr_review.sh"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "SKIP (scripts/archive_plans.py not found)" in result.stdout


def _valid_pr_body(plan: str) -> str:
    return "\n".join([
        f"Plan: {plan}",
        "Slice phase: Production hardening",
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
        "- Changed: docs/example.md:1 records the workflow rule.",
        "- Contract match: traces to the process contract.",
        "- Gaps: none.",
        "",
        "## Verification",
        "- pytest passed",
        "",
        "## Mechanical verification",
        "- Command: pytest tests/test_local_pr_review.py - Result: passed - Environment: local",
        "",
        "## Diff size",
        "2 files, +10 / -2",
    ])


def _write_fixture_repo(repo: Path) -> None:
    (repo / "scripts").mkdir(parents=True)
    _write_executable(
        repo / "scripts" / "local_pr_review.sh",
        (REPO_ROOT / "scripts" / "local_pr_review.sh").read_text(encoding="utf-8"),
    )
    _write_executable(
        repo / "scripts" / "check_guard_class_closure.py",
        (REPO_ROOT / "scripts" / "check_guard_class_closure.py").read_text(encoding="utf-8"),
    )
    _write_executable(
        repo / "scripts" / "pre_push_audit.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\necho pre-push ok\n",
    )
    _write_executable(
        repo / "scripts" / "audit_plan_code_consistency.py",
        "#!/usr/bin/env python3\nprint('plan ok')\n",
    )
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    _git(repo, "branch", "-M", "main")
    _git(repo, "remote", "add", "origin", str(repo))
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(repo, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main")


def _add_guard_shaped_change(repo: Path) -> None:
    guard = repo / "extracted_content_pipeline" / "support_ticket_privacy.py"
    guard.parent.mkdir(parents=True, exist_ok=True)
    guard.write_text(
        "def is_private_marker(value):\n"
        "    if isinstance(value, (str, dict)):\n"
        "        return value.strip().lower() in {'not published'}\n"
        "    return False\n",
        encoding="utf-8",
    )
    _git(repo, "add", "extracted_content_pipeline/support_ticket_privacy.py")
    _git(repo, "commit", "-m", "add guard-shaped symptom patch")


def _write_unit_gate_fixture(
    repo: Path,
    *,
    selector_script: str,
    checker_script: str | None = None,
) -> None:
    _write_executable(repo / "scripts" / "select_impacted_tests.py", selector_script)
    _write_executable(
        repo / "scripts" / "check_unit_gate.py",
        checker_script
        or "#!/usr/bin/env python3\nimport sys\nprint('unit gate args=' + ' '.join(sys.argv[1:]))\n",
    )
    (repo / "tests").mkdir(exist_ok=True)
    (repo / "tests" / "test_example.py").write_text("def test_example():\n    assert True\n", encoding="utf-8")
    (repo / "tests" / "unit_gate_baseline.txt").write_text("", encoding="utf-8")
    _git(
        repo,
        "add",
        "scripts/select_impacted_tests.py",
        "scripts/check_unit_gate.py",
        "tests/test_example.py",
        "tests/unit_gate_baseline.txt",
    )
    _git(repo, "commit", "-m", "add unit gate stubs")


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
