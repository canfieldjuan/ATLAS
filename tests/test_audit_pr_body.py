"""Tests for the AGENTS.md section 1b PR-body contract audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_pr_body.py"
SPEC = importlib.util.spec_from_file_location("audit_pr_body", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit_pr_body_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit_pr_body_module
SPEC.loader.exec_module(audit_pr_body_module)

audit_pr_body = audit_pr_body_module.audit_pr_body
is_dependabot_author = audit_pr_body_module.is_dependabot_author


def _valid_body(plan: str = "plans/PR-Example.md") -> str:
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
        "## Diff size",
        "2 files, +10 / -2",
    ])


def _write_plan(tmp_path: Path, plan: str = "plans/PR-Example.md") -> Path:
    plan_path = tmp_path / plan
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("# PR-Example\n", encoding="utf-8")
    return tmp_path


def test_valid_body_passes(tmp_path: Path) -> None:
    root = _write_plan(tmp_path)

    assert audit_pr_body(_valid_body(), root=root) == []


def test_empty_body_fails() -> None:
    assert audit_pr_body("  \n\n") == ["PR body is empty"]


def test_missing_plan_lead_line_fails(tmp_path: Path) -> None:
    root = _write_plan(tmp_path)
    body = _valid_body().replace("Plan: plans/PR-Example.md", "Overview first")

    failures = audit_pr_body(body, root=root)

    assert any("first non-empty line" in failure for failure in failures)


def test_nonexistent_plan_doc_fails(tmp_path: Path) -> None:
    failures = audit_pr_body(_valid_body(), root=tmp_path)

    assert any("does not exist" in failure for failure in failures)


def test_symlinked_working_tree_plan_doc_fails(tmp_path: Path) -> None:
    target = tmp_path / "real-plan.md"
    target.write_text("# real target\n", encoding="utf-8")
    plan = tmp_path / "plans" / "PR-Example.md"
    plan.parent.mkdir(parents=True)
    plan.symlink_to(target)

    failures = audit_pr_body(_valid_body(), root=tmp_path)

    assert any("does not exist" in failure for failure in failures)


def test_working_tree_plan_under_symlinked_parent_fails(tmp_path: Path) -> None:
    target_dir = tmp_path / "actual-plans"
    target_dir.mkdir()
    (target_dir / "PR-Example.md").write_text("# real target\n", encoding="utf-8")
    (tmp_path / "plans").symlink_to(target_dir, target_is_directory=True)

    failures = audit_pr_body(_valid_body(), root=tmp_path)

    assert any("does not exist" in failure for failure in failures)


def test_missing_one_paragraph_why_fails(tmp_path: Path) -> None:
    root = _write_plan(tmp_path)
    body = "\n".join([
        "Plan: plans/PR-Example.md",
        "Slice phase: Production hardening",
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
        "- Gaps: none.",
        "",
        "## Verification",
        "- pytest passed",
        "",
        "## Diff size",
        "2 files, +10 / -2",
    ])

    failures = audit_pr_body(body, root=root)

    assert any("one-paragraph why" in failure for failure in failures)


def test_missing_slice_phase_fails(tmp_path: Path) -> None:
    root = _write_plan(tmp_path)
    body = _valid_body().replace("Slice phase: Production hardening\n", "")

    failures = audit_pr_body(body, root=root)

    assert any("Slice phase" in failure for failure in failures)


@pytest.mark.parametrize(
    ("replacement", "expected"),
    (
        ("", "exactly one 'Ownership lane:'"),
        ("Ownership lane: Dev Workflow", "lowercase letters"),
        (
            "Ownership lane: dev-workflow/process-gate-enrollment\n"
            "Ownership lane: dev-workflow/other",
            "exactly one 'Ownership lane:'",
        ),
    ),
)
def test_lane_header_contract_rejects_invalid_shapes(
    tmp_path: Path, replacement: str, expected: str
) -> None:
    root = _write_plan(tmp_path)
    body = _valid_body().replace(
        "Ownership lane: dev-workflow/process-gate-enrollment", replacement,
        1,
    )

    failures = audit_pr_body(body, root=root)

    assert any(expected in failure for failure in failures)


def test_lane_header_must_immediately_follow_slice_phase(tmp_path: Path) -> None:
    root = _write_plan(tmp_path)
    body = _valid_body().replace(
        "Slice phase: Production hardening\nOwnership lane:",
        "Slice phase: Production hardening\n\nOwnership lane:",
    )

    failures = audit_pr_body(body, root=root)

    assert any("immediately after" in failure for failure in failures)


@pytest.mark.parametrize("fence", ("```", "~~~"))
def test_fenced_example_cannot_supply_body_contract_structure(
    tmp_path: Path, fence: str
) -> None:
    root = _write_plan(tmp_path)
    body = "\n".join([
        "Plan: plans/PR-Example.md",
        "",
        f"{fence}md",
        *(_valid_body().splitlines()[1:]),
        fence,
    ])

    failures = audit_pr_body(body, root=root)

    assert any("missing canonical 'Slice phase" in failure for failure in failures)
    assert "missing required section: ## Intentional" in failures


def test_metadata_after_why_paragraph_cannot_supply_canonical_header(tmp_path: Path) -> None:
    root = _write_plan(tmp_path)
    body = _valid_body().replace(
        "Plan: plans/PR-Example.md\nSlice phase: Production hardening\n"
        "Ownership lane: dev-workflow/process-gate-enrollment",
        "Plan: plans/PR-Example.md\nOne-paragraph why.\n"
        "Slice phase: Production hardening\n"
        "Ownership lane: dev-workflow/process-gate-enrollment",
    )

    failures = audit_pr_body(body, root=root)

    assert any("missing canonical 'Slice phase" in failure for failure in failures)


def test_missing_section_fails(tmp_path: Path) -> None:
    root = _write_plan(tmp_path)
    body = _valid_body().replace("## Parked hardening\nNone.\n", "")

    failures = audit_pr_body(body, root=root)

    assert "missing required section: ## Parked hardening" in failures


def test_missing_cold_diff_reconstruction_fails(tmp_path: Path) -> None:
    root = _write_plan(tmp_path)
    body = _valid_body().replace(
        "## Cold diff reconstruction\n"
        "- Changed: docs/example.md:1 records the workflow rule.\n"
        "- Contract match: traces to the process contract.\n"
        "- Gaps: none.\n\n",
        "",
    )

    failures = audit_pr_body(body, root=root)

    assert "missing required section: ## Cold diff reconstruction" in failures


def test_out_of_order_sections_fail(tmp_path: Path) -> None:
    root = _write_plan(tmp_path)
    body = "\n".join([
        "Plan: plans/PR-Example.md",
        "Slice phase: Production hardening",
        "",
        "One-paragraph why.",
        "",
        "## Deferred",
        "- a follow-up",
        "",
        "## Intentional",
        "- a trade-off",
        "",
        "## Parked hardening",
        "None.",
        "",
        "## Cold diff reconstruction",
        "- Gaps: none.",
        "",
        "## Verification",
        "- pytest passed",
        "",
        "## Diff size",
        "2 files, +10 / -2",
    ])

    failures = audit_pr_body(body, root=root)

    assert any("out of order" in failure for failure in failures)


def test_extra_sections_between_required_ones_pass(tmp_path: Path) -> None:
    root = _write_plan(tmp_path)
    body = _valid_body().replace(
        "## Verification",
        "## Review notes\nExtra context.\n\n## Verification",
    )

    assert audit_pr_body(body, root=root) == []


def test_slice_phase_after_first_heading_fails(tmp_path: Path) -> None:
    root = _write_plan(tmp_path)
    body = _valid_body().replace("Slice phase: Production hardening\n", "")
    body = body.replace(
        "## Intentional",
        "## Intentional\nSlice phase: Production hardening",
    )

    failures = audit_pr_body(body, root=root)

    assert any("Slice phase" in failure for failure in failures)


def test_dependabot_author_detection() -> None:
    assert is_dependabot_author("app/dependabot")
    assert is_dependabot_author("dependabot[bot]")
    assert is_dependabot_author(" dependabot ")
    assert not is_dependabot_author("canfieldjuan")
    assert not is_dependabot_author(None)


def test_dependabot_cli_exempts_invalid_body() -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as body_file:
        body_file.write("Dependabot generated body without the Atlas plan contract.\n")
        body_path = Path(body_file.name)

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--pr-author",
                "app/dependabot",
                str(body_path),
            ],
            check=False,
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0
        assert "Dependabot PR body exempt" in result.stdout
    finally:
        body_path.unlink(missing_ok=True)


def test_normal_author_cli_rejects_same_invalid_body() -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as body_file:
        body_file.write("Dependabot generated body without the Atlas plan contract.\n")
        body_path = Path(body_file.name)

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--pr-author",
                "canfieldjuan",
                str(body_path),
            ],
            check=False,
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 1
        assert "AGENTS.md section 1b contract" in result.stdout
    finally:
        body_path.unlink(missing_ok=True)


def test_docs_only_marker_passes_for_markdown_only_human_diff(tmp_path: Path) -> None:
    repo = _git_repo_with_changed_path(tmp_path, "docs/example.md", "# changed\n")
    body = tmp_path / "body.md"
    body.write_text("Docs-only: true\n\nCorrect a documentation typo.\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--base-ref",
            "origin/main",
            str(body),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "explicit Markdown-only body exemption" in result.stdout


def test_docs_only_marker_passes_for_deleted_markdown_file(tmp_path: Path) -> None:
    repo = _git_repo_with_deleted_markdown_path(tmp_path)
    body = tmp_path / "body.md"
    body.write_text("Docs-only: true\n\nArchive a completed plan.\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--base-ref",
            "origin/main",
            str(body),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "explicit Markdown-only body exemption" in result.stdout


def test_docs_only_marker_passes_for_archived_plan_move(tmp_path: Path) -> None:
    repo = _git_repo_with_archived_plan_move(tmp_path)
    body = tmp_path / "body.md"
    body.write_text("Docs-only: true\n\nArchive a completed plan.\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--base-ref",
            "origin/main",
            str(body),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "explicit Markdown-only body exemption" in result.stdout


def test_docs_only_marker_can_inspect_a_fetched_head_ref(tmp_path: Path) -> None:
    repo = _git_repo_with_changed_path(tmp_path, "docs/example.md", "# changed\n")
    _git(repo, "update-ref", "refs/remotes/origin/pr-1", "HEAD")
    _git(repo, "checkout", "-q", "--detach", "origin/main")
    body = tmp_path / "body.md"
    body.write_text("Docs-only: true\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--base-ref",
            "origin/main",
            "--head-ref",
            "refs/remotes/origin/pr-1",
            str(body),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "explicit Markdown-only body exemption" in result.stdout


def test_docs_only_marker_rejects_a_branch_added_plan(tmp_path: Path) -> None:
    repo = _git_repo_with_changed_path(tmp_path, "plans/PR-Example.md", "# Plan\n")
    body = tmp_path / "body.md"
    body.write_text("Docs-only: true\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--base-ref",
            "origin/main",
            str(body),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "with no branch-added plan" in result.stdout


def test_docs_only_marker_rejects_non_markdown_diff(tmp_path: Path) -> None:
    repo = _git_repo_with_changed_path(tmp_path, "scripts/example.py", "print('changed')\n")
    body = tmp_path / "body.md"
    body.write_text("Docs-only: true\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--base-ref",
            "origin/main",
            str(body),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "valid only for a non-empty Markdown-only human diff" in result.stdout


def test_docs_only_marker_requires_base_ref(tmp_path: Path) -> None:
    body = tmp_path / "body.md"
    body.write_text("Docs-only: true\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(body)],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "requires --base-ref" in result.stderr


@pytest.mark.parametrize(
    "existing_plan",
    (
        "plans/PR-Archived.md",
        "plans/PR-Existing.md",
        "plans/PR-Legacy.md",
        "plans/PR-Previous.md",
        "plans/PR-Reference.md",
        "plans/PR-Unrelated.md",
    ),
)
def test_full_body_rejects_each_unrelated_existing_plan(
    tmp_path: Path, existing_plan: str
) -> None:
    repo = _git_repo_with_plan_required_diff(tmp_path, existing_plan=existing_plan)
    body = tmp_path / "body.md"
    body.write_text(_valid_body(plan=existing_plan), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--base-ref",
            "origin/main",
            str(body),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "sole branch-added plan" in result.stdout
    assert "plans/PR-Added.md" in result.stdout


def test_full_body_accepts_the_sole_branch_added_plan(tmp_path: Path) -> None:
    repo = _git_repo_with_plan_required_diff(
        tmp_path, existing_plan="plans/PR-Existing.md"
    )
    body = tmp_path / "body.md"
    body.write_text(_valid_body(plan="plans/PR-Added.md"), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--base-ref",
            "origin/main",
            str(body),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_full_body_requires_docs_only_marker_without_a_branch_added_plan(
    tmp_path: Path,
) -> None:
    repo = _git_repo_with_docs_only_diff(
        tmp_path, existing_plan="plans/PR-Existing.md", add_plan=False
    )
    body = tmp_path / "body.md"
    body.write_text(_valid_body(plan="plans/PR-Existing.md"), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--base-ref",
            "origin/main",
            str(body),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "Docs-only: true" in result.stdout


def test_full_body_accepts_a_branch_added_plan_for_markdown_only_diff(tmp_path: Path) -> None:
    repo = _git_repo_with_docs_only_diff(
        tmp_path, existing_plan="plans/PR-Existing.md", add_plan=True
    )
    body = tmp_path / "body.md"
    body.write_text(_valid_body(plan="plans/PR-Added.md"), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--base-ref",
            "origin/main",
            str(body),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_repo_root_checks_plan_in_inspected_checkout(tmp_path: Path) -> None:
    repo = _write_plan(tmp_path / "inspected", "plans/PR-OnlyInInspected.md")
    body = tmp_path / "body.md"
    body.write_text(_valid_body(plan="plans/PR-OnlyInInspected.md"), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(repo),
            str(body),
        ],
        check=False,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "pr body audit: PASS" in result.stdout


def test_cli_repo_root_rejects_symlinked_plan_in_inspected_checkout(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "inspected"
    target = repo / "real-plan.md"
    target.parent.mkdir()
    target.write_text("# real target\n", encoding="utf-8")
    plan = repo / "plans" / "PR-Example.md"
    plan.parent.mkdir()
    plan.symlink_to(target)
    body = tmp_path / "body.md"
    body.write_text(_valid_body(), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(repo),
            str(body),
        ],
        check=False,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "plan doc named in the PR body does not exist" in result.stdout


# -- trusted-base plan-doc inspection (--plan-git-ref) ---------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _git_repo_with_plan(tmp_path: Path, plan: str = "plans/PR-Example.md") -> Path:
    repo = tmp_path / "repo"
    (repo / plan).parent.mkdir(parents=True)
    (repo / plan).write_text("# PR-Example\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "seed")
    return repo


def _git_repo_with_changed_path(tmp_path: Path, relative_path: str, text: str) -> Path:
    repo = tmp_path / "repo"
    (repo / "README.md").parent.mkdir(parents=True)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "branch", "-M", "main")
    _git(repo, "remote", "add", "origin", str(repo))
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    changed = repo / relative_path
    changed.parent.mkdir(parents=True, exist_ok=True)
    changed.write_text(text, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "change")
    return repo


def _git_repo_with_deleted_markdown_path(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "example.md").write_text("# archived\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base documentation")
    _git(repo, "branch", "-M", "main")
    _git(repo, "remote", "add", "origin", str(repo))
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(repo, "rm", "docs/example.md")
    _git(repo, "commit", "-qm", "archive documentation")
    return repo


def _git_repo_with_archived_plan_move(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "plans").mkdir(parents=True)
    (repo / "plans" / "PR-Completed.md").write_text("# completed\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base plan")
    _git(repo, "branch", "-M", "main")
    _git(repo, "remote", "add", "origin", str(repo))
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    (repo / "plans" / "archive").mkdir()
    _git(repo, "mv", "plans/PR-Completed.md", "plans/archive/PR-Completed.md")
    _git(repo, "commit", "-qm", "archive completed plan")
    return repo


def _git_repo_with_plan_required_diff(tmp_path: Path, *, existing_plan: str) -> Path:
    repo = tmp_path / "repo"
    (repo / existing_plan).parent.mkdir(parents=True)
    (repo / existing_plan).write_text("# existing plan\n", encoding="utf-8")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "branch", "-M", "main")
    _git(repo, "remote", "add", "origin", str(repo))
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "example.py").write_text("print('changed')\n", encoding="utf-8")
    (repo / "plans" / "PR-Added.md").write_text("# added plan\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "planned code change")
    return repo


def _git_repo_with_docs_only_diff(
    tmp_path: Path, *, existing_plan: str, add_plan: bool
) -> Path:
    repo = tmp_path / "repo"
    (repo / existing_plan).parent.mkdir(parents=True)
    (repo / existing_plan).write_text("# existing plan\n", encoding="utf-8")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "branch", "-M", "main")
    _git(repo, "remote", "add", "origin", str(repo))
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    (repo / "docs").mkdir()
    (repo / "docs" / "example.md").write_text("# changed\n", encoding="utf-8")
    if add_plan:
        (repo / "plans" / "PR-Added.md").write_text("# added plan\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "planned documentation change")
    return repo


def test_plan_exists_callable_is_injectable() -> None:
    failures = audit_pr_body(_valid_body(), plan_exists=lambda plan: False)
    assert any("does not exist" in failure for failure in failures)
    assert audit_pr_body(_valid_body(), plan_exists=lambda plan: True) == []


def test_plan_exists_at_ref_sees_committed_plan_and_misses_absent_one(
    tmp_path: Path,
) -> None:
    repo = _git_repo_with_plan(tmp_path)
    exists = audit_pr_body_module.plan_exists_at_ref("HEAD", repo_root=repo)
    assert exists("plans/PR-Example.md") is True
    assert exists("plans/PR-Not-There.md") is False


def test_audit_against_ref_fails_when_plan_missing_at_ref(tmp_path: Path) -> None:
    repo = _git_repo_with_plan(tmp_path)
    exists = audit_pr_body_module.plan_exists_at_ref("HEAD", repo_root=repo)
    failures = audit_pr_body(
        _valid_body(plan="plans/PR-Not-There.md"), plan_exists=exists
    )
    assert any("does not exist" in failure for failure in failures)


def test_cli_plan_git_ref_uses_repo_root(tmp_path: Path) -> None:
    repo = _git_repo_with_plan(tmp_path)
    body = tmp_path / "body.md"
    body.write_text(_valid_body(), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--plan-git-ref",
            "HEAD",
            str(body),
        ],
        check=False,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "pr body audit: PASS" in result.stdout


def test_resolve_git_ref_true_for_head_false_for_unfetched(tmp_path: Path) -> None:
    repo = _git_repo_with_plan(tmp_path)
    assert audit_pr_body_module.resolve_git_ref("HEAD", repo_root=repo) is True
    assert (
        audit_pr_body_module.resolve_git_ref(
            "refs/remotes/origin/pr-999", repo_root=repo
        )
        is False
    )


def test_cli_unresolvable_plan_ref_is_infra_failure_exit_2() -> None:
    """A trusted-base run whose PR-head fetch did not happen must exit 2
    (infrastructure), never 0 -- the gate cannot silently pass."""
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as handle:
        handle.write(_valid_body())
        body_file = handle.name
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--plan-git-ref",
            "refs/remotes/origin/pr-0",
            body_file,
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "not resolvable" in proc.stderr


def test_plan_path_that_is_a_tree_at_ref_is_not_a_plan_doc(tmp_path: Path) -> None:
    """cat-file -e would accept a TREE at the plan path (a PR shipping
    plans/PR-Example.md/child); only a blob counts as a plan doc."""
    repo = tmp_path / "repo"
    nested = repo / "plans" / "PR-Example.md"
    nested.mkdir(parents=True)
    (nested / "child").write_text("not a plan doc\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "seed")
    exists = audit_pr_body_module.plan_exists_at_ref("HEAD", repo_root=repo)
    assert exists("plans/PR-Example.md") is False
    failures = audit_pr_body(_valid_body(), plan_exists=exists)
    assert any("does not exist" in failure for failure in failures)


def test_plan_path_that_is_a_symlink_at_ref_is_not_a_plan_doc(tmp_path: Path) -> None:
    """cat-file -t reports symlinks as blobs (mode 120000, possibly
    dangling); the working-tree is_file() rejects dangling symlinks, so
    the ref checker requires a regular-file mode."""
    import os

    repo = tmp_path / "repo"
    (repo / "plans").mkdir(parents=True)
    os.symlink("nowhere-real.md", repo / "plans" / "PR-Example.md")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "seed")
    exists = audit_pr_body_module.plan_exists_at_ref("HEAD", repo_root=repo)
    assert exists("plans/PR-Example.md") is False
