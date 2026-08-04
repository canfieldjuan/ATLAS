from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_pr_branch_name.py"
SPEC = importlib.util.spec_from_file_location("check_pr_branch_name", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def _body(plan: str = "plans/PR-Branch-Naming-Gate.md") -> str:
    return f"Plan: {plan}\n\n## Intentional\n- none\n"


def test_plan_slug_normalizes_plan_name_to_branch_slug() -> None:
    assert checker.plan_slug("EOM_First.Clean-Won") == "eom-first-clean-won"


def test_branch_name_accepts_matching_plan_branch() -> None:
    assert checker.branch_name_errors(
        branch="claude/pr-branch-naming-gate",
        body=_body(),
    ) == []


def test_branch_name_rejects_non_pr_prefix() -> None:
    errors = checker.branch_name_errors(
        branch="claude/branch-naming-gate",
        body=_body(),
    )

    assert "branch must match claude/pr-<slice-name>" in errors[0]


def test_branch_name_rejects_plan_slug_mismatch() -> None:
    errors = checker.branch_name_errors(
        branch="claude/pr-other",
        body=_body(),
    )

    assert errors == [
        "branch 'claude/pr-other' does not match PR plan branch "
        "'claude/pr-branch-naming-gate'"
    ]


def test_docs_only_body_requires_pr_prefix_only() -> None:
    assert checker.branch_name_errors(
        branch="claude/pr-docs-typo",
        body="Docs-only: true\n\nFix docs.\n",
    ) == []


def test_body_without_plan_or_docs_only_marker_fails() -> None:
    errors = checker.branch_name_errors(
        branch="claude/pr-branch-naming-gate",
        body="No marker here\n",
    )

    assert errors == [
        "PR body must begin with Plan: plans/PR-<Slice>.md or Docs-only: true"
    ]


def test_detached_branch_fails() -> None:
    assert checker.branch_name_errors(branch="", body=_body()) == [
        "current checkout is detached; switch to a PR branch first"
    ]


def test_cli_parser_rejects_missing_required_arguments() -> None:
    with pytest.raises(SystemExit):
        checker.main([])
