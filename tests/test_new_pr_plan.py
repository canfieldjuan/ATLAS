from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "new_pr_plan.sh"
AUDIT_PLAN_DOC = REPO_ROOT / "scripts" / "audit_plan_doc.py"


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)


def _run_new_plan(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )


def test_new_pr_plan_creates_agents_plan_skeleton(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)

    result = _run_new_plan(
        tmp_path,
        "Dev-Workflow-Example",
        "--lane",
        "dev-workflow/test",
        "--phase",
        "Workflow/process",
    )

    assert result.returncode == 0
    assert "created plan scaffold: plans/PR-Dev-Workflow-Example.md" in result.stdout
    plan_path = tmp_path / "plans" / "PR-Dev-Workflow-Example.md"
    text = plan_path.read_text(encoding="utf-8")
    assert text.startswith("# PR-Dev-Workflow-Example\n")
    assert "Ownership lane: dev-workflow/test" in text
    assert "Slice phase: Workflow/process" in text
    assert "### Files touched" in text
    assert "| **Total** | **0** |" in text

    audit = subprocess.run(
        ["python", str(AUDIT_PLAN_DOC), str(plan_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert audit.returncode == 0


def test_new_pr_plan_does_not_double_prefix_existing_pr_name(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)

    result = _run_new_plan(tmp_path, "PR-Already-Prefixed")

    assert result.returncode == 0
    assert (tmp_path / "plans" / "PR-Already-Prefixed.md").exists()
    assert not (tmp_path / "plans" / "PR-PR-Already-Prefixed.md").exists()


def test_new_pr_plan_refuses_existing_plan_without_force(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    first = _run_new_plan(tmp_path, "Overwrite-Check", "--phase", "Workflow/process")
    assert first.returncode == 0
    plan_path = tmp_path / "plans" / "PR-Overwrite-Check.md"
    original = plan_path.read_text(encoding="utf-8")

    second = _run_new_plan(tmp_path, "Overwrite-Check", "--phase", "Vertical slice")

    assert second.returncode == 2
    assert "plan already exists" in second.stderr
    assert plan_path.read_text(encoding="utf-8") == original


def test_new_pr_plan_force_overwrites_existing_plan(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    first = _run_new_plan(tmp_path, "Force-Check", "--phase", "Workflow/process")
    assert first.returncode == 0

    second = _run_new_plan(tmp_path, "Force-Check", "--phase", "Vertical slice", "--force")

    assert second.returncode == 0
    text = (tmp_path / "plans" / "PR-Force-Check.md").read_text(encoding="utf-8")
    assert "Slice phase: Vertical slice" in text


def test_new_pr_plan_rejects_missing_and_unsafe_slice_names(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)

    missing = _run_new_plan(tmp_path)
    traversal = _run_new_plan(tmp_path, "../Bad")
    separator = _run_new_plan(tmp_path, "bad/name")
    empty_prefix = _run_new_plan(tmp_path, "PR-")

    assert missing.returncode == 2
    assert "missing slice name" in missing.stderr
    assert traversal.returncode == 2
    assert "unsafe slice name" in traversal.stderr
    assert separator.returncode == 2
    assert "unsafe slice name" in separator.stderr
    assert empty_prefix.returncode == 2
    assert "slice name must include text after PR-" in empty_prefix.stderr


def test_new_pr_plan_requires_option_values(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)

    lane = _run_new_plan(tmp_path, "Option-Check", "--lane")
    phase = _run_new_plan(tmp_path, "Option-Check", "--phase")

    assert lane.returncode == 2
    assert "--lane requires a value" in lane.stderr
    assert phase.returncode == 2
    assert "--phase requires a value" in phase.stderr
