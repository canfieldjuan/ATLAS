from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "new_pr_plan.sh"
AUDIT_PLAN_DOC = REPO_ROOT / "scripts" / "audit_plan_doc.py"


def _compact_ws(text: str) -> str:
    return " ".join(text.split())


def _init_git_repo(path: Path, *, with_state: bool = True) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    if with_state:
        _write_state(path)


def _write_state(path: Path, lane: str = "dev-workflow/test") -> Path:
    state = path / "SESSION_STATE.local.md"
    state.write_text(
        "# Atlas Builder Session State\n\n"
        f"Current lane: {lane}\n\n"
        "## Owned Active PR\n\nPR: none\n",
        encoding="utf-8",
    )
    return state


def _run_new_plan(
    path: Path,
    *args: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=cwd or path,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "ATLAS_SESSION_STATE_FILE": "", **(env or {})},
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
    compact_text = _compact_ws(text)
    assert text.startswith("# PR-Dev-Workflow-Example\n")
    assert "Ownership lane: dev-workflow/test" in text
    assert "Slice phase: Workflow/process" in text
    assert "### Problem-derived contract" in text
    assert "- Root cause:" in text
    assert "- Correct fix must touch/change:" in text
    assert "- Must not change:" in text
    assert "### Review Contract" in text
    assert "- Acceptance criteria:" in text
    assert "- Reachability proof:" in text
    assert "- Affected surfaces:" in text
    assert "- Risk areas:" in text
    assert "- Reviewer rules triggered:" in text
    # The scaffold must keep PROMPTING for a settleable contract, not merely
    # emit the headings: a newly authored acceptance criterion with no code
    # claim or settling evidence is an authoring defect, while risk areas remain
    # exempt because the review matrix does not disposition them.
    # Asserting only the headings let those prompts be deleted while green.
    assert "names a claim about the code, or the evidence that settles it" in text
    assert "Do NOT name a BARE risk category" in text
    assert "fail authoring until the builder names the code claim" in compact_text
    assert "Naming the evidence rescues it" in compact_text
    assert "3k.3 evidence-gated mechanism" in compact_text
    assert "sampled fixture list alone is not enough" in compact_text
    assert "3k.4 execution model and property-level invariant" in compact_text
    assert "sampled concurrent test alone is not enough" in compact_text
    assert "not\n  dispositioned by the review matrix" in text
    assert "### Boundary-change enumeration" in text
    assert "Required when this diff changes a guard, validator, resolver, or admission" in text
    assert "- Replaced-path behaviors: TODO/N/A." in text
    assert "- Guard-relevant fields: TODO/N/A." in text
    assert "- Caller x input shape: TODO/N/A." in text
    assert "### Files touched" in text
    assert "| **Total** | **0** |" in text

    audit = subprocess.run(
        ["python", str(AUDIT_PLAN_DOC), str(plan_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert audit.returncode == 0


def test_contract_rule_preserves_legacy_disposition_behavior() -> None:
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    reviewer_rules = (REPO_ROOT / "docs" / "REVIEWER_RULES.md").read_text(
        encoding="utf-8"
    )

    assert "Do not retroactively\nre-disposition a legacy criterion" in agents
    assert "review against the contract as authored" in reviewer_rules
    assert (
        "Do not mark a legacy\n  criterion `could-not-determine` solely because"
        in reviewer_rules
    )


def test_contract_rule_qualifies_risk_category_prohibitions(tmp_path: Path) -> None:
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    reviewer_rules = (REPO_ROOT / "docs" / "REVIEWER_RULES.md").read_text(
        encoding="utf-8"
    )
    _init_git_repo(tmp_path)
    result = _run_new_plan(
        tmp_path,
        "Qualifier-Regression",
        "--lane",
        "dev-workflow/test",
    )
    assert result.returncode == 0
    scaffold = (tmp_path / "plans" / "PR-Qualifier-Regression.md").read_text(
        encoding="utf-8"
    )

    for source in (agents, reviewer_rules, scaffold):
        assert "never a risk category" not in source
        assert "forbids a risk category" not in source
        assert "criterion sits at could-not-determine" not in source
        assert "test_concurrent_claim" not in source

    assert "never a bare risk category" in agents
    assert "BARE risk category with no referent" in reviewer_rules
    assert "Do NOT name a BARE risk category" in scaffold
    assert "3k.3 evidence-gated mechanism" in _compact_ws(agents)
    assert "3k.3 evidence-gated mechanism" in _compact_ws(reviewer_rules)
    assert "3k.3 evidence-gated mechanism" in _compact_ws(scaffold)
    assert "3k.4 execution model" in _compact_ws(agents)
    assert "3k.4 execution model" in _compact_ws(reviewer_rules)
    assert "3k.4 execution model and property-level invariant" in _compact_ws(scaffold)


def test_new_pr_plan_does_not_double_prefix_existing_pr_name(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)

    result = _run_new_plan(tmp_path, "PR-Already-Prefixed", "--lane", "dev-workflow/test")

    assert result.returncode == 0
    assert (tmp_path / "plans" / "PR-Already-Prefixed.md").exists()
    assert not (tmp_path / "plans" / "PR-PR-Already-Prefixed.md").exists()


def test_new_pr_plan_refuses_existing_plan_without_force(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    first = _run_new_plan(
        tmp_path, "Overwrite-Check", "--lane", "dev-workflow/test", "--phase", "Workflow/process"
    )
    assert first.returncode == 0
    plan_path = tmp_path / "plans" / "PR-Overwrite-Check.md"
    original = plan_path.read_text(encoding="utf-8")

    second = _run_new_plan(
        tmp_path, "Overwrite-Check", "--lane", "dev-workflow/test", "--phase", "Vertical slice"
    )

    assert second.returncode == 2
    assert "plan already exists" in second.stderr
    assert plan_path.read_text(encoding="utf-8") == original


def test_new_pr_plan_force_overwrites_existing_plan(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    first = _run_new_plan(
        tmp_path, "Force-Check", "--lane", "dev-workflow/test", "--phase", "Workflow/process"
    )
    assert first.returncode == 0

    second = _run_new_plan(
        tmp_path,
        "Force-Check",
        "--lane",
        "dev-workflow/test",
        "--phase",
        "Vertical slice",
        "--force",
    )

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


def test_new_pr_plan_requires_a_session_state_file_before_writing(tmp_path: Path) -> None:
    _init_git_repo(tmp_path, with_state=False)

    result = _run_new_plan(tmp_path, "Missing-State", "--lane", "dev-workflow/test")

    assert result.returncode == 2
    assert "session state file not found" in result.stderr
    assert not (tmp_path / "plans" / "PR-Missing-State.md").exists()


def test_new_pr_plan_requires_exactly_one_nonempty_current_lane(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    state = _write_state(tmp_path, "")

    empty = _run_new_plan(tmp_path, "Empty-Lane", "--lane", "dev-workflow/test")

    assert empty.returncode == 2
    assert "Current lane: must be non-empty" in empty.stderr
    assert not (tmp_path / "plans" / "PR-Empty-Lane.md").exists()

    state.write_text(
        "Current lane: dev-workflow/test\nCurrent lane: another-lane\n",
        encoding="utf-8",
    )
    duplicate = _run_new_plan(tmp_path, "Duplicate-Lane", "--lane", "dev-workflow/test")

    assert duplicate.returncode == 2
    assert "exactly one top-level Current lane:" in duplicate.stderr
    assert not (tmp_path / "plans" / "PR-Duplicate-Lane.md").exists()


def test_new_pr_plan_rejects_missing_or_placeholder_current_lane(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    state = tmp_path / "SESSION_STATE.local.md"
    state.write_text("Operator-assigned lane: dev-workflow/test\n", encoding="utf-8")

    missing = _run_new_plan(tmp_path, "Missing-Lane", "--lane", "dev-workflow/test")

    assert missing.returncode == 2
    assert "exactly one top-level Current lane:" in missing.stderr
    assert not (tmp_path / "plans" / "PR-Missing-Lane.md").exists()

    state.write_text("Current lane: <one sentence>\n", encoding="utf-8")
    placeholder = _run_new_plan(tmp_path, "Placeholder-Lane", "--lane", "<one sentence>")

    assert placeholder.returncode == 2
    assert "must name an assigned lane" in placeholder.stderr
    assert not (tmp_path / "plans" / "PR-Placeholder-Lane.md").exists()


def test_new_pr_plan_rejects_omitted_or_mismatched_lane_before_force_write(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    plan = tmp_path / "plans" / "PR-Lane-Check.md"
    plan.parent.mkdir()
    plan.write_text("original\n", encoding="utf-8")

    missing = _run_new_plan(tmp_path, "Lane-Check")
    mismatch = _run_new_plan(
        tmp_path, "Lane-Check", "--lane", "dev-workflow/other", "--force"
    )

    assert missing.returncode == 2
    assert "--lane is required" in missing.stderr
    assert mismatch.returncode == 2
    assert "lane mismatch" in mismatch.stderr
    assert plan.read_text(encoding="utf-8") == "original\n"


def test_new_pr_plan_state_file_flag_overrides_environment(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    explicit_state = tmp_path / "explicit-state.md"
    explicit_state.write_text("Current lane: dev-workflow/explicit\n", encoding="utf-8")

    result = _run_new_plan(
        tmp_path,
        "Explicit-State",
        "--lane",
        "dev-workflow/explicit",
        "--state-file",
        str(explicit_state),
        env={"ATLAS_SESSION_STATE_FILE": str(tmp_path / "missing-state.md")},
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "plans" / "PR-Explicit-State.md").exists()


def test_new_pr_plan_uses_state_file_from_environment(tmp_path: Path) -> None:
    _init_git_repo(tmp_path, with_state=False)
    state = _write_state(tmp_path, "dev-workflow/environment")
    state.rename(tmp_path / "session-state.md")
    state = tmp_path / "session-state.md"

    result = _run_new_plan(
        tmp_path,
        "Environment-State",
        "--lane",
        "dev-workflow/environment",
        env={"ATLAS_SESSION_STATE_FILE": str(state)},
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "plans" / "PR-Environment-State.md").exists()


def test_new_pr_plan_ignores_ambient_session_state_file(tmp_path: Path, monkeypatch) -> None:
    _init_git_repo(tmp_path)
    monkeypatch.setenv("ATLAS_SESSION_STATE_FILE", str(tmp_path / "wrong-state.md"))

    result = _run_new_plan(tmp_path, "Ambient-State", "--lane", "dev-workflow/test")

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "plans" / "PR-Ambient-State.md").exists()


def test_new_pr_plan_resolves_relative_state_paths_from_repo_root(tmp_path: Path) -> None:
    _init_git_repo(tmp_path, with_state=False)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state = _write_state(state_dir, "dev-workflow/relative")
    nested = tmp_path / "nested"
    nested.mkdir()

    flag = _run_new_plan(
        tmp_path,
        "Relative-Flag",
        "--lane",
        "dev-workflow/relative",
        "--state-file",
        "state/SESSION_STATE.local.md",
        cwd=nested,
    )
    environment = _run_new_plan(
        tmp_path,
        "Relative-Environment",
        "--lane",
        "dev-workflow/relative",
        cwd=nested,
        env={"ATLAS_SESSION_STATE_FILE": "state/SESSION_STATE.local.md"},
    )

    assert state.exists()
    assert flag.returncode == 0, flag.stderr
    assert environment.returncode == 0, environment.stderr
    assert (tmp_path / "plans" / "PR-Relative-Flag.md").exists()
    assert (tmp_path / "plans" / "PR-Relative-Environment.md").exists()
