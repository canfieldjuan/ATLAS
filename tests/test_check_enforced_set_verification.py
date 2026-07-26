from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_enforced_set_verification",
    Path(__file__).resolve().parent.parent / "scripts" / "check_enforced_set_verification.py",
)
mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)

COMPLIANT_PLAN = """
## Verification

- CI-equivalent command copied from enforcing workflow: `bash scripts/run_eom_lead_pipeline_checks.sh`.
- Copied from enforcing workflow: `.github/workflows/atlas_eom_lead_pipeline_checks.yml`.
"""
NO_WORKFLOW_PLAN = """
## Verification

- No enforcing workflow applies: documentation-only surface has no workflow owner.
- Closest local command: `python scripts/audit_plan_doc.py plans/PR-Docs.md`.
"""
SCAFFOLD_PLACEHOLDER = """
## Verification

- Pending before push: TODO.
- CI-equivalent command copied from enforcing workflow: TODO/N/A.
- Copied from enforcing workflow: TODO/N/A.
- No enforcing workflow applies: TODO/N/A.
- Closest local command: TODO/N/A.
"""


def test_plan_without_ci_equivalent_verification_is_flagged() -> None:
    findings = mod.scan_plans({"plans/PR-Thing.md": "## Verification\n- pytest tests/test_one.py"})
    assert len(findings) == 1
    assert findings[0].path == "plans/PR-Thing.md"
    assert mod.RULE in findings[0].reason


def test_plan_with_ci_equivalent_verification_is_clean() -> None:
    assert mod.scan_plans({"plans/PR-Thing.md": COMPLIANT_PLAN}) == []


def test_plan_with_no_enforcing_workflow_path_is_clean() -> None:
    assert mod.scan_plans({"plans/PR-Docs.md": NO_WORKFLOW_PLAN}) == []


def test_markers_must_be_in_verification_section() -> None:
    misplaced = COMPLIANT_PLAN.replace("## Verification", "## Mechanism") + "\n## Verification\n- pytest"
    assert not mod.plan_has_enforced_set_verification(misplaced)


def test_scaffold_placeholders_do_not_satisfy_verification() -> None:
    assert not mod.plan_has_enforced_set_verification(SCAFFOLD_PLACEHOLDER)


def test_missing_workflow_source_line_is_flagged() -> None:
    missing_source = """
## Verification

- CI-equivalent command copied from enforcing workflow: `pytest tests/test_one.py`.
"""
    assert len(mod.scan_plans({"plans/PR-Thing.md": missing_source})) == 1


def test_negative_execution_attestation_is_flagged() -> None:
    text = """
## Verification

- CI-equivalent command copied from enforcing workflow: `bash scripts/run_eom_lead_pipeline_checks.sh` (not run locally).
- Copied from enforcing workflow: `.github/workflows/atlas_eom_lead_pipeline_checks.yml`.
"""
    assert len(mod.scan_plans({"plans/PR-Thing.md": text})) == 1


def test_verification_section_stops_at_next_top_level_heading() -> None:
    text = COMPLIANT_PLAN + "\n## Estimated diff size\n- copied from enforcing workflow: no"
    assert mod.plan_has_enforced_set_verification(text)


def test_cli_entrypoint_warns_advisory_and_fails_strict(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        mod,
        "changed_plan_texts",
        lambda base: {"plans/PR-Thing.md": "## Verification\n- pytest tests/test_one.py"},
    )

    assert mod.main(["--base", "ignored"]) == 0
    out = capsys.readouterr().out
    assert "::warning file=plans/PR-Thing.md::" in out
    assert mod.RULE in out

    assert mod.main(["--base", "ignored", "--strict"]) == 1


def test_git_failure_raises_system_exit() -> None:
    with pytest.raises(SystemExit, match="git .* failed"):
        mod._git(["rev-parse", "--verify", "definitely-not-a-ref-xyz"])
