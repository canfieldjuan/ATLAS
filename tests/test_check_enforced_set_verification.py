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

- bash scripts/run_eom_lead_pipeline_checks.sh - 189 passed.
- CI-equivalent command copied from enforcing workflow: `bash scripts/run_eom_lead_pipeline_checks.sh`.
- Copied from enforcing workflow: `.github/workflows/atlas_eom_lead_pipeline_checks.yml`.
"""
NO_WORKFLOW_PLAN = """
## Verification

- python scripts/audit_plan_doc.py plans/PR-Docs.md - OK.
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


def test_skipped_execution_attestation_is_flagged() -> None:
    text = """
## Verification

- CI-equivalent command copied from enforcing workflow: `bash scripts/run_eom_lead_pipeline_checks.sh` (skipped locally).
- Copied from enforcing workflow: `.github/workflows/atlas_eom_lead_pipeline_checks.yml`.
"""
    assert len(mod.scan_plans({"plans/PR-Thing.md": text})) == 1


def test_command_markers_without_affirmative_result_are_flagged() -> None:
    text = """
## Verification

- CI-equivalent command copied from enforcing workflow: `bash scripts/run_eom_lead_pipeline_checks.sh`.
- Copied from enforcing workflow: `.github/workflows/atlas_eom_lead_pipeline_checks.yml`.
"""
    assert len(mod.scan_plans({"plans/PR-Thing.md": text})) == 1


def test_unrelated_pass_does_not_satisfy_required_command() -> None:
    text = """
## Verification

- python -m pytest tests/test_one.py - passed.
- CI-equivalent command copied from enforcing workflow: `bash scripts/run_eom_lead_pipeline_checks.sh`.
- Copied from enforcing workflow: `.github/workflows/atlas_eom_lead_pipeline_checks.yml`.
"""
    assert len(mod.scan_plans({"plans/PR-Thing.md": text})) == 1


def test_prefix_subset_command_does_not_satisfy_required_command() -> None:
    text = """
## Verification

- python -m pytest tests/test_one.py::test_only - 1 passed.
- CI-equivalent command copied from enforcing workflow: `python -m pytest tests/test_one.py`.
- Copied from enforcing workflow: `.github/workflows/example.yml`.
"""
    assert len(mod.scan_plans({"plans/PR-Thing.md": text})) == 1


def test_unrelated_skipped_optional_check_does_not_poison_command_result() -> None:
    text = COMPLIANT_PLAN + "\n- optional browser check - skipped."
    assert mod.scan_plans({"plans/PR-Thing.md": text}) == []


def test_no_workflow_fallback_rejects_workflow_enrolled_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "path_has_enforcing_workflow", lambda path: path == "atlas_brain/api/leads.py")
    findings = mod.scan_plans(
        {"plans/PR-Thing.md": NO_WORKFLOW_PLAN},
        changed_paths=["atlas_brain/api/leads.py"],
    )
    assert len(findings) == 1


def test_verification_section_stops_at_next_top_level_heading() -> None:
    text = COMPLIANT_PLAN + "\n## Estimated diff size\n- copied from enforcing workflow: no"
    assert mod.plan_has_enforced_set_verification(text)


def test_cli_entrypoint_warns_advisory_and_fails_strict(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        mod,
        "changed_plan_texts",
        lambda base: {"plans/PR-Thing.md": "## Verification\n- pytest tests/test_one.py"},
    )
    monkeypatch.setattr(mod, "changed_paths", lambda base: ["plans/PR-Thing.md"])

    assert mod.main(["--base", "ignored"]) == 0
    out = capsys.readouterr().out
    assert "::warning file=plans/PR-Thing.md::" in out
    assert mod.RULE in out

    assert mod.main(["--base", "ignored", "--strict"]) == 1


def test_git_failure_raises_system_exit() -> None:
    with pytest.raises(SystemExit, match="git .* failed"):
        mod._git(["rev-parse", "--verify", "definitely-not-a-ref-xyz"])


# ---------------------------------------------------------------------------
# Single-source manifest: this rule applied to its own mechanism.
#
# The mirror script used to carry its own copy of the workflow's pytest list,
# so "run what CI runs" was a claim maintained by hand -- the exact drift this
# rule exists to prevent, one level down. Both consumers now read
# tests/eom_lead_pipeline_files.txt and neither keeps a copy.
# ---------------------------------------------------------------------------

import subprocess  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = _REPO_ROOT / "tests" / "eom_lead_pipeline_files.txt"
MIRROR_SCRIPT = _REPO_ROOT / "scripts" / "run_eom_lead_pipeline_checks.sh"
GATE_WORKFLOW = (
    _REPO_ROOT / ".github" / "workflows" / "atlas_eom_lead_pipeline_checks.yml"
)


def _manifest_entries() -> list[str]:
    return [
        line.strip()
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_manifest_lists_real_test_files() -> None:
    entries = _manifest_entries()
    assert entries, "manifest must not be empty"
    missing = [path for path in entries if not (_REPO_ROOT / path).is_file()]
    assert missing == [], f"manifest names files that do not exist: {missing}"


def test_manifest_is_shell_safe() -> None:
    """Both consumers pipe the manifest through xargs, so an entry containing
    whitespace or a quote would split into bogus pytest arguments."""
    for entry in _manifest_entries():
        assert not any(ch in entry for ch in " \t'\"\\"), f"unsafe entry: {entry!r}"


def test_both_consumers_read_the_manifest() -> None:
    """Drift is structurally impossible only while both consumers read the
    manifest instead of listing files themselves."""
    for consumer in (MIRROR_SCRIPT, GATE_WORKFLOW):
        text = consumer.read_text(encoding="utf-8")
        assert "eom_lead_pipeline_files.txt" in text, (
            f"{consumer.name} does not read the manifest"
        )


def test_mirror_script_does_not_relist_the_test_set() -> None:
    """A second copy of the set inside the mirror is the original defect."""
    text = MIRROR_SCRIPT.read_text(encoding="utf-8")
    relisted = [
        line for line in text.splitlines()
        if "tests/test_" in line and "eom_lead_pipeline_files" not in line
    ]
    assert relisted == [], f"mirror re-lists test files: {relisted}"


def test_mirror_script_parses() -> None:
    subprocess.run(["bash", "-n", str(MIRROR_SCRIPT)], check=True)
