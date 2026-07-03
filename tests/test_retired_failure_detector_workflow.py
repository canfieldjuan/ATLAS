from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "retired_failure_detector.yml"
MATURITY_WORKFLOW = ROOT / ".github" / "workflows" / "maturity_sweep_advisory.yml"
UPLOAD_ARTIFACT_SHA = "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"


def workflow_text(path: Path = WORKFLOW) -> str:
    return path.read_text(encoding="utf-8")


def test_workflow_runs_on_pull_request_without_target_token_posture() -> None:
    workflow = workflow_text()

    assert "\n  pull_request:\n" in workflow
    assert "pull_request_target:" not in workflow
    assert "\npermissions:\n  contents: read\n" in workflow


def test_workflow_fetches_base_ref_and_writes_detector_json_artifact() -> None:
    workflow = workflow_text()

    assert "- name: Resolve base ref" in workflow
    assert "refs/remotes/origin/${base_ref}" in workflow
    assert "- name: Run retired failure detector" in workflow
    assert "scripts/detect_retired_failure_modes.py" in workflow
    assert '--base "origin/${base_ref}"' in workflow
    assert "--json-out artifacts/retired-failure-detector/retired-failure-signals.json" in workflow


def test_workflow_uploads_pinned_detector_artifact_without_signal_failure_step() -> None:
    workflow = workflow_text()

    assert "- name: Upload retired failure detector artifact" in workflow
    assert f"uses: actions/upload-artifact@{UPLOAD_ARTIFACT_SHA}" in workflow
    assert "name: retired-failure-detector-signals-${{ github.run_id }}" in workflow
    assert "path: artifacts/retired-failure-detector" in workflow
    assert "if-no-files-found: error" in workflow
    assert "retention-days: 14" in workflow
    assert "jq" not in workflow
    assert "signals | length" not in workflow


def test_workflow_contract_test_is_enrolled_in_maturity_sweep_pr_ci() -> None:
    workflow = MATURITY_WORKFLOW.read_text(encoding="utf-8")

    assert "tests/test_retired_failure_detector_workflow.py" in workflow
