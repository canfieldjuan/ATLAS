from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "extracted_pipeline_checks.yml"
CHECKS = ROOT / "scripts" / "run_extracted_pipeline_checks.sh"


def test_extracted_checks_uploads_deflection_pii_recall_advisory_artifact() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "Write deflection PII recall advisory artifact" in workflow
    assert "python scripts/score_deflection_pii_recall.py" in workflow
    assert (
        "--output artifacts/deflection-pii-recall/deflection-pii-recall-advisory.json"
        in workflow
    )
    assert "Upload deflection PII recall advisory artifact" in workflow
    assert (
        "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"
        in workflow
    )
    assert "name: deflection-pii-recall-advisory" in workflow
    assert (
        "path: artifacts/deflection-pii-recall/deflection-pii-recall-advisory.json"
        in workflow
    )
    assert "if-no-files-found: error" in workflow


def test_deflection_pii_recall_artifact_stays_advisory_not_gating() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "--require-free-high-severity-pass" not in workflow
    assert "free_high_severity_pass" not in workflow
    assert "free_high_severity_leak_count" not in workflow


def test_advisory_artifact_contract_test_is_ci_enrolled() -> None:
    checks = CHECKS.read_text(encoding="utf-8")

    assert "tests/test_extracted_pipeline_pii_recall_advisory_artifact.py" in checks
