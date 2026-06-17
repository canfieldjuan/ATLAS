from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "pre_push_audit.yml"


def test_pre_push_audit_workflow_passes_pr_body_to_local_review() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Write current PR body" in text
    assert "github.event_name == 'pull_request'" in text
    assert 'Path(os.environ["RUNNER_TEMP"], "current-pr-body.md")' in text
    assert 'EVENT_NAME: ${{ github.event_name }}' in text
    assert 'bash scripts/local_pr_review.sh --current-pr-body-file "$RUNNER_TEMP/current-pr-body.md"' in text


def test_pre_push_audit_workflow_keeps_push_to_main_without_pr_body() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert 'if [ "$EVENT_NAME" = "pull_request" ]; then' in text
    assert "else\n            bash scripts/local_pr_review.sh\n          fi" in text


def test_pre_push_audit_workflow_enrolls_push_pr_wrapper_tests() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "tests/test_push_pr_wrapper.py" in text


def test_pre_push_audit_workflow_enrolls_full_report_redaction_checker_tests() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "tests/test_check_deflection_full_report_proof_bundle.py" in text


def test_pre_push_audit_workflow_enrolls_gitleaks_baseline_rotation_tests() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "tests/test_check_gitleaks_baseline_rotation.py" in text


def test_pre_push_audit_workflow_enrolls_security_guardrails_workflow_tests() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "tests/test_security_guardrails_workflow.py" in text
