from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "pre_push_audit.yml"


def test_pre_push_audit_workflow_pr_event_runs_trusted_base_split() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request_target:" in text
    assert "if: github.event_name == 'pull_request_target'" in text
    assert "Checkout trusted base" in text
    assert "actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0" in text
    assert "ref: ${{ github.event.pull_request.base.sha }}" in text
    assert 'git worktree add "$RUNNER_TEMP/pr-tree" "refs/remotes/origin/pr-${PR_NUMBER}"' in text
    assert '--repo-root "$RUNNER_TEMP/pr-tree"' in text
    assert '--script-root "$GITHUB_WORKSPACE"' in text
    assert "bash scripts/local_pr_review.sh \\" in text
    assert 'python scripts/audit_workflow_security_posture.py "$RUNNER_TEMP/pr-tree/.github/workflows"' in text


def test_pre_push_audit_workflow_passes_pr_body_to_trusted_local_review() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Write current PR body" in text
    assert 'Path(os.environ["RUNNER_TEMP"], "current-pr-body.md")' in text
    assert '--current-pr-body-file "$RUNNER_TEMP/current-pr-body.md"' in text
    assert "PR_AUTHOR: ${{ github.event.pull_request.user.login }}" in text
    assert '--pr-author "$PR_AUTHOR"' in text


def test_pre_push_audit_workflow_keeps_push_to_main_without_pr_body() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "pre-push-audit-main:" in text
    assert "if: github.event_name == 'push'" in text
    assert "run: bash scripts/local_pr_review.sh" in text


def test_pre_push_audit_workflow_enrolls_push_pr_wrapper_tests() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "tests/test_push_pr_wrapper.py" in text


def test_pre_push_audit_workflow_enrolls_open_pr_wrapper_tests() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "tests/test_open_pr_wrapper.py" in text


def test_pre_push_audit_workflow_enrolls_plan_doc_audit_tests() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "tests/test_audit_plan_doc.py" in text


def test_pre_push_audit_workflow_enrolls_plan_admission_and_body_gate_tests() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert text.count("tests/test_audit_pr_plan_presence.py") == 2
    assert text.count("tests/test_plan_admission_workflow.py") == 2
    assert text.count("tests/test_pr_body_contract_workflow.py") == 2


def test_pre_push_audit_workflow_enrolls_session_drift_tests() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert text.count("tests/test_audit_pr_session_drift.py") == 2


def test_pre_push_audit_workflow_enrolls_session_lane_workflow_tests() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert text.count("tests/test_session_lane_workflow.py") == 2


def test_pre_push_audit_workflow_enrolls_full_report_redaction_checker_tests() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "tests/test_check_deflection_full_report_proof_bundle.py" in text


def test_pre_push_audit_workflow_enrolls_gitleaks_baseline_rotation_tests() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "tests/test_check_gitleaks_baseline_rotation.py" in text


def test_pre_push_audit_workflow_enrolls_security_guardrails_workflow_tests() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "tests/test_security_guardrails_workflow.py" in text


def test_pre_push_audit_workflow_enrolls_workflow_security_posture_audit() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "tests/test_audit_workflow_security_posture.py" in text
    assert "pytest pytest-asyncio pyyaml" in text
    assert "python scripts/audit_workflow_security_posture.py .github/workflows" in text


def test_pre_push_audit_workflow_enrolls_claude_workflow_security_tests() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "tests/test_claude_workflow_security.py" in text


def test_pre_push_audit_workflow_enrolls_deflection_report_ttl_workflow_tests() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "tests/test_deflection_report_ttl_workflow.py" in text
