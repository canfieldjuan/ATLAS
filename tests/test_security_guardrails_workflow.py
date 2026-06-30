from __future__ import annotations

import importlib.util
from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "security_guardrails.yml"
BASELINE_GUARD_WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "gitleaks_baseline_growth_guard.yml"
)
BRANCH_PROTECTION_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "branch_protection_required_checks.yml"
PRE_COMMIT_CONFIG = Path(__file__).resolve().parents[1] / ".pre-commit-config.yaml"
SECURITY_GUARDRAILS_DOC = Path(__file__).resolve().parents[1] / "docs" / "SECURITY_GUARDRAILS.md"
REQUIRED_STATUS_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_required_status_checks.py"


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _baseline_guard_workflow_text() -> str:
    return BASELINE_GUARD_WORKFLOW.read_text(encoding="utf-8")


def _load_required_status_script():
    spec = importlib.util.spec_from_file_location("check_required_status_checks", REQUIRED_STATUS_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_baseline_guard_runs_on_pull_request_target_label_changes() -> None:
    text = _baseline_guard_workflow_text()

    assert "pull_request_target:" in text
    assert "types: [opened, synchronize, reopened, ready_for_review, labeled, unlabeled]" in text
    assert "pull_request:" not in text


def test_baseline_guard_checks_out_trusted_base_and_fetches_pr_as_data() -> None:
    text = _baseline_guard_workflow_text()

    assert "Checkout trusted base" in text
    assert "ref: ${{ github.event.pull_request.base.sha }}" in text
    assert '"pull/${PR_NUMBER}/head:refs/remotes/origin/pr-${PR_NUMBER}"' in text
    assert "--head-ref \"refs/remotes/origin/pr-${PR_NUMBER}\"" in text


def test_baseline_guard_uses_json_labels() -> None:
    text = _baseline_guard_workflow_text()

    assert "PR_LABELS_JSON: ${{ toJson(github.event.pull_request.labels.*.name) }}" in text
    assert "--labels-json \"${PR_LABELS_JSON}\"" in text


def test_security_guardrails_workflow_does_not_emit_skipped_baseline_context() -> None:
    text = _workflow_text()

    assert "Gitleaks baseline growth guard" not in text


def test_heavy_scans_do_not_run_on_pull_request_target() -> None:
    text = _workflow_text()

    assert "if: github.event_name != 'pull_request' && github.event_name != 'pull_request_target'" in text


def test_gitleaks_pre_commit_scans_staged_changes_without_echoing_secrets() -> None:
    text = PRE_COMMIT_CONFIG.read_text(encoding="utf-8")

    assert 'minimum_pre_commit_version: "3.2.0"' in text
    assert "repo: local" in text
    assert "id: gitleaks-protect" in text
    assert "entry: gitleaks protect --staged --redact --verbose" in text
    assert "language: system" in text
    assert "pass_filenames: false" in text
    assert "stages: [pre-commit]" in text


def test_security_guardrails_docs_explain_gitleaks_pre_commit_install() -> None:
    text = SECURITY_GUARDRAILS_DOC.read_text(encoding="utf-8")

    assert ".pre-commit-config.yaml" in text
    assert "`pre-commit` 3.2 or newer" in text
    assert "pre-commit install" in text
    assert "gitleaks protect --staged --redact --verbose" in text
    assert "does not rotate historical credentials" in text


def test_security_guardrails_docs_name_required_gitleaks_checks() -> None:
    text = SECURITY_GUARDRAILS_DOC.read_text(encoding="utf-8")

    assert "`Gitleaks PR secret scan`" in text
    assert "`Gitleaks baseline growth guard`" in text
    assert "`Branch Protection Required Checks` workflow" in text


def test_branch_protection_workflow_audits_live_required_checks() -> None:
    text = BRANCH_PROTECTION_WORKFLOW.read_text(encoding="utf-8")

    assert "branches/main/protection/required_status_checks" in text
    assert "ATLAS_BRANCH_PROTECTION_READ_TOKEN" in text
    assert "BRANCH_PROTECTION_READ_TOKEN != ''" in text
    assert ".github/workflows/gitleaks_baseline_growth_guard.yml" in text
    assert ".github/workflows/security_guardrails.yml" in text
    assert (
        "if: github.event_name != 'workflow_dispatch' || github.ref == 'refs/heads/main'"
        in text
    )
    assert "ref: ${{ github.event.repository.default_branch }}" in text
    assert "scripts/check_required_status_checks.py" in text
    assert "workflow_dispatch:" in text
    assert "schedule:" in text


def test_branch_protection_workflow_ref_guard_precedes_admin_read_token() -> None:
    text = BRANCH_PROTECTION_WORKFLOW.read_text(encoding="utf-8")
    token_index = text.index(
        "BRANCH_PROTECTION_READ_TOKEN: ${{ secrets.ATLAS_BRANCH_PROTECTION_READ_TOKEN }}"
    )

    assert text.index("if: github.event_name != 'workflow_dispatch'") < token_index
    assert text.index("ref: ${{ github.event.repository.default_branch }}") < text.index(
        "GH_TOKEN: ${{ env.BRANCH_PROTECTION_READ_TOKEN }}"
    )


def test_required_status_check_audit_accepts_contexts_and_checks_shapes() -> None:
    checker = _load_required_status_script()
    payload = {
        "contexts": ["live-reconciliation"],
        "checks": [
            {"context": "Gitleaks PR secret scan"},
            {"context": "Gitleaks baseline growth guard"},
        ],
    }

    assert checker.missing_required_contexts(payload) == []


def test_required_status_check_audit_accepts_github_actions_source() -> None:
    checker = _load_required_status_script()
    payload = {
        "contexts": [
            "live-reconciliation",
            "Gitleaks PR secret scan",
            "Gitleaks baseline growth guard",
        ],
        "checks": [
            {"context": "live-reconciliation", "app_id": checker.GITHUB_ACTIONS_APP_ID},
            {"context": "Gitleaks PR secret scan", "app_id": checker.GITHUB_ACTIONS_APP_ID},
            {"context": "Gitleaks baseline growth guard", "app_id": checker.GITHUB_ACTIONS_APP_ID},
        ],
    }

    assert checker.required_status_check_failures(payload) == []


def test_required_status_check_audit_rejects_legacy_only_contexts() -> None:
    checker = _load_required_status_script()
    payload = {
        "contexts": [
            "live-reconciliation",
            "Gitleaks PR secret scan",
            "Gitleaks baseline growth guard",
        ],
    }

    failures = checker.required_status_check_failures(payload)

    assert [failure.context for failure in failures] == [
        "live-reconciliation",
        "Gitleaks PR secret scan",
        "Gitleaks baseline growth guard",
    ]
    assert all("expected app_id" in failure.reason for failure in failures)


def test_required_status_check_audit_rejects_wrong_check_source() -> None:
    checker = _load_required_status_script()
    payload = {
        "checks": [
            {"context": "live-reconciliation", "app_id": checker.GITHUB_ACTIONS_APP_ID},
            {"context": "Gitleaks PR secret scan", "app_id": -1},
            {"context": "Gitleaks baseline growth guard", "app_id": None},
        ],
    }

    failures = checker.required_status_check_failures(payload)

    assert [failure.context for failure in failures] == [
        "Gitleaks PR secret scan",
        "Gitleaks baseline growth guard",
    ]
    assert "found app_id -1" in failures[0].reason
    assert "found legacy/unpinned" in failures[1].reason


def test_required_status_check_audit_fails_when_gitleaks_context_missing() -> None:
    checker = _load_required_status_script()
    payload = {
        "required_status_checks": {
            "checks": [{"context": "live-reconciliation"}],
        },
    }

    assert checker.missing_required_contexts(payload) == [
        "Gitleaks PR secret scan",
        "Gitleaks baseline growth guard",
    ]
