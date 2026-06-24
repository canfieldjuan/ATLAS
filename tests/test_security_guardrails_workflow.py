from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "security_guardrails.yml"
PRE_COMMIT_CONFIG = Path(__file__).resolve().parents[1] / ".pre-commit-config.yaml"
SECURITY_GUARDRAILS_DOC = Path(__file__).resolve().parents[1] / "docs" / "SECURITY_GUARDRAILS.md"


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_baseline_guard_runs_on_pull_request_target_label_changes() -> None:
    text = _workflow_text()

    assert "pull_request_target:" in text
    assert "types: [opened, synchronize, reopened, ready_for_review, labeled, unlabeled]" in text
    assert "if: github.event_name == 'pull_request_target'" in text


def test_baseline_guard_checks_out_trusted_base_and_fetches_pr_as_data() -> None:
    text = _workflow_text()

    assert "Checkout trusted base" in text
    assert "ref: ${{ github.event.pull_request.base.sha }}" in text
    assert '"pull/${PR_NUMBER}/head:refs/remotes/origin/pr-${PR_NUMBER}"' in text
    assert "--head-ref \"refs/remotes/origin/pr-${PR_NUMBER}\"" in text


def test_baseline_guard_uses_json_labels() -> None:
    text = _workflow_text()

    assert "PR_LABELS_JSON: ${{ toJson(github.event.pull_request.labels.*.name) }}" in text
    assert "--labels-json \"${PR_LABELS_JSON}\"" in text


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
