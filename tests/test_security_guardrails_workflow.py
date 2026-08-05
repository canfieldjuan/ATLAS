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
GATE_REGISTRY = Path(__file__).resolve().parents[1] / "ci" / "gates.yml"

REQUIRED_STATUS_CONTEXTS = (
    "live-reconciliation",
    "diff-budget",
    "plan-admission",
    "session-lane",
    "review-contract",
    "pr-body-contract",
    "Gitleaks PR secret scan",
    "Gitleaks baseline growth guard",
)

REQUIRED_STATUS_WORKFLOW_PATHS = (
    ".github/workflows/ai_reconciliation_live.yml",
    ".github/workflows/diff_budget.yml",
    ".github/workflows/plan_admission.yml",
    ".github/workflows/pr_body_contract.yml",
    ".github/workflows/review_contract.yml",
    ".github/workflows/session_lane.yml",
    ".github/workflows/gitleaks_baseline_growth_guard.yml",
    ".github/workflows/security_guardrails.yml",
    "ci/gates.yml",
)


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
    normalized_text = " ".join(text.split())

    for context in REQUIRED_STATUS_CONTEXTS:
        assert f"`{context}`" in text
    assert "`Branch Protection Required Checks` workflow" in text
    # 2026-08-04 status alignment: live settings now carry every
    # registry-required context, so the doc records that completed state
    # (plus the verification command) instead of the old partial-set /
    # separate-REST-PATCH interim wording.
    assert (
        "live GitHub settings contain every registry-required context pinned "
        "to the GitHub Actions app source" in normalized_text
    )
    assert "scripts/check_required_status_checks.py" in text


def test_branch_protection_workflow_audits_live_required_checks() -> None:
    text = BRANCH_PROTECTION_WORKFLOW.read_text(encoding="utf-8")

    assert "branches/main/protection/required_status_checks" in text
    assert "ATLAS_BRANCH_PROTECTION_READ_TOKEN" in text
    assert "BRANCH_PROTECTION_READ_TOKEN != ''" in text
    for workflow_path in REQUIRED_STATUS_WORKFLOW_PATHS:
        assert workflow_path in text
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
        "contexts": [
            "live-reconciliation",
            "diff-budget",
            "plan-admission",
            "session-lane",
            "pr-body-contract",
        ],
        "checks": [
            {"context": "review-contract"},
            {"context": "Gitleaks PR secret scan"},
            {"context": "Gitleaks baseline growth guard"},
        ],
    }

    assert checker.missing_required_contexts(payload) == []


def test_required_status_check_defaults_are_registry_derived() -> None:
    checker = _load_required_status_script()

    assert checker.default_required_contexts(GATE_REGISTRY) == REQUIRED_STATUS_CONTEXTS
    assert "DEFAULT_REQUIRED_CONTEXTS = (" not in REQUIRED_STATUS_SCRIPT.read_text(
        encoding="utf-8"
    )


def test_gate_registry_excludes_advisory_from_required_contexts() -> None:
    checker = _load_required_status_script()
    gates = checker.load_gate_registry(GATE_REGISTRY)
    advisory_contexts = {
        gate["context"]
        for gate in gates
        if gate["enforcement"] == "advisory" and gate["context"]
    }

    assert "seam-convergence" in advisory_contexts
    assert "guard-class-closure-lint" in advisory_contexts
    assert advisory_contexts.isdisjoint(checker.default_required_contexts(GATE_REGISTRY))


def test_gate_registry_workflow_paths_exist() -> None:
    checker = _load_required_status_script()
    root = GATE_REGISTRY.parents[1]

    for gate in checker.load_gate_registry(GATE_REGISTRY):
        workflow = gate["workflow"]
        assert isinstance(workflow, str)
        assert (root / workflow).is_file(), gate["id"]


def test_gate_registry_fails_closed_for_malformed_branch_required_gate() -> None:
    checker = _load_required_status_script()
    malformed = """\
gates:
  - id: missing-context
    name: Missing Context
    context: null
    enforcement: branch_required
    trusted_base: true
    workflow: .github/workflows/missing.yml
    local_command: null
"""

    try:
        checker.parse_gate_registry(malformed)
    except ValueError as exc:
        assert "branch_required needs context" in str(exc)
    else:
        raise AssertionError("malformed branch_required gate passed")


def test_gate_registry_fails_closed_for_unknown_enforcement_class() -> None:
    checker = _load_required_status_script()
    malformed = """\
gates:
  - id: unknown-class
    name: Unknown Class
    context: unknown-class
    enforcement: maybe_required
    trusted_base: true
    workflow: .github/workflows/unknown.yml
    local_command: null
"""

    try:
        checker.parse_gate_registry(malformed)
    except ValueError as exc:
        assert "invalid enforcement" in str(exc)
    else:
        raise AssertionError("unknown enforcement class passed")


def test_gate_registry_fails_closed_without_branch_required_gate() -> None:
    checker = _load_required_status_script()
    malformed = """\
gates:
  - id: advisory-only
    name: Advisory Only
    context: advisory-only
    enforcement: advisory
    trusted_base: false
    workflow: .github/workflows/advisory.yml
    local_command: null
"""

    try:
        checker.parse_gate_registry(malformed)
    except ValueError as exc:
        assert "at least one branch_required gate required" in str(exc)
    else:
        raise AssertionError("registry without branch_required gate passed")


def test_gate_registry_preserves_hash_inside_quoted_scalar() -> None:
    checker = _load_required_status_script()
    registry = """\
gates:
  - id: quoted-context
    name: Quoted Context
    context: "Gate # 1"
    enforcement: branch_required # supported inline comment
    trusted_base: true
    workflow: .github/workflows/quoted.yml
    local_command: null
"""

    assert checker.parse_gate_registry(registry)[0]["context"] == "Gate # 1"


def test_gate_registry_preserves_hash_inside_plain_scalar() -> None:
    checker = _load_required_status_script()
    registry = """\
gates:
  - id: plain-hash-context
    name: Plain Hash Context
    context: Gate#1
    enforcement: branch_required
    trusted_base: true
    workflow: .github/workflows/plain.yml
    local_command: null
"""

    assert checker.parse_gate_registry(registry)[0]["context"] == "Gate#1"


def test_gate_registry_preserves_apostrophe_inside_plain_scalar() -> None:
    checker = _load_required_status_script()
    registry = """\
gates:
  - id: plain-apostrophe-name
    name: Owner's Gate
    context: owner-gate
    enforcement: branch_required
    trusted_base: true
    workflow: .github/workflows/plain.yml
    local_command: null
"""

    assert checker.parse_gate_registry(registry)[0]["name"] == "Owner's Gate"


def test_gate_registry_strips_plain_scalar_comment_after_space_hash() -> None:
    checker = _load_required_status_script()
    registry = """\
gates:
  - id: plain-comment-context
    name: Plain Comment Context
    context: Gate # supported inline comment
    enforcement: branch_required
    trusted_base: true
    workflow: .github/workflows/plain.yml
    local_command: null
"""

    assert checker.parse_gate_registry(registry)[0]["context"] == "Gate"


def test_gate_registry_preserves_escaped_quote_before_hash() -> None:
    checker = _load_required_status_script()
    registry = """\
gates:
  - id: escaped-double-quote
    name: Escaped Double Quote
    context: "Gate \\"# 1"
    enforcement: branch_required
    trusted_base: true
    workflow: .github/workflows/escaped.yml
    local_command: null
"""

    assert checker.parse_gate_registry(registry)[0]["context"] == 'Gate "# 1'


def test_gate_registry_rejects_unsupported_escape_sequence() -> None:
    checker = _load_required_status_script()
    malformed = """\
gates:
  - id: unsupported-escape
    name: Unsupported Escape
    context: "Gate\\u0020One"
    enforcement: branch_required
    trusted_base: true
    workflow: .github/workflows/escaped.yml
    local_command: null
"""

    try:
        checker.parse_gate_registry(malformed)
    except ValueError as exc:
        assert "unsupported escape sequence" in str(exc)
    else:
        raise AssertionError("unsupported escape sequence passed")


def test_gate_registry_preserves_doubled_single_quote_before_hash() -> None:
    checker = _load_required_status_script()
    registry = """\
gates:
  - id: escaped-single-quote
    name: Escaped Single Quote
    context: 'Owner''s # Gate'
    enforcement: branch_required
    trusted_base: true
    workflow: .github/workflows/escaped.yml
    local_command: null
"""

    assert checker.parse_gate_registry(registry)[0]["context"] == "Owner's # Gate"


def test_gate_registry_rejects_malformed_quoted_scalar() -> None:
    checker = _load_required_status_script()
    malformed = """\
gates:
  - id: malformed-quote
    name: Malformed Quote
    context: "Gate # 1
    enforcement: branch_required
    trusted_base: true
    workflow: .github/workflows/malformed.yml
    local_command: null
"""

    try:
        checker.parse_gate_registry(malformed)
    except ValueError as exc:
        assert "malformed quoted scalar" in str(exc)
    else:
        raise AssertionError("malformed quoted scalar passed")


def test_gate_registry_rejects_unknown_fields() -> None:
    checker = _load_required_status_script()
    malformed = """\
gates:
  - id: typo
    name: Typo
    context: typo
    enforcement: branch_required
    enforcment: advisory
    trusted_base: true
    workflow: .github/workflows/typo.yml
    local_command: null
"""

    try:
        checker.parse_gate_registry(malformed)
    except ValueError as exc:
        assert "unsupported fields: enforcment" in str(exc)
    else:
        raise AssertionError("unknown registry field passed")


def test_gate_registry_rejects_invalid_field_types() -> None:
    checker = _load_required_status_script()
    malformed = """\
gates:
  - id: invalid-types
    name: true
    context: false
    enforcement: advisory
    trusted_base: true
    workflow: .github/workflows/invalid.yml
    local_command: false
"""

    try:
        checker.parse_gate_registry(malformed)
    except ValueError as exc:
        assert "has invalid name" in str(exc)
    else:
        raise AssertionError("invalid registry field types passed")


def test_gate_registry_rejects_boolean_context_on_non_required_gate() -> None:
    checker = _load_required_status_script()
    malformed = """\
gates:
  - id: invalid-context
    name: Invalid Context
    context: false
    enforcement: advisory
    trusted_base: true
    workflow: .github/workflows/invalid.yml
    local_command: null
"""

    try:
        checker.parse_gate_registry(malformed)
    except ValueError as exc:
        assert "has invalid context" in str(exc)
    else:
        raise AssertionError("boolean context passed")


def test_gate_registry_rejects_boolean_local_command() -> None:
    checker = _load_required_status_script()
    malformed = """\
gates:
  - id: invalid-local-command
    name: Invalid Local Command
    context: invalid-local-command
    enforcement: advisory
    trusted_base: true
    workflow: .github/workflows/invalid.yml
    local_command: false
"""

    try:
        checker.parse_gate_registry(malformed)
    except ValueError as exc:
        assert "has invalid local_command" in str(exc)
    else:
        raise AssertionError("boolean local_command passed")


def test_required_status_cli_reports_registry_errors(tmp_path, capsys) -> None:
    checker = _load_required_status_script()
    registry = tmp_path / "gates.yml"
    registry.write_text(
        """\
gates:
  - id: missing-context
    name: Missing Context
    context: null
    enforcement: branch_required
    trusted_base: true
    workflow: .github/workflows/missing.yml
    local_command: null
""",
        encoding="utf-8",
    )
    payload = tmp_path / "payload.json"
    payload.write_text("{}", encoding="utf-8")

    code = checker.main(
        ["--registry-file", str(registry), "--payload-file", str(payload)]
    )
    captured = capsys.readouterr()

    assert code == 2
    assert "branch_required needs context" in captured.err


def test_required_status_check_audit_accepts_github_actions_source() -> None:
    checker = _load_required_status_script()
    payload = {
        "checks": [
            {"context": context, "app_id": checker.GITHUB_ACTIONS_APP_ID}
            for context in REQUIRED_STATUS_CONTEXTS
        ],
    }

    assert checker.required_status_check_failures(payload) == []


def test_required_status_check_audit_rejects_legacy_only_contexts() -> None:
    checker = _load_required_status_script()
    payload = {
        "contexts": list(REQUIRED_STATUS_CONTEXTS),
    }

    failures = checker.required_status_check_failures(payload)

    assert [failure.context for failure in failures] == list(REQUIRED_STATUS_CONTEXTS)
    assert all("expected app_id" in failure.reason for failure in failures)


def test_required_status_check_audit_rejects_wrong_check_source() -> None:
    checker = _load_required_status_script()
    payload = {
        "checks": [
            {"context": context, "app_id": checker.GITHUB_ACTIONS_APP_ID}
            for context in REQUIRED_STATUS_CONTEXTS[:-2]
        ] + [
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


def test_required_status_check_audit_fails_when_target_contexts_missing() -> None:
    checker = _load_required_status_script()
    payload = {
        "required_status_checks": {
            "checks": [{"context": "live-reconciliation"}],
        },
    }

    assert checker.missing_required_contexts(payload) == [
        "diff-budget",
        "plan-admission",
        "session-lane",
        "review-contract",
        "pr-body-contract",
        "Gitleaks PR secret scan",
        "Gitleaks baseline growth guard",
    ]


def test_required_status_check_audit_fails_current_live_payload_until_enrolled() -> None:
    checker = _load_required_status_script()
    payload = {
        "checks": [
            {"context": "live-reconciliation", "app_id": checker.GITHUB_ACTIONS_APP_ID},
            {"context": "Gitleaks PR secret scan", "app_id": checker.GITHUB_ACTIONS_APP_ID},
            {
                "context": "Gitleaks baseline growth guard",
                "app_id": checker.GITHUB_ACTIONS_APP_ID,
            },
        ],
    }

    failures = checker.required_status_check_failures(payload)

    assert [failure.context for failure in failures] == [
        "diff-budget",
        "plan-admission",
        "session-lane",
        "review-contract",
        "pr-body-contract",
    ]
    assert all(failure.reason == "missing required check" for failure in failures)
