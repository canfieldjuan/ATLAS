from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_auditor():
    path = REPO_ROOT / "scripts" / "audit_pr_side_docs_test_consistency.py"
    spec = importlib.util.spec_from_file_location("audit_pr_side_docs_test_consistency", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


auditor = _load_auditor()
AUDITOR_SCRIPT = REPO_ROOT / "scripts" / "audit_pr_side_docs_test_consistency.py"


def _write_fixture(
    repo: Path,
    *,
    contexts: tuple[str, ...] = ("live-reconciliation", "unit-gate"),
    doc_contexts: tuple[str, ...] | None = None,
    test_contexts: tuple[str, ...] | None = None,
    workflow_paths: tuple[str, ...] | None = None,
    test_workflow_paths: tuple[str, ...] | None = None,
) -> None:
    doc_contexts = contexts if doc_contexts is None else doc_contexts
    test_contexts = contexts if test_contexts is None else test_contexts
    branch_required_by_context = {
        "live-reconciliation": (
            "live-reconciliation",
            "AI reconciliation live",
            "live-reconciliation",
            ".github/workflows/ai_reconciliation_live.yml",
        ),
        "unit-gate": ("unit-gate", "Unit Gate", "unit-gate", ".github/workflows/unit_gate.yml"),
    }
    branch_required = [branch_required_by_context[context] for context in contexts]
    test_workflow_paths = (
        tuple(workflow for *_prefix, workflow in branch_required)
        + auditor.EXTRA_REQUIRED_STATUS_TEST_PATHS
        if test_workflow_paths is None
        else test_workflow_paths
    )
    workflow_paths = (
        tuple(test_workflow_paths) + auditor.EXTRA_BRANCH_PROTECTION_TRIGGER_PATHS
        if workflow_paths is None
        else workflow_paths
    )

    (repo / "ci").mkdir(parents=True)
    (repo / "docs").mkdir()
    (repo / ".github" / "workflows").mkdir(parents=True)
    (repo / "tests").mkdir()

    gates = ["gates:"]
    for gate_id, name, context, workflow in branch_required:
        gates.extend(
            [
                f"  - id: {gate_id}",
                f"    name: {name}",
                f"    context: {context}",
                "    enforcement: branch_required",
                "    trusted_base: true",
                f"    workflow: {workflow}",
                "    local_command: null",
                "",
            ]
        )
    (repo / "ci" / "gates.yml").write_text("\n".join(gates), encoding="utf-8")
    for *_prefix, workflow in branch_required:
        workflow_path = repo / workflow
        workflow_path.parent.mkdir(parents=True, exist_ok=True)
        workflow_path.write_text("name: fixture\n", encoding="utf-8")

    doc_lines = [
        "# Security Guardrails",
        "",
        "Target branch protection for `main` is derived from `ci/gates.yml` entries",
        "marked `branch_required`: "
        + ", ".join(f"`{context}`" for context in doc_contexts)
        + ", all pinned to the GitHub Actions app source.",
        "",
        "Other incidental references must not define the canonical inventory.",
    ]
    (repo / "docs" / "SECURITY_GUARDRAILS.md").write_text(
        "\n".join(doc_lines),
        encoding="utf-8",
    )

    workflow_lines = [
        "on:",
        "  push:",
        "    branches:",
        "      - main",
        "    paths:",
    ]
    workflow_lines.extend(f'      - "{path}"' for path in workflow_paths)
    (repo / ".github" / "workflows" / "branch_protection_required_checks.yml").write_text(
        "\n".join(workflow_lines),
        encoding="utf-8",
    )

    test_lines = [
        "REQUIRED_STATUS_CONTEXTS = (",
        *(f'    "{context}",' for context in test_contexts),
        ")",
        "REQUIRED_STATUS_WORKFLOW_PATHS = (",
        *(f'    "{path}",' for path in test_workflow_paths),
        ")",
    ]
    (repo / "tests" / "test_security_guardrails_workflow.py").write_text(
        "\n".join(test_lines),
        encoding="utf-8",
    )


def test_audit_accepts_synchronized_pr_side_docs_tests_and_workflow(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    assert auditor.audit_repo(tmp_path) == []


def test_cli_accepts_synchronized_pr_side_docs_tests_and_workflow(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    result = subprocess.run(
        [sys.executable, str(AUDITOR_SCRIPT), "--repo-root", str(tmp_path)],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "OK: branch-required registry, docs, workflow triggers, and tests agree." in result.stdout


def test_audit_fails_when_docs_omit_branch_required_context(tmp_path: Path) -> None:
    _write_fixture(tmp_path, doc_contexts=("live-reconciliation",))

    failures = auditor.audit_repo(tmp_path)

    assert failures == [
        "docs/SECURITY_GUARDRAILS.md: missing branch-required context(s): unit-gate"
    ]


def test_audit_fails_when_docs_keep_stale_branch_required_context(
    tmp_path: Path,
) -> None:
    _write_fixture(
        tmp_path,
        contexts=("live-reconciliation",),
        doc_contexts=("live-reconciliation", "unit-gate"),
        test_contexts=("live-reconciliation",),
        test_workflow_paths=(".github/workflows/ai_reconciliation_live.yml", "ci/gates.yml"),
        workflow_paths=(
            ".github/workflows/ai_reconciliation_live.yml",
            "ci/gates.yml",
            *auditor.EXTRA_BRANCH_PROTECTION_TRIGGER_PATHS,
        ),
    )

    failures = auditor.audit_repo(tmp_path)

    assert failures == [
        "docs/SECURITY_GUARDRAILS.md: extra branch-required context(s): unit-gate"
    ]


def test_audit_fails_when_docs_reorder_branch_required_contexts(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path, doc_contexts=("unit-gate", "live-reconciliation"))

    failures = auditor.audit_repo(tmp_path)

    assert failures == [
        "docs/SECURITY_GUARDRAILS.md: branch-required context order differs from ci/gates.yml branch_required contexts"
    ]


def test_cli_reports_failure_when_docs_omit_branch_required_context(tmp_path: Path) -> None:
    _write_fixture(tmp_path, doc_contexts=("live-reconciliation",))

    result = subprocess.run(
        [sys.executable, str(AUDITOR_SCRIPT), "--repo-root", str(tmp_path)],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert (
        "FAIL: docs/SECURITY_GUARDRAILS.md: missing branch-required context(s): unit-gate"
        in result.stdout
    )


def test_audit_fails_when_workflow_omits_required_gate_path(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        workflow_paths=(
            ".github/workflows/ai_reconciliation_live.yml",
            *auditor.EXTRA_REQUIRED_STATUS_TEST_PATHS,
            *auditor.EXTRA_BRANCH_PROTECTION_TRIGGER_PATHS,
        ),
    )

    failures = auditor.audit_repo(tmp_path)

    assert any(
        ".github/workflows/branch_protection_required_checks.yml: missing push path trigger(s): .github/workflows/unit_gate.yml"
        == failure
        for failure in failures
    )


def test_audit_fails_when_workflow_path_is_only_in_comment(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        workflow_paths=(
            ".github/workflows/ai_reconciliation_live.yml",
            *auditor.EXTRA_REQUIRED_STATUS_TEST_PATHS,
            *auditor.EXTRA_BRANCH_PROTECTION_TRIGGER_PATHS,
        ),
    )
    workflow = tmp_path / ".github" / "workflows" / "branch_protection_required_checks.yml"
    workflow.write_text(
        workflow.read_text(encoding="utf-8")
        + '\n#      - ".github/workflows/unit_gate.yml"\n',
        encoding="utf-8",
    )

    failures = auditor.audit_repo(tmp_path)

    assert (
        ".github/workflows/branch_protection_required_checks.yml: missing push path trigger(s): .github/workflows/unit_gate.yml"
        in failures
    )


def test_audit_fails_when_workflow_excludes_required_gate_path(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        workflow_paths=(
            ".github/workflows/ai_reconciliation_live.yml",
            ".github/workflows/unit_gate.yml",
            "!.github/workflows/unit_gate.yml",
            *auditor.EXTRA_REQUIRED_STATUS_TEST_PATHS,
            *auditor.EXTRA_BRANCH_PROTECTION_TRIGGER_PATHS,
        ),
    )

    failures = auditor.audit_repo(tmp_path)

    assert (
        ".github/workflows/branch_protection_required_checks.yml: unsupported negative push path trigger(s): !.github/workflows/unit_gate.yml"
        in failures
    )


def test_audit_raises_when_workflow_paths_are_nested_under_branches(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    workflow = tmp_path / ".github" / "workflows" / "branch_protection_required_checks.yml"
    workflow.write_text(
        "\n".join(
            [
                "on:",
                "  push:",
                "    branches:",
                "      paths:",
                *(
                    f'        - "{path}"'
                    for path in (
                        ".github/workflows/ai_reconciliation_live.yml",
                        ".github/workflows/unit_gate.yml",
                        *auditor.EXTRA_REQUIRED_STATUS_TEST_PATHS,
                        *auditor.EXTRA_BRANCH_PROTECTION_TRIGGER_PATHS,
                    )
                ),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(auditor.AuditFailure, match="on.push.paths must be a direct child"):
        auditor.audit_repo(tmp_path)


def test_audit_raises_when_push_mapping_is_not_directly_under_on(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    workflow = tmp_path / ".github" / "workflows" / "branch_protection_required_checks.yml"
    workflow.write_text(
        "\n".join(
            [
                "on:",
                "  workflow_dispatch:",
                "    push:",
                "      paths:",
                *(
                    f'        - "{path}"'
                    for path in (
                        ".github/workflows/ai_reconciliation_live.yml",
                        ".github/workflows/unit_gate.yml",
                        *auditor.EXTRA_REQUIRED_STATUS_TEST_PATHS,
                        *auditor.EXTRA_BRANCH_PROTECTION_TRIGGER_PATHS,
                    )
                ),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(auditor.AuditFailure, match="on.push must be a direct child"):
        auditor.audit_repo(tmp_path)


def test_audit_raises_when_push_branches_exclude_main(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    workflow = tmp_path / ".github" / "workflows" / "branch_protection_required_checks.yml"
    workflow.write_text(
        workflow.read_text(encoding="utf-8").replace(
            "    branches:\n      - main",
            "    branches-ignore:\n      - main",
        ),
        encoding="utf-8",
    )

    with pytest.raises(auditor.AuditFailure, match="branches-ignore must not exclude main"):
        auditor.audit_repo(tmp_path)


def test_audit_raises_when_flow_style_push_branches_exclude_main(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    workflow = tmp_path / ".github" / "workflows" / "branch_protection_required_checks.yml"
    workflow.write_text(
        workflow.read_text(encoding="utf-8").replace(
            "    branches:\n      - main",
            "    branches-ignore: [main]",
        ),
        encoding="utf-8",
    )

    with pytest.raises(auditor.AuditFailure, match="branches-ignore must not exclude main"):
        auditor.audit_repo(tmp_path)


def test_audit_raises_when_push_branches_do_not_include_main(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    workflow = tmp_path / ".github" / "workflows" / "branch_protection_required_checks.yml"
    workflow.write_text(
        workflow.read_text(encoding="utf-8").replace(
            "    branches:\n      - main",
            "    branches:\n      - release",
        ),
        encoding="utf-8",
    )

    with pytest.raises(auditor.AuditFailure, match="branches must admit main"):
        auditor.audit_repo(tmp_path)


def test_audit_raises_when_push_branch_patterns_exclude_main(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    workflow = tmp_path / ".github" / "workflows" / "branch_protection_required_checks.yml"
    workflow.write_text(
        workflow.read_text(encoding="utf-8").replace(
            "    branches:\n      - main",
            "    branches: [main, '!main']",
        ),
        encoding="utf-8",
    )

    with pytest.raises(auditor.AuditFailure, match="branches must admit main"):
        auditor.audit_repo(tmp_path)


def test_audit_raises_when_push_branch_ignore_pattern_excludes_main(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    workflow = tmp_path / ".github" / "workflows" / "branch_protection_required_checks.yml"
    workflow.write_text(
        workflow.read_text(encoding="utf-8").replace(
            "    branches:\n      - main",
            "    branches-ignore: ['m*']",
        ),
        encoding="utf-8",
    )

    with pytest.raises(auditor.AuditFailure, match="branches-ignore must not exclude main"):
        auditor.audit_repo(tmp_path)


@pytest.mark.parametrize(
    "pattern",
    [
        "!m+ain",
        "!ma?n",
        "!m[ae]in",
        r"!m\ain",
    ],
)
def test_audit_raises_when_push_branch_pattern_uses_unsupported_github_glob(
    tmp_path: Path,
    pattern: str,
) -> None:
    _write_fixture(tmp_path)
    workflow = tmp_path / ".github" / "workflows" / "branch_protection_required_checks.yml"
    workflow.write_text(
        workflow.read_text(encoding="utf-8").replace(
            "    branches:\n      - main",
            f"    branches: [main, '{pattern}']",
        ),
        encoding="utf-8",
    )

    with pytest.raises(auditor.AuditFailure, match="unsupported on.push branch pattern syntax"):
        auditor.audit_repo(tmp_path)


@pytest.mark.parametrize(
    "pattern",
    [
        "m+ain",
        "ma?n",
        "m[ae]in",
        r"m\ain",
    ],
)
def test_audit_raises_when_push_branch_ignore_pattern_uses_unsupported_github_glob(
    tmp_path: Path,
    pattern: str,
) -> None:
    _write_fixture(tmp_path)
    workflow = tmp_path / ".github" / "workflows" / "branch_protection_required_checks.yml"
    workflow.write_text(
        workflow.read_text(encoding="utf-8").replace(
            "    branches:\n      - main",
            f"    branches-ignore: ['{pattern}']",
        ),
        encoding="utf-8",
    )

    with pytest.raises(auditor.AuditFailure, match="unsupported on.push branch pattern syntax"):
        auditor.audit_repo(tmp_path)


def test_audit_fails_when_test_context_tuple_is_stale(tmp_path: Path) -> None:
    _write_fixture(tmp_path, test_contexts=("live-reconciliation",))

    failures = auditor.audit_repo(tmp_path)

    assert (
        "tests/test_security_guardrails_workflow.py: REQUIRED_STATUS_CONTEXTS differs from ci/gates.yml branch_required contexts"
        in failures
    )


def test_audit_fails_when_test_context_tuple_is_reordered_or_duplicated(tmp_path: Path) -> None:
    _write_fixture(tmp_path, test_contexts=("unit-gate", "live-reconciliation", "unit-gate"))

    failures = auditor.audit_repo(tmp_path)

    assert (
        "tests/test_security_guardrails_workflow.py: REQUIRED_STATUS_CONTEXTS differs from ci/gates.yml branch_required contexts"
        in failures
    )


def test_audit_raises_when_test_context_tuple_is_assigned_twice(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    test_path = tmp_path / "tests" / "test_security_guardrails_workflow.py"
    test_path.write_text(
        test_path.read_text(encoding="utf-8")
        + '\nREQUIRED_STATUS_CONTEXTS = ("live-reconciliation",)\n',
        encoding="utf-8",
    )

    with pytest.raises(
        auditor.AuditFailure,
        match="multiple assignments for REQUIRED_STATUS_CONTEXTS",
    ):
        auditor.audit_repo(tmp_path)


def test_audit_raises_when_test_context_constant_is_a_list(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    test_path = tmp_path / "tests" / "test_security_guardrails_workflow.py"
    test_path.write_text(
        "\n".join(
            [
                'REQUIRED_STATUS_CONTEXTS = ["live-reconciliation", "unit-gate"]',
                "REQUIRED_STATUS_WORKFLOW_PATHS = (",
                '    ".github/workflows/ai_reconciliation_live.yml",',
                '    ".github/workflows/unit_gate.yml",',
                '    "ci/gates.yml",',
                ")",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(auditor.AuditFailure, match="must be a literal tuple of strings"):
        auditor.audit_repo(tmp_path)


def test_audit_raises_when_test_context_tuple_is_mutated(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    test_path = tmp_path / "tests" / "test_security_guardrails_workflow.py"
    test_path.write_text(
        test_path.read_text(encoding="utf-8")
        + '\nREQUIRED_STATUS_CONTEXTS += ("shadow-required",)\n',
        encoding="utf-8",
    )

    with pytest.raises(
        auditor.AuditFailure,
        match="mutating assignment for REQUIRED_STATUS_CONTEXTS",
    ):
        auditor.audit_repo(tmp_path)


def test_audit_raises_when_test_context_tuple_is_mutated_in_nested_block(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    test_path = tmp_path / "tests" / "test_security_guardrails_workflow.py"
    test_path.write_text(
        test_path.read_text(encoding="utf-8")
        + '\nif True:\n    REQUIRED_STATUS_CONTEXTS += ("shadow-required",)\n',
        encoding="utf-8",
    )

    with pytest.raises(
        auditor.AuditFailure,
        match="mutating assignment for REQUIRED_STATUS_CONTEXTS",
    ):
        auditor.audit_repo(tmp_path)


def test_audit_raises_when_test_context_tuple_is_assigned_in_nested_block(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    test_path = tmp_path / "tests" / "test_security_guardrails_workflow.py"
    test_path.write_text(
        "\n".join(
            [
                "if False:",
                '    REQUIRED_STATUS_CONTEXTS = ("live-reconciliation", "unit-gate")',
                "REQUIRED_STATUS_WORKFLOW_PATHS = (",
                '    ".github/workflows/ai_reconciliation_live.yml",',
                '    ".github/workflows/unit_gate.yml",',
                '    "ci/gates.yml",',
                ")",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        auditor.AuditFailure,
        match="runtime binding for REQUIRED_STATUS_CONTEXTS",
    ):
        auditor.audit_repo(tmp_path)


def test_audit_raises_when_test_context_constant_is_rebound_by_loop(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    test_path = tmp_path / "tests" / "test_security_guardrails_workflow.py"
    test_path.write_text(
        test_path.read_text(encoding="utf-8")
        + '\nfor REQUIRED_STATUS_CONTEXTS in [("shadow-required",)]:\n    pass\n',
        encoding="utf-8",
    )

    with pytest.raises(
        auditor.AuditFailure,
        match="runtime binding for REQUIRED_STATUS_CONTEXTS",
    ):
        auditor.audit_repo(tmp_path)


@pytest.mark.parametrize(
    "pattern_source",
    [
        'match ("shadow-required",):\n    case REQUIRED_STATUS_CONTEXTS:\n        pass\n',
        'match {"contexts": ("shadow-required",)}:\n    case {"contexts": _, **REQUIRED_STATUS_CONTEXTS}:\n        pass\n',
    ],
)
def test_audit_raises_when_test_context_constant_is_rebound_by_pattern_capture(
    tmp_path: Path,
    pattern_source: str,
) -> None:
    _write_fixture(tmp_path)
    test_path = tmp_path / "tests" / "test_security_guardrails_workflow.py"
    test_path.write_text(
        test_path.read_text(encoding="utf-8") + "\n" + pattern_source,
        encoding="utf-8",
    )

    with pytest.raises(
        auditor.AuditFailure,
        match="runtime binding for REQUIRED_STATUS_CONTEXTS",
    ):
        auditor.audit_repo(tmp_path)


def test_audit_raises_when_test_context_constant_is_rebound_indirectly(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    test_path = tmp_path / "tests" / "test_security_guardrails_workflow.py"
    test_path.write_text(
        test_path.read_text(encoding="utf-8")
        + '\nglobals()["REQUIRED_STATUS_CONTEXTS"] = ("shadow-required",)\n',
        encoding="utf-8",
    )

    with pytest.raises(
        auditor.AuditFailure,
        match="runtime binding for REQUIRED_STATUS_CONTEXTS",
    ):
        auditor.audit_repo(tmp_path)


def test_audit_raises_when_test_context_constant_is_rebound_through_namespace_alias(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    test_path = tmp_path / "tests" / "test_security_guardrails_workflow.py"
    test_path.write_text(
        test_path.read_text(encoding="utf-8")
        + '\nnamespace = globals()\nnamespace["REQUIRED_STATUS_CONTEXTS"] = ("shadow-required",)\n',
        encoding="utf-8",
    )

    with pytest.raises(
        auditor.AuditFailure,
        match="runtime binding for REQUIRED_STATUS_CONTEXTS",
    ):
        auditor.audit_repo(tmp_path)


def test_audit_raises_when_test_context_constant_is_rebound_through_namespace_attribute(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    test_path = tmp_path / "tests" / "test_security_guardrails_workflow.py"
    test_path.write_text(
        test_path.read_text(encoding="utf-8")
        + '\nimport builtins\nbuiltins.globals()["REQUIRED_STATUS_CONTEXTS"] = ("shadow-required",)\n',
        encoding="utf-8",
    )

    with pytest.raises(
        auditor.AuditFailure,
        match="runtime binding for REQUIRED_STATUS_CONTEXTS",
    ):
        auditor.audit_repo(tmp_path)


def test_audit_fails_when_test_workflow_tuple_is_stale(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        test_workflow_paths=(
            ".github/workflows/ai_reconciliation_live.yml",
            *auditor.EXTRA_REQUIRED_STATUS_TEST_PATHS,
        ),
    )

    failures = auditor.audit_repo(tmp_path)

    assert (
        "tests/test_security_guardrails_workflow.py: REQUIRED_STATUS_WORKFLOW_PATHS differs from ci/gates.yml branch_required workflows"
        in failures
    )


def test_audit_fails_when_registry_workflow_path_is_missing(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    (tmp_path / ".github" / "workflows" / "unit_gate.yml").unlink()

    failures = auditor.audit_repo(tmp_path)

    assert (
        "ci/gates.yml: registry workflow path(s) missing from PR tree: .github/workflows/unit_gate.yml"
        in failures
    )


def test_audit_fails_when_registry_workflow_path_names_docs_file(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    replacements = (
        tmp_path / "ci" / "gates.yml",
        tmp_path / ".github" / "workflows" / "branch_protection_required_checks.yml",
        tmp_path / "tests" / "test_security_guardrails_workflow.py",
    )
    for path in replacements:
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                ".github/workflows/unit_gate.yml",
                "docs/SECURITY_GUARDRAILS.md",
            ),
            encoding="utf-8",
        )

    failures = auditor.audit_repo(tmp_path)

    assert (
        "ci/gates.yml: registry workflow path(s) must be regular workflow files under .github/workflows: docs/SECURITY_GUARDRAILS.md"
        in failures
    )


def test_audit_fails_when_registry_workflow_path_has_non_workflow_suffix(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    replacement = ".github/workflows/unit_gate.txt"
    (tmp_path / replacement).write_text("name: not a workflow\n", encoding="utf-8")
    replacements = (
        tmp_path / "ci" / "gates.yml",
        tmp_path / ".github" / "workflows" / "branch_protection_required_checks.yml",
        tmp_path / "tests" / "test_security_guardrails_workflow.py",
    )
    for path in replacements:
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                ".github/workflows/unit_gate.yml",
                replacement,
            ),
            encoding="utf-8",
        )

    failures = auditor.audit_repo(tmp_path)

    assert (
        "ci/gates.yml: registry workflow path(s) must be regular workflow files under .github/workflows: .github/workflows/unit_gate.txt"
        in failures
    )


def test_audit_fails_when_registry_workflow_path_is_symlink(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    workflow = tmp_path / ".github" / "workflows" / "unit_gate.yml"
    workflow.unlink()
    workflow.symlink_to("ai_reconciliation_live.yml")

    failures = auditor.audit_repo(tmp_path)

    assert (
        "ci/gates.yml: registry workflow path(s) must be regular workflow files under .github/workflows: .github/workflows/unit_gate.yml"
        in failures
    )


def test_audit_fails_when_non_required_registry_workflow_path_is_missing(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    gates = tmp_path / "ci" / "gates.yml"
    gates.write_text(
        gates.read_text(encoding="utf-8")
        + "\n".join(
            [
                "  - id: advisory-missing-workflow",
                "    name: Advisory missing workflow",
                "    context: advisory-missing-workflow",
                "    enforcement: advisory",
                "    trusted_base: false",
                "    workflow: .github/workflows/missing_advisory.yml",
                "    local_command: null",
                "",
            ]
        ),
        encoding="utf-8",
    )

    failures = auditor.audit_repo(tmp_path)

    assert (
        "ci/gates.yml: registry workflow path(s) missing from PR tree: .github/workflows/missing_advisory.yml"
        in failures
    )


def test_audit_fails_when_registry_workflow_path_escapes_repo_root(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    replacements = (
        tmp_path / "ci" / "gates.yml",
        tmp_path / ".github" / "workflows" / "branch_protection_required_checks.yml",
        tmp_path / "tests" / "test_security_guardrails_workflow.py",
    )
    for path in replacements:
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                ".github/workflows/unit_gate.yml",
                "/etc/passwd",
            ),
            encoding="utf-8",
        )

    failures = auditor.audit_repo(tmp_path)

    assert (
        "ci/gates.yml: registry workflow path(s) must stay inside the PR tree: /etc/passwd"
        in failures
    )


def test_audit_raises_when_required_registry_is_missing(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    (tmp_path / "ci" / "gates.yml").unlink()

    with pytest.raises(auditor.AuditFailure, match="ci/gates.yml: could not read file"):
        auditor.audit_repo(tmp_path)
