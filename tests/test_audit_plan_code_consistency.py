"""Fixture tests for scripts/audit_plan_code_consistency.py."""
from __future__ import annotations

import textwrap

import pytest

from tests.audit_helpers import load_auditor


@pytest.fixture(scope="module")
def auditor():
    return load_auditor("audit_plan_code_consistency")


def test_parse_claims_uses_exact_section_titles(auditor):
    plan = textwrap.dedent("""\
        # Example

        ## Out of scope

        `scripts/imaginary.py`

        ## Scope (this PR)

        `scripts/audit_plan_code_consistency.py`
    """)

    paths, funcs = auditor.parse_claims(plan)

    assert paths == {"scripts/audit_plan_code_consistency.py"}
    assert funcs == set()


def test_parse_claims_accepts_root_and_hyphenated_paths(auditor):
    plan = textwrap.dedent("""\
        # Example

        ## Scope (this PR)

        `AGENTS.md`
        `plans/PR-Audit-The-Auditors-1.md`
    """)

    paths, _ = auditor.parse_claims(plan)

    assert "AGENTS.md" in paths
    assert "plans/PR-Audit-The-Auditors-1.md" in paths


def test_parse_claims_reads_only_backticked_paths(auditor):
    plan = textwrap.dedent("""\
        # Example

        ## Scope (this PR)

        This prose mentions scripts/not_backticked.py but should not enforce it.
        `scripts/audit_plan_code_consistency.py`
    """)

    paths, _ = auditor.parse_claims(plan)

    assert paths == {"scripts/audit_plan_code_consistency.py"}


def test_parse_claims_reads_backticked_function_calls(auditor):
    plan = textwrap.dedent("""\
        # Example

        ## Mechanism

        Calls `parse_claims()` but ignores get() because it is too short.
    """)

    _, funcs = auditor.parse_claims(plan)

    assert funcs == {"parse_claims"}


def test_audit_claims_reports_missing_path_and_function(auditor):
    plan = textwrap.dedent("""\
        # Example

        ## Scope (this PR)

        `scripts/does_not_exist.py`

        ## Mechanism

        Calls `function_that_does_not_exist()`.
    """)

    missing_paths, missing_functions = auditor.audit_claims(plan)

    assert missing_paths == ["scripts/does_not_exist.py"]
    assert missing_functions == ["function_that_does_not_exist"]


def test_audit_claims_accepts_existing_root_path_and_function(auditor):
    plan = textwrap.dedent("""\
        # Example

        ## Scope (this PR)

        `AGENTS.md`

        ## Mechanism

        Calls `parse_claims()`.
    """)

    missing_paths, missing_functions = auditor.audit_claims(plan)

    assert missing_paths == []
    assert missing_functions == []
