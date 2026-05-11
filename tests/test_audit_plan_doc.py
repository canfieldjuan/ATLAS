"""Fixture tests for scripts/audit_plan_doc.py.

Locks down the regression Copilot caught on PR #483: the original
required-section check used `required.lower() in heading.lower()`
which is substring matching. A heading like "## Out of scope" would
spuriously satisfy the "Scope" requirement because "scope" is a
substring of "out of scope". The fix was an allowlist of acceptable
normalized heading variants per slot.

This test must continue to pass: a future refactor that loosens the
match back to substring containment will fail
`test_out_of_scope_must_not_satisfy_scope`.
"""
from __future__ import annotations

import textwrap

import pytest

from tests.audit_helpers import load_auditor


@pytest.fixture(scope="module")
def auditor():
    return load_auditor("audit_plan_doc")


HAPPY_PLAN = textwrap.dedent("""\
    # plan

    ## Why this slice exists

    foo

    ## Scope (this PR)

    foo

    ## Mechanism

    foo

    ## Intentional

    foo

    ## Deferred

    foo

    ## Verification

    foo

    ## Estimated diff size

    foo
""")


SCOPE_TRAP_PLAN = textwrap.dedent("""\
    # plan

    ## Why this slice exists

    foo

    ## Out of scope

    not a real Scope section -- substring match would falsely accept this.

    ## Mechanism

    foo

    ## Intentional

    foo

    ## Deferred

    foo

    ## Verification

    foo

    ## Estimated diff size

    foo
""")


def test_happy_path_seven_sections_pass(auditor, tmp_path, capsys):
    p = tmp_path / "happy.md"
    p.write_text(HAPPY_PLAN)
    rc = auditor.main_args(p) if hasattr(auditor, "main_args") else None
    # The script's main() reads sys.argv; call via monkeypatch.
    import sys
    saved = sys.argv
    sys.argv = ["audit_plan_doc.py", str(p)]
    try:
        rc = auditor.main()
    finally:
        sys.argv = saved
    assert rc == 0


def test_out_of_scope_must_not_satisfy_scope(auditor, tmp_path):
    """Regression for PR #483 Copilot catch: substring matching is
    forbidden; only allowlisted variants of each section title pass.
    """
    p = tmp_path / "trap.md"
    p.write_text(SCOPE_TRAP_PLAN)
    import sys
    saved = sys.argv
    sys.argv = ["audit_plan_doc.py", str(p)]
    try:
        rc = auditor.main()
    finally:
        sys.argv = saved
    # The "Out of scope" heading must NOT satisfy the "Scope" slot,
    # so the audit reports Scope as MISSING and exits 1.
    assert rc == 1


def test_pathological_missing_path(auditor):
    """Pathological input: invocation with a path that does not exist
    must return a non-zero exit code, not silently pass."""
    import sys
    saved = sys.argv
    sys.argv = ["audit_plan_doc.py", "/nonexistent/path/to/plan.md"]
    try:
        rc = auditor.main()
    finally:
        sys.argv = saved
    assert rc != 0
