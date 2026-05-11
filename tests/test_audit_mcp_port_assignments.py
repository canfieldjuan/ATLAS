"""Fixture tests for scripts/audit_mcp_port_assignments.py.

Locks down the regression Copilot caught on PR #485: the ENV_VAR_LINE
regex was `[A-Z][A-Z_]+` which rejected digits, silently dropping
`ATLAS_MCP_B2B_CHURN_PORT=8062` (because of the "2" in "B2B"). The
auditor then reported b2b_churn as MISSING-IN-DOC even though
CLAUDE.md documented it.

This test must continue to pass: a future "small tweak" to the regex
that re-introduces the digit-rejection bug will fail
`test_env_var_line_matches_digit_in_name`.
"""
from __future__ import annotations

import pytest

from tests.audit_helpers import load_auditor


@pytest.fixture(scope="module")
def auditor():
    return load_auditor("audit_mcp_port_assignments")


def test_env_var_line_matches_digit_in_name(auditor):
    """Regression for PR #485 Copilot catch: ENV_VAR_LINE must accept
    digits in the env-var name part. The previous `[A-Z][A-Z_]+`
    class rejected the "2" in B2B_CHURN, silently dropping the row.
    """
    text = (
        "ATLAS_MCP_CRM_PORT=8056\n"
        "ATLAS_MCP_B2B_CHURN_PORT=8062\n"
        "ATLAS_MCP_INTELLIGENCE_PORT=8061\n"
    )
    claims = auditor.doc_claims(text)
    names = {c[1] for c in claims}
    assert "b2b_churn" in names, (
        "ENV_VAR_LINE regex must allow digits in the env-var name "
        "(see PR #485 Copilot review on the B2B_CHURN miss)"
    )
    # All three expected ports should be present.
    assert names >= {"crm", "b2b_churn", "intelligence"}


def test_happy_path_all_documented_match_config(auditor):
    """Happy path: every port claim matches MCPConfig's default."""
    text = "ATLAS_MCP_CRM_PORT=8056\n"
    claims = auditor.doc_claims(text)
    assert len(claims) == 1
    line_no, name, port, kind = claims[0]
    assert name == "crm"
    assert port == 8056
    assert kind == "env"


def test_unknown_env_var_name_surfaces_as_drift(auditor):
    """Per AGENTS.md section 3e: an env-var-style port claim whose
    name is not in NAME_NORMALIZER must surface (not silently skip).
    """
    text = "ATLAS_MCP_NEW_SERVER_PORT=9999\n"
    claims = auditor.doc_claims(text)
    assert len(claims) == 1
    _, name, port, _ = claims[0]
    # The raw lowercased name should land in the claim list so
    # main() can render it as UNKNOWN drift.
    assert name == "new_server"
    assert port == 9999
