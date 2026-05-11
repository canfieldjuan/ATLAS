"""Fixture tests for scripts/audit_mcp_tool_names_match_docs.py.

Locks down the regression Copilot caught on PR #484: doc_claims()
originally silently dropped `### <Name> MCP Server` headers whose
name was not in HEADER_TO_FILE. A renamed or newly added server
disappeared from coverage silently. The fix made doc_claims()
return (claims, unknown_headers) so main() can surface unknowns
as DRIFT.

This test continues to pin that contract: an unknown header in the
fixture must show up in the returned `unknown_headers` list.
"""
from __future__ import annotations

import textwrap

import pytest

from tests.audit_helpers import load_auditor


@pytest.fixture(scope="module")
def auditor():
    return load_auditor("audit_mcp_tool_names_match_docs")


KNOWN_SECTION = textwrap.dedent("""\
    ### Email MCP Server (9 tools)

    Tools: `send_email`, `read_inbox`, `list_folders`
""")

UNKNOWN_SECTION = textwrap.dedent("""\
    ### Foobar MCP Server (5 tools)

    Tools: `do_thing`, `do_other_thing`
""")


def test_doc_claims_returns_tuple_of_known_and_unknown(auditor):
    """The function returns a (claims, unknown) tuple per PR #484
    Copilot review."""
    out = auditor.doc_claims(KNOWN_SECTION)
    assert isinstance(out, tuple)
    assert len(out) == 2
    claims, unknown = out
    assert isinstance(claims, dict)
    assert isinstance(unknown, list)


def test_known_server_header_lands_in_claims(auditor):
    """Happy path: a known server name populates the claims dict."""
    claims, unknown = auditor.doc_claims(KNOWN_SECTION)
    assert "Email" in claims
    assert unknown == []
    # The Tools-line backticked snake_case identifiers (>=4 chars)
    # are gathered as the inventory.
    assert "send_email" in claims["Email"]
    assert "list_folders" in claims["Email"]


def test_unknown_server_header_surfaces_in_unknown_list(auditor):
    """Regression for PR #484 Copilot catch: a header whose name is
    not in HEADER_TO_FILE must land in `unknown_headers`, not be
    silently dropped (per AGENTS.md section 3e)."""
    claims, unknown = auditor.doc_claims(UNKNOWN_SECTION)
    assert "Foobar" in unknown
    # The unknown header should NOT pollute the claims dict either.
    assert "Foobar" not in claims


def test_mixed_known_and_unknown(auditor):
    """Both kinds of headers in the same doc are partitioned."""
    text = KNOWN_SECTION + "\n" + UNKNOWN_SECTION
    claims, unknown = auditor.doc_claims(text)
    assert "Email" in claims
    assert "Foobar" in unknown
