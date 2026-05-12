"""Fixture tests for scripts/audit_mcp_tool_names_match_docs.py."""
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
    out = auditor.doc_claims(KNOWN_SECTION)

    assert isinstance(out, tuple)
    assert len(out) == 2
    claims, unknown = out
    assert isinstance(claims, dict)
    assert isinstance(unknown, list)


def test_known_server_header_lands_in_claims(auditor):
    claims, unknown = auditor.doc_claims(KNOWN_SECTION)

    assert "Email" in claims
    assert unknown == []
    assert "send_email" in claims["Email"]
    assert "list_folders" in claims["Email"]


def test_unknown_server_header_surfaces_in_unknown_list(auditor):
    claims, unknown = auditor.doc_claims(UNKNOWN_SECTION)

    assert "Foobar" in unknown
    assert "Foobar" not in claims


def test_mixed_known_and_unknown(auditor):
    text = KNOWN_SECTION + "\n" + UNKNOWN_SECTION

    claims, unknown = auditor.doc_claims(text)

    assert "Email" in claims
    assert "Foobar" in unknown
