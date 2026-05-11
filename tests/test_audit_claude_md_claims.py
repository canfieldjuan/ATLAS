"""Fixture tests for scripts/audit_claude_md_claims.py.

Locks down two regressions Copilot caught on PR #483:

  1. HEADER_PATTERN must require the closing ")". The original regex
     `^###\\s+(?P<name>.+?)\\s+MCP Server\\s*\\(\\s*(?P<count>\\d+\\+?)`
     accepted malformed headers missing the closing paren. The fix
     anchored the pattern to a literal `)` so `(9 tools` (no close)
     no longer matches.

  2. When a server file/dir is missing, count_decorators() returns a
     `MISSING_FILE` / `MISSING_DIR` string sentinel, NOT a -1 int.
     Previously -1 leaked into output as a fake "actual count" --
     hard to interpret in failure mode.
"""
from __future__ import annotations

import pytest

from tests.audit_helpers import load_auditor


@pytest.fixture(scope="module")
def auditor():
    return load_auditor("audit_claude_md_claims")


def test_header_pattern_matches_well_formed(auditor):
    """Happy path: a well-formed section header parses cleanly."""
    text = "### Email MCP Server (9 tools)\n"
    matches = list(auditor.HEADER_PATTERN.finditer(text))
    assert len(matches) == 1
    assert matches[0].group("name") == "Email"
    assert matches[0].group("count") == "9"


def test_header_pattern_matches_soft_count(auditor):
    """Soft count ("60+") format is still supported."""
    text = "### B2B Churn Intelligence MCP Server (60+ tools)\n"
    matches = list(auditor.HEADER_PATTERN.finditer(text))
    assert len(matches) == 1
    assert matches[0].group("count") == "60+"


def test_header_pattern_matches_multi_column(auditor):
    """The `(83 tools, 17 modules)` shape (post-PR-#457) parses."""
    text = "### B2B Churn Intelligence MCP Server (83 tools, 17 modules)\n"
    matches = list(auditor.HEADER_PATTERN.finditer(text))
    assert len(matches) == 1
    assert matches[0].group("count") == "83"


def test_header_pattern_rejects_missing_close_paren(auditor):
    """Regression for PR #483 Copilot catch: malformed header without
    closing ")" must not match. Previously matched and polluted output.
    """
    text = "### Email MCP Server (9 tools\n"
    matches = list(auditor.HEADER_PATTERN.finditer(text))
    assert matches == []


def test_count_decorators_missing_file_returns_sentinel(auditor, tmp_path):
    """Regression for PR #483 Copilot catch: missing file returns
    the string sentinel "MISSING_FILE", not -1."""
    nonexistent = tmp_path / "does_not_exist.py"
    result = auditor.count_decorators(nonexistent)
    assert result == auditor.MISSING_FILE
    assert result != -1


def test_count_decorators_real_file(auditor, tmp_path):
    """Happy path: counting decorators in a real file works."""
    p = tmp_path / "fake_server.py"
    p.write_text(
        "@mcp.tool\n"
        "def alpha():\n    pass\n\n"
        "@mcp.tool()\n"
        "def beta():\n    pass\n",
        encoding="utf-8",
    )
    assert auditor.count_decorators(p) == 2
