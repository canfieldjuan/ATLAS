from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_claude_md_claims.py"


def load_auditor():
    spec = importlib.util.spec_from_file_location("audit_claude_md_claims", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_header_pattern_matches_supported_shapes():
    auditor = load_auditor()

    standard = "### Email MCP Server (9 tools)\n"
    soft = "### B2B Churn Intelligence MCP Server (60+ tools)\n"
    multi = "### B2B Churn Intelligence MCP Server (83 tools, 17 modules)\n"

    assert auditor.HEADER_PATTERN.search(standard).group("count") == "9"
    assert auditor.HEADER_PATTERN.search(soft).group("count") == "60+"
    assert auditor.HEADER_PATTERN.search(multi).group("count") == "83"


def test_header_pattern_rejects_missing_close_paren():
    auditor = load_auditor()

    assert auditor.HEADER_PATTERN.search("### Email MCP Server (9 tools\n") is None


def test_count_decorators_missing_file_returns_sentinel(tmp_path):
    auditor = load_auditor()

    result = auditor.count_decorators(tmp_path / "missing.py")

    assert result == auditor.MISSING_FILE
    assert result != -1


def test_count_decorators_counts_bare_and_called_tool_decorators(tmp_path):
    auditor = load_auditor()
    server = tmp_path / "server.py"
    server.write_text(
        "@mcp.tool\n"
        "def alpha():\n"
        "    pass\n\n"
        "@mcp.tool()\n"
        "def beta():\n"
        "    pass\n",
        encoding="utf-8",
    )

    assert auditor.count_decorators(server) == 2


def test_audit_claims_surfaces_unknown_server_header():
    auditor = load_auditor()

    rows = auditor.audit_claims("### New Thing MCP Server (1 tools)\n")

    assert rows == [("New Thing", "1", "?", "UNKNOWN")]
