from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "audit_mcp_port_assignments.py"


def load_auditor():
    name = "audit_mcp_port_assignments"
    if name in sys.modules:
        return sys.modules[name]

    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _config() -> str:
    return textwrap.dedent(
        """\
        from pydantic import Field

        class MCPConfig:
            crm_port: int = Field(default=8056)
            b2b_churn_port: int = Field(default=8062)
            memory_port: int = Field(default=8064)
            transport: str = Field(default="stdio")
        """
    )


def test_mcp_config_ports_extracts_field_defaults():
    auditor = load_auditor()

    assert auditor.mcp_config_ports(_config()) == {
        "crm": 8056,
        "b2b_churn": 8062,
        "memory": 8064,
    }


def test_documented_port_claims_parse_env_and_sse_examples():
    auditor = load_auditor()
    doc = textwrap.dedent(
        """\
        ATLAS_MCP_CRM_PORT=8056
        ATLAS_MCP_B2B_CHURN_PORT=8062

        # SSE HTTP mode (port 8064)
        python -m atlas_brain.mcp.memory_server --sse
        """
    )

    claims = auditor.documented_port_claims(doc)

    assert [claim.port for claim in claims["crm"]] == [8056]
    assert [claim.port for claim in claims["b2b_churn"]] == [8062]
    assert [claim.port for claim in claims["memory"]] == [8064]


def test_audit_ports_reports_missing_drift_conflict_and_extra():
    auditor = load_auditor()
    claims = auditor.documented_port_claims(
        textwrap.dedent(
            """\
            ATLAS_MCP_CRM_PORT=8057
            ATLAS_MCP_MEMORY_PORT=8064
            ATLAS_MCP_MEMORY_PORT=8065
            ATLAS_MCP_EXTRA_PORT=9000
            """
        )
    )

    rows = {
        row.name: row.status
        for row in auditor.audit_ports(
            {"crm": 8056, "b2b_churn": 8062, "memory": 8064},
            claims,
        )
    }

    assert rows == {
        "b2b_churn": "MISSING",
        "crm": "DRIFT",
        "extra": "EXTRA",
        "memory": "CONFLICT",
    }


def test_cli_accepts_matching_docs(tmp_path):
    doc = tmp_path / "CLAUDE.md"
    config = tmp_path / "config.py"
    config.write_text(_config(), encoding="utf-8")
    doc.write_text(
        textwrap.dedent(
            """\
            ATLAS_MCP_CRM_PORT=8056
            ATLAS_MCP_B2B_CHURN_PORT=8062

            # SSE HTTP mode (port 8064)
            python -m atlas_brain.mcp.memory_server --sse
            """
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(doc), str(config)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "OK" in result.stdout


def test_cli_rejects_port_drift(tmp_path):
    doc = tmp_path / "CLAUDE.md"
    config = tmp_path / "config.py"
    config.write_text(_config(), encoding="utf-8")
    doc.write_text(
        textwrap.dedent(
            """\
            ATLAS_MCP_CRM_PORT=9999
            ATLAS_MCP_B2B_CHURN_PORT=8062

            # SSE HTTP mode (port 8064)
            python -m atlas_brain.mcp.memory_server --sse
            """
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(doc), str(config)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "DRIFT" in result.stdout
    assert "crm" in result.stdout
