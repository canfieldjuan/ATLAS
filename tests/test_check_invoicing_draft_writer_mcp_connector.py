from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_invoicing_draft_writer_mcp_connector.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "check_invoicing_draft_writer_mcp_connector",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tool_surface_errors_accept_exact_draft_writer_tools() -> None:
    module = _load_script_module()

    assert module._tool_surface_errors(set(module.EXPECTED_DRAFT_WRITER_TOOLS)) == []


def test_tool_surface_errors_reject_missing_draft_writer_tool() -> None:
    module = _load_script_module()
    tools = set(module.EXPECTED_DRAFT_WRITER_TOOLS)
    tools.remove("create_draft_invoice")

    errors = module._tool_surface_errors(tools)

    assert errors == ["missing draft-writer tools: create_draft_invoice"]


def test_tool_surface_errors_reject_extra_and_denied_tools() -> None:
    module = _load_script_module()
    tools = set(module.EXPECTED_DRAFT_WRITER_TOOLS)
    tools.update({"send_invoice", "surprise_tool"})

    errors = module._tool_surface_errors(tools)

    assert "unexpected tools exposed: send_invoice, surprise_tool" in errors
    assert "denied tools exposed: send_invoice" in errors


def test_main_requires_token_before_network(monkeypatch, capsys) -> None:
    module = _load_script_module()

    def fail_probe(*_args, **_kwargs):
        raise AssertionError("network touched")

    monkeypatch.delenv("ATLAS_MCP_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(module, "_unauth_status_code", fail_probe)

    result = module._main(["--url", "http://example.invalid/mcp"])

    captured = capsys.readouterr()
    assert result == 2
    assert "ATLAS_MCP_AUTH_TOKEN or --token is required" in captured.err


def test_main_rejects_non_401_unauthenticated_probe(monkeypatch, capsys) -> None:
    module = _load_script_module()
    monkeypatch.setattr(module, "_unauth_status_code", lambda *_args: 200)

    result = module._main(["--url", "http://example.invalid/mcp", "--token", "token-value"])

    captured = capsys.readouterr()
    assert result == 1
    assert "expected unauthenticated request to return 401, got 200" in captured.err


def test_main_reports_authenticated_tool_surface(monkeypatch, capsys) -> None:
    module = _load_script_module()
    monkeypatch.setattr(module, "_unauth_status_code", lambda *_args: 401)

    async def fake_list_tools(*_args, **_kwargs):
        return set(module.EXPECTED_DRAFT_WRITER_TOOLS)

    monkeypatch.setattr(module, "_list_tools", fake_list_tools)

    result = module._main(["--url", "http://example.invalid/mcp", "--token", "token-value"])

    captured = capsys.readouterr()
    assert result == 0
    assert "exposes 4 draft-writer tools" in captured.out
    assert "- create_draft_invoice" in captured.out
