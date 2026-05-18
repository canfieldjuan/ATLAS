from __future__ import annotations

import pytest

from atlas_brain.mcp.auth import BearerAuthMiddleware
from atlas_brain.mcp import invoicing_readonly_server as readonly


READONLY_TOOLS = {
    "get_invoice",
    "list_invoices",
    "search_invoices",
    "list_pending_drafts",
    "customer_balance",
    "payment_history",
    "list_services",
    "get_service",
}

MUTATING_TOOLS = {
    "approve_and_send",
    "create_invoice",
    "create_service",
    "export_invoice_pdf",
    "mark_void",
    "record_payment",
    "send_invoice",
    "set_service_status",
    "update_invoice",
    "update_service",
}


def _tool_names() -> set[str]:
    return set(readonly.mcp._tool_manager._tools)


def test_invoicing_readonly_mcp_exposes_exact_read_tool_surface():
    assert _tool_names() == READONLY_TOOLS
    assert _tool_names().isdisjoint(MUTATING_TOOLS)


@pytest.mark.asyncio
async def test_invoicing_readonly_mcp_delegates_to_full_read_tool(monkeypatch):
    calls = []

    async def fake_list_invoices(**kwargs):
        calls.append(kwargs)
        return '{"ok": true}'

    monkeypatch.setattr(readonly._full, "list_invoices", fake_list_invoices)

    result = await readonly.list_invoices(
        status="draft",
        contact_id="contact-1",
        business_context_id="firefly",
        limit=3,
    )

    assert result == '{"ok": true}'
    assert calls == [
        {
            "status": "draft",
            "contact_id": "contact-1",
            "business_context_id": "firefly",
            "limit": 3,
        }
    ]


def test_invoicing_readonly_http_requires_bearer_token(monkeypatch):
    monkeypatch.delenv("ATLAS_MCP_AUTH_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="ATLAS_MCP_AUTH_TOKEN is required"):
        readonly._streamable_http_app()


def test_invoicing_readonly_http_wraps_with_bearer_auth(monkeypatch):
    monkeypatch.setenv("ATLAS_MCP_AUTH_TOKEN", "test-token")

    app = readonly._streamable_http_app()

    assert isinstance(app, BearerAuthMiddleware)
