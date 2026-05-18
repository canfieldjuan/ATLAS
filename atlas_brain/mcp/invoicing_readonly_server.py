"""
Read-only Atlas Invoicing MCP Server.

This server is intentionally separate from atlas_brain.mcp.invoicing_server.
It exposes invoice/service review tools only, with no mutation, send, payment,
PDF-export, or approval tools. Use it for authenticated MCP clients such as
ChatGPT connectors that need to inspect invoices without the ability to change
state.

Run:
    python -m atlas_brain.mcp.invoicing_readonly_server
    ATLAS_MCP_AUTH_TOKEN=<token> python -m atlas_brain.mcp.invoicing_readonly_server --sse
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from mcp.server.fastmcp import FastMCP

from . import invoicing_server as _full
from ..config_defaults import DEFAULT_INVOICING_READONLY_PORT, DEFAULT_MCP_HOST

logger = logging.getLogger("atlas.mcp.invoicing.readonly")


@asynccontextmanager
async def _lifespan(server):
    """Initialize DB pool on startup, close on shutdown."""
    from ..storage.database import close_database, init_database

    await init_database()
    logger.info("Read-only invoicing MCP: DB pool initialized")
    yield
    await close_database()


mcp = FastMCP(
    "atlas-invoicing-readonly",
    instructions=(
        "Read-only invoicing server for Atlas. "
        "Review invoices, draft blockers, services, balances, and payment "
        "history. This server cannot create, update, approve, send, void, "
        "record payments, or export PDFs."
    ),
    lifespan=_lifespan,
)


@mcp.tool()
async def get_invoice(invoice_id: str) -> str:
    """Fetch an invoice by UUID or invoice number."""
    return await _full.get_invoice(invoice_id)


@mcp.tool()
async def list_invoices(
    status: Optional[str] = None,
    contact_id: Optional[str] = None,
    business_context_id: Optional[str] = None,
    limit: int = 50,
) -> str:
    """List invoices with optional status/contact/business-context filters."""
    return await _full.list_invoices(
        status=status,
        contact_id=contact_id,
        business_context_id=business_context_id,
        limit=limit,
    )


@mcp.tool()
async def search_invoices(
    keyword: Optional[str] = None,
    status: Optional[str] = None,
    contact_id: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: int = 50,
) -> str:
    """Search invoices by keyword, status, contact, or issue-date range."""
    return await _full.search_invoices(
        keyword=keyword,
        status=status,
        contact_id=contact_id,
        from_date=from_date,
        to_date=to_date,
        limit=limit,
    )


@mcp.tool()
async def list_pending_drafts(
    contact_id: Optional[str] = None,
    business_context_id: Optional[str] = None,
    only_blocked: bool = False,
    limit: int = 100,
) -> str:
    """List draft invoices annotated with send blockers and warnings."""
    return await _full.list_pending_drafts(
        contact_id=contact_id,
        business_context_id=business_context_id,
        only_blocked=only_blocked,
        limit=limit,
    )


@mcp.tool()
async def customer_balance(
    contact_id: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
) -> str:
    """Get outstanding balance for a customer."""
    return await _full.customer_balance(contact_id=contact_id, phone=phone, email=email)


@mcp.tool()
async def payment_history(
    contact_id: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
) -> str:
    """Get payment behavior analytics for a customer."""
    return await _full.payment_history(contact_id=contact_id, phone=phone, email=email)


@mcp.tool()
async def list_services(
    contact_id: Optional[str] = None,
    status: Optional[str] = None,
) -> str:
    """List customer service agreements."""
    return await _full.list_services(contact_id=contact_id, status=status)


@mcp.tool()
async def get_service(service_id: str) -> str:
    """Get a customer service agreement by UUID."""
    return await _full.get_service(service_id)


def _streamable_http_app():
    """Build the authenticated streamable HTTP app for read-only tools."""
    from .auth import apply_auth_middleware

    if not os.environ.get("ATLAS_MCP_AUTH_TOKEN", "").strip():
        raise RuntimeError(
            "ATLAS_MCP_AUTH_TOKEN is required for read-only invoicing HTTP mode; "
            "these tools expose customer financial data."
        )
    return apply_auth_middleware(mcp.streamable_http_app())


if __name__ == "__main__":
    if "--sse" in sys.argv:
        # Streamable HTTP transport for read tools only. It still requires
        # bearer auth because these tools expose customer financial data.
        import anyio
        import uvicorn
        from mcp.server.transport_security import TransportSecuritySettings

        host = os.environ.get("ATLAS_MCP_HOST", DEFAULT_MCP_HOST)
        port = int(os.environ.get("ATLAS_MCP_INVOICING_READONLY_PORT", str(DEFAULT_INVOICING_READONLY_PORT)))

        mcp.settings.host = host
        mcp.settings.port = port
        mcp.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=False,
        )

        async def _serve():
            config = uvicorn.Config(
                _streamable_http_app(),
                host=host,
                port=port,
                log_level="info",
            )
            server = uvicorn.Server(config)
            await server.serve()

        anyio.run(_serve)
    else:
        mcp.run(transport="stdio")
