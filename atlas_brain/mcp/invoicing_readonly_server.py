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
from .invoicing_readonly_oauth import (
    DEFAULT_READONLY_SCOPE,
    InvoicingReadonlyOAuthProvider,
    as_any_http_url,
    handle_approval_request,
    validate_oauth_settings,
)
from ..config_defaults import DEFAULT_INVOICING_READONLY_PORT, DEFAULT_MCP_HOST

logger = logging.getLogger("atlas.mcp.invoicing.readonly")

_AUTH_MODE_BEARER = "bearer"
_AUTH_MODE_OAUTH = "oauth"
_MIN_HTTP_AUTH_TOKEN_LENGTH = 24
_PLACEHOLDER_HTTP_AUTH_TOKENS = {
    "<token>",
    "changeme",
    "change-me",
    "password",
    "secret",
    "test-readonly-token",
    "test-token",
    "token",
}
_oauth_provider: InvoicingReadonlyOAuthProvider | None = None


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


@mcp.custom_route("/oauth/approve", methods=["GET", "POST"], include_in_schema=False)
async def _oauth_approve(request):
    """Operator approval page for ChatGPT-style OAuth connectors."""
    if _oauth_provider is None:
        from starlette.responses import HTMLResponse

        return HTMLResponse("<h1>OAuth mode is not enabled</h1>", status_code=404)
    return await handle_approval_request(_oauth_provider, request)


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

    if _http_auth_mode() == _AUTH_MODE_OAUTH:
        _configure_oauth_auth()
        return mcp.streamable_http_app()

    _require_http_auth_token()
    return apply_auth_middleware(mcp.streamable_http_app())


def _http_auth_mode() -> str:
    mode = os.environ.get("ATLAS_MCP_INVOICING_READONLY_AUTH_MODE", _AUTH_MODE_BEARER).strip().lower()
    if mode not in {_AUTH_MODE_BEARER, _AUTH_MODE_OAUTH}:
        raise RuntimeError(
            "ATLAS_MCP_INVOICING_READONLY_AUTH_MODE must be either 'bearer' or 'oauth'"
        )
    return mode


def _require_http_auth_token() -> str:
    """Return a production-shaped auth token or fail before serving HTTP."""
    token = os.environ.get("ATLAS_MCP_AUTH_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "ATLAS_MCP_AUTH_TOKEN is required for read-only invoicing HTTP mode; "
            "these tools expose customer financial data."
        )
    if token.lower() in _PLACEHOLDER_HTTP_AUTH_TOKENS or token.startswith("<"):
        raise RuntimeError(
            "ATLAS_MCP_AUTH_TOKEN must not be a placeholder value for read-only "
            "invoicing HTTP mode."
        )
    if len(token) < _MIN_HTTP_AUTH_TOKEN_LENGTH:
        raise RuntimeError(
            "ATLAS_MCP_AUTH_TOKEN must be at least "
            f"{_MIN_HTTP_AUTH_TOKEN_LENGTH} characters for read-only invoicing HTTP mode."
        )
    return token


def _configure_oauth_auth() -> InvoicingReadonlyOAuthProvider:
    """Configure FastMCP's built-in OAuth auth for ChatGPT connectors."""
    global _oauth_provider

    if _oauth_provider is not None:
        return _oauth_provider

    from mcp.server.auth.provider import ProviderTokenVerifier
    from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions

    issuer_url = os.environ.get("ATLAS_MCP_INVOICING_READONLY_OAUTH_ISSUER_URL", "").strip()
    resource_url = os.environ.get("ATLAS_MCP_INVOICING_READONLY_OAUTH_RESOURCE_URL", "").strip()
    approval_token = os.environ.get("ATLAS_MCP_INVOICING_READONLY_OAUTH_APPROVAL_TOKEN", "").strip()
    validate_oauth_settings(
        issuer_url=issuer_url,
        resource_server_url=resource_url,
        approval_token=approval_token,
    )

    provider = InvoicingReadonlyOAuthProvider(
        issuer_url=issuer_url,
        approval_token=approval_token,
        scopes=[DEFAULT_READONLY_SCOPE],
    )
    mcp.settings.auth = AuthSettings(
        issuer_url=as_any_http_url(issuer_url),
        resource_server_url=as_any_http_url(resource_url),
        required_scopes=[DEFAULT_READONLY_SCOPE],
        client_registration_options=ClientRegistrationOptions(
            enabled=True,
            valid_scopes=[DEFAULT_READONLY_SCOPE],
            default_scopes=[DEFAULT_READONLY_SCOPE],
        ),
    )
    mcp._auth_server_provider = provider
    mcp._token_verifier = ProviderTokenVerifier(provider)
    _oauth_provider = provider
    return provider


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
