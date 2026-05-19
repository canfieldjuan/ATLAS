"""
Draft-only Atlas Invoicing MCP Server.

This server is intentionally separate from atlas_brain.mcp.invoicing_server.
It exposes a narrow write surface for remote connector clients that may create
or update draft invoices, but cannot send, approve, void, record payments,
mutate services, or export PDFs.

Run:
    python -m atlas_brain.mcp.invoicing_draft_writer_server
    ATLAS_MCP_AUTH_TOKEN=<token> python -m atlas_brain.mcp.invoicing_draft_writer_server --sse
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
from contextlib import asynccontextmanager
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from . import invoicing_server as _full
from .invoicing_draft_writer_oauth import (
    DEFAULT_DRAFT_WRITE_SCOPE,
    InvoicingDraftWriterOAuthProvider,
    as_any_http_url,
    handle_approval_request,
    validate_oauth_settings,
)
from ..config_defaults import DEFAULT_MCP_HOST

logger = logging.getLogger("atlas.mcp.invoicing.draft_writer")

DEFAULT_DRAFT_WRITER_PORT = 8066
_AUTH_MODE_BEARER = "bearer"
_AUTH_MODE_OAUTH = "oauth"
_MIN_HTTP_AUTH_TOKEN_LENGTH = 24
_PLACEHOLDER_HTTP_AUTH_TOKENS = {
    "<token>",
    "changeme",
    "change-me",
    "password",
    "secret",
    "test-token",
    "token",
}
_MAX_LINE_ITEMS = 100
_SOURCE = "chatgpt_draft_writer"
_SOURCE_REF_PREFIX = "chatgpt_draft_writer"
_oauth_provider: InvoicingDraftWriterOAuthProvider | None = None


@asynccontextmanager
async def _lifespan(server):
    """Initialize DB pool on startup, close on shutdown."""
    from ..storage.database import close_database, init_database

    await init_database()
    logger.info("Draft-writer invoicing MCP: DB pool initialized")
    yield
    await close_database()


mcp = FastMCP(
    "atlas-invoicing-draft-writer",
    instructions=(
        "Draft-only invoicing server for Atlas. Create and update draft "
        "invoices for operator review. This server cannot send, approve, "
        "void, record payments, mutate services, or export PDFs."
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


def _repo():
    from ..storage.repositories.invoice import get_invoice_repo

    return get_invoice_repo()


def _json_error(message: str) -> str:
    return json.dumps({"success": False, "error": message})


def _parse_line_items(line_items: str | list[dict[str, Any]]) -> tuple[list[dict[str, Any]] | None, str | None]:
    if isinstance(line_items, str):
        try:
            parsed = json.loads(line_items)
        except json.JSONDecodeError:
            return None, "Invalid line_items JSON"
    else:
        parsed = line_items

    if not isinstance(parsed, list):
        return None, "line_items must be a JSON array"
    if not parsed:
        return None, "At least one line item is required"
    if len(parsed) > _MAX_LINE_ITEMS:
        return None, f"Max {_MAX_LINE_ITEMS} line items per invoice"

    items: list[dict[str, Any]] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            return None, f"Line item {index + 1} must be an object"
        description = str(item.get("description") or "").strip()
        if not description:
            return None, f"Line item {index + 1} description is required"
        quantity, quantity_error = _coerce_non_negative_decimal(
            item.get("quantity", 1),
            f"Line item {index + 1} quantity",
        )
        if quantity_error:
            return None, quantity_error
        unit_price, price_error = _coerce_non_negative_decimal(
            item.get("unit_price", 0),
            f"Line item {index + 1} unit_price",
        )
        if price_error:
            return None, price_error
        amount = quantity * unit_price
        items.append(
            {
                **item,
                "description": description,
                "quantity": float(quantity),
                "unit_price": float(unit_price),
                "amount": float(amount),
            }
        )
    return items, None


def _coerce_non_negative_decimal(value: Any, label: str) -> tuple[Decimal, str | None]:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal("0"), f"{label} must be numeric"
    if not parsed.is_finite():
        return Decimal("0"), f"{label} must be finite"
    if parsed < 0:
        return Decimal("0"), f"{label} must be non-negative"
    return parsed, None


def _coerce_bounded_float(value: Any, label: str, lower: float, upper: float) -> tuple[float, str | None]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return lower, f"{label} must be numeric"
    if not math.isfinite(parsed):
        return lower, f"{label} must be finite"
    return max(lower, min(parsed, upper)), None


def _source_ref_for_idempotency_key(idempotency_key: str) -> str:
    digest = hashlib.sha256(idempotency_key.strip().encode("utf-8")).hexdigest()
    return f"{_SOURCE_REF_PREFIX}:{digest}"


def _draft_metadata(idempotency_key: str) -> dict[str, Any]:
    return {
        "mcp_connector": "invoicing_draft_writer",
        "operator_review_required": True,
        "created_by_remote_connector": True,
        "idempotency_key": idempotency_key,
    }


@mcp.tool()
async def create_draft_invoice(
    customer_name: str,
    line_items: str,
    idempotency_key: str,
    due_days: int = 30,
    customer_email: Optional[str] = None,
    customer_phone: Optional[str] = None,
    customer_address: Optional[str] = None,
    tax_rate: float = 0.0,
    discount_amount: float = 0.0,
    invoice_for: Optional[str] = None,
    contact_name: Optional[str] = None,
    notes: Optional[str] = None,
    business_context_id: Optional[str] = None,
) -> str:
    """Create an idempotent draft invoice for operator review."""
    customer = customer_name.strip() if customer_name else ""
    if not customer:
        return _json_error("customer_name is required")

    key = idempotency_key.strip() if idempotency_key else ""
    if not key:
        return _json_error("idempotency_key is required")

    items, item_error = _parse_line_items(line_items)
    if item_error:
        return _json_error(item_error)
    assert items is not None

    due_days = max(0, min(int(due_days), 365))
    tax_rate, tax_error = _coerce_bounded_float(tax_rate, "tax_rate", 0.0, 1.0)
    if tax_error:
        return _json_error(tax_error)
    discount_amount, discount_error = _coerce_bounded_float(
        discount_amount,
        "discount_amount",
        0.0,
        float("inf"),
    )
    if discount_error:
        return _json_error(discount_error)
    source_ref = _source_ref_for_idempotency_key(key)

    try:
        repo = _repo()
        existing = await repo.get_by_source_ref(source_ref)
        if existing is not None:
            return json.dumps(
                {"success": True, "created": False, "invoice": existing},
                default=str,
            )

        invoice = await repo.create(
            customer_name=customer,
            due_date=date.today() + timedelta(days=due_days),
            line_items=items,
            customer_email=customer_email,
            customer_phone=customer_phone,
            customer_address=customer_address,
            tax_rate=tax_rate,
            discount_amount=discount_amount,
            invoice_for=invoice_for,
            contact_name=contact_name,
            source=_SOURCE,
            source_ref=source_ref,
            business_context_id=business_context_id,
            notes=notes,
            metadata=_draft_metadata(key),
        )
        return json.dumps({"success": True, "created": True, "invoice": invoice}, default=str)
    except Exception:
        logger.exception("create_draft_invoice error")
        return _json_error("Internal error")


@mcp.tool()
async def update_draft_invoice(
    invoice_id: str,
    line_items: Optional[str] = None,
    due_date: Optional[str] = None,
    notes: Optional[str] = None,
    tax_rate: Optional[float] = None,
    discount_amount: Optional[float] = None,
    invoice_for: Optional[str] = None,
    contact_name: Optional[str] = None,
) -> str:
    """Update an existing draft invoice. Non-draft invoices are rejected."""
    try:
        repo = _repo()
        if _full._is_uuid(invoice_id):
            invoice = await repo.get_by_id(_full._uuid.UUID(invoice_id))
        else:
            invoice = await repo.get_by_number(invoice_id)
        if invoice is None:
            return _json_error("Invoice not found")
        if invoice.get("status") != "draft":
            return _json_error("Invoice is not in draft status")

        return await _full.update_invoice(
            invoice_id=invoice_id,
            line_items=line_items,
            due_date=due_date,
            notes=notes,
            tax_rate=tax_rate,
            discount_amount=discount_amount,
            invoice_for=invoice_for,
            contact_name=contact_name,
        )
    except Exception:
        logger.exception("update_draft_invoice error")
        return _json_error("Internal error")


@mcp.tool()
async def get_invoice(invoice_id: str) -> str:
    """Fetch an invoice by UUID or invoice number."""
    return await _full.get_invoice(invoice_id)


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


def _streamable_http_app():
    """Build the authenticated streamable HTTP app for draft-write tools."""
    from .auth import apply_auth_middleware

    if _http_auth_mode() == _AUTH_MODE_OAUTH:
        _configure_oauth_auth()
        return mcp.streamable_http_app()

    _require_http_auth_token()
    return apply_auth_middleware(mcp.streamable_http_app())


def _http_auth_mode() -> str:
    mode = os.environ.get(
        "ATLAS_MCP_INVOICING_DRAFT_WRITER_AUTH_MODE",
        _AUTH_MODE_BEARER,
    ).strip().lower()
    if mode not in {_AUTH_MODE_BEARER, _AUTH_MODE_OAUTH}:
        raise RuntimeError(
            "ATLAS_MCP_INVOICING_DRAFT_WRITER_AUTH_MODE must be either 'bearer' or 'oauth'"
        )
    return mode


def _require_http_auth_token() -> str:
    """Return a production-shaped auth token or fail before serving HTTP."""
    token = os.environ.get("ATLAS_MCP_AUTH_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "ATLAS_MCP_AUTH_TOKEN is required for draft-writer invoicing HTTP "
            "mode; these tools can create and update draft invoices."
        )
    if token.lower() in _PLACEHOLDER_HTTP_AUTH_TOKENS or token.startswith("<"):
        raise RuntimeError(
            "ATLAS_MCP_AUTH_TOKEN must not be a placeholder value for "
            "draft-writer invoicing HTTP mode."
        )
    if len(token) < _MIN_HTTP_AUTH_TOKEN_LENGTH:
        raise RuntimeError(
            "ATLAS_MCP_AUTH_TOKEN must be at least "
            f"{_MIN_HTTP_AUTH_TOKEN_LENGTH} characters for draft-writer "
            "invoicing HTTP mode."
        )
    return token


def _configure_oauth_auth() -> InvoicingDraftWriterOAuthProvider:
    """Configure FastMCP's built-in OAuth auth for ChatGPT connectors."""
    global _oauth_provider

    if _oauth_provider is not None:
        return _oauth_provider

    from mcp.server.auth.provider import ProviderTokenVerifier
    from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions

    issuer_url = os.environ.get("ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_ISSUER_URL", "").strip()
    resource_url = os.environ.get("ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_RESOURCE_URL", "").strip()
    approval_token = os.environ.get("ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_APPROVAL_TOKEN", "").strip()
    validate_oauth_settings(
        issuer_url=issuer_url,
        resource_server_url=resource_url,
        approval_token=approval_token,
    )

    provider = InvoicingDraftWriterOAuthProvider(
        issuer_url=issuer_url,
        approval_token=approval_token,
        scopes=[DEFAULT_DRAFT_WRITE_SCOPE],
    )
    mcp.settings.auth = AuthSettings(
        issuer_url=as_any_http_url(issuer_url),
        resource_server_url=as_any_http_url(resource_url),
        required_scopes=[DEFAULT_DRAFT_WRITE_SCOPE],
        client_registration_options=ClientRegistrationOptions(
            enabled=True,
            valid_scopes=[DEFAULT_DRAFT_WRITE_SCOPE],
            default_scopes=[DEFAULT_DRAFT_WRITE_SCOPE],
        ),
    )
    mcp._auth_server_provider = provider
    mcp._token_verifier = ProviderTokenVerifier(provider)
    _oauth_provider = provider
    return provider


if __name__ == "__main__":
    if "--sse" in sys.argv:
        import uvicorn

        host = os.environ.get("ATLAS_MCP_HOST", DEFAULT_MCP_HOST)
        port = int(os.environ.get("ATLAS_MCP_INVOICING_DRAFT_WRITER_PORT", DEFAULT_DRAFT_WRITER_PORT))
        uvicorn.run(_streamable_http_app(), host=host, port=port)
    else:
        mcp.run()
