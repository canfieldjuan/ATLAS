"""
Atlas Invoicing MCP Server.

Exposes invoice creation, payment tracking, and customer balance queries
to any MCP-compatible client (Claude Desktop, Cursor, custom agents).

Tools:
    create_invoice      -- create invoice with line items, contact lookup
    get_invoice         -- fetch by UUID or invoice number
    list_invoices       -- filter by status, contact_id, business_context_id
    update_invoice      -- edit draft invoices
    send_invoice        -- mark sent; optionally send via email
    record_payment      -- record manual payment, auto-update status
    mark_void           -- void/cancel an invoice with reason
    customer_balance    -- outstanding balance by contact_id/phone/email
    payment_history     -- payment behavior analytics
    search_invoices     -- full-text search across invoices

Run:
    python -m atlas_brain.mcp.invoicing_server          # stdio
    python -m atlas_brain.mcp.invoicing_server --sse    # SSE HTTP
"""

import json
import logging
import sys
import uuid as _uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("atlas.mcp.invoicing")


@asynccontextmanager
async def _lifespan(server):
    """Initialize DB pool on startup, close on shutdown."""
    from ..storage.database import init_database, close_database
    await init_database()
    logger.info("Invoicing MCP: DB pool initialized")
    yield
    await close_database()


mcp = FastMCP(
    "atlas-invoicing",
    instructions=(
        "Invoicing server for Atlas. "
        "Create invoices, record payments, track customer balances. "
        "Manual payment tracking only (no auto-charge). "
        "Invoice numbers follow INV-YYYY-NNNN format."
    ),
    lifespan=_lifespan,
)


def _repo():
    from ..storage.repositories.invoice import get_invoice_repo
    return get_invoice_repo()


def _crm():
    from ..services.crm_provider import get_crm_provider
    return get_crm_provider()


def _is_uuid(value: str) -> bool:
    try:
        _uuid.UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


async def _resolve_contact_id(
    contact_id: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
) -> Optional[str]:
    """Resolve a contact_id from phone or email if not provided directly."""
    if contact_id and _is_uuid(contact_id):
        return contact_id
    crm = _crm()
    try:
        if phone:
            results = await crm.search_contacts(phone=phone, limit=1)
            if results:
                return str(results[0].get("id"))
        if email:
            results = await crm.search_contacts(email=email, limit=1)
            if results:
                return str(results[0].get("id"))
    except Exception as e:
        logger.warning("Contact resolution failed: %s", e)
    return None


async def _log_crm(contact_id: Optional[str], interaction_type: str, summary: str):
    """Log a CRM interaction if contact_id is set."""
    if not contact_id:
        return
    try:
        await _crm().log_interaction(
            contact_id=contact_id,
            interaction_type=interaction_type,
            summary=summary,
        )
    except Exception as e:
        logger.warning("CRM interaction log failed: %s", e)


# ---------------------------------------------------------------------------
# Tool: create_invoice
# ---------------------------------------------------------------------------

@mcp.tool()
async def create_invoice(
    customer_name: str,
    line_items: str,
    due_days: int = 30,
    contact_id: Optional[str] = None,
    customer_email: Optional[str] = None,
    customer_phone: Optional[str] = None,
    customer_address: Optional[str] = None,
    tax_rate: float = 0.0,
    discount_amount: float = 0.0,
    invoice_for: Optional[str] = None,
    contact_name: Optional[str] = None,
    notes: Optional[str] = None,
    source: str = "mcp_tool",
    business_context_id: Optional[str] = None,
) -> str:
    """
    Create a new invoice.

    line_items: JSON array of [{description, quantity, unit_price}]
    due_days: days from today until due (default 30)
    contact_id: UUID of CRM contact (auto-resolved from phone/email if omitted)
    tax_rate: decimal tax rate (e.g. 0.0825 for 8.25%)
    invoice_for: description of what the invoice covers (e.g. "Office Cleaning - January 2026")
    contact_name: name of the contact person at the customer
    """
    if not customer_name or not customer_name.strip():
        return json.dumps({"success": False, "error": "customer_name is required"})

    try:
        items = json.loads(line_items) if isinstance(line_items, str) else line_items
    except json.JSONDecodeError:
        return json.dumps({"success": False, "error": "Invalid line_items JSON"})

    if isinstance(items, list) and len(items) > 100:
        return json.dumps({"success": False, "error": "Max 100 line items per invoice"})

    # Resolve contact
    resolved_contact = await _resolve_contact_id(contact_id, customer_phone, customer_email)
    cid = _uuid.UUID(resolved_contact) if resolved_contact else None

    due_days = max(0, min(due_days, 365))
    tax_rate = max(0.0, min(tax_rate, 1.0))
    discount_amount = max(0.0, discount_amount)
    due = date.today() + timedelta(days=due_days)

    try:
        invoice = await _repo().create(
            customer_name=customer_name,
            due_date=due,
            line_items=items,
            contact_id=cid,
            customer_email=customer_email,
            customer_phone=customer_phone,
            customer_address=customer_address,
            tax_rate=tax_rate,
            discount_amount=discount_amount,
            invoice_for=invoice_for,
            contact_name=contact_name,
            source=source,
            business_context_id=business_context_id,
            notes=notes,
        )

        # CRM log
        await _log_crm(
            resolved_contact, "invoice",
            f"Invoice {invoice['invoice_number']} created for ${invoice['total_amount']:.2f}",
        )

        return json.dumps({"success": True, "invoice": invoice}, default=str)
    except Exception as exc:
        logger.exception("create_invoice error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: get_invoice
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_invoice(invoice_id: str) -> str:
    """
    Fetch an invoice by UUID or invoice number (e.g. INV-2026-0001).
    """
    try:
        repo = _repo()
        if _is_uuid(invoice_id):
            inv = await repo.get_by_id(_uuid.UUID(invoice_id))
        else:
            inv = await repo.get_by_number(invoice_id)

        if inv is None:
            return json.dumps({"found": False, "invoice": None})

        payments = await repo.get_payments(inv["id"])
        inv["payments"] = payments
        return json.dumps({"found": True, "invoice": inv}, default=str)
    except Exception as exc:
        logger.exception("get_invoice error")
        return json.dumps({"error": "Internal error", "found": False, "invoice": None})


# ---------------------------------------------------------------------------
# Tool: list_invoices
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_invoices(
    status: Optional[str] = None,
    contact_id: Optional[str] = None,
    business_context_id: Optional[str] = None,
    limit: int = 50,
) -> str:
    """
    List invoices with optional filters.

    status: draft | sent | partial | paid | overdue | void
    """
    try:
        repo = _repo()
        cid = _uuid.UUID(contact_id) if contact_id and _is_uuid(contact_id) else None

        invoices = await repo.search(
            contact_id=cid,
            status=status,
            limit=min(limit, 200),
        )

        if business_context_id:
            invoices = [i for i in invoices if i.get("business_context_id") == business_context_id]

        return json.dumps({"invoices": invoices, "count": len(invoices)}, default=str)
    except Exception as exc:
        logger.exception("list_invoices error")
        return json.dumps({"error": "Internal error", "invoices": [], "count": 0})


# ---------------------------------------------------------------------------
# Tool: update_invoice
# ---------------------------------------------------------------------------

@mcp.tool()
async def update_invoice(
    invoice_id: str,
    line_items: Optional[str] = None,
    due_date: Optional[str] = None,
    notes: Optional[str] = None,
    tax_rate: Optional[float] = None,
    discount_amount: Optional[float] = None,
    invoice_for: Optional[str] = None,
    contact_name: Optional[str] = None,
) -> str:
    """
    Edit a draft invoice. Only draft invoices can be modified.

    line_items: JSON array of [{description, quantity, unit_price}]
    due_date: ISO date string (YYYY-MM-DD)
    invoice_for: description of what the invoice covers
    contact_name: name of the contact person at the customer
    """
    try:
        items = json.loads(line_items) if line_items else None
    except json.JSONDecodeError:
        return json.dumps({"success": False, "error": "Invalid line_items JSON"})

    if isinstance(items, list) and len(items) > 100:
        return json.dumps({"success": False, "error": "Max 100 line items per invoice"})

    dd = date.fromisoformat(due_date) if due_date else None
    if tax_rate is not None:
        tax_rate = max(0.0, min(tax_rate, 1.0))
    if discount_amount is not None:
        discount_amount = max(0.0, discount_amount)

    try:
        repo = _repo()
        if _is_uuid(invoice_id):
            iid = _uuid.UUID(invoice_id)
        else:
            inv = await repo.get_by_number(invoice_id)
            if not inv:
                return json.dumps({"success": False, "error": "Invoice not found"})
            iid = inv["id"]

        updated = await repo.update_invoice(
            invoice_id=iid,
            line_items=items,
            due_date=dd,
            notes=notes,
            tax_rate=tax_rate,
            discount_amount=discount_amount,
            invoice_for=invoice_for,
            contact_name=contact_name,
        )
        if updated is None:
            return json.dumps({"success": False, "error": "Invoice not found or not in draft status"})
        return json.dumps({"success": True, "invoice": updated}, default=str)
    except Exception as exc:
        logger.exception("update_invoice error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: send_invoice
# ---------------------------------------------------------------------------

@mcp.tool()
async def send_invoice(
    invoice_id: str,
    method: str = "manual",
) -> str:
    """
    Mark an invoice as sent.

    method: 'manual' (just mark sent) or 'email' (send via email provider)
    """
    try:
        repo = _repo()
        iid = _uuid.UUID(invoice_id) if _is_uuid(invoice_id) else None

        if not iid:
            inv = await repo.get_by_number(invoice_id)
            if not inv:
                return json.dumps({"success": False, "error": "Invoice not found"})
            iid = inv["id"]
        else:
            inv = await repo.get_by_id(iid)
            if not inv:
                return json.dumps({"success": False, "error": "Invoice not found"})

        now = datetime.now(timezone.utc)
        await repo.update_status(iid, "sent", sent_at=now, sent_via=method)

        # Send email if requested
        if method == "email" and inv.get("customer_email"):
            try:
                from ..services.email_provider import get_email_provider
                from ..templates.email.invoice import render_invoice_html, render_invoice_text
                email_provider = get_email_provider()

                html_body = render_invoice_html(inv)
                text_body = render_invoice_text(inv)

                await email_provider.send(
                    to=[inv["customer_email"]],
                    subject=f"Invoice {inv['invoice_number']} - ${inv['total_amount']:.2f}",
                    body=text_body,
                    html=html_body,
                )
                logger.info("Invoice email sent to %s for %s", inv["customer_email"], inv["invoice_number"])
            except Exception as e:
                logger.error("Failed to send invoice email: %s", e)
                return json.dumps({
                    "success": True, "status": "sent",
                    "email_sent": False, "email_error": str(e),
                    "invoice_number": inv["invoice_number"],
                }, default=str)

        # CRM log
        contact_id = inv.get("contact_id")
        await _log_crm(
            str(contact_id) if contact_id else None, "invoice",
            f"Invoice {inv['invoice_number']} sent via {method}",
        )

        return json.dumps({
            "success": True, "status": "sent",
            "email_sent": method == "email",
            "invoice_number": inv["invoice_number"],
        }, default=str)
    except Exception as exc:
        logger.exception("send_invoice error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: record_payment
# ---------------------------------------------------------------------------

@mcp.tool()
async def record_payment(
    invoice_id: str,
    amount: float,
    payment_method: str = "other",
    reference: Optional[str] = None,
    notes: Optional[str] = None,
    payment_date: Optional[str] = None,
) -> str:
    """
    Record a manual payment on an invoice. Auto-updates status to partial/paid.

    payment_method: cash | check | card | zelle | venmo | other
    reference: check number, transaction ID, etc.
    payment_date: ISO date (YYYY-MM-DD), defaults to today
    """
    try:
        repo = _repo()
        # Resolve invoice by UUID or number
        if _is_uuid(invoice_id):
            iid = _uuid.UUID(invoice_id)
            inv = await repo.get_by_id(iid)
        else:
            inv = await repo.get_by_number(invoice_id)
            iid = inv["id"] if inv else None

        if not inv:
            return json.dumps({"success": False, "error": "Invoice not found"})

        pd = date.fromisoformat(payment_date) if payment_date else None

        payment = await repo.record_payment(
            invoice_id=iid,
            amount=amount,
            payment_method=payment_method,
            payment_date=pd,
            reference=reference,
            notes=notes,
        )

        # Refresh invoice to get updated status
        updated_inv = await repo.get_by_id(iid)

        # CRM log
        contact_id = inv.get("contact_id")
        ref_text = f" ({reference})" if reference else ""
        await _log_crm(
            str(contact_id) if contact_id else None, "invoice",
            f"Payment ${amount:.2f} on {inv['invoice_number']} via {payment_method}{ref_text}",
        )

        return json.dumps({
            "success": True,
            "payment": payment,
            "invoice_status": updated_inv["status"] if updated_inv else "unknown",
            "amount_due": updated_inv["amount_due"] if updated_inv else None,
        }, default=str)
    except Exception as exc:
        logger.exception("record_payment error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: mark_void
# ---------------------------------------------------------------------------

@mcp.tool()
async def mark_void(invoice_id: str, reason: str = "") -> str:
    """
    Void/cancel an invoice. Provide a reason for the void.
    """
    try:
        repo = _repo()
        if _is_uuid(invoice_id):
            iid = _uuid.UUID(invoice_id)
            inv = await repo.get_by_id(iid)
        else:
            inv = await repo.get_by_number(invoice_id)
            iid = inv["id"] if inv else None

        if not inv:
            return json.dumps({"success": False, "error": "Invoice not found"})

        now = datetime.now(timezone.utc)
        await repo.update_status(iid, "void", voided_at=now, void_reason=reason)

        # CRM log
        contact_id = inv.get("contact_id")
        await _log_crm(
            str(contact_id) if contact_id else None, "invoice",
            f"Invoice {inv['invoice_number']} voided: {reason}",
        )

        return json.dumps({"success": True, "invoice_number": inv["invoice_number"], "status": "void"})
    except Exception as exc:
        logger.exception("mark_void error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: customer_balance
# ---------------------------------------------------------------------------

@mcp.tool()
async def customer_balance(
    contact_id: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
) -> str:
    """
    Get outstanding balance for a customer.

    Provide contact_id (UUID), phone, or email to identify the customer.
    """
    resolved = await _resolve_contact_id(contact_id, phone, email)
    if not resolved:
        return json.dumps({"found": False, "error": "Customer not found"})

    try:
        balance = await _repo().get_customer_balance(_uuid.UUID(resolved))
        return json.dumps({"found": True, "balance": balance}, default=str)
    except Exception as exc:
        logger.exception("customer_balance error")
        return json.dumps({"error": "Internal error", "found": False, "balance": None})


# ---------------------------------------------------------------------------
# Tool: payment_history
# ---------------------------------------------------------------------------

@mcp.tool()
async def payment_history(
    contact_id: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
) -> str:
    """
    Get payment behavior analytics for a customer.

    Returns: total invoices, on-time count, late count, avg days to pay,
    outstanding balance.
    """
    resolved = await _resolve_contact_id(contact_id, phone, email)
    if not resolved:
        return json.dumps({"found": False, "error": "Customer not found"})

    try:
        behavior = await _repo().get_payment_behavior(_uuid.UUID(resolved))
        return json.dumps({"found": True, "behavior": behavior}, default=str)
    except Exception as exc:
        logger.exception("payment_history error")
        return json.dumps({"error": "Internal error", "found": False, "behavior": None})


# ---------------------------------------------------------------------------
# Tool: create_service
# ---------------------------------------------------------------------------

@mcp.tool()
async def create_service(
    service_name: str,
    rate: float,
    calendar_keyword: str,
    contact_id: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    rate_label: str = "Per Visit",
    tax_rate: float = 0.0,
    calendar_id: Optional[str] = None,
    start_date: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    """
    Create a recurring service agreement for a customer.

    Links a CRM contact to a calendar keyword for monthly auto-invoicing.
    The keyword is matched against calendar event summaries (case-insensitive).

    contact_id: UUID of CRM contact (auto-resolved from phone/email if omitted)
    rate: per-visit charge
    calendar_keyword: substring to match in calendar event summaries (e.g. "Smith")
    calendar_id: specific Google Calendar ID (omit for primary calendar)
    start_date: ISO date (YYYY-MM-DD), defaults to today
    """
    if not service_name or not service_name.strip():
        return json.dumps({"success": False, "error": "service_name is required"})
    if not calendar_keyword or not calendar_keyword.strip():
        return json.dumps({"success": False, "error": "calendar_keyword is required"})

    resolved = await _resolve_contact_id(contact_id, phone, email)
    if not resolved:
        return json.dumps({"success": False, "error": "Customer not found. Provide contact_id, phone, or email."})

    from ..storage.repositories.customer_service import get_customer_service_repo
    repo = get_customer_service_repo()

    sd = date.fromisoformat(start_date) if start_date else None

    try:
        service = await repo.create(
            contact_id=_uuid.UUID(resolved),
            service_name=service_name,
            rate=rate,
            calendar_keyword=calendar_keyword,
            rate_label=rate_label,
            tax_rate=tax_rate,
            calendar_id=calendar_id,
            start_date=sd,
            notes=notes,
        )

        await _log_crm(
            resolved, "service",
            f"Service agreement created: {service_name} @ ${rate:.2f}/{rate_label}",
        )

        return json.dumps({"success": True, "service": service}, default=str)
    except Exception as exc:
        logger.exception("create_service error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: list_services
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_services(
    contact_id: Optional[str] = None,
    status: Optional[str] = None,
) -> str:
    """
    List customer service agreements.

    contact_id: filter by CRM contact UUID
    status: filter by status (active, paused, cancelled). Default: active only.
    """
    from ..storage.repositories.customer_service import get_customer_service_repo
    repo = get_customer_service_repo()

    try:
        if contact_id and _is_uuid(contact_id):
            services = await repo.get_by_contact(_uuid.UUID(contact_id))
            if status:
                services = [s for s in services if s["status"] == status]
        else:
            services = await repo.list_active()
            if status and status != "active":
                services = await repo.list_by_status(status)

        return json.dumps({"services": services, "count": len(services)}, default=str)
    except Exception as exc:
        logger.exception("list_services error")
        return json.dumps({"error": "Internal error", "services": [], "count": 0})


# ---------------------------------------------------------------------------
# Tool: get_service
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_service(service_id: str) -> str:
    """
    Get a customer service agreement by UUID.
    """
    if not _is_uuid(service_id):
        return json.dumps({"found": False, "error": "Invalid UUID"})

    from ..storage.repositories.customer_service import get_customer_service_repo
    repo = get_customer_service_repo()

    try:
        svc = await repo.get_by_id(_uuid.UUID(service_id))
        if svc is None:
            return json.dumps({"found": False, "service": None})
        return json.dumps({"found": True, "service": svc}, default=str)
    except Exception as exc:
        logger.exception("get_service error")
        return json.dumps({"error": "Internal error", "found": False})


# ---------------------------------------------------------------------------
# Tool: update_service
# ---------------------------------------------------------------------------

@mcp.tool()
async def update_service(
    service_id: str,
    service_name: Optional[str] = None,
    rate: Optional[float] = None,
    calendar_keyword: Optional[str] = None,
    rate_label: Optional[str] = None,
    tax_rate: Optional[float] = None,
    notes: Optional[str] = None,
) -> str:
    """
    Update a customer service agreement. Only active/paused services can be edited.
    """
    if not _is_uuid(service_id):
        return json.dumps({"success": False, "error": "Invalid UUID"})

    from ..storage.repositories.customer_service import get_customer_service_repo
    repo = get_customer_service_repo()

    fields = {}
    if service_name is not None:
        fields["service_name"] = service_name
    if rate is not None:
        fields["rate"] = rate
    if calendar_keyword is not None:
        fields["calendar_keyword"] = calendar_keyword
    if rate_label is not None:
        fields["rate_label"] = rate_label
    if tax_rate is not None:
        fields["tax_rate"] = tax_rate
    if notes is not None:
        fields["notes"] = notes

    if not fields:
        return json.dumps({"success": False, "error": "No fields to update"})

    try:
        updated = await repo.update(_uuid.UUID(service_id), **fields)
        if updated is None:
            return json.dumps({"success": False, "error": "Service not found or not editable (must be active/paused)"})

        # CRM log
        contact_id = updated.get("contact_id")
        changes = ", ".join(f"{k}={v}" for k, v in fields.items())
        await _log_crm(
            str(contact_id) if contact_id else None, "service",
            f"Service {updated['service_name']} updated: {changes}",
        )

        return json.dumps({"success": True, "service": updated}, default=str)
    except Exception as exc:
        logger.exception("update_service error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: set_service_status
# ---------------------------------------------------------------------------

@mcp.tool()
async def set_service_status(
    service_id: str,
    status: str,
) -> str:
    """
    Set service agreement status.

    status: active | paused | cancelled
    """
    if not _is_uuid(service_id):
        return json.dumps({"success": False, "error": "Invalid UUID"})

    if status not in ("active", "paused", "cancelled"):
        return json.dumps({"success": False, "error": "Status must be active, paused, or cancelled"})

    from ..storage.repositories.customer_service import get_customer_service_repo
    repo = get_customer_service_repo()

    try:
        svc = await repo.get_by_id(_uuid.UUID(service_id))
        if svc is None:
            return json.dumps({"success": False, "error": "Service not found"})

        await repo.update_status(_uuid.UUID(service_id), status)

        # CRM log
        contact_id = svc.get("contact_id")
        await _log_crm(
            str(contact_id) if contact_id else None, "service",
            f"Service {svc['service_name']} status changed to {status}",
        )

        return json.dumps({"success": True, "service_id": service_id, "status": status})
    except Exception as exc:
        logger.exception("set_service_status error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: search_invoices
# ---------------------------------------------------------------------------

@mcp.tool()
async def search_invoices(
    keyword: Optional[str] = None,
    status: Optional[str] = None,
    contact_id: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: int = 50,
) -> str:
    """
    Search invoices by keyword, status, contact, or date range.

    keyword: searches invoice number, customer name, notes
    from_date / to_date: ISO date (YYYY-MM-DD) for issue_date range
    """
    try:
        cid = _uuid.UUID(contact_id) if contact_id and _is_uuid(contact_id) else None
        fd = date.fromisoformat(from_date) if from_date else None
        td = date.fromisoformat(to_date) if to_date else None

        results = await _repo().search(
            keyword=keyword,
            contact_id=cid,
            status=status,
            from_date=fd,
            to_date=td,
            limit=min(limit, 200),
        )
        return json.dumps({"invoices": results, "count": len(results)}, default=str)
    except Exception as exc:
        logger.exception("search_invoices error")
        return json.dumps({"error": "Internal error", "invoices": [], "count": 0})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    transport = "sse" if "--sse" in sys.argv else "stdio"
    if transport == "sse":
        from ..config import settings
        from .auth import run_sse_with_auth

        mcp.settings.host = settings.mcp.host
        mcp.settings.port = settings.mcp.invoicing_port
        run_sse_with_auth(mcp, settings.mcp.host, settings.mcp.invoicing_port)
    else:
        mcp.run(transport="stdio")
