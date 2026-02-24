"""
REST API for invoice action endpoints.

Provides endpoints triggered by ntfy action buttons:
- GET  /{invoice_id}              -- view invoice details
- POST /{invoice_id}/send         -- send invoice via email
- POST /{invoice_id}/send-reminder -- send payment reminder
- POST /{invoice_id}/mark-paid    -- quick-pay: record full amount
"""

import logging
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, HTTPException

logger = logging.getLogger("atlas.api.invoicing")

router = APIRouter(prefix="/invoicing", tags=["invoicing"])


async def _load_invoice(invoice_id: str) -> dict:
    """Load an invoice by UUID or invoice number. Raises 404 if not found."""
    from ...storage.repositories.invoice import get_invoice_repo

    repo = get_invoice_repo()
    try:
        uid = UUID(invoice_id)
        inv = await repo.get_by_id(uid)
    except (ValueError, AttributeError):
        inv = await repo.get_by_number(invoice_id)

    if not inv:
        raise HTTPException(404, f"Invoice {invoice_id} not found")
    return inv


# ---------------------------------------------------------------------------
# GET /invoicing/{invoice_id}  -- View Details
# ---------------------------------------------------------------------------

@router.get("/{invoice_id}")
async def view_invoice(invoice_id: str):
    """View invoice details including payments."""
    from ...storage.repositories.invoice import get_invoice_repo

    inv = await _load_invoice(invoice_id)
    payments = await get_invoice_repo().get_payments(inv["id"])

    return {
        "invoice": inv,
        "payments": payments,
    }


# ---------------------------------------------------------------------------
# POST /invoicing/{invoice_id}/send  -- Send Invoice
# ---------------------------------------------------------------------------

@router.post("/{invoice_id}/send")
async def send_invoice(invoice_id: str):
    """Send an invoice via email."""
    from ...storage.repositories.invoice import get_invoice_repo

    inv = await _load_invoice(invoice_id)
    repo = get_invoice_repo()

    if inv["status"] == "void":
        raise HTTPException(400, "Cannot send a voided invoice")

    # Mark as sent
    now = datetime.now(timezone.utc)
    await repo.update_status(inv["id"], "sent", sent_at=now, sent_via="email")

    # Send email if customer has email
    email_sent = False
    if inv.get("customer_email"):
        try:
            from ...services.email_provider import get_email_provider
            from ...templates.email.invoice import render_invoice_html, render_invoice_text
            email_provider = get_email_provider()

            html_body = render_invoice_html(inv)
            text_body = render_invoice_text(inv)

            await email_provider.send(
                to=[inv["customer_email"]],
                subject=f"Invoice {inv['invoice_number']} - ${inv['total_amount']:.2f}",
                body=text_body,
                html=html_body,
            )
            email_sent = True
        except Exception as e:
            logger.error("Invoice email send failed for %s: %s", inv["invoice_number"], e)

    # CRM log
    contact_id = inv.get("contact_id")
    if contact_id:
        try:
            from ...services.crm_provider import get_crm_provider
            crm = get_crm_provider()
            await crm.log_interaction(
                contact_id=str(contact_id),
                interaction_type="invoice",
                summary=f"Invoice {inv['invoice_number']} sent via email",
            )
        except Exception as e:
            logger.warning("CRM log failed: %s", e)

    logger.info("Invoice %s sent (email=%s)", inv["invoice_number"], email_sent)
    return {
        "action": "send",
        "invoice_number": inv["invoice_number"],
        "status": "sent",
        "email_sent": email_sent,
    }


# ---------------------------------------------------------------------------
# POST /invoicing/{invoice_id}/send-reminder  -- Send Reminder
# ---------------------------------------------------------------------------

@router.post("/{invoice_id}/send-reminder")
async def send_reminder(invoice_id: str):
    """Send a payment reminder for an overdue invoice."""
    from ...config import settings
    from ...storage.repositories.invoice import get_invoice_repo

    inv = await _load_invoice(invoice_id)
    repo = get_invoice_repo()

    if inv["status"] not in ("sent", "partial", "overdue"):
        raise HTTPException(400, f"Cannot send reminder for invoice with status '{inv['status']}'")

    # Check reminder limits
    max_reminders = settings.invoicing.reminder_max_count
    if inv["reminder_count"] >= max_reminders:
        raise HTTPException(400, f"Max reminders ({max_reminders}) already sent")

    # Send email reminder if customer has email
    email_sent = False
    if inv.get("customer_email"):
        try:
            from ...services.email_provider import get_email_provider
            from ...templates.email.invoice import render_invoice_html, render_invoice_text
            email_provider = get_email_provider()

            # Render the full invoice as HTML for the reminder attachment
            html_body = render_invoice_html(inv)
            text_body = render_invoice_text(inv)

            await email_provider.send(
                to=[inv["customer_email"]],
                subject=f"Payment Reminder: Invoice {inv['invoice_number']} - ${inv['amount_due']:.2f}",
                body=text_body,
                html=html_body,
            )
            email_sent = True
        except Exception as e:
            logger.error("Reminder email failed for %s: %s", inv["invoice_number"], e)

    # Update reminder count
    await repo.update_reminder(inv["id"])

    # CRM log
    contact_id = inv.get("contact_id")
    if contact_id:
        try:
            from ...services.crm_provider import get_crm_provider
            crm = get_crm_provider()
            await crm.log_interaction(
                contact_id=str(contact_id),
                interaction_type="invoice",
                summary=f"Payment reminder #{inv['reminder_count'] + 1} sent for {inv['invoice_number']} (${inv['amount_due']:.2f})",
            )
        except Exception as e:
            logger.warning("CRM log failed: %s", e)

    logger.info("Reminder sent for %s (count=%d)", inv["invoice_number"], inv["reminder_count"] + 1)
    return {
        "action": "send_reminder",
        "invoice_number": inv["invoice_number"],
        "reminder_count": inv["reminder_count"] + 1,
        "email_sent": email_sent,
    }


# ---------------------------------------------------------------------------
# POST /invoicing/{invoice_id}/mark-paid  -- Quick Pay
# ---------------------------------------------------------------------------

@router.post("/{invoice_id}/mark-paid")
async def mark_paid(invoice_id: str):
    """Quick-pay: record full remaining amount as paid."""
    from ...storage.repositories.invoice import get_invoice_repo

    inv = await _load_invoice(invoice_id)
    repo = get_invoice_repo()

    if inv["status"] in ("paid", "void"):
        raise HTTPException(400, f"Invoice already {inv['status']}")

    amount_due = float(inv["amount_due"])
    if amount_due <= 0:
        raise HTTPException(400, "No amount due")

    # Record payment for the full remaining amount
    payment = await repo.record_payment(
        invoice_id=inv["id"],
        amount=amount_due,
        payment_method="other",
        notes="Quick-pay via ntfy action",
    )

    # CRM log
    contact_id = inv.get("contact_id")
    if contact_id:
        try:
            from ...services.crm_provider import get_crm_provider
            crm = get_crm_provider()
            await crm.log_interaction(
                contact_id=str(contact_id),
                interaction_type="invoice",
                summary=f"Payment ${amount_due:.2f} on {inv['invoice_number']} (quick-pay)",
            )
        except Exception as e:
            logger.warning("CRM log failed: %s", e)

    logger.info("Invoice %s marked paid ($%.2f)", inv["invoice_number"], amount_due)
    return {
        "action": "mark_paid",
        "invoice_number": inv["invoice_number"],
        "amount_paid": amount_due,
        "status": "paid",
    }
