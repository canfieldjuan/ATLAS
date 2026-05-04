"""
Invoice payment reminder -- daily autonomous task.

Sends email reminders for overdue invoices, respecting max reminder count
and interval between reminders. Returns results for LLM synthesis.
"""

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.invoice_payment_reminders")


def _should_send_reminder(
    reminder_count: int,
    due_date: date,
    last_reminder_at: Optional[datetime],
    today: date,
    now: datetime,
    intervals: list[int],
    legacy_max_count: int,
    legacy_interval_days: int,
) -> tuple[bool, Optional[str]]:
    """Decide whether a reminder is due now.

    Returns (should_send, skip_reason). When intervals is non-empty, uses an
    escalating cadence: intervals[0] days from due_date -> reminder #1, then
    intervals[i] days from last_reminder_at -> reminder #(i+1). Total reminders
    capped at len(intervals).

    Empty intervals falls back to the legacy flat cadence (same gap between
    every reminder, capped at legacy_max_count).
    """
    if not intervals:
        if reminder_count >= legacy_max_count:
            return False, f"Max reminders ({legacy_max_count}) reached"
        if last_reminder_at:
            gap = now - last_reminder_at
            if gap < timedelta(days=legacy_interval_days):
                return False, f"Too soon (last: {last_reminder_at.strftime('%Y-%m-%d')})"
        return True, None

    if reminder_count >= len(intervals):
        return False, f"Max reminders ({len(intervals)}) reached"

    required_gap_days = intervals[reminder_count]

    if reminder_count == 0:
        days_since_due = (today - due_date).days
        if days_since_due < required_gap_days:
            wait = required_gap_days - days_since_due
            return False, f"Wait {wait} more day(s); first reminder due {required_gap_days}d after due_date"
        return True, None

    if not last_reminder_at:
        # reminder_count > 0 with no last_reminder_at is anomalous; allow send.
        return True, None
    gap = now - last_reminder_at
    if gap < timedelta(days=required_gap_days):
        wait = required_gap_days - gap.days
        return False, f"Wait {wait} more day(s) since last reminder"
    return True, None


async def run(task: ScheduledTask) -> dict:
    """Send payment reminders for overdue invoices.

    Respects invoicing.reminder_max_count and invoicing.reminder_interval_days.
    Returns dict for synthesis, or _skip_synthesis when no reminders needed.
    """
    from ...config import settings

    if not settings.invoicing.enabled:
        return {"_skip_synthesis": "Invoicing disabled"}

    cfg = settings.invoicing

    if not cfg.reminders_enabled:
        return {"_skip_synthesis": "Payment reminders disabled (reminders_enabled=False)"}

    from ...storage.repositories.invoice import get_invoice_repo

    repo = get_invoice_repo()

    try:
        overdue = await repo.get_overdue(as_of_date=date.today())
    except Exception as e:
        logger.error("Failed to query overdue invoices: %s", e)
        return {"_skip_synthesis": f"Error: {e}"}

    if not overdue:
        return {"_skip_synthesis": "No overdue invoices to remind"}

    today = date.today()
    now = datetime.now(timezone.utc)
    intervals = list(cfg.reminder_intervals or [])
    reminders_sent = []
    reminders_skipped = []

    for inv in overdue:
        should_send, skip_reason = _should_send_reminder(
            reminder_count=inv["reminder_count"],
            due_date=inv["due_date"],
            last_reminder_at=inv.get("last_reminder_at"),
            today=today,
            now=now,
            intervals=intervals,
            legacy_max_count=cfg.reminder_max_count,
            legacy_interval_days=cfg.reminder_interval_days,
        )
        if not should_send:
            reminders_skipped.append({
                "invoice_number": inv["invoice_number"],
                "reason": skip_reason or "skipped",
            })
            continue

        # Send email reminder if customer has email
        email_sent = False
        customer_email = inv.get("customer_email")
        if customer_email:
            try:
                import base64
                from ...services.email_provider import get_email_provider
                from ...services.invoice_pdf import render_invoice_pdf
                email_provider = get_email_provider()

                body = (
                    f"Payment Reminder - Invoice {inv['invoice_number']}\n\n"
                    f"Dear {inv['customer_name']},\n\n"
                    f"This is a friendly reminder that invoice {inv['invoice_number']} "
                    f"for ${inv['amount_due']:.2f} is past due.\n\n"
                    f"Original Due Date: {inv['due_date']}\n"
                    f"Amount Due: ${inv['amount_due']:.2f}\n\n"
                    f"A copy of the invoice is attached for your reference.\n\n"
                    f"Please arrange payment at your earliest convenience.\n\n"
                    f"Thank you."
                )

                # Attach invoice PDF; fall back to text-only if render fails so
                # the reminder still goes out (a missing nudge is worse than
                # a missing attachment).
                attachments = None
                try:
                    pdf_bytes = render_invoice_pdf(inv)
                    attachments = [{
                        "filename": f"{inv['invoice_number']}.pdf",
                        "content": base64.b64encode(pdf_bytes).decode("ascii"),
                    }]
                except Exception as pdf_err:
                    logger.warning(
                        "PDF render failed for reminder %s, sending text-only: %s",
                        inv["invoice_number"], pdf_err,
                    )
                    body = body.replace(
                        "A copy of the invoice is attached for your reference.\n\n",
                        "",
                    )

                await email_provider.send(
                    to=[customer_email],
                    subject=f"Payment Reminder: Invoice {inv['invoice_number']} - ${inv['amount_due']:.2f}",
                    body=body,
                    attachments=attachments,
                )
                email_sent = True
            except Exception as e:
                logger.error("Reminder email failed for %s: %s", inv["invoice_number"], e)

        # Update reminder tracking
        try:
            await repo.update_reminder(inv["id"])
        except Exception as e:
            logger.error("Failed to update reminder count for %s: %s", inv["invoice_number"], e)

        # CRM log
        contact_id = inv.get("contact_id")
        if contact_id:
            try:
                from ...services.crm_provider import get_crm_provider
                crm = get_crm_provider()
                await crm.log_interaction(
                    contact_id=str(contact_id),
                    interaction_type="invoice",
                    summary=f"Payment reminder #{inv['reminder_count'] + 1} for {inv['invoice_number']} (${inv['amount_due']:.2f})",
                )
            except Exception as e:
                logger.warning("CRM log failed: %s", e)

        reminders_sent.append({
            "invoice_number": inv["invoice_number"],
            "customer_name": inv["customer_name"],
            "amount_due": float(inv.get("amount_due", 0)),
            "reminder_number": inv["reminder_count"] + 1,
            "email_sent": email_sent,
        })

    if not reminders_sent:
        return {"_skip_synthesis": "No reminders needed (all at limit or too recent)"}

    result = {
        "reminders_sent": len(reminders_sent),
        "reminders_skipped": len(reminders_skipped),
        "details": reminders_sent,
    }

    logger.info(
        "Payment reminders: %d sent, %d skipped",
        len(reminders_sent), len(reminders_skipped),
    )
    return result
