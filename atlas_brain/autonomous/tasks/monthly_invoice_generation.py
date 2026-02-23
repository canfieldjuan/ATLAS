"""
Monthly invoice generation -- autonomous task.

On the 1st of each month, pulls the prior month's calendar events,
matches them to customer service agreements via calendar_keyword,
builds invoices with per-visit line items, and optionally sends them.
"""

import logging
from calendar import monthrange
from datetime import date, datetime, timedelta, timezone
from uuid import UUID

from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.monthly_invoice_generation")


async def run(task: ScheduledTask) -> dict:
    """Generate monthly invoices from calendar events matched to service agreements.

    Returns dict for ntfy notification, or _skip_synthesis when nothing to do.
    """
    from ...config import settings

    if not settings.invoicing.enabled:
        return {"_skip_synthesis": "Invoicing disabled"}

    if not settings.invoicing.auto_invoice_enabled:
        return {"_skip_synthesis": "Auto-invoicing disabled"}

    from ...services.calendar_provider import get_calendar_provider
    from ...storage.repositories.customer_service import get_customer_service_repo
    from ...storage.repositories.invoice import get_invoice_repo
    from ...services.crm_provider import get_crm_provider

    cal = get_calendar_provider()
    svc_repo = get_customer_service_repo()
    inv_repo = get_invoice_repo()
    crm = get_crm_provider()

    # Compute billing period: prior calendar month
    today = date.today()
    if today.month == 1:
        period_year = today.year - 1
        period_month = 12
    else:
        period_year = today.year
        period_month = today.month - 1

    _, last_day = monthrange(period_year, period_month)
    period_start = datetime(period_year, period_month, 1, tzinfo=timezone.utc)
    period_end = datetime(period_year, period_month, last_day, 23, 59, 59, tzinfo=timezone.utc)
    period_label = f"{period_year}-{period_month:02d}"

    auto_send = settings.invoicing.auto_invoice_send_email
    due_days = settings.invoicing.auto_invoice_due_days

    # Load active auto-invoice services
    try:
        services = await svc_repo.list_active(auto_invoice_only=True)
    except Exception as e:
        logger.error("Failed to load services: %s", e)
        return {"_skip_synthesis": f"Error loading services: {e}"}

    if not services:
        return {"_skip_synthesis": "No active auto-invoice services"}

    # Pull all calendar events for the billing period
    try:
        events = await cal.list_events(period_start, period_end)
    except Exception as e:
        logger.error("Failed to fetch calendar events: %s", e)
        return {"_skip_synthesis": f"Error fetching calendar: {e}"}

    # Filter to confirmed only
    confirmed_events = [e for e in events if e.status == "confirmed"]

    results = {
        "period": period_label,
        "services_checked": len(services),
        "invoices_created": 0,
        "invoices_sent": 0,
        "invoices_skipped_dedup": 0,
        "invoices_skipped_no_events": 0,
        "total_amount": 0.0,
        "details": [],
    }

    for svc in services:
        svc_id = svc["id"]
        keyword = svc["calendar_keyword"].lower()
        source_ref = f"{svc_id}_{period_label}"

        # Match events by keyword in summary
        matching = [
            e for e in confirmed_events
            if keyword in e.summary.lower()
        ]

        if not matching:
            results["invoices_skipped_no_events"] += 1
            logger.debug(
                "Service %s (%s): no matching events for keyword '%s'",
                svc_id, svc["service_name"], svc["calendar_keyword"],
            )
            continue

        # Dedup: check if invoice already exists for this service+period
        try:
            existing = await inv_repo.get_by_source_ref(source_ref)
            if existing:
                results["invoices_skipped_dedup"] += 1
                logger.info(
                    "Service %s: invoice for %s already exists (%s), skipping",
                    svc_id, period_label, existing["invoice_number"],
                )
                continue
        except Exception as e:
            logger.warning("Dedup check failed for %s: %s", svc_id, e)

        # Look up contact details via CRM get_contact (direct ID fetch)
        contact = None
        try:
            contact = await crm.get_contact(str(svc["contact_id"]))
        except Exception as e:
            logger.warning("Contact lookup failed for %s: %s", svc["contact_id"], e)

        customer_name = (contact or {}).get("full_name", "Customer")
        customer_email = (contact or {}).get("email")
        customer_phone = (contact or {}).get("phone")
        customer_address = (contact or {}).get("address")

        # Build line items: one per matching event
        rate = float(svc["rate"])
        line_items = []
        for event in sorted(matching, key=lambda e: e.start):
            event_date = event.start.date() if isinstance(event.start, datetime) else event.start
            line_items.append({
                "description": svc["service_name"],
                "date": event_date.isoformat(),
                "quantity": 1,
                "unit_price": rate,
                "rate_label": svc.get("rate_label", "Per Visit"),
            })

        # Create the invoice
        due_date = date.today() + timedelta(days=due_days)
        invoice_for = f"{svc['service_name']} - {_month_name(period_month)} {period_year}"
        tax_rate = float(svc.get("tax_rate", 0))

        try:
            invoice = await inv_repo.create(
                customer_name=customer_name,
                due_date=due_date,
                line_items=line_items,
                contact_id=UUID(str(svc["contact_id"])),
                customer_email=customer_email,
                customer_phone=customer_phone,
                customer_address=customer_address,
                tax_rate=tax_rate,
                invoice_for=invoice_for,
                source="monthly_auto",
                source_ref=source_ref,
                business_context_id=svc.get("business_context_id"),
                notes=f"Auto-generated for {period_label}. {len(matching)} visit(s).",
            )
        except Exception as e:
            logger.error("Failed to create invoice for service %s: %s", svc_id, e)
            results["details"].append({
                "service": svc["service_name"],
                "error": str(e),
            })
            continue

        results["invoices_created"] += 1
        results["total_amount"] += float(invoice.get("total_amount", 0))

        detail = {
            "service": svc["service_name"],
            "customer": customer_name,
            "visits": len(matching),
            "invoice_number": invoice["invoice_number"],
            "total": float(invoice.get("total_amount", 0)),
        }

        # Send email if configured
        if auto_send and customer_email:
            try:
                from ...services.email_provider import get_email_provider
                from ...templates.email.invoice import render_invoice_html, render_invoice_text

                email_provider = get_email_provider()
                html_body = render_invoice_html(invoice)
                text_body = render_invoice_text(invoice)

                await email_provider.send(
                    to=[customer_email],
                    subject=f"Invoice {invoice['invoice_number']} - {_money(invoice['total_amount'])}",
                    body=text_body,
                    html=html_body,
                )

                # Mark as sent
                now = datetime.now(timezone.utc)
                await inv_repo.update_status(
                    invoice["id"], "sent", sent_at=now, sent_via="email"
                )
                results["invoices_sent"] += 1
                detail["sent"] = True

                logger.info(
                    "Sent invoice %s to %s for %s",
                    invoice["invoice_number"], customer_email, svc["service_name"],
                )
            except Exception as e:
                logger.error(
                    "Failed to send invoice %s: %s",
                    invoice["invoice_number"], e,
                )
                detail["send_error"] = str(e)

        # Mark service as invoiced
        next_first = _next_month_first(period_year, period_month)
        try:
            await svc_repo.mark_invoiced(
                UUID(str(svc_id)),
                date(period_year, period_month, last_day),
                next_first,
            )
        except Exception as e:
            logger.warning("Failed to mark service %s invoiced: %s", svc_id, e)

        # Log CRM interaction
        try:
            await crm.log_interaction(
                contact_id=str(svc["contact_id"]),
                interaction_type="invoice",
                summary=(
                    f"Auto-invoice {invoice['invoice_number']} for {period_label}: "
                    f"{len(matching)} visit(s), {_money(invoice['total_amount'])}"
                ),
            )
        except Exception as e:
            logger.warning("CRM log failed for %s: %s", svc_id, e)

        results["details"].append(detail)

    results["total_amount"] = round(results["total_amount"], 2)

    if results["invoices_created"] == 0 and results["invoices_skipped_dedup"] == 0:
        return {"_skip_synthesis": f"No invoices generated for {period_label} (no matching events)"}

    logger.info(
        "Monthly invoicing for %s: %d created, %d sent, %d dedup-skipped, $%.2f total",
        period_label,
        results["invoices_created"],
        results["invoices_sent"],
        results["invoices_skipped_dedup"],
        results["total_amount"],
    )

    # Send ntfy notification directly (no synthesis_skill on this task)
    await _send_notification(results, task)

    return results


async def _send_notification(results: dict, task: ScheduledTask) -> None:
    """Send ntfy push notification with invoice generation summary."""
    try:
        from ...autonomous.config import autonomous_config
        from ...config import settings

        if not autonomous_config.notify_results or not settings.alerts.ntfy_enabled:
            return
        if (task.metadata or {}).get("notify") is False:
            return

        from ...tools.notify import notify_tool

        period = results["period"]
        created = results["invoices_created"]
        sent = results["invoices_sent"]
        total = results["total_amount"]

        lines = [f"Period: {period}", f"Invoices created: {created}"]
        if sent:
            lines.append(f"Sent via email: {sent}")
        lines.append(f"Total: {_money(total)}")

        for d in results.get("details", []):
            lines.append(f"  {d['customer']}: {d['invoice_number']} ({d['visits']} visits, {_money(d['total'])})")

        priority = (task.metadata or {}).get("notify_priority") or autonomous_config.notify_priority
        await notify_tool._send_notification(
            message="\n".join(lines),
            title="Atlas: Monthly Invoice Generation",
            priority=priority,
            tags="invoice,billing",
        )
        logger.info("Sent ntfy notification for monthly invoicing")
    except Exception:
        logger.warning("Failed to send ntfy notification", exc_info=True)


def _month_name(month: int) -> str:
    """Return full month name."""
    import calendar
    return calendar.month_name[month]


def _money(val) -> str:
    """Format as $X.XX."""
    try:
        return f"${float(val):,.2f}"
    except (TypeError, ValueError):
        return "$0.00"


def _next_month_first(year: int, month: int) -> date:
    """Return the 1st of the month after year/month."""
    if month == 12:
        return date(year + 1, 1, 1)
    return date(year, month + 1, 1)
