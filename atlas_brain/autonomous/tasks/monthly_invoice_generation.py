"""
Monthly invoice generation -- autonomous task.

On the 1st of each month, pulls the prior month's calendar events,
matches them to customer service agreements via calendar_keyword,
builds invoices with per-visit line items, and optionally sends them.

Supports:
- Per Visit: one line item per date, QTY = events-per-day
- Per Month: single flat-rate line item regardless of event count
- Per Hour: skipped (needs manual hours input)
- Review mode: creates drafts + PDFs without emailing
- PDF generation + save to disk + email attachment
- Custom tax_label for service charges (e.g., credit card surcharge)
"""

import base64
import logging
import os
from calendar import monthrange
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.monthly_invoice_generation")


async def run(task: ScheduledTask) -> dict:
    """Generate monthly invoices from calendar events matched to service agreements.

    Supports task.metadata overrides:
        billing_month: "YYYY-MM" to invoice a specific month instead of prior month
        contact_ids: ["uuid1", ...] to invoice only specific customers
    """
    from ...config import settings

    if not settings.invoicing.enabled:
        return {"_skip_synthesis": "Invoicing disabled"}

    if not settings.invoicing.auto_invoice_enabled:
        return {"_skip_synthesis": "Auto-invoicing disabled"}

    from ...services.calendar_provider import get_calendar_provider
    from ...services.crm_provider import get_crm_provider
    from ...storage.repositories.customer_service import get_customer_service_repo
    from ...storage.repositories.invoice import get_invoice_repo

    cal = get_calendar_provider()
    svc_repo = get_customer_service_repo()
    inv_repo = get_invoice_repo()
    crm = get_crm_provider()

    meta = task.metadata or {}

    # Compute billing period: metadata override or prior calendar month
    today = date.today()
    billing_month_override = meta.get("billing_month")
    if billing_month_override:
        try:
            parts = billing_month_override.split("-")
            period_year = int(parts[0])
            period_month = int(parts[1])
        except (ValueError, IndexError):
            return {"_skip_synthesis": f"Invalid billing_month format: {billing_month_override!r} (expected YYYY-MM)"}
    elif today.month == 1:
        period_year = today.year - 1
        period_month = 12
    else:
        period_year = today.year
        period_month = today.month - 1

    # Optional contact_ids filter
    contact_id_filter: set[str] | None = None
    raw_cids = meta.get("contact_ids")
    if raw_cids:
        contact_id_filter = {str(c) for c in raw_cids}

    _, last_day = monthrange(period_year, period_month)
    period_start = datetime(period_year, period_month, 1, tzinfo=timezone.utc)
    # Extend end into next day UTC to capture late-evening events in US timezones
    # (e.g., 9pm CDT on March 31 = 2am UTC April 1)
    period_end = datetime(period_year, period_month, last_day, tzinfo=timezone.utc) + timedelta(hours=30)
    period_label = f"{period_year}-{period_month:02d}"

    review_mode = settings.invoicing.auto_invoice_review_mode
    auto_send = settings.invoicing.auto_invoice_send_email and not review_mode
    due_days = settings.invoicing.auto_invoice_due_days
    calendar_id = settings.invoicing.auto_invoice_calendar_id or None
    save_base = os.path.expanduser(settings.invoicing.auto_invoice_save_path)

    # Load active auto-invoice services
    try:
        services = await svc_repo.list_active(auto_invoice_only=True)
    except Exception as e:
        logger.error("Failed to load services: %s", e)
        return {"_skip_synthesis": f"Error loading services: {e}"}

    if not services:
        return {"_skip_synthesis": "No active auto-invoice services"}

    # Pull all calendar events for the billing period from the commercial calendar
    try:
        events = await cal.list_events(period_start, period_end, calendar_id=calendar_id)
    except Exception as e:
        logger.error("Failed to fetch calendar events: %s", e)
        return {"_skip_synthesis": f"Error fetching calendar: {e}"}

    confirmed_events = [e for e in events if e.status == "confirmed"]

    results: dict[str, Any] = {
        "period": period_label,
        "review_mode": review_mode,
        "services_checked": len(services),
        "invoices_created": 0,
        "invoices_sent": 0,
        "invoices_skipped_dedup": 0,
        "invoices_skipped_no_events": 0,
        "needs_hours": [],
        "total_amount": 0.0,
        "details": [],
    }

    # -- Phase 1: Build per-customer bundles from service agreements ----------
    # Group services by contact_id so customers with multiple services
    # (e.g. Kinder Morgan St. Elmo + Altamont) get ONE combined invoice.

    from collections import defaultdict

    customer_bundles: dict[str, dict[str, Any]] = {}  # contact_id -> bundle

    for svc in services:
        svc_id = svc["id"]
        contact_id = str(svc["contact_id"])

        # Skip if contact_ids filter is set and this service isn't in it
        if contact_id_filter and contact_id not in contact_id_filter:
            continue

        keyword = svc["calendar_keyword"].lower()
        rate_label = svc.get("rate_label", "Per Visit")

        # Per Hour services need manual input -- skip and flag
        if rate_label == "Per Hour":
            results["needs_hours"].append({
                "service": svc["service_name"],
                "contact_id": contact_id,
                "rate": float(svc["rate"]),
                "keyword": svc["calendar_keyword"],
            })
            logger.info(
                "Service %s (%s): Per Hour rate, skipping -- needs manual hours",
                svc_id, svc["service_name"],
            )
            continue

        # Match events by keyword in summary, filtered to billing month dates
        billing_start = date(period_year, period_month, 1)
        billing_end = date(period_year, period_month, last_day)
        matching = [
            e for e in confirmed_events
            if keyword in e.summary.lower()
            and billing_start <= (e.start.date() if isinstance(e.start, datetime) else e.start) <= billing_end
        ]

        # Per Month services don't need events (flat rate)
        if not matching and rate_label != "Per Month":
            results["invoices_skipped_no_events"] += 1
            logger.debug(
                "Service %s (%s): no matching events for keyword '%s'",
                svc_id, svc["service_name"], svc["calendar_keyword"],
            )
            continue

        # Build line items for this service
        rate = float(svc["rate"])
        svc_line_items = []
        visit_count = 0

        if rate_label == "Per Month":
            svc_line_items.append({
                "date": date(period_year, period_month, 1).strftime("%m/%d/%Y"),
                "description": f"{_month_name(period_month)} {svc['service_name']}",
                "quantity": 1,
                "unit_price": rate,
            })
            visit_count = len(matching) if matching else 0
        else:
            day_counts: Counter[date] = Counter()
            for event in matching:
                event_date = event.start.date() if isinstance(event.start, datetime) else event.start
                day_counts[event_date] += 1
            for event_date in sorted(day_counts.keys()):
                qty = day_counts[event_date]
                svc_line_items.append({
                    "date": event_date.strftime("%m/%d/%Y"),
                    "description": svc["service_name"],
                    "quantity": qty,
                    "unit_price": rate,
                })
            visit_count = sum(day_counts.values())

        # Add to customer bundle
        if contact_id not in customer_bundles:
            customer_bundles[contact_id] = {
                "contact_id": contact_id,
                "service_ids": [],
                "line_items": [],
                "visit_count": 0,
                "service_names": [],
                "tax_rate": 0.0,
                "invoice_metadata": {},
            }
        bundle = customer_bundles[contact_id]
        bundle["service_ids"].append(str(svc_id))
        bundle["line_items"].extend(svc_line_items)
        bundle["visit_count"] += visit_count
        bundle["service_names"].append(svc["service_name"])

        # Use highest tax_rate across grouped services
        svc_tax = float(svc.get("tax_rate", 0))
        if svc_tax > bundle["tax_rate"]:
            bundle["tax_rate"] = svc_tax
            svc_notes = str(svc.get("notes") or "").lower()
            if "service charge" in svc_notes or "credit card" in svc_notes:
                bundle["invoice_metadata"]["tax_label"] = f"Service Charge ({svc_tax * 100:.1f}%)"

    # -- Phase 2: Create one invoice per customer bundle --------------------

    for contact_id, bundle in customer_bundles.items():
        line_items = bundle["line_items"]
        if not line_items:
            continue

        # Dedup by combined service IDs + period
        source_ref = f"{'_'.join(sorted(bundle['service_ids']))}_{period_label}"
        try:
            existing = await inv_repo.get_by_source_ref(source_ref)
            if existing:
                results["invoices_skipped_dedup"] += 1
                logger.info(
                    "Customer %s: invoice for %s already exists (%s), skipping",
                    contact_id, period_label, existing["invoice_number"],
                )
                continue
        except Exception as e:
            logger.warning("Dedup check failed for %s: %s", contact_id, e)

        # Look up contact details via CRM
        contact = None
        try:
            contact = await crm.get_contact(contact_id)
        except Exception as e:
            logger.warning("Contact lookup failed for %s: %s", contact_id, e)

        customer_name = (contact or {}).get("full_name", "Customer")
        customer_email = (contact or {}).get("email")
        customer_phone = (contact or {}).get("phone")
        customer_address = (contact or {}).get("address")

        # Sort line items by date
        line_items.sort(key=lambda li: li.get("date") or "")

        tax_rate = bundle["tax_rate"]
        invoice_metadata = bundle["invoice_metadata"]
        service_names = bundle["service_names"]
        visit_count = bundle["visit_count"]

        due_date = today + timedelta(days=due_days)
        if len(service_names) == 1:
            invoice_for = f"{service_names[0]} - {_month_name(period_month)} {period_year}"
        else:
            invoice_for = f"{', '.join(service_names)} - {_month_name(period_month)} {period_year}"

        try:
            invoice = await inv_repo.create(
                customer_name=customer_name,
                due_date=due_date,
                line_items=line_items,
                contact_id=UUID(contact_id),
                customer_email=customer_email,
                customer_phone=customer_phone,
                customer_address=customer_address,
                tax_rate=tax_rate,
                invoice_for=invoice_for,
                source="monthly_auto",
                source_ref=source_ref,
                notes=f"Auto-generated for {period_label}.",
                metadata=invoice_metadata if invoice_metadata else None,
            )
        except Exception as e:
            logger.error("Failed to create invoice for customer %s: %s", customer_name, e)
            results["details"].append({"customer": customer_name, "error": str(e)})
            continue

        results["invoices_created"] += 1
        results["total_amount"] += float(invoice.get("total_amount", 0))

        detail: dict[str, Any] = {
            "service": ", ".join(service_names),
            "customer": customer_name,
            "visits": visit_count,
            "invoice_number": invoice["invoice_number"],
            "total": float(invoice.get("total_amount", 0)),
        }

        # Generate PDF and save to disk
        pdf_saved = False
        pdf_bytes = b""
        pdf_filename = ""
        try:
            from ...services.invoice_pdf import render_invoice_pdf

            if invoice_metadata:
                invoice["metadata"] = invoice_metadata

            pdf_bytes = render_invoice_pdf(invoice)
            customer_folder = os.path.join(save_base, str(period_year), customer_name)
            os.makedirs(customer_folder, exist_ok=True)
            pdf_filename = f"{invoice['invoice_number']} - {_month_name(period_month)} {period_year}.pdf"
            pdf_path = os.path.join(customer_folder, pdf_filename)
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
            detail["pdf_path"] = pdf_path
            pdf_saved = True
            logger.info("Saved PDF: %s", pdf_path)
        except Exception as e:
            logger.error("Failed to generate PDF for %s: %s", invoice["invoice_number"], e)
            detail["pdf_error"] = str(e)

        # Send email with PDF attachment (only if not review mode)
        if auto_send and customer_email:
            try:
                from ...services.email_provider import get_email_provider
                from ...templates.email.invoice import (
                    BUSINESS_NAME, BUSINESS_PHONE, BUSINESS_EMAIL,
                )

                email_provider = get_email_provider()
                email_body = (
                    f"Please find attached invoice {invoice['invoice_number']} "
                    f"for {invoice_for}.\n\n"
                    f"Amount Due: {_money(invoice['total_amount'])}\n"
                    f"Due Date: {due_date.strftime('%m/%d/%Y')}\n\n"
                    f"Make all checks payable to {BUSINESS_NAME}.\n\n"
                    f"Thank you for your business!\n\n"
                    f"{BUSINESS_NAME}\n"
                    f"{BUSINESS_PHONE}\n"
                    f"{BUSINESS_EMAIL}"
                )

                attachments = []
                if pdf_saved and pdf_bytes:
                    attachments.append({
                        "filename": pdf_filename,
                        "content": base64.b64encode(pdf_bytes).decode("ascii"),
                    })

                await email_provider.send(
                    to=[customer_email],
                    subject=f"Invoice {invoice['invoice_number']} - {BUSINESS_NAME} - {_money(invoice['total_amount'])}",
                    body=email_body,
                    attachments=attachments if attachments else None,
                )

                now = datetime.now(timezone.utc)
                await inv_repo.update_status(
                    invoice["id"], "sent", sent_at=now, sent_via="email",
                )
                results["invoices_sent"] += 1
                detail["sent"] = True
                logger.info("Sent invoice %s to %s", invoice["invoice_number"], customer_email)
            except Exception as e:
                logger.error("Failed to send invoice %s: %s", invoice["invoice_number"], e)
                detail["send_error"] = str(e)

        # Mark all grouped services as invoiced
        next_first = _next_month_first(period_year, period_month)
        for svc_id_str in bundle["service_ids"]:
            try:
                await svc_repo.mark_invoiced(
                    UUID(svc_id_str),
                    date(period_year, period_month, last_day),
                    next_first,
                )
            except Exception as e:
                logger.warning("Failed to mark service %s invoiced: %s", svc_id_str, e)

        # Log CRM interaction
        try:
            await crm.log_interaction(
                contact_id=contact_id,
                interaction_type="invoice",
                summary=(
                    f"Auto-invoice {invoice['invoice_number']} for {period_label}: "
                    f"{visit_count} visit(s), {_money(invoice['total_amount'])}"
                ),
            )
        except Exception as e:
            logger.warning("CRM log failed for %s: %s", contact_id, e)

        results["details"].append(detail)

    results["total_amount"] = round(results["total_amount"], 2)

    if results["invoices_created"] == 0 and results["invoices_skipped_dedup"] == 0:
        if results["needs_hours"]:
            pass  # still notify about hourly services
        else:
            return {"_skip_synthesis": f"No invoices generated for {period_label}"}

    logger.info(
        "Monthly invoicing for %s: %d created, %d sent, %d dedup-skipped, %d needs-hours, $%.2f total",
        period_label,
        results["invoices_created"],
        results["invoices_sent"],
        results["invoices_skipped_dedup"],
        len(results["needs_hours"]),
        results["total_amount"],
    )

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
        review = results.get("review_mode", False)

        lines = [f"Period: {period}"]
        if review:
            lines.append(f"REVIEW MODE: {created} invoices ready for review (not sent)")
        else:
            lines.append(f"Invoices created: {created}")
            if sent:
                lines.append(f"Sent via email: {sent}")
        lines.append(f"Total: {_money(total)}")

        for d in results.get("details", []):
            status = "DRAFT" if review else ("SENT" if d.get("sent") else "created")
            lines.append(f"  {d['customer']}: {d['invoice_number']} ({d['visits']} visits, {_money(d['total'])}) [{status}]")

        needs_hours = results.get("needs_hours", [])
        if needs_hours:
            lines.append("")
            lines.append(f"NEEDS HOURS ({len(needs_hours)}):")
            for nh in needs_hours:
                lines.append(f"  {nh['service']} @ ${nh['rate']:.2f}/hr")

        priority = (task.metadata or {}).get("notify_priority") or autonomous_config.notify_priority
        await notify_tool._send_notification(
            message="\n".join(lines),
            title="Atlas: Monthly Invoice Generation",
            priority=priority,
            tags="invoice,billing",
        )
    except Exception:
        logger.warning("Failed to send ntfy notification", exc_info=True)


def _month_name(month: int) -> str:
    import calendar
    return calendar.month_name[month]


def _money(val) -> str:
    try:
        return f"${float(val):,.2f}"
    except (TypeError, ValueError):
        return "$0.00"


def _next_month_first(year: int, month: int) -> date:
    if month == 12:
        return date(year + 1, 1, 1)
    return date(year, month + 1, 1)
