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
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.monthly_invoice_generation")

# Placeholder values for Per Hour service drafts.
# Drafts are created with quantity=0 (subtotal=$0) and a clearly-flagged
# description; the user fills in actual hours via update_invoice before
# approve_and_send.
_PER_HOUR_PLACEHOLDER_DESC_SUFFIX = " (hours TBD - update before sending)"
_PER_HOUR_PLACEHOLDER_QTY = 0


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
        "skipped_no_events_details": [],
        "needs_hours": [],
        "keyword_collisions": [],
        "total_amount": 0.0,
        "details": [],
    }

    # -- Phase 1a: Assign each calendar event to exactly one service ----------
    # Substring matching can produce collisions (e.g. event "Smith Corp" matches
    # both a "Smith" and a "Smith Corp" service, double-counting the visit).
    # Resolve by longest matching keyword (most specific); record every multi-
    # match so the user can rename keywords after the fact.

    billing_start = date(period_year, period_month, 1)
    billing_end = date(period_year, period_month, last_day)
    month_events = [
        e for e in confirmed_events
        if billing_start <= (e.start.date() if isinstance(e.start, datetime) else e.start) <= billing_end
    ]

    # Per Hour services participate in event assignment too: if Brookstone
    # has events this month, those events should land on the Brookstone
    # service (and produce a placeholder draft) rather than be dropped.
    eligible_services = [
        s for s in services
        if not contact_id_filter or str(s["contact_id"]) in contact_id_filter
    ]

    events_by_service, collisions = _assign_events_to_services(month_events, eligible_services)
    results["keyword_collisions"] = collisions
    for c in collisions:
        logger.warning(
            "Keyword collision: '%s' on %s matched %d services (%s); assigned to '%s' via %s",
            c["event_summary"], c["event_date"], len(c["matched_services"]),
            [m["keyword"] for m in c["matched_services"]],
            c["assigned_to"], c["resolution"],
        )

    # -- Phase 1b: Build per-customer bundles from service agreements ---------
    # Group services by contact_id so customers with multiple services
    # (e.g. Kinder Morgan St. Elmo + Altamont) get ONE combined invoice.

    customer_bundles: dict[str, dict[str, Any]] = {}  # contact_id -> bundle

    for svc in services:
        svc_id = svc["id"]
        contact_id = str(svc["contact_id"])

        # Skip if contact_ids filter is set and this service isn't in it
        if contact_id_filter and contact_id not in contact_id_filter:
            continue

        rate_label = svc.get("rate_label", "Per Visit")

        # Per Hour services always get flagged in needs_hours so the user
        # sees them in the ntfy summary even if no events were logged.
        if rate_label == "Per Hour":
            results["needs_hours"].append({
                "service": svc["service_name"],
                "contact_id": contact_id,
                "rate": float(svc["rate"]),
                "keyword": svc["calendar_keyword"],
            })

        matching = events_by_service.get(str(svc_id), [])

        # Per Month services don't need events (flat rate). Per Hour and
        # Per Visit both require at least one matching event for the month.
        if not matching and rate_label != "Per Month":
            results["invoices_skipped_no_events"] += 1
            last_invoiced = svc.get("last_invoiced_at")
            results["skipped_no_events_details"].append({
                "service": svc["service_name"],
                "contact_id": contact_id,
                "keyword": svc["calendar_keyword"],
                "rate_label": rate_label,
                "last_invoiced_at": last_invoiced.isoformat() if hasattr(last_invoiced, "isoformat") else last_invoiced,
            })
            logger.debug(
                "Service %s (%s): no matching events for keyword '%s'",
                svc_id, svc["service_name"], svc["calendar_keyword"],
            )
            continue

        # Build line items for this service
        rate = float(svc["rate"])
        svc_line_items: list[dict] = []
        visit_count = 0

        if rate_label == "Per Month":
            svc_line_items.append({
                "date": date(period_year, period_month, 1).strftime("%m/%d/%Y"),
                "description": f"{_month_name(period_month)} {svc['service_name']}",
                "quantity": 1,
                "unit_price": rate,
            })
            visit_count = len(matching) if matching else 0
        elif rate_label == "Per Hour":
            svc_line_items, visit_count = _build_per_hour_line_items(
                matching, rate, svc["service_name"],
            )
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

        # Flag the whole invoice as needing hours when any contributing
        # service is Per Hour. Picked up by review tooling / list_pending_drafts.
        if rate_label == "Per Hour":
            bundle["invoice_metadata"]["needs_hours"] = True

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
            pdf_filename = f"{invoice['invoice_number']}.pdf"
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


def _build_notification_lines(results: dict) -> list[str]:
    """Build the ntfy message body for a monthly_invoice_generation result.

    Pure function so it can be unit-tested in isolation; _send_notification
    wraps this with the actual ntfy delivery + config gating.
    """
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

    collisions = results.get("keyword_collisions", [])
    if collisions:
        lines.append("")
        lines.append(f"KEYWORD COLLISIONS ({len(collisions)}) -- review service keywords:")
        for c in collisions:
            matched = ", ".join(m["keyword"] for m in c["matched_services"])
            lines.append(
                f"  '{c['event_summary']}' ({c['event_date']}) -> {c['assigned_to']} "
                f"[{c['resolution']}; matched: {matched}]"
            )

    skipped = results.get("skipped_no_events_details", [])
    if skipped:
        lines.append("")
        lines.append(f"NO EVENTS THIS MONTH ({len(skipped)}):")
        for s in skipped:
            last = s.get("last_invoiced_at")
            tail = f"last invoiced {last}" if last else "no prior invoices"
            lines.append(f"  {s['service']} ({tail})")

    return lines


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

        lines = _build_notification_lines(results)
        priority = (task.metadata or {}).get("notify_priority") or autonomous_config.notify_priority
        await notify_tool._send_notification(
            message="\n".join(lines),
            title="Atlas: Monthly Invoice Generation",
            priority=priority,
            tags="invoice,billing",
        )
    except Exception:
        logger.warning("Failed to send ntfy notification", exc_info=True)


def _build_per_hour_line_items(
    events: list,
    rate: float,
    service_name: str,
) -> tuple[list[dict], int]:
    """Build placeholder line items for a Per Hour service.

    One line item per event date, quantity=_PER_HOUR_PLACEHOLDER_QTY (0)
    so the draft renders at $0 until the user fills hours via update_invoice.
    Returns (line_items, visit_count).
    """
    day_counts: Counter[date] = Counter()
    for event in events:
        event_date = event.start.date() if isinstance(event.start, datetime) else event.start
        day_counts[event_date] += 1

    line_items: list[dict] = []
    for event_date in sorted(day_counts.keys()):
        line_items.append({
            "date": event_date.strftime("%m/%d/%Y"),
            "description": service_name + _PER_HOUR_PLACEHOLDER_DESC_SUFFIX,
            "quantity": _PER_HOUR_PLACEHOLDER_QTY,
            "unit_price": rate,
        })
    return line_items, sum(day_counts.values())


def _assign_events_to_services(
    events: list,
    eligible_services: list[dict],
) -> tuple[dict[str, list], list[dict]]:
    """Assign each event to exactly one service.

    For events whose summary contains multiple service keywords, picks the
    LONGEST keyword (most specific). Equal-length keywords break alphabetically
    to stay deterministic, and every multi-match is recorded as a collision.

    Returns (events_by_service_id, collisions).
    """
    events_by_service: dict[str, list] = defaultdict(list)
    collisions: list[dict] = []

    for event in events:
        summary_lower = event.summary.lower()
        matchers = [
            s for s in eligible_services
            if s["calendar_keyword"].lower() in summary_lower
        ]
        if not matchers:
            continue

        if len(matchers) == 1:
            events_by_service[str(matchers[0]["id"])].append(event)
            continue

        matchers.sort(key=lambda s: (-len(s["calendar_keyword"]), s["calendar_keyword"]))
        chosen = matchers[0]
        events_by_service[str(chosen["id"])].append(event)

        is_tie = len(matchers[0]["calendar_keyword"]) == len(matchers[1]["calendar_keyword"])
        event_date = event.start.date() if isinstance(event.start, datetime) else event.start
        collisions.append({
            "event_summary": event.summary,
            "event_date": event_date.isoformat() if hasattr(event_date, "isoformat") else str(event_date),
            "matched_services": [
                {"service": m["service_name"], "keyword": m["calendar_keyword"]}
                for m in matchers
            ],
            "assigned_to": chosen["service_name"],
            "resolution": "alphabetical_tiebreak" if is_tie else "longest_keyword",
        })

    return dict(events_by_service), collisions


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
