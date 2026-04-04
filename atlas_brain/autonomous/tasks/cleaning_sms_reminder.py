"""
Cleaning SMS reminder -- autonomous task.

Runs daily, looks 3 days ahead on the residential calendar, matches events
to CRM contacts by calendar_keyword, and sends a friendly SMS reminder
to the customer's phone number.

Dedup: tracks sent reminders in-memory keyed by "{contact_phone}:{event_date}"
so each customer gets at most one reminder per cleaning date.
"""

import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.cleaning_sms_reminder")

# In-memory dedup: "{phone}:{date}" -> True
_sent_reminders: set[str] = set()
_MAX_DEDUP = 1000


def _e164(number: str) -> str:
    """Normalize phone to E.164."""
    n = re.sub(r"[\s\-\(\)\.]", "", number.strip())
    if not n.startswith("+"):
        n = "+1" + n if len(n) == 10 else "+" + n
    return n


async def run(task: ScheduledTask) -> dict[str, Any] | str:
    """Send SMS reminders to residential customers 3 days before their cleaning.

    Configurable via task.metadata:
        lead_days (int): Days before cleaning to send reminder (default: 3)
        calendar_id (str): Google Calendar ID for residential cleanings
        from_number (str): SignalWire toll-free number to send from
        dry_run (bool): If true, list what would be sent without sending
    """
    global _sent_reminders

    from ...config import settings

    metadata = task.metadata or {}
    lead_days = metadata.get("lead_days", 3)
    dry_run = metadata.get("dry_run", False)
    calendar_id = metadata.get("calendar_id")
    from_number = metadata.get("from_number")

    if not calendar_id:
        return {"_skip_synthesis": "No calendar_id configured for cleaning SMS reminders"}

    if not from_number:
        return {"_skip_synthesis": "No from_number configured for cleaning SMS reminders"}

    # Gate: calendar must be configured
    if not settings.tools.calendar_enabled or not settings.tools.calendar_refresh_token:
        return {"_skip_synthesis": "Calendar not configured"}

    from ...services.calendar_provider import get_calendar_provider
    from ...services.crm_provider import get_crm_provider
    from ...storage.repositories.customer_service import get_customer_service_repo

    cal = get_calendar_provider()
    crm = get_crm_provider()
    svc_repo = get_customer_service_repo()

    # Target date: lead_days from now
    today = date.today()
    target_date = today + timedelta(days=lead_days)
    target_start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
    # Extend window to cover US timezone events that spill into next UTC day
    target_end = target_start + timedelta(hours=30)

    # Fetch calendar events for the target day
    try:
        events = await cal.list_events(target_start, target_end, calendar_id=calendar_id)
    except Exception as e:
        logger.error("Failed to fetch calendar events: %s", e)
        return {"_skip_synthesis": f"Calendar fetch error: {e}"}

    confirmed = [e for e in events if e.status == "confirmed"]
    if not confirmed:
        return {"_skip_synthesis": f"No cleaning events on {target_date}"}

    # Load service agreements to match events to contacts
    try:
        services = await svc_repo.list_active()
    except Exception as e:
        logger.error("Failed to load services: %s", e)
        return {"_skip_synthesis": f"Error loading services: {e}"}

    # Build keyword -> service lookup
    keyword_map: dict[str, dict] = {}
    for svc in services:
        kw = svc.get("calendar_keyword", "").lower().strip()
        if kw:
            keyword_map[kw] = svc

    # Dedup housekeeping
    if len(_sent_reminders) > _MAX_DEDUP:
        _sent_reminders.clear()

    results: dict[str, Any] = {
        "target_date": target_date.isoformat(),
        "lead_days": lead_days,
        "events_found": len(confirmed),
        "reminders_sent": 0,
        "reminders_skipped": 0,
        "dry_run": dry_run,
        "details": [],
    }

    # Match events to contacts and send SMS
    for event in confirmed:
        event_date = event.start.date() if isinstance(event.start, datetime) else event.start
        # Only send for events on the target date
        if event_date != target_date:
            continue

        summary = event.summary or ""
        summary_lower = summary.lower()

        # Try to match to a service agreement by keyword
        matched_svc = None
        for kw, svc in keyword_map.items():
            if kw in summary_lower:
                matched_svc = svc
                break

        # Look up CRM contact for phone number
        contact = None
        contact_phone = None
        contact_id_for_crm = None

        if matched_svc and matched_svc.get("contact_id"):
            contact_id_for_crm = str(matched_svc["contact_id"])
            try:
                contact = await crm.get_contact(contact_id_for_crm)
                contact_phone = (contact or {}).get("phone")
            except Exception as e:
                logger.warning("Contact lookup failed for %s: %s", contact_id_for_crm, e)

        # Fallback: match event summary against CRM contact names
        # (residential customers may not have service agreements)
        if not contact_phone and summary.strip():
            try:
                matches = await crm.search_contacts(query=summary.strip(), limit=1)
                if matches:
                    contact = matches[0]
                    contact_phone = contact.get("phone")
                    contact_id_for_crm = str(contact["id"]) if contact.get("id") else None
            except Exception as e:
                logger.debug("Name-based contact search failed for '%s': %s", summary, e)

        if not contact_phone:
            results["reminders_skipped"] += 1
            results["details"].append({
                "event": summary,
                "date": event_date.isoformat(),
                "status": "skipped",
                "reason": "no phone number",
            })
            continue

        # Dedup check
        dedup_key = f"{contact_phone}:{event_date}"
        if dedup_key in _sent_reminders:
            results["reminders_skipped"] += 1
            results["details"].append({
                "event": summary,
                "date": event_date.isoformat(),
                "status": "skipped",
                "reason": "already reminded",
            })
            continue

        # Build the message
        contact_name = (contact or {}).get("full_name", summary)
        day_name = target_date.strftime("%A, %B %d").replace(" 0", " ")
        business_name = metadata.get("business_name", "Effingham Office Maids")
        message = (
            f"Hi {contact_name}! This is a friendly reminder from "
            f"{business_name} that your cleaning is scheduled "
            f"for {day_name}. Please let us know if you need to "
            f"reschedule. Thank you!"
        )

        if dry_run:
            _sent_reminders.add(dedup_key)
            results["reminders_sent"] += 1
            results["details"].append({
                "event": summary,
                "date": event_date.isoformat(),
                "phone": contact_phone,
                "contact": contact_name,
                "status": "would_send",
                "message": message,
            })
            continue

        # Send SMS via SignalWire
        try:
            from ...mcp.twilio_server import _client, _run_sync

            msg = await _run_sync(_client().messages.create,
                to=_e164(contact_phone),
                from_=_e164(from_number),
                body=message,
            )

            # Persist outbound SMS
            try:
                from ...storage.repositories.sms_message import get_sms_message_repo
                sms_repo = get_sms_message_repo()
                await sms_repo.create(
                    message_sid=msg.sid,
                    from_number=msg.from_,
                    to_number=msg.to,
                    direction="outbound",
                    body=message,
                    status=msg.status or "queued",
                    source="cleaning_sms_reminder",
                )
            except Exception as persist_err:
                logger.warning("Failed to persist SMS %s: %s", msg.sid, persist_err)

            _sent_reminders.add(dedup_key)
            results["reminders_sent"] += 1
            results["details"].append({
                "event": summary,
                "date": event_date.isoformat(),
                "phone": contact_phone,
                "contact": contact_name,
                "status": "sent",
                "message_sid": msg.sid,
            })

            # CRM log
            if contact_id_for_crm:
                try:
                    await crm.log_interaction(
                        contact_id=contact_id_for_crm,
                        interaction_type="sms",
                        summary=f"Cleaning reminder sent for {day_name}",
                    )
                except Exception as e:
                    logger.warning("CRM log failed: %s", e)

        except Exception as e:
            logger.error("SMS send failed for %s: %s", contact_phone, e)
            results["details"].append({
                "event": summary,
                "date": event_date.isoformat(),
                "phone": contact_phone,
                "status": "error",
                "error": str(e),
            })

    if results["reminders_sent"] == 0 and results["reminders_skipped"] == 0:
        return {"_skip_synthesis": f"No matching cleaning events on {target_date}"}

    # Notify
    await _send_notification(results, task)
    return results


async def _send_notification(results: dict, task: ScheduledTask) -> None:
    """Send ntfy push notification with reminder summary."""
    try:
        from ...autonomous.config import autonomous_config
        from ...config import settings

        if not autonomous_config.notify_results or not settings.alerts.ntfy_enabled:
            return
        if (task.metadata or {}).get("notify") is False:
            return

        from ...tools.notify import notify_tool

        sent = results["reminders_sent"]
        target = results["target_date"]
        lines = [f"Date: {target}", f"Reminders sent: {sent}"]

        for d in results.get("details", []):
            if d["status"] in ("sent", "would_send"):
                lines.append(f"  {d['contact']}: {d['phone']}")

        await notify_tool._send_notification(
            message="\n".join(lines),
            title="Atlas: Cleaning SMS Reminders",
            priority="default",
            tags="sms,cleaning",
        )
    except Exception:
        logger.warning("Failed to send ntfy notification", exc_info=True)
