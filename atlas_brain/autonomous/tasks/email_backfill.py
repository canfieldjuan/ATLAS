"""
CRM email backfill task.

Manually-triggered task that scans inbox history via IMAP, classifies
emails, and populates the CRM with contacts and interactions from
historical commercial/customer emails.

Idempotent: find_or_create_contact deduplicates by email; processed_emails
has ON CONFLICT DO NOTHING.  Safe to re-run.
"""

import asyncio
import email.utils
import logging
from datetime import timezone
from typing import Any

from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from .email_classifier import get_email_classifier
from .gmail_digest import _get_processed_message_ids

logger = logging.getLogger("atlas.autonomous.tasks.email_backfill")


async def run(task: ScheduledTask) -> dict:
    """Scan inbox history and populate CRM contacts from historical emails."""
    from ...services.email_provider import IMAPEmailProvider

    metadata = task.metadata or {}
    query = metadata.get("query", "newer_than:90d")
    max_days = metadata.get("max_days", 90)
    batch_size = metadata.get("batch_size", 10)
    window_days = 30

    provider = IMAPEmailProvider()
    if not provider.is_configured():
        return {"_skip_synthesis": "IMAP not configured"}

    classifier = get_email_classifier()

    # Date-range chunking: split into windows to work around 200-result IMAP cap.
    # Work backwards from today in `window_days`-day chunks.
    all_emails: list[dict[str, Any]] = []
    windows_scanned = 0

    for window_start_offset in range(0, max_days, window_days):
        window_end_offset = min(window_start_offset + window_days, max_days)

        # Build window query: newer_than for outer bound, older_than for inner bound
        # e.g. for days 0-30: newer_than:30d (no older_than needed for most recent)
        # e.g. for days 30-60: newer_than:60d older_than:30d
        window_query_parts = [f"newer_than:{window_end_offset}d"]
        if window_start_offset > 0:
            window_query_parts.append(f"older_than:{window_start_offset}d")

        # Merge with user query filters (e.g. from:customer@domain.com)
        extra_filters = _extract_non_date_filters(query)
        if extra_filters:
            window_query_parts.extend(extra_filters)

        window_query = " ".join(window_query_parts)

        try:
            messages = await provider.list_messages(
                query=window_query, max_results=200,
            )
        except Exception as e:
            logger.warning(
                "Backfill window %d-%dd failed: %s",
                window_start_offset, window_end_offset, e,
            )
            continue

        if not messages:
            windows_scanned += 1
            continue

        # Dedup against already-processed
        msg_ids = [m["id"] for m in messages]
        already = await _get_processed_message_ids(msg_ids)
        new_messages = [m for m in messages if m["id"] not in already]

        if not new_messages:
            windows_scanned += 1
            continue

        # Fetch full body in batches
        for i in range(0, len(new_messages), batch_size):
            batch = new_messages[i : i + batch_size]
            tasks = [provider.get_message(m["id"]) for m in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.warning("Backfill fetch failed: %s", r)
                elif r.get("error"):
                    logger.warning("Backfill IMAP error: %s", r["error"])
                else:
                    all_emails.append(r)

        windows_scanned += 1
        logger.info(
            "Backfill window %d-%dd: %d messages, %d new, %d fetched so far",
            window_start_offset, window_end_offset,
            len(messages), len(new_messages), len(all_emails),
        )

    if not all_emails:
        return {
            "total_scanned": 0,
            "windows_scanned": windows_scanned,
            "_skip_synthesis": True,
        }

    # Classify all
    classifier.classify_batch(all_emails)

    # Filter to real human-replyable emails only.
    # Must be replyable (not noreply/automated senders) AND from a
    # category that indicates actual person-to-person correspondence.
    _CONTACT_CATEGORIES = {"lead", "personal"}
    actionable = [
        e for e in all_emails
        if e.get("replyable") is True
        and e.get("category") in _CONTACT_CATEGORIES
    ]

    if not actionable:
        return {
            "total_scanned": len(all_emails),
            "actionable": 0,
            "windows_scanned": windows_scanned,
            "_skip_synthesis": True,
        }

    # CRM backfill: create contacts and log interactions
    contacts_linked = 0
    interactions_logged = 0

    from ...services.crm_provider import get_crm_provider

    crm = get_crm_provider()

    for e in actionable:
        try:
            _, sender_email = email.utils.parseaddr(e.get("from", ""))
            if not sender_email:
                continue

            # Skip automated/noreply senders that slipped through classification
            local_part = sender_email.split("@")[0].lower().replace("_", "").replace("-", "").replace(".", "")
            if local_part in ("noreply", "donotreply", "maildaemon", "postmaster"):
                continue

            sender_name = e.get("from", "").split("<")[0].strip().strip('"')
            if not sender_name:
                sender_name = sender_email

            # find_or_create_contact deduplicates by email
            contact = await crm.find_or_create_contact(
                full_name=sender_name,
                email=sender_email,
                source="email_backfill",
                contact_type="customer",
            )
            if not contact.get("id"):
                continue

            contact_id = str(contact["id"])
            e["_contact_id"] = contact_id
            contacts_linked += 1

            # Log the email as a CRM interaction
            subject = e.get("subject", "(no subject)")
            body_preview = (e.get("body_text") or "")[:200]
            date_str = e.get("date", "")
            occurred_at = None
            if date_str:
                try:
                    occurred_at = _parse_email_date(date_str)
                except Exception:
                    pass

            await crm.log_interaction(
                contact_id=contact_id,
                interaction_type="email",
                summary=f"Received email: {subject}. {body_preview}".strip(),
                occurred_at=occurred_at,
            )
            interactions_logged += 1

        except Exception as exc:
            logger.warning(
                "Backfill CRM failed for email %s: %s", e.get("id"), exc,
            )

    # Record to processed_emails (ON CONFLICT DO NOTHING for idempotency)
    await _record_backfill_emails(actionable)

    logger.info(
        "Backfill complete: %d scanned, %d actionable, %d contacts linked, "
        "%d interactions",
        len(all_emails), len(actionable), contacts_linked, interactions_logged,
    )

    return {
        "total_scanned": len(all_emails),
        "actionable": len(actionable),
        "contacts_linked": contacts_linked,
        "interactions_logged": interactions_logged,
        "windows_scanned": windows_scanned,
        "_skip_synthesis": True,
    }


def _extract_non_date_filters(query: str) -> list[str]:
    """Extract non-date filter tokens from a query string.

    Strips newer_than:, older_than:, is:unread etc. and returns remaining
    tokens like from:customer@domain.com.
    """
    skip_prefixes = ("newer_than:", "older_than:", "is:")
    tokens = query.split()
    return [t for t in tokens if not any(t.lower().startswith(p) for p in skip_prefixes)]


def _parse_email_date(date_str: str) -> str | None:
    """Parse an email date string to ISO format for log_interaction."""
    from email.utils import parsedate_to_datetime

    try:
        dt = parsedate_to_datetime(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception:
        return None


async def _record_backfill_emails(emails: list[dict[str, Any]]) -> None:
    """Record processed emails to DB with ON CONFLICT DO NOTHING."""
    pool = get_db_pool()
    if not pool.is_initialized:
        return

    records = []
    for e in emails:
        records.append((
            e.get("id", ""),
            e.get("from", ""),
            e.get("subject", ""),
            e.get("category"),
            e.get("priority"),
            e.get("replyable"),
            e.get("_contact_id"),
        ))

    if not records:
        return

    try:
        async with pool.transaction() as conn:
            await conn.executemany(
                """
                INSERT INTO processed_emails
                    (gmail_message_id, sender, subject, category, priority,
                     replyable, contact_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (gmail_message_id) DO NOTHING
                """,
                records,
            )
        logger.info("Recorded %d backfill emails to processed_emails", len(records))
    except Exception as e:
        logger.warning("Failed to record backfill emails: %s", e)
