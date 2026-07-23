"""
CRM email backfill task.

Manually-triggered task that scans the **Sent Mail** folder via IMAP,
extracts recipients, and populates the CRM with contacts and interactions
from historical customer correspondence.

Strategy: if you sent someone an email, they are a real contact. The Sent
folder is a far more reliable signal than the inbox (which is mostly
automated services).

Idempotent: find_or_create_contact deduplicates by email; processed_emails
has ON CONFLICT DO NOTHING.  Safe to re-run.
"""

import asyncio
import email.utils
import logging
import re
from datetime import timezone
from typing import Any

from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from .gmail_digest import _get_processed_message_ids

logger = logging.getLogger("atlas.autonomous.tasks.email_backfill")

# Recipients to skip -- automated addresses, self, and junk patterns
_SKIP_LOCAL_PARTS = frozenset({
    "noreply", "donotreply", "maildaemon", "postmaster",
    "unsubscribe", "bounce", "abuse", "support", "help",
    "info", "notifications", "alerts", "reply",
    "customerservice", "tuningsupport", "affiliates",
    "contractor", "youremail", "your_email",
})

_SKIP_DOMAIN_KEYWORDS = frozenset({
    "optout", "unsubscribe", "getblueshift.com",
    "example.invalid", "docs.google.com",
    "redfinoutsourcing.com", "americanmuscle.com",
    "pedalcommander.com", "webpartners.co", "lincare.com",
})

_SKIP_SUBJECT_RE = re.compile(
    r"^(unsubscribe|uid=|https?://)", re.IGNORECASE,
)


def _is_junk_recipient(addr: str, subject: str, owner_email: str) -> bool:
    """Return True if the recipient address should be skipped."""
    addr_lower = addr.lower()
    if addr_lower == owner_email.lower():
        return True

    local = addr_lower.split("@")[0].replace("_", "").replace("-", "").replace(".", "")
    if local in _SKIP_LOCAL_PARTS:
        return True

    domain = addr_lower.split("@")[-1] if "@" in addr_lower else ""
    if any(kw in domain for kw in _SKIP_DOMAIN_KEYWORDS):
        return True

    # Skip gibberish addresses (long base64-like local parts with no real name)
    if len(local) > 40:
        return True

    if _SKIP_SUBJECT_RE.match(subject.strip()):
        return True

    return False


async def run(task: ScheduledTask) -> dict:
    """Scan Sent Mail and populate CRM contacts from email recipients."""
    from ...services.email_provider import IMAPEmailProvider
    from ...config import settings

    metadata = task.metadata or {}
    max_days = metadata.get("max_days", 730)
    batch_size = metadata.get("batch_size", 10)
    mailbox = metadata.get("mailbox", "[Gmail]/Sent Mail")
    window_days = 30

    provider = IMAPEmailProvider()
    if not provider.is_configured():
        return {"_skip_synthesis": "IMAP not configured"}

    owner_email = settings.email.imap_username or ""

    # Date-range chunking: split into windows to work around 200-result IMAP cap.
    all_emails: list[dict[str, Any]] = []
    windows_scanned = 0

    for window_start_offset in range(0, max_days, window_days):
        window_end_offset = min(window_start_offset + window_days, max_days)

        window_query_parts = [f"newer_than:{window_end_offset}d"]
        if window_start_offset > 0:
            window_query_parts.append(f"older_than:{window_start_offset}d")

        window_query = " ".join(window_query_parts)

        try:
            messages = await provider.list_messages(
                query=window_query, max_results=200, mailbox=mailbox,
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

        # Filter out junk recipients before fetching full bodies
        filtered = []
        for m in new_messages:
            _, recipient = email.utils.parseaddr(m.get("to", ""))
            subject = m.get("subject", "")
            if not recipient:
                continue
            if _is_junk_recipient(recipient, subject, owner_email):
                continue
            filtered.append(m)

        if not filtered:
            windows_scanned += 1
            continue

        # Fetch full body in batches (only for non-junk)
        for i in range(0, len(filtered), batch_size):
            batch = filtered[i : i + batch_size]
            fetch_tasks = [
                provider.get_message(m["id"], mailbox=mailbox) for m in batch
            ]
            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.warning("Backfill fetch failed: %s", r)
                elif r.get("error"):
                    logger.warning("Backfill IMAP error: %s", r["error"])
                else:
                    all_emails.append(r)

        windows_scanned += 1
        logger.info(
            "Backfill window %d-%dd: %d messages, %d new, %d non-junk, %d fetched so far",
            window_start_offset, window_end_offset,
            len(messages), len(new_messages), len(filtered), len(all_emails),
        )

    if not all_emails:
        return {
            "total_scanned": 0,
            "windows_scanned": windows_scanned,
            "_skip_synthesis": True,
        }

    # CRM backfill: create contacts from recipients and log interactions
    contacts_linked = 0
    interactions_logged = 0

    from ...services.crm_provider import get_crm_provider

    crm = get_crm_provider()

    for e in all_emails:
        try:
            _, recipient_email = email.utils.parseaddr(e.get("to", ""))
            if not recipient_email:
                continue

            # Double-check junk filter (full message might have different To:)
            if _is_junk_recipient(
                recipient_email, e.get("subject", ""), owner_email,
            ):
                continue

            recipient_name = e.get("to", "").split("<")[0].strip().strip('"')
            if not recipient_name or recipient_name.lower() == recipient_email.lower():
                # No display name -- use email local part as name
                recipient_name = recipient_email.split("@")[0].replace(".", " ").title()

            # find_or_create_contact deduplicates by email
            contact = await crm.find_or_create_contact(
                full_name=recipient_name,
                email=recipient_email,
                source="email_backfill",
                contact_type="customer",
                business_context_id="effingham_maids",
            )
            if not contact.get("id"):
                continue

            contact_id = str(contact["id"])
            e["_contact_id"] = contact_id
            contacts_linked += 1

            # Log the sent email as a CRM interaction
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
                summary=f"Sent email: {subject}. {body_preview}".strip(),
                occurred_at=occurred_at,
            )
            interactions_logged += 1

        except Exception as exc:
            logger.warning(
                "Backfill CRM failed for email %s: %s", e.get("id"), exc,
            )

    # Record to processed_emails (ON CONFLICT DO NOTHING for idempotency)
    await _record_backfill_emails(all_emails)

    logger.info(
        "Backfill complete: %d fetched, %d contacts linked, %d interactions",
        len(all_emails), contacts_linked, interactions_logged,
    )

    return {
        "total_scanned": len(all_emails),
        "contacts_linked": contacts_linked,
        "interactions_logged": interactions_logged,
        "windows_scanned": windows_scanned,
        "_skip_synthesis": True,
    }


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
            e.get("to", ""),  # sender column stores recipient for sent mail
            e.get("subject", ""),
            "sent",  # category = sent (distinguishes from inbox records)
            "backfill",  # priority = backfill (audit trail)
            True,  # replyable
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
