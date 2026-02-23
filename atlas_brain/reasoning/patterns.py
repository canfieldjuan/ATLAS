"""Pattern matchers for proactive reflection.

Each pattern detector queries recent data and returns findings
that the reflection node can act on.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("atlas.reasoning.patterns")


async def detect_stale_threads() -> list[dict[str, Any]]:
    """Find threads where Atlas sent a reply but received no response in 3+ days."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return []

    rows = await pool.fetch(
        """
        SELECT ed.id AS draft_id, ed.original_from, ed.draft_subject,
               ed.sent_at, pe.intent, pe.contact_id
        FROM email_drafts ed
        LEFT JOIN processed_emails pe
            ON pe.gmail_message_id = ed.gmail_message_id
        WHERE ed.status = 'sent'
          AND ed.sent_at < NOW() - INTERVAL '3 days'
          AND NOT EXISTS (
              SELECT 1 FROM processed_emails pe2
              WHERE pe2.followup_of_draft_id = ed.id
          )
        ORDER BY ed.sent_at ASC
        LIMIT 20
        """
    )
    return [
        {
            "pattern": "stale_thread",
            "description": (
                f"Reply to {r['original_from']} re: {r['draft_subject']} "
                f"sent {r['sent_at'].strftime('%b %d')} with no response"
            ),
            "entity_type": "contact",
            "entity_id": r.get("contact_id"),
            "draft_id": str(r["draft_id"]),
            "original_intent": r.get("intent"),
            "days_since_reply": (
                (__import__("datetime").datetime.now(__import__("datetime").timezone.utc)
                 - r["sent_at"]).days
            ),
        }
        for r in rows
    ]


async def detect_scheduling_gaps() -> list[dict[str, Any]]:
    """Find estimates sent but no booking made within 5+ days."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return []

    rows = await pool.fetch(
        """
        SELECT ed.id AS draft_id, ed.original_from, ed.draft_subject,
               ed.sent_at, pe.contact_id
        FROM email_drafts ed
        LEFT JOIN processed_emails pe
            ON pe.gmail_message_id = ed.gmail_message_id
        WHERE ed.status = 'sent'
          AND pe.intent = 'estimate_request'
          AND ed.sent_at < NOW() - INTERVAL '5 days'
          AND pe.contact_id IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM appointments a
              WHERE a.contact_id = pe.contact_id
                AND a.created_at > ed.sent_at
          )
        ORDER BY ed.sent_at ASC
        LIMIT 20
        """
    )
    return [
        {
            "pattern": "scheduling_gap",
            "description": (
                f"Estimate for {r['original_from']} sent {r['sent_at'].strftime('%b %d')} "
                f"but no appointment booked"
            ),
            "entity_type": "contact",
            "entity_id": r.get("contact_id"),
            "draft_id": str(r["draft_id"]),
        }
        for r in rows
    ]


async def detect_missing_followups() -> list[dict[str, Any]]:
    """Find completed appointments with no invoice sent."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return []

    try:
        rows = await pool.fetch(
            """
            SELECT a.id AS appointment_id, a.contact_id, a.service_type,
                   a.scheduled_at, c.full_name
            FROM appointments a
            LEFT JOIN contacts c ON c.id::text = a.contact_id
            WHERE a.status = 'completed'
              AND a.scheduled_at < NOW() - INTERVAL '1 day'
              AND NOT EXISTS (
                  SELECT 1 FROM invoices i
                  WHERE i.contact_id = a.contact_id
                    AND i.created_at > a.scheduled_at
              )
            ORDER BY a.scheduled_at DESC
            LIMIT 10
            """
        )
    except Exception:
        # invoices table may not exist
        return []

    return [
        {
            "pattern": "missing_followup",
            "description": (
                f"Appointment with {r.get('full_name', 'unknown')} on "
                f"{r['scheduled_at'].strftime('%b %d')} completed but no invoice"
            ),
            "entity_type": "contact",
            "entity_id": r.get("contact_id"),
            "appointment_id": str(r["appointment_id"]),
        }
        for r in rows
    ]


async def run_all_pattern_detectors() -> list[dict[str, Any]]:
    """Run all pattern detectors and aggregate findings."""
    import asyncio

    results = await asyncio.gather(
        detect_stale_threads(),
        detect_scheduling_gaps(),
        detect_missing_followups(),
        return_exceptions=True,
    )

    findings = []
    for result in results:
        if isinstance(result, list):
            findings.extend(result)
        elif isinstance(result, Exception):
            logger.warning("Pattern detector failed: %s", result)

    return findings
