"""
Auto-approve email drafts after a configurable delay window.

Runs every 2 minutes.  For each pending root draft whose delay window has
elapsed, checks intent + confidence eligibility and calls approve_draft()
to send automatically.  Complaint drafts are NEVER auto-approved.
"""

import json
import logging
from uuid import UUID

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.email_auto_approve")


async def run(task: ScheduledTask) -> dict:
    """Auto-approve eligible pending drafts past the delay window."""
    cfg = settings.email_draft
    if not cfg.auto_approve_enabled:
        return {"_skip_synthesis": "Auto-approve disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "Database not available"}

    delay = cfg.auto_approve_delay_seconds

    # Find pending root drafts past the delay window
    rows = await pool.fetch(
        """
        SELECT ed.id, ed.gmail_message_id, ed.created_at,
               pe.action_plan, pe.intent, pe.intent_confidence
        FROM email_drafts ed
        JOIN processed_emails pe ON ed.gmail_message_id = pe.gmail_message_id
        WHERE ed.status = 'pending'
          AND ed.parent_draft_id IS NULL
          AND ed.created_at <= NOW() - make_interval(secs => $1)
        ORDER BY ed.created_at ASC
        """,
        float(delay),
    )

    if not rows:
        return {"_skip_synthesis": True, "auto_approved": 0, "checked": 0}

    from ...api.email_drafts import approve_draft

    approved = 0
    skipped = 0

    for row in rows:
        draft_id = row["id"]
        intent = row["intent"]
        action_plan_raw = row["action_plan"]

        # Hard safety rail: never auto-approve complaint drafts
        if intent == "complaint":
            skipped += 1
            continue

        # Check intent eligibility
        if intent and intent not in cfg.auto_approve_intents:
            skipped += 1
            continue

        # Use indexed column; fall back to action_plan JSON for legacy rows
        confidence = float(row["intent_confidence"]) if row["intent_confidence"] is not None else 0.0
        if confidence == 0.0 and action_plan_raw:
            try:
                plan = json.loads(action_plan_raw) if isinstance(action_plan_raw, str) else action_plan_raw
                if isinstance(plan, dict):
                    confidence = float(plan.get("confidence", 0.0))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        # Skip if no action_plan at all (no intent classification happened)
        if not action_plan_raw:
            skipped += 1
            continue

        if confidence < cfg.auto_approve_min_confidence:
            skipped += 1
            continue

        try:
            await approve_draft(UUID(str(draft_id)))
            approved += 1
            logger.info(
                "Auto-approved draft %s (intent=%s, confidence=%.2f)",
                draft_id, intent, confidence,
            )
        except Exception as exc:
            logger.warning(
                "Auto-approve failed for draft %s: %s", draft_id, exc,
            )
            skipped += 1

    logger.info(
        "Auto-approve cycle: %d checked, %d approved, %d skipped",
        len(rows), approved, skipped,
    )
    return {
        "auto_approved": approved,
        "skipped": skipped,
        "checked": len(rows),
        "_skip_synthesis": True,
    }
