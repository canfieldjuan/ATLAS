"""
Campaign send task.

Runs every 2 minutes.  Finds queued campaign emails past their cancel
window and sends them via the configured CampaignSender (Resend).
Updates campaign + sequence state on success/failure.
"""

import logging
from datetime import datetime, timedelta, timezone

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from .campaign_audit import log_campaign_event

logger = logging.getLogger("atlas.autonomous.tasks.campaign_send")

# After this many consecutive send failures, mark campaign as draft
# so it stops being retried.
_MAX_SEND_ATTEMPTS = 3


async def run(task: ScheduledTask) -> dict:
    """Send queued campaign emails past the cancel window."""
    cfg = settings.campaign_sequence
    if not cfg.enabled:
        return {"_skip_synthesis": "Campaign sequences disabled"}
    if not cfg.auto_send_enabled:
        return {"_skip_synthesis": "Auto-send disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "Database unavailable"}

    from ...services.campaign_sender import get_campaign_sender

    try:
        sender = get_campaign_sender()
    except RuntimeError as exc:
        return {"_skip_synthesis": f"Sender not configured: {exc}"}

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=cfg.auto_send_delay_seconds)

    # Find campaigns ready to send:
    # - status = 'queued'
    # - approved_at <= cutoff (cancel window has passed)
    # - recipient_email set
    campaigns = await pool.fetch(
        """
        SELECT bc.id, bc.sequence_id, bc.step_number,
               bc.recipient_email, bc.from_email,
               bc.subject, bc.body, bc.company_name,
               bc.metadata
        FROM b2b_campaigns bc
        WHERE bc.status = 'queued'
          AND bc.approved_at <= $1
          AND bc.recipient_email IS NOT NULL
        ORDER BY bc.approved_at ASC
        LIMIT 20
        """,
        cutoff,
    )

    if not campaigns:
        return {"_skip_synthesis": True, "sent": 0}

    sent_count = 0
    failed_count = 0

    for campaign in campaigns:
        c = dict(campaign)
        campaign_id = c["id"]
        sequence_id = c.get("sequence_id")
        from_email = c.get("from_email") or cfg.resend_from_email

        if not from_email:
            logger.warning("No from_email for campaign %s, skipping", campaign_id)
            continue

        try:
            result = await sender.send(
                to=c["recipient_email"],
                from_email=from_email,
                subject=c.get("subject", ""),
                body=c.get("body", ""),
                tags=[
                    {"name": "company", "value": c.get("company_name", "")},
                    {"name": "step", "value": str(c.get("step_number", 1))},
                ],
            )

            esp_message_id = result.get("id", "")

            # Update campaign: sent
            await pool.execute(
                """
                UPDATE b2b_campaigns
                SET status = 'sent',
                    sent_at = $1,
                    esp_message_id = $2,
                    updated_at = $1
                WHERE id = $3
                """,
                now, esp_message_id, campaign_id,
            )

            # Update sequence: last_sent_at, last_campaign_id, next_step_after
            if sequence_id:
                step = c.get("step_number", 1)

                # Check if this was the final step
                seq_row = await pool.fetchrow(
                    "SELECT current_step, max_steps FROM campaign_sequences WHERE id = $1",
                    sequence_id,
                )
                is_final_step = (
                    seq_row and seq_row["current_step"] >= seq_row["max_steps"]
                )

                if is_final_step:
                    # Mark sequence as completed -- no more steps
                    await pool.execute(
                        """
                        UPDATE campaign_sequences
                        SET last_sent_at = $1,
                            last_campaign_id = $2,
                            next_step_after = NULL,
                            status = 'completed',
                            updated_at = $1
                        WHERE id = $3
                        """,
                        now, campaign_id, sequence_id,
                    )
                else:
                    delays = cfg.step_delay_days
                    delay_idx = min(step - 1, len(delays) - 1)
                    delay_days = delays[delay_idx] if delays else 3
                    next_step_time = now + timedelta(days=delay_days)

                    await pool.execute(
                        """
                        UPDATE campaign_sequences
                        SET last_sent_at = $1,
                            last_campaign_id = $2,
                            next_step_after = $3,
                            updated_at = $1
                        WHERE id = $4
                        """,
                        now, campaign_id, next_step_time, sequence_id,
                    )

            # Audit log
            await log_campaign_event(
                pool, event_type="sent", source="system",
                campaign_id=campaign_id, sequence_id=sequence_id,
                step_number=c.get("step_number"),
                subject=c.get("subject"),
                recipient_email=c["recipient_email"],
                esp_message_id=esp_message_id,
            )

            sent_count += 1
            logger.info(
                "Sent campaign %s (step %s) to %s via Resend: %s",
                campaign_id, c.get("step_number"), c["recipient_email"], esp_message_id,
            )

        except Exception as exc:
            failed_count += 1

            # Audit log the failure
            await log_campaign_event(
                pool, event_type="send_failed", source="system",
                campaign_id=campaign_id, sequence_id=sequence_id,
                step_number=c.get("step_number"),
                recipient_email=c["recipient_email"],
                error_detail=str(exc),
            )

            # Check if we've exceeded max attempts -- revert to draft
            fail_count = await pool.fetchval(
                """
                SELECT COUNT(*) FROM campaign_audit_log
                WHERE campaign_id = $1 AND event_type = 'send_failed'
                """,
                campaign_id,
            )
            if fail_count >= _MAX_SEND_ATTEMPTS:
                await pool.execute(
                    "UPDATE b2b_campaigns SET status = 'draft', updated_at = $1 WHERE id = $2",
                    now, campaign_id,
                )
                logger.error(
                    "Campaign %s reverted to draft after %d send failures",
                    campaign_id, fail_count,
                )
            else:
                logger.warning(
                    "Send failed for campaign %s (attempt %d/%d): %s",
                    campaign_id, fail_count, _MAX_SEND_ATTEMPTS, exc,
                )

    return {
        "_skip_synthesis": True,
        "sent": sent_count,
        "failed": failed_count,
    }
