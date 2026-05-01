"""
Campaign audit log helper.

Single function used by all campaign code to record state changes
into the campaign_audit_log table.  Never raises.
"""

import json
import logging
from typing import Any
from uuid import UUID

logger = logging.getLogger("atlas.autonomous.tasks.campaign_audit")


async def log_campaign_event(
    pool,
    *,
    event_type: str,
    source: str = "system",
    campaign_id: UUID | str | None = None,
    sequence_id: UUID | str | None = None,
    step_number: int | None = None,
    subject: str | None = None,
    body: str | None = None,
    recipient_email: str | None = None,
    esp_message_id: str | None = None,
    error_detail: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Insert a row into campaign_audit_log.  Never raises."""
    try:
        await pool.execute(
            """
            INSERT INTO campaign_audit_log
                (campaign_id, sequence_id, event_type, step_number,
                 subject, body, recipient_email, esp_message_id,
                 error_detail, source, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
            _to_uuid(campaign_id),
            _to_uuid(sequence_id),
            event_type,
            step_number,
            subject,
            body,
            recipient_email,
            esp_message_id,
            error_detail,
            source,
            json.dumps(metadata or {}),
        )
    except Exception as exc:
        logger.warning(
            "Failed to write audit log (event=%s, seq=%s): %s",
            event_type, sequence_id, exc,
        )


def _to_uuid(val: UUID | str | None) -> UUID | None:
    if val is None:
        return None
    if isinstance(val, UUID):
        return val
    return UUID(val)
