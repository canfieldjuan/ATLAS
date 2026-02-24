"""Atlas event types and emission helper."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

logger = logging.getLogger("atlas.reasoning.events")


class EventType:
    """Known event type constants."""

    # Email
    EMAIL_RECEIVED = "email.received"
    EMAIL_DRAFT_GENERATED = "email.draft_generated"
    EMAIL_DRAFT_SENT = "email.draft_sent"
    EMAIL_FOLLOWUP_RECEIVED = "email.followup_received"

    # Voice
    VOICE_TURN_COMPLETED = "voice.turn_completed"

    # CRM
    CRM_CONTACT_CREATED = "crm.contact_created"
    CRM_INTERACTION_LOGGED = "crm.interaction_logged"

    # Calendar
    CALENDAR_EVENT_CREATED = "calendar.event_created"

    # SMS / Calls
    SMS_RECEIVED = "sms.received"
    CALL_COMPLETED = "call.completed"

    # Appointments / Invoicing
    APPOINTMENT_BOOKED = "appointment.booked"
    INVOICE_SENT = "invoice.sent"
    INVOICE_OVERDUE = "invoice.overdue"

    # News
    NEWS_SIGNIFICANT = "news.significant"
    NEWS_MARKET_MOVING = "news.market_moving"

    # Market
    MARKET_SIGNIFICANT_MOVE = "market.significant_move"
    MARKET_ALERT = "market.alert"


@dataclass
class AtlasEvent:
    """In-memory representation of a persisted atlas_events row."""

    id: Optional[UUID] = None
    event_type: str = ""
    source: str = ""
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    processing_result: Optional[dict[str, Any]] = None

    @classmethod
    def from_row(cls, row: dict) -> AtlasEvent:
        """Build from an asyncpg Record/dict."""
        return cls(
            id=row.get("id"),
            event_type=row.get("event_type", ""),
            source=row.get("source", ""),
            entity_type=row.get("entity_type"),
            entity_id=row.get("entity_id"),
            payload=row.get("payload") or {},
            created_at=row.get("created_at"),
            processed_at=row.get("processed_at"),
            processing_result=row.get("processing_result"),
        )


async def emit_event(
    event_type: str,
    source: str,
    payload: dict[str, Any],
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
) -> UUID:
    """Persist an event to atlas_events. NOTIFY fires via trigger."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    row_id = await pool.fetchval(
        """
        INSERT INTO atlas_events (event_type, source, entity_type, entity_id, payload)
        VALUES ($1, $2, $3, $4, $5::jsonb)
        RETURNING id
        """,
        event_type,
        source,
        entity_type,
        entity_id,
        __import__("json").dumps(payload),
    )
    logger.debug("Emitted event %s (%s) -> %s", event_type, source, row_id)
    return row_id
