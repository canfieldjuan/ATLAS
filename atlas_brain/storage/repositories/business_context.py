"""
Business context repository.

Loads business identities (phone number -> persona/services/hours) from
PostgreSQL so the comms service can route calls dynamically.
"""

import logging
from typing import Optional

from ..database import get_db_pool
from ..exceptions import DatabaseUnavailableError, DatabaseOperationError

logger = logging.getLogger("atlas.storage.business_context")


class BusinessContextRepository:
    """Repository for business context CRUD."""

    async def list_enabled(self) -> list[dict]:
        """Load all enabled business contexts."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("list business contexts")

        try:
            rows = await pool.fetch(
                "SELECT * FROM business_contexts WHERE enabled = TRUE LIMIT 100"
            )
            return [dict(row) for row in rows]
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            logger.error("Failed to list business contexts: %s", e)
            raise DatabaseOperationError("list business contexts", e)

    async def get(self, context_id: str) -> Optional[dict]:
        """Load a single business context by ID."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get business context")

        try:
            row = await pool.fetchrow(
                "SELECT * FROM business_contexts WHERE id = $1",
                context_id,
            )
            return dict(row) if row else None
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            logger.error("Failed to get business context %s: %s", context_id, e)
            raise DatabaseOperationError("get business context", e)

    async def upsert(self, context_id: str, data: dict) -> dict:
        """Insert or update a business context."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("upsert business context")

        try:
            row = await pool.fetchrow(
                """
                INSERT INTO business_contexts (
                    id, name, description, phone_numbers,
                    greeting, voice_name, persona,
                    business_type, services, service_area, pricing_info,
                    monday_open, monday_close,
                    tuesday_open, tuesday_close,
                    wednesday_open, wednesday_close,
                    thursday_open, thursday_close,
                    friday_open, friday_close,
                    saturday_open, saturday_close,
                    sunday_open, sunday_close,
                    timezone, after_hours_message,
                    scheduling_enabled, scheduling_calendar_id,
                    scheduling_min_notice_hours, scheduling_max_advance_days,
                    scheduling_default_duration, scheduling_buffer_minutes,
                    transfer_number, take_messages, max_call_duration_minutes,
                    sms_enabled, sms_auto_reply, enabled
                ) VALUES (
                    $1, $2, $3, $4,
                    $5, $6, $7,
                    $8, $9, $10, $11,
                    $12, $13, $14, $15, $16, $17, $18, $19, $20, $21,
                    $22, $23, $24, $25,
                    $26, $27,
                    $28, $29, $30, $31, $32, $33,
                    $34, $35, $36,
                    $37, $38, $39
                )
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    phone_numbers = EXCLUDED.phone_numbers,
                    greeting = EXCLUDED.greeting,
                    voice_name = EXCLUDED.voice_name,
                    persona = EXCLUDED.persona,
                    business_type = EXCLUDED.business_type,
                    services = EXCLUDED.services,
                    service_area = EXCLUDED.service_area,
                    pricing_info = EXCLUDED.pricing_info,
                    monday_open = EXCLUDED.monday_open,
                    monday_close = EXCLUDED.monday_close,
                    tuesday_open = EXCLUDED.tuesday_open,
                    tuesday_close = EXCLUDED.tuesday_close,
                    wednesday_open = EXCLUDED.wednesday_open,
                    wednesday_close = EXCLUDED.wednesday_close,
                    thursday_open = EXCLUDED.thursday_open,
                    thursday_close = EXCLUDED.thursday_close,
                    friday_open = EXCLUDED.friday_open,
                    friday_close = EXCLUDED.friday_close,
                    saturday_open = EXCLUDED.saturday_open,
                    saturday_close = EXCLUDED.saturday_close,
                    sunday_open = EXCLUDED.sunday_open,
                    sunday_close = EXCLUDED.sunday_close,
                    timezone = EXCLUDED.timezone,
                    after_hours_message = EXCLUDED.after_hours_message,
                    scheduling_enabled = EXCLUDED.scheduling_enabled,
                    scheduling_calendar_id = EXCLUDED.scheduling_calendar_id,
                    scheduling_min_notice_hours = EXCLUDED.scheduling_min_notice_hours,
                    scheduling_max_advance_days = EXCLUDED.scheduling_max_advance_days,
                    scheduling_default_duration = EXCLUDED.scheduling_default_duration,
                    scheduling_buffer_minutes = EXCLUDED.scheduling_buffer_minutes,
                    transfer_number = EXCLUDED.transfer_number,
                    take_messages = EXCLUDED.take_messages,
                    max_call_duration_minutes = EXCLUDED.max_call_duration_minutes,
                    sms_enabled = EXCLUDED.sms_enabled,
                    sms_auto_reply = EXCLUDED.sms_auto_reply,
                    enabled = EXCLUDED.enabled,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING *
                """,
                context_id,
                data.get("name", ""),
                data.get("description", ""),
                data.get("phone_numbers", []),
                data.get("greeting", "Hello, how can I help you today?"),
                data.get("voice_name", "Atlas"),
                data.get("persona", ""),
                data.get("business_type", ""),
                data.get("services", []),
                data.get("service_area", ""),
                data.get("pricing_info", ""),
                data.get("monday_open", "09:00"),
                data.get("monday_close", "17:00"),
                data.get("tuesday_open", "09:00"),
                data.get("tuesday_close", "17:00"),
                data.get("wednesday_open", "09:00"),
                data.get("wednesday_close", "17:00"),
                data.get("thursday_open", "09:00"),
                data.get("thursday_close", "17:00"),
                data.get("friday_open", "09:00"),
                data.get("friday_close", "17:00"),
                data.get("saturday_open"),
                data.get("saturday_close"),
                data.get("sunday_open"),
                data.get("sunday_close"),
                data.get("timezone", "America/Chicago"),
                data.get("after_hours_message", ""),
                data.get("scheduling_enabled", True),
                data.get("scheduling_calendar_id"),
                data.get("scheduling_min_notice_hours", 24),
                data.get("scheduling_max_advance_days", 30),
                data.get("scheduling_default_duration", 60),
                data.get("scheduling_buffer_minutes", 15),
                data.get("transfer_number"),
                data.get("take_messages", True),
                data.get("max_call_duration_minutes", 10),
                data.get("sms_enabled", True),
                data.get("sms_auto_reply", True),
                data.get("enabled", True),
            )
            if row:
                return dict(row)
            raise DatabaseOperationError(
                "upsert business context", Exception("No row returned")
            )
        except (DatabaseUnavailableError, DatabaseOperationError):
            raise
        except Exception as e:
            logger.error("Failed to upsert business context: %s", e)
            raise DatabaseOperationError("upsert business context", e)


_repo: Optional[BusinessContextRepository] = None


def get_business_context_repo() -> BusinessContextRepository:
    """Get the global business context repository."""
    global _repo
    if _repo is None:
        _repo = BusinessContextRepository()
    return _repo
