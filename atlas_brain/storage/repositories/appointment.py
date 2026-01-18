"""
Appointment repository for persistence and retrieval.

Provides CRUD operations for appointments stored in PostgreSQL.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from ..database import get_db_pool
from ..exceptions import DatabaseUnavailableError, DatabaseOperationError

logger = logging.getLogger("atlas.storage.appointment")


class AppointmentRepository:
    """
    Repository for appointment storage and retrieval.

    Handles persistence of customer appointments with efficient queries
    for availability checking and customer lookup.
    """

    async def create(
        self,
        start_time: datetime,
        end_time: datetime,
        service_type: str,
        customer_name: str,
        customer_phone: str,
        business_context_id: str,
        customer_email: Optional[str] = None,
        customer_address: Optional[str] = None,
        calendar_event_id: Optional[str] = None,
        call_id: Optional[UUID] = None,
        notes: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict:
        """Create a new appointment. Returns the created appointment dict."""
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("create appointment")

        appointment_id = uuid4()
        duration_minutes = int((end_time - start_time).total_seconds() / 60)
        metadata_json = json.dumps(metadata or {})
        now = datetime.now(timezone.utc)

        try:
            row = await pool.fetchrow(
                """
                INSERT INTO appointments (
                    id, start_time, end_time, duration_minutes,
                    service_type, notes, customer_name, customer_phone,
                    customer_email, customer_address, calendar_event_id,
                    business_context_id, call_id, created_at, updated_at, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $14, $15::jsonb)
                RETURNING *
                """,
                appointment_id,
                start_time,
                end_time,
                duration_minutes,
                service_type,
                notes,
                customer_name,
                customer_phone,
                customer_email,
                customer_address,
                calendar_event_id,
                business_context_id,
                call_id,
                now,
                metadata_json,
            )

            if row:
                logger.info(
                    "Created appointment %s for %s at %s",
                    appointment_id,
                    customer_name,
                    start_time,
                )
                return self._row_to_dict(row)

            raise DatabaseOperationError("create appointment", Exception("No row returned"))

        except (DatabaseUnavailableError, DatabaseOperationError):
            raise
        except Exception as e:
            logger.error("Failed to create appointment: %s", e)
            raise DatabaseOperationError("create appointment", e)

    async def get_by_id(self, appointment_id: UUID) -> Optional[dict]:
        """Get an appointment by ID."""
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("get appointment by id")

        try:
            row = await pool.fetchrow(
                "SELECT * FROM appointments WHERE id = $1",
                appointment_id,
            )
            return self._row_to_dict(row) if row else None

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get appointment by id", e)

    async def get_by_phone(
        self,
        phone: str,
        status: str = "confirmed",
        upcoming_only: bool = True,
        limit: int = 10,
    ) -> list[dict]:
        """
        Get appointments by customer phone number.

        Used for reschedule/cancel flows where customer calls back.
        """
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("get appointments by phone")

        try:
            # Normalize phone (strip non-digits for comparison)
            phone_digits = "".join(c for c in phone if c.isdigit())

            conditions = ["REGEXP_REPLACE(customer_phone, '[^0-9]', '', 'g') = $1"]
            params = [phone_digits]
            param_idx = 2

            if status:
                conditions.append(f"status = ${param_idx}")
                params.append(status)
                param_idx += 1

            if upcoming_only:
                conditions.append(f"start_time > ${param_idx}")
                params.append(datetime.now(timezone.utc))
                param_idx += 1

            params.append(limit)

            rows = await pool.fetch(
                f"""
                SELECT * FROM appointments
                WHERE {' AND '.join(conditions)}
                ORDER BY start_time ASC
                LIMIT ${param_idx}
                """,
                *params,
            )

            return [self._row_to_dict(row) for row in rows]

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get appointments by phone", e)

    async def search_by_name(
        self,
        name: str,
        include_history: bool = True,
        limit: int = 10,
    ) -> list[dict]:
        """
        Search appointments by customer name (case-insensitive partial match).

        Args:
            name: Customer name to search for (partial match)
            include_history: If True, include past appointments
            limit: Maximum results to return

        Returns:
            List of matching appointments, most recent first
        """
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("search appointments by name")

        try:
            # Use ILIKE for case-insensitive partial match
            search_pattern = f"%{name}%"
            conditions = ["customer_name ILIKE $1"]
            params: list[Any] = [search_pattern]
            param_idx = 2

            if not include_history:
                conditions.append(f"start_time > ${param_idx}")
                params.append(datetime.now(timezone.utc))
                param_idx += 1

            params.append(limit)

            rows = await pool.fetch(
                f"""
                SELECT * FROM appointments
                WHERE {' AND '.join(conditions)}
                ORDER BY start_time DESC
                LIMIT ${param_idx}
                """,
                *params,
            )

            return [self._row_to_dict(row) for row in rows]

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("search appointments by name", e)

    async def get_in_range(
        self,
        start: datetime,
        end: datetime,
        business_context_id: Optional[str] = None,
        status: str = "confirmed",
    ) -> list[dict]:
        """
        Get appointments within a date range.

        Used for availability checking and calendar views.
        """
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("get appointments in range")

        try:
            conditions = [
                "start_time < $2",  # Starts before range ends
                "end_time > $1",    # Ends after range starts
            ]
            params = [start, end]
            param_idx = 3

            if status:
                conditions.append(f"status = ${param_idx}")
                params.append(status)
                param_idx += 1

            if business_context_id:
                conditions.append(f"business_context_id = ${param_idx}")
                params.append(business_context_id)
                param_idx += 1

            rows = await pool.fetch(
                f"""
                SELECT * FROM appointments
                WHERE {' AND '.join(conditions)}
                ORDER BY start_time ASC
                """,
                *params,
            )

            return [self._row_to_dict(row) for row in rows]

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get appointments in range", e)

    async def check_conflict(
        self,
        start: datetime,
        end: datetime,
        business_context_id: str,
        exclude_id: Optional[UUID] = None,
    ) -> bool:
        """
        Check if a time slot has conflicts.

        Returns True if there ARE conflicts (slot is NOT available).
        """
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("check appointment conflict")

        try:
            conditions = [
                "start_time < $2",  # Starts before new slot ends
                "end_time > $1",    # Ends after new slot starts
                "business_context_id = $3",
                "status = 'confirmed'",
            ]
            params = [start, end, business_context_id]

            if exclude_id:
                conditions.append("id != $4")
                params.append(exclude_id)

            result = await pool.fetchval(
                f"""
                SELECT EXISTS(
                    SELECT 1 FROM appointments
                    WHERE {' AND '.join(conditions)}
                )
                """,
                *params,
            )

            return bool(result)

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("check appointment conflict", e)

    async def cancel(
        self,
        appointment_id: UUID,
        reason: Optional[str] = None,
    ) -> bool:
        """Cancel an appointment."""
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("cancel appointment")

        try:
            now = datetime.now(timezone.utc)

            result = await pool.execute(
                """
                UPDATE appointments
                SET status = 'cancelled',
                    cancelled_at = $2,
                    cancellation_reason = $3,
                    updated_at = $2
                WHERE id = $1 AND status = 'confirmed'
                """,
                appointment_id,
                now,
                reason,
            )

            success = self._parse_row_count(result) > 0
            if success:
                logger.info("Cancelled appointment: %s", appointment_id)
            return success

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("cancel appointment", e)

    async def update(
        self,
        appointment_id: UUID,
        **updates,
    ) -> Optional[dict]:
        """
        Update appointment fields.

        Allowed fields: start_time, end_time, service_type, notes,
        customer_name, customer_phone, customer_email, customer_address,
        calendar_event_id, status
        """
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("update appointment")

        allowed_fields = {
            "start_time", "end_time", "service_type", "notes",
            "customer_name", "customer_phone", "customer_email",
            "customer_address", "calendar_event_id", "status",
            "confirmation_sent", "confirmation_sent_at",
            "reminder_sent", "reminder_sent_at",
        }

        # Filter to allowed fields only
        updates = {k: v for k, v in updates.items() if k in allowed_fields}
        if not updates:
            return await self.get_by_id(appointment_id)

        # Recalculate duration if times changed
        if "start_time" in updates and "end_time" in updates:
            updates["duration_minutes"] = int(
                (updates["end_time"] - updates["start_time"]).total_seconds() / 60
            )

        updates["updated_at"] = datetime.now(timezone.utc)

        try:
            set_clauses = []
            params = [appointment_id]
            for i, (field, value) in enumerate(updates.items(), start=2):
                set_clauses.append(f"{field} = ${i}")
                params.append(value)

            row = await pool.fetchrow(
                f"""
                UPDATE appointments
                SET {', '.join(set_clauses)}
                WHERE id = $1
                RETURNING *
                """,
                *params,
            )

            if row:
                logger.info("Updated appointment: %s", appointment_id)
                return self._row_to_dict(row)
            return None

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("update appointment", e)

    async def mark_confirmation_sent(self, appointment_id: UUID) -> bool:
        """Mark that confirmation was sent."""
        result = await self.update(
            appointment_id,
            confirmation_sent=True,
            confirmation_sent_at=datetime.now(timezone.utc),
        )
        return result is not None

    async def mark_reminder_sent(self, appointment_id: UUID) -> bool:
        """Mark that reminder was sent."""
        result = await self.update(
            appointment_id,
            reminder_sent=True,
            reminder_sent_at=datetime.now(timezone.utc),
        )
        return result is not None

    # === Message methods ===

    async def create_message(
        self,
        caller_phone: str,
        message_text: str,
        business_context_id: str,
        caller_name: Optional[str] = None,
        call_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict:
        """Create a voicemail/callback message."""
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("create message")

        message_id = uuid4()
        metadata_json = json.dumps(metadata or {})

        try:
            row = await pool.fetchrow(
                """
                INSERT INTO appointment_messages (
                    id, caller_phone, caller_name, message_text,
                    business_context_id, call_id, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                RETURNING *
                """,
                message_id,
                caller_phone,
                caller_name,
                message_text,
                business_context_id,
                call_id,
                metadata_json,
            )

            if row:
                logger.info("Created message %s from %s", message_id, caller_phone)
                return dict(row)

            raise DatabaseOperationError("create message", Exception("No row returned"))

        except (DatabaseUnavailableError, DatabaseOperationError):
            raise
        except Exception as e:
            logger.error("Failed to create message: %s", e)
            raise DatabaseOperationError("create message", e)

    async def get_unread_messages(
        self,
        business_context_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Get unread messages."""
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("get unread messages")

        try:
            conditions = ["read = FALSE"]
            params = []
            param_idx = 1

            if business_context_id:
                conditions.append(f"business_context_id = ${param_idx}")
                params.append(business_context_id)
                param_idx += 1

            params.append(limit)

            rows = await pool.fetch(
                f"""
                SELECT * FROM appointment_messages
                WHERE {' AND '.join(conditions)}
                ORDER BY created_at DESC
                LIMIT ${param_idx}
                """,
                *params,
            )

            return [dict(row) for row in rows]

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get unread messages", e)

    def _parse_row_count(self, result: str) -> int:
        """Parse row count from PostgreSQL command result."""
        if not result:
            return 0
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0

    def _row_to_dict(self, row) -> dict:
        """Convert a database row to a dict."""
        result = dict(row)
        # Parse metadata if it's a string
        if isinstance(result.get("metadata"), str):
            result["metadata"] = json.loads(result["metadata"])
        elif result.get("metadata") is None:
            result["metadata"] = {}
        return result


# Singleton instance
_appointment_repo: Optional[AppointmentRepository] = None


def get_appointment_repo() -> AppointmentRepository:
    """Get the global appointment repository."""
    global _appointment_repo
    if _appointment_repo is None:
        _appointment_repo = AppointmentRepository()
    return _appointment_repo
