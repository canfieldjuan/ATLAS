"""
Reminder repository for persistence and retrieval.

Provides CRUD operations for reminders stored in PostgreSQL.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from ..database import get_db_pool
from ..exceptions import DatabaseUnavailableError, DatabaseOperationError
from ..models import Reminder

logger = logging.getLogger("atlas.storage.reminder")


class ReminderRepository:
    """
    Repository for reminder storage and retrieval.

    Handles persistence of user reminders with efficient queries
    for finding due reminders.
    """

    async def create(
        self,
        message: str,
        due_at: datetime,
        user_id: Optional[UUID] = None,
        repeat_pattern: Optional[str] = None,
        source: str = "voice",
        metadata: Optional[dict[str, Any]] = None,
    ) -> Reminder:
        """Create a new reminder. Raises DatabaseUnavailableError if DB not ready."""
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("create reminder")

        reminder_id = uuid4()
        metadata_json = json.dumps(metadata or {})

        try:
            row = await pool.fetchrow(
                """
                INSERT INTO reminders (
                    id, message, due_at, user_id, created_at,
                    repeat_pattern, source, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
                RETURNING id, created_at
                """,
                reminder_id,
                message,
                due_at,
                user_id,
                datetime.now(timezone.utc),
                repeat_pattern,
                source,
                metadata_json,
            )

            if row:
                logger.info("Created reminder %s: %s (due: %s)", reminder_id, message, due_at)
                return Reminder(
                    id=row["id"],
                    message=message,
                    due_at=due_at,
                    user_id=user_id,
                    created_at=row["created_at"],
                    repeat_pattern=repeat_pattern,
                    source=source,
                    metadata=metadata or {},
                )

            raise DatabaseOperationError("create reminder", Exception("No row returned"))

        except (DatabaseUnavailableError, DatabaseOperationError):
            raise
        except Exception as e:
            logger.error("Failed to create reminder: %s", e)
            raise DatabaseOperationError("create reminder", e)

    async def get_by_id(self, reminder_id: UUID) -> Optional[Reminder]:
        """Get a reminder by ID. Returns None if not found."""
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("get reminder by id")

        try:
            row = await pool.fetchrow(
                """
                SELECT id, message, due_at, user_id, created_at,
                       completed, completed_at, delivered, delivered_at,
                       repeat_pattern, source, metadata
                FROM reminders
                WHERE id = $1
                """,
                reminder_id,
            )

            if row:
                return self._row_to_reminder(row)
            return None

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get reminder by id", e)

    async def get_pending(
        self,
        before: Optional[datetime] = None,
        user_id: Optional[UUID] = None,
        limit: int = 100,
    ) -> list[Reminder]:
        """
        Get pending reminders (not completed, not delivered).

        Args:
            before: Only get reminders due before this time
            user_id: Filter by user
            limit: Maximum number to return
        """
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("get pending reminders")

        try:
            conditions = ["completed = FALSE", "delivered = FALSE"]
            params = []
            param_idx = 1

            if before:
                conditions.append(f"due_at <= ${param_idx}")
                params.append(before)
                param_idx += 1

            if user_id:
                conditions.append(f"user_id = ${param_idx}")
                params.append(user_id)
                param_idx += 1

            where_clause = " AND ".join(conditions)
            params.append(limit)

            rows = await pool.fetch(
                f"""
                SELECT id, message, due_at, user_id, created_at,
                       completed, completed_at, delivered, delivered_at,
                       repeat_pattern, source, metadata
                FROM reminders
                WHERE {where_clause}
                ORDER BY due_at ASC
                LIMIT ${param_idx}
                """,
                *params,
            )

            return [self._row_to_reminder(row) for row in rows]

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get pending reminders", e)

    async def get_due(self, as_of: Optional[datetime] = None) -> list[Reminder]:
        """
        Get reminders that are due for delivery.

        This is the critical query for the scheduler.
        """
        if as_of is None:
            as_of = datetime.now(timezone.utc)

        return await self.get_pending(before=as_of)

    async def get_user_reminders(
        self,
        user_id: Optional[UUID] = None,
        include_completed: bool = False,
        limit: int = 50,
    ) -> list[Reminder]:
        """Get reminders for a user."""
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("get user reminders")

        try:
            conditions = []
            params = []
            param_idx = 1

            if user_id:
                conditions.append(f"user_id = ${param_idx}")
                params.append(user_id)
                param_idx += 1

            if not include_completed:
                conditions.append("completed = FALSE")

            where_clause = ""
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)

            params.append(limit)

            rows = await pool.fetch(
                f"""
                SELECT id, message, due_at, user_id, created_at,
                       completed, completed_at, delivered, delivered_at,
                       repeat_pattern, source, metadata
                FROM reminders
                {where_clause}
                ORDER BY due_at ASC
                LIMIT ${param_idx}
                """,
                *params,
            )

            return [self._row_to_reminder(row) for row in rows]

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get user reminders", e)

    def _parse_row_count(self, result: str) -> int:
        """Parse row count from PostgreSQL command result."""
        if not result:
            return 0
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0

    async def mark_delivered(
        self,
        reminder_id: UUID,
        delivered_at: Optional[datetime] = None,
    ) -> bool:
        """Mark a reminder as delivered."""
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("mark reminder delivered")

        try:
            if delivered_at is None:
                delivered_at = datetime.now(timezone.utc)

            result = await pool.execute(
                """
                UPDATE reminders
                SET delivered = TRUE, delivered_at = $2
                WHERE id = $1 AND delivered = FALSE
                """,
                reminder_id,
                delivered_at,
            )

            success = self._parse_row_count(result) > 0
            if success:
                logger.info("Marked reminder as delivered: %s", reminder_id)
            return success

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("mark reminder delivered", e)

    async def mark_completed(
        self,
        reminder_id: UUID,
        completed_at: Optional[datetime] = None,
    ) -> bool:
        """Mark a reminder as completed (acknowledged by user)."""
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("mark reminder completed")

        try:
            if completed_at is None:
                completed_at = datetime.now(timezone.utc)

            result = await pool.execute(
                """
                UPDATE reminders
                SET completed = TRUE, completed_at = $2
                WHERE id = $1 AND completed = FALSE
                """,
                reminder_id,
                completed_at,
            )

            success = self._parse_row_count(result) > 0
            if success:
                logger.info("Marked reminder as completed: %s", reminder_id)
            return success

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("mark reminder completed", e)

    async def delete(self, reminder_id: UUID) -> bool:
        """Delete a reminder."""
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("delete reminder")

        try:
            result = await pool.execute(
                "DELETE FROM reminders WHERE id = $1",
                reminder_id,
            )

            success = self._parse_row_count(result) > 0
            if success:
                logger.info("Deleted reminder: %s", reminder_id)
            return success

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("delete reminder", e)

    async def delete_old_completed(
        self,
        older_than_days: int = 30,
    ) -> int:
        """Delete completed reminders older than N days."""
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("delete old completed reminders")

        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)

            result = await pool.execute(
                """
                DELETE FROM reminders
                WHERE completed = TRUE AND completed_at < $1
                """,
                cutoff,
            )

            count = self._parse_row_count(result)
            if count > 0:
                logger.info("Deleted %d old completed reminders", count)
            return count

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("delete old completed reminders", e)

    def _calculate_next_due(
        self,
        reminder: Reminder,
    ) -> Optional[datetime]:
        """
        Calculate the next due time for a recurring reminder.

        Uses dateutil.relativedelta for accurate month/year arithmetic.
        """
        from dateutil.relativedelta import relativedelta

        if not reminder.repeat_pattern:
            return None

        if reminder.repeat_pattern == "daily":
            return reminder.due_at + relativedelta(days=1)
        elif reminder.repeat_pattern == "weekly":
            return reminder.due_at + relativedelta(weeks=1)
        elif reminder.repeat_pattern == "monthly":
            # Proper month arithmetic - handles month-end dates correctly
            # e.g., Jan 31 + 1 month = Feb 28 (or 29 in leap year)
            return reminder.due_at + relativedelta(months=1)
        elif reminder.repeat_pattern == "yearly":
            return reminder.due_at + relativedelta(years=1)
        else:
            logger.warning("Unknown repeat pattern: %s", reminder.repeat_pattern)
            return None

    async def reschedule_recurring(
        self,
        reminder: Reminder,
    ) -> Optional[Reminder]:
        """
        Create the next occurrence of a recurring reminder.

        Called after a recurring reminder is delivered.
        NOTE: For atomic delivery + reschedule, use deliver_recurring() instead.
        """
        next_due = self._calculate_next_due(reminder)
        if next_due is None:
            return None

        return await self.create(
            message=reminder.message,
            due_at=next_due,
            user_id=reminder.user_id,
            repeat_pattern=reminder.repeat_pattern,
            source="scheduled",
            metadata=reminder.metadata,
        )

    async def deliver_recurring(
        self,
        reminder: Reminder,
        delivered_at: Optional[datetime] = None,
    ) -> Optional[Reminder]:
        """
        Atomically mark reminder as delivered and create next occurrence.

        This ensures that recurring reminders are never lost due to partial
        failures. Either both operations succeed or neither does.

        Args:
            reminder: The reminder being delivered
            delivered_at: Override delivery timestamp (defaults to now)

        Returns:
            The newly created next occurrence, or None if not recurring
        """
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("deliver recurring reminder")

        if not reminder.repeat_pattern:
            # Not recurring, just mark delivered
            await self.mark_delivered(reminder.id, delivered_at)
            return None

        next_due = self._calculate_next_due(reminder)
        if next_due is None:
            await self.mark_delivered(reminder.id, delivered_at)
            return None

        if delivered_at is None:
            delivered_at = datetime.now(timezone.utc)

        try:
            async with pool.transaction() as conn:
                # Mark current reminder as delivered
                await conn.execute(
                    """
                    UPDATE reminders
                    SET delivered = TRUE, delivered_at = $2
                    WHERE id = $1 AND delivered = FALSE
                    """,
                    reminder.id,
                    delivered_at,
                )
                logger.info("Marked reminder as delivered: %s", reminder.id)

                # Create next occurrence
                next_id = uuid4()
                metadata_json = json.dumps(reminder.metadata or {})
                created_at = datetime.now(timezone.utc)

                row = await conn.fetchrow(
                    """
                    INSERT INTO reminders (
                        id, message, due_at, user_id, created_at,
                        repeat_pattern, source, metadata
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
                    RETURNING id, created_at
                    """,
                    next_id,
                    reminder.message,
                    next_due,
                    reminder.user_id,
                    created_at,
                    reminder.repeat_pattern,
                    "scheduled",
                    metadata_json,
                )

                if row:
                    logger.info(
                        "Created next recurring reminder %s (due: %s)",
                        next_id,
                        next_due,
                    )
                    return Reminder(
                        id=row["id"],
                        message=reminder.message,
                        due_at=next_due,
                        user_id=reminder.user_id,
                        created_at=row["created_at"],
                        repeat_pattern=reminder.repeat_pattern,
                        source="scheduled",
                        metadata=reminder.metadata or {},
                    )

                raise DatabaseOperationError(
                    "deliver recurring reminder",
                    Exception("Failed to create next occurrence"),
                )

        except (DatabaseUnavailableError, DatabaseOperationError):
            raise
        except Exception as e:
            logger.error("Failed to deliver recurring reminder: %s", e)
            raise DatabaseOperationError("deliver recurring reminder", e)

    def _row_to_reminder(self, row) -> Reminder:
        """Convert a database row to a Reminder object."""
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        elif metadata is None:
            metadata = {}

        return Reminder(
            id=row["id"],
            message=row["message"],
            due_at=row["due_at"],
            user_id=row["user_id"],
            created_at=row["created_at"],
            completed=row["completed"],
            completed_at=row["completed_at"],
            delivered=row["delivered"],
            delivered_at=row["delivered_at"],
            repeat_pattern=row["repeat_pattern"],
            source=row["source"] or "voice",
            metadata=metadata,
        )


_reminder_repo: Optional[ReminderRepository] = None


def get_reminder_repo() -> ReminderRepository:
    """Get the global reminder repository."""
    global _reminder_repo
    if _reminder_repo is None:
        _reminder_repo = ReminderRepository()
    return _reminder_repo
