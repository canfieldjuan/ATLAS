"""
SMS message repository for inbound/outbound SMS persistence.

Provides CRUD operations for SMS messages stored in PostgreSQL.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from ..database import get_db_pool
from ..exceptions import DatabaseUnavailableError, DatabaseOperationError

logger = logging.getLogger("atlas.storage.sms_message")


class SMSMessageRepository:
    """Repository for SMS message storage and retrieval."""

    async def create(
        self,
        message_sid: str,
        from_number: str,
        to_number: str,
        direction: str = "inbound",
        body: str = "",
        media_urls: Optional[list] = None,
        business_context_id: Optional[str] = None,
        status: Optional[str] = None,
        source: Optional[str] = None,
        source_ref: Optional[str] = None,
    ) -> dict:
        """Create a new SMS message record. Returns the created row as dict."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("create SMS message")

        sms_id = uuid4()
        now = datetime.now(timezone.utc)
        if status is None:
            status = "received" if direction == "inbound" else "pending"

        try:
            row = await pool.fetchrow(
                """
                INSERT INTO sms_messages (
                    id, message_sid, from_number, to_number, direction,
                    body, media_urls, business_context_id, status,
                    source, source_ref, created_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11, $12)
                RETURNING *
                """,
                sms_id,
                message_sid,
                from_number,
                to_number,
                direction,
                body,
                json.dumps(media_urls or []),
                business_context_id,
                status,
                source,
                source_ref,
                now,
            )
            if row:
                logger.info("Created SMS message %s (%s) sid=%s", sms_id, direction, message_sid)
                return self._row_to_dict(row)
            raise DatabaseOperationError("create SMS message", Exception("No row returned"))
        except (DatabaseUnavailableError, DatabaseOperationError):
            raise
        except Exception as e:
            logger.error("Failed to create SMS message: %s", e)
            raise DatabaseOperationError("create SMS message", e)

    async def update_status(
        self,
        sms_id: UUID,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        """Update message status."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("update SMS status")

        try:
            await pool.execute(
                """
                UPDATE sms_messages
                SET status = $2, error_message = $3
                WHERE id = $1
                """,
                sms_id,
                status,
                error_message,
            )
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("update SMS status", e)

    async def update_delivery(self, sms_id: UUID, delivered_at: datetime) -> None:
        """Mark message as delivered with timestamp."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("update SMS delivery")

        try:
            await pool.execute(
                """
                UPDATE sms_messages
                SET status = 'delivered', delivered_at = $2
                WHERE id = $1
                """,
                sms_id,
                delivered_at,
            )
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("update SMS delivery", e)

    async def update_extraction(
        self,
        sms_id: UUID,
        summary: Optional[str] = None,
        extracted_data: Optional[dict] = None,
        intent: Optional[str] = None,
    ) -> None:
        """Store LLM extraction results."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("update SMS extraction")

        try:
            await pool.execute(
                """
                UPDATE sms_messages
                SET summary = COALESCE($2, summary),
                    extracted_data = COALESCE($3::jsonb, extracted_data),
                    intent = COALESCE($4, intent),
                    processed_at = $5
                WHERE id = $1
                """,
                sms_id,
                summary,
                json.dumps(extracted_data) if extracted_data is not None else None,
                intent,
                datetime.now(timezone.utc),
            )
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("update SMS extraction", e)

    async def link_contact(self, sms_id: UUID, contact_id: str) -> None:
        """Set the CRM contact_id on an SMS message."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("link SMS contact")

        try:
            await pool.execute(
                "UPDATE sms_messages SET contact_id = $2 WHERE id = $1",
                sms_id,
                contact_id,
            )
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("link SMS contact", e)

    async def mark_notified(self, sms_id: UUID) -> None:
        """Mark message as notified."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("mark SMS notified")

        try:
            await pool.execute(
                "UPDATE sms_messages SET notified = TRUE WHERE id = $1",
                sms_id,
            )
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("mark SMS notified", e)

    async def get_by_message_sid(self, message_sid: str) -> Optional[dict]:
        """Get a message by provider message SID."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get SMS by message sid")

        try:
            row = await pool.fetchrow(
                "SELECT * FROM sms_messages WHERE message_sid = $1",
                message_sid,
            )
            return self._row_to_dict(row) if row else None
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get SMS by message sid", e)

    async def get_by_id(self, sms_id: UUID) -> Optional[dict]:
        """Get a message by ID."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get SMS by id")

        try:
            row = await pool.fetchrow(
                "SELECT * FROM sms_messages WHERE id = $1",
                sms_id,
            )
            return self._row_to_dict(row) if row else None
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get SMS by id", e)

    async def get_by_contact_id(
        self, contact_id: str, limit: int = 20,
    ) -> list[dict]:
        """Get SMS messages linked to a CRM contact."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get SMS by contact id")

        try:
            rows = await pool.fetch(
                """
                SELECT * FROM sms_messages
                WHERE contact_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                contact_id,
                limit,
            )
            return [self._row_to_dict(row) for row in rows]
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get SMS by contact id", e)

    async def get_by_phone_pair(
        self,
        phone_a: str,
        phone_b: str,
        limit: int = 50,
    ) -> list[dict]:
        """Get conversation between two phone numbers (both directions)."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get SMS by phone pair")

        try:
            rows = await pool.fetch(
                """
                SELECT * FROM sms_messages
                WHERE (from_number = $1 AND to_number = $2)
                   OR (from_number = $2 AND to_number = $1)
                ORDER BY created_at DESC
                LIMIT $3
                """,
                phone_a,
                phone_b,
                limit,
            )
            return [self._row_to_dict(row) for row in rows]
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get SMS by phone pair", e)

    async def get_recent(
        self,
        business_context_id: Optional[str] = None,
        direction: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """Get recent messages, optionally filtered by business context and direction."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get recent SMS")

        conditions = []
        params: list = []
        idx = 1

        if business_context_id:
            conditions.append(f"business_context_id = ${idx}")
            params.append(business_context_id)
            idx += 1
        if direction:
            conditions.append(f"direction = ${idx}")
            params.append(direction)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        try:
            rows = await pool.fetch(
                f"""
                SELECT * FROM sms_messages
                {where}
                ORDER BY created_at DESC
                LIMIT ${idx}
                """,
                *params,
            )
            return [self._row_to_dict(row) for row in rows]
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get recent SMS", e)

    def _row_to_dict(self, row) -> dict:
        """Convert a database row to a dict."""
        result = dict(row)
        # Handle JSONB fields (asyncpg returns dicts, but some wrappers may return strings)
        for key in ("extracted_data",):
            val = result.get(key)
            if val is None:
                result[key] = {}
            elif isinstance(val, str):
                try:
                    result[key] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    result[key] = {}
        for list_key in ("media_urls",):
            val = result.get(list_key)
            if val is None:
                result[list_key] = []
            elif isinstance(val, str):
                try:
                    result[list_key] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    result[list_key] = []
        return result


_sms_message_repo: Optional[SMSMessageRepository] = None


def get_sms_message_repo() -> SMSMessageRepository:
    """Get the global SMS message repository."""
    global _sms_message_repo
    if _sms_message_repo is None:
        _sms_message_repo = SMSMessageRepository()
    return _sms_message_repo
