"""
SMS message repository for inbound/outbound SMS persistence.

Provides CRUD operations for SMS messages stored in PostgreSQL.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID, uuid4

from ..database import get_db_pool
from ..exceptions import DatabaseUnavailableError, DatabaseOperationError

logger = logging.getLogger("atlas.storage.sms_message")

SMS_CONTACT_PROCESSING_LEASE_SECONDS = 300


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

    async def update_contact_processing_status(
        self,
        sms_id: UUID,
        status: str,
        *,
        owner_token: str,
        error_message: Optional[str] = None,
        clear_owner: bool = False,
    ) -> bool:
        """Update a leased contact-processing row without losing ownership.

        Returns True only when the caller still owns the processing lease.
        Non-terminal progress/status writes keep ``error_message`` as the
        owner token; terminal writes may clear it explicitly.
        """
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("update SMS contact processing status")

        try:
            row = await pool.fetchrow(
                """
                UPDATE sms_messages
                SET status = $2,
                    error_message = CASE WHEN $5 THEN NULL ELSE COALESCE($4, $3) END,
                    processed_at = $6
                WHERE id = $1
                  AND status = 'processing'
                  AND error_message = $3
                RETURNING id
                """,
                sms_id,
                status,
                owner_token,
                error_message,
                clear_owner,
                datetime.now(timezone.utc),
            )
            return row is not None
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("update SMS contact processing status", e)

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

    async def claim_contact_processing(
        self,
        sms_id: UUID,
        *,
        owner_token: Optional[str] = None,
    ) -> Optional[dict]:
        """Atomically claim CRM/contact processing and return authoritative row state.

        Returns the claimed row with ``_claim_acquired=True`` for the worker that
        moves an incomplete or abandoned row into the processing state. If the
        row exists but is not claimable, returns the current row with
        ``_claim_acquired=False`` so callers can distinguish active leases from
        already-terminal rows without racing a separate reload. Missing rows
        return ``None``.
        """
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("claim SMS contact processing")

        now = datetime.now(timezone.utc)
        lease_cutoff = now - timedelta(seconds=SMS_CONTACT_PROCESSING_LEASE_SECONDS)
        try:
            row = await pool.fetchrow(
                """
                WITH attempted AS (
                    UPDATE sms_messages
                    SET status = 'processing',
                        processed_at = $2,
                        error_message = $4
                    WHERE id = $1
                      AND (
                        status IN ('received', 'retry_pending')
                        OR (status = 'processing' AND (processed_at IS NULL OR processed_at < $3))
                      )
                    RETURNING sms_messages.*, TRUE AS claim_acquired
                ),
                current_row AS (
                    SELECT sms_messages.*, FALSE AS claim_acquired
                    FROM sms_messages
                    WHERE id = $1
                      AND NOT EXISTS (SELECT 1 FROM attempted)
                )
                SELECT * FROM attempted
                UNION ALL
                SELECT * FROM current_row
                LIMIT 1
                """,
                sms_id,
                now,
                lease_cutoff,
                owner_token,
            )
            if row is None:
                return None
            result = self._row_to_dict(row)
            result["_claim_acquired"] = bool(result.pop("claim_acquired", False))
            return result
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("claim SMS contact processing", e)

    async def claim_contact_processing_bool(
        self,
        sms_id: UUID,
        *,
        owner_token: Optional[str] = None,
    ) -> bool:
        """Compatibility helper for callers that only need claim ownership."""
        row = await self.claim_contact_processing(sms_id, owner_token=owner_token)
        return bool(row and row.get("_claim_acquired"))

    async def owns_contact_processing(self, sms_id: UUID, owner_token: str) -> bool:
        """Return True when the current processing row is still owned by this token."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("check SMS contact processing owner")

        try:
            row = await pool.fetchrow(
                """
                SELECT id
                FROM sms_messages
                WHERE id = $1
                  AND status = 'processing'
                  AND error_message = $2
                """,
                sms_id,
                owner_token,
            )
            return row is not None
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("check SMS contact processing owner", e)

    async def touch_contact_processing_owner(
        self,
        sms_id: UUID,
        owner_token: str,
    ) -> bool:
        """Heartbeat the processing lease only when this worker still owns it."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("touch SMS contact processing owner")

        try:
            row = await pool.fetchrow(
                """
                UPDATE sms_messages
                SET processed_at = $3
                WHERE id = $1
                  AND status = 'processing'
                  AND error_message = $2
                RETURNING id
                """,
                sms_id,
                owner_token,
                datetime.now(timezone.utc),
            )
            return row is not None
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("touch SMS contact processing owner", e)

    async def mark_contact_processing_retry_pending(
        self,
        sms_id: UUID,
        error_message: Optional[str] = None,
        *,
        owner_token: Optional[str] = None,
    ) -> None:
        """Record that SMS contact processing should be retried by a future delivery."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("mark SMS contact processing retry pending")

        try:
            now = datetime.now(timezone.utc)
            if owner_token is not None:
                await pool.execute(
                    """
                    UPDATE sms_messages
                    SET status = 'retry_pending',
                        error_message = $2,
                        processed_at = $3
                    WHERE id = $1
                      AND status = 'processing'
                      AND error_message = $4
                    """,
                    sms_id,
                    error_message,
                    now,
                    owner_token,
                )
            else:
                await pool.execute(
                    """
                    UPDATE sms_messages
                    SET status = 'retry_pending',
                        error_message = $2,
                        processed_at = $3
                    WHERE id = $1
                      AND contact_id IS NULL
                      AND status IN ('received', 'retry_pending')
                    """,
                    sms_id,
                    error_message,
                    now,
                )
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("mark SMS contact processing retry pending", e)

    async def mark_contact_processing_complete(
        self,
        sms_id: UUID,
        *,
        owner_token: Optional[str] = None,
    ) -> bool:
        """Mark a still-processing SMS row complete without changing linked/notify state.

        Returns True only when the terminal transition was durably written.
        """
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("mark SMS contact processing complete")

        try:
            now = datetime.now(timezone.utc)
            if owner_token is not None:
                row = await pool.fetchrow(
                    """
                    UPDATE sms_messages
                    SET status = 'ready',
                        processed_at = $2,
                        error_message = NULL
                    WHERE id = $1
                      AND status = 'processing'
                      AND error_message = $3
                    RETURNING id
                    """,
                    sms_id,
                    now,
                    owner_token,
                )
            else:
                row = await pool.fetchrow(
                    """
                    UPDATE sms_messages
                    SET status = 'ready',
                        processed_at = $2,
                        error_message = NULL
                    WHERE id = $1
                      AND status = 'processing'
                    RETURNING id
                    """,
                    sms_id,
                    now,
                )
            return row is not None
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("mark SMS contact processing complete", e)

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

    async def has_auto_reply_for_inbound(self, inbound_sms_id: UUID) -> bool:
        """Return True when an auto-reply send decision already exists."""
        return await self.get_auto_reply_for_inbound(inbound_sms_id) is not None

    async def get_auto_reply_for_inbound(self, inbound_sms_id: UUID) -> Optional[dict]:
        """Return the durable auto-reply outbox row for an inbound SMS, if any."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("check SMS auto-reply outbox")

        try:
            row = await pool.fetchrow(
                """
                SELECT *
                FROM sms_messages
                WHERE direction = 'outbound'
                  AND source = 'auto_reply'
                  AND source_ref = $1
                LIMIT 1
                """,
                str(inbound_sms_id),
            )
            return self._row_to_dict(row) if row else None
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("check SMS auto-reply outbox", e)

    async def reserve_auto_reply_for_inbound(
        self,
        *,
        inbound_sms_id: UUID,
        from_number: str,
        to_number: str,
        body: str,
        business_context_id: Optional[str],
    ) -> Optional[dict]:
        """Persist an outbound auto-reply send decision before provider contact."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("reserve SMS auto-reply")

        sms_id = uuid4()
        now = datetime.now(timezone.utc)
        message_sid = f"auto_reply_{inbound_sms_id}"
        try:
            row = await pool.fetchrow(
                """
                INSERT INTO sms_messages (
                    id, message_sid, from_number, to_number, direction,
                    body, media_urls, business_context_id, status,
                    source, source_ref, created_at
                )
                VALUES ($1, $2, $3, $4, 'outbound', $5, '[]'::jsonb, $6, 'pending', 'auto_reply', $7, $8)
                ON CONFLICT (message_sid) DO NOTHING
                RETURNING *
                """,
                sms_id,
                message_sid,
                from_number,
                to_number,
                body,
                business_context_id,
                str(inbound_sms_id),
                now,
            )
            return self._row_to_dict(row) if row else None
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("reserve SMS auto-reply", e)

    async def mark_auto_reply_sent(
        self,
        auto_reply_sms_id: UUID,
        *,
        provider_message_id: Optional[str] = None,
    ) -> bool:
        """Mark a reserved auto-reply as sent after provider acceptance.

        Returns True only when the pending outbox row was durably finalized.
        """
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("mark SMS auto-reply sent")

        try:
            row = await pool.fetchrow(
                """
                UPDATE sms_messages
                SET status = 'sent',
                    error_message = $2,
                    processed_at = $3
                WHERE id = $1
                  AND direction = 'outbound'
                  AND source = 'auto_reply'
                RETURNING id
                """,
                auto_reply_sms_id,
                provider_message_id,
                datetime.now(timezone.utc),
            )
            return row is not None
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("mark SMS auto-reply sent", e)

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
