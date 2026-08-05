"""
Email repository for persistence and retrieval.

Provides CRUD operations for sent email history stored in PostgreSQL.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from ..database import get_db_pool
from ..exceptions import DatabaseUnavailableError, DatabaseOperationError
from ..models import SentEmail

logger = logging.getLogger("atlas.storage.email")


class EmailRepository:
    """
    Repository for sent email storage and retrieval.

    Handles persistence of sent emails for history queries.
    """

    def __init__(self, *, pool: Any | None = None) -> None:
        """Use the configured pool, or a caller-supplied store adapter.

        The EOM funnel runs against its own connection string; evidence
        writes must land in the same store that owns the record they
        describe, not whatever the global pool points at.
        """
        self._pool_override = pool

    def _get_pool(self) -> Any:
        if self._pool_override is not None:
            return self._pool_override
        return get_db_pool()

    async def create(
        self,
        to_addresses: list[str],
        subject: str,
        body: str,
        template_type: Optional[str] = None,
        session_id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
        cc_addresses: Optional[list[str]] = None,
        attachments: Optional[list[str]] = None,
        resend_message_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        business_context_id: Optional[str] = None,
    ) -> SentEmail:
        """Create a new sent email record."""
        pool = self._get_pool()

        if not getattr(pool, "is_initialized", True):
            raise DatabaseUnavailableError("create sent email")

        email_id = uuid4()
        metadata_json = json.dumps(metadata or {})
        normalized_context = self._normalize_business_context_id(
            business_context_id
        )

        try:
            row = await pool.fetchrow(
                """
                INSERT INTO sent_emails (
                    id, to_addresses, cc_addresses, subject, body,
                    template_type, session_id, user_id, attachments,
                    resend_message_id, sent_at, metadata, business_context_id
                )
                VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                    $12::jsonb, $13
                )
                RETURNING id, sent_at
                """,
                email_id,
                to_addresses,
                cc_addresses or [],
                subject,
                body,
                template_type,
                session_id,
                user_id,
                attachments or [],
                resend_message_id,
                datetime.now(timezone.utc),
                metadata_json,
                normalized_context,
            )

            if row:
                logger.info(
                    "Saved sent email %s: template=%s context=%s",
                    email_id,
                    template_type,
                    normalized_context,
                )
                return SentEmail(
                    id=row["id"],
                    to_addresses=to_addresses,
                    cc_addresses=cc_addresses or [],
                    subject=subject,
                    body=body,
                    template_type=template_type,
                    session_id=session_id,
                    user_id=user_id,
                    attachments=attachments or [],
                    resend_message_id=resend_message_id,
                    sent_at=row["sent_at"],
                    metadata=metadata or {},
                    business_context_id=normalized_context,
                )

            raise DatabaseOperationError("create sent email", Exception("No row returned"))

        except (DatabaseUnavailableError, DatabaseOperationError):
            raise
        except Exception as e:
            logger.error("Failed to create sent email: %s", e)
            raise DatabaseOperationError("create sent email", e)

    async def get_by_id(
        self,
        email_id: UUID,
        business_context_id: Optional[str] = None,
    ) -> Optional[SentEmail]:
        """Get a sent email by ID."""
        pool = self._get_pool()

        if not getattr(pool, "is_initialized", True):
            raise DatabaseUnavailableError("get sent email by id")

        try:
            params: list[Any] = [email_id]
            context_condition = ""
            if business_context_id is not None:
                context_condition = " AND business_context_id = $2"
                params.append(
                    self._normalize_business_context_id(business_context_id)
                )

            row = await pool.fetchrow(
                f"""
                SELECT id, to_addresses, cc_addresses, subject, body,
                       template_type, session_id, user_id, attachments,
                       resend_message_id, sent_at, metadata,
                       business_context_id
                FROM sent_emails
                WHERE id = $1{context_condition}
                """,
                *params,
            )

            if row:
                return self._row_to_email(row)
            return None

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get sent email by id", e)

    async def query(
        self,
        user_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None,
        template_type: Optional[str] = None,
        to_address: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0,
        business_context_id: Optional[str] = None,
        contact_id: Optional[str] = None,
    ) -> list[SentEmail]:
        """
        Query sent emails with filters.

        Args:
            user_id: Filter by user
            session_id: Filter by session
            template_type: Filter by template type
            to_address: Filter by recipient (partial match)
            since: Filter emails sent after this time
            until: Filter emails sent before this time
            limit: Maximum number to return
            offset: Number of records to skip
            business_context_id: Exact tenant scope; excludes NULL legacy rows
            contact_id: Match history linked through metadata; when combined
                with to_address, either identity may match
        """
        limit = min(limit, 500)
        pool = self._get_pool()

        if not getattr(pool, "is_initialized", True):
            raise DatabaseUnavailableError("query sent emails")

        try:
            conditions = []
            params = []
            param_idx = 1

            if user_id:
                conditions.append(f"user_id = ${param_idx}")
                params.append(user_id)
                param_idx += 1

            if session_id:
                conditions.append(f"session_id = ${param_idx}")
                params.append(session_id)
                param_idx += 1

            if template_type:
                conditions.append(f"template_type = ${param_idx}")
                params.append(template_type)
                param_idx += 1

            identity_conditions = []
            if to_address:
                identity_conditions.append(f"${param_idx} = ANY(to_addresses)")
                params.append(to_address)
                param_idx += 1

            if contact_id:
                identity_conditions.append(
                    f"metadata->>'contact_id' = ${param_idx}"
                )
                params.append(str(contact_id))
                param_idx += 1

            if identity_conditions:
                conditions.append(
                    "(" + " OR ".join(identity_conditions) + ")"
                )

            if since:
                conditions.append(f"sent_at >= ${param_idx}")
                params.append(since)
                param_idx += 1

            if until:
                conditions.append(f"sent_at <= ${param_idx}")
                params.append(until)
                param_idx += 1

            if business_context_id is not None:
                conditions.append(f"business_context_id = ${param_idx}")
                params.append(
                    self._normalize_business_context_id(business_context_id)
                )
                param_idx += 1

            where_clause = ""
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)

            params.append(limit)
            limit_idx = param_idx
            param_idx += 1

            params.append(offset)
            offset_idx = param_idx

            rows = await pool.fetch(
                f"""
                SELECT id, to_addresses, cc_addresses, subject, body,
                       template_type, session_id, user_id, attachments,
                       resend_message_id, sent_at, metadata,
                       business_context_id
                FROM sent_emails
                {where_clause}
                ORDER BY sent_at DESC
                LIMIT ${limit_idx} OFFSET ${offset_idx}
                """,
                *params,
            )

            return [self._row_to_email(row) for row in rows]

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("query sent emails", e)

    async def get_recent(
        self,
        hours: int = 24,
        user_id: Optional[UUID] = None,
        limit: int = 20,
        business_context_id: Optional[str] = None,
    ) -> list[SentEmail]:
        """Get emails sent in the last N hours."""
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        return await self.query(
            user_id=user_id,
            since=since,
            limit=limit,
            business_context_id=business_context_id,
        )

    async def get_today(
        self,
        user_id: Optional[UUID] = None,
        limit: int = 50,
        business_context_id: Optional[str] = None,
    ) -> list[SentEmail]:
        """Get emails sent today."""
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return await self.query(
            user_id=user_id,
            since=today_start,
            limit=limit,
            business_context_id=business_context_id,
        )

    async def count(
        self,
        user_id: Optional[UUID] = None,
        template_type: Optional[str] = None,
        since: Optional[datetime] = None,
        business_context_id: Optional[str] = None,
    ) -> int:
        """Count emails matching filters."""
        pool = self._get_pool()

        if not getattr(pool, "is_initialized", True):
            raise DatabaseUnavailableError("count sent emails")

        try:
            conditions = []
            params = []
            param_idx = 1

            if user_id:
                conditions.append(f"user_id = ${param_idx}")
                params.append(user_id)
                param_idx += 1

            if template_type:
                conditions.append(f"template_type = ${param_idx}")
                params.append(template_type)
                param_idx += 1

            if since:
                conditions.append(f"sent_at >= ${param_idx}")
                params.append(since)
                param_idx += 1

            if business_context_id is not None:
                conditions.append(f"business_context_id = ${param_idx}")
                params.append(
                    self._normalize_business_context_id(business_context_id)
                )
                param_idx += 1

            where_clause = ""
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)

            row = await pool.fetchrow(
                f"SELECT COUNT(*) as count FROM sent_emails {where_clause}",
                *params,
            )

            return row["count"] if row else 0

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("count sent emails", e)

    async def delete_old(self, older_than_days: int = 90) -> int:
        """Delete emails older than N days."""
        pool = self._get_pool()

        if not getattr(pool, "is_initialized", True):
            raise DatabaseUnavailableError("delete old sent emails")

        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)

            result = await pool.execute(
                "DELETE FROM sent_emails WHERE sent_at < $1",
                cutoff,
            )

            count = self._parse_row_count(result)
            if count > 0:
                logger.info("Deleted %d old sent emails", count)
            return count

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("delete old sent emails", e)

    def _parse_row_count(self, result: str) -> int:
        """Parse row count from PostgreSQL command result."""
        if not result:
            return 0
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0

    @staticmethod
    def _normalize_business_context_id(value: Optional[str]) -> Optional[str]:
        """Validate and normalize an explicitly supplied tenant key."""
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("business_context_id must not be blank")
        if len(normalized) > 64:
            raise ValueError("business_context_id must be at most 64 characters")
        return normalized

    def _row_to_email(self, row) -> SentEmail:
        """Convert a database row to a SentEmail object."""
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        elif metadata is None:
            metadata = {}

        return SentEmail(
            id=row["id"],
            to_addresses=list(row["to_addresses"]) if row["to_addresses"] else [],
            cc_addresses=list(row["cc_addresses"]) if row["cc_addresses"] else [],
            subject=row["subject"],
            body=row["body"],
            template_type=row["template_type"],
            session_id=row["session_id"],
            user_id=row["user_id"],
            attachments=list(row["attachments"]) if row["attachments"] else [],
            resend_message_id=row["resend_message_id"],
            sent_at=row["sent_at"],
            metadata=metadata,
            business_context_id=row["business_context_id"],
        )


_email_repo: Optional[EmailRepository] = None


def get_email_repo() -> EmailRepository:
    """Get the global email repository."""
    global _email_repo
    if _email_repo is None:
        _email_repo = EmailRepository()
    return _email_repo
