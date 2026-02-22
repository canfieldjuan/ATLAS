"""
Call transcript repository for post-call intelligence.

Provides CRUD operations for call transcripts stored in PostgreSQL.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from ..database import get_db_pool
from ..exceptions import DatabaseUnavailableError, DatabaseOperationError

logger = logging.getLogger("atlas.storage.call_transcript")


class CallTranscriptRepository:
    """Repository for call transcript storage and retrieval."""

    async def create(
        self,
        call_sid: str,
        from_number: str,
        to_number: str,
        context_id: str,
        duration: int,
    ) -> dict:
        """Create a new call transcript record. Returns the created row as dict."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("create call transcript")

        transcript_id = uuid4()
        now = datetime.now(timezone.utc)

        try:
            row = await pool.fetchrow(
                """
                INSERT INTO call_transcripts (
                    id, call_sid, from_number, to_number,
                    business_context_id, duration_seconds, created_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING *
                """,
                transcript_id,
                call_sid,
                from_number,
                to_number,
                context_id,
                duration,
                now,
            )
            if row:
                logger.info("Created call transcript %s for call %s", transcript_id, call_sid)
                return self._row_to_dict(row)
            raise DatabaseOperationError("create call transcript", Exception("No row returned"))
        except (DatabaseUnavailableError, DatabaseOperationError):
            raise
        except Exception as e:
            logger.error("Failed to create call transcript: %s", e)
            raise DatabaseOperationError("create call transcript", e)

    async def update_transcript(self, transcript_id: UUID, transcript: str) -> None:
        """Store the ASR transcript text."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("update transcript")

        try:
            await pool.execute(
                """
                UPDATE call_transcripts
                SET transcript = $2, transcribed_at = $3
                WHERE id = $1
                """,
                transcript_id,
                transcript,
                datetime.now(timezone.utc),
            )
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("update transcript", e)

    async def update_extraction(
        self,
        transcript_id: UUID,
        summary: str,
        extracted_data: dict,
        proposed_actions: list,
    ) -> None:
        """Store LLM extraction results."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("update extraction")

        try:
            await pool.execute(
                """
                UPDATE call_transcripts
                SET summary = $2, extracted_data = $3::jsonb,
                    proposed_actions = $4::jsonb, processed_at = $5
                WHERE id = $1
                """,
                transcript_id,
                summary,
                json.dumps(extracted_data),
                json.dumps(proposed_actions),
                datetime.now(timezone.utc),
            )
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("update extraction", e)

    async def update_status(
        self,
        transcript_id: UUID,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        """Update processing status."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("update status")

        try:
            await pool.execute(
                """
                UPDATE call_transcripts
                SET status = $2, error_message = $3
                WHERE id = $1
                """,
                transcript_id,
                status,
                error_message,
            )
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("update status", e)

    async def save_draft(self, transcript_id: UUID, draft_type: str, content: str) -> None:
        """Merge a draft ('email' or 'sms') into the drafts JSONB column."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("save draft")

        try:
            await pool.execute(
                """
                UPDATE call_transcripts
                SET drafts = COALESCE(drafts, '{}'::jsonb)
                          || jsonb_build_object($2::text, $3::text)
                WHERE id = $1
                """,
                transcript_id,
                draft_type,
                content,
            )
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("save draft", e)

    async def link_contact(self, transcript_id: UUID, contact_id: str) -> None:
        """Set the CRM contact_id on a call transcript."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("link contact")

        try:
            await pool.execute(
                "UPDATE call_transcripts SET contact_id = $2 WHERE id = $1",
                transcript_id,
                contact_id,
            )
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("link contact", e)

    async def mark_notified(self, transcript_id: UUID) -> None:
        """Mark transcript as notified."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("mark notified")

        try:
            await pool.execute(
                "UPDATE call_transcripts SET notified = TRUE WHERE id = $1",
                transcript_id,
            )
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("mark notified", e)

    async def get_by_call_sid(self, call_sid: str) -> Optional[dict]:
        """Get a transcript by call SID."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get by call sid")

        try:
            row = await pool.fetchrow(
                "SELECT * FROM call_transcripts WHERE call_sid = $1",
                call_sid,
            )
            return self._row_to_dict(row) if row else None
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get by call sid", e)

    async def get_by_id(self, transcript_id: UUID) -> Optional[dict]:
        """Get a transcript by ID."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get by id")

        try:
            row = await pool.fetchrow(
                "SELECT * FROM call_transcripts WHERE id = $1",
                transcript_id,
            )
            return self._row_to_dict(row) if row else None
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get by id", e)

    async def get_by_contact_id(
        self, contact_id: str, limit: int = 20,
    ) -> list[dict]:
        """Get call transcripts linked to a CRM contact."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get by contact id")

        try:
            rows = await pool.fetch(
                """
                SELECT * FROM call_transcripts
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
            raise DatabaseOperationError("get by contact id", e)

    async def get_recent(
        self,
        business_context_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """Get recent transcripts, optionally filtered by business context."""
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("get recent")

        try:
            if business_context_id:
                rows = await pool.fetch(
                    """
                    SELECT * FROM call_transcripts
                    WHERE business_context_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    business_context_id,
                    limit,
                )
            else:
                rows = await pool.fetch(
                    """
                    SELECT * FROM call_transcripts
                    ORDER BY created_at DESC
                    LIMIT $1
                    """,
                    limit,
                )
            return [self._row_to_dict(row) for row in rows]
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get recent", e)

    async def search(
        self,
        keyword: Optional[str] = None,
        contact_id: Optional[str] = None,
        intent: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Search call transcripts with filters.

        Supports full-text search on transcript and summary fields,
        plus structured filters on contact, intent, and date range.
        """
        pool = get_db_pool()
        if not pool.is_initialized:
            raise DatabaseUnavailableError("search transcripts")

        conditions: list[str] = []
        params: list = []
        idx = 1

        if keyword:
            conditions.append(
                f"(transcript ILIKE ${idx} OR summary ILIKE ${idx})"
            )
            params.append(f"%{keyword}%")
            idx += 1
        if contact_id:
            conditions.append(f"contact_id = ${idx}")
            params.append(contact_id)
            idx += 1
        if intent:
            conditions.append(f"extracted_data->>'intent' = ${idx}")
            params.append(intent)
            idx += 1
        if from_date:
            conditions.append(f"created_at >= ${idx}")
            params.append(from_date)
            idx += 1
        if to_date:
            conditions.append(f"created_at <= ${idx}")
            params.append(to_date)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        try:
            rows = await pool.fetch(
                f"""
                SELECT * FROM call_transcripts
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
            raise DatabaseOperationError("search transcripts", e)

    def _row_to_dict(self, row) -> dict:
        """Convert a database row to a dict."""
        result = dict(row)
        # asyncpg returns JSONB as Python dicts/lists, but some pool wrappers
        # (e.g. pgbouncer, custom wrappers) may return JSON strings instead.
        for key in ("extracted_data", "drafts"):
            val = result.get(key)
            if val is None:
                result[key] = {}
            elif isinstance(val, str):
                try:
                    result[key] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    result[key] = {}
        val = result.get("proposed_actions")
        if val is None:
            result["proposed_actions"] = []
        elif isinstance(val, str):
            try:
                result["proposed_actions"] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                result["proposed_actions"] = []
        return result


_call_transcript_repo: Optional[CallTranscriptRepository] = None


def get_call_transcript_repo() -> CallTranscriptRepository:
    """Get the global call transcript repository."""
    global _call_transcript_repo
    if _call_transcript_repo is None:
        _call_transcript_repo = CallTranscriptRepository()
    return _call_transcript_repo
