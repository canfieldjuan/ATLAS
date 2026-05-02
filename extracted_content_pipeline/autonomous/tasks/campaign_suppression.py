"""Campaign suppression compatibility helpers for standalone extraction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any
from uuid import UUID


@dataclass(frozen=True)
class AssignmentResult:
    assigned: bool
    sequence_id: UUID
    conflict_with_sequence_id: UUID | None = None
    reason: str | None = None


def _normalize_email(email: str | None) -> str:
    return str(email or "").strip().lower()


def _email_domain(email: str) -> str:
    _, _, domain = email.partition("@")
    return domain.strip().lower()


def _recipient_lock_key(email: str) -> int:
    digest = hashlib.blake2b(email.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=True)


async def is_suppressed(pool: Any, *, email: str) -> dict[str, Any] | None:
    """Return an active suppression row for an email or its domain."""
    normalized = _normalize_email(email)
    if not normalized:
        return None
    row = await pool.fetchrow(
        """
        SELECT *
        FROM campaign_suppressions
        WHERE (expires_at IS NULL OR expires_at > NOW())
          AND LOWER(email) = $1
        ORDER BY created_at DESC
        LIMIT 1
        """,
        normalized,
    )
    if row:
        return dict(row)

    domain = _email_domain(normalized)
    if not domain:
        return None
    row = await pool.fetchrow(
        """
        SELECT *
        FROM campaign_suppressions
        WHERE (expires_at IS NULL OR expires_at > NOW())
          AND LOWER(domain) = $1
        ORDER BY created_at DESC
        LIMIT 1
        """,
        domain,
    )
    return dict(row) if row else None


async def _assign_recipient_to_sequence_locked(
    conn: Any,
    *,
    sequence_id: UUID,
    email: str,
) -> AssignmentResult:
    conflict_id = await conn.fetchval(
        """
        SELECT id
        FROM campaign_sequences
        WHERE LOWER(BTRIM(recipient_email)) = $1
          AND status = 'active'
          AND id != $2
        LIMIT 1
        """,
        email,
        sequence_id,
    )
    if conflict_id:
        return AssignmentResult(
            False,
            sequence_id,
            conflict_with_sequence_id=(
                conflict_id if isinstance(conflict_id, UUID) else UUID(str(conflict_id))
            ),
            reason="recipient_already_assigned",
        )

    result = await conn.execute(
        """
        UPDATE campaign_sequences
        SET recipient_email = $2,
            updated_at = NOW()
        WHERE id = $1
          AND status = 'active'
        """,
        sequence_id,
        email,
    )
    return AssignmentResult(
        str(result).upper() == "UPDATE 1",
        sequence_id,
        reason=None if str(result).upper() == "UPDATE 1" else "sequence_not_active",
    )


async def assign_recipient_to_sequence(
    pool: Any,
    sequence_id: UUID | str,
    email: str,
) -> AssignmentResult:
    """Assign a recipient to one active sequence unless another owns it."""
    sid = sequence_id if isinstance(sequence_id, UUID) else UUID(str(sequence_id))
    normalized = _normalize_email(email)
    if not normalized:
        return AssignmentResult(False, sid, reason="empty_email")

    if hasattr(pool, "acquire"):
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT pg_advisory_xact_lock($1)",
                    _recipient_lock_key(normalized),
                )
                return await _assign_recipient_to_sequence_locked(
                    conn,
                    sequence_id=sid,
                    email=normalized,
                )

    return await _assign_recipient_to_sequence_locked(
        pool,
        sequence_id=sid,
        email=normalized,
    )
