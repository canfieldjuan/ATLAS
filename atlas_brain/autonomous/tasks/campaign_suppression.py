"""
Campaign suppression helpers.

Provides add_suppression() and is_suppressed() for the global
do-not-contact / blocklist system, and assign_recipient_to_sequence()
for cross-sequence person dedup at recipient-attachment time.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

logger = logging.getLogger("atlas.autonomous.tasks.campaign_suppression")


@dataclass
class AssignmentResult:
    """Outcome of attaching a recipient_email to a campaign_sequences row."""

    assigned: bool
    sequence_id: UUID
    conflict_with_sequence_id: UUID | None = None
    reason: str | None = None


async def add_suppression(
    pool,
    *,
    email: str | None = None,
    domain: str | None = None,
    reason: str,
    source: str = "system",
    campaign_id: UUID | str | None = None,
    notes: str | None = None,
    expires_at: datetime | None = None,
) -> UUID | None:
    """Insert a suppression record. Returns ID, or None if duplicate."""
    if not email and not domain:
        return None

    cid = None
    if campaign_id:
        cid = campaign_id if isinstance(campaign_id, UUID) else UUID(str(campaign_id))

    try:
        email_val = email.lower().strip() if email else None
        domain_val = domain.lower().strip() if domain else None

        row = await pool.fetchrow(
            """
            INSERT INTO campaign_suppressions
                (email, domain, reason, source, campaign_id, notes, expires_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT ((LOWER(email))) WHERE email IS NOT NULL
            DO UPDATE SET
                reason = EXCLUDED.reason,
                source = EXCLUDED.source,
                campaign_id = COALESCE(EXCLUDED.campaign_id, campaign_suppressions.campaign_id),
                notes = COALESCE(EXCLUDED.notes, campaign_suppressions.notes),
                expires_at = CASE
                    WHEN EXCLUDED.expires_at IS NULL THEN NULL
                    WHEN campaign_suppressions.expires_at IS NULL THEN NULL
                    ELSE GREATEST(EXCLUDED.expires_at, campaign_suppressions.expires_at)
                END
            WHERE EXCLUDED.expires_at IS NULL
               OR campaign_suppressions.expires_at IS NOT NULL
            RETURNING id
            """,
            email_val,
            domain_val,
            reason,
            source,
            cid,
            notes,
            expires_at,
        )
        if row:
            logger.info(
                "Suppression added/updated: email=%s domain=%s reason=%s source=%s",
                email, domain, reason, source,
            )
            return row["id"]
        return None
    except Exception as exc:
        logger.error("Failed to add suppression (address may remain eligible): %s", exc)
        return None


async def is_suppressed(pool, *, email: str) -> dict | None:
    """Check if an email or its domain is suppressed.

    Returns the suppression row as a dict if blocked, None if clear.
    Respects expires_at (expired entries don't count).
    """
    if not email:
        return None

    email_lower = email.lower().strip()
    now = datetime.now(timezone.utc)

    # Check exact email match first
    row = await pool.fetchrow(
        """
        SELECT id, email, domain, reason, source, notes, created_at, expires_at
        FROM campaign_suppressions
        WHERE LOWER(email) = $1
          AND (expires_at IS NULL OR expires_at > $2)
        ORDER BY created_at DESC
        LIMIT 1
        """,
        email_lower,
        now,
    )
    if row:
        return dict(row)

    # Check domain match
    parts = email_lower.split("@")
    if len(parts) == 2:
        domain = parts[1]
        row = await pool.fetchrow(
            """
            SELECT id, email, domain, reason, source, notes, created_at, expires_at
            FROM campaign_suppressions
            WHERE LOWER(domain) = $1
              AND (expires_at IS NULL OR expires_at > $2)
            ORDER BY created_at DESC
            LIMIT 1
            """,
            domain,
            now,
        )
        if row:
            return dict(row)

    return None


async def assign_recipient_to_sequence(
    pool,
    sequence_id: UUID | str,
    email: str,
) -> AssignmentResult:
    """Atomically attach a recipient email to a campaign_sequences row.

    If another active sequence already holds this email, the target sequence
    is marked 'superseded' (when still 'active'), an entry is written to
    campaign_audit_log, and the email is NOT assigned. Otherwise the email
    is set on the sequence.

    The conflict probe + state mutation run in a single transaction. The
    partial index idx_campaign_seq_active_email (migration 307) makes the
    probe O(1) for typical recipient cardinalities.
    """
    sid = sequence_id if isinstance(sequence_id, UUID) else UUID(str(sequence_id))

    if not email or not email.strip():
        return AssignmentResult(
            assigned=False,
            sequence_id=sid,
            reason="empty_email",
        )

    email_clean = email.strip()
    email_lower = email_clean.lower()

    async with pool.acquire() as conn:
        async with conn.transaction():
            conflict = await conn.fetchval(
                """
                SELECT id FROM campaign_sequences
                WHERE LOWER(recipient_email) = $1
                  AND status = 'active'
                  AND id != $2
                LIMIT 1
                """,
                email_lower,
                sid,
            )

            if conflict:
                await conn.execute(
                    """
                    UPDATE campaign_sequences
                    SET status = 'superseded', updated_at = NOW()
                    WHERE id = $1 AND status = 'active'
                    """,
                    sid,
                )
                await conn.execute(
                    """
                    INSERT INTO campaign_audit_log
                        (sequence_id, event_type, recipient_email, source, metadata)
                    VALUES ($1, 'recipient_superseded', $2, 'system', $3::jsonb)
                    """,
                    sid,
                    email_clean,
                    json.dumps({
                        "conflict_with_sequence_id": str(conflict),
                        "reason": "active_sequence_exists_for_recipient",
                    }),
                )
                logger.info(
                    "Sequence %s superseded: recipient %s already in active sequence %s",
                    sid, email_clean, conflict,
                )
                return AssignmentResult(
                    assigned=False,
                    sequence_id=sid,
                    conflict_with_sequence_id=conflict,
                    reason="active_sequence_exists_for_recipient",
                )

            await conn.execute(
                """
                UPDATE campaign_sequences
                SET recipient_email = $1, updated_at = NOW()
                WHERE id = $2
                """,
                email_clean,
                sid,
            )
            return AssignmentResult(assigned=True, sequence_id=sid)
