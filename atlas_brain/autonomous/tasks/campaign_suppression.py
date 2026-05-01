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

import asyncpg

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


async def _supersede_sequence(
    conn,
    *,
    sequence_id: UUID,
    email: str,
    conflict_with: UUID | None,
    reason: str,
) -> None:
    """Mark a sequence superseded, audit it, and cancel its draft/queued
    campaigns so they cannot ship after the recipient lost the dedup race.

    Must be called inside an open transaction on ``conn``. Idempotent on
    sequence status: only flips 'active' -> 'superseded' to avoid
    overwriting terminal states like 'replied' or 'bounced'. Campaign
    cancellation runs unconditionally because a campaign linked to a
    superseded sequence must never be sent regardless of the sequence's
    prior state when this helper is invoked.
    """
    await conn.execute(
        """
        UPDATE campaign_sequences
        SET status = 'superseded', updated_at = NOW()
        WHERE id = $1 AND status = 'active'
        """,
        sequence_id,
    )
    cancel_status = await conn.execute(
        """
        UPDATE b2b_campaigns
        SET status = 'cancelled', updated_at = NOW()
        WHERE sequence_id = $1
          AND status IN ('draft', 'approved', 'queued')
        """,
        sequence_id,
    )
    cancelled_count = 0
    if isinstance(cancel_status, str) and cancel_status.startswith("UPDATE "):
        try:
            cancelled_count = int(cancel_status.split()[1])
        except (ValueError, IndexError):
            cancelled_count = 0
    await conn.execute(
        """
        INSERT INTO campaign_audit_log
            (sequence_id, event_type, recipient_email, source, metadata)
        VALUES ($1, 'recipient_superseded', $2, 'system', $3::jsonb)
        """,
        sequence_id,
        email,
        json.dumps({
            "conflict_with_sequence_id": str(conflict_with) if conflict_with else None,
            "reason": reason,
            "cancelled_campaigns": cancelled_count,
        }),
    )
    logger.info(
        "Sequence %s superseded: recipient %s already in active sequence %s "
        "(reason=%s, cancelled %d campaigns)",
        sequence_id, email, conflict_with, reason, cancelled_count,
    )


async def _try_set_active_recipient(conn, *, sequence_id: UUID, email: str) -> bool:
    """Attempt to set recipient_email on a sequence that is still active.

    Returns True iff the UPDATE matched exactly the target row; False when
    the row no longer exists or has transitioned out of 'active' (e.g. it
    was just superseded/bounced/unsubscribed/replied by another worker
    between caller selection and assignment). Raises UniqueViolationError
    when the partial unique index rejects the write -- the caller decides
    whether to enter race-recovery.
    """
    status = await conn.execute(
        """
        UPDATE campaign_sequences
        SET recipient_email = $1, updated_at = NOW()
        WHERE id = $2 AND status = 'active'
        """,
        email,
        sequence_id,
    )
    if isinstance(status, str) and status.startswith("UPDATE "):
        try:
            return int(status.split()[1]) > 0
        except (ValueError, IndexError):
            return False
    return False


_PROBE_ACTIVE_CONFLICT_SQL = """
SELECT id FROM campaign_sequences
WHERE LOWER(recipient_email) = $1
  AND status = 'active'
  AND id != $2
LIMIT 1
"""


async def attach_recipient_strict(
    pool,
    sequence_id: UUID | str,
    email: str,
) -> AssignmentResult:
    """Strict variant of ``assign_recipient_to_sequence`` for explicit
    callers (REST APIs, manual admin tools) that want a deterministic
    conflict result without the side effect of superseding their own
    sequence.

    Behaviour:

      * No conflict -> UPDATE the recipient on the active sequence and
        return ``assigned=True``.
      * Active conflict (visible at probe time *or* surfaced as
        ``UniqueViolationError`` when a worker raced us) -> return
        ``assigned=False`` with reason
        ``'active_sequence_exists_for_recipient'`` (probe-detected) or
        ``'active_sequence_exists_for_recipient_race'`` (race-detected)
        and ``conflict_with_sequence_id`` set to the winner's id when
        re-probable.
      * Target sequence not active (already terminal, or row missing) ->
        ``assigned=False`` with reason ``'sequence_not_active'``.
      * Empty email -> ``assigned=False`` with reason ``'empty_email'``.

    The strict variant never modifies any row other than the target
    sequence's recipient_email, so a 409 mapper in the route handler can
    safely retry or report back to the caller without unwinding state.
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
                _PROBE_ACTIVE_CONFLICT_SQL, email_lower, sid,
            )
            if conflict:
                return AssignmentResult(
                    assigned=False,
                    sequence_id=sid,
                    conflict_with_sequence_id=conflict,
                    reason="active_sequence_exists_for_recipient",
                )

            try:
                updated = await _try_set_active_recipient(
                    conn, sequence_id=sid, email=email_clean,
                )
            except asyncpg.UniqueViolationError:
                # Race: a concurrent worker just assigned this email.
                # Re-probe so we can hand the winner's id back to the
                # caller for a deterministic 409 response.
                conflict = await conn.fetchval(
                    _PROBE_ACTIVE_CONFLICT_SQL, email_lower, sid,
                )
                return AssignmentResult(
                    assigned=False,
                    sequence_id=sid,
                    conflict_with_sequence_id=conflict,
                    reason="active_sequence_exists_for_recipient_race",
                )

            if not updated:
                return AssignmentResult(
                    assigned=False,
                    sequence_id=sid,
                    reason="sequence_not_active",
                )

            return AssignmentResult(assigned=True, sequence_id=sid)


async def assign_recipient_to_sequence(
    pool,
    sequence_id: UUID | str,
    email: str,
) -> AssignmentResult:
    """Atomically attach a recipient email to a campaign_sequences row.

    If another active sequence already holds this email, the target sequence
    is marked 'superseded', any draft/approved/queued b2b_campaigns rows
    attached to it are cancelled (so they cannot leak past the dedup gate),
    an entry is written to campaign_audit_log, and the email is NOT
    assigned.

    Race-safety. Migration 309 enforces a UNIQUE partial index on
    LOWER(recipient_email) WHERE status='active', so the database is the
    source of truth. The happy-path SELECT-then-UPDATE handles the common
    case in one transaction; if a concurrent worker wins the race between
    our SELECT and UPDATE, the UPDATE raises UniqueViolationError and we
    fall through to a fresh transaction that re-probes the conflict and
    supersedes our sequence the same way.
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
        try:
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
                    await _supersede_sequence(
                        conn,
                        sequence_id=sid,
                        email=email_clean,
                        conflict_with=conflict,
                        reason="active_sequence_exists_for_recipient",
                    )
                    return AssignmentResult(
                        assigned=False,
                        sequence_id=sid,
                        conflict_with_sequence_id=conflict,
                        reason="active_sequence_exists_for_recipient",
                    )

                updated = await _try_set_active_recipient(
                    conn, sequence_id=sid, email=email_clean,
                )
                if not updated:
                    return AssignmentResult(
                        assigned=False,
                        sequence_id=sid,
                        reason="sequence_not_active",
                    )
                return AssignmentResult(assigned=True, sequence_id=sid)
        except asyncpg.UniqueViolationError:
            pass  # fall through to race-recovery path

        # Race lost: a concurrent worker assigned this email between our
        # SELECT and UPDATE. The unique index rejected our write; the
        # outer transaction has already rolled back.
        #
        # On recovery we re-probe for an active conflict. The competitor
        # may have transitioned out of 'active' (bounced, replied, etc.)
        # between the violation and now -- in which case the partial
        # unique index no longer covers them and the email is claimable.
        # We try the UPDATE inside a savepoint up to _MAX_RACE_RETRIES
        # times; each retry begins with a fresh probe, so if a *new*
        # winner has settled into 'active' we'll see it and supersede.
        # If both probes show no winner but the UPDATE still raises
        # UniqueViolationError, the email is in an unstable state (a
        # competitor entered and left 'active' rapidly between our
        # probe and write); we return assigned=False with reason
        # 'concurrent_state_unstable' rather than supersede ourselves
        # against a phantom winner.
        _MAX_RACE_RETRIES = 2
        async with conn.transaction():
            for _attempt in range(_MAX_RACE_RETRIES):
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
                if conflict is not None:
                    await _supersede_sequence(
                        conn,
                        sequence_id=sid,
                        email=email_clean,
                        conflict_with=conflict,
                        reason="active_sequence_exists_for_recipient_race",
                    )
                    return AssignmentResult(
                        assigned=False,
                        sequence_id=sid,
                        conflict_with_sequence_id=conflict,
                        reason="active_sequence_exists_for_recipient_race",
                    )

                # No active competitor visible. Try to claim the email.
                try:
                    async with conn.transaction():  # savepoint
                        updated = await _try_set_active_recipient(
                            conn, sequence_id=sid, email=email_clean,
                        )
                    if updated:
                        return AssignmentResult(assigned=True, sequence_id=sid)
                    # Our own sequence has transitioned out of 'active'
                    # since the caller selected it. Don't supersede --
                    # the row is already terminal and any cascade is
                    # owned by the path that moved it there.
                    return AssignmentResult(
                        assigned=False,
                        sequence_id=sid,
                        reason="sequence_not_active",
                    )
                except asyncpg.UniqueViolationError:
                    # A new competitor just claimed the email during
                    # our retry. Loop back to re-probe -- if it's still
                    # active we supersede ourselves against the new
                    # winner; if it has already transitioned out we
                    # try one more UPDATE before giving up.
                    continue

            return AssignmentResult(
                assigned=False,
                sequence_id=sid,
                reason="concurrent_state_unstable",
            )
