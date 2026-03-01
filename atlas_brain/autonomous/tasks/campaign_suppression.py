"""
Campaign suppression helpers.

Provides add_suppression() and is_suppressed() for the global
do-not-contact / blocklist system.
"""

import logging
from datetime import datetime, timezone
from uuid import UUID

logger = logging.getLogger("atlas.autonomous.tasks.campaign_suppression")


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
        logger.warning("Failed to add suppression: %s", exc)
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
