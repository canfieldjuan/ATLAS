"""Repository helpers for Competitive Intelligence vendor briefing state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PendingVendorBriefing:
    vendor_name: str
    recipient_email: str
    subject: str
    briefing_data: dict[str, Any]
    briefing_html: str


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


async def has_prior_deliverable_briefing(pool: Any, vendor_name: str) -> bool:
    count = await pool.fetchval(
        """
        SELECT COUNT(*) FROM b2b_vendor_briefings
        WHERE LOWER(vendor_name) = LOWER($1)
          AND status NOT IN ('failed', 'suppressed', 'rejected')
        """,
        vendor_name,
    )
    return int(count or 0) > 0


async def has_recent_deliverable_briefing(
    pool: Any,
    vendor_name: str,
    cooldown_days: int,
) -> bool:
    row = await pool.fetchval(
        """
        SELECT EXISTS(
            SELECT 1 FROM b2b_vendor_briefings
            WHERE LOWER(vendor_name) = LOWER($1)
              AND status NOT IN ('failed', 'suppressed', 'rejected')
              AND created_at > NOW() - make_interval(days => $2)
        )
        """,
        vendor_name,
        cooldown_days,
    )
    return bool(row)


async def insert_suppressed_briefing_record(
    pool: Any,
    *,
    vendor_name: str,
    recipient_email: str,
    subject: str,
    briefing_data: dict[str, Any],
) -> None:
    await pool.execute(
        """
        INSERT INTO b2b_vendor_briefings
            (vendor_name, recipient_email, subject, briefing_data, status)
        VALUES ($1, $2, $3, $4::jsonb, 'suppressed')
        """,
        vendor_name,
        recipient_email,
        subject,
        json.dumps(briefing_data, default=str),
    )


async def insert_delivery_briefing_record(
    pool: Any,
    *,
    vendor_name: str,
    recipient_email: str,
    subject: str,
    briefing_data: dict[str, Any],
    resend_id: str | None,
    status: str,
) -> None:
    await pool.execute(
        """
        INSERT INTO b2b_vendor_briefings
            (vendor_name, recipient_email, subject, briefing_data, resend_id, status)
        VALUES ($1, $2, $3, $4::jsonb, $5, $6)
        """,
        vendor_name,
        recipient_email,
        subject,
        json.dumps(briefing_data, default=str),
        resend_id,
        status,
    )


async def insert_pending_approval_briefing_record(
    pool: Any,
    *,
    vendor_name: str,
    recipient_email: str,
    subject: str,
    briefing_data: dict[str, Any],
    briefing_html: str,
    target_mode: str,
) -> None:
    await pool.execute(
        """
        INSERT INTO b2b_vendor_briefings
            (vendor_name, recipient_email, subject, briefing_data,
             briefing_html, status, target_mode)
        VALUES ($1, $2, $3, $4::jsonb, $5, 'pending_approval', $6)
        """,
        vendor_name,
        recipient_email,
        subject,
        json.dumps(briefing_data, default=str),
        briefing_html,
        target_mode,
    )


async def fetch_pending_approval_briefing(
    pool: Any,
    briefing_id: str,
) -> PendingVendorBriefing | None:
    row = await pool.fetchrow(
        """
        SELECT id, vendor_name, recipient_email, subject,
               briefing_data, briefing_html
        FROM b2b_vendor_briefings
        WHERE id = $1 AND status = 'pending_approval'
        """,
        briefing_id,
    )
    if not row:
        return None
    return PendingVendorBriefing(
        vendor_name=row["vendor_name"],
        recipient_email=row["recipient_email"],
        subject=row["subject"],
        briefing_data=_json_dict(row["briefing_data"]),
        briefing_html=row["briefing_html"] or "",
    )


async def mark_pending_briefing_sent(
    pool: Any,
    *,
    briefing_id: str,
    resend_id: str | None,
) -> None:
    await pool.execute(
        """
        UPDATE b2b_vendor_briefings
        SET status = $1, resend_id = $2, approved_at = NOW()
        WHERE id = $3
        """,
        "sent",
        resend_id,
        briefing_id,
    )


async def mark_pending_briefing_failed(pool: Any, briefing_id: str) -> None:
    await pool.execute(
        """
        UPDATE b2b_vendor_briefings
        SET status = $1
        WHERE id = $2
        """,
        "failed",
        briefing_id,
    )


async def reject_pending_briefing(
    pool: Any,
    *,
    briefing_id: str,
    reason: str | None = None,
) -> bool:
    result = await pool.execute(
        """
        UPDATE b2b_vendor_briefings
        SET status = 'rejected', rejected_at = NOW(), reject_reason = $1
        WHERE id = $2 AND status = 'pending_approval'
        """,
        reason,
        briefing_id,
    )
    return result != "UPDATE 0"
