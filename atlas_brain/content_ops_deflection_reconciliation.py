"""Paid-but-report-missing reconciliation ledger writes (#1462).

When a Stripe deflection checkout completes (paid) but no report row exists and
the event has aged past the write-ordering race window, the billing webhook
returns 2xx (to stop Stripe retrying a non-2xx for hours) and records the
paid-but-undelivered case here so it is surfaced for manual reconciliation
rather than retried into the void.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("atlas.content_ops_deflection_reconciliation")

_INSERT_PAID_RECONCILIATION_SQL = """
INSERT INTO content_ops_deflection_paid_reconciliation
    (account_id, request_id, stripe_session_id, event_type, reason)
VALUES ($1, $2, $3, $4, $5)
ON CONFLICT (account_id, request_id, stripe_session_id) DO NOTHING
"""


async def record_paid_report_missing(
    pool: Any,
    *,
    account_id: str,
    request_id: str,
    stripe_session_id: str | None,
    event_type: str,
    reason: str = "paid_report_missing",
) -> None:
    """Record a paid Stripe deflection checkout that has no report row.

    Idempotent on (account_id, request_id, stripe_session_id) so Stripe retries
    -- or a later genuine arrival of the same event -- do not create duplicate
    ledger rows.
    """

    await pool.execute(
        _INSERT_PAID_RECONCILIATION_SQL,
        account_id,
        request_id,
        stripe_session_id,
        event_type,
        reason,
    )
