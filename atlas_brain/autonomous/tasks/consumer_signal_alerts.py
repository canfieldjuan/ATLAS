"""Real-time signal alerts for consumer (Amazon Seller Intelligence) accounts.

Runs every 6 hours. Finds deep-enriched reviews with pain_score >= 8 or
safety_flag set, enriched in the last 6 hours, and matching tracked ASINs
for active accounts. Groups alerts by account and sends one email per account.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.tasks.consumer_signal_alerts")


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Check for high-severity signals and alert affected accounts."""
    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    cfg = settings.campaign_sequence
    if not cfg.resend_api_key or not cfg.resend_from_email:
        return {"_skip_synthesis": "Resend not configured"}

    # Lookback window: 6 hours (matches task interval)
    since = datetime.now(timezone.utc) - timedelta(hours=6)

    # Find high-severity signals from the lookback window
    signals = await pool.fetch(
        """
        SELECT pr.id, pr.asin, pr.rating, pr.summary,
               pr.pain_score, pr.root_cause,
               (pr.deep_extraction->'safety_flag'->>'flagged')::boolean AS safety_flagged,
               pr.deep_enriched_at
        FROM product_reviews pr
        WHERE pr.deep_enriched_at >= $1
          AND (
              pr.pain_score >= 8
              OR (pr.deep_extraction->'safety_flag'->>'flagged')::boolean IS TRUE
          )
        """,
        since,
    )

    if not signals:
        return {"_skip_synthesis": "No high-severity signals in window"}

    # Get all signal ASINs
    signal_asins = list({s["asin"] for s in signals})

    # Find accounts tracking these ASINs
    account_asins = await pool.fetch(
        """
        SELECT ta.account_id, ta.asin,
               sa.name AS account_name,
               su.email AS owner_email, su.full_name
        FROM tracked_asins ta
        JOIN saas_accounts sa ON sa.id = ta.account_id
        JOIN saas_users su ON su.account_id = sa.id AND su.role = 'owner'
        WHERE ta.asin = ANY($1)
          AND sa.product = 'consumer'
          AND sa.plan_status IN ('trialing', 'active')
        """,
        signal_asins,
    )

    if not account_asins:
        return {"_skip_synthesis": "No accounts tracking affected ASINs"}

    # Group by account
    account_map: dict[str, dict[str, Any]] = {}
    for row in account_asins:
        aid = str(row["account_id"])
        if aid not in account_map:
            account_map[aid] = {
                "account_name": row["account_name"],
                "owner_email": row["owner_email"],
                "full_name": row["full_name"],
                "asins": set(),
            }
        account_map[aid]["asins"].add(row["asin"])

    # Index signals by ASIN for fast lookup
    asin_signals: dict[str, list[dict]] = {}
    for s in signals:
        asin_signals.setdefault(s["asin"], []).append(s)

    emails_sent = 0
    for aid, acct in account_map.items():
        # Collect signals relevant to this account's tracked ASINs
        acct_signals = []
        for asin in acct["asins"]:
            acct_signals.extend(asin_signals.get(asin, []))

        if not acct_signals:
            continue

        await _send_alert_email(
            owner_email=acct["owner_email"],
            full_name=acct["full_name"],
            account_name=acct["account_name"],
            signals=acct_signals,
            cfg=cfg,
        )
        emails_sent += 1

    return {
        "_skip_synthesis": "Consumer signal alerts complete",
        "total_signals": len(signals),
        "accounts_alerted": emails_sent,
    }


async def _send_alert_email(
    *,
    owner_email: str,
    full_name: str | None,
    account_name: str,
    signals: list[dict],
    cfg,
) -> None:
    """Send signal alert email via Resend."""
    name = full_name or "there"

    safety_count = sum(1 for s in signals if s.get("safety_flagged"))
    high_pain = [s for s in signals if s.get("pain_score") and s["pain_score"] >= 8]

    subject_parts = []
    if safety_count:
        subject_parts.append(f"{safety_count} safety alert{'s' if safety_count != 1 else ''}")
    if high_pain:
        subject_parts.append(f"{len(high_pain)} high-pain signal{'s' if len(high_pain) != 1 else ''}")
    subject = f"Alert: {', '.join(subject_parts)}" if subject_parts else "New Signal Alerts"

    # Build signal rows
    tag = settings.external_data.amazon_associate_tag
    rows = ""
    for s in sorted(signals, key=lambda x: x.get("pain_score") or 0, reverse=True)[:10]:
        is_safety = s.get("safety_flagged")
        badge = '<span style="color:red;font-weight:bold;">SAFETY</span> ' if is_safety else ""
        pain = f"Pain: {s['pain_score']}" if s.get("pain_score") else ""
        asin = s["asin"]
        asin_url = f"https://www.amazon.com/dp/{asin}?tag={tag}" if tag else f"https://www.amazon.com/dp/{asin}"
        asin_link = f"<a href='{asin_url}' style='color:#60a5fa;text-decoration:none;'>{asin}</a>"
        rows += (
            f"<tr>"
            f"<td style='padding:4px 8px;'>{badge}{asin_link}</td>"
            f"<td style='padding:4px 8px;'>{s.get('root_cause') or 'N/A'}</td>"
            f"<td style='padding:4px 8px;'>{pain}</td>"
            f"<td style='padding:4px 8px;'>{s.get('summary', '')[:80]}</td>"
            f"</tr>"
        )

    body = (
        f"<h2>Signal Alert</h2>"
        f"<p>Hi {name},</p>"
        f"<p>New high-severity signals detected for <strong>{account_name}</strong>:</p>"
        f"<table cellpadding='4' style='border-collapse:collapse; font-size:14px;'>"
        f"<tr style='background:#1e293b;color:#e2e8f0;'>"
        f"<th style='padding:6px 8px;text-align:left;'>ASIN</th>"
        f"<th style='padding:6px 8px;text-align:left;'>Root Cause</th>"
        f"<th style='padding:6px 8px;text-align:left;'>Score</th>"
        f"<th style='padding:6px 8px;text-align:left;'>Summary</th>"
        f"</tr>"
        f"{rows}"
        f"</table>"
        f"<hr>"
        f"<p><em>View full details in your Atlas Intel dashboard.</em></p>"
    )

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {cfg.resend_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": cfg.resend_from_email,
                    "to": [owner_email],
                    "subject": subject,
                    "html": body,
                },
            )
            resp.raise_for_status()
    except Exception as exc:
        logger.warning("Failed to send signal alert to %s: %s", owner_email, exc)
