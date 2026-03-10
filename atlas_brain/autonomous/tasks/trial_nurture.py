"""Daily trial-to-paid nurture emails at key milestones.

Runs daily at 10 AM. Queries saas_accounts for trialing consumer accounts,
calculates trial age, and sends template emails at days 3, 7, 11, and 13.
A trial_nurture_log table prevents duplicate sends.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.tasks.trial_nurture")

# (key, trigger_day, only_if_inactive)
_MILESTONES = [
    ("day_3", 3, True),    # Only if 0 ASINs tracked
    ("day_7", 7, False),   # Always
    ("day_11", 11, False), # Always
    ("day_13", 13, False), # Always
]


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Send nurture emails to trialing consumer accounts at key milestones."""
    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    cfg = settings.campaign_sequence
    if not cfg.resend_api_key or not cfg.resend_from_email:
        return {"_skip_synthesis": "Resend not configured"}

    trial_days = settings.saas_auth.trial_days
    base_url = (settings.external_data.blog_base_url or "").rstrip("/")

    # Fetch all trialing consumer accounts with owner email and ASIN count
    rows = await pool.fetch(
        """
        SELECT sa.id AS account_id, sa.name AS account_name,
               sa.trial_ends_at,
               su.email AS owner_email, su.full_name,
               COALESCE(ta.cnt, 0) AS asin_count
        FROM saas_accounts sa
        JOIN saas_users su ON su.account_id = sa.id AND su.role = 'owner'
        LEFT JOIN LATERAL (
            SELECT COUNT(*) AS cnt FROM tracked_asins WHERE account_id = sa.id
        ) ta ON TRUE
        WHERE sa.product = 'consumer'
          AND sa.plan = 'trial'
          AND sa.plan_status = 'trialing'
          AND sa.trial_ends_at IS NOT NULL
        """
    )

    if not rows:
        return {"_skip_synthesis": "No trialing accounts"}

    # Fetch already-sent milestones for all these accounts
    account_ids = [r["account_id"] for r in rows]
    sent_rows = await pool.fetch(
        "SELECT account_id, milestone FROM trial_nurture_log WHERE account_id = ANY($1)",
        account_ids,
    )
    sent_set: set[tuple[str, str]] = {
        (str(r["account_id"]), r["milestone"]) for r in sent_rows
    }

    now = datetime.now(timezone.utc)
    emails_sent = 0

    for row in rows:
        aid = str(row["account_id"])
        trial_ends_at = row["trial_ends_at"]
        trial_start = trial_ends_at - timedelta(days=trial_days)
        trial_age = (now - trial_start).days

        for milestone_key, trigger_day, only_if_inactive in _MILESTONES:
            if trial_age < trigger_day:
                continue
            if (aid, milestone_key) in sent_set:
                continue
            # day_3: skip if user has tracked ASINs (they're engaged)
            if only_if_inactive and row["asin_count"] > 0:
                continue

            subject, body = _render_email(
                milestone_key,
                full_name=row["full_name"],
                account_name=row["account_name"],
                trial_days_left=max(0, (trial_ends_at - now).days),
                asin_count=row["asin_count"],
                base_url=base_url,
            )

            await _send_email(
                to=row["owner_email"],
                subject=subject,
                body=body,
                cfg=cfg,
            )

            # Record to prevent duplicate sends
            await pool.execute(
                """
                INSERT INTO trial_nurture_log (account_id, milestone)
                VALUES ($1, $2)
                ON CONFLICT DO NOTHING
                """,
                row["account_id"],
                milestone_key,
            )
            emails_sent += 1

    return {
        "_skip_synthesis": f"Trial nurture complete: {emails_sent} emails sent",
        "accounts_checked": len(rows),
        "emails_sent": emails_sent,
    }


def _render_email(
    milestone: str,
    *,
    full_name: str | None,
    account_name: str,
    trial_days_left: int,
    asin_count: int,
    base_url: str,
) -> tuple[str, str]:
    """Return (subject, html_body) for a milestone email."""
    name = full_name or "there"

    if milestone == "day_3":
        subject = "Start collecting competitive intelligence"
        body = (
            f"<h2>Your Intelligence Feed Awaits</h2>"
            f"<p>Hi {name},</p>"
            f"<p>You signed up 3 days ago but haven't added any ASINs to your watchlist yet.</p>"
            f"<p>Add your first ASIN to start collecting displacement intelligence, "
            f"evidence-backed pain-point signals, and safety alerts delivered straight to your inbox.</p>"
            f"<p><a href='{base_url}/dashboard' "
            f"style='display:inline-block;padding:10px 24px;background:#2563eb;"
            f"color:#fff;text-decoration:none;border-radius:6px;'>Start Your Watchlist</a></p>"
            f"<p>You have {trial_days_left} days left in your trial.</p>"
        )
    elif milestone == "day_7":
        subject = "Your week 1 intelligence summary"
        body = (
            f"<h2>Your First Week of Intelligence Collection</h2>"
            f"<p>Hi {name},</p>"
            f"<p>You're one week into your trial"
            f"{f' and tracking {asin_count} ASIN(s) on your watchlist' if asin_count else ''}. "
            f"Here's what your intelligence feed delivers:</p>"
            f"<ul>"
            f"<li><strong>Evidence-backed pain signals</strong> - Know exactly what customers complain about, with proof</li>"
            f"<li><strong>Displacement maps</strong> - See which brands customers switch to and from</li>"
            f"<li><strong>Safety signal alerts</strong> - Get notified about product safety concerns instantly</li>"
            f"<li><strong>Feature gap intelligence</strong> - Discover unmet needs in your category</li>"
            f"</ul>"
            f"<p><a href='{base_url}/dashboard' "
            f"style='display:inline-block;padding:10px 24px;background:#2563eb;"
            f"color:#fff;text-decoration:none;border-radius:6px;'>View Your Intelligence Feed</a></p>"
            f"<p>You have {trial_days_left} days left in your trial.</p>"
        )
    elif milestone == "day_11":
        subject = "Your trial ends in 3 days"
        body = (
            f"<h2>3 Days Left</h2>"
            f"<p>Hi {name},</p>"
            f"<p>Your trial for <strong>{account_name}</strong> ends in 3 days.</p>"
            f"<p>Upgrade now to keep your watchlist, signal alerts, and intelligence feeds active:</p>"
            f"<ul>"
            f"<li><strong>Starter</strong> ($99/mo) - 10 ASINs, weekly intelligence digests</li>"
            f"<li><strong>Growth</strong> ($199/mo) - 50 ASINs, API access, displacement maps</li>"
            f"<li><strong>Pro</strong> ($299/mo) - 200 ASINs, full API, priority enrichment</li>"
            f"</ul>"
            f"<p><a href='{base_url}/settings/billing' "
            f"style='display:inline-block;padding:10px 24px;background:#2563eb;"
            f"color:#fff;text-decoration:none;border-radius:6px;'>Upgrade Now</a></p>"
        )
    else:  # day_13
        subject = "Last day of your free trial"
        body = (
            f"<h2>Your Trial Ends Tomorrow</h2>"
            f"<p>Hi {name},</p>"
            f"<p>This is your last day to upgrade <strong>{account_name}</strong> "
            f"before your trial expires.</p>"
            f"<p>After expiration, you'll lose access to:</p>"
            f"<ul>"
            f"<li>Your ASIN watchlist and intelligence feeds</li>"
            f"<li>Displacement signal alerts and weekly digests</li>"
            f"<li>Historical brand memory and competitive flow maps</li>"
            f"</ul>"
            f"<p>Don't lose your data. Upgrade in one click:</p>"
            f"<p><a href='{base_url}/settings/billing' "
            f"style='display:inline-block;padding:10px 24px;background:#dc2626;"
            f"color:#fff;text-decoration:none;border-radius:6px;font-weight:bold;'>"
            f"Upgrade Before It's Too Late</a></p>"
        )

    return subject, body


async def _send_email(*, to: str, subject: str, body: str, cfg) -> None:
    """Send an email via Resend."""
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
                    "to": [to],
                    "subject": subject,
                    "html": body,
                },
            )
            resp.raise_for_status()
    except Exception as exc:
        logger.warning("Failed to send nurture email to %s: %s", to, exc)
