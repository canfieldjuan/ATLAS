"""Weekly digest email for consumer (Amazon Seller Intelligence) accounts.

Runs Monday 8 AM. For each consumer account with tracked ASINs and an
active/trialing subscription, gathers review metrics from the past 7 days
and sends an HTML summary via Resend.
"""

import logging
from datetime import date, timedelta, timezone
from typing import Any

import httpx

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.tasks.consumer_weekly_digest")


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Generate and email weekly digest for each consumer account."""
    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    cfg = settings.campaign_sequence
    if not cfg.resend_api_key or not cfg.resend_from_email:
        return {"_skip_synthesis": "Resend not configured"}

    accounts = await pool.fetch(
        """
        SELECT sa.id AS account_id, sa.name AS account_name,
               su.email AS owner_email, su.full_name,
               array_agg(ta.asin) AS asins
        FROM saas_accounts sa
        JOIN saas_users su ON su.account_id = sa.id AND su.role = 'owner'
        JOIN tracked_asins ta ON ta.account_id = sa.id
        WHERE sa.product = 'consumer'
          AND sa.plan_status IN ('trialing', 'active')
        GROUP BY sa.id, sa.name, su.email, su.full_name
        """
    )

    if not accounts:
        return {"_skip_synthesis": "No eligible consumer accounts for digest"}

    since = date.today() - timedelta(days=7)
    emails_sent = 0

    for acct in accounts:
        account_id = acct["account_id"]
        asins = [a for a in (acct["asins"] or []) if a]
        if not asins:
            continue

        try:
            metrics = await _gather_metrics(pool, asins, since)
        except Exception as exc:
            logger.error("Metrics gathering failed for account %s: %s", account_id, exc)
            continue

        if metrics["new_reviews"] == 0:
            continue

        await _send_digest_email(
            owner_email=acct["owner_email"],
            full_name=acct["full_name"],
            account_name=acct["account_name"],
            metrics=metrics,
            cfg=cfg,
        )
        emails_sent += 1

    return {
        "_skip_synthesis": "Consumer weekly digest complete",
        "accounts_processed": len(accounts),
        "emails_sent": emails_sent,
    }


async def _gather_metrics(pool, asins: list[str], since: date) -> dict[str, Any]:
    """Gather review metrics for tracked ASINs over the past week."""
    asin_list = list(set(asins))

    row = await pool.fetchrow(
        """
        SELECT
            COUNT(*)::int AS new_reviews,
            ROUND(AVG(pr.rating)::numeric, 2) AS avg_rating,
            COUNT(*) FILTER (
                WHERE pr.deep_enrichment_status = 'enriched'
            )::int AS enriched,
            COUNT(*) FILTER (WHERE pr.pain_score >= 8)::int AS high_pain,
            COUNT(*) FILTER (
                WHERE (pr.deep_extraction->'safety_flag'->>'flagged')::boolean IS TRUE
            )::int AS safety_flagged
        FROM product_reviews pr
        WHERE pr.asin = ANY($1)
          AND pr.imported_at >= $2
        """,
        asin_list,
        since,
    )

    top_pain = await pool.fetch(
        """
        SELECT pr.asin, pr.root_cause, pr.pain_score
        FROM product_reviews pr
        WHERE pr.asin = ANY($1) AND pr.imported_at >= $2
          AND pr.pain_score IS NOT NULL
        ORDER BY pr.pain_score DESC
        LIMIT 5
        """,
        asin_list,
        since,
    )

    # Brand summary from materialized view (brand_health is computed in Python)
    brand_rows = await pool.fetch(
        """
        SELECT brand, review_count,
               deep_count, repurchase_yes, repurchase_total,
               retention_positive, retention_negative,
               trajectory_positive, trajectory_negative,
               safety_count
        FROM mv_brand_summary
        WHERE brand IN (
            SELECT DISTINCT pm.brand FROM product_metadata pm WHERE pm.asin = ANY($1)
        )
        ORDER BY review_count DESC
        LIMIT 5
        """,
        asin_list,
    )

    return {
        "new_reviews": row["new_reviews"] if row else 0,
        "avg_rating": float(row["avg_rating"]) if row and row["avg_rating"] else None,
        "enriched": row["enriched"] if row else 0,
        "high_pain": row["high_pain"] if row else 0,
        "safety_flagged": row["safety_flagged"] if row else 0,
        "top_pain_points": [
            {"asin": r["asin"], "root_cause": r["root_cause"], "pain_score": r["pain_score"]}
            for r in top_pain
        ],
        "brand_health": [
            {"brand": r["brand"], "health": _compute_health(r), "reviews": r["review_count"]}
            for r in brand_rows
        ],
    }


def _compute_health(r) -> int | None:
    """Compute brand health from mv_brand_summary row (mirrors consumer_dashboard)."""
    deep = r["deep_count"]
    if deep < 5:
        return None
    scores: list[float] = []
    rp_total = r["repurchase_total"]
    scores.append(r["repurchase_yes"] / rp_total if rp_total > 0 else 0.5)
    ret_total = r["retention_positive"] + r["retention_negative"]
    scores.append(r["retention_positive"] / ret_total if ret_total > 0 else 0.5)
    traj_total = r["trajectory_positive"] + r["trajectory_negative"]
    scores.append(r["trajectory_positive"] / traj_total if traj_total > 0 else 0.5)
    safety_rate = max(0.0, 1.0 - (r["safety_count"] / deep) * 10)
    scores.append(safety_rate)
    return round(sum(scores) / len(scores) * 100)


async def _send_digest_email(
    *,
    owner_email: str,
    full_name: str | None,
    account_name: str,
    metrics: dict[str, Any],
    cfg,
) -> None:
    """Send weekly digest via Resend."""
    name = full_name or "there"
    subject = f"Weekly Intel Digest: {metrics['new_reviews']} new reviews"

    # Build pain points section
    tag = settings.external_data.amazon_associate_tag
    pain_html = ""
    if metrics["top_pain_points"]:
        items = ""
        for p in metrics["top_pain_points"]:
            asin = p["asin"]
            asin_url = f"https://www.amazon.com/dp/{asin}?tag={tag}" if tag else f"https://www.amazon.com/dp/{asin}"
            asin_link = f"<a href='{asin_url}' style='color:#60a5fa;text-decoration:none;'>{asin}</a>"
            items += (
                f"<li><strong>{asin_link}</strong>: {p['root_cause'] or 'Unknown'} "
                f"(pain score: {p['pain_score']})</li>"
            )
        pain_html = f"<h3>Top Pain Points</h3><ul>{items}</ul>"

    # Build brand health section
    brand_html = ""
    if metrics["brand_health"]:
        items = "".join(
            f"<li><strong>{b['brand']}</strong>: health {b['health']:.0f}% "
            f"({b['reviews']} reviews)</li>"
            for b in metrics["brand_health"]
            if b["health"] is not None
        )
        if items:
            brand_html = f"<h3>Brand Health</h3><ul>{items}</ul>"

    body = (
        f"<h2>Weekly Intelligence Digest</h2>"
        f"<p>Hi {name},</p>"
        f"<p>Here's your weekly summary for <strong>{account_name}</strong>:</p>"
        f"<table cellpadding='6' style='border-collapse:collapse;'>"
        f"<tr><td>New Reviews</td><td><strong>{metrics['new_reviews']}</strong></td></tr>"
        f"<tr><td>Avg Rating</td><td><strong>{metrics['avg_rating'] or 'N/A'}</strong></td></tr>"
        f"<tr><td>High-Pain Signals</td><td><strong>{metrics['high_pain']}</strong></td></tr>"
        f"<tr><td>Safety Flags</td><td><strong>{metrics['safety_flagged']}</strong></td></tr>"
        f"</table>"
        f"{pain_html}"
        f"{brand_html}"
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
        logger.warning("Failed to send weekly digest to %s: %s", owner_email, exc)
