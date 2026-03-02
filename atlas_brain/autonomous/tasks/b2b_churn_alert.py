"""Hourly churn signal spike alerts for B2B tenants.

Compares current signal counts / avg urgency against stored baselines
for each tenant's tracked vendors. Fires ntfy (and optionally email)
when thresholds are exceeded.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.tasks.b2b_churn_alert")


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Check tracked vendors for churn signal spikes and alert tenants."""
    cfg = settings.b2b_alert
    if not cfg.enabled:
        return {"_skip_synthesis": "B2B churn alerts disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    cooldown_cutoff = datetime.now(timezone.utc) - timedelta(hours=cfg.cooldown_hours)
    alerts_sent = 0
    vendors_checked = 0

    # Get all accounts with tracked vendors
    accounts = await pool.fetch(
        """
        SELECT sa.id AS account_id, sa.name AS account_name,
               tv.vendor_name,
               su.email AS owner_email
        FROM tracked_vendors tv
        JOIN saas_accounts sa ON sa.id = tv.account_id
        JOIN saas_users su ON su.account_id = sa.id AND su.role = 'owner'
        WHERE sa.product IN ('b2b_retention', 'b2b_challenger')
        """
    )

    if not accounts:
        return {"_skip_synthesis": "No B2B accounts with tracked vendors"}

    for row in accounts:
        account_id = row["account_id"]
        vendor_name = row["vendor_name"]
        vendors_checked += 1

        # Current metrics for this vendor
        current = await pool.fetchrow(
            """
            SELECT
                count(*) AS signal_count,
                COALESCE(avg((enrichment->>'urgency_score')::numeric), 0) AS avg_urgency,
                count(*) FILTER (
                    WHERE jsonb_array_length(COALESCE(enrichment->'competitors_mentioned', '[]'::jsonb)) > 0
                ) AS displacement_count
            FROM b2b_reviews
            WHERE enrichment_status = 'enriched'
              AND vendor_name ILIKE '%' || $1 || '%'
              AND enriched_at > NOW() - INTERVAL '7 days'
            """,
            vendor_name,
        )

        if not current or current["signal_count"] == 0:
            continue

        metrics = {
            "signal_count": float(current["signal_count"]),
            "avg_urgency": float(current["avg_urgency"]),
            "displacement_count": float(current["displacement_count"]),
        }

        for metric_name, current_value in metrics.items():
            # Get or create baseline
            baseline_row = await pool.fetchrow(
                """
                SELECT baseline_value, last_alerted_at
                FROM b2b_alert_baselines
                WHERE account_id = $1 AND vendor_name = $2 AND metric = $3
                """,
                account_id, vendor_name, metric_name,
            )

            if baseline_row is None:
                # First run: seed baseline, no alert
                await pool.execute(
                    """
                    INSERT INTO b2b_alert_baselines (account_id, vendor_name, metric, baseline_value)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (account_id, vendor_name, metric) DO NOTHING
                    """,
                    account_id, vendor_name, metric_name, current_value,
                )
                continue

            baseline_value = float(baseline_row["baseline_value"])
            last_alerted = baseline_row["last_alerted_at"]

            # Check cooldown
            if last_alerted and last_alerted > cooldown_cutoff:
                continue

            # Check thresholds
            should_alert = False
            delta = current_value - baseline_value

            if metric_name == "signal_count" and delta >= cfg.signal_count_threshold:
                should_alert = True
            elif metric_name == "avg_urgency" and delta >= cfg.urgency_spike_threshold:
                should_alert = True
            elif metric_name == "displacement_count" and delta >= cfg.signal_count_threshold:
                should_alert = True

            if should_alert:
                await _send_alert(
                    account_name=row["account_name"],
                    vendor_name=vendor_name,
                    metric=metric_name,
                    baseline=baseline_value,
                    current=current_value,
                    owner_email=row["owner_email"],
                )
                alerts_sent += 1

                # Update baseline and last_alerted_at
                await pool.execute(
                    """
                    UPDATE b2b_alert_baselines
                    SET baseline_value = $1, last_alerted_at = NOW(), updated_at = NOW()
                    WHERE account_id = $2 AND vendor_name = $3 AND metric = $4
                    """,
                    current_value, account_id, vendor_name, metric_name,
                )

    return {
        "_skip_synthesis": "B2B churn alert check complete",
        "vendors_checked": vendors_checked,
        "alerts_sent": alerts_sent,
    }


async def _send_alert(
    *,
    account_name: str,
    vendor_name: str,
    metric: str,
    baseline: float,
    current: float,
    owner_email: str,
) -> None:
    """Send ntfy notification for a churn signal spike."""
    if not settings.alerts.ntfy_enabled:
        return

    metric_labels = {
        "signal_count": "new churn signals",
        "avg_urgency": "average urgency score",
        "displacement_count": "competitive displacement mentions",
    }
    label = metric_labels.get(metric, metric)
    delta = current - baseline

    message = (
        f"Spike detected for {vendor_name}\n"
        f"{label}: {baseline:.1f} -> {current:.1f} (+{delta:.1f})\n"
        f"Account: {account_name}"
    )

    ntfy_url = f"{settings.alerts.ntfy_url.rstrip('/')}/{settings.alerts.ntfy_topic}"
    headers = {
        "Title": f"Churn Alert: {vendor_name}",
        "Priority": "high",
        "Tags": "warning,b2b,churn",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(ntfy_url, content=message, headers=headers)
            resp.raise_for_status()
    except Exception as exc:
        logger.warning("ntfy alert failed for %s/%s: %s", account_name, vendor_name, exc)
