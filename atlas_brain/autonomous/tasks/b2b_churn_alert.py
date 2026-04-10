"""Hourly churn signal spike alerts for B2B tenants.

Compares current signal counts / avg urgency against stored baselines
for each tenant's tracked vendors. Fires ntfy (and optionally email)
when thresholds are exceeded.
"""

import html
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from ...config import settings
from ...services.campaign_sender import get_campaign_sender
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
    alerts_failed = 0
    baselines_seeded = 0
    email_alerts_sent = 0
    ntfy_alerts_sent = 0
    vendors_checked = 0

    # Get all accounts with tracked vendors
    accounts = await pool.fetch(
        """
        SELECT sa.id AS account_id, sa.name AS account_name,
               tv.vendor_name,
               su.email AS owner_email,
               su.full_name AS owner_name
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
        # APPROVED-ENRICHMENT-READ: urgency_score, competitors_mentioned
        # Reason: 7-day signal aggregation for churn alerts
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
              AND duplicate_of_review_id IS NULL
              AND vendor_name ILIKE '%' || $1 || '%'
              AND enriched_at > NOW() - INTERVAL '7 days'
            """,
            vendor_name,
        )

        if not current or current["signal_count"] == 0:
            continue

        signal_count = float(current["signal_count"])
        metrics = {
            "signal_count": signal_count,
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
                baselines_seeded += 1
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
                if signal_count >= cfg.min_reviews_for_urgency:
                    should_alert = True
                else:
                    logger.info(
                        "Suppressed urgency alert for %s: delta=%.1f but signal_count=%d < min=%d",
                        vendor_name, delta, int(signal_count), cfg.min_reviews_for_urgency,
                    )
            elif metric_name == "displacement_count" and delta >= cfg.signal_count_threshold:
                should_alert = True

            if should_alert:
                delivery = await _send_alert_notifications(
                    account_name=row["account_name"],
                    vendor_name=vendor_name,
                    metric=metric_name,
                    baseline=baseline_value,
                    current=current_value,
                    owner_email=row["owner_email"],
                    owner_name=row["owner_name"],
                )
                if not delivery["delivered"]:
                    alerts_failed += 1
                    logger.warning(
                        "No churn alert delivery succeeded for %s/%s/%s",
                        row["account_name"],
                        vendor_name,
                        metric_name,
                    )
                    continue
                alerts_sent += 1
                email_alerts_sent += 1 if delivery["email"] else 0
                ntfy_alerts_sent += 1 if delivery["ntfy"] else 0

                # Dispatch webhook for churn alert (vendor-level data only, no account info)
                try:
                    await _dispatch_alert_webhook(pool, vendor_name, {
                        "metric": metric_name,
                        "baseline": baseline_value,
                        "current": current_value,
                        "delta": delta,
                    })
                except Exception:
                    logger.debug("Webhook dispatch skipped for churn alert")

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
        "alerts_failed": alerts_failed,
        "baselines_seeded": baselines_seeded,
        "email_alerts_sent": email_alerts_sent,
        "ntfy_alerts_sent": ntfy_alerts_sent,
    }


def _email_channel_configured() -> bool:
    """Return True when owner-email alert delivery is configured."""
    if not settings.b2b_alert.email_enabled:
        return False
    campaign_cfg = settings.campaign_sequence
    if campaign_cfg.sender_type == "ses":
        return bool(campaign_cfg.ses_from_email)
    return bool(campaign_cfg.resend_api_key and campaign_cfg.resend_from_email)


async def _send_alert_notifications(
    *,
    account_name: str,
    vendor_name: str,
    metric: str,
    baseline: float,
    current: float,
    owner_email: str,
    owner_name: str | None = None,
) -> dict[str, bool]:
    """Send churn alert over all configured notification channels."""
    email_sent = await _send_alert_email(
        account_name=account_name,
        vendor_name=vendor_name,
        metric=metric,
        baseline=baseline,
        current=current,
        owner_email=owner_email,
        owner_name=owner_name,
    )
    ntfy_sent = await _send_ntfy_alert(
        account_name=account_name,
        vendor_name=vendor_name,
        metric=metric,
        baseline=baseline,
        current=current,
    )
    return {
        "delivered": email_sent or ntfy_sent,
        "email": email_sent,
        "ntfy": ntfy_sent,
    }


async def _send_alert_email(
    *,
    account_name: str,
    vendor_name: str,
    metric: str,
    baseline: float,
    current: float,
    owner_email: str,
    owner_name: str | None,
) -> bool:
    """Send a churn alert email to the tenant owner."""
    if not owner_email or not _email_channel_configured():
        return False

    metric_labels = {
        "signal_count": "new churn signals",
        "avg_urgency": "average urgency score",
        "displacement_count": "competitive displacement mentions",
    }
    label = metric_labels.get(metric, metric)
    delta = current - baseline
    campaign_cfg = settings.campaign_sequence
    from_email = (
        campaign_cfg.ses_from_email
        if campaign_cfg.sender_type == "ses"
        else campaign_cfg.resend_from_email
    )
    sender_name = settings.b2b_alert.sender_name.strip()
    from_addr = f"{sender_name} <{from_email}>" if sender_name else from_email
    recipient_name = html.escape((owner_name or "there").strip() or "there")
    safe_account_name = html.escape(account_name)
    safe_vendor_name = html.escape(vendor_name)
    safe_label = html.escape(label)
    subject = f"Churn Alert: {vendor_name} signal spike"
    body = (
        f"<h2>Churn Signal Spike Detected</h2>"
        f"<p>Hi {recipient_name},</p>"
        f"<p>A material signal change was detected for <strong>{safe_vendor_name}</strong> "
        f"in <strong>{safe_account_name}</strong>.</p>"
        f"<ul>"
        f"<li><strong>Metric:</strong> {safe_label}</li>"
        f"<li><strong>Previous baseline:</strong> {baseline:.1f}</li>"
        f"<li><strong>Current value:</strong> {current:.1f}</li>"
        f"<li><strong>Delta:</strong> +{delta:.1f}</li>"
        f"</ul>"
    )
    dashboard_url = settings.b2b_alert.dashboard_base_url.strip()
    if dashboard_url:
        safe_dashboard_url = html.escape(dashboard_url.rstrip("/"), quote=True)
        body += (
            f"<p><a href=\"{safe_dashboard_url}\" "
            f"style=\"color:#2563eb;text-decoration:none;\">Open churn dashboard</a></p>"
        )
    body += "<p><em>Review the latest vendor intelligence to confirm whether pressure is accelerating.</em></p>"

    try:
        sender = get_campaign_sender()
        await sender.send(
            to=owner_email,
            from_email=from_addr,
            subject=subject,
            body=body,
            tags=[
                {"name": "task", "value": "b2b_churn_alert"},
                {"name": "vendor", "value": vendor_name},
                {"name": "metric", "value": metric},
            ],
        )
        return True
    except Exception as exc:
        logger.warning("Alert email failed for %s/%s: %s", account_name, vendor_name, exc)
        return False


async def _send_ntfy_alert(
    *,
    account_name: str,
    vendor_name: str,
    metric: str,
    baseline: float,
    current: float,
) -> bool:
    """Send ntfy notification for a churn signal spike."""
    if not settings.alerts.ntfy_enabled:
        return False

    ntfy_url = f"{settings.alerts.ntfy_url.rstrip('/')}/{settings.alerts.ntfy_topic}"
    headers = {
        "Title": f"Churn Alert: {vendor_name}",
        "Priority": "high",
        "Tags": "warning,b2b,churn",
    }
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

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(ntfy_url, content=message, headers=headers)
            resp.raise_for_status()
        return True
    except Exception as exc:
        logger.warning("ntfy alert failed for %s/%s: %s", account_name, vendor_name, exc)
        return False


async def _dispatch_alert_webhook(pool: Any, vendor_name: str, payload: dict[str, Any]) -> None:
    """Dispatch churn alert webhooks if webhook delivery is available."""
    from ...services.b2b.webhook_dispatcher import dispatch_webhooks

    await dispatch_webhooks(pool, "churn_alert", vendor_name, payload)
