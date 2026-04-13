"""Recurring delivery for saved-view watchlist alert emails."""

from __future__ import annotations

import logging
import uuid as _uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from ...api.b2b_tenant_dashboard import (
    list_tenant_accounts_in_motion_feed,
    list_tenant_slow_burn_watchlist,
)
from ...auth.dependencies import AuthUser
from ...config import settings
from ...services.b2b import watchlist_alerts as watchlist_alert_service
from ...services.campaign_sender import get_campaign_sender
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.tasks.b2b_watchlist_alert_delivery")


def _synthetic_account_user(*, account_id: Any, product: str) -> AuthUser:
    return AuthUser(
        user_id="00000000-0000-0000-0000-000000000000",
        account_id=str(account_id),
        plan="b2b_trial",
        plan_status="active",
        role="owner",
        product=product or "b2b_retention",
        is_admin=True,
    )


async def _fetch_due_views(pool, limit: int) -> list[Any]:
    return await pool.fetch(
        """
        SELECT v.id, v.account_id, v.name, v.vendor_names, v.category, v.source, v.min_urgency,
               v.include_stale, v.named_accounts_only, v.changed_wedges_only,
               v.vendor_alert_threshold, v.account_alert_threshold,
               v.preview_alerts_enabled, v.preview_alert_min_confidence,
               v.preview_alert_require_budget_authority, v.stale_days_threshold,
               v.alert_email_enabled, v.alert_delivery_frequency, v.next_alert_delivery_at,
               v.last_alert_delivery_at, v.last_alert_delivery_status, v.last_alert_delivery_summary,
               v.created_at, v.updated_at,
               sa.name AS account_name,
               sa.product,
               sa.plan_status
        FROM b2b_watchlist_views v
        JOIN saas_accounts sa ON sa.id = v.account_id
        WHERE v.alert_email_enabled = TRUE
          AND v.next_alert_delivery_at IS NOT NULL
          AND v.next_alert_delivery_at <= NOW()
          AND sa.product IN ('b2b_retention', 'b2b_challenger')
          AND sa.plan_status IN ('trialing', 'active', 'past_due')
        ORDER BY v.next_alert_delivery_at ASC, v.updated_at DESC, v.created_at DESC
        LIMIT $1
        """,
        limit,
    )


async def _claim_delivery_attempt(pool, row) -> Any | None:
    cfg = settings.b2b_watchlist_delivery
    now = datetime.now(timezone.utc)
    return await pool.fetchrow(
        """
        INSERT INTO b2b_watchlist_alert_email_log (
            id, account_id, watchlist_view_id, event_ids, recipient_emails, message_ids,
            event_count, status, summary, error, delivered_at, created_at, updated_at,
            scheduled_for, delivery_frequency, delivery_mode
        )
        VALUES (
            $1, $2, $3, '{}'::uuid[], '{}'::text[], '{}'::text[],
            0, 'processing', 'Processing scheduled watchlist alert delivery', NULL, NULL, $4, $4,
            $5, $6, 'scheduled'
        )
        ON CONFLICT (watchlist_view_id, scheduled_for, delivery_mode)
        WHERE scheduled_for IS NOT NULL
        DO UPDATE
        SET status = 'processing',
            summary = EXCLUDED.summary,
            error = NULL,
            updated_at = EXCLUDED.updated_at
        WHERE (
                b2b_watchlist_alert_email_log.status = 'processing'
                AND b2b_watchlist_alert_email_log.updated_at < NOW() - ($7 || ' seconds')::interval
              )
           OR b2b_watchlist_alert_email_log.status = 'failed'
        RETURNING id
        """,
        _uuid.uuid4(),
        row["account_id"],
        row["id"],
        now,
        row["next_alert_delivery_at"],
        str(row["alert_delivery_frequency"] or "daily"),
        str(int(cfg.stale_claim_seconds)),
    )


async def _advance_view_schedule(
    pool,
    *,
    view_id: Any,
    account_id: Any,
    frequency: str,
    status: str,
    summary: str,
    scheduled_for: datetime | None,
) -> None:
    now = datetime.now(timezone.utc)
    if status == "failed":
        next_due = now + timedelta(seconds=int(settings.b2b_watchlist_delivery.failed_retry_seconds))
    else:
        next_due = watchlist_alert_service.next_watchlist_alert_delivery_at(
            frequency,
            anchor=scheduled_for,
            from_now=now,
        )
    await pool.execute(
        """
        UPDATE b2b_watchlist_views
        SET next_alert_delivery_at = $3,
            last_alert_delivery_at = $4,
            last_alert_delivery_status = $5,
            last_alert_delivery_summary = $6,
            updated_at = $4
        WHERE id = $1
          AND account_id = $2
        """,
        view_id,
        account_id,
        next_due,
        now,
        status,
        summary[:1000],
    )


async def _send_scheduled_watchlist_email(pool, sender, row, delivery_log_id: _uuid.UUID) -> str:
    account_id = row["account_id"]
    view_id = row["id"]
    scheduled_for = row["next_alert_delivery_at"]
    frequency = str(row["alert_delivery_frequency"] or "daily")
    account_name = str(row["account_name"] or "your account")
    user = _synthetic_account_user(account_id=account_id, product=str(row["product"] or "b2b_retention"))

    evaluation = await watchlist_alert_service.evaluate_watchlist_alert_events_for_view(
        pool,
        account_id=account_id,
        view_id=view_id,
        view_row=row,
        user=user,
        slow_burn_loader=list_tenant_slow_burn_watchlist,
        accounts_loader=list_tenant_accounts_in_motion_feed,
    )
    events = evaluation["events"]
    suppressed_preview_summary = evaluation.get("suppressed_preview_summary")
    event_ids = [_uuid.UUID(str(event["id"])) for event in events]
    recipients = await watchlist_alert_service.resolve_watchlist_alert_recipients(pool, account_id=account_id)
    recipient_emails = [item["email"] for item in recipients]
    now = datetime.now(timezone.utc)

    if not events:
        summary = (
            "No open alert events to deliver"
            f"{watchlist_alert_service.suppressed_preview_summary_suffix(suppressed_preview_summary)}"
        )
        await watchlist_alert_service.update_watchlist_alert_email_log(
            pool,
            log_id=delivery_log_id,
            event_ids=event_ids,
            recipient_emails=recipient_emails,
            message_ids=[],
            event_count=0,
            status="no_events",
            summary=summary,
            error=None,
            delivered_at=now,
            recorded_at=now,
        )
        await _advance_view_schedule(
            pool,
            view_id=view_id,
            account_id=account_id,
            frequency=frequency,
            status="no_events",
            summary=summary,
            scheduled_for=scheduled_for,
        )
        return "no_events"

    if not recipient_emails:
        summary = "Watchlist alert email delivery failed before send"
        error = "No active owner email is configured for this account"
        await watchlist_alert_service.update_watchlist_alert_email_log(
            pool,
            log_id=delivery_log_id,
            event_ids=event_ids,
            recipient_emails=[],
            message_ids=[],
            event_count=len(events),
            status="failed",
            summary=summary,
            error=error,
            delivered_at=None,
            recorded_at=now,
        )
        await _advance_view_schedule(
            pool,
            view_id=view_id,
            account_id=account_id,
            frequency=frequency,
            status="failed",
            summary=summary,
            scheduled_for=scheduled_for,
        )
        return "failed"

    from_addr = watchlist_alert_service.watchlist_alert_from_email()
    if not from_addr:
        summary = "Watchlist alert email delivery failed before send"
        error = "Campaign sender from-address is not configured"
        await watchlist_alert_service.update_watchlist_alert_email_log(
            pool,
            log_id=delivery_log_id,
            event_ids=event_ids,
            recipient_emails=recipient_emails,
            message_ids=[],
            event_count=len(events),
            status="failed",
            summary=summary,
            error=error,
            delivered_at=None,
            recorded_at=now,
        )
        await _advance_view_schedule(
            pool,
            view_id=view_id,
            account_id=account_id,
            frequency=frequency,
            status="failed",
            summary=summary,
            scheduled_for=scheduled_for,
        )
        return "failed"

    html_body = watchlist_alert_service.render_watchlist_alert_email_html(
        account_name=account_name,
        view_name=str(row["name"]),
        summary_line=f"{len(events)} open saved-view alert{'s' if len(events) != 1 else ''} are ready for review.",
        events=events,
    )
    subject = f"Watchlist alerts: {row['name']}"

    delivered_count = 0
    suppressed_count = 0
    message_ids: list[str] = []
    errors: list[str] = []
    for recipient in recipient_emails:
        sent, message_id, send_error, suppressed = await watchlist_alert_service.send_watchlist_alert_email(
            pool,
            sender,
            recipient=recipient,
            from_addr=from_addr,
            subject=subject,
            html_body=html_body,
            view_name=str(row["name"]),
        )
        if sent:
            delivered_count += 1
            if message_id:
                message_ids.append(message_id)
        elif suppressed:
            suppressed_count += 1
            if send_error:
                errors.append(send_error)
        elif send_error:
            errors.append(send_error)

    if suppressed_count == len(recipient_emails):
        status = "skipped"
        summary = "Skipped delivery because every saved-view recipient is currently suppressed."
        error_text = "; ".join(errors) if errors else None
        delivered_at = now
    else:
        status = "sent"
        if delivered_count == 0:
            status = "failed"
        elif delivered_count < len(recipient_emails):
            status = "partial"
        summary = (
            f"Delivered watchlist alert email to {delivered_count} of {len(recipient_emails)} recipient"
            f"{'' if len(recipient_emails) == 1 else 's'}"
        )
        if suppressed_count > 0:
            summary += f" ({suppressed_count} suppressed)"
        error_text = "; ".join(errors) if errors else None
        delivered_at = now if delivered_count > 0 else None

    await watchlist_alert_service.update_watchlist_alert_email_log(
        pool,
        log_id=delivery_log_id,
        event_ids=event_ids,
        recipient_emails=recipient_emails,
        message_ids=message_ids,
        event_count=len(events),
        status=status,
        summary=summary,
        error=error_text,
        delivered_at=delivered_at,
        recorded_at=now,
    )
    await _advance_view_schedule(
        pool,
        view_id=view_id,
        account_id=account_id,
        frequency=frequency,
        status=status,
        summary=summary,
        scheduled_for=scheduled_for,
    )
    return status


async def run(task: ScheduledTask) -> dict[str, Any]:
    cfg = settings.b2b_watchlist_delivery
    if not cfg.enabled:
        return {"_skip_synthesis": "Watchlist alert delivery disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    if not watchlist_alert_service.watchlist_alert_sender_configured():
        return {"_skip_synthesis": "Campaign sender not configured for watchlist alert delivery"}

    try:
        sender = get_campaign_sender()
    except Exception as exc:
        logger.warning("Watchlist alert delivery sender unavailable: %s", exc)
        return {"_skip_synthesis": "Campaign sender unavailable"}

    due_rows = await _fetch_due_views(pool, int(cfg.max_views_per_run))
    if not due_rows:
        return {"_skip_synthesis": "No due watchlist alert deliveries"}

    attempts_claimed = 0
    sent = 0
    partial = 0
    failed = 0
    no_events = 0
    skipped = 0

    for row in due_rows:
        claim = await _claim_delivery_attempt(pool, row)
        if not claim:
            continue
        attempts_claimed += 1
        try:
            status = await _send_scheduled_watchlist_email(pool, sender, row, claim["id"])
        except Exception as exc:
            logger.exception("Scheduled watchlist alert delivery failed for view %s", row["id"])
            await watchlist_alert_service.update_watchlist_alert_email_log(
                pool,
                log_id=claim["id"],
                event_ids=[],
                recipient_emails=[],
                message_ids=[],
                event_count=0,
                status="failed",
                summary="Watchlist alert email delivery failed before completion",
                error=str(exc),
                delivered_at=None,
                recorded_at=datetime.now(timezone.utc),
            )
            await _advance_view_schedule(
                pool,
                view_id=row["id"],
                account_id=row["account_id"],
                frequency=str(row["alert_delivery_frequency"] or "daily"),
                status="failed",
                summary="Watchlist alert email delivery failed before completion",
                scheduled_for=row["next_alert_delivery_at"],
            )
            failed += 1
            continue

        if status == "sent":
            sent += 1
        elif status == "partial":
            partial += 1
        elif status == "no_events":
            no_events += 1
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1

    if attempts_claimed == 0:
        return {"_skip_synthesis": "No watchlist alert deliveries were claimable"}

    return {
        "_skip_synthesis": "Watchlist alert delivery complete",
        "due_views": len(due_rows),
        "attempts_claimed": attempts_claimed,
        "sent": sent,
        "partial": partial,
        "no_events": no_events,
        "skipped": skipped,
        "failed": failed,
    }
