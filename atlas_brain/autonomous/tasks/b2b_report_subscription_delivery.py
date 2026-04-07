"""Recurring delivery for persisted B2B report subscriptions."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

from ...config import settings
from ...services.campaign_sender import get_campaign_sender
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from ...templates.email.report_subscription_delivery import (
    render_report_subscription_delivery_html,
)

logger = logging.getLogger("atlas.tasks.b2b_report_subscription_delivery")

_TIMESTAMP_KEYS = {
    "report_date",
    "created_at",
    "updated_at",
    "generated_at",
    "last_computed_at",
    "data_as_of_date",
    "as_of_date",
    "computed_at",
}
_REPORT_TYPE_FOCUS_MAP = {
    "battle_cards": {"battle_card", "challenger_brief"},
    "executive_reports": {
        "weekly_churn_feed",
        "vendor_scorecard",
        "vendor_deep_dive",
        "vendor_retention",
        "category_overview",
        "exploratory_overview",
    },
    "comparison_packs": {
        "vendor_comparison",
        "account_comparison",
        "account_deep_dive",
        "displacement_report",
        "challenger_intel",
        "challenger_brief",
    },
}
_REPORT_FREQUENCY_DAYS = {
    "weekly": 7,
    "monthly": 30,
    "quarterly": 90,
}


def _safe_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        import json

        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _campaign_from_email() -> str:
    campaign_cfg = settings.campaign_sequence
    if campaign_cfg.sender_type == "ses":
        return str(campaign_cfg.ses_from_email or "").strip()
    return str(campaign_cfg.resend_from_email or "").strip()


def _frontend_base_url() -> str:
    configured = str(settings.b2b_report_delivery.dashboard_base_url or "").strip()
    if configured:
        return configured.rstrip("/")
    return str(settings.saas_auth.frontend_base_url or "").strip().rstrip("/")


def _scope_manage_url(scope_type: str, scope_key: str) -> str | None:
    base_url = _frontend_base_url()
    if not base_url:
        return None
    if scope_type == "report":
        return f"{base_url}/reports/{quote(scope_key, safe='')}"
    return f"{base_url}/reports"


def _report_url(report_id: Any) -> str | None:
    base_url = _frontend_base_url()
    if not base_url or not report_id:
        return None
    return f"{base_url}/reports/{quote(str(report_id), safe='')}"


def _format_report_type_label(report_type: str) -> str:
    return str(report_type or "report").replace("_", " ").title()


def _report_display_title(row) -> str:
    report_type = str(row["report_type"] or "")
    vendor_filter = str(row["vendor_filter"] or "").strip()
    category_filter = str(row["category_filter"] or "").strip()
    if report_type in {"vendor_comparison", "account_comparison"} and vendor_filter and category_filter:
        return f"{vendor_filter} vs {category_filter}"
    if report_type == "challenger_brief" and vendor_filter and category_filter:
        return f"{vendor_filter} -> {category_filter}"
    return vendor_filter or category_filter or _format_report_type_label(report_type)


def _report_trust_label(row, intelligence_data: dict[str, Any]) -> str:
    quality_record = _safe_json(intelligence_data.get("battle_card_quality"))
    quality_status = str(row["quality_status"] or intelligence_data.get("quality_status") or "").strip().lower()
    if not quality_status and isinstance(quality_record, dict):
        quality_status = str(quality_record.get("status") or "").strip().lower()
    report_type = str(row["report_type"] or "")
    status = str(row["status"] or "").strip().lower()

    if report_type == "battle_card" and quality_status:
        if quality_status == "sales_ready":
            return "Evidence-backed"
        if quality_status == "needs_review":
            return "Needs review"
        if quality_status == "thin_evidence":
            return "Thin evidence"
        return "Fallback render"

    if status == "completed":
        return "Persisted artifact"
    return "In workflow"


def _parse_timestamp_candidate(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _collect_timestamp_candidates(value: Any, results: list[datetime], *, depth: int = 0, seen: set[int] | None = None) -> None:
    if depth > 5 or value is None:
        return
    if seen is None:
        seen = set()
    if isinstance(value, (dict, list)):
        marker = id(value)
        if marker in seen:
            return
        seen.add(marker)

    if isinstance(value, list):
        for item in value:
            _collect_timestamp_candidates(item, results, depth=depth + 1, seen=seen)
        return

    if not isinstance(value, dict):
        return

    for key, child in value.items():
        if key in _TIMESTAMP_KEYS:
            parsed = _parse_timestamp_candidate(child)
            if parsed is not None:
                results.append(parsed)
        if isinstance(child, (dict, list)):
            _collect_timestamp_candidates(child, results, depth=depth + 1, seen=seen)


def _relative_age_label(age_hours: float) -> str:
    rounded = max(1, int(round(age_hours)))
    if rounded < 24:
        return f"{rounded}h ago"
    if rounded < 24 * 7:
        return f"{max(1, int(round(rounded / 24)))}d ago"
    return f"{max(1, int(round(rounded / (24 * 7))))}w ago"


def _derive_freshness(row, intelligence_data: dict[str, Any], data_density: dict[str, Any]) -> dict[str, Any]:
    fresh_hours = settings.b2b_report_delivery.fresh_hours
    monitor_hours = settings.b2b_report_delivery.monitor_hours
    candidates: list[datetime] = []
    for candidate in (
        row["report_date"],
        row["created_at"],
    ):
        parsed = _parse_timestamp_candidate(candidate)
        if parsed is not None:
            candidates.append(parsed)

    _collect_timestamp_candidates(intelligence_data, candidates)
    _collect_timestamp_candidates(data_density, candidates)

    if not candidates:
        return {
            "state": "unknown",
            "label": "Unknown",
            "detail": "No freshness timestamp was attached to this artifact.",
            "anchor": None,
        }

    anchor = max(candidates)
    age_hours = max(0.0, (datetime.now(timezone.utc) - anchor).total_seconds() / 3600.0)
    if age_hours <= fresh_hours:
        return {
            "state": "fresh",
            "label": "Fresh",
            "detail": f"Evidence window looks current ({_relative_age_label(age_hours)}).",
            "anchor": anchor,
        }
    if age_hours <= monitor_hours:
        return {
            "state": "monitor",
            "label": "Monitor",
            "detail": f"Artifact is aging ({_relative_age_label(age_hours)}). Recheck before external use.",
            "anchor": anchor,
        }
    return {
        "state": "stale",
        "label": "Stale",
        "detail": f"Refresh recommended. Latest evidence is {_relative_age_label(age_hours)} old.",
        "anchor": anchor,
    }


def _freshness_allowed(policy: str, freshness_state: str) -> bool:
    if policy == "any":
        return True
    if policy == "fresh_only":
        return freshness_state == "fresh"
    return freshness_state in {"fresh", "monitor"}


def _tracked_vendor_allowed(row, tracked_vendors: set[str], account_id: Any) -> bool:
    vendor_filter = str(row["vendor_filter"] or "").strip().lower()
    row_account_id = row["account_id"]
    if row_account_id is not None and str(row_account_id) == str(account_id):
        return True
    if not vendor_filter:
        return True
    return vendor_filter in tracked_vendors


def _focus_allowed(scope_type: str, focus: str, report_type: str) -> bool:
    if scope_type == "report" or focus == "all":
        return True
    allowed_types = _REPORT_TYPE_FOCUS_MAP.get(focus)
    if not allowed_types:
        return True
    return report_type in allowed_types


def _shorten(value: str, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _evidence_highlight_from_item(item: Any) -> str | None:
    if not isinstance(item, dict):
        text = str(item or "").strip()
        return _shorten(text) if text else None

    excerpt = str(item.get("excerpt_text") or item.get("quote") or item.get("text") or "").strip()
    if not excerpt:
        return None

    meta: list[str] = []
    reviewer_company = str(item.get("reviewer_company") or "").strip()
    reviewer_title = str(item.get("reviewer_title") or item.get("role") or "").strip()
    competitor = str(item.get("competitor") or "").strip()
    pain_category = str(item.get("pain_category") or "").strip()
    time_anchor = str(item.get("time_anchor") or "").strip()

    if reviewer_company:
        meta.append(reviewer_company)
    if reviewer_title:
        meta.append(reviewer_title)
    if competitor:
        meta.append(f"vs {competitor}")
    if pain_category:
        meta.append(f"pain: {pain_category}")
    if time_anchor:
        meta.append(time_anchor)

    if meta:
        return f"{_shorten(excerpt)} ({'; '.join(meta)})"
    return _shorten(excerpt)


def _extract_evidence_highlights(intelligence_data: dict[str, Any], *, limit: int = 3) -> list[str]:
    for key in ("witness_highlights", "reasoning_witness_highlights"):
        values = intelligence_data.get(key)
        if not isinstance(values, list):
            continue
        highlights: list[str] = []
        for item in values:
            line = _evidence_highlight_from_item(item)
            if not line:
                continue
            highlights.append(line)
            if len(highlights) >= limit:
                return highlights

    values = intelligence_data.get("customer_pain_quotes")
    if isinstance(values, list):
        highlights = []
        for item in values:
            line = _evidence_highlight_from_item(item)
            if not line:
                continue
            highlights.append(line)
            if len(highlights) >= limit:
                return highlights
    return []


def _frequency_label(frequency: str) -> str:
    return str(frequency or "weekly").replace("_", " ").title()


def _next_delivery_at(frequency: str) -> datetime:
    days = _REPORT_FREQUENCY_DAYS.get(str(frequency or "weekly"), 7)
    return datetime.now(timezone.utc).replace(microsecond=0) + timedelta(days=days)


def _subscription_summary(scope_type: str, artifacts: list[dict[str, Any]]) -> str:
    if scope_type == "report" and artifacts:
        return f"{artifacts[0]['title']} is ready from your saved recurring report subscription."
    count = len(artifacts)
    if count == 1:
        return "1 persisted artifact is ready from your saved recurring report subscription."
    return f"{count} persisted artifacts are ready from your saved recurring report subscription."


def _aggregate_freshness_state(artifacts: list[dict[str, Any]]) -> str:
    states = {str(artifact["freshness_state"]) for artifact in artifacts if artifact.get("freshness_state")}
    if not states:
        return "none"
    if len(states) == 1:
        return next(iter(states))
    return "mixed"


def _delivery_status_summary(
    status: str,
    artifact_count: int,
    recipient_count: int,
    successful_recipient_count: int,
) -> str:
    if status == "sent":
        return f"Delivered {artifact_count} artifact(s) to {recipient_count} recipient(s)."
    if status == "partial":
        return (
            f"Partially delivered {artifact_count} artifact(s) to "
            f"{successful_recipient_count} of {recipient_count} recipient(s)."
        )
    if status == "skipped":
        return "Skipped delivery because no eligible persisted artifacts matched the saved policy."
    return "Recurring delivery failed before any recipient received the package."


async def _tracked_vendors(pool, account_id: Any) -> set[str]:
    rows = await pool.fetch(
        """
        SELECT vendor_name
        FROM tracked_vendors
        WHERE account_id = $1
        """,
        account_id,
    )
    return {
        str(row["vendor_name"] or "").strip().lower()
        for row in rows
        if str(row["vendor_name"] or "").strip()
    }


async def _claim_delivery_attempt(pool, row) -> Any | None:
    stale_claim_seconds = settings.b2b_report_delivery.stale_claim_seconds
    return await pool.fetchrow(
        """
        INSERT INTO b2b_report_subscription_delivery_log (
            subscription_id,
            account_id,
            scheduled_for,
            scope_type,
            scope_key,
            recipient_emails,
            delivery_frequency,
            deliverable_focus,
            freshness_policy,
            status,
            delivered_at
        )
        VALUES (
            $1,
            $2,
            $3,
            $4,
            $5,
            $6::text[],
            $7,
            $8,
            $9,
            'processing',
            NOW()
        )
        ON CONFLICT (subscription_id, scheduled_for) DO UPDATE
        SET recipient_emails = EXCLUDED.recipient_emails,
            delivery_frequency = EXCLUDED.delivery_frequency,
            deliverable_focus = EXCLUDED.deliverable_focus,
            freshness_policy = EXCLUDED.freshness_policy,
            delivered_report_ids = '{}'::uuid[],
            message_ids = '{}'::text[],
            freshness_state = 'none',
            status = 'processing',
            summary = '',
            error = NULL,
            delivered_at = NOW()
        WHERE b2b_report_subscription_delivery_log.status IN ('failed', 'processing')
          AND b2b_report_subscription_delivery_log.delivered_at < NOW() - ($10 || ' seconds')::interval
        RETURNING id
        """,
        row["id"],
        row["account_id"],
        row["next_delivery_at"],
        row["scope_type"],
        row["scope_key"],
        list(row["recipient_emails"] or []),
        row["delivery_frequency"],
        row["deliverable_focus"],
        row["freshness_policy"],
        str(stale_claim_seconds),
    )


async def _finalize_delivery_attempt(
    pool,
    *,
    delivery_log_id: Any,
    status: str,
    freshness_state: str,
    delivered_report_ids: list[Any],
    message_ids: list[str],
    summary: str,
    error: str | None,
) -> None:
    await pool.execute(
        """
        UPDATE b2b_report_subscription_delivery_log
        SET status = $2,
            freshness_state = $3,
            delivered_report_ids = $4::uuid[],
            message_ids = $5::text[],
            summary = $6,
            error = $7,
            delivered_at = NOW()
        WHERE id = $1
        """,
        delivery_log_id,
        status,
        freshness_state,
        delivered_report_ids,
        message_ids,
        summary[:1000],
        (error or "")[:2000] or None,
    )


async def _advance_subscription(pool, subscription_id: Any, delivery_frequency: str) -> None:
    await pool.execute(
        """
        UPDATE b2b_report_subscriptions
        SET next_delivery_at = $2
        WHERE id = $1
        """,
        subscription_id,
        _next_delivery_at(delivery_frequency),
    )


async def _fetch_report_row(pool, report_id: Any) -> Any | None:
    return await pool.fetchrow(
        """
        SELECT id, account_id, report_date, report_type, executive_summary,
               vendor_filter, category_filter, status, created_at,
               intelligence_data, data_density,
               CASE
                 WHEN report_type = 'battle_card'
                 THEN COALESCE(intelligence_data->>'quality_status', intelligence_data->'battle_card_quality'->>'status')
                 ELSE NULL
               END AS quality_status
        FROM b2b_intelligence
        WHERE id = $1
        """,
        report_id,
    )


async def _fetch_library_rows(pool, account_id: Any, tracked_vendors: set[str], focus: str) -> list[Any]:
    allowed_types = _REPORT_TYPE_FOCUS_MAP.get(focus)
    scan_limit = settings.b2b_report_delivery.max_reports_per_delivery * 5
    return await pool.fetch(
        """
        SELECT id, account_id, report_date, report_type, executive_summary,
               vendor_filter, category_filter, status, created_at,
               intelligence_data, data_density,
               CASE
                 WHEN report_type = 'battle_card'
                 THEN COALESCE(intelligence_data->>'quality_status', intelligence_data->'battle_card_quality'->>'status')
                 ELSE NULL
               END AS quality_status
        FROM b2b_intelligence
        WHERE status = 'completed'
          AND (
                vendor_filter IS NULL
                OR vendor_filter = ''
                OR LOWER(vendor_filter) = ANY($2::text[])
                OR account_id = $1
          )
          AND ($3::text[] IS NULL OR report_type = ANY($3::text[]))
        ORDER BY report_date DESC NULLS LAST, created_at DESC NULLS LAST, id DESC
        LIMIT $4
        """,
        account_id,
        list(tracked_vendors),
        list(allowed_types) if allowed_types else None,
        scan_limit,
    )


def _build_delivery_artifact(row) -> dict[str, Any]:
    intelligence_data = _safe_json(row["intelligence_data"])
    if not isinstance(intelligence_data, dict):
        intelligence_data = {}
    data_density = _safe_json(row["data_density"])
    if not isinstance(data_density, dict):
        data_density = {}
    freshness = _derive_freshness(row, intelligence_data, data_density)
    executive_summary = str(
        row["executive_summary"]
        or intelligence_data.get("executive_summary")
        or "No executive summary is attached to this artifact yet."
    ).strip()
    return {
        "report_id": row["id"],
        "report_type": str(row["report_type"] or ""),
        "title": _report_display_title(row),
        "type_label": _format_report_type_label(str(row["report_type"] or "")),
        "trust_label": _report_trust_label(row, intelligence_data),
        "freshness_state": freshness["state"],
        "freshness_label": freshness["label"],
        "freshness_detail": freshness["detail"],
        "executive_summary": executive_summary,
        "evidence_highlights": _extract_evidence_highlights(intelligence_data),
        "report_url": _report_url(row["id"]),
    }


async def _resolve_artifacts(pool, row, tracked_vendors: set[str]) -> list[dict[str, Any]]:
    scope_type = str(row["scope_type"] or "")
    freshness_policy = str(row["freshness_policy"] or "fresh_or_monitor")

    if scope_type == "report":
        report_row = await _fetch_report_row(pool, row["report_id"])
        if not report_row:
            return []
        if str(report_row["status"] or "").strip().lower() != "completed":
            return []
        if not _tracked_vendor_allowed(report_row, tracked_vendors, row["account_id"]):
            return []
        artifact = _build_delivery_artifact(report_row)
        if not _freshness_allowed(freshness_policy, artifact["freshness_state"]):
            return []
        return [artifact]

    rows = await _fetch_library_rows(
        pool,
        row["account_id"],
        tracked_vendors,
        str(row["deliverable_focus"] or "all"),
    )
    artifacts: list[dict[str, Any]] = []
    for report_row in rows:
        report_type = str(report_row["report_type"] or "")
        if not _focus_allowed(scope_type, str(row["deliverable_focus"] or "all"), report_type):
            continue
        artifact = _build_delivery_artifact(report_row)
        if not _freshness_allowed(freshness_policy, artifact["freshness_state"]):
            continue
        artifacts.append(artifact)
        if len(artifacts) >= settings.b2b_report_delivery.max_reports_per_delivery:
            break
    return artifacts


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Deliver due persisted report subscriptions through the campaign sender."""
    delivery_cfg = settings.b2b_report_delivery
    if not delivery_cfg.enabled:
        return {"_skip_synthesis": "Recurring report delivery disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    from_email = _campaign_from_email()
    if not from_email:
        return {"_skip_synthesis": "Campaign sender not configured"}

    try:
        sender = get_campaign_sender()
    except Exception as exc:
        logger.warning("Recurring report delivery sender unavailable: %s", exc)
        return {"_skip_synthesis": "Campaign sender unavailable"}

    due_rows = await pool.fetch(
        """
        SELECT s.id, s.account_id, sa.name AS account_name,
               s.report_id, s.scope_type, s.scope_key, s.scope_label,
               s.delivery_frequency, s.deliverable_focus, s.freshness_policy,
               s.recipient_emails, s.delivery_note, s.next_delivery_at
        FROM b2b_report_subscriptions s
        JOIN saas_accounts sa ON sa.id = s.account_id
        WHERE s.enabled = TRUE
          AND s.next_delivery_at IS NOT NULL
          AND s.next_delivery_at <= NOW()
          AND sa.product IN ('b2b_retention', 'b2b_challenger')
          AND sa.plan_status IN ('trialing', 'active', 'past_due')
        ORDER BY s.next_delivery_at ASC, s.created_at ASC
        LIMIT $1
        """,
        delivery_cfg.max_subscriptions_per_run,
    )

    if not due_rows:
        return {"_skip_synthesis": "No due report subscriptions"}

    attempts_claimed = 0
    delivered = 0
    partially_delivered = 0
    skipped = 0
    failed = 0

    for row in due_rows:
        claim = await _claim_delivery_attempt(pool, row)
        if not claim:
            continue

        attempts_claimed += 1
        delivery_log_id = claim["id"]
        recipients = list(row["recipient_emails"] or [])
        status = "failed"
        summary = "Recurring delivery failed before any recipient received the package."
        error_message = None
        message_ids: list[str] = []
        delivered_report_ids: list[Any] = []
        freshness_state = "none"

        try:
            tracked_vendors = await _tracked_vendors(pool, row["account_id"])
            artifacts = await _resolve_artifacts(pool, row, tracked_vendors)
            delivered_report_ids = [artifact["report_id"] for artifact in artifacts]
            freshness_state = _aggregate_freshness_state(artifacts)

            if not recipients:
                error_message = "Subscription has no recipients"
                status = "failed"
            elif not artifacts:
                status = "skipped"
                summary = _delivery_status_summary(status, 0, len(recipients))
            else:
                subject = (
                    f"{artifacts[0]['title']} | Recurring delivery"
                    if row["scope_type"] == "report" and len(artifacts) == 1
                    else f"{row['account_name']}: {len(artifacts)} report artifact(s) ready"
                )
                summary_line = _subscription_summary(str(row["scope_type"] or ""), artifacts)
                manage_url = _scope_manage_url(str(row["scope_type"] or ""), str(row["scope_key"] or ""))
                html_body = render_report_subscription_delivery_html(
                    account_name=str(row["account_name"] or ""),
                    scope_label=str(row["scope_label"] or "Recurring delivery"),
                    summary_line=summary_line,
                    frequency_label=_frequency_label(str(row["delivery_frequency"] or "")),
                    manage_url=manage_url,
                    delivery_note=str(row["delivery_note"] or ""),
                    artifacts=artifacts,
                )
                send_failures: list[str] = []
                sender_name = str(delivery_cfg.sender_name or "").strip()
                from_addr = f"{sender_name} <{from_email}>" if sender_name else from_email
                for recipient in recipients:
                    try:
                        result = await sender.send(
                            to=recipient,
                            from_email=from_addr,
                            subject=subject,
                            body=html_body,
                            tags=[
                                {"name": "task", "value": "b2b_report_subscription_delivery"},
                                {"name": "subscription_id", "value": str(row["id"])},
                                {"name": "scope_type", "value": str(row["scope_type"] or "")},
                            ],
                        )
                        if result.get("id"):
                            message_ids.append(str(result["id"]))
                    except Exception as exc:
                        send_failures.append(f"{recipient}: {exc}")

                if send_failures and len(send_failures) == len(recipients):
                    status = "failed"
                    error_message = "; ".join(send_failures)
                elif send_failures:
                    status = "partial"
                    error_message = "; ".join(send_failures)
                else:
                    status = "sent"

                summary = _delivery_status_summary(
                    status,
                    len(artifacts),
                    len(recipients),
                    len(message_ids),
                )

            await _finalize_delivery_attempt(
                pool,
                delivery_log_id=delivery_log_id,
                status=status,
                freshness_state=freshness_state,
                delivered_report_ids=delivered_report_ids,
                message_ids=message_ids,
                summary=summary,
                error=error_message,
            )

            if status in {"sent", "partial", "skipped"}:
                await _advance_subscription(pool, row["id"], str(row["delivery_frequency"] or "weekly"))

            if status == "sent":
                delivered += 1
            elif status == "partial":
                partially_delivered += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
        except Exception as exc:
            logger.exception("Recurring report delivery failed for subscription %s", row["id"])
            await _finalize_delivery_attempt(
                pool,
                delivery_log_id=delivery_log_id,
                status="failed",
                freshness_state=freshness_state,
                delivered_report_ids=delivered_report_ids,
                message_ids=message_ids,
                summary=summary,
                error=str(exc),
            )
            failed += 1

    if attempts_claimed == 0:
        return {"_skip_synthesis": "No report subscriptions were claimable"}

    return {
        "_skip_synthesis": "Recurring report delivery complete",
        "due_subscriptions": len(due_rows),
        "attempts_claimed": attempts_claimed,
        "delivered": delivered,
        "partially_delivered": partially_delivered,
        "skipped": skipped,
        "failed": failed,
    }
