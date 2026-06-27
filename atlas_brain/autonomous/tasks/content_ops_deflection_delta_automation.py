"""Scheduled generation for paid Content Ops deflection report deltas."""

from __future__ import annotations

import logging
from typing import Any

from extracted_content_pipeline.campaign_ports import SendRequest, SendResult
from extracted_content_pipeline.campaign_sender import create_campaign_sender
from extracted_content_pipeline.deflection_report_access import (
    PostgresDeflectionReportArtifactStore,
    compute_and_save_recent_deflection_deltas,
)

from ...config import settings
from ...content_ops_deflection_delivery import (
    DeflectionDeltaDeliveryConfig,
    pending_deflection_delta_delivery_count,
    send_pending_deflection_delta_deliveries,
)
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.tasks.content_ops_deflection_delta_automation")


class _DryRunSender:
    async def send(self, _request: SendRequest) -> SendResult:
        raise RuntimeError("dry-run deflection delta delivery should not send email")


def _task_metadata(task: ScheduledTask | Any) -> dict[str, Any]:
    metadata = getattr(task, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _metadata_int(task: ScheduledTask | Any, key: str, default: int) -> int:
    value = _task_metadata(task).get(key, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(parsed, 100))


def _metadata_bool(task: ScheduledTask | Any, key: str, default: bool) -> bool:
    metadata = _task_metadata(task)
    if key not in metadata:
        return default
    return bool(metadata.get(key))


def _configured(value: Any) -> bool:
    return bool(str(value or "").strip())


def _delta_delivery_missing_config(*, dry_run: bool) -> list[str]:
    cfg = settings.deflection_delivery
    missing: list[str] = []
    if not _configured(cfg.from_email):
        missing.append("from_email")
    if not dry_run and not _configured(cfg.resend_api_key):
        missing.append("resend_api_key")
    return missing


def _delta_delivery_config(*, dry_run: bool) -> DeflectionDeltaDeliveryConfig:
    cfg = settings.deflection_delivery
    return DeflectionDeltaDeliveryConfig(
        from_email=str(cfg.from_email or "").strip(),
        reply_to=str(cfg.reply_to or "").strip() or None,
        limit=int(cfg.limit),
        dry_run=dry_run,
    )


def _sender(*, dry_run: bool) -> Any:
    cfg = settings.deflection_delivery
    if dry_run:
        return _DryRunSender()
    return create_campaign_sender(
        "resend",
        {
            "api_key": cfg.resend_api_key,
            "api_url": cfg.resend_api_url,
            "timeout_seconds": cfg.resend_timeout_seconds,
        },
    )


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Generate persisted deltas for recent paid deflection reports."""

    cfg = settings.deflection_delta
    if not cfg.enabled:
        return {"_skip_synthesis": "Deflection delta automation disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    try:
        summary = await compute_and_save_recent_deflection_deltas(
            PostgresDeflectionReportArtifactStore(pool=pool),
            account_limit=_metadata_int(task, "account_limit", int(cfg.account_limit)),
            reports_per_account=_metadata_int(
                task,
                "reports_per_account",
                int(cfg.reports_per_account),
            ),
        )
    except Exception:
        logger.exception("Deflection delta automation failed")
        raise

    dry_run = _metadata_bool(
        task,
        "delivery_dry_run",
        bool(settings.deflection_delivery.dry_run),
    )
    delivery_summary = None
    missing_delivery = _delta_delivery_missing_config(dry_run=dry_run)
    pending_delivery_count = 0
    if not missing_delivery:
        delivery_summary = await send_pending_deflection_delta_deliveries(
            pool,
            sender=_sender(dry_run=dry_run),
            config=_delta_delivery_config(dry_run=dry_run),
        )
    else:
        pending_delivery_count = await pending_deflection_delta_delivery_count(pool)

    payload = {
        "accounts_scanned": summary.accounts_scanned,
        "reports_scanned": summary.reports_scanned,
        "deltas_saved": summary.deltas_saved,
        "delta_deliveries_enqueued": summary.delta_deliveries_enqueued,
        "skipped_no_delta": summary.skipped_no_delta,
        "failed": summary.failed,
        "delivery_scanned": delivery_summary.scanned if delivery_summary else 0,
        "delivery_sent": delivery_summary.sent if delivery_summary else 0,
        "delivery_failed": delivery_summary.failed if delivery_summary else 0,
        "delivery_deferred": (
            getattr(delivery_summary, "deferred", 0) if delivery_summary else 0
        ),
        "delivery_dry_run": delivery_summary.dry_run if delivery_summary else 0,
        "delivery_dry_run_enabled": dry_run,
        "account_limit_reached": summary.account_limit_reached,
        "account_limit_overflow": summary.account_limit_overflow,
        "reports_per_account_limit_reached": (
            summary.reports_per_account_limit_reached
        ),
        "reports_per_account_limit_overflow": (
            summary.reports_per_account_limit_overflow
        ),
        "report_limit_reached_accounts": list(summary.report_limit_reached_accounts),
        "report_limit_overflow_accounts": list(
            summary.report_limit_overflow_accounts
        ),
    }
    has_blocked_delivery = bool(
        missing_delivery
        and (summary.delta_deliveries_enqueued or pending_delivery_count)
    )
    if has_blocked_delivery:
        payload["delivery_missing_config"] = missing_delivery
        payload["delivery_pending"] = pending_delivery_count
        return {
            "_skip_synthesis": "Deflection delta delivery config missing",
            **payload,
        }
    if summary.reports_scanned == 0:
        return {
            "_skip_synthesis": "No paid deflection reports found for delta automation",
            **payload,
        }
    if summary.failed >= summary.reports_scanned:
        raise RuntimeError("Deflection delta automation failed for all scanned reports")
    if delivery_summary and delivery_summary.failed:
        if (
            not dry_run
            and delivery_summary.scanned > 0
            and delivery_summary.sent == 0
            and delivery_summary.failed >= delivery_summary.scanned
        ):
            raise RuntimeError("Deflection delta delivery failed for all scanned deliveries")
        return {"_skip_synthesis": "Deflection delta delivery degraded", **payload}
    if summary.failed:
        return {"_skip_synthesis": "Deflection delta automation degraded", **payload}
    if delivery_summary and getattr(delivery_summary, "deferred", 0):
        return {
            "_skip_synthesis": "Deflection delta delivery pending retries",
            **payload,
        }
    if summary.account_limit_overflow or summary.reports_per_account_limit_overflow:
        return {
            "_skip_synthesis": "Deflection delta automation scan window overflow",
            **payload,
        }
    if summary.account_limit_reached or summary.reports_per_account_limit_reached:
        return {
            "_skip_synthesis": "Deflection delta automation scan window saturated",
            **payload,
        }
    return {"_skip_synthesis": "Deflection delta automation complete", **payload}
