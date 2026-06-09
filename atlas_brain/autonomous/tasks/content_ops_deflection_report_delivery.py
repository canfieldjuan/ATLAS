"""Scheduled delivery for queued paid Content Ops deflection reports."""

from __future__ import annotations

import logging
from typing import Any

from extracted_content_pipeline.campaign_ports import SendRequest, SendResult
from extracted_content_pipeline.campaign_sender import create_campaign_sender

from ...config import settings
from ...content_ops_deflection_delivery import (
    DeflectionReportDeliveryConfig,
    send_pending_deflection_report_deliveries,
)
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.tasks.content_ops_deflection_report_delivery")


class _DryRunSender:
    async def send(self, _request: SendRequest) -> SendResult:
        raise RuntimeError("dry-run deflection delivery should not send email")


def _task_metadata(task: ScheduledTask | Any) -> dict[str, Any]:
    metadata = getattr(task, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _dry_run_enabled(task: ScheduledTask | Any) -> bool:
    metadata = _task_metadata(task)
    if "dry_run" in metadata:
        return bool(metadata.get("dry_run"))
    return bool(settings.deflection_delivery.dry_run)


def _configured(value: Any) -> bool:
    return bool(str(value or "").strip())


def _missing_config(*, dry_run: bool) -> list[str]:
    cfg = settings.deflection_delivery
    missing: list[str] = []
    if not _configured(cfg.from_email):
        missing.append("from_email")
    if not _configured(cfg.subject):
        missing.append("subject")
    if not (_configured(cfg.result_base_url) or _configured(cfg.result_url_template)):
        missing.append("result_base_url_or_template")
    if not dry_run and not _configured(cfg.resend_api_key):
        missing.append("resend_api_key")
    return missing


def _delivery_config(*, dry_run: bool) -> DeflectionReportDeliveryConfig:
    cfg = settings.deflection_delivery
    return DeflectionReportDeliveryConfig(
        from_email=str(cfg.from_email or "").strip(),
        result_base_url=str(cfg.result_base_url or "").strip(),
        result_url_template=str(cfg.result_url_template or "").strip(),
        reply_to=str(cfg.reply_to or "").strip() or None,
        subject=str(cfg.subject or "").strip(),
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
    """Drain queued paid deflection report deliveries through the configured ESP."""

    cfg = settings.deflection_delivery
    if not cfg.enabled:
        return {"_skip_synthesis": "Deflection report delivery disabled"}

    dry_run = _dry_run_enabled(task)
    missing = _missing_config(dry_run=dry_run)
    if missing:
        return {
            "_skip_synthesis": "Deflection report delivery config missing",
            "missing": missing,
            "dry_run": dry_run,
        }

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready", "dry_run": dry_run}

    try:
        summary = await send_pending_deflection_report_deliveries(
            pool,
            sender=_sender(dry_run=dry_run),
            config=_delivery_config(dry_run=dry_run),
        )
    except Exception as exc:
        logger.exception("Queued deflection report delivery failed")
        return {
            "_skip_synthesis": "Deflection report delivery failed",
            "error": str(exc)[:500],
            "dry_run": dry_run,
        }

    payload = {
        "scanned": summary.scanned,
        "sent": summary.sent,
        "failed": summary.failed,
        "dry_run": summary.dry_run,
        "dry_run_enabled": dry_run,
    }
    if summary.scanned == 0:
        return {
            "_skip_synthesis": "No pending deflection report deliveries",
            **payload,
        }
    return {"_skip_synthesis": "Deflection report delivery complete", **payload}
