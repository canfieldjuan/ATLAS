"""Scheduled generation for paid Content Ops deflection report deltas."""

from __future__ import annotations

import logging
from typing import Any

from extracted_content_pipeline.deflection_report_access import (
    PostgresDeflectionReportArtifactStore,
    compute_and_save_recent_deflection_deltas,
)

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.tasks.content_ops_deflection_delta_automation")


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
    except Exception as exc:
        logger.exception("Deflection delta automation failed")
        return {
            "_skip_synthesis": "Deflection delta automation failed",
            "error": str(exc)[:500],
        }

    payload = {
        "accounts_scanned": summary.accounts_scanned,
        "reports_scanned": summary.reports_scanned,
        "deltas_saved": summary.deltas_saved,
        "skipped_no_delta": summary.skipped_no_delta,
        "failed": summary.failed,
    }
    if summary.reports_scanned == 0:
        return {
            "_skip_synthesis": "No paid deflection reports found for delta automation",
            **payload,
        }
    return {"_skip_synthesis": "Deflection delta automation complete", **payload}
