"""Daily maintenance task for B2B witness-quality field propagation.

Runs after the daily synthesis and report cron has produced new
``b2b_reasoning_synthesis`` / ``b2b_intelligence`` rows. Two responsibilities:

1. Drain any new historical gap by running the backfill in apply mode --
   keeps the Phase 7/8 propagation invariant
   (``fillable_missing_fields == 0`` across all surfaces) closed across
   newly-created artifacts.

2. Re-audit the post-backfill state and alert when the invariant breaks.
   A non-zero ``fillable_missing_fields`` after the backfill ran indicates
   a write-path leak the backfill cannot keep up with -- typically a new
   report generator that doesn't decorate witnesses with quality fields.

Notifications fire only on the failure path, via ntfy. The success path is
silent so the operator inbox does not fill up with no-op pings every day.
"""

from __future__ import annotations

import logging
from typing import Any

from atlas_brain.config import settings
from atlas_brain.services.witness_quality_maintenance import (
    run_witness_quality_maintenance,
)
from atlas_brain.storage.database import get_db_pool

logger = logging.getLogger("atlas.autonomous.tasks.b2b_witness_quality_maintenance")


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1", "on")
    return default


async def _send_alert(*, fillable_total: int, leaking: list[dict[str, Any]]) -> None:
    """Send an ntfy alert when the propagation invariant breaks."""
    if not getattr(settings.alerts, "ntfy_enabled", False):
        logger.warning(
            "witness_quality_maintenance: invariant breached but ntfy disabled "
            "(fillable=%d surfaces=%d)",
            fillable_total,
            len(leaking),
        )
        return
    try:
        from atlas_brain.tools.notify import notify_tool

        worst = sorted(
            leaking,
            key=lambda item: int(item.get("fillable_missing_fields", 0) or 0),
            reverse=True,
        )[:5]
        bullets = "\n".join(
            f"- {item['surface']}: {item['fillable_missing_fields']} "
            f"(of {item['witness_objects']} witnesses)"
            for item in worst
        )
        message = (
            "Witness quality propagation regressed: "
            f"{fillable_total} fillable_missing fields after backfill.\n\n"
            f"Top offenders:\n{bullets}\n\n"
            "A write path is dropping fields the source has. Investigate "
            "scripts/audit_witness_field_propagation.py and the most "
            "recently-changed report generator."
        )
        await notify_tool._send_notification(
            message=message,
            title="Atlas: Witness Quality Invariant Breached",
            priority="4",
            tags="warning,b2b,witness",
        )
    except Exception:
        logger.error(
            "witness_quality_maintenance: failed to send ntfy alert",
            exc_info=True,
        )


async def run(task: Any) -> dict[str, Any]:
    """Builtin task entry point.

    Reads optional metadata overrides from the scheduled task:
      - ``days`` (int, default 30): lookback window for backfill + audit.
      - ``apply`` (bool, default True): when False, runs as dry-run only.
      - ``overwrite`` (bool, default False): replace existing quality
        fields instead of only filling missing.
      - ``audit_row_limit`` (int, default 500): row cap for the
        propagation audit. Backfill always scans the full window.

    Returns the same dict the maintenance service returns, plus an
    ``_skip_synthesis`` hint so the runner does not try to LLM-summarise
    a structured operational result.
    """
    metadata = getattr(task, "metadata", None) or {}
    days = _coerce_int(metadata.get("days"), 30)
    apply = _coerce_bool(metadata.get("apply"), True)
    overwrite = _coerce_bool(metadata.get("overwrite"), False)
    audit_row_limit = _coerce_int(metadata.get("audit_row_limit"), 500)

    pool = get_db_pool()
    if pool is None:
        logger.warning("witness_quality_maintenance: no DB pool available")
        return {"_skip_synthesis": "DB pool unavailable", "alert_triggered": False}

    result = await run_witness_quality_maintenance(
        pool,
        days=days,
        apply=apply,
        overwrite=overwrite,
        audit_row_limit=audit_row_limit,
    )

    # Operational telemetry: log the headline numbers regardless of alert.
    backfill_tables = result.get("backfill", {}).get("tables", {})
    fields_written = sum(
        int(stats.get("fields_written", 0) or 0)
        for stats in backfill_tables.values()
    )
    changed_rows = sum(
        int(stats.get("changed_rows", 0) or 0)
        for stats in backfill_tables.values()
    )
    logger.info(
        "witness_quality_maintenance: days=%d apply=%s changed_rows=%d "
        "fields_written=%d fillable_missing=%d alert=%s",
        days,
        apply,
        changed_rows,
        fields_written,
        int(result.get("fillable_missing_fields", 0) or 0),
        bool(result.get("alert_triggered", False)),
    )

    if result.get("alert_triggered"):
        await _send_alert(
            fillable_total=int(result.get("fillable_missing_fields", 0) or 0),
            leaking=result.get("leaking_surfaces") or [],
        )

    # Attach the skip hint so the autonomous runner does not try to feed
    # this structured maintenance result into an LLM synthesis skill.
    result["_skip_synthesis"] = (
        f"witness_quality_maintenance: {fields_written} fields written, "
        f"{int(result.get('fillable_missing_fields', 0) or 0)} fillable_missing"
    )
    return result
