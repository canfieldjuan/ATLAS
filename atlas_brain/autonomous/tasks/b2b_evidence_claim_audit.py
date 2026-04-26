"""Daily audit task for the EvidenceClaim contract shadow capture.

Phase 9 step 6. Runs after the nightly synthesis cycle has populated
b2b_evidence_claims and reads back a structured summary for the operator
inbox: total claims, valid/invalid/cannot_validate breakdown,
rejection-reason distribution, per-vendor / per-source / per-pain_category
slices, and a small invalid-claim sample for hand-audit triage.

Output goes to:

  - the runner log (always, headline line at INFO level)
  - the autonomous-task return dict (always, the full summary so
    digest skills or other consumers can pretty-print it)
  - ntfy (only on alert conditions, see _is_alertable)

The task is silent on the success path so the operator inbox does not
fill with no-op pings. It alerts when the shadow capture appears to be
broken or trending the wrong way:

  - zero rows captured for the day (synthesis ran but did not write)
  - cannot_validate share above 60% (pipeline producing tagless rows)
  - any single rejection_reason exceeds 25% of invalid total (a gate
    is rejecting at an unexpected rate, likely a regression)

These thresholds are deliberately conservative for v1; the canary
hand-audit at rollout step 7 is where we tune them.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from atlas_brain.config import settings
from atlas_brain.services.reasoning_delivery_audit import summarize_claim_validation
from atlas_brain.storage.database import get_db_pool


logger = logging.getLogger("atlas.autonomous.tasks.b2b_evidence_claim_audit")


def _coerce_as_of(metadata: dict[str, Any] | None) -> date:
    raw = (metadata or {}).get("as_of_date")
    if raw:
        if isinstance(raw, date):
            return raw
        try:
            return date.fromisoformat(str(raw))
        except (TypeError, ValueError):
            logger.warning("Invalid as_of_date metadata %r; falling back to today", raw)
    return date.today()


def _is_alertable(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return (alert, reasons). Triggers when the shadow capture looks
    broken: no rows captured at all, very high cannot_validate share, or
    a single rejection_reason dominating the invalid bucket."""
    reasons: list[str] = []
    total = int(summary.get("scope", {}).get("total_rows", 0) or 0)
    if total <= 0:
        reasons.append("zero claim rows captured for as_of_date")
        return True, reasons

    totals = summary.get("totals", {})
    cannot = int(totals.get("cannot_validate", 0) or 0)
    if total > 0 and cannot / total > 0.60:
        reasons.append(
            f"cannot_validate share {cannot}/{total} exceeds 60%"
        )

    invalid_total = int(totals.get("invalid", 0) or 0)
    if invalid_total > 0:
        reasons_by_type = summary.get("rejection_reasons_by_claim_type", {})
        flat: dict[str, int] = {}
        for entries in reasons_by_type.values():
            for entry in entries:
                flat[entry["rejection_reason"]] = (
                    flat.get(entry["rejection_reason"], 0)
                    + int(entry["count"] or 0)
                )
        for reason, count in flat.items():
            if count / invalid_total > 0.25:
                reasons.append(
                    f"rejection_reason '{reason}' = {count}/{invalid_total} "
                    f"({count / invalid_total:.0%}) of invalid"
                )
                break  # One example is enough for the alert payload.

    return bool(reasons), reasons


async def _maybe_alert(summary: dict[str, Any], alert_reasons: list[str]) -> bool:
    """Send an ntfy if the alert conditions fired and ntfy is enabled."""
    if not alert_reasons:
        return False
    if not getattr(settings.alerts, "ntfy_enabled", False):
        logger.warning(
            "evidence_claim_audit: alert reasons present but ntfy disabled: %s",
            "; ".join(alert_reasons),
        )
        return False
    try:
        from atlas_brain.tools.notify import notify_tool

        scope = summary.get("scope", {})
        totals = summary.get("totals", {})
        message = (
            f"EvidenceClaim shadow audit flagged for "
            f"{summary.get('as_of_date')}:\n\n"
            + "\n".join(f"- {reason}" for reason in alert_reasons)
            + f"\n\ntotals: valid={totals.get('valid', 0)} "
            f"invalid={totals.get('invalid', 0)} "
            f"cannot_validate={totals.get('cannot_validate', 0)} "
            f"(rows={scope.get('total_rows', 0)} "
            f"vendors={scope.get('distinct_vendors', 0)})"
        )
        await notify_tool._send_notification(
            title=f"Evidence claim audit: {summary.get('as_of_date')}",
            message=message,
            priority="default",
            tags=["b2b", "evidence_claim", "audit"],
        )
        return True
    except Exception:
        logger.warning("evidence_claim_audit: ntfy send failed", exc_info=True)
        return False


async def run(task: Any) -> dict[str, Any]:
    """Builtin task entry point.

    Reads optional metadata overrides from the scheduled task:
      - ``as_of_date`` (ISO date string, default today): which day's
        claim rows to summarize. Use yesterday's date when running
        immediately after midnight to capture the prior synthesis cycle.
      - ``invalid_examples_per_reason`` (int, default 3): cap on
        sampled invalid rows per (claim_type, rejection_reason) pair.
      - ``rejection_reasons_per_claim_type`` (int, default 10): top-N
        reasons surfaced per claim_type.

    Returns the summary dict plus _skip_synthesis so the runner does not
    LLM-summarise structured operational output.
    """
    metadata = getattr(task, "metadata", None) or {}
    as_of = _coerce_as_of(metadata)
    invalid_examples_per_reason = int(metadata.get("invalid_examples_per_reason", 3))
    rejection_reasons_per_claim_type = int(metadata.get("rejection_reasons_per_claim_type", 10))

    pool = get_db_pool()
    if pool is None:
        logger.warning("evidence_claim_audit: no DB pool available")
        return {
            "_skip_synthesis": "DB pool unavailable",
            "alert_triggered": False,
        }

    summary = await summarize_claim_validation(
        pool,
        as_of_date=as_of,
        invalid_examples_per_reason=invalid_examples_per_reason,
        rejection_reasons_per_claim_type=rejection_reasons_per_claim_type,
    )

    scope = summary.get("scope", {})
    totals = summary.get("totals", {})
    logger.info(
        "evidence_claim_audit: as_of=%s rows=%d vendors=%d valid=%d invalid=%d cannot_validate=%d",
        summary["as_of_date"],
        scope.get("total_rows", 0),
        scope.get("distinct_vendors", 0),
        totals.get("valid", 0),
        totals.get("invalid", 0),
        totals.get("cannot_validate", 0),
    )

    alert, alert_reasons = _is_alertable(summary)
    notified = False
    if alert:
        notified = await _maybe_alert(summary, alert_reasons)

    summary["alert_triggered"] = alert
    summary["alert_reasons"] = alert_reasons
    summary["alert_notified"] = notified
    summary["_skip_synthesis"] = (
        f"evidence_claim_audit: {totals.get('valid', 0)} valid / "
        f"{totals.get('invalid', 0)} invalid / "
        f"{totals.get('cannot_validate', 0)} cannot_validate "
        f"(rows={scope.get('total_rows', 0)})"
    )
    return summary
