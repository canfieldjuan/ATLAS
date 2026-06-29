"""Send paid FAQ deflection report delivery emails from the webhook queue."""

from __future__ import annotations

import base64
from dataclasses import dataclass, replace
import logging
from typing import Any, Literal, Mapping, Protocol, Sequence
from urllib.parse import quote

from .content_ops_deflection_incidents import emit_deflection_paid_funnel_incident_alert
from extracted_content_pipeline.campaign_ports import (
    IdempotentReplayConflict,
    SendRequest,
    SendResult,
)
from extracted_content_pipeline.deflection_report_access import (
    DeflectionDeltaAccessRecord,
    deflection_delta_read_payload,
    stored_deflection_report_model,
)
from extracted_content_pipeline.faq_deflection_report import (
    deflection_report_email_action_rows,
)
from extracted_content_pipeline.storage._jsonb_helpers import decode_jsonb_field


DEFAULT_DEFLECTION_DELIVERY_SUBJECT = "Your FAQ deflection report is ready"
DEFAULT_DEFLECTION_DELTA_DELIVERY_SUBJECT = "Your support deflection delta is ready"
DELIVERY_CLAIM_STALE_AFTER = "15 minutes"
MAX_DELIVERY_ERROR_CHARS = 500
EMAIL_ACTION_SECTION_LIMIT = 3
DELTA_DELIVERY_ACTION_LIMIT = 3
_DELTA_DELIVERY_SIGNAL_FIELDS = (
    "support_cost_delta",
    "new_count",
    "resolved_count",
    "resurfaced_count",
    "growing_count",
    "shrinking_count",
    "still_unresolved_count",
    "status_changed_count",
    "cost_changed_count",
    "csat_changed_count",
    "low_confidence_identity_count",
)

logger = logging.getLogger("atlas.content_ops_deflection_delivery")


class DeflectionReportDeliverySender(Protocol):
    async def send(self, request: SendRequest) -> SendResult:
        """Send a delivery email."""


@dataclass(frozen=True)
class DeflectionReportDeliveryConfig:
    from_email: str
    result_base_url: str = ""
    result_url_template: str = ""
    reply_to: str | None = None
    subject: str = DEFAULT_DEFLECTION_DELIVERY_SUBJECT
    limit: int = 20
    dry_run: bool = False


@dataclass(frozen=True)
class DeflectionDeltaDeliveryConfig:
    from_email: str
    reply_to: str | None = None
    subject: str = DEFAULT_DEFLECTION_DELTA_DELIVERY_SUBJECT
    limit: int = 20
    dry_run: bool = False


@dataclass(frozen=True)
class DeflectionReportDeliverySummary:
    scanned: int
    sent: int
    failed: int
    dry_run: int


@dataclass(frozen=True)
class DeflectionDeltaDeliveryRunSummary:
    scanned: int
    sent: int
    failed: int
    dry_run: int
    deferred: int = 0


@dataclass(frozen=True)
class DeflectionDeltaDeliverySummary:
    subject: str
    text_body: str
    html_body: str


@dataclass(frozen=True)
class _DeliveryDeltaActionItem:
    question: str
    change_types: tuple[str, ...]
    ticket_count_delta: int
    support_cost_delta: float
    owner_lane: str
    current_status: str
    baseline_status: str


@dataclass(frozen=True)
class _DeliveryEmailActionItem:
    question: str
    ticket_count: int
    estimated_support_cost: float | None
    owner_lane: str
    evidence_tier: str
    product_gap_summary: str
    customer_vocabulary: tuple[str, ...]
    cost_period: str
    cost_confidence: str
    status: str
    recommended_action: str


@dataclass(frozen=True)
class _DeliveryEmailSummary:
    repeat_ticket_count: int
    generated_question_count: int
    estimated_support_cost: float
    drafted_answer_count: int
    no_proven_answer_count: int
    ticket_source_count: int
    priority_fix_items: tuple[_DeliveryEmailActionItem, ...] = ()
    drafted_resolution_items: tuple[_DeliveryEmailActionItem, ...] = ()


class PaidReportPdfRenderError(RuntimeError):
    """Raised when the paid report PDF renderer fails."""


def deflection_delivery_email_surface_observation(
    text_body: str | None,
) -> dict[str, dict[str, int]]:
    """Return sanitized scorecard observations from a rendered delivery email."""

    return {"displayed_rows": _delivery_email_displayed_rows(text_body or "")}


def _delivery_email_displayed_rows(text_body: str) -> dict[str, int]:
    rows = {
        "priority_fix_queue": 0,
        "drafted_resolutions": 0,
    }
    active_section: str | None = None
    for raw_line in text_body.splitlines():
        line = raw_line.strip()
        if line == "Next actions:":
            active_section = "priority_fix_queue"
            continue
        if line == "Ready to publish:":
            active_section = "drafted_resolutions"
            continue
        if not line:
            active_section = None
            continue
        if active_section is not None and line.startswith("- "):
            rows[active_section] += 1
    return rows


async def send_pending_deflection_report_deliveries(
    pool: Any,
    *,
    sender: DeflectionReportDeliverySender,
    config: DeflectionReportDeliveryConfig,
) -> DeflectionReportDeliverySummary:
    """Send pending paid-report delivery emails and update queue status."""

    _validate_config(config)
    rows = await pool.fetch(
        _PENDING_SQL if config.dry_run else _CLAIM_PENDING_SQL,
        int(config.limit),
    )
    sent = failed = dry_run = 0
    for row in rows:
        data = _row_to_dict(row)
        account_id = _required_text(data.get("account_id"), "account_id")
        request_id = _required_text(data.get("request_id"), "request_id")
        report_url = deflection_report_result_url(request_id=request_id, config=config)
        email = _clean(data.get("delivery_email"))
        if not bool(data.get("paid")):
            await _emit_delivery_incident(
                "paid_report_delivery_report_not_paid",
                account_id=account_id,
                request_id=request_id,
                severity="warning",
            )
            await _mark_failed(pool, account_id, request_id, "report_not_paid")
            failed += 1
            continue
        if not email:
            await _emit_delivery_incident(
                "paid_report_delivery_missing_email",
                account_id=account_id,
                request_id=request_id,
                severity="error",
            )
            await _mark_failed(pool, account_id, request_id, "missing_delivery_email")
            failed += 1
            continue
        if config.dry_run:
            dry_run += 1
            continue
        try:
            artifact = _decoded_artifact(data.get("artifact"))
            attachments = _pdf_attachments(
                artifact=artifact,
                request_id=request_id,
            )
            email_summary = _delivery_email_summary(artifact)
            if not await _confirm_delivery_still_sendable(pool, account_id, request_id):
                await _emit_delivery_incident(
                    "paid_report_delivery_no_longer_sendable",
                    account_id=account_id,
                    request_id=request_id,
                    severity="warning",
                )
                logger.warning(
                    "Deflection report delivery skipped before send: "
                    "account=%s request=%s",
                    account_id,
                    request_id,
                )
                failed += 1
                continue
            result = await sender.send(
                _send_request(
                    account_id=account_id,
                    request_id=request_id,
                    email=email,
                    report_url=report_url,
                    attachments=attachments,
                    email_summary=email_summary,
                    config=config,
                )
            )
        except IdempotentReplayConflict:
            # The original send already went out for this idempotency key (the
            # provider rejected the retried payload). Mark delivered, not failed
            # -- no second email is sent. Re-rendered PDF attachments differ
            # byte-for-byte on re-claim, so this is the normal re-claim path.
            await _emit_delivery_incident(
                "paid_report_delivery_idempotent_replay",
                account_id=account_id,
                request_id=request_id,
                severity="info",
            )
            await _mark_delivered(
                pool,
                account_id,
                request_id,
                _provider_message_id("resend", "idempotent-replay"),
            )
            sent += 1
            continue
        except PaidReportPdfRenderError as exc:
            error = _bounded_error(exc)
            if _claimed_previous_delivery_status(data) == "sending":
                await _emit_delivery_incident(
                    "paid_report_delivery_pdf_render_reclaim_deferred",
                    account_id=account_id,
                    request_id=request_id,
                    severity="warning",
                    error=error,
                )
                await _defer_reclaimed_sending(pool, account_id, request_id, error)
                failed += 1
                continue
            await _emit_delivery_incident(
                "paid_report_delivery_send_failed",
                account_id=account_id,
                request_id=request_id,
                severity="error",
                error=error,
            )
            await _mark_failed(pool, account_id, request_id, error)
            failed += 1
            continue
        except Exception as exc:
            error = _bounded_error(exc)
            await _emit_delivery_incident(
                "paid_report_delivery_send_failed",
                account_id=account_id,
                request_id=request_id,
                severity="error",
                error=error,
            )
            await _mark_failed(pool, account_id, request_id, error)
            failed += 1
            continue
        await _mark_delivered(
            pool,
            account_id,
            request_id,
            _provider_message_id(result.provider, result.message_id),
        )
        sent += 1
    return DeflectionReportDeliverySummary(
        scanned=len(rows),
        sent=sent,
        failed=failed,
        dry_run=dry_run,
    )


async def send_pending_deflection_delta_deliveries(
    pool: Any,
    *,
    sender: DeflectionReportDeliverySender,
    config: DeflectionDeltaDeliveryConfig,
    account_id: str | None = None,
    current_request_id: str | None = None,
    entitled_account_ids: Sequence[str] | None = None,
) -> DeflectionDeltaDeliveryRunSummary:
    """Send pending paid-report delta delivery emails and update queue status."""

    _validate_delta_config(config)
    scoped_account_id = _clean(account_id) or None
    scoped_current_request_id = _clean(current_request_id) or None
    entitled_accounts = _optional_text_list(entitled_account_ids)
    rows = await pool.fetch(
        _PENDING_DELTA_SQL if config.dry_run else _CLAIM_PENDING_DELTA_SQL,
        int(config.limit),
        scoped_account_id,
        scoped_current_request_id,
        entitled_accounts,
    )
    sent = failed = deferred = dry_run = 0
    for row in rows:
        data = _row_to_dict(row)
        account_id = _required_text(data.get("account_id"), "account_id")
        current_request_id = _required_text(
            data.get("current_request_id"),
            "current_request_id",
        )
        baseline_request_id = _required_text(
            data.get("baseline_request_id"),
            "baseline_request_id",
        )
        email = _clean(data.get("delivery_email"))
        if config.dry_run:
            dry_run += 1
            continue
        if not email:
            await _fail_delta_delivery(
                pool,
                incident_type="delta_delivery_missing_email",
                account_id=account_id,
                current_request_id=current_request_id,
                baseline_request_id=baseline_request_id,
                error="missing_delivery_email",
                severity="error",
            )
            failed += 1
            continue
        if not bool(data.get("current_paid")) or not bool(data.get("baseline_paid")):
            await _defer_delta_delivery(
                pool,
                incident_type="delta_delivery_source_report_not_paid",
                account_id=account_id,
                current_request_id=current_request_id,
                baseline_request_id=baseline_request_id,
                error="source_report_not_paid",
                severity="warning",
            )
            deferred += 1
            continue
        try:
            record = _delta_record_from_delivery_row(data)
            payload = deflection_delta_read_payload(record)
            if not _delta_payload_has_delivery_content(payload):
                await _fail_delta_delivery(
                    pool,
                    incident_type="delta_delivery_empty_payload",
                    account_id=account_id,
                    current_request_id=current_request_id,
                    baseline_request_id=baseline_request_id,
                    error="empty_delta_payload",
                    severity="warning",
                )
                failed += 1
                continue
            rendered = deflection_delta_delivery_summary(
                payload,
                subject=config.subject,
            )
            if not await _confirm_delta_delivery_still_sendable(
                pool,
                account_id,
                current_request_id,
                baseline_request_id,
            ):
                await _defer_delta_delivery(
                    pool,
                    incident_type="delta_delivery_no_longer_sendable",
                    account_id=account_id,
                    current_request_id=current_request_id,
                    baseline_request_id=baseline_request_id,
                    error="delta_no_longer_sendable",
                    severity="warning",
                )
                deferred += 1
                continue
            result = await sender.send(
                _delta_send_request(
                    account_id=account_id,
                    current_request_id=current_request_id,
                    baseline_request_id=baseline_request_id,
                    email=email,
                    rendered=rendered,
                    config=config,
                )
            )
        except IdempotentReplayConflict:
            await _mark_delta_delivered(
                pool,
                account_id,
                current_request_id,
                baseline_request_id,
                _provider_message_id("resend", "idempotent-replay"),
            )
            sent += 1
            continue
        except Exception as exc:
            logger.exception(
                "Deflection delta delivery failed: account=%s current=%s baseline=%s",
                account_id,
                current_request_id,
                baseline_request_id,
            )
            await _fail_delta_delivery(
                pool,
                incident_type="delta_delivery_send_failed",
                account_id=account_id,
                current_request_id=current_request_id,
                baseline_request_id=baseline_request_id,
                error=_bounded_error(exc),
                severity="error",
            )
            failed += 1
            continue
        await _mark_delta_delivered(
            pool,
            account_id,
            current_request_id,
            baseline_request_id,
            _provider_message_id(result.provider, result.message_id),
        )
        sent += 1
    return DeflectionDeltaDeliveryRunSummary(
        scanned=len(rows),
        sent=sent,
        failed=failed,
        dry_run=dry_run,
        deferred=deferred,
    )


async def pending_deflection_delta_delivery_count(
    pool: Any,
    *,
    account_id: str | None = None,
    current_request_id: str | None = None,
    entitled_account_ids: Sequence[str] | None = None,
) -> int:
    """Return queued delta deliveries that still require a configured sender."""

    scoped_account_id = _clean(account_id) or None
    scoped_current_request_id = _clean(current_request_id) or None
    entitled_accounts = _optional_text_list(entitled_account_ids)
    count = await pool.fetchval(
        f"""
        SELECT COUNT(*)
        FROM content_ops_deflection_delta_deliveries
        WHERE (
                delivery_status = 'pending'
                OR (
                delivery_status = 'sending'
                AND updated_at < NOW() - INTERVAL '{DELIVERY_CLAIM_STALE_AFTER}'
                )
              )
          AND ($1::text IS NULL OR account_id = $1)
          AND ($2::text IS NULL OR current_request_id = $2)
          AND ($3::text[] IS NULL OR account_id = ANY($3::text[]))
        """,
        scoped_account_id,
        scoped_current_request_id,
        entitled_accounts,
    )
    parsed = _strict_int(count)
    return parsed if parsed is not None else 0


def deflection_report_result_url(
    *,
    request_id: str,
    config: DeflectionReportDeliveryConfig,
) -> str:
    encoded = quote(_required_text(request_id, "request_id"), safe="")
    template = _clean(config.result_url_template)
    if template:
        if "{request_id}" not in template:
            raise ValueError("result_url_template must include {request_id}")
        return template.format(request_id=encoded)
    base = _clean(config.result_base_url)
    if not base:
        raise ValueError("result_base_url or result_url_template is required")
    return (
        f"{base.rstrip('/')}/systems/support-ticket-deflection/results/"
        f"{encoded}?checkout=success"
    )


def deflection_delta_delivery_summary(
    payload: Mapping[str, Any],
    *,
    max_items: int = DELTA_DELIVERY_ACTION_LIMIT,
    subject: str = DEFAULT_DEFLECTION_DELTA_DELIVERY_SUBJECT,
) -> DeflectionDeltaDeliverySummary:
    """Render bounded delivery copy from the paid delta read payload."""

    delta = payload.get("delta") if isinstance(payload.get("delta"), Mapping) else {}
    summary = delta.get("summary") if isinstance(delta.get("summary"), Mapping) else {}
    current = delta.get("current") if isinstance(delta.get("current"), Mapping) else {}
    baseline = delta.get("baseline") if isinstance(delta.get("baseline"), Mapping) else {}
    items = _delivery_delta_action_items(delta.get("items"), max_items=max_items)
    resolved_subject = _required_text(subject, "subject")
    return DeflectionDeltaDeliverySummary(
        subject=resolved_subject,
        text_body=_render_delta_summary_text(
            current=current,
            baseline=baseline,
            summary=summary,
            items=items,
        ),
        html_body=_render_delta_summary_html(
            current=current,
            baseline=baseline,
            summary=summary,
            items=items,
        ),
    )


def _delta_payload_has_delivery_content(payload: Mapping[str, Any]) -> bool:
    delta = payload.get("delta") if isinstance(payload.get("delta"), Mapping) else {}
    current = delta.get("current") if isinstance(delta.get("current"), Mapping) else {}
    baseline = delta.get("baseline") if isinstance(delta.get("baseline"), Mapping) else {}
    if not _delta_source_window_label(current) or not _delta_source_window_label(baseline):
        return False
    items = delta.get("items")
    if isinstance(items, Sequence) and not isinstance(items, (str, bytes, bytearray)):
        if any(isinstance(item, Mapping) and _strict_text(item.get("question")) for item in items):
            return True
    summary = delta.get("summary") if isinstance(delta.get("summary"), Mapping) else {}
    for field in _DELTA_DELIVERY_SIGNAL_FIELDS:
        value = _strict_number(summary.get(field))
        if value is not None and value != 0:
            return True
    return False


def _delivery_idempotency_key(account_id: str, request_id: str) -> str:
    """Deterministic Resend idempotency key for a paid report delivery (#1461).

    Derived purely from (account_id, request_id) so a re-claimed 'sending' row
    recomputes the identical key. Resend dedupes identical keys server-side for
    24h, and the claim re-tries after DELIVERY_CLAIM_STALE_AFTER (15 minutes) --
    well inside that window -- so a crash between send and mark cannot produce a
    second email on re-claim.
    """

    return f"deflection-report:{account_id}:{request_id}"


def _delta_delivery_idempotency_key(
    account_id: str,
    current_request_id: str,
    baseline_request_id: str,
) -> str:
    return (
        "deflection-delta:"
        f"{account_id}:{current_request_id}:{baseline_request_id}"
    )


def _send_request(
    *,
    account_id: str,
    request_id: str,
    email: str,
    report_url: str,
    attachments: tuple[dict[str, str], ...] = (),
    email_summary: _DeliveryEmailSummary | None = None,
    config: DeflectionReportDeliveryConfig,
) -> SendRequest:
    has_attachment = bool(attachments)
    return SendRequest(
        campaign_id=f"content_ops_deflection_report:{account_id}:{request_id}",
        idempotency_key=_delivery_idempotency_key(account_id, request_id),
        to_email=email,
        from_email=_required_text(config.from_email, "from_email"),
        reply_to=_clean(config.reply_to) or None,
        subject=_required_text(config.subject, "subject"),
        html_body=_render_html(
            report_url,
            has_attachment=has_attachment,
            email_summary=email_summary,
        ),
        text_body=_render_text(
            report_url,
            has_attachment=has_attachment,
            email_summary=email_summary,
        ),
        attachments=attachments,
        tags=(
            {"name": "source", "value": "content_ops_deflection_report"},
            {"name": "request_id", "value": request_id},
        ),
        metadata={
            "account_id": account_id,
            "request_id": request_id,
            "source": "content_ops_deflection_report",
        },
    )


def _delta_send_request(
    *,
    account_id: str,
    current_request_id: str,
    baseline_request_id: str,
    email: str,
    rendered: DeflectionDeltaDeliverySummary,
    config: DeflectionDeltaDeliveryConfig,
) -> SendRequest:
    return SendRequest(
        campaign_id=(
            "content_ops_deflection_delta:"
            f"{account_id}:{current_request_id}:{baseline_request_id}"
        ),
        idempotency_key=_delta_delivery_idempotency_key(
            account_id,
            current_request_id,
            baseline_request_id,
        ),
        to_email=email,
        from_email=_required_text(config.from_email, "from_email"),
        reply_to=_clean(config.reply_to) or None,
        subject=rendered.subject,
        html_body=rendered.html_body,
        text_body=rendered.text_body,
        attachments=(),
        tags=(
            {"name": "source", "value": "content_ops_deflection_delta"},
            {"name": "current_request_id", "value": current_request_id},
            {"name": "baseline_request_id", "value": baseline_request_id},
        ),
        metadata={
            "account_id": account_id,
            "current_request_id": current_request_id,
            "baseline_request_id": baseline_request_id,
            "source": "content_ops_deflection_delta",
        },
    )


def _render_html(
    report_url: str,
    *,
    has_attachment: bool = False,
    email_summary: _DeliveryEmailSummary | None = None,
) -> str:
    if email_summary is not None:
        return _render_model_summary_html(
            report_url,
            has_attachment=has_attachment,
            summary=email_summary,
        )
    safe_url = _escape(report_url)
    attachment_copy = (
        "<p>The full report PDF is attached for sharing and offline review.</p>"
        if has_attachment
        else ""
    )
    return (
        "<h1>Your FAQ deflection report is ready</h1>"
        "<p>Your paid report is available at the secure results page below.</p>"
        f"{attachment_copy}"
        f'<p><a href="{safe_url}">Open your report</a></p>'
        "<p>The secure results page remains the system of record for this report.</p>"
    )


def _render_text(
    report_url: str,
    *,
    has_attachment: bool = False,
    email_summary: _DeliveryEmailSummary | None = None,
) -> str:
    if email_summary is not None:
        return _render_model_summary_text(
            report_url,
            has_attachment=has_attachment,
            summary=email_summary,
        )
    attachment_copy = (
        "The full report PDF is attached for sharing and offline review.\n\n"
        if has_attachment
        else ""
    )
    return (
        "Your FAQ deflection report is ready\n\n"
        "Your paid report is available at the secure results page below:\n\n"
        f"{attachment_copy}"
        f"{report_url}\n\n"
        "The secure results page remains the system of record for this report.\n"
    )


def _render_model_summary_html(
    report_url: str,
    *,
    has_attachment: bool,
    summary: _DeliveryEmailSummary,
) -> str:
    safe_url = _escape(report_url)
    attachment_copy = (
        "<p>The curated report PDF is attached for sharing and offline review.</p>"
        if has_attachment
        else ""
    )
    return (
        "<h1>Your FAQ deflection report is ready</h1>"
        "<p>Your paid report is available at the secure results page below.</p>"
        "<h2>Key numbers</h2>"
        "<ul>"
        f"<li>{_html_count(summary.repeat_ticket_count)} repeat tickets across "
        f"{_html_count(summary.generated_question_count)} ranked questions.</li>"
        f"<li>{_html_money(summary.estimated_support_cost)} estimated assisted-contact "
        "handling in this upload.</li>"
        f"<li>{_html_count(summary.drafted_answer_count)} publishable answers drafted "
        "from proven resolutions.</li>"
        f"<li>{_html_count(summary.no_proven_answer_count)} questions still need "
        "approved resolution evidence.</li>"
        f"<li>{_html_count(summary.ticket_source_count)} ticket sources represented.</li>"
        "</ul>"
        f"{_render_action_summary_html(summary)}"
        f"{attachment_copy}"
        "<p>The secure results page has the consolidated report, PDF, and complete "
        "evidence export.</p>"
        f'<p><a href="{safe_url}">Open your report</a></p>'
    )


def _render_model_summary_text(
    report_url: str,
    *,
    has_attachment: bool,
    summary: _DeliveryEmailSummary,
) -> str:
    attachment_copy = (
        "The curated report PDF is attached for sharing and offline review.\n\n"
        if has_attachment
        else ""
    )
    return (
        "Your FAQ deflection report is ready\n\n"
        "Key numbers:\n"
        f"- {_email_count(summary.repeat_ticket_count)} repeat tickets across "
        f"{_email_count(summary.generated_question_count)} ranked questions.\n"
        f"- {_email_money(summary.estimated_support_cost)} estimated assisted-contact "
        "handling in this upload.\n"
        f"- {_email_count(summary.drafted_answer_count)} publishable answers drafted "
        "from proven resolutions.\n"
        f"- {_email_count(summary.no_proven_answer_count)} questions still need approved "
        "resolution evidence.\n"
        f"- {_email_count(summary.ticket_source_count)} ticket sources represented.\n\n"
        f"{_render_action_summary_text(summary)}"
        f"{attachment_copy}"
        "The secure results page has the consolidated report, PDF, and complete "
        "evidence export:\n\n"
        f"{report_url}\n"
    )


def _delivery_email_summary(artifact: Any) -> _DeliveryEmailSummary | None:
    model = stored_deflection_report_model(
        artifact if isinstance(artifact, Mapping) else None
    )
    if model is None:
        return None
    sections = _model_sections(model, "email_summary")
    for section in sections:
        if _clean(section.get("id")) != "support_tax":
            continue
        summary = _support_tax_email_summary(section)
        if summary is not None:
            selected_rows = deflection_report_email_action_rows(
                model,
                cap=EMAIL_ACTION_SECTION_LIMIT,
            )
            return replace(
                summary,
                priority_fix_items=_email_action_items(
                    selected_rows.get("priority_fix_queue", ()),
                ),
                drafted_resolution_items=_email_action_items(
                    selected_rows.get("drafted_resolutions", ()),
                ),
            )
    return None


def _render_delta_summary_html(
    *,
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
    summary: Mapping[str, Any],
    items: tuple[_DeliveryDeltaActionItem, ...],
) -> str:
    blocks = [
        "<h1>Your support deflection delta is ready</h1>",
        "<p>This compares the latest paid support-ticket report against the "
        "selected paid baseline.</p>",
        "<h2>Period compared</h2>",
        "<p>",
        _escape(_delta_period_line(current=current, baseline=baseline)),
        "</p>",
        "<h2>What changed</h2>",
        "<ul>",
    ]
    for line in _delta_summary_lines(summary):
        blocks.append(f"<li>{_escape(line)}</li>")
    blocks.append("</ul>")
    if items:
        blocks.append("<h2>Top changes to review</h2><ul>")
        for item in items:
            blocks.append(f"<li>{_escape(_delta_action_item_text(item))}</li>")
        blocks.append("</ul>")
    return "".join(blocks)


def _render_delta_summary_text(
    *,
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
    summary: Mapping[str, Any],
    items: tuple[_DeliveryDeltaActionItem, ...],
) -> str:
    blocks = [
        "Your support deflection delta is ready",
        "",
        "Period compared:",
        f"- {_delta_period_line(current=current, baseline=baseline)}",
        "",
        "What changed:",
    ]
    blocks.extend(f"- {line}" for line in _delta_summary_lines(summary))
    if items:
        blocks.extend(["", "Top changes to review:"])
        blocks.extend(f"- {_delta_action_item_text(item)}" for item in items)
    return "\n".join(blocks) + "\n"


def _delta_period_line(
    *,
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> str:
    current_window = _delta_source_window_label(current) or "current report"
    baseline_window = _delta_source_window_label(baseline) or "baseline report"
    return f"{current_window} vs {baseline_window}"


def _delta_source_window_label(value: Mapping[str, Any]) -> str:
    start = _strict_text(value.get("source_date_start"))
    end = _strict_text(value.get("source_date_end"))
    if start and end:
        return f"{start} to {end}"
    return start or end or ""


def _delta_summary_lines(summary: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        f"{_email_count(_strict_int(summary.get('new_count')) or 0)} new repeats",
        f"{_email_count(_strict_int(summary.get('resolved_count')) or 0)} resolved repeats",
        f"{_email_count(_strict_int(summary.get('growing_count')) or 0)} growing repeats",
        f"{_email_count(_strict_int(summary.get('shrinking_count')) or 0)} shrinking repeats",
        f"{_email_count(_strict_int(summary.get('still_unresolved_count')) or 0)} still unresolved repeats",
        (
            "Support-cost movement: "
            f"{_signed_email_money(_strict_number(summary.get('support_cost_delta')) or 0.0)}"
            " estimated assisted-contact handling"
        ),
    )


def _delivery_delta_action_items(
    value: Any,
    *,
    max_items: int,
) -> tuple[_DeliveryDeltaActionItem, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    items = [
        item
        for row in value
        if isinstance(row, Mapping)
        and (item := _delivery_delta_action_item(row)) is not None
    ]
    items.sort(
        key=lambda item: (
            abs(item.support_cost_delta),
            abs(item.ticket_count_delta),
            item.question,
        ),
        reverse=True,
    )
    try:
        limit = int(max_items)
    except (TypeError, ValueError):
        limit = DELTA_DELIVERY_ACTION_LIMIT
    return tuple(items[: max(0, min(limit, 10))])


def _delivery_delta_action_item(
    row: Mapping[str, Any],
) -> _DeliveryDeltaActionItem | None:
    question = _strict_text(row.get("question"))
    if not question:
        return None
    return _DeliveryDeltaActionItem(
        question=question,
        change_types=tuple(_strict_texts(row.get("change_types"))[:4]),
        ticket_count_delta=_strict_int(row.get("ticket_count_delta")) or 0,
        support_cost_delta=_strict_number(row.get("support_cost_delta")) or 0.0,
        owner_lane=_strict_text(row.get("owner_lane")) or "",
        current_status=_strict_text(row.get("current_status")) or "",
        baseline_status=_strict_text(row.get("baseline_status")) or "",
    )


def _delta_action_item_text(item: _DeliveryDeltaActionItem) -> str:
    parts = [
        item.question,
        ", ".join(_change_type_label(change) for change in item.change_types)
        or "Changed",
        f"{_signed_count(item.ticket_count_delta)} tickets",
        f"{_signed_email_money(item.support_cost_delta)} estimated handling",
    ]
    if item.owner_lane:
        parts.append(f"Owner: {item.owner_lane}")
    if item.baseline_status or item.current_status:
        parts.append(
            "Status: "
            + " -> ".join(
                part
                for part in (item.baseline_status, item.current_status)
                if part
            )
        )
    return " - ".join(parts)


def _change_type_label(value: str) -> str:
    return value.replace("_", " ").title()


def _model_sections(model: Mapping[str, Any], surface: str) -> tuple[dict[str, Any], ...]:
    sections = model.get("sections")
    if not isinstance(sections, list):
        return ()
    filtered = [
        dict(section)
        for section in sections
        if isinstance(section, Mapping)
        and surface in {str(item).strip() for item in section.get("surfaces") or ()}
    ]
    filtered.sort(key=lambda section: _strict_int(section.get("priority")) or 0)
    return tuple(filtered)


def _support_tax_email_summary(
    section: Mapping[str, Any],
) -> _DeliveryEmailSummary | None:
    data = section.get("data")
    if not isinstance(data, Mapping):
        return None
    repeat_ticket_count = _strict_int(data.get("repeat_ticket_count"))
    generated_question_count = _strict_int(data.get("generated_question_count"))
    estimated_support_cost = _strict_number(data.get("estimated_support_cost"))
    drafted_answer_count = _strict_int(data.get("drafted_answer_count"))
    no_proven_answer_count = _strict_int(data.get("no_proven_answer_count"))
    ticket_source_count = _strict_int(data.get("ticket_source_count"))
    if (
        repeat_ticket_count is None
        or generated_question_count is None
        or estimated_support_cost is None
        or drafted_answer_count is None
        or no_proven_answer_count is None
        or ticket_source_count is None
    ):
        return None
    return _DeliveryEmailSummary(
        repeat_ticket_count=repeat_ticket_count,
        generated_question_count=generated_question_count,
        estimated_support_cost=estimated_support_cost,
        drafted_answer_count=drafted_answer_count,
        no_proven_answer_count=no_proven_answer_count,
        ticket_source_count=ticket_source_count,
    )


def _email_action_items(
    rows: tuple[Mapping[str, Any], ...],
) -> tuple[_DeliveryEmailActionItem, ...]:
    return tuple(
        item
        for row in rows
        if (item := _email_action_item(row)) is not None
    )


def _email_action_item(row: Mapping[str, Any]) -> _DeliveryEmailActionItem | None:
    question = _strict_text(row.get("question"))
    ticket_count = _strict_int(row.get("ticket_count"))
    if not question or ticket_count is None:
        return None
    estimated_support_cost = _strict_number(row.get("estimated_support_cost"))
    return _DeliveryEmailActionItem(
        question=question,
        ticket_count=ticket_count,
        estimated_support_cost=estimated_support_cost,
        owner_lane=_strict_text(row.get("owner_lane")) or "",
        evidence_tier=_strict_text(row.get("evidence_tier")) or "",
        product_gap_summary=_strict_text(row.get("product_gap_summary")) or "",
        customer_vocabulary=tuple(_strict_texts(row.get("customer_vocabulary"))[:3]),
        cost_period=_strict_text(row.get("cost_period")) or "",
        cost_confidence=_strict_text(row.get("cost_confidence")) or "",
        status=_strict_text(row.get("status")) or "",
        recommended_action=_strict_text(row.get("recommended_action")) or "",
    )


def _render_action_summary_html(summary: _DeliveryEmailSummary) -> str:
    blocks: list[str] = []
    if summary.priority_fix_items:
        blocks.append("<h2>Next actions</h2><ul>")
        for item in summary.priority_fix_items:
            blocks.append(f"<li>{_html_action_item(item, include_action=True)}</li>")
        blocks.append("</ul>")
    if summary.drafted_resolution_items:
        blocks.append("<h2>Ready to publish</h2><ul>")
        for item in summary.drafted_resolution_items:
            blocks.append(f"<li>{_html_action_item(item, include_action=False)}</li>")
        blocks.append("</ul>")
    return "".join(blocks)


def _render_action_summary_text(summary: _DeliveryEmailSummary) -> str:
    blocks: list[str] = []
    if summary.priority_fix_items:
        blocks.append("Next actions:\n")
        for item in summary.priority_fix_items:
            blocks.append(f"- {_text_action_item(item, include_action=True)}\n")
        blocks.append("\n")
    if summary.drafted_resolution_items:
        blocks.append("Ready to publish:\n")
        for item in summary.drafted_resolution_items:
            blocks.append(f"- {_text_action_item(item, include_action=False)}\n")
        blocks.append("\n")
    return "".join(blocks)


def _html_action_item(item: _DeliveryEmailActionItem, *, include_action: bool) -> str:
    return _escape(_text_action_item(item, include_action=include_action))


def _text_action_item(item: _DeliveryEmailActionItem, *, include_action: bool) -> str:
    parts = [
        item.question,
        f"{_email_count(item.ticket_count)} repeat tickets",
    ]
    if item.estimated_support_cost is not None:
        parts.append(f"{_email_money(item.estimated_support_cost)} estimated handling")
    if item.owner_lane:
        parts.append(f"Owner: {item.owner_lane}")
    if item.evidence_tier:
        parts.append(f"Evidence: {_evidence_tier_label(item.evidence_tier)}")
    if item.product_gap_summary:
        parts.append(item.product_gap_summary)
    if item.customer_vocabulary:
        parts.append("Customer vocabulary: " + ", ".join(item.customer_vocabulary))
    if item.cost_period or item.cost_confidence:
        parts.append(
            "Cost basis: "
            + " / ".join(
                part
                for part in (
                    _cost_period_label(item.cost_period),
                    _cost_confidence_label(item.cost_confidence),
                )
                if part
            )
        )
    if item.status:
        parts.append(item.status)
    if include_action and item.recommended_action:
        parts.append(item.recommended_action)
    return " - ".join(parts)


def _evidence_tier_label(value: str) -> str:
    if not value:
        return "Unknown"
    return {
        "csv_customer_text": "CSV customer text",
        "csv_index_metadata_only": "CSV index metadata only",
        "csv_full_thread_resolution_evidence": "CSV full-thread resolution evidence",
    }.get(value, value.replace("_", " "))


def _cost_period_label(value: str) -> str:
    return {
        "batch_upload": "this upload",
    }.get(value, value.replace("_", " ") if value else "")


def _cost_confidence_label(value: str) -> str:
    return {
        "benchmark_with_resolution_evidence": "benchmark cost with resolution evidence",
        "benchmark_with_customer_text": "benchmark cost with customer text",
        "benchmark_index_metadata_only": "benchmark cost from index metadata",
    }.get(value, value.replace("_", " ") if value else "")


def _strict_texts(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = list(value)
    else:
        values = []
    out: list[str] = []
    for item in values:
        text = _strict_text(item)
        if text and text not in out:
            out.append(text)
    return out


def _strict_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _strict_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _strict_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _decoded_artifact(artifact: Any) -> Any:
    return decode_jsonb_field(artifact, default={})


def _claimed_previous_delivery_status(data: Mapping[str, Any]) -> str:
    return _clean(data.get("previous_delivery_status")) or "pending"


def _pdf_attachments(
    *,
    artifact: Any,
    request_id: str,
) -> tuple[dict[str, str], ...]:
    if not isinstance(artifact, Mapping):
        raise ValueError("paid_report_pdf_missing_artifact")
    try:
        from .deflection_pdf_renderer import render_deflection_full_report_pdf

        pdf_bytes = render_deflection_full_report_pdf(artifact)
    except Exception as exc:
        logger.exception(
            "Deflection report PDF render failed for %s",
            request_id,
        )
        raise PaidReportPdfRenderError(
            f"paid_report_pdf_render_failed: {_bounded_error(exc)}"
        ) from exc
    if not pdf_bytes:
        raise PaidReportPdfRenderError("paid_report_pdf_empty")
    return ({
        "filename": f"{_attachment_slug(request_id)}-support-deflection-report.pdf",
        "content": base64.b64encode(pdf_bytes).decode("ascii"),
    },)


def _attachment_slug(value: str) -> str:
    slug = "".join(
        char if char.isalnum() or char in {"-", "_"} else "-"
        for char in value
    )
    return slug.strip("-_") or "deflection-report"


async def _mark_delivered(
    pool: Any,
    account_id: str,
    request_id: str,
    provider_message_id: str,
) -> None:
    await pool.execute(
        """
        UPDATE content_ops_deflection_report_deliveries
        SET delivery_status = 'delivered',
            delivery_error = NULL,
            provider_message_id = $3,
            delivered_at = NOW(),
            updated_at = NOW()
        WHERE account_id = $1
          AND request_id = $2
          AND delivery_status = 'sending'
        """,
        account_id,
        request_id,
        provider_message_id,
    )


async def _mark_failed(pool: Any, account_id: str, request_id: str, error: str) -> None:
    await pool.execute(
        """
        UPDATE content_ops_deflection_report_deliveries
        SET delivery_status = 'failed',
            delivery_error = $3,
            updated_at = NOW()
        WHERE account_id = $1
          AND request_id = $2
          AND delivery_status = 'sending'
        """,
        account_id,
        request_id,
        _bounded_text(error),
    )


async def _defer_reclaimed_sending(
    pool: Any,
    account_id: str,
    request_id: str,
    error: str,
) -> None:
    await pool.execute(
        """
        UPDATE content_ops_deflection_report_deliveries
        SET delivery_status = 'sending',
            delivery_error = $3,
            updated_at = NOW()
        WHERE account_id = $1
          AND request_id = $2
          AND delivery_status = 'sending'
        """,
        account_id,
        request_id,
        _bounded_text(error),
    )


async def _mark_delta_delivered(
    pool: Any,
    account_id: str,
    current_request_id: str,
    baseline_request_id: str,
    provider_message_id: str,
) -> None:
    await pool.execute(
        """
        UPDATE content_ops_deflection_delta_deliveries
        SET delivery_status = 'delivered',
            delivery_error = NULL,
            provider_message_id = $4,
            delivered_at = NOW(),
            updated_at = NOW()
        WHERE account_id = $1
          AND current_request_id = $2
          AND baseline_request_id = $3
          AND delivery_status = 'sending'
        """,
        account_id,
        current_request_id,
        baseline_request_id,
        provider_message_id,
    )


async def _mark_delta_failed(
    pool: Any,
    account_id: str,
    current_request_id: str,
    baseline_request_id: str,
    error: str,
) -> None:
    await pool.execute(
        """
        UPDATE content_ops_deflection_delta_deliveries
        SET delivery_status = 'failed',
            delivery_error = $4,
            updated_at = NOW()
        WHERE account_id = $1
          AND current_request_id = $2
          AND baseline_request_id = $3
          AND delivery_status IN ('pending', 'sending')
        """,
        account_id,
        current_request_id,
        baseline_request_id,
        _bounded_text(error),
    )


async def _mark_delta_pending(
    pool: Any,
    account_id: str,
    current_request_id: str,
    baseline_request_id: str,
    error: str,
) -> None:
    await pool.execute(
        """
        UPDATE content_ops_deflection_delta_deliveries
        SET delivery_status = 'pending',
            delivery_error = $4,
            updated_at = NOW()
        WHERE account_id = $1
          AND current_request_id = $2
          AND baseline_request_id = $3
          AND delivery_status IN ('pending', 'sending')
        """,
        account_id,
        current_request_id,
        baseline_request_id,
        _bounded_text(error),
    )


async def _fail_delta_delivery(
    pool: Any,
    *,
    incident_type: str,
    account_id: str,
    current_request_id: str,
    baseline_request_id: str,
    error: str,
    severity: Literal["error", "warning", "info"] = "error",
) -> None:
    logger.log(
        logging.ERROR if severity == "error" else logging.WARNING,
        "Deflection delta delivery failed: account=%s current=%s baseline=%s error=%s",
        account_id,
        current_request_id,
        baseline_request_id,
        error,
    )
    await _mark_delta_failed(
        pool,
        account_id,
        current_request_id,
        baseline_request_id,
        error,
    )
    await _emit_delta_delivery_incident(
        incident_type,
        account_id=account_id,
        current_request_id=current_request_id,
        baseline_request_id=baseline_request_id,
        severity=severity,
        error=error,
    )


async def _defer_delta_delivery(
    pool: Any,
    *,
    incident_type: str,
    account_id: str,
    current_request_id: str,
    baseline_request_id: str,
    error: str,
    severity: Literal["error", "warning", "info"] = "warning",
) -> None:
    logger.log(
        logging.ERROR if severity == "error" else logging.WARNING,
        "Deflection delta delivery deferred: account=%s current=%s baseline=%s error=%s",
        account_id,
        current_request_id,
        baseline_request_id,
        error,
    )
    await _mark_delta_pending(
        pool,
        account_id,
        current_request_id,
        baseline_request_id,
        error,
    )
    await _emit_delta_delivery_incident(
        incident_type,
        account_id=account_id,
        current_request_id=current_request_id,
        baseline_request_id=baseline_request_id,
        severity=severity,
        error=error,
    )


async def _confirm_delivery_still_sendable(
    pool: Any,
    account_id: str,
    request_id: str,
) -> bool:
    row = await pool.fetchrow(
        """
        UPDATE content_ops_deflection_report_deliveries d
        SET updated_at = NOW()
        FROM content_ops_deflection_reports r
        WHERE d.account_id = $1
          AND d.request_id = $2
          AND r.account_id = d.account_id
          AND r.request_id = d.request_id
          AND d.delivery_status = 'sending'
          AND r.paid = true
        RETURNING d.request_id
        """,
        account_id,
        request_id,
    )
    return row is not None


async def _confirm_delta_delivery_still_sendable(
    pool: Any,
    account_id: str,
    current_request_id: str,
    baseline_request_id: str,
) -> bool:
    row = await pool.fetchrow(
        """
        UPDATE content_ops_deflection_delta_deliveries d
        SET updated_at = NOW()
        FROM content_ops_deflection_reports current_report,
             content_ops_deflection_reports baseline_report
        WHERE d.account_id = $1
          AND d.current_request_id = $2
          AND d.baseline_request_id = $3
          AND current_report.account_id = d.account_id
          AND current_report.request_id = d.current_request_id
          AND current_report.paid = true
          AND baseline_report.account_id = d.account_id
          AND baseline_report.request_id = d.baseline_request_id
          AND baseline_report.paid = true
          AND d.delivery_status = 'sending'
        RETURNING d.current_request_id
        """,
        account_id,
        current_request_id,
        baseline_request_id,
    )
    return row is not None


async def _emit_delivery_incident(
    incident_type: str,
    *,
    account_id: str,
    request_id: str,
    severity: Literal["error", "warning", "info"] = "error",
    **fields: Any,
) -> None:
    await emit_deflection_paid_funnel_incident_alert(
        logger,
        incident_type=incident_type,
        severity=severity,
        account_id=account_id,
        request_id=request_id,
        **fields,
    )


async def _emit_delta_delivery_incident(
    incident_type: str,
    *,
    account_id: str,
    current_request_id: str,
    baseline_request_id: str,
    severity: Literal["error", "warning", "info"] = "error",
    **fields: Any,
) -> None:
    await emit_deflection_paid_funnel_incident_alert(
        logger,
        incident_type=incident_type,
        severity=severity,
        account_id=account_id,
        request_id=current_request_id,
        current_request_id=current_request_id,
        baseline_request_id=baseline_request_id,
        **fields,
    )


def _validate_config(config: DeflectionReportDeliveryConfig) -> None:
    _required_text(config.from_email, "from_email")
    _required_text(config.subject, "subject")
    if int(config.limit) <= 0:
        raise ValueError("limit must be greater than 0")
    deflection_report_result_url(request_id="config-check", config=config)


def _validate_delta_config(config: DeflectionDeltaDeliveryConfig) -> None:
    _required_text(config.from_email, "from_email")
    _required_text(config.subject, "subject")
    if int(config.limit) <= 0:
        raise ValueError("limit must be greater than 0")


def _provider_message_id(provider: str, message_id: str) -> str:
    return f"{_required_text(provider, 'provider')}:{_required_text(message_id, 'message_id')}"


def _bounded_error(exc: Exception) -> str:
    return _bounded_text(f"{type(exc).__name__}: {exc}")


def _bounded_text(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) <= MAX_DELIVERY_ERROR_CHARS:
        return text
    return text[: MAX_DELIVERY_ERROR_CHARS - 3] + "..."


def _row_to_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    return dict(row)


def _delta_record_from_delivery_row(data: Mapping[str, Any]) -> DeflectionDeltaAccessRecord:
    delta = decode_jsonb_field(data.get("delta"), default={})
    return DeflectionDeltaAccessRecord(
        account_id=_required_text(data.get("account_id"), "account_id"),
        current_request_id=_required_text(
            data.get("current_request_id"),
            "current_request_id",
        ),
        baseline_request_id=_required_text(
            data.get("baseline_request_id"),
            "baseline_request_id",
        ),
        delta=dict(delta) if isinstance(delta, Mapping) else {},
        created_at=data.get("delta_created_at"),
        updated_at=data.get("delta_updated_at"),
    )


def _required_text(value: Any, label: str) -> str:
    cleaned = _clean(value)
    if not cleaned:
        raise ValueError(f"{label} is required")
    return cleaned


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _optional_text_list(values: Sequence[str] | None) -> list[str] | None:
    if values is None:
        return None
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = _clean(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _email_count(value: int) -> str:
    return f"{int(value):,}"


def _html_count(value: int) -> str:
    return _escape(_email_count(value))


def _email_money(value: float) -> str:
    return f"${float(value):,.0f}"


def _signed_email_money(value: float) -> str:
    amount = float(value)
    if amount < 0:
        return f"-${abs(amount):,.0f}"
    prefix = "+" if amount > 0 else ""
    return f"{prefix}{_email_money(amount)}"


def _signed_count(value: int) -> str:
    prefix = "+" if int(value) > 0 else ""
    return f"{prefix}{_email_count(value)}"


def _html_money(value: float) -> str:
    return _escape(_email_money(value))


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


_PENDING_SQL = """
SELECT
    d.account_id,
    d.request_id,
    d.delivery_status AS previous_delivery_status,
    r.delivery_email,
    r.paid,
    r.artifact
FROM content_ops_deflection_report_deliveries d
JOIN content_ops_deflection_reports r
  ON r.account_id = d.account_id
 AND r.request_id = d.request_id
WHERE d.delivery_status = 'pending'
ORDER BY d.created_at
LIMIT $1
"""

_CLAIM_PENDING_SQL = f"""
WITH claimed AS (
    SELECT
        d.account_id,
        d.request_id,
        d.created_at,
        d.delivery_status AS previous_delivery_status
    FROM content_ops_deflection_report_deliveries d
    WHERE d.delivery_status = 'pending'
       OR (
            d.delivery_status = 'sending'
            AND d.updated_at < NOW() - INTERVAL '{DELIVERY_CLAIM_STALE_AFTER}'
       )
    ORDER BY d.created_at
    FOR UPDATE SKIP LOCKED
    LIMIT $1
)
UPDATE content_ops_deflection_report_deliveries d
SET delivery_status = 'sending',
    delivery_error = NULL,
    updated_at = NOW()
FROM claimed c
JOIN content_ops_deflection_reports r
  ON r.account_id = c.account_id
 AND r.request_id = c.request_id
WHERE d.account_id = c.account_id
  AND d.request_id = c.request_id
RETURNING
    d.account_id,
    d.request_id,
    c.previous_delivery_status,
    r.delivery_email,
    r.paid,
    r.artifact
"""

_PENDING_DELTA_SQL = """
SELECT
    d.account_id,
    d.current_request_id,
    d.baseline_request_id,
    current_report.delivery_email,
    current_report.paid AS current_paid,
    baseline_report.paid AS baseline_paid,
    deltas.delta,
    deltas.created_at AS delta_created_at,
    deltas.updated_at AS delta_updated_at
FROM content_ops_deflection_delta_deliveries d
JOIN content_ops_deflection_deltas deltas
  ON deltas.account_id = d.account_id
 AND deltas.current_request_id = d.current_request_id
 AND deltas.baseline_request_id = d.baseline_request_id
JOIN content_ops_deflection_reports current_report
  ON current_report.account_id = d.account_id
 AND current_report.request_id = d.current_request_id
JOIN content_ops_deflection_reports baseline_report
  ON baseline_report.account_id = d.account_id
 AND baseline_report.request_id = d.baseline_request_id
WHERE d.delivery_status = 'pending'
  AND ($2::text IS NULL OR d.account_id = $2)
  AND ($3::text IS NULL OR d.current_request_id = $3)
  AND ($4::text[] IS NULL OR d.account_id = ANY($4::text[]))
ORDER BY d.created_at
LIMIT $1
"""

_CLAIM_PENDING_DELTA_SQL = f"""
WITH claimed AS (
    SELECT
        d.account_id,
        d.current_request_id,
        d.baseline_request_id,
        d.created_at
    FROM content_ops_deflection_delta_deliveries d
    WHERE (
            d.delivery_status = 'pending'
            OR (
            d.delivery_status = 'sending'
            AND d.updated_at < NOW() - INTERVAL '{DELIVERY_CLAIM_STALE_AFTER}'
            )
          )
      AND ($2::text IS NULL OR d.account_id = $2)
      AND ($3::text IS NULL OR d.current_request_id = $3)
      AND ($4::text[] IS NULL OR d.account_id = ANY($4::text[]))
    ORDER BY d.created_at
    FOR UPDATE SKIP LOCKED
    LIMIT $1
)
UPDATE content_ops_deflection_delta_deliveries d
SET delivery_status = 'sending',
    delivery_error = NULL,
    updated_at = NOW()
FROM claimed c
JOIN content_ops_deflection_deltas deltas
  ON deltas.account_id = c.account_id
 AND deltas.current_request_id = c.current_request_id
 AND deltas.baseline_request_id = c.baseline_request_id
JOIN content_ops_deflection_reports current_report
  ON current_report.account_id = c.account_id
 AND current_report.request_id = c.current_request_id
JOIN content_ops_deflection_reports baseline_report
  ON baseline_report.account_id = c.account_id
 AND baseline_report.request_id = c.baseline_request_id
WHERE d.account_id = c.account_id
  AND d.current_request_id = c.current_request_id
  AND d.baseline_request_id = c.baseline_request_id
RETURNING
    d.account_id,
    d.current_request_id,
    d.baseline_request_id,
    current_report.delivery_email,
    current_report.paid AS current_paid,
    baseline_report.paid AS baseline_paid,
    deltas.delta,
    deltas.created_at AS delta_created_at,
    deltas.updated_at AS delta_updated_at
"""
