"""Send paid FAQ deflection report delivery emails from the webhook queue."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol
from urllib.parse import quote

from extracted_content_pipeline.campaign_ports import SendRequest, SendResult


DEFAULT_DEFLECTION_DELIVERY_SUBJECT = "Your FAQ deflection report is ready"
MAX_DELIVERY_ERROR_CHARS = 500


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
class DeflectionReportDeliverySummary:
    scanned: int
    sent: int
    failed: int
    dry_run: int


async def send_pending_deflection_report_deliveries(
    pool: Any,
    *,
    sender: DeflectionReportDeliverySender,
    config: DeflectionReportDeliveryConfig,
) -> DeflectionReportDeliverySummary:
    """Send pending paid-report delivery emails and update queue status."""

    _validate_config(config)
    rows = await pool.fetch(_PENDING_SQL, int(config.limit))
    sent = failed = dry_run = 0
    for row in rows:
        data = _row_to_dict(row)
        account_id = _required_text(data.get("account_id"), "account_id")
        request_id = _required_text(data.get("request_id"), "request_id")
        report_url = deflection_report_result_url(request_id=request_id, config=config)
        email = _clean(data.get("delivery_email"))
        if not bool(data.get("paid")):
            await _mark_failed(pool, account_id, request_id, "report_not_paid")
            failed += 1
            continue
        if not email:
            await _mark_failed(pool, account_id, request_id, "missing_delivery_email")
            failed += 1
            continue
        if config.dry_run:
            dry_run += 1
            continue
        try:
            result = await sender.send(
                _send_request(
                    account_id=account_id,
                    request_id=request_id,
                    email=email,
                    report_url=report_url,
                    config=config,
                )
            )
        except Exception as exc:
            await _mark_failed(pool, account_id, request_id, _bounded_error(exc))
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


def _send_request(
    *,
    account_id: str,
    request_id: str,
    email: str,
    report_url: str,
    config: DeflectionReportDeliveryConfig,
) -> SendRequest:
    return SendRequest(
        campaign_id=f"content_ops_deflection_report:{account_id}:{request_id}",
        to_email=email,
        from_email=_required_text(config.from_email, "from_email"),
        reply_to=_clean(config.reply_to) or None,
        subject=_required_text(config.subject, "subject"),
        html_body=_render_html(report_url),
        text_body=_render_text(report_url),
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


def _render_html(report_url: str) -> str:
    safe_url = _escape(report_url)
    return (
        "<h1>Your FAQ deflection report is ready</h1>"
        "<p>Your paid report is available at the secure results page below.</p>"
        f'<p><a href="{safe_url}">Open your report</a></p>'
        "<p>This email only carries the access link; the report stays behind "
        "the paid results page.</p>"
    )


def _render_text(report_url: str) -> str:
    return (
        "Your FAQ deflection report is ready\n\n"
        "Your paid report is available at the secure results page below:\n\n"
        f"{report_url}\n\n"
        "This email only carries the access link; the report stays behind "
        "the paid results page.\n"
    )


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
          AND delivery_status = 'pending'
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
          AND delivery_status = 'pending'
        """,
        account_id,
        request_id,
        _bounded_text(error),
    )


def _validate_config(config: DeflectionReportDeliveryConfig) -> None:
    _required_text(config.from_email, "from_email")
    _required_text(config.subject, "subject")
    if int(config.limit) <= 0:
        raise ValueError("limit must be greater than 0")
    deflection_report_result_url(request_id="config-check", config=config)


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


def _required_text(value: Any, label: str) -> str:
    cleaned = _clean(value)
    if not cleaned:
        raise ValueError(f"{label} is required")
    return cleaned


def _clean(value: Any) -> str:
    return str(value or "").strip()


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
    r.delivery_email,
    r.paid
FROM content_ops_deflection_report_deliveries d
JOIN content_ops_deflection_reports r
  ON r.account_id = d.account_id
 AND r.request_id = d.request_id
WHERE d.delivery_status = 'pending'
ORDER BY d.created_at
LIMIT $1
"""
