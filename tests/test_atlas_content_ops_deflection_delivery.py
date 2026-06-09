from __future__ import annotations

from typing import Any

import pytest

from atlas_brain.content_ops_deflection_delivery import (
    DELIVERY_CLAIM_STALE_AFTER,
    DeflectionReportDeliveryConfig,
    deflection_report_result_url,
    send_pending_deflection_report_deliveries,
)
from extracted_content_pipeline.campaign_ports import SendRequest, SendResult


class _Pool:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetch_calls.append((query, args))
        assert "content_ops_deflection_report_deliveries" in query
        assert "content_ops_deflection_reports" in query
        return self.rows[: int(args[0])]

    async def execute(self, query: str, *args: Any) -> str:
        self.execute_calls.append((query, args))
        return "UPDATE 1"


class _Sender:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.requests: list[SendRequest] = []

    async def send(self, request: SendRequest) -> SendResult:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return SendResult(provider="resend", message_id="email-123", raw={"id": "email-123"})


def _row(**overrides: Any) -> dict[str, Any]:
    row = {
        "account_id": "acct-123",
        "request_id": "content-ops-abc123",
        "delivery_email": "buyer@example.com",
        "paid": True,
    }
    row.update(overrides)
    return row


def _config(**overrides: Any) -> DeflectionReportDeliveryConfig:
    values = {
        "from_email": "Atlas Content Ops <reports@example.com>",
        "result_base_url": "https://portfolio.example.com",
        "reply_to": "support@example.com",
        "limit": 20,
    }
    values.update(overrides)
    return DeflectionReportDeliveryConfig(**values)


@pytest.mark.asyncio
async def test_delivery_worker_sends_pending_paid_report_link() -> None:
    pool = _Pool([_row()])
    sender = _Sender()

    summary = await send_pending_deflection_report_deliveries(
        pool,
        sender=sender,
        config=_config(),
    )

    assert summary.scanned == 1
    assert summary.sent == 1
    assert summary.failed == 0
    claim_query, claim_args = pool.fetch_calls[0]
    assert "FOR UPDATE SKIP LOCKED" in claim_query
    assert "SET delivery_status = 'sending'" in claim_query
    assert claim_args == (20,)
    assert len(sender.requests) == 1
    request = sender.requests[0]
    assert request.to_email == "buyer@example.com"
    assert request.from_email == "Atlas Content Ops <reports@example.com>"
    assert request.reply_to == "support@example.com"
    assert request.subject == "Your FAQ deflection report is ready"
    expected_url = (
        "https://portfolio.example.com/systems/support-ticket-deflection/results/"
        "content-ops-abc123?checkout=success"
    )
    assert expected_url in request.html_body
    assert expected_url in (request.text_body or "")
    assert "markdown" not in request.html_body.lower()
    assert "resolution_evidence" not in request.html_body

    update_query, update_args = pool.execute_calls[0]
    assert "delivery_status = 'delivered'" in update_query
    assert "delivery_status = 'sending'" in update_query
    assert update_args == ("acct-123", "content-ops-abc123", "resend:email-123")
    assert "buyer@example.com" not in str(update_args)


@pytest.mark.asyncio
async def test_delivery_worker_dry_run_does_not_send_or_update() -> None:
    pool = _Pool([_row()])
    sender = _Sender()

    summary = await send_pending_deflection_report_deliveries(
        pool,
        sender=sender,
        config=_config(dry_run=True),
    )

    assert summary.scanned == 1
    assert summary.sent == 0
    assert summary.dry_run == 1
    pending_query, pending_args = pool.fetch_calls[0]
    assert "FOR UPDATE SKIP LOCKED" not in pending_query
    assert "WHERE d.delivery_status = 'pending'" in pending_query
    assert pending_args == (20,)
    assert sender.requests == []
    assert pool.execute_calls == []


@pytest.mark.asyncio
async def test_delivery_worker_claim_query_retries_stale_sending_rows() -> None:
    pool = _Pool([_row()])
    sender = _Sender()

    await send_pending_deflection_report_deliveries(
        pool,
        sender=sender,
        config=_config(),
    )

    claim_query, _claim_args = pool.fetch_calls[0]
    assert "d.delivery_status = 'pending'" in claim_query
    assert "d.delivery_status = 'sending'" in claim_query
    assert f"INTERVAL '{DELIVERY_CLAIM_STALE_AFTER}'" in claim_query


@pytest.mark.asyncio
async def test_delivery_worker_marks_missing_email_failed() -> None:
    pool = _Pool([_row(delivery_email=" ")])
    sender = _Sender()

    summary = await send_pending_deflection_report_deliveries(
        pool,
        sender=sender,
        config=_config(),
    )

    assert summary.failed == 1
    assert sender.requests == []
    query, args = pool.execute_calls[0]
    assert "delivery_status = 'failed'" in query
    assert "delivery_status = 'sending'" in query
    assert args == ("acct-123", "content-ops-abc123", "missing_delivery_email")


@pytest.mark.asyncio
async def test_delivery_worker_marks_unpaid_report_failed() -> None:
    pool = _Pool([_row(paid=False)])
    sender = _Sender()

    summary = await send_pending_deflection_report_deliveries(
        pool,
        sender=sender,
        config=_config(),
    )

    assert summary.failed == 1
    assert sender.requests == []
    query, args = pool.execute_calls[0]
    assert "delivery_status = 'failed'" in query
    assert "delivery_status = 'sending'" in query
    assert args == ("acct-123", "content-ops-abc123", "report_not_paid")


@pytest.mark.asyncio
async def test_delivery_worker_marks_provider_failure_failed_with_bounded_error() -> None:
    pool = _Pool([_row()])
    sender = _Sender(error=RuntimeError("provider down " + ("x" * 700)))

    summary = await send_pending_deflection_report_deliveries(
        pool,
        sender=sender,
        config=_config(),
    )

    assert summary.failed == 1
    query, args = pool.execute_calls[0]
    assert "delivery_status = 'failed'" in query
    assert "delivery_status = 'sending'" in query
    assert args[0:2] == ("acct-123", "content-ops-abc123")
    assert args[2].startswith("RuntimeError: provider down")
    assert len(args[2]) == 500


def test_deflection_report_result_url_uses_template_and_quotes_request_id() -> None:
    url = deflection_report_result_url(
        request_id="content ops/abc",
        config=_config(
            result_base_url="",
            result_url_template="https://portfolio.example.com/r/{request_id}?checkout=success",
        ),
    )

    assert url == "https://portfolio.example.com/r/content%20ops%2Fabc?checkout=success"


def test_deflection_report_result_url_defaults_to_live_portfolio_result_route() -> None:
    url = deflection_report_result_url(
        request_id="content ops/abc",
        config=_config(result_base_url="https://juancanfield.com/"),
    )

    assert (
        url
        == "https://juancanfield.com/systems/support-ticket-deflection/results/"
        "content%20ops%2Fabc?checkout=success"
    )
    assert "/services/faq-deflection" not in url


def test_deflection_report_result_url_requires_configured_destination() -> None:
    with pytest.raises(ValueError, match="result_base_url or result_url_template"):
        deflection_report_result_url(request_id="req-123", config=_config(result_base_url=""))
