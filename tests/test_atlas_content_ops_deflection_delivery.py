from __future__ import annotations

import base64
import json
import sys
from types import ModuleType
from typing import Any

import pytest

from atlas_brain.content_ops_deflection_delivery import (
    DELIVERY_CLAIM_STALE_AFTER,
    DeflectionReportDeliveryConfig,
    deflection_report_result_url,
    send_pending_deflection_report_deliveries,
)
from atlas_brain.content_ops_deflection_incidents import INCIDENT_LOG_MARKER
from extracted_content_pipeline.campaign_ports import (
    IdempotentReplayConflict,
    SendRequest,
    SendResult,
)


def _incident_payloads(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for record in caplog.records:
        message = record.getMessage()
        if INCIDENT_LOG_MARKER in message:
            payloads.append(json.loads(message.split(INCIDENT_LOG_MARKER, 1)[1].strip()))
    return payloads


class _Pool:
    def __init__(self, rows: list[dict[str, Any]], *, sendable: bool = True) -> None:
        self.rows = rows
        self.sendable = sendable
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchrow_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetch_calls.append((query, args))
        assert "content_ops_deflection_report_deliveries" in query
        assert "content_ops_deflection_reports" in query
        assert "r.artifact" in query
        return self.rows[: int(args[0])]

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.fetchrow_calls.append((query, args))
        assert "content_ops_deflection_report_deliveries" in query
        assert "content_ops_deflection_reports" in query
        assert "delivery_status = 'sending'" in query
        assert "r.paid = true" in query
        if not self.sendable:
            return None
        return {"request_id": args[1]}

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
        "artifact": json.dumps({
            "markdown": (
                "# Support Ticket Deflection Report\n\n"
                "## Support Tax Confirmation\n\n"
                "Customers ask about invoices repeatedly.\n"
            ),
        }),
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


def _install_fake_pdf_renderer(
    monkeypatch: pytest.MonkeyPatch,
    renderer: Any,
) -> None:
    module = ModuleType("atlas_brain.deflection_pdf_renderer")
    module.render_deflection_full_report_pdf = renderer
    monkeypatch.setitem(sys.modules, "atlas_brain.deflection_pdf_renderer", module)


@pytest.mark.asyncio
async def test_delivery_worker_sends_pending_paid_report_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _Pool([_row()])
    sender = _Sender()
    _install_fake_pdf_renderer(monkeypatch, lambda _artifact: b"%PDF-fake-bytes")

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
    confirm_query, confirm_args = pool.fetchrow_calls[0]
    assert "RETURNING d.request_id" in confirm_query
    assert confirm_args == ("acct-123", "content-ops-abc123")
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
    assert request.attachments
    assert request.attachments[0]["filename"] == (
        "content-ops-abc123-support-deflection-report.pdf"
    )
    assert base64.b64decode(request.attachments[0]["content"]) == b"%PDF-fake-bytes"
    assert "full report PDF is attached" in request.html_body
    assert "full report PDF is attached" in (request.text_body or "")

    update_query, update_args = pool.execute_calls[0]
    assert "delivery_status = 'delivered'" in update_query
    assert "delivery_status = 'sending'" in update_query
    assert update_args == ("acct-123", "content-ops-abc123", "resend:email-123")
    assert "buyer@example.com" not in str(update_args)


@pytest.mark.asyncio
async def test_delivery_worker_renders_pdf_with_lazy_renderer_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _Pool([_row()])
    sender = _Sender()

    _install_fake_pdf_renderer(monkeypatch, lambda _artifact: b"%PDF-lazy-renderer")

    summary = await send_pending_deflection_report_deliveries(
        pool,
        sender=sender,
        config=_config(),
    )

    assert summary.sent == 1
    attachment = sender.requests[0].attachments[0]
    assert base64.b64decode(attachment["content"]) == b"%PDF-lazy-renderer"


@pytest.mark.asyncio
async def test_delivery_worker_falls_back_to_link_only_when_pdf_render_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _Pool([_row()])
    sender = _Sender()

    def _raise_pdf(_artifact: Any) -> bytes:
        raise RuntimeError("pdf down")

    _install_fake_pdf_renderer(monkeypatch, _raise_pdf)

    summary = await send_pending_deflection_report_deliveries(
        pool,
        sender=sender,
        config=_config(),
    )

    assert summary.sent == 1
    request = sender.requests[0]
    assert request.attachments == ()
    assert "full report PDF is attached" not in request.html_body
    assert "full report PDF is attached" not in (request.text_body or "")
    update_query, update_args = pool.execute_calls[0]
    assert "delivery_status = 'delivered'" in update_query
    assert update_args == ("acct-123", "content-ops-abc123", "resend:email-123")


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
async def test_delivery_worker_rechecks_paid_and_status_before_sending(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING", logger="atlas.content_ops_deflection_delivery")
    pool = _Pool([_row()], sendable=False)
    sender = _Sender()
    _install_fake_pdf_renderer(monkeypatch, lambda _artifact: b"%PDF-fake-bytes")

    summary = await send_pending_deflection_report_deliveries(
        pool,
        sender=sender,
        config=_config(),
    )

    assert summary.scanned == 1
    assert summary.sent == 0
    assert summary.failed == 1
    assert sender.requests == []
    assert pool.fetchrow_calls
    assert pool.execute_calls == []
    assert _incident_payloads(caplog) == [
        {
            "account_id": "acct-123",
            "incident_type": "paid_report_delivery_no_longer_sendable",
            "request_id": "content-ops-abc123",
            "severity": "warning",
        }
    ]


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
async def test_delivery_worker_does_not_double_send_on_reclaim_after_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # #1461: a crash between send() and _mark_delivered leaves the row
    # 'sending'; the claim re-tries stale 'sending' rows, so the worker calls
    # send() again. A deterministic idempotency key keeps the resend deduped.
    class _IdempotentResendSender:
        """Resend-style sender: identical Idempotency-Key values are deduped."""

        def __init__(self) -> None:
            self.calls: list[SendRequest] = []
            self.delivered: dict[str, str] = {}

        async def send(self, request: SendRequest) -> SendResult:
            self.calls.append(request)
            key = request.idempotency_key or ""
            if key in self.delivered:
                message_id = self.delivered[key]
                return SendResult(
                    provider="resend",
                    message_id=message_id,
                    raw={"id": message_id, "deduped": True},
                )
            message_id = f"email-{len(self.delivered) + 1}"
            self.delivered[key] = message_id
            return SendResult(
                provider="resend", message_id=message_id, raw={"id": message_id}
            )

    _install_fake_pdf_renderer(monkeypatch, lambda _artifact: b"%PDF-fake-bytes")
    sender = _IdempotentResendSender()
    row = _row()

    # Run 1: the delivered-mark UPDATE raises, so the process dies before
    # finalizing and the row stays 'sending'.
    class _CrashOnMarkPool(_Pool):
        async def execute(self, query: str, *args: Any) -> str:
            raise RuntimeError("crash between send and mark")

    crash_pool = _CrashOnMarkPool([row])
    with pytest.raises(RuntimeError, match="crash between send and mark"):
        await send_pending_deflection_report_deliveries(
            crash_pool, sender=sender, config=_config()
        )
    assert len(sender.calls) == 1
    assert len(sender.delivered) == 1  # one real email so far

    # Run 2: the claim re-tries the still-'sending' row; the resend carries the
    # same deterministic key, so Resend dedupes it and no second email is sent.
    reclaim_pool = _Pool([row])
    summary = await send_pending_deflection_report_deliveries(
        reclaim_pool, sender=sender, config=_config()
    )

    assert summary.sent == 1
    assert len(sender.calls) == 2
    assert sender.calls[0].idempotency_key == sender.calls[1].idempotency_key
    assert (
        sender.calls[1].idempotency_key
        == "deflection-report:acct-123:content-ops-abc123"
    )
    assert len(sender.delivered) == 1  # still one unique email -- no duplicate


@pytest.mark.asyncio
async def test_delivery_worker_marks_idempotent_replay_delivered_not_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # On re-claim the PDF re-renders to different bytes, so Resend rejects the
    # retry with 409 invalid_idempotent_request -> IdempotentReplayConflict. The
    # original email already went out, so the row must be delivered (not failed,
    # which would fire a false send-failure incident).
    class _ConflictSender:
        async def send(self, request: SendRequest) -> SendResult:
            raise IdempotentReplayConflict(request.idempotency_key or "")

    _install_fake_pdf_renderer(monkeypatch, lambda _artifact: b"%PDF-fake-bytes")
    pool = _Pool([_row()])

    summary = await send_pending_deflection_report_deliveries(
        pool, sender=_ConflictSender(), config=_config()
    )

    assert summary.sent == 1
    assert summary.failed == 0
    update_query, update_args = pool.execute_calls[0]
    assert "delivery_status = 'delivered'" in update_query
    assert "delivery_status = 'sending'" in update_query
    assert update_args == ("acct-123", "content-ops-abc123", "resend:idempotent-replay")


@pytest.mark.asyncio
async def test_reclaim_changing_attachment_payload_marks_delivered_via_real_resend_sender(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # End-to-end regression for the R8 BLOCKER (real ResendCampaignSender + the
    # real render->link-only fallback transition): first send carries the PDF
    # attachment; on re-claim the render fails so the email is link-only -- a
    # DIFFERENT payload under the SAME idempotency key. A Resend-like client
    # returns 409 invalid_idempotent_request for that mismatch, and the delivery
    # path must mark the row delivered (idempotent replay), not failed.
    from extracted_content_pipeline.campaign_sender import (
        ResendCampaignSender,
        ResendSenderConfig,
    )

    class _Resp:
        def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"http {self.status_code}")

        def json(self) -> dict[str, Any]:
            return self._payload

    class _IdempotentResendHTTP:
        """Records the first body per key; a different body on the same key 409s."""

        def __init__(self) -> None:
            self.seen: dict[str, Any] = {}
            self.posts: list[dict[str, Any]] = []

        async def post(self, url: str, *, json: Any, headers: Any) -> _Resp:
            key = headers.get("Idempotency-Key")
            self.posts.append({"key": key, "json": json})
            if key and key in self.seen and self.seen[key] != json:
                return _Resp(
                    {"name": "invalid_idempotent_request", "message": "different payload"},
                    status_code=409,
                )
            self.seen.setdefault(key, json)
            return _Resp({"id": f"email-{len(self.seen)}"})

    http = _IdempotentResendHTTP()
    sender = ResendCampaignSender(ResendSenderConfig(api_key="re_key"), http_client=http)

    render_calls = {"n": 0}

    def _renderer(_artifact: Any) -> bytes:
        render_calls["n"] += 1
        if render_calls["n"] == 1:
            return b"%PDF-first-render"
        raise RuntimeError("render failed on reclaim")

    _install_fake_pdf_renderer(monkeypatch, _renderer)
    row = _row()

    first = await send_pending_deflection_report_deliveries(
        _Pool([row]), sender=sender, config=_config()
    )
    assert first.sent == 1 and first.failed == 0

    # Re-claim: render now fails -> link-only -> different payload, same key.
    second = await send_pending_deflection_report_deliveries(
        _Pool([row]), sender=sender, config=_config()
    )
    assert second.sent == 1  # delivered via idempotent replay, NOT failed
    assert second.failed == 0
    assert len(http.posts) == 2
    assert http.posts[0]["key"] == http.posts[1]["key"]  # same idempotency key
    assert http.posts[0]["json"] != http.posts[1]["json"]  # attachment vs link-only


@pytest.mark.asyncio
async def test_delivery_worker_marks_missing_email_failed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("ERROR", logger="atlas.content_ops_deflection_delivery")
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
    payloads = _incident_payloads(caplog)
    assert payloads == [
        {
            "account_id": "acct-123",
            "incident_type": "paid_report_delivery_missing_email",
            "request_id": "content-ops-abc123",
            "severity": "error",
        }
    ]
    assert "buyer@example.com" not in caplog.text


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
async def test_delivery_worker_marks_provider_failure_failed_with_bounded_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("ERROR", logger="atlas.content_ops_deflection_delivery")
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
    payloads = _incident_payloads(caplog)
    assert payloads[0]["incident_type"] == "paid_report_delivery_send_failed"
    assert payloads[0]["account_id"] == "acct-123"
    assert payloads[0]["request_id"] == "content-ops-abc123"
    assert payloads[0]["severity"] == "error"
    assert payloads[0]["error"].startswith("RuntimeError: provider down")
    assert "buyer@example.com" not in caplog.text


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
