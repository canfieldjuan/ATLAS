from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from atlas_brain.content_ops_deflection_delivery import (
    DELIVERY_CLAIM_STALE_AFTER,
    DeflectionDeltaDeliveryConfig,
    DeflectionReportDeliveryConfig,
    deflection_delta_delivery_summary,
    deflection_delivery_email_surface_observation,
    deflection_report_result_url,
    pending_deflection_delta_delivery_count,
    send_pending_deflection_delta_deliveries,
    send_pending_deflection_report_deliveries,
)
from atlas_brain.content_ops_deflection_incidents import INCIDENT_LOG_MARKER
from extracted_content_pipeline.campaign_ports import (
    IdempotentReplayConflict,
    SendRequest,
    SendResult,
)
from extracted_content_pipeline.faq_deflection_report import (
    build_deflection_full_report_qa_scorecard,
)


ROOT = Path(__file__).resolve().parents[1]
_DEFLECTION_DELTA_DELIVERY_MIGRATIONS = (
    ROOT / "atlas_brain" / "storage" / "migrations" / "328_content_ops_deflection_reports.sql",
    ROOT / "atlas_brain" / "storage" / "migrations" / "331_content_ops_deflection_report_delivery_email.sql",
    ROOT / "atlas_brain" / "storage" / "migrations" / "340_content_ops_deflection_deltas.sql",
    ROOT / "atlas_brain" / "storage" / "migrations" / "341_content_ops_deflection_delta_deliveries.sql",
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


class _DeltaPool:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        sendable: bool = True,
    ) -> None:
        self.rows = rows
        self.sendable = sendable
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchval_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchrow_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetch_calls.append((query, args))
        assert "content_ops_deflection_delta_deliveries" in query
        assert "content_ops_deflection_deltas" in query
        assert "content_ops_deflection_reports current_report" in query
        assert "content_ops_deflection_reports baseline_report" in query
        assert "current_report.delivery_email" in query
        assert "d.delivery_email" not in query
        return self.rows[: int(args[0])]

    async def fetchval(self, query: str, *args: Any) -> int:
        self.fetchval_calls.append((query, args))
        assert "content_ops_deflection_delta_deliveries" in query
        assert "delivery_status = 'pending'" in query
        assert DELIVERY_CLAIM_STALE_AFTER in query
        return len(self.rows)

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.fetchrow_calls.append((query, args))
        assert "content_ops_deflection_delta_deliveries" in query
        assert "delivery_status = 'sending'" in query
        assert "current_report.paid = true" in query
        assert "baseline_report.paid = true" in query
        if not self.sendable:
            return None
        return {"current_request_id": args[1]}

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


def _delivery_report_model_artifact(
    *,
    schema_version: str = "deflection.v1",
    support_tax_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    support_tax_data = {
        "repeat_ticket_count": 1234,
        "non_repeat_ticket_count": 17,
        "generated_question_count": 8,
        "assisted_contact_cost": 13.50,
        "estimated_support_cost": 16659.0,
        "source_date_window": None,
        "drafted_answer_count": 3,
        "no_proven_answer_count": 5,
        "ticket_source_count": 42,
    }
    support_tax_data.update(support_tax_overrides or {})
    return {
        "markdown": (
            "# Support Ticket Deflection Report\n\n"
            "RAW MARKDOWN BODY SHOULD NOT ENTER THE EMAIL\n"
        ),
        "report_model": {
            "schema_version": schema_version,
            "title": "Support Ticket Deflection Report",
            "summary": {},
            "sections": [
                {
                    "id": "support_tax",
                    "title": "Support Tax Confirmation",
                    "priority": 10,
                    "surfaces": ["web", "pdf", "email_summary", "markdown"],
                    "default_limit": None,
                    "required_data": [
                        "repeat_ticket_count",
                        "non_repeat_ticket_count",
                        "generated_question_count",
                        "assisted_contact_cost",
                        "estimated_support_cost",
                        "source_date_window",
                        "drafted_answer_count",
                        "no_proven_answer_count",
                        "ticket_source_count",
                    ],
                    "data": support_tax_data,
                },
                {
                    "id": "question_details",
                    "title": "Question Details and Evidence",
                    "priority": 50,
                    "surfaces": ["web", "pdf", "markdown"],
                    "default_limit": None,
                    "required_data": ["rows"],
                    "data": {
                        "rows": [
                            {
                                "source_ids": ["ticket-secret-123"],
                                "evidence_quotes": [
                                    "private evidence quote should stay out",
                                ],
                                "answer_evidence_status": "resolution_evidence",
                            }
                        ]
                    },
                },
                {
                    "id": "priority_fix_queue",
                    "title": "Priority Fix Queue",
                    "priority": 35,
                    "surfaces": ["web", "pdf", "email_summary"],
                    "default_limit": 3,
                    "required_data": ["items", "result_page_limit", "pdf_limit"],
                    "data": {
                        "result_page_limit": 2,
                        "pdf_limit": 10,
                        "items": [
                            {
                                "question": "How do I enable SSO?",
                                "ticket_count": 6,
                                "estimated_support_cost": 81.0,
                                "status": "Draft ready",
                                "recommended_action": "Review and publish",
                                "source_ids": ["ticket-secret-priority-draft"],
                            },
                            {
                                "question": "How do I export attribution reports?",
                                "ticket_count": 9,
                                "estimated_support_cost": 121.5,
                                "owner_lane": "Reporting",
                                "evidence_tier": "csv_customer_text",
                                "product_gap_summary": (
                                    "Repeated support friction routes to Reporting. "
                                    "9 support tickets in this upload."
                                ),
                                "customer_vocabulary": [
                                    "export attribution reports",
                                    "download attribution CSV",
                                ],
                                "cost_period": "batch_upload",
                                "cost_confidence": "benchmark_with_customer_text",
                                "status": "Needs answer",
                                "recommended_action": "Create a help-center answer",
                                "representative_phrasing": [
                                    "RAW_REPRESENTATIVE_PHRASE_SHOULD_STAY_OUT",
                                ],
                                "source_ids": ["ticket-secret-action-1"],
                                "top_evidence": [
                                    {
                                        "source_id": "ticket-secret-action-2",
                                        "evidence_quote": (
                                            "raw action evidence should stay out"
                                        ),
                                    },
                                ],
                            },
                            {
                                "question": "How do I update invoice contacts?",
                                "ticket_count": 4,
                                "estimated_support_cost": 54.0,
                                "status": "Needs review",
                                "recommended_action": "Approve the billing macro",
                            },
                            {
                                "question": "How do I invite an auditor?",
                                "ticket_count": 3,
                                "estimated_support_cost": 40.5,
                                "status": "Needs answer",
                                "recommended_action": "Create an admin answer",
                            },
                        ],
                    },
                },
                {
                    "id": "drafted_resolutions",
                    "title": "Drafted Resolutions Ready to Publish",
                    "priority": 37,
                    "surfaces": ["web", "pdf", "email_summary"],
                    "default_limit": 3,
                    "required_data": ["items", "result_page_limit", "pdf_limit"],
                    "data": {
                        "result_page_limit": 1,
                        "pdf_limit": 10,
                        "items": [
                            {
                                "question": "How do I enable SSO?",
                                "ticket_count": 6,
                                "estimated_support_cost": 81.0,
                                "status": "Draft ready",
                                "recommended_action": "Review and publish",
                                "source_ids": ["ticket-secret-draft-1"],
                            },
                            {
                                "question": "How do I change workspace owners?",
                                "ticket_count": 2,
                                "estimated_support_cost": 27.0,
                                "status": "Draft ready",
                                "recommended_action": "Publish ownership FAQ",
                            },
                        ],
                    },
                },
            ],
        },
    }


def _config(**overrides: Any) -> DeflectionReportDeliveryConfig:
    values = {
        "from_email": "Atlas Content Ops <reports@example.com>",
        "result_base_url": "https://portfolio.example.com",
        "reply_to": "support@example.com",
        "limit": 20,
    }
    values.update(overrides)
    return DeflectionReportDeliveryConfig(**values)


def _delta_read_payload() -> dict[str, Any]:
    return {
        "schema_version": "deflection_delta_read.v1",
        "current_request_id": "current-report",
        "baseline_request_id": "baseline-report",
        "delta": {
            "schema_version": "deflection_delta.v1",
            "current": {
                "source_date_start": "2026-05-01",
                "source_date_end": "2026-05-31",
            },
            "baseline": {
                "source_date_start": "2026-04-01",
                "source_date_end": "2026-04-30",
            },
            "summary": {
                "new_count": 2,
                "resolved_count": 1,
                "growing_count": 3,
                "shrinking_count": 1,
                "still_unresolved_count": 4,
                "support_cost_delta": 243.0,
            },
            "items": [
                {
                    "question": "How do I export attribution reports?",
                    "owner_lane": "Reporting",
                    "baseline_status": "Needs answer",
                    "current_status": "Needs answer",
                    "ticket_count_delta": 4,
                    "support_cost_delta": 54.0,
                    "change_types": ["NEW", "GROWING"],
                    "source_ids": ["ticket-secret-delta-1"],
                    "top_evidence": [
                        {"evidence_quote": "raw delta quote should stay out"}
                    ],
                    "representative_phrasing": [
                        "raw representative phrasing should stay out"
                    ],
                },
                {
                    "question": "How do I close a workspace?",
                    "owner_lane": "Admin",
                    "baseline_status": "Needs answer",
                    "current_status": "Draft ready",
                    "ticket_count_delta": -2,
                    "support_cost_delta": -27.0,
                    "change_types": ["RESOLVED", "SHRINKING"],
                },
                {
                    "question": "How do I enable SSO?",
                    "owner_lane": "Security",
                    "baseline_status": "Needs answer",
                    "current_status": "Needs answer",
                    "ticket_count_delta": 1,
                    "support_cost_delta": 13.5,
                    "change_types": ["STILL_UNRESOLVED"],
                },
                {
                    "question": "How do I invite auditors?",
                    "owner_lane": "Admin",
                    "baseline_status": "Needs answer",
                    "current_status": "Needs answer",
                    "ticket_count_delta": 1,
                    "support_cost_delta": 10.0,
                    "change_types": ["GROWING"],
                },
            ],
        },
    }


def _delta_delivery_row(**overrides: Any) -> dict[str, Any]:
    payload = _delta_read_payload()
    row = {
        "account_id": "acct-123",
        "current_request_id": payload["current_request_id"],
        "baseline_request_id": payload["baseline_request_id"],
        "delivery_email": "buyer@example.com",
        "current_paid": True,
        "baseline_paid": True,
        "delta": json.dumps(payload["delta"]),
        "delta_created_at": "2026-06-01T00:00:00Z",
        "delta_updated_at": "2026-06-01T00:00:00Z",
    }
    row.update(overrides)
    return row


def _install_fake_pdf_renderer(
    monkeypatch: pytest.MonkeyPatch,
    renderer: Any,
) -> None:
    module = ModuleType("atlas_brain.deflection_pdf_renderer")
    module.render_deflection_full_report_pdf = renderer
    monkeypatch.setitem(sys.modules, "atlas_brain.deflection_pdf_renderer", module)


def test_deflection_delta_delivery_summary_renders_bounded_action_copy() -> None:
    rendered = deflection_delta_delivery_summary(_delta_read_payload(), max_items=2)

    assert rendered.subject == "Your support deflection delta is ready"
    assert "2026-05-01 to 2026-05-31 vs 2026-04-01 to 2026-04-30" in rendered.text_body
    assert "2 new repeats" in rendered.text_body
    assert "1 resolved repeats" in rendered.text_body
    assert "3 growing repeats" in rendered.text_body
    assert "4 still unresolved repeats" in rendered.text_body
    assert "+$243 estimated assisted-contact handling" in rendered.text_body
    assert "How do I export attribution reports?" in rendered.text_body
    assert "New, Growing" in rendered.text_body
    assert "+4 tickets" in rendered.text_body
    assert "+$54 estimated handling" in rendered.text_body
    assert "Owner: Reporting" in rendered.text_body
    assert "Status: Needs answer -> Needs answer" in rendered.text_body
    assert "How do I close a workspace?" in rendered.text_body
    assert "-2 tickets" in rendered.text_body
    assert "-$27 estimated handling" in rendered.text_body
    assert "How do I enable SSO?" not in rendered.text_body
    assert "How do I invite auditors?" not in rendered.text_body
    assert "ticket-secret-delta-1" not in rendered.text_body
    assert "raw delta quote should stay out" not in rendered.text_body
    assert "raw representative phrasing should stay out" not in rendered.text_body
    assert "Top changes to review" in rendered.html_body
    assert "How do I export attribution reports?" in rendered.html_body
    assert "ticket-secret-delta-1" not in rendered.html_body


def test_deflection_delta_delivery_summary_escapes_html_and_handles_savings() -> None:
    payload = _delta_read_payload()
    delta = payload["delta"]
    assert isinstance(delta, dict)
    summary = delta["summary"]
    assert isinstance(summary, dict)
    summary["support_cost_delta"] = -135.0
    items = delta["items"]
    assert isinstance(items, list)
    items[0]["question"] = "Can <script>alert('x')</script> export data?"

    rendered = deflection_delta_delivery_summary(payload, max_items=1)

    assert "-$135 estimated assisted-contact handling" in rendered.text_body
    assert "Can <script>alert('x')</script> export data?" in rendered.text_body
    assert "Can &lt;script&gt;alert('x')&lt;/script&gt; export data?" in rendered.html_body
    assert "<script>" not in rendered.html_body


@pytest.mark.asyncio
async def test_send_pending_deflection_delta_deliveries_sends_allowlisted_payload() -> None:
    pool = _DeltaPool([_delta_delivery_row()])
    sender = _Sender()

    summary = await send_pending_deflection_delta_deliveries(
        pool,
        sender=sender,
        config=DeflectionDeltaDeliveryConfig(
            from_email="reports@example.com",
            reply_to="support@example.com",
            limit=5,
            dry_run=False,
        ),
    )

    assert summary.sent == 1
    assert summary.failed == 0
    assert summary.dry_run == 0
    assert len(sender.requests) == 1
    request = sender.requests[0]
    assert request.to_email == "buyer@example.com"
    assert request.from_email == "reports@example.com"
    assert request.reply_to == "support@example.com"
    assert request.subject == "Your support deflection delta is ready"
    assert request.campaign_id == (
        "content_ops_deflection_delta:acct-123:current-report:baseline-report"
    )
    assert request.idempotency_key == (
        "deflection-delta:acct-123:current-report:baseline-report"
    )
    assert request.tags == (
        {"name": "source", "value": "content_ops_deflection_delta"},
        {"name": "current_request_id", "value": "current-report"},
        {"name": "baseline_request_id", "value": "baseline-report"},
    )
    assert "ticket-secret-delta-1" not in request.text_body
    assert "raw evidence" not in request.text_body
    assert "raw phrasing" not in request.html_body
    assert len(pool.fetchrow_calls) == 1
    delivered_query, delivered_args = pool.execute_calls[-1]
    assert "delivery_status = 'delivered'" in delivered_query
    assert delivered_args == (
        "acct-123",
        "current-report",
        "baseline-report",
        "resend:email-123",
    )


@pytest.mark.asyncio
async def test_send_pending_deflection_delta_deliveries_dry_run_does_not_mutate() -> None:
    pool = _DeltaPool([_delta_delivery_row()])
    sender = _Sender()

    summary = await send_pending_deflection_delta_deliveries(
        pool,
        sender=sender,
        config=DeflectionDeltaDeliveryConfig(
            from_email="reports@example.com",
            limit=5,
            dry_run=True,
        ),
    )

    assert summary.scanned == 1
    assert summary.sent == 0
    assert summary.failed == 0
    assert summary.deferred == 0
    assert summary.dry_run == 1
    assert sender.requests == []
    assert pool.fetchrow_calls == []
    assert pool.execute_calls == []


@pytest.mark.asyncio
async def test_send_pending_deflection_delta_deliveries_dry_run_skips_invalid_rows() -> None:
    pool = _DeltaPool([
        _delta_delivery_row(
            delivery_email="",
            current_paid=False,
            baseline_paid=False,
        )
    ])
    sender = _Sender()

    summary = await send_pending_deflection_delta_deliveries(
        pool,
        sender=sender,
        config=DeflectionDeltaDeliveryConfig(
            from_email="reports@example.com",
            limit=5,
            dry_run=True,
        ),
    )

    assert summary.scanned == 1
    assert summary.sent == 0
    assert summary.failed == 0
    assert summary.dry_run == 1
    assert sender.requests == []
    assert pool.fetchrow_calls == []
    assert pool.execute_calls == []


@pytest.mark.asyncio
async def test_send_pending_deflection_delta_deliveries_defers_unpaid_sources(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pool = _DeltaPool([_delta_delivery_row(current_paid=False)])
    sender = _Sender()
    caplog.set_level("WARNING", logger="atlas.content_ops_deflection_delivery")

    summary = await send_pending_deflection_delta_deliveries(
        pool,
        sender=sender,
        config=DeflectionDeltaDeliveryConfig(
            from_email="reports@example.com",
            limit=5,
            dry_run=False,
        ),
    )

    assert summary.failed == 0
    assert summary.deferred == 1
    assert sender.requests == []
    deferred_query, deferred_args = pool.execute_calls[-1]
    assert "delivery_status = 'pending'" in deferred_query
    assert "delivery_status = 'failed'" not in deferred_query
    assert deferred_args[-1] == "source_report_not_paid"
    incidents = _incident_payloads(caplog)
    assert incidents[-1]["incident_type"] == "delta_delivery_source_report_not_paid"
    assert "Deflection delta delivery deferred" in caplog.text


@pytest.mark.asyncio
async def test_send_pending_deflection_delta_deliveries_defers_when_no_longer_sendable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pool = _DeltaPool([_delta_delivery_row()], sendable=False)
    sender = _Sender()
    caplog.set_level("WARNING", logger="atlas.content_ops_deflection_delivery")

    summary = await send_pending_deflection_delta_deliveries(
        pool,
        sender=sender,
        config=DeflectionDeltaDeliveryConfig(
            from_email="reports@example.com",
            limit=5,
            dry_run=False,
        ),
    )

    assert summary.failed == 0
    assert summary.deferred == 1
    assert sender.requests == []
    deferred_query, deferred_args = pool.execute_calls[-1]
    assert "delivery_status = 'pending'" in deferred_query
    assert deferred_args[-1] == "delta_no_longer_sendable"
    incidents = _incident_payloads(caplog)
    assert incidents[-1]["incident_type"] == "delta_delivery_no_longer_sendable"


@pytest.mark.asyncio
async def test_pending_deflection_delta_delivery_count_includes_stale_sending() -> None:
    pool = _DeltaPool([_delta_delivery_row(), _delta_delivery_row()])

    count = await pending_deflection_delta_delivery_count(pool)

    assert count == 2
    query, args = pool.fetchval_calls[0]
    assert args == (None, None, None)
    assert "delivery_status = 'pending'" in query
    assert "delivery_status = 'sending'" in query
    assert DELIVERY_CLAIM_STALE_AFTER in query


@pytest.mark.asyncio
async def test_delta_delivery_queue_can_be_scoped_to_one_account() -> None:
    pool = _DeltaPool([_delta_delivery_row()])
    sender = _Sender()

    summary = await send_pending_deflection_delta_deliveries(
        pool,
        sender=sender,
        config=DeflectionDeltaDeliveryConfig(
            from_email="reports@example.com",
            limit=5,
            dry_run=True,
        ),
        account_id=" acct-target ",
    )
    count = await pending_deflection_delta_delivery_count(
        pool,
        account_id=" acct-target ",
    )

    assert summary.scanned == 1
    assert count == 1
    fetch_query, fetch_args = pool.fetch_calls[0]
    assert fetch_args == (5, "acct-target", None, None)
    assert "AND ($2::text IS NULL OR d.account_id = $2)" in fetch_query
    count_query, count_args = pool.fetchval_calls[0]
    assert count_args == ("acct-target", None, None)
    assert "AND ($1::text IS NULL OR account_id = $1)" in count_query


@pytest.mark.asyncio
async def test_delta_delivery_queue_can_be_scoped_to_one_current_request() -> None:
    pool = _DeltaPool([_delta_delivery_row()])
    sender = _Sender()

    summary = await send_pending_deflection_delta_deliveries(
        pool,
        sender=sender,
        config=DeflectionDeltaDeliveryConfig(
            from_email="reports@example.com",
            limit=5,
            dry_run=False,
        ),
        account_id="acct-target",
        current_request_id="checked-current",
    )
    count = await pending_deflection_delta_delivery_count(
        pool,
        account_id="acct-target",
        current_request_id="checked-current",
    )

    assert summary.scanned == 1
    assert count == 1
    claim_query, claim_args = pool.fetch_calls[0]
    assert claim_args == (5, "acct-target", "checked-current", None)
    assert "AND ($2::text IS NULL OR d.account_id = $2)" in claim_query
    assert "AND ($3::text IS NULL OR d.current_request_id = $3)" in claim_query
    count_query, count_args = pool.fetchval_calls[0]
    assert count_args == ("acct-target", "checked-current", None)
    assert "AND ($2::text IS NULL OR current_request_id = $2)" in count_query


@pytest.mark.asyncio
async def test_delta_delivery_queue_filters_entitled_accounts_on_global_drain() -> None:
    pool = _DeltaPool([_delta_delivery_row()])
    sender = _Sender()

    summary = await send_pending_deflection_delta_deliveries(
        pool,
        sender=sender,
        config=DeflectionDeltaDeliveryConfig(
            from_email="reports@example.com",
            limit=5,
            dry_run=True,
        ),
        entitled_account_ids=("acct-active", " ", "acct-active"),
    )
    count = await pending_deflection_delta_delivery_count(
        pool,
        entitled_account_ids=("acct-active",),
    )

    assert summary.scanned == 1
    assert count == 1
    fetch_query, fetch_args = pool.fetch_calls[0]
    assert fetch_args == (5, None, None, ["acct-active"])
    assert "AND ($4::text[] IS NULL OR d.account_id = ANY($4::text[]))" in fetch_query
    count_query, count_args = pool.fetchval_calls[0]
    assert count_args == (None, None, ["acct-active"])
    assert "AND ($3::text[] IS NULL OR account_id = ANY($3::text[]))" in count_query


async def _apply_delta_delivery_migrations(conn: Any) -> None:
    for path in _DEFLECTION_DELTA_DELIVERY_MIGRATIONS:
        await conn.execute(path.read_text(encoding="utf-8"))


async def _cleanup_delta_delivery_scope_rows(conn: Any, accounts: tuple[str, ...]) -> None:
    await conn.execute(
        "DELETE FROM content_ops_deflection_reports WHERE account_id = ANY($1::text[])",
        list(accounts),
    )


async def _insert_delta_delivery_fixture(
    conn: Any,
    *,
    account_id: str,
    current_request_id: str,
    baseline_request_id: str,
    current_email: str,
) -> None:
    delta = _delta_read_payload()["delta"]
    await conn.execute(
        """
        INSERT INTO content_ops_deflection_reports (
            account_id,
            request_id,
            snapshot,
            artifact,
            paid,
            paid_at,
            delivery_email
        )
        VALUES
            ($1, $2, '{}'::jsonb, '{}'::jsonb, true, NOW(), $4),
            ($1, $3, '{}'::jsonb, '{}'::jsonb, true, NOW(), 'baseline@example.com')
        ON CONFLICT (account_id, request_id) DO UPDATE
        SET paid = EXCLUDED.paid,
            paid_at = EXCLUDED.paid_at,
            delivery_email = EXCLUDED.delivery_email
        """,
        account_id,
        current_request_id,
        baseline_request_id,
        current_email,
    )
    await conn.execute(
        """
        INSERT INTO content_ops_deflection_deltas (
            account_id,
            current_request_id,
            baseline_request_id,
            delta
        )
        VALUES ($1, $2, $3, $4::jsonb)
        ON CONFLICT (account_id, current_request_id, baseline_request_id)
        DO UPDATE SET delta = EXCLUDED.delta
        """,
        account_id,
        current_request_id,
        baseline_request_id,
        json.dumps(delta),
    )
    await conn.execute(
        """
        INSERT INTO content_ops_deflection_delta_deliveries (
            account_id,
            current_request_id,
            baseline_request_id,
            delivery_email
        )
        VALUES ($1, $2, $3, 'queued-address-should-not-send@example.com')
        ON CONFLICT (account_id, current_request_id, baseline_request_id)
        DO UPDATE SET delivery_status = 'pending',
                      delivery_error = NULL,
                      provider_message_id = NULL,
                      delivered_at = NULL,
                      updated_at = NOW()
        """,
        account_id,
        current_request_id,
        baseline_request_id,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delta_delivery_scope_live_postgres_drains_only_target_account_current_request() -> None:
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.environ.get("ATLAS_MIGRATION_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ATLAS_MIGRATION_TEST_DATABASE_URL not set")

    target_account = "acct-delta-scope-target"
    other_account = "acct-delta-scope-other"
    target_current = "current-delta-scope-target"
    target_baseline = "baseline-delta-scope-target"
    other_current = "current-delta-scope-other"
    other_baseline = "baseline-delta-scope-other"
    conn = await asyncpg.connect(database_url)
    try:
        await _apply_delta_delivery_migrations(conn)
        await _cleanup_delta_delivery_scope_rows(conn, (target_account, other_account))
        await _insert_delta_delivery_fixture(
            conn,
            account_id=target_account,
            current_request_id=target_current,
            baseline_request_id=target_baseline,
            current_email="target-buyer@example.com",
        )
        await _insert_delta_delivery_fixture(
            conn,
            account_id=target_account,
            current_request_id=other_current,
            baseline_request_id=target_baseline,
            current_email="same-account-other-current@example.com",
        )
        await _insert_delta_delivery_fixture(
            conn,
            account_id=other_account,
            current_request_id=target_current,
            baseline_request_id=other_baseline,
            current_email="other-account-same-current@example.com",
        )

        sender = _Sender()
        summary = await send_pending_deflection_delta_deliveries(
            conn,
            sender=sender,
            config=DeflectionDeltaDeliveryConfig(
                from_email="reports@example.com",
                limit=10,
                dry_run=False,
            ),
            account_id=target_account,
            current_request_id=target_current,
        )

        rows = await conn.fetch(
            """
            SELECT account_id,
                   current_request_id,
                   baseline_request_id,
                   delivery_status,
                   provider_message_id,
                   delivered_at
            FROM content_ops_deflection_delta_deliveries
            WHERE account_id = ANY($1::text[])
            ORDER BY account_id, current_request_id, baseline_request_id
            """,
            [target_account, other_account],
        )
    finally:
        await _cleanup_delta_delivery_scope_rows(conn, (target_account, other_account))
        await conn.close()

    assert summary.scanned == 1
    assert summary.sent == 1
    assert summary.failed == 0
    assert summary.deferred == 0
    assert [request.to_email for request in sender.requests] == ["target-buyer@example.com"]
    by_key = {
        (row["account_id"], row["current_request_id"], row["baseline_request_id"]): row
        for row in rows
    }
    target = by_key[(target_account, target_current, target_baseline)]
    assert target["delivery_status"] == "delivered"
    assert target["provider_message_id"] == "resend:email-123"
    assert target["delivered_at"] is not None

    same_account_other_current = by_key[(target_account, other_current, target_baseline)]
    assert same_account_other_current["delivery_status"] == "pending"
    assert same_account_other_current["provider_message_id"] is None
    assert same_account_other_current["delivered_at"] is None

    other_account_same_current = by_key[(other_account, target_current, other_baseline)]
    assert other_account_same_current["delivery_status"] == "pending"
    assert other_account_same_current["provider_message_id"] is None
    assert other_account_same_current["delivered_at"] is None


@pytest.mark.asyncio
async def test_send_pending_deflection_delta_deliveries_rejects_empty_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = _delta_read_payload()
    delta = payload["delta"]
    assert isinstance(delta, dict)
    delta["items"] = []
    summary = delta["summary"]
    assert isinstance(summary, dict)
    for key in list(summary):
        summary[key] = 0
    pool = _DeltaPool([_delta_delivery_row(delta=json.dumps(delta))])
    sender = _Sender()
    caplog.set_level("WARNING", logger="atlas.content_ops_deflection_delivery")

    summary_result = await send_pending_deflection_delta_deliveries(
        pool,
        sender=sender,
        config=DeflectionDeltaDeliveryConfig(
            from_email="reports@example.com",
            limit=5,
            dry_run=False,
        ),
    )

    assert summary_result.failed == 1
    assert summary_result.deferred == 0
    assert sender.requests == []
    failed_query, failed_args = pool.execute_calls[-1]
    assert "delivery_status = 'failed'" in failed_query
    assert failed_args[-1] == "empty_delta_payload"
    incidents = _incident_payloads(caplog)
    assert incidents[-1]["incident_type"] == "delta_delivery_empty_payload"
    assert incidents[-1]["current_request_id"] == "current-report"
    assert incidents[-1]["baseline_request_id"] == "baseline-report"
    assert "empty_delta_payload" in caplog.text


@pytest.mark.asyncio
async def test_send_pending_deflection_delta_deliveries_logs_and_incidents_on_send_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pool = _DeltaPool([_delta_delivery_row()])
    sender = _Sender(error=RuntimeError("resend down"))
    caplog.set_level("ERROR", logger="atlas.content_ops_deflection_delivery")

    summary = await send_pending_deflection_delta_deliveries(
        pool,
        sender=sender,
        config=DeflectionDeltaDeliveryConfig(
            from_email="reports@example.com",
            limit=5,
            dry_run=False,
        ),
    )

    assert summary.failed == 1
    assert summary.deferred == 0
    failed_query, failed_args = pool.execute_calls[-1]
    assert "delivery_status = 'failed'" in failed_query
    assert failed_args[-1] == "RuntimeError: resend down"
    incidents = _incident_payloads(caplog)
    assert incidents[-1]["incident_type"] == "delta_delivery_send_failed"
    assert incidents[-1]["error"] == "RuntimeError: resend down"
    assert "Deflection delta delivery failed" in caplog.text


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
    assert "Key numbers" not in request.html_body
    assert "full report PDF is attached" in request.html_body
    assert "full report PDF is attached" in (request.text_body or "")

    update_query, update_args = pool.execute_calls[0]
    assert "delivery_status = 'delivered'" in update_query
    assert "delivery_status = 'sending'" in update_query
    assert update_args == ("acct-123", "content-ops-abc123", "resend:email-123")
    assert "buyer@example.com" not in str(update_args)


@pytest.mark.asyncio
async def test_delivery_worker_renders_model_backed_email_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _delivery_report_model_artifact()
    pool = _Pool([_row(artifact=json.dumps(artifact))])
    sender = _Sender()
    _install_fake_pdf_renderer(monkeypatch, lambda _artifact: b"%PDF-model-bytes")

    summary = await send_pending_deflection_report_deliveries(
        pool,
        sender=sender,
        config=_config(),
    )

    assert summary.sent == 1
    request = sender.requests[0]
    assert "Key numbers" in request.html_body
    assert "1,234 repeat tickets across 8 ranked questions" in request.html_body
    assert "$16,659 estimated assisted-contact handling" in request.html_body
    assert "3 publishable answers drafted" in request.html_body
    assert "5 questions still need approved resolution evidence" in request.html_body
    assert "42 ticket sources represented" in request.html_body
    assert "Next actions" in request.html_body
    assert "How do I export attribution reports?" in request.html_body
    assert "9 repeat tickets" in request.html_body
    assert "$122 estimated handling" in request.html_body
    assert "Owner: Reporting" in request.html_body
    assert "Evidence: CSV customer text" in request.html_body
    assert "Repeated support friction routes to Reporting" in request.html_body
    assert "Customer vocabulary: export attribution reports, download attribution CSV" in request.html_body
    assert "Cost basis: this upload / benchmark cost with customer text" in request.html_body
    assert "Create a help-center answer" in request.html_body
    assert "How do I update invoice contacts?" in request.html_body
    assert "How do I invite an auditor?" not in request.html_body
    assert "Ready to publish" in request.html_body
    assert "How do I enable SSO?" in request.html_body
    assert request.html_body.count("How do I enable SSO?") == 1
    assert "How do I change workspace owners?" not in request.html_body
    assert "curated report PDF is attached" in request.html_body
    assert "full report PDF is attached" not in request.html_body
    assert "The secure results page has the consolidated report" in request.html_body
    assert "RAW MARKDOWN BODY" not in request.html_body
    assert "private evidence quote" not in request.html_body
    assert "ticket-secret-123" not in request.html_body
    assert "RAW_REPRESENTATIVE_PHRASE_SHOULD_STAY_OUT" not in request.html_body
    assert "ticket-secret-action-1" not in request.html_body
    assert "raw action evidence should stay out" not in request.html_body
    assert "ticket-secret-priority-draft" not in request.html_body
    assert "ticket-secret-draft-1" not in request.html_body
    assert "resolution_evidence" not in request.html_body
    assert request.text_body is not None
    assert "1,234 repeat tickets across 8 ranked questions" in request.text_body
    assert "$16,659 estimated assisted-contact handling" in request.text_body
    assert "Next actions" in request.text_body
    assert "How do I export attribution reports?" in request.text_body
    assert "Owner: Reporting" in request.text_body
    assert "Evidence: CSV customer text" in request.text_body
    assert "Repeated support friction routes to Reporting" in request.text_body
    assert "Customer vocabulary: export attribution reports, download attribution CSV" in request.text_body
    assert "Cost basis: this upload / benchmark cost with customer text" in request.text_body
    assert "How do I update invoice contacts?" in request.text_body
    assert "How do I invite an auditor?" not in request.text_body
    assert "Ready to publish" in request.text_body
    assert "How do I enable SSO?" in request.text_body
    assert request.text_body.count("How do I enable SSO?") == 1
    assert "How do I change workspace owners?" not in request.text_body
    assert "curated report PDF is attached" in request.text_body
    assert "RAW MARKDOWN BODY" not in request.text_body
    assert "private evidence quote" not in request.text_body
    assert "ticket-secret-123" not in request.text_body
    assert "ticket-secret-action-1" not in request.text_body
    assert "raw action evidence should stay out" not in request.text_body
    assert "ticket-secret-priority-draft" not in request.text_body

    observation = deflection_delivery_email_surface_observation(request.text_body)
    scorecard = build_deflection_full_report_qa_scorecard(
        artifact["report_model"],
        surface_observations={"email": observation},
    )
    bad_scorecard = build_deflection_full_report_qa_scorecard(
        artifact["report_model"],
        surface_observations={
            "email": {
                "displayed_rows": {
                    "priority_fix_queue": 3,
                    "drafted_resolutions": 1,
                },
            },
        },
    )

    assert observation == {
        "displayed_rows": {
            "priority_fix_queue": 2,
            "drafted_resolutions": 1,
        },
    }
    failed = {
        assertion["id"]
        for assertion in scorecard["assertions"]
        if not assertion["ok"]
    }
    bad_failed = {
        assertion["id"]
        for assertion in bad_scorecard["assertions"]
        if not assertion["ok"]
    }
    assert "surface.email.displayed_rows.priority_fix_queue" not in failed
    assert "surface.email.displayed_rows.drafted_resolutions" not in failed
    assert "surface.email.displayed_rows.priority_fix_queue" in bad_failed


@pytest.mark.asyncio
async def test_delivery_worker_omits_malformed_action_summary_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _delivery_report_model_artifact()
    sections = artifact["report_model"]["sections"]
    for section in sections:
        if section["id"] == "priority_fix_queue":
            section["data"]["items"] = [
                {"question": "Missing ticket count"},
                {"ticket_count": 3},
                {
                    "question": {
                        "source_ids": ["ticket-object-leak"],
                        "evidence_quote": "object evidence leak",
                    },
                    "ticket_count": 3,
                    "estimated_support_cost": 30.0,
                },
                "not-a-row",
            ]
        if section["id"] == "drafted_resolutions":
            section["data"]["items"] = "not-a-list"
    pool = _Pool([_row(artifact=json.dumps(artifact))])
    sender = _Sender()
    _install_fake_pdf_renderer(monkeypatch, lambda _artifact: b"%PDF-model-bytes")

    summary = await send_pending_deflection_report_deliveries(
        pool,
        sender=sender,
        config=_config(),
    )

    assert summary.sent == 1
    request = sender.requests[0]
    assert "Key numbers" in request.html_body
    assert "Next actions" not in request.html_body
    assert "Ready to publish" not in request.html_body
    assert "Missing ticket count" not in request.html_body
    assert "ticket-object-leak" not in request.html_body
    assert "object evidence leak" not in request.html_body
    assert "not-a-list" not in request.html_body
    assert request.text_body is not None
    assert "Next actions" not in request.text_body
    assert "Ready to publish" not in request.text_body
    assert "ticket-object-leak" not in request.text_body
    assert "object evidence leak" not in request.text_body


@pytest.mark.asyncio
async def test_delivery_worker_omits_non_scalar_action_optional_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _delivery_report_model_artifact()
    sections = artifact["report_model"]["sections"]
    for section in sections:
        if section["id"] == "priority_fix_queue":
            section["data"]["items"] = [
                {
                    "question": "How do I update audit exports?",
                    "ticket_count": 5,
                    "estimated_support_cost": 67.5,
                    "status": ["status evidence leak"],
                    "recommended_action": {
                        "evidence_quote": "action evidence leak",
                    },
                },
            ]
        if section["id"] == "drafted_resolutions":
            section["data"]["items"] = []
    pool = _Pool([_row(artifact=json.dumps(artifact))])
    sender = _Sender()
    _install_fake_pdf_renderer(monkeypatch, lambda _artifact: b"%PDF-model-bytes")

    summary = await send_pending_deflection_report_deliveries(
        pool,
        sender=sender,
        config=_config(),
    )

    assert summary.sent == 1
    request = sender.requests[0]
    assert "Next actions" in request.html_body
    assert "How do I update audit exports?" in request.html_body
    assert "status evidence leak" not in request.html_body
    assert "action evidence leak" not in request.html_body
    assert request.text_body is not None
    assert "How do I update audit exports?" in request.text_body
    assert "status evidence leak" not in request.text_body
    assert "action evidence leak" not in request.text_body


@pytest.mark.asyncio
async def test_delivery_worker_honors_zero_action_summary_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _delivery_report_model_artifact()
    sections = artifact["report_model"]["sections"]
    for section in sections:
        if section["id"] in {"priority_fix_queue", "drafted_resolutions"}:
            section["data"]["result_page_limit"] = 0
    pool = _Pool([_row(artifact=json.dumps(artifact))])
    sender = _Sender()
    _install_fake_pdf_renderer(monkeypatch, lambda _artifact: b"%PDF-model-bytes")

    summary = await send_pending_deflection_report_deliveries(
        pool,
        sender=sender,
        config=_config(),
    )

    assert summary.sent == 1
    request = sender.requests[0]
    assert "Key numbers" in request.html_body
    assert "Next actions" not in request.html_body
    assert "Ready to publish" not in request.html_body
    assert "How do I export attribution reports?" not in request.html_body
    assert "How do I enable SSO?" not in request.html_body
    assert request.text_body is not None
    assert "Next actions" not in request.text_body
    assert "Ready to publish" not in request.text_body


@pytest.mark.asyncio
async def test_delivery_worker_falls_back_for_future_report_model_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _Pool([
        _row(
            artifact=json.dumps(
                _delivery_report_model_artifact(schema_version="deflection.v2")
            )
        )
    ])
    sender = _Sender()
    _install_fake_pdf_renderer(monkeypatch, lambda _artifact: b"%PDF-model-bytes")

    summary = await send_pending_deflection_report_deliveries(
        pool,
        sender=sender,
        config=_config(),
    )

    assert summary.sent == 1
    request = sender.requests[0]
    assert "Key numbers" not in request.html_body
    assert "full report PDF is attached" in request.html_body
    assert "secure results page remains the system of record" in request.html_body


@pytest.mark.asyncio
async def test_delivery_worker_falls_back_for_malformed_email_summary_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _Pool([
        _row(
            artifact=json.dumps(
                _delivery_report_model_artifact(
                    support_tax_overrides={"repeat_ticket_count": "not-a-number"}
                )
            )
        )
    ])
    sender = _Sender()
    _install_fake_pdf_renderer(monkeypatch, lambda _artifact: b"%PDF-model-bytes")

    summary = await send_pending_deflection_report_deliveries(
        pool,
        sender=sender,
        config=_config(),
    )

    assert summary.sent == 1
    request = sender.requests[0]
    assert "Key numbers" not in request.html_body
    assert "0 repeat tickets" not in request.html_body
    assert "full report PDF is attached" in request.html_body


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
async def test_delivery_worker_fails_when_paid_pdf_render_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pool = _Pool([_row()])
    sender = _Sender()
    caplog.set_level("ERROR", logger="atlas.content_ops_deflection_delivery")

    def _raise_pdf(_artifact: Any) -> bytes:
        raise RuntimeError("pdf down")

    _install_fake_pdf_renderer(monkeypatch, _raise_pdf)

    summary = await send_pending_deflection_report_deliveries(
        pool,
        sender=sender,
        config=_config(),
    )

    assert summary.scanned == 1
    assert summary.sent == 0
    assert summary.failed == 1
    assert sender.requests == []
    update_query, update_args = pool.execute_calls[0]
    assert "delivery_status = 'failed'" in update_query
    assert update_args == (
        "acct-123",
        "content-ops-abc123",
        "PaidReportPdfRenderError: paid_report_pdf_render_failed: RuntimeError: pdf down",
    )
    incidents = _incident_payloads(caplog)
    assert incidents[-1]["incident_type"] == "paid_report_delivery_send_failed"
    assert incidents[-1]["error"] == update_args[-1]
    assert "Deflection report PDF render failed" in caplog.text


@pytest.mark.asyncio
async def test_delivery_worker_preserves_stale_reclaim_when_paid_pdf_render_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pool = _Pool([_row(previous_delivery_status="sending")])
    sender = _Sender()
    caplog.set_level("WARNING", logger="atlas.content_ops_deflection_delivery")

    def _raise_pdf(_artifact: Any) -> bytes:
        raise RuntimeError("pdf down")

    _install_fake_pdf_renderer(monkeypatch, _raise_pdf)

    summary = await send_pending_deflection_report_deliveries(
        pool,
        sender=sender,
        config=_config(),
    )

    assert summary.scanned == 1
    assert summary.sent == 0
    assert summary.failed == 1
    assert sender.requests == []
    update_query, update_args = pool.execute_calls[0]
    assert "delivery_status = 'sending'" in update_query
    assert "delivery_status = 'pending'" not in update_query
    assert "delivery_status = 'failed'" not in update_query
    assert update_args == (
        "acct-123",
        "content-ops-abc123",
        "PaidReportPdfRenderError: paid_report_pdf_render_failed: RuntimeError: pdf down",
    )
    incidents = _incident_payloads(caplog)
    assert incidents[-1]["incident_type"] == (
        "paid_report_delivery_pdf_render_reclaim_deferred"
    )
    assert incidents[-1]["severity"] == "warning"
    assert incidents[-1]["error"] == update_args[-1]


@pytest.mark.asyncio
async def test_delivery_worker_repeated_stale_render_outage_remains_reclaim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _Pool([
        _row(
            previous_delivery_status="sending",
            delivery_error="PaidReportPdfRenderError: prior render outage",
        )
    ])
    sender = _Sender()

    def _raise_pdf(_artifact: Any) -> bytes:
        raise RuntimeError("pdf still down")

    _install_fake_pdf_renderer(monkeypatch, _raise_pdf)

    summary = await send_pending_deflection_report_deliveries(
        pool,
        sender=sender,
        config=_config(),
    )

    assert summary.sent == 0
    assert summary.failed == 1
    assert sender.requests == []
    update_query, update_args = pool.execute_calls[0]
    assert "delivery_status = 'sending'" in update_query
    assert "delivery_status = 'failed'" not in update_query
    assert update_args == (
        "acct-123",
        "content-ops-abc123",
        "PaidReportPdfRenderError: paid_report_pdf_render_failed: RuntimeError: pdf still down",
    )


@pytest.mark.asyncio
async def test_delivery_worker_fails_when_paid_pdf_artifact_is_malformed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pool = _Pool([_row(artifact=json.dumps(["not", "a", "report"]))])
    sender = _Sender()
    caplog.set_level("ERROR", logger="atlas.content_ops_deflection_delivery")

    summary = await send_pending_deflection_report_deliveries(
        pool,
        sender=sender,
        config=_config(),
    )

    assert summary.scanned == 1
    assert summary.sent == 0
    assert summary.failed == 1
    assert sender.requests == []
    update_query, update_args = pool.execute_calls[0]
    assert "delivery_status = 'failed'" in update_query
    assert update_args == (
        "acct-123",
        "content-ops-abc123",
        "ValueError: paid_report_pdf_missing_artifact",
    )
    incidents = _incident_payloads(caplog)
    assert incidents[-1]["incident_type"] == "paid_report_delivery_send_failed"
    assert incidents[-1]["error"] == update_args[-1]


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
    assert "d.delivery_status AS previous_delivery_status" in claim_query
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
    # real render->attachment transition): both attempts carry a PDF, but a
    # regenerated attachment can still differ byte-for-byte under the SAME
    # idempotency key. A Resend-like client returns 409 invalid_idempotent_request
    # for that mismatch, and the delivery path must mark the row delivered
    # (idempotent replay), not failed.
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
        return b"%PDF-second-render"

    _install_fake_pdf_renderer(monkeypatch, _renderer)
    row = _row()

    first = await send_pending_deflection_report_deliveries(
        _Pool([row]), sender=sender, config=_config()
    )
    assert first.sent == 1 and first.failed == 0

    # Re-claim: render returns a different PDF payload under the same key.
    second = await send_pending_deflection_report_deliveries(
        _Pool([row]), sender=sender, config=_config()
    )
    assert second.sent == 1  # delivered via idempotent replay, NOT failed
    assert second.failed == 0
    assert len(http.posts) == 2
    assert http.posts[0]["key"] == http.posts[1]["key"]  # same idempotency key
    assert http.posts[0]["json"] != http.posts[1]["json"]  # regenerated attachment


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
