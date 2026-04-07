from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import atlas_brain.autonomous.tasks.b2b_report_subscription_delivery as mod


class _DuePool:
    def __init__(self, row):
        self.is_initialized = True
        self._row = row

    async def fetch(self, *_args, **_kwargs):
        return [self._row]


@pytest.mark.asyncio
async def test_send_subscription_email_retries_and_wraps_footer(monkeypatch):
    calls: list[dict] = []
    sleep_calls: list[int] = []

    class Sender:
        async def send(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("temporary failure")
            return {"id": "msg-1"}

    async def _fake_sleep(seconds: int):
        sleep_calls.append(seconds)

    monkeypatch.setattr(mod, "is_suppressed", AsyncMock(return_value=None))
    monkeypatch.setattr(mod.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(
        mod.settings.campaign_sequence,
        "unsubscribe_base_url",
        "https://example.com/unsubscribe",
        raising=False,
    )
    monkeypatch.setattr(
        mod.settings.campaign_sequence,
        "company_address",
        "123 Example St",
        raising=False,
    )
    monkeypatch.setattr(
        mod.settings.b2b_report_delivery,
        "max_send_attempts_per_recipient",
        2,
        raising=False,
    )
    monkeypatch.setattr(
        mod.settings.b2b_report_delivery,
        "retry_backoff_seconds",
        3,
        raising=False,
    )

    sent, message_id, error, suppressed = await mod._send_subscription_email(
        None,
        Sender(),
        recipient="buyer@example.com",
        from_addr="Atlas <atlas@example.com>",
        subject="Recurring delivery",
        html_body="<p>Hello</p>",
        tags=[{"name": "task", "value": "b2b_report_subscription_delivery"}],
    )

    assert sent is True
    assert message_id == "msg-1"
    assert error is None
    assert suppressed is False
    assert sleep_calls == [3]
    assert len(calls) == 2
    assert "List-Unsubscribe" in (calls[0]["headers"] or {})
    assert "Unsubscribe" in calls[0]["body"]


@pytest.mark.asyncio
async def test_run_skips_unchanged_delivery_without_sending(monkeypatch):
    scheduled_for = datetime.now(timezone.utc) - timedelta(minutes=5)
    pool = _DuePool(
        {
            "id": "sub-1",
            "account_id": "acct-1",
            "account_name": "Smoke Co",
            "report_id": "report-1",
            "scope_type": "report",
            "scope_key": "report-1",
            "scope_label": "Smoke recurring report",
            "delivery_frequency": "weekly",
            "deliverable_focus": "all",
            "freshness_policy": "any",
            "recipient_emails": ["buyer@example.com"],
            "delivery_note": "note",
            "next_delivery_at": scheduled_for,
        }
    )
    row = {
        "id": "sub-1",
        "account_id": "acct-1",
        "account_name": "Smoke Co",
        "report_id": "report-1",
        "scope_type": "report",
        "scope_key": "report-1",
        "scope_label": "Smoke recurring report",
        "delivery_frequency": "weekly",
        "deliverable_focus": "all",
        "freshness_policy": "any",
        "recipient_emails": ["buyer@example.com"],
        "delivery_note": "note",
        "next_delivery_at": scheduled_for,
    }
    artifact = {
        "report_id": "report-1",
        "report_type": "vendor_scorecard",
        "title": "Smoke recurring report",
        "trust_label": "Evidence-backed",
        "quality_status": "sales_ready",
        "blocker_count": 0,
        "unresolved_issue_count": 0,
        "freshness_state": "fresh",
        "executive_summary": "summary",
        "evidence_highlights": ["highlight"],
    }
    finalize = AsyncMock()
    advance = AsyncMock()
    sender = SimpleNamespace(send=AsyncMock(side_effect=AssertionError("sender should not be called")))

    monkeypatch.setattr(mod.settings.b2b_report_delivery, "enabled", True, raising=False)
    monkeypatch.setattr(mod.settings.b2b_report_delivery, "max_subscriptions_per_run", 1, raising=False)
    monkeypatch.setattr(mod.settings.b2b_report_delivery, "suppress_unchanged_deliveries", True, raising=False)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "get_campaign_sender", lambda: sender)
    monkeypatch.setattr(mod, "_claim_delivery_attempt", AsyncMock(return_value={"id": "log-1"}))
    monkeypatch.setattr(mod, "_tracked_vendors", AsyncMock(return_value=set()))
    monkeypatch.setattr(mod, "_resolve_artifacts", AsyncMock(return_value=[artifact]))
    monkeypatch.setattr(mod, "_latest_delivery_content_hash", AsyncMock(return_value="same-hash"))
    monkeypatch.setattr(mod, "_delivery_content_hash", lambda *_args, **_kwargs: "same-hash")
    monkeypatch.setattr(mod, "_finalize_delivery_attempt", finalize)
    monkeypatch.setattr(mod, "_advance_subscription", advance)

    result = await mod.run(SimpleNamespace())

    assert result["delivered"] == 0
    assert result["skipped"] == 1
    assert result["failed"] == 0
    assert finalize.await_args.kwargs["content_hash"] == "same-hash"
    assert "has not materially changed" in finalize.await_args.kwargs["summary"]
    assert advance.await_args.args == (
        pool,
        "sub-1",
        "weekly",
        scheduled_for,
    )


@pytest.mark.asyncio
async def test_run_sends_and_advances_subscription_with_anchor(monkeypatch):
    scheduled_for = datetime.now(timezone.utc) - timedelta(minutes=5)
    row = {
        "id": "sub-2",
        "account_id": "acct-2",
        "account_name": "Smoke Co",
        "report_id": "report-2",
        "scope_type": "report",
        "scope_key": "report-2",
        "scope_label": "Smoke recurring report",
        "delivery_frequency": "weekly",
        "deliverable_focus": "all",
        "freshness_policy": "any",
        "recipient_emails": ["buyer@example.com"],
        "delivery_note": "note",
        "next_delivery_at": scheduled_for,
    }
    artifact = {
        "report_id": "report-2",
        "report_type": "vendor_scorecard",
        "title": "Smoke recurring report",
        "trust_label": "Evidence-backed",
        "quality_status": "sales_ready",
        "blocker_count": 0,
        "unresolved_issue_count": 0,
        "freshness_state": "fresh",
        "executive_summary": "summary",
        "evidence_highlights": ["highlight"],
    }
    pool = _DuePool(row)
    finalize = AsyncMock()
    advance = AsyncMock()
    send_helper = AsyncMock(return_value=(True, "msg-1", None, False))

    monkeypatch.setattr(mod.settings.b2b_report_delivery, "enabled", True, raising=False)
    monkeypatch.setattr(mod.settings.b2b_report_delivery, "max_subscriptions_per_run", 1, raising=False)
    monkeypatch.setattr(mod.settings.b2b_report_delivery, "suppress_unchanged_deliveries", True, raising=False)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "get_campaign_sender", lambda: SimpleNamespace())
    monkeypatch.setattr(mod, "_claim_delivery_attempt", AsyncMock(return_value={"id": "log-2"}))
    monkeypatch.setattr(mod, "_tracked_vendors", AsyncMock(return_value=set()))
    monkeypatch.setattr(mod, "_resolve_artifacts", AsyncMock(return_value=[artifact]))
    monkeypatch.setattr(mod, "_latest_delivery_content_hash", AsyncMock(return_value=None))
    monkeypatch.setattr(mod, "_delivery_content_hash", lambda *_args, **_kwargs: "new-hash")
    monkeypatch.setattr(mod, "_send_subscription_email", send_helper)
    monkeypatch.setattr(mod, "_finalize_delivery_attempt", finalize)
    monkeypatch.setattr(mod, "_advance_subscription", advance)

    result = await mod.run(SimpleNamespace())

    assert result["delivered"] == 1
    assert result["skipped"] == 0
    assert result["failed"] == 0
    assert send_helper.await_count == 1
    assert finalize.await_args.kwargs["content_hash"] == "new-hash"
    assert finalize.await_args.kwargs["status"] == "sent"
    assert advance.await_args.args == (
        pool,
        "sub-2",
        "weekly",
        scheduled_for,
    )


@pytest.mark.asyncio
async def test_latest_delivery_content_hash_ignores_partial_and_skipped(monkeypatch):
    seen_args: list[tuple[object, ...]] = []

    class Pool:
        async def fetchval(self, query, *args):
            seen_args.append(args)
            assert "status = ANY($2::text[])" in query
            return "hash-1"

    result = await mod._latest_delivery_content_hash(Pool(), "sub-1")

    assert result == "hash-1"
    assert seen_args == [("sub-1", ["sent"])]


@pytest.mark.asyncio
async def test_run_does_not_skip_unchanged_after_partial_delivery(monkeypatch):
    scheduled_for = datetime.now(timezone.utc) - timedelta(minutes=5)
    row = {
        "id": "sub-3",
        "account_id": "acct-3",
        "account_name": "Smoke Co",
        "report_id": "report-3",
        "scope_type": "report",
        "scope_key": "report-3",
        "scope_label": "Smoke recurring report",
        "delivery_frequency": "weekly",
        "deliverable_focus": "all",
        "freshness_policy": "any",
        "recipient_emails": ["buyer@example.com"],
        "delivery_note": "note",
        "next_delivery_at": scheduled_for,
    }
    artifact = {
        "report_id": "report-3",
        "report_type": "vendor_scorecard",
        "title": "Smoke recurring report",
        "trust_label": "Evidence-backed",
        "quality_status": "sales_ready",
        "blocker_count": 0,
        "unresolved_issue_count": 0,
        "freshness_state": "fresh",
        "executive_summary": "summary",
        "evidence_highlights": ["highlight"],
    }
    pool = _DuePool(row)
    finalize = AsyncMock()
    advance = AsyncMock()
    send_helper = AsyncMock(return_value=(True, "msg-3", None, False))

    monkeypatch.setattr(mod.settings.b2b_report_delivery, "enabled", True, raising=False)
    monkeypatch.setattr(mod.settings.b2b_report_delivery, "max_subscriptions_per_run", 1, raising=False)
    monkeypatch.setattr(mod.settings.b2b_report_delivery, "suppress_unchanged_deliveries", True, raising=False)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "get_campaign_sender", lambda: SimpleNamespace())
    monkeypatch.setattr(mod, "_claim_delivery_attempt", AsyncMock(return_value={"id": "log-3"}))
    monkeypatch.setattr(mod, "_tracked_vendors", AsyncMock(return_value=set()))
    monkeypatch.setattr(mod, "_resolve_artifacts", AsyncMock(return_value=[artifact]))
    monkeypatch.setattr(mod, "_latest_delivery_content_hash", AsyncMock(return_value=None))
    monkeypatch.setattr(mod, "_send_subscription_email", send_helper)
    monkeypatch.setattr(mod, "_finalize_delivery_attempt", finalize)
    monkeypatch.setattr(mod, "_advance_subscription", advance)

    result = await mod.run(SimpleNamespace())

    assert result["delivered"] == 1
    assert result["skipped"] == 0
    send_helper.assert_awaited_once()
    assert finalize.await_args.kwargs["status"] == "sent"
