from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import atlas_brain.autonomous.tasks.b2b_report_subscription_delivery as mod


class _DuePool:
    def __init__(self, row):
        self.is_initialized = True
        self._rows = row if isinstance(row, list) else [row]

    async def fetch(self, *_args, **_kwargs):
        return self._rows


def _selection(
    artifacts,
    *,
    eligible_before_freshness_count: int | None = None,
    freshness_blocked_count: int = 0,
):
    return mod._ArtifactSelectionResult(
        artifacts=list(artifacts),
        eligible_before_freshness_count=(
            len(artifacts) if eligible_before_freshness_count is None else eligible_before_freshness_count
        ),
        freshness_blocked_count=freshness_blocked_count,
    )


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
async def test_send_subscription_email_skips_suppressed_recipient(monkeypatch):
    sender = SimpleNamespace(send=AsyncMock(side_effect=AssertionError("sender should not be called")))

    monkeypatch.setattr(
        mod,
        "is_suppressed",
        AsyncMock(return_value={"reason": "unsubscribe"}),
    )

    sent, message_id, error, suppressed = await mod._send_subscription_email(
        None,
        sender,
        recipient="buyer@example.com",
        from_addr="Atlas <atlas@example.com>",
        subject="Recurring delivery",
        html_body="<p>Hello</p>",
        tags=[{"name": "task", "value": "b2b_report_subscription_delivery"}],
    )

    assert sent is False
    assert message_id is None
    assert suppressed is True
    assert "suppressed" in (error or "")


def test_delivery_eligibility_reason_blocks_competitive_quality_and_reviews(monkeypatch):
    monkeypatch.setattr(
        mod.settings.b2b_report_delivery,
        "require_sales_ready_for_competitive",
        True,
        raising=False,
    )
    monkeypatch.setattr(mod.settings.b2b_report_delivery, "max_blocker_count", 0, raising=False)
    monkeypatch.setattr(mod.settings.b2b_report_delivery, "max_open_review_count", 0, raising=False)

    quality_reason = mod._delivery_eligibility_reason(
        {
            "report_type": "battle_card",
            "quality_status": "needs_review",
            "blocker_count": 0,
            "unresolved_issue_count": 0,
        },
        {},
    )
    blocker_reason = mod._delivery_eligibility_reason(
        {
            "report_type": "vendor_scorecard",
            "quality_status": None,
            "blocker_count": 1,
            "unresolved_issue_count": 0,
        },
        {},
    )
    review_reason = mod._delivery_eligibility_reason(
        {
            "report_type": "vendor_scorecard",
            "quality_status": None,
            "blocker_count": 0,
            "unresolved_issue_count": 2,
        },
        {},
    )

    assert quality_reason == "quality_status=needs_review"
    assert blocker_reason == "blocker_count=1 exceeds max_blocker_count=0"
    assert review_reason == "unresolved_issue_count=2 exceeds max_open_review_count=0"


def test_artifact_ready_accepts_published_and_sales_ready():
    assert mod._artifact_ready(
        {"report_type": "vendor_scorecard", "status": "published", "quality_status": None},
        {},
    ) is True
    assert mod._artifact_ready(
        {"report_type": "battle_card", "status": "sales_ready", "quality_status": None},
        {},
    ) is True
    assert mod._artifact_ready(
        {"report_type": "vendor_scorecard", "status": "failed", "quality_status": None},
        {},
    ) is False


def test_build_delivery_artifact_includes_normalized_trust_fields():
    row = {
        "id": "report-1",
        "report_type": "battle_card",
        "vendor_filter": "Zendesk",
        "category_filter": None,
        "status": "sales_ready",
        "quality_status": "sales_ready",
        "blocker_count": 0,
        "warning_count": 1,
        "unresolved_issue_count": 0,
        "report_date": datetime.now(timezone.utc) - timedelta(hours=8),
        "created_at": datetime.now(timezone.utc) - timedelta(hours=8),
        "executive_summary": "Summary",
        "intelligence_data": {},
        "data_density": {},
    }

    artifact = mod._build_delivery_artifact(row)

    assert artifact["artifact_state"] == "ready"
    assert artifact["artifact_label"] == "Ready"
    assert artifact["review_state"] == "warnings"
    assert artifact["review_label"] == "Warnings"
    assert artifact["freshness_state"] == "fresh"
    assert artifact["trust"]["review_state"] == "warnings"
    assert artifact["trust"]["artifact_state"] == "ready"


@pytest.mark.asyncio
async def test_resolve_artifacts_skips_overridden_library_reports(monkeypatch):
    monkeypatch.setattr(
        mod.settings.b2b_report_delivery,
        "report_scope_overrides_library",
        True,
        raising=False,
    )
    monkeypatch.setattr(mod, "_fetch_library_rows", AsyncMock(return_value=[
        {
            "id": "report-1",
            "report_type": "vendor_scorecard",
            "status": "published",
            "quality_status": None,
            "intelligence_data": {},
        },
        {
            "id": "report-2",
            "report_type": "vendor_scorecard",
            "status": "published",
            "quality_status": None,
            "intelligence_data": {},
        },
    ]))
    monkeypatch.setattr(mod, "_overridden_report_ids", AsyncMock(return_value={"report-1"}))
    monkeypatch.setattr(mod, "_delivery_eligibility_reason", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        mod,
        "_build_delivery_artifact",
        lambda report_row: {"report_id": report_row["id"], "freshness_state": "fresh"},
    )

    artifacts = await mod._resolve_artifacts(
        None,
        {
            "account_id": "acct-1",
            "scope_type": "library",
            "deliverable_focus": "all",
            "freshness_policy": "any",
        },
        set(),
    )

    assert [artifact["report_id"] for artifact in artifacts] == ["report-2"]


@pytest.mark.asyncio
async def test_resolve_artifacts_applies_library_view_filters(monkeypatch):
    monkeypatch.setattr(mod, "_fetch_library_rows", AsyncMock(return_value=[
        {
            "id": "report-1",
            "report_type": "battle_card",
            "vendor_filter": "Zendesk",
            "status": "sales_ready",
            "quality_status": "sales_ready",
            "intelligence_data": {},
            "data_density": {},
            "created_at": datetime.now(timezone.utc) - timedelta(days=10),
            "report_date": datetime.now(timezone.utc) - timedelta(days=10),
            "blocker_count": 1,
            "warning_count": 0,
            "unresolved_issue_count": 0,
        },
        {
            "id": "report-2",
            "report_type": "battle_card",
            "vendor_filter": "Intercom",
            "status": "sales_ready",
            "quality_status": "sales_ready",
            "intelligence_data": {},
            "data_density": {},
            "created_at": datetime.now(timezone.utc) - timedelta(hours=6),
            "report_date": datetime.now(timezone.utc) - timedelta(hours=6),
            "blocker_count": 0,
            "warning_count": 0,
            "unresolved_issue_count": 0,
        },
    ]))
    monkeypatch.setattr(
        mod.settings.b2b_report_delivery,
        "report_scope_overrides_library",
        False,
        raising=False,
    )
    monkeypatch.setattr(mod, "_delivery_eligibility_reason", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        mod,
        "_build_delivery_artifact",
        lambda report_row: {"report_id": report_row["id"], "freshness_state": "fresh"},
    )

    artifacts = await mod._resolve_artifacts(
        None,
        {
            "account_id": "acct-1",
            "scope_type": "library_view",
            "scope_key": "library-view--type-battle_card--vendor-zendesk--quality-sales_ready--freshness-stale--review-blocked",
            "filter_payload": {
                "report_type": "battle_card",
                "vendor_filter": "Zendesk",
                "quality_status": "sales_ready",
                "freshness_state": "stale",
                "review_state": "blocked",
            },
            "deliverable_focus": "all",
            "freshness_policy": "any",
        },
        {"zendesk", "intercom"},
    )

    assert [artifact["report_id"] for artifact in artifacts] == ["report-1"]


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
    monkeypatch.setattr(mod, "_resolve_artifact_selection", AsyncMock(return_value=_selection([artifact])))
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
async def test_run_does_not_advance_when_no_artifacts_match_policy(monkeypatch):
    scheduled_for = datetime.now(timezone.utc) - timedelta(minutes=5)
    row = {
        "id": "sub-no-artifacts",
        "account_id": "acct-no-artifacts",
        "account_name": "Smoke Co",
        "report_id": "report-no-artifacts",
        "scope_type": "library",
        "scope_key": "library",
        "scope_label": "Smoke recurring library",
        "delivery_frequency": "weekly",
        "deliverable_focus": "all",
        "freshness_policy": "any",
        "recipient_emails": ["buyer@example.com"],
        "delivery_note": "note",
        "next_delivery_at": scheduled_for,
    }
    pool = _DuePool(row)
    finalize = AsyncMock()
    advance = AsyncMock()

    monkeypatch.setattr(mod.settings.b2b_report_delivery, "enabled", True, raising=False)
    monkeypatch.setattr(mod.settings.b2b_report_delivery, "max_subscriptions_per_run", 1, raising=False)
    monkeypatch.setattr(mod.settings.b2b_report_delivery, "suppress_unchanged_deliveries", True, raising=False)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "get_campaign_sender", lambda: SimpleNamespace())
    monkeypatch.setattr(mod, "_claim_delivery_attempt", AsyncMock(return_value={"id": "log-no-artifacts"}))
    monkeypatch.setattr(mod, "_tracked_vendors", AsyncMock(return_value=set()))
    monkeypatch.setattr(mod, "_resolve_artifact_selection", AsyncMock(return_value=_selection([])))
    monkeypatch.setattr(mod, "_finalize_delivery_attempt", finalize)
    monkeypatch.setattr(mod, "_advance_subscription", advance)

    result = await mod.run(SimpleNamespace())

    assert result["skipped"] == 1
    assert finalize.await_args.kwargs["status"] == "skipped"
    assert "no eligible persisted artifacts matched the saved policy" in finalize.await_args.kwargs["summary"]
    advance.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_marks_core_incomplete_freshness_skip_as_blocked(monkeypatch):
    scheduled_for = datetime.now(timezone.utc) - timedelta(minutes=5)
    row = {
        "id": "sub-core-blocked",
        "account_id": "acct-core-blocked",
        "account_name": "Blocked Co",
        "report_id": "report-core-blocked",
        "scope_type": "library",
        "scope_key": "library",
        "scope_label": "Blocked recurring library",
        "delivery_frequency": "weekly",
        "deliverable_focus": "all",
        "freshness_policy": "fresh_only",
        "recipient_emails": ["buyer@example.com"],
        "delivery_note": "note",
        "next_delivery_at": scheduled_for,
    }
    pool = _DuePool(row)
    finalize = AsyncMock()
    advance = AsyncMock()

    monkeypatch.setattr(mod.settings.b2b_report_delivery, "enabled", True, raising=False)
    monkeypatch.setattr(mod.settings.b2b_report_delivery, "max_subscriptions_per_run", 1, raising=False)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "get_campaign_sender", lambda: SimpleNamespace())
    monkeypatch.setattr(mod, "_claim_delivery_attempt", AsyncMock(return_value={"id": "log-core-blocked"}))
    monkeypatch.setattr(mod, "_tracked_vendors", AsyncMock(return_value=set()))
    monkeypatch.setattr(
        mod,
        "_resolve_artifact_selection",
        AsyncMock(
            return_value=_selection(
                [],
                eligible_before_freshness_count=2,
                freshness_blocked_count=2,
            )
        ),
    )
    monkeypatch.setattr(mod, "has_complete_core_run_marker", AsyncMock(return_value=False))
    monkeypatch.setattr(mod, "_finalize_delivery_attempt", finalize)
    monkeypatch.setattr(mod, "_advance_subscription", advance)

    result = await mod.run(SimpleNamespace())

    assert result["skipped"] == 1
    assert finalize.await_args.kwargs["status"] == "skipped"
    assert finalize.await_args.kwargs["freshness_state"] == "blocked"
    assert "core churn materialization is incomplete" in finalize.await_args.kwargs["summary"]
    advance.assert_not_awaited()


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
    monkeypatch.setattr(mod, "_resolve_artifact_selection", AsyncMock(return_value=_selection([artifact])))
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
async def test_run_dry_run_skips_send_and_does_not_advance(monkeypatch):
    scheduled_for = datetime.now(timezone.utc) - timedelta(minutes=5)
    row = {
        "id": "sub-dry",
        "account_id": "acct-dry",
        "account_name": "Dry Run Co",
        "report_id": "report-dry",
        "scope_type": "report",
        "scope_key": "report-dry",
        "scope_label": "Dry recurring report",
        "delivery_frequency": "weekly",
        "deliverable_focus": "all",
        "freshness_policy": "any",
        "recipient_emails": ["buyer@example.com"],
        "delivery_note": "note",
        "next_delivery_at": scheduled_for,
    }
    artifact = {
        "report_id": "report-dry",
        "report_type": "vendor_scorecard",
        "title": "Dry recurring report",
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
    send_helper = AsyncMock(side_effect=AssertionError("send helper should not be called in dry run"))

    monkeypatch.setattr(mod.settings.b2b_report_delivery, "enabled", True, raising=False)
    monkeypatch.setattr(mod.settings.b2b_report_delivery, "dry_run", False, raising=False)
    monkeypatch.setattr(mod.settings.b2b_report_delivery, "max_subscriptions_per_run", 1, raising=False)
    monkeypatch.setattr(mod.settings.b2b_report_delivery, "suppress_unchanged_deliveries", True, raising=False)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(
        mod,
        "get_campaign_sender",
        lambda: (_ for _ in ()).throw(AssertionError("campaign sender should not be created in dry run")),
    )
    monkeypatch.setattr(mod, "_count_due_non_canary_subscriptions", AsyncMock(return_value=0))
    monkeypatch.setattr(mod, "_fetch_due_rows", AsyncMock(return_value=[row]))
    monkeypatch.setattr(mod, "_claim_delivery_attempt", AsyncMock(return_value={"id": "log-dry"}))
    monkeypatch.setattr(mod, "_tracked_vendors", AsyncMock(return_value=set()))
    monkeypatch.setattr(mod, "_resolve_artifact_selection", AsyncMock(return_value=_selection([artifact])))
    monkeypatch.setattr(mod, "_latest_delivery_content_hash", AsyncMock(return_value=None))
    monkeypatch.setattr(mod, "_delivery_content_hash", lambda *_args, **_kwargs: "dry-hash")
    monkeypatch.setattr(mod, "_send_subscription_email", send_helper)
    monkeypatch.setattr(mod, "_finalize_delivery_attempt", finalize)
    monkeypatch.setattr(mod, "_advance_subscription", advance)

    result = await mod.run(SimpleNamespace(metadata={"dry_run": True}))

    assert result["dry_run"] is True
    assert result["delivered"] == 0
    assert result["dry_run_deliveries"] == 1
    assert result["skipped"] == 0
    assert finalize.await_args.kwargs["status"] == "dry_run"
    assert "Dry run: would deliver 1 artifact(s) to 1 recipient(s)." == finalize.await_args.kwargs["summary"]
    advance.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_advances_when_all_recipients_are_suppressed(monkeypatch):
    scheduled_for = datetime.now(timezone.utc) - timedelta(minutes=5)
    row = {
        "id": "sub-suppressed",
        "account_id": "acct-suppressed",
        "account_name": "Suppressed Co",
        "report_id": "report-suppressed",
        "scope_type": "report",
        "scope_key": "report-suppressed",
        "scope_label": "Suppressed recurring report",
        "delivery_frequency": "weekly",
        "deliverable_focus": "all",
        "freshness_policy": "any",
        "recipient_emails": ["buyer@example.com"],
        "delivery_note": "note",
        "next_delivery_at": scheduled_for,
    }
    artifact = {
        "report_id": "report-suppressed",
        "report_type": "vendor_scorecard",
        "title": "Suppressed recurring report",
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
    send_helper = AsyncMock(return_value=(False, None, "buyer@example.com: suppressed (manual)", True))

    monkeypatch.setattr(mod.settings.b2b_report_delivery, "enabled", True, raising=False)
    monkeypatch.setattr(mod.settings.b2b_report_delivery, "max_subscriptions_per_run", 1, raising=False)
    monkeypatch.setattr(mod.settings.b2b_report_delivery, "suppress_unchanged_deliveries", False, raising=False)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "get_campaign_sender", lambda: SimpleNamespace())
    monkeypatch.setattr(mod, "_claim_delivery_attempt", AsyncMock(return_value={"id": "log-suppressed"}))
    monkeypatch.setattr(mod, "_tracked_vendors", AsyncMock(return_value=set()))
    monkeypatch.setattr(mod, "_resolve_artifact_selection", AsyncMock(return_value=_selection([artifact])))
    monkeypatch.setattr(mod, "_delivery_content_hash", lambda *_args, **_kwargs: "suppressed-hash")
    monkeypatch.setattr(mod, "_send_subscription_email", send_helper)
    monkeypatch.setattr(mod, "_finalize_delivery_attempt", finalize)
    monkeypatch.setattr(mod, "_advance_subscription", advance)

    result = await mod.run(SimpleNamespace())

    assert result["skipped"] == 1
    assert finalize.await_args.kwargs["status"] == "skipped"
    assert "every recipient on the subscription is currently suppressed" in finalize.await_args.kwargs["summary"]
    assert advance.await_args.args == (
        pool,
        "sub-suppressed",
        "weekly",
        scheduled_for,
    )


@pytest.mark.asyncio
async def test_run_canary_scope_defers_non_canary_accounts(monkeypatch):
    scheduled_for = datetime.now(timezone.utc) - timedelta(minutes=5)
    due_rows = [
        {
            "id": "sub-live",
            "account_id": "acct-live",
            "account_name": "Canary Co",
            "report_id": "report-live",
            "scope_type": "report",
            "scope_key": "report-live",
            "scope_label": "Canary recurring report",
            "delivery_frequency": "weekly",
            "deliverable_focus": "all",
            "freshness_policy": "any",
            "recipient_emails": ["buyer@example.com"],
            "delivery_note": "note",
            "next_delivery_at": scheduled_for,
        },
        {
            "id": "sub-deferred",
            "account_id": "acct-deferred",
            "account_name": "Deferred Co",
            "report_id": "report-deferred",
            "scope_type": "report",
            "scope_key": "report-deferred",
            "scope_label": "Deferred recurring report",
            "delivery_frequency": "weekly",
            "deliverable_focus": "all",
            "freshness_policy": "any",
            "recipient_emails": ["buyer@example.com"],
            "delivery_note": "note",
            "next_delivery_at": scheduled_for,
        },
    ]
    artifact = {
        "report_id": "report-live",
        "report_type": "vendor_scorecard",
        "title": "Canary recurring report",
        "trust_label": "Evidence-backed",
        "quality_status": "sales_ready",
        "blocker_count": 0,
        "unresolved_issue_count": 0,
        "freshness_state": "fresh",
        "executive_summary": "summary",
        "evidence_highlights": ["highlight"],
    }
    pool = _DuePool(due_rows)
    claim = AsyncMock(return_value={"id": "log-live"})
    finalize = AsyncMock()
    advance = AsyncMock()
    send_helper = AsyncMock(return_value=(True, "msg-live", None, False))

    monkeypatch.setattr(mod.settings.b2b_report_delivery, "enabled", True, raising=False)
    monkeypatch.setattr(mod.settings.b2b_report_delivery, "dry_run", False, raising=False)
    monkeypatch.setattr(mod.settings.b2b_report_delivery, "canary_account_ids", "", raising=False)
    monkeypatch.setattr(mod.settings.b2b_report_delivery, "max_subscriptions_per_run", 5, raising=False)
    monkeypatch.setattr(mod.settings.b2b_report_delivery, "suppress_unchanged_deliveries", False, raising=False)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "get_campaign_sender", lambda: SimpleNamespace())
    monkeypatch.setattr(mod, "_count_due_non_canary_subscriptions", AsyncMock(return_value=1))
    monkeypatch.setattr(mod, "_fetch_due_rows", AsyncMock(return_value=[due_rows[0]]))
    monkeypatch.setattr(mod, "_claim_delivery_attempt", claim)
    monkeypatch.setattr(mod, "_tracked_vendors", AsyncMock(return_value=set()))
    monkeypatch.setattr(mod, "_resolve_artifact_selection", AsyncMock(return_value=_selection([artifact])))
    monkeypatch.setattr(mod, "_delivery_content_hash", lambda *_args, **_kwargs: "live-hash")
    monkeypatch.setattr(mod, "_send_subscription_email", send_helper)
    monkeypatch.setattr(mod, "_finalize_delivery_attempt", finalize)
    monkeypatch.setattr(mod, "_advance_subscription", advance)

    result = await mod.run(SimpleNamespace(metadata={"canary_account_ids": ["acct-live"]}))

    assert result["dry_run"] is False
    assert result["deferred_non_canary"] == 1
    assert result["due_subscriptions"] == 1
    assert claim.await_count == 1
    assert claim.await_args.args[1]["id"] == "sub-live"
    assert send_helper.await_count == 1
    assert finalize.await_args.kwargs["status"] == "sent"
    assert advance.await_count == 1


@pytest.mark.asyncio
async def test_fetch_due_rows_scopes_canary_before_limit():
    seen: list[tuple[str, tuple[object, ...]]] = []

    class Pool:
        async def fetch(self, query, *args):
            seen.append((query, args))
            if "s.account_id::text = ANY($2::text[])" in query:
                return [{"id": "sub-live", "account_id": "acct-live"}]
            raise AssertionError("expected canary-scoped due-row query")

    rows = await mod._fetch_due_rows(Pool(), 1, {"acct-live"})

    assert rows == [{"id": "sub-live", "account_id": "acct-live"}]
    assert len(seen) == 1
    assert seen[0][1] == (1, ["acct-live"])


@pytest.mark.asyncio
async def test_claim_delivery_attempt_reclaims_dry_run_immediately():
    seen_args: list[tuple[object, ...]] = []

    class Pool:
        async def fetchrow(self, query, *args):
            seen_args.append(args)
            assert "delivery_mode" in query
            assert "ON CONFLICT (subscription_id, scheduled_for, delivery_mode)" in query
            assert "status = ANY($11::text[])" in query
            assert "status = ANY($12::text[])" in query
            return {"id": "log-claim"}

    row = {
        "id": "sub-claim",
        "account_id": "acct-claim",
        "next_delivery_at": datetime.now(timezone.utc),
        "scope_type": "report",
        "scope_key": "report-claim",
        "recipient_emails": ["buyer@example.com"],
        "delivery_frequency": "weekly",
        "deliverable_focus": "all",
        "freshness_policy": "any",
    }

    result = await mod._claim_delivery_attempt(Pool(), row, dry_run=True)

    assert result == {"id": "log-claim"}
    assert seen_args
    assert seen_args[0][3] == "dry_run"
    assert seen_args[0][10] == ["failed", "processing"]
    assert seen_args[0][11] == ["dry_run"]


@pytest.mark.asyncio
async def test_claim_delivery_attempt_uses_live_mode_for_live_runs():
    seen_args: list[tuple[object, ...]] = []

    class Pool:
        async def fetchrow(self, query, *args):
            seen_args.append(args)
            assert "ON CONFLICT (subscription_id, scheduled_for, delivery_mode)" in query
            return {"id": "log-live"}

    row = {
        "id": "sub-live",
        "account_id": "acct-live",
        "next_delivery_at": datetime.now(timezone.utc),
        "scope_type": "report",
        "scope_key": "report-live",
        "recipient_emails": ["buyer@example.com"],
        "delivery_frequency": "weekly",
        "deliverable_focus": "all",
        "freshness_policy": "any",
    }

    result = await mod._claim_delivery_attempt(Pool(), row, dry_run=False)

    assert result == {"id": "log-live"}
    assert seen_args
    assert seen_args[0][3] == "live"


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
    monkeypatch.setattr(mod, "_resolve_artifact_selection", AsyncMock(return_value=_selection([artifact])))
    monkeypatch.setattr(mod, "_latest_delivery_content_hash", AsyncMock(return_value=None))
    monkeypatch.setattr(mod, "_send_subscription_email", send_helper)
    monkeypatch.setattr(mod, "_finalize_delivery_attempt", finalize)
    monkeypatch.setattr(mod, "_advance_subscription", advance)

    result = await mod.run(SimpleNamespace())

    assert result["delivered"] == 1
    assert result["skipped"] == 0
    send_helper.assert_awaited_once()
    assert finalize.await_args.kwargs["status"] == "sent"
