from __future__ import annotations

from datetime import datetime, timezone

import pytest

from atlas_brain.services.report_subscription_delivery_audit import (
    _normalize_account_ids,
    summarize_report_subscription_delivery_health,
)


def test_normalize_account_ids_deduplicates_and_strips():
    assert _normalize_account_ids([" acct-1 ", "", "acct-1", "acct-2"]) == ["acct-1", "acct-2"]
    assert _normalize_account_ids([]) is None


@pytest.mark.asyncio
async def test_summarize_report_subscription_delivery_health_shapes_results():
    now = datetime(2026, 4, 7, 22, 0, tzinfo=timezone.utc)

    class Pool:
        async def fetchrow(self, query, *args):
            if "COUNT(*) AS total_attempts" in query:
                assert args == (14, ["acct-1"])
                return {
                    "total_attempts": 5,
                    "live_attempts": 3,
                    "dry_run_attempts": 2,
                    "sent_attempts": 2,
                    "partial_attempts": 1,
                    "skipped_attempts": 1,
                    "failed_attempts": 1,
                    "processing_attempts": 0,
                    "blocked_core_incomplete_attempts": 1,
                    "unchanged_skip_attempts": 1,
                    "fully_suppressed_attempts": 1,
                    "suppressed_recipient_failures": 2,
                    "latest_delivered_at": now,
                }
            if "COUNT(*) FILTER (WHERE enabled = TRUE)" in query:
                assert args == (["acct-1"],)
                return {
                    "enabled_subscriptions": 2,
                    "due_now": 1,
                }
            raise AssertionError(f"unexpected fetchrow query: {query}")

        async def fetch(self, query, *args):
            if "GROUP BY delivery_mode, status" in query:
                assert args == (14, ["acct-1"])
                return [
                    {"delivery_mode": "dry_run", "status": "skipped", "attempt_count": 1},
                    {"delivery_mode": "live", "status": "sent", "attempt_count": 2},
                ]
            if "GROUP BY l.account_id, sa.name" in query:
                assert args == (14, ["acct-1"], 3)
                return [
                    {
                        "account_id": "acct-1",
                        "account_name": "Canary Co",
                        "attempt_count": 5,
                        "live_attempt_count": 3,
                        "failed_attempt_count": 1,
                        "partial_attempt_count": 1,
                        "blocked_core_incomplete_attempt_count": 1,
                        "unchanged_skip_attempt_count": 1,
                        "latest_delivered_at": now,
                    }
                ]
            if "ORDER BY l.delivered_at DESC, l.id DESC" in query:
                assert args == (14, ["acct-1"], 3)
                return [
                    {
                        "id": "log-1",
                        "subscription_id": "sub-1",
                        "account_id": "acct-1",
                        "account_name": "Canary Co",
                        "scheduled_for": now,
                        "delivered_at": now,
                        "delivery_mode": "live",
                        "status": "sent",
                        "freshness_state": "fresh",
                        "scope_type": "report",
                        "scope_key": "report-1",
                        "report_count": 1,
                        "recipient_count": 1,
                        "summary": "Delivered 1 artifact(s) to 1 recipient(s).",
                        "error": "",
                    }
                ]
            raise AssertionError(f"unexpected fetch query: {query}")

    result = await summarize_report_subscription_delivery_health(
        Pool(),
        days=14,
        top_n=3,
        account_ids=["acct-1"],
    )

    assert result["days"] == 14
    assert result["filters"] == {"account_ids": ["acct-1"], "top_n": 3}
    assert result["delivery_summary"]["total_attempts"] == 5
    assert result["delivery_summary"]["live_attempts"] == 3
    assert result["delivery_summary"]["dry_run_attempts"] == 2
    assert result["delivery_summary"]["blocked_core_incomplete_attempts"] == 1
    assert result["delivery_summary"]["unchanged_skip_attempts"] == 1
    assert result["delivery_summary"]["suppressed_recipient_failures"] == 2
    assert result["active_subscription_summary"] == {"enabled_subscriptions": 2, "due_now": 1}
    assert result["status_breakdown"][0]["delivery_mode"] == "dry_run"
    assert result["top_accounts"][0]["account_name"] == "Canary Co"
    assert result["recent_attempts"][0]["status"] == "sent"
    assert result["recent_attempts"][0]["freshness_state"] == "fresh"
