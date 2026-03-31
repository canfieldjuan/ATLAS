from datetime import datetime, timezone

import pytest

from atlas_brain.services.reasoning_retry_audit import summarize_reasoning_retry_churn


class _FakePool:
    def __init__(self):
        self.fetchrow_calls = []
        self.fetch_calls = []

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append((query, args))
        return {
            "recovered_retries": 5,
            "escalations": 2,
            "repeated_rule_escalations": 1,
            "costly_retry_escalations": 1,
            "affected_vendors": 3,
            "retry_tokens": 91000,
        }

    async def fetch(self, query, *args):
        self.fetch_calls.append((query, args))
        if "FROM synthesis_validation_results" in query:
            return [
                {
                    "rule_code": "unknown_packet_citation",
                    "retry_findings": 3,
                    "vendor_count": 2,
                    "last_seen_at": datetime(2026, 3, 30, 23, 0, tzinfo=timezone.utc),
                }
            ]
        if "event_type = 'validation_retry_rejected'" in query:
            return [
                {
                    "vendor_name": "Shopify",
                    "retry_count": 2,
                    "retry_tokens": 45000,
                    "last_seen_at": datetime(2026, 3, 30, 22, 30, tzinfo=timezone.utc),
                }
            ]
        return [
            {
                "review_id": "r1",
                "status": "open",
                "occurrence_count": 2,
                "last_seen_at": datetime(2026, 3, 30, 22, 45, tzinfo=timezone.utc),
                "vendor_name": "Shopify",
                "reason_code": "costly_validation_retry",
                "rule_code": None,
                "summary": "Recovered validation retries costly for Shopify",
                "detail": {"retry_tokens": 90000},
            }
        ]


@pytest.mark.asyncio
async def test_summarize_reasoning_retry_churn_returns_normalized_sections():
    pool = _FakePool()

    result = await summarize_reasoning_retry_churn(
        pool,
        hours=48,
        top_n=5,
        queue_limit=10,
    )

    assert result["hours"] == 48
    assert result["summary"]["recovered_retries"] == 5
    assert result["summary"]["retry_tokens"] == 91000
    assert result["top_rules"][0]["rule_code"] == "unknown_packet_citation"
    assert result["top_vendors"][0]["vendor_name"] == "Shopify"
    assert result["open_retry_escalations"][0]["reason_code"] == "costly_validation_retry"
    assert pool.fetchrow_calls[0][1] == (48,)
    assert pool.fetch_calls[0][1] == (48, 5)
    assert pool.fetch_calls[1][1] == (48, 5)
    assert pool.fetch_calls[2][1] == (10,)
