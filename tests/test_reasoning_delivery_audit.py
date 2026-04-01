from datetime import datetime, timezone

import pytest

from atlas_brain.services.reasoning_delivery_audit import summarize_reasoning_delivery_health


class _FakePool:
    def __init__(self):
        self.fetchrow_calls = []
        self.fetch_calls = []

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append((query, args))
        return {
            "vendor_rows": 12,
            "latest_created_at": datetime(2026, 3, 31, 6, 25, tzinfo=timezone.utc),
            "avg_tokens_used": 24810.5,
            "metric_ref_rows": 12,
            "witness_ref_rows": 11,
        }

    async def fetch(self, query, *args):
        self.fetch_calls.append((query, args))
        if "FROM b2b_cross_vendor_reasoning_synthesis" in query:
            return [
                {
                    "analysis_type": "pairwise_battle",
                    "row_count": 11,
                    "latest_created_at": datetime(2026, 3, 31, 6, 0, tzinfo=timezone.utc),
                    "avg_tokens_used": 4295.2,
                    "metric_ref_rows": 11,
                    "witness_ref_rows": 10,
                }
            ]
        if "FROM synthesis_validation_results" in query:
            return [
                {
                    "rule_code": "missing_citations",
                    "severity": "warning",
                    "finding_count": 4,
                    "vendor_count": 2,
                    "latest_created_at": datetime(2026, 3, 31, 5, 45, tzinfo=timezone.utc),
                }
            ]
        if "FROM latest_attempt" in query:
            return [
                {
                    "artifact_type": "reasoning_synthesis",
                    "status": "succeeded",
                    "row_count": 12,
                },
                {
                    "artifact_type": "cross_vendor_reasoning",
                    "status": "succeeded",
                    "row_count": 11,
                },
            ]
        return [
            {
                "report_type": "battle_card",
                "row_count": 2,
                "latest_created_at": datetime(2026, 3, 31, 6, 10, tzinfo=timezone.utc),
                "rows_with_reference_ids": 2,
            },
            {
                "report_type": "weekly_churn_feed",
                "row_count": 1,
                "latest_created_at": datetime(2026, 3, 31, 6, 20, tzinfo=timezone.utc),
                "rows_with_reference_ids": 1,
            },
            {
                "report_type": "displacement_report",
                "row_count": 1,
                "latest_created_at": datetime(2026, 3, 31, 6, 25, tzinfo=timezone.utc),
                "rows_with_reference_ids": 1,
            },
            {
                "report_type": "category_overview",
                "row_count": 1,
                "latest_created_at": datetime(2026, 3, 31, 6, 30, tzinfo=timezone.utc),
                "rows_with_reference_ids": 1,
            },
        ]


@pytest.mark.asyncio
async def test_summarize_reasoning_delivery_health_returns_normalized_sections():
    pool = _FakePool()

    result = await summarize_reasoning_delivery_health(
        pool,
        days=14,
        top_n=5,
    )

    assert result["days"] == 14
    assert result["vendor_synthesis"]["rows"] == 12
    assert result["vendor_synthesis"]["rows_with_metric_refs"] == 12
    assert result["cross_vendor_synthesis"][0]["analysis_type"] == "pairwise_battle"
    assert result["top_validation_findings"][0]["rule_code"] == "missing_citations"
    assert result["latest_attempt_statuses"][0]["artifact_type"] == "reasoning_synthesis"
    assert result["downstream_reference_coverage"][0]["report_type"] == "battle_card"
    assert any(
        row["report_type"] == "displacement_report"
        for row in result["downstream_reference_coverage"]
    )
    assert any(
        row["report_type"] == "category_overview"
        for row in result["downstream_reference_coverage"]
    )
    assert pool.fetchrow_calls[0][1] == (14,)
    assert pool.fetch_calls[0][1] == (14,)
    assert pool.fetch_calls[1][1] == (14, 5)
    assert pool.fetch_calls[2][1] == (14,)
    assert pool.fetch_calls[3][1] == (14,)
