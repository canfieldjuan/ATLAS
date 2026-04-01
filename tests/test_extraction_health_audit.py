from datetime import date
from datetime import datetime, timezone

import pytest

from atlas_brain.services.extraction_health_audit import summarize_extraction_health


class _FakePool:
    def __init__(self):
        self.fetchrow_calls = []
        self.fetch_calls = []

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append((query, args))
        return {
            "enriched_rows": 25277,
            "rows_with_spans": 21000,
            "span_count": 50554,
            "repair_triggered_rows": 3200,
            "repair_promoted_rows": 1400,
            "hard_gap_rows": 12,
            "phrase_arrays_without_spans": 4,
            "blank_replacement_mode": 3,
            "blank_operating_model_shift": 2,
            "blank_productivity_delta_claim": 1,
            "blank_org_pressure_type": 1,
            "missing_or_empty_evidence_spans": 4,
            "blank_evidence_map_hash": 1,
            "empty_salience_flags": 10624,
            "strategic_candidate_rows": 4452,
            "money_without_pricing_span": 210,
            "competitor_without_displacement_framing": 890,
            "named_company_without_named_account_evidence": 11,
            "timeline_language_without_timing_anchor": 1234,
            "workflow_language_without_replacement_mode": 601,
        }

    async def fetch(self, query, *args):
        self.fetch_calls.append((query, args))
        if "GROUP BY 1" in query:
            return [
                {
                    "day": date(2026, 3, 31),
                    "enriched_rows": 150,
                    "rows_with_spans": 120,
                    "span_count": 360,
                    "repair_triggered_rows": 30,
                    "hard_gap_rows": 12,
                    "phrase_arrays_without_spans": 4,
                    "blank_replacement_mode": 3,
                    "blank_operating_model_shift": 2,
                    "missing_or_empty_evidence_spans": 4,
                    "strategic_candidate_rows": 51,
                }
            ]
        if "GROUP BY source" in query:
            return [
                {
                    "source": "reddit",
                    "enriched_rows": 800,
                    "repair_triggered_rows": 120,
                    "repair_promoted_rows": 60,
                    "rows_with_spans": 760,
                    "span_count": 1900,
                }
            ]
        if "FROM task_executions e" in query:
            return [
                {
                    "run_id": "run-123",
                    "task_name": "b2b_enrichment_repair",
                    "started_at": datetime(2026, 3, 31, 22, 0, tzinfo=timezone.utc),
                    "result_text": '{"reviews_processed": 5, "witness_rows": 2, "witness_count": 8, "secondary_write_hits": 1, "exact_cache_hits": 2, "generated": 3}',
                }
            ]
        if "COALESCE(enriched_at, imported_at) AS activity_at" in query:
            return [
                {
                    "vendor_name": "Zendesk",
                    "source": "trustpilot",
                    "content_type": "review",
                    "reviewer_title": "",
                    "reviewer_company": "",
                    "review_text": "We switched to another provider because Zendesk was not worth the money and support was a nightmare.",
                    "summary": "Switched away from Zendesk",
                    "pros": "",
                    "cons": "",
                    "product_name": "Zendesk",
                    "enrichment": {
                        "salience_flags": [],
                        "replacement_mode": "none",
                        "timeline": {"decision_timeline": "unknown"},
                        "churn_signals": {
                            "intent_to_leave": True,
                            "actively_evaluating": False,
                            "migration_in_progress": True,
                            "contract_renewal_mentioned": False,
                        },
                        "specific_complaints": ["support was a nightmare"],
                        "pricing_phrases": ["not worth the money"],
                        "feature_gaps": [],
                        "competitors_mentioned": [],
                        "evidence_spans": [],
                    },
                    "activity_at": datetime(2026, 3, 31, 22, 0, tzinfo=timezone.utc),
                }
            ]
        return [
            {
                "vendor_name": "Zendesk",
                "hard_gap_rows": 5,
                "phrase_arrays_without_spans": 2,
                "blank_replacement_mode": 1,
                "blank_operating_model_shift": 1,
                "missing_or_empty_evidence_spans": 2,
                "empty_salience_flags": 18,
                "strategic_candidate_rows": 22,
                "enriched_rows": 300,
            }
        ]


@pytest.mark.asyncio
async def test_summarize_extraction_health_returns_snapshot_trend_and_vendors():
    pool = _FakePool()

    result = await summarize_extraction_health(
        pool,
        days=14,
        top_n=5,
    )

    assert result["days"] == 14
    assert result["top_n"] == 5
    assert result["current_snapshot"]["enriched_rows"] == 25277
    assert result["current_snapshot"]["rows_with_spans"] == 21000
    assert result["current_snapshot"]["span_count"] == 50554
    assert result["current_snapshot"]["repair_triggered_rows"] == 3200
    assert result["current_snapshot"]["repair_trigger_rate"] == pytest.approx(3200 / 25277, rel=1e-4)
    assert result["current_snapshot"]["hard_gap_rows"] == 12
    assert result["current_snapshot"]["strategic_candidate_rows"] == 1
    assert result["current_snapshot"]["competitor_without_displacement_framing"] == 1
    assert result["current_snapshot"]["money_without_pricing_span"] == 0
    assert result["current_snapshot"]["workflow_language_without_replacement_mode"] == 0
    assert result["daily_trend"][0]["enriched_rows"] == 150
    assert result["daily_trend"][0]["span_count"] == 360
    assert result["daily_trend"][0]["witness_yield_rate"] == pytest.approx(2.4)
    assert result["daily_trend"][0]["repair_trigger_rate"] == pytest.approx(0.2)
    assert result["daily_trend"][0]["hard_gap_rows"] == 12
    assert result["daily_trend"][0]["strategic_candidate_rows"] == 51
    assert result["top_vendors"][0]["vendor_name"] == "Zendesk"
    assert result["top_vendors"][0]["strategic_candidate_rows"] == 1
    assert result["top_sources"][0]["source"] == "reddit"
    assert result["top_sources"][0]["witness_yield_rate"] == pytest.approx(2.375)
    assert result["recent_runs"][0]["run_id"] == "run-123"
    assert result["recent_runs"][0]["witness_count"] == 8
    assert result["recent_runs"][0]["secondary_write_hits"] == 1
    assert result["current_snapshot"]["secondary_write_hits_window"] == 1
    assert pool.fetch_calls[0][1] == (14,)
    assert pool.fetch_calls[1][1] == (14, 5)
    assert pool.fetch_calls[2][1] == ()
    assert pool.fetch_calls[3][1] == (5,)
    assert pool.fetch_calls[4][1] == (14, ["b2b_enrichment", "b2b_enrichment_repair"], 5)
