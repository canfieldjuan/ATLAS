from datetime import date

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
                    "hard_gap_rows": 12,
                    "phrase_arrays_without_spans": 4,
                    "blank_replacement_mode": 3,
                    "blank_operating_model_shift": 2,
                    "missing_or_empty_evidence_spans": 4,
                    "strategic_candidate_rows": 51,
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
    assert result["current_snapshot"]["hard_gap_rows"] == 12
    assert result["current_snapshot"]["strategic_candidate_rows"] == 4452
    assert result["current_snapshot"]["workflow_language_without_replacement_mode"] == 601
    assert result["daily_trend"][0]["hard_gap_rows"] == 12
    assert result["daily_trend"][0]["strategic_candidate_rows"] == 51
    assert result["top_vendors"][0]["vendor_name"] == "Zendesk"
    assert result["top_vendors"][0]["strategic_candidate_rows"] == 22
    assert pool.fetch_calls[0][1] == (14,)
    assert pool.fetch_calls[1][1] == (5,)
