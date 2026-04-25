import json
from datetime import datetime, timezone

import pytest

from atlas_brain.services.reasoning_delivery_audit import (
    iter_witness_objects,
    summarize_reasoning_delivery_health,
    summarize_witness_field_propagation,
)


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


def test_iter_witness_objects_finds_nested_witnesses_but_not_reference_ids():
    payload = {
        "reasoning_reference_ids": {"witness_ids": ["w1"]},
        "coverage_gaps": [{"witness_id": "gap:thin_segment:size_smb"}],
        "reasoning_anchor_examples": {
            "common_pattern": [
                {
                    "witness_id": "w1",
                    "excerpt_text": "Support was slow.",
                    "grounding_status": "grounded",
                }
            ]
        },
        "sections": [
            {
                "quote": {
                    "excerpt_text": "Pricing was the blocker.",
                    "witness_type": "pricing",
                }
            }
        ],
    }

    found = list(iter_witness_objects(payload))

    assert [path for path, _witness in found] == [
        "$.reasoning_anchor_examples.common_pattern[0]",
        "$.sections[0].quote",
    ]
    assert found[0][1]["witness_id"] == "w1"


class _PropagationPool:
    async def fetchrow(self, query, *args):
        assert "FROM b2b_vendor_witnesses" in query
        return {
            "witness_objects": 2,
            "grounding_status_present": 2,
            "phrase_polarity_present": 1,
            "phrase_subject_present": 1,
            "phrase_role_present": 1,
            "phrase_verbatim_present": 2,
            "pain_confidence_present": 1,
            "full_quality_objects": 1,
        }

    async def fetch(self, query, *args):
        assert args == (7, 50)
        if "FROM b2b_reasoning_synthesis" in query:
            return [
                {
                    "surface": "b2b_reasoning_synthesis",
                    "artifact_key": "Slack:2026-04-25:30:v2",
                    "payload": json.dumps({
                        "reasoning_witness_highlights": [
                            {
                                "witness_id": "w1",
                                "excerpt_text": "Support was slow.",
                                "grounding_status": "grounded",
                                "phrase_polarity": "negative",
                                "phrase_subject": "subject_vendor",
                                "phrase_role": "primary_driver",
                                "phrase_verbatim": False,
                                "pain_confidence": "strong",
                            }
                        ]
                    }),
                }
            ]
        if "FROM b2b_cross_vendor_reasoning_synthesis" in query:
            return []
        if "FROM b2b_intelligence" in query:
            return [
                {
                    "surface": "b2b_intelligence:battle_card",
                    "artifact_key": "report-1",
                    "payload": {
                        "reasoning_witness_highlights": [
                            {
                                "witness_id": "w2",
                                "excerpt_text": "Pricing was the blocker.",
                                "witness_type": "common_pattern",
                            }
                        ]
                    },
                }
            ]
        raise AssertionError(query)


@pytest.mark.asyncio
async def test_summarize_witness_field_propagation_counts_surface_drops():
    result = await summarize_witness_field_propagation(
        _PropagationPool(),
        days=7,
        row_limit=50,
    )

    by_surface = {surface["surface"]: surface for surface in result["surfaces"]}

    source = by_surface["b2b_vendor_witnesses"]
    assert source["witness_objects"] == 2
    assert source["field_counts"]["phrase_verbatim"]["present"] == 2
    assert source["field_counts"]["pain_confidence"]["missing"] == 1

    synthesis = by_surface["b2b_reasoning_synthesis"]
    assert synthesis["witness_objects"] == 1
    assert synthesis["full_quality_objects"] == 1
    assert synthesis["field_counts"]["phrase_verbatim"]["present"] == 1

    battle_card = by_surface["b2b_intelligence:battle_card"]
    assert battle_card["witness_objects"] == 1
    assert battle_card["full_quality_objects"] == 0
    assert battle_card["field_counts"]["grounding_status"]["missing"] == 1
    assert battle_card["drop_examples"][0]["missing_fields"] == result["quality_fields"]
