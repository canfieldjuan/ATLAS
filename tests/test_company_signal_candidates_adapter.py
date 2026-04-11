from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod
from atlas_brain.autonomous.tasks._b2b_shared import (
    read_company_signal_candidate_groups,
    read_company_signal_candidate_group_summary,
    read_company_signal_review_impact_summary,
    read_company_signal_candidates,
)


def _make_review_row(**overrides):
    row = {
        "review_id": uuid4(),
        "source": "g2",
        "reviewer_company": "Acme Corp",
        "raw_reviewer_company": "Acme Corp",
        "resolution_confidence": "medium",
        "vendor_name": "Zendesk",
        "product_category": "Customer Support",
        "reviewed_at": datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc),
        "reviewer_title": "VP Operations",
        "company_size_raw": "201-500",
        "industry": "SaaS",
        "company_domain": "acme.example",
        "role_level": "vp",
        "is_dm": True,
        "urgency": Decimal("8.2"),
        "pain": "pricing",
        "alternatives": json.dumps([{"name": "Freshdesk"}]),
        "seat_count": "120",
        "contract_end": "2026-07-01",
        "buying_stage": "evaluation",
        "relevance_score": Decimal("0.71"),
        "author_churn_score": Decimal("0.65"),
        "intent_to_leave": False,
        "actively_evaluating": True,
        "contract_renewal_mentioned": False,
        "indicator_cancel": False,
        "indicator_migration": False,
        "indicator_evaluation": True,
        "indicator_switch": False,
    }
    row.update(overrides)
    return row


def _make_candidate_row(**overrides):
    row = {
        "review_id": uuid4(),
        "company_name": "Acme Corp",
        "company_name_raw": "Acme Corp",
        "vendor_name": "Zendesk",
        "product_category": "Customer Support",
        "source": "reddit",
        "reviewed_at": datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc),
        "urgency_score": Decimal("6.8"),
        "relevance_score": Decimal("0.71"),
        "pain_category": "pricing",
        "buyer_role": "vp",
        "decision_maker": True,
        "seat_count": 120,
        "contract_end": "2026-07-01",
        "buying_stage": "evaluation",
        "resolution_confidence": "medium",
        "confidence_score": Decimal("0.26"),
        "confidence_tier": "low",
        "signal_evidence_present": False,
        "canonical_gap_reason": "low_confidence_low_trust_source",
        "candidate_bucket": "analyst_review",
        "review_status": "pending",
        "review_status_updated_at": None,
        "reviewed_by": None,
        "review_notes": None,
        "materialization_run_id": "run-123",
        "first_seen_at": datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc),
        "last_seen_at": datetime(2026, 4, 10, 12, 5, tzinfo=timezone.utc),
        "summary": "Considering a switch this quarter.",
        "review_excerpt": "We are actively evaluating alternatives because pricing keeps climbing.",
        "pros": "Easy setup",
        "cons": "Too expensive",
        "content_type": "review",
        "source_url": "https://example.com/review/1",
        "quotable_phrases": json.dumps(["Pricing keeps climbing"]),
    }
    row.update(overrides)
    return row


def _make_group_row(**overrides):
    sample_review_id = uuid4()
    row = {
        "id": uuid4(),
        "company_name": "acme",
        "display_company_name": "Acme Corp",
        "vendor_name": "Zendesk",
        "product_category": "Customer Support",
        "review_count": 3,
        "distinct_source_count": 2,
        "decision_maker_count": 1,
        "signal_evidence_count": 2,
        "canonical_ready_review_count": 1,
        "avg_urgency_score": Decimal("7.4"),
        "max_urgency_score": Decimal("8.8"),
        "avg_confidence_score": Decimal("0.41"),
        "max_confidence_score": Decimal("0.58"),
        "corroborated_confidence_score": Decimal("0.68"),
        "confidence_tier": "high",
        "source_distribution": json.dumps({"reddit": 2, "g2": 1}),
        "gap_reason_distribution": json.dumps({"canonical_ready": 1, "low_confidence_low_trust_source": 2}),
        "sample_review_ids": [sample_review_id],
        "representative_review_id": sample_review_id,
        "representative_source": "reddit",
        "representative_pain_category": "pricing",
        "representative_buyer_role": "vp",
        "representative_decision_maker": True,
        "representative_seat_count": 120,
        "representative_contract_end": "2026-07-01",
        "representative_buying_stage": "evaluation",
        "representative_confidence_score": Decimal("0.58"),
        "representative_urgency_score": Decimal("8.8"),
        "canonical_gap_reason": None,
        "candidate_bucket": "canonical_ready",
        "review_status": "pending",
        "review_status_updated_at": None,
        "reviewed_by": None,
        "review_notes": None,
        "materialization_run_id": "run-789",
        "first_seen_at": datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc),
        "last_seen_at": datetime(2026, 4, 10, 12, 5, tzinfo=timezone.utc),
    }
    row.update(overrides)
    return row


def _make_support_review_row(review_id, **overrides):
    row = {
        "id": review_id,
        "source": "reddit",
        "summary": "Three reviewers are evaluating replacements.",
        "review_excerpt": "Pricing pressure and renewal risk keep coming up across teams.",
        "source_url": "https://example.com/review/2",
        "reviewed_at": datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc),
        "quotable_phrases": json.dumps(["Renewal risk keeps coming up"]),
    }
    row.update(overrides)
    return row


class FakePool:
    def __init__(self, *responses):
        if len(responses) == 1:
            self.fetch = AsyncMock(return_value=responses[0])
        else:
            self.fetch = AsyncMock(side_effect=list(responses))


@pytest.fixture(autouse=True)
def _patch_shared_helpers():
    with patch(
        "atlas_brain.autonomous.tasks._b2b_shared._intelligence_source_allowlist",
        return_value=["g2", "reddit"],
    ), patch(
        "atlas_brain.autonomous.tasks._b2b_shared._eligible_review_filters",
        return_value=(
            "r.enrichment_status = 'enriched'"
            " AND r.enriched_at > NOW() - make_interval(days => $1)"
            " AND r.source = ANY($2::text[])"
        ),
    ):
        yield


@pytest.mark.asyncio
async def test_review_candidates_keep_low_trust_rows_for_analyst_review():
    pool = FakePool([_make_review_row(source="reddit")])

    results = await shared_mod._fetch_company_signal_review_candidates(
        pool,
        window_days=90,
        urgency_threshold=7.0,
    )

    assert len(results) == 1
    assert results[0]["candidate_bucket"] == "analyst_review"
    assert results[0]["canonical_gap_reason"] == "low_confidence_low_trust_source"


@pytest.mark.asyncio
async def test_review_candidates_mark_trusted_ready_rows_as_canonical_ready():
    pool = FakePool([_make_review_row(source="g2")])

    results = await shared_mod._fetch_company_signal_review_candidates(
        pool,
        window_days=90,
        urgency_threshold=7.0,
    )

    assert len(results) == 1
    assert results[0]["candidate_bucket"] == "canonical_ready"
    assert results[0]["canonical_gap_reason"] is None


@pytest.mark.asyncio
async def test_review_candidates_flag_below_threshold_rows():
    pool = FakePool([_make_review_row(source="g2", urgency=Decimal("5.4"))])

    results = await shared_mod._fetch_company_signal_review_candidates(
        pool,
        window_days=90,
        urgency_threshold=7.0,
    )

    assert len(results) == 1
    assert results[0]["candidate_bucket"] == "analyst_review"
    assert results[0]["canonical_gap_reason"] == "below_high_intent_threshold"


@pytest.mark.asyncio
async def test_review_candidates_reject_ineligible_company_names():
    pool = FakePool([_make_review_row(
        reviewer_company="https://chatgpt.com/g/g-LsO4PHxnv-robert-on-ai-and-craftsmanship",
    )])

    results = await shared_mod._fetch_company_signal_review_candidates(
        pool,
        window_days=90,
        urgency_threshold=7.0,
    )

    assert results == []


@pytest.mark.asyncio
async def test_read_company_signal_candidates_maps_rows_and_filters_in_sql():
    pool = FakePool([_make_candidate_row()])

    results = await read_company_signal_candidates(
        pool,
        window_days=90,
        company_name="Acme",
        scoped_vendors=["Zendesk"],
        candidate_bucket="analyst_review",
        review_status="pending",
        canonical_gap_reason="low_confidence_low_trust_source",
        min_urgency=6.0,
        min_confidence=0.2,
        decision_makers_only=True,
        signal_evidence_present=False,
        limit=25,
    )

    sql = pool.fetch.call_args[0][0]
    assert "candidate_bucket =" in sql
    assert "review_status =" in sql
    assert "company_name ILIKE" in sql
    assert "canonical_gap_reason =" in sql
    assert "urgency_score" in sql
    assert "confidence_score" in sql
    assert "decision_maker = true" in sql
    assert "signal_evidence_present =" in sql
    assert "ANY(" in sql
    assert "LIMIT $10" in sql
    assert len(results) == 1
    assert results[0]["company"] == "Acme Corp"
    assert results[0]["candidate_bucket"] == "analyst_review"
    assert results[0]["confidence_score"] == 0.26
    assert results[0]["review_status"] == "pending"
    assert results[0]["review_status_updated_at"] is None
    assert results[0]["quote_excerpt"] == "Pricing keeps climbing"
    assert results[0]["review_excerpt"].startswith("We are actively evaluating")


@pytest.mark.asyncio
async def test_read_company_signal_candidates_scoped_empty_short_circuits():
    pool = FakePool([_make_candidate_row()])

    results = await read_company_signal_candidates(
        pool,
        window_days=90,
        scoped_vendors=[],
    )

    assert results == []
    pool.fetch.assert_not_called()


@pytest.mark.asyncio
async def test_read_company_signal_candidate_groups_maps_rows_and_support_reviews():
    group_row = _make_group_row()
    support_row = _make_support_review_row(group_row["sample_review_ids"][0])
    pool = FakePool([group_row], [support_row])

    results = await read_company_signal_candidate_groups(
        pool,
        window_days=90,
        company_name="Acme",
        scoped_vendors=["Zendesk"],
        candidate_bucket="canonical_ready",
        review_status="pending",
        min_urgency=6.0,
        min_confidence=0.6,
        min_reviews=2,
        decision_makers_only=True,
        signal_evidence_present=True,
        review_priority_band="medium",
        review_priority_reason="cross_source_corroboration",
        limit=25,
    )

    primary_sql = pool.fetch.call_args_list[0][0][0]
    support_sql = pool.fetch.call_args_list[1][0][0]
    assert "candidate_bucket =" in primary_sql
    assert "review_status =" in primary_sql
    assert "review_count >=" in primary_sql
    assert "corroborated_confidence_score" in primary_sql
    assert "signal_evidence_count > 0" in primary_sql
    assert "decision_maker_count > 0" in primary_sql
    assert "promote_now" in primary_sql
    assert "cross_source_corroboration" in primary_sql
    assert "ANY(" in primary_sql
    assert "WHERE id = ANY($1::uuid[])" in support_sql
    assert len(results) == 1
    assert results[0]["group_id"] == str(group_row["id"])
    assert results[0]["display_company"] == "Acme Corp"
    assert results[0]["review_count"] == 3
    assert results[0]["corroborated_confidence_score"] == 0.68
    assert results[0]["review_priority_band"] == "promote_now"
    assert results[0]["review_priority_reason"] == "canonical_ready"
    assert results[0]["supporting_reviews"][0]["quote_excerpt"] == "Renewal risk keeps coming up"


def test_company_signal_queue_sla_days_normalizes_misordered_thresholds():
    with patch.object(shared_mod.settings.b2b_churn, "company_signal_queue_promote_now_sla_days", 3.0), patch.object(
        shared_mod.settings.b2b_churn,
        "company_signal_queue_high_sla_days",
        2.0,
    ), patch.object(
        shared_mod.settings.b2b_churn,
        "company_signal_queue_medium_sla_days",
        1.0,
    ), patch.object(
        shared_mod.settings.b2b_churn,
        "company_signal_queue_low_sla_days",
        5.0,
    ):
        assert shared_mod._company_signal_queue_sla_days() == {
            "promote_now": 3.0,
            "high": 3.0,
            "medium": 3.0,
            "low": 5.0,
        }


@pytest.mark.asyncio
async def test_read_company_signal_candidate_group_summary_aggregates_queue_health():
    pool = type("SummaryPool", (), {})()
    pool.fetchrow = AsyncMock(
        return_value={
            "total_groups": 12,
            "total_reviews": 37,
            "canonical_ready_reviews": 5,
            "pending_groups": 8,
            "actionable_pending_groups": 6,
            "actionable_pending_reviews": 17,
            "blocked_pending_groups": 2,
            "blocked_pending_reviews": 4,
            "near_threshold_blocked_groups": 1,
            "near_threshold_blocked_reviews": 2,
            "approved_groups": 3,
            "suppressed_groups": 1,
            "canonical_ready_groups": 2,
            "analyst_review_groups": 10,
            "pending_canonical_ready_groups": 2,
            "pending_analyst_review_groups": 6,
            "decision_maker_groups": 7,
            "signal_evidence_groups": 4,
            "avg_pending_age_days": 2.75,
            "oldest_pending_age_days": 5.5,
            "overdue_pending_groups": 3,
            "overdue_pending_reviews": 9,
        }
    )
    pool.fetch = AsyncMock(
        side_effect=[
            [
                {
                    "gap_reason": "low_confidence_low_trust_source",
                    "group_count": 7,
                    "review_count": 21,
                }
            ],
            [
                {
                    "vendor_name": "Zendesk",
                    "group_count": 5,
                    "review_count": 14,
                    "pending_groups": 4,
                    "canonical_ready_groups": 1,
                }
            ],
            [
                {
                    "vendor_name": "Zendesk",
                    "actionable_group_count": 3,
                    "actionable_review_count": 9,
                    "promote_now_group_count": 1,
                    "high_group_count": 1,
                    "medium_group_count": 1,
                    "actionable_signal_evidence_groups": 2,
                    "actionable_decision_maker_groups": 1,
                }
            ],
            [
                {
                    "vendor_name": "Zendesk",
                    "review_priority_band": "high",
                    "review_priority_reason": "has_signal_evidence_and_decision_maker",
                    "actionable_group_count": 2,
                    "actionable_review_count": 5,
                }
            ],
            [
                {
                    "vendor_name": "Copper",
                    "blocked_group_count": 2,
                    "blocked_review_count": 4,
                    "low_confidence_group_count": 2,
                    "below_threshold_group_count": 0,
                    "missing_signal_evidence_group_count": 0,
                }
            ],
            [
                {
                    "vendor_name": "Copper",
                    "canonical_gap_reason": "low_confidence_low_trust_source",
                    "review_priority_reason": "low_confidence_low_trust_source",
                    "blocked_group_count": 2,
                    "blocked_review_count": 4,
                }
            ],
            [
                {
                    "vendor_name": "Copper",
                    "near_threshold_group_count": 1,
                    "near_threshold_review_count": 2,
                    "low_confidence_near_threshold_groups": 1,
                    "below_threshold_near_threshold_groups": 0,
                }
            ],
            [
                {
                    "canonical_gap_reason": "low_confidence_low_trust_source",
                    "near_threshold_group_count": 1,
                    "near_threshold_review_count": 2,
                }
            ],
            [
                {
                    "id": uuid4(),
                    "company_name": "coppercorp",
                    "display_company_name": "Copper Corp",
                    "vendor_name": "Copper",
                    "review_count": 2,
                    "canonical_gap_reason": "low_confidence_low_trust_source",
                    "candidate_bucket": "analyst_review",
                    "distinct_source_count": 1,
                    "max_urgency_score": Decimal("5.0"),
                    "corroborated_confidence_score": Decimal("0.41"),
                    "representative_source": "reddit",
                }
            ],
            [
                {
                    "source": "reddit",
                    "group_count": 2,
                    "review_count": 4,
                }
            ],
            [
                {
                    "source": "reddit",
                    "group_count": 1,
                    "review_count": 2,
                }
            ],
            [
                {
                    "confidence_tier": "high",
                    "group_count": 2,
                },
                {
                    "confidence_tier": "low",
                    "group_count": 10,
                },
            ],
            [
                {
                    "id": uuid4(),
                    "company_name": "acme",
                    "display_company_name": "Acme Corp",
                    "vendor_name": "Zendesk",
                    "review_count": 3,
                    "distinct_source_count": 2,
                    "decision_maker_count": 1,
                    "signal_evidence_count": 2,
                    "canonical_ready_review_count": 1,
                    "max_urgency_score": Decimal("8.8"),
                    "corroborated_confidence_score": Decimal("0.68"),
                    "representative_source": "reddit",
                    "candidate_bucket": "canonical_ready",
                    "canonical_gap_reason": None,
                    "review_priority_band": "promote_now",
                    "review_priority_reason": "canonical_ready",
                }
            ],
            [
                {
                    "review_priority_band": "promote_now",
                    "group_count": 2,
                    "review_count": 5,
                },
                {
                    "review_priority_band": "medium",
                    "group_count": 4,
                    "review_count": 12,
                },
            ],
            [
                {
                    "review_priority_band": "promote_now",
                    "review_priority_reason": "canonical_ready",
                    "group_count": 2,
                    "review_count": 5,
                },
                {
                    "review_priority_band": "medium",
                    "review_priority_reason": "cross_source_corroboration",
                    "group_count": 4,
                    "review_count": 12,
                },
            ],
            [
                {
                    "review_priority_band": "promote_now",
                    "sla_days": 1.0,
                    "pending_group_count": 2,
                    "pending_review_count": 5,
                    "overdue_group_count": 1,
                    "overdue_review_count": 3,
                    "oldest_pending_age_days": 5.5,
                    "oldest_overdue_age_days": 5.5,
                },
                {
                    "review_priority_band": "medium",
                    "sla_days": 7.0,
                    "pending_group_count": 4,
                    "pending_review_count": 12,
                    "overdue_group_count": 0,
                    "overdue_review_count": 0,
                    "oldest_pending_age_days": 4.0,
                    "oldest_overdue_age_days": 0.0,
                },
            ],
            [
                {
                    "review_priority_band": "promote_now",
                    "review_priority_reason": "canonical_ready",
                    "sla_days": 1.0,
                    "pending_group_count": 2,
                    "pending_review_count": 5,
                    "overdue_group_count": 1,
                    "overdue_review_count": 3,
                    "oldest_pending_age_days": 5.5,
                    "oldest_overdue_age_days": 5.5,
                },
                {
                    "review_priority_band": "medium",
                    "review_priority_reason": "cross_source_corroboration",
                    "sla_days": 7.0,
                    "pending_group_count": 4,
                    "pending_review_count": 12,
                    "overdue_group_count": 0,
                    "overdue_review_count": 0,
                    "oldest_pending_age_days": 4.0,
                    "oldest_overdue_age_days": 0.0,
                },
            ],
            [
                {
                    "id": uuid4(),
                    "company_name": "acme",
                    "display_company_name": "Acme Corp",
                    "vendor_name": "Zendesk",
                    "review_count": 3,
                    "candidate_bucket": "canonical_ready",
                    "canonical_gap_reason": None,
                    "review_priority_band": "promote_now",
                    "review_priority_reason": "canonical_ready",
                    "pending_age_days": 5.5,
                }
            ],
        ]
    )

    summary = await read_company_signal_candidate_group_summary(
        pool,
        window_days=90,
        company_name="Acme",
        scoped_vendors=["Zendesk"],
        candidate_bucket="analyst_review",
        review_status="pending",
        min_urgency=6.0,
        min_confidence=0.25,
        min_reviews=2,
        decision_makers_only=True,
        signal_evidence_present=True,
        review_priority_band="medium",
        review_priority_reason="cross_source_corroboration",
        top_n=5,
    )

    totals_sql = pool.fetchrow.call_args[0][0]
    gap_sql = pool.fetch.call_args_list[0][0][0]
    vendor_sql = pool.fetch.call_args_list[1][0][0]
    actionable_vendor_sql = pool.fetch.call_args_list[2][0][0]
    actionable_vendor_reason_sql = pool.fetch.call_args_list[3][0][0]
    blocked_vendor_sql = pool.fetch.call_args_list[4][0][0]
    blocked_vendor_reason_sql = pool.fetch.call_args_list[5][0][0]
    near_threshold_vendor_sql = pool.fetch.call_args_list[6][0][0]
    near_threshold_reason_sql = pool.fetch.call_args_list[7][0][0]
    near_threshold_group_sql = pool.fetch.call_args_list[8][0][0]
    blocked_source_sql = pool.fetch.call_args_list[9][0][0]
    near_threshold_source_sql = pool.fetch.call_args_list[10][0][0]
    confidence_sql = pool.fetch.call_args_list[11][0][0]
    priority_sql = pool.fetch.call_args_list[12][0][0]
    pending_band_sql = pool.fetch.call_args_list[13][0][0]
    pending_reason_sql = pool.fetch.call_args_list[14][0][0]
    pending_sla_band_sql = pool.fetch.call_args_list[15][0][0]
    pending_sla_reason_sql = pool.fetch.call_args_list[16][0][0]
    oldest_pending_sql = pool.fetch.call_args_list[17][0][0]
    pending_sla_band_args = pool.fetch.call_args_list[15][0][1:]
    pending_sla_reason_args = pool.fetch.call_args_list[16][0][1:]
    oldest_pending_args = pool.fetch.call_args_list[17][0][1:]
    assert "candidate_bucket =" in totals_sql
    assert "review_status =" in totals_sql
    assert "review_count >=" in totals_sql
    assert "corroborated_confidence_score" in totals_sql
    assert "signal_evidence_count > 0" in totals_sql
    assert "decision_maker_count > 0" in totals_sql
    assert "avg_pending_age_days" in totals_sql
    assert "oldest_pending_age_days" in totals_sql
    assert "overdue_pending_groups" in totals_sql
    assert "overdue_pending_reviews" in totals_sql
    assert "near_threshold_blocked_groups" in totals_sql
    assert "near_threshold_blocked_reviews" in totals_sql
    assert "promote_now" in totals_sql
    assert "cross_source_corroboration" in totals_sql
    assert "vendor_name ILIKE" not in totals_sql
    assert "ANY(" in totals_sql
    assert "GROUP BY 1" in gap_sql
    assert "GROUP BY 1" in vendor_sql
    assert "actionable_group_count" in actionable_vendor_sql
    assert "review_priority_band IN ('promote_now', 'high', 'medium')" in actionable_vendor_sql
    assert "HAVING COUNT(*) FILTER" in actionable_vendor_sql
    assert "review_priority_reason" in actionable_vendor_reason_sql
    assert "GROUP BY 1, 2, 3" in actionable_vendor_reason_sql
    assert "blocked_group_count" in blocked_vendor_sql
    assert "= 'low'" in blocked_vendor_sql
    assert "canonical_gap_reason" in blocked_vendor_reason_sql
    assert "review_priority_reason" in blocked_vendor_reason_sql
    assert "near_threshold_group_count" in near_threshold_vendor_sql
    assert "below_threshold_near_threshold_groups" in near_threshold_vendor_sql
    assert "near_threshold_group_count" in near_threshold_reason_sql
    assert "canonical_gap_reason" in near_threshold_reason_sql
    assert "COALESCE(corroborated_confidence_score, 0) DESC" in near_threshold_group_sql
    assert "representative_source" in near_threshold_group_sql
    assert "jsonb_each_text" in blocked_source_sql
    assert "representative_source" in blocked_source_sql
    assert "jsonb_each_text" in near_threshold_source_sql
    assert "source_review_count" in near_threshold_source_sql
    assert "GROUP BY 1" in confidence_sql
    assert "review_priority_band" in priority_sql
    assert "review_priority_reason" in priority_sql
    assert "canonical_ready_review_count" in priority_sql
    assert "representative_source" in priority_sql
    assert "WHERE review_status = 'pending'" in pending_band_sql
    assert "GROUP BY 1, 2" in pending_reason_sql
    assert "review_priority_reason" in pending_reason_sql
    assert "WHERE review_status = 'pending'" in pending_reason_sql
    assert "sla_days" in pending_sla_band_sql
    assert "overdue_group_count" in pending_sla_band_sql
    assert "GROUP BY 1" in pending_sla_band_sql
    assert "sla_days" in pending_sla_reason_sql
    assert "overdue_group_count" in pending_sla_reason_sql
    assert "GROUP BY 1, 2" in pending_sla_reason_sql
    assert pending_sla_band_args
    assert pending_sla_band_args == oldest_pending_args
    assert pending_sla_reason_args[:-1] == pending_sla_band_args
    assert pending_sla_reason_args[-1] == 5
    assert "pending_age_days" in oldest_pending_sql
    assert "WHERE review_status = 'pending'" in oldest_pending_sql
    assert summary["totals"]["total_groups"] == 12
    assert summary["totals"]["avg_pending_age_days"] == 2.75
    assert summary["totals"]["actionable_pending_groups"] == 6
    assert summary["totals"]["actionable_pending_reviews"] == 17
    assert summary["totals"]["blocked_pending_groups"] == 2
    assert summary["totals"]["blocked_pending_reviews"] == 4
    assert summary["totals"]["near_threshold_blocked_groups"] == 1
    assert summary["totals"]["near_threshold_blocked_reviews"] == 2
    assert summary["totals"]["overdue_pending_groups"] == 3
    assert summary["totals"]["overdue_pending_reviews"] == 9
    assert summary["gap_reasons"][0]["gap_reason"] == "low_confidence_low_trust_source"
    assert summary["top_vendors"][0]["vendor_name"] == "Zendesk"
    assert summary["actionable_top_vendors"][0]["vendor_name"] == "Zendesk"
    assert summary["actionable_top_vendors"][0]["actionable_group_count"] == 3
    assert summary["actionable_top_vendor_reasons"][0]["review_priority_reason"] == "has_signal_evidence_and_decision_maker"
    assert summary["blocked_top_vendors"][0]["vendor_name"] == "Copper"
    assert summary["blocked_top_vendor_reasons"][0]["canonical_gap_reason"] == "low_confidence_low_trust_source"
    assert summary["blocked_source_mix"][0]["source"] == "reddit"
    assert summary["blocked_source_mix"][0]["group_count"] == 2
    assert summary["near_threshold_top_vendors"][0]["vendor_name"] == "Copper"
    assert summary["near_threshold_gap_reasons"][0]["canonical_gap_reason"] == "low_confidence_low_trust_source"
    assert summary["near_threshold_groups"][0]["vendor"] == "Copper"
    assert summary["near_threshold_groups"][0]["company"] == "coppercorp"
    assert summary["near_threshold_groups"][0]["representative_source"] == "reddit"
    assert summary["near_threshold_groups"][0]["confidence_gap_to_canonical"] == 0.19
    assert summary["near_threshold_groups"][0]["urgency_gap_to_high_intent"] is None
    assert summary["near_threshold_source_mix"][0]["source"] == "reddit"
    assert summary["near_threshold_source_mix"][0]["review_count"] == 2
    assert summary["confidence_tiers"][0]["confidence_tier"] == "high"
    assert summary["priority_groups"][0]["review_priority_band"] == "promote_now"
    assert summary["priority_groups"][0]["vendor"] == "Zendesk"
    assert summary["pending_priority_bands"][0]["review_priority_band"] == "promote_now"
    assert summary["pending_priority_reasons"][0]["review_priority_reason"] == "canonical_ready"
    assert summary["pending_sla_bands"][0]["sla_days"] == 1.0
    assert summary["pending_sla_bands"][0]["overdue_group_count"] == 1
    assert summary["pending_sla_reasons"][0]["review_priority_reason"] == "canonical_ready"
    assert summary["pending_sla_reasons"][0]["overdue_review_count"] == 3
    assert summary["oldest_pending_group"]["vendor"] == "Zendesk"
    assert summary["oldest_pending_group"]["pending_age_days"] == 5.5


@pytest.mark.asyncio
async def test_read_company_signal_review_impact_summary_aggregates_actions_and_rebuilds():
    pool = type("ImpactSummaryPool", (), {})()
    pool.fetchrow = AsyncMock(
        return_value={
            "total_actions": 6,
            "total_batches": 4,
            "distinct_vendors": 2,
            "approvals": 4,
            "suppressions": 2,
            "company_signal_creations": 3,
            "company_signal_updates": 1,
            "company_signal_deletions": 2,
            "company_signal_noops": 0,
            "rebuild_requests": 4,
            "rebuild_triggered": 3,
            "rebuild_blocked": 1,
            "rebuild_persisted_runs": 2,
            "rebuild_persisted_reports": 2,
            "rebuild_total_accounts": 9,
        }
    )
    pool.fetch = AsyncMock(
        side_effect=[
            [
                {
                    "review_scope": "bulk_group",
                    "action_count": 4,
                }
            ],
            [
                {
                    "review_priority_band": "promote_now",
                    "action_count": 3,
                    "approvals": 3,
                    "suppressions": 0,
                    "company_signal_creations": 2,
                    "company_signal_updates": 1,
                    "company_signal_deletions": 0,
                    "company_signal_noops": 0,
                    "rebuild_requests": 2,
                    "rebuild_triggered": 2,
                    "rebuild_persisted_reports": 2,
                    "rebuild_total_accounts": 9,
                }
            ],
            [
                {
                    "review_priority_band": "promote_now",
                    "review_priority_reason": "canonical_ready",
                    "action_count": 3,
                    "approvals": 3,
                    "suppressions": 0,
                    "company_signal_creations": 2,
                    "company_signal_updates": 1,
                    "company_signal_deletions": 0,
                    "company_signal_noops": 0,
                    "rebuild_requests": 2,
                    "rebuild_triggered": 2,
                    "rebuild_persisted_reports": 2,
                    "rebuild_total_accounts": 9,
                }
            ],
            [
                {
                    "vendor_name": "Zendesk",
                    "action_count": 5,
                    "approvals": 3,
                    "suppressions": 2,
                    "company_signal_creations": 2,
                    "company_signal_updates": 1,
                    "company_signal_deletions": 2,
                    "rebuild_requests": 3,
                    "rebuild_triggered": 2,
                    "rebuild_persisted_reports": 2,
                    "rebuild_total_accounts": 9,
                }
            ],
            [
                {
                    "vendor_name": "Zendesk",
                    "review_priority_band": "promote_now",
                    "review_priority_reason": "canonical_ready",
                    "action_count": 3,
                    "approvals": 3,
                    "suppressions": 0,
                    "company_signal_creations": 2,
                    "company_signal_updates": 1,
                    "company_signal_deletions": 0,
                    "company_signal_noops": 0,
                    "rebuild_requests": 2,
                    "rebuild_triggered": 2,
                    "rebuild_persisted_reports": 2,
                    "rebuild_total_accounts": 9,
                }
            ],
        ]
    )

    summary = await read_company_signal_review_impact_summary(
        pool,
        window_days=30,
        scoped_vendors=["Zendesk"],
        review_action="approved",
        review_priority_band="promote_now",
        review_priority_reason="canonical_ready",
        top_n=5,
    )

    totals_sql = pool.fetchrow.call_args[0][0]
    scopes_sql = pool.fetch.call_args_list[0][0][0]
    priority_sql = pool.fetch.call_args_list[1][0][0]
    priority_reason_sql = pool.fetch.call_args_list[2][0][0]
    vendors_sql = pool.fetch.call_args_list[3][0][0]
    vendor_reasons_sql = pool.fetch.call_args_list[4][0][0]
    assert "review_action =" in totals_sql
    assert "vendor_name = ANY(" in totals_sql
    assert "review_priority_band" in totals_sql
    assert "review_priority_reason" in totals_sql
    assert "COUNT(DISTINCT review_batch_id)" in totals_sql
    assert "FROM b2b_company_signal_review_events" in totals_sql
    assert "GROUP BY 1" in scopes_sql
    assert "review_priority_band" in priority_sql
    assert "band_rebuilds" in priority_sql
    assert "review_priority_reason" in priority_reason_sql
    assert "reason_rebuilds" in priority_reason_sql
    assert "vendor_actions" in vendors_sql
    assert "vendor_rebuilds" in vendors_sql
    assert "vendor_reason_actions" in vendor_reasons_sql
    assert "vendor_reason_rebuilds" in vendor_reasons_sql
    assert summary["totals"]["total_actions"] == 6
    assert summary["totals"]["company_signal_effect_rate"] == 1.0
    assert summary["totals"]["company_signal_creation_rate"] == 0.5
    assert summary["totals"]["rebuild_trigger_rate"] == 0.75
    assert summary["totals"]["avg_rebuild_reports_per_triggered"] == 2 / 3
    assert summary["totals"]["avg_rebuild_accounts_per_triggered"] == 3.0
    assert summary["scopes"][0]["review_scope"] == "bulk_group"
    assert summary["priority_bands"][0]["review_priority_band"] == "promote_now"
    assert summary["priority_bands"][0]["company_signal_effect_rate"] == 1.0
    assert summary["priority_bands"][0]["company_signal_creation_rate"] == 2 / 3
    assert summary["priority_bands"][0]["rebuild_trigger_rate"] == 1.0
    assert summary["priority_bands"][0]["avg_rebuild_accounts_per_triggered"] == 4.5
    assert summary["priority_reasons"][0]["review_priority_reason"] == "canonical_ready"
    assert summary["priority_reasons"][0]["company_signal_effect_rate"] == 1.0
    assert summary["top_vendors"][0]["vendor_name"] == "Zendesk"
    assert summary["top_vendors"][0]["company_signal_effect_rate"] == 1.0
    assert summary["top_vendors"][0]["rebuild_trigger_rate"] == 2 / 3
    assert summary["top_vendor_reasons"][0]["vendor_name"] == "Zendesk"
    assert summary["top_vendor_reasons"][0]["review_priority_reason"] == "canonical_ready"
    assert summary["top_vendor_reasons"][0]["company_signal_effect_rate"] == 1.0
