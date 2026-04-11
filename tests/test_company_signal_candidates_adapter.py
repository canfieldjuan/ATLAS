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
    assert "ANY(" in primary_sql
    assert "WHERE id = ANY($1::uuid[])" in support_sql
    assert len(results) == 1
    assert results[0]["group_id"] == str(group_row["id"])
    assert results[0]["display_company"] == "Acme Corp"
    assert results[0]["review_count"] == 3
    assert results[0]["corroborated_confidence_score"] == 0.68
    assert results[0]["supporting_reviews"][0]["quote_excerpt"] == "Renewal risk keeps coming up"
