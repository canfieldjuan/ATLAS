from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod
from atlas_brain.autonomous.tasks._b2b_shared import read_company_signal_candidates


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
        "materialization_run_id": "run-123",
        "first_seen_at": datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc),
        "last_seen_at": datetime(2026, 4, 10, 12, 5, tzinfo=timezone.utc),
    }
    row.update(overrides)
    return row


class FakePool:
    def __init__(self, rows):
        self.fetch = AsyncMock(return_value=rows)


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
        scoped_vendors=["Zendesk"],
        candidate_bucket="analyst_review",
        limit=25,
    )

    sql = pool.fetch.call_args[0][0]
    assert "candidate_bucket =" in sql
    assert "ANY(" in sql
    assert "LIMIT $4" in sql
    assert len(results) == 1
    assert results[0]["company"] == "Acme Corp"
    assert results[0]["candidate_bucket"] == "analyst_review"
    assert results[0]["confidence_score"] == 0.26


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
