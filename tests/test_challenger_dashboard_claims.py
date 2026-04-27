"""Backend contract tests for Challenger ProductClaim aggregation."""

from __future__ import annotations

from datetime import date
from uuid import uuid4

import pytest

from atlas_brain.services.b2b.challenger_dashboard_claims import (
    DIRECT_DISPLACEMENT_CLAIM_TYPE,
    aggregate_direct_displacement_claim,
    aggregate_direct_displacement_claims_for_challenger,
)
from atlas_brain.services.b2b.product_claim import (
    ClaimScope,
    EvidencePosture,
    SuppressionReason,
)


class FakePool:
    def __init__(self, *, fetchrow_result=None, fetch_results=None):
        self.fetchrow_result = fetchrow_result
        self.fetch_results = list(fetch_results or [])
        self.fetchrow_calls = []
        self.fetch_calls = []

    async def fetchrow(self, sql, *args):
        self.fetchrow_calls.append((sql, args))
        return self.fetchrow_result

    async def fetch(self, sql, *args):
        self.fetch_calls.append((sql, args))
        return self.fetch_results


def _row(**overrides):
    base = {
        "incumbent": "Google Workspace",
        "supporting_count": 3,
        "direct_evidence_count": 3,
        "witness_count": 2,
        "contradiction_count": 0,
        "evidence_links": [str(uuid4()), str(uuid4())],
        "contradicting_links": [],
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_direct_displacement_pair_flips_incumbent_query_to_challenger_claim():
    pool = FakePool(fetchrow_result=_row())

    claim = await aggregate_direct_displacement_claim(
        pool,
        challenger="Microsoft 365",
        incumbent="Google Workspace",
        as_of_date=date(2026, 4, 26),
        analysis_window_days=90,
    )

    assert pool.fetchrow_calls
    sql, args = pool.fetchrow_calls[0]
    assert args == ("Google Workspace", "Microsoft 365", date(2026, 4, 26), 90)
    assert "claim_type = 'displacement_proof_to_competitor'" in sql
    assert "lower(ec.vendor_name) = lower($1)" in sql
    assert "lower(ec.secondary_target) = lower($2)" in sql
    assert "LEFT JOIN b2b_reviews r ON r.id = ec.source_review_id" in sql
    assert "NULLIF(lower(btrim(r.reviewer_name)), '') AS reviewer_key" in sql
    assert "count(DISTINCT reviewer_key)" in sql
    assert "count(DISTINCT source_review_id) FILTER" not in sql
    assert "lower(ec.vendor_name) = lower($2)" in sql
    assert "lower(ec.secondary_target) = lower($1)" in sql
    assert "PARTITION BY" in sql
    assert "COALESCE(ec.source_excerpt_fingerprint, ec.id::text)" in sql

    assert claim.claim_scope == ClaimScope.COMPETITOR_PAIR
    assert claim.claim_type == DIRECT_DISPLACEMENT_CLAIM_TYPE
    assert claim.claim_key == "incumbent:Google Workspace"
    assert claim.target_entity == "Microsoft 365"
    assert claim.secondary_target == "Google Workspace"
    assert claim.supporting_count == 3
    assert claim.direct_evidence_count == 3
    assert claim.witness_count == 2
    assert claim.render_allowed is True
    assert claim.report_allowed is True


@pytest.mark.asyncio
async def test_direct_displacement_pair_without_rows_returns_suppressed_claim():
    pool = FakePool(fetchrow_result=None)

    claim = await aggregate_direct_displacement_claim(
        pool,
        challenger="Microsoft 365",
        incumbent="Google Workspace",
        as_of_date=date(2026, 4, 26),
        analysis_window_days=90,
    )

    assert claim.target_entity == "Microsoft 365"
    assert claim.secondary_target == "Google Workspace"
    assert claim.supporting_count == 0
    assert claim.direct_evidence_count == 0
    assert claim.evidence_posture == EvidencePosture.INSUFFICIENT
    assert claim.render_allowed is False
    assert claim.report_allowed is False
    assert claim.suppression_reason == SuppressionReason.INSUFFICIENT_SUPPORTING_COUNT


@pytest.mark.asyncio
async def test_direct_displacement_list_returns_challenger_centric_rows():
    pool = FakePool(
        fetch_results=[
            _row(incumbent="Google Workspace", supporting_count=3, direct_evidence_count=3, witness_count=2),
            _row(incumbent="Slack", supporting_count=1, direct_evidence_count=1, witness_count=1),
        ]
    )

    rows = await aggregate_direct_displacement_claims_for_challenger(
        pool,
        challenger="Microsoft 365",
        as_of_date=date(2026, 4, 26),
        analysis_window_days=90,
        limit=10,
    )

    assert pool.fetch_calls
    sql, args = pool.fetch_calls[0]
    assert args == ("Microsoft 365", date(2026, 4, 26), 90, 10)
    assert "lower(ec.secondary_target) = lower($1)" in sql
    assert "LEFT JOIN b2b_reviews r ON r.id = ec.source_review_id" in sql
    assert "count(DISTINCT reviewer_key)" in sql
    assert "GROUP BY vendor_name" in sql
    assert "lower(ec.vendor_name) = lower($1)" in sql

    assert [r.incumbent for r in rows] == ["Google Workspace", "Slack"]
    assert all(r.challenger == "Microsoft 365" for r in rows)
    assert rows[0].claim.target_entity == "Microsoft 365"
    assert rows[0].claim.secondary_target == "Google Workspace"
    assert rows[0].claim.report_allowed is True
    assert rows[1].claim.target_entity == "Microsoft 365"
    assert rows[1].claim.secondary_target == "Slack"
    assert rows[1].claim.render_allowed is True
    assert rows[1].claim.report_allowed is False


@pytest.mark.asyncio
async def test_direct_displacement_repeated_reviews_do_not_publish_without_distinct_reviewers():
    pool = FakePool(
        fetchrow_result=_row(
            supporting_count=3,
            direct_evidence_count=3,
            witness_count=1,
            evidence_links=[str(uuid4()), str(uuid4()), str(uuid4())],
        )
    )

    claim = await aggregate_direct_displacement_claim(
        pool,
        challenger="Microsoft 365",
        incumbent="Google Workspace",
        as_of_date=date(2026, 4, 26),
        analysis_window_days=90,
    )

    assert claim.supporting_count == 3
    assert claim.witness_count == 1
    assert claim.render_allowed is True
    assert claim.report_allowed is False
    assert claim.suppression_reason == SuppressionReason.LOW_CONFIDENCE


@pytest.mark.asyncio
async def test_direct_displacement_policy_renders_single_valid_witness_but_does_not_publish():
    pool = FakePool(
        fetchrow_result=_row(
            supporting_count=1,
            direct_evidence_count=1,
            witness_count=1,
            evidence_links=[str(uuid4())],
        )
    )

    claim = await aggregate_direct_displacement_claim(
        pool,
        challenger="Microsoft 365",
        incumbent="Google Workspace",
        as_of_date=date(2026, 4, 26),
        analysis_window_days=90,
    )

    assert claim.render_allowed is True
    assert claim.report_allowed is False
    assert claim.suppression_reason == SuppressionReason.LOW_CONFIDENCE


@pytest.mark.asyncio
async def test_direct_displacement_inverse_direction_counts_as_contradiction():
    pool = FakePool(
        fetchrow_result=_row(
            supporting_count=5,
            direct_evidence_count=5,
            witness_count=3,
            contradiction_count=3,
            contradicting_links=["11111111-1111-4111-8111-111111111111"],
        )
    )

    claim = await aggregate_direct_displacement_claim(
        pool,
        challenger="Microsoft 365",
        incumbent="Google Workspace",
        as_of_date=date(2026, 4, 26),
        analysis_window_days=90,
    )

    assert claim.contradiction_count == 3
    assert claim.contradicting_links == ("11111111-1111-4111-8111-111111111111",)
    assert claim.render_allowed is True
    assert claim.report_allowed is False
    assert claim.suppression_reason == SuppressionReason.CONTRADICTORY_EVIDENCE
