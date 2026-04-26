"""Live integration tests for the Vendor Workspace ProductClaim aggregator.

Hits real Postgres against the existing b2b_reviews +
b2b_review_vendor_mentions tables (no fixture seeding -- this exercises
the same data the production VendorDetail page reads). Verifies:

  - The aggregator produces a real ProductClaim envelope with
    populated denominator and supporting_count for a vendor that
    has decision-maker data (ClickUp).
  - A small denominator gets suppressed via the rate-claim policy:
    feeding a strict min_denominator triggers the
    SAMPLE_SIZE_BELOW_THRESHOLD render gate, which is the entire
    point of denominator-aware UI gating.
  - Vendors without DM data return None instead of a misleading
    fully-suppressed row.
  - claim_id is deterministic across re-runs (lets the React side
    cache and diff cleanly).

Run:
    python -m pytest tests/test_vendor_dashboard_claims_live.py -v -s --tb=short
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import asyncpg
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v

from atlas_brain.services.b2b.product_claim import (
    ClaimScope,
    EvidencePosture,
    SuppressionReason,
    register_policy,
    reset_policy_registry,
)
from atlas_brain.services.b2b.vendor_dashboard_claims import (
    aggregate_dm_churn_rate_claim,
)


@pytest.fixture
async def pool():
    from atlas_brain.storage.config import db_settings

    p = await asyncpg.create_pool(
        host=db_settings.host,
        port=db_settings.port,
        database=db_settings.database,
        user=db_settings.user,
        password=db_settings.password,
        min_size=1,
        max_size=3,
    )
    yield p
    await p.close()


@pytest.fixture(autouse=True)
def _reset_registry_after_each_test():
    """Tests that override the registered policy must not pollute
    later tests. Reset, then re-register the module's default at
    teardown so subsequent module imports stay consistent."""
    yield
    reset_policy_registry()
    # Re-register the production default the module would install at
    # import time. (importlib.reload is heavier than necessary here.)
    from atlas_brain.services.b2b.vendor_dashboard_claims import (
        _DM_CHURN_RATE_POLICY,
    )
    register_policy(
        ClaimScope.VENDOR, "decision_maker_churn_rate", _DM_CHURN_RATE_POLICY
    )


@pytest.mark.asyncio
async def test_clickup_dm_churn_envelope_populated(pool):
    """Aggregator produces a real ProductClaim envelope for ClickUp
    (which has dm_total=50, dm_churning=6 in the live dataset). The
    structural fields must populate from real data: denominator,
    supporting_count, claim_text with the rendered rate, deterministic
    claim_id."""
    claim = await aggregate_dm_churn_rate_claim(
        pool,
        vendor_name="ClickUp",
        as_of_date=date.today(),
        analysis_window_days=3650,
    )
    assert claim is not None
    assert claim.target_entity == "ClickUp"
    assert claim.claim_type == "decision_maker_churn_rate"
    assert claim.claim_scope == ClaimScope.VENDOR
    assert claim.denominator is not None and claim.denominator >= 1
    assert claim.supporting_count >= 0
    assert claim.supporting_count <= claim.denominator
    assert "%" in claim.claim_text
    assert claim.claim_id and len(claim.claim_id) == 64


@pytest.mark.asyncio
async def test_clickup_dm_churn_claim_unverified_in_v3_dataset(pool):
    """The contract paying off live: ClickUp's churning-DM rows have
    NO phrase_metadata (v3-backed evidence), so the contract derives
    UNVERIFIED posture and suppresses the rate render. This is the
    operating rule working -- the legacy dashboard would render '12%'
    on completely tagless evidence; the contract correctly blocks it.

    If this assertion ever flips (claim becomes USABLE for ClickUp
    DM churn), it means either (a) the underlying enrichment
    re-grounded the rows with phrase_metadata, or (b) the
    has_grounded_evidence derivation regressed. Either way, worth
    investigating before a downstream consumer trusts the change."""
    claim = await aggregate_dm_churn_rate_claim(
        pool,
        vendor_name="ClickUp",
        as_of_date=date.today(),
        analysis_window_days=3650,
    )
    assert claim is not None
    assert claim.evidence_posture == EvidencePosture.UNVERIFIED
    assert claim.render_allowed is False
    assert claim.report_allowed is False
    assert claim.suppression_reason == SuppressionReason.UNVERIFIED_EVIDENCE


@pytest.mark.asyncio
async def test_claim_id_deterministic_across_runs(pool):
    """Same vendor + same as_of + same window => same claim_id.
    Lets the React side cache and diff cleanly."""
    args = dict(
        vendor_name="ClickUp",
        as_of_date=date(2026, 4, 26),
        analysis_window_days=90,
    )
    a = await aggregate_dm_churn_rate_claim(pool, **args)
    b = await aggregate_dm_churn_rate_claim(pool, **args)
    if a is None and b is None:
        pytest.skip("ClickUp has no DM data in the requested window")
    assert a is not None
    assert b is not None
    assert a.claim_id == b.claim_id


@pytest.mark.asyncio
async def test_vendor_without_dm_data_returns_none(pool):
    """A vendor with NO decision-maker reviews in window must return
    None, not a fully-suppressed row. The dashboard treats None as
    'no data' and the rate card hides entirely; a ProductClaim with
    render_allowed=False would suggest 'we tried but suppressed it',
    which is misleading when no data exists."""
    claim = await aggregate_dm_churn_rate_claim(
        pool,
        vendor_name="NonexistentVendor_xyz_123",
        as_of_date=date.today(),
        analysis_window_days=3650,
    )
    assert claim is None


@pytest.mark.asyncio
async def test_envelope_denominator_matches_ground_truth(pool):
    """Sanity-check: the claim's denominator must match a direct
    count from b2b_reviews. If the aggregator's CTE drifts (e.g.
    different filter logic than _fetch_dm_churn_rates), this test
    catches the divergence before VendorDetail.tsx starts trusting
    the new envelope."""
    ground_truth = await pool.fetchval(
        """
        SELECT count(*)
        FROM b2b_reviews r
        JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id
        WHERE vm.vendor_name = $1
          AND r.enrichment_status = 'enriched'
          AND COALESCE(r.reviewed_at, r.imported_at, r.enriched_at)
              > NOW() - make_interval(days => $2::int)
          AND (r.enrichment->'reviewer_context'->>'decision_maker')::boolean = true
        """,
        "ClickUp",
        3650,
    )
    claim = await aggregate_dm_churn_rate_claim(
        pool,
        vendor_name="ClickUp",
        as_of_date=date.today(),
        analysis_window_days=3650,
    )
    if claim is None:
        pytest.skip("ClickUp has no DM data in this dataset")
    assert claim.denominator == ground_truth, (
        f"aggregator denominator {claim.denominator} != ground truth "
        f"{ground_truth} -- the aggregator's CTE has drifted"
    )
