"""Vendor Workspace dashboard ProductClaim aggregator.

Phase 10 Patch 2a (UI first; reports inherit). Produces ProductClaim
envelopes for the vendor-detail rate cards so VendorDetail.tsx can
gate renders on render_allowed instead of pretending a 50% rate from
a denominator of 2 reviews is meaningful.

Initial scope is intentionally narrow: one claim_type
(decision_maker_churn_rate) so the contract proves itself against
real data before the surface fan-out. Patches 2b/2c add the API
endpoint and the React-side migration. Subsequent patches add
price_complaint_rate, weakness_theme, strength_theme, and the
remaining VENDOR-scope claims.

Read-only. The aggregator computes ProductClaim envelopes at request
time from existing tables (b2b_reviews + b2b_review_vendor_mentions);
no migration, no persistence layer in Patch 2a. The persistence
decision (open decision #1 in the design doc) waits on the canary
review.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from .product_claim import (
    ClaimGatePolicy,
    ClaimScope,
    ProductClaim,
    build_product_claim,
    register_policy,
)


logger = logging.getLogger("atlas.b2b.vendor_dashboard_claims")


# Decision-maker churn rate is a rate claim: the gate must enforce a
# minimum denominator so a "100% churn rate from 1 DM" never publishes.
_DM_CHURN_RATE_POLICY = ClaimGatePolicy(
    is_rate_claim=True,
    min_denominator_for_rate=10,
    # Tighter coverage thresholds than the default to reflect that a
    # rate claim needs a wider sample to meaningfully grade HIGH.
    high_confidence_min_supporting=15,
    high_confidence_min_witnesses=10,
    medium_confidence_min_supporting=5,
    medium_confidence_min_witnesses=3,
)
register_policy(
    ClaimScope.VENDOR, "decision_maker_churn_rate", _DM_CHURN_RATE_POLICY
)


_DM_CHURN_QUERY = """
WITH dm_rows AS (
    SELECT
        r.id AS review_id,
        r.reviewer_name,
        (r.enrichment->'churn_signals'->>'intent_to_leave')::boolean AS intent_to_leave,
        (r.enrichment->'reviewer_context'->>'decision_maker')::boolean AS decision_maker,
        COALESCE(NULLIF(r.enrichment->>'enrichment_schema_version', ''), '0')::int AS schema_version,
        r.enrichment ? 'phrase_metadata' AS has_phrase_metadata
    FROM b2b_reviews r
    JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id
    WHERE vm.vendor_name = $1
      AND r.enrichment_status = 'enriched'
      AND COALESCE(r.reviewed_at, r.imported_at, r.enriched_at)
          > NOW() - make_interval(days => $2::int)
)
SELECT
    -- Numerator: DMs signaling intent_to_leave.
    count(*) FILTER (WHERE decision_maker = true AND intent_to_leave = true) AS dm_churning,
    -- Denominator: total DMs in the window for this vendor.
    count(*) FILTER (WHERE decision_maker = true) AS dm_total,
    -- Direct evidence: DM-churn rows backed by grounded phrase metadata.
    count(*) FILTER (
        WHERE decision_maker = true
          AND intent_to_leave = true
          AND has_phrase_metadata = true
    ) AS dm_churning_direct,
    -- Distinct reviewers among the churning DMs (witness diversity).
    count(DISTINCT reviewer_name) FILTER (
        WHERE decision_maker = true AND intent_to_leave = true
    ) AS dm_churning_witnesses,
    -- Contradictions: DMs explicitly NOT churning (intent_to_leave=false)
    -- in the same window. Higher counts indicate split sentiment.
    count(*) FILTER (
        WHERE decision_maker = true AND intent_to_leave = false
    ) AS dm_not_churning,
    -- Backing review_ids for evidence_links provenance.
    array_agg(review_id) FILTER (
        WHERE decision_maker = true AND intent_to_leave = true
    ) AS churning_review_ids,
    -- Whether ANY backing churn-DM row has phrase metadata. False -> UNVERIFIED.
    bool_or(has_phrase_metadata) FILTER (
        WHERE decision_maker = true AND intent_to_leave = true
    ) AS has_grounded_evidence
FROM dm_rows
"""


async def aggregate_dm_churn_rate_claim(
    pool,
    *,
    vendor_name: str,
    as_of_date: date,
    analysis_window_days: int,
) -> ProductClaim | None:
    """Build the VENDOR.decision_maker_churn_rate ProductClaim for a
    vendor, or None if the vendor has no DM data at all.

    The contract is: render_allowed reflects whether the rate is
    SAFE to display. A rate from a denominator below 10 is not
    renderable; the dashboard should show 'Insufficient evidence'
    instead of a percentage. Reports never publish unless the rate
    also passes the (usable + high/medium confidence) gate.
    """
    row = await pool.fetchrow(_DM_CHURN_QUERY, vendor_name, int(analysis_window_days))
    if row is None:
        return None
    dm_total = int(row["dm_total"] or 0)
    if dm_total <= 0:
        # Vendor has no decision-maker reviews in window. Nothing to
        # render; signal absence to the caller.
        return None

    dm_churning = int(row["dm_churning"] or 0)
    dm_churning_direct = int(row["dm_churning_direct"] or 0)
    dm_churning_witnesses = int(row["dm_churning_witnesses"] or 0)
    dm_not_churning = int(row["dm_not_churning"] or 0)
    churning_review_ids = list(row["churning_review_ids"] or [])
    has_grounded = bool(row["has_grounded_evidence"]) if row["has_grounded_evidence"] is not None else False

    # If there are zero churning DMs, supporting_count=0 means
    # INSUFFICIENT posture -- the dashboard correctly shows "no
    # churn signal" rather than implying a 0% rate is meaningful.
    rate = dm_churning / dm_total if dm_total > 0 else 0.0

    return build_product_claim(
        claim_scope=ClaimScope.VENDOR,
        claim_type="decision_maker_churn_rate",
        # claim_key is a stable label; we don't need disambiguation
        # within (VENDOR, decision_maker_churn_rate, vendor_name)
        # because there is exactly one such claim per vendor.
        claim_key="decision_maker_churn",
        claim_text=(
            f"{int(round(rate * 100))}% of decision-makers signaled "
            f"intent to leave"
        ),
        target_entity=vendor_name,
        secondary_target=None,
        supporting_count=dm_churning,
        direct_evidence_count=dm_churning_direct,
        witness_count=dm_churning_witnesses,
        contradiction_count=dm_not_churning,
        denominator=dm_total,
        sample_size=dm_total,
        has_grounded_evidence=has_grounded if dm_churning > 0 else True,
        evidence_links=tuple(str(rid) for rid in churning_review_ids),
        contradicting_links=(),
        as_of_date=as_of_date,
        analysis_window_days=analysis_window_days,
    )


__all__ = [
    "aggregate_dm_churn_rate_claim",
]
