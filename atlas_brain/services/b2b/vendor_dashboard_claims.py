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

Eligibility parity: the query reuses _eligible_review_filters from
the legacy intelligence pipeline so the new ProductClaim envelope
sees the same row population the existing dashboard rate computation
sees -- duplicate suppression, source allowlist, jsonld_aggregate
exclusion, applied data-correction filters all included.
"""

from __future__ import annotations

import logging
from datetime import date

from ...autonomous.tasks._b2b_shared import (
    _eligible_review_filters,
    _intelligence_source_allowlist,
)
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

# Price complaint rate: same rate-claim shape as DM churn. The
# denominator is "all eligible reviews for the vendor" (not just DMs),
# so meaningful rates start to mean something at a higher absolute
# review count. Using the same min_denominator_for_rate=10 as a
# floor; canary review can tighten it once we see rate distributions.
_PRICE_COMPLAINT_RATE_POLICY = ClaimGatePolicy(
    is_rate_claim=True,
    min_denominator_for_rate=10,
    high_confidence_min_supporting=15,
    high_confidence_min_witnesses=10,
    medium_confidence_min_supporting=5,
    medium_confidence_min_witnesses=3,
)
register_policy(
    ClaimScope.VENDOR, "price_complaint_rate", _PRICE_COMPLAINT_RATE_POLICY
)


def _build_dm_churn_query() -> str:
    """Build the DM-churn aggregation query using the shared eligible-
    review filter so the ProductClaim denominator matches what the
    legacy dashboard rate would compute. Vendor name is $1, window
    days is $2, source allowlist is $3."""
    eligible = _eligible_review_filters(
        window_param=2,
        source_param=3,
        alias="r",
        vendor_expr="vm.vendor_name",
    )
    return f"""
WITH dm_rows AS (
    SELECT
        r.id AS review_id,
        r.reviewer_name,
        (r.enrichment->'churn_signals'->>'intent_to_leave')::boolean AS intent_to_leave,
        (r.enrichment->'reviewer_context'->>'decision_maker')::boolean AS decision_maker,
        -- Direct evidence test for THIS row: does the row carry at
        -- least one phrase tagged as grounded? Stricter than just
        -- "v4 existence" because v4 rows can still have all phrases
        -- ungrounded. v2 of this check (Patch 3+) joins to
        -- b2b_evidence_claims (Phase 9 shadow table) so direct
        -- evidence requires that the GROUNDED PHRASE specifically
        -- supports a churn-relevant claim_type for this vendor, not
        -- just that the row has any grounded phrase. The current row-
        -- level grounding test is a v1 approximation that overcounts
        -- when a row has grounded phrases unrelated to DM churn (e.g.
        -- a positive UX phrase grounding next to a v3-tagged churn
        -- signal). Replace once Phase 9 capture is steady-state.
        EXISTS (
            SELECT 1
            FROM jsonb_array_elements(
                COALESCE(r.enrichment->'phrase_metadata', '[]'::jsonb)
            ) p
            WHERE p->>'grounding_status' = 'grounded'
        ) AS has_grounded_phrase
    FROM b2b_reviews r
    JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id
    WHERE vm.vendor_name = $1
      AND {eligible}
)
SELECT
    -- Numerator: DMs signaling intent_to_leave.
    count(*) FILTER (WHERE decision_maker = true AND intent_to_leave = true) AS dm_churning,
    -- Denominator: total DMs in the window for this vendor.
    count(*) FILTER (WHERE decision_maker = true) AS dm_total,
    -- Direct evidence: churning DM rows with at least one grounded
    -- phrase. Tighter than the prior "any phrase_metadata exists"
    -- check; ties direct evidence to the row's actual grounding,
    -- not just to v4 existence.
    count(*) FILTER (
        WHERE decision_maker = true
          AND intent_to_leave = true
          AND has_grounded_phrase = true
    ) AS dm_churning_direct,
    -- Distinct reviewers among the churning DMs (witness diversity).
    count(DISTINCT reviewer_name) FILTER (
        WHERE decision_maker = true AND intent_to_leave = true
    ) AS dm_churning_witnesses,
    -- Backing review_ids for evidence_links provenance.
    array_agg(review_id) FILTER (
        WHERE decision_maker = true AND intent_to_leave = true
    ) AS churning_review_ids
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

    For rate claims, contradiction_count is 0 by design. The legacy
    "DMs NOT signaling intent_to_leave" measure conflates denominator
    context with contradicting evidence: those rows are exactly what
    the rate is computed against, not a refutation of it. A meaningful
    contradiction would be evidence that the rate computation itself
    is wrong (sample skew, signal-flip from a prior period, etc.).
    Defining that precisely waits on a real product need.
    """
    sources = _intelligence_source_allowlist()
    row = await pool.fetchrow(
        _build_dm_churn_query(),
        vendor_name,
        int(analysis_window_days),
        sources,
    )
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
    churning_review_ids = list(row["churning_review_ids"] or [])
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
        # See docstring: rate-claim contradictions are not non-numerator
        # rows. Set to 0 until a defensible rate-specific definition
        # is in place.
        contradiction_count=0,
        denominator=dm_total,
        sample_size=dm_total,
        # Grounded iff at least one churning-DM row has a grounded
        # phrase. False -> UNVERIFIED via the validator's path. When
        # the numerator is 0 we have nothing to ground in either
        # direction; default True so the claim renders the "0%"
        # rate honestly with no false UNVERIFIED label.
        has_grounded_evidence=(dm_churning_direct > 0) if dm_churning > 0 else True,
        evidence_links=tuple(str(rid) for rid in churning_review_ids),
        contradicting_links=(),
        as_of_date=as_of_date,
        analysis_window_days=analysis_window_days,
    )


def _build_price_complaint_query() -> str:
    """Build the price-complaint aggregation query using the shared
    eligible-review filter so the ProductClaim denominator matches the
    legacy churn-signal aggregator. Vendor name is $1, window days is
    $2, source allowlist is $3.

    Numerator semantic: a 'price complaint' is a row where the
    enrichment tagged pain_category='pricing' OR set the explicit
    contract_context.price_complaint flag. The legacy
    _fetch_price_complaint_rates uses max(pricing_rate,
    explicit_complaint_rate); we count the UNION exactly once via
    SQL OR, which is closer to the underlying claim ('this review
    contains a price complaint') than an overlapping max."""
    eligible = _eligible_review_filters(
        window_param=2,
        source_param=3,
        alias="r",
        vendor_expr="vm.vendor_name",
    )
    return f"""
WITH review_rows AS (
    SELECT
        r.id AS review_id,
        r.reviewer_name,
        (
            r.enrichment->>'pain_category' = 'pricing'
            OR (r.enrichment->'contract_context'->>'price_complaint')::boolean = true
        ) AS is_price_complaint,
        EXISTS (
            SELECT 1
            FROM jsonb_array_elements(
                COALESCE(r.enrichment->'phrase_metadata', '[]'::jsonb)
            ) p
            WHERE p->>'grounding_status' = 'grounded'
        ) AS has_grounded_phrase
    FROM b2b_reviews r
    JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id
    WHERE vm.vendor_name = $1
      AND {eligible}
)
SELECT
    -- Numerator: rows containing a price complaint.
    count(*) FILTER (WHERE is_price_complaint = true) AS complaint_count,
    -- Denominator: all eligible reviews for the vendor in window.
    count(*) AS reviews_total,
    -- Direct evidence: complaint rows with at least one grounded phrase.
    -- v1 approximation -- see Open Decision #6 in the design doc; the
    -- v2 path joins to b2b_evidence_claims for true claim-level
    -- lineage once Phase 9 capture is steady-state.
    count(*) FILTER (
        WHERE is_price_complaint = true
          AND has_grounded_phrase = true
    ) AS complaint_direct,
    -- Distinct reviewers among the complaining set.
    count(DISTINCT reviewer_name) FILTER (
        WHERE is_price_complaint = true
    ) AS complaint_witnesses,
    -- Backing review_ids for evidence_links provenance.
    array_agg(review_id) FILTER (
        WHERE is_price_complaint = true
    ) AS complaint_review_ids
FROM review_rows
"""


async def aggregate_price_complaint_rate_claim(
    pool,
    *,
    vendor_name: str,
    as_of_date: date,
    analysis_window_days: int,
) -> ProductClaim | None:
    """Build the VENDOR.price_complaint_rate ProductClaim for a
    vendor, or None if the vendor has no eligible reviews at all.

    Same contract semantics as DM churn: render_allowed reflects
    whether the rate is SAFE to display. A rate from a denominator
    below 10 is not renderable; the dashboard should show
    'Insufficient evidence' instead of a percentage. Reports never
    publish unless the rate also passes the (usable + high/medium
    confidence) gate.

    Rate-claim contradictions are 0 by design (see DM-churn docstring).
    Reviews without price complaints are denominator context, not
    contradiction.
    """
    sources = _intelligence_source_allowlist()
    row = await pool.fetchrow(
        _build_price_complaint_query(),
        vendor_name,
        int(analysis_window_days),
        sources,
    )
    if row is None:
        return None
    reviews_total = int(row["reviews_total"] or 0)
    if reviews_total <= 0:
        return None

    complaint_count = int(row["complaint_count"] or 0)
    complaint_direct = int(row["complaint_direct"] or 0)
    complaint_witnesses = int(row["complaint_witnesses"] or 0)
    complaint_review_ids = list(row["complaint_review_ids"] or [])
    rate = complaint_count / reviews_total if reviews_total > 0 else 0.0

    return build_product_claim(
        claim_scope=ClaimScope.VENDOR,
        claim_type="price_complaint_rate",
        # Stable label; one such claim per vendor.
        claim_key="price_complaint",
        claim_text=(
            f"{int(round(rate * 100))}% of reviews contain a price complaint"
        ),
        target_entity=vendor_name,
        secondary_target=None,
        supporting_count=complaint_count,
        direct_evidence_count=complaint_direct,
        witness_count=complaint_witnesses,
        contradiction_count=0,
        denominator=reviews_total,
        sample_size=reviews_total,
        has_grounded_evidence=(
            complaint_direct > 0 if complaint_count > 0 else True
        ),
        evidence_links=tuple(str(rid) for rid in complaint_review_ids),
        contradicting_links=(),
        as_of_date=as_of_date,
        analysis_window_days=analysis_window_days,
    )


__all__ = [
    "aggregate_dm_churn_rate_claim",
    "aggregate_price_complaint_rate_claim",
]
