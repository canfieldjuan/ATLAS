"""COMPETITOR_PAIR ProductClaim aggregators for Challenger UI.

Phase 10 Patch 5a1 starts with direct displacement only. The raw
EvidenceClaim substrate is incumbent-centric:

    vendor_name = incumbent losing share
    claim_type = displacement_proof_to_competitor
    secondary_target = challenger winning consideration

The dashboard row is challenger-centric, so the ProductClaim envelope
flips the public pair:

    target_entity = challenger
    secondary_target = incumbent

The query direction stays encoded in claim_key so a later inverse
`displacement_proof_from_competitor` slice cannot collide or invert the
pair accidentally.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from .product_claim import (
    ClaimGatePolicy,
    ClaimScope,
    ProductClaim,
    build_product_claim,
    register_policy,
)


DIRECT_DISPLACEMENT_CLAIM_TYPE = "direct_displacement"


_DIRECT_DISPLACEMENT_POLICY = ClaimGatePolicy(
    # A single validated displacement witness is safe to render in the UI,
    # but reports/campaigns still need corroboration before publishing.
    min_supporting_count=1,
    min_direct_evidence=1,
    high_confidence_min_supporting=8,
    high_confidence_min_witnesses=5,
    medium_confidence_min_supporting=3,
    medium_confidence_min_witnesses=2,
)
register_policy(
    ClaimScope.COMPETITOR_PAIR,
    DIRECT_DISPLACEMENT_CLAIM_TYPE,
    _DIRECT_DISPLACEMENT_POLICY,
)


@dataclass(frozen=True)
class DirectDisplacementClaimRow:
    challenger: str
    incumbent: str
    claim: ProductClaim


_SELECT_DIRECT_DISPLACEMENT_PAIR_SQL = """
WITH direct_ranked AS (
    SELECT
        ec.id,
        ec.vendor_name,
        ec.target_entity,
        ec.secondary_target,
        ec.witness_id,
        ec.source_review_id,
        ec.source_excerpt_fingerprint,
        ec.salience_score,
        ec.grounding_rank,
        ec.pain_confidence_rank,
        NULLIF(lower(btrim(r.reviewer_name)), '') AS reviewer_key,
        ROW_NUMBER() OVER (
            PARTITION BY
                ec.vendor_name,
                COALESCE(ec.source_excerpt_fingerprint, ec.id::text)
            ORDER BY
                ec.salience_score DESC,
                ec.grounding_rank ASC,
                ec.pain_confidence_rank ASC,
                ec.witness_id ASC
        ) AS dedup_rn
    FROM b2b_evidence_claims ec
    LEFT JOIN b2b_reviews r ON r.id = ec.source_review_id
    WHERE ec.status = 'valid'
      AND ec.claim_type = 'displacement_proof_to_competitor'
      AND ec.as_of_date = $3
      AND ec.analysis_window_days = $4
      AND lower(ec.vendor_name) = lower($1)
      AND lower(ec.secondary_target) = lower($2)
),
inverse_ranked AS (
    SELECT
        ec.id,
        ec.vendor_name,
        ec.secondary_target,
        ec.witness_id,
        ec.source_review_id,
        ec.source_excerpt_fingerprint,
        ec.salience_score,
        ec.grounding_rank,
        ec.pain_confidence_rank,
        ROW_NUMBER() OVER (
            PARTITION BY
                ec.vendor_name,
                COALESCE(ec.source_excerpt_fingerprint, ec.id::text)
            ORDER BY
                ec.salience_score DESC,
                ec.grounding_rank ASC,
                ec.pain_confidence_rank ASC,
                ec.witness_id ASC
        ) AS dedup_rn
    FROM b2b_evidence_claims ec
    WHERE ec.status = 'valid'
      AND ec.claim_type = 'displacement_proof_to_competitor'
      AND ec.as_of_date = $3
      AND ec.analysis_window_days = $4
      AND lower(ec.vendor_name) = lower($2)
      AND lower(ec.secondary_target) = lower($1)
),
direct AS (
    SELECT
        count(*) AS supporting_count,
        count(*) AS direct_evidence_count,
        count(DISTINCT reviewer_key) FILTER (
            WHERE reviewer_key IS NOT NULL
        ) AS witness_count,
        array_agg(DISTINCT source_review_id::text) FILTER (
            WHERE source_review_id IS NOT NULL
        ) AS evidence_links
    FROM direct_ranked
    WHERE dedup_rn = 1
),
inverse AS (
    SELECT
        count(*) AS contradiction_count,
        array_agg(DISTINCT source_review_id::text) FILTER (
            WHERE source_review_id IS NOT NULL
        ) AS contradicting_links
    FROM inverse_ranked
    WHERE dedup_rn = 1
)
SELECT
    direct.supporting_count,
    direct.direct_evidence_count,
    direct.witness_count,
    direct.evidence_links,
    inverse.contradiction_count,
    inverse.contradicting_links
FROM direct
CROSS JOIN inverse
"""


_SELECT_DIRECT_DISPLACEMENT_FOR_CHALLENGER_SQL = """
WITH direct_ranked AS (
    SELECT
        ec.id,
        ec.vendor_name,
        ec.target_entity,
        ec.secondary_target,
        ec.witness_id,
        ec.source_review_id,
        ec.source_excerpt_fingerprint,
        ec.salience_score,
        ec.grounding_rank,
        ec.pain_confidence_rank,
        NULLIF(lower(btrim(r.reviewer_name)), '') AS reviewer_key,
        ROW_NUMBER() OVER (
            PARTITION BY
                ec.vendor_name,
                COALESCE(ec.source_excerpt_fingerprint, ec.id::text)
            ORDER BY
                ec.salience_score DESC,
                ec.grounding_rank ASC,
                ec.pain_confidence_rank ASC,
                ec.witness_id ASC
        ) AS dedup_rn
    FROM b2b_evidence_claims ec
    LEFT JOIN b2b_reviews r ON r.id = ec.source_review_id
    WHERE ec.status = 'valid'
      AND ec.claim_type = 'displacement_proof_to_competitor'
      AND ec.as_of_date = $2
      AND ec.analysis_window_days = $3
      AND lower(ec.secondary_target) = lower($1)
),
inverse_ranked AS (
    SELECT
        ec.id,
        ec.vendor_name,
        ec.secondary_target,
        ec.witness_id,
        ec.source_review_id,
        ec.source_excerpt_fingerprint,
        ec.salience_score,
        ec.grounding_rank,
        ec.pain_confidence_rank,
        ROW_NUMBER() OVER (
            PARTITION BY
                ec.vendor_name,
                COALESCE(ec.source_excerpt_fingerprint, ec.id::text)
            ORDER BY
                ec.salience_score DESC,
                ec.grounding_rank ASC,
                ec.pain_confidence_rank ASC,
                ec.witness_id ASC
        ) AS dedup_rn
    FROM b2b_evidence_claims ec
    WHERE ec.status = 'valid'
      AND ec.claim_type = 'displacement_proof_to_competitor'
      AND ec.as_of_date = $2
      AND ec.analysis_window_days = $3
      AND lower(ec.vendor_name) = lower($1)
),
direct AS (
    SELECT
        vendor_name AS incumbent,
        count(*) AS supporting_count,
        count(*) AS direct_evidence_count,
        count(DISTINCT reviewer_key) FILTER (
            WHERE reviewer_key IS NOT NULL
        ) AS witness_count,
        array_agg(DISTINCT source_review_id::text) FILTER (
            WHERE source_review_id IS NOT NULL
        ) AS evidence_links
    FROM direct_ranked
    WHERE dedup_rn = 1
    GROUP BY vendor_name
),
inverse AS (
    SELECT
        secondary_target AS incumbent,
        count(*) AS contradiction_count,
        array_agg(DISTINCT source_review_id::text) FILTER (
            WHERE source_review_id IS NOT NULL
        ) AS contradicting_links
    FROM inverse_ranked
    WHERE dedup_rn = 1
    GROUP BY secondary_target
)
SELECT
    direct.incumbent,
    direct.supporting_count,
    direct.direct_evidence_count,
    direct.witness_count,
    direct.evidence_links,
    COALESCE(inverse.contradiction_count, 0) AS contradiction_count,
    inverse.contradicting_links
FROM direct
LEFT JOIN inverse ON lower(inverse.incumbent) = lower(direct.incumbent)
ORDER BY direct.supporting_count DESC, direct.incumbent ASC
LIMIT $4
"""


def _clean_entity(value: str) -> str:
    return str(value or "").strip()


def _int_value(row: Any, field: str) -> int:
    try:
        return int(row[field] or 0)
    except (KeyError, TypeError, ValueError):
        return 0


def _evidence_links(row: Any) -> tuple[str, ...]:
    try:
        raw = row["evidence_links"] or []
    except (KeyError, TypeError):
        raw = []
    return tuple(str(item) for item in raw if item)


def _contradicting_links(row: Any) -> tuple[str, ...]:
    try:
        raw = row["contradicting_links"] or []
    except (KeyError, TypeError):
        raw = []
    return tuple(str(item) for item in raw if item)


def _build_direct_displacement_claim(
    *,
    challenger: str,
    incumbent: str,
    row: Any | None,
    as_of_date: date,
    analysis_window_days: int,
) -> ProductClaim:
    supporting_count = _int_value(row, "supporting_count") if row else 0
    direct_evidence_count = _int_value(row, "direct_evidence_count") if row else 0
    # Witness diversity is reviewer identity, not review count. Missing
    # reviewer_name is conservative (0 distinct witnesses) so repeat /
    # cross-posted reviews cannot manufacture MEDIUM confidence.
    witness_count = _int_value(row, "witness_count") if row else 0
    contradiction_count = _int_value(row, "contradiction_count") if row else 0
    evidence_links = _evidence_links(row) if row else ()
    contradicting_links = _contradicting_links(row) if row else ()

    return build_product_claim(
        claim_scope=ClaimScope.COMPETITOR_PAIR,
        claim_type=DIRECT_DISPLACEMENT_CLAIM_TYPE,
        claim_key=f"incumbent:{incumbent}",
        claim_text=f"{incumbent} shows direct displacement pressure toward {challenger}",
        target_entity=challenger,
        secondary_target=incumbent,
        supporting_count=supporting_count,
        direct_evidence_count=direct_evidence_count,
        witness_count=witness_count,
        contradiction_count=contradiction_count,
        denominator=None,
        sample_size=supporting_count,
        has_grounded_evidence=direct_evidence_count > 0 if supporting_count > 0 else True,
        evidence_links=evidence_links,
        contradicting_links=contradicting_links,
        as_of_date=as_of_date,
        analysis_window_days=analysis_window_days,
        policy=_DIRECT_DISPLACEMENT_POLICY,
    )


async def aggregate_direct_displacement_claim(
    pool,
    *,
    challenger: str,
    incumbent: str,
    as_of_date: date,
    analysis_window_days: int,
) -> ProductClaim:
    """Build one challenger-centric direct-displacement claim for a pair.

    Query direction is incumbent -> challenger because Phase 9 stores
    `DISPLACEMENT_PROOF_TO_COMPETITOR` under the incumbent vendor. The
    returned ProductClaim is challenger-centric for the UI:
    target_entity=challenger, secondary_target=incumbent.

    No rows still returns a ProductClaim with supporting_count=0 so a
    caller with a configured pair can render an explicit insufficient
    state instead of silently omitting the pair.
    """
    challenger_name = _clean_entity(challenger)
    incumbent_name = _clean_entity(incumbent)
    row = await pool.fetchrow(
        _SELECT_DIRECT_DISPLACEMENT_PAIR_SQL,
        incumbent_name,
        challenger_name,
        as_of_date,
        int(analysis_window_days),
    )
    return _build_direct_displacement_claim(
        challenger=challenger_name,
        incumbent=incumbent_name,
        row=row,
        as_of_date=as_of_date,
        analysis_window_days=analysis_window_days,
    )


async def aggregate_direct_displacement_claims_for_challenger(
    pool,
    *,
    challenger: str,
    as_of_date: date,
    analysis_window_days: int,
    limit: int = 25,
) -> list[DirectDisplacementClaimRow]:
    """Return validated incumbent rows for one challenger.

    Unlike aggregate_direct_displacement_claim(), this list shape only
    returns incumbents that have at least one valid direct displacement
    EvidenceClaim. API 5a2 can merge this with configured targets if it
    needs explicit suppressed rows for tracked-but-unproven pairs.
    """
    challenger_name = _clean_entity(challenger)
    rows = await pool.fetch(
        _SELECT_DIRECT_DISPLACEMENT_FOR_CHALLENGER_SQL,
        challenger_name,
        as_of_date,
        int(analysis_window_days),
        int(limit),
    )
    out: list[DirectDisplacementClaimRow] = []
    for row in rows:
        incumbent = _clean_entity(row["incumbent"])
        out.append(
            DirectDisplacementClaimRow(
                challenger=challenger_name,
                incumbent=incumbent,
                claim=_build_direct_displacement_claim(
                    challenger=challenger_name,
                    incumbent=incumbent,
                    row=row,
                    as_of_date=as_of_date,
                    analysis_window_days=analysis_window_days,
                ),
            )
        )
    return out


__all__ = [
    "DIRECT_DISPLACEMENT_CLAIM_TYPE",
    "DirectDisplacementClaimRow",
    "aggregate_direct_displacement_claim",
    "aggregate_direct_displacement_claims_for_challenger",
]
