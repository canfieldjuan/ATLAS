"""ACCOUNT-scope ProductClaim helpers for opportunity rows.

Patch 4a starts with the high-intent / Opportunities feed. The feed is
row-level, so the first contract is intentionally narrow:

  - one ProductClaim per high-intent opportunity row
  - UI render can pass with one direct source review
  - report/campaign publish remains blocked until corroboration exists

This keeps the dashboard useful without letting a single review become a
strong downstream recommendation.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from .product_claim import (
    ClaimGatePolicy,
    ClaimScope,
    ProductClaim,
    build_product_claim,
    register_policy,
)


ACCOUNT_OPPORTUNITY_CLAIM_TYPE = "account_opportunity_readiness"


_ACCOUNT_OPPORTUNITY_POLICY = ClaimGatePolicy(
    min_supporting_count=1,
    min_direct_evidence=1,
    high_confidence_min_supporting=5,
    high_confidence_min_witnesses=3,
    medium_confidence_min_supporting=2,
    medium_confidence_min_witnesses=2,
)
register_policy(
    ClaimScope.ACCOUNT,
    ACCOUNT_OPPORTUNITY_CLAIM_TYPE,
    _ACCOUNT_OPPORTUNITY_POLICY,
)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _has_intent_signal(row: dict[str, Any]) -> bool:
    signals = row.get("intent_signals")
    if isinstance(signals, dict) and any(bool(value) for value in signals.values()):
        return True
    stage = _clean_text(row.get("buying_stage")).lower()
    if stage in {"evaluation", "active_purchase", "renewal_decision"}:
        return True
    if _clean_text(row.get("contract_signal")):
        return True
    if _safe_list(row.get("alternatives")):
        return True
    return False


def _source_review_ids(row: dict[str, Any]) -> tuple[str, ...]:
    review_id = _clean_text(row.get("review_id"))
    if review_id:
        return (review_id,)
    return tuple(
        _clean_text(item)
        for item in _safe_list(row.get("source_review_ids"))
        if _clean_text(item)
    )


def _has_quote_evidence(row: dict[str, Any]) -> bool:
    quotes = _safe_list(row.get("quotes")) or _safe_list(row.get("evidence"))
    for item in quotes:
        if isinstance(item, dict):
            if _clean_text(item.get("text") or item.get("quote") or item.get("excerpt")):
                return True
        elif _clean_text(item):
            return True
    return False


def _claim_text(company: str, vendor: str, row: dict[str, Any]) -> str:
    stage = _clean_text(row.get("buying_stage")).replace("_", " ")
    if stage:
        return f"{company} shows {stage} pressure away from {vendor}"
    return f"{company} shows churn pressure away from {vendor}"


def serialize_product_claim(claim: ProductClaim) -> dict[str, Any]:
    """Dict form matching the public ProductClaim API envelope.

    ACCOUNT v1 adds source_review_count because reviewer-dedup lineage
    is not plumbed into high-intent rows yet. ProductClaim.witness_count
    stays conservative (0/1 for this row), while source_review_count is
    the count callers can display as the row's evidence denominator.
    """
    return {
        "claim_id": claim.claim_id,
        "claim_key": claim.claim_key,
        "claim_scope": claim.claim_scope.value,
        "claim_type": claim.claim_type,
        "claim_text": claim.claim_text,
        "target_entity": claim.target_entity,
        "secondary_target": claim.secondary_target,
        "supporting_count": claim.supporting_count,
        "direct_evidence_count": claim.direct_evidence_count,
        "witness_count": claim.witness_count,
        "contradiction_count": claim.contradiction_count,
        "denominator": claim.denominator,
        "sample_size": claim.sample_size,
        "source_review_count": claim.sample_size,
        "has_grounded_evidence": claim.has_grounded_evidence,
        "confidence": claim.confidence.value,
        "evidence_posture": claim.evidence_posture.value,
        "render_allowed": claim.render_allowed,
        "report_allowed": claim.report_allowed,
        "suppression_reason": (
            claim.suppression_reason.value if claim.suppression_reason else None
        ),
        "evidence_links": list(claim.evidence_links),
        "contradicting_links": list(claim.contradicting_links),
        "as_of_date": claim.as_of_date.isoformat(),
        "analysis_window_days": claim.analysis_window_days,
        "schema_version": claim.schema_version,
    }


def build_account_opportunity_claim(
    row: dict[str, Any],
    *,
    as_of_date: date,
    analysis_window_days: int,
) -> ProductClaim:
    """Build the ACCOUNT opportunity-readiness claim for one feed row.

    Direct evidence is intentionally conservative: the row must identify
    a real company/vendor pair, have a source review or quote, and carry
    an explicit intent signal. Missing any of those makes the row
    monitor-only through the ProductClaim gate.

    Reviewer-dedup data is not present on high-intent rows yet. Until it
    is, source_count feeds supporting_count/sample_size, while
    witness_count is capped at 1 for this row so confidence thresholds
    do not accidentally treat multiple source reviews as multiple
    people.
    """
    company = _clean_text(row.get("company"))
    vendor = _clean_text(row.get("vendor"))
    source_review_ids = _source_review_ids(row)
    source_count = len(source_review_ids) if source_review_ids else int(bool(_has_quote_evidence(row)))
    witness_count = 1 if source_count > 0 else 0
    has_identity = bool(company and vendor)
    has_source = source_count > 0
    has_intent = _has_intent_signal(row)
    direct_count = 1 if has_identity and has_source and has_intent else 0

    return build_product_claim(
        claim_scope=ClaimScope.ACCOUNT,
        claim_type=ACCOUNT_OPPORTUNITY_CLAIM_TYPE,
        claim_key="opportunity_readiness",
        claim_text=_claim_text(company or "Unknown account", vendor or "unknown vendor", row),
        target_entity=company or "unknown_account",
        secondary_target=vendor or None,
        supporting_count=source_count,
        direct_evidence_count=direct_count,
        witness_count=witness_count,
        contradiction_count=0,
        denominator=source_count,
        sample_size=source_count,
        has_grounded_evidence=direct_count > 0,
        evidence_links=source_review_ids,
        contradicting_links=(),
        as_of_date=as_of_date,
        analysis_window_days=analysis_window_days,
        require_registered_policy=True,
    )


def attach_account_opportunity_claim(
    payload: dict[str, Any],
    *,
    as_of_date: date,
    analysis_window_days: int,
) -> dict[str, Any]:
    """Append the nested ProductClaim to an existing opportunity payload.

    Do not mirror gate fields flat on the row. Patch 4b must consume
    opportunity_claim.* only; a second partial surface would recreate
    the drift class this contract is meant to remove.
    """
    claim = build_account_opportunity_claim(
        payload,
        as_of_date=as_of_date,
        analysis_window_days=analysis_window_days,
    )
    return {
        **payload,
        "opportunity_claim": serialize_product_claim(claim),
    }


__all__ = [
    "ACCOUNT_OPPORTUNITY_CLAIM_TYPE",
    "attach_account_opportunity_claim",
    "build_account_opportunity_claim",
    "serialize_product_claim",
]
