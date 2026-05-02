"""ACCOUNT-scope claim helpers for standalone campaign opportunity rows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
from typing import Any


ACCOUNT_OPPORTUNITY_CLAIM_TYPE = "account_opportunity_readiness"


@dataclass(frozen=True)
class ProductClaim:
    claim_id: str
    claim_key: str
    claim_scope: str
    claim_type: str
    claim_text: str
    target_entity: str
    secondary_target: str | None
    supporting_count: int
    direct_evidence_count: int
    witness_count: int
    contradiction_count: int
    denominator: int | None
    sample_size: int | None
    has_grounded_evidence: bool
    confidence: str
    evidence_posture: str
    render_allowed: bool
    report_allowed: bool
    suppression_reason: str | None
    evidence_links: tuple[str, ...]
    contradicting_links: tuple[str, ...]
    as_of_date: date
    analysis_window_days: int
    schema_version: str = "v1"


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


def _claim_id(
    *,
    claim_key: str,
    target_entity: str,
    secondary_target: str | None,
    as_of_date: date,
    analysis_window_days: int,
) -> str:
    raw = "|".join(
        [
            "account",
            ACCOUNT_OPPORTUNITY_CLAIM_TYPE,
            claim_key,
            target_entity,
            secondary_target or "",
            as_of_date.isoformat(),
            str(analysis_window_days),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _confidence(*, supporting_count: int, witness_count: int) -> str:
    if supporting_count >= 5 and witness_count >= 3:
        return "high"
    if supporting_count >= 2 and witness_count >= 2:
        return "medium"
    return "low"


def _posture(
    *,
    supporting_count: int,
    direct_evidence_count: int,
    has_grounded_evidence: bool,
) -> str:
    if not has_grounded_evidence:
        return "unverified"
    if supporting_count <= 0:
        return "insufficient"
    if direct_evidence_count <= 0:
        return "weak"
    return "usable"


def _gates(
    *,
    evidence_posture: str,
    confidence: str,
    supporting_count: int,
    direct_evidence_count: int,
) -> tuple[bool, bool, str | None]:
    if evidence_posture == "unverified":
        return False, False, "unverified_evidence"
    if evidence_posture == "insufficient" or supporting_count < 1:
        return False, False, "insufficient_supporting_count"
    if direct_evidence_count < 1:
        return False, False, "weak_evidence_only"
    render_allowed = True
    if evidence_posture != "usable":
        return render_allowed, False, "weak_evidence_only"
    if confidence == "low":
        return render_allowed, False, "low_confidence"
    return True, True, None


def account_opportunity_source_review_count(row: dict[str, Any]) -> int:
    """Return the row's source-review count independent of claim sample size."""
    source_review_ids = _source_review_ids(row)
    if source_review_ids:
        return len(source_review_ids)
    return int(bool(_has_quote_evidence(row)))


def serialize_product_claim(
    claim: ProductClaim,
    *,
    source_review_count: int,
) -> dict[str, Any]:
    """Return the public ProductClaim-style envelope expected by copied tasks."""
    return {
        "claim_id": claim.claim_id,
        "claim_key": claim.claim_key,
        "claim_scope": claim.claim_scope,
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
        "source_review_count": source_review_count,
        "has_grounded_evidence": claim.has_grounded_evidence,
        "confidence": claim.confidence,
        "evidence_posture": claim.evidence_posture,
        "render_allowed": claim.render_allowed,
        "report_allowed": claim.report_allowed,
        "suppression_reason": claim.suppression_reason,
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
    """Build the ACCOUNT opportunity-readiness claim for one feed row."""
    company = _clean_text(row.get("company"))
    vendor = _clean_text(row.get("vendor"))
    source_review_ids = _source_review_ids(row)
    source_count = account_opportunity_source_review_count(row)
    witness_count = 1 if source_count > 0 else 0
    has_identity = bool(company and vendor)
    has_source = source_count > 0
    has_intent = _has_intent_signal(row)
    direct_count = 1 if has_identity and has_source and has_intent else 0
    has_grounded_evidence = direct_count > 0
    evidence_posture = _posture(
        supporting_count=source_count,
        direct_evidence_count=direct_count,
        has_grounded_evidence=has_grounded_evidence,
    )
    confidence = _confidence(
        supporting_count=source_count,
        witness_count=witness_count,
    )
    render_allowed, report_allowed, suppression_reason = _gates(
        evidence_posture=evidence_posture,
        confidence=confidence,
        supporting_count=source_count,
        direct_evidence_count=direct_count,
    )
    target_entity = company or "unknown_account"
    secondary_target = vendor or None
    return ProductClaim(
        claim_id=_claim_id(
            claim_key="opportunity_readiness",
            target_entity=target_entity,
            secondary_target=secondary_target,
            as_of_date=as_of_date,
            analysis_window_days=analysis_window_days,
        ),
        claim_key="opportunity_readiness",
        claim_scope="account",
        claim_type=ACCOUNT_OPPORTUNITY_CLAIM_TYPE,
        claim_text=_claim_text(company or "Unknown account", vendor or "unknown vendor", row),
        target_entity=target_entity,
        secondary_target=secondary_target,
        supporting_count=source_count,
        direct_evidence_count=direct_count,
        witness_count=witness_count,
        contradiction_count=0,
        denominator=source_count,
        sample_size=source_count,
        has_grounded_evidence=has_grounded_evidence,
        confidence=confidence,
        evidence_posture=evidence_posture,
        render_allowed=render_allowed,
        report_allowed=report_allowed,
        suppression_reason=suppression_reason,
        evidence_links=source_review_ids,
        contradicting_links=(),
        as_of_date=as_of_date,
        analysis_window_days=analysis_window_days,
    )


def attach_account_opportunity_claim(
    payload: dict[str, Any],
    *,
    as_of_date: date,
    analysis_window_days: int,
) -> dict[str, Any]:
    claim = build_account_opportunity_claim(
        payload,
        as_of_date=as_of_date,
        analysis_window_days=analysis_window_days,
    )
    return {
        **payload,
        "opportunity_claim": serialize_product_claim(
            claim,
            source_review_count=account_opportunity_source_review_count(payload),
        ),
    }


__all__ = [
    "ACCOUNT_OPPORTUNITY_CLAIM_TYPE",
    "ProductClaim",
    "account_opportunity_source_review_count",
    "attach_account_opportunity_claim",
    "build_account_opportunity_claim",
    "serialize_product_claim",
]
