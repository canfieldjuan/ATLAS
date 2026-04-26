"""Vendor ProductClaims API.

Phase 10 Patch 2b. Exposes the VENDOR-scope ProductClaim envelopes the
React side will consume to gate dashboard renders. Read-only; preserves
every field of ProductClaim verbatim so the React side cannot drift
from the contract.

Initial scope is intentionally narrow: ONE claim_type
(decision_maker_churn_rate) so the API contract proves itself end to
end before the surface fan-out. Subsequent patches add price
complaint rate, weakness theme, strength theme, and the remaining
VENDOR-scope claims to the same endpoint shape.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..auth.dependencies import AuthUser, require_b2b_plan
from ..config import settings
from ..services.b2b.product_claim import ProductClaim
from ..services.b2b.vendor_dashboard_claims import (
    aggregate_dm_churn_rate_claim,
    aggregate_price_complaint_rate_claim,
)
from ..storage.database import get_db_pool


logger = logging.getLogger("atlas.api.b2b_vendor_claims")


router = APIRouter(
    prefix="/b2b/vendor-claims",
    tags=["b2b-vendor-claims"],
)


# Window bounds match the existing b2b_evidence router for consistency.
_DEFAULT_WINDOW_DAYS = 30
_MIN_WINDOW_DAYS = 7
_MAX_WINDOW_DAYS = 365


def _default_analysis_window_days() -> int:
    return int(
        getattr(
            settings.b2b_churn,
            "evidence_default_analysis_window_days",
            _DEFAULT_WINDOW_DAYS,
        )
    )


def _pool_or_503():
    pool = get_db_pool()
    if pool is None or not getattr(pool, "is_initialized", False):
        raise HTTPException(status_code=503, detail="Database not ready")
    return pool


def _parse_target_date(as_of_date: Optional[str]) -> date:
    if not as_of_date:
        return date.today()
    try:
        return date.fromisoformat(as_of_date)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="Invalid as_of_date; expected YYYY-MM-DD"
        ) from exc


# ----------------------------------------------------------------------------
# Response shape: every ProductClaim gate field exposed verbatim.
# ----------------------------------------------------------------------------


class VendorClaimResponse(BaseModel):
    """Verbatim mirror of ProductClaim. The React side is meant to bind
    one-to-one against this; fields here that diverge from ProductClaim
    will silently break the contract."""

    claim_id: str
    claim_key: str
    claim_scope: str
    claim_type: str
    claim_text: str
    target_entity: str
    secondary_target: Optional[str] = None

    # Numerator / denominator / context.
    supporting_count: int
    direct_evidence_count: int
    witness_count: int
    contradiction_count: int
    denominator: Optional[int] = None
    sample_size: Optional[int] = None
    has_grounded_evidence: bool

    # Derived quality.
    confidence: str
    evidence_posture: str

    # Render gates -- the WHOLE point of this contract.
    render_allowed: bool
    report_allowed: bool
    suppression_reason: Optional[str] = None

    # Provenance.
    evidence_links: list[str] = Field(default_factory=list)
    contradicting_links: list[str] = Field(default_factory=list)

    as_of_date: str
    analysis_window_days: int
    schema_version: str


class VendorClaimsResponse(BaseModel):
    vendor_name: str
    as_of_date: str
    analysis_window_days: int
    claims: list[VendorClaimResponse]


def _serialize_claim(claim: ProductClaim) -> VendorClaimResponse:
    """Round-trip a ProductClaim into the API response shape. Field
    names match ProductClaim 1:1 except where the dataclass uses an
    enum that's serialized as its .value string. The internal `policy`
    field is intentionally NOT exposed (it's an implementation detail
    of the gate logic, not a consumer-facing field)."""
    return VendorClaimResponse(
        claim_id=claim.claim_id,
        claim_key=claim.claim_key,
        claim_scope=claim.claim_scope.value,
        claim_type=claim.claim_type,
        claim_text=claim.claim_text,
        target_entity=claim.target_entity,
        secondary_target=claim.secondary_target,
        supporting_count=claim.supporting_count,
        direct_evidence_count=claim.direct_evidence_count,
        witness_count=claim.witness_count,
        contradiction_count=claim.contradiction_count,
        denominator=claim.denominator,
        sample_size=claim.sample_size,
        has_grounded_evidence=claim.has_grounded_evidence,
        confidence=claim.confidence.value,
        evidence_posture=claim.evidence_posture.value,
        render_allowed=claim.render_allowed,
        report_allowed=claim.report_allowed,
        suppression_reason=(
            claim.suppression_reason.value if claim.suppression_reason else None
        ),
        evidence_links=list(claim.evidence_links),
        contradicting_links=list(claim.contradicting_links),
        as_of_date=claim.as_of_date.isoformat(),
        analysis_window_days=claim.analysis_window_days,
        schema_version=claim.schema_version,
    )


# ----------------------------------------------------------------------------
# Endpoints.
# ----------------------------------------------------------------------------


@router.get("/{vendor_name}", response_model=VendorClaimsResponse)
async def get_vendor_claims(
    vendor_name: str,
    as_of_date: Optional[str] = Query(default=None),
    analysis_window_days: int = Query(
        default=None,  # type: ignore[arg-type]
        ge=_MIN_WINDOW_DAYS,
        le=_MAX_WINDOW_DAYS,
    ),
    user: AuthUser = Depends(require_b2b_plan("b2b_trial")),
) -> VendorClaimsResponse:
    """Return the validated VENDOR-scope ProductClaims for a vendor.

    The React dashboard reads `render_allowed` to decide whether each
    rate card displays the value or shows 'Insufficient evidence';
    server-side reports read `report_allowed` (strictly tighter).
    Both come from the same ProductClaim, so the report can never
    publish a claim the UI suppressed.

    Currently emits decision_maker_churn_rate and price_complaint_rate.
    Subsequent patches extend the same endpoint with weakness_theme,
    strength_theme, recommend_ratio, and the remaining VENDOR-scope
    claim_types -- the response shape stays.
    """
    vendor = (vendor_name or "").strip()
    if not vendor:
        raise HTTPException(status_code=422, detail="vendor_name is required")
    pool = _pool_or_503()

    target_date = _parse_target_date(as_of_date)
    window_days = int(analysis_window_days or _default_analysis_window_days())

    claims: list[ProductClaim] = []

    dm_claim = await aggregate_dm_churn_rate_claim(
        pool,
        vendor_name=vendor,
        as_of_date=target_date,
        analysis_window_days=window_days,
    )
    if dm_claim is not None:
        claims.append(dm_claim)

    price_claim = await aggregate_price_complaint_rate_claim(
        pool,
        vendor_name=vendor,
        as_of_date=target_date,
        analysis_window_days=window_days,
    )
    if price_claim is not None:
        claims.append(price_claim)

    return VendorClaimsResponse(
        vendor_name=vendor,
        as_of_date=target_date.isoformat(),
        analysis_window_days=window_days,
        claims=[_serialize_claim(c) for c in claims],
    )
