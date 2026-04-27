"""Challenger ProductClaims API.

Phase 10 Patch 5a2. Exposes challenger-centric COMPETITOR_PAIR ProductClaim
envelopes the React Challenger UI will consume to gate winner-call
language, head-to-head proof, and challenger action affordances.

Initial scope is intentionally narrow: ONE claim_type
(direct_displacement) so the API contract proves itself end to end
before incumbent_weakness, buyer_preference, and category_overlap
slices land. Subsequent patches add the remaining COMPETITOR_PAIR
claim_types to the same response shape.

Direction convention (pinned by Patch 5a1 backend):

  - Phase 9 evidence query is incumbent-centric: vendor_name=incumbent,
    secondary_target=challenger, claim_type=displacement_proof_to_competitor.
  - This API is challenger-centric: target_entity=challenger,
    secondary_target=incumbent, claim_key="incumbent:{incumbent}".
  - The flip happens inside the aggregator; this endpoint surfaces the
    challenger-centric view verbatim.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..auth.dependencies import AuthUser, require_b2b_plan
from ..config import settings
from ..services.b2b.challenger_dashboard_claims import (
    aggregate_direct_displacement_claims_for_challenger,
)
from ..storage.database import get_db_pool
from .b2b_vendor_claims import VendorClaimResponse, _serialize_claim


logger = logging.getLogger("atlas.api.b2b_challenger_claims")


router = APIRouter(
    prefix="/b2b/challenger-claims",
    tags=["b2b-challenger-claims"],
)


# Window bounds match the existing b2b_evidence and b2b_vendor_claims routers.
_DEFAULT_WINDOW_DAYS = 30
_MIN_WINDOW_DAYS = 7
_MAX_WINDOW_DAYS = 365
_DEFAULT_LIMIT = 25
_MAX_LIMIT = 100


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
# Response shape: one row per (challenger, incumbent) pair, each row carries
# the verbatim ProductClaim envelope.
# ----------------------------------------------------------------------------


class ChallengerClaimRow(BaseModel):
    """One incumbent row for the challenger view. The challenger field
    repeats the URL path parameter for parity; the incumbent + claim
    fields are the per-pair payload."""

    challenger: str
    incumbent: str
    claim: VendorClaimResponse


class ChallengerClaimsResponse(BaseModel):
    challenger: str
    as_of_date: str
    analysis_window_days: int
    rows: list[ChallengerClaimRow]


# ----------------------------------------------------------------------------
# Endpoint.
# ----------------------------------------------------------------------------


@router.get("/{challenger}", response_model=ChallengerClaimsResponse)
async def get_challenger_claims(
    challenger: str,
    as_of_date: Optional[str] = Query(default=None),
    analysis_window_days: int = Query(
        default=None,  # type: ignore[arg-type]
        ge=_MIN_WINDOW_DAYS,
        le=_MAX_WINDOW_DAYS,
    ),
    limit: int = Query(default=_DEFAULT_LIMIT, ge=1, le=_MAX_LIMIT),
    user: AuthUser = Depends(require_b2b_plan("b2b_trial")),
) -> ChallengerClaimsResponse:
    """Return validated direct-displacement claims for a challenger.

    The React Challenger UI reads `claim.render_allowed` to decide
    whether a "Losing From" / head-to-head row displays winner-call
    language; server-side challenger reports read `claim.report_allowed`
    (strictly tighter). Both come from the same ProductClaim, so a
    report can never publish a winner call the UI suppressed.

    Currently emits direct_displacement only. Subsequent patches add
    incumbent_weakness, buyer_preference, and category_overlap to the
    same row shape (one per incumbent, multiple claim_types per row).
    """
    challenger_name = (challenger or "").strip()
    if not challenger_name:
        raise HTTPException(status_code=422, detail="challenger is required")
    pool = _pool_or_503()

    target_date = _parse_target_date(as_of_date)
    window_days = int(analysis_window_days or _default_analysis_window_days())

    rows = await aggregate_direct_displacement_claims_for_challenger(
        pool,
        challenger=challenger_name,
        as_of_date=target_date,
        analysis_window_days=window_days,
        limit=int(limit),
    )

    return ChallengerClaimsResponse(
        challenger=challenger_name,
        as_of_date=target_date.isoformat(),
        analysis_window_days=window_days,
        rows=[
            ChallengerClaimRow(
                challenger=row.challenger,
                incumbent=row.incumbent,
                claim=_serialize_claim(row.claim),
            )
            for row in rows
        ],
    )
