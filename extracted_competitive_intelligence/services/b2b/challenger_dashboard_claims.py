"""Host integration port for challenger dashboard ProductClaim aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol


DIRECT_DISPLACEMENT_CLAIM_TYPE = "direct_displacement"
DEFAULT_DIRECT_DISPLACEMENT_LIMIT = 25


class ChallengerDashboardClaimsPortNotConfigured(RuntimeError):
    """Raised when a host has not registered challenger claim aggregation."""


@dataclass(frozen=True)
class DirectDisplacementClaimRow:
    challenger: str
    incumbent: str
    claim: Any


class DirectDisplacementClaimReader(Protocol):
    async def __call__(
        self,
        pool: Any,
        *,
        challenger: str,
        incumbent: str,
        as_of_date: date,
        analysis_window_days: int,
    ) -> Any:
        """Return one challenger-centric direct displacement ProductClaim."""


class DirectDisplacementClaimsForChallengerReader(Protocol):
    async def __call__(
        self,
        pool: Any,
        *,
        challenger: str,
        as_of_date: date,
        analysis_window_days: int,
        limit: int = DEFAULT_DIRECT_DISPLACEMENT_LIMIT,
    ) -> list[DirectDisplacementClaimRow]:
        """Return incumbent rows for one challenger."""


class DirectDisplacementClaimsForIncumbentReader(Protocol):
    async def __call__(
        self,
        pool: Any,
        *,
        incumbent: str,
        as_of_date: date,
        analysis_window_days: int,
        limit: int = DEFAULT_DIRECT_DISPLACEMENT_LIMIT,
    ) -> list[DirectDisplacementClaimRow]:
        """Return challenger rows for one incumbent-losing-share card."""


_direct_displacement_claim_reader: DirectDisplacementClaimReader | None = None
_challenger_claims_reader: DirectDisplacementClaimsForChallengerReader | None = None
_incumbent_claims_reader: DirectDisplacementClaimsForIncumbentReader | None = None


def configure_direct_displacement_claim_reader(
    reader: DirectDisplacementClaimReader | None,
) -> None:
    """Register the host adapter for single pair direct-displacement claims."""
    global _direct_displacement_claim_reader
    _direct_displacement_claim_reader = reader


def configure_direct_displacement_claims_for_challenger_reader(
    reader: DirectDisplacementClaimsForChallengerReader | None,
) -> None:
    """Register the host adapter for challenger-side displacement rows."""
    global _challenger_claims_reader
    _challenger_claims_reader = reader


def configure_direct_displacement_claims_for_incumbent_reader(
    reader: DirectDisplacementClaimsForIncumbentReader | None,
) -> None:
    """Register the host adapter for incumbent-side displacement rows."""
    global _incumbent_claims_reader
    _incumbent_claims_reader = reader


async def aggregate_direct_displacement_claim(
    pool: Any,
    *,
    challenger: str,
    incumbent: str,
    as_of_date: date,
    analysis_window_days: int,
) -> Any:
    """Build one challenger-centric direct displacement claim for a pair."""
    if _direct_displacement_claim_reader is None:
        raise ChallengerDashboardClaimsPortNotConfigured(
            "No direct displacement claim reader has been configured"
        )
    return await _direct_displacement_claim_reader(
        pool,
        challenger=challenger,
        incumbent=incumbent,
        as_of_date=as_of_date,
        analysis_window_days=analysis_window_days,
    )


async def aggregate_direct_displacement_claims_for_challenger(
    pool: Any,
    *,
    challenger: str,
    as_of_date: date,
    analysis_window_days: int,
    limit: int = DEFAULT_DIRECT_DISPLACEMENT_LIMIT,
) -> list[DirectDisplacementClaimRow]:
    """Return validated incumbent rows for one challenger."""
    if _challenger_claims_reader is None:
        raise ChallengerDashboardClaimsPortNotConfigured(
            "No challenger-side direct displacement claims reader has been configured"
        )
    return await _challenger_claims_reader(
        pool,
        challenger=challenger,
        as_of_date=as_of_date,
        analysis_window_days=analysis_window_days,
        limit=limit,
    )


async def aggregate_direct_displacement_claims_for_incumbent(
    pool: Any,
    *,
    incumbent: str,
    as_of_date: date,
    analysis_window_days: int,
    limit: int = DEFAULT_DIRECT_DISPLACEMENT_LIMIT,
) -> list[DirectDisplacementClaimRow]:
    """Return challenger rows for one incumbent-losing-share card."""
    if _incumbent_claims_reader is None:
        raise ChallengerDashboardClaimsPortNotConfigured(
            "No incumbent-side direct displacement claims reader has been configured"
        )
    return await _incumbent_claims_reader(
        pool,
        incumbent=incumbent,
        as_of_date=as_of_date,
        analysis_window_days=analysis_window_days,
        limit=limit,
    )


__all__ = [
    "ChallengerDashboardClaimsPortNotConfigured",
    "DEFAULT_DIRECT_DISPLACEMENT_LIMIT",
    "DIRECT_DISPLACEMENT_CLAIM_TYPE",
    "DirectDisplacementClaimReader",
    "DirectDisplacementClaimRow",
    "DirectDisplacementClaimsForChallengerReader",
    "DirectDisplacementClaimsForIncumbentReader",
    "aggregate_direct_displacement_claim",
    "aggregate_direct_displacement_claims_for_challenger",
    "aggregate_direct_displacement_claims_for_incumbent",
    "configure_direct_displacement_claim_reader",
    "configure_direct_displacement_claims_for_challenger_reader",
    "configure_direct_displacement_claims_for_incumbent_reader",
]
