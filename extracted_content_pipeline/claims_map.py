"""Content-ops claims map (operating-model slice 3).

The deterministic core of the stage-3B claims/compliance gate from
``docs/content_ops_operating_model.md``: map each claim a draft makes onto the
messaging registry and status-check it, so a draft can't ship wording that
conflicts with (or outlives) the approved claim. Pure value types + pure
functions -- no I/O, no Atlas imports, no DB, no LLM.

The part that *requires* an LLM -- extracting claims from prose into
``ExtractedClaim`` rows -- is a later slice; this module takes already-extracted
claims and decides their status against a registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Mapping

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 CI compatibility
    from enum import Enum

    class StrEnum(str, Enum):
        pass

from .review_contract import RiskTier


class ClaimStatus(StrEnum):
    """Status of one mapped claim against the messaging registry."""

    MATCH = "match"  # wording matches the approved registry wording
    MISMATCH = "mismatch"  # references a registry entry but the wording differs
    UNREGISTERED = "unregistered"  # no known registry entry -- needs human review
    EXPIRED = "expired"  # matched an entry whose approved wording has lapsed


def _normalize(value: object) -> str:
    """Casefold + collapse whitespace for wording comparison.

    Non-``str`` (e.g. ``None`` from JSON null) normalizes to ``""`` rather than
    raising, so callers can build these from decoded/untrusted input.
    """

    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).casefold()


@dataclass(frozen=True)
class RegistryClaim:
    """An entry in the messaging/claim registry (the source of truth)."""

    id: str
    approved_wording: str
    risk_tier: RiskTier | None = None
    expiration: date | None = None

    def is_expired(self, as_of: date) -> bool:
        """True if the approved wording has lapsed on ``as_of`` (exclusive of the
        expiration day -- the day itself is still valid)."""

        return self.expiration is not None and as_of > self.expiration


@dataclass(frozen=True)
class ExtractedClaim:
    """One claim as an extractor would emit it.

    ``registry_id`` is the extractor's candidate match (``None`` if it found no
    registry entry). Producing these from prose is the deferred LLM step.
    """

    text: str | None
    location: str = ""
    registry_id: str | None = None


@dataclass(frozen=True)
class MappedClaim:
    """A claims-map row: an extracted claim resolved against the registry."""

    text: str | None
    location: str
    registry_id: str | None
    approved_wording: str | None
    status: ClaimStatus
    risk_tier: RiskTier | None = None


def map_claim(
    claim: ExtractedClaim,
    registry: Mapping[str, RegistryClaim],
    *,
    as_of: date,
) -> MappedClaim:
    """Resolve one ``ExtractedClaim`` against ``registry`` as of ``as_of``."""

    entry = registry.get(claim.registry_id) if claim.registry_id else None
    if entry is None:
        # Preserve the extractor's candidate id (None if it had none, or the
        # stale/typo'd id if it pointed at a missing registry entry) so a
        # reviewer can tell "no candidate" apart from "bad candidate".
        return MappedClaim(
            text=claim.text,
            location=claim.location,
            registry_id=claim.registry_id,
            approved_wording=None,
            status=ClaimStatus.UNREGISTERED,
        )
    if entry.is_expired(as_of):
        status = ClaimStatus.EXPIRED
    elif _normalize(claim.text) == _normalize(entry.approved_wording):
        status = ClaimStatus.MATCH
    else:
        status = ClaimStatus.MISMATCH
    return MappedClaim(
        text=claim.text,
        location=claim.location,
        registry_id=entry.id,
        approved_wording=entry.approved_wording,
        status=status,
        risk_tier=entry.risk_tier,
    )


def build_claims_map(
    claims: Iterable[ExtractedClaim],
    registry: Mapping[str, RegistryClaim],
    *,
    as_of: date,
) -> tuple[MappedClaim, ...]:
    """Map every extracted claim, preserving input order."""

    return tuple(map_claim(claim, registry, as_of=as_of) for claim in claims)


# Statuses that should stop a draft at the claims gate. UNREGISTERED is
# deliberately excluded -- an unknown claim is a human-review signal, not an
# automatic block (per the doc's routing).
_BLOCKING_STATUSES = frozenset({ClaimStatus.MISMATCH, ClaimStatus.EXPIRED})


def blocking_claims(mapped: Iterable[MappedClaim]) -> tuple[MappedClaim, ...]:
    """The mapped claims whose status blocks publish (MISMATCH or EXPIRED)."""

    return tuple(m for m in mapped if m.status in _BLOCKING_STATUSES)


def is_clear(mapped: Iterable[MappedClaim]) -> bool:
    """True if no mapped claim blocks (no MISMATCH/EXPIRED)."""

    return not blocking_claims(mapped)
