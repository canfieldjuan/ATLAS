"""Content-ops review vocabulary (operating-model slice 1).

Typed vocabulary for the content review subsystem described in
``docs/content_ops_operating_model.md``. This module is intentionally pure:
value enums, one frozen record, and a few small helpers, with no I/O, no Atlas
imports, no database, and no LLM. Later slices (risk-tier routing, the claims
map, the Content-PR coverage matrix, the calibration library) consume these
names so they do not each reinvent their own.

Nothing here changes existing behavior. The generated-asset review API keeps
its host-extensible string statuses; mapping those onto ``ReviewDecision`` is a
later slice.
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


class RiskTier(StrEnum):
    """Editorial review-burden tier, set on the brief.

    Distinct from ``extracted_quality_gate.RiskLevel``: that is a *computed*
    safety score (with auto-approve eligibility); this is an author-declared
    routing tier that decides how much review an asset must clear. The labels
    deliberately match so the two stay legible together.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewDecision(StrEnum):
    """Accountable editor decision on a draft.

    ``APPROVED_WITH_EXCEPTION`` lets a piece ship despite a soft-rule violation,
    but never invisibly -- it pairs with an :class:`ExceptionRecord`.
    """

    BLOCKED = "blocked"
    REVISION_REQUIRED = "revision_required"
    APPROVED = "approved"
    APPROVED_WITH_EXCEPTION = "approved_with_exception"
    ESCALATED = "escalated"


class FailureCategory(StrEnum):
    """Structured ``why`` for a post-publish verdict (the failure taxonomy).

    A structured reason makes verdicts reusable instead of anecdotal: you learn
    *which type* of failure recurs, which feeds the "3+ -> new gate" flywheel
    (see :func:`recurring_failure_categories`).
    """

    WRONG_AUDIENCE = "wrong_audience"
    WEAK_OFFER = "weak_offer"
    WEAK_HOOK = "weak_hook"
    UNCLEAR_PROMISE = "unclear_promise"
    CLAIM_CREDIBILITY_GAP = "claim_credibility_gap"
    CTA_FRICTION = "cta_friction"
    LANDING_MISMATCH = "landing_mismatch"
    TIMING = "timing"
    CHANNEL_MISMATCH = "channel_mismatch"
    DELIVERABILITY = "deliverability"
    TOO_GENERIC = "too_generic"
    TOO_COMPLEX = "too_complex"
    INSUFFICIENT_PROOF = "insufficient_proof"
    COMPLIANCE_WEAKENED = "compliance_weakened"
    INCONCLUSIVE_DATA = "inconclusive_data"


class GateStage(StrEnum):
    """The four-part gate stack.

    Deterministic stages (``SCHEMA``, ``CLAIMS_COMPLIANCE``) are decided by the
    system; ``MODEL_ASSISTED`` produces evidence the editor resolves; the editor
    owns the accountable call at ``HUMAN_EDITOR``. The market settles
    effectiveness after publish -- not represented here.
    """

    SCHEMA = "schema"  # 3A
    CLAIMS_COMPLIANCE = "claims_compliance"  # 3B
    MODEL_ASSISTED = "model_assisted"  # 3C
    HUMAN_EDITOR = "human_editor"  # 3D


# Risk-tier -> ordered required gate stages (the doc's risk-tier table as data).
# This is pure routing data; the code that *enforces* it is a later slice.
REQUIRED_REVIEW_BY_TIER: Mapping[RiskTier, tuple[GateStage, ...]] = {
    RiskTier.LOW: (
        GateStage.SCHEMA,
        GateStage.MODEL_ASSISTED,
        GateStage.HUMAN_EDITOR,
    ),
    RiskTier.MEDIUM: (
        GateStage.SCHEMA,
        GateStage.CLAIMS_COMPLIANCE,
        GateStage.MODEL_ASSISTED,
        GateStage.HUMAN_EDITOR,
    ),
    RiskTier.HIGH: (
        GateStage.SCHEMA,
        GateStage.CLAIMS_COMPLIANCE,
        GateStage.MODEL_ASSISTED,
        GateStage.HUMAN_EDITOR,
    ),
    RiskTier.CRITICAL: (
        GateStage.SCHEMA,
        GateStage.CLAIMS_COMPLIANCE,
        GateStage.MODEL_ASSISTED,
        GateStage.HUMAN_EDITOR,
    ),
}


def required_stages_for(tier: RiskTier) -> tuple[GateStage, ...]:
    """Return the ordered gate stages a draft at ``tier`` must clear.

    HIGH and CRITICAL share the same stage set today; their extra burden
    (compliance/legal sign-off, postmortem, experiment contract) is expressed by
    later slices, not by adding gate *stages* here.
    """

    return REQUIRED_REVIEW_BY_TIER[tier]


@dataclass(frozen=True)
class ExceptionRecord:
    """An ``APPROVED_WITH_EXCEPTION`` audit entry.

    Keeps an override out of Slack and in the system: which rule was waived, why,
    who owns it, when it lapses, and whether the rule itself should change.
    """

    rule: str
    reason: str
    owner: str
    expiration: date | None = None
    should_update_rule: bool = False

    def is_active(self, as_of: date) -> bool:
        """True if the exception still applies on ``as_of`` (inclusive of the
        expiration day). An exception with no expiration never lapses."""

        if self.expiration is None:
            return True
        return as_of <= self.expiration


def recurring_failure_categories(
    categories: Iterable[FailureCategory],
    *,
    threshold: int = 3,
) -> frozenset[FailureCategory]:
    """Categories that have failed ``threshold`` or more times.

    The negative-flywheel trigger: when the same failure recurs at or above the
    threshold it is a candidate for a new deterministic gate. Pure counting --
    deciding what gate to add is a human/later-slice call.
    """

    if threshold < 1:
        raise ValueError("threshold must be >= 1")
    counts: dict[FailureCategory, int] = {}
    for category in categories:
        counts[category] = counts.get(category, 0) + 1
    return frozenset(
        category for category, count in counts.items() if count >= threshold
    )
