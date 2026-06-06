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
from types import MappingProxyType
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
    routing tier that decides how much review an asset must clear. The member
    *names* deliberately mirror ``RiskLevel`` so the two read alike, but the
    string *values* differ in case (``RiskTier.LOW == "low"`` vs
    ``RiskLevel.LOW == "LOW"``) -- do not compare the two enums' values
    directly.
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
# Wrapped in MappingProxyType (per the extracted_content_pipeline constant-table
# convention, e.g. control_surfaces.OUTPUT_CATALOG) so accidental mutation of the
# routing table fails loudly instead of silently corrupting review burden.
REQUIRED_REVIEW_BY_TIER: Mapping[RiskTier, tuple[GateStage, ...]] = MappingProxyType(
    {
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
)


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


# --------------------------------------------------------------------------
# Slice 2: stage-0 triage + stage-7 experiment contract.
# Additive schemas; the routing that *requires* them is a later slice.
# --------------------------------------------------------------------------


def _missing_text(value: object) -> bool:
    """True if a required text value should count as missing.

    A value is present only if it is a non-empty, non-whitespace ``str``.
    ``None`` or any non-``str`` (e.g. JSON ``null`` deserialized into the field)
    counts as missing rather than raising ``AttributeError`` on ``.strip()`` --
    these schemas are typed by convention, not enforced at runtime.
    """

    return not (isinstance(value, str) and value.strip())


def _missing_positive_int(value: object) -> bool:
    """True if a required positive-int value should count as missing.

    Present only if it is a real ``int`` (not ``bool``) greater than zero.
    ``None`` or any non-int counts as missing rather than raising ``TypeError``
    on the ``> 0`` comparison -- same robustness contract as ``_missing_text``.
    """

    return not (isinstance(value, int) and not isinstance(value, bool) and value > 0)


class TriageDecision(StrEnum):
    """Stage-0 verdict: should this asset exist at all?

    The gate that keeps the pipeline from becoming an efficient machine for
    producing polished landfill. ``CLONE_WINNER`` reuses an existing proven
    asset instead of creating net-new.
    """

    CREATE = "create"
    CLONE_WINNER = "clone_winner"
    DEFER = "defer"
    REJECT = "reject"


# Required stage-0 text inputs, in declaration order. opportunity_size and
# reuse_potential are optional context, not completeness-gating.
_TRIAGE_REQUIRED_TEXT = (
    "audience_segment",
    "lifecycle_stage",
    "business_goal",
    "expected_behavior_change",
    "channel",
    "why_now",
)


@dataclass(frozen=True)
class TriageRequest:
    """Stage-0 triage inputs (what a triage decision is made from).

    Completeness only: ``missing_fields`` reports blank required inputs so a
    triage can't be waved through empty. It does not judge whether the asset is
    *worth* making -- that is the human ``TriageDecision``.
    """

    audience_segment: str = ""
    lifecycle_stage: str = ""
    business_goal: str = ""
    expected_behavior_change: str = ""
    channel: str = ""
    why_now: str = ""
    opportunity_size: str = ""  # optional context
    reuse_potential: str = ""  # optional context
    risk_tier: RiskTier | None = None

    def missing_fields(self) -> tuple[str, ...]:
        missing = [
            name
            for name in _TRIAGE_REQUIRED_TEXT
            if _missing_text(getattr(self, name))
        ]
        if not isinstance(self.risk_tier, RiskTier):
            missing.append("risk_tier")
        return tuple(missing)

    def is_complete(self) -> bool:
        return not self.missing_fields()


# Required stage-7 text fields, in declaration order. secondary_metric is
# required per docs/content_ops_operating_model.md (stage 7 "Required fields");
# the two numeric fields are checked separately (must be > 0).
_EXPERIMENT_REQUIRED_TEXT = (
    "hypothesis",
    "primary_metric",
    "secondary_metric",
    "audience",
    "comparison",
    "success_definition",
    "inconclusive_definition",
    "decision_if_works",
    "decision_if_not",
)


@dataclass(frozen=True)
class ExperimentContract:
    """Stage-7 measurement plan, frozen *before* publish.

    Defining this up front is what keeps the verdict ledger from rotting into
    "the graph went up and nobody wants to argue." Completeness only: it checks
    the plan is filled, not that the hypothesis is sound. The required field set
    mirrors stage 7 in ``docs/content_ops_operating_model.md``.
    """

    hypothesis: str = ""
    primary_metric: str = ""
    secondary_metric: str = ""  # required (a guardrail metric, per the doc)
    audience: str = ""
    comparison: str = ""  # control or baseline being compared against
    success_definition: str = ""  # what counts as "worked"
    inconclusive_definition: str = ""  # what counts as "inconclusive"
    decision_if_works: str = ""
    decision_if_not: str = ""
    attribution_window_days: int = 0
    min_sample_size: int = 0

    def missing_fields(self) -> tuple[str, ...]:
        missing = [
            name
            for name in _EXPERIMENT_REQUIRED_TEXT
            if _missing_text(getattr(self, name))
        ]
        if _missing_positive_int(self.attribution_window_days):
            missing.append("attribution_window_days")
        if _missing_positive_int(self.min_sample_size):
            missing.append("min_sample_size")
        return tuple(missing)

    def is_complete(self) -> bool:
        return not self.missing_fields()
