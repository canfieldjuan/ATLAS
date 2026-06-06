"""Content-PR review contract (operating-model slice 4).

The deterministic core of the review contract from
``docs/content_ops_operating_model.md``: a reviewer is never handed a draft and
asked "thoughts?" -- they are handed a **Content-PR** (frozen rule-pack versions
+ claims map + a required coverage matrix), and approval is *computed* from it
under one rule -- **no required rule passes silently**.

This is the slice where the earlier slices compose: slice 1's ``ReviewDecision``
is the verdict, slice 3's ``blocking_claims`` feeds it, and slice 4 adds the
coverage matrix + comment discipline. Pure value types + pure functions -- no
I/O, no Atlas imports, no DB, no LLM. The adversarial pass, calibration library,
and any LLM-driven coverage generation are later slices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 CI compatibility
    from enum import Enum

    class StrEnum(str, Enum):
        pass

from .claims_map import MappedClaim, blocking_claims
from .review_contract import ReviewDecision


def _nonempty(value: object) -> bool:
    """True only for a non-empty, non-whitespace ``str`` (None/non-str -> False)."""

    return isinstance(value, str) and bool(value.strip())


class CoverageStatus(StrEnum):
    """Status of one required-rule row in the coverage matrix."""

    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    UNRESOLVED = "unresolved"  # default -- a required row left here is a silent gap


class CommentCategory(StrEnum):
    """The category every reviewer comment must attach to.

    ``NIT`` is the escape hatch for drive-by polish: still categorized, never
    blocking. The rest mirror the doc's comment-discipline list.
    """

    BRIEF = "brief"
    BRAND_RULE = "brand_rule"
    CLAIM_REGISTRY = "claim_registry"
    COMPLIANCE = "compliance"
    CHANNEL_CONSTRAINT = "channel_constraint"
    PERFORMANCE_HYPOTHESIS = "performance_hypothesis"
    EDITORIAL_JUDGMENT = "editorial_judgment"
    NIT = "nit"


@dataclass(frozen=True)
class RulePacketVersions:
    """The frozen rule-pack version stamps a Content-PR is reviewed against.

    A review can't be trusted unless every pack version is pinned, so an
    unpinned packet blocks the verdict (see ``review_verdict``).
    """

    brief: str = ""
    brand_voice: str = ""
    claim_registry: str = ""
    compliance: str = ""
    channel_schema: str = ""

    def missing(self) -> tuple[str, ...]:
        names = ("brief", "brand_voice", "claim_registry", "compliance", "channel_schema")
        return tuple(name for name in names if not _nonempty(getattr(self, name)))

    def is_pinned(self) -> bool:
        return not self.missing()


@dataclass(frozen=True)
class CoverageRow:
    """One required-rule row of the coverage matrix.

    A required row is *resolved* only when the reviewer recorded a real status
    with cited evidence (PASS/FAIL) or marked it NOT_APPLICABLE. UNRESOLVED -- or
    a PASS/FAIL with no evidence -- is a silent gap and blocks approval.
    """

    rule_id: str
    requirement: str = ""
    required: bool = True
    status: CoverageStatus = CoverageStatus.UNRESOLVED
    evidence: str = ""

    def is_resolved(self) -> bool:
        if self.status is CoverageStatus.NOT_APPLICABLE:
            return True
        if self.status in (CoverageStatus.PASS, CoverageStatus.FAIL):
            return _nonempty(self.evidence)
        return False


@dataclass(frozen=True)
class ReviewComment:
    """A categorized, evidenced reviewer comment.

    Every comment attaches to a ``CommentCategory``; ``blocking`` is the
    reviewer's call (a NIT must never be blocking -- see ``__post_init__``).
    """

    category: CommentCategory
    message: str = ""
    evidence: str = ""
    blocking: bool = False

    def __post_init__(self) -> None:
        if self.category is CommentCategory.NIT and self.blocking:
            raise ValueError("a NIT comment cannot be blocking")


@dataclass(frozen=True)
class ContentPR:
    """What a reviewer is handed instead of a bare draft.

    Brief snapshot + frozen rule packet + the slice-3 claims map + gate results
    (the coverage matrix) + categorized comments.
    """

    asset_id: str = ""
    rule_packet: RulePacketVersions = RulePacketVersions()
    coverage: tuple[CoverageRow, ...] = ()
    claims: tuple[MappedClaim, ...] = ()
    comments: tuple[ReviewComment, ...] = ()


def unresolved_required_rows(rows: Iterable[CoverageRow]) -> tuple[CoverageRow, ...]:
    """Required coverage rows that are not resolved (the silent-gap detector)."""

    return tuple(r for r in rows if r.required and not r.is_resolved())


def failing_required_rows(rows: Iterable[CoverageRow]) -> tuple[CoverageRow, ...]:
    """Required coverage rows the reviewer explicitly failed."""

    return tuple(r for r in rows if r.required and r.status is CoverageStatus.FAIL)


def blocking_comments(comments: Iterable[ReviewComment]) -> tuple[ReviewComment, ...]:
    """Comments the reviewer marked blocking."""

    return tuple(c for c in comments if c.blocking)


def verdict_reasons(pr: ContentPR) -> tuple[str, ...]:
    """Human-readable reasons the verdict is not a clean approval (in order)."""

    reasons: list[str] = []
    missing = pr.rule_packet.missing()
    if missing:
        reasons.append(f"rule packet not pinned: {', '.join(missing)}")
    unresolved = unresolved_required_rows(pr.coverage)
    if unresolved:
        reasons.append(
            "unresolved required coverage: "
            + ", ".join(r.rule_id for r in unresolved)
        )
    failing = failing_required_rows(pr.coverage)
    if failing:
        reasons.append("failed required coverage: " + ", ".join(r.rule_id for r in failing))
    blocked_claims = blocking_claims(pr.claims)
    if blocked_claims:
        reasons.append(f"{len(blocked_claims)} blocking claim(s)")
    blocked_comments = blocking_comments(pr.comments)
    if blocked_comments:
        reasons.append(f"{len(blocked_comments)} blocking comment(s)")
    return tuple(reasons)


def review_verdict(pr: ContentPR) -> ReviewDecision:
    """Compute the verdict from a Content-PR (no required rule passes silently).

    Auto-derives only BLOCKED / REVISION_REQUIRED / APPROVED; the human-only
    states (APPROVED_WITH_EXCEPTION, ESCALATED) are never produced here.
    """

    # The review must be complete and trustworthy before it can approve at all.
    if not pr.rule_packet.is_pinned():
        return ReviewDecision.BLOCKED
    if unresolved_required_rows(pr.coverage):
        return ReviewDecision.BLOCKED
    # Trustworthy review, but content has hard failures -> back for revision.
    if (
        failing_required_rows(pr.coverage)
        or blocking_claims(pr.claims)
        or blocking_comments(pr.comments)
    ):
        return ReviewDecision.REVISION_REQUIRED
    return ReviewDecision.APPROVED
