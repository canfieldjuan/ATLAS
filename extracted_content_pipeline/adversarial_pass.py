"""Content-ops adversarial review pass (operating-model slice 5b).

The deterministic core of the **adversarial pass** from
``docs/content_ops_operating_model.md``: a second, independent model-assisted
pass whose only job is to find the strongest reason a draft should *not* ship
-- overclaims, ambiguity, reader objections, promise/CTA mismatch, generic
stretches, missing proof, voice slips. This module is the data model for that
pass plus the deterministic disagreement/merge helpers between two independent
passes. Pure value types + pure functions -- no I/O, no Atlas imports, no DB,
no LLM.

The pass is **not a judge**: findings are evidence the accountable editor
resolves. That invariant is enforced at the seam -- a finding converts to a
*non-blocking* :class:`~.content_pr.ReviewComment`; only the editor decides
what blocks. Running the prompts that *produce* findings, and the
model-disagreement orchestration (route-to-human on an A-pass/B-fail split),
stay parked per the doc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 CI compatibility
    from enum import Enum

    class StrEnum(str, Enum):
        # Match CPython's real StrEnum: str(member) is the value, so the
        # [adversarial:...] prefix formats identically on both interpreter
        # paths (3.10's plain str+Enum would format as the class-qualified
        # name under str()).
        __str__ = str.__str__

from .content_pr import CommentCategory, ReviewComment


class AdversarialFindingCategory(StrEnum):
    """The doc's adversarial-pass target list: reasons a draft should not ship."""

    OVERCLAIM = "overclaim"
    AMBIGUITY = "ambiguity"
    READER_OBJECTION = "reader_objection"
    PROMISE_CTA_MISMATCH = "promise_cta_mismatch"
    GENERIC_STRETCH = "generic_stretch"
    MISSING_PROOF = "missing_proof"
    VOICE_SLIP = "voice_slip"


def _nonempty(value: object) -> bool:
    """True only for a non-empty, non-whitespace ``str`` (None/non-str -> False).

    Same robustness contract as the sibling slices: findings may be decoded from
    a model's JSON output, so a ``None`` message counts as missing, never raises.
    """

    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class AdversarialFinding:
    """One "strongest reason this should not ship" from an adversarial pass.

    ``message`` is the objection; ``evidence`` is the quoted draft text it
    points at; ``location`` is where in the draft. A finding with no message or
    no evidence is noise, not a finding -- see :meth:`is_substantiated`.
    """

    category: AdversarialFindingCategory
    message: str = ""
    evidence: str = ""
    location: str = ""

    def is_substantiated(self) -> bool:
        """True only when the finding carries both an objection and evidence.

        An unsubstantiated finding cannot anchor an editor decision; the
        substantiated filter is what keeps a chatty pass from flooding review.
        """

        return _nonempty(self.message) and _nonempty(self.evidence)


@dataclass(frozen=True)
class AdversarialPass:
    """One independent adversarial pass over a draft.

    ``source`` records the prompt/model identity that produced the pass (e.g.
    ``"adversarial-prompt@v2 / model-b"``) so two passes are distinguishable;
    independence of the two sources is the caller's responsibility.
    """

    pass_id: str
    source: str = ""
    findings: tuple[AdversarialFinding, ...] = ()

    def categories(self) -> frozenset[AdversarialFindingCategory]:
        """The distinct finding categories this pass raised."""

        return frozenset(f.category for f in self.findings)

    def substantiated(self) -> tuple[AdversarialFinding, ...]:
        """Only the findings carrying both message and evidence, in order."""

        return tuple(f for f in self.findings if f.is_substantiated())


def corroborated_categories(
    first: AdversarialPass, second: AdversarialPass
) -> frozenset[AdversarialFindingCategory]:
    """Categories BOTH passes raised independently (the strongest signal).

    StrEnum set intersection is value-based, so a category decoded as a plain
    string in one pass still corroborates the enum member in the other.
    """

    return first.categories() & second.categories()


def disagreement_categories(
    first: AdversarialPass, second: AdversarialPass
) -> frozenset[AdversarialFindingCategory]:
    """Categories exactly one pass raised (the disagreement surface).

    This is the *data* for the parked orchestration: what to do about a
    disagreement (route to human, log the override) is deliberately not decided
    here -- deterministic blockers always block and model findings always go to
    a human regardless, per the doc.
    """

    return first.categories() ^ second.categories()


def merge_findings(
    first: AdversarialPass, second: AdversarialPass
) -> tuple[AdversarialFinding, ...]:
    """All findings from both passes, first-pass order then second, de-duplicated.

    Only *exact* duplicates (same category, message, evidence, location) merge;
    two differently-worded objections in the same category are both kept --
    collapsing them would be a judgment call, and this module does not judge.
    """

    merged: list[AdversarialFinding] = []
    seen: set[AdversarialFinding] = set()
    for finding in first.findings + second.findings:
        if finding not in seen:
            seen.add(finding)
            merged.append(finding)
    return tuple(merged)


# Finding category -> review-comment category. VOICE_SLIP maps to the brand-rule
# lane; every other adversarial objection lands in the editor's judgment lane
# rather than pretending to be a deterministic gate result. Data, not branches.
_FINDING_COMMENT_CATEGORY: Mapping[
    AdversarialFindingCategory, CommentCategory
] = {
    AdversarialFindingCategory.VOICE_SLIP: CommentCategory.BRAND_RULE,
}


def comment_from_finding(finding: AdversarialFinding) -> ReviewComment:
    """Convert an adversarial finding into a reviewer comment -- never blocking.

    The "still not a judge" invariant lives here: a model-found objection
    enters the Content-PR as evidence (``blocking=False``); only the
    accountable editor escalates it to blocking. The finding's category is
    carried in the message prefix so the editor sees which failure mode the
    pass claimed.
    """

    category = _FINDING_COMMENT_CATEGORY.get(
        finding.category, CommentCategory.EDITORIAL_JUDGMENT
    )
    prefix = f"[adversarial:{finding.category}]"
    # A decoded None/blank message or evidence counts as missing -- the prefix
    # alone is the message, and "None" never leaks into editor-facing text.
    message = f"{prefix} {finding.message}" if _nonempty(finding.message) else prefix
    return ReviewComment(
        category=category,
        message=message,
        evidence=finding.evidence if _nonempty(finding.evidence) else "",
        blocking=False,
    )
