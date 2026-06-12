"""Content-ops review calibration library (operating-model slice 5a).

The deterministic core of the **review calibration library** from
``docs/content_ops_operating_model.md``: a curated reference set of worked
review examples (verdict + reasoning + label) that anchors *both* human and
model reviewers. Without examples, "brand voice" is a seance; this is the
living anti-drift set every brief is checked against, and overrides
(``APPROVED_WITH_EXCEPTION``) feed new examples back into it.

Pure value types + pure functions -- no I/O, no Atlas imports, no DB, no LLM.
The part that *requires* an LLM -- scoring a fresh draft *against* these
anchors, or generating new example text -- is a later slice. This module only
stores, classifies, and queries already-curated examples, and converts a
slice-1 :class:`ExceptionRecord` override into one.

The adversarial pass (slice 5b) and model-disagreement orchestration stay
parked per the doc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 CI compatibility
    from enum import Enum

    class StrEnum(str, Enum):
        # Match CPython's real StrEnum: str(member) is the value, so a member
        # formatted via str()/f-string yields the value, not the class-qualified
        # name, on the 3.10 fallback path too.
        __str__ = str.__str__

from .review_contract import ExceptionRecord, FailureCategory, ReviewDecision


class CalibrationLabel(StrEnum):
    """The teaching label on a calibration example (the doc's example list).

    The polarity of a label (does this example show *good* or *bad* work?) is
    decided by :data:`_POSITIVE_LABELS` / :data:`_NEGATIVE_LABELS`;
    ``BORDERLINE`` is deliberately neither -- it is the judgment-call case a
    reviewer is meant to study, not a clean exemplar.
    """

    APPROVED = "approved"
    REJECTED = "rejected"
    BORDERLINE = "borderline"
    KNOWN_DEFECT = "known_defect"
    GOOD_VOICE = "good_voice"
    VOICE_DRIFT = "voice_drift"
    OVERCLAIM = "overclaim"
    WEAK_PERSUASION = "weak_persuasion"
    STRONG_PERSUASION = "strong_persuasion"


# Label polarity as data. A label is positive (a "do this" anchor), negative (a
# "not this" anchor), or neither (BORDERLINE -- the studied judgment call).
# frozenset membership is value-based for StrEnum, so a plain decoded string
# such as "overclaim" classifies the same as CalibrationLabel.OVERCLAIM.
_POSITIVE_LABELS: frozenset[CalibrationLabel] = frozenset(
    {
        CalibrationLabel.APPROVED,
        CalibrationLabel.GOOD_VOICE,
        CalibrationLabel.STRONG_PERSUASION,
    }
)
_NEGATIVE_LABELS: frozenset[CalibrationLabel] = frozenset(
    {
        CalibrationLabel.REJECTED,
        CalibrationLabel.KNOWN_DEFECT,
        CalibrationLabel.VOICE_DRIFT,
        CalibrationLabel.OVERCLAIM,
        CalibrationLabel.WEAK_PERSUASION,
    }
)


def is_positive_label(label: object) -> bool:
    """True if ``label`` anchors *good* work (compares by value, not identity)."""

    return label in _POSITIVE_LABELS


def is_negative_label(label: object) -> bool:
    """True if ``label`` anchors a *defect* to avoid (compares by value)."""

    return label in _NEGATIVE_LABELS


def _nonempty(value: object) -> bool:
    """True only for a non-empty, non-whitespace ``str`` (None/non-str -> False).

    Mirrors the robustness contract in the sibling slices: these records may be
    built from decoded/untrusted input, so a ``None`` excerpt or reasoning
    counts as missing rather than raising on ``.strip()``.
    """

    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class CalibrationExample:
    """One curated review example: the anchor a reviewer studies.

    ``excerpt`` is the worked copy (or the relevant fragment), ``label`` is its
    teaching classification, and ``reasoning`` is *why* it earned that label --
    the part that makes the set anchor judgment instead of just listing verdicts.
    ``verdict`` and ``failure_category`` are optional cross-links into slice 1's
    vocabulary; ``source`` records provenance (``"curated"`` vs ``"override"``).
    """

    example_id: str
    excerpt: str = ""
    label: CalibrationLabel = CalibrationLabel.BORDERLINE
    reasoning: str = ""
    verdict: ReviewDecision | None = None
    failure_category: FailureCategory | None = None
    source: str = "curated"

    def is_positive(self) -> bool:
        return is_positive_label(self.label)

    def is_negative(self) -> bool:
        return is_negative_label(self.label)

    def is_teachable(self) -> bool:
        """True only when the example can actually anchor a reviewer.

        An anchor with no copy to look at, or no reasoning for its label, is
        decoration -- it teaches nothing. Both ``excerpt`` and ``reasoning``
        must be present, non-whitespace text.
        """

        return _nonempty(self.excerpt) and _nonempty(self.reasoning)


@dataclass(frozen=True)
class CalibrationLibrary:
    """An immutable, queryable set of calibration examples.

    The living reference set queried before every brief. Pure container: every
    query returns a new tuple in input order and never mutates the set (curation
    happens by building a new library, per the frozen-record convention).
    """

    examples: tuple[CalibrationExample, ...] = ()

    def by_label(self, label: CalibrationLabel) -> tuple[CalibrationExample, ...]:
        """Examples carrying ``label`` (value comparison handles decoded input)."""

        return tuple(e for e in self.examples if e.label == label)

    def by_failure_category(
        self, category: FailureCategory | None
    ) -> tuple[CalibrationExample, ...]:
        """Examples cross-linked to ``category``.

        A ``None`` query returns empty, and ``None``-category examples never
        match: an example with no recorded failure category is not silently
        swept into any category's anchor set.
        """

        if category is None:
            return ()
        return tuple(e for e in self.examples if e.failure_category == category)

    def by_verdict(
        self, verdict: ReviewDecision | None
    ) -> tuple[CalibrationExample, ...]:
        """Examples whose recorded verdict equals ``verdict`` (None never matches)."""

        if verdict is None:
            return ()
        return tuple(e for e in self.examples if e.verdict == verdict)

    def positives(self) -> tuple[CalibrationExample, ...]:
        """The "do this" anchors (good-voice / strong-persuasion / approved)."""

        return tuple(e for e in self.examples if e.is_positive())

    def negatives(self) -> tuple[CalibrationExample, ...]:
        """The "not this" anchors (drift / overclaim / weak / rejected / defect)."""

        return tuple(e for e in self.examples if e.is_negative())

    def teachable(self) -> tuple[CalibrationExample, ...]:
        """Only the anchors that actually carry copy + reasoning."""

        return tuple(e for e in self.examples if e.is_teachable())

    def labels_covered(self) -> frozenset[CalibrationLabel]:
        """The distinct labels with at least one *teachable* example.

        Only teachable anchors count as coverage: a decoded record that carries
        a label but no excerpt/reasoning cannot anchor a reviewer, so it must
        not hide a blind spot (see :meth:`missing_labels`).
        """

        return frozenset(e.label for e in self.examples if e.is_teachable())

    def missing_labels(
        self, required: Iterable[CalibrationLabel]
    ) -> tuple[CalibrationLabel, ...]:
        """Required labels with no teachable example -- the blind spots, in order.

        A calibration set that has no overclaim or voice-drift example cannot
        anchor a reviewer on those failure modes; this reports the gap so
        curation is driven by coverage, not vibes. Coverage is derived from
        teachable examples only -- a label represented solely by decoration
        (no excerpt or no reasoning) is still a gap. Preserves ``required``
        order and de-duplicates.
        """

        covered = self.labels_covered()
        seen: set[CalibrationLabel] = set()
        gaps: list[CalibrationLabel] = []
        for label in required:
            if label not in covered and label not in seen:
                seen.add(label)
                gaps.append(label)
        return tuple(gaps)


def example_from_exception(
    record: ExceptionRecord,
    *,
    excerpt: str,
    label: CalibrationLabel = CalibrationLabel.BORDERLINE,
    failure_category: FailureCategory | None = None,
    suffix: str = "",
) -> CalibrationExample:
    """Turn an ``APPROVED_WITH_EXCEPTION`` override into a calibration example.

    The doc's compounding rule: *overrides feed the calibration set, so the
    judgment layer itself compounds.* An override is by construction a
    judgment call -- a piece shipped despite a soft-rule violation -- so it
    defaults to a ``BORDERLINE`` anchor with the override's own reason as the
    teaching reasoning, the waived ``rule`` carried into the id, and provenance
    stamped ``"override"`` so curated and override-fed anchors stay
    distinguishable. ``verdict`` is fixed to ``APPROVED_WITH_EXCEPTION`` (the
    only decision an :class:`ExceptionRecord` represents).

    The same soft rule is often waived repeatedly (that is what
    ``should_update_rule`` exists for), and repeats would collide on
    ``override:{rule}`` -- ``example_id`` is **not** unique across a library
    unless the caller passes a distinguishing ``suffix`` (an asset id, date,
    or sequence number), which is appended as ``override:{rule}:{suffix}``.
    """

    example_id = f"override:{record.rule}"
    if _nonempty(suffix):
        example_id = f"{example_id}:{suffix}"
    return CalibrationExample(
        example_id=example_id,
        excerpt=excerpt,
        label=label,
        reasoning=record.reason,
        verdict=ReviewDecision.APPROVED_WITH_EXCEPTION,
        failure_category=failure_category,
        source="override",
    )
