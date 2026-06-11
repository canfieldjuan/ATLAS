"""Content-ops calibration-anchor selection (operating-model slice 7).

Bridges slice 5b's adversarial findings to slice 5a's calibration library: given
the finding categories an adversarial pass raised and a curated
:class:`CalibrationLibrary`, select the teachable anchors that illustrate those
failure modes. This is what finally makes the calibration set do work in a live
review -- when a draft trips an ``overclaim`` finding, the editor is shown the
curated overclaim anchor (a worked example) instead of just a bare objection.

Pure value functions -- no I/O, no Atlas imports, no DB, no LLM. The mapping from
finding category to calibration label is deliberately *partial*: only finding
categories that have a corresponding curated calibration label map; the rest
contribute no anchor rather than a forced/misleading one.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from .adversarial_pass import AdversarialFindingCategory
from .calibration_library import CalibrationExample, CalibrationLabel, CalibrationLibrary


# Finding category -> the calibration label that anchors that failure mode.
# Intentionally conservative: only the unambiguous correspondences are mapped.
# Extending this (e.g. generic_stretch -> weak_persuasion) is a follow-up once
# the finding and calibration vocabularies are reconciled.
_FINDING_LABEL_MAP: Mapping[AdversarialFindingCategory, CalibrationLabel] = {
    AdversarialFindingCategory.OVERCLAIM: CalibrationLabel.OVERCLAIM,
    AdversarialFindingCategory.VOICE_SLIP: CalibrationLabel.VOICE_DRIFT,
}


def label_for_finding_category(category: object) -> CalibrationLabel | None:
    """The calibration label anchoring ``category``, or ``None`` if unmapped.

    Lookup is value-based, so a category decoded as a plain string (e.g.
    ``"overclaim"``) resolves the same as the enum member.
    """

    for finding_category, label in _FINDING_LABEL_MAP.items():
        if finding_category == category:
            return label
    return None


def anchors_for_finding_categories(
    library: CalibrationLibrary,
    categories: Iterable[object],
) -> tuple[CalibrationExample, ...]:
    """Teachable calibration anchors for the mapped ``categories``, de-duplicated.

    For each finding category that maps to a calibration label, the library's
    teachable examples for that label are collected. Only teachable anchors
    (carrying both excerpt and reasoning) are returned -- an anchor with no copy
    to look at teaches nothing. Order follows first appearance of a category,
    then the library's order within a label; duplicates (by ``example_id``) are
    dropped so the same anchor is never shown twice.
    """

    selected: list[CalibrationExample] = []
    seen_ids: set[str] = set()
    seen_labels: set[CalibrationLabel] = set()
    for category in categories:
        label = label_for_finding_category(category)
        if label is None or label in seen_labels:
            continue
        seen_labels.add(label)
        for example in library.by_label(label):
            if example.is_teachable() and example.example_id not in seen_ids:
                seen_ids.add(example.example_id)
                selected.append(example)
    return tuple(selected)
