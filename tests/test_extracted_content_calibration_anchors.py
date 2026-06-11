"""Unit tests for slice 7: calibration-anchor selection.

Pure value functions bridging slice 5b findings to slice 5a anchors; no DB,
no async, no Atlas imports, no LLM.
"""

from __future__ import annotations

from extracted_content_pipeline.adversarial_pass import AdversarialFindingCategory
from extracted_content_pipeline.calibration_anchors import (
    anchors_for_finding_categories,
    label_for_finding_category,
)
from extracted_content_pipeline.calibration_library import (
    CalibrationExample,
    CalibrationLabel,
    CalibrationLibrary,
)


def _anchor(
    example_id: str,
    label: CalibrationLabel,
    *,
    excerpt: str = "worked example copy",
    reasoning: str = "why it earned the label",
) -> CalibrationExample:
    return CalibrationExample(
        example_id=example_id, excerpt=excerpt, label=label, reasoning=reasoning
    )


# -- label_for_finding_category ----------------------------------------------


def test_mapped_finding_categories_resolve_to_labels() -> None:
    assert label_for_finding_category(AdversarialFindingCategory.OVERCLAIM) is CalibrationLabel.OVERCLAIM
    assert label_for_finding_category(AdversarialFindingCategory.VOICE_SLIP) is CalibrationLabel.VOICE_DRIFT


def test_unmapped_finding_category_returns_none() -> None:
    assert label_for_finding_category(AdversarialFindingCategory.AMBIGUITY) is None
    assert label_for_finding_category(AdversarialFindingCategory.MISSING_PROOF) is None


def test_label_lookup_matches_decoded_string_category() -> None:
    assert label_for_finding_category("overclaim") is CalibrationLabel.OVERCLAIM
    assert label_for_finding_category("totally_unknown") is None


# -- anchors_for_finding_categories ------------------------------------------


def _library() -> CalibrationLibrary:
    return CalibrationLibrary(
        examples=(
            _anchor("oc1", CalibrationLabel.OVERCLAIM),
            _anchor("oc2", CalibrationLabel.OVERCLAIM),
            _anchor("vd1", CalibrationLabel.VOICE_DRIFT),
            _anchor("gv1", CalibrationLabel.GOOD_VOICE),
            _anchor("oc_blank", CalibrationLabel.OVERCLAIM, excerpt="", reasoning=""),
        )
    )


def test_anchors_selected_for_mapped_fired_categories() -> None:
    picked = anchors_for_finding_categories(
        _library(), (AdversarialFindingCategory.OVERCLAIM,)
    )
    # Both teachable overclaim anchors; the blank one is dropped.
    assert tuple(a.example_id for a in picked) == ("oc1", "oc2")


def test_unmapped_category_contributes_no_anchor() -> None:
    assert anchors_for_finding_categories(
        _library(), (AdversarialFindingCategory.MISSING_PROOF,)
    ) == ()


def test_anchors_dedup_repeated_category_and_keep_first_order() -> None:
    picked = anchors_for_finding_categories(
        _library(),
        (
            AdversarialFindingCategory.OVERCLAIM,
            AdversarialFindingCategory.VOICE_SLIP,
            AdversarialFindingCategory.OVERCLAIM,  # repeat -> not re-collected
        ),
    )
    assert tuple(a.example_id for a in picked) == ("oc1", "oc2", "vd1")


def test_good_voice_anchor_never_selected_without_a_mapped_finding() -> None:
    # No fired category maps to GOOD_VOICE, so that anchor never surfaces here.
    picked = anchors_for_finding_categories(
        _library(), (AdversarialFindingCategory.OVERCLAIM,)
    )
    assert all(a.label == CalibrationLabel.OVERCLAIM for a in picked)


def test_empty_library_or_no_categories_is_empty() -> None:
    assert anchors_for_finding_categories(CalibrationLibrary(), (AdversarialFindingCategory.OVERCLAIM,)) == ()
    assert anchors_for_finding_categories(_library(), ()) == ()


def test_decoded_string_category_selects_anchors() -> None:
    picked = anchors_for_finding_categories(_library(), ("overclaim",))
    assert tuple(a.example_id for a in picked) == ("oc1", "oc2")
