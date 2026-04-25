from atlas_brain.services.witness_quality_propagation import (
    collect_quote_witness_ids,
    decorate_witness_quality_fields,
    normalize_witness_quality_row,
)


def test_collect_quote_witness_ids_ignores_id_only_gap_objects():
    payload = {
        "coverage_gaps": [{"witness_id": "gap:thin_segment:size_smb"}],
        "reasoning_witness_highlights": [
            {
                "witness_id": "witness:1",
                "excerpt_text": "Pricing changed at renewal.",
            }
        ],
        "anchor_examples": {
            "common_pattern": [
                {
                    "_sid": "witness:2",
                    "quote": "Support took weeks.",
                }
            ]
        },
    }

    assert collect_quote_witness_ids(payload) == {"witness:1", "witness:2"}


def test_normalize_witness_quality_row_preserves_false_verbatim():
    witness_id, quality = normalize_witness_quality_row(
        {
            "witness_id": "witness:1",
            "grounding_status": "not_grounded",
            "phrase_polarity": "",
            "phrase_subject": None,
            "phrase_role": "primary_driver",
            "phrase_verbatim": False,
            "pain_confidence": "weak",
        }
    )

    assert witness_id == "witness:1"
    assert quality == {
        "grounding_status": "not_grounded",
        "phrase_role": "primary_driver",
        "phrase_verbatim": False,
        "pain_confidence": "weak",
    }


def test_decorate_witness_quality_fields_fills_missing_fields_only():
    payload = {
        "reasoning_witness_highlights": [
            {
                "witness_id": "witness:1",
                "excerpt_text": "Pricing changed at renewal.",
                "pain_confidence": "weak",
            }
        ],
        "coverage_gaps": [{"witness_id": "witness:1"}],
    }

    decorated, stats = decorate_witness_quality_fields(
        payload,
        {
            "witness:1": {
                "grounding_status": "grounded",
                "phrase_polarity": "negative",
                "phrase_subject": "subject_vendor",
                "phrase_role": "primary_driver",
                "phrase_verbatim": True,
                "pain_confidence": "strong",
            }
        },
    )

    witness = decorated["reasoning_witness_highlights"][0]
    assert witness["grounding_status"] == "grounded"
    assert witness["phrase_polarity"] == "negative"
    assert witness["phrase_subject"] == "subject_vendor"
    assert witness["phrase_role"] == "primary_driver"
    assert witness["phrase_verbatim"] is True
    assert witness["pain_confidence"] == "weak"
    assert decorated["coverage_gaps"][0] == {"witness_id": "witness:1"}
    assert stats == {
        "witness_objects_seen": 1,
        "witness_objects_matched": 1,
        "witness_objects_updated": 1,
        "fields_written": 5,
    }


def test_decorate_witness_quality_fields_can_overwrite_when_requested():
    payload = {
        "reasoning_witness_highlights": [
            {
                "witness_id": "witness:1",
                "excerpt_text": "Pricing changed at renewal.",
                "pain_confidence": "weak",
            }
        ],
    }

    decorated, stats = decorate_witness_quality_fields(
        payload,
        {"witness:1": {"pain_confidence": "strong"}},
        overwrite=True,
    )

    assert decorated["reasoning_witness_highlights"][0]["pain_confidence"] == "strong"
    assert stats["fields_written"] == 1
