"""Tests for v2 phrase_metadata normalization + reader helpers (Phase 1a).

Covers:
- `_coerce_legacy_phrase_arrays`: legacy arrays stay list[str] under all inputs
- `_normalize_phrase_metadata`: canonicalization, text invariant, enum coercion
- Reader helpers in `_b2b_phrase_metadata`: v1 safe defaults, v2 lookups
- `_compute_derived_fields` integration: conditional version bump 3 vs 4
"""

from __future__ import annotations

import copy

from atlas_brain.autonomous.tasks._b2b_phrase_metadata import (
    enrichment_schema_version,
    is_v2_tagged,
    phrase_metadata_by_field,
    phrase_metadata_map,
    phrase_tag,
)
from atlas_brain.autonomous.tasks.b2b_enrichment import (
    _PHRASE_METADATA_FIELDS,
    _coerce_legacy_phrase_arrays,
    _compute_derived_fields,
    _normalize_phrase_metadata,
    _normalize_tag_value,
)


# ---------------------------------------------------------------------------
# _coerce_legacy_phrase_arrays
# ---------------------------------------------------------------------------


def _empty_arrays() -> dict:
    return {field: [] for field in _PHRASE_METADATA_FIELDS}


def test_coerce_preserves_clean_list_of_strings():
    r = _empty_arrays()
    r["specific_complaints"] = ["a", "b"]
    r["pricing_phrases"] = ["c"]
    _coerce_legacy_phrase_arrays(r)
    assert r["specific_complaints"] == ["a", "b"]
    assert r["pricing_phrases"] == ["c"]


def test_coerce_extracts_text_from_dict_entries():
    r = _empty_arrays()
    r["specific_complaints"] = [{"text": "complaint1"}, {"text": "  complaint2  "}]
    _coerce_legacy_phrase_arrays(r)
    assert r["specific_complaints"] == ["complaint1", "complaint2"]


def test_coerce_drops_dicts_without_text():
    r = _empty_arrays()
    r["specific_complaints"] = [{"text": "kept"}, {"no_text": "dropped"}, {"text": ""}]
    _coerce_legacy_phrase_arrays(r)
    assert r["specific_complaints"] == ["kept"]


def test_coerce_handles_mixed_and_garbage():
    r = _empty_arrays()
    r["specific_complaints"] = ["ok", {"text": "dict-ok"}, 42, None, "  trim  "]
    _coerce_legacy_phrase_arrays(r)
    assert r["specific_complaints"] == ["ok", "dict-ok", "trim"]


def test_coerce_creates_missing_fields_as_empty():
    r = {}
    _coerce_legacy_phrase_arrays(r)
    for field in _PHRASE_METADATA_FIELDS:
        assert r[field] == []


def test_coerce_non_list_values_become_empty():
    r = {
        "specific_complaints": "not a list",
        "pricing_phrases": None,
        "feature_gaps": 42,
        "quotable_phrases": [],
        "recommendation_language": [],
        "positive_aspects": [],
    }
    _coerce_legacy_phrase_arrays(r)
    for field in ("specific_complaints", "pricing_phrases", "feature_gaps"):
        assert r[field] == []


# ---------------------------------------------------------------------------
# _normalize_tag_value
# ---------------------------------------------------------------------------

_SUBJECT_ENUM = ("subject_vendor", "alternative", "self", "third_party", "unclear")


def test_normalize_tag_accepts_valid_value():
    result, coerced = _normalize_tag_value("subject_vendor", _SUBJECT_ENUM)
    assert result == "subject_vendor"
    assert coerced is False


def test_normalize_tag_lowercases_without_counting_as_coercion():
    result, coerced = _normalize_tag_value("SUBJECT_VENDOR", _SUBJECT_ENUM)
    assert result == "subject_vendor"
    assert coerced is False


def test_normalize_tag_coerces_unknown_to_unclear_and_counts():
    result, coerced = _normalize_tag_value("WEIRD_VALUE", _SUBJECT_ENUM)
    assert result == "unclear"
    assert coerced is True


def test_normalize_tag_none_is_unclear_not_coerced():
    result, coerced = _normalize_tag_value(None, _SUBJECT_ENUM)
    assert result == "unclear"
    assert coerced is False


def test_normalize_tag_non_string_is_unclear_not_coerced():
    result, coerced = _normalize_tag_value(42, _SUBJECT_ENUM)
    assert result == "unclear"
    assert coerced is False


# ---------------------------------------------------------------------------
# _normalize_phrase_metadata
# ---------------------------------------------------------------------------


def test_normalize_metadata_all_provided():
    r = _empty_arrays()
    r["specific_complaints"] = ["c1", "c2"]
    r["pricing_phrases"] = ["p1"]
    r["phrase_metadata"] = [
        {"field": "specific_complaints", "index": 0, "text": "c1",
         "subject": "subject_vendor", "polarity": "negative",
         "role": "primary_driver", "verbatim": True},
        {"field": "specific_complaints", "index": 1, "text": "c2",
         "subject": "subject_vendor", "polarity": "negative",
         "role": "supporting_context", "verbatim": True},
        {"field": "pricing_phrases", "index": 0, "text": "p1",
         "subject": "subject_vendor", "polarity": "negative",
         "role": "primary_driver", "verbatim": True,
         "category_hint": "pricing"},
    ]
    canonical, counters = _normalize_phrase_metadata(r)
    assert len(canonical) == 3
    assert canonical[2]["category_hint"] == "pricing"
    assert counters["llm_provided_rows"] == 3
    assert counters["llm_missing_rows"] == 0
    assert counters["text_mismatch_rows"] == 0
    assert counters["unknown_tag_coercions"] == 0


def test_normalize_metadata_unknown_tags_coerced_to_unclear():
    r = _empty_arrays()
    r["specific_complaints"] = ["c1"]
    r["phrase_metadata"] = [
        {"field": "specific_complaints", "index": 0, "text": "c1",
         "subject": "WEIRD", "polarity": "BOGUS", "role": "bad_role",
         "verbatim": True},
    ]
    canonical, counters = _normalize_phrase_metadata(r)
    assert canonical[0]["subject"] == "unclear"
    assert canonical[0]["polarity"] == "unclear"
    assert canonical[0]["role"] == "unclear"
    assert counters["unknown_tag_coercions"] == 3


def test_normalize_metadata_text_mismatch_forces_verbatim_false():
    r = _empty_arrays()
    r["specific_complaints"] = ["real text"]
    r["phrase_metadata"] = [
        {"field": "specific_complaints", "index": 0, "text": "WRONG TEXT",
         "subject": "subject_vendor", "polarity": "negative",
         "role": "primary_driver", "verbatim": True},
    ]
    canonical, counters = _normalize_phrase_metadata(r)
    assert canonical[0]["text"] == "real text"  # overwritten with legacy value
    assert canonical[0]["verbatim"] is False  # forced off on mismatch
    assert counters["text_mismatch_rows"] == 1


def test_normalize_metadata_fills_defaults_for_missing_rows():
    r = _empty_arrays()
    r["specific_complaints"] = ["c1", "c2"]
    r["phrase_metadata"] = [
        {"field": "specific_complaints", "index": 0, "text": "c1",
         "subject": "subject_vendor", "polarity": "negative",
         "role": "primary_driver", "verbatim": True},
    ]
    canonical, counters = _normalize_phrase_metadata(r)
    assert len(canonical) == 2
    assert canonical[1]["subject"] == "unclear"
    assert canonical[1]["verbatim"] is False
    assert counters["llm_missing_rows"] == 1


def test_normalize_metadata_missing_entirely_defaults_all():
    r = _empty_arrays()
    r["specific_complaints"] = ["c1"]
    r["pricing_phrases"] = ["p1"]
    # phrase_metadata absent entirely
    canonical, counters = _normalize_phrase_metadata(r)
    assert len(canonical) == 2
    assert counters["llm_missing_rows"] == 2
    assert counters["llm_provided_rows"] == 0


# ---------------------------------------------------------------------------
# Reader helpers (_b2b_phrase_metadata)
# ---------------------------------------------------------------------------


def test_reader_helpers_v1_short_circuit():
    v1 = {
        "enrichment_schema_version": 3,
        "specific_complaints": ["I can't cancel"],
    }
    assert enrichment_schema_version(v1) == 3
    assert is_v2_tagged(v1) is False
    assert phrase_metadata_by_field(v1, "specific_complaints") == []
    assert phrase_metadata_map(v1) == {}
    assert phrase_tag(v1, "specific_complaints", 0, "subject", "fallback") == "fallback"


def test_reader_helpers_none_input_is_safe():
    assert enrichment_schema_version(None) == 0
    assert is_v2_tagged(None) is False
    assert phrase_metadata_by_field(None, "any") == []
    assert phrase_metadata_map(None) == {}


def test_reader_helpers_v2_returns_rows_in_index_order():
    v2 = {
        "enrichment_schema_version": 4,
        "specific_complaints": ["a", "b", "c"],
        "phrase_metadata": [
            {"field": "specific_complaints", "index": 2, "text": "c"},
            {"field": "specific_complaints", "index": 0, "text": "a"},
            {"field": "specific_complaints", "index": 1, "text": "b"},
        ],
    }
    rows = phrase_metadata_by_field(v2, "specific_complaints")
    assert [r["index"] for r in rows] == [0, 1, 2]


def test_reader_helpers_map_keyed_by_field_index():
    v2 = {
        "enrichment_schema_version": 4,
        "specific_complaints": ["a"],
        "pricing_phrases": ["p"],
        "phrase_metadata": [
            {"field": "specific_complaints", "index": 0, "text": "a",
             "subject": "subject_vendor"},
            {"field": "pricing_phrases", "index": 0, "text": "p",
             "subject": "self"},
        ],
    }
    mp = phrase_metadata_map(v2)
    assert mp[("specific_complaints", 0)]["subject"] == "subject_vendor"
    assert mp[("pricing_phrases", 0)]["subject"] == "self"


def test_reader_helpers_drop_malformed_rows():
    v2 = {
        "enrichment_schema_version": 4,
        "specific_complaints": ["a"],
        "phrase_metadata": [
            None,
            "string-not-dict",
            {"field": "", "index": 0, "text": "bad"},
            {"field": "specific_complaints", "index": "nope", "text": "bad"},
            {"field": "specific_complaints", "index": 0, "text": "a",
             "subject": "subject_vendor"},
        ],
    }
    mp = phrase_metadata_map(v2)
    assert list(mp.keys()) == [("specific_complaints", 0)]


# ---------------------------------------------------------------------------
# _compute_derived_fields integration
# ---------------------------------------------------------------------------


def _baseline_v1_result() -> dict:
    return {
        "specific_complaints": [],
        "pricing_phrases": [],
        "feature_gaps": [],
        "quotable_phrases": [],
        "recommendation_language": [],
        "positive_aspects": [],
        "competitors_mentioned": [],
        "event_mentions": [],
        "churn_signals": {},
        "reviewer_context": {},
        "budget_signals": {},
        "use_case": {},
        "urgency_indicators": {},
        "sentiment_trajectory": {},
        "buyer_authority": {},
        "timeline": {},
        "contract_context": {},
    }


def _baseline_source_row() -> dict:
    return {
        "id": "unit-review",
        "vendor_name": "TestVendor",
        "source": "g2",
        "rating": 2.0,
        "rating_max": 5,
        "content_type": "review",
        "review_text": "test",
        "raw_metadata": {"source_weight": 0.7},
        "summary": None,
        "reviewer_company": None,
        "reviewer_title": None,
    }


def test_compute_derived_fields_v1_path_keeps_schema_3():
    r = _baseline_v1_result()
    r["specific_complaints"] = ["too slow"]
    out = _compute_derived_fields(copy.deepcopy(r), _baseline_source_row())
    assert enrichment_schema_version(out) == 3
    assert "phrase_metadata" not in out
    assert not is_v2_tagged(out)


def test_compute_derived_fields_v2_path_bumps_to_schema_4():
    r = _baseline_v1_result()
    r["specific_complaints"] = ["too slow"]
    r["phrase_metadata"] = [
        {"field": "specific_complaints", "index": 0, "text": "too slow",
         "subject": "subject_vendor", "polarity": "negative",
         "role": "primary_driver", "verbatim": True},
    ]
    out = _compute_derived_fields(copy.deepcopy(r), _baseline_source_row())
    assert enrichment_schema_version(out) == 4
    assert is_v2_tagged(out)
    mp = phrase_metadata_map(out)
    assert mp[("specific_complaints", 0)]["subject"] == "subject_vendor"


def test_compute_derived_fields_coerces_adversarial_dicts_in_legacy_arrays():
    r = _baseline_v1_result()
    # LLM mistakenly put dicts into the legacy string array
    r["specific_complaints"] = [{"text": "too slow", "subject": "subject_vendor"}]
    r["phrase_metadata"] = [
        {"field": "specific_complaints", "index": 0, "text": "too slow",
         "subject": "subject_vendor", "polarity": "negative",
         "role": "primary_driver", "verbatim": True},
    ]
    out = _compute_derived_fields(copy.deepcopy(r), _baseline_source_row())
    # Legacy array MUST be coerced back to list[str]
    assert out["specific_complaints"] == ["too slow"]
    assert all(isinstance(x, str) for x in out["specific_complaints"])
    # v2 metadata still populated
    assert enrichment_schema_version(out) == 4


def test_compute_derived_fields_malformed_metadata_with_phrases_falls_back_to_v3():
    """LLM returned a metadata list but zero valid rows, while legacy arrays
    hold phrases. That's a malformed attempt; route to v3 legacy path."""
    r = _baseline_v1_result()
    r["specific_complaints"] = ["real phrase"]
    # All source metadata rows are garbage (no valid field/index)
    r["phrase_metadata"] = [
        {"not_field": "specific_complaints", "not_index": 0, "text": "real phrase"},
        "not-a-dict",
        None,
        {},
    ]
    out = _compute_derived_fields(copy.deepcopy(r), _baseline_source_row())
    assert enrichment_schema_version(out) == 3, \
        "malformed metadata with phrases should fall back to v3"
    assert "phrase_metadata" not in out
    assert not is_v2_tagged(out)


def test_compute_derived_fields_empty_metadata_with_phrases_falls_back_to_v3():
    """Legacy arrays have phrases but metadata is an empty list -- the LLM
    forgot to tag anything. Fall back to v3."""
    r = _baseline_v1_result()
    r["specific_complaints"] = ["something"]
    r["phrase_metadata"] = []
    out = _compute_derived_fields(copy.deepcopy(r), _baseline_source_row())
    assert enrichment_schema_version(out) == 3
    assert "phrase_metadata" not in out


def test_compute_derived_fields_zero_phrase_review_stays_v4_on_empty_metadata():
    """Valid empty extraction: the review had no phrases worth extracting,
    LLM correctly returned an empty metadata array. This IS v4."""
    r = _baseline_v1_result()  # all six legacy arrays empty
    r["phrase_metadata"] = []
    out = _compute_derived_fields(copy.deepcopy(r), _baseline_source_row())
    assert enrichment_schema_version(out) == 4, \
        "zero-phrase review with empty metadata list should be valid v4"
    assert out.get("phrase_metadata") == []
    assert is_v2_tagged(out), \
        "v4 row with empty phrase_metadata should still be schema-aware"


def test_compute_derived_fields_partial_metadata_coverage_bumps_to_v4():
    """LLM tagged some phrases but not others. As long as one valid row was
    provided, bump to v4 and fill defaults for the rest."""
    r = _baseline_v1_result()
    r["specific_complaints"] = ["tagged", "untagged"]
    r["phrase_metadata"] = [
        {"field": "specific_complaints", "index": 0, "text": "tagged",
         "subject": "subject_vendor", "polarity": "negative",
         "role": "primary_driver", "verbatim": True},
    ]
    out = _compute_derived_fields(copy.deepcopy(r), _baseline_source_row())
    assert enrichment_schema_version(out) == 4
    mp = phrase_metadata_map(out)
    assert mp[("specific_complaints", 0)]["subject"] == "subject_vendor"
    assert mp[("specific_complaints", 1)]["subject"] == "unclear"


def test_is_v2_tagged_accepts_empty_metadata_list():
    """v4 row with an empty phrase_metadata list is still v2-tagged --
    it represents a legitimate zero-phrase valid extraction."""
    v2_empty = {"enrichment_schema_version": 4, "phrase_metadata": []}
    assert is_v2_tagged(v2_empty) is True


def test_is_v2_tagged_rejects_missing_metadata_field():
    """v4 declared but no phrase_metadata key -- malformed; not v2-tagged."""
    v4_broken = {"enrichment_schema_version": 4}
    assert is_v2_tagged(v4_broken) is False


def test_is_v2_tagged_rejects_non_list_metadata():
    """v4 with non-list phrase_metadata is malformed."""
    v4_bad = {"enrichment_schema_version": 4, "phrase_metadata": "not-a-list"}
    assert is_v2_tagged(v4_bad) is False


# ---------------------------------------------------------------------------
# Phase 1b: write-time grounding gate via _normalize_phrase_metadata
# ---------------------------------------------------------------------------


def test_normalize_metadata_grounding_pass_keeps_verbatim_true():
    """LLM said verbatim=True and the phrase IS in the source -- keep True."""
    r = _empty_arrays()
    r["specific_complaints"] = ["too expensive"]
    r["phrase_metadata"] = [
        {"field": "specific_complaints", "index": 0, "text": "too expensive",
         "subject": "subject_vendor", "polarity": "negative",
         "role": "primary_driver", "verbatim": True},
    ]
    source_row = {
        "summary": None,
        "review_text": "We tried it but it was too expensive for us.",
    }
    canonical, counters = _normalize_phrase_metadata(r, source_row)
    assert canonical[0]["verbatim"] is True
    assert counters["verbatim_grounding_wins"] == 1
    assert counters["verbatim_grounding_failures"] == 0


def test_normalize_metadata_grounding_fail_coerces_verbatim_false():
    """LLM said verbatim=True but the phrase is NOT in the source. Coerce."""
    r = _empty_arrays()
    r["specific_complaints"] = ["paraphrased complaint"]
    r["phrase_metadata"] = [
        {"field": "specific_complaints", "index": 0, "text": "paraphrased complaint",
         "subject": "subject_vendor", "polarity": "negative",
         "role": "primary_driver", "verbatim": True},
    ]
    source_row = {
        "summary": None,
        "review_text": "Something completely different about the product.",
    }
    canonical, counters = _normalize_phrase_metadata(r, source_row)
    assert canonical[0]["verbatim"] is False
    assert counters["verbatim_grounding_failures"] == 1
    assert counters["verbatim_grounding_wins"] == 0


def test_normalize_metadata_grounding_normalization_recovers_curly_quotes():
    """Phrase has typographic apostrophe; source has ASCII. Normalized
    grounding should still pass and keep verbatim=True."""
    r = _empty_arrays()
    # Phrase as extracted by LLM with curly quote (\u2019)
    r["specific_complaints"] = ["it\u2019s broken"]
    r["phrase_metadata"] = [
        {"field": "specific_complaints", "index": 0, "text": "it\u2019s broken",
         "subject": "subject_vendor", "polarity": "negative",
         "role": "primary_driver", "verbatim": True},
    ]
    source_row = {
        "summary": None,
        "review_text": "I noticed it's broken when I tried it last week.",
    }
    canonical, counters = _normalize_phrase_metadata(r, source_row)
    assert canonical[0]["verbatim"] is True
    assert counters["verbatim_grounding_wins"] == 1


def test_normalize_metadata_grounding_skipped_when_source_row_omitted():
    """Phase 1a backwards compat: when source_row is None, no grounding
    runs. LLM verbatim flag is honored as declared (modulo text-invariant)."""
    r = _empty_arrays()
    r["specific_complaints"] = ["never appears in any review"]
    r["phrase_metadata"] = [
        {"field": "specific_complaints", "index": 0,
         "text": "never appears in any review",
         "subject": "subject_vendor", "polarity": "negative",
         "role": "primary_driver", "verbatim": True},
    ]
    canonical, counters = _normalize_phrase_metadata(r)  # no source_row
    assert canonical[0]["verbatim"] is True  # not coerced
    assert counters["verbatim_grounding_failures"] == 0
    assert counters["verbatim_grounding_wins"] == 0


def test_normalize_metadata_grounding_skipped_when_llm_said_verbatim_false():
    """LLM declared verbatim=False; grounding is unnecessary. Counters
    untouched and verbatim stays False without invoking the helper."""
    r = _empty_arrays()
    r["specific_complaints"] = ["a phrase"]
    r["phrase_metadata"] = [
        {"field": "specific_complaints", "index": 0, "text": "a phrase",
         "subject": "subject_vendor", "polarity": "negative",
         "role": "primary_driver", "verbatim": False},
    ]
    source_row = {"summary": None, "review_text": "a phrase appears here"}
    canonical, counters = _normalize_phrase_metadata(r, source_row)
    assert canonical[0]["verbatim"] is False
    assert counters["verbatim_grounding_wins"] == 0
    assert counters["verbatim_grounding_failures"] == 0


def test_normalize_metadata_strict_verbatim_rejects_string_false():
    """Defensive: bool('false') is True, which would silently flip a denial
    into an affirmative. Only literal True counts as the LLM saying yes."""
    r = _empty_arrays()
    r["specific_complaints"] = ["a phrase"]
    r["phrase_metadata"] = [
        {"field": "specific_complaints", "index": 0, "text": "a phrase",
         "subject": "subject_vendor", "polarity": "negative",
         "role": "primary_driver", "verbatim": "false"},  # string, not bool
    ]
    canonical, _ = _normalize_phrase_metadata(r)
    assert canonical[0]["verbatim"] is False


def test_normalize_metadata_strict_verbatim_rejects_zero_int():
    r = _empty_arrays()
    r["specific_complaints"] = ["a phrase"]
    r["phrase_metadata"] = [
        {"field": "specific_complaints", "index": 0, "text": "a phrase",
         "subject": "subject_vendor", "polarity": "negative",
         "role": "primary_driver", "verbatim": 1},  # int, not True
    ]
    canonical, _ = _normalize_phrase_metadata(r)
    assert canonical[0]["verbatim"] is False  # `1 is True` is False


def test_normalize_metadata_strict_verbatim_accepts_only_true():
    r = _empty_arrays()
    r["specific_complaints"] = ["a phrase"]
    r["phrase_metadata"] = [
        {"field": "specific_complaints", "index": 0, "text": "a phrase",
         "subject": "subject_vendor", "polarity": "negative",
         "role": "primary_driver", "verbatim": True},
    ]
    # No source_row -> no grounding gate, so True passes through
    canonical, _ = _normalize_phrase_metadata(r)
    assert canonical[0]["verbatim"] is True


def test_normalize_metadata_grounding_after_text_mismatch_does_not_double_count():
    """Text mismatch already forces verbatim=False before grounding runs.
    Don't double-count by invoking the grounding gate too."""
    r = _empty_arrays()
    r["specific_complaints"] = ["real text"]
    r["phrase_metadata"] = [
        {"field": "specific_complaints", "index": 0, "text": "WRONG TEXT",
         "subject": "subject_vendor", "polarity": "negative",
         "role": "primary_driver", "verbatim": True},
    ]
    source_row = {"summary": None, "review_text": "real text appears here"}
    canonical, counters = _normalize_phrase_metadata(r, source_row)
    assert canonical[0]["verbatim"] is False
    assert counters["text_mismatch_rows"] == 1
    # Grounding never ran because verbatim was already False at that point
    assert counters["verbatim_grounding_failures"] == 0
    assert counters["verbatim_grounding_wins"] == 0


def test_compute_derived_fields_passes_source_row_to_grounding():
    """End-to-end: _compute_derived_fields uses source_row's review_text to
    validate verbatim claims. Phrase that the LLM marked verbatim=True but
    isn't in review_text should come out verbatim=False on the persisted
    metadata row."""
    r = _baseline_v1_result()
    r["specific_complaints"] = ["phrase that does not appear"]
    r["phrase_metadata"] = [
        {"field": "specific_complaints", "index": 0,
         "text": "phrase that does not appear",
         "subject": "subject_vendor", "polarity": "negative",
         "role": "primary_driver", "verbatim": True},
    ]
    source = _baseline_source_row()
    source["review_text"] = "Something completely unrelated."
    out = _compute_derived_fields(copy.deepcopy(r), source)
    mp = phrase_metadata_map(out)
    assert mp[("specific_complaints", 0)]["verbatim"] is False, \
        "grounding gate should have flipped verbatim=True to False"
    # Still v4 -- the LLM tried, just got the verbatim claim wrong
    assert enrichment_schema_version(out) == 4
