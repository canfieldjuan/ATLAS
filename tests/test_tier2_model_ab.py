"""Sanity tests for the Tier 2 model A/B harness pure helpers.

The harness drives a model-routing decision; its metric helpers must be
pinned before we trust the output. Tests target the pure-function surface:
JSON validity, enum violations, explicit-promotion counting, the
attribution proxy ("pain without Tier 1 complaints"), and Jaccard.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

# asyncpg is mocked so the harness's import chain
# (atlas_brain.storage.database -> asyncpg) succeeds without a real driver.
_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

import tier2_model_ab as harness  # noqa: E402


# -- is_valid_top_level -------------------------------------------------------


def test_is_valid_top_level_accepts_payload_with_all_required_keys():
    payload = {key: None for key in harness.REQUIRED_TOP_LEVEL_KEYS}
    valid, missing = harness.is_valid_top_level(payload)
    assert valid is True
    assert missing == []


def test_is_valid_top_level_flags_missing_required_keys():
    payload = {key: None for key in harness.REQUIRED_TOP_LEVEL_KEYS}
    payload.pop("pain_categories")
    payload.pop("competitors_mentioned")
    valid, missing = harness.is_valid_top_level(payload)
    assert valid is False
    assert "pain_categories" in missing
    assert "competitors_mentioned" in missing


def test_is_valid_top_level_rejects_non_dict():
    valid_none, missing_none = harness.is_valid_top_level(None)
    valid_list, missing_list = harness.is_valid_top_level([])
    assert valid_none is False
    assert valid_list is False
    assert set(missing_none) == set(harness.REQUIRED_TOP_LEVEL_KEYS)
    assert set(missing_list) == set(harness.REQUIRED_TOP_LEVEL_KEYS)


# -- enum_violations ----------------------------------------------------------


def test_enum_violations_passes_canonical_payload():
    payload = {
        "competitors_mentioned": [
            {
                "name": "Freshdesk",
                "evidence_type": "explicit_switch",
                "displacement_confidence": "high",
                "reason_category": "pricing",
            }
        ],
        "pain_categories": [
            {"category": "pricing", "severity": "primary"},
            {"category": "support", "severity": "secondary"},
        ],
        "buyer_authority": {
            "role_type": "champion",
            "executive_sponsor_mentioned": True,
            "buying_stage": "evaluation",
        },
        "timeline": {
            "contract_end": None,
            "evaluation_deadline": None,
            "decision_timeline": "within_quarter",
        },
        "contract_context": {
            "contract_value_signal": "mid_market",
            "usage_duration": None,
        },
    }
    assert harness.enum_violations(payload) == []


def test_enum_violations_catches_invalid_pain_category_and_severity():
    payload = {
        "pain_categories": [
            {"category": "totally_made_up", "severity": "primary"},
            {"category": "pricing", "severity": "weird"},
        ],
    }
    violations = harness.enum_violations(payload)
    assert any("pain_categories[0].category=" in v for v in violations)
    assert any("pain_categories[1].severity=" in v for v in violations)


def test_enum_violations_catches_invalid_evidence_type_and_displacement_confidence():
    payload = {
        "competitors_mentioned": [
            {
                "name": "Freshdesk",
                "evidence_type": "made_up_evidence",
                "displacement_confidence": "definitely",
                "reason_category": "pricing",
            },
        ],
    }
    violations = harness.enum_violations(payload)
    assert any("competitors_mentioned[0].evidence_type=" in v for v in violations)
    assert any("competitors_mentioned[0].displacement_confidence=" in v for v in violations)


def test_enum_violations_catches_invalid_buyer_role_and_buying_stage():
    payload = {
        "buyer_authority": {
            "role_type": "ceo_overlord",
            "buying_stage": "negotiating_hard",
        },
    }
    violations = harness.enum_violations(payload)
    assert any("buyer_authority.role_type=" in v for v in violations)
    assert any("buyer_authority.buying_stage=" in v for v in violations)


def test_enum_violations_catches_invalid_timeline_and_contract_signal():
    payload = {
        "timeline": {"decision_timeline": "real_soon_now"},
        "contract_context": {"contract_value_signal": "huge"},
    }
    violations = harness.enum_violations(payload)
    assert any("timeline.decision_timeline=" in v for v in violations)
    assert any("contract_context.contract_value_signal=" in v for v in violations)


def test_enum_violations_returns_marker_for_non_dict_output():
    assert harness.enum_violations(None) == ["non-dict output"]
    assert harness.enum_violations([]) == ["non-dict output"]


# -- count_explicit_promotions ------------------------------------------------


def test_count_explicit_promotions_counts_only_explicit_switch_and_active_evaluation():
    payload = {
        "competitors_mentioned": [
            {"name": "A", "evidence_type": "explicit_switch"},
            {"name": "B", "evidence_type": "active_evaluation"},
            {"name": "C", "evidence_type": "implied_preference"},
            {"name": "D", "evidence_type": "neutral_mention"},
            {"name": "E", "evidence_type": "reverse_flow"},
        ],
    }
    assert harness.count_explicit_promotions(payload) == 2


def test_count_explicit_promotions_returns_zero_for_empty_or_invalid_input():
    assert harness.count_explicit_promotions({"competitors_mentioned": []}) == 0
    assert harness.count_explicit_promotions({"competitors_mentioned": None}) == 0
    assert harness.count_explicit_promotions(None) == 0
    assert harness.count_explicit_promotions({}) == 0


# -- has_pain_without_tier1_complaints ---------------------------------------


def test_has_pain_without_tier1_flags_pain_when_tier1_complaints_empty():
    parsed = {"pain_categories": [{"category": "pricing", "severity": "primary"}]}
    assert harness.has_pain_without_tier1_complaints(parsed, []) is True


def test_has_pain_without_tier1_clears_when_tier1_has_complaints():
    parsed = {"pain_categories": [{"category": "pricing", "severity": "primary"}]}
    assert harness.has_pain_without_tier1_complaints(
        parsed, ["the price is too damn high"]
    ) is False


def test_has_pain_without_tier1_clears_when_pain_categories_empty():
    parsed = {"pain_categories": []}
    assert harness.has_pain_without_tier1_complaints(parsed, []) is False


def test_has_pain_without_tier1_handles_none_inputs():
    assert harness.has_pain_without_tier1_complaints(None, []) is False
    assert harness.has_pain_without_tier1_complaints({}, []) is False


# -- jaccard ------------------------------------------------------------------


def test_jaccard_empty_sets_are_treated_as_match():
    # Two empty pain-category sets agree perfectly: both are saying "no pain".
    assert harness.jaccard(set(), set()) == 1.0


def test_jaccard_full_overlap_is_one():
    a = {"pricing", "support"}
    b = {"pricing", "support"}
    assert harness.jaccard(a, b) == 1.0


def test_jaccard_partial_overlap_is_proportional():
    a = {"pricing", "support", "ux"}
    b = {"pricing", "support", "integration"}
    # |a & b| = 2, |a | b| = 4 -> 0.5
    assert harness.jaccard(a, b) == 0.5


def test_jaccard_disjoint_sets_are_zero():
    a = {"pricing"}
    b = {"support"}
    assert harness.jaccard(a, b) == 0.0


def test_jaccard_one_empty_one_not_is_zero():
    assert harness.jaccard({"pricing"}, set()) == 0.0
    assert harness.jaccard(set(), {"pricing"}) == 0.0


# -- pain_category_set --------------------------------------------------------


def test_pain_category_set_extracts_only_canonical_strings():
    parsed = {
        "pain_categories": [
            {"category": "pricing", "severity": "primary"},
            {"category": "support", "severity": "secondary"},
            {"category": None, "severity": "minor"},  # missing category dropped
            {"severity": "minor"},  # no category dropped
            {"category": 42, "severity": "minor"},  # non-string dropped
        ],
    }
    assert harness.pain_category_set(parsed) == {"pricing", "support"}


def test_pain_category_set_returns_empty_for_invalid_input():
    assert harness.pain_category_set(None) == set()
    assert harness.pain_category_set({}) == set()
    assert harness.pain_category_set({"pain_categories": "not a list"}) == set()
