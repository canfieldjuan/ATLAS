"""Tests for vendor_briefing pain-label sanitization (Phase 2.3 first commit).

The bad v3 pattern was: missing/empty pain_category synthesized to
'overall_dissatisfaction' and rendered as 'Overall Dissatisfaction' in
customer-facing output. Phase 2.3 kills the synthetic default, branches
the CTA on the formatted label, and sanitizes any blank-key entries that
upstream synthesis may have left in briefing['pain_labels'].

This commit does NOT add contract-helper imports. The narrow scope is:
remove synthetic generic defaults; leave rollup-fed reads alone (those
were gated upstream in Phase 2.2).
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)

from atlas_brain.autonomous.tasks.b2b_vendor_briefing import (  # noqa: E402
    _build_default_cta_hook,
    _default_pain_label,
    _finalize_briefing_presentation,
)


# ---------------------------------------------------------------------------
# _default_pain_label -- the synthetic default is gone
# ---------------------------------------------------------------------------


def test_default_pain_label_returns_none_for_none_input():
    assert _default_pain_label(None) is None


def test_default_pain_label_returns_none_for_empty_string():
    assert _default_pain_label("") is None


def test_default_pain_label_returns_none_for_whitespace():
    assert _default_pain_label("   ") is None
    assert _default_pain_label("\t\n") is None


def test_default_pain_label_renders_specific_category():
    # Real category survives -- the helper still formats it via the
    # _PAIN_LABEL_FALLBACKS map or title-cases it.
    assert _default_pain_label("pricing") == "Pricing and Contract Value Fatigue"


def test_default_pain_label_preserves_real_overall_dissatisfaction():
    # Critical: a row that ACTUALLY has pain_category='overall_dissatisfaction'
    # still maps to its display label. The migration only refuses to
    # SYNTHESIZE that label from a missing input.
    assert _default_pain_label("overall_dissatisfaction") == "Overall Dissatisfaction"
    assert _default_pain_label("general_dissatisfaction") == "Overall Dissatisfaction"


def test_default_pain_label_titlecases_unknown_category():
    # Unknown specific categories title-case via the fallback, e.g.
    # 'data_migration' -> 'Data Migration'.
    assert _default_pain_label("data_migration") == "Data Migration"


# ---------------------------------------------------------------------------
# _build_default_cta_hook -- branches on pain_label, never on raw whitespace
# ---------------------------------------------------------------------------


def _briefing(**overrides: Any) -> dict[str, Any]:
    base = {
        "vendor_name": "Acme",
        "challenger_mode": False,
        "pain_breakdown": [],
        "top_displacement_targets": [],
    }
    base.update(overrides)
    return base


def test_cta_hook_with_empty_pain_breakdown_is_generic():
    out = _build_default_cta_hook(_briefing(pain_breakdown=[]))
    assert "overall dissatisfaction" not in out.lower()
    assert "Acme" in out


def test_cta_hook_with_blank_category_in_pain_breakdown_does_not_render_pain():
    """A pain_breakdown row with category='' must NOT render as
    'Review the overall dissatisfaction signals...'."""
    out = _build_default_cta_hook(_briefing(pain_breakdown=[{"category": ""}]))
    assert "overall dissatisfaction" not in out.lower()


def test_cta_hook_with_whitespace_category_does_not_render_pain():
    """A pain_breakdown row with category='   ' was previously truthy
    enough to enter the pain branch and render a broken phrase. Branch on
    the formatted label so whitespace categories cannot leak."""
    out = _build_default_cta_hook(_briefing(pain_breakdown=[{"category": "   "}]))
    # The render must not contain any pain-phrase from a whitespace input
    assert "Review the   signals" not in out
    assert "  signals" not in out  # no double-space artifact
    assert "overall dissatisfaction" not in out.lower()


def test_cta_hook_with_real_category_does_render_pain():
    out = _build_default_cta_hook(_briefing(
        pain_breakdown=[{"category": "pricing"}],
    ))
    assert "pricing" in out.lower()


def test_cta_hook_challenger_with_real_category_renders_pain_label():
    out = _build_default_cta_hook(_briefing(
        challenger_mode=True,
        pain_breakdown=[{"category": "pricing"}],
        top_displacement_targets=[{"competitor": "Shopify"}],
    ))
    assert "pricing" in out.lower()
    assert "Shopify" in out


def test_cta_hook_with_only_top_target_falls_through_when_pain_label_none():
    """Empty pain_breakdown but real top_target should hit the
    target-only branch, not synthesize a pain phrase."""
    out = _build_default_cta_hook(_briefing(
        pain_breakdown=[{"category": ""}],
        top_displacement_targets=[{"competitor": "Shopify"}],
    ))
    assert "Shopify" in out
    assert "overall dissatisfaction" not in out.lower()


# ---------------------------------------------------------------------------
# _finalize_briefing_presentation -- sanitizes existing pain_labels
# ---------------------------------------------------------------------------


def test_finalize_drops_blank_key_from_existing_pain_labels():
    """Upstream synthesis may produce {'': 'Overall Dissatisfaction'} when
    the source row had no pain category. Sanitize before render."""
    briefing = {
        "pain_labels": {
            "": "Overall Dissatisfaction",
            "pricing": "Pricing and Contract Value Fatigue",
        },
        "pain_breakdown": [],
    }
    _finalize_briefing_presentation(briefing)
    assert "" not in briefing["pain_labels"]
    assert briefing["pain_labels"].get("pricing") == "Pricing and Contract Value Fatigue"


def test_finalize_drops_whitespace_key_from_existing_pain_labels():
    briefing = {
        "pain_labels": {
            "   ": "Overall Dissatisfaction",
            "ux": "User Experience Friction",
        },
        "pain_breakdown": [],
    }
    _finalize_briefing_presentation(briefing)
    assert "   " not in briefing["pain_labels"]
    assert briefing["pain_labels"].get("ux") == "User Experience Friction"


def test_finalize_ignores_malformed_existing_pain_labels():
    briefing = {
        "pain_labels": ["not", "a", "dict"],
        "pain_breakdown": [{"category": "pricing"}],
    }
    _finalize_briefing_presentation(briefing)
    assert briefing["pain_labels"] == {
        "pricing": "Pricing and Contract Value Fatigue",
    }


def test_finalize_preserves_real_overall_dissatisfaction_entry():
    """A non-empty 'overall_dissatisfaction' key is a real category and
    must NOT be dropped -- only blank/whitespace keys are sanitized."""
    briefing = {
        "pain_labels": {
            "overall_dissatisfaction": "Overall Dissatisfaction",
        },
        "pain_breakdown": [],
    }
    _finalize_briefing_presentation(briefing)
    assert briefing["pain_labels"].get("overall_dissatisfaction") == "Overall Dissatisfaction"


def test_finalize_adds_label_for_real_pain_breakdown_category():
    briefing = {
        "pain_labels": {},
        "pain_breakdown": [{"category": "pricing"}],
    }
    _finalize_briefing_presentation(briefing)
    assert briefing["pain_labels"].get("pricing") == "Pricing and Contract Value Fatigue"


def test_finalize_skips_blank_pain_breakdown_category_quietly():
    """Pain breakdown rows with empty category should NOT inject a blank
    key. The existing loop already gates on raw_category truthiness, but
    this test locks in the behavior post-migration."""
    briefing = {
        "pain_labels": {},
        "pain_breakdown": [{"category": ""}, {"category": "   "}],
    }
    _finalize_briefing_presentation(briefing)
    assert briefing["pain_labels"] == {}


def test_finalize_does_not_override_existing_label_for_same_category():
    """If pain_labels already has a label for a category, do not
    overwrite it from _default_pain_label."""
    briefing = {
        "pain_labels": {"pricing": "Custom Pricing Label"},
        "pain_breakdown": [{"category": "pricing"}],
    }
    _finalize_briefing_presentation(briefing)
    assert briefing["pain_labels"]["pricing"] == "Custom Pricing Label"
