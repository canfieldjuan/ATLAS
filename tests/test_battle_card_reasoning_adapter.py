"""Tests for battle card reasoning adapter and deterministic field population."""

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

# Pre-mock heavy deps
_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)

for _mod in (
    "torch", "torchaudio", "transformers", "accelerate", "bitsandbytes",
    "PIL", "PIL.Image", "numpy", "cv2", "sounddevice", "soundfile",
    "nemo.collections", "nemo.collections.asr",
    "nemo.collections.asr.models",
    "playwright", "playwright.async_api",
    "playwright_stealth",
    "curl_cffi", "curl_cffi.requests",
    "pytrends", "pytrends.request",
):
    sys.modules.setdefault(_mod, MagicMock())

from atlas_brain.autonomous.tasks._b2b_shared import (
    _get_battle_card_reasoning_state,
    _get_vendor_reasoning,
)
from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
    SynthesisView,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synthesis_view(
    vendor: str = "TestVendor",
    wedge: str = "price_squeeze",
    confidence: str = "high",
) -> SynthesisView:
    raw = {
        "reasoning_contracts": {
            "vendor_core_reasoning": {
                "causal_narrative": {
                    "primary_wedge": wedge,
                    "confidence": confidence,
                    "trigger": "Q1 price hike",
                    "why_now": "Budget cycle alignment",
                    "data_gaps": [],
                    "what_would_weaken_thesis": [
                        {"condition": "Price rollback", "signal_source": "temporal", "monitorable": True},
                    ],
                },
            },
        },
    }
    from datetime import date
    return SynthesisView(vendor, raw, "v2", date(2026, 3, 29))


def _make_legacy_lookup(
    vendor: str = "TestVendor",
    archetype: str = "pricing_shock",
    confidence: float = 0.8,
) -> dict[str, dict]:
    return {
        vendor: {
            "archetype": archetype,
            "confidence": confidence,
            "mode": "",
            "risk_level": "high",
            "executive_summary": "Legacy summary",
            "key_signals": ["Legacy signal A"],
            "falsification_conditions": ["Legacy condition"],
            "uncertainty_sources": ["Legacy uncertainty"],
        },
    }


def _make_raw_synthesis(wedge: str = "price_squeeze", confidence: str = "high") -> dict:
    return {
        "reasoning_contracts": {
            "vendor_core_reasoning": {
                "causal_narrative": {
                    "primary_wedge": wedge,
                    "confidence": confidence,
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Tests: _get_battle_card_reasoning_state
# ---------------------------------------------------------------------------

class TestBattleCardReasoningState:
    def test_prefers_synthesis_view(self):
        views = {"TestVendor": _make_synthesis_view()}
        state = _get_battle_card_reasoning_state(
            "TestVendor", synthesis_views=views,
        )
        assert state["archetype"] == "price_squeeze"
        assert state["reasoning_source"] == "synthesis"
        assert state["has_confident_reasoning"] is True

    def test_falls_back_to_reasoning_lookup(self):
        lookup = _make_legacy_lookup()
        state = _get_battle_card_reasoning_state(
            "TestVendor", reasoning_lookup=lookup,
        )
        assert state["archetype"] == "pricing_shock"
        assert state["confidence"] == 0.8
        assert state["reasoning_source"] == "legacy"
        assert state["has_confident_reasoning"] is True

    def test_low_confidence_not_confident(self):
        lookup = _make_legacy_lookup(confidence=0.3)
        state = _get_battle_card_reasoning_state(
            "TestVendor", reasoning_lookup=lookup,
        )
        assert state["has_confident_reasoning"] is False

    def test_synthesis_fallback_for_confident(self):
        """When _rc has low confidence but raw synthesis has confident wedge,
        has_confident_reasoning should still be True."""
        lookup = _make_legacy_lookup(confidence=0.3)
        raw_synth = {"TestVendor": _make_raw_synthesis()}
        state = _get_battle_card_reasoning_state(
            "TestVendor",
            reasoning_lookup=lookup,
            reasoning_synthesis_lookup=raw_synth,
        )
        assert state["has_confident_reasoning"] is True

    def test_empty_vendor_returns_defaults(self):
        state = _get_battle_card_reasoning_state("Unknown")
        assert state["archetype"] == ""
        assert state["confidence"] == 0
        assert state["has_confident_reasoning"] is False
        assert state["reasoning_source"] == "legacy"

    def test_includes_all_required_fields(self):
        views = {"V": _make_synthesis_view(vendor="V")}
        state = _get_battle_card_reasoning_state("V", synthesis_views=views)
        required = {
            "archetype", "confidence", "risk_level", "key_signals",
            "falsification_conditions", "uncertainty_sources",
            "reasoning_source", "has_confident_reasoning",
            "executive_summary", "reasoning_mode",
        }
        assert required.issubset(state.keys())

    def test_canonicalized_key_lookup(self):
        """Views keyed by different casing should still resolve."""
        views = {"TestVendor": _make_synthesis_view()}
        state = _get_battle_card_reasoning_state(
            "testvendor", synthesis_views=views,
        )
        assert state["archetype"] == "price_squeeze"


# ---------------------------------------------------------------------------
# Tests: deterministic battle card fields populated from state
# ---------------------------------------------------------------------------

class TestBattleCardFieldPopulation:
    def test_archetype_fields_set_from_synthesis(self):
        """When synthesis provides reasoning, the card entry should have
        archetype, confidence, key_signals, and reasoning_source."""
        views = {"V": _make_synthesis_view(vendor="V")}
        state = _get_battle_card_reasoning_state("V", synthesis_views=views)
        # Simulate what the builder does
        card: dict[str, Any] = {}
        if state.get("archetype"):
            card["archetype"] = state["archetype"]
            card["archetype_confidence"] = state["confidence"]
            card["archetype_risk_level"] = state["risk_level"]
            card["archetype_key_signals"] = state["key_signals"]
            card["falsification_conditions"] = state["falsification_conditions"]
            card["uncertainty_sources"] = state["uncertainty_sources"]
            card["reasoning_source"] = state["reasoning_source"]

        assert card["archetype"] == "price_squeeze"
        assert card["reasoning_source"] == "synthesis"

    def test_no_archetype_fields_when_empty(self):
        state = _get_battle_card_reasoning_state("Unknown")
        card: dict[str, Any] = {}
        if state.get("archetype"):
            card["archetype"] = state["archetype"]
        assert "archetype" not in card

    def test_qualification_gate_uses_has_confident(self):
        """has_confident_reasoning should lower qualification thresholds."""
        views = {"V": _make_synthesis_view(vendor="V")}
        state = _get_battle_card_reasoning_state("V", synthesis_views=views)
        density_gate = 10 if state["has_confident_reasoning"] else 15
        assert density_gate == 10

        empty_state = _get_battle_card_reasoning_state("Unknown")
        density_gate_empty = 10 if empty_state["has_confident_reasoning"] else 15
        assert density_gate_empty == 15


# ---------------------------------------------------------------------------
# Tests: coexistence with synthesis injection
# ---------------------------------------------------------------------------

class TestCoexistenceWithSynthesisInjection:
    def test_state_does_not_mutate_views(self):
        """The adapter should not modify the SynthesisView."""
        view = _make_synthesis_view()
        views = {"TestVendor": view}
        _get_battle_card_reasoning_state("TestVendor", synthesis_views=views)
        # View should be unchanged
        assert view.primary_wedge.value == "price_squeeze"
        assert view.confidence("causal_narrative") == "high"

    def test_reasoning_synthesis_lookup_not_required(self):
        """Adapter should work without reasoning_synthesis_lookup."""
        views = {"V": _make_synthesis_view(vendor="V")}
        state = _get_battle_card_reasoning_state(
            "V", synthesis_views=views,
        )
        assert state["has_confident_reasoning"] is True


# ---------------------------------------------------------------------------
# Tests: synthesis injection prefers typed views over raw dicts
# ---------------------------------------------------------------------------

from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
    inject_synthesis_into_card,
)


class TestInjectionPrefersTypedViews:
    def test_inject_from_typed_view(self):
        """inject_synthesis_into_card should work with a SynthesisView directly."""
        view = _make_synthesis_view()
        card: dict[str, Any] = {"vendor": "TestVendor"}
        inject_synthesis_into_card(card, view, requested_as_of=None)

        assert card.get("reasoning_source") == "b2b_reasoning_synthesis"
        assert card.get("synthesis_wedge") == "price_squeeze"
        assert "reasoning_contracts" in card
        contracts = card["reasoning_contracts"]
        assert "vendor_core_reasoning" in contracts

    def test_inject_sets_freshness(self):
        """Injected card should have data_as_of_date from the view."""
        view = _make_synthesis_view()
        card: dict[str, Any] = {"vendor": "TestVendor"}
        from datetime import date
        inject_synthesis_into_card(
            card, view, requested_as_of=date(2026, 3, 29),
        )
        assert "data_as_of_date" in card
        assert card["data_as_of_date"] == "2026-03-29"

    def test_inject_marks_stale(self):
        """When view as_of < requested_as_of, data_stale should be True."""
        from datetime import date
        view = _make_synthesis_view()  # as_of = 2026-03-29
        card: dict[str, Any] = {"vendor": "TestVendor"}
        inject_synthesis_into_card(
            card, view, requested_as_of=date(2026, 3, 30),
        )
        assert card.get("data_stale") is True

    def test_inject_not_stale_when_current(self):
        """When view as_of == requested_as_of, data_stale should be False."""
        from datetime import date
        view = _make_synthesis_view()  # as_of = 2026-03-29
        card: dict[str, Any] = {"vendor": "TestVendor"}
        inject_synthesis_into_card(
            card, view, requested_as_of=date(2026, 3, 29),
        )
        assert card.get("data_stale") is False


class TestConfidentReasoningFromViews:
    def test_confident_from_synthesis_view_medium(self):
        """Medium-confidence synthesis view should count as confident."""
        view = _make_synthesis_view(confidence="medium")
        views = {"V": view}
        state = _get_battle_card_reasoning_state("V", synthesis_views=views)
        assert state["has_confident_reasoning"] is True

    def test_not_confident_from_synthesis_view_low(self):
        """Low-confidence synthesis view should not count as confident."""
        view = _make_synthesis_view(confidence="low")
        views = {"V": view}
        state = _get_battle_card_reasoning_state("V", synthesis_views=views)
        # Low confidence in synthesis_view_to_reasoning_entry maps to 0.25
        # which is < 0.7, so _rc path says no. But view has wedge + low
        # confidence, so the synthesis_views path checks medium/high only.
        assert state["has_confident_reasoning"] is False

    def test_confident_prefers_view_over_raw_synthesis(self):
        """When both synthesis_views and raw synthesis exist, view wins."""
        view = _make_synthesis_view(confidence="high")
        views = {"V": view}
        raw_synth = {"V": _make_raw_synthesis(confidence="low")}
        state = _get_battle_card_reasoning_state(
            "V", synthesis_views=views, reasoning_synthesis_lookup=raw_synth,
        )
        assert state["has_confident_reasoning"] is True
