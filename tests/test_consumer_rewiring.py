"""Tests for consumer rewiring: verify vendor briefings, challenger briefs,
and accounts in motion extract Phase 3 governance fields through the shared
synthesis loader."""

import sys
from datetime import date
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


# ---------------------------------------------------------------------------
# Shared fixture: feed entry with full Phase 3 contracts
# ---------------------------------------------------------------------------

def _feed_entry_with_governance() -> dict[str, Any]:
    return {
        "vendor": "TestVendor",
        "reasoning_contracts": {
            "schema_version": "v1",
            "vendor_core_reasoning": {
                "causal_narrative": {
                    "primary_wedge": "price_squeeze",
                    "confidence": "medium",
                    "summary": "Pricing pressure after Q1 hike",
                },
                "timing_intelligence": {
                    "best_timing_window": "Q2 renewal cycle",
                    "immediate_triggers": [
                        {"type": "deadline", "trigger": "Q2 renewal"},
                    ],
                    "confidence": "medium",
                },
                "why_they_stay": {
                    "summary": "Ecosystem lock-in",
                    "strengths": [
                        {"area": "integrations", "evidence": "Broad API"},
                    ],
                },
                "confidence_posture": {
                    "overall": "medium",
                    "limits": ["thin enterprise sample"],
                },
            },
            "displacement_reasoning": {
                "switch_triggers": [
                    {"type": "deadline", "description": "Q2 renewal"},
                ],
            },
            "account_reasoning": {
                "market_summary": "Active evaluation in mid-market ops",
                "total_accounts": {"value": 12, "source_id": "accounts:summary:total_accounts"},
            },
            "evidence_governance": {
                "coverage_gaps": [
                    {"type": "missing_pool", "area": "displacement", "_sid": "gap:x"},
                ],
                "metric_ledger": [
                    {"label": "total_reviews", "value": 150, "_sid": "vault:metric:total_reviews"},
                ],
            },
        },
        "synthesis_wedge": "price_squeeze",
        "synthesis_wedge_label": "Price Squeeze",
        "reasoning_source": "b2b_reasoning_synthesis",
    }


# ---------------------------------------------------------------------------
# Vendor Briefing: _apply_reasoning_synthesis_to_briefing
# ---------------------------------------------------------------------------

class TestVendorBriefingRewiring:
    def test_extracts_why_they_stay(self):
        from atlas_brain.autonomous.tasks.b2b_vendor_briefing import (
            _apply_reasoning_synthesis_to_briefing,
        )
        briefing: dict[str, Any] = {}
        feed = _feed_entry_with_governance()
        _apply_reasoning_synthesis_to_briefing(briefing, feed)

        assert "why_they_stay" in briefing
        assert briefing["why_they_stay"]["summary"] == "Ecosystem lock-in"

    def test_extracts_confidence_posture(self):
        from atlas_brain.autonomous.tasks.b2b_vendor_briefing import (
            _apply_reasoning_synthesis_to_briefing,
        )
        briefing: dict[str, Any] = {}
        feed = _feed_entry_with_governance()
        _apply_reasoning_synthesis_to_briefing(briefing, feed)

        assert "confidence_posture" in briefing
        assert briefing["confidence_limits"] == ["thin enterprise sample"]

    def test_extracts_switch_triggers(self):
        from atlas_brain.autonomous.tasks.b2b_vendor_briefing import (
            _apply_reasoning_synthesis_to_briefing,
        )
        briefing: dict[str, Any] = {}
        feed = _feed_entry_with_governance()
        _apply_reasoning_synthesis_to_briefing(briefing, feed)

        assert "switch_triggers" in briefing
        assert briefing["switch_triggers"][0]["type"] == "deadline"

    def test_extracts_coverage_gaps(self):
        from atlas_brain.autonomous.tasks.b2b_vendor_briefing import (
            _apply_reasoning_synthesis_to_briefing,
        )
        briefing: dict[str, Any] = {}
        feed = _feed_entry_with_governance()
        _apply_reasoning_synthesis_to_briefing(briefing, feed)

        assert "coverage_gaps" in briefing

    def test_no_governance_when_no_contracts(self):
        from atlas_brain.autonomous.tasks.b2b_vendor_briefing import (
            _apply_reasoning_synthesis_to_briefing,
        )
        briefing: dict[str, Any] = {}
        _apply_reasoning_synthesis_to_briefing(briefing, {"vendor": "X"})

        assert "why_they_stay" not in briefing
        assert "switch_triggers" not in briefing


# ---------------------------------------------------------------------------
# Challenger Brief: _fetch_synthesis_view uses shared loader
# ---------------------------------------------------------------------------

class TestChallengerBriefRewiring:
    def test_fetch_synthesis_view_uses_shared_loader(self):
        """Verify _fetch_synthesis_view delegates to load_best_reasoning_view."""
        import inspect
        from atlas_brain.autonomous.tasks.b2b_challenger_brief import (
            _fetch_synthesis_view,
        )
        source = inspect.getsource(_fetch_synthesis_view)
        assert "load_best_reasoning_view" in source
        assert "b2b_reasoning_synthesis" not in source  # no direct query


# ---------------------------------------------------------------------------
# Accounts in Motion: synthesis-first reasoning_lookup
# ---------------------------------------------------------------------------

class TestAccountsInMotionRewiring:
    def test_reasoning_lookup_uses_synthesis_first_pattern(self):
        """Verify accounts in motion builds reasoning_lookup from synthesis views."""
        import inspect
        from atlas_brain.autonomous.tasks import b2b_accounts_in_motion as mod
        source = inspect.getsource(mod)
        assert "build_reasoning_lookup_from_views" in source
        assert "{**legacy_lookup, **synth_lookup}" in source
        assert "load_best_cross_vendor_lookup" in source

    def test_fetch_latest_synthesis_views_accepts_vendor_names(self):
        """When vendor_names provided, should use load_best_reasoning_views."""
        import inspect
        from atlas_brain.autonomous.tasks.b2b_accounts_in_motion import (
            _fetch_latest_synthesis_views,
        )
        source = inspect.getsource(_fetch_latest_synthesis_views)
        assert "load_best_reasoning_views" in source
        assert "vendor_names" in source
