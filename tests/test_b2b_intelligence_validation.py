"""Regression tests for B2B churn intelligence quality controls.

Tests canonicalization, self-flow filtering, re-aggregation, quote
verification, stale date detection, displacement pair validation,
and executive source list config parsing.
"""

import sys
from datetime import date
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Pre-mock heavy deps before importing the task module
# ---------------------------------------------------------------------------
for _mod in (
    "torch", "torchaudio", "transformers", "accelerate", "bitsandbytes",
    "PIL", "PIL.Image", "numpy", "cv2", "sounddevice", "soundfile",
    "nemo.collections", "nemo.collections.asr",
    "nemo.collections.asr.models",
    "starlette", "starlette.requests",
    "asyncpg",
    "playwright", "playwright.async_api",
    "playwright_stealth",
    "curl_cffi", "curl_cffi.requests",
    "pytrends", "pytrends.request",
):
    sys.modules.setdefault(_mod, MagicMock())

from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
    _canonicalize_competitor,
    _executive_source_list,
    _validate_report,
)


# ---------------------------------------------------------------------------
# Phase 2: Canonicalization
# ---------------------------------------------------------------------------


class TestCanonicalizeCompetitor:
    def test_alias_map_hit(self):
        assert _canonicalize_competitor("gcp") == "Google Cloud Platform"

    def test_alias_map_case_insensitive(self):
        assert _canonicalize_competitor("GCP") == "Google Cloud Platform"

    def test_passthrough_proper_case(self):
        """Already-cased names should pass through unchanged."""
        assert _canonicalize_competitor("Asana") == "Asana"

    def test_lowercase_gets_titlecased(self):
        assert _canonicalize_competitor("notion") == "Notion"

    def test_empty_string(self):
        assert _canonicalize_competitor("") == ""

    def test_whitespace_stripped(self):
        assert _canonicalize_competitor("  gcp  ") == "Google Cloud Platform"


# ---------------------------------------------------------------------------
# Phase 2: Self-flow filtering + re-aggregation
# ---------------------------------------------------------------------------


class TestDisplacementNormalization:
    def test_self_flow_filtered(self):
        """Self-flows are filtered in the fetcher's post-processing, not in _validate_report.

        After canonicalization in _fetch_competitive_displacement, vendor==competitor
        rows are dropped, so they never appear in source_displacement passed to the LLM.
        This test verifies the fetcher logic via _canonicalize_competitor.
        """
        # Simulate fetcher post-processing
        raw_rows = [
            {"vendor_name": "Jira", "competitor": "Jira", "direction": "switching_to", "mention_count": 5},
            {"vendor_name": "Jira", "competitor": "Asana", "direction": "switching_to", "mention_count": 3},
        ]
        merged: dict[tuple[str, str, str | None], int] = {}
        for r in raw_rows:
            canon = _canonicalize_competitor(r["competitor"])
            vendor = r["vendor_name"]
            if canon and vendor and canon.lower() == vendor.lower():
                continue  # self-flow filtered
            key = (vendor, canon, r["direction"])
            merged[key] = merged.get(key, 0) + r["mention_count"]

        results = [
            {"vendor": k[0], "competitor": k[1], "direction": k[2], "mention_count": cnt}
            for k, cnt in merged.items()
            if cnt >= 2
        ]
        vendors = [(e["vendor"], e["competitor"]) for e in results]
        assert ("Jira", "Jira") not in vendors
        assert ("Jira", "Asana") in vendors

    def test_reaggregation(self):
        """Two rows that canonicalize to the same competitor merge their counts.

        This tests the _fetch_competitive_displacement post-processing logic
        indirectly via _canonicalize_competitor.
        """
        # Simulate what the fetcher does:
        raw_rows = [
            {"vendor_name": "Jira", "competitor": "gcp", "direction": "switching_to", "mention_count": 3},
            {"vendor_name": "Jira", "competitor": "Google Cloud Platform", "direction": "switching_to", "mention_count": 2},
        ]
        merged: dict[tuple[str, str, str | None], int] = {}
        for r in raw_rows:
            canon = _canonicalize_competitor(r["competitor"])
            key = (r["vendor_name"], canon, r["direction"])
            merged[key] = merged.get(key, 0) + r["mention_count"]

        results = [
            {"vendor": k[0], "competitor": k[1], "direction": k[2], "mention_count": cnt}
            for k, cnt in merged.items()
            if cnt >= 2
        ]
        assert len(results) == 1
        assert results[0]["competitor"] == "Google Cloud Platform"
        assert results[0]["mention_count"] == 5


# ---------------------------------------------------------------------------
# Phase 3: _validate_report
# ---------------------------------------------------------------------------


class TestValidateReportQuotes:
    def test_fabricated_quote_nulled(self):
        """A key_quote not in source data gets set to null."""
        parsed = {
            "weekly_churn_feed": [
                {"company": "Acme Corp", "key_quote": "This is fabricated"},
            ],
            "displacement_map": [],
            "timeline_hot_list": [],
        }
        warnings = _validate_report(
            parsed,
            source_high_intent=[
                {"company": "Acme Corp", "quotes": ["Real quote from source"]},
            ],
            source_quotable=[],
            source_displacement=[],
            report_date=date(2026, 3, 7),
        )
        assert parsed["weekly_churn_feed"][0]["key_quote"] is None
        assert any("Fabricated" in w for w in warnings)

    def test_real_quote_preserved(self):
        """A key_quote matching source data is kept."""
        parsed = {
            "weekly_churn_feed": [
                {"company": "Acme Corp", "key_quote": "Real quote from source"},
            ],
            "displacement_map": [],
            "timeline_hot_list": [],
        }
        warnings = _validate_report(
            parsed,
            source_high_intent=[
                {"company": "Acme Corp", "quotes": ["Real quote from source"]},
            ],
            source_quotable=[],
            source_displacement=[],
            report_date=date(2026, 3, 7),
        )
        assert parsed["weekly_churn_feed"][0]["key_quote"] == "Real quote from source"
        assert not any("Fabricated" in w for w in warnings)


class TestValidateReportSummaryCheck:
    def test_summary_company_not_in_feed_warns(self):
        """Company in exec summary but missing from weekly_churn_feed => warning."""
        parsed = {
            "executive_summary": "Acme Corp shows high churn intent this period.",
            "weekly_churn_feed": [],  # Acme not in feed
            "displacement_map": [],
            "timeline_hot_list": [],
        }
        warnings = _validate_report(
            parsed,
            source_high_intent=[{"company": "Acme Corp", "quotes": []}],
            source_quotable=[],
            source_displacement=[],
            report_date=date(2026, 3, 7),
        )
        assert any("Acme Corp" in w and "not in weekly_churn_feed" in w for w in warnings)

    def test_summary_company_in_feed_no_warn(self):
        """Company in both exec summary and feed => no warning."""
        parsed = {
            "executive_summary": "Acme Corp shows high churn intent this period.",
            "weekly_churn_feed": [{"company": "Acme Corp"}],
            "displacement_map": [],
            "timeline_hot_list": [],
        }
        warnings = _validate_report(
            parsed,
            source_high_intent=[{"company": "Acme Corp", "quotes": []}],
            source_quotable=[],
            source_displacement=[],
            report_date=date(2026, 3, 7),
        )
        assert not any("not in weekly_churn_feed" in w for w in warnings)


class TestValidateReportStaleTimeline:
    def test_stale_contract_end_dropped(self):
        """contract_end before report date => entry dropped."""
        parsed = {
            "weekly_churn_feed": [],
            "displacement_map": [],
            "timeline_hot_list": [
                {"company": "OldCo", "contract_end": "2025-06-30", "urgency": 8},
                {"company": "FutureCo", "contract_end": "2026-06-30", "urgency": 9},
            ],
        }
        warnings = _validate_report(
            parsed,
            source_high_intent=[],
            source_quotable=[],
            source_displacement=[],
            report_date=date(2026, 3, 7),
        )
        assert len(parsed["timeline_hot_list"]) == 1
        assert parsed["timeline_hot_list"][0]["company"] == "FutureCo"
        assert any("Stale" in w for w in warnings)

    def test_unparseable_date_kept_with_warning(self):
        """Non-ISO contract_end kept in timeline but generates a warning."""
        parsed = {
            "weekly_churn_feed": [],
            "displacement_map": [],
            "timeline_hot_list": [
                {"company": "VagueCo", "contract_end": "Q3 2026", "urgency": 7},
            ],
        }
        warnings = _validate_report(
            parsed,
            source_high_intent=[],
            source_quotable=[],
            source_displacement=[],
            report_date=date(2026, 3, 7),
        )
        assert len(parsed["timeline_hot_list"]) == 1  # kept
        assert any("Unparseable" in w for w in warnings)


class TestValidateReportDisplacement:
    def test_unmatched_displacement_dropped(self):
        """Displacement pair not in source data => dropped."""
        parsed = {
            "weekly_churn_feed": [],
            "displacement_map": [
                {"from_vendor": "Slack", "to_vendor": "Teams", "mention_count": 4},
                {"from_vendor": "Notion", "to_vendor": "Confluence", "mention_count": 3},
            ],
            "timeline_hot_list": [],
        }
        source_displacement = [
            {"vendor": "Slack", "competitor": "Teams", "mention_count": 4},
        ]
        warnings = _validate_report(
            parsed,
            source_high_intent=[],
            source_quotable=[],
            source_displacement=source_displacement,
            report_date=date(2026, 3, 7),
        )
        assert len(parsed["displacement_map"]) == 1
        assert parsed["displacement_map"][0]["from_vendor"] == "Slack"
        assert any("Unmatched" in w for w in warnings)

    def test_reversed_direction_pair_accepted(self):
        """LLM may reverse vendor/competitor based on direction -- both orderings valid."""
        parsed = {
            "weekly_churn_feed": [],
            "displacement_map": [
                {"from_vendor": "Teams", "to_vendor": "Slack", "mention_count": 3},
            ],
            "timeline_hot_list": [],
        }
        source_displacement = [
            {"vendor": "Slack", "competitor": "Teams", "mention_count": 3},
        ]
        warnings = _validate_report(
            parsed,
            source_high_intent=[],
            source_quotable=[],
            source_displacement=source_displacement,
            report_date=date(2026, 3, 7),
        )
        assert len(parsed["displacement_map"]) == 1
        assert not any("Unmatched" in w for w in warnings)


# ---------------------------------------------------------------------------
# Phase 1: Executive source list
# ---------------------------------------------------------------------------


class TestExecutiveSourceList:
    @patch("atlas_brain.autonomous.tasks.b2b_churn_intelligence.settings")
    def test_parses_config(self, mock_settings):
        mock_settings.b2b_churn.intelligence_executive_sources = "g2,capterra,trustradius"
        result = _executive_source_list()
        assert result == ["g2", "capterra", "trustradius"]

    @patch("atlas_brain.autonomous.tasks.b2b_churn_intelligence.settings")
    def test_separate_from_broad_allowlist(self, mock_settings):
        """Executive sources should be a subset of the broad allowlist."""
        mock_settings.b2b_churn.intelligence_executive_sources = "g2,capterra"
        exec_sources = _executive_source_list()
        assert "reddit" not in exec_sources
        assert "trustpilot" not in exec_sources

    @patch("atlas_brain.autonomous.tasks.b2b_churn_intelligence.settings")
    def test_handles_whitespace(self, mock_settings):
        mock_settings.b2b_churn.intelligence_executive_sources = " g2 , capterra , trustradius "
        result = _executive_source_list()
        assert result == ["g2", "capterra", "trustradius"]
