"""Regression tests for B2B churn intelligence quality controls.

Tests canonicalization, self-flow filtering, re-aggregation, quote
verification, stale date detection, displacement pair validation,
and executive source list config parsing.
"""

import json
import sys
from datetime import date
from types import SimpleNamespace
from typing import Any
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Pre-mock heavy deps before importing the task module
# ---------------------------------------------------------------------------
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

import atlas_brain.autonomous.tasks.b2b_churn_intelligence as churn_intel_mod
from atlas_brain.autonomous.tasks._b2b_shared import (
    _battle_card_companies_from_evidence_vault,
    _build_battle_card_locked_facts,
    _build_company_signal_blocked_names_by_vendor,
    _build_deterministic_battle_cards,
    _build_deterministic_battle_card_competitive_landscape,
    _build_deterministic_battle_card_weakness_analysis,
    _battle_card_fallback_recommended_plays,
    _merge_canonical_company_signals,
    build_account_intelligence,
    _build_scorecard_locked_facts,
    _build_deterministic_vendor_feed,
    _build_validated_executive_summary,
    _canonicalize_competitor,
    _compute_churn_pressure_score,
    _compute_evidence_confidence,
    _executive_source_list,
    _fallback_scorecard_expert_take,
    _sanitize_battle_card_sales_copy,
    _battle_card_allowed_claims,
    _validate_battle_card_sales_copy,
    _validate_scorecard_expert_take,
    _validate_report,
)
from atlas_brain.autonomous.tasks._b2b_synthesis_reader import load_synthesis_view
from atlas_brain.config import settings


def _load_synth_view(vendor: str, raw: dict) -> Any:
    """Wrap a raw synthesis dict in a SynthesisView for test use."""
    return load_synthesis_view(raw, vendor, schema_version="v2")
from atlas_brain.autonomous.tasks.b2b_battle_cards import (
    _BATTLE_CARD_LLM_FIELDS,
    _apply_battle_card_quality,
    _battle_card_row_status,
    _build_battle_card_render_payload,
    _evaluate_battle_card_quality,
    _prioritize_seller_usable_primary_weakness,
    _promote_account_reasoning_to_battle_card,
    _battle_card_llm_options,
    _battle_card_prior_attempt,
    _battle_card_seller_usable_battles,
    _pair_opponent,
    _persist_battle_card,
    _parse_battle_card_sales_copy,
    _retire_gated_out_battle_cards,
    _update_execution_progress,
)
from atlas_brain.autonomous.tasks.b2b_churn_reports import (
    _pair_opponent as _report_pair_opponent,
    _attach_synthesis_contracts_to_report_entry,
)
from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
    contract_gaps_for_consumer,
    load_synthesis_view,
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


class TestChurnReportReasoningContracts:
    def test_attach_vendor_core_reasoning_to_feed_entry(self):
        entry = {"vendor": "Zendesk"}
        view = load_synthesis_view(
            {
                "schema_version": "2.1",
                "reasoning_contracts": {
                    "vendor_core_reasoning": {
                        "schema_version": "v1",
                        "causal_narrative": {
                            "primary_wedge": "price_squeeze",
                            "confidence": "high",
                            "trigger": "Price hike",
                            "why_now": "AI upsell packaging change",
                        },
                        "segment_playbook": {"confidence": "medium"},
                        "timing_intelligence": {
                            "confidence": "medium",
                            "best_timing_window": "Before renewal",
                            "active_eval_signals": {
                                "value": 2,
                                "source_id": "accounts:summary:active_eval_signal_count",
                            },
                            "sentiment_direction": "declining",
                            "immediate_triggers": [
                                {"trigger": "Q2 renewal", "type": "deadline"},
                            ],
                        },
                    },
                    "account_reasoning": {
                        "schema_version": "v1",
                        "market_summary": "Two accounts are actively evaluating alternatives.",
                        "total_accounts": {
                            "value": 6,
                            "source_id": "accounts:summary:total_accounts",
                        },
                        "high_intent_count": {
                            "value": 4,
                            "source_id": "accounts:summary:high_intent_count",
                        },
                        "active_eval_count": {
                            "value": 2,
                            "source_id": "accounts:summary:active_eval_signal_count",
                        },
                        "top_accounts": [],
                    },
                },
                "meta": {
                    "evidence_window_start": "2026-03-01",
                    "evidence_window_end": "2026-03-18",
                },
            },
            "Zendesk",
        )

        _attach_synthesis_contracts_to_report_entry(
            entry,
            view,
            consumer_name="weekly_churn_feed",
            requested_as_of=date(2026, 3, 20),
            include_displacement=False,
        )

        assert entry["reasoning_contracts"]["vendor_core_reasoning"]["causal_narrative"]["trigger"] == "Price hike"
        assert "causal_narrative" not in entry
        assert entry["synthesis_wedge"] == "price_squeeze"
        assert entry["reasoning_source"] == "b2b_reasoning_synthesis"
        assert entry["evidence_window_days"] == 17
        assert entry["timing_intelligence"]["best_timing_window"] == "Before renewal"
        assert entry["timing_summary"] == (
            "Before renewal. 2 active evaluation signals are visible right now. "
            "Review sentiment is skewing more negative."
        )
        assert entry["timing_metrics"]["active_eval_signals"] == 2
        assert entry["priority_timing_triggers"] == ["Q2 renewal"]
        assert "displacement_reasoning" not in entry

    def test_attach_displacement_reasoning_to_scorecard_entry(self):
        entry = {"vendor": "Zendesk"}
        view = load_synthesis_view(
            {
                "schema_version": "2.1",
                "reasoning_contracts": {
                    "vendor_core_reasoning": {
                        "schema_version": "v1",
                        "causal_narrative": {
                            "primary_wedge": "price_squeeze",
                            "confidence": "high",
                        },
                    },
                    "account_reasoning": {
                        "schema_version": "v1",
                        "market_summary": "Two accounts are actively evaluating alternatives.",
                        "total_accounts": {"value": 6, "source_id": "accounts:summary:total_accounts"},
                        "high_intent_count": {"value": 4, "source_id": "accounts:summary:high_intent_count"},
                        "active_eval_count": {
                            "value": 2,
                            "source_id": "accounts:summary:active_eval_signal_count",
                        },
                        "top_accounts": [],
                    },
                    "displacement_reasoning": {
                        "schema_version": "v1",
                        "competitive_reframes": {"confidence": "medium"},
                        "migration_proof": {
                            "confidence": "medium",
                            "switch_volume": {
                                "value": 0,
                                "source_id": "displacement:aggregate:total_explicit_switches",
                            },
                        },
                    },
                },
            },
            "Zendesk",
        )

        _attach_synthesis_contracts_to_report_entry(
            entry,
            view,
            consumer_name="vendor_scorecard",
            requested_as_of=date(2026, 3, 20),
            include_displacement=True,
        )

        assert "reasoning_contracts" in entry
        assert entry["reasoning_contracts"]["displacement_reasoning"]["migration_proof"]["switch_volume"]["value"] == 0
        assert "migration_proof" not in entry

    def test_consumers_require_account_reasoning_contract(self):
        view = load_synthesis_view(
            {
                "schema_version": "2.1",
                "reasoning_contracts": {
                    "vendor_core_reasoning": {
                        "schema_version": "v1",
                        "causal_narrative": {
                            "primary_wedge": "price_squeeze",
                            "confidence": "high",
                        },
                    },
                    "displacement_reasoning": {
                        "schema_version": "v1",
                        "competitive_reframes": {"confidence": "medium"},
                    },
                },
            },
            "Zendesk",
        )

        battle_gaps = contract_gaps_for_consumer(view, "battle_card")
        weekly_gaps = contract_gaps_for_consumer(view, "weekly_churn_feed")
        scorecard_gaps = contract_gaps_for_consumer(view, "vendor_scorecard")
        motion_gaps = contract_gaps_for_consumer(view, "accounts_in_motion")
        brief_gaps = contract_gaps_for_consumer(view, "challenger_brief")

        assert "account_reasoning" in battle_gaps
        assert "account_reasoning" in weekly_gaps
        assert "account_reasoning" in scorecard_gaps
        assert "account_reasoning" in motion_gaps
        assert "account_reasoning" in brief_gaps

    def test_attach_account_reasoning_promotes_summary_and_named_accounts(self):
        entry = {"vendor": "Zendesk", "named_accounts": []}
        view = load_synthesis_view(
            {
                "schema_version": "2.1",
                "reasoning_contracts": {
                    "vendor_core_reasoning": {
                        "schema_version": "v1",
                        "causal_narrative": {
                            "primary_wedge": "price_squeeze",
                            "confidence": "high",
                        },
                    },
                    "account_reasoning": {
                        "schema_version": "v1",
                        "market_summary": "Two accounts are actively evaluating alternatives.",
                        "total_accounts": {"value": 6, "source_id": "accounts:summary:total_accounts"},
                        "high_intent_count": {"value": 4, "source_id": "accounts:summary:high_intent_count"},
                        "active_eval_count": {
                            "value": 2,
                            "source_id": "accounts:summary:active_eval_signal_count",
                        },
                        "top_accounts": [
                            {"name": "Acme Corp", "intent_score": 0.9, "source_id": "accounts:item:acme"},
                            {"name": "Globex", "intent_score": 0.6, "source_id": "accounts:item:globex"},
                        ],
                    },
                },
            },
            "Zendesk",
        )

        _attach_synthesis_contracts_to_report_entry(
            entry,
            view,
            consumer_name="weekly_churn_feed",
            requested_as_of=date(2026, 3, 20),
            include_displacement=False,
        )

        assert entry["reasoning_contracts"]["account_reasoning"]["top_accounts"][0]["name"] == "Acme Corp"
        assert entry["account_pressure_summary"] == "Two accounts are actively evaluating alternatives."
        assert entry["account_pressure_metrics"]["high_intent_count"] == 4
        assert entry["priority_account_names"] == ["Acme Corp", "Globex"]
        assert entry["named_accounts"][0]["company"] == "Acme Corp"
        assert entry["named_accounts"][0]["reasoning_backed"] is True

    def test_promote_account_reasoning_to_battle_card(self):
        card = _sample_battle_card()
        card["reasoning_contracts"] = {
            "schema_version": "v1",
            "account_reasoning": {
                "schema_version": "v1",
                "market_summary": "Two accounts are actively evaluating alternatives.",
                "total_accounts": {"value": 6, "source_id": "accounts:summary:total_accounts"},
                "high_intent_count": {"value": 4, "source_id": "accounts:summary:high_intent_count"},
                "active_eval_count": {
                    "value": 2,
                    "source_id": "accounts:summary:active_eval_signal_count",
                },
                "top_accounts": [
                    {"name": "Acme Corp", "intent_score": 0.9, "source_id": "accounts:item:acme"},
                    {"name": "Globex", "intent_score": 0.6, "source_id": "accounts:item:globex"},
                ],
            },
        }

        _promote_account_reasoning_to_battle_card(card)

        assert card["account_reasoning"]["top_accounts"][0]["name"] == "Acme Corp"
        assert card["account_pressure_summary"] == "Two accounts are actively evaluating alternatives."
        assert card["account_pressure_metrics"]["high_intent_count"] == 4
        assert card["priority_account_names"] == ["Acme Corp", "Globex"]


class TestChurnReportCrossVendorHelpers:
    def test_pair_opponent_returns_empty_for_self_only_pair(self):
        assert _report_pair_opponent(("Zendesk", "Zendesk"), "Zendesk") == ""

    def test_pair_opponent_returns_other_vendor_for_normal_pair(self):
        assert _report_pair_opponent(("Zendesk", "Freshdesk"), "Zendesk") == "Freshdesk"


class TestChurnVendorScoping:
    def test_apply_vendor_scope_filters_rows_mappings_and_tuple_bundles(self):
        scoped, names = churn_intel_mod._apply_vendor_scope_to_churn_inputs(
            {
                "vendor_scores": [
                    {"vendor_name": "Zendesk", "score": 1},
                    {"vendor_name": "Freshdesk", "score": 2},
                ],
                "vendor_scores_from_signals": [
                    {"vendor_name": "Zendesk", "score": 10},
                    {"vendor_name": "Freshdesk", "score": 20},
                ],
                "high_intent": [
                    {"vendor": "Zendesk", "company": "Acme"},
                    {"vendor": "Freshdesk", "company": "Beta"},
                ],
                "existing_company_signals": {
                    "Zendesk": [{"company_name": "Acme"}],
                    "Freshdesk": [{"company_name": "Beta"}],
                },
                "displacement_provenance": {
                    ("Zendesk", "Freshdesk"): {"count": 3},
                    ("Intercom", "Help Scout"): {"count": 1},
                },
                "review_text_aggs": (
                    [
                        {"vendor_name": "Zendesk", "quote": "z"},
                        {"vendor_name": "Freshdesk", "quote": "f"},
                    ],
                    [
                        {"vendor_name": "Freshdesk", "quote": "p"},
                    ],
                ),
                "contract_ctx_aggs": (
                    [
                        {"vendor_name": "Zendesk", "value": "annual"},
                    ],
                    [
                        {"vendor_name": "Freshdesk", "value": "monthly"},
                    ],
                ),
            },
            ["Zendesk"],
        )

        assert names == ["Zendesk"]
        assert [row["vendor_name"] for row in scoped["vendor_scores"]] == ["Zendesk"]
        assert [row["vendor_name"] for row in scoped["vendor_scores_from_signals"]] == ["Zendesk"]
        assert [row["vendor"] for row in scoped["high_intent"]] == ["Zendesk"]
        assert list(scoped["existing_company_signals"].keys()) == ["Zendesk"]
        assert list(scoped["displacement_provenance"].keys()) == [("Zendesk", "Freshdesk")]
        assert [row["vendor_name"] for row in scoped["review_text_aggs"][0]] == ["Zendesk"]
        assert scoped["review_text_aggs"][1] == []
        assert [row["vendor_name"] for row in scoped["contract_ctx_aggs"][0]] == ["Zendesk"]
        assert scoped["contract_ctx_aggs"][1] == []

    def test_apply_vendor_scope_normalizes_string_input(self):
        scoped, names = churn_intel_mod._apply_vendor_scope_to_churn_inputs(
            {
                "vendor_scores": [
                    {"vendor_name": "Zendesk", "score": 1},
                    {"vendor_name": "Freshdesk", "score": 2},
                ],
            },
            " zendesk ",
        )

        assert names == ["Zendesk"]
        assert [row["vendor_name"] for row in scoped["vendor_scores"]] == ["Zendesk"]


class TestAccountIntelligenceHygiene:
    def test_build_company_signal_blocked_names_by_vendor_includes_vendors_integrations_and_alternatives(self):
        blocked = _build_company_signal_blocked_names_by_vendor(
            ["Zendesk", "Freshdesk", "Salesforce"],
            high_intent_entries=[
                {
                    "vendor": "Zendesk",
                    "alternatives": [{"name": "Halo"}, {"name": "BoldDesk"}],
                }
            ],
            integration_lookup={
                "Zendesk": [{"integration_name": "Salesforce"}],
            },
        )

        assert "freshdesk" in blocked["Zendesk"]
        assert "salesforce" in blocked["Zendesk"]
        assert "halo" in blocked["Zendesk"]
        assert "bolddesk" in blocked["Zendesk"]

    def test_build_account_intelligence_filters_polluted_company_names(self):
        acct = build_account_intelligence(
            "Zendesk",
            high_intent_entries=[
                {
                    "company": "ourdomain.com",
                    "urgency": 8.0,
                    "buying_stage": "evaluation",
                },
                {
                    "company": "Government agency",
                    "urgency": 9.0,
                    "buying_stage": "evaluation",
                },
                {
                    "company": "Costa Rica",
                    "urgency": 8.0,
                    "buying_stage": "evaluation",
                },
                {
                    "company": "ClonePartner",
                    "urgency": 8.0,
                    "buying_stage": "evaluation",
                },
                {
                    "company": "Freshdesk",
                    "urgency": 8.0,
                    "buying_stage": "evaluation",
                    "alternatives": [{"name": "Freshdesk"}],
                },
                {
                    "company": "B2B E-Commerce Startup",
                    "urgency": 8.0,
                    "buying_stage": "evaluation",
                },
                {
                    "company": "Video Studio",
                    "urgency": 7.0,
                    "buying_stage": "evaluation",
                },
                {
                    "company": "Acme Corp",
                    "urgency": 7.5,
                    "buying_stage": "evaluation",
                    "decision_maker": True,
                },
            ],
            persisted_signals=[
                {
                    "company_name": "Zendesk",
                    "urgency_score": 9.0,
                    "buying_stage": "evaluation",
                }
            ],
            blocked_names={"salesforce", "freshdesk", "halo", "bolddesk"},
        )

        assert [a["company_name"] for a in acct["accounts"]] == ["Acme Corp"]
        assert acct["summary"]["total_accounts"] == 1
        assert acct["summary"]["decision_maker_count"] == 1
        assert acct["summary"]["active_eval_signal_count"] == 1

    def test_build_account_intelligence_preserves_account_context_from_persisted_signals(self):
        acct = build_account_intelligence(
            "Zendesk",
            persisted_signals=[
                {
                    "company_name": "Acme Corp",
                    "urgency_score": 8.0,
                    "buying_stage": "evaluation",
                    "source": "g2",
                    "review_id": "r-1",
                    "title": "VP Support",
                    "company_size": "500",
                    "industry": "SaaS",
                    "alternatives": [{"name": "Freshdesk"}],
                    "quotes": [{"quote": "We need to switch ASAP"}],
                    "first_seen_at": "2026-03-01T00:00:00+00:00",
                    "last_seen_at": "2026-03-18T00:00:00+00:00",
                    "confidence_score": 0.9,
                },
            ],
        )

        account = acct["accounts"][0]
        assert account["title"] == "VP Support"
        assert account["company_size"] == "500"
        assert account["industry"] == "SaaS"
        assert account["alternatives"] == [{"name": "Freshdesk"}]
        assert account["quotes"] == [{"quote": "We need to switch ASAP"}]
        assert account["first_seen_at"] == "2026-03-01T00:00:00+00:00"
        assert account["last_seen_at"] == "2026-03-18T00:00:00+00:00"
        assert account["confidence_score"] == 0.9

    def test_merge_canonical_company_signals_filters_persisted_vendor_names_from_review_alternatives(self):
        merged = _merge_canonical_company_signals(
            [],
            {
                "Zendesk": [
                    {
                        "company_name": "Halo",
                        "urgency_score": 7.0,
                        "alternatives": [{"name": "Halo"}],
                    },
                    {
                        "company_name": "Acme Corp",
                        "urgency_score": 8.0,
                        "alternatives": [{"name": "Freshdesk"}],
                    },
                ]
            },
            blocked_names_by_vendor={"Zendesk": {"freshdesk", "salesforce"}},
        )

        assert [row["company_name"] for row in merged["Zendesk"]] == ["acme"]


class TestValidateReportSummaryCheck:
    def test_summary_vendor_in_feed_no_warn(self):
        """Vendor in both exec summary and feed => no warning."""
        parsed = {
            "executive_summary": "Zendesk shows elevated churn pressure this period.",
            "weekly_churn_feed": [{"vendor": "Zendesk"}],
            "displacement_map": [],
            "timeline_hot_list": [],
        }
        warnings = _validate_report(
            parsed,
            source_high_intent=[],
            source_quotable=[],
            source_displacement=[],
            report_date=date(2026, 3, 7),
        )
        assert not any("not in weekly_churn_feed" in w for w in warnings)

    def test_empty_feed_no_crash(self):
        """Empty feed with summary mentioning a vendor => no crash."""
        parsed = {
            "executive_summary": "Zendesk shows elevated churn pressure.",
            "weekly_churn_feed": [],
            "displacement_map": [],
            "timeline_hot_list": [],
        }
        warnings = _validate_report(
            parsed,
            source_high_intent=[],
            source_quotable=[],
            source_displacement=[],
            report_date=date(2026, 3, 7),
        )
        # Should not crash, no warnings about vendors since feed is empty
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
# Battle-card sales copy validation
# ---------------------------------------------------------------------------


def _sample_battle_card() -> dict[str, Any]:
    return {
        "vendor": "Shopify",
        "churn_pressure_score": 43.1,
        "total_reviews": 1156,
        "vendor_weaknesses": [{"area": "pricing", "count": 246}],
        "customer_pain_quotes": [
            {"quote": "We need better control over rising platform costs.", "urgency": 8},
            {"quote": "Support quality has become inconsistent during peak periods.", "urgency": 7},
        ],
        "competitor_differentiators": [
            {"competitor": "WooCommerce", "mentions": 53, "switch_count": 0},
            {"competitor": "BigCommerce", "mentions": 45, "switch_count": 0},
        ],
        "objection_data": {
            "avg_urgency": 4.3,
            "dm_churn_rate": 0.39,
            "total_reviews": 1156,
            "churn_signal_density": 24.7,
            "price_complaint_rate": 0.213,
            "budget_context": {
                "avg_seat_count": 52.6,
                "max_seat_count": 150,
                "median_seat_count": 12,
                "price_increase_rate": 0.020761245674740483,
                "price_increase_count": 24,
            },
            "top_feature_gaps": [{"feature": "Better SEO tools", "mentions": 3}],
        },
    }


class TestBattleCardSalesCopyValidation:
    def test_rejects_unsupported_numeric_claims(self):
        generated = {"executive_summary": "Costs drop by 30% and 55% of buyers are leaving."}
        warnings = _validate_battle_card_sales_copy(_sample_battle_card(), generated)
        assert any("unsupported numeric claims" in w for w in warnings)

    def test_allows_supported_rounded_percentage_variants(self):
        generated = {
            "executive_summary": "A recent 2% price increase wave is creating budget pressure."
        }
        warnings = _validate_battle_card_sales_copy(_sample_battle_card(), generated)
        assert not any("unsupported numeric claims" in w for w in warnings)

    def test_allows_supported_decimal_percentage_variant_equivalent(self):
        card = _sample_battle_card()
        card["objection_data"]["price_complaint_rate"] = 0.27
        generated = {
            "weakness_analysis": [
                {
                    "weakness": "Pricing pressure is rising.",
                    "evidence": "27.0% price complaint rate across 1156 reviews",
                    "customer_quote": "",
                    "winning_position": "Emphasize pricing clarity.",
                }
            ]
        }
        warnings = _validate_battle_card_sales_copy(card, generated)
        assert not any("unsupported numeric claims" in w for w in warnings)

    def test_rejects_numeric_claims_present_only_in_cross_vendor_prose(self):
        card = _sample_battle_card()
        card["cross_vendor_battles"] = [{
            "opponent": "WooCommerce",
            "conclusion": "Shopify is winning in the 52% SMB segment with 43 explicit switches.",
            "durability": "structural",
            "confidence": 0.85,
            "winner": "Shopify",
            "key_insights": ["52% SMB share", "43 explicit switches"],
        }]
        generated = {
            "competitive_landscape": {
                "top_alternatives": "WooCommerce remains relevant, but Shopify is winning in the 52% SMB segment."
            }
        }
        warnings = _validate_battle_card_sales_copy(card, generated)
        assert any("unsupported numeric claims" in w for w in warnings)

    def test_allows_numeric_claims_from_structured_reasoning_contracts(self):
        card = _sample_battle_card() | {
            "reasoning_contracts": {
                "schema_version": "v1",
                "vendor_core_reasoning": {
                    "schema_version": "v1",
                    "timing_intelligence": {
                        "active_eval_signals": {
                            "value": 9,
                            "source_id": "segment:aggregate:active_eval_signal_count",
                        },
                    },
                },
                "category_reasoning": {
                    "schema_version": "v1",
                    "vendor_count": 12,
                    "displacement_flow_count": 43,
                },
            },
        }
        generated = {
            "executive_summary": "There are 9 active evaluation signals across a 12-vendor category."
        }
        warnings = _validate_battle_card_sales_copy(card, generated)
        assert not any("unsupported numeric claims" in w for w in warnings)

    def test_rejects_stale_years(self):
        generated = {"landmine_questions": ["Should these app features still require plugins in 2024?"]}
        warnings = _validate_battle_card_sales_copy(_sample_battle_card(), generated)
        assert any("references a year" in w for w in warnings)

    def test_rejects_leaving_language_without_switches(self):
        generated = {"competitive_landscape": {"top_alternatives": ["Customers are leaving for WooCommerce."]}}
        warnings = _validate_battle_card_sales_copy(_sample_battle_card(), generated)
        assert any("implies switching" in w for w in warnings)

    def test_rejects_high_priority_when_score_is_moderate(self):
        generated = {"executive_summary": "HIGH PRIORITY TARGET: Shopify is under pressure."}
        warnings = _validate_battle_card_sales_copy(_sample_battle_card(), generated)
        assert any("overstates urgency" in w for w in warnings)

    def test_rejects_low_evidence_feature_gap_as_headline(self):
        generated = {"executive_summary": "Shopify is vulnerable because of Better SEO tools gaps."}
        warnings = _validate_battle_card_sales_copy(_sample_battle_card(), generated)
        assert any("low-evidence feature gap" in w for w in warnings)

    def test_rejects_non_exact_customer_quote(self):
        generated = {
            "weakness_analysis": [
                {
                    "weakness": "Pricing pressure is rising.",
                    "evidence": "246 pricing mentions",
                    "customer_quote": "We need better cost control right now.",
                    "winning_position": "Emphasize pricing clarity.",
                }
            ]
        }
        warnings = _validate_battle_card_sales_copy(_sample_battle_card(), generated)
        assert any("customer_quote is not an exact source quote" in w for w in warnings)

    def test_rejects_unsupported_numeric_claims_in_recommended_play(self):
        generated = {
            "recommended_plays": [
                {
                    "play": "Target support teams with a renewal benchmark.",
                    "target_segment": "Support department leaders showing 21% churn rates and 4.1 urgency.",
                    "key_message": "Lead with cost control.",
                    "timing": "During renewal planning.",
                }
            ]
        }
        warnings = _validate_battle_card_sales_copy(_sample_battle_card(), generated)
        assert any("unsupported numeric claims" in w for w in warnings)

    def test_rejects_overcertain_play_when_account_intelligence_is_missing(self):
        card = _sample_battle_card() | {
            "reasoning_contracts": {
                "schema_version": "v1",
                "vendor_core_reasoning": {
                    "schema_version": "v1",
                    "segment_playbook": {
                        "confidence": "medium",
                        "data_gaps": ["No account-level intelligence for targeted outreach"],
                    },
                },
            },
            "high_intent_companies": [],
        }
        generated = {
            "recommended_plays": [
                {
                    "play": "Target SMB support teams with cost audit messaging.",
                    "target_segment": "Support leaders already dealing with service friction and renewal pressure.",
                    "key_message": "Lead with cost control.",
                    "timing": "During renewal planning.",
                }
            ]
        }
        warnings = _validate_battle_card_sales_copy(card, generated)
        assert any("overstates segment certainty" in w for w in warnings)

    def test_rejects_duplicate_recommended_play_segments(self):
        generated = {
            "recommended_plays": [
                {
                    "play": "Best tested on SMB support teams with cost audit messaging.",
                    "target_segment": "SMB support teams",
                    "key_message": "Lead with cost control.",
                    "timing": "During renewal planning.",
                },
                {
                    "play": "Best tested on SMB support teams with migration benchmarking.",
                    "target_segment": "SMB support teams",
                    "key_message": "Lead with migration risk reduction.",
                    "timing": "During annual planning.",
                },
            ]
        }
        warnings = _validate_battle_card_sales_copy(_sample_battle_card(), generated)
        assert any("repeat the same target segment" in w for w in warnings)

    def test_rejects_duplicate_recommended_play_segments_even_with_sample_suffix(self):
        generated = {
            "recommended_plays": [
                {
                    "play": "Best tested on SMB support teams with cost audit messaging.",
                    "target_segment": "SMB support teams (sample n=12)",
                    "key_message": "Lead with cost control.",
                    "timing": "During renewal planning.",
                },
                {
                    "play": "Best tested on SMB support teams with migration benchmarking.",
                    "target_segment": "SMB support teams (sample n=18)",
                    "key_message": "Lead with migration risk reduction.",
                    "timing": "During annual planning.",
                },
            ]
        }
        warnings = _validate_battle_card_sales_copy(_sample_battle_card(), generated)
        assert any("repeat the same target segment" in w for w in warnings)


class TestBattleCardSalesCopySanitization:
    def test_sanitizes_unsupported_numeric_summary(self):
        card = _sample_battle_card()
        generated = {"executive_summary": "HIGH PRIORITY TARGET: Costs drop by 30% and 55% of buyers are leaving."}
        sanitized = _sanitize_battle_card_sales_copy(card, generated)
        warnings = _validate_battle_card_sales_copy(card, sanitized)
        assert not warnings
        assert "30%" not in sanitized["executive_summary"]
        assert "55%" not in sanitized["executive_summary"]
        assert "HIGH PRIORITY TARGET" not in sanitized["executive_summary"]

    def test_sanitizes_switching_language_without_switch_evidence(self):
        card = _sample_battle_card()
        generated = {
            "competitive_landscape": {
                "top_alternatives": ["Customers are leaving for WooCommerce in 2024."]
            }
        }
        sanitized = _sanitize_battle_card_sales_copy(card, generated)
        warnings = _validate_battle_card_sales_copy(card, sanitized)
        assert not warnings
        assert "leaving" not in sanitized["competitive_landscape"]["top_alternatives"][0].lower()
        assert "2024" not in sanitized["competitive_landscape"]["top_alternatives"][0]

    def test_sanitizes_low_evidence_headline(self):
        card = _sample_battle_card()
        generated = {"weakness_analysis": [{"headline": "Better SEO tools is the main reason Shopify is vulnerable."}]}
        sanitized = _sanitize_battle_card_sales_copy(card, generated)
        warnings = _validate_battle_card_sales_copy(card, sanitized)
        assert not warnings
        assert "better seo tools" not in sanitized["weakness_analysis"][0]["headline"].lower()

    def test_sanitizes_mid_sentence_high_priority_phrase_cleanly(self):
        card = _sample_battle_card()
        generated = {
            "executive_summary": (
                "Magento has a strong churn pattern, making it a high-priority target "
                "for mid-market teams."
            )
        }
        sanitized = _sanitize_battle_card_sales_copy(card, generated)
        warnings = _validate_battle_card_sales_copy(card, sanitized)
        assert not warnings
        assert "a Emerging vulnerability" not in sanitized["executive_summary"]

    def test_sanitizes_customer_quote_to_exact_source_text(self):
        card = _sample_battle_card()
        generated = {
            "weakness_analysis": [
                {
                    "weakness": "Pricing pressure is rising.",
                    "evidence": "246 pricing mentions",
                    "customer_quote": "We need better cost control right now.",
                    "winning_position": "Emphasize pricing clarity.",
                }
            ]
        }
        sanitized = _sanitize_battle_card_sales_copy(card, generated)
        warnings = _validate_battle_card_sales_copy(card, sanitized)
        assert not warnings
        assert sanitized["weakness_analysis"][0]["customer_quote"] in {
            "We need better control over rising platform costs.",
            "Support quality has become inconsistent during peak periods.",
        }

    def test_sanitizes_proof_point_to_supported_structured_count(self):
        card = _sample_battle_card() | {
            "reasoning_contracts": {
                "schema_version": "v1",
                "vendor_core_reasoning": {
                    "schema_version": "v1",
                    "timing_intelligence": {
                        "active_eval_signals": {
                            "value": 9,
                            "source_id": "segment:aggregate:active_eval_signal_count",
                        },
                    },
                },
            },
        }
        generated = {
            "objection_handlers": [
                {
                    "objection": "Our team is already trained on Shopify.",
                    "acknowledge": "Training investment is real.",
                    "pivot": "The day-to-day friction may already outweigh the retraining cost.",
                    "proof_point": "Support departments show a 21% churn rate with average urgency of 4.1.",
                }
            ]
        }
        sanitized = _sanitize_battle_card_sales_copy(card, generated)
        warnings = _validate_battle_card_sales_copy(card, sanitized)
        assert not warnings
        assert "21%" not in sanitized["objection_handlers"][0]["proof_point"]
        assert "4.1" not in sanitized["objection_handlers"][0]["proof_point"]
        assert "9 active evaluation signals" in sanitized["objection_handlers"][0]["proof_point"]

    def test_sanitizes_recommended_play_numeric_claims(self):
        card = _sample_battle_card()
        generated = {
            "recommended_plays": [
                {
                    "play": "Target support teams with a renewal benchmark.",
                    "target_segment": "Support department leaders showing 21% churn rates and 4.1 urgency.",
                    "key_message": "Reduce costs by 30% with a simpler support stack.",
                    "timing": "During the next 14 days.",
                }
            ]
        }
        sanitized = _sanitize_battle_card_sales_copy(card, generated)
        warnings = _validate_battle_card_sales_copy(card, sanitized)
        assert not warnings
        target_segment = sanitized["recommended_plays"][0]["target_segment"]
        key_message = sanitized["recommended_plays"][0]["key_message"]
        timing = sanitized["recommended_plays"][0]["timing"]
        assert "21%" not in target_segment
        assert "4.1" not in target_segment
        assert "30%" not in key_message
        assert "14" not in timing

    def test_sanitizes_recommended_play_numeric_timing_windows(self):
        card = _sample_battle_card()
        generated = {
            "recommended_plays": [
                {
                    "play": "Target support teams with a renewal benchmark.",
                    "target_segment": "Support leaders already dealing with service friction and renewal pressure.",
                    "key_message": "Lead with pricing clarity.",
                    "timing": "Over the next 90 days before renewal review.",
                }
            ]
        }
        sanitized = _sanitize_battle_card_sales_copy(card, generated)
        warnings = _validate_battle_card_sales_copy(card, sanitized)
        assert not warnings
        assert sanitized["recommended_plays"][0]["timing"] == (
            "Best tested during active evaluation windows, renewal review, or planning cycles."
        )

    def test_sanitizes_overcertain_play_when_account_intelligence_is_missing(self):
        card = _sample_battle_card() | {
            "reasoning_contracts": {
                "schema_version": "v1",
                "vendor_core_reasoning": {
                    "schema_version": "v1",
                    "segment_playbook": {
                        "confidence": "medium",
                        "data_gaps": ["No account-level intelligence for targeted outreach"],
                    },
                },
            },
            "high_intent_companies": [],
        }
        generated = {
            "recommended_plays": [
                {
                    "play": "Target SMB support teams with cost audit messaging.",
                    "target_segment": "Support leaders already dealing with service friction and renewal pressure.",
                    "key_message": "Lead with cost control.",
                    "timing": "During renewal planning.",
                }
            ]
        }
        sanitized = _sanitize_battle_card_sales_copy(card, generated)
        warnings = _validate_battle_card_sales_copy(card, sanitized)
        assert not warnings
        assert sanitized["recommended_plays"][0]["play"].startswith("Best tested on")

    def test_sanitizes_duplicate_recommended_play_segments_from_segment_playbook(self):
        card = _sample_battle_card() | {
            "reasoning_contracts": {
                "schema_version": "v1",
                "vendor_core_reasoning": {
                    "schema_version": "v1",
                    "segment_playbook": {
                        "confidence": "medium",
                        "priority_segments": [
                            {
                                "segment": "SMB support teams (50 or fewer employees)",
                                "best_opening_angle": "Cost-effective support without forced AI upgrades",
                            },
                            {
                                "segment": "Support departments with moderate urgency tolerance",
                                "best_opening_angle": "Reliable support tooling without complexity overhead",
                            },
                        ],
                    },
                    "timing_intelligence": {
                        "best_timing_window": "During renewal review cycles",
                    },
                },
            },
            "high_intent_companies": [],
        }
        generated = {
            "recommended_plays": [
                {
                    "play": "Best tested on SMB support teams with cost audit messaging.",
                    "target_segment": "SMB support teams",
                    "key_message": "Lead with cost control.",
                    "timing": "During renewal planning.",
                },
                {
                    "play": "Best tested on SMB support teams with migration benchmarking.",
                    "target_segment": "SMB support teams",
                    "key_message": "Lead with migration risk reduction.",
                    "timing": "During annual planning.",
                },
            ]
        }
        sanitized = _sanitize_battle_card_sales_copy(card, generated)
        warnings = _validate_battle_card_sales_copy(card, sanitized)
        assert not warnings
        assert sanitized["recommended_plays"][0]["target_segment"] == "SMB support teams"
        assert sanitized["recommended_plays"][1]["target_segment"] == "Support departments with moderate urgency tolerance"

    def test_fallback_recommended_plays_surface_segment_sample_size(self):
        card = _sample_battle_card() | {
            "reasoning_contracts": {
                "schema_version": "v1",
                "vendor_core_reasoning": {
                    "schema_version": "v1",
                    "segment_playbook": {
                        "confidence": "medium",
                        "priority_segments": [
                            {
                                "segment": "Mid-Market operations teams",
                                "best_opening_angle": "TCO comparison",
                                "estimated_reach": {
                                    "value": 22,
                                    "source_id": "segment:reach:size:mid_market",
                                },
                                "sample_size": 22,
                            },
                        ],
                    },
                    "timing_intelligence": {
                        "best_timing_window": "During renewal review cycles",
                    },
                },
            },
            "high_intent_companies": [],
        }

        plays = _battle_card_fallback_recommended_plays(card)

        assert plays[0]["target_segment"] == "Mid-Market operations teams (sample n=22)"

    def test_fallback_recommended_plays_use_strategic_roles_when_segments_missing(self):
        card = _sample_battle_card() | {
            "reasoning_contracts": {
                "schema_version": "v1",
                "vendor_core_reasoning": {
                    "schema_version": "v1",
                    "segment_playbook": {
                        "confidence": "medium",
                        "priority_segments": [],
                        "supporting_evidence": {
                            "top_strategic_roles": [
                                {
                                    "role_type": "economic_buyer",
                                    "priority_score": 45,
                                    "review_count": 15,
                                    "source_id": "segment:role:economic_buyer",
                                },
                            ],
                            "top_departments": [
                                {"department": "finance", "source_id": "segment:department:finance"},
                            ],
                            "top_contract_segments": [
                                {"segment": "enterprise", "source_id": "segment:contract:enterprise"},
                            ],
                        },
                    },
                    "timing_intelligence": {
                        "best_timing_window": "During renewal review cycles",
                    },
                },
            },
            "high_intent_companies": [],
        }

        plays = _battle_card_fallback_recommended_plays(card)

        assert plays[0]["target_segment"] == "economic buyers in Finance teams"
        assert "pricing predictability" in plays[0]["key_message"].lower()
        assert plays[0]["timing"] == "During renewal review cycles."


class TestBattleCardQualityGate:
    def test_quality_gate_rejects_conflicting_active_eval_signals(self):
        card = _sample_battle_card() | {
            "evidence_window_is_thin": False,
            "data_stale": False,
            "account_pressure_metrics": {"active_eval_count": 0},
            "active_evaluation_deadlines": [{"company": "Acme Transit", "decision_timeline": "within_quarter"}],
        }
        quality = _evaluate_battle_card_quality(card, phase="deterministic")
        assert quality["status"] == "deterministic_fallback"
        assert any("active-evaluation signal conflict" in item for item in quality["failed_checks"])

    def test_quality_gate_does_not_conflict_across_eval_signal_families(self):
        card = _sample_battle_card() | {
            "evidence_window_is_thin": False,
            "data_stale": False,
            "data_as_of_date": date.today().isoformat(),
            "account_pressure_metrics": {"active_eval_count": 2},
            "timing_metrics": {"active_eval_signals": 228},
            "high_intent_companies": [
                {"company": "Acme Transit", "urgency": 9, "buying_stage": "evaluation"},
            ],
        }
        quality = _evaluate_battle_card_quality(card, phase="deterministic")
        assert quality["status"] != "deterministic_fallback"
        assert not any("active-evaluation signal conflict" in item for item in quality["failed_checks"])

    def test_quality_gate_requires_high_intent_accounts_with_stage_and_urgency(self):
        card = _sample_battle_card() | {
            "evidence_window_is_thin": False,
            "data_stale": False,
            "high_intent_companies": [
                {"company": "Acme Transit", "urgency": 9, "buying_stage": "post_purchase"},
            ],
            "recommended_plays": [
                {
                    "play": "Run a support health-check audit before renewal",
                    "target_segment": "IT directors",
                    "key_message": "Expose hidden support risk before commitment.",
                    "timing": "Renewal planning workshop this quarter",
                },
                {
                    "play": "Schedule a reliability benchmark workshop with security leadership",
                    "target_segment": "Security managers",
                    "key_message": "Benchmark incident response readiness.",
                    "timing": "After support escalation",
                },
            ],
            "executive_summary": "Support erosion is driving evaluation risk right now.",
        }
        quality = _evaluate_battle_card_quality(card, phase="final")
        assert quality["status"] == "deterministic_fallback"
        assert any("high-intent account" in item for item in quality["failed_checks"])

    def test_quality_gate_allows_global_eval_fallback_for_stage_gap(self):
        card = _sample_battle_card() | {
            "evidence_window_is_thin": False,
            "data_stale": False,
            "data_as_of_date": date.today().isoformat(),
            "timing_metrics": {"active_eval_signals": 8},
            "high_intent_companies": [
                {"company": "Acme Transit", "urgency": 9, "buying_stage": "post_purchase"},
            ],
        }
        quality = _evaluate_battle_card_quality(card, phase="deterministic")
        assert quality["status"] != "deterministic_fallback"
        assert any("global active-evaluation evidence fallback" in item for item in quality["warnings"])

    def test_quality_gate_uses_data_as_of_stale_days_threshold(self):
        stale_date = date.fromordinal(date.today().toordinal() - 4).isoformat()
        card = _sample_battle_card() | {
            "evidence_window_is_thin": False,
            "data_stale": False,
            "data_as_of_date": stale_date,
            "high_intent_companies": [
                {"company": "Acme Transit", "urgency": 9, "buying_stage": "evaluation"},
            ],
        }
        quality = _evaluate_battle_card_quality(card, phase="deterministic")
        assert quality["status"] == "deterministic_fallback"
        assert any("source data is stale for requested report date" in item for item in quality["failed_checks"])

    def test_quality_gate_clears_data_stale_within_allowed_lag(self):
        lag_date = date.fromordinal(date.today().toordinal() - 1).isoformat()
        card = _sample_battle_card() | {
            "evidence_window_is_thin": False,
            "data_stale": True,
            "data_as_of_date": lag_date,
            "high_intent_companies": [
                {"company": "Acme Transit", "urgency": 9, "buying_stage": "evaluation"},
            ],
        }
        quality = _evaluate_battle_card_quality(card, phase="deterministic")
        assert quality["status"] != "deterministic_fallback"
        assert card["data_stale"] is False
        assert any("source data lag detected" in item for item in quality["warnings"])

    def test_quality_gate_sales_ready_when_strict_requirements_pass(self):
        card = _sample_battle_card() | {
            "evidence_window_is_thin": False,
            "data_stale": False,
            "evidence_window_days": 21,
            "high_intent_companies": [
                {"company": "Acme Transit", "urgency": 9, "buying_stage": "evaluation"},
            ],
            "recommended_plays": [
                {
                    "play": "Run a support health-check audit for Acme Transit",
                    "target_segment": "IT directors at Acme Transit",
                    "key_message": "Use an audit to expose incident-response gaps before renewal.",
                    "timing": "Renewal review session this quarter",
                },
                {
                    "play": "Host a reliability benchmark workshop with security evaluators",
                    "target_segment": "Security evaluators in enterprise accounts",
                    "key_message": "Benchmark uptime and escalation readiness with a structured assessment.",
                    "timing": "Within two weeks of support incidents",
                },
            ],
            "executive_summary": "Support and reliability pressure is creating an immediate opening.",
            "discovery_questions": [
                "How are support escalations affecting security operations this quarter?",
                "What is your threshold for switching if reliability misses continue?",
            ],
            "landmine_questions": [
                "When did you last benchmark support SLAs before renewal?",
                "What happens if alert coverage drops during peak periods?",
            ],
            "objection_handlers": [
                {
                    "objection": "Switching is disruptive.",
                    "acknowledge": "That concern is valid.",
                    "pivot": "The disruption of staying can be larger when support issues persist.",
                    "proof_point": "Recent evaluation pressure is visible in account signals.",
                }
            ],
            "talk_track": {
                "opening": "We are seeing support friction during renewal windows.",
                "mid_call_pivot": "A benchmark workshop can validate reliability risk quickly.",
                "closing": "Open to a short audit session this week?",
            },
        }
        quality = _evaluate_battle_card_quality(card, phase="final")
        assert quality["status"] == "sales_ready"
        assert quality["failed_checks"] == []

    def test_quality_gate_counts_segment_targeted_play_as_actionable(self):
        card = _sample_battle_card() | {
            "evidence_window_is_thin": False,
            "data_stale": False,
            "data_as_of_date": date.today().isoformat(),
            "evidence_window_days": 21,
            "high_intent_companies": [
                {"company": "Acme Transit", "urgency": 9, "buying_stage": "evaluation"},
            ],
            "recommended_plays": [
                {
                    "play": "Run a benchmark workshop for renewal planning.",
                    "target_segment": "SMB transportation accounts evaluating CRM alternatives",
                    "key_message": "Focus on cost control and lower operating friction.",
                    "timing": "During active evaluation windows this quarter.",
                },
                {
                    "play": "Launch a pricing assessment for accounts with renewal scrutiny.",
                    "target_segment": "Mid-market operations teams with budget pressure",
                    "key_message": "Show transparent pricing paths and migration support.",
                    "timing": "Immediately after pricing objections in pipeline reviews.",
                },
            ],
            "executive_summary": "Pricing pressure is driving immediate evaluation activity.",
            "discovery_questions": ["Which renewal accounts are currently evaluating alternatives?"],
            "landmine_questions": ["What happens if renewal pricing rises again next quarter?"],
            "objection_handlers": [
                {
                    "objection": "Switching cost is too high.",
                    "acknowledge": "The concern is common in active evaluations.",
                    "pivot": "A benchmark clarifies near-term ROI and migration risk.",
                    "proof_point": "Current signals show active evaluation pressure in this segment.",
                },
            ],
            "talk_track": {
                "opening": "Renewal pricing pressure is visible right now.",
                "mid_call_pivot": "A benchmark workshop will quantify fit and cost risk quickly.",
                "closing": "Can we schedule the workshop this week?",
            },
        }
        quality = _evaluate_battle_card_quality(card, phase="final")
        assert quality["status"] == "sales_ready"
        assert quality["required_signals"]["actionable_play_count"] >= 1

    def test_quality_gate_repairs_non_actionable_plays_with_deterministic_fallback(self):
        card = _sample_battle_card() | {
            "evidence_window_is_thin": False,
            "data_stale": False,
            "data_as_of_date": date.today().isoformat(),
            "evidence_window_days": 21,
            "high_intent_companies": [
                {"company": "Acme Transit", "urgency": 9, "buying_stage": "evaluation"},
            ],
            "recommended_plays": [
                {
                    "play": "Consider this platform at some point.",
                    "target_segment": "all",
                    "key_message": "It could help.",
                    "timing": "sometime",
                },
            ],
            "reasoning_contracts": {
                "schema_version": "v1",
                "vendor_core_reasoning": {
                    "schema_version": "v1",
                    "segment_playbook": {
                        "confidence": "medium",
                        "supporting_evidence": {
                            "top_strategic_roles": [
                                {"role_type": "economic_buyer", "source_id": "segment:role:economic_buyer"},
                                {"role_type": "evaluator", "source_id": "segment:role:evaluator"},
                            ],
                            "top_departments": [
                                {"department": "finance", "source_id": "segment:department:finance"},
                            ],
                        },
                    },
                    "timing_intelligence": {
                        "best_timing_window": "During renewal planning this quarter",
                    },
                },
            },
            "executive_summary": "Pricing pressure is driving evaluation activity.",
            "discovery_questions": ["What will drive your renewal decision this quarter?"],
            "landmine_questions": ["What happens if spend rises again at renewal?"],
            "objection_handlers": [
                {
                    "objection": "Switching seems risky.",
                    "acknowledge": "That concern is valid.",
                    "pivot": "A benchmark session lowers migration risk quickly.",
                    "proof_point": "Current evaluation pressure is visible in account signals.",
                },
            ],
            "talk_track": {
                "opening": "We are seeing renewal pricing pressure.",
                "mid_call_pivot": "A benchmark can validate fit and cost risk quickly.",
                "closing": "Can we book a workshop this week?",
            },
        }
        quality = _evaluate_battle_card_quality(card, phase="final")
        assert quality["status"] in {"sales_ready", "needs_review"}
        assert not any(
            "recommended plays are missing role/account targeting + timing + CTA" in item
            for item in quality["failed_checks"]
        )
        assert quality["required_signals"]["actionable_play_count"] >= 1

    def test_prioritize_seller_usable_primary_weakness_skips_other(self):
        card = _sample_battle_card() | {
            "weakness_analysis": [
                {"weakness": "Other concerns keep resurfacing in buyer feedback"},
                {"weakness": "Pricing pressure is creating renewal scrutiny"},
                {"weakness": "Ux concerns keep resurfacing in buyer feedback"},
            ],
        }
        _prioritize_seller_usable_primary_weakness(card)
        assert card["weakness_analysis"][0]["weakness"].lower().startswith("pricing")

    def test_apply_quality_sets_card_contract_and_row_status(self):
        card = _sample_battle_card() | {
            "evidence_window_is_thin": False,
            "data_stale": True,
        }
        quality = _apply_battle_card_quality(card, phase="deterministic")
        assert card["battle_card_quality"]["schema_version"] == "v1"
        assert card["quality_status"] == quality["status"] == "deterministic_fallback"
        assert _battle_card_row_status(card) == "deterministic_fallback"


class TestBattleCardSalesCopyParsing:
    def test_recovers_truncated_json(self):
        parsed = _parse_battle_card_sales_copy(
            '{"executive_summary":"Pressure is rising.","discovery_questions":["How are renewals going?"]'
        )
        assert parsed["executive_summary"] == "Pressure is rising."
        assert parsed["discovery_questions"] == ["How are renewals going?"]
        assert not parsed.get("_parse_fallback")

    def test_retry_payload_uses_raw_text_for_invalid_json(self):
        prior_attempt = _battle_card_prior_attempt(
            {"analysis_text": "Executive summary: support is slipping", "_parse_fallback": True}
        )
        assert prior_attempt == "Executive summary: support is slipping"


class TestBattleCardDeterministicSections:
    def test_builds_deterministic_weakness_analysis(self):
        analysis = _build_deterministic_battle_card_weakness_analysis(_sample_battle_card())
        assert len(analysis) == 1
        assert analysis[0]["weakness"] == "Pricing pressure is creating renewal scrutiny"
        assert "21.3% price complaint rate" in analysis[0]["evidence"]
        assert analysis[0]["customer_quote"] == "We need better control over rising platform costs."

    def test_skips_irrelevant_quote_for_unmatched_weakness(self):
        card = {
            **_sample_battle_card(),
            "vendor_weaknesses": [{"area": "pricing", "count": 246}, {"area": "reliability", "count": 120}],
            "customer_pain_quotes": [
                {"quote": "Amazon Web Services went down for 24 hours. So our business had to shut down for 24 hours", "urgency": 9},
            ],
        }
        analysis = _build_deterministic_battle_card_weakness_analysis(card)
        assert analysis[0]["customer_quote"] == ""
        assert analysis[1]["customer_quote"] == "Amazon Web Services went down for 24 hours. So our business had to shut down for 24 hours"

    def test_builds_deterministic_competitive_landscape(self):
        landscape = _build_deterministic_battle_card_competitive_landscape(_sample_battle_card())
        assert "recent price increases are creating renewal scrutiny" in landscape["vulnerability_window"].lower()
        assert landscape["top_alternatives"][0].startswith("WooCommerce (53 mentions in evaluation sets")
        assert any("Renewal cycles after recent price increases" in item for item in landscape["displacement_triggers"])

    def test_competitive_landscape_uses_category_reasoning_when_council_missing(self):
        card = _sample_battle_card()
        card.pop("category_council", None)
        card["reasoning_contracts"] = {
            "schema_version": "v1",
            "category_reasoning": {"schema_version": "v1", "market_regime": "consolidating"},
        }
        landscape = _build_deterministic_battle_card_competitive_landscape(card)
        assert "consolidating" in landscape["vulnerability_window"].lower()

    def test_dedupes_competitor_aliases_in_top_alternatives(self):
        card = {
            **_sample_battle_card(),
            "competitor_differentiators": [
                {"competitor": "Azure", "mentions": 9, "switch_count": 0, "primary_driver": "pricing"},
                {"competitor": "Microsoft Azure", "mentions": 7, "switch_count": 0, "primary_driver": "pricing"},
                {"competitor": "Google Cloud Platform", "mentions": 16, "switch_count": 0, "primary_driver": "pricing"},
            ],
        }
        landscape = _build_deterministic_battle_card_competitive_landscape(card)
        azure_entries = [item for item in landscape["top_alternatives"] if "Azure (" in item]
        assert len(azure_entries) == 1
        assert "(16 mentions in evaluation sets; primary driver: pricing)" in azure_entries[0]

    def test_allowed_claims_include_aggregated_competitor_mentions(self):
        card = {
            **_sample_battle_card(),
            "competitor_differentiators": [
                {"competitor": "Azure", "mentions": 9, "switch_count": 0, "primary_driver": "pricing"},
                {"competitor": "Microsoft Azure", "mentions": 7, "switch_count": 0, "primary_driver": "pricing"},
            ],
        }
        claims = _battle_card_allowed_claims(card)
        assert "16" in claims

    def test_skips_non_competitor_integration_labels_in_top_alternatives(self):
        card = {
            **_sample_battle_card(),
            "competitor_differentiators": [
                {"competitor": "Custom ChatGPT integration", "mentions": 8, "switch_count": 0, "primary_driver": "features"},
                {"competitor": "Custom ChatGPT Integration", "mentions": 5, "switch_count": 0, "primary_driver": "features"},
                {"competitor": "WooCommerce", "mentions": 53, "switch_count": 0, "primary_driver": "pricing"},
            ],
        }
        landscape = _build_deterministic_battle_card_competitive_landscape(card)
        assert all("Custom ChatGPT integration" not in item for item in landscape["top_alternatives"])
        assert any("WooCommerce" in item for item in landscape["top_alternatives"])

    def test_skips_zero_evidence_competitors_in_top_alternatives(self):
        card = {
            **_sample_battle_card(),
            "competitor_differentiators": [
                {"competitor": "Freshworks", "mentions": 0, "switch_count": 0, "primary_driver": "features"},
                {"competitor": "WooCommerce", "mentions": 53, "switch_count": 0, "primary_driver": "pricing"},
            ],
        }
        landscape = _build_deterministic_battle_card_competitive_landscape(card)
        assert all("Freshworks" not in item for item in landscape["top_alternatives"])


class _FakePersistPool:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    async def execute(self, *args: Any) -> None:
        self.calls.append(args)

    async def fetchval(self, *args: Any) -> int:
        self.calls.append(args)
        return 0


class TestBattleCardPersistence:
    @pytest.mark.asyncio
    async def test_persist_battle_card_writes_deterministic_sections(self):
        pool = _FakePersistPool()
        card = _sample_battle_card()
        card["weakness_analysis"] = _build_deterministic_battle_card_weakness_analysis(card)
        card["competitive_landscape"] = _build_deterministic_battle_card_competitive_landscape(card)
        card["reasoning_contracts"] = {"schema_version": "v1", "vendor_core_reasoning": {"schema_version": "v1"}}
        card["vendor_core_reasoning"] = {"schema_version": "v1"}
        card["render_packet_version"] = "contract_first_v1"
        card["render_contracts_used"] = True
        card["render_packet_hash"] = "abc123"
        card["llm_render_status"] = "pending"

        persisted = await _persist_battle_card(
            pool,
            today=date(2026, 3, 18),
            card=card,
            data_density=json.dumps({"vendors_analyzed": 1}),
            report_source_review_count=42,
            report_source_dist={"g2": 30, "reddit": 12},
            llm_model="pipeline_deterministic",
        )

        assert persisted is True
        payload = json.loads(pool.calls[0][4])
        assert payload["llm_render_status"] == "pending"
        assert payload["weakness_analysis"][0]["customer_quote"] == "We need better control over rising platform costs."
        assert "top_alternatives" in payload["competitive_landscape"]
        assert payload["render_packet_version"] == "contract_first_v1"
        assert payload["render_contracts_used"] is True
        assert payload["render_packet_hash"] == "abc123"

    @pytest.mark.asyncio
    async def test_overlay_persist_keeps_deterministic_sections_on_failure(self):
        pool = _FakePersistPool()
        card = _sample_battle_card()
        card["weakness_analysis"] = _build_deterministic_battle_card_weakness_analysis(card)
        card["competitive_landscape"] = _build_deterministic_battle_card_competitive_landscape(card)
        card["render_packet_version"] = "contract_first_v1"
        card["render_contracts_used"] = False
        card["llm_render_status"] = "failed"
        card["llm_render_error"] = "LLM did not return valid JSON"

        persisted = await _persist_battle_card(
            pool,
            today=date(2026, 3, 18),
            card=card,
            data_density=json.dumps({"vendors_analyzed": 1}),
            report_source_review_count=42,
            report_source_dist={"g2": 30, "reddit": 12},
            llm_model="pipeline_deterministic",
        )

        assert persisted is True
        payload = json.loads(pool.calls[0][4])
        assert payload["llm_render_status"] == "failed"
        assert payload["llm_render_error"] == "LLM did not return valid JSON"
        assert payload["weakness_analysis"][0]["weakness"] == "Pricing pressure is creating renewal scrutiny"
        assert payload["render_packet_version"] == "contract_first_v1"
        assert payload["render_contracts_used"] is False
        assert pool.calls[0][8] == "pipeline_deterministic"

    @pytest.mark.asyncio
    async def test_retire_gated_out_battle_cards_deletes_latest_vendor_rows(self):
        pool = _FakePersistPool()

        deleted = await _retire_gated_out_battle_cards(
            pool,
            ["Google Cloud Platform", " google cloud platform ", ""],
        )

        assert deleted == 0
        assert "DELETE FROM b2b_intelligence" in pool.calls[0][0]
        assert pool.calls[0][1] == ["google cloud platform"]


class TestBattleCardLlmRouting:
    def test_pair_opponent_skips_self_pairs(self):
        assert _pair_opponent(("Zendesk", "Zendesk"), "Zendesk") == ""
        assert _pair_opponent(("Freshdesk", "Zendesk"), "Zendesk") == "Freshdesk"

    def test_deterministic_sections_removed_from_llm_field_set(self):
        assert "weakness_analysis" not in _BATTLE_CARD_LLM_FIELDS
        assert "competitive_landscape" not in _BATTLE_CARD_LLM_FIELDS

    def test_auto_backend_uses_battle_card_model_when_present(self):
        cfg = type("Cfg", (), {
            "battle_card_llm_backend": "auto",
            "battle_card_openrouter_model": "openai/o4-mini",
        })()
        opts = _battle_card_llm_options(cfg)
        assert opts == {
            "workload": "synthesis",
            "try_openrouter": True,
            "openrouter_model": "openai/o4-mini",
        }

    def test_auto_backend_falls_back_to_global_reasoning_model_when_blank(self, monkeypatch):
        import atlas_brain.autonomous.tasks.b2b_battle_cards as battle_cards_mod

        monkeypatch.setattr(
            battle_cards_mod.settings.llm,
            "openrouter_reasoning_model",
            "anthropic/claude-sonnet-4",
        )
        cfg = type("Cfg", (), {})()
        cfg.battle_card_llm_backend = "auto"
        cfg.battle_card_openrouter_model = ""
        opts = _battle_card_llm_options(cfg)
        assert opts["openrouter_model"] == "anthropic/claude-sonnet-4"

    def test_anthropic_backend_disables_openrouter(self):
        cfg = type("Cfg", (), {
            "battle_card_llm_backend": "anthropic",
            "battle_card_openrouter_model": "openai/o4-mini",
        })()
        opts = _battle_card_llm_options(cfg)
        assert opts == {
            "workload": "anthropic",
            "try_openrouter": False,
            "openrouter_model": None,
        }

    def test_render_payload_prefers_reasoning_contracts(self):
        card = _sample_battle_card() | {
            "causal_narrative": {"trigger": "Old flat trigger", "primary_wedge": "support_erosion"},
            "vendor_core_reasoning": {
                "causal_narrative": {"trigger": "Price hike", "primary_wedge": "price_squeeze"},
                "segment_playbook": {"confidence": "medium"},
                "timing_intelligence": {"best_timing_window": "Before renewal", "confidence": "medium"},
            },
            "displacement_reasoning": {
                "migration_proof": {"confidence": "medium", "switching_is_real": False},
                "competitive_reframes": {"confidence": "medium", "reframes": []},
            },
            "category_reasoning": {"market_regime": "fragmented", "schema_version": "v1"},
            "reasoning_contracts": {
                "schema_version": "v1",
                "vendor_core_reasoning": {
                    "schema_version": "v1",
                    "causal_narrative": {"trigger": "Contract trigger", "primary_wedge": "price_squeeze"},
                    "segment_playbook": {"confidence": "high"},
                    "timing_intelligence": {"best_timing_window": "During planning", "confidence": "medium"},
                },
                "displacement_reasoning": {
                    "schema_version": "v1",
                    "migration_proof": {"confidence": "high", "switching_is_real": True},
                    "competitive_reframes": {"confidence": "medium", "reframes": []},
                },
                "category_reasoning": {"schema_version": "v1", "market_regime": "consolidating"},
                "account_reasoning": {
                    "schema_version": "v1",
                    "market_summary": "Two accounts are actively evaluating alternatives.",
                    "total_accounts": {"value": 6, "source_id": "accounts:summary:total_accounts"},
                    "high_intent_count": {"value": 4, "source_id": "accounts:summary:high_intent_count"},
                    "active_eval_count": {
                        "value": 2,
                        "source_id": "accounts:summary:active_eval_signal_count",
                    },
                    "top_accounts": [
                        {"name": "Acme Corp", "intent_score": 0.9, "source_id": "accounts:item:acme"},
                    ],
                },
            },
        }

        payload = _build_battle_card_render_payload(card)

        assert payload["render_packet_version"] == "contract_first_v1"
        assert payload["vendor_core_reasoning"]["causal_narrative"]["trigger"] == "Contract trigger"
        assert payload["displacement_reasoning"]["migration_proof"]["switching_is_real"] is True
        assert payload["category_reasoning"]["market_regime"] == "consolidating"
        assert payload["account_reasoning"]["top_accounts"][0]["name"] == "Acme Corp"
        assert payload["locked_facts"]["vendor"] == "Shopify"
        assert "causal_narrative" not in payload
        assert "timing_intelligence" not in payload
        assert "migration_proof" not in payload

    def test_render_payload_uses_raw_category_reasoning_when_contract_missing(self):
        card = _sample_battle_card() | {
            "category_reasoning": {"schema_version": "v1", "market_regime": "fragmented"},
            "reasoning_contracts": {
                "schema_version": "v1",
                "vendor_core_reasoning": {"schema_version": "v1"},
            },
        }

        payload = _build_battle_card_render_payload(card)

        assert payload["category_reasoning"]["market_regime"] == "fragmented"

    def test_render_payload_includes_retry_fields(self):
        payload = _build_battle_card_render_payload(
            _sample_battle_card(),
            prior_attempt="Previous draft",
            validation_feedback=["unsupported numeric claims", "overstates urgency"],
        )

        assert payload["prior_attempt"] == "Previous draft"
        assert payload["validation_feedback"] == [
            "unsupported numeric claims",
            "overstates urgency",
        ]

    def test_render_payload_hash_changes_with_contract_reasoning(self):
        from atlas_brain.reasoning.semantic_cache import compute_evidence_hash

        base_card = _sample_battle_card()
        base_hash = compute_evidence_hash(_build_battle_card_render_payload(base_card))

        changed_card = base_card | {
            "vendor_core_reasoning": {
                "causal_narrative": {"trigger": "Price hike", "primary_wedge": "price_squeeze"},
            },
            "reasoning_contracts": {
                "schema_version": "v1",
                "vendor_core_reasoning": {
                    "schema_version": "v1",
                    "causal_narrative": {"trigger": "Price hike", "primary_wedge": "price_squeeze"},
                },
            },
        }
        changed_hash = compute_evidence_hash(_build_battle_card_render_payload(changed_card))

        assert base_hash != changed_hash

    def test_seller_usable_battles_keep_only_target_losing_pairs(self):
        battles = [
            {
                "opponent": "WooCommerce",
                "winner": "Shopify",
                "loser": "WooCommerce",
                "conclusion": "Shopify is winning this battle.",
            },
            {
                "opponent": "BigCommerce",
                "winner": "BigCommerce",
                "loser": "Shopify",
                "conclusion": "Shopify is losing this battle.",
            },
        ]
        usable = _battle_card_seller_usable_battles("Shopify", battles)
        assert len(usable) == 1
        assert usable[0]["opponent"] == "BigCommerce"

    def test_render_payload_filters_target_winning_battles(self):
        card = _sample_battle_card() | {
            "cross_vendor_battles": [
                {
                    "opponent": "WooCommerce",
                    "winner": "Shopify",
                    "loser": "WooCommerce",
                    "conclusion": "Shopify is winning this battle.",
                },
                {
                    "opponent": "BigCommerce",
                    "winner": "BigCommerce",
                    "loser": "Shopify",
                    "conclusion": "Shopify is losing this battle.",
                },
            ]
        }

        payload = _build_battle_card_render_payload(card)
        assert [b["opponent"] for b in payload["cross_vendor_battles"]] == ["BigCommerce"]

    def test_validator_prefers_contract_claims_over_stale_flat_sections(self):
        card = _sample_battle_card() | {
            "causal_narrative": {
                "primary_wedge": "support_erosion",
                "trigger": "Stale flat trigger 2024",
            },
            "reasoning_contracts": {
                "schema_version": "v1",
                "vendor_core_reasoning": {
                    "schema_version": "v1",
                    "causal_narrative": {
                        "primary_wedge": "price_squeeze",
                        "trigger": "Price hike",
                    },
                },
            },
        }

        claims = _battle_card_allowed_claims(card)
        assert "2024" not in claims
        warnings = _validate_battle_card_sales_copy(
            card,
            {"executive_summary": "The key issue showed up in 2024."},
        )
        assert any("references a year" in warning for warning in warnings)

    def test_validator_does_not_backfill_missing_contract_section_from_flat_mirror(self):
        card = _sample_battle_card() | {
            "timing_intelligence": {
                "best_timing_window": "Legacy 2024 window",
            },
            "reasoning_contracts": {
                "schema_version": "v1",
                "vendor_core_reasoning": {
                    "schema_version": "v1",
                    "causal_narrative": {
                        "primary_wedge": "price_squeeze",
                        "trigger": "Price hike",
                    },
                },
            },
        }

        claims = _battle_card_allowed_claims(card)
        assert "2024" not in claims

    def test_validator_does_not_allow_cross_vendor_numeric_tokens_from_prose(self):
        card = _sample_battle_card() | {
            "cross_vendor_battles": [
                {
                    "opponent": "WooCommerce",
                    "conclusion": "Shopify is winning in the 52% SMB segment with 43 explicit switches.",
                    "key_insights": [
                        {"insight": "52% SMB share", "evidence": "43 explicit switches"},
                    ],
                }
            ]
        }

        claims = _battle_card_allowed_claims(card)
        assert "52%" not in claims
        assert "43" not in claims


class TestBattleCardExecutionProgress:
    @pytest.mark.asyncio
    @patch("atlas_brain.storage.repositories.scheduled_task.get_scheduled_task_repo")
    async def test_update_execution_progress_uses_injected_execution_id(self, mock_repo_fn):
        repo = AsyncMock()
        mock_repo_fn.return_value = repo
        task = type("Task", (), {"name": "b2b_battle_cards", "metadata": {"_execution_id": str(uuid4())}})()

        await _update_execution_progress(
            task,
            stage="llm_overlay",
            progress_current=2,
            progress_total=15,
            cards_built=15,
        )

        repo.update_execution_metadata.assert_awaited_once()
        called_exec_id = repo.update_execution_metadata.call_args[0][0]
        payload = repo.update_execution_metadata.call_args[0][1]
        assert str(called_exec_id) == task.metadata["_execution_id"]
        assert payload["stage"] == "llm_overlay"
        assert payload["progress_current"] == 2
        assert payload["cards_built"] == 15


class TestDeterministicBattleCardBuild:
    def test_build_deterministic_battle_cards_is_not_capped_at_fifteen_by_default(self):
        vendor_scores = [
            {
                "vendor_name": f"Vendor {idx}",
                "product_category": "Testing",
                "total_reviews": 30,
                "churn_intent": 6,
                "avg_urgency": 6.5,
            }
            for idx in range(20)
        ]

        cards = _build_deterministic_battle_cards(
            vendor_scores,
            pain_lookup={},
            competitor_lookup={},
            feature_gap_lookup={},
            quote_lookup={},
            price_lookup={},
            budget_lookup={},
            sentiment_lookup={},
            dm_lookup={},
            company_lookup={},
            product_profile_lookup={},
            competitive_disp=[],
            competitor_reasons=[],
        )

        assert len(cards) == 20

    def test_build_deterministic_battle_cards_respects_explicit_limit(self):
        vendor_scores = [
            {
                "vendor_name": f"Vendor {idx}",
                "product_category": "Testing",
                "total_reviews": 30,
                "churn_intent": 6,
                "avg_urgency": 6.5,
            }
            for idx in range(8)
        ]

        cards = _build_deterministic_battle_cards(
            vendor_scores,
            pain_lookup={},
            competitor_lookup={},
            feature_gap_lookup={},
            quote_lookup={},
            price_lookup={},
            budget_lookup={},
            sentiment_lookup={},
            dm_lookup={},
            company_lookup={},
            product_profile_lookup={},
            competitive_disp=[],
            competitor_reasons=[],
            limit=3,
        )

        assert len(cards) == 3

    def test_build_deterministic_battle_cards_promote_segment_and_timing_summaries(self):
        vendor_scores = [
            {
                "vendor_name": "Zendesk",
                "product_category": "Helpdesk",
                "total_reviews": 40,
                "churn_intent": 8,
                "avg_urgency": 6.8,
            },
        ]

        cards = _build_deterministic_battle_cards(
            vendor_scores,
            pain_lookup={},
            competitor_lookup={},
            feature_gap_lookup={},
            quote_lookup={},
            price_lookup={},
            budget_lookup={},
            sentiment_lookup={},
            dm_lookup={},
            company_lookup={},
            product_profile_lookup={},
            competitive_disp=[],
            competitor_reasons=[],
            synthesis_views={
                "Zendesk": _load_synth_view("Zendesk", {
                    "schema_version": "2.1",
                    "reasoning_contracts": {
                        "schema_version": "v1",
                        "vendor_core_reasoning": {
                            "schema_version": "v1",
                            "segment_playbook": {
                                "confidence": "medium",
                                "supporting_evidence": {
                                    "top_strategic_roles": [
                                        {
                                            "role_type": "economic_buyer",
                                            "source_id": "segment:role:economic_buyer",
                                        },
                                    ],
                                    "top_departments": [
                                        {
                                            "department": "finance",
                                            "source_id": "segment:department:finance",
                                        },
                                    ],
                                    "top_contract_segments": [
                                        {
                                            "segment": "enterprise",
                                            "source_id": "segment:contract:enterprise",
                                        },
                                    ],
                                },
                            },
                            "timing_intelligence": {
                                "confidence": "medium",
                                "best_timing_window": "immediate - pricing pressure is current",
                                "active_eval_signals": {
                                    "value": 2,
                                    "source_id": "accounts:summary:active_eval_signal_count",
                                },
                                "immediate_triggers": [
                                    {"trigger": "Q2 renewal", "type": "deadline"},
                                ],
                            },
                        },
                    },
                }),
            },
        )

        assert len(cards) == 1
        assert "economic buyers" in cards[0]["segment_targeting_summary"]
        assert "Finance teams" in cards[0]["segment_targeting_summary"]
        assert "enterprise contracts" in cards[0]["segment_targeting_summary"]
        assert cards[0]["timing_summary"].startswith(
            "Immediate - pricing pressure is current."
        )
        assert cards[0]["priority_timing_triggers"] == ["Q2 renewal"]

    def test_build_deterministic_battle_cards_normalize_priority_segment_and_add_trigger_detail(self):
        cards = _build_deterministic_battle_cards(
            vendor_scores=[
                {
                    "vendor_name": "Close",
                    "product_category": "CRM",
                    "total_reviews": 40,
                    "churn_intent": 8,
                    "avg_urgency": 6.2,
                },
            ],
            pain_lookup={},
            competitor_lookup={},
            feature_gap_lookup={},
            quote_lookup={},
            price_lookup={},
            budget_lookup={},
            sentiment_lookup={},
            dm_lookup={},
            company_lookup={},
            product_profile_lookup={},
            competitive_disp=[],
            competitor_reasons=[],
            synthesis_views={
                "Close": _load_synth_view("Close", {
                    "schema_version": "2.1",
                    "reasoning_contracts": {
                        "schema_version": "v1",
                        "vendor_core_reasoning": {
                            "schema_version": "v1",
                            "segment_playbook": {
                                "confidence": "medium",
                                "priority_segments": [
                                    {
                                        "segment": "Small Business (1-10 employees)",
                                        "best_opening_angle": "Highlight a pricing benchmark",
                                    },
                                ],
                            },
                            "timing_intelligence": {
                                "confidence": "medium",
                                "best_timing_window": "Immediate - active evaluation signals are already present",
                                "active_eval_signals": {
                                    "value": 3,
                                    "source_id": "accounts:summary:active_eval_signal_count",
                                },
                                "immediate_triggers": [
                                    {"trigger": "Active evaluation of BambooHR (3 accounts)", "type": "signal"},
                                ],
                            },
                        },
                    },
                }),
            },
        )

        assert cards[0]["segment_targeting_summary"].startswith(
            "Best current segment wedge is Small Business"
        )
        assert "(1 10 employees)" not in cards[0]["segment_targeting_summary"]
        assert "led with highlight a pricing benchmark." in cards[0]["segment_targeting_summary"]
        assert "Key trigger: Active evaluation of BambooHR (3 accounts)." in cards[0]["timing_summary"]


class TestBattleCardCompanyPrioritization:
    def test_companies_from_vault_prioritize_required_stage_accounts(self):
        min_urgency = float(settings.b2b_churn.battle_card_quality_min_high_intent_urgency)
        vault = {
            "company_signals": [
                {"company_name": "Account A", "urgency_score": min_urgency + 2.0, "buyer_role": "ic", "buying_stage": "post_purchase", "source": "reddit"},
                {"company_name": "Account B", "urgency_score": min_urgency + 1.5, "buyer_role": "ic", "buying_stage": "post_purchase", "source": "reddit"},
                {"company_name": "Account C", "urgency_score": min_urgency + 1.0, "buyer_role": "ic", "buying_stage": "post_purchase", "source": "reddit"},
                {"company_name": "Account D", "urgency_score": min_urgency + 0.5, "buyer_role": "ic", "buying_stage": "unknown", "source": "reddit"},
                {"company_name": "Account E", "urgency_score": min_urgency + 0.2, "buyer_role": "ic", "buying_stage": "post_purchase", "source": "reddit"},
                {"company_name": "Account Eval", "urgency_score": min_urgency, "buyer_role": "evaluator", "buying_stage": "evaluation", "source": "reddit"},
            ],
        }

        companies = _battle_card_companies_from_evidence_vault(vault, limit=5)

        assert len(companies) == 5
        assert any(str(item.get("buying_stage") or "").lower() == "evaluation" for item in companies)

    def test_companies_from_vault_do_not_promote_low_urgency_required_stage(self):
        min_urgency = float(settings.b2b_churn.battle_card_quality_min_high_intent_urgency)
        vault = {
            "company_signals": [
                {"company_name": "High Urgency Post", "urgency_score": min_urgency + 2.0, "buyer_role": "ic", "buying_stage": "post_purchase", "source": "reddit"},
                {"company_name": "Low Urgency Eval", "urgency_score": max(min_urgency - 2.0, 0.0), "buyer_role": "evaluator", "buying_stage": "evaluation", "source": "reddit"},
            ],
        }

        companies = _battle_card_companies_from_evidence_vault(vault, limit=2)

        assert companies[0]["company"] == "High Urgency Post"


class TestChurnIntelligenceExecutionProgress:
    @pytest.mark.asyncio
    async def test_run_emits_loading_progress_before_heavy_fetch(self, monkeypatch):
        progress = AsyncMock()
        pool = type("Pool", (), {"is_initialized": True})()

        async def stop_after_progress():
            raise RuntimeError("stop_after_progress")

        monkeypatch.setattr(churn_intel_mod.settings.b2b_churn, "enabled", True, raising=False)
        monkeypatch.setattr(churn_intel_mod.settings.b2b_churn, "intelligence_enabled", True, raising=False)
        monkeypatch.setattr(churn_intel_mod, "_update_execution_progress", progress)
        monkeypatch.setattr(churn_intel_mod, "get_db_pool", lambda: pool)
        monkeypatch.setattr(churn_intel_mod, "_warm_vendor_cache", stop_after_progress)

        task = type("Task", (), {"metadata": {"_execution_id": str(uuid4())}})()
        with pytest.raises(RuntimeError, match="stop_after_progress"):
            await churn_intel_mod.run(task)

        progress.assert_awaited()
        assert progress.await_args_list[0].kwargs["stage"] == churn_intel_mod._STAGE_LOADING_INPUTS

    @pytest.mark.asyncio
    async def test_run_updates_reasoning_progress_per_completed_vendor(self, monkeypatch):
        progress = AsyncMock()
        pool = type("Pool", (), {"is_initialized": True, "fetch": AsyncMock(return_value=[])})()
        reasoning_pkg = __import__("atlas_brain.reasoning", fromlist=["dummy"])
        tiers_mod = __import__("atlas_brain.reasoning.tiers", fromlist=["dummy"])

        class StopAfterReasoning(RuntimeError):
            pass

        class FakeReasoner:
            def __init__(self):
                self._cache = object()

            async def analyze(self, **kwargs):
                vendor = kwargs["vendor_name"]
                return SimpleNamespace(
                    conclusion={"archetype": f"{vendor}_shape", "risk_level": "high", "executive_summary": vendor, "key_signals": []},
                    confidence=0.8,
                    mode="reason",
                    tokens_used=11,
                    reasoning_steps=[],
                    boundary_conditions={},
                )

        async def fake_gather(*coros, **kwargs):
            if len(coros) == 34:
                for coro in coros:
                    close = getattr(coro, "close", None)
                    if close:
                        close()
                vendor_scores = [
                    {"vendor_name": "Zendesk", "product_category": "Helpdesk", "avg_urgency": 7.0},
                    {"vendor_name": "Intercom", "product_category": "Helpdesk", "avg_urgency": 6.0},
                ]
                return (
                    vendor_scores,                    # vendor_scores
                    [{"vendor_name": "Zendesk"}],    # high_intent
                    {},                               # existing_company_signals
                    [],                               # competitive_disp
                    [],                               # pain_dist
                    [],                               # feature_gaps
                    [],                               # negative_counts
                    [],                               # price_rates
                    [],                               # dm_rates
                    [],                               # churning_companies
                    [],                               # quotable_evidence
                    [],                               # evidence_vault_review_rows
                    [],                               # budget_signals
                    [],                               # use_case_dist
                    [],                               # sentiment_traj
                    [],                               # buyer_auth
                    [],                               # timeline_signals
                    [],                               # competitor_reasons
                    [],                               # keyword_spikes
                    {},                               # data_context
                    {},                               # vendor_provenance
                    {},                               # displacement_provenance
                    {},                               # pain_provenance
                    {},                               # use_case_provenance
                    {},                               # integration_provenance
                    {},                               # buyer_profile_provenance
                    [],                               # insider_aggregates_raw
                    [],                               # product_profiles_raw
                    ([], []),                         # _review_text_aggs
                    [],                               # _department_dist
                    [],                               # _company_size_dist
                    ([], []),                         # _contract_ctx_aggs
                    [],                               # _sentiment_tenure_raw
                    [],                               # _turning_points_raw
                )
            return [await coro for coro in coros]

        monkeypatch.setattr(churn_intel_mod.settings.b2b_churn, "enabled", True, raising=False)
        monkeypatch.setattr(churn_intel_mod.settings.b2b_churn, "intelligence_enabled", True, raising=False)
        monkeypatch.setattr(churn_intel_mod.settings.b2b_churn, "stratified_reasoning_enabled", True, raising=False)
        monkeypatch.setattr(churn_intel_mod.settings.b2b_churn, "cross_vendor_reasoning_enabled", False, raising=False)
        monkeypatch.setattr(churn_intel_mod.settings.b2b_churn, "stratified_reasoning_vendor_cap", 2, raising=False)
        monkeypatch.setattr(churn_intel_mod.settings.b2b_churn, "stratified_reasoning_vendor_limit", 0, raising=False)
        monkeypatch.setattr(churn_intel_mod, "_update_execution_progress", progress)
        monkeypatch.setattr(churn_intel_mod, "get_db_pool", lambda: pool)
        monkeypatch.setattr(churn_intel_mod, "_warm_vendor_cache", AsyncMock())
        monkeypatch.setattr(churn_intel_mod, "_sync_vendor_firmographics", AsyncMock(return_value=42))
        monkeypatch.setattr(churn_intel_mod, "_fetch_prior_reports", AsyncMock(return_value=[]))
        monkeypatch.setattr(churn_intel_mod, "_fetch_prior_archetypes", AsyncMock(return_value={}))
        monkeypatch.setattr(churn_intel_mod, "_build_exploratory_payload", lambda *args, **kwargs: ({}, 0))
        monkeypatch.setattr(churn_intel_mod, "_build_vendor_evidence", lambda vs, **kwargs: {"vendor_name": vs["vendor_name"], "product_category": vs.get("product_category", "")})
        monkeypatch.setattr(churn_intel_mod, "_build_deterministic_displacement_map", lambda *args, **kwargs: (_ for _ in ()).throw(StopAfterReasoning()))
        monkeypatch.setattr(churn_intel_mod.asyncio, "gather", fake_gather)
        monkeypatch.setattr(reasoning_pkg, "get_stratified_reasoner", lambda: FakeReasoner())
        monkeypatch.setattr(reasoning_pkg, "init_stratified_reasoner", AsyncMock())
        monkeypatch.setattr(tiers_mod, "gather_tier_context", AsyncMock(return_value={}))

        task = type("Task", (), {"metadata": {"_execution_id": str(uuid4())}})()
        with pytest.raises(StopAfterReasoning):
            await churn_intel_mod.run(task)

        reasoning_calls = [
            call.kwargs for call in progress.await_args_list
            if call.kwargs.get("stage") == churn_intel_mod._STAGE_REASONING
        ]
        progress_values = [call["progress_current"] for call in reasoning_calls if "progress_current" in call]
        assert progress_values[:3] == [0, 1, 2]
        assert progress_values[-1] == 2
        assert reasoning_calls[0]["progress_total"] == 2

    def test_openrouter_backend_uses_override_when_present(self):
        cfg = type("Cfg", (), {
            "battle_card_llm_backend": "openrouter",
            "battle_card_openrouter_model": "openai/o4-mini-high",
        })()
        opts = _battle_card_llm_options(cfg)
        assert opts == {
            "workload": "synthesis",
            "try_openrouter": True,
            "openrouter_model": "openai/o4-mini-high",
        }


def _sample_scorecard() -> dict[str, Any]:
    return {
        "vendor": "Zendesk",
        "risk_level": "high",
        "churn_pressure_score": 72.0,
        "avg_urgency": 6.4,
        "trend": "worsening",
        "top_pain": "support",
        "reasoning_summary": "Buyers considering Zendesk should scrutinize support reliability as the pressure pattern continues to worsen.",
        "archetype": "support_collapse",
        "archetype_confidence": 0.82,
        "cross_vendor_comparisons": [
            {
                "opponent": "Freshdesk",
                "conclusion": "Freshdesk is gaining on support responsiveness.",
                "confidence": 0.84,
                "resource_advantage": "Freshdesk holds the relative service-speed edge in recent comparisons.",
            },
            {
                "opponent": "Intercom",
                "conclusion": "Intercom mention is weak.",
                "confidence": 0.42,
                "resource_advantage": "Intercom has broader AI branding.",
            },
        ],
        "competitor_overlap": [{"competitor": "Freshdesk", "mentions": 18}],
    }


class TestScorecardNarrativeGuardrails:
    def test_build_scorecard_locked_facts_filters_low_confidence_refs(self):
        locked = _build_scorecard_locked_facts(_sample_scorecard())
        assert locked["vendor"] == "Zendesk"
        assert locked["archetype"] == "support_collapse"
        assert locked["allowed_opponents"] == ["Freshdesk"]
        assert locked["comparison"]["opponent"] == "Freshdesk"

    def test_validate_scorecard_expert_take_rejects_wrong_archetype(self):
        warnings = _validate_scorecard_expert_take(
            _sample_scorecard(),
            "Buyers considering Zendesk should treat this as a pricing_shock pattern right now.",
        )
        assert any("pricing_shock" in warning for warning in warnings)

    def test_validate_scorecard_expert_take_rejects_low_confidence_opponent(self):
        warnings = _validate_scorecard_expert_take(
            _sample_scorecard(),
            "Buyers considering Zendesk should assume Intercom now has the stronger position.",
        )
        assert any("Intercom" in warning for warning in warnings)

    def test_fallback_scorecard_expert_take_stays_short_and_buyer_facing(self):
        text = _fallback_scorecard_expert_take(_sample_scorecard())
        assert len(text.split()) <= 80
        assert "buyers considering zendesk" in text.lower()

    def test_build_battle_card_locked_facts_captures_allowed_opponents(self):
        locked = _build_battle_card_locked_facts(_sample_battle_card() | {
            "archetype": "pricing_shock",
            "archetype_risk_level": "high",
            "cross_vendor_battles": [{"opponent": "BigCommerce"}],
            "resource_asymmetry": {"opponent": "WooCommerce", "resource_advantage": "WooCommerce has the broader plugin ecosystem."},
        })
        assert locked["archetype"] == "pricing_shock"
        assert locked["priority_language_allowed"] is False
        assert "WooCommerce" in locked["allowed_opponents"]
        assert "BigCommerce" in locked["allowed_opponents"]


# ---------------------------------------------------------------------------
# Phase 1: Executive source list
# ---------------------------------------------------------------------------


class TestExecutiveSourceList:
    @patch("atlas_brain.autonomous.tasks._b2b_shared.settings")
    def test_parses_config(self, mock_settings):
        mock_settings.b2b_churn.intelligence_executive_sources = "g2,capterra,trustradius"
        result = _executive_source_list()
        assert result == ["g2", "capterra", "trustradius"]

    @patch("atlas_brain.autonomous.tasks._b2b_shared.settings")
    def test_separate_from_broad_allowlist(self, mock_settings):
        """Executive sources should be a subset of the broad allowlist."""
        mock_settings.b2b_churn.intelligence_executive_sources = "g2,capterra"
        exec_sources = _executive_source_list()
        assert "reddit" not in exec_sources
        assert "trustpilot" not in exec_sources

    @patch("atlas_brain.autonomous.tasks._b2b_shared.settings")
    def test_handles_whitespace(self, mock_settings):
        mock_settings.b2b_churn.intelligence_executive_sources = " g2 , capterra , trustradius "
        result = _executive_source_list()
        assert result == ["g2", "capterra", "trustradius"]


# ---------------------------------------------------------------------------
# Vendor-level churn pressure score
# ---------------------------------------------------------------------------


def _make_vendor_score(vendor: str, category: str = "Helpdesk",
                       total_reviews: int = 60, churn_intent: int = 20,
                       avg_urgency: float = 5.0) -> dict:
    return {
        "vendor_name": vendor,
        "product_category": category,
        "total_reviews": total_reviews,
        "churn_intent": churn_intent,
        "avg_urgency": avg_urgency,
        "avg_rating_normalized": 0.5,
        "recommend_yes": 10,
        "recommend_no": 5,
        "positive_review_pct": 50.0,
    }


def _empty_lookups(**overrides) -> dict:
    """Return a dict of all empty lookups for _build_deterministic_vendor_feed."""
    defaults = {
        "pain_lookup": {},
        "competitor_lookup": {},
        "feature_gap_lookup": {},
        "quote_lookup": {},
        "budget_lookup": {},
        "sentiment_lookup": {},
        "buyer_auth_lookup": {},
        "dm_lookup": {},
        "price_lookup": {},
        "company_lookup": {},
        "keyword_spike_lookup": {},
        "prior_reports": [],
    }
    defaults.update(overrides)
    return defaults


class TestChurnPressureScore:
    def test_zero_inputs(self):
        score = _compute_churn_pressure_score(
            churn_density=0, avg_urgency=0, dm_churn_rate=0,
            displacement_mention_count=0, price_complaint_rate=0,
            total_reviews=100,
        )
        assert score == 0.0

    def test_maximum_inputs(self):
        score = _compute_churn_pressure_score(
            churn_density=100.0, avg_urgency=10.0, dm_churn_rate=1.0,
            displacement_mention_count=50, price_complaint_rate=1.0,
            total_reviews=100,
        )
        assert score == 100.0

    def test_low_confidence_penalty(self):
        kwargs = dict(churn_density=50.0, avg_urgency=7.0, dm_churn_rate=0.5,
                      displacement_mention_count=10, price_complaint_rate=0.3)
        high = _compute_churn_pressure_score(**kwargs, total_reviews=100)
        low = _compute_churn_pressure_score(**kwargs, total_reviews=10)
        assert low < high

    def test_medium_confidence(self):
        kwargs = dict(churn_density=40.0, avg_urgency=5.0, dm_churn_rate=0.4,
                      displacement_mention_count=5, price_complaint_rate=0.2)
        high = _compute_churn_pressure_score(**kwargs, total_reviews=60)
        med = _compute_churn_pressure_score(**kwargs, total_reviews=30)
        assert med < high
        assert med == round(high * 0.85, 1)


class TestVendorFeedRanking:
    def test_descending_score_order(self):
        """Vendor feed entries must be sorted by churn_pressure_score DESC."""
        vendors = [
            _make_vendor_score("LowVendor", total_reviews=60, churn_intent=10, avg_urgency=3.0),
            _make_vendor_score("HighVendor", total_reviews=60, churn_intent=30, avg_urgency=8.0),
            _make_vendor_score("MidVendor", total_reviews=60, churn_intent=20, avg_urgency=6.0),
        ]
        lookups = _empty_lookups(
            dm_lookup={"LowVendor": 0.1, "HighVendor": 0.7, "MidVendor": 0.4},
            price_lookup={"LowVendor": 0.1, "HighVendor": 0.5, "MidVendor": 0.3},
        )
        feed = _build_deterministic_vendor_feed(vendors, **lookups)
        scores = [e["churn_pressure_score"] for e in feed]
        assert scores == sorted(scores, reverse=True)
        assert feed[0]["vendor"] == "HighVendor"

    def test_multi_category_aggregation(self):
        """Same vendor in two categories should merge into one entry."""
        vendors = [
            _make_vendor_score("Mailchimp", category="Marketing Automation",
                               total_reviews=90, churn_intent=40, avg_urgency=4.3),
            _make_vendor_score("Mailchimp", category="Email Marketing",
                               total_reviews=26, churn_intent=12, avg_urgency=5.6),
        ]
        lookups = _empty_lookups(
            dm_lookup={"Mailchimp": 0.5},
            price_lookup={"Mailchimp": 0.4},
        )
        feed = _build_deterministic_vendor_feed(vendors, **lookups)
        assert len(feed) == 1
        entry = feed[0]
        assert entry["vendor"] == "Mailchimp"
        # Reviews summed: 90 + 26 = 116
        assert entry["total_reviews"] == 116
        # Churn density from merged totals: (40+12)/116 * 100 = 44.8%
        assert entry["churn_signal_density"] == 44.8
        # Weighted avg urgency: (4.3*90 + 5.6*26) / 116 = 4.59 -> 4.6
        assert entry["avg_urgency"] == 4.6
        # Dominant category: Marketing Automation (90 > 26)
        assert entry["category"] == "Marketing Automation"


class TestVendorFeedNamedAccounts:
    def test_named_accounts_populated(self):
        """When company_lookup has data, named_accounts should be populated."""
        vendors = [_make_vendor_score("Zendesk", avg_urgency=7.0, churn_intent=25)]
        lookups = _empty_lookups(
            company_lookup={"Zendesk": [{
                "company": "BrightPath",
                "urgency": 9,
                "source": "reddit",
                "buying_stage": "evaluation",
                "confidence_score": 0.82,
                "decision_maker": True,
                "first_seen_at": "2026-03-01T00:00:00+00:00",
                "last_seen_at": "2026-03-19T00:00:00+00:00",
            }]},
            dm_lookup={"Zendesk": 0.5},
            price_lookup={"Zendesk": 0.4},
        )
        feed = _build_deterministic_vendor_feed(vendors, **lookups)
        assert len(feed) == 1
        accts = feed[0]["named_accounts"]
        assert len(accts) == 1
        assert accts[0]["company"] == "BrightPath"
        assert accts[0]["urgency"] == 9
        assert accts[0]["source"] == "reddit"
        assert accts[0]["buying_stage"] == "evaluation"
        assert accts[0]["confidence_score"] == 0.82
        assert accts[0]["decision_maker"] is True


class TestVendorFeedNoCompanyRequired:
    def test_works_with_empty_company_lookup(self):
        """Feed should work fine with no company data at all."""
        vendors = [_make_vendor_score("Jira", avg_urgency=7.0, churn_intent=25)]
        lookups = _empty_lookups(
            dm_lookup={"Jira": 0.5},
            price_lookup={"Jira": 0.3},
        )
        feed = _build_deterministic_vendor_feed(vendors, **lookups)
        assert len(feed) == 1
        assert feed[0]["vendor"] == "Jira"
        assert feed[0]["named_accounts"] == []
        assert feed[0]["churn_pressure_score"] > 0


class TestVendorExecutiveSummary:
    @patch("atlas_brain.autonomous.tasks._b2b_shared.settings")
    def test_uses_vendor_language(self, mock_settings):
        """Executive summary should say 'vendors under elevated churn pressure'."""
        mock_settings.b2b_churn.intelligence_executive_sources = "g2,capterra"
        feed = [
            {
                "vendor": "Zendesk",
                "churn_signal_density": 38.6,
                "total_reviews": 59,
                "avg_urgency": 7.2,
                "top_pain": "pricing",
                "pain_breakdown": [{"category": "pricing", "count": 42}],
                "top_displacement_targets": [{"competitor": "Freshdesk", "mentions": 12}],
                "key_quote": "We switched because pricing doubled.",
                "named_accounts": [],
            },
        ]
        summary = _build_validated_executive_summary(
            {"weekly_churn_feed": feed},
            data_context={
                "enrichment_period": {"earliest": "2026-03-01", "latest": "2026-03-07"},
                "source_distribution": {"g2": {"reviews": 40, "high_urgency": 5}},
            },
            executive_sources=["g2", "capterra"],
        )
        assert "vendors under elevated churn pressure" in summary
        assert "account signals" not in summary
        assert "Strongest vendor-level churn signals" in summary
        assert "Zendesk" in summary


# ---------------------------------------------------------------------------
# Evidence confidence scoring (renamed from _compute_displacement_confidence)
# ---------------------------------------------------------------------------


class TestEvidenceConfidence:
    def test_zero_mentions(self):
        """Zero mentions should return 0.0 confidence."""
        assert _compute_evidence_confidence(0, {}) == 0.0

    def test_high_confidence(self):
        """20+ mentions from 3+ verified sources should yield >= 0.9."""
        dist = {"g2": 10, "capterra": 8, "trustradius": 5}
        score = _compute_evidence_confidence(23, dist)
        assert score >= 0.9

    def test_single_unverified_source(self):
        """Single unverified source should yield < 0.4."""
        score = _compute_evidence_confidence(3, {"reddit": 3})
        assert score < 0.4

    def test_mixed_sources_mid_range(self):
        """Mix of verified + unverified at moderate volume -> mid-range."""
        dist = {"g2": 4, "reddit": 3}
        score = _compute_evidence_confidence(7, dist)
        assert 0.3 < score < 0.8

    def test_returns_float_in_range(self):
        """Score should always be a float in [0.0, 1.0]."""
        for mentions in (0, 1, 5, 20, 100):
            for dist in ({}, {"g2": mentions}, {"reddit": 1, "g2": 2}):
                score = _compute_evidence_confidence(mentions, dist)
                assert isinstance(score, float)
                assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Company name normalization
# ---------------------------------------------------------------------------

from atlas_brain.services.company_normalization import normalize_company_name


class TestNormalizeCompanyName:
    def test_strips_inc(self):
        assert normalize_company_name("Acme Inc") == "acme"

    def test_strips_incorporated_dot(self):
        assert normalize_company_name("Acme Incorporated.") == "acme"

    def test_strips_llc(self):
        assert normalize_company_name("ACME LLC") == "acme"

    def test_empty_string(self):
        assert normalize_company_name("") == ""

    def test_preserves_multi_word(self):
        assert normalize_company_name("BrightPath Software") == "brightpath software"


@pytest.mark.asyncio
async def test_fetch_company_signal_review_context_filters_quarantined_reviews():
    pool = SimpleNamespace(fetch=AsyncMock(return_value=[]))

    result = await churn_intel_mod._fetch_company_signal_review_context(
        pool,
        [uuid4()],
    )

    assert result == {}
    query = pool.fetch.await_args.args[0]
    assert "enrichment_status = 'enriched'" in query


@pytest.mark.asyncio
async def test_head_to_head_from_edges_filters_quarantined_review_companies():
    sample_review_id = uuid4()
    pool = SimpleNamespace(
        fetch=AsyncMock(
            side_effect=[
                [{
                    "from_vendor": "CrowdStrike",
                    "to_vendor": "SentinelOne",
                    "mention_count": 2,
                    "sample_review_ids": [sample_review_id],
                }],
                [],
                [],
            ],
        ),
    )

    result = await churn_intel_mod._head_to_head_from_edges(
        pool,
        "CrowdStrike",
        "SentinelOne",
        window_days=None,
    )

    assert len(result) == 2
    review_query = pool.fetch.await_args_list[2].args[0]
    assert "enrichment_status = 'enriched'" in review_query
