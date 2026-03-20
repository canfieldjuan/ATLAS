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
    _build_battle_card_locked_facts,
    _build_deterministic_battle_card_competitive_landscape,
    _build_deterministic_battle_card_weakness_analysis,
    _build_scorecard_locked_facts,
    _build_deterministic_vendor_feed,
    _build_validated_executive_summary,
    _canonicalize_competitor,
    _compute_churn_pressure_score,
    _compute_evidence_confidence,
    _executive_source_list,
    _fallback_scorecard_expert_take,
    _sanitize_battle_card_sales_copy,
    _validate_battle_card_sales_copy,
    _validate_scorecard_expert_take,
    _validate_report,
)
from atlas_brain.autonomous.tasks.b2b_battle_cards import (
    _BATTLE_CARD_LLM_FIELDS,
    _battle_card_llm_options,
    _battle_card_prior_attempt,
    _persist_battle_card,
    _parse_battle_card_sales_copy,
    _update_execution_progress,
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

    def test_allows_numeric_claims_present_in_cross_vendor_input(self):
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


class _FakePersistPool:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    async def execute(self, *args: Any) -> None:
        self.calls.append(args)


class TestBattleCardPersistence:
    @pytest.mark.asyncio
    async def test_persist_battle_card_writes_deterministic_sections(self):
        pool = _FakePersistPool()
        card = _sample_battle_card()
        card["weakness_analysis"] = _build_deterministic_battle_card_weakness_analysis(card)
        card["competitive_landscape"] = _build_deterministic_battle_card_competitive_landscape(card)
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

    @pytest.mark.asyncio
    async def test_overlay_persist_keeps_deterministic_sections_on_failure(self):
        pool = _FakePersistPool()
        card = _sample_battle_card()
        card["weakness_analysis"] = _build_deterministic_battle_card_weakness_analysis(card)
        card["competitive_landscape"] = _build_deterministic_battle_card_competitive_landscape(card)
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
        assert pool.calls[0][8] == "pipeline_deterministic"


class TestBattleCardLlmRouting:
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

    def test_auto_backend_falls_back_to_model_default_when_blank(self):
        field = type("FieldInfo", (), {"default": "openai/o4-mini"})
        cfg_type = type("Cfg", (), {"model_fields": {"battle_card_openrouter_model": field}})
        cfg = cfg_type()
        cfg.battle_card_llm_backend = "auto"
        cfg.battle_card_openrouter_model = ""
        opts = _battle_card_llm_options(cfg)
        assert opts["openrouter_model"] == "openai/o4-mini"

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
            if len(coros) == 33:
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
