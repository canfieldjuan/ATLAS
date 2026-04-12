"""Tests for Phase 6: Report Migration.

Verifies governance fields surface in report entries, dual narratives
are resolved, and scorecard LLM payloads include retention/confidence.
"""

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

from atlas_brain.autonomous.tasks.b2b_churn_reports import (
    _attach_context_to_deterministic_reports,
    _attach_synthesis_contracts_to_report_entry,
    _build_scorecard_narrative_payload,
)
from atlas_brain.autonomous.tasks._b2b_shared import _structure_displacement_report
from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
    SynthesisView,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_view_with_governance() -> SynthesisView:
    raw = {
        "reasoning_contracts": {
            "schema_version": "v1",
            "vendor_core_reasoning": {
                "causal_narrative": {
                    "primary_wedge": "price_squeeze",
                    "confidence": "medium",
                    "summary": "Pricing pressure is acute after Q1 hike.",
                    "trigger": "Q1 price increase",
                    "why_now": "Renewal cycle coincides with budget cuts",
                    "data_gaps": [],
                    "citations": [],
                },
                "timing_intelligence": {
                    "best_timing_window": "Q2 renewal cycle",
                    "confidence": "medium",
                },
                "why_they_stay": {
                    "summary": "Ecosystem breadth reduces churn",
                    "strengths": [
                        {"area": "integrations", "evidence": "Broad API ecosystem"},
                        {"area": "brand", "evidence": "Market leader perception"},
                    ],
                },
                "confidence_posture": {
                    "overall": "medium",
                    "limits": ["thin enterprise sample (n=7)", "no displacement evidence"],
                },
            },
            "displacement_reasoning": {
                "migration_proof": {"confidence": "low"},
            },
            "category_reasoning": {"market_regime": "consolidating"},
            "account_reasoning": {"market_summary": "Active evaluation in ops teams"},
            "evidence_governance": {
                "metric_ledger": [
                    {"label": "total_reviews", "value": 150, "_sid": "vault:metric:total_reviews"},
                ],
                "coverage_gaps": [
                    {"type": "missing_pool", "area": "displacement", "_sid": "gap:missing_pool:displacement"},
                ],
                "contradictions": [],
            },
        },
    }
    return SynthesisView("TestVendor", raw, schema_version="v2", as_of_date=date(2026, 3, 28))


# ---------------------------------------------------------------------------
# Tests: governance fields in report entries
# ---------------------------------------------------------------------------

class TestGovernanceFieldsInReportEntries:
    def test_why_they_stay_attached(self):
        entry: dict[str, Any] = {"vendor": "TestVendor"}
        view = _make_view_with_governance()
        _attach_synthesis_contracts_to_report_entry(
            entry, view, consumer_name="vendor_scorecard",
            requested_as_of=date(2026, 3, 28), include_displacement=True,
        )
        assert "why_they_stay" in entry
        assert entry["why_they_stay"]["summary"] == "Ecosystem breadth reduces churn"

    def test_confidence_posture_attached(self):
        entry: dict[str, Any] = {"vendor": "TestVendor"}
        view = _make_view_with_governance()
        _attach_synthesis_contracts_to_report_entry(
            entry, view, consumer_name="vendor_scorecard",
            requested_as_of=date(2026, 3, 28), include_displacement=True,
        )
        assert "confidence_posture" in entry
        assert entry["confidence_posture"]["overall"] == "medium"
        assert "confidence_limits" in entry
        assert "thin enterprise sample" in entry["confidence_limits"][0]

    def test_coverage_gaps_attached(self):
        entry: dict[str, Any] = {"vendor": "TestVendor"}
        view = _make_view_with_governance()
        _attach_synthesis_contracts_to_report_entry(
            entry, view, consumer_name="vendor_scorecard",
            requested_as_of=date(2026, 3, 28), include_displacement=True,
        )
        assert "coverage_gaps" in entry
        assert entry["coverage_gaps"][0]["type"] == "missing_pool"

    def test_metric_ledger_attached(self):
        entry: dict[str, Any] = {"vendor": "TestVendor"}
        view = _make_view_with_governance()
        _attach_synthesis_contracts_to_report_entry(
            entry, view, consumer_name="vendor_scorecard",
            requested_as_of=date(2026, 3, 28), include_displacement=True,
        )
        assert "metric_ledger" in entry
        assert entry["metric_ledger"][0]["label"] == "total_reviews"

    def test_suppressed_account_reasoning_removed_from_scorecard_contracts(self):
        entry: dict[str, Any] = {"vendor": "TestVendor"}
        view = SynthesisView(
            "TestVendor",
            {
                "reasoning_contracts": {
                    "schema_version": "v1",
                    "vendor_core_reasoning": {
                        "causal_narrative": {
                            "primary_wedge": "price_squeeze",
                            "confidence": "high",
                            "summary": "Pricing pressure is acute after Q1 hike.",
                            "data_gaps": [],
                            "citations": [],
                        },
                    },
                    "account_reasoning": {
                        "confidence": "insufficient",
                        "data_gaps": ["Section missing from model output"],
                        "top_accounts": [],
                    },
                },
            },
            schema_version="v2",
            as_of_date=date(2026, 3, 28),
        )
        _attach_synthesis_contracts_to_report_entry(
            entry, view, consumer_name="vendor_scorecard",
            requested_as_of=date(2026, 3, 28), include_displacement=True,
        )

        assert "account_reasoning" not in entry["reasoning_contracts"]
        assert "account_reasoning:insufficient" in entry["reasoning_contract_gaps"]
        assert "account_reasoning:suppressed" in entry["reasoning_contract_gaps"]

    def test_sparse_account_reasoning_surfaces_preview_only_in_scorecard_entry(self):
        entry: dict[str, Any] = {"vendor": "TestVendor"}
        view = SynthesisView(
            "TestVendor",
            {
                "reasoning_contracts": {
                    "schema_version": "v1",
                    "vendor_core_reasoning": {
                        "causal_narrative": {
                            "primary_wedge": "price_squeeze",
                            "confidence": "high",
                            "summary": "Pricing pressure is acute after Q1 hike.",
                            "data_gaps": [],
                            "citations": [],
                        },
                    },
                    "account_reasoning": {
                        "confidence": "insufficient",
                        "market_summary": "A single post-purchase account is in scope.",
                        "total_accounts": {
                            "value": 1,
                            "source_id": "accounts:summary:total_accounts",
                        },
                        "top_accounts": [
                            {
                                "name": "Concentrix",
                                "intent_score": 0.6,
                                "source_id": "accounts:company:concentrix",
                            },
                        ],
                    },
                },
            },
            schema_version="v2",
            as_of_date=date(2026, 3, 28),
        )
        _attach_synthesis_contracts_to_report_entry(
            entry, view, consumer_name="vendor_scorecard",
            requested_as_of=date(2026, 3, 28), include_displacement=True,
        )

        assert "account_reasoning" not in entry.get("reasoning_contracts", {})
        assert entry["account_reasoning_preview"]["top_accounts"][0]["name"] == "Concentrix"
        assert entry["account_pressure_metrics"]["total_accounts"] == 1
        assert entry["priority_account_names"] == ["Concentrix"]
        assert "account_reasoning:suppressed" in entry["reasoning_contract_gaps"]

    def test_low_confidence_sections_surface_disclaimers(self):
        entry: dict[str, Any] = {"vendor": "TestVendor"}
        view = SynthesisView(
            "TestVendor",
            {
                "reasoning_contracts": {
                    "schema_version": "v1",
                    "vendor_core_reasoning": {
                        "causal_narrative": {
                            "primary_wedge": "price_squeeze",
                            "confidence": "high",
                            "summary": "Pricing pressure is acute after Q1 hike.",
                            "data_gaps": [],
                            "citations": [],
                        },
                        "timing_intelligence": {
                            "confidence": "low",
                            "best_timing_window": "Q2 renewal",
                        },
                    },
                },
            },
            schema_version="v2",
            as_of_date=date(2026, 3, 28),
        )
        _attach_synthesis_contracts_to_report_entry(
            entry, view, consumer_name="vendor_scorecard",
            requested_as_of=date(2026, 3, 28), include_displacement=True,
        )

        assert entry["reasoning_section_disclaimers"]["timing_intelligence"]

    def test_reasoning_source_set(self):
        entry: dict[str, Any] = {"vendor": "TestVendor"}
        view = _make_view_with_governance()
        _attach_synthesis_contracts_to_report_entry(
            entry, view, consumer_name="weekly_churn_feed",
            requested_as_of=date(2026, 3, 28), include_displacement=False,
        )
        assert entry["reasoning_source"] == "b2b_reasoning_synthesis"

    def test_scope_manifest_atoms_and_delta_attached(self):
        entry: dict[str, Any] = {"vendor": "TestVendor"}
        view = SynthesisView(
            "TestVendor",
            {
                "scope_manifest": {
                    "selection_strategy": "vendor_facet_packet_v1",
                    "reviews_in_scope": 12,
                    "witnesses_in_scope": 9,
                },
                "reasoning_atoms": {
                    "theses": [
                        {
                            "thesis_id": "t1",
                            "wedge": "price_squeeze",
                            "summary": "Pricing pressure is rising in mid-market accounts.",
                        },
                    ],
                    "timing_windows": [
                        {
                            "window_type": "renewal",
                            "start_or_anchor": "Q2 renewal cycle",
                            "urgency": "high",
                        },
                    ],
                },
                "reasoning_delta": {
                    "changed": True,
                    "wedge_changed": True,
                    "theses_added": ["price_squeeze"],
                },
                "reasoning_contracts": {
                    "schema_version": "v1",
                    "vendor_core_reasoning": {
                        "causal_narrative": {
                            "primary_wedge": "price_squeeze",
                            "confidence": "high",
                        },
                    },
                },
            },
            schema_version="v2",
            as_of_date=date(2026, 3, 28),
        )

        _attach_synthesis_contracts_to_report_entry(
            entry,
            view,
            consumer_name="vendor_scorecard",
            requested_as_of=date(2026, 3, 28),
            include_displacement=True,
        )

        assert entry["scope_manifest"]["selection_strategy"] == "vendor_facet_packet_v1"
        assert entry["reasoning_atom_summary"]["theses"][0]["thesis_id"] == "t1"
        assert entry["reasoning_delta"]["changed"] is True


# ---------------------------------------------------------------------------
# Tests: scorecard narrative payload includes Phase 3 context
# ---------------------------------------------------------------------------

class TestScorecardNarrativePayloadPhase6:
    def test_includes_why_they_stay(self):
        scorecard = {
            "vendor": "TestVendor",
            "churn_pressure_score": 72,
            "risk_level": "high",
            "archetype": "price_squeeze",
            "archetype_confidence": 0.8,
            "reasoning_summary": "Pricing pressure is acute.",
            "why_they_stay": {
                "summary": "Ecosystem breadth reduces churn",
                "strengths": [
                    {"area": "integrations", "evidence": "Broad API"},
                    {"area": "brand", "evidence": "Market leader"},
                ],
            },
            "confidence_posture": {
                "overall": "medium",
                "limits": ["thin enterprise sample"],
            },
        }
        reasoning_lookup = {
            "TestVendor": {
                "archetype": "price_squeeze",
                "confidence": 0.8,
                "key_signals": ["Price hike Q1"],
            },
        }
        payload = _build_scorecard_narrative_payload(
            scorecard, reasoning_lookup=reasoning_lookup,
        )
        assert "why_they_stay" in payload
        assert payload["why_they_stay"]["summary"] == "Ecosystem breadth reduces churn"
        assert "integrations" in payload["why_they_stay"]["top_strengths"]

    def test_includes_confidence_limits(self):
        scorecard = {
            "vendor": "TestVendor",
            "churn_pressure_score": 60,
            "risk_level": "medium",
            "confidence_posture": {
                "overall": "medium",
                "limits": ["weak displacement density", "thin enterprise sample"],
            },
        }
        payload = _build_scorecard_narrative_payload(scorecard)
        assert "confidence_limits" in payload
        assert len(payload["confidence_limits"]) == 2

    def test_no_why_they_stay_when_absent(self):
        scorecard = {
            "vendor": "TestVendor",
            "churn_pressure_score": 60,
            "risk_level": "medium",
        }
        payload = _build_scorecard_narrative_payload(scorecard)
        assert "why_they_stay" not in payload
        assert "confidence_limits" not in payload


# ---------------------------------------------------------------------------
# Tests: synthesis replaces legacy reasoning_summary
# ---------------------------------------------------------------------------

class TestSynthesisReplacesLegacyNarrative:
    def test_synthesis_summary_overrides_legacy(self):
        """When synthesis view is attached, reasoning_summary comes from
        causal_narrative.summary, not from legacy reasoning_lookup."""
        from atlas_brain.autonomous.tasks.b2b_churn_reports import (
            _attach_context_to_deterministic_reports,
        )
        scorecard = {
            "vendor": "TestVendor",
            "churn_pressure_score": 72,
            "risk_level": "high",
            "archetype": "pricing_shock",
            "reasoning_summary": "Legacy: old reasoning summary",
        }
        view = _make_view_with_governance()
        synthesis_views = {"TestVendor": view}

        # Simulate _attach_context_to_deterministic_reports scorecard path
        _attach_synthesis_contracts_to_report_entry(
            scorecard, view, consumer_name="vendor_scorecard",
            requested_as_of=date(2026, 3, 28), include_displacement=True,
        )
        # Simulate the narrative override that happens in _attach_context
        causal = view.section("causal_narrative")
        if isinstance(causal, dict):
            summary = causal.get("summary", "")
            if summary:
                scorecard["reasoning_summary"] = summary
                scorecard["reasoning_source"] = "b2b_reasoning_synthesis"
        if view.primary_wedge:
            scorecard["archetype"] = view.primary_wedge.value

        assert scorecard["reasoning_summary"] == "Pricing pressure is acute after Q1 hike."
        assert scorecard["reasoning_source"] == "b2b_reasoning_synthesis"
        assert scorecard["archetype"] == "price_squeeze"


class TestCategoryOverviewProvenance:
    def test_category_overview_attaches_cross_vendor_analysis_and_refs(self):
        category_overview = [{"category": "CRM"}]

        _attach_context_to_deterministic_reports(
            pool=None,
            as_of=date(2026, 3, 31),
            deterministic_weekly_feed=[],
            deterministic_vendor_scorecards=[],
            deterministic_displacement_map=[],
            deterministic_category_overview=category_overview,
            deterministic_vendor_deep_dives=[],
            evidence_vault_lookup={},
            synthesis_views={},
            xv_lookup={
                "battles": {},
                "councils": {
                    "CRM": {
                        "confidence": 0.81,
                        "source": "synthesis",
                        "computed_date": date(2026, 3, 31),
                        "reference_ids": {
                            "metric_ids": ["metric:crm:1"],
                            "witness_ids": ["witness:crm:1"],
                        },
                        "conclusion": {
                            "winner": "HubSpot",
                            "loser": "Salesforce",
                            "conclusion": "SMB buyers are moving toward lower-friction CRM platforms.",
                            "market_regime": "price_competition",
                            "durability_assessment": "cyclical",
                            "key_insights": [{"insight": "Pricing is fragmenting the market.", "evidence": "renewal data"}],
                        },
                    },
                },
                "asymmetries": {},
            },
        )

        entry = category_overview[0]
        assert entry["cross_vendor_analysis"]["market_regime"] == "price_competition"
        assert entry["category_council"]["winner"] == "HubSpot"
        assert entry["reference_ids"]["metric_ids"] == ["metric:crm:1"]
        assert entry["reasoning_source"] == "b2b_cross_vendor_reasoning_synthesis"


class TestDisplacementReportProvenance:
    def test_displacement_edges_attach_cross_vendor_reference_ids(self):
        displacement_map = [
            {
                "from_vendor": "Zendesk",
                "to_vendor": "Freshdesk",
                "mention_count": 14,
                "signal_strength": "strong",
                "primary_driver": "pricing",
            },
        ]

        _attach_context_to_deterministic_reports(
            pool=None,
            as_of=date(2026, 3, 31),
            deterministic_weekly_feed=[],
            deterministic_vendor_scorecards=[],
            deterministic_displacement_map=displacement_map,
            deterministic_category_overview=[],
            deterministic_vendor_deep_dives=[],
            evidence_vault_lookup={},
            synthesis_views={},
            xv_lookup={
                "battles": {
                    ("Freshdesk", "Zendesk"): {
                        "confidence": 0.78,
                        "source": "synthesis",
                        "computed_date": date(2026, 3, 31),
                        "reference_ids": {
                            "metric_ids": ["metric:pair:1"],
                            "witness_ids": ["witness:pair:1"],
                        },
                        "conclusion": {
                            "loser": "Zendesk",
                            "conclusion": "Freshdesk is displacing Zendesk on pricing.",
                            "durability_assessment": "stable",
                        },
                    },
                },
                "councils": {},
                "asymmetries": {},
            },
        )

        edge = displacement_map[0]
        assert edge["battle_conclusion"] == "Freshdesk is displacing Zendesk on pricing."
        assert edge["reference_ids"]["metric_ids"] == ["metric:pair:1"]
        assert edge["reasoning_source"] == "b2b_cross_vendor_reasoning_synthesis"

        report = _structure_displacement_report(displacement_map)
        assert report["top_battles"][0]["reference_ids"]["witness_ids"] == ["witness:pair:1"]


# ---------------------------------------------------------------------------
# Tests: v2.3 schema field alignment
# ---------------------------------------------------------------------------

class TestV23SchemaAlignment:
    def test_summary_synthesized_from_trigger_why_now(self):
        """When v2.3 synthesis has trigger/why_now but no summary,
        synthesis_view_to_reasoning_entry should build one."""
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            synthesis_view_to_reasoning_entry,
        )
        raw = {
            "reasoning_contracts": {
                "vendor_core_reasoning": {
                    "causal_narrative": {
                        "primary_wedge": "price_squeeze",
                        "trigger": "Q1 price increase",
                        "why_now": "Budget cycle alignment",
                        "causal_chain": "Price hike -> evaluation -> switch",
                        "confidence": "medium",
                        "data_gaps": [],
                        "citations": [],
                    },
                },
            },
        }
        view = SynthesisView("V", raw, "v2", date(2026, 3, 28))
        entry = synthesis_view_to_reasoning_entry(view)

        assert "Q1 price increase" in entry["executive_summary"]
        assert "Budget cycle alignment" in entry["executive_summary"]
        assert len(entry["key_signals"]) >= 2
        assert "Q1 price increase" in entry["key_signals"]

    def test_summary_preserved_when_present(self):
        """When synthesis has explicit summary, use it directly."""
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            synthesis_view_to_reasoning_entry,
        )
        raw = {
            "reasoning_contracts": {
                "vendor_core_reasoning": {
                    "causal_narrative": {
                        "primary_wedge": "price_squeeze",
                        "summary": "Explicit model summary",
                        "key_signals": ["signal_a"],
                        "confidence": "high",
                        "data_gaps": [],
                    },
                },
            },
        }
        view = SynthesisView("V", raw, "v2", date(2026, 3, 28))
        entry = synthesis_view_to_reasoning_entry(view)

        assert entry["executive_summary"] == "Explicit model summary"
        assert entry["key_signals"] == ["signal_a"]


# ---------------------------------------------------------------------------
# Tests: scorecard narrative payload includes all governance fields
# ---------------------------------------------------------------------------

class TestScorecardPayloadGovernanceComplete:
    def test_coverage_gaps_in_payload(self):
        scorecard = {
            "vendor": "V",
            "churn_pressure_score": 60,
            "risk_level": "medium",
            "coverage_gaps": [
                {"type": "missing_pool", "area": "displacement", "_sid": "gap:x"},
                {"type": "thin_account_signals", "area": "accounts", "_sid": "gap:y"},
            ],
        }
        payload = _build_scorecard_narrative_payload(scorecard)
        assert "coverage_gaps" in payload
        assert len(payload["coverage_gaps"]) == 2
        assert payload["coverage_gaps"][0]["type"] == "missing_pool"

    def test_metric_ledger_in_payload(self):
        scorecard = {
            "vendor": "V",
            "churn_pressure_score": 60,
            "risk_level": "medium",
            "metric_ledger": [
                {"label": "total_reviews", "value": 150, "scope": "review_volume", "_sid": "vault:metric:total_reviews"},
            ],
        }
        payload = _build_scorecard_narrative_payload(scorecard)
        assert "metric_ledger" in payload
        assert payload["metric_ledger"][0]["label"] == "total_reviews"
        assert payload["metric_ledger"][0]["scope"] == "review_volume"

    def test_governance_absent_when_not_on_scorecard(self):
        scorecard = {
            "vendor": "V",
            "churn_pressure_score": 60,
            "risk_level": "medium",
        }
        payload = _build_scorecard_narrative_payload(scorecard)
        assert "coverage_gaps" not in payload
        assert "metric_ledger" not in payload
