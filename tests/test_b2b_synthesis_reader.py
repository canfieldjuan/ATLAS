"""Tests for _b2b_synthesis_reader: load_best_reasoning_view, legacy fallback,
and consumer-required contract mapping."""

import sys
from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Pre-mock heavy deps before importing the module
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

from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
    SynthesisView,
    contract_gaps_for_consumer,
    legacy_reasoning_to_contracts,
    load_best_reasoning_view,
    load_best_reasoning_views,
    load_prior_reasoning_snapshots,
    load_synthesis_view,
    required_contracts_for_consumer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synthesis_row(
    vendor_name: str = "Acme",
    as_of_date: date | None = None,
    schema_version: str = "v2",
    synthesis: dict | None = None,
) -> dict[str, Any]:
    """Build a fake b2b_reasoning_synthesis row."""
    if as_of_date is None:
        as_of_date = date(2026, 3, 28)
    if synthesis is None:
        synthesis = {
            "reasoning_contracts": {
                "schema_version": "2.2",
                "vendor_core_reasoning": {
                    "causal_narrative": {
                        "primary_wedge": "price_squeeze",
                        "summary": "Pricing pressure is acute.",
                        "confidence": "high",
                        "key_signals": ["Price complaints up 40%"],
                        "what_would_weaken_thesis": [
                            {"condition": "Price cut announced", "signal_source": "", "monitorable": True}
                        ],
                        "data_gaps": [],
                    },
                    "segment_playbook": {"segments": [], "confidence": "medium"},
                    "timing_intelligence": {"confidence": "medium"},
                },
                "displacement_reasoning": {
                    "competitive_reframes": {"confidence": "medium"},
                    "migration_proof": {"confidence": "medium"},
                },
                "category_reasoning": {"market_regime": "consolidating"},
                "account_reasoning": {"total_accounts": 12},
            },
        }
    return {
        "vendor_name": vendor_name,
        "as_of_date": as_of_date,
        "schema_version": schema_version,
        "synthesis": synthesis,
    }


def _make_legacy_row(
    vendor_name: str = "Acme",
    archetype: str = "pricing_shock",
    archetype_confidence: float = 0.8,
    executive_summary: str = "Churn pressure from pricing.",
    key_signals: list | None = None,
    falsification: list | None = None,
    uncertainty: list | None = None,
) -> dict[str, Any]:
    """Build a fake b2b_churn_signals row."""
    return {
        "vendor_name": vendor_name,
        "archetype": archetype,
        "archetype_confidence": archetype_confidence,
        "falsification_conditions": falsification or ["Price cut announced"],
        "reasoning_executive_summary": executive_summary,
        "reasoning_key_signals": key_signals or ["Price complaints up 40%"],
        "reasoning_uncertainty_sources": uncertainty or ["Only 30 reviews"],
        "last_computed_at": datetime(2026, 3, 27, tzinfo=timezone.utc),
    }


def _mock_pool(
    synth_row: dict | None = None,
    legacy_row: dict | None = None,
    synth_rows: list[dict] | None = None,
    legacy_rows: list[dict] | None = None,
    packet_row: dict | None = None,
) -> AsyncMock:
    """Create a mock pool that returns the given rows for fetchrow/fetch."""
    pool = AsyncMock()

    async def _fetchrow(query: str, *args):
        if "b2b_reasoning_synthesis" in query:
            return synth_row
        if "b2b_vendor_reasoning_packets" in query:
            return packet_row
        if "b2b_churn_signals" in query:
            return legacy_row
        return None

    async def _fetch(query: str, *args):
        if "b2b_reasoning_synthesis" in query:
            return synth_rows or []
        if "b2b_churn_signals" in query:
            return legacy_rows or []
        return []

    pool.fetchrow = _fetchrow
    pool.fetch = _fetch
    return pool


# ---------------------------------------------------------------------------
# Tests: legacy_reasoning_to_contracts
# ---------------------------------------------------------------------------

class TestLegacyReasoningToContracts:
    def test_basic_conversion(self):
        legacy = _make_legacy_row()
        result = legacy_reasoning_to_contracts(legacy, vendor_name="Acme")

        assert result["schema_version"] == "legacy"
        assert result["_legacy_source"] == "b2b_churn_signals"

        contracts = result["reasoning_contracts"]
        assert contracts["schema_version"] == "legacy"

        vendor_core = contracts["vendor_core_reasoning"]
        assert vendor_core["schema_version"] == "legacy"

        cn = vendor_core["causal_narrative"]
        assert cn["primary_wedge"] == "price_squeeze"
        assert cn["confidence"] == "high"  # 0.8 >= 0.7
        assert cn["summary"] == "Churn pressure from pricing."
        assert cn["key_signals"] == ["Price complaints up 40%"]
        assert len(cn["what_would_weaken_thesis"]) == 1
        assert cn["data_gaps"] == ["Only 30 reviews"]

    def test_low_confidence(self):
        legacy = _make_legacy_row(archetype_confidence=0.3)
        result = legacy_reasoning_to_contracts(legacy, vendor_name="Acme")
        cn = result["reasoning_contracts"]["vendor_core_reasoning"]["causal_narrative"]
        assert cn["confidence"] == "low"

    def test_medium_confidence(self):
        legacy = _make_legacy_row(archetype_confidence=0.5)
        result = legacy_reasoning_to_contracts(legacy, vendor_name="Acme")
        cn = result["reasoning_contracts"]["vendor_core_reasoning"]["causal_narrative"]
        assert cn["confidence"] == "medium"

    def test_missing_confidence_defaults_non_empty_section_to_medium(self):
        view = load_synthesis_view(
            {
                "reasoning_contracts": {
                    "schema_version": "v1",
                    "account_reasoning": {
                        "market_summary": "Two accounts are actively evaluating alternatives.",
                        "top_accounts": [{"name": "Acme Corp"}],
                    },
                },
            },
            "Acme",
        )
        assert view.confidence("account_reasoning") == "medium"
        assert view.should_suppress("account_reasoning") is False

    def test_archetype_mapped_to_wedge(self):
        """pricing_shock -> price_squeeze, feature_gap -> feature_parity."""
        legacy_pricing = _make_legacy_row(archetype="pricing_shock")
        cn = legacy_reasoning_to_contracts(legacy_pricing)["reasoning_contracts"]["vendor_core_reasoning"]["causal_narrative"]
        assert cn["primary_wedge"] == "price_squeeze"
        assert cn["_legacy_archetype"] == "pricing_shock"

        legacy_feature = _make_legacy_row(archetype="feature_gap")
        cn2 = legacy_reasoning_to_contracts(legacy_feature)["reasoning_contracts"]["vendor_core_reasoning"]["causal_narrative"]
        assert cn2["primary_wedge"] == "feature_parity"
        assert cn2["_legacy_archetype"] == "feature_gap"

    def test_missing_fields_produce_minimal_output(self):
        legacy = {
            "archetype": "feature_gap",
            "archetype_confidence": None,
            "falsification_conditions": None,
            "reasoning_executive_summary": None,
            "reasoning_key_signals": None,
            "reasoning_uncertainty_sources": None,
        }
        result = legacy_reasoning_to_contracts(legacy, vendor_name="X")
        cn = result["reasoning_contracts"]["vendor_core_reasoning"]["causal_narrative"]
        assert cn["primary_wedge"] == "feature_parity"  # feature_gap -> feature_parity
        assert cn["confidence"] == "low"
        assert cn["data_gaps"] == []
        assert "summary" not in cn
        assert "key_signals" not in cn

    def test_json_string_fields_decoded(self):
        import json
        legacy = _make_legacy_row(
            key_signals=json.dumps(["signal_a", "signal_b"]),
            falsification=json.dumps(["cond_1"]),
            uncertainty=json.dumps(["small sample"]),
        )
        result = legacy_reasoning_to_contracts(legacy, vendor_name="Acme")
        cn = result["reasoning_contracts"]["vendor_core_reasoning"]["causal_narrative"]
        assert cn["key_signals"] == ["signal_a", "signal_b"]
        assert len(cn["what_would_weaken_thesis"]) == 1
        assert cn["data_gaps"] == ["small sample"]


# ---------------------------------------------------------------------------
# Tests: load_best_reasoning_view
# ---------------------------------------------------------------------------

class TestLoadBestReasoningView:
    @pytest.mark.asyncio
    async def test_prefers_synthesis_over_legacy(self):
        synth = _make_synthesis_row()
        legacy = _make_legacy_row()
        pool = _mock_pool(synth_row=synth, legacy_row=legacy)

        view = await load_best_reasoning_view(pool, "Acme")

        assert view is not None
        assert view.schema_version == "v2"
        assert view.primary_wedge is not None
        assert view.primary_wedge.value == "price_squeeze"


def test_consumer_context_surfaces_scope_manifest_atoms_and_delta():
    raw = {
        "reasoning_contracts": {
            "schema_version": "v1",
            "vendor_core_reasoning": {
                "causal_narrative": {
                    "primary_wedge": "price_squeeze",
                    "confidence": "high",
                },
            },
        },
        "scope_manifest": {
            "selection_strategy": "vendor_facet_packet_v1",
            "witnesses_in_scope": 8,
        },
        "reasoning_atoms": {
            "schema_version": "v1",
            "theses": [{"thesis_id": "primary_wedge"}],
            "timing_windows": [{"window_id": "trigger_1"}],
            "proof_points": [{"label": "switch_volume"}],
            "account_signals": [{"company": "Acme"}],
            "counterevidence": [{"counterevidence_id": "counterevidence_1"}],
            "coverage_limits": [{"coverage_limit_id": "limit_1"}],
        },
        "reasoning_delta": {
            "schema_version": "v1",
            "changed": True,
        },
    }
    view = load_synthesis_view(raw, "Acme")

    context = view.consumer_context("vendor_scorecard")

    assert context["scope_manifest"]["selection_strategy"] == "vendor_facet_packet_v1"
    assert context["reasoning_atoms"]["schema_version"] == "v1"
    assert context["theses"][0]["thesis_id"] == "primary_wedge"
    assert context["timing_windows"][0]["window_id"] == "trigger_1"
    assert context["reasoning_delta"]["changed"] is True

    @pytest.mark.asyncio
    async def test_falls_back_to_legacy(self):
        from atlas_brain.autonomous import visibility as visibility_mod

        legacy = _make_legacy_row()
        pool = _mock_pool(synth_row=None, legacy_row=legacy)
        emit = AsyncMock()
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(visibility_mod, "emit_event", emit)

        view = await load_best_reasoning_view(pool, "Acme", allow_legacy_fallback=True)

        assert view is not None
        assert view.schema_version == "legacy"
        cn = view.section("causal_narrative")
        assert cn["primary_wedge"] == "price_squeeze"
        assert cn["summary"] == "Churn pressure from pricing."
        emit.assert_awaited_once()
        assert emit.await_args.kwargs["reason_code"] == "legacy_reasoning_view_fallback"
        monkeypatch.undo()

    @pytest.mark.asyncio
    async def test_returns_none_when_no_data(self):
        pool = _mock_pool(synth_row=None, legacy_row=None)
        view = await load_best_reasoning_view(pool, "Acme")
        assert view is None

    @pytest.mark.asyncio
    async def test_default_does_not_fallback_to_legacy(self):
        legacy = _make_legacy_row()
        pool = _mock_pool(synth_row=None, legacy_row=legacy)

        view = await load_best_reasoning_view(pool, "Acme")

        assert view is None

    @pytest.mark.asyncio
    async def test_hydrates_packet_artifacts_from_packet_table_when_inline_missing(self):
        synth = _make_synthesis_row(
            synthesis={
                "reasoning_contracts": {
                    "vendor_core_reasoning": {
                        "causal_narrative": {
                            "primary_wedge": "price_squeeze",
                            "confidence": "high",
                            "summary": "Pricing pressure is acute.",
                        },
                    },
                },
            },
        )
        packet_row = {
            "packet": {
                "payload": {
                    "metric_ledger": [
                        {"label": "Price complaints", "_sid": "metric:price"},
                    ],
                    "witness_pack": [
                        {
                            "witness_id": "witness:1",
                            "_sid": "witness:1",
                            "excerpt_text": "Budget pressure at renewal.",
                        },
                    ],
                    "section_packets": {
                        "anchor_examples": {"common_pattern": ["witness:1"]},
                    },
                },
            },
        }
        synth["synthesis"]["reasoning_contracts"]["vendor_core_reasoning"]["causal_narrative"]["citations"] = [
            "metric:price",
            "witness:1",
        ]
        pool = _mock_pool(synth_row=synth, packet_row=packet_row)

        view = await load_best_reasoning_view(pool, "Acme")

        assert view is not None
        assert view.reference_ids == {
            "metric_ids": ["metric:price"],
            "witness_ids": ["witness:1"],
        }
        assert view.packet_artifacts["section_packets"]["anchor_examples"]["common_pattern"] == ["witness:1"]
        assert view.witness_pack[0]["witness_id"] == "witness:1"

    @pytest.mark.asyncio
    async def test_legacy_view_primary_wedge_resolves(self):
        """Legacy pricing_shock should resolve to price_squeeze Wedge via validate_wedge."""
        legacy = _make_legacy_row()
        pool = _mock_pool(synth_row=None, legacy_row=legacy)

        view = await load_best_reasoning_view(pool, "Acme", allow_legacy_fallback=True)
        assert view.primary_wedge is not None
        assert view.primary_wedge.value == "price_squeeze"
        assert view.wedge_label == "Price Squeeze"

    @pytest.mark.asyncio
    async def test_legacy_view_has_materialized_contracts(self):
        legacy = _make_legacy_row()
        pool = _mock_pool(synth_row=None, legacy_row=legacy)

        view = await load_best_reasoning_view(pool, "Acme", allow_legacy_fallback=True)
        contracts = view.materialized_contracts()
        assert "vendor_core_reasoning" in contracts

    @pytest.mark.asyncio
    async def test_legacy_view_falsification_conditions(self):
        legacy = _make_legacy_row(falsification=["Price cut", "Market shift"])
        pool = _mock_pool(synth_row=None, legacy_row=legacy)

        view = await load_best_reasoning_view(pool, "Acme", allow_legacy_fallback=True)
        fcs = view.falsification_conditions()
        assert len(fcs) == 2
        assert fcs[0]["condition"] == "Price cut"

    @pytest.mark.asyncio
    async def test_synthesis_json_string_decoded(self):
        """Synthesis stored as JSON string should be decoded."""
        import json
        synth = _make_synthesis_row()
        synth["synthesis"] = json.dumps(synth["synthesis"])
        pool = _mock_pool(synth_row=synth)

        view = await load_best_reasoning_view(pool, "Acme")
        assert view is not None
        assert view.schema_version == "v2"

    @pytest.mark.asyncio
    async def test_as_of_date_passed_through(self):
        synth = _make_synthesis_row(as_of_date=date(2026, 3, 20))
        pool = _mock_pool(synth_row=synth)

        view = await load_best_reasoning_view(pool, "Acme", as_of=date(2026, 3, 28))
        assert view.as_of_date == date(2026, 3, 20)


# ---------------------------------------------------------------------------
# Tests: load_best_reasoning_views (batch)
# ---------------------------------------------------------------------------

class TestLoadBestReasoningViews:
    @pytest.mark.asyncio
    async def test_batch_mixed_sources(self):
        """Vendor A from synthesis, Vendor B from legacy fallback."""
        from atlas_brain.autonomous import visibility as visibility_mod

        synth_rows = [_make_synthesis_row(vendor_name="VendorA")]
        legacy_rows = [_make_legacy_row(vendor_name="VendorB")]
        pool = _mock_pool(synth_rows=synth_rows, legacy_rows=legacy_rows)
        emit = AsyncMock()
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(visibility_mod, "emit_event", emit)

        views = await load_best_reasoning_views(pool, ["VendorA", "VendorB"], allow_legacy_fallback=True)

        assert "VendorA" in views
        assert views["VendorA"].schema_version == "v2"
        assert "VendorB" in views
        assert views["VendorB"].schema_version == "legacy"
        emit.assert_awaited_once()
        assert emit.await_args.kwargs["reason_code"] == "legacy_reasoning_batch_fallback"
        monkeypatch.undo()

    @pytest.mark.asyncio
    async def test_batch_empty(self):
        pool = _mock_pool()
        views = await load_best_reasoning_views(pool, [])
        assert views == {}

    @pytest.mark.asyncio
    async def test_synthesis_covers_all(self):
        """When synthesis covers all vendors, no legacy query needed."""
        synth_rows = [
            _make_synthesis_row(vendor_name="A"),
            _make_synthesis_row(vendor_name="B"),
        ]
        pool = _mock_pool(synth_rows=synth_rows, legacy_rows=[])

        views = await load_best_reasoning_views(pool, ["A", "B"])
        assert len(views) == 2
        assert all(v.schema_version == "v2" for v in views.values())

    @pytest.mark.asyncio
    async def test_batch_default_does_not_fallback_to_legacy(self):
        legacy_rows = [_make_legacy_row(vendor_name="VendorB")]
        pool = _mock_pool(synth_rows=[], legacy_rows=legacy_rows)

        views = await load_best_reasoning_views(pool, ["VendorB"])

        assert views == {}


class TestLoadPriorReasoningSnapshots:
    @pytest.mark.asyncio
    async def test_prefers_prior_synthesis_snapshot(self):
        synth_rows = [
            _make_synthesis_row(
                vendor_name="Acme",
                as_of_date=date(2026, 3, 27),
            ),
        ]
        pool = _mock_pool(synth_rows=synth_rows, legacy_rows=[])

        snapshots = await load_prior_reasoning_snapshots(
            pool,
            ["Acme"],
            before_date=date(2026, 3, 28),
        )

        assert snapshots["Acme"]["archetype"] == "price_squeeze"
        assert snapshots["Acme"]["confidence"] == 0.85
        assert snapshots["Acme"]["snapshot_date"] == "2026-03-27"

    @pytest.mark.asyncio
    async def test_falls_back_to_prior_legacy_snapshot(self):
        legacy_rows = [
            _make_legacy_row(
                vendor_name="Beta",
                archetype="feature_gap",
                archetype_confidence=0.5,
            ),
        ]
        pool = _mock_pool(synth_rows=[], legacy_rows=legacy_rows)

        snapshots = await load_prior_reasoning_snapshots(
            pool,
            ["Beta"],
            before_date=date(2026, 3, 28),
        )

        assert snapshots["Beta"]["archetype"] == "feature_gap"
        assert snapshots["Beta"]["confidence"] == 0.5
        assert snapshots["Beta"]["snapshot_date"] == "2026-03-27"


# ---------------------------------------------------------------------------
# Tests: consumer contract requirements
# ---------------------------------------------------------------------------

class TestConsumerContracts:
    def test_campaign_contracts(self):
        required = required_contracts_for_consumer("campaign")
        assert "vendor_core_reasoning" in required
        assert "displacement_reasoning" in required

    def test_blog_reranker_contracts(self):
        required = required_contracts_for_consumer("blog_reranker")
        assert "vendor_core_reasoning" in required
        assert "category_reasoning" in required

    def test_vendor_briefing_contracts(self):
        required = required_contracts_for_consumer("vendor_briefing")
        assert "vendor_core_reasoning" in required
        assert "displacement_reasoning" in required
        assert "account_reasoning" in required

    def test_unknown_consumer_returns_empty(self):
        required = required_contracts_for_consumer("nonexistent")
        assert required == ()

    def test_contract_gaps_with_legacy_view(self):
        """Legacy views only have vendor_core_reasoning, so other contracts
        should show as gaps."""
        legacy = _make_legacy_row()
        raw = legacy_reasoning_to_contracts(legacy, vendor_name="Acme")
        view = load_synthesis_view(raw, "Acme", schema_version="legacy")

        gaps = contract_gaps_for_consumer(view, "battle_card")
        # Legacy only provides vendor_core_reasoning, so displacement/category/account
        # should be gaps.
        assert "displacement_reasoning" in gaps
        assert "category_reasoning" in gaps
        assert "account_reasoning" in gaps
        assert "vendor_core_reasoning" not in gaps

    def test_contract_gaps_with_full_synthesis(self):
        synth = _make_synthesis_row()
        view = load_synthesis_view(
            synth["synthesis"], "Acme", schema_version="v2",
        )
        gaps = contract_gaps_for_consumer(view, "battle_card")
        assert gaps == []

    def test_consumer_context_resolves_anchor_examples_and_reference_ids(self):
        synth = _make_synthesis_row(
            synthesis={
                "reasoning_contracts": {
                    "schema_version": "2.2",
                    "vendor_core_reasoning": {
                        "causal_narrative": {
                            "primary_wedge": "price_squeeze",
                            "confidence": "high",
                        },
                    },
                    "displacement_reasoning": {
                        "migration_proof": {"confidence": "medium"},
                    },
                    "category_reasoning": {"market_regime": "consolidating"},
                    "account_reasoning": {"total_accounts": 12},
                },
                "reference_ids": {
                    "metric_ids": ["vault:metric:total_reviews"],
                    "witness_ids": ["witness:r1:0"],
                },
                "packet_artifacts": {
                    "witness_pack": [
                        {
                            "witness_id": "witness:r1:0",
                            "_sid": "witness:r1:0",
                            "excerpt_text": "Hack Club said the renewal jumped to $200k/year.",
                            "reviewer_company": "Hack Club",
                            "time_anchor": "Q2 renewal",
                            "salience_score": 9.4,
                        },
                    ],
                    "section_packets": {
                        "anchor_examples": {
                            "outlier_or_named_account": ["witness:r1:0"],
                        },
                    },
                },
            },
        )
        view = load_synthesis_view(synth["synthesis"], "Acme", schema_version="v2")

        context = view.consumer_context("battle_card")

        assert context["reference_ids"]["witness_ids"] == ["witness:r1:0"]
        assert context["anchor_examples"]["outlier_or_named_account"][0]["reviewer_company"] == "Hack Club"
        assert context["witness_highlights"][0]["witness_id"] == "witness:r1:0"


# ---------------------------------------------------------------------------
# Tests: synthesis_view_to_reasoning_entry and build_reasoning_lookup_from_views
# ---------------------------------------------------------------------------

from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
    build_reasoning_lookup_from_views,
    synthesis_view_to_reasoning_entry,
)


class TestSynthesisViewToReasoningEntry:
    def test_synthesis_view_produces_valid_entry(self):
        synth = _make_synthesis_row()
        view = load_synthesis_view(synth["synthesis"], "Acme", schema_version="v2")

        entry = synthesis_view_to_reasoning_entry(view)

        assert entry["archetype"] == "price_squeeze"
        assert entry["mode"] == "synthesis"
        assert entry["confidence"] == 0.85  # high -> 0.85
        assert entry["executive_summary"] == "Pricing pressure is acute."
        assert entry["key_signals"] == ["Price complaints up 40%"]
        assert len(entry["falsification_conditions"]) == 1
        assert entry["risk_level"] == "high"

    def test_legacy_view_produces_valid_entry(self):
        legacy = _make_legacy_row()
        raw = legacy_reasoning_to_contracts(legacy, vendor_name="Acme")
        view = load_synthesis_view(raw, "Acme", schema_version="legacy")

        entry = synthesis_view_to_reasoning_entry(view)

        assert entry["archetype"] == "price_squeeze"  # mapped from pricing_shock
        assert entry["mode"] == "synthesis_fallback"
        assert entry["executive_summary"] == "Churn pressure from pricing."

    def test_build_reasoning_lookup_from_views(self):
        synth_a = _make_synthesis_row(vendor_name="VendorA")
        synth_b = _make_synthesis_row(vendor_name="VendorB")
        views = {
            "VendorA": load_synthesis_view(synth_a["synthesis"], "VendorA", schema_version="v2"),
            "VendorB": load_synthesis_view(synth_b["synthesis"], "VendorB", schema_version="v2"),
        }

        lookup = build_reasoning_lookup_from_views(views)

        assert "VendorA" in lookup
        assert "VendorB" in lookup
        assert lookup["VendorA"]["archetype"] == "price_squeeze"
        assert lookup["VendorB"]["mode"] == "synthesis"

    def test_synthesis_overrides_legacy_in_merged_lookup(self):
        """Verify that when synthesis and legacy both exist for a vendor,
        the synthesis entry wins in a merged reasoning_lookup."""
        synth = _make_synthesis_row(vendor_name="Acme")
        synth_view = load_synthesis_view(synth["synthesis"], "Acme", schema_version="v2")
        synth_lookup = build_reasoning_lookup_from_views({"Acme": synth_view})

        legacy_lookup = {
            "Acme": {
                "archetype": "feature_gap",
                "confidence": 0.4,
                "mode": "",
                "risk_level": "low",
                "executive_summary": "Old legacy summary",
                "key_signals": [],
                "falsification_conditions": [],
                "uncertainty_sources": [],
            }
        }

        # Synthesis wins via {**legacy, **synth} merge pattern
        merged = {**legacy_lookup, **synth_lookup}
        assert merged["Acme"]["archetype"] == "price_squeeze"
        assert merged["Acme"]["mode"] == "synthesis"
        assert merged["Acme"]["executive_summary"] == "Pricing pressure is acute."

    def test_legacy_fills_gaps_in_merged_lookup(self):
        """Vendors without synthesis still get legacy reasoning."""
        synth_lookup = build_reasoning_lookup_from_views({})

        legacy_lookup = {
            "OnlyLegacy": {
                "archetype": "pricing_shock",
                "confidence": 0.7,
                "mode": "",
                "executive_summary": "Legacy only",
                "key_signals": [],
                "falsification_conditions": [],
                "uncertainty_sources": [],
                "risk_level": "high",
            }
        }

        merged = {**legacy_lookup, **synth_lookup}
        assert "OnlyLegacy" in merged
        assert merged["OnlyLegacy"]["archetype"] == "pricing_shock"


# ---------------------------------------------------------------------------
# Tests: downstream adoption - vendor briefing uses loader
# ---------------------------------------------------------------------------

class TestVendorBriefingAdoption:
    @pytest.mark.asyncio
    async def test_briefing_gets_mapped_wedge_from_legacy(self):
        """When only legacy exists, vendor briefing archetype should be a valid wedge."""
        legacy = _make_legacy_row(vendor_name="TestVendor", archetype="support_collapse")
        pool = _mock_pool(synth_row=None, legacy_row=legacy)

        view = await load_best_reasoning_view(pool, "TestVendor", allow_legacy_fallback=True)
        assert view is not None

        # Simulate what vendor briefing does
        wedge = view.primary_wedge
        cn = view.section("causal_narrative")
        archetype = wedge.value if wedge else cn.get("primary_wedge", "")

        # support_collapse maps to support_erosion
        assert archetype == "support_erosion"

    @pytest.mark.asyncio
    async def test_briefing_gets_synthesis_wedge(self):
        """When synthesis exists, vendor briefing archetype comes from synthesis."""
        synth = _make_synthesis_row(vendor_name="TestVendor")
        pool = _mock_pool(synth_row=synth)

        view = await load_best_reasoning_view(pool, "TestVendor")
        assert view is not None
        assert view.primary_wedge is not None
        assert view.primary_wedge.value == "price_squeeze"
