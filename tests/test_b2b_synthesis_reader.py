"""Tests for synthesis-only reasoning readers and consumer contracts."""

import sys
from datetime import date
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
    discover_reasoning_vendor_names,
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


def _mock_pool(
    synth_row: dict | None = None,
    synth_rows: list[dict] | None = None,
    packet_row: dict | None = None,
) -> AsyncMock:
    """Create a mock pool that returns the given synthesis rows."""
    pool = AsyncMock()

    async def _fetchrow(query: str, *args):
        if "b2b_reasoning_synthesis" in query:
            return synth_row
        if "b2b_vendor_reasoning_packets" in query:
            return packet_row
        return None

    async def _fetch(query: str, *args):
        if "b2b_reasoning_synthesis" in query:
            return synth_rows or []
        return []

    pool.fetchrow = _fetchrow
    pool.fetch = _fetch
    return pool


# ---------------------------------------------------------------------------
# Tests: load_best_reasoning_view
# ---------------------------------------------------------------------------

class TestLoadBestReasoningView:
    @pytest.mark.asyncio
    async def test_returns_synthesis_view(self):
        synth = _make_synthesis_row()
        pool = _mock_pool(synth_row=synth)

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


def test_filtered_consumer_context_surfaces_sparse_account_preview():
    raw = {
        "reasoning_contracts": {
            "schema_version": "v2",
            "vendor_core_reasoning": {
                "causal_narrative": {
                    "primary_wedge": "price_squeeze",
                    "confidence": "high",
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
    }
    view = load_synthesis_view(raw, "Salesforce", schema_version="v2")

    context = view.filtered_consumer_context("vendor_scorecard")

    assert "account_reasoning" not in context["reasoning_contracts"]
    assert "account_reasoning:insufficient" in context["reasoning_contract_gaps"]
    assert "account_reasoning:suppressed" in context["reasoning_contract_gaps"]
    assert context["account_reasoning_preview"]["account_pressure_metrics"]["total_accounts"] == 1
    assert context["account_reasoning_preview"]["priority_account_names"] == ["Concentrix"]
    assert (
        context["account_reasoning_preview"]["account_reasoning"]["top_accounts"][0]["name"]
        == "Concentrix"
    )
    assert context["reasoning_section_disclaimers"]["account_reasoning"]


@pytest.mark.asyncio
async def test_returns_none_when_no_data():
    pool = _mock_pool(synth_row=None)
    view = await load_best_reasoning_view(pool, "Acme")
    assert view is None


class TestLoadBestReasoningViewPackets:
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
    async def test_prefers_prompt_payload_from_packet_table_when_present(self):
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
                    "section_packets": {
                        "anchor_examples": {"common_pattern": ["witness:full"]},
                    },
                },
                "prompt_payload": {
                    "witness_pack": [
                        {
                            "_sid": "witness:prompt",
                            "excerpt_text": "Prompt witness.",
                        },
                    ],
                    "section_packets": {
                        "anchor_examples": {"common_pattern": ["witness:prompt"]},
                    },
                },
            },
        }
        synth["synthesis"]["reasoning_contracts"]["vendor_core_reasoning"]["causal_narrative"]["citations"] = [
            "witness:prompt",
        ]
        pool = _mock_pool(synth_row=synth, packet_row=packet_row)

        view = await load_best_reasoning_view(pool, "Acme")

        assert view is not None
        assert view.packet_artifacts["section_packets"]["anchor_examples"]["common_pattern"] == ["witness:prompt"]
        assert view.witness_pack[0]["witness_id"] == "witness:prompt"

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
    async def test_batch_empty(self):
        pool = _mock_pool()
        views = await load_best_reasoning_views(pool, [])
        assert views == {}

    @pytest.mark.asyncio
    async def test_synthesis_covers_all(self):
        """Batch loading returns only synthesis rows."""
        synth_rows = [
            _make_synthesis_row(vendor_name="A"),
            _make_synthesis_row(vendor_name="B"),
        ]
        pool = _mock_pool(synth_rows=synth_rows)

        views = await load_best_reasoning_views(pool, ["A", "B"])
        assert len(views) == 2
        assert all(v.schema_version == "v2" for v in views.values())

@pytest.mark.asyncio
async def test_discover_reasoning_vendor_names_returns_synthesis_vendors():
    pool = _mock_pool(
        synth_rows=[{"vendor_name": "VendorA"}],
    )

    vendor_names = await discover_reasoning_vendor_names(
        pool,
        as_of=date(2026, 4, 7),
    )

    assert vendor_names == ["VendorA"]


class TestLoadPriorReasoningSnapshots:
    @pytest.mark.asyncio
    async def test_prefers_prior_synthesis_snapshot(self):
        synth_rows = [
            _make_synthesis_row(
                vendor_name="Acme",
                as_of_date=date(2026, 3, 27),
            ),
        ]
        pool = _mock_pool(synth_rows=synth_rows)

        snapshots = await load_prior_reasoning_snapshots(
            pool,
            ["Acme"],
            before_date=date(2026, 3, 28),
        )

        assert snapshots["Acme"]["archetype"] == "price_squeeze"
        assert snapshots["Acme"]["confidence"] == 0.85
        assert snapshots["Acme"]["snapshot_date"] == "2026-03-27"

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
        """Synthesis entries win when merged into an older reasoning lookup."""
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

# ---------------------------------------------------------------------------
# Tests: downstream adoption - vendor briefing uses loader
# ---------------------------------------------------------------------------

class TestVendorBriefingAdoption:
    @pytest.mark.asyncio
    async def test_briefing_gets_synthesis_wedge(self):
        """When synthesis exists, vendor briefing archetype comes from synthesis."""
        synth = _make_synthesis_row(vendor_name="TestVendor")
        pool = _mock_pool(synth_row=synth)

        view = await load_best_reasoning_view(pool, "TestVendor")
        assert view is not None
        assert view.primary_wedge is not None
        assert view.primary_wedge.value == "price_squeeze"
