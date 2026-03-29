"""Tests for Phase 3: contract expansion, reader accessors, and governance validation."""

import sys
from dataclasses import dataclass, field
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

from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
    _build_confidence_posture,
    _build_evidence_governance,
    _build_switch_triggers,
    _build_why_they_stay,
    build_reasoning_contracts,
)
from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
    SynthesisView,
    load_synthesis_view,
)
from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
    validate_synthesis,
)
from atlas_brain.autonomous.tasks._b2b_pool_compression import (
    CompressedPacket,
    ScoredItem,
    SourceRef,
    TrackedAggregate,
)


# ---------------------------------------------------------------------------
# Fixtures: minimal packet with governance signals
# ---------------------------------------------------------------------------

def _make_packet(
    *,
    retention_proof: list[dict] | None = None,
    coverage_gaps: list[dict] | None = None,
    contradiction_rows: list[dict] | None = None,
    metric_ledger: list[dict] | None = None,
    minority_signals: list[dict] | None = None,
    temporal_items: list[ScoredItem] | None = None,
) -> CompressedPacket:
    pools: dict[str, list[ScoredItem]] = {}
    if temporal_items:
        pools["temporal"] = temporal_items
    return CompressedPacket(
        vendor_name="TestVendor",
        pools=pools,
        aggregates=[
            TrackedAggregate("total_reviews", 100, "vault:metric:total_reviews"),
        ],
        retention_proof=retention_proof or [],
        coverage_gaps=coverage_gaps or [],
        contradiction_rows=contradiction_rows or [],
        metric_ledger=metric_ledger or [],
        minority_signals=minority_signals or [],
    )


def _make_synthesis(
    *,
    wedge: str = "price_squeeze",
    confidence: str = "medium",
    has_why_they_stay: bool = False,
    has_confidence_posture: bool = False,
    schema_version: str = "2.3",
) -> dict[str, Any]:
    cn: dict[str, Any] = {
        "primary_wedge": wedge,
        "confidence": confidence,
        "summary": "Test narrative",
        "data_gaps": [],
    }
    vc: dict[str, Any] = {"causal_narrative": cn}
    if has_why_they_stay:
        vc["why_they_stay"] = {
            "summary": "Retained by integrations",
            "strengths": [{"area": "integrations", "evidence": "broad ecosystem"}],
        }
    if has_confidence_posture:
        vc["confidence_posture"] = {
            "overall": confidence,
            "limits": ["thin enterprise sample"],
        }
    return {
        "schema_version": schema_version,
        "reasoning_contracts": {
            "vendor_core_reasoning": vc,
            "displacement_reasoning": {},
            "category_reasoning": {},
            "account_reasoning": {},
        },
        "causal_narrative": cn,
        "segment_playbook": {"confidence": "medium"},
        "timing_intelligence": {"confidence": "medium"},
        "competitive_reframes": {"confidence": "medium"},
        "migration_proof": {"confidence": "medium"},
    }


# ---------------------------------------------------------------------------
# Tests: _build_why_they_stay
# ---------------------------------------------------------------------------

class TestBuildWhyTheyStay:
    def test_builds_from_retention_proof(self):
        packet = _make_packet(retention_proof=[
            {"area": "integrations", "strength": "Broad ecosystem", "mention_count": 15, "_sid": "vault:strength:integrations"},
            {"area": "brand", "strength": "Strong brand loyalty", "mention_count": 8, "_sid": "vault:strength:brand"},
        ])
        result = _build_why_they_stay(packet)
        assert result is not None
        assert "integrations" in result["summary"]
        assert len(result["strengths"]) == 2
        assert result["strengths"][0]["area"] == "integrations"
        assert result["strengths"][0]["_sid"] == "vault:strength:integrations"

    def test_returns_none_when_empty(self):
        packet = _make_packet(retention_proof=[])
        assert _build_why_they_stay(packet) is None

    def test_skips_entries_without_area(self):
        packet = _make_packet(retention_proof=[
            {"strength": "orphan", "_sid": "x"},
        ])
        assert _build_why_they_stay(packet) is None


# ---------------------------------------------------------------------------
# Tests: _build_confidence_posture
# ---------------------------------------------------------------------------

class TestBuildConfidencePosture:
    def test_builds_from_coverage_gaps(self):
        packet = _make_packet(coverage_gaps=[
            {"type": "thin_segment_sample", "area": "role_finance", "sample_size": 7, "_sid": "gap:thin_segment:role_finance"},
            {"type": "missing_pool", "area": "displacement", "sample_size": 0, "_sid": "gap:missing_pool:displacement"},
        ])
        cn = {"confidence": "medium"}
        result = _build_confidence_posture(packet, cn)
        assert result is not None
        assert result["overall"] == "medium"
        assert len(result["limits"]) == 2
        assert "thin role finance sample" in result["limits"][0]
        assert "no displacement evidence" in result["limits"][1]

    def test_returns_none_when_no_gaps_and_high_confidence(self):
        packet = _make_packet(coverage_gaps=[])
        cn = {"confidence": "high"}
        assert _build_confidence_posture(packet, cn) is None

    def test_returns_posture_when_low_confidence_even_without_gaps(self):
        packet = _make_packet(coverage_gaps=[])
        cn = {"confidence": "low"}
        result = _build_confidence_posture(packet, cn)
        assert result is not None
        assert result["overall"] == "low"


# ---------------------------------------------------------------------------
# Tests: _build_switch_triggers
# ---------------------------------------------------------------------------

class TestBuildSwitchTriggers:
    def test_extracts_from_timing_intelligence(self):
        timing = {
            "immediate_triggers": [
                {"type": "deadline", "description": "Contract renewal Q2", "_sid": "temporal:trigger:renewal"},
                {"type": "spike", "description": "Complaint spike March"},
            ],
        }
        packet = _make_packet()
        triggers = _build_switch_triggers(timing, packet)
        assert len(triggers) == 2
        assert triggers[0]["type"] == "deadline"
        assert triggers[0]["_sid"] == "temporal:trigger:renewal"

    def test_extracts_from_temporal_spikes(self):
        timing: dict = {"immediate_triggers": []}
        spike_item = ScoredItem(
            data={"spike_type": "pricing_complaint_surge", "magnitude": 3.5},
            score=0.8,
            source_ref=SourceRef("temporal", "spike", "pricing_surge"),
        )
        packet = _make_packet(temporal_items=[spike_item])
        triggers = _build_switch_triggers(timing, packet)
        assert len(triggers) == 1
        assert triggers[0]["type"] == "spike"

    def test_deduplicates(self):
        timing = {
            "immediate_triggers": [
                {"type": "deadline", "description": "Renewal"},
                {"type": "deadline", "description": "Renewal again"},
            ],
        }
        packet = _make_packet()
        triggers = _build_switch_triggers(timing, packet)
        assert len(triggers) == 1

    def test_empty_when_no_triggers(self):
        timing: dict = {}
        packet = _make_packet()
        assert _build_switch_triggers(timing, packet) == []


# ---------------------------------------------------------------------------
# Tests: _build_evidence_governance
# ---------------------------------------------------------------------------

class TestBuildEvidenceGovernance:
    def test_passes_through_all_sections(self):
        packet = _make_packet(
            metric_ledger=[{"label": "total_reviews", "value": 100, "_sid": "vault:metric:total_reviews"}],
            contradiction_rows=[{"dimension": "support", "_sid": "segment:contradiction:support"}],
            coverage_gaps=[{"type": "missing_pool", "area": "displacement", "_sid": "gap:missing_pool:displacement"}],
            minority_signals=[{"label": "security", "urgency": 9.0, "_sid": "vault:minority:security"}],
        )
        result = _build_evidence_governance(packet)
        assert result is not None
        assert "metric_ledger" in result
        assert "contradictions" in result
        assert "coverage_gaps" in result
        assert "minority_signals" in result

    def test_returns_none_when_all_empty(self):
        packet = _make_packet()
        assert _build_evidence_governance(packet) is None

    def test_partial_governance(self):
        packet = _make_packet(
            metric_ledger=[{"label": "x", "value": 1, "_sid": "y"}],
        )
        result = _build_evidence_governance(packet)
        assert result is not None
        assert "metric_ledger" in result
        assert "contradictions" not in result


# ---------------------------------------------------------------------------
# Tests: SynthesisView accessors
# ---------------------------------------------------------------------------

class TestSynthesisViewPhase3Accessors:
    def test_why_they_stay_accessor(self):
        raw = {
            "reasoning_contracts": {
                "vendor_core_reasoning": {
                    "why_they_stay": {
                        "summary": "Integrations",
                        "strengths": [{"area": "integrations"}],
                    },
                },
            },
        }
        view = load_synthesis_view(raw, "V", schema_version="v2")
        assert view.why_they_stay["summary"] == "Integrations"
        assert len(view.retention_strengths) == 1

    def test_confidence_posture_accessor(self):
        raw = {
            "reasoning_contracts": {
                "vendor_core_reasoning": {
                    "confidence_posture": {
                        "overall": "medium",
                        "limits": ["thin enterprise sample"],
                    },
                },
            },
        }
        view = load_synthesis_view(raw, "V", schema_version="v2")
        assert view.confidence_posture["overall"] == "medium"
        assert view.confidence_limits == ["thin enterprise sample"]

    def test_switch_triggers_accessor(self):
        raw = {
            "reasoning_contracts": {
                "displacement_reasoning": {
                    "switch_triggers": [
                        {"type": "deadline", "description": "Q2 renewal"},
                    ],
                },
            },
        }
        view = load_synthesis_view(raw, "V", schema_version="v2")
        assert len(view.switch_triggers) == 1
        assert view.switch_triggers[0]["type"] == "deadline"

    def test_evidence_governance_accessor(self):
        raw = {
            "reasoning_contracts": {
                "evidence_governance": {
                    "metric_ledger": [{"label": "total_reviews", "value": 100}],
                    "contradictions": [{"dimension": "support"}],
                    "coverage_gaps": [{"type": "missing_pool"}],
                },
            },
        }
        view = load_synthesis_view(raw, "V", schema_version="v2")
        assert len(view.metric_ledger) == 1
        assert len(view.contradictions) == 1
        assert len(view.coverage_gaps) == 1

    def test_empty_defaults(self):
        view = load_synthesis_view({}, "V", schema_version="v2")
        assert view.why_they_stay == {}
        assert view.confidence_posture == {}
        assert view.switch_triggers == []
        assert view.evidence_governance == {}
        assert view.metric_ledger == []
        assert view.contradictions == []
        assert view.coverage_gaps == []
        assert view.confidence_limits == []
        assert view.retention_strengths == []


# ---------------------------------------------------------------------------
# Tests: Validation rules
# ---------------------------------------------------------------------------

class TestValidationContradictionHedging:
    def test_blocks_high_confidence_with_contradictions(self):
        synthesis = _make_synthesis(confidence="high")
        packet = _make_packet(contradiction_rows=[
            {"dimension": "support", "_sid": "segment:contradiction:support"},
        ])
        result = validate_synthesis(synthesis, packet)
        error_codes = [e.code for e in result.errors]
        assert "contradiction_confidence_cap" in error_codes
        assert not result.is_valid

    def test_blocks_missing_contradiction_dims_in_data_gaps(self):
        synthesis = _make_synthesis(confidence="medium")
        # data_gaps is empty, contradictions on "support" -> error
        packet = _make_packet(contradiction_rows=[
            {"dimension": "support", "_sid": "segment:contradiction:support"},
        ])
        result = validate_synthesis(synthesis, packet)
        error_codes = [e.code for e in result.errors]
        assert "contradiction_dims_missing_from_data_gaps" in error_codes

    def test_passes_when_contradictions_reflected_in_data_gaps(self):
        synthesis = _make_synthesis(confidence="medium")
        cn = synthesis["reasoning_contracts"]["vendor_core_reasoning"]["causal_narrative"]
        cn["data_gaps"] = ["contradictory support signals across segments"]
        synthesis["reasoning_contracts"]["vendor_core_reasoning"]["confidence_posture"] = {
            "overall": "medium",
            "limits": ["support contradiction between segments"],
        }
        packet = _make_packet(contradiction_rows=[
            {"dimension": "support", "_sid": "segment:contradiction:support"},
        ])
        result = validate_synthesis(synthesis, packet)
        error_codes = [e.code for e in result.errors]
        assert "contradiction_confidence_cap" not in error_codes
        assert "contradiction_dims_missing_from_data_gaps" not in error_codes
        assert "contradiction_dims_missing_from_limits" not in error_codes

    def test_no_errors_without_contradictions(self):
        synthesis = _make_synthesis(confidence="high")
        packet = _make_packet(contradiction_rows=[])
        result = validate_synthesis(synthesis, packet)
        error_codes = [e.code for e in result.errors]
        assert "contradiction_confidence_cap" not in error_codes


class TestValidationWhyTheyStay:
    def test_blocks_missing_why_they_stay(self):
        synthesis = _make_synthesis(has_why_they_stay=False)
        packet = _make_packet(retention_proof=[
            {"area": "integrations", "strength": "broad", "_sid": "vault:strength:integrations"},
        ])
        result = validate_synthesis(synthesis, packet)
        error_codes = [e.code for e in result.errors]
        assert "missing_why_they_stay" in error_codes
        assert not result.is_valid

    def test_passes_when_why_they_stay_present(self):
        synthesis = _make_synthesis(has_why_they_stay=True)
        packet = _make_packet(retention_proof=[
            {"area": "integrations", "strength": "broad", "_sid": "vault:strength:integrations"},
        ])
        result = validate_synthesis(synthesis, packet)
        error_codes = [e.code for e in result.errors]
        assert "missing_why_they_stay" not in error_codes

    def test_no_error_when_no_retention_proof(self):
        synthesis = _make_synthesis(has_why_they_stay=False)
        packet = _make_packet(retention_proof=[])
        result = validate_synthesis(synthesis, packet)
        error_codes = [e.code for e in result.errors]
        assert "missing_why_they_stay" not in error_codes


class TestValidationConfidenceLimits:
    def test_blocks_missing_limits_with_gaps(self):
        synthesis = _make_synthesis(has_confidence_posture=False)
        packet = _make_packet(coverage_gaps=[
            {"type": "missing_pool", "area": "displacement", "_sid": "gap:missing_pool:displacement"},
        ])
        result = validate_synthesis(synthesis, packet)
        error_codes = [e.code for e in result.errors]
        assert "missing_confidence_limits" in error_codes
        assert not result.is_valid

    def test_passes_when_limits_present(self):
        synthesis = _make_synthesis(has_confidence_posture=True)
        packet = _make_packet(coverage_gaps=[
            {"type": "missing_pool", "area": "displacement", "_sid": "gap:missing_pool:displacement"},
        ])
        result = validate_synthesis(synthesis, packet)
        error_codes = [e.code for e in result.errors]
        assert "missing_confidence_limits" not in error_codes

    def test_no_error_when_no_gaps(self):
        synthesis = _make_synthesis(has_confidence_posture=False)
        packet = _make_packet(coverage_gaps=[])
        result = validate_synthesis(synthesis, packet)
        error_codes = [e.code for e in result.errors]
        assert "missing_confidence_limits" not in error_codes


# ---------------------------------------------------------------------------
# Tests: materialized_contracts includes evidence_governance
# ---------------------------------------------------------------------------

class TestMaterializedContractsGovernance:
    def test_evidence_governance_in_materialized(self):
        raw = {
            "reasoning_contracts": {
                "vendor_core_reasoning": {
                    "causal_narrative": {"primary_wedge": "price_squeeze", "confidence": "high"},
                },
                "evidence_governance": {
                    "metric_ledger": [{"label": "total_reviews", "value": 100}],
                    "contradictions": [{"dimension": "support"}],
                },
            },
        }
        view = load_synthesis_view(raw, "V", schema_version="v2")
        contracts = view.materialized_contracts()
        assert "evidence_governance" in contracts
        assert len(contracts["evidence_governance"]["metric_ledger"]) == 1

    def test_evidence_governance_absent_when_empty(self):
        raw = {
            "reasoning_contracts": {
                "vendor_core_reasoning": {
                    "causal_narrative": {"primary_wedge": "price_squeeze"},
                },
            },
        }
        view = load_synthesis_view(raw, "V", schema_version="v2")
        contracts = view.materialized_contracts()
        assert "evidence_governance" not in contracts


# ---------------------------------------------------------------------------
# Tests: metric_ledger validation distinguishes ledger vs packet
# ---------------------------------------------------------------------------

class TestValidationMetricLedgerStrict:
    def test_warns_number_in_packet_but_not_in_ledger(self):
        """A source_id in packet aggregates but not in metric_ledger should warn."""
        synthesis = _make_synthesis()
        # Add a numeric wrapper into the contracts-path causal_narrative
        cn = synthesis["reasoning_contracts"]["vendor_core_reasoning"]["causal_narrative"]
        cn["some_metric"] = {
            "value": 42,
            "source_id": "vault:metric:total_reviews",
        }
        # Packet has the aggregate but metric_ledger is empty
        packet = _make_packet(metric_ledger=[])
        result = validate_synthesis(synthesis, packet)
        codes = [w.code for w in result.warnings]
        assert "number_not_in_ledger" in codes

    def test_no_warning_when_in_ledger(self):
        synthesis = _make_synthesis()
        cn = synthesis["reasoning_contracts"]["vendor_core_reasoning"]["causal_narrative"]
        cn["some_metric"] = {
            "value": 100,
            "source_id": "vault:metric:total_reviews",
        }
        packet = _make_packet(metric_ledger=[
            {"label": "total_reviews", "value": 100, "_sid": "vault:metric:total_reviews"},
        ])
        result = validate_synthesis(synthesis, packet)
        warning_codes = [w.code for w in result.warnings]
        error_codes = [e.code for e in result.errors]
        assert "number_not_in_ledger" not in warning_codes
        assert "unsupported_number" not in error_codes


# ---------------------------------------------------------------------------
# Tests: contract builders merge instead of overwrite
# ---------------------------------------------------------------------------

class TestContractBuildersMerge:
    def test_why_they_stay_merges_model_and_deterministic(self):
        """Model-provided strengths preserved, deterministic fills gaps."""
        packet = _make_packet(retention_proof=[
            {"area": "integrations", "strength": "Broad ecosystem", "mention_count": 15, "_sid": "vault:strength:integrations"},
            {"area": "brand", "strength": "Brand loyalty", "mention_count": 8, "_sid": "vault:strength:brand"},
        ])
        synthesis = _make_synthesis()
        # Simulate model providing why_they_stay with a richer neutralization field
        synthesis["reasoning_contracts"]["vendor_core_reasoning"]["why_they_stay"] = {
            "summary": "Model-provided summary with neutralization context",
            "strengths": [
                {"area": "integrations", "evidence": "Model evidence", "neutralization": "But migration tools exist"},
            ],
        }
        contracts = build_reasoning_contracts(synthesis, packet)
        wts = contracts["vendor_core_reasoning"]["why_they_stay"]
        # Model summary preserved
        assert "Model-provided summary" in wts["summary"]
        # Model's integrations strength preserved with neutralization
        integrations = [s for s in wts["strengths"] if s["area"] == "integrations"]
        assert integrations[0].get("neutralization") == "But migration tools exist"
        # Deterministic "brand" was not in model output, so it gets merged in
        brand = [s for s in wts["strengths"] if s["area"] == "brand"]
        assert len(brand) == 1

    def test_confidence_posture_merges_limits(self):
        """Model-provided limits preserved, deterministic adds new ones."""
        packet = _make_packet(coverage_gaps=[
            {"type": "missing_pool", "area": "displacement", "sample_size": 0, "_sid": "gap:missing_pool:displacement"},
        ])
        synthesis = _make_synthesis()
        synthesis["reasoning_contracts"]["vendor_core_reasoning"]["confidence_posture"] = {
            "overall": "medium",
            "limits": ["model-identified limitation"],
        }
        contracts = build_reasoning_contracts(synthesis, packet)
        cp = contracts["vendor_core_reasoning"]["confidence_posture"]
        assert "model-identified limitation" in cp["limits"]
        assert any("no displacement" in lim for lim in cp["limits"])

    def test_switch_triggers_merges(self):
        """Model-provided triggers preserved, deterministic adds new types."""
        packet = _make_packet(temporal_items=[
            ScoredItem(
                data={"spike_type": "complaint_surge", "magnitude": 3.0},
                score=0.8,
                source_ref=SourceRef("temporal", "spike", "complaint_surge"),
            ),
        ])
        synthesis = _make_synthesis()
        synthesis["reasoning_contracts"]["displacement_reasoning"]["switch_triggers"] = [
            {"type": "deadline", "description": "Q2 contract renewal", "rationale": "70% renew in Q2"},
        ]
        synthesis["timing_intelligence"] = {
            "confidence": "medium",
            "immediate_triggers": [],
        }
        contracts = build_reasoning_contracts(synthesis, packet)
        st = contracts["displacement_reasoning"]["switch_triggers"]
        # Model's deadline trigger preserved with rationale
        deadline = [t for t in st if t.get("type") == "deadline"]
        assert deadline[0].get("rationale") == "70% renew in Q2"
        # Deterministic spike merged in
        spike = [t for t in st if t.get("type") == "spike"]
        assert len(spike) == 1
