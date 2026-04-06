"""Tests for Phase 8: Battle Card Governance Enforcement."""

import sys
from datetime import date
from types import SimpleNamespace
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

from atlas_brain.autonomous.tasks.b2b_battle_cards import (
    _build_battle_card_render_payload,
    _battle_card_trace_metadata,
    _evaluate_battle_card_quality,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _base_card(**overrides) -> dict[str, Any]:
    card: dict[str, Any] = {
        "vendor": "TestVendor",
        "category": "CRM",
        "churn_pressure_score": 72,
        "risk_level": "high",
        "total_reviews": 150,
        "confidence": "medium",
        "data_as_of_date": date.today().isoformat(),
        "data_stale": False,
        "evidence_window_days": 90,
        "evidence_window_is_thin": False,
    }
    card.update(overrides)
    return card


def _card_with_governance(**overrides) -> dict[str, Any]:
    return _base_card(
        reasoning_contracts={
            "vendor_core_reasoning": {
                "causal_narrative": {
                    "primary_wedge": "price_squeeze",
                    "confidence": "medium",
                },
                "why_they_stay": {
                    "summary": "Ecosystem breadth",
                    "strengths": [
                        {"area": "integrations", "evidence": "Broad API"},
                    ],
                },
                "confidence_posture": {
                    "overall": "medium",
                    "limits": ["thin enterprise sample"],
                },
            },
            "evidence_governance": {
                "contradictions": [
                    {"dimension": "support", "_sid": "segment:contradiction:support"},
                ],
                "coverage_gaps": [
                    {"type": "missing_pool", "area": "displacement", "_sid": "gap:missing_pool:displacement"},
                ],
                "metric_ledger": [
                    {"label": "total_reviews", "value": 150, "_sid": "vault:metric:total_reviews"},
                ],
            },
        },
        **overrides,
    )


def _anchor_support() -> dict[str, Any]:
    return {
        "anchor_examples": {
            "outlier_or_named_account": [
                {
                    "witness_id": "witness:r1:0",
                    "excerpt_text": "Hack Club said Slack tried to charge $200k/year at renewal.",
                    "reviewer_company": "Hack Club",
                    "time_anchor": "Q2 renewal",
                    "numeric_literals": {"currency_mentions": ["$200k/year"]},
                    "competitor": "Google Chat",
                },
            ],
        },
        "witness_highlights": [
            {
                "witness_id": "witness:r1:0",
                "excerpt_text": "Hack Club said Slack tried to charge $200k/year at renewal.",
                "reviewer_company": "Hack Club",
                "time_anchor": "Q2 renewal",
                "numeric_literals": {"currency_mentions": ["$200k/year"]},
                "competitor": "Google Chat",
            },
        ],
        "reference_ids": {"witness_ids": ["witness:r1:0"]},
    }


# ---------------------------------------------------------------------------
# Tests: render payload includes governance context
# ---------------------------------------------------------------------------

class TestRenderPayloadGovernance:
    def test_retention_context_in_payload(self):
        card = _card_with_governance()
        payload = _build_battle_card_render_payload(card)
        assert "retention_context" in payload
        assert payload["retention_context"]["summary"] == "Ecosystem breadth"

    def test_confidence_posture_in_payload(self):
        card = _card_with_governance()
        payload = _build_battle_card_render_payload(card)
        assert "confidence_posture" in payload
        assert payload["confidence_posture"]["overall"] == "medium"

    def test_contradictions_in_payload(self):
        card = _card_with_governance()
        payload = _build_battle_card_render_payload(card)
        assert "contradictions" in payload
        assert len(payload["contradictions"]) == 1

    def test_coverage_gaps_in_payload(self):
        card = _card_with_governance()
        payload = _build_battle_card_render_payload(card)
        assert "coverage_gaps" in payload
        assert payload["coverage_gaps"][0]["type"] == "missing_pool"

    def test_no_governance_when_no_contracts(self):
        card = _base_card()
        payload = _build_battle_card_render_payload(card)
        assert "retention_context" not in payload
        assert "contradictions" not in payload
        assert "coverage_gaps" not in payload

    def test_metric_ledger_in_payload(self):
        card = _card_with_governance()
        payload = _build_battle_card_render_payload(card)
        assert "metric_ledger" in payload

    def test_anchor_examples_and_witness_highlights_in_payload(self):
        card = _card_with_governance(**_anchor_support())
        payload = _build_battle_card_render_payload(card)
        assert payload["anchor_examples"]["outlier_or_named_account"][0]["reviewer_company"] == "Hack Club"
        assert payload["witness_highlights"][0]["witness_id"] == "witness:r1:0"
        assert payload["reference_ids"]["witness_ids"] == ["witness:r1:0"]

    def test_full_contract_bundle_omitted_when_section_contracts_present(self):
        card = _card_with_governance()
        payload = _build_battle_card_render_payload(card)
        assert "reasoning_contracts" not in payload
        assert "vendor_core_reasoning" in payload

    def test_compacts_reasoning_contract_sections_for_prompt_payload(self):
        card = _card_with_governance()
        card["reasoning_contracts"] = {
            "vendor_core_reasoning": {
                "causal_narrative": {
                    "primary_wedge": "price_squeeze",
                    "trigger": "Budget shock",
                    "why_now": "Renewals are active",
                    "who_most_affected": "Ops leaders",
                    "confidence": "medium",
                    "data_gaps": [f"gap-{i}" for i in range(6)],
                    "citations": [f"cite-{i}" for i in range(10)],
                },
                "segment_playbook": {
                    "confidence": "low",
                    "data_gaps": [f"segment-gap-{i}" for i in range(6)],
                    "priority_segments": [
                        {"segment": f"seg-{i}", "why_now": "now", "sample_size": i + 1}
                        for i in range(6)
                    ],
                    "supporting_evidence": {
                        "top_roles": [
                            {"role_type": f"role-{i}", "review_count": i + 1}
                            for i in range(5)
                        ],
                    },
                },
                "timing_intelligence": {
                    "confidence": "medium",
                    "best_timing_window": "Q2",
                    "immediate_triggers": [f"trigger-{i}" for i in range(6)],
                    "data_gaps": [f"time-gap-{i}" for i in range(6)],
                    "supporting_evidence": {"large_blob": "drop-me"},
                },
                "why_they_stay": {
                    "summary": "They stay for integrations.",
                    "strengths": [
                        {"area": f"strength-{i}", "evidence": "proof", "neutralization": "counter"}
                        for i in range(5)
                    ],
                },
                "confidence_posture": {
                    "overall": "medium",
                    "limits": [f"limit-{i}" for i in range(6)],
                },
            },
        }
        payload = _build_battle_card_render_payload(card)
        vendor_core = payload["vendor_core_reasoning"]
        assert "citations" not in vendor_core
        assert len(vendor_core["causal_narrative"]["data_gaps"]) == 4
        assert len(vendor_core["segment_playbook"]["priority_segments"]) == 3
        assert len(vendor_core["segment_playbook"]["supporting_evidence"]["top_roles"]) == 3
        assert "supporting_evidence" not in vendor_core["timing_intelligence"]
        assert len(vendor_core["timing_intelligence"]["immediate_triggers"]) == 4
        assert len(vendor_core["why_they_stay"]["strengths"]) == 3
        assert len(vendor_core["confidence_posture"]["limits"]) == 4

    def test_limits_anchor_witness_reference_and_account_payload_inputs(self):
        card = _card_with_governance(
            cross_vendor_battles=[
                {"opponent": f"Opp-{i}", "conclusion": "x", "confidence": 0.9}
                for i in range(5)
            ],
            high_intent_companies=[
                {"company": f"Acct-{i}", "urgency": 8.0, "role": "VP"}
                for i in range(5)
            ],
            reference_ids={
                "witness_ids": [f"w-{i}" for i in range(20)],
                "metric_ids": [f"m-{i}" for i in range(20)],
            },
            anchor_examples={
                "outlier_or_named_account": [
                    {"witness_id": f"w-{i}", "reviewer_company": f"Co-{i}", "excerpt_text": "quote"}
                    for i in range(3)
                ],
            },
            witness_highlights=[
                {"witness_id": f"w-{i}", "excerpt_text": "quote"}
                for i in range(8)
            ],
        )
        payload = _build_battle_card_render_payload(card)
        assert len(payload["cross_vendor_battles"]) == 2
        assert len(payload["high_intent_companies"]) == 3
        assert len(payload["reference_ids"]["witness_ids"]) == 12
        assert len(payload["reference_ids"]["metric_ids"]) == 12
        assert len(payload["anchor_examples"]["outlier_or_named_account"]) == 1
        assert len(payload["witness_highlights"]) == 4


class TestTraceMetadata:
    def test_battle_card_trace_metadata_includes_vendor_and_run_identity(self):
        task = SimpleNamespace(id="run-123")
        metadata = _battle_card_trace_metadata(
            task,
            {"vendor": "HubSpot"},
            attempt=2,
        )
        assert metadata["vendor_name"] == "HubSpot"
        assert metadata["run_id"] == "run-123"
        assert metadata["entity_type"] == "battle_card"
        assert metadata["entity_id"] == "HubSpot"
        assert metadata["source_name"] == "b2b_battle_cards"
        assert metadata["event_type"] == "llm_overlay"
        assert metadata["attempt_no"] == 2


# ---------------------------------------------------------------------------
# Tests: governance quality checks
# ---------------------------------------------------------------------------

class TestGovernanceQualityChecks:
    def test_contradiction_overreach_blocks_absolute_language(self):
        card = _card_with_governance(
            executive_summary="TestVendor is clearly losing market share across all segments.",
        )
        quality = _evaluate_battle_card_quality(card, phase="final")
        assert any("absolute language" in b for b in quality["failed_checks"])

    def test_contradiction_warns_when_no_absolute_language(self):
        card = _card_with_governance(
            executive_summary="TestVendor faces pricing pressure in some segments.",
        )
        quality = _evaluate_battle_card_quality(card, phase="final")
        assert not any("absolute language" in b for b in quality["failed_checks"])
        assert any("contradictory evidence" in w for w in quality["warnings"])

    def test_coverage_gap_warns(self):
        card = _card_with_governance()
        quality = _evaluate_battle_card_quality(card, phase="deterministic")
        assert any("thin evidence" in w for w in quality["warnings"])

    def test_missing_retention_warns(self):
        """When synthesis has retention strengths but card doesn't, warn."""
        card = _card_with_governance()
        # Card has no why_they_stay output (LLM didn't generate it yet)
        card.pop("why_they_stay", None)
        quality = _evaluate_battle_card_quality(card, phase="deterministic")
        assert any("retention strengths" in w for w in quality["warnings"])

    def test_no_governance_warnings_without_contracts(self):
        card = _base_card()
        quality = _evaluate_battle_card_quality(card, phase="deterministic")
        assert not any("contradictory" in w for w in quality["warnings"])
        assert not any("thin evidence" in w for w in quality["warnings"])
        assert not any("retention" in w for w in quality["warnings"])

    def test_governance_reduces_quality_score(self):
        """Cards with governance warnings should score lower."""
        clean_card = _base_card()
        gov_card = _card_with_governance()

        clean_q = _evaluate_battle_card_quality(clean_card, phase="deterministic")
        gov_q = _evaluate_battle_card_quality(gov_card, phase="deterministic")

        # Governance card has contradiction + coverage_gap + retention warnings
        assert gov_q["score"] < clean_q["score"]


class TestCopyLevelGovernance:
    def test_absolute_language_in_talk_track_blocked(self):
        card = _card_with_governance(
            executive_summary="Moderate pressure observed.",
            talk_track="TestVendor is clearly the worst in its category.",
        )
        quality = _evaluate_battle_card_quality(card, phase="final")
        assert any(
            "talk_track" in b and "absolute language" in b
            for b in quality["failed_checks"]
        )

    def test_absolute_language_in_objection_handlers_blocked(self):
        card = _card_with_governance(
            executive_summary="Moderate pressure observed.",
            objection_handlers=[
                {"objection": "Price", "response": "We are undeniably cheaper."},
            ],
        )
        quality = _evaluate_battle_card_quality(card, phase="final")
        assert any(
            "objection_handlers" in b and "absolute language" in b
            for b in quality["failed_checks"]
        )

    def test_coverage_gap_strong_claims_warned(self):
        card = _card_with_governance(
            executive_summary="Some pressure.",
            talk_track="Our displacement advantage is proven and guaranteed.",
        )
        quality = _evaluate_battle_card_quality(card, phase="final")
        assert any(
            "strong claims" in w and "displacement" in w
            for w in quality["warnings"]
        )

    def test_no_copy_warnings_when_hedged(self):
        card = _card_with_governance(
            executive_summary="Some pressure, though evidence is mixed.",
            talk_track="Based on available data, pricing pressure appears significant.",
        )
        quality = _evaluate_battle_card_quality(card, phase="final")
        assert not any("absolute language" in b for b in quality["failed_checks"])
        assert not any("strong claims" in w for w in quality["warnings"])

    def test_anchor_backed_specificity_blocks_generic_copy(self):
        card = _card_with_governance(
            executive_summary="The vendor faces broad commercial pressure right now.",
            talk_track="Lead with a pricing benchmark and test urgency in discovery.",
            **_anchor_support(),
        )
        quality = _evaluate_battle_card_quality(card, phase="final")
        assert any(
            "witness-backed anchor" in blocker
            for blocker in quality["failed_checks"]
        )

    def test_anchor_backed_specificity_accepts_concrete_copy(self):
        card = _card_with_governance(
            executive_summary="Hack Club flagged a $200k/year renewal shock, which makes pricing pressure concrete right now.",
            talk_track="Use the Hack Club renewal story to test whether this account is facing a similar Q2 renewal or Google Chat evaluation.",
            **_anchor_support(),
        )
        quality = _evaluate_battle_card_quality(card, phase="final")
        assert not any(
            "witness-backed anchor" in blocker
            for blocker in quality["failed_checks"]
        )
        assert not any(
            "named-account anchor exists" in warning
            for warning in quality["warnings"]
        )
