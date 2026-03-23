"""Direct tests for cross-vendor contradiction finders and multi-pass wiring."""

import asyncio
import json

import pytest

from atlas_brain.reasoning.cross_vendor import (
    CrossVendorReasoner,
    _find_asymmetry_contradictions,
    _find_battle_contradictions,
    _find_category_contradictions,
)
from atlas_brain.reasoning.multi_pass import MultiPassResult, PassResult


@pytest.fixture(autouse=True)
def _stub_trace_llm_call(monkeypatch):
    monkeypatch.setattr("atlas_brain.pipelines.llm.trace_llm_call", lambda *args, **kwargs: None)


class _FakeCache:
    def __init__(self):
        self.stored = []

    async def lookup(self, pattern_sig):
        return None

    async def store(self, entry):
        self.stored.append(entry)


class _MockLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self._call_count = 0
        self.model = "mock-model"
        self.name = "mock"

    def chat(self, **kwargs):
        resp = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return {
            "response": json.dumps(resp),
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "_trace_meta": {},
        }

    @property
    def call_count(self):
        return self._call_count


def _make_xv_conclusion(**overrides):
    base = {
        "analysis_type": "pairwise_battle",
        "vendors": ["Acme", "Bravo"],
        "conclusion": "Acme is winning on current metrics.",
        "confidence": 0.8,
        "key_insights": [{"insight": "recommend_ratio gap", "evidence": "recommend_ratio: 10"}],
        "durability_assessment": "temporary",
        "falsification_conditions": ["Bravo improves sentiment"],
        "winner": "Acme",
        "loser": "Bravo",
        "segment_dynamics": {
            "enterprise_winner": "",
            "smb_winner": "",
            "segment_divergence": False,
        },
        "market_regime": "",
        "uncertainty_sources": [],
        "resource_advantage": "",
    }
    base.update(overrides)
    return base


def test_battle_contradictions_flag_locked_direction_mismatch():
    conclusion = {"winner": "Acme", "loser": "Bravo"}
    payload = {
        "vendor_a": {"name": "Acme"},
        "vendor_b": {"name": "Bravo"},
        "locked_direction": {"winner": "Bravo", "loser": "Acme"},
    }
    contradictions = _find_battle_contradictions(conclusion, payload)
    assert any("locked displacement direction" in c for c in contradictions)


def test_category_contradictions_flag_hhi_and_vendor_count():
    conclusion = {"market_regime": "platform_consolidation"}
    payload = {"ecosystem": {"hhi": 900, "vendor_count": 25}}
    contradictions = _find_category_contradictions(conclusion, payload)
    assert any("HHI is low" in c for c in contradictions)
    assert any("vendor count is high" in c for c in contradictions)


def test_asymmetry_contradictions_match_suffix_variation_and_insider_gap():
    conclusion = {"resource_advantage": "Shopify holds the resource advantage due to brand trust."}
    payload = {
        "vendor_a": {"name": "Shopify Inc.", "total_reviews": 10, "insider_signal_count": 1},
        "vendor_b": {"name": "BigCommerce", "total_reviews": 80, "insider_signal_count": 10},
    }
    contradictions = _find_asymmetry_contradictions(conclusion, payload)
    assert any("5x more reviews" in c for c in contradictions)
    assert any("more insider signals" in c for c in contradictions)


def test_cross_vendor_reasoner_runs_classify_challenge_ground(monkeypatch):
    # Enable multi-pass for this test (default is now False)
    monkeypatch.setenv("ATLAS_REASONING__MULTI_PASS_ENABLED", "true")
    monkeypatch.setenv("ATLAS_REASONING__MULTI_PASS_GROUND_ALWAYS", "true")

    cache = _FakeCache()
    llm = _MockLLM([_make_xv_conclusion()])
    seen = {}

    async def _fake_multi_pass_reason(**kwargs):
        seen["finder"] = kwargs.get("contradiction_finder")
        return MultiPassResult(
            final_conclusion=_make_xv_conclusion(winner="Acme", loser="Bravo", confidence=0.72),
            passes=[
                PassResult(1, "classify", _make_xv_conclusion(winner="Acme", loser="Bravo", confidence=0.8), 150, 1.0, False),
                PassResult(2, "challenge", _make_xv_conclusion(winner="Bravo", loser="Acme", confidence=0.7), 150, 1.0, True),
                PassResult(3, "ground", _make_xv_conclusion(winner="Acme", loser="Bravo", confidence=0.72), 150, 1.0, False),
            ],
            total_tokens=450,
            total_duration_ms=3.0,
            passes_executed=3,
        )

    monkeypatch.setattr("atlas_brain.reasoning.cross_vendor.resolve_stratified_llm", lambda cfg: llm)
    monkeypatch.setattr("atlas_brain.reasoning.llm_utils.resolve_stratified_llm_light", lambda cfg: llm)
    monkeypatch.setattr("atlas_brain.reasoning.cross_vendor.multi_pass_reason", _fake_multi_pass_reason)

    async def _run():
        reasoner = CrossVendorReasoner(cache)
        return await reasoner.analyze_battle(
            "Acme",
            "Bravo",
            evidence_a={"velocity_churn_density": 0.3, "recommend_ratio": 10, "total_reviews": 50},
            evidence_b={"velocity_churn_density": -0.2, "recommend_ratio": 35, "total_reviews": 55},
            displacement_edge={"mention_count": 12},
        )

    result = asyncio.run(_run())

    assert seen["finder"] is not None
    assert result.conclusion["winner"] == "Bravo"
    assert result.conclusion["loser"] == "Acme"
    assert "net gainer of defectors" in result.conclusion["conclusion"]
    assert len(cache.stored) == 1


def test_cross_vendor_reasoner_skips_ground_when_challenge_unchanged(monkeypatch):
    # Enable multi-pass for this test (default is now False)
    monkeypatch.setenv("ATLAS_REASONING__MULTI_PASS_ENABLED", "true")
    monkeypatch.setenv("ATLAS_REASONING__MULTI_PASS_GROUND_ALWAYS", "true")

    cache = _FakeCache()
    llm = _MockLLM([_make_xv_conclusion()])

    async def _fake_multi_pass_reason(**kwargs):
        return MultiPassResult(
            final_conclusion=_make_xv_conclusion(winner="Acme", loser="Bravo", confidence=0.79),
            passes=[
                PassResult(1, "classify", _make_xv_conclusion(winner="Acme", loser="Bravo", confidence=0.8), 150, 1.0, False),
                PassResult(2, "challenge", _make_xv_conclusion(winner="Acme", loser="Bravo", confidence=0.79), 150, 1.0, False),
            ],
            total_tokens=300,
            total_duration_ms=2.0,
            passes_executed=2,
        )

    monkeypatch.setattr("atlas_brain.reasoning.cross_vendor.resolve_stratified_llm", lambda cfg: llm)
    monkeypatch.setattr("atlas_brain.reasoning.llm_utils.resolve_stratified_llm_light", lambda cfg: llm)
    monkeypatch.setattr("atlas_brain.reasoning.cross_vendor.multi_pass_reason", _fake_multi_pass_reason)

    async def _run():
        reasoner = CrossVendorReasoner(cache)
        return await reasoner.analyze_battle(
            "Acme",
            "Bravo",
            evidence_a={"velocity_churn_density": 0.3, "recommend_ratio": 10, "total_reviews": 50},
            evidence_b={"velocity_churn_density": -0.2, "recommend_ratio": 35, "total_reviews": 55},
            displacement_edge={"mention_count": 12},
        )

    result = asyncio.run(_run())

    assert result.conclusion["winner"] == "Bravo"
    assert result.conclusion["loser"] == "Acme"
    assert len(cache.stored) == 1
