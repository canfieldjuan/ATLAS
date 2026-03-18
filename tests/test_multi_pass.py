"""Tests for the multi-pass reasoning engine."""

import asyncio
import json
import pytest

from atlas_brain.reasoning.multi_pass import (
    MultiPassResult,
    PassResult,
    _find_contradicting_evidence,
    multi_pass_reason,
)


@pytest.fixture(autouse=True)
def _stub_trace_llm_call(monkeypatch):
    monkeypatch.setattr("atlas_brain.pipelines.llm.trace_llm_call", lambda *args, **kwargs: None)


@pytest.fixture(autouse=True)
def _stub_to_thread(monkeypatch):
    async def _direct_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("atlas_brain.reasoning.multi_pass.asyncio.to_thread", _direct_to_thread)


# ---------------------------------------------------------------------------
# Contradiction detection tests
# ---------------------------------------------------------------------------


class TestFindContradictingEvidence:
    def test_top_pain_maps_to_different_archetype(self):
        """UX as top pain + pricing_shock archetype -> contradiction."""
        conclusion = {"archetype": "pricing_shock", "confidence": 0.75}
        evidence = {
            "pain_categories": [
                {"category": "ux", "count": 15},
                {"category": "pricing", "count": 5},
            ]
        }
        contradictions = _find_contradicting_evidence(conclusion, evidence)
        assert len(contradictions) >= 1
        assert "ux" in contradictions[0].lower()
        assert "leadership_redesign" in contradictions[0]

    def test_no_contradiction_when_pain_matches_archetype(self):
        """Pricing as top pain + pricing_shock -> no pain contradiction."""
        conclusion = {"archetype": "pricing_shock", "confidence": 0.75}
        evidence = {
            "pain_categories": [
                {"category": "pricing", "count": 15},
                {"category": "ux", "count": 5},
            ]
        }
        contradictions = _find_contradicting_evidence(conclusion, evidence)
        # Should not have a pain-category contradiction
        pain_contradictions = [c for c in contradictions if "pain category" in c.lower()]
        assert len(pain_contradictions) == 0

    def test_strong_alternative_archetype_score(self):
        """Pre-score > 0.4 for different archetype -> contradiction."""
        conclusion = {"archetype": "pricing_shock", "confidence": 0.7}
        evidence = {
            "archetype_scores": [
                {"archetype": "feature_gap", "score": 0.55},
                {"archetype": "pricing_shock", "score": 0.3},
            ]
        }
        contradictions = _find_contradicting_evidence(conclusion, evidence)
        assert any("feature_gap" in c for c in contradictions)

    def test_low_reviews_high_confidence(self):
        """<20 reviews + confidence > 0.7 -> contradiction."""
        conclusion = {"archetype": "pricing_shock", "confidence": 0.85}
        evidence = {"total_reviews": 8}
        contradictions = _find_contradicting_evidence(conclusion, evidence)
        assert any("reviews" in c.lower() for c in contradictions)

    def test_no_contradiction_low_confidence(self):
        """Low confidence doesn't trigger the review count contradiction."""
        conclusion = {"archetype": "pricing_shock", "confidence": 0.5}
        evidence = {"total_reviews": 8}
        contradictions = _find_contradicting_evidence(conclusion, evidence)
        review_contradictions = [c for c in contradictions if "reviews" in c.lower()]
        assert len(review_contradictions) == 0

    def test_high_displacement_non_displacement_archetype(self):
        """High displacement count + non-displacement archetype -> contradiction."""
        conclusion = {"archetype": "pricing_shock", "confidence": 0.7}
        evidence = {"displacement_mention_count": 12}
        contradictions = _find_contradicting_evidence(conclusion, evidence)
        assert any("displacement" in c.lower() for c in contradictions)

    def test_max_three_contradictions(self):
        """Never returns more than 3 contradictions."""
        conclusion = {"archetype": "pricing_shock", "confidence": 0.85}
        evidence = {
            "pain_categories": [{"category": "ux", "count": 15}],
            "archetype_scores": [{"archetype": "feature_gap", "score": 0.55}],
            "total_reviews": 5,
            "displacement_mention_count": 10,
        }
        contradictions = _find_contradicting_evidence(conclusion, evidence)
        assert len(contradictions) <= 3

    def test_mixed_archetype_skips_pain_contradiction(self):
        """Mixed archetype doesn't trigger pain-category contradiction."""
        conclusion = {"archetype": "mixed", "confidence": 0.6}
        evidence = {
            "pain_categories": [{"category": "ux", "count": 15}],
        }
        contradictions = _find_contradicting_evidence(conclusion, evidence)
        pain_contradictions = [c for c in contradictions if "maps to" in c]
        assert len(pain_contradictions) == 0

    def test_empty_evidence(self):
        """Empty evidence -> no contradictions."""
        conclusion = {"archetype": "pricing_shock", "confidence": 0.7}
        contradictions = _find_contradicting_evidence(conclusion, {})
        assert contradictions == []


# ---------------------------------------------------------------------------
# Multi-pass engine tests (with mock LLM)
# ---------------------------------------------------------------------------


class MockLLM:
    """Mock LLM that returns pre-configured conclusions."""

    def __init__(self, responses: list[dict]):
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


def _make_conclusion(archetype="pricing_shock", confidence=0.75, **overrides):
    base = {
        "archetype": archetype,
        "secondary_archetype": None,
        "confidence": confidence,
        "risk_level": "medium",
        "executive_summary": "Test summary with some specific metrics like churn_density: 25% and avg_urgency: 6.5.",
        "key_signals": ["churn_density: 25%", "price_complaint_rate: 0.18"],
        "falsification_conditions": ["price reduction reverses churn trend"],
        "uncertainty_sources": ["limited temporal data", "small sample size"],
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_single_pass_when_disabled():
    """When enabled=False, only Pass 1 runs."""
    llm = MockLLM([_make_conclusion()])
    result = await multi_pass_reason(
        llm=llm,
        system_prompt="test",
        evidence_payload={"evidence": {}},
        json_schema={},
        max_tokens=1024,
        temperature=0.3,
        enabled=False,
    )
    assert result.passes_executed == 1
    assert result.passes[0].pass_type == "classify"
    assert llm.call_count == 1


@pytest.mark.asyncio
async def test_skip_challenge_low_confidence():
    """Low-confidence cases skip challenge but still get lightweight grounding."""
    p1 = _make_conclusion(confidence=0.25)
    p_ground = _make_conclusion(confidence=0.3)
    llm = MockLLM([p1, p_ground])
    result = await multi_pass_reason(
        llm=llm,
        system_prompt="test",
        evidence_payload={"evidence": {"pain_categories": [{"category": "ux", "count": 15}]}},
        json_schema={},
        max_tokens=1024,
        temperature=0.3,
        challenge_confidence_floor=0.3,
        ground_always=True,
    )
    assert result.passes_executed == 2
    assert result.passes[1].pass_type == "ground"
    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_skip_challenge_no_contradictions():
    """No contradictions skips challenge but still runs grounding by default."""
    conclusion = _make_conclusion(archetype="pricing_shock", confidence=0.75)
    p_ground = _make_conclusion(archetype="pricing_shock", confidence=0.75)
    llm = MockLLM([conclusion, p_ground])
    result = await multi_pass_reason(
        llm=llm,
        system_prompt="test",
        evidence_payload={
            "evidence": {
                "pain_categories": [{"category": "pricing", "count": 15}],
                "total_reviews": 50,
            }
        },
        json_schema={},
        max_tokens=1024,
        temperature=0.3,
        ground_always=True,
    )
    assert result.passes_executed == 2
    assert result.passes[1].pass_type == "ground"
    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_challenge_fires_on_contradiction():
    """Pass 2 fires when contradictions exist and confidence > floor."""
    p1 = _make_conclusion(archetype="pricing_shock", confidence=0.75)
    p2 = _make_conclusion(archetype="leadership_redesign", confidence=0.70)
    llm = MockLLM([p1, p2])
    result = await multi_pass_reason(
        llm=llm,
        system_prompt="test",
        evidence_payload={
            "evidence": {
                "pain_categories": [{"category": "ux", "count": 15}],
                "total_reviews": 50,
            }
        },
        json_schema={},
        max_tokens=1024,
        temperature=0.3,
    )
    assert result.passes_executed >= 2
    assert result.passes[1].pass_type == "challenge"
    assert llm.call_count >= 2


@pytest.mark.asyncio
async def test_challenge_fires_for_low_confidence_high_impact_case():
    """Low-confidence but high-impact cases still get challenged."""
    p1 = _make_conclusion(archetype="pricing_shock", confidence=0.25)
    p2 = _make_conclusion(archetype="leadership_redesign", confidence=0.42)
    p3 = _make_conclusion(archetype="leadership_redesign", confidence=0.43)
    llm = MockLLM([p1, p2, p3])
    result = await multi_pass_reason(
        llm=llm,
        system_prompt="test",
        evidence_payload={
            "evidence": {
                "pain_categories": [{"category": "ux", "count": 15}],
                "total_reviews": 40,
                "churn_density": 28.0,
            }
        },
        json_schema={},
        max_tokens=1024,
        temperature=0.3,
        challenge_confidence_floor=0.3,
        ground_always=True,
    )
    assert result.passes_executed == 3
    assert result.passes[1].pass_type == "challenge"
    assert llm.call_count == 3


def test_ground_fires_when_challenge_changes():
    """Pass 3 fires when challenge changed the archetype."""
    p1 = _make_conclusion(archetype="pricing_shock", confidence=0.75)
    p2 = _make_conclusion(archetype="leadership_redesign", confidence=0.70)
    p3 = _make_conclusion(archetype="leadership_redesign", confidence=0.72)
    llm = MockLLM([p1, p2, p3])
    async def _run():
        return await multi_pass_reason(
            llm=llm,
            system_prompt="test",
            evidence_payload={
                "evidence": {
                    "pain_categories": [{"category": "ux", "count": 15}],
                    "total_reviews": 50,
                }
            },
            json_schema={},
            max_tokens=1024,
            temperature=0.3,
        )
    result = asyncio.run(_run())
    assert result.passes_executed == 3
    assert result.passes[2].pass_type == "ground"
    assert result.final_conclusion["archetype"] == "leadership_redesign"


def test_ground_runs_when_challenge_unchanged():
    """Ground still runs after a challenge pass even if the conclusion holds."""
    p1 = _make_conclusion(archetype="pricing_shock", confidence=0.75)
    # Challenge keeps same archetype and confidence within threshold
    p2 = _make_conclusion(archetype="pricing_shock", confidence=0.74)
    p3 = _make_conclusion(archetype="pricing_shock", confidence=0.74)
    llm = MockLLM([p1, p2, p3])
    async def _run():
        return await multi_pass_reason(
            llm=llm,
            system_prompt="test",
            evidence_payload={
                "evidence": {
                    "pain_categories": [{"category": "ux", "count": 15}],
                    "total_reviews": 50,
                }
            },
            json_schema={},
            max_tokens=1024,
            temperature=0.3,
            ground_change_threshold=0.05,
        )
    result = asyncio.run(_run())
    assert result.passes_executed == 3
    assert result.passes[2].pass_type == "ground"
    assert llm.call_count == 3


def test_ground_can_be_disabled_when_challenge_skips():
    """ground_always=False preserves the old skip behavior when challenge does not run."""
    p1 = _make_conclusion(confidence=0.25)
    llm = MockLLM([p1])

    async def _run():
        return await multi_pass_reason(
            llm=llm,
            system_prompt="test",
            evidence_payload={"evidence": {"pain_categories": [{"category": "ux", "count": 15}]}},
            json_schema={},
            max_tokens=1024,
            temperature=0.3,
            challenge_confidence_floor=0.3,
            ground_always=False,
        )

    result = asyncio.run(_run())
    assert result.passes_executed == 1
    assert llm.call_count == 1


@pytest.mark.asyncio
async def test_total_tokens_accumulated():
    """Total tokens should sum across all passes."""
    p1 = _make_conclusion(archetype="pricing_shock", confidence=0.75)
    p2 = _make_conclusion(archetype="leadership_redesign", confidence=0.70)
    p3 = _make_conclusion(archetype="leadership_redesign", confidence=0.72)
    llm = MockLLM([p1, p2, p3])
    result = await multi_pass_reason(
        llm=llm,
        system_prompt="test",
        evidence_payload={
            "evidence": {
                "pain_categories": [{"category": "ux", "count": 15}],
                "total_reviews": 50,
            }
        },
        json_schema={},
        max_tokens=1024,
        temperature=0.3,
    )
    # Each mock call returns 100 + 50 = 150 tokens
    assert result.total_tokens == 150 * result.passes_executed


@pytest.mark.asyncio
async def test_normalize_fn_applied():
    """normalize_fn is applied to each pass conclusion."""
    calls = []

    def track_normalize(c):
        calls.append(c.get("archetype"))
        return c

    p1 = _make_conclusion(archetype="pricing_shock", confidence=0.25)
    llm = MockLLM([p1])
    await multi_pass_reason(
        llm=llm,
        system_prompt="test",
        evidence_payload={"evidence": {}},
        json_schema={},
        max_tokens=1024,
        temperature=0.3,
        normalize_fn=track_normalize,
    )
    assert "pricing_shock" in calls


def test_ground_only_skips_challenge():
    """ground_only=True runs classify -> ground, no challenge."""
    p1 = _make_conclusion(archetype="pricing_shock", confidence=0.75)
    p_ground = _make_conclusion(archetype="pricing_shock", confidence=0.72)
    llm = MockLLM([p1, p_ground])
    async def _run():
        return await multi_pass_reason(
            llm=llm,
            system_prompt="test",
            evidence_payload={
                "evidence": {
                    "pain_categories": [{"category": "ux", "count": 15}],
                    "total_reviews": 50,
                }
            },
            json_schema={},
            max_tokens=1024,
            temperature=0.3,
            ground_only=True,
        )
    result = asyncio.run(_run())
    assert result.passes_executed == 2
    assert result.passes[0].pass_type == "classify"
    assert result.passes[1].pass_type == "ground"
    # No challenge pass present
    assert all(p.pass_type != "challenge" for p in result.passes)
    assert llm.call_count == 2


def test_ground_only_when_disabled_is_single_pass():
    """ground_only + enabled=False -> single classify pass only."""
    p1 = _make_conclusion(archetype="pricing_shock", confidence=0.75)
    llm = MockLLM([p1])
    async def _run():
        return await multi_pass_reason(
            llm=llm,
            system_prompt="test",
            evidence_payload={"evidence": {}},
            json_schema={},
            max_tokens=1024,
            temperature=0.3,
            enabled=False,
            ground_only=True,
        )
    result = asyncio.run(_run())
    assert result.passes_executed == 1
    assert llm.call_count == 1
