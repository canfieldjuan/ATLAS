from __future__ import annotations

from typing import Any, Mapping

from extracted_reasoning_core.api import build_narrative_plan
from extracted_reasoning_core.types import NarrativePlan, ReasoningPack


def _state_with_claims() -> dict[str, Any]:
    return {
        "status": "completed",
        "depth": "L2",
        "raw_synthesis": {
            "summary": "Pricing pressure dominates.",
            "claims": [
                {
                    "claim_id": "c1",
                    "claim": "Renewal pricing drives displacement.",
                    "confidence": 0.85,
                    "source_ids": ["r1", "r2"],
                    "section": "drivers",
                },
                {
                    "claim_id": "c2",
                    "claim": "Onboarding friction is secondary.",
                    "confidence": 0.55,
                    "source_ids": ["r3"],
                    "section": "drivers",
                },
                {
                    "claim_id": "c3",
                    "claim": "Sales team mentions competitor X frequently.",
                    "confidence": 0.7,
                    "source_ids": ["r4"],
                    "section": "competitive",
                },
            ],
            "confidence": 0.78,
        },
    }


def test_build_narrative_plan_happy_path_orders_by_confidence_and_groups_sections() -> None:
    pack = ReasoningPack(name="default_narrative", policies={})

    plan = build_narrative_plan(_state_with_claims(), pack=pack)

    assert isinstance(plan, NarrativePlan)
    # Claims confidence-sorted (c1 0.85, c3 0.7, c2 0.55) then grouped by
    # section so renderers get section-coherent runs:
    #   drivers: c1, c2  (c1 first because it sorted ahead of c2)
    #   competitive: c3  (section first seen after c1, before c2)
    assert tuple(c["claim_id"] for c in plan.claims) == ("c1", "c2", "c3")
    section_ids = [s["id"] for s in plan.sections]
    assert "drivers" in section_ids and "competitive" in section_ids
    drivers = next(s for s in plan.sections if s["id"] == "drivers")
    assert drivers["claim_count"] == 2
    assert drivers["title"] == "Drivers"
    drivers_evidence = next(e for e in plan.evidence_requirements if e["section_id"] == "drivers")
    assert set(drivers_evidence["required_source_ids"]) == {"r1", "r2", "r3"}
    assert plan.state_hints["overall_confidence"] == 0.78
    assert plan.state_hints["claim_count"] == 3
    assert plan.state_hints["dropped_below_confidence"] == 0
    assert plan.state_hints["depth"] == "L2"


def test_build_narrative_plan_min_confidence_drops_low_claims() -> None:
    pack = ReasoningPack(name="strict", policies={"min_confidence": 0.6})

    plan = build_narrative_plan(_state_with_claims(), pack=pack)

    # c2 (0.55) drops; c1 (0.85) and c3 (0.7) survive.
    assert tuple(c["claim_id"] for c in plan.claims) == ("c1", "c3")
    assert plan.state_hints["dropped_below_confidence"] == 1


def test_build_narrative_plan_max_sections_caps_section_count() -> None:
    state = _state_with_claims()
    state["raw_synthesis"]["claims"].append({
        "claim_id": "c4",
        "claim": "Customer success is understaffed.",
        "confidence": 0.6,
        "source_ids": ["r5"],
        "section": "operations",
    })
    pack = ReasoningPack(name="capped", policies={"max_sections": 2})

    plan = build_narrative_plan(state, pack=pack)

    assert len(plan.sections) == 2
    # Sections are inserted in claim-iteration order after sorting; first two distinct
    # section names from the confidence-sorted list survive.
    section_ids = [s["id"] for s in plan.sections]
    assert section_ids == ["drivers", "competitive"]


def test_build_narrative_plan_accepts_raw_synthesis_directly_without_state_wrapper() -> None:
    raw = _state_with_claims()["raw_synthesis"]
    pack = ReasoningPack(name="default", policies={})

    plan = build_narrative_plan(raw, pack=pack)

    assert plan.state_hints["claim_count"] == 3
    assert plan.state_hints["overall_confidence"] == 0.78


def test_build_narrative_plan_empty_context_returns_empty_plan() -> None:
    pack = ReasoningPack(name="default", policies={})

    plan = build_narrative_plan({}, pack=pack)

    assert plan.claims == ()
    assert plan.sections == ()
    assert plan.evidence_requirements == ()
    assert plan.state_hints["claim_count"] == 0
    assert plan.state_hints["section_count"] == 0
