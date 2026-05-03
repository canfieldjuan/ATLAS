from __future__ import annotations

import pytest

import extracted_reasoning_core.api as api
from extracted_reasoning_core.api import (
    EvidenceItem,
    FalsificationResult,
    NarrativePlan,
    OutputPolicy,
    ReasoningInput,
    ReasoningPack,
    ReasoningPorts,
    ReasoningResult,
    TemporalEvidence,
    ValidationReport,
    Wedge,
)


def test_public_api_exports_reasoning_contract_types() -> None:
    expected = {
        "EvidenceItem",
        "ReasoningInput",
        "ReasoningResult",
        "ReasoningPorts",
        "ReasoningPack",
        "ValidationReport",
        "Wedge",
        "score_archetypes",
        "run_reasoning",
        "validate_reasoning_output",
    }

    assert expected <= set(api.__all__)


def test_reasoning_public_dataclasses_instantiate_with_defaults() -> None:
    evidence = EvidenceItem(
        source_type="review",
        source_id="r1",
        text="Pricing became too expensive after renewal",
    )
    reasoning_input = ReasoningInput(
        entity_id="vendor:acme",
        entity_type="vendor",
        goal="build a campaign narrative",
        evidence=(evidence,),
        pack_name="content_pipeline",
    )
    result = ReasoningResult(
        summary="Pricing pressure is the strongest wedge.",
        claims=({"claim": "price squeeze"},),
        confidence=0.82,
        tier="L2",
        state={"wedge": Wedge.PRICE_SQUEEZE.value},
    )

    assert reasoning_input.evidence == (evidence,)
    assert result.state["wedge"] == "price_squeeze"
    assert ReasoningPorts() == ReasoningPorts()
    assert ReasoningPack(name="content_pipeline").version == "v1"
    # PR-C1c amended PR #79's TemporalEvidence from a coarse
    # `Mapping[str, Any]` placeholder to the rich shape with the four
    # public sub-types. The contract now requires `vendor_name` and
    # `snapshot_days`; collections default to empty lists.
    te = TemporalEvidence(vendor_name="acme", snapshot_days=30)
    assert te.velocities == []
    assert te.trends == []
    assert te.anomalies == []
    assert te.category_baselines == []
    assert te.insufficient_data is False
    assert NarrativePlan().sections == ()
    assert FalsificationResult().should_invalidate is False
    assert OutputPolicy().require_citations is True


def test_validation_report_has_explicit_terminal_verdict() -> None:
    passed = ValidationReport(passed=True)
    failed = ValidationReport(
        passed=False,
        blockers=("missing citation",),
        warnings=("low confidence",),
    )

    assert passed.passed is True
    assert passed.blockers == ()
    assert failed.passed is False
    assert failed.blockers == ("missing citation",)
    assert failed.warnings == ("low confidence",)


def test_stubbed_public_entry_points_fail_closed_until_consolidated() -> None:
    evidence = EvidenceItem(source_type="review", source_id="r1")
    reasoning_input = ReasoningInput(
        entity_id="vendor:acme",
        entity_type="vendor",
        goal="test",
        evidence=(evidence,),
    )
    result = ReasoningResult(
        summary="",
        claims=(),
        confidence=0.0,
        tier="L1",
        state={},
    )

    sync_calls = [
        lambda: api.score_archetypes({}),
        lambda: api.evaluate_evidence({}),
        lambda: api.build_temporal_evidence(()),
        lambda: api.build_narrative_plan({}, pack=ReasoningPack(name="default")),
        lambda: api.compute_evidence_hash({}),
        lambda: api.build_semantic_cache_key(reasoning_input, tier="L1"),
        lambda: api.load_reasoning_pack("content_pipeline"),
        lambda: api.validate_reasoning_output(result),
    ]

    for call in sync_calls:
        with pytest.raises(NotImplementedError):
            call()


@pytest.mark.asyncio
async def test_stubbed_async_entry_points_fail_closed_until_consolidated() -> None:
    evidence = EvidenceItem(source_type="review", source_id="r1")
    reasoning_input = ReasoningInput(
        entity_id="vendor:acme",
        entity_type="vendor",
        goal="test",
        evidence=(evidence,),
    )

    with pytest.raises(NotImplementedError):
        await api.run_reasoning(reasoning_input)

    with pytest.raises(NotImplementedError):
        await api.continue_reasoning({}, {})

    with pytest.raises(NotImplementedError):
        await api.check_falsification({}, (evidence,))
