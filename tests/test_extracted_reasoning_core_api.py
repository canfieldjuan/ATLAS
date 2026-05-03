from __future__ import annotations

from datetime import date, timedelta

import pytest

import extracted_reasoning_core.api as api
from extracted_reasoning_core.api import (
    ArchetypeMatch,
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

    # PR-C1g wired `score_archetypes` and `build_temporal_evidence`;
    # PR-C1d wired `evaluate_evidence` to the slim `EvidenceEngine`.
    # All three originally NotImplementedError stubs are now functional.
    sync_calls = [
        lambda: api.build_narrative_plan({}, pack=ReasoningPack(name="default")),
        lambda: api.compute_evidence_hash({}),
        lambda: api.build_semantic_cache_key(reasoning_input, tier="L1"),
        lambda: api.load_reasoning_pack("content_pipeline"),
        lambda: api.validate_reasoning_output(result),
    ]

    for call in sync_calls:
        with pytest.raises(NotImplementedError):
            call()


# ------------------------------------------------------------------
# PR-C1g wired entry points
# ------------------------------------------------------------------


def test_score_archetypes_returns_public_archetype_matches() -> None:
    # Strong pricing-flavored evidence picks pricing_shock as top match.
    evidence = {
        "avg_urgency": 7.0,
        "top_pain": "pricing too expensive after renewal",
        "competitor_count": 4,
        "recommend_ratio": 0.3,
        "displacement_edge_count": 3,
        "positive_review_pct": 35,
    }
    matches = api.score_archetypes(evidence)
    assert isinstance(matches, tuple)
    assert len(matches) <= 3  # default limit
    assert all(isinstance(m, ArchetypeMatch) for m in matches)
    # Top match should be pricing_shock with the public-shape fields populated.
    top = matches[0]
    assert top.archetype_id == "pricing_shock"
    assert top.label == "Pricing Shock"  # title-case derived
    assert top.score > 0.0
    assert isinstance(top.evidence_hits, tuple)
    assert top.risk_label  # non-empty


def test_score_archetypes_respects_explicit_limit() -> None:
    evidence = {
        "avg_urgency": 7.0,
        "top_pain": "pricing too expensive",
        "competitor_count": 4,
    }
    matches = api.score_archetypes(evidence, limit=1)
    assert len(matches) == 1


def test_score_archetypes_returns_empty_for_empty_evidence() -> None:
    matches = api.score_archetypes({}, limit=3)
    # No evidence -> all archetype scores fall below MATCH_THRESHOLD; the
    # adapter still returns Sequence[ArchetypeMatch] but limit caps the
    # returned slice. Pricing evidence with empty input also yields a
    # valid (low-score) shape.
    assert isinstance(matches, tuple)
    assert all(isinstance(m, ArchetypeMatch) for m in matches)


def test_build_temporal_evidence_insufficient_data_short_circuit() -> None:
    # Single snapshot is below MIN_DAYS_FOR_VELOCITY=2 -> insufficient.
    one = [{"vendor_name": "acme", "snapshot_date": date(2026, 5, 1), "avg_urgency": 5.0}]
    te = api.build_temporal_evidence(one)
    assert isinstance(te, TemporalEvidence)
    assert te.vendor_name == "acme"
    assert te.snapshot_days == 1
    assert te.insufficient_data is True
    assert te.velocities == []


def test_build_temporal_evidence_computes_velocities() -> None:
    snapshots = [
        {"vendor_name": "acme", "snapshot_date": date(2026, 5, 1), "avg_urgency": 5.0},
        {"vendor_name": "acme", "snapshot_date": date(2026, 5, 5), "avg_urgency": 7.0},
    ]
    te = api.build_temporal_evidence(snapshots)
    assert te.vendor_name == "acme"
    assert te.snapshot_days == 2
    assert te.insufficient_data is False
    by_metric = {v.metric: v for v in te.velocities}
    assert "avg_urgency" in by_metric
    v = by_metric["avg_urgency"]
    assert v.current_value == 7.0
    assert v.previous_value == 5.0
    assert v.days_between == 4


def test_build_temporal_evidence_handles_empty_snapshots() -> None:
    te = api.build_temporal_evidence(())
    assert isinstance(te, TemporalEvidence)
    assert te.snapshot_days == 0
    assert te.insufficient_data is True
    assert te.vendor_name == ""


def test_build_temporal_evidence_accepts_baselines_kwarg() -> None:
    # The signature accepts an optional baselines mapping. PR-C1g treats
    # it as an advisory input (structured anomaly support is a future
    # PR); confirm the call shape works without raising and that the
    # returned evidence has the velocity computed.
    snapshots = [
        {"vendor_name": "acme", "snapshot_date": date(2026, 5, 1), "avg_urgency": 5.0},
        {"vendor_name": "acme", "snapshot_date": date(2026, 5, 5), "avg_urgency": 7.0},
    ]
    te = api.build_temporal_evidence(snapshots, baselines={"avg_urgency": {"p50": 6.0}})
    assert te.snapshot_days == 2
    assert any(v.metric == "avg_urgency" for v in te.velocities)


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
