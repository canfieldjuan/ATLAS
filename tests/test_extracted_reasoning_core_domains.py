"""Tests for extracted_reasoning_core.domains.

Validates the domain-agnostic reasoning protocols, the typed result
envelope, the lineage block, and the domain registry contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from extracted_reasoning_core import domains as _domains
from extracted_reasoning_core.domains import (
    DomainDescriptor,
    DomainReasoningResult,
    ReasoningConsumerPort,
    ReasoningProducerPort,
    ReasoningSubject,
    ReferenceIds,
    _project_universal_fields,
    get_domain,
    list_domains,
    register_domain,
)


# ---------------------------------------------------------------------
# Fixtures: tiny sample subject/payload pair for a hypothetical domain
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class _SamplePayload:
    """Stand-in domain payload for tests."""

    headline: str
    score: float


@dataclass(frozen=True)
class _SampleSubject:
    """Stand-in subject. Satisfies ReasoningSubject structurally."""

    id: str
    domain: str
    payload: Mapping[str, Any]


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Snapshot the domain registry, give the test a clean slate, then restore.

    Earlier versions cleared before AND after, but that wiped registrations
    other modules made via import-time side effects (e.g.
    ``atlas_brain.reasoning.vendor_pressure`` calling ``register_domain``
    when imported). When the full extracted-pipeline check runs both this
    file and the vendor-pressure tests in the same process, those tests
    would then look up ``get_domain('vendor_pressure')`` and find nothing.

    Snapshot-and-restore preserves any other module's registrations across
    this file's test runs while still giving each individual test a clean
    registry to validate its own register/get/list semantics.
    """
    snapshot = dict(_domains._REGISTRY)
    _domains._REGISTRY.clear()
    try:
        yield
    finally:
        _domains._REGISTRY.clear()
        _domains._REGISTRY.update(snapshot)


# ---------------------------------------------------------------------
# Subject Protocol
# ---------------------------------------------------------------------


def test_reasoning_subject_protocol_is_runtime_checkable() -> None:
    subject = _SampleSubject(id="opp-1", domain="vendor_pressure", payload={"x": 1})
    assert isinstance(subject, ReasoningSubject)


def test_reasoning_subject_rejects_missing_attributes() -> None:
    class _Bad:
        id = "x"
        # missing 'domain' and 'payload'

    assert not isinstance(_Bad(), ReasoningSubject)


# ---------------------------------------------------------------------
# ReferenceIds + DomainReasoningResult
# ---------------------------------------------------------------------


def test_reference_ids_defaults_are_empty() -> None:
    refs = ReferenceIds()
    assert refs.metric_ids == ()
    assert refs.witness_ids == ()


def test_reference_ids_carries_provided_values() -> None:
    refs = ReferenceIds(metric_ids=("m1", "m2"), witness_ids=("w1",))
    assert refs.metric_ids == ("m1", "m2")
    assert refs.witness_ids == ("w1",)


def test_domain_reasoning_result_required_fields() -> None:
    payload = _SamplePayload(headline="renewal pressure", score=0.82)
    result: DomainReasoningResult[_SamplePayload] = DomainReasoningResult(
        subject_id="opp-1",
        domain="vendor_pressure",
        confidence=0.82,
        domain_payload=payload,
    )

    assert result.subject_id == "opp-1"
    assert result.domain == "vendor_pressure"
    assert result.confidence == pytest.approx(0.82)
    assert result.domain_payload is payload
    # Optional fields default to None / empty tuples / empty ReferenceIds
    assert result.confidence_label is None
    assert result.as_of is None
    assert result.executive_summary is None
    assert result.key_signals == ()
    assert result.uncertainty_sources == ()
    assert result.falsification_conditions == ()
    assert result.mode is None
    assert result.risk_level is None
    assert result.archetype is None
    assert result.reference_ids == ReferenceIds()


def test_domain_reasoning_result_carries_optional_fields() -> None:
    result = DomainReasoningResult(
        subject_id="opp-1",
        domain="vendor_pressure",
        confidence=0.9,
        domain_payload=_SamplePayload(headline="x", score=0.9),
        confidence_label="high",
        as_of="2026-05-05T00:00:00Z",
        executive_summary="Acme is reviewing vendors before renewal.",
        key_signals=("pricing_mentions", "exec_change"),
        uncertainty_sources=("limited transcript coverage",),
        falsification_conditions=("renewal signed within 7 days",),
        mode="synthesis",
        risk_level="high",
        archetype="renewal_at_risk",
        reference_ids=ReferenceIds(metric_ids=("m1",), witness_ids=("w1", "w2")),
    )

    assert result.confidence_label == "high"
    assert result.executive_summary == "Acme is reviewing vendors before renewal."
    assert result.key_signals == ("pricing_mentions", "exec_change")
    assert result.uncertainty_sources == ("limited transcript coverage",)
    assert result.falsification_conditions == ("renewal signed within 7 days",)
    assert result.mode == "synthesis"
    assert result.risk_level == "high"
    assert result.archetype == "renewal_at_risk"
    assert result.reference_ids.metric_ids == ("m1",)
    assert result.reference_ids.witness_ids == ("w1", "w2")


def test_domain_reasoning_result_is_frozen() -> None:
    result = DomainReasoningResult(
        subject_id="opp-1",
        domain="vendor_pressure",
        confidence=0.5,
        domain_payload=_SamplePayload(headline="x", score=0.5),
    )
    with pytest.raises(Exception):
        result.confidence = 0.99  # type: ignore[misc]


# ---------------------------------------------------------------------
# Producer / Consumer Protocols
# ---------------------------------------------------------------------


class _SampleProducer:
    """Concrete class that satisfies ReasoningProducerPort structurally."""

    async def produce(
        self,
        subject: _SampleSubject,
        /,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> DomainReasoningResult[_SamplePayload] | None:
        if subject.domain != "vendor_pressure":
            return None
        return DomainReasoningResult(
            subject_id=subject.id,
            domain=subject.domain,
            confidence=0.7,
            domain_payload=_SamplePayload(headline="ok", score=0.7),
        )


class _SampleConsumer:
    """Concrete class that satisfies ReasoningConsumerPort structurally."""

    def to_summary_fields(
        self,
        result: DomainReasoningResult[_SamplePayload],
    ) -> Mapping[str, Any]:
        return {
            "headline": result.domain_payload.headline,
            "confidence": result.confidence,
        }

    def to_detail_fields(
        self,
        result: DomainReasoningResult[_SamplePayload],
    ) -> Mapping[str, Any]:
        return {
            "headline": result.domain_payload.headline,
            "score": result.domain_payload.score,
            "confidence": result.confidence,
            "executive_summary": result.executive_summary,
        }


def test_producer_satisfies_protocol() -> None:
    assert isinstance(_SampleProducer(), ReasoningProducerPort)


def test_consumer_satisfies_protocol() -> None:
    assert isinstance(_SampleConsumer(), ReasoningConsumerPort)


@pytest.mark.asyncio
async def test_producer_returns_none_for_other_domain() -> None:
    producer = _SampleProducer()
    out = await producer.produce(
        _SampleSubject(id="t-1", domain="call_transcript", payload={}),
    )
    assert out is None


@pytest.mark.asyncio
async def test_producer_returns_typed_result() -> None:
    producer = _SampleProducer()
    out = await producer.produce(
        _SampleSubject(id="opp-1", domain="vendor_pressure", payload={}),
    )
    assert out is not None
    assert out.subject_id == "opp-1"
    assert out.domain_payload.score == pytest.approx(0.7)


def test_consumer_projections_are_independent() -> None:
    consumer = _SampleConsumer()
    result = DomainReasoningResult(
        subject_id="opp-1",
        domain="vendor_pressure",
        confidence=0.7,
        domain_payload=_SamplePayload(headline="renewal pressure", score=0.7),
        executive_summary="Sample summary.",
    )
    summary = consumer.to_summary_fields(result)
    detail = consumer.to_detail_fields(result)

    assert summary == {"headline": "renewal pressure", "confidence": 0.7}
    assert detail["score"] == pytest.approx(0.7)
    assert detail["executive_summary"] == "Sample summary."
    # Detail should be a strict superset of summary's domain field.
    assert detail["headline"] == summary["headline"]


# ---------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------


def test_register_domain_returns_descriptor() -> None:
    desc = register_domain(
        "vendor_pressure",
        subject_type=_SampleSubject,
        payload_type=_SamplePayload,
    )
    assert isinstance(desc, DomainDescriptor)
    assert desc.name == "vendor_pressure"
    assert desc.subject_type is _SampleSubject
    assert desc.payload_type is _SamplePayload


def test_get_domain_returns_registered() -> None:
    register_domain(
        "vendor_pressure",
        subject_type=_SampleSubject,
        payload_type=_SamplePayload,
    )
    desc = get_domain("vendor_pressure")
    assert desc is not None
    assert desc.name == "vendor_pressure"


def test_get_domain_missing_returns_none() -> None:
    assert get_domain("never_registered") is None


def test_register_domain_is_idempotent_on_identical_types() -> None:
    a = register_domain(
        "vendor_pressure",
        subject_type=_SampleSubject,
        payload_type=_SamplePayload,
    )
    b = register_domain(
        "vendor_pressure",
        subject_type=_SampleSubject,
        payload_type=_SamplePayload,
    )
    assert a == b


def test_register_domain_conflict_raises() -> None:
    register_domain(
        "vendor_pressure",
        subject_type=_SampleSubject,
        payload_type=_SamplePayload,
    )

    @dataclass(frozen=True)
    class _OtherPayload:
        x: int

    with pytest.raises(ValueError, match="already registered"):
        register_domain(
            "vendor_pressure",
            subject_type=_SampleSubject,
            payload_type=_OtherPayload,
        )


def test_register_domain_empty_name_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        register_domain("", subject_type=_SampleSubject, payload_type=_SamplePayload)


def test_list_domains_returns_sorted_tuple() -> None:
    register_domain(
        "vendor_pressure",
        subject_type=_SampleSubject,
        payload_type=_SamplePayload,
    )
    register_domain(
        "call_transcript",
        subject_type=_SampleSubject,
        payload_type=_SamplePayload,
    )
    register_domain(
        "company_entity",
        subject_type=_SampleSubject,
        payload_type=_SamplePayload,
    )
    domains = list_domains()
    assert isinstance(domains, tuple)
    names = [d.name for d in domains]
    assert names == ["call_transcript", "company_entity", "vendor_pressure"]


# ---------------------------------------------------------------------
# Universal-field projection helper
# ---------------------------------------------------------------------


def test_project_universal_fields_includes_all_universal_keys() -> None:
    result = DomainReasoningResult(
        subject_id="opp-1",
        domain="vendor_pressure",
        confidence=0.7,
        domain_payload=_SamplePayload(headline="x", score=0.7),
        confidence_label="medium",
        as_of="2026-05-05T00:00:00Z",
        executive_summary="exec",
        key_signals=("a", "b"),
        uncertainty_sources=("u1",),
        falsification_conditions=("f1",),
        mode="synthesis",
        risk_level="medium",
        archetype="renewal_at_risk",
        reference_ids=ReferenceIds(metric_ids=("m1",), witness_ids=("w1",)),
    )
    flat = _project_universal_fields(result)
    assert flat["subject_id"] == "opp-1"
    assert flat["domain"] == "vendor_pressure"
    assert flat["confidence"] == pytest.approx(0.7)
    assert flat["confidence_label"] == "medium"
    assert flat["executive_summary"] == "exec"
    assert flat["key_signals"] == ["a", "b"]
    assert flat["uncertainty_sources"] == ["u1"]
    assert flat["falsification_conditions"] == ["f1"]
    assert flat["reference_ids"] == {"metric_ids": ["m1"], "witness_ids": ["w1"]}
    # No domain_payload in the universal projection -- domain-specific
    # consumers add that themselves.
    assert "domain_payload" not in flat


def test_project_universal_fields_returns_fresh_dict() -> None:
    result = DomainReasoningResult(
        subject_id="opp-1",
        domain="vendor_pressure",
        confidence=0.7,
        domain_payload=_SamplePayload(headline="x", score=0.7),
    )
    a = _project_universal_fields(result)
    b = _project_universal_fields(result)
    assert a is not b
    a["mutated"] = True
    assert "mutated" not in b
