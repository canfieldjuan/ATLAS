"""Tests for the vendor-pressure specialization of the typed envelope.

Validates that ``atlas_brain.reasoning.vendor_pressure`` correctly
expresses the existing churn / vendor-pressure reasoning flow as a
``DomainReasoningResult[VendorPressurePayload]`` and that its consumer
projection matches the legacy ``_b2b_reasoning_consumer_adapter``
output bit-for-bit (including the sparse-entry contract from PR #184).

These tests deliberately exercise the ``_from_entry`` path instead of
the synthesis-view wrapper. The wrapper does a deferred import of
``atlas_brain.autonomous.tasks._b2b_synthesis_reader`` (which pulls
in the storage/scheduler chain), and that's not in the standalone-CI
dep set. The dict-in path covers the same logic without crossing that
boundary; the wrapper itself is just a thin import + delegate.
"""

from __future__ import annotations

import pytest

from atlas_brain.reasoning.vendor_pressure import (
    VendorOpportunitySubject,
    VendorPressureConsumer,
    VendorPressurePayload,
    vendor_pressure_result_from_entry,
    vendor_pressure_result_from_synthesis_view,
)
from extracted_reasoning_core.domains import (
    DomainReasoningResult,
    ReasoningConsumerPort,
    ReasoningSubject,
    get_domain,
)


# ---------------------------------------------------------------------
# Subject / payload basic shape
# ---------------------------------------------------------------------


def test_vendor_opportunity_subject_satisfies_reasoning_subject_protocol() -> None:
    subject = VendorOpportunitySubject(id="opp-1", payload={"vendor_name": "Acme"})
    assert isinstance(subject, ReasoningSubject)
    assert subject.id == "opp-1"
    assert subject.domain == "vendor_pressure"
    assert subject.payload == {"vendor_name": "Acme"}


def test_vendor_pressure_payload_default() -> None:
    payload = VendorPressurePayload()
    assert payload.wedge is None


def test_vendor_pressure_payload_with_wedge() -> None:
    payload = VendorPressurePayload(wedge="renewal_pressure")
    assert payload.wedge == "renewal_pressure"


# ---------------------------------------------------------------------
# Domain registration (module import side effect)
# ---------------------------------------------------------------------


def test_vendor_pressure_domain_is_registered() -> None:
    desc = get_domain("vendor_pressure")
    assert desc is not None
    assert desc.name == "vendor_pressure"
    assert desc.subject_type is VendorOpportunitySubject
    assert desc.payload_type is VendorPressurePayload


# ---------------------------------------------------------------------
# vendor_pressure_result_from_entry
# ---------------------------------------------------------------------


def test_result_from_full_entry() -> None:
    entry = {
        "archetype": "price_squeeze",
        "confidence": 0.82,
        "mode": "synthesis",
        "risk_level": "high",
        "executive_summary": "Acme is reviewing vendors before renewal.",
        "key_signals": ["pricing_mentions", "exec_change"],
        "uncertainty_sources": ["limited transcript coverage"],
        "falsification_conditions": ["renewal signed within 7 days"],
    }

    result = vendor_pressure_result_from_entry(entry, subject_id="opp-1")

    assert isinstance(result, DomainReasoningResult)
    assert result.subject_id == "opp-1"
    assert result.domain == "vendor_pressure"
    assert result.confidence == pytest.approx(0.82)
    assert result.mode == "synthesis"
    assert result.risk_level == "high"
    assert result.archetype == "price_squeeze"
    assert result.executive_summary == "Acme is reviewing vendors before renewal."
    assert result.key_signals == ("pricing_mentions", "exec_change")
    assert result.uncertainty_sources == ("limited transcript coverage",)
    assert result.falsification_conditions == ("renewal signed within 7 days",)
    assert result.domain_payload.wedge == "price_squeeze"


def test_result_from_empty_entry() -> None:
    """Sparse entries must produce stable result fields (PR #184 contract)."""
    result = vendor_pressure_result_from_entry({})

    assert result.subject_id == ""
    assert result.domain == "vendor_pressure"
    assert result.confidence is None
    assert result.mode is None
    assert result.risk_level is None
    assert result.archetype is None
    assert result.executive_summary is None
    assert result.key_signals == ()
    assert result.uncertainty_sources == ()
    assert result.falsification_conditions == ()
    assert result.domain_payload.wedge is None


def test_result_from_none_entry_is_sparse() -> None:
    """``None`` entry -> same result shape as empty dict."""
    result = vendor_pressure_result_from_entry(None)
    assert result.confidence is None
    assert result.archetype is None
    assert result.key_signals == ()


def test_result_handles_explicit_null_lists() -> None:
    """Producer dict may set list keys to explicit None; envelope still gets empty tuples."""
    entry = {
        "archetype": "support_erosion",
        "confidence": 0.44,
        "mode": "synthesis",
        "risk_level": "medium",
        "executive_summary": "summary",
        "key_signals": None,
        "uncertainty_sources": None,
        "falsification_conditions": None,
    }

    result = vendor_pressure_result_from_entry(entry)

    assert result.key_signals == ()
    assert result.uncertainty_sources == ()
    assert result.falsification_conditions == ()


# ---------------------------------------------------------------------
# vendor_pressure_result_from_synthesis_view (thin wrapper)
# ---------------------------------------------------------------------


def test_synthesis_view_wrapper_short_circuits_on_none() -> None:
    """``view=None`` must not trigger the deferred synthesis-reader import."""
    result = vendor_pressure_result_from_synthesis_view(None, subject_id="opp-1")
    assert result.subject_id == "opp-1"
    assert result.confidence is None
    assert result.archetype is None


# ---------------------------------------------------------------------
# VendorPressureConsumer projections
# ---------------------------------------------------------------------


def test_consumer_satisfies_reasoning_consumer_port() -> None:
    assert isinstance(VendorPressureConsumer(), ReasoningConsumerPort)


def _full_result() -> DomainReasoningResult[VendorPressurePayload]:
    return DomainReasoningResult(
        subject_id="opp-1",
        domain="vendor_pressure",
        confidence=0.82,
        domain_payload=VendorPressurePayload(wedge="price_squeeze"),
        mode="synthesis",
        risk_level="high",
        archetype="price_squeeze",
        executive_summary="exec",
        key_signals=("a", "b"),
        uncertainty_sources=("u1",),
        falsification_conditions=("f1",),
    )


def test_consumer_summary_projection() -> None:
    consumer = VendorPressureConsumer()
    out = consumer.to_summary_fields(_full_result())
    assert out == {
        "archetype": "price_squeeze",
        "archetype_confidence": 0.82,
        "reasoning_mode": "synthesis",
        "reasoning_risk_level": "high",
    }


def test_consumer_detail_projection() -> None:
    consumer = VendorPressureConsumer()
    out = consumer.to_detail_fields(_full_result())
    assert out == {
        "archetype": "price_squeeze",
        "archetype_confidence": 0.82,
        "reasoning_mode": "synthesis",
        "reasoning_risk_level": "high",
        "reasoning_executive_summary": "exec",
        "reasoning_key_signals": ["a", "b"],
        "reasoning_uncertainty_sources": ["u1"],
        "falsification_conditions": ["f1"],
    }


def test_consumer_sparse_result_keeps_stable_keys() -> None:
    """Sparse result -> all 8 detail keys present, scalars None, lists empty.

    Mirrors the PR #184 sparse-entry contract that signals.py relies on.
    """
    sparse = DomainReasoningResult(
        subject_id="",
        domain="vendor_pressure",
        confidence=None,
        domain_payload=VendorPressurePayload(),
    )
    consumer = VendorPressureConsumer()

    summary = consumer.to_summary_fields(sparse)
    detail = consumer.to_detail_fields(sparse)

    assert summary == {
        "archetype": None,
        "archetype_confidence": None,
        "reasoning_mode": None,
        "reasoning_risk_level": None,
    }
    assert set(detail.keys()) == {
        "archetype",
        "archetype_confidence",
        "reasoning_mode",
        "reasoning_risk_level",
        "reasoning_executive_summary",
        "reasoning_key_signals",
        "reasoning_uncertainty_sources",
        "falsification_conditions",
    }
    assert detail["archetype"] is None
    assert detail["archetype_confidence"] is None
    assert detail["reasoning_mode"] is None
    assert detail["reasoning_risk_level"] is None
    assert detail["reasoning_executive_summary"] is None
    assert detail["reasoning_key_signals"] == []
    assert detail["reasoning_uncertainty_sources"] == []
    assert detail["falsification_conditions"] == []
