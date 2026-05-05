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
    _enrich_entry_with_view_metadata,
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
# Lineage / freshness ingestion via the entry-dict path
# ---------------------------------------------------------------------


def test_result_carries_reference_ids_from_entry() -> None:
    """Nested ``reference_ids`` round-trips through the envelope.

    The synthesis-view wrapper enriches the entry with
    ``view.reference_ids``; this test exercises the underlying
    builder with a synthetic entry to keep the test off the
    atlas_brain.autonomous import chain.
    """
    entry = {
        "archetype": "renewal_pressure",
        "confidence": 0.74,
        "reference_ids": {
            "metric_ids": ["m_pricing_mentions", "m_exec_change"],
            "witness_ids": ["w_segment_7"],
        },
    }
    result = vendor_pressure_result_from_entry(entry, subject_id="opp-1")
    assert result.reference_ids.metric_ids == (
        "m_pricing_mentions",
        "m_exec_change",
    )
    assert result.reference_ids.witness_ids == ("w_segment_7",)


def test_result_carries_as_of_from_entry() -> None:
    entry = {"as_of": "2026-05-04"}
    assert vendor_pressure_result_from_entry(entry).as_of == "2026-05-04"


def test_result_carries_confidence_label_from_entry() -> None:
    entry = {"confidence": 0.74, "confidence_label": "high"}
    result = vendor_pressure_result_from_entry(entry)
    assert result.confidence == 0.74
    assert result.confidence_label == "high"


def test_result_normalizes_lineage_ids_strip_and_drop_empty() -> None:
    """Lineage IDs are str-coerced, .strip()-ed, and empty-after-strip
    is dropped. Mirrors the call_transcript domain's normalization.
    """
    entry = {
        "reference_ids": {
            "metric_ids": ["  m1  ", "", "   ", None, "m2"],
            "witness_ids": [None, "w1"],
        },
    }
    result = vendor_pressure_result_from_entry(entry)
    assert result.reference_ids.metric_ids == ("m1", "m2")
    assert result.reference_ids.witness_ids == ("w1",)


def test_result_collapses_empty_strings_for_freshness_and_label() -> None:
    """Empty-string ``as_of`` / ``confidence_label`` collapse to None.

    Mirrors ``view.as_of_date_iso()`` returning "" when the synthesis
    has no date — consumers should see "no date", not "the empty string
    date".
    """
    entry = {"as_of": "", "confidence_label": "   "}
    result = vendor_pressure_result_from_entry(entry)
    assert result.as_of is None
    assert result.confidence_label is None


def test_result_ignores_non_mapping_reference_ids() -> None:
    """Defensive: malformed ``reference_ids`` (e.g. a list) leaves
    lineage empty rather than raising.
    """
    entry = {"reference_ids": ["m1", "m2"]}
    result = vendor_pressure_result_from_entry(entry)
    assert result.reference_ids.metric_ids == ()
    assert result.reference_ids.witness_ids == ()


# ---------------------------------------------------------------------
# _enrich_entry_with_view_metadata: pure helper, exercised with stub
# views that match the real SynthesisView API shape (reference_ids and
# as_of_date_iso are @property, confidence is a method). These tests
# stay off the atlas_brain.autonomous import chain.
# ---------------------------------------------------------------------


class _StubSynthesisViewPropertyShape:
    """Stub matching the real SynthesisView shape exactly.

    - ``reference_ids`` is a ``@property`` (returns dict).
    - ``as_of_date_iso`` is a ``@property`` (returns string).
    - ``confidence`` is a method taking ``section`` arg.
    """

    def __init__(
        self,
        *,
        reference_ids=None,
        as_of_iso="",
        confidence_label_for_causal="",
    ):
        self._reference_ids = reference_ids or {}
        self._as_of_iso = as_of_iso
        self._confidence_label = confidence_label_for_causal

    @property
    def reference_ids(self):
        return self._reference_ids

    @property
    def as_of_date_iso(self):
        return self._as_of_iso

    def confidence(self, section):
        if section == "causal_narrative":
            return self._confidence_label
        return ""


class _StubSynthesisViewCallableShape:
    """Stub modelling ``as_of_date_iso`` as a callable instead of a
    property — covers the wrapper's defensive accept-both branch so a
    future view-API refactor doesn't regress us.
    """

    def __init__(self, *, as_of_iso=""):
        self._as_of_iso = as_of_iso

    @property
    def reference_ids(self):
        return {}

    def as_of_date_iso(self):
        return self._as_of_iso

    def confidence(self, section):
        return ""


def test_enrich_with_property_shaped_view() -> None:
    """Real SynthesisView has ``as_of_date_iso`` as a @property — must
    populate ``as_of`` from the property value, not via a method call.
    """
    view = _StubSynthesisViewPropertyShape(
        reference_ids={
            "metric_ids": ["m_pricing_mentions"],
            "witness_ids": ["w_segment_7"],
        },
        as_of_iso="2026-05-04",
        confidence_label_for_causal="high",
    )

    enriched = _enrich_entry_with_view_metadata({}, view)

    assert enriched["reference_ids"] == {
        "metric_ids": ["m_pricing_mentions"],
        "witness_ids": ["w_segment_7"],
    }
    assert enriched["as_of"] == "2026-05-04"
    assert enriched["confidence_label"] == "high"


def test_enrich_with_callable_shaped_view() -> None:
    """Defensive: if the view exposes ``as_of_date_iso`` as a method
    (zero-arg callable), the wrapper still picks up the value.
    """
    view = _StubSynthesisViewCallableShape(as_of_iso="2026-05-04")
    enriched = _enrich_entry_with_view_metadata({}, view)
    assert enriched["as_of"] == "2026-05-04"


def test_enrich_skips_empty_string_as_of() -> None:
    """``view.as_of_date_iso`` returns "" when the synthesis has no
    date — that should NOT show up as ``as_of=""`` on the envelope.
    """
    view = _StubSynthesisViewPropertyShape(as_of_iso="")
    enriched = _enrich_entry_with_view_metadata({}, view)
    assert "as_of" not in enriched


def test_enrich_skips_empty_reference_ids() -> None:
    """Empty / missing reference_ids dict should leave the entry's
    ``reference_ids`` key unset (so the builder falls back to the
    sparse default).
    """
    view = _StubSynthesisViewPropertyShape(reference_ids={})
    enriched = _enrich_entry_with_view_metadata({}, view)
    assert "reference_ids" not in enriched


def test_enrich_skips_empty_confidence_label() -> None:
    view = _StubSynthesisViewPropertyShape(confidence_label_for_causal="")
    enriched = _enrich_entry_with_view_metadata({}, view)
    assert "confidence_label" not in enriched


def test_enrich_uses_setdefault_so_explicit_entry_keys_win() -> None:
    """If the entry already carries one of the view-derived fields
    (e.g. a producer that pre-computed reference_ids), the wrapper
    must NOT overwrite it.
    """
    view = _StubSynthesisViewPropertyShape(
        reference_ids={"metric_ids": ["from_view"], "witness_ids": []},
        as_of_iso="2026-05-04",
        confidence_label_for_causal="high",
    )
    entry = {
        "reference_ids": {"metric_ids": ["from_entry"], "witness_ids": []},
        "as_of": "2025-01-01",
        "confidence_label": "low",
    }
    enriched = _enrich_entry_with_view_metadata(entry, view)
    assert enriched["reference_ids"]["metric_ids"] == ["from_entry"]
    assert enriched["as_of"] == "2025-01-01"
    assert enriched["confidence_label"] == "low"


def test_enrich_handles_view_missing_attributes_gracefully() -> None:
    """A bare object with none of the expected attributes should not
    raise — the entry just stays unenriched.
    """
    enriched = _enrich_entry_with_view_metadata({}, object())
    assert enriched == {}


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
