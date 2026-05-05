"""Tests for the call-transcript reasoning domain (M5-gamma).

Validates that the typed-envelope abstraction generalizes to a second
domain — ``call_transcript`` registers, builds, and projects without
touching ``extracted_reasoning_core`` or ``vendor_pressure``. Also
demonstrates **coexistence**: both domains live in the registry at
once and their consumers project to entirely different overlay-field
sets from envelopes of the same Python type.
"""

from __future__ import annotations

import pytest

from atlas_brain.reasoning.call_transcript import (
    CallTranscriptConsumer,
    CallTranscriptPayload,
    CallTranscriptSubject,
    call_transcript_result_from_entry,
)
from atlas_brain.reasoning.vendor_pressure import (
    VendorOpportunitySubject,
    VendorPressureConsumer,
    VendorPressurePayload,
    vendor_pressure_result_from_entry,
)
from extracted_reasoning_core.domains import (
    DomainReasoningResult,
    ReasoningConsumerPort,
    ReasoningSubject,
    get_domain,
    list_domains,
)


# ---------------------------------------------------------------------
# Subject / payload basic shape
# ---------------------------------------------------------------------


def test_call_transcript_subject_satisfies_reasoning_subject_protocol() -> None:
    subject = CallTranscriptSubject(
        id="call-42",
        payload={
            "duration_seconds": 1820,
            "participants": ["alice@example.com", "bob@acme.com"],
        },
    )
    assert isinstance(subject, ReasoningSubject)
    assert subject.id == "call-42"
    assert subject.domain == "call_transcript"
    assert subject.payload["duration_seconds"] == 1820


def test_call_transcript_payload_default() -> None:
    payload = CallTranscriptPayload()
    assert payload.topics == ()
    assert payload.sentiment is None
    assert payload.action_items == ()
    assert payload.participants == ()


def test_call_transcript_payload_with_values() -> None:
    payload = CallTranscriptPayload(
        topics=("renewal", "pricing"),
        sentiment="negative",
        action_items=("send proposal",),
        participants=("alice@example.com",),
    )
    assert payload.topics == ("renewal", "pricing")
    assert payload.sentiment == "negative"


# ---------------------------------------------------------------------
# Domain registration (module import side effect)
# ---------------------------------------------------------------------


def test_call_transcript_domain_is_registered() -> None:
    desc = get_domain("call_transcript")
    assert desc is not None
    assert desc.name == "call_transcript"
    assert desc.subject_type is CallTranscriptSubject
    assert desc.payload_type is CallTranscriptPayload


def test_both_domains_coexist_in_registry() -> None:
    """The whole point of the abstraction: vendor_pressure + call_transcript
    coexist in the same registry, identified by ``domain`` string."""
    names = {d.name for d in list_domains()}
    assert {"vendor_pressure", "call_transcript"}.issubset(names)


# ---------------------------------------------------------------------
# Builder: full + sparse + lineage
# ---------------------------------------------------------------------


def test_result_from_full_entry() -> None:
    entry = {
        "archetype": "renewal_at_risk",
        "confidence": 0.74,
        "mode": "synthesis",
        "risk_level": "high",
        "executive_summary": "Customer asked twice about pricing flexibility.",
        "key_signals": ["pricing_pushback", "competitor_mention"],
        "uncertainty_sources": ["only one decision-maker on the call"],
        "falsification_conditions": ["renewal signed within 14 days"],
        "topics": ["renewal", "pricing", "competition"],
        "sentiment": "negative",
        "action_items": ["send pricing comparison", "schedule follow-up"],
        "participants": ["alice@example.com", "bob@acme.com"],
        "metric_ids": ["price_mentions", "exec_change"],
        "witness_ids": ["transcript-42-segment-7"],
    }

    result = call_transcript_result_from_entry(entry, subject_id="call-42")

    assert isinstance(result, DomainReasoningResult)
    assert result.subject_id == "call-42"
    assert result.domain == "call_transcript"
    assert result.confidence == pytest.approx(0.74)
    assert result.archetype == "renewal_at_risk"
    assert result.executive_summary == "Customer asked twice about pricing flexibility."
    assert result.key_signals == ("pricing_pushback", "competitor_mention")
    assert result.uncertainty_sources == ("only one decision-maker on the call",)
    assert result.falsification_conditions == ("renewal signed within 14 days",)
    # Domain payload carries the call-specific shape
    assert result.domain_payload.topics == ("renewal", "pricing", "competition")
    assert result.domain_payload.sentiment == "negative"
    assert result.domain_payload.action_items == (
        "send pricing comparison",
        "schedule follow-up",
    )
    assert result.domain_payload.participants == (
        "alice@example.com",
        "bob@acme.com",
    )
    # Lineage round-trips through the envelope
    assert result.reference_ids.metric_ids == ("price_mentions", "exec_change")
    assert result.reference_ids.witness_ids == ("transcript-42-segment-7",)


def test_result_from_empty_entry_is_sparse() -> None:
    result = call_transcript_result_from_entry({})

    assert result.subject_id == ""
    assert result.domain == "call_transcript"
    assert result.confidence is None
    assert result.archetype is None
    assert result.executive_summary is None
    assert result.key_signals == ()
    assert result.uncertainty_sources == ()
    assert result.falsification_conditions == ()
    assert result.domain_payload.topics == ()
    assert result.domain_payload.sentiment is None
    assert result.domain_payload.action_items == ()
    assert result.domain_payload.participants == ()
    assert result.reference_ids.metric_ids == ()
    assert result.reference_ids.witness_ids == ()


def test_result_from_none_entry_is_sparse() -> None:
    result = call_transcript_result_from_entry(None)
    assert result.confidence is None
    assert result.domain_payload.topics == ()


def test_result_handles_explicit_null_lists() -> None:
    """Producer dict may set list keys to explicit None; envelope still
    gets empty tuples (mirrors the PR #184 sparse-entry guard pattern
    used in vendor_pressure)."""
    entry = {
        "topics": None,
        "action_items": None,
        "participants": None,
        "key_signals": None,
        "metric_ids": None,
        "witness_ids": None,
    }
    result = call_transcript_result_from_entry(entry)
    assert result.domain_payload.topics == ()
    assert result.domain_payload.action_items == ()
    assert result.domain_payload.participants == ()
    assert result.key_signals == ()
    assert result.reference_ids.metric_ids == ()
    assert result.reference_ids.witness_ids == ()


def test_result_coerces_non_string_list_items_and_drops_none() -> None:
    """Defensive coercion: ``None`` items are filtered, non-strings are
    coerced via ``str()`` (e.g. ``42`` -> ``"42"``).

    Domain-payload lists keep the same shape as their producer (so
    integers in ``topics`` survive as their stringified form). Lineage
    IDs go through a tighter ``_normalize_ids`` path that additionally
    strips whitespace and drops empties — see
    :func:`test_lineage_ids_normalize_and_filter_empty`.
    """
    entry = {
        "topics": ["renewal", None, 42, "pricing"],
        "metric_ids": [None, "real_metric_id"],
    }
    result = call_transcript_result_from_entry(entry)
    assert result.domain_payload.topics == ("renewal", "42", "pricing")
    assert result.reference_ids.metric_ids == ("real_metric_id",)


def test_lineage_ids_strip_whitespace_and_drop_empty() -> None:
    """Lineage IDs are normalized: whitespace stripped, empties dropped.

    Unlike domain-payload lists (which preserve coerced strings even if
    short or padded), lineage IDs feed downstream lookups; whitespace
    or empty IDs would silently miss-match, so the envelope filters
    them at the boundary.
    """
    entry = {
        "metric_ids": ["  m1  ", "", "   ", None, "m2"],
        "witness_ids": ["w1", "  w2", ""],
    }
    result = call_transcript_result_from_entry(entry)
    assert result.reference_ids.metric_ids == ("m1", "m2")
    assert result.reference_ids.witness_ids == ("w1", "w2")


def test_lineage_prefers_nested_reference_ids_object() -> None:
    """When the entry carries a nested ``reference_ids`` object (the
    canonical synthesis-payload shape), the builder reads from there
    and the top-level ``metric_ids``/``witness_ids`` fallback is
    ignored.
    """
    entry = {
        "reference_ids": {
            "metric_ids": ["nested_m1", "nested_m2"],
            "witness_ids": ["nested_w1"],
        },
        # Top-level keys are present but should be ignored when the
        # nested object exists.
        "metric_ids": ["top_level_m_should_be_ignored"],
        "witness_ids": ["top_level_w_should_be_ignored"],
    }
    result = call_transcript_result_from_entry(entry)
    assert result.reference_ids.metric_ids == ("nested_m1", "nested_m2")
    assert result.reference_ids.witness_ids == ("nested_w1",)


def test_lineage_falls_back_to_top_level_when_nested_absent() -> None:
    """Without a nested ``reference_ids`` object, the flat shape used
    by the vendor-pressure-style entry path still round-trips.
    """
    entry = {
        "metric_ids": ["flat_m1"],
        "witness_ids": ["flat_w1"],
    }
    result = call_transcript_result_from_entry(entry)
    assert result.reference_ids.metric_ids == ("flat_m1",)
    assert result.reference_ids.witness_ids == ("flat_w1",)


# ---------------------------------------------------------------------
# CallTranscriptConsumer projection
# ---------------------------------------------------------------------


def test_consumer_satisfies_reasoning_consumer_port() -> None:
    assert isinstance(CallTranscriptConsumer(), ReasoningConsumerPort)


def _full_result() -> DomainReasoningResult[CallTranscriptPayload]:
    return DomainReasoningResult(
        subject_id="call-42",
        domain="call_transcript",
        confidence=0.74,
        domain_payload=CallTranscriptPayload(
            topics=("renewal", "pricing"),
            sentiment="negative",
            action_items=("send proposal",),
            participants=("alice@example.com",),
        ),
        confidence_label="medium",
        executive_summary="exec",
        key_signals=("a",),
        uncertainty_sources=("u1",),
        falsification_conditions=("f1",),
        archetype="renewal_at_risk",
    )


def test_consumer_summary_projection() -> None:
    out = CallTranscriptConsumer().to_summary_fields(_full_result())
    assert out == {
        "transcript_archetype": "renewal_at_risk",
        "transcript_confidence": 0.74,
        "transcript_sentiment": "negative",
        "transcript_topics": ["renewal", "pricing"],
    }


def test_consumer_detail_projection() -> None:
    out = CallTranscriptConsumer().to_detail_fields(_full_result())
    assert out == {
        "transcript_archetype": "renewal_at_risk",
        "transcript_confidence": 0.74,
        "transcript_confidence_label": "medium",
        "transcript_summary": "exec",
        "transcript_topics": ["renewal", "pricing"],
        "transcript_sentiment": "negative",
        "transcript_action_items": ["send proposal"],
        "transcript_participants": ["alice@example.com"],
        "transcript_key_signals": ["a"],
        "transcript_uncertainty_sources": ["u1"],
        "transcript_falsification_conditions": ["f1"],
    }


def test_consumer_sparse_result_keeps_stable_keys() -> None:
    sparse = DomainReasoningResult(
        subject_id="",
        domain="call_transcript",
        confidence=None,
        domain_payload=CallTranscriptPayload(),
    )
    consumer = CallTranscriptConsumer()
    summary = consumer.to_summary_fields(sparse)
    detail = consumer.to_detail_fields(sparse)

    assert summary == {
        "transcript_archetype": None,
        "transcript_confidence": None,
        "transcript_sentiment": None,
        "transcript_topics": [],
    }
    assert set(detail.keys()) == {
        "transcript_archetype",
        "transcript_confidence",
        "transcript_confidence_label",
        "transcript_summary",
        "transcript_topics",
        "transcript_sentiment",
        "transcript_action_items",
        "transcript_participants",
        "transcript_key_signals",
        "transcript_uncertainty_sources",
        "transcript_falsification_conditions",
    }


# ---------------------------------------------------------------------
# Domain isolation: vendor-pressure and call-transcript don't bleed
# ---------------------------------------------------------------------


def test_consumers_project_to_disjoint_field_sets() -> None:
    """Belt-and-suspenders proof that the typed envelope keeps domains
    isolated: a vendor-pressure consumer and a call-transcript consumer
    fed isomorphic envelopes produce overlay dicts with disjoint keys.

    This is the value of the abstraction in one test: an MCP/API
    surface can pick the right consumer for its domain without any
    coordination between the two domain modules.
    """
    vp_result = vendor_pressure_result_from_entry(
        {
            "archetype": "price_squeeze",
            "confidence": 0.82,
            "mode": "synthesis",
            "risk_level": "high",
        }
    )
    tx_result = call_transcript_result_from_entry(
        {
            "archetype": "renewal_at_risk",
            "confidence": 0.74,
            "topics": ["renewal"],
        }
    )

    vp_summary = VendorPressureConsumer().to_summary_fields(vp_result)
    tx_summary = CallTranscriptConsumer().to_summary_fields(tx_result)

    # No key overlap between the two summaries
    assert set(vp_summary.keys()).isdisjoint(set(tx_summary.keys()))


def test_subject_protocol_satisfied_by_both_subject_types() -> None:
    """Both subject types satisfy the same Protocol — that's how a
    generic dispatcher could route to either domain producer without
    knowing about either dataclass concretely."""
    subjects = (
        VendorOpportunitySubject(id="opp-1"),
        CallTranscriptSubject(id="call-42"),
    )
    for s in subjects:
        assert isinstance(s, ReasoningSubject)
