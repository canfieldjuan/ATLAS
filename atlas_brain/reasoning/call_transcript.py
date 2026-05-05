"""Call-transcript reasoning domain — second-domain proof-of-life.

This module demonstrates that the typed-envelope abstraction shipped in
M5-alpha (#211) and validated by the vendor-pressure specialization in
M5-beta (#215) is genuinely domain-agnostic. ``call_transcript`` is a
second concrete domain registered against the same
``DomainReasoningResult`` envelope and the same producer/consumer
Protocols, with **zero changes** to ``extracted_reasoning_core``.

Pattern parity with ``vendor_pressure``:

    vendor_pressure                    call_transcript
    ---------------                    ---------------
    VendorOpportunitySubject           CallTranscriptSubject
    VendorPressurePayload              CallTranscriptPayload
    vendor_pressure_result_from_entry  call_transcript_result_from_entry
    VendorPressureConsumer             CallTranscriptConsumer

Both register through the same ``register_domain`` entry point and emit
the same envelope type; only the domain-specific ``domain_payload``
shape and the consumer field-name mapping differ.

Scope: this is a *proof-of-life* slice, not a full production wiring.
It proves the abstraction generalizes; it does not pull transcripts
from any real source. A future product PR can build a producer that
takes a real transcript -> ``call_transcript_result_from_entry`` ->
overlay fields on a transcript-detail API/UI.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from extracted_reasoning_core.domains import (
    DomainReasoningResult,
    ReasoningConsumerPort,
    ReferenceIds,
    register_domain,
)


@dataclass(frozen=True)
class CallTranscriptSubject:
    """A recorded conversation that reasoning runs against.

    Satisfies ``extracted_reasoning_core.domains.ReasoningSubject``
    structurally (``id`` / ``domain`` / ``payload``). The ``payload``
    carries call-shaped metadata (call leg ids, participant emails,
    duration, source provider, etc.) that the producer needs.
    """

    id: str
    domain: str = "call_transcript"
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CallTranscriptPayload:
    """Call-transcript-specific payload carried in ``domain_payload``.

    Universal fields (confidence, executive_summary, key_signals,
    falsification_conditions, ...) live on the envelope itself. This
    payload is for transcript semantics that don't fit a universal
    slot:

    - ``topics`` -- ranked list of conversation topics surfaced.
    - ``sentiment`` -- categorical sentiment label
      (``"positive"`` / ``"neutral"`` / ``"negative"`` / ``None``).
    - ``action_items`` -- explicit follow-ups extracted from the call.
    - ``participants`` -- normalized participant identifiers (emails,
      phone numbers, internal user ids) the conversation involved.

    A vendor-pressure consumer would never read these; a transcript
    consumer never reads ``wedge``. That separation is the point of
    the typed envelope.
    """

    topics: tuple[str, ...] = ()
    sentiment: str | None = None
    action_items: tuple[str, ...] = ()
    participants: tuple[str, ...] = ()


def call_transcript_result_from_entry(
    entry: Mapping[str, Any] | None,
    *,
    subject_id: str = "",
) -> DomainReasoningResult[CallTranscriptPayload]:
    """Build a typed reasoning result from a flat transcript-entry dict.

    Mirrors :func:`atlas_brain.reasoning.vendor_pressure.vendor_pressure_result_from_entry`
    -- pure dict-in / envelope-out conversion that tests can drive
    directly with synthetic entries.

    Universal envelope fields (``confidence``, ``executive_summary``,
    ``key_signals``, ``uncertainty_sources``, ``falsification_conditions``,
    ``mode``, ``risk_level``, ``archetype``, ``reference_ids``) are
    populated from same-named keys in ``entry`` when present. The
    domain-specific ``CallTranscriptPayload`` is built from
    ``topics`` / ``sentiment`` / ``action_items`` / ``participants``
    keys.

    A ``None`` or empty entry produces a result with ``confidence=None``
    and empty-tuple list fields, matching the envelope's sparse contract.
    """
    entry = entry or {}

    def _as_tuple(key: str) -> tuple[str, ...]:
        raw = entry.get(key) or []
        if not isinstance(raw, list):
            return ()
        return tuple(str(item) for item in raw if item is not None)

    metric_ids = entry.get("metric_ids") or []
    witness_ids = entry.get("witness_ids") or []
    if not isinstance(metric_ids, list):
        metric_ids = []
    if not isinstance(witness_ids, list):
        witness_ids = []

    return DomainReasoningResult(
        subject_id=subject_id,
        domain="call_transcript",
        confidence=entry.get("confidence"),
        domain_payload=CallTranscriptPayload(
            topics=_as_tuple("topics"),
            sentiment=entry.get("sentiment"),
            action_items=_as_tuple("action_items"),
            participants=_as_tuple("participants"),
        ),
        executive_summary=entry.get("executive_summary"),
        key_signals=_as_tuple("key_signals"),
        uncertainty_sources=_as_tuple("uncertainty_sources"),
        falsification_conditions=_as_tuple("falsification_conditions"),
        mode=entry.get("mode"),
        risk_level=entry.get("risk_level"),
        archetype=entry.get("archetype"),
        reference_ids=ReferenceIds(
            metric_ids=tuple(str(m) for m in metric_ids if m is not None),
            witness_ids=tuple(str(w) for w in witness_ids if w is not None),
        ),
    )


class CallTranscriptConsumer:
    """Consumer adapter for the ``"call_transcript"`` domain.

    Satisfies ``ReasoningConsumerPort[CallTranscriptPayload]``. Projects
    the typed envelope into flat overlay dicts suitable for an
    MCP/API/UI surface that wants to render transcript reasoning. The
    field-name mapping is independent of the vendor-pressure consumer
    -- a transcript-detail API exposes ``transcript_topics``, not
    ``archetype``; ``transcript_sentiment``, not ``reasoning_risk_level``.

    Field-name mapping from envelope -> overlay output:

        envelope.confidence                  -> ``transcript_confidence``
        envelope.confidence_label            -> ``transcript_confidence_label``
        envelope.executive_summary           -> ``transcript_summary``
        envelope.key_signals                 -> ``transcript_key_signals``
        envelope.uncertainty_sources         -> ``transcript_uncertainty_sources``
        envelope.falsification_conditions    -> ``transcript_falsification_conditions``
        envelope.archetype                   -> ``transcript_archetype``
        envelope.domain_payload.topics       -> ``transcript_topics``
        envelope.domain_payload.sentiment    -> ``transcript_sentiment``
        envelope.domain_payload.action_items -> ``transcript_action_items``
        envelope.domain_payload.participants -> ``transcript_participants``

    Summary returns the four most-quoted fields for a list view; detail
    returns the full set for a transcript-detail view.
    """

    def to_summary_fields(
        self,
        result: DomainReasoningResult[CallTranscriptPayload],
    ) -> Mapping[str, Any]:
        return {
            "transcript_archetype": result.archetype,
            "transcript_confidence": result.confidence,
            "transcript_sentiment": result.domain_payload.sentiment,
            "transcript_topics": list(result.domain_payload.topics),
        }

    def to_detail_fields(
        self,
        result: DomainReasoningResult[CallTranscriptPayload],
    ) -> Mapping[str, Any]:
        return {
            "transcript_archetype": result.archetype,
            "transcript_confidence": result.confidence,
            "transcript_confidence_label": result.confidence_label,
            "transcript_summary": result.executive_summary,
            "transcript_topics": list(result.domain_payload.topics),
            "transcript_sentiment": result.domain_payload.sentiment,
            "transcript_action_items": list(result.domain_payload.action_items),
            "transcript_participants": list(result.domain_payload.participants),
            "transcript_key_signals": list(result.key_signals),
            "transcript_uncertainty_sources": list(result.uncertainty_sources),
            "transcript_falsification_conditions": list(
                result.falsification_conditions
            ),
        }


# Module-import side effect: register the domain. Idempotent on
# identical re-registration; conflict raises.
register_domain(
    "call_transcript",
    subject_type=CallTranscriptSubject,
    payload_type=CallTranscriptPayload,
)


__all__ = [
    "CallTranscriptConsumer",
    "CallTranscriptPayload",
    "CallTranscriptSubject",
    "call_transcript_result_from_entry",
]
