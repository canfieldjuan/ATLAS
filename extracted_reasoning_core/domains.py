"""Domain-agnostic reasoning protocols, envelope, and registry.

Reasoning is performed over many kinds of subjects -- vendors under
churn pressure, call transcripts, internal company entities, and
whatever else a host wants to plug in. The existing ``ReasoningInput``
and ``ReasoningResult`` types in ``extracted_reasoning_core.types``
already use generic ``entity_id`` / ``entity_type`` fields, but two
pieces have been missing:

1.  A **typed envelope** that pairs the universal reasoning core
    (confidence, summary, claims, lineage) with a domain-typed
    payload, so domain-specific conclusions don't have to live in an
    untyped ``state: Mapping[str, Any]`` bag.
2.  **High-level Producer / Consumer Protocols** at the domain level
    so a host can register a reasoning surface for a new domain
    purely additively, without touching the core.

This module supplies those two pieces plus a small domain registry.
Existing types in ``extracted_reasoning_core.types`` are unchanged;
M5-beta will refactor the vendor-pressure consumer adapter and the
campaign provider port to be typed specializations of the protocols
defined here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import (
    Any,
    Generic,
    NamedTuple,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from .types import ReasoningResult


PayloadT_co = TypeVar("PayloadT_co", covariant=True)
PayloadT = TypeVar("PayloadT")
SubjectT_contra = TypeVar("SubjectT_contra", contravariant=True)


@runtime_checkable
class ReasoningSubject(Protocol):
    """Anything reasoning is performed over.

    A subject identifies *what* is being reasoned about, scoped by a
    domain string. The ``payload`` carries domain-specific input fields
    (e.g. opportunity row, transcript metadata, company facts) that the
    producer needs to compute a result.

    Domain conventions:

    * ``"vendor_pressure"`` -- vendor under churn pressure (Atlas churn
      reasoning, current).
    * ``"call_transcript"`` -- a recorded conversation (future).
    * ``"company_entity"`` -- internal company record (future).

    Any class with these three attributes satisfies the protocol; this
    is ``@runtime_checkable`` so adapter wiring can validate at startup
    via ``isinstance(obj, ReasoningSubject)``.
    """

    id: str
    domain: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class ReferenceIds:
    """Universal evidence-lineage block.

    Mirrors the ``reference_ids`` block of the canonical reasoning
    payload from ``docs/reasoning_interface_contract.md``. Carried on
    every result so consumers can resolve back to source evidence
    regardless of domain.
    """

    metric_ids: tuple[str, ...] = ()
    witness_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DomainReasoningResult(Generic[PayloadT_co]):
    """Domain-agnostic reasoning result with a domain-typed payload.

    The fields outside ``domain_payload`` are universal across domains
    and form a stable consumer contract: confidence, summary, signals,
    uncertainty, falsification conditions, lineage. The
    ``domain_payload`` carries the domain-specific conclusions
    (wedge / displacement / account_signals for vendor pressure;
    topics / sentiment / action_items for call transcripts; etc.).

    This sits alongside ``ReasoningResult`` in ``types.py``: the
    existing ``ReasoningResult`` keeps its untyped ``state: Mapping``
    bag for backward compatibility; this typed envelope is the new
    contract for anything built after M5-alpha.
    """

    # ``confidence`` is required positionally to preserve the original
    # M5-alpha constructor contract -- moving it out of position would
    # silently corrupt downstream callers that use positional construction
    # (a payload would slide into ``confidence`` and vice versa). The
    # type widens to ``float | None`` because real producers have cases
    # where confidence is genuinely unknown (sparse evidence, insufficient
    # pool depth, etc.); a producer that is confident asserts a float in
    # [0, 1], one with no signal explicitly passes ``None``.
    subject_id: str
    domain: str
    confidence: float | None
    domain_payload: PayloadT_co

    # Optional universal fields -- present when the producer can compute
    # them, omitted (None / empty tuple) otherwise.
    confidence_label: str | None = None
    as_of: str | None = None  # ISO 8601 datetime
    executive_summary: str | None = None
    key_signals: tuple[str, ...] = ()
    uncertainty_sources: tuple[str, ...] = ()
    falsification_conditions: tuple[str, ...] = ()
    mode: str | None = None
    risk_level: str | None = None
    archetype: str | None = None
    reference_ids: ReferenceIds = field(default_factory=ReferenceIds)


@runtime_checkable
class ReasoningProducerPort(Protocol[SubjectT_contra, PayloadT_co]):
    """Host-owned port for producing a reasoning result for a subject.

    A producer takes a fully-formed ``ReasoningSubject`` (or any
    domain-specific subclass) and returns a ``DomainReasoningResult``
    typed for the producer's domain payload. Returning ``None`` means
    the producer found no reasoning to assert (e.g. insufficient
    evidence for the contract floor); callers fall back to defaults.

    The single ``produce`` method is async because real producers
    typically do LLM calls or DB lookups. Pure-sync producers can wrap
    in ``asyncio.to_thread`` or just declare ``async def`` and return
    immediately -- both satisfy the protocol.
    """

    async def produce(
        self,
        subject: SubjectT_contra,
        /,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> DomainReasoningResult[PayloadT_co] | None:
        """Compute reasoning for ``subject`` and return the typed result."""


@runtime_checkable
class ReasoningConsumerPort(Protocol[PayloadT]):
    """Host-owned port for projecting a result into stable overlay fields.

    Consumers are MCP tool responses, API endpoints, UI views -- any
    surface that needs flat ``dict[str, Any]`` overlay fields rather
    than the full envelope. The two methods mirror the
    summary/detail split currently implemented in
    ``atlas_brain/autonomous/tasks/_b2b_reasoning_consumer_adapter.py``;
    M5-beta will refactor that file to be the
    ``"vendor_pressure"`` consumer.
    """

    def to_summary_fields(
        self,
        result: DomainReasoningResult[PayloadT],
    ) -> Mapping[str, Any]:
        """Project a result into compact summary overlay fields."""

    def to_detail_fields(
        self,
        result: DomainReasoningResult[PayloadT],
    ) -> Mapping[str, Any]:
        """Project a result into the full detail overlay fields."""


class DomainDescriptor(NamedTuple):
    """Static metadata for a registered reasoning domain."""

    name: str
    subject_type: type
    payload_type: type


_REGISTRY: dict[str, DomainDescriptor] = {}


def register_domain(
    name: str,
    *,
    subject_type: type,
    payload_type: type,
) -> DomainDescriptor:
    """Register a reasoning domain.

    Idempotent on identical re-registration -- the same ``name`` with
    the same ``subject_type`` and ``payload_type`` is a no-op.
    A name conflict (same name, different types) raises
    ``ValueError`` so accidental shadowing fails closed at import
    time. Producer and consumer instances are *not* held in the
    registry: the descriptor is purely typing/discovery metadata, and
    actual instances are wired by the host at startup.
    """
    if not name:
        raise ValueError("domain name must be a non-empty string")
    existing = _REGISTRY.get(name)
    if existing is not None:
        if (
            existing.subject_type is subject_type
            and existing.payload_type is payload_type
        ):
            return existing
        raise ValueError(
            f"domain {name!r} already registered with "
            f"subject_type={existing.subject_type.__name__}, "
            f"payload_type={existing.payload_type.__name__}; "
            f"refusing to shadow with "
            f"subject_type={subject_type.__name__}, "
            f"payload_type={payload_type.__name__}"
        )
    descriptor = DomainDescriptor(
        name=name,
        subject_type=subject_type,
        payload_type=payload_type,
    )
    _REGISTRY[name] = descriptor
    return descriptor


def get_domain(name: str) -> DomainDescriptor | None:
    """Return the descriptor registered under ``name``, if any."""
    return _REGISTRY.get(name)


def list_domains() -> Sequence[DomainDescriptor]:
    """Return all registered domain descriptors, sorted by name."""
    return tuple(sorted(_REGISTRY.values(), key=lambda d: d.name))


def _clear_registry_for_tests() -> None:
    """Test-only: drop all registered domains.

    Not part of the public API. Used by tests that need a clean
    registry to validate registration semantics without depending on
    test-execution order.
    """
    _REGISTRY.clear()


def _project_universal_fields(
    result: DomainReasoningResult[Any],
) -> dict[str, Any]:
    """Helper: extract the universal (non-domain) fields as a flat dict.

    Useful for consumers that want to default-include the universal
    fields and only customize the domain-specific ones. Returns a new
    dict each call so callers can mutate it freely.
    """
    return {
        "subject_id": result.subject_id,
        "domain": result.domain,
        "confidence": result.confidence,
        "confidence_label": result.confidence_label,
        "as_of": result.as_of,
        "executive_summary": result.executive_summary,
        "key_signals": list(result.key_signals),
        "uncertainty_sources": list(result.uncertainty_sources),
        "falsification_conditions": list(result.falsification_conditions),
        "mode": result.mode,
        "risk_level": result.risk_level,
        "archetype": result.archetype,
        "reference_ids": {
            "metric_ids": list(result.reference_ids.metric_ids),
            "witness_ids": list(result.reference_ids.witness_ids),
        },
    }


# Re-export ReasoningResult so callers that want both the legacy
# untyped result and the new typed envelope can import from one place.
__all__ = [
    "DomainDescriptor",
    "DomainReasoningResult",
    "ReasoningConsumerPort",
    "ReasoningProducerPort",
    "ReasoningResult",
    "ReasoningSubject",
    "ReferenceIds",
    "get_domain",
    "list_domains",
    "register_domain",
]
