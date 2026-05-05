"""Vendor-pressure specialization of the domain-agnostic reasoning envelope.

This module expresses the existing churn / vendor-pressure reasoning
flow as a typed specialization of the protocols defined in M5-alpha
(``extracted_reasoning_core.domains``). It is the M5-beta deliverable
from ``docs/hybrid_extraction_plan_status_2026-05-05.md``.

Three pieces:

1.  ``VendorPressurePayload`` -- the vendor-specific payload that
    travels under ``DomainReasoningResult.domain_payload`` for this
    domain. Initially carries the canonical wedge identifier so a
    vendor-pressure-aware consumer can route on it without losing
    signal to the universal ``archetype`` slot. Future producer-side
    work can grow this to include displacement narratives, account
    signals, etc., without changing the envelope.

2.  ``vendor_pressure_result_from_synthesis_view(view)`` -- the
    producer-side bridge that builds a typed
    ``DomainReasoningResult[VendorPressurePayload]`` from a
    ``_b2b_synthesis_reader.SynthesisView``. Internally still goes
    through ``synthesis_view_to_reasoning_entry`` so the universal
    fields (confidence, summary, signals, uncertainty, falsification,
    mode, risk_level, archetype) come from the same place they always
    did.

3.  ``VendorPressureConsumer`` -- the consumer-side projection that
    implements ``ReasoningConsumerPort[VendorPressurePayload]``. Its
    ``to_summary_fields`` and ``to_detail_fields`` produce the same
    flat dicts that the legacy ``_b2b_reasoning_consumer_adapter``
    produced. Wire shape is preserved bit-for-bit; the legacy adapter
    delegates here.

The domain is registered at module import via ``register_domain``,
making it discoverable via ``extracted_reasoning_core.domains.get_domain
('vendor_pressure')``.
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
class VendorOpportunitySubject:
    """A vendor (or vendor opportunity) that vendor-pressure reasoning runs on.

    Satisfies ``extracted_reasoning_core.domains.ReasoningSubject``
    structurally: ``id``, ``domain``, and a ``payload`` mapping carrying
    whatever opportunity-shaped data the producer needs (vendor name,
    target mode, opportunity row fields).
    """

    id: str
    domain: str = "vendor_pressure"
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VendorPressurePayload:
    """Vendor-pressure-specific payload carried in ``domain_payload``.

    Currently minimal: the canonical wedge identifier (e.g. ``"renewal_pressure"``,
    ``"price_squeeze"``). Universal fields like the archetype label,
    confidence, and risk level live on the envelope itself; the wedge here
    is the raw vendor-pressure semantic identifier without losing fidelity
    when the universal ``archetype`` slot may rename it.

    The dataclass is intentionally narrow today and will grow with
    producer-side enrichment (displacement narrative, account signals,
    proof points) as those move under the typed payload pathway.
    """

    wedge: str | None = None


def vendor_pressure_result_from_entry(
    entry: Mapping[str, Any] | None,
    *,
    subject_id: str = "",
) -> DomainReasoningResult[VendorPressurePayload]:
    """Build a typed reasoning result from a reasoning-entry dict.

    Pure dict-in / envelope-out conversion. Tests use this directly with
    synthetic entries, avoiding the ``atlas_brain.autonomous.tasks``
    import chain (which pulls in storage / asyncpg / scheduler that
    aren't in the standalone-CI dep set).

    A ``None`` or empty entry produces a result with all-None scalar
    fields and empty-tuple list fields, matching the PR #184 sparse
    contract that ``signals.py`` overlays rely on.
    """
    entry = entry or {}

    key_signals_raw = entry.get("key_signals") or []
    if not isinstance(key_signals_raw, list):
        key_signals_raw = []
    uncertainty_raw = entry.get("uncertainty_sources") or []
    if not isinstance(uncertainty_raw, list):
        uncertainty_raw = []
    falsification_raw = entry.get("falsification_conditions") or []
    if not isinstance(falsification_raw, list):
        falsification_raw = []

    archetype = entry.get("archetype")

    return DomainReasoningResult(
        subject_id=subject_id,
        domain="vendor_pressure",
        confidence=entry.get("confidence"),
        domain_payload=VendorPressurePayload(wedge=archetype),
        executive_summary=entry.get("executive_summary"),
        key_signals=tuple(key_signals_raw),
        uncertainty_sources=tuple(uncertainty_raw),
        falsification_conditions=tuple(falsification_raw),
        mode=entry.get("mode"),
        risk_level=entry.get("risk_level"),
        archetype=archetype,
        reference_ids=ReferenceIds(),
    )


def vendor_pressure_result_from_synthesis_view(
    view: object,
    *,
    subject_id: str = "",
) -> DomainReasoningResult[VendorPressurePayload]:
    """Build a typed reasoning result from a vendor synthesis view.

    Production wrapper that routes through
    ``_b2b_synthesis_reader.synthesis_view_to_reasoning_entry`` then
    delegates to :func:`vendor_pressure_result_from_entry`. The deferred
    import keeps this module loadable without the ``atlas_brain.autonomous``
    dep chain when callers only need the dict-in path.
    """
    if view is None:
        return vendor_pressure_result_from_entry({}, subject_id=subject_id)

    from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
        synthesis_view_to_reasoning_entry,
    )

    entry = synthesis_view_to_reasoning_entry(view)
    return vendor_pressure_result_from_entry(entry, subject_id=subject_id)


class VendorPressureConsumer:
    """Consumer adapter for the ``"vendor_pressure"`` domain.

    Satisfies ``ReasoningConsumerPort[VendorPressurePayload]``. The two
    projections exactly mirror the dict shapes the legacy
    ``_b2b_reasoning_consumer_adapter.reasoning_*_fields_from_view``
    functions produced, so MCP signal overlays and any other consumers
    keep working without code changes.

    Field-name mapping from envelope -> overlay output:

        envelope.archetype          -> ``archetype``
        envelope.confidence         -> ``archetype_confidence``
        envelope.mode               -> ``reasoning_mode``
        envelope.risk_level         -> ``reasoning_risk_level``
        envelope.executive_summary  -> ``reasoning_executive_summary``
        envelope.key_signals        -> ``reasoning_key_signals``
        envelope.uncertainty_sources-> ``reasoning_uncertainty_sources``
        envelope.falsification_conditions -> ``falsification_conditions``

    The summary projection includes the first four fields; the detail
    projection includes all eight.
    """

    def to_summary_fields(
        self,
        result: DomainReasoningResult[VendorPressurePayload],
    ) -> Mapping[str, Any]:
        return {
            "archetype": result.archetype,
            "archetype_confidence": result.confidence,
            "reasoning_mode": result.mode,
            "reasoning_risk_level": result.risk_level,
        }

    def to_detail_fields(
        self,
        result: DomainReasoningResult[VendorPressurePayload],
    ) -> Mapping[str, Any]:
        return {
            "archetype": result.archetype,
            "archetype_confidence": result.confidence,
            "reasoning_mode": result.mode,
            "reasoning_risk_level": result.risk_level,
            "reasoning_executive_summary": result.executive_summary,
            "reasoning_key_signals": list(result.key_signals),
            "reasoning_uncertainty_sources": list(result.uncertainty_sources),
            "falsification_conditions": list(result.falsification_conditions),
        }


# Module-import side effect: register the domain. Idempotent on
# identical re-registration (e.g. re-import in tests), conflict raises.
register_domain(
    "vendor_pressure",
    subject_type=VendorOpportunitySubject,
    payload_type=VendorPressurePayload,
)


__all__ = [
    "VendorOpportunitySubject",
    "VendorPressureConsumer",
    "VendorPressurePayload",
    "vendor_pressure_result_from_entry",
    "vendor_pressure_result_from_synthesis_view",
]
