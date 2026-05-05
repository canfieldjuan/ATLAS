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
    ``_b2b_synthesis_reader.SynthesisView``. Goes through
    ``synthesis_view_to_reasoning_entry`` for the universal-narrative
    fields, then enriches with view-only metadata (lineage,
    freshness, categorical confidence label) before delegating to
    :func:`vendor_pressure_result_from_entry`.

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


def _normalize_lineage_ids(raw: Any) -> tuple[str, ...]:
    """Coerce to str, strip whitespace, drop None / empty.

    Matches the call_transcript domain's lineage normalization. Lineage
    IDs feed downstream lookups; whitespace IDs and empties would silently
    miss-match, so filtering at the envelope boundary is the right place.
    """
    if not isinstance(raw, list):
        return ()
    out: list[str] = []
    for item in raw:
        if item is None:
            continue
        normalized = str(item).strip()
        if normalized:
            out.append(normalized)
    return tuple(out)


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

    Recognized keys (all optional; absent or null falls back to envelope
    defaults):

    Universal-narrative fields (from ``synthesis_view_to_reasoning_entry``):
        ``archetype``, ``confidence`` (numeric), ``mode``, ``risk_level``,
        ``executive_summary``, ``key_signals``, ``uncertainty_sources``,
        ``falsification_conditions``.

    Lineage / freshness fields (from ``SynthesisView`` direct accessors;
    see :func:`vendor_pressure_result_from_synthesis_view`):
        ``confidence_label`` (categorical: ``"high"`` / ``"medium"`` /
        ``"low"`` / ...),
        ``as_of`` (ISO date string),
        ``reference_ids`` (nested object: ``{"metric_ids": [...],
        "witness_ids": [...]}``).

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

    # Lineage: nested ``reference_ids`` object is the canonical shape
    # surfaced by ``SynthesisView.reference_ids``. The synthesis-view
    # wrapper enriches the entry with this; tests can also drive it
    # directly with a synthetic dict.
    ref_ids_raw = entry.get("reference_ids")
    if isinstance(ref_ids_raw, Mapping):
        metric_ids = _normalize_lineage_ids(ref_ids_raw.get("metric_ids"))
        witness_ids = _normalize_lineage_ids(ref_ids_raw.get("witness_ids"))
    else:
        metric_ids = ()
        witness_ids = ()

    # Freshness: empty string from view.as_of_date_iso() collapses to None
    # so consumers see "no date" rather than "the empty-string date".
    as_of_raw = entry.get("as_of")
    if isinstance(as_of_raw, str) and as_of_raw.strip():
        as_of = as_of_raw.strip()
    else:
        as_of = None

    # Categorical confidence label (e.g. "high"/"medium"/"low"). Empty
    # strings normalize to None for the same reason.
    confidence_label_raw = entry.get("confidence_label")
    if isinstance(confidence_label_raw, str) and confidence_label_raw.strip():
        confidence_label = confidence_label_raw.strip()
    else:
        confidence_label = None

    return DomainReasoningResult(
        subject_id=subject_id,
        domain="vendor_pressure",
        confidence=entry.get("confidence"),
        domain_payload=VendorPressurePayload(wedge=archetype),
        confidence_label=confidence_label,
        as_of=as_of,
        executive_summary=entry.get("executive_summary"),
        key_signals=tuple(key_signals_raw),
        uncertainty_sources=tuple(uncertainty_raw),
        falsification_conditions=tuple(falsification_raw),
        mode=entry.get("mode"),
        risk_level=entry.get("risk_level"),
        archetype=archetype,
        reference_ids=ReferenceIds(
            metric_ids=metric_ids,
            witness_ids=witness_ids,
        ),
    )


def _enrich_entry_with_view_metadata(
    entry: dict[str, Any],
    view: object,
) -> dict[str, Any]:
    """Top up ``entry`` with view-only metadata via ``setdefault``.

    Pulled out as a pure helper so tests can exercise it directly with
    a stub view, without crossing into the ``atlas_brain.autonomous``
    import chain that the lean CI tier doesn't carry. Also makes the
    property-vs-method handling for each accessor explicit.

    Three accessors, each with its own quirk in the real
    ``_b2b_synthesis_reader.SynthesisView``:

    - ``view.reference_ids`` is a ``@property`` returning a dict
      (already deduplicated). Read directly via ``getattr``.
    - ``view.as_of_date_iso`` is a ``@property`` returning a string
      ("" when the synthesis has no date). The earlier draft of this
      wrapper treated it as a method, which silently dropped the
      value in production -- ``getattr`` on a property returns the
      *value*, ``callable(str)`` is ``False``, branch skipped. Both
      shapes are now accepted (property value or zero-arg callable)
      so a future refactor of the view doesn't regress us either way.
    - ``view.confidence(section)`` is a regular method taking a
      section-name argument. Always callable in the real view.
    """
    # Lineage: property returning the dict directly.
    reference_ids = getattr(view, "reference_ids", None)
    if isinstance(reference_ids, Mapping) and reference_ids:
        entry.setdefault("reference_ids", reference_ids)

    # Freshness: property in the real view (returns ""). Accept either
    # the property value (str | None) or a zero-arg callable that
    # returns a string. Empty / whitespace-only collapses to absent.
    as_of_attr = getattr(view, "as_of_date_iso", None)
    if callable(as_of_attr):
        try:
            as_of_value = as_of_attr()
        except Exception:  # pragma: no cover -- defensive against view-API drift
            as_of_value = None
    else:
        as_of_value = as_of_attr
    if isinstance(as_of_value, str) and as_of_value.strip():
        entry.setdefault("as_of", as_of_value)

    # Categorical confidence label for the causal-narrative section
    # (complements the numeric confidence already in the entry).
    confidence_method = getattr(view, "confidence", None)
    if callable(confidence_method):
        try:
            label = confidence_method("causal_narrative")
        except Exception:  # pragma: no cover -- defensive against view-API drift
            label = None
        if isinstance(label, str) and label.strip():
            entry.setdefault("confidence_label", label)

    return entry


def vendor_pressure_result_from_synthesis_view(
    view: object,
    *,
    subject_id: str = "",
) -> DomainReasoningResult[VendorPressurePayload]:
    """Build a typed reasoning result from a vendor synthesis view.

    Production wrapper. Two-step:

    1. Calls ``synthesis_view_to_reasoning_entry(view)`` for the
       universal narrative fields (archetype / confidence / summary /
       signals / mode / risk_level / falsification / uncertainty).
    2. Enriches the entry dict via :func:`_enrich_entry_with_view_metadata`
       with view-only metadata the reasoning-entry helper doesn't
       surface:

       - ``reference_ids`` from ``view.reference_ids`` -- the nested
         ``{"metric_ids": [...], "witness_ids": [...]}`` lineage block.
       - ``as_of`` from ``view.as_of_date_iso`` (property) -- ISO date
         string describing when the synthesis was generated.
       - ``confidence_label`` from ``view.confidence("causal_narrative")``
         -- the categorical band ("high" / "medium" / "low" / ...).

    Then delegates to :func:`vendor_pressure_result_from_entry` so the
    sparse-contract guards and field normalization happen in one place.

    The deferred import keeps this module loadable without the
    ``atlas_brain.autonomous`` dep chain when callers only need the
    dict-in path.
    """
    if view is None:
        return vendor_pressure_result_from_entry({}, subject_id=subject_id)

    from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
        synthesis_view_to_reasoning_entry,
    )

    entry: dict[str, Any] = dict(synthesis_view_to_reasoning_entry(view))
    entry = _enrich_entry_with_view_metadata(entry, view)

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
    projection includes all eight. Lineage / freshness fields are
    available on the envelope (``reference_ids``, ``as_of``,
    ``confidence_label``) but intentionally not in this consumer's
    overlay output to preserve the legacy ``signals.py`` wire shape.
    Future consumers (or this one in a versioned variant) can surface
    them when there's a UI/API surface ready to render them.
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
