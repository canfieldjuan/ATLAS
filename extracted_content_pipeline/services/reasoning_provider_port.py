"""Host-owned provider port for campaign reasoning context.

Role distinction (M5-alpha / M5-beta context):

This port and ``extracted_reasoning_core.domains.ReasoningProducerPort``
are **different architectural roles** that happen to share the word
"reasoning":

* ``CampaignReasoningProviderPort`` (this module) is a **data-input
  provider**. It returns a ``CampaignReasoningContext`` -- a
  vendor-pressure-domain-shaped bundle of pre-computed inputs
  (``anchor_examples``, ``witness_highlights``, ``top_theses``,
  ``account_signals``, ``proof_points``, ``timing_windows``, ...)
  that the campaign generator feeds into its prompts. The host has
  already done the upstream reasoning; this port just supplies the
  per-target payload.

* ``ReasoningProducerPort[SubjectT, PayloadT]`` (M5-alpha) is a
  **reasoning compute port**. It takes a subject and produces a
  typed ``DomainReasoningResult[PayloadT]`` envelope. The producer
  is what *runs* the reasoning; consumers project the envelope into
  overlay fields.

So a typical end-to-end flow looks like:

    1. A reasoning producer (``vendor_pressure``, ``call_transcript``,
       ...) implements ``ReasoningProducerPort`` and emits a typed
       envelope per subject.
    2. A host pipeline persists / indexes that envelope.
    3. ``CampaignReasoningProviderPort`` (a *separate* component)
       reads back the persisted ``vendor_pressure`` data shaped as
       a ``CampaignReasoningContext`` so the campaign generator can
       consume it.

A future enrichment can grow ``CampaignReasoningContext`` to carry a
``DomainReasoningResult[VendorPressurePayload]`` directly (or convert
between the two shapes) once the producer-side typed envelope is in
production use. That work is tracked in the plan-status doc as a
follow-up; it is intentionally not done here because it would touch
every campaign-generator caller and isn't required for the
domain-agnostic reasoning abstraction itself to be useful.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from ..campaign_ports import CampaignReasoningContext, TenantScope


@runtime_checkable
class CampaignReasoningProviderPort(Protocol):
    """Port for reading per-target reasoning context from a host provider.

    Implementers fetch the prepared per-campaign-target reasoning bundle
    keyed by ``scope`` (tenant) and ``target_id``. Returning ``None``
    signals that no context is available for that target; the campaign
    generator falls back to its zero-context defaults rather than
    failing.

    See module docstring for how this port relates to the M5-alpha
    ``ReasoningProducerPort``.
    """

    async def read_campaign_reasoning_context(
        self,
        *,
        scope: TenantScope,
        target_id: str,
        target_mode: str,
        opportunity: Mapping[str, Any],
    ) -> CampaignReasoningContext | None:
        ...
