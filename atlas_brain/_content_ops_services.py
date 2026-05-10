"""Factory for the Content Ops execution-services bundle.

Builds a `ContentOpsExecutionServices` populated with the
generators the host has fully wired. This v0 slot wires
`signal_extraction` only -- a deterministic generator with no
external dependencies (no `IntelligenceRepository`, no
`LLMClient`, no `SkillStore`). Other slots stay `None`; the
executor's per-step dispatcher returns `service_not_configured`
for unset slots, which the route layer maps to a per-step error
the UI can render.

Follow-up slices will plug the remaining 5 generators
(`campaign`, `blog_post`, `report`, `landing_page`,
`sales_brief`) into the same bundle once their host-side
repository factories land.

See `plans/PR-Content-Ops-Execution-Services-Wire-1.md` for the
slice contract.
"""

from __future__ import annotations

from extracted_content_pipeline.content_ops_execution import (
    ContentOpsExecutionServices,
)
from extracted_content_pipeline.signal_extraction import (
    SignalExtractionService,
)


# Module-level singleton: SignalExtractionService is stateless, so
# re-creating it per request would just churn allocations.
_SIGNAL_EXTRACTION_SERVICE: SignalExtractionService = SignalExtractionService()


def build_content_ops_execution_services() -> ContentOpsExecutionServices:
    """Return the host's Content Ops execution-services bundle.

    Slots not yet populated remain `None`; the executor returns
    `service_not_configured` per output for unset slots. As host
    repositories / LLM / skills factories arrive, follow-up
    slices populate the remaining slots in this same bundle.
    """

    return ContentOpsExecutionServices(
        signal_extraction=_SIGNAL_EXTRACTION_SERVICE,
    )


__all__ = ["build_content_ops_execution_services"]
