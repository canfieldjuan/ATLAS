"""Constrained Atlas side of the office-approved EOM customer handoff."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class EOMLeadConversionError(ValueError):
    """A caller-correctable finalization rejection with an HTTP status."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class EOMCustomerHandoff:
    """Opaque cross-system facts Atlas may persist for an approval."""

    contact_id: str
    tracker_customer_id: int
    tracker_site_id: int
    approval_key: str
    actor_id: int
    actor_name: str


async def finalize_eom_customer_handoff(
    crm: Any,
    handoff: EOMCustomerHandoff,
) -> dict[str, Any]:
    """Delegate to the authoritative CRM transaction implementation."""
    finalizer = getattr(crm, "finalize_eom_customer_handoff", None)
    if not callable(finalizer):
        raise RuntimeError("Configured CRM provider cannot finalize EOM customer handoffs")
    return await finalizer(
        contact_id=handoff.contact_id,
        tracker_customer_id=handoff.tracker_customer_id,
        tracker_site_id=handoff.tracker_site_id,
        approval_key=handoff.approval_key,
        actor_id=handoff.actor_id,
        actor_name=handoff.actor_name,
    )
