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


@dataclass(frozen=True)
class EOMLeadLost:
    """The office's disposition of a lead that will not convert."""

    contact_id: str
    reason_code: str
    note: str | None
    operation_key: str
    actor_id: int
    actor_name: str


@dataclass(frozen=True)
class EOMLeadReopen:
    """Return a previously-lost lead to the active review queue."""

    contact_id: str
    operation_key: str
    actor_id: int
    actor_name: str


async def mark_eom_lead_lost(crm: Any, command: EOMLeadLost) -> dict[str, Any]:
    """Delegate to the authoritative CRM transaction implementation."""
    marker = getattr(crm, "mark_eom_lead_lost", None)
    if not callable(marker):
        raise RuntimeError("Configured CRM provider cannot mark EOM leads lost")
    return await marker(
        contact_id=command.contact_id,
        reason_code=command.reason_code,
        note=command.note,
        operation_key=command.operation_key,
        actor_id=command.actor_id,
        actor_name=command.actor_name,
    )


async def reopen_eom_lead(crm: Any, command: EOMLeadReopen) -> dict[str, Any]:
    """Delegate to the authoritative CRM transaction implementation."""
    reopener = getattr(crm, "reopen_eom_lead", None)
    if not callable(reopener):
        raise RuntimeError("Configured CRM provider cannot reopen EOM leads")
    return await reopener(
        contact_id=command.contact_id,
        operation_key=command.operation_key,
        actor_id=command.actor_id,
        actor_name=command.actor_name,
    )
