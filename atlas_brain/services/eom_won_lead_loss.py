"""Safe disposition of an EOM lead after its first clean was booked.

The existing pre-won loss writer is intentionally a single database
transaction.  A won lead has two extra effects: a persisted Google Calendar
event and an approval-sendable onboarding draft.  This module owns the narrow
prepare -> Calendar DELETE -> complete protocol required to remove those
effects without claiming a loss after an uncertain external result.
"""

from __future__ import annotations

from typing import Any

from .eom_lead_conversion import EOMLeadConversionError, EOMLeadLost, mark_eom_lead_lost


def _won_loss_callables(crm: Any) -> tuple[Any, Any, Any, Any]:
    """Return the provider's closed won-loss lifecycle surface.

    Unlike a compatibility fallback, requiring this surface prevents a future
    provider from silently admitting won leads through the legacy direct writer
    without owning the Calendar and draft teardown protocol.
    """

    execution_lock = getattr(crm, "eom_won_lead_loss_execution_lock", None)
    prepare = getattr(crm, "prepare_eom_won_lead_loss", None)
    complete = getattr(crm, "complete_eom_won_lead_loss", None)
    unsettled = getattr(crm, "mark_eom_won_lead_loss_calendar_unsettled", None)
    if not all(
        callable(value) for value in (execution_lock, prepare, complete, unsettled)
    ):
        raise RuntimeError("Configured CRM provider cannot safely lose won EOM leads")
    return execution_lock, prepare, complete, unsettled


async def mark_eom_lead_lost_with_won_teardown(
    crm: Any,
    calendar: Any,
    command: EOMLeadLost,
) -> dict[str, Any]:
    """Run the canonical direct loss or the safe won-lead teardown.

    The authoritative provider decides whether the current contact is pre-won,
    already-completed, or needs the Calendar lifecycle.  The execution lock is
    held across the only external step.  It deliberately is not a database
    transaction: the provider writes durable prepare evidence before DELETE and
    atomically commits the draft revoke plus lost transition only afterwards.
    """

    execution_lock, prepare, complete, unsettled = _won_loss_callables(crm)
    async with execution_lock(contact_id=command.contact_id) as execution_crm:
        scoped_crm = execution_crm if execution_crm is not None else crm
        _, scoped_prepare, scoped_complete, scoped_unsettled = _won_loss_callables(
            scoped_crm
        )
        prepared = await scoped_prepare(
            contact_id=command.contact_id,
            reason_code=command.reason_code,
            note=command.note,
            operation_key=command.operation_key,
            actor_id=command.actor_id,
            actor_name=command.actor_name,
        )
        mode = str(prepared.get("mode") or "")
        if mode == "pre_won":
            return await mark_eom_lead_lost(scoped_crm, command)
        if mode == "completed":
            result = prepared.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("CRM provider returned an invalid won-loss replay")
            return result
        if mode != "won":
            raise RuntimeError("CRM provider returned an invalid won-loss preparation")

        calendar_id = str(prepared.get("calendar_id") or "").strip()
        calendar_event_id = str(prepared.get("calendar_event_id") or "").strip()
        if not calendar_id or not calendar_event_id:
            raise RuntimeError(
                "CRM provider prepared incomplete first-clean Calendar facts"
            )
        delete_event = getattr(calendar, "delete_event", None)
        if not callable(delete_event):
            raise RuntimeError(
                "Configured Calendar provider cannot delete EOM first cleans"
            )
        result = await delete_event(
            calendar_id=calendar_id,
            event_id=calendar_event_id,
        )
        if not bool(getattr(result, "success", False)):
            await scoped_unsettled(
                contact_id=command.contact_id,
                operation_key=command.operation_key,
                calendar_id=calendar_id,
                calendar_event_id=calendar_event_id,
                calendar_error=getattr(result, "error", None),
                calendar_message=str(
                    getattr(result, "message", None)
                    or "Calendar event cancellation failed"
                ),
                actor_id=command.actor_id,
                actor_name=command.actor_name,
            )
            raise EOMLeadConversionError(
                502,
                str(
                    getattr(result, "message", None)
                    or "Calendar event cancellation failed"
                ),
            )
        return await scoped_complete(
            contact_id=command.contact_id,
            reason_code=command.reason_code,
            note=command.note,
            operation_key=command.operation_key,
            calendar_id=calendar_id,
            calendar_event_id=calendar_event_id,
            actor_id=command.actor_id,
            actor_name=command.actor_name,
        )
