"""Office-owned EOM estimate booking command."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any


class EOMEstimateBookingError(Exception):
    """HTTP-mappable failure for the private EOM estimate booking command."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class EOMEstimateBooking:
    contact_id: str
    scheduled_start: datetime
    scheduled_end: datetime
    calendar_id: str | None
    notes: str | None
    booking_key: str
    actor_id: int
    actor_name: str


def deterministic_eom_estimate_calendar_event_id(
    *,
    contact_id: str,
    booking_key: str,
) -> str:
    """Return a Google-safe deterministic event ID for one booking operation."""
    digest = hashlib.sha256(f"{contact_id}:{booking_key}".encode("utf-8")).hexdigest()
    return f"eomest{digest[:56]}"


def _calendar_id(value: str | None) -> str:
    return (value or "primary").strip() or "primary"


def _configured_calendar_id(calendar: Any) -> str:
    configured = getattr(calendar, "configured_calendar_id", None)
    if isinstance(configured, str) and configured.strip():
        return _calendar_id(configured)
    # Fallback for providers/fakes that predate the public accessor.
    config = getattr(calendar, "_config", None)
    configured = getattr(config, "calendar_id", None)
    return _calendar_id(configured if isinstance(configured, str) else None)


def _effective_calendar_id(value: str | None, calendar: Any) -> str:
    requested = (value or "").strip()
    if requested:
        return requested
    return _configured_calendar_id(calendar)


def _event_summary(contact: dict[str, Any]) -> str:
    name = str(contact.get("full_name") or "").strip() or "EOM lead"
    return f"Estimate: {name}"


def _event_description(command: EOMEstimateBooking) -> str:
    parts = ["Scheduled from the private EOM lead funnel."]
    if command.notes:
        parts.append(command.notes)
    return "\n\n".join(parts)


def _prepared_calendar_event(
    prepared: dict[str, Any],
    *,
    command: EOMEstimateBooking,
    effective_calendar_id: str,
    expected_event_id: str,
) -> dict[str, Any]:
    """Return the immutable Calendar payload claimed by CRM preparation."""
    event = prepared.get("calendar_event")
    if isinstance(event, dict):
        return {
            "summary": str(event.get("summary") or "").strip()
            or _event_summary(prepared.get("contact", {})),
            "location": event.get("location"),
            "description": str(event.get("description") or ""),
            "calendar_id": _calendar_id(
                str(event.get("calendar_id") or effective_calendar_id)
            ),
            "event_id": str(event.get("event_id") or "").strip() or expected_event_id,
        }
    return {
        "summary": _event_summary(prepared.get("contact", {})),
        "location": prepared.get("contact", {}).get("address"),
        "description": _event_description(command),
        "calendar_id": effective_calendar_id,
        "event_id": expected_event_id,
    }


def _calendar_failure_proves_no_write(result: Any) -> bool:
    """Classify Calendar create failures before choosing failed vs ambiguous."""
    data = result.data if isinstance(result.data, dict) else {}
    if data.get("request_phase") == "conflict_verification":
        return False
    if result.error in {"TOOL_DISABLED", "NOT_CONFIGURED"}:
        # CalendarTool returns these before issuing any Google request, so no
        # event can exist; recording ambiguity here would wedge the lead with
        # no reconciliation surface (e.g. service booted before Calendar
        # secrets were populated).
        return True
    if result.error == "AUTH_ERROR":
        return True
    if result.error != "API_ERROR":
        return False
    status_code = data.get("status_code")
    return status_code in {400, 401, 403, 404}


def _booking_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "contact_id": str(result.get("contact_id") or ""),
        "lead_stage": str(result.get("lead_stage") or ""),
        "status": str(result.get("status") or ""),
        "calendar_event_id": (
            str(result["calendar_event_id"])
            if result.get("calendar_event_id") is not None
            else None
        ),
        "expected_calendar_event_id": (
            str(result["expected_calendar_event_id"])
            if result.get("expected_calendar_event_id") is not None
            else None
        ),
        "idempotent": bool(result.get("idempotent")),
    }


async def schedule_eom_estimate_booking(
    crm: Any,
    calendar: Any,
    command: EOMEstimateBooking,
) -> dict[str, Any]:
    """Prepare, create the Calendar event, then complete the CRM transition."""
    preparer = getattr(crm, "prepare_eom_estimate_booking", None)
    completer = getattr(crm, "complete_eom_estimate_booking", None)
    marker = getattr(crm, "mark_eom_estimate_booking_calendar_ambiguous", None)
    failure_marker = getattr(crm, "mark_eom_estimate_booking_calendar_failed", None)
    if (
        not callable(preparer)
        or not callable(completer)
        or not callable(marker)
        or not callable(failure_marker)
    ):
        raise RuntimeError(
            "Configured CRM provider cannot schedule EOM estimate bookings"
        )
    execution_lock = getattr(crm, "eom_estimate_booking_execution_lock", None)
    if not callable(execution_lock):
        raise RuntimeError(
            "Configured CRM provider cannot serialize EOM estimate bookings"
        )

    # The lock spans prepare -> Calendar -> complete so customer handoff can
    # detect an in-flight same-key execution and stay fenced until the
    # strongest outcome for this key is on the ledger.
    async with execution_lock(booking_key=command.booking_key):
        return await _run_estimate_booking(
            calendar=calendar,
            command=command,
            preparer=preparer,
            completer=completer,
            marker=marker,
            failure_marker=failure_marker,
        )


async def _run_estimate_booking(
    *,
    calendar: Any,
    command: EOMEstimateBooking,
    preparer: Any,
    completer: Any,
    marker: Any,
    failure_marker: Any,
) -> dict[str, Any]:
    """Execute one estimate-booking attempt while the execution lock is held."""
    expected_event_id = deterministic_eom_estimate_calendar_event_id(
        contact_id=command.contact_id,
        booking_key=command.booking_key,
    )
    requested_calendar_id = (command.calendar_id or "").strip()
    effective_calendar_id = _effective_calendar_id(command.calendar_id, calendar)
    prepared = await preparer(
        contact_id=command.contact_id,
        scheduled_start=command.scheduled_start,
        scheduled_end=command.scheduled_end,
        calendar_id=effective_calendar_id,
        calendar_id_explicit=bool(requested_calendar_id),
        notes=command.notes,
        booking_key=command.booking_key,
        expected_calendar_event_id=expected_event_id,
        actor_id=command.actor_id,
        actor_name=command.actor_name,
    )
    if bool(prepared.get("idempotent")) and prepared.get("status") == "estimate_booked":
        return _booking_result(prepared)

    create_event = getattr(calendar, "create_event", None)
    if not callable(create_event):
        raise RuntimeError(
            "Configured Calendar provider cannot create EOM estimate events"
        )

    calendar_event = _prepared_calendar_event(
        prepared,
        command=command,
        effective_calendar_id=effective_calendar_id,
        expected_event_id=expected_event_id,
    )
    if calendar_event["event_id"] != expected_event_id:
        await marker(
            contact_id=command.contact_id,
            booking_key=command.booking_key,
            expected_calendar_event_id=expected_event_id,
            observed_calendar_event_id=str(calendar_event["event_id"]),
            actor_id=command.actor_id,
            actor_name=command.actor_name,
        )
        raise EOMEstimateBookingError(
            502,
            "Prepared calendar event id does not match estimate booking",
        )

    result = await create_event(
        summary=calendar_event["summary"],
        start=command.scheduled_start,
        end=command.scheduled_end,
        location=calendar_event["location"],
        description=calendar_event["description"],
        calendar_id=calendar_event["calendar_id"],
        event_id=expected_event_id,
    )
    if not result.success:
        if result.error == "IDEMPOTENCY_CONFLICT" or not _calendar_failure_proves_no_write(
            result
        ):
            observed_event_id = ""
            if isinstance(result.data, dict):
                observed_event_id = str(result.data.get("event_id") or "")
            await marker(
                contact_id=command.contact_id,
                booking_key=command.booking_key,
                expected_calendar_event_id=expected_event_id,
                observed_calendar_event_id=observed_event_id,
                actor_id=command.actor_id,
                actor_name=command.actor_name,
            )
        else:
            await failure_marker(
                contact_id=command.contact_id,
                booking_key=command.booking_key,
                expected_calendar_event_id=expected_event_id,
                calendar_error=result.error,
                calendar_message=result.message,
                actor_id=command.actor_id,
                actor_name=command.actor_name,
            )
        raise EOMEstimateBookingError(
            502,
            result.message or "Calendar event creation failed",
        )

    calendar_event_id = str(result.data.get("event_id") or "").strip()
    if calendar_event_id != expected_event_id:
        await marker(
            contact_id=command.contact_id,
            booking_key=command.booking_key,
            expected_calendar_event_id=expected_event_id,
            observed_calendar_event_id=calendar_event_id,
            actor_id=command.actor_id,
            actor_name=command.actor_name,
        )
        raise EOMEstimateBookingError(
            502,
            "Calendar returned an unexpected event id; booking requires reconciliation",
        )

    completed = await completer(
        contact_id=command.contact_id,
        scheduled_start=command.scheduled_start,
        scheduled_end=command.scheduled_end,
        calendar_id=str(calendar_event["calendar_id"]),
        notes=command.notes,
        booking_key=command.booking_key,
        expected_calendar_event_id=expected_event_id,
        calendar_event_id=calendar_event_id,
        actor_id=command.actor_id,
        actor_name=command.actor_name,
    )
    return _booking_result(completed)
