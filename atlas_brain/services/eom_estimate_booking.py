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
            "calendar_id": _calendar_id(str(event.get("calendar_id") or "")),
            "event_id": str(event.get("event_id") or "").strip() or expected_event_id,
        }
    return {
        "summary": _event_summary(prepared.get("contact", {})),
        "location": prepared.get("contact", {}).get("address"),
        "description": _event_description(command),
        "calendar_id": _calendar_id(command.calendar_id),
        "event_id": expected_event_id,
    }


def _calendar_failure_proves_no_write(result: Any) -> bool:
    """Classify Calendar create failures before choosing failed vs ambiguous."""
    if result.error == "AUTH_ERROR":
        return True
    if result.error != "API_ERROR":
        return False
    if not isinstance(result.data, dict):
        return False
    status_code = result.data.get("status_code")
    return status_code in {400, 401, 403, 404}


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

    expected_event_id = deterministic_eom_estimate_calendar_event_id(
        contact_id=command.contact_id,
        booking_key=command.booking_key,
    )
    prepared = await preparer(
        contact_id=command.contact_id,
        scheduled_start=command.scheduled_start,
        scheduled_end=command.scheduled_end,
        calendar_id=_calendar_id(command.calendar_id),
        notes=command.notes,
        booking_key=command.booking_key,
        expected_calendar_event_id=expected_event_id,
        actor_id=command.actor_id,
        actor_name=command.actor_name,
    )
    if bool(prepared.get("idempotent")) and prepared.get("status") == "estimate_booked":
        return prepared

    create_event = getattr(calendar, "create_event", None)
    if not callable(create_event):
        raise RuntimeError(
            "Configured Calendar provider cannot create EOM estimate events"
        )

    calendar_event = _prepared_calendar_event(
        prepared,
        command=command,
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

    return await completer(
        contact_id=command.contact_id,
        scheduled_start=command.scheduled_start,
        scheduled_end=command.scheduled_end,
        calendar_id=_calendar_id(command.calendar_id),
        notes=command.notes,
        booking_key=command.booking_key,
        expected_calendar_event_id=expected_event_id,
        calendar_event_id=calendar_event_id,
        actor_id=command.actor_id,
        actor_name=command.actor_name,
    )
