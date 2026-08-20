"""Office-owned EOM booking commands (estimate and first clean).

Both booking families run the same durable prepare -> Calendar -> complete
engine; a service binding carries only what differs: the deterministic
event-ID prefix, the CRM provider method names, and the completed-status
token. The first-clean completion additionally enqueues the onboarding
email draft inside the CRM transaction (see
DatabaseCRMProvider._enqueue_eom_onboarding_email_draft).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any


class EOMEstimateBookingError(Exception):
    """HTTP-mappable failure for the private EOM booking commands."""

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


@dataclass(frozen=True)
class EOMFirstCleanBooking(EOMEstimateBooking):
    """Same payload shape; the distinct type names the office action."""


@dataclass(frozen=True)
class _EOMBookingServiceBinding:
    """Family-specific constants for the shared booking engine."""

    label: str
    summary_prefix: str
    event_id_prefix: str
    booked_status: str
    preparer_name: str
    completer_name: str
    ambiguous_marker_name: str
    failed_marker_name: str
    requires_concrete_calendar_identity: bool


_ESTIMATE_SERVICE_BINDING = _EOMBookingServiceBinding(
    label="estimate",
    summary_prefix="Estimate",
    event_id_prefix="eomest",
    booked_status="estimate_booked",
    preparer_name="prepare_eom_estimate_booking",
    completer_name="complete_eom_estimate_booking",
    ambiguous_marker_name="mark_eom_estimate_booking_calendar_ambiguous",
    failed_marker_name="mark_eom_estimate_booking_calendar_failed",
    requires_concrete_calendar_identity=False,
)

_FIRST_CLEAN_SERVICE_BINDING = _EOMBookingServiceBinding(
    label="first clean",
    summary_prefix="First clean",
    event_id_prefix="eomfcl",
    booked_status="first_clean_booked",
    preparer_name="prepare_eom_first_clean_booking",
    completer_name="complete_eom_first_clean_booking",
    ambiguous_marker_name="mark_eom_first_clean_booking_calendar_ambiguous",
    failed_marker_name="mark_eom_first_clean_booking_calendar_failed",
    requires_concrete_calendar_identity=True,
)


def _deterministic_eom_booking_event_id(
    prefix: str, *, contact_id: str, booking_key: str
) -> str:
    digest = hashlib.sha256(f"{contact_id}:{booking_key}".encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:56]}"


def deterministic_eom_estimate_calendar_event_id(
    *,
    contact_id: str,
    booking_key: str,
) -> str:
    """Return a Google-safe deterministic event ID for one booking operation."""
    return _deterministic_eom_booking_event_id(
        "eomest", contact_id=contact_id, booking_key=booking_key
    )


def deterministic_eom_first_clean_calendar_event_id(
    *,
    contact_id: str,
    booking_key: str,
) -> str:
    """Return a Google-safe deterministic event ID for one first-clean booking."""
    return _deterministic_eom_booking_event_id(
        "eomfcl", contact_id=contact_id, booking_key=booking_key
    )


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


async def _resolve_concrete_calendar_identity(calendar: Any, *, calendar_id: str) -> str:
    """Resolve a first-clean target before CRM records an external event.

    ``primary`` belongs to whichever Google principal owns the current refresh
    token. A first-clean booking therefore cannot persist it as if it were the
    historical event's concrete resource identifier. The Calendar boundary
    supplies the canonical ID before this service writes preparation evidence.
    """

    resolve_calendar_id = getattr(calendar, "resolve_calendar_id", None)
    if not callable(resolve_calendar_id):
        raise RuntimeError(
            "Configured Calendar provider cannot resolve EOM first-clean identity"
        )
    result = await resolve_calendar_id(calendar_id=calendar_id)
    if not bool(getattr(result, "success", False)):
        raise EOMEstimateBookingError(
            502,
            str(
                getattr(result, "message", None)
                or "Calendar identity resolution failed"
            ),
        )
    data = getattr(result, "data", None)
    resolved_calendar_id = (
        str(data.get("calendar_id") or "").strip()
        if isinstance(data, dict)
        else ""
    )
    if not resolved_calendar_id or resolved_calendar_id.casefold() == "primary":
        raise EOMEstimateBookingError(
            502,
            "Calendar identity resolution did not return a concrete calendar id",
        )
    return resolved_calendar_id


def _event_summary(
    binding: _EOMBookingServiceBinding, contact: dict[str, Any]
) -> str:
    name = str(contact.get("full_name") or "").strip() or "EOM lead"
    return f"{binding.summary_prefix}: {name}"


def _event_description(command: EOMEstimateBooking) -> str:
    parts = ["Scheduled from the private EOM lead funnel."]
    if command.notes:
        parts.append(command.notes)
    return "\n\n".join(parts)


def _prepared_calendar_event(
    prepared: dict[str, Any],
    *,
    binding: _EOMBookingServiceBinding,
    command: EOMEstimateBooking,
    effective_calendar_id: str,
    expected_event_id: str,
) -> dict[str, Any]:
    """Return the immutable Calendar payload claimed by CRM preparation."""
    event = prepared.get("calendar_event")
    if isinstance(event, dict):
        return {
            "summary": str(event.get("summary") or "").strip()
            or _event_summary(binding, prepared.get("contact", {})),
            "location": event.get("location"),
            "description": str(event.get("description") or ""),
            "calendar_id": _calendar_id(
                str(event.get("calendar_id") or effective_calendar_id)
            ),
            "event_id": str(event.get("event_id") or "").strip() or expected_event_id,
        }
    return {
        "summary": _event_summary(binding, prepared.get("contact", {})),
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
    if data.get("request_phase") == "auth":
        # The failure happened while acquiring the OAuth token, before any
        # Google Calendar event request was issued, so no event write can
        # exist; recording ambiguity would wedge the lead behind a token
        # outage with no reconciliation surface.
        return True
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
    closed: dict[str, Any] = {
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
    if "onboarding_draft_id" in result:
        draft_id = result.get("onboarding_draft_id")
        closed["onboarding_draft_id"] = (
            str(draft_id) if draft_id is not None else None
        )
    return closed


def _booking_lifecycle_callables(
    crm: Any, binding: _EOMBookingServiceBinding
) -> tuple[Any, Any, Any, Any]:
    preparer = getattr(crm, binding.preparer_name, None)
    completer = getattr(crm, binding.completer_name, None)
    marker = getattr(crm, binding.ambiguous_marker_name, None)
    failure_marker = getattr(crm, binding.failed_marker_name, None)
    if (
        not callable(preparer)
        or not callable(completer)
        or not callable(marker)
        or not callable(failure_marker)
    ):
        raise RuntimeError(
            f"Configured CRM provider cannot schedule EOM {binding.label} bookings"
        )
    return preparer, completer, marker, failure_marker


async def schedule_eom_estimate_booking(
    crm: Any,
    calendar: Any,
    command: EOMEstimateBooking,
) -> dict[str, Any]:
    """Prepare, create the Calendar event, then complete the CRM transition."""
    return await _schedule_eom_booking(
        crm, calendar, command, _ESTIMATE_SERVICE_BINDING
    )


async def schedule_eom_first_clean_booking(
    crm: Any,
    calendar: Any,
    command: EOMFirstCleanBooking,
) -> dict[str, Any]:
    """Book the first cleaning: Calendar event + lead_stage -> won + draft."""
    return await _schedule_eom_booking(
        crm, calendar, command, _FIRST_CLEAN_SERVICE_BINDING
    )


async def _schedule_eom_booking(
    crm: Any,
    calendar: Any,
    command: EOMEstimateBooking,
    binding: _EOMBookingServiceBinding,
) -> dict[str, Any]:
    _booking_lifecycle_callables(crm, binding)
    execution_lock = getattr(crm, "eom_estimate_booking_execution_lock", None)
    if not callable(execution_lock):
        raise RuntimeError(
            f"Configured CRM provider cannot serialize EOM {binding.label} bookings"
        )

    # The lock spans prepare -> Calendar -> complete so customer handoff can
    # detect an in-flight same-key execution and stay fenced until the
    # strongest outcome for this key is on the ledger. The lock may yield an
    # execution-scoped provider bound to its own session connection; running
    # the lifecycle steps through that provider keeps the whole booking on
    # one pooled connection, so max_size concurrent bookings cannot exhaust
    # the pool by each reserving a lock connection plus a transaction
    # connection.
    async with execution_lock(booking_key=command.booking_key) as execution_crm:
        scoped_crm = execution_crm if execution_crm is not None else crm
        preparer, completer, marker, failure_marker = _booking_lifecycle_callables(
            scoped_crm, binding
        )
        return await _run_eom_booking(
            calendar=calendar,
            command=command,
            binding=binding,
            preparer=preparer,
            completer=completer,
            marker=marker,
            failure_marker=failure_marker,
        )


async def _run_eom_booking(
    *,
    calendar: Any,
    command: EOMEstimateBooking,
    binding: _EOMBookingServiceBinding,
    preparer: Any,
    completer: Any,
    marker: Any,
    failure_marker: Any,
) -> dict[str, Any]:
    """Execute one booking attempt while the execution lock is held."""
    expected_event_id = _deterministic_eom_booking_event_id(
        binding.event_id_prefix,
        contact_id=command.contact_id,
        booking_key=command.booking_key,
    )
    requested_calendar_id = (command.calendar_id or "").strip()
    effective_calendar_id = _effective_calendar_id(command.calendar_id, calendar)

    async def _prepare(calendar_id: str) -> dict[str, Any]:
        return await preparer(
            contact_id=command.contact_id,
            scheduled_start=command.scheduled_start,
            scheduled_end=command.scheduled_end,
            calendar_id=calendar_id,
            calendar_id_explicit=bool(requested_calendar_id),
            notes=command.notes,
            booking_key=command.booking_key,
            expected_calendar_event_id=expected_event_id,
            actor_id=command.actor_id,
            actor_name=command.actor_name,
        )

    prepared = await _prepare(effective_calendar_id)
    if bool(prepared.get("requires_calendar_identity")):
        if not binding.requires_concrete_calendar_identity:
            raise RuntimeError(
                "CRM provider requested Calendar identity for an unsupported booking"
            )
        effective_calendar_id = await _resolve_concrete_calendar_identity(
            calendar, calendar_id=effective_calendar_id
        )
        prepared = await _prepare(effective_calendar_id)
        if bool(prepared.get("requires_calendar_identity")):
            raise RuntimeError("CRM provider did not accept resolved Calendar identity")
    if bool(prepared.get("idempotent")) and prepared.get("status") == binding.booked_status:
        return _booking_result(prepared)

    create_event = getattr(calendar, "create_event", None)
    if not callable(create_event):
        raise RuntimeError(
            f"Configured Calendar provider cannot create EOM {binding.label} events"
        )

    calendar_event = _prepared_calendar_event(
        prepared,
        binding=binding,
        command=command,
        effective_calendar_id=effective_calendar_id,
        expected_event_id=expected_event_id,
    )
    if (
        binding.requires_concrete_calendar_identity
        and str(calendar_event["calendar_id"]).strip().casefold() == "primary"
    ):
        raise EOMEstimateBookingError(
            409,
            "EOM first-clean booking requires Calendar identity reconciliation",
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
            f"Prepared calendar event id does not match {binding.label} booking",
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

    try:
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
    except Exception:
        # The Calendar event exists (create returned the expected ID), so a
        # completion rejection must leave reconciliation evidence instead of
        # an orphaned appointment with a forever-pending ledger.
        await marker(
            contact_id=command.contact_id,
            booking_key=command.booking_key,
            expected_calendar_event_id=expected_event_id,
            observed_calendar_event_id=calendar_event_id,
            actor_id=command.actor_id,
            actor_name=command.actor_name,
        )
        raise
    return _booking_result(completed)
