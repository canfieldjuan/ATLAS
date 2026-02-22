"""
Scheduling tools for Atlas voice interface.

Enables natural language appointment booking via "Hey Atlas, book an appointment..."
Wraps existing SchedulingService and AppointmentRepository.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

import dateparser

from ..config import settings
from ..comms import EFFINGHAM_MAIDS_CONTEXT, BusinessContext, get_context_router
from ..storage.repositories.appointment import get_appointment_repo
from ..storage.exceptions import DatabaseUnavailableError, DatabaseOperationError
from .base import ToolParameter, ToolResult

logger = logging.getLogger("atlas.tools.scheduling")


def _get_scheduling_service():
    """Lazy import to avoid circular dependency."""
    from atlas_comms.services import scheduling_service
    return scheduling_service


def _get_time_slot_class():
    """Lazy import TimeSlot to avoid circular dependency."""
    from atlas_comms.services import TimeSlot
    return TimeSlot


def _get_default_context() -> Optional[BusinessContext]:
    """Get the default business context for scheduling."""
    router = get_context_router()

    # Try effingham_maids first
    ctx = router.get_context("effingham_maids")
    if ctx:
        return ctx

    # If not registered, try to register it (for standalone tool usage)
    try:
        from ..comms import EFFINGHAM_MAIDS_CONTEXT
        if EFFINGHAM_MAIDS_CONTEXT.scheduling.enabled:
            router.register_context(EFFINGHAM_MAIDS_CONTEXT)
            logger.info("Auto-registered effingham_maids context for scheduling")
            return EFFINGHAM_MAIDS_CONTEXT
    except Exception as e:
        logger.debug("Could not auto-register effingham_maids: %s", e)

    # Fall back to first registered context
    contexts = router.list_contexts()
    return contexts[0] if contexts else None


def _parse_datetime(text: str, timezone: str = settings.reminder.default_timezone) -> Optional[datetime]:
    """Parse natural language datetime using dateparser."""
    settings_dict = {
        "PREFER_DATES_FROM": "future",
        "PREFER_DAY_OF_MONTH": "first",
        "RETURN_AS_TIMEZONE_AWARE": True,
        "TIMEZONE": timezone,
    }
    result = dateparser.parse(text, settings=settings_dict)

    # Fallback: dateparser struggles with "next Tuesday at 10:30 AM"
    # but handles "Tuesday at 10:30 AM" fine (PREFER_DATES_FROM: future handles it)
    if result is None and text.lower().startswith("next "):
        result = dateparser.parse(text[5:], settings=settings_dict)

    return result


async def _send_confirmation_email(
    customer_email: str,
    customer_name: str,
    service_type: str,
    start_time: datetime,
    address: Optional[str],
    context: BusinessContext,
) -> bool:
    """
    Send appointment confirmation email.

    Returns True if email sent successfully.
    """
    from .email import email_tool

    # Format date/time for email
    date_str = start_time.strftime("%A, %B %d, %Y")
    time_str = start_time.strftime("%I:%M %p")

    # Build email body
    body_parts = [
        f"Dear {customer_name},",
        "",
        f"Your appointment has been confirmed!",
        "",
        f"Service: {service_type}",
        f"Date: {date_str}",
        f"Time: {time_str}",
    ]

    if address:
        body_parts.append(f"Location: {address}")

    body_parts.extend([
        "",
        f"If you need to reschedule or cancel, please call us.",
        "",
        f"Thank you for choosing {context.name}!",
        "",
        "Best regards,",
        context.name,
    ])

    email_body = "\n".join(body_parts)
    subject = f"Appointment Confirmed - {date_str} at {time_str}"

    try:
        result = await email_tool.execute({
            "to": customer_email,
            "subject": subject,
            "body": email_body,
        })

        if result.success:
            logger.info("Confirmation email sent to %s", customer_email)
            return True
        else:
            logger.warning("Failed to send confirmation email: %s", result.message)
            return False

    except Exception as e:
        logger.error("Error sending confirmation email: %s", e)
        return False


class CheckAvailabilityTool:
    """Check available appointment slots."""

    @property
    def name(self) -> str:
        return "check_availability"

    @property
    def description(self) -> str:
        return (
            "Check available appointment times. Use when user asks "
            "'what times are available', 'when can I schedule', etc."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="date",
                param_type="string",
                description="Date to check (e.g., 'tomorrow', 'next Tuesday', 'January 20')",
                required=False,
            ),
            ToolParameter(
                name="days_ahead",
                param_type="int",
                description="Number of days to search (default: 7)",
                required=False,
                default=7,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["availability", "available", "open slots", "free times"]

    @property
    def category(self) -> str:
        return "scheduling"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Find available appointment slots."""
        context = _get_default_context()
        if not context:
            return ToolResult(
                success=False,
                error="NO_CONTEXT",
                message="No business context configured for scheduling.",
            )

        if not context.scheduling.enabled:
            return ToolResult(
                success=False,
                error="SCHEDULING_DISABLED",
                message="Scheduling is not enabled for this business.",
            )

        if not context.scheduling.calendar_id:
            return ToolResult(
                success=False,
                error="CALENDAR_NOT_CONFIGURED",
                message="Calendar not configured. Set ATLAS_COMMS_EFFINGHAM_MAIDS_CALENDAR_ID.",
            )

        # Parse date if provided
        target_date = None
        date_text = params.get("date")
        if date_text:
            target_date = _parse_datetime(date_text, context.hours.timezone)

        days_ahead = params.get("days_ahead", 7)

        try:
            slots = await _get_scheduling_service().get_available_slots(
                context=context,
                date=target_date,
                days_ahead=days_ahead,
            )

            if not slots:
                return ToolResult(
                    success=True,
                    data={"slots": [], "count": 0},
                    message="No available appointments found in that time frame.",
                )

            # Format slots for response
            formatted_slots = []
            for slot in slots[:10]:  # Limit to 10 slots
                formatted_slots.append({
                    "start": slot.start.isoformat(),
                    "end": slot.end.isoformat(),
                    "display": str(slot),
                    "duration_minutes": slot.duration_minutes,
                })

            # Generate speech-friendly message
            speech_msg = _get_scheduling_service().format_slots_for_speech(slots)

            return ToolResult(
                success=True,
                data={"slots": formatted_slots, "count": len(slots)},
                message=speech_msg,
            )

        except Exception as e:
            logger.exception("Error checking availability")
            return ToolResult(
                success=False,
                error="AVAILABILITY_ERROR",
                message=f"Failed to check availability: {e}",
            )


class BookAppointmentTool:
    """Book a new appointment."""

    @property
    def name(self) -> str:
        return "book_appointment"

    @property
    def description(self) -> str:
        return (
            "Book an appointment for a customer. Use when user says "
            "'book an appointment', 'schedule for', 'set up appointment'."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="customer_name",
                param_type="string",
                description="Customer's full name",
                required=True,
            ),
            ToolParameter(
                name="customer_phone",
                param_type="string",
                description="Customer's phone number",
                required=True,
            ),
            ToolParameter(
                name="date",
                param_type="string",
                description="Appointment date (e.g., 'tomorrow', 'next Tuesday')",
                required=True,
            ),
            ToolParameter(
                name="time",
                param_type="string",
                description="Appointment time (e.g., '2pm', '10:30 AM')",
                required=True,
            ),
            ToolParameter(
                name="service_type",
                param_type="string",
                description="Type of service (e.g., 'cleaning estimate', 'deep clean')",
                required=False,
                default="Cleaning Estimate",
            ),
            ToolParameter(
                name="customer_email",
                param_type="string",
                description="Customer's email for confirmation",
                required=False,
            ),
            ToolParameter(
                name="address",
                param_type="string",
                description="Service address",
                required=False,
            ),
            ToolParameter(
                name="notes",
                param_type="string",
                description="Additional notes about the appointment",
                required=False,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["appointment", "booking", "book", "schedule appointment"]

    @property
    def category(self) -> str:
        return "scheduling"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Book an appointment."""
        context = _get_default_context()
        if not context:
            return ToolResult(
                success=False,
                error="NO_CONTEXT",
                message="No business context configured for scheduling.",
            )

        if not context.scheduling.calendar_id:
            return ToolResult(
                success=False,
                error="CALENDAR_NOT_CONFIGURED",
                message="Calendar not configured. Set ATLAS_COMMS_EFFINGHAM_MAIDS_CALENDAR_ID.",
            )

        # Validate required params
        customer_name = params.get("customer_name")
        customer_phone = params.get("customer_phone")
        date_text = params.get("date")
        time_text = params.get("time")

        if not all([customer_name, customer_phone, date_text, time_text]):
            return ToolResult(
                success=False,
                error="MISSING_PARAMS",
                message="Need customer name, phone, date, and time to book.",
            )

        # Parse datetime
        datetime_text = f"{date_text} at {time_text}"
        tz = ZoneInfo(context.hours.timezone)
        start_time = _parse_datetime(datetime_text, context.hours.timezone)

        if not start_time:
            return ToolResult(
                success=False,
                error="INVALID_DATETIME",
                message=f"Could not understand the date/time: {datetime_text}",
            )

        # Create time slot
        duration = context.scheduling.default_duration_minutes
        end_time = start_time + timedelta(minutes=duration)
        slot = _get_time_slot_class()(start=start_time, end=end_time)

        # Optional params
        service_type = params.get("service_type", "Cleaning Estimate")
        customer_email = params.get("customer_email")
        address = params.get("address")
        notes = params.get("notes")

        try:
            # Book via scheduling service (creates calendar event)
            appointment = await _get_scheduling_service().book_appointment(
                context=context,
                slot=slot,
                customer_name=customer_name,
                customer_phone=customer_phone,
                customer_email=customer_email,
                service_type=service_type,
                location=address,
                notes=notes,
            )

            if not appointment:
                return ToolResult(
                    success=False,
                    error="BOOKING_FAILED",
                    message="Failed to create calendar event. The slot may no longer be available.",
                )

            # Also save to database for tracking
            repo = get_appointment_repo()
            try:
                db_record = await repo.create(
                    start_time=start_time,
                    end_time=end_time,
                    service_type=service_type,
                    customer_name=customer_name,
                    customer_phone=customer_phone,
                    business_context_id=context.id,
                    customer_email=customer_email,
                    customer_address=address,
                    calendar_event_id=str(appointment.id),
                    notes=notes or "",
                )
                db_id = str(db_record.get("id", ""))
            except (DatabaseUnavailableError, DatabaseOperationError) as e:
                logger.warning("DB save failed but calendar event created: %s", e)
                db_id = None

            # Send confirmation email if email provided
            email_sent = False
            if customer_email:
                email_sent = await _send_confirmation_email(
                    customer_email=customer_email,
                    customer_name=customer_name,
                    service_type=service_type,
                    start_time=start_time,
                    address=address,
                    context=context,
                )

            # Format confirmation message
            time_str = start_time.strftime("%A, %B %d at %I:%M %p")
            confirm_msg = (
                f"Appointment booked for {customer_name} on {time_str}. "
                f"Service: {service_type}."
            )
            if email_sent:
                confirm_msg += " Confirmation email sent."
            elif customer_email:
                confirm_msg += " Could not send confirmation email."

            return ToolResult(
                success=True,
                data={
                    "appointment_id": db_id,
                    "calendar_event_id": appointment.id,
                    "customer_name": customer_name,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "service_type": service_type,
                    "confirmation_email_sent": email_sent,
                },
                message=confirm_msg,
            )

        except Exception as e:
            logger.exception("Error booking appointment")
            return ToolResult(
                success=False,
                error="BOOKING_ERROR",
                message=f"Failed to book appointment: {e}",
            )


class CancelAppointmentTool:
    """Cancel an existing appointment."""

    @property
    def name(self) -> str:
        return "cancel_appointment"

    @property
    def description(self) -> str:
        return (
            "Cancel an existing appointment. Use when user says "
            "'cancel appointment', 'cancel booking for'."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="customer_phone",
                param_type="string",
                description="Customer's phone number to find their appointment",
                required=False,
            ),
            ToolParameter(
                name="appointment_id",
                param_type="string",
                description="Specific appointment ID to cancel",
                required=False,
            ),
            ToolParameter(
                name="reason",
                param_type="string",
                description="Reason for cancellation",
                required=False,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["cancel", "cancel booking", "cancel appointment"]

    @property
    def category(self) -> str:
        return "scheduling"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Cancel an appointment."""
        context = _get_default_context()
        if not context:
            return ToolResult(
                success=False,
                error="NO_CONTEXT",
                message="No business context configured.",
            )

        # Calendar needed for deleting calendar event (optional but recommended)
        if not context.scheduling.calendar_id:
            logger.warning("No calendar_id configured - calendar event won't be deleted")

        customer_phone = params.get("customer_phone")
        appointment_id = params.get("appointment_id")
        reason = params.get("reason")

        if not customer_phone and not appointment_id:
            return ToolResult(
                success=False,
                error="MISSING_PARAMS",
                message="Need either customer phone or appointment ID to cancel.",
            )

        repo = get_appointment_repo()

        try:
            # Find appointment
            if appointment_id:
                from uuid import UUID
                appt = await repo.get_by_id(UUID(appointment_id))
                if not appt:
                    return ToolResult(
                        success=False,
                        error="NOT_FOUND",
                        message=f"No appointment found with ID {appointment_id}",
                    )
                appointments = [appt]
            else:
                appointments = await repo.get_by_phone(
                    customer_phone, status="confirmed", upcoming_only=True
                )
                if not appointments:
                    return ToolResult(
                        success=False,
                        error="NOT_FOUND",
                        message=f"No upcoming appointments found for {customer_phone}",
                    )

            # Cancel the first/only appointment
            appt = appointments[0]
            appt_id = appt["id"]
            calendar_event_id = appt.get("calendar_event_id")
            customer_name = appt.get("customer_name", "Customer")

            # Cancel in database
            from uuid import UUID
            cancelled = await repo.cancel(UUID(str(appt_id)), reason)

            # Cancel calendar event if exists
            if calendar_event_id and context.scheduling.calendar_id:
                try:
                    await _get_scheduling_service().cancel_appointment(
                        context, calendar_event_id
                    )
                except Exception as e:
                    logger.warning("Failed to cancel calendar event: %s", e)

            if cancelled:
                start_time = appt.get("start_time")
                if isinstance(start_time, datetime):
                    time_str = start_time.strftime("%A, %B %d at %I:%M %p")
                else:
                    time_str = str(start_time)

                return ToolResult(
                    success=True,
                    data={
                        "appointment_id": str(appt_id),
                        "customer_name": customer_name,
                        "was_scheduled_for": time_str,
                    },
                    message=f"Cancelled appointment for {customer_name} that was scheduled for {time_str}.",
                )
            else:
                return ToolResult(
                    success=False,
                    error="CANCEL_FAILED",
                    message="Could not cancel the appointment.",
                )

        except Exception as e:
            logger.exception("Error cancelling appointment")
            return ToolResult(
                success=False,
                error="CANCEL_ERROR",
                message=f"Failed to cancel appointment: {e}",
            )


class RescheduleAppointmentTool:
    """Reschedule an existing appointment."""

    @property
    def name(self) -> str:
        return "reschedule_appointment"

    @property
    def description(self) -> str:
        return (
            "Reschedule an existing appointment to a new time. "
            "Use when user says 'reschedule', 'move appointment to'."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="customer_phone",
                param_type="string",
                description="Customer's phone number to find their appointment",
                required=True,
            ),
            ToolParameter(
                name="new_date",
                param_type="string",
                description="New date (e.g., 'tomorrow', 'next Monday')",
                required=True,
            ),
            ToolParameter(
                name="new_time",
                param_type="string",
                description="New time (e.g., '2pm', '10:30 AM')",
                required=True,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["reschedule", "move appointment", "change appointment"]

    @property
    def category(self) -> str:
        return "scheduling"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Reschedule an appointment."""
        context = _get_default_context()
        if not context:
            return ToolResult(
                success=False,
                error="NO_CONTEXT",
                message="No business context configured.",
            )

        if not context.scheduling.calendar_id:
            return ToolResult(
                success=False,
                error="CALENDAR_NOT_CONFIGURED",
                message="Calendar not configured. Set ATLAS_COMMS_EFFINGHAM_MAIDS_CALENDAR_ID.",
            )

        customer_phone = params.get("customer_phone")
        new_date = params.get("new_date")
        new_time = params.get("new_time")

        if not all([customer_phone, new_date, new_time]):
            return ToolResult(
                success=False,
                error="MISSING_PARAMS",
                message="Need customer phone, new date, and new time to reschedule.",
            )

        repo = get_appointment_repo()

        try:
            # Find existing appointment
            appointments = await repo.get_by_phone(
                customer_phone, status="confirmed", upcoming_only=True
            )
            if not appointments:
                return ToolResult(
                    success=False,
                    error="NOT_FOUND",
                    message=f"No upcoming appointments found for {customer_phone}",
                )

            old_appt = appointments[0]
            customer_name = old_appt.get("customer_name")
            customer_email = old_appt.get("customer_email")
            service_type = old_appt.get("service_type", "Cleaning Estimate")
            address = old_appt.get("customer_address")
            notes = old_appt.get("notes")
            old_calendar_id = old_appt.get("calendar_event_id")

            # Parse new datetime
            datetime_text = f"{new_date} at {new_time}"
            new_start = _parse_datetime(datetime_text, context.hours.timezone)

            if not new_start:
                return ToolResult(
                    success=False,
                    error="INVALID_DATETIME",
                    message=f"Could not understand the new date/time: {datetime_text}",
                )

            duration = context.scheduling.default_duration_minutes
            new_end = new_start + timedelta(minutes=duration)
            new_slot = _get_time_slot_class()(start=new_start, end=new_end)

            # Book new slot FIRST (before cancelling old, for rollback safety)
            new_appointment = await _get_scheduling_service().book_appointment(
                context=context,
                slot=new_slot,
                customer_name=customer_name,
                customer_phone=customer_phone,
                customer_email=customer_email,
                service_type=service_type,
                location=address,
                notes=notes,
            )

            if not new_appointment:
                return ToolResult(
                    success=False,
                    error="RESCHEDULE_FAILED",
                    message="Could not book the new time slot. It may not be available.",
                )

            # Cancel old calendar event only after new booking succeeded
            if old_calendar_id and context.scheduling.calendar_id:
                try:
                    await _get_scheduling_service().cancel_appointment(
                        context, old_calendar_id
                    )
                except Exception as e:
                    logger.warning(
                        "New booking created but failed to cancel old event %s: %s",
                        old_calendar_id, e,
                    )

            # Update database record
            from uuid import UUID
            old_id = UUID(str(old_appt["id"]))
            await repo.update(
                old_id,
                start_time=new_start,
                end_time=new_end,
                calendar_event_id=new_appointment.id,
            )

            time_str = new_start.strftime("%A, %B %d at %I:%M %p")
            return ToolResult(
                success=True,
                data={
                    "appointment_id": str(old_id),
                    "calendar_event_id": new_appointment.id,
                    "customer_name": customer_name,
                    "new_start_time": new_start.isoformat(),
                    "new_end_time": new_end.isoformat(),
                },
                message=f"Rescheduled appointment for {customer_name} to {time_str}.",
            )

        except Exception as e:
            logger.exception("Error rescheduling appointment")
            return ToolResult(
                success=False,
                error="RESCHEDULE_ERROR",
                message=f"Failed to reschedule: {e}",
            )


class LookupCustomerTool:
    """Look up customer by phone or name."""

    @property
    def name(self) -> str:
        return "lookup_customer"

    @property
    def description(self) -> str:
        return (
            "Look up a customer's information and appointment history. "
            "Use when user asks 'find customer', 'look up', 'customer info for'."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="phone",
                param_type="string",
                description="Customer's phone number",
                required=False,
            ),
            ToolParameter(
                name="name",
                param_type="string",
                description="Customer's name (partial match supported)",
                required=False,
            ),
            ToolParameter(
                name="include_history",
                param_type="boolean",
                description="Include past appointments (default: true)",
                required=False,
                default=True,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["customer", "find customer", "lookup", "customer info"]

    @property
    def category(self) -> str:
        return "scheduling"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """
        Look up a customer.

        Strategy (in order):
          1. CRM contacts table — the single source of truth for customer data.
          2. Appointment rows — fallback for legacy records not yet in the CRM.

        When a match is found in the CRM the response includes full contact
        details plus linked appointments.  When found only via appointment rows
        the response matches the legacy format so callers behave identically.
        """
        phone = params.get("phone")
        name = params.get("name")
        include_history = params.get("include_history", True)

        if not phone and not name:
            return ToolResult(
                success=False,
                error="MISSING_PARAMS",
                message="Need either phone number or name to look up customer.",
            )

        # ------------------------------------------------------------------
        # Step 1: CRM lookup (contacts table via configured provider)
        # ------------------------------------------------------------------
        try:
            from ..services.crm_provider import get_crm_provider

            crm = get_crm_provider()
            crm_results = await crm.search_contacts(
                query=name,
                phone=phone,
                limit=5,
            )
            if crm_results:
                contact = crm_results[0]
                contact_id = str(contact.get("id", ""))
                appointments = []
                if contact_id:
                    appointments = await crm.get_contact_appointments(contact_id)

                # Build appointment summaries
                context = _get_default_context()
                tz_name = (
                    context.hours.timezone
                    if context
                    else settings.reminder.default_timezone
                )
                now = datetime.now(ZoneInfo(tz_name))
                upcoming, past = [], []
                for appt in appointments:
                    start_time = appt.get("start_time")
                    appt_summary = {
                        "id": str(appt.get("id", "")),
                        "date": start_time.strftime("%A, %B %d") if start_time else "Unknown",
                        "time": start_time.strftime("%I:%M %p") if start_time else "Unknown",
                        "service": appt.get("service_type"),
                        "status": appt.get("status"),
                    }
                    if start_time and start_time > now:
                        upcoming.append(appt_summary)
                    else:
                        past.append(appt_summary)

                customer_info = {
                    "name": contact.get("full_name"),
                    "phone": contact.get("phone"),
                    "email": contact.get("email"),
                    "address": contact.get("address"),
                    "contact_id": contact_id,
                    "source": "crm",
                }

                msg_parts = [f"Found customer: {customer_info['name']}"]
                if customer_info.get("phone"):
                    msg_parts.append(f"Phone: {customer_info['phone']}")
                if upcoming:
                    msg_parts.append(f"Has {len(upcoming)} upcoming appointment(s).")
                    next_appt = upcoming[0]
                    msg_parts.append(
                        f"Next: {next_appt['service']} on {next_appt['date']} at {next_appt['time']}."
                    )
                if past and include_history:
                    msg_parts.append(f"Has {len(past)} past appointment(s).")

                return ToolResult(
                    success=True,
                    data={
                        "found": True,
                        "customer": customer_info,
                        "contact": contact,
                        "upcoming_appointments": upcoming,
                        "past_appointments": past[:5],
                        "total_appointments": len(appointments),
                    },
                    message=" ".join(msg_parts),
                )
        except Exception as crm_exc:
            logger.debug("CRM lookup failed, falling back to appointment rows: %s", crm_exc)

        # ------------------------------------------------------------------
        # Step 2: Legacy fallback — scrape customer data from appointment rows
        # (covers contacts not yet in the CRM)
        # ------------------------------------------------------------------
        repo = get_appointment_repo()

        try:
            appointments = []

            # Search by phone first (more precise)
            if phone:
                appointments = await repo.get_by_phone(
                    phone,
                    status=None,  # All statuses
                    upcoming_only=not include_history,
                    limit=20,
                )

            # If no results by phone, try name search
            if not appointments and name:
                appointments = await repo.search_by_name(
                    name,
                    include_history=include_history,
                    limit=20,
                )

            if not appointments:
                search_term = phone or name
                return ToolResult(
                    success=True,
                    data={"found": False, "appointments": []},
                    message=f"No customer found matching '{search_term}'.",
                )

            # Extract customer info from most recent appointment
            latest = appointments[0]
            customer_info = {
                "name": latest.get("customer_name"),
                "phone": latest.get("customer_phone"),
                "email": latest.get("customer_email"),
                "address": latest.get("customer_address"),
                "source": "appointments",
            }

            # Separate upcoming vs past appointments
            context = _get_default_context()
            tz_name = context.hours.timezone if context else settings.reminder.default_timezone
            now = datetime.now(ZoneInfo(tz_name))
            upcoming = []
            past = []

            for appt in appointments:
                start_time = appt.get("start_time")
                appt_summary = {
                    "id": str(appt.get("id")),
                    "date": start_time.strftime("%A, %B %d") if start_time else "Unknown",
                    "time": start_time.strftime("%I:%M %p") if start_time else "Unknown",
                    "service": appt.get("service_type"),
                    "status": appt.get("status"),
                }

                if start_time and start_time > now:
                    upcoming.append(appt_summary)
                else:
                    past.append(appt_summary)

            # Build response message
            msg_parts = [f"Found customer: {customer_info['name']}"]
            if customer_info.get("phone"):
                msg_parts.append(f"Phone: {customer_info['phone']}")
            if upcoming:
                msg_parts.append(f"Has {len(upcoming)} upcoming appointment(s).")
                next_appt = upcoming[0]
                msg_parts.append(
                    f"Next: {next_appt['service']} on {next_appt['date']} at {next_appt['time']}."
                )
            if past and include_history:
                msg_parts.append(f"Has {len(past)} past appointment(s).")

            return ToolResult(
                success=True,
                data={
                    "found": True,
                    "customer": customer_info,
                    "upcoming_appointments": upcoming,
                    "past_appointments": past[:5],  # Limit history
                    "total_appointments": len(appointments),
                },
                message=" ".join(msg_parts),
            )

        except (DatabaseUnavailableError, DatabaseOperationError) as e:
            logger.error("Database error in customer lookup: %s", e)
            return ToolResult(
                success=False,
                error="DATABASE_ERROR",
                message="Could not access customer records.",
            )
        except Exception as e:
            logger.exception("Error looking up customer")
            return ToolResult(
                success=False,
                error="LOOKUP_ERROR",
                message=f"Failed to look up customer: {e}",
            )


# Module-level instances
check_availability_tool = CheckAvailabilityTool()
book_appointment_tool = BookAppointmentTool()
cancel_appointment_tool = CancelAppointmentTool()
reschedule_appointment_tool = RescheduleAppointmentTool()
lookup_customer_tool = LookupCustomerTool()
