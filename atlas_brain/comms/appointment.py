"""
Appointment scheduling state machine.

Manages the conversation flow for booking appointments over phone/SMS.
Integrates with calendar, email, and SMS services.

Flow:
    GREETING → COLLECT_SERVICE → COLLECT_DATE → CHECK_AVAILABILITY →
    OFFER_SLOTS → CONFIRM_SLOT → COLLECT_CUSTOMER_INFO → CONFIRM_DETAILS →
    CREATE_BOOKING → SEND_CONFIRMATIONS → COMPLETE
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Optional, TYPE_CHECKING
from uuid import UUID

from .config import BusinessContext, BusinessHours, SchedulingConfig
from .services import (
    Appointment,
    CalendarService,
    EmailService,
    SMSService,
    TimeSlot,
    StubCalendarService,
    StubEmailService,
    StubSMSService,
)

if TYPE_CHECKING:
    from .protocols import TelephonyProvider, Call

logger = logging.getLogger("atlas.comms.appointment")


# === Scheduling Helpers ===


def parse_time(time_str: str) -> Optional[time]:
    """Parse a time string like '09:00' to a time object."""
    if not time_str:
        return None
    try:
        parts = time_str.split(":")
        return time(int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except (ValueError, IndexError):
        return None


def is_within_business_hours(
    dt: datetime,
    hours: BusinessHours,
) -> bool:
    """Check if a datetime falls within business hours.

    Converts the datetime to business timezone before comparing
    to ensure correct evaluation regardless of input timezone.
    """
    from zoneinfo import ZoneInfo

    # Convert to business timezone for accurate time comparison
    if dt.tzinfo is not None:
        business_tz = ZoneInfo(hours.timezone)
        dt_local = dt.astimezone(business_tz)
    else:
        dt_local = dt

    day_name = dt_local.strftime("%A").lower()
    open_attr = f"{day_name}_open"
    close_attr = f"{day_name}_close"

    open_time = parse_time(getattr(hours, open_attr, None))
    close_time = parse_time(getattr(hours, close_attr, None))

    if open_time is None or close_time is None:
        return False  # Closed this day

    dt_time = dt_local.time()
    return open_time <= dt_time < close_time


def get_business_hours_for_day(
    dt: datetime,
    hours: BusinessHours,
) -> tuple[Optional[time], Optional[time]]:
    """Get open/close times for a specific day."""
    day_name = dt.strftime("%A").lower()
    open_time = parse_time(getattr(hours, f"{day_name}_open", None))
    close_time = parse_time(getattr(hours, f"{day_name}_close", None))
    return open_time, close_time


# === States ===


class AppointmentState(Enum):
    """States of the appointment scheduling conversation."""

    # Initial state
    IDLE = auto()  # Not in an appointment flow
    GREETING = auto()  # Initial greeting, detecting intent

    # Information gathering
    COLLECT_SERVICE = auto()  # Asking what service they need
    COLLECT_DATE = auto()  # Asking when they'd like to schedule
    CHECK_AVAILABILITY = auto()  # Querying calendar for slots
    OFFER_SLOTS = auto()  # Presenting available times
    CONFIRM_SLOT = auto()  # User selecting a time

    # Customer information
    COLLECT_NAME = auto()  # Getting customer name
    COLLECT_PHONE = auto()  # Getting/confirming phone number
    COLLECT_EMAIL = auto()  # Getting email for confirmation
    COLLECT_ADDRESS = auto()  # Getting service address

    # Confirmation and booking
    CONFIRM_DETAILS = auto()  # Reviewing all details
    CREATE_BOOKING = auto()  # Creating calendar event
    SEND_CONFIRMATIONS = auto()  # Sending email/SMS confirmations
    COMPLETE = auto()  # Successfully completed

    # Alternative flows
    TRANSFER = auto()  # Transferring to human
    TAKE_MESSAGE = auto()  # Taking a message instead
    RESCHEDULE = auto()  # Rescheduling existing appointment
    CANCEL_APPOINTMENT = auto()  # Cancelling appointment

    # Terminal states
    ENDED = auto()  # Conversation ended
    ERROR = auto()  # Error occurred


class AppointmentEvent(Enum):
    """Events that trigger state transitions."""

    # Intent detection
    INTENT_SCHEDULE = auto()  # User wants to schedule
    INTENT_RESCHEDULE = auto()  # User wants to reschedule
    INTENT_CANCEL = auto()  # User wants to cancel
    INTENT_QUOTE = auto()  # User wants pricing info
    INTENT_QUESTION = auto()  # User has a question
    INTENT_TRANSFER = auto()  # User wants to speak to human
    INTENT_MESSAGE = auto()  # User wants to leave message

    # Data collection
    SERVICE_PROVIDED = auto()  # User provided service type
    DATE_PROVIDED = auto()  # User provided date preference
    SLOT_SELECTED = auto()  # User selected a time slot
    NAME_PROVIDED = auto()  # User provided name
    PHONE_PROVIDED = auto()  # User provided phone
    EMAIL_PROVIDED = auto()  # User provided email
    ADDRESS_PROVIDED = auto()  # User provided address
    INFO_SKIPPED = auto()  # User skipped optional info

    # Availability
    AVAILABILITY_FOUND = auto()  # Slots available
    NO_AVAILABILITY = auto()  # No slots in requested window
    SLOT_UNAVAILABLE = auto()  # Selected slot no longer available

    # Confirmation
    CONFIRMED = auto()  # User confirmed
    DECLINED = auto()  # User declined/wants changes
    DETAILS_CHANGED = auto()  # User wants to change something

    # Booking outcomes
    BOOKING_SUCCESS = auto()  # Calendar event created
    BOOKING_FAILED = auto()  # Failed to create event
    CONFIRMATIONS_SENT = auto()  # Email/SMS sent
    CONFIRMATION_FAILED = auto()  # Failed to send confirmations

    # Control events
    TIMEOUT = auto()  # User didn't respond
    CANCEL = auto()  # Cancel current operation
    RESET = auto()  # Reset to start
    ERROR_OCCURRED = auto()  # Error happened
    HANGUP = auto()  # User hung up
    INVALID_INPUT = auto()  # Could not understand user input
    TOO_MANY_RETRIES = auto()  # Max retry attempts exceeded


# === Transition Table ===


APPOINTMENT_TRANSITIONS: dict[tuple[AppointmentState, AppointmentEvent], AppointmentState] = {
    # From IDLE
    (AppointmentState.IDLE, AppointmentEvent.INTENT_SCHEDULE): AppointmentState.GREETING,
    (AppointmentState.IDLE, AppointmentEvent.RESET): AppointmentState.IDLE,

    # From GREETING - intent routing
    (AppointmentState.GREETING, AppointmentEvent.INTENT_SCHEDULE): AppointmentState.COLLECT_SERVICE,
    (AppointmentState.GREETING, AppointmentEvent.INTENT_RESCHEDULE): AppointmentState.RESCHEDULE,
    (AppointmentState.GREETING, AppointmentEvent.INTENT_CANCEL): AppointmentState.CANCEL_APPOINTMENT,
    (AppointmentState.GREETING, AppointmentEvent.INTENT_QUOTE): AppointmentState.COLLECT_SERVICE,
    (AppointmentState.GREETING, AppointmentEvent.INTENT_TRANSFER): AppointmentState.TRANSFER,
    (AppointmentState.GREETING, AppointmentEvent.INTENT_MESSAGE): AppointmentState.TAKE_MESSAGE,
    (AppointmentState.GREETING, AppointmentEvent.TIMEOUT): AppointmentState.ENDED,
    (AppointmentState.GREETING, AppointmentEvent.HANGUP): AppointmentState.ENDED,

    # From COLLECT_SERVICE
    (AppointmentState.COLLECT_SERVICE, AppointmentEvent.SERVICE_PROVIDED): AppointmentState.COLLECT_DATE,
    (AppointmentState.COLLECT_SERVICE, AppointmentEvent.INTENT_TRANSFER): AppointmentState.TRANSFER,
    (AppointmentState.COLLECT_SERVICE, AppointmentEvent.TIMEOUT): AppointmentState.ENDED,
    (AppointmentState.COLLECT_SERVICE, AppointmentEvent.HANGUP): AppointmentState.ENDED,

    # From COLLECT_DATE
    (AppointmentState.COLLECT_DATE, AppointmentEvent.DATE_PROVIDED): AppointmentState.CHECK_AVAILABILITY,
    (AppointmentState.COLLECT_DATE, AppointmentEvent.INTENT_TRANSFER): AppointmentState.TRANSFER,
    (AppointmentState.COLLECT_DATE, AppointmentEvent.TIMEOUT): AppointmentState.ENDED,
    (AppointmentState.COLLECT_DATE, AppointmentEvent.HANGUP): AppointmentState.ENDED,

    # From CHECK_AVAILABILITY
    (AppointmentState.CHECK_AVAILABILITY, AppointmentEvent.AVAILABILITY_FOUND): AppointmentState.OFFER_SLOTS,
    (AppointmentState.CHECK_AVAILABILITY, AppointmentEvent.NO_AVAILABILITY): AppointmentState.COLLECT_DATE,
    (AppointmentState.CHECK_AVAILABILITY, AppointmentEvent.ERROR_OCCURRED): AppointmentState.ERROR,

    # From OFFER_SLOTS
    (AppointmentState.OFFER_SLOTS, AppointmentEvent.SLOT_SELECTED): AppointmentState.CONFIRM_SLOT,
    (AppointmentState.OFFER_SLOTS, AppointmentEvent.DECLINED): AppointmentState.COLLECT_DATE,
    (AppointmentState.OFFER_SLOTS, AppointmentEvent.INTENT_TRANSFER): AppointmentState.TRANSFER,
    (AppointmentState.OFFER_SLOTS, AppointmentEvent.TIMEOUT): AppointmentState.ENDED,
    (AppointmentState.OFFER_SLOTS, AppointmentEvent.HANGUP): AppointmentState.ENDED,

    # From CONFIRM_SLOT
    (AppointmentState.CONFIRM_SLOT, AppointmentEvent.CONFIRMED): AppointmentState.COLLECT_NAME,
    (AppointmentState.CONFIRM_SLOT, AppointmentEvent.DECLINED): AppointmentState.OFFER_SLOTS,
    (AppointmentState.CONFIRM_SLOT, AppointmentEvent.SLOT_UNAVAILABLE): AppointmentState.CHECK_AVAILABILITY,

    # From COLLECT_NAME
    (AppointmentState.COLLECT_NAME, AppointmentEvent.NAME_PROVIDED): AppointmentState.COLLECT_PHONE,
    (AppointmentState.COLLECT_NAME, AppointmentEvent.TIMEOUT): AppointmentState.ENDED,
    (AppointmentState.COLLECT_NAME, AppointmentEvent.HANGUP): AppointmentState.ENDED,

    # From COLLECT_PHONE
    (AppointmentState.COLLECT_PHONE, AppointmentEvent.PHONE_PROVIDED): AppointmentState.COLLECT_EMAIL,
    (AppointmentState.COLLECT_PHONE, AppointmentEvent.INFO_SKIPPED): AppointmentState.COLLECT_EMAIL,
    (AppointmentState.COLLECT_PHONE, AppointmentEvent.TIMEOUT): AppointmentState.ENDED,
    (AppointmentState.COLLECT_PHONE, AppointmentEvent.HANGUP): AppointmentState.ENDED,

    # From COLLECT_EMAIL
    (AppointmentState.COLLECT_EMAIL, AppointmentEvent.EMAIL_PROVIDED): AppointmentState.COLLECT_ADDRESS,
    (AppointmentState.COLLECT_EMAIL, AppointmentEvent.INFO_SKIPPED): AppointmentState.COLLECT_ADDRESS,
    (AppointmentState.COLLECT_EMAIL, AppointmentEvent.TIMEOUT): AppointmentState.ENDED,
    (AppointmentState.COLLECT_EMAIL, AppointmentEvent.HANGUP): AppointmentState.ENDED,

    # From COLLECT_ADDRESS
    (AppointmentState.COLLECT_ADDRESS, AppointmentEvent.ADDRESS_PROVIDED): AppointmentState.CONFIRM_DETAILS,
    (AppointmentState.COLLECT_ADDRESS, AppointmentEvent.INFO_SKIPPED): AppointmentState.CONFIRM_DETAILS,
    (AppointmentState.COLLECT_ADDRESS, AppointmentEvent.TIMEOUT): AppointmentState.ENDED,
    (AppointmentState.COLLECT_ADDRESS, AppointmentEvent.HANGUP): AppointmentState.ENDED,

    # From CONFIRM_DETAILS
    (AppointmentState.CONFIRM_DETAILS, AppointmentEvent.CONFIRMED): AppointmentState.CREATE_BOOKING,
    (AppointmentState.CONFIRM_DETAILS, AppointmentEvent.DECLINED): AppointmentState.COLLECT_SERVICE,
    (AppointmentState.CONFIRM_DETAILS, AppointmentEvent.DETAILS_CHANGED): AppointmentState.COLLECT_SERVICE,

    # From CREATE_BOOKING
    (AppointmentState.CREATE_BOOKING, AppointmentEvent.BOOKING_SUCCESS): AppointmentState.SEND_CONFIRMATIONS,
    (AppointmentState.CREATE_BOOKING, AppointmentEvent.BOOKING_FAILED): AppointmentState.ERROR,

    # From SEND_CONFIRMATIONS
    (AppointmentState.SEND_CONFIRMATIONS, AppointmentEvent.CONFIRMATIONS_SENT): AppointmentState.COMPLETE,
    (AppointmentState.SEND_CONFIRMATIONS, AppointmentEvent.CONFIRMATION_FAILED): AppointmentState.COMPLETE,

    # From COMPLETE
    (AppointmentState.COMPLETE, AppointmentEvent.RESET): AppointmentState.IDLE,
    (AppointmentState.COMPLETE, AppointmentEvent.HANGUP): AppointmentState.ENDED,

    # From TRANSFER
    (AppointmentState.TRANSFER, AppointmentEvent.CONFIRMED): AppointmentState.ENDED,
    (AppointmentState.TRANSFER, AppointmentEvent.DECLINED): AppointmentState.GREETING,
    (AppointmentState.TRANSFER, AppointmentEvent.ERROR_OCCURRED): AppointmentState.TAKE_MESSAGE,

    # From TAKE_MESSAGE
    (AppointmentState.TAKE_MESSAGE, AppointmentEvent.CONFIRMED): AppointmentState.COMPLETE,
    (AppointmentState.TAKE_MESSAGE, AppointmentEvent.HANGUP): AppointmentState.ENDED,

    # From ERROR
    (AppointmentState.ERROR, AppointmentEvent.RESET): AppointmentState.IDLE,
    (AppointmentState.ERROR, AppointmentEvent.HANGUP): AppointmentState.ENDED,

    # From RESCHEDULE - need to find existing appointment first
    (AppointmentState.RESCHEDULE, AppointmentEvent.CONFIRMED): AppointmentState.COLLECT_DATE,
    (AppointmentState.RESCHEDULE, AppointmentEvent.DECLINED): AppointmentState.GREETING,
    (AppointmentState.RESCHEDULE, AppointmentEvent.HANGUP): AppointmentState.ENDED,
    (AppointmentState.RESCHEDULE, AppointmentEvent.ERROR_OCCURRED): AppointmentState.ERROR,

    # From CANCEL_APPOINTMENT
    (AppointmentState.CANCEL_APPOINTMENT, AppointmentEvent.CONFIRMED): AppointmentState.COMPLETE,
    (AppointmentState.CANCEL_APPOINTMENT, AppointmentEvent.DECLINED): AppointmentState.GREETING,
    (AppointmentState.CANCEL_APPOINTMENT, AppointmentEvent.HANGUP): AppointmentState.ENDED,
    (AppointmentState.CANCEL_APPOINTMENT, AppointmentEvent.ERROR_OCCURRED): AppointmentState.ERROR,

    # Too many retries escalation - from any collection state, go to transfer
    (AppointmentState.GREETING, AppointmentEvent.TOO_MANY_RETRIES): AppointmentState.TRANSFER,
    (AppointmentState.COLLECT_SERVICE, AppointmentEvent.TOO_MANY_RETRIES): AppointmentState.TRANSFER,
    (AppointmentState.COLLECT_DATE, AppointmentEvent.TOO_MANY_RETRIES): AppointmentState.TRANSFER,
    (AppointmentState.OFFER_SLOTS, AppointmentEvent.TOO_MANY_RETRIES): AppointmentState.TRANSFER,
    (AppointmentState.CONFIRM_SLOT, AppointmentEvent.TOO_MANY_RETRIES): AppointmentState.TRANSFER,
    (AppointmentState.COLLECT_NAME, AppointmentEvent.TOO_MANY_RETRIES): AppointmentState.TRANSFER,
    (AppointmentState.COLLECT_PHONE, AppointmentEvent.TOO_MANY_RETRIES): AppointmentState.TRANSFER,
    (AppointmentState.COLLECT_EMAIL, AppointmentEvent.TOO_MANY_RETRIES): AppointmentState.TRANSFER,
    (AppointmentState.COLLECT_ADDRESS, AppointmentEvent.TOO_MANY_RETRIES): AppointmentState.TRANSFER,
    (AppointmentState.CONFIRM_DETAILS, AppointmentEvent.TOO_MANY_RETRIES): AppointmentState.TRANSFER,
    (AppointmentState.RESCHEDULE, AppointmentEvent.TOO_MANY_RETRIES): AppointmentState.TRANSFER,
    (AppointmentState.CANCEL_APPOINTMENT, AppointmentEvent.TOO_MANY_RETRIES): AppointmentState.TRANSFER,
}


# === Context ===


@dataclass
class AppointmentContext:
    """
    Context passed through the appointment flow.

    Accumulates data as the conversation progresses.
    """

    # Session info
    session_id: UUID = field(default_factory=lambda: __import__("uuid").uuid4())
    started_at: datetime = field(default_factory=datetime.now)
    call_id: Optional[UUID] = None
    business_context_id: str = ""

    # Caller info (from caller ID or collected)
    caller_phone: str = ""
    caller_name: Optional[str] = None

    # Service request
    service_type: Optional[str] = None
    service_notes: str = ""

    # Date/time preferences
    preferred_date: Optional[datetime] = None
    preferred_time_of_day: Optional[str] = None  # "morning", "afternoon", "evening"
    flexibility: str = "flexible"  # "flexible", "specific"

    # Available slots (from calendar check)
    available_slots: list[TimeSlot] = field(default_factory=list)
    selected_slot: Optional[TimeSlot] = None

    # Customer information
    customer_name: str = ""
    customer_phone: str = ""
    customer_email: str = ""
    customer_address: str = ""

    # Created appointment
    appointment: Optional[Appointment] = None

    # For reschedule/cancel flows - existing appointment
    existing_appointments: list[dict] = field(default_factory=list)
    appointment_to_modify: Optional[dict] = None

    # Conversation tracking
    transcript: list[dict] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3

    # Error info
    error: Optional[str] = None
    error_details: Optional[dict] = None

    # Message (for TAKE_MESSAGE flow)
    message_text: str = ""

    def elapsed_seconds(self) -> float:
        """Time since session started."""
        return (datetime.now() - self.started_at).total_seconds()

    def add_turn(self, role: str, text: str) -> None:
        """Add a conversation turn to the transcript."""
        self.transcript.append({
            "role": role,
            "text": text,
            "timestamp": datetime.now().isoformat(),
        })

    def build_appointment(self) -> Appointment:
        """Build an Appointment from collected context."""
        if not self.selected_slot:
            raise ValueError("No time slot selected")

        return Appointment(
            start=self.selected_slot.start,
            end=self.selected_slot.end,
            service_type=self.service_type or "Appointment",
            duration_minutes=self.selected_slot.duration_minutes,
            customer_name=self.customer_name,
            customer_phone=self.customer_phone or self.caller_phone,
            customer_email=self.customer_email,
            customer_address=self.customer_address,
            notes=self.service_notes,
            business_context_id=self.business_context_id,
            call_id=self.call_id,
        )


# === State Machine ===


# Type alias for state handlers
StateHandler = Callable[["AppointmentStateMachine"], Coroutine[Any, Any, Optional[str]]]


class AppointmentStateMachine:
    """
    State machine for appointment scheduling conversations.

    Manages transitions and executes state-specific logic.

    Usage:
        machine = AppointmentStateMachine(
            business_context=context,
            calendar_service=calendar,
            email_service=email,
            sms_service=sms,
        )

        # Start appointment flow
        response = await machine.start(caller_phone="+1234567890")

        # Process user input
        response = await machine.process_input("I'd like to schedule a cleaning")

        # Check state
        if machine.state == AppointmentState.COMPLETE:
            print(f"Booked: {machine.context.appointment}")
    """

    def __init__(
        self,
        business_context: BusinessContext,
        calendar_service: Optional[CalendarService] = None,
        email_service: Optional[EmailService] = None,
        sms_service: Optional[SMSService] = None,
        telephony_provider: Optional["TelephonyProvider"] = None,
        call: Optional["Call"] = None,
        intent_parser: Optional[Callable] = None,  # LLM-based parser
    ):
        self.business_context = business_context
        self.scheduling_config = business_context.scheduling

        # Services (use stubs if not provided)
        self.calendar = calendar_service or StubCalendarService()
        self.email = email_service or StubEmailService()
        self.sms = sms_service or StubSMSService()

        # Telephony for transfer operations
        self._telephony_provider = telephony_provider
        self._call = call

        # Optional LLM-based intent parser
        self._intent_parser = intent_parser

        # State
        self._state = AppointmentState.IDLE
        self._context = AppointmentContext(
            business_context_id=business_context.id,
        )

        # Listeners for state changes
        self._listeners: list[Callable] = []

        # State handlers - each state has a handler that generates the response
        self._state_handlers: dict[AppointmentState, StateHandler] = {
            AppointmentState.IDLE: self._handle_idle,
            AppointmentState.GREETING: self._handle_greeting,
            AppointmentState.COLLECT_SERVICE: self._handle_collect_service,
            AppointmentState.COLLECT_DATE: self._handle_collect_date,
            AppointmentState.CHECK_AVAILABILITY: self._handle_check_availability,
            AppointmentState.OFFER_SLOTS: self._handle_offer_slots,
            AppointmentState.CONFIRM_SLOT: self._handle_confirm_slot,
            AppointmentState.COLLECT_NAME: self._handle_collect_name,
            AppointmentState.COLLECT_PHONE: self._handle_collect_phone,
            AppointmentState.COLLECT_EMAIL: self._handle_collect_email,
            AppointmentState.COLLECT_ADDRESS: self._handle_collect_address,
            AppointmentState.CONFIRM_DETAILS: self._handle_confirm_details,
            AppointmentState.CREATE_BOOKING: self._handle_create_booking,
            AppointmentState.SEND_CONFIRMATIONS: self._handle_send_confirmations,
            AppointmentState.COMPLETE: self._handle_complete,
            AppointmentState.TRANSFER: self._handle_transfer,
            AppointmentState.TAKE_MESSAGE: self._handle_take_message,
            AppointmentState.RESCHEDULE: self._handle_reschedule,
            AppointmentState.CANCEL_APPOINTMENT: self._handle_cancel_appointment,
            AppointmentState.ERROR: self._handle_error,
        }

    @property
    def state(self) -> AppointmentState:
        return self._state

    @property
    def context(self) -> AppointmentContext:
        return self._context

    def reset(self) -> None:
        """Reset to idle state with fresh context."""
        self._state = AppointmentState.IDLE
        self._context = AppointmentContext(
            business_context_id=self.business_context.id,
        )
        self._notify(AppointmentEvent.RESET, None, AppointmentState.IDLE)

    async def start(
        self,
        caller_phone: str = "",
        caller_name: Optional[str] = None,
        call_id: Optional[UUID] = None,
    ) -> str:
        """
        Start a new appointment scheduling session.

        Returns the initial greeting response.
        """
        self._context.caller_phone = caller_phone
        self._context.caller_name = caller_name
        self._context.call_id = call_id
        self._context.customer_phone = caller_phone  # Pre-fill from caller ID

        # Transition to greeting
        self._transition(AppointmentEvent.INTENT_SCHEDULE)

        # Generate greeting
        return await self._execute_handler()

    async def process_input(self, user_input: str) -> str:
        """
        Process user input and return response.

        This is the main method called for each turn of conversation.
        Includes retry logic - after max_retries invalid inputs, escalates to transfer.
        """
        self._context.add_turn("user", user_input)

        # Parse intent and extract data based on current state
        event, data = await self._parse_input(user_input)

        # Apply extracted data to context
        self._apply_data(data)

        # Handle retry logic for invalid/unrecognized input
        if event is None:
            self._context.retry_count += 1
            logger.debug(
                "No event parsed, retry count: %d/%d",
                self._context.retry_count,
                self._context.max_retries,
            )

            if self._context.retry_count >= self._context.max_retries:
                # Too many failed attempts - escalate to transfer
                logger.info(
                    "Max retries reached in state %s, escalating to transfer",
                    self._state.name,
                )
                event = AppointmentEvent.TOO_MANY_RETRIES
                self._context.retry_count = 0
            else:
                # Return a retry prompt without transitioning
                response = self._get_retry_prompt()
                self._context.add_turn("assistant", response)
                return response
        else:
            # Valid input - reset retry counter
            self._context.retry_count = 0

        # Perform transition
        if event:
            success = self._transition(event)
            if not success:
                logger.warning(
                    "Invalid transition: %s + %s",
                    self._state.name,
                    event.name,
                )

        # Execute handler for current state and get response
        response = await self._execute_handler()

        self._context.add_turn("assistant", response)
        return response

    def _get_retry_prompt(self) -> str:
        """Get a context-appropriate retry prompt based on current state."""
        state_prompts = {
            AppointmentState.GREETING: "I'm sorry, I didn't quite catch that. Would you like to schedule an appointment, reschedule, or speak with someone?",
            AppointmentState.COLLECT_SERVICE: "I'm sorry, I didn't understand. What type of service are you looking for?",
            AppointmentState.COLLECT_DATE: "I'm sorry, I didn't catch that. When would you like to schedule? You can say things like 'tomorrow' or 'next Monday'.",
            AppointmentState.OFFER_SLOTS: "I'm sorry, I didn't understand your selection. Could you tell me which time slot number you'd prefer?",
            AppointmentState.CONFIRM_SLOT: "I'm sorry, I didn't catch that. Does this time work for you? Please say yes or no.",
            AppointmentState.COLLECT_NAME: "I'm sorry, I didn't catch your name. Could you please tell me your name?",
            AppointmentState.COLLECT_PHONE: "I'm sorry, I couldn't get that. Could you please provide your phone number?",
            AppointmentState.COLLECT_EMAIL: "I'm sorry, I didn't catch that. What's your email address? Or say 'skip' if you don't want to provide one.",
            AppointmentState.COLLECT_ADDRESS: "I'm sorry, I didn't understand. What's the service address? Or say 'skip' to continue.",
            AppointmentState.CONFIRM_DETAILS: "I'm sorry, I didn't catch that. Does everything look correct? Please say yes to confirm or no to make changes.",
            AppointmentState.RESCHEDULE: "I'm sorry, I didn't understand. Which appointment would you like to reschedule? Please say the number.",
            AppointmentState.CANCEL_APPOINTMENT: "I'm sorry, I didn't catch that. Which appointment would you like to cancel? Please say the number.",
        }
        return state_prompts.get(
            self._state,
            "I'm sorry, I didn't understand. Could you please repeat that?"
        )

    def _transition(self, event: AppointmentEvent) -> bool:
        """Attempt a state transition."""
        key = (self._state, event)
        new_state = APPOINTMENT_TRANSITIONS.get(key)

        if new_state is None:
            return False

        old_state = self._state
        self._state = new_state

        logger.debug(
            "Transition: %s -[%s]-> %s",
            old_state.name,
            event.name,
            new_state.name,
        )

        self._notify(event, old_state, new_state)
        return True

    def can_transition(self, event: AppointmentEvent) -> bool:
        """Check if a transition is valid without performing it."""
        return (self._state, event) in APPOINTMENT_TRANSITIONS

    def add_listener(self, callback: Callable) -> None:
        """Add a listener for state changes."""
        self._listeners.append(callback)

    def _notify(
        self,
        event: AppointmentEvent,
        old_state: Optional[AppointmentState],
        new_state: AppointmentState,
    ) -> None:
        """Notify listeners of state change."""
        for listener in self._listeners:
            try:
                result = listener(event, old_state, new_state, self._context)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as e:
                logger.error("Listener error: %s", e)

    async def _execute_handler(self) -> str:
        """Execute the handler for the current state."""
        handler = self._state_handlers.get(self._state)
        if handler:
            response = await handler()
            return response or ""
        return ""

    async def _parse_input(self, user_input: str) -> tuple[Optional[AppointmentEvent], dict]:
        """
        Parse user input to determine event and extract data.

        Returns (event, extracted_data).

        Uses LLM parser if available, otherwise falls back to keyword matching.
        """
        # Try LLM-based parsing first if available
        if self._intent_parser:
            try:
                result = await self._intent_parser(
                    user_input,
                    self._state,
                    self._context,
                    self.business_context,
                )
                if result:
                    return result
            except Exception as e:
                logger.warning("LLM intent parser failed, using fallback: %s", e)

        # Fallback to keyword-based parsing
        text = user_input.lower().strip()
        data = {}

        # Intent detection keywords
        if self._state == AppointmentState.GREETING:
            if any(w in text for w in ["schedule", "book", "appointment", "cleaning", "service"]):
                return AppointmentEvent.INTENT_SCHEDULE, data
            elif any(w in text for w in ["reschedule", "change", "move"]):
                return AppointmentEvent.INTENT_RESCHEDULE, data
            elif any(w in text for w in ["cancel", "nevermind"]):
                return AppointmentEvent.INTENT_CANCEL, data
            elif any(w in text for w in ["price", "cost", "quote", "how much"]):
                return AppointmentEvent.INTENT_QUOTE, data
            elif any(w in text for w in ["speak", "talk", "person", "human", "owner"]):
                return AppointmentEvent.INTENT_TRANSFER, data
            elif any(w in text for w in ["message", "call back"]):
                return AppointmentEvent.INTENT_MESSAGE, data

        # Service collection
        elif self._state == AppointmentState.COLLECT_SERVICE:
            services = self.business_context.services
            for service in services:
                if service.lower() in text:
                    data["service_type"] = service
                    return AppointmentEvent.SERVICE_PROVIDED, data
            # Accept any response as service description
            if len(text) > 2:
                data["service_type"] = user_input.strip()
                return AppointmentEvent.SERVICE_PROVIDED, data

        # Date collection
        elif self._state == AppointmentState.COLLECT_DATE:
            # Simple date parsing - in production use dateparser
            from datetime import date
            from zoneinfo import ZoneInfo

            # Use business timezone for date calculations
            tz = ZoneInfo(self.business_context.hours.timezone)
            today = date.today()

            if "today" in text:
                data["preferred_date"] = datetime.combine(today, datetime.min.time(), tzinfo=tz)
            elif "tomorrow" in text:
                data["preferred_date"] = datetime.combine(today + timedelta(days=1), datetime.min.time(), tzinfo=tz)
            elif "next week" in text:
                data["preferred_date"] = datetime.combine(today + timedelta(days=7), datetime.min.time(), tzinfo=tz)
            elif any(w in text for w in ["monday", "tuesday", "wednesday", "thursday", "friday"]):
                # Find next occurrence of day
                days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                for i, day in enumerate(days):
                    if day in text:
                        days_ahead = i - today.weekday()
                        if days_ahead <= 0:
                            days_ahead += 7
                        data["preferred_date"] = datetime.combine(today + timedelta(days=days_ahead), datetime.min.time(), tzinfo=tz)
                        break
            else:
                # Default to this week
                data["preferred_date"] = datetime.combine(today, datetime.min.time(), tzinfo=tz)

            # Time of day preference
            if any(w in text for w in ["morning", "am"]):
                data["preferred_time_of_day"] = "morning"
            elif any(w in text for w in ["afternoon"]):
                data["preferred_time_of_day"] = "afternoon"
            elif any(w in text for w in ["evening", "pm"]):
                data["preferred_time_of_day"] = "evening"

            return AppointmentEvent.DATE_PROVIDED, data

        # Slot selection
        elif self._state == AppointmentState.OFFER_SLOTS:
            slots = self._context.available_slots
            # Check for slot number selection
            for i, slot in enumerate(slots[:5], 1):
                if str(i) in text or f"option {i}" in text or f"number {i}" in text:
                    data["selected_slot"] = slot
                    return AppointmentEvent.SLOT_SELECTED, data

            # Check for "first", "second", etc.
            ordinals = ["first", "second", "third", "fourth", "fifth"]
            for i, ordinal in enumerate(ordinals):
                if ordinal in text and i < len(slots):
                    data["selected_slot"] = slots[i]
                    return AppointmentEvent.SLOT_SELECTED, data

            if any(w in text for w in ["none", "different", "other"]):
                return AppointmentEvent.DECLINED, data

        # Slot confirmation
        elif self._state == AppointmentState.CONFIRM_SLOT:
            if any(w in text for w in ["yes", "yeah", "correct", "right", "good", "perfect", "works"]):
                return AppointmentEvent.CONFIRMED, data
            elif any(w in text for w in ["no", "nope", "wrong", "different"]):
                return AppointmentEvent.DECLINED, data

        # Name collection
        elif self._state == AppointmentState.COLLECT_NAME:
            # Accept any reasonable input as a name
            if len(text) > 1:
                # Clean up common prefixes
                name = user_input.strip()
                for prefix in ["my name is ", "this is ", "it's ", "i'm ", "i am "]:
                    if name.lower().startswith(prefix):
                        name = name[len(prefix):]
                data["customer_name"] = name.strip().title()
                return AppointmentEvent.NAME_PROVIDED, data

        # Phone collection
        elif self._state == AppointmentState.COLLECT_PHONE:
            # Extract digits
            digits = "".join(c for c in text if c.isdigit())
            if len(digits) >= 10:
                data["customer_phone"] = digits
                return AppointmentEvent.PHONE_PROVIDED, data
            elif any(w in text for w in ["same", "that's", "yes", "correct"]) and self._context.caller_phone:
                data["customer_phone"] = self._context.caller_phone
                return AppointmentEvent.PHONE_PROVIDED, data
            elif any(w in text for w in ["skip", "no email", "none"]):
                return AppointmentEvent.INFO_SKIPPED, data

        # Email collection
        elif self._state == AppointmentState.COLLECT_EMAIL:
            # Simple email detection
            if "@" in text and "." in text:
                # Extract email-like string
                words = text.split()
                for word in words:
                    if "@" in word and "." in word:
                        data["customer_email"] = word.strip(".,!?")
                        return AppointmentEvent.EMAIL_PROVIDED, data
            elif any(w in text for w in ["skip", "no email", "none", "don't have"]):
                return AppointmentEvent.INFO_SKIPPED, data

        # Address collection
        elif self._state == AppointmentState.COLLECT_ADDRESS:
            if len(text) > 5:
                data["customer_address"] = user_input.strip()
                return AppointmentEvent.ADDRESS_PROVIDED, data
            elif any(w in text for w in ["skip", "later", "none"]):
                return AppointmentEvent.INFO_SKIPPED, data

        # Details confirmation
        elif self._state == AppointmentState.CONFIRM_DETAILS:
            if any(w in text for w in ["yes", "correct", "right", "good", "book", "confirm"]):
                return AppointmentEvent.CONFIRMED, data
            elif any(w in text for w in ["no", "change", "wrong", "different"]):
                return AppointmentEvent.DECLINED, data

        # Transfer confirmation
        elif self._state == AppointmentState.TRANSFER:
            if any(w in text for w in ["yes", "please"]):
                return AppointmentEvent.CONFIRMED, data
            elif any(w in text for w in ["no", "nevermind"]):
                return AppointmentEvent.DECLINED, data

        # Take message
        elif self._state == AppointmentState.TAKE_MESSAGE:
            if len(text) > 2:
                data["message_text"] = user_input.strip()
                return AppointmentEvent.CONFIRMED, data

        # Reschedule - selecting from multiple appointments or confirming
        elif self._state == AppointmentState.RESCHEDULE:
            appointments = self._context.existing_appointments
            # If single appointment was found, user confirms to proceed
            if self._context.appointment_to_modify:
                # Any response with date info means they want to proceed
                return AppointmentEvent.CONFIRMED, data
            # Multiple appointments - select by number
            if appointments and len(appointments) > 1:
                for i, appt in enumerate(appointments[:5], 1):
                    if str(i) in text or f"option {i}" in text or f"number {i}" in text:
                        data["appointment_to_modify"] = appt
                        return AppointmentEvent.CONFIRMED, data
                ordinals = ["first", "second", "third", "fourth", "fifth"]
                for i, ordinal in enumerate(ordinals):
                    if ordinal in text and i < len(appointments):
                        data["appointment_to_modify"] = appointments[i]
                        return AppointmentEvent.CONFIRMED, data
            if any(w in text for w in ["no", "nevermind", "cancel"]):
                return AppointmentEvent.DECLINED, data

        # Cancel appointment - selecting from multiple or confirming cancellation
        elif self._state == AppointmentState.CANCEL_APPOINTMENT:
            appointments = self._context.existing_appointments
            # If we have appointment_to_modify, user is confirming cancellation
            if self._context.appointment_to_modify:
                if any(w in text for w in ["yes", "yeah", "correct", "sure", "confirm"]):
                    return AppointmentEvent.CONFIRMED, data
                elif any(w in text for w in ["no", "nevermind", "keep"]):
                    return AppointmentEvent.DECLINED, data
            # Multiple appointments - select by number
            elif appointments and len(appointments) > 1:
                for i, appt in enumerate(appointments[:5], 1):
                    if str(i) in text or f"option {i}" in text or f"number {i}" in text:
                        data["appointment_to_modify"] = appt
                        return AppointmentEvent.CONFIRMED, data
                ordinals = ["first", "second", "third", "fourth", "fifth"]
                for i, ordinal in enumerate(ordinals):
                    if ordinal in text and i < len(appointments):
                        data["appointment_to_modify"] = appointments[i]
                        return AppointmentEvent.CONFIRMED, data
            if any(w in text for w in ["no", "nevermind"]):
                return AppointmentEvent.DECLINED, data

        # Global: check for hangup/cancel
        if any(w in text for w in ["bye", "goodbye", "hang up", "end call"]):
            return AppointmentEvent.HANGUP, data

        if any(w in text for w in ["transfer", "speak to someone", "talk to owner"]):
            return AppointmentEvent.INTENT_TRANSFER, data

        return None, data

    def _apply_data(self, data: dict) -> None:
        """Apply extracted data to context."""
        if "service_type" in data:
            self._context.service_type = data["service_type"]
        if "preferred_date" in data:
            self._context.preferred_date = data["preferred_date"]
        if "preferred_time_of_day" in data:
            self._context.preferred_time_of_day = data["preferred_time_of_day"]
        if "selected_slot" in data:
            self._context.selected_slot = data["selected_slot"]
        if "customer_name" in data:
            self._context.customer_name = data["customer_name"]
        if "customer_phone" in data:
            self._context.customer_phone = data["customer_phone"]
        if "customer_email" in data:
            self._context.customer_email = data["customer_email"]
        if "customer_address" in data:
            self._context.customer_address = data["customer_address"]
        if "message_text" in data:
            self._context.message_text = data["message_text"]
        if "appointment_to_modify" in data:
            self._context.appointment_to_modify = data["appointment_to_modify"]
            # Pre-fill context with existing appointment data for reschedule
            appt = data["appointment_to_modify"]
            self._context.service_type = appt.get("service_type")
            self._context.customer_name = appt.get("customer_name", "")
            self._context.customer_email = appt.get("customer_email", "")
            self._context.customer_address = appt.get("customer_address", "")

    # === State Handlers ===

    async def _handle_idle(self) -> str:
        """Handle idle state - waiting for conversation to start."""
        # IDLE state typically doesn't produce output - it waits for start_session()
        # This handler is here for completeness; transitions happen via events
        return ""

    async def _handle_greeting(self) -> str:
        """Generate greeting response."""
        return self.business_context.greeting

    async def _handle_collect_service(self) -> str:
        """Ask about service type."""
        services = self.business_context.services
        if services:
            service_list = ", ".join(services[:-1]) + f", or {services[-1]}" if len(services) > 1 else services[0]
            return f"What type of service are you looking for? We offer {service_list}."
        return "What type of service can I help you schedule today?"

    async def _handle_collect_date(self) -> str:
        """Ask about preferred date/time."""
        config = self.scheduling_config

        if self._context.available_slots and not self._context.selected_slot:
            # We had no availability last time
            return (
                "I'm sorry, I couldn't find availability for that time. "
                "Would you like to try a different day or time?"
            )

        return (
            f"When would you like to schedule your {self._context.service_type or 'appointment'}? "
            f"We can book up to {config.max_advance_days} days in advance."
        )

    async def _handle_check_availability(self) -> str:
        """Check calendar for available slots with scheduling rule enforcement."""
        from zoneinfo import ZoneInfo

        config = self.scheduling_config
        hours = self.business_context.hours
        business_tz = ZoneInfo(hours.timezone)
        now = datetime.now(timezone.utc)
        now_local = now.astimezone(business_tz)

        # Enforce minimum notice
        min_start = now + timedelta(hours=config.min_notice_hours)
        # Use business timezone for fallback to ensure consistent slot generation
        preferred = self._context.preferred_date or now_local

        # If preferred date is before minimum notice, adjust
        if preferred < min_start:
            preferred = min_start.astimezone(business_tz)

        # Enforce maximum advance booking
        max_end = now + timedelta(days=config.max_advance_days)

        # Search window
        start = preferred
        end = min(preferred + timedelta(days=7), max_end)

        try:
            slots = await self.calendar.get_available_slots(
                date_start=start,
                date_end=end,
                duration_minutes=config.default_duration_minutes,
                buffer_minutes=config.buffer_minutes,
                calendar_id=config.calendar_id,
            )

            # Filter slots by business hours
            filtered_slots = []
            for slot in slots:
                if is_within_business_hours(slot.start, hours):
                    # Also check end time is within hours
                    if is_within_business_hours(
                        slot.end - timedelta(minutes=1), hours
                    ):
                        filtered_slots.append(slot)

            slots = filtered_slots

            # Filter by time of day preference
            if self._context.preferred_time_of_day and slots:
                preference_filtered = []
                for slot in slots:
                    hour = slot.start.hour
                    if self._context.preferred_time_of_day == "morning" and 6 <= hour < 12:
                        preference_filtered.append(slot)
                    elif self._context.preferred_time_of_day == "afternoon" and 12 <= hour < 17:
                        preference_filtered.append(slot)
                    elif self._context.preferred_time_of_day == "evening" and 17 <= hour < 21:
                        preference_filtered.append(slot)
                slots = preference_filtered or slots  # Fall back to all if no matches

            # Double-check against database for conflicts
            try:
                from ..storage.repositories.appointment import get_appointment_repo
                repo = get_appointment_repo()
                final_slots = []
                for slot in slots[:10]:  # Check up to 10 slots
                    has_conflict = await repo.check_conflict(
                        start=slot.start,
                        end=slot.end,
                        business_context_id=self._context.business_context_id,
                    )
                    if not has_conflict:
                        final_slots.append(slot)
                slots = final_slots
            except Exception as e:
                logger.warning("Could not check database conflicts: %s", e)
                # Continue with calendar-only slots

            self._context.available_slots = slots

            if slots:
                self._transition(AppointmentEvent.AVAILABILITY_FOUND)
            else:
                self._transition(AppointmentEvent.NO_AVAILABILITY)

        except Exception as e:
            logger.error("Failed to check availability: %s", e)
            self._context.error = str(e)
            self._transition(AppointmentEvent.ERROR_OCCURRED)

        return ""  # Handler for next state will generate response

    async def _handle_offer_slots(self) -> str:
        """Present available time slots."""
        slots = self._context.available_slots[:5]  # Max 5 options

        if not slots:
            return "I apologize, but I couldn't find any available times. Would you like to try a different week?"

        response = "I have the following times available:\n"
        for i, slot in enumerate(slots, 1):
            response += f"{i}. {slot}\n"

        response += "\nWhich time works best for you?"
        return response

    async def _handle_confirm_slot(self) -> str:
        """Confirm selected time slot."""
        slot = self._context.selected_slot
        if slot:
            return (
                f"Just to confirm, you'd like to schedule for {slot}. "
                "Is that correct?"
            )
        return "I'm sorry, I didn't catch which time you'd prefer. Could you tell me again?"

    async def _handle_collect_name(self) -> str:
        """Ask for customer name."""
        if self._context.caller_name:
            return f"I have your name as {self._context.caller_name}. Is that correct, or would you like to use a different name?"
        return "May I have your name for the appointment?"

    async def _handle_collect_phone(self) -> str:
        """Ask for phone number."""
        if self._context.caller_phone:
            # Format phone for readability
            phone = self._context.caller_phone
            if len(phone) == 10:
                formatted = f"({phone[:3]}) {phone[3:6]}-{phone[6:]}"
            else:
                formatted = phone
            return f"Is {formatted} the best number to reach you?"
        return "What's the best phone number to reach you?"

    async def _handle_collect_email(self) -> str:
        """Ask for email address."""
        return "What email address should I send the confirmation to? Or say 'skip' if you prefer not to provide one."

    async def _handle_collect_address(self) -> str:
        """Ask for service address."""
        if self.business_context.business_type in ["cleaning service", "home service"]:
            return "What's the address where you'd like the service performed?"
        return "Is there anything else you'd like me to note for your appointment?"

    async def _handle_confirm_details(self) -> str:
        """Review all appointment details."""
        ctx = self._context
        slot = ctx.selected_slot

        details = [
            f"Service: {ctx.service_type}",
            f"Date/Time: {slot}",
            f"Name: {ctx.customer_name}",
            f"Phone: {ctx.customer_phone}",
        ]

        if ctx.customer_email:
            details.append(f"Email: {ctx.customer_email}")
        if ctx.customer_address:
            details.append(f"Address: {ctx.customer_address}")

        return (
            "Let me confirm your appointment details:\n"
            + "\n".join(f"• {d}" for d in details)
            + "\n\nShall I go ahead and book this?"
        )

    async def _handle_create_booking(self) -> str:
        """Create the calendar event and persist to database."""
        try:
            appointment = self._context.build_appointment()

            # Final conflict check before booking
            try:
                from ..storage.repositories.appointment import get_appointment_repo
                repo = get_appointment_repo()
                has_conflict = await repo.check_conflict(
                    start=appointment.start,
                    end=appointment.end,
                    business_context_id=self._context.business_context_id,
                )
                if has_conflict:
                    logger.warning("Slot became unavailable during booking flow")
                    self._transition(AppointmentEvent.SLOT_UNAVAILABLE)
                    return ""
            except Exception as e:
                logger.warning("Could not verify slot availability: %s", e)

            # Create calendar event
            event_id = await self.calendar.create_event(
                appointment=appointment,
                calendar_id=self.scheduling_config.calendar_id,
            )
            appointment.calendar_event_id = event_id

            # Persist to database
            try:
                from ..storage.repositories.appointment import get_appointment_repo
                repo = get_appointment_repo()
                db_appointment = await repo.create(
                    start_time=appointment.start,
                    end_time=appointment.end,
                    service_type=appointment.service_type,
                    customer_name=appointment.customer_name,
                    customer_phone=appointment.customer_phone,
                    customer_email=appointment.customer_email,
                    customer_address=appointment.customer_address,
                    calendar_event_id=event_id,
                    business_context_id=self._context.business_context_id,
                    call_id=self._context.call_id,
                    notes=appointment.notes,
                )
                # Update appointment with database ID
                appointment.id = db_appointment["id"]
                logger.info(
                    "Persisted appointment %s to database",
                    appointment.id,
                )
            except Exception as e:
                logger.error("Failed to persist appointment to database: %s", e)
                # Continue anyway - calendar event was created

            self._context.appointment = appointment

            logger.info(
                "Created appointment %s for %s at %s",
                appointment.id,
                appointment.customer_name,
                appointment.start,
            )

            # If this is a reschedule, cancel the old appointment
            if self._context.appointment_to_modify:
                old_appt = self._context.appointment_to_modify
                try:
                    from ..storage.repositories.appointment import get_appointment_repo
                    repo = get_appointment_repo()
                    await repo.cancel(
                        appointment_id=old_appt["id"],
                        reason="Rescheduled to new time",
                    )
                    logger.info(
                        "Cancelled old appointment %s (rescheduled)",
                        old_appt["id"],
                    )

                    # Also delete from calendar
                    if old_appt.get("calendar_event_id"):
                        try:
                            await self.calendar.delete_event(
                                event_id=old_appt["calendar_event_id"],
                                calendar_id=self.scheduling_config.calendar_id,
                            )
                        except Exception as e:
                            logger.warning("Failed to delete old calendar event: %s", e)
                except Exception as e:
                    logger.error("Failed to cancel old appointment: %s", e)
                    # Continue anyway - new appointment was created

            self._transition(AppointmentEvent.BOOKING_SUCCESS)

        except Exception as e:
            logger.error("Failed to create booking: %s", e)
            self._context.error = str(e)
            self._transition(AppointmentEvent.BOOKING_FAILED)

        return ""  # Next handler will generate response

    async def _handle_send_confirmations(self) -> str:
        """Send confirmation email and SMS."""
        appointment = self._context.appointment
        if not appointment:
            self._transition(AppointmentEvent.CONFIRMATION_FAILED)
            return ""

        success = True

        # Send email if provided
        if appointment.customer_email:
            try:
                await self.email.send_appointment_confirmation(
                    appointment=appointment,
                    business_name=self.business_context.name,
                    business_phone=self.business_context.phone_numbers[0] if self.business_context.phone_numbers else None,
                )
                appointment.confirmation_sent = True
                logger.info("Sent confirmation email to %s", appointment.customer_email)
            except Exception as e:
                logger.error("Failed to send confirmation email: %s", e)
                success = False

        # Send SMS
        if appointment.customer_phone:
            try:
                await self.sms.send_appointment_confirmation_sms(
                    appointment=appointment,
                    business_name=self.business_context.name,
                )
                logger.info("Sent confirmation SMS to %s", appointment.customer_phone)
            except Exception as e:
                logger.error("Failed to send confirmation SMS: %s", e)
                success = False

        if success:
            self._transition(AppointmentEvent.CONFIRMATIONS_SENT)
        else:
            self._transition(AppointmentEvent.CONFIRMATION_FAILED)

        return ""

    async def _handle_complete(self) -> str:
        """Generate completion response."""
        appointment = self._context.appointment
        if appointment:
            response = (
                f"Your appointment has been booked for {appointment.start.strftime('%A, %B %d at %I:%M %p')}. "
            )
            if appointment.customer_email:
                response += "You'll receive a confirmation email shortly. "
            response += f"Thank you for choosing {self.business_context.name}. Have a great day!"
            return response

        return f"Thank you for calling {self.business_context.name}. Have a great day!"

    async def _handle_transfer(self) -> str:
        """Handle transfer to human."""
        transfer_number = self.business_context.transfer_number

        if transfer_number and self._telephony_provider and self._call:
            # Actually perform the transfer
            try:
                success = await self._telephony_provider.transfer(
                    self._call,
                    transfer_number,
                )
                if success:
                    self._transition(AppointmentEvent.CONFIRMED)
                    return "Please hold while I transfer you to someone who can help."
                else:
                    self._transition(AppointmentEvent.ERROR_OCCURRED)
            except Exception as e:
                logger.error("Transfer failed: %s", e)
                self._transition(AppointmentEvent.ERROR_OCCURRED)

        if transfer_number:
            # No telephony provider, but transfer number exists
            return "Please hold while I transfer you to someone who can help."

        return (
            "I apologize, but there's no one available to take your call right now. "
            "Would you like to leave a message instead?"
        )

    async def _handle_take_message(self) -> str:
        """Handle taking a message."""
        if self._context.message_text:
            # Persist the message to database
            try:
                from ..storage.repositories.appointment import get_appointment_repo
                repo = get_appointment_repo()
                await repo.create_message(
                    caller_phone=self._context.caller_phone,
                    caller_name=self._context.caller_name,
                    message_text=self._context.message_text,
                    business_context_id=self._context.business_context_id,
                    call_id=self._context.call_id,
                )
                logger.info("Saved message from %s", self._context.caller_phone)
            except Exception as e:
                logger.error("Failed to save message: %s", e)

            return (
                "Thank you. I've noted your message and someone will get back to you soon. "
                "Is there anything else I can help you with?"
            )
        return "Of course, what message would you like to leave?"

    async def _handle_reschedule(self) -> str:
        """Handle rescheduling an existing appointment."""
        # First time entering - look up existing appointments
        if not self._context.existing_appointments:
            try:
                from ..storage.repositories.appointment import get_appointment_repo
                repo = get_appointment_repo()
                appointments = await repo.get_by_phone(
                    phone=self._context.caller_phone,
                    status="confirmed",
                    upcoming_only=True,
                )
                self._context.existing_appointments = appointments
            except Exception as e:
                logger.error("Failed to lookup appointments: %s", e)
                self._transition(AppointmentEvent.ERROR_OCCURRED)
                return ""

        if not self._context.existing_appointments:
            return (
                "I don't see any upcoming appointments for this phone number. "
                "Would you like to schedule a new appointment instead?"
            )

        if len(self._context.existing_appointments) == 1:
            appt = self._context.existing_appointments[0]
            self._context.appointment_to_modify = appt
            # Pre-fill context with existing appointment data
            self._context.service_type = appt.get("service_type")
            self._context.customer_name = appt.get("customer_name", "")
            self._context.customer_email = appt.get("customer_email", "")
            self._context.customer_address = appt.get("customer_address", "")

            start_time = appt.get("start_time")
            if isinstance(start_time, datetime):
                formatted = start_time.strftime("%A, %B %d at %I:%M %p")
            else:
                formatted = str(start_time)

            return (
                f"I found your appointment for {appt.get('service_type')} "
                f"on {formatted}. "
                "When would you like to reschedule it to?"
            )

        # Multiple appointments - list them
        response = "I found multiple upcoming appointments:\n"
        for i, appt in enumerate(self._context.existing_appointments[:5], 1):
            start_time = appt.get("start_time")
            if isinstance(start_time, datetime):
                formatted = start_time.strftime("%m/%d at %I:%M %p")
            else:
                formatted = str(start_time)
            response += f"{i}. {appt.get('service_type')} on {formatted}\n"

        response += "\nWhich appointment would you like to reschedule?"
        return response

    async def _handle_cancel_appointment(self) -> str:
        """Handle cancelling an existing appointment."""
        # First time entering - look up existing appointments
        if not self._context.existing_appointments:
            try:
                from ..storage.repositories.appointment import get_appointment_repo
                repo = get_appointment_repo()
                appointments = await repo.get_by_phone(
                    phone=self._context.caller_phone,
                    status="confirmed",
                    upcoming_only=True,
                )
                self._context.existing_appointments = appointments
            except Exception as e:
                logger.error("Failed to lookup appointments: %s", e)
                self._transition(AppointmentEvent.ERROR_OCCURRED)
                return ""

        if not self._context.existing_appointments:
            return (
                "I don't see any upcoming appointments for this phone number. "
                "Is there anything else I can help you with?"
            )

        # If we have an appointment to cancel (selected or only one)
        if self._context.appointment_to_modify:
            appt = self._context.appointment_to_modify
            try:
                from ..storage.repositories.appointment import get_appointment_repo
                repo = get_appointment_repo()
                await repo.cancel(
                    appointment_id=appt["id"],
                    reason="Customer requested cancellation",
                )

                # Also delete from calendar if we have the event ID
                if appt.get("calendar_event_id"):
                    try:
                        await self.calendar.delete_event(
                            event_id=appt["calendar_event_id"],
                            calendar_id=self.scheduling_config.calendar_id,
                        )
                    except Exception as e:
                        logger.warning("Failed to delete calendar event: %s", e)

                self._transition(AppointmentEvent.CONFIRMED)
                return (
                    f"Your appointment for {appt.get('service_type')} has been cancelled. "
                    "Is there anything else I can help you with?"
                )
            except Exception as e:
                logger.error("Failed to cancel appointment: %s", e)
                self._transition(AppointmentEvent.ERROR_OCCURRED)
                return ""

        if len(self._context.existing_appointments) == 1:
            self._context.appointment_to_modify = self._context.existing_appointments[0]
            appt = self._context.appointment_to_modify

            start_time = appt.get("start_time")
            if isinstance(start_time, datetime):
                formatted = start_time.strftime("%A, %B %d at %I:%M %p")
            else:
                formatted = str(start_time)

            return (
                f"I found your appointment for {appt.get('service_type')} "
                f"on {formatted}. "
                "Are you sure you want to cancel this appointment?"
            )

        # Multiple appointments
        response = "I found multiple upcoming appointments:\n"
        for i, appt in enumerate(self._context.existing_appointments[:5], 1):
            start_time = appt.get("start_time")
            if isinstance(start_time, datetime):
                formatted = start_time.strftime("%m/%d at %I:%M %p")
            else:
                formatted = str(start_time)
            response += f"{i}. {appt.get('service_type')} on {formatted}\n"

        response += "\nWhich appointment would you like to cancel?"
        return response

    async def _handle_error(self) -> str:
        """Handle error state."""
        return (
            "I apologize, but I'm having some technical difficulties. "
            "Would you like to leave a message, or would you prefer to call back later?"
        )


# === Factory Functions ===


def create_appointment_machine(
    business_context: BusinessContext,
    calendar_service: Optional[CalendarService] = None,
    email_service: Optional[EmailService] = None,
    sms_service: Optional[SMSService] = None,
    telephony_provider: Optional["TelephonyProvider"] = None,
    call: Optional["Call"] = None,
    intent_parser: Optional[Callable] = None,
) -> AppointmentStateMachine:
    """
    Factory function to create an appointment state machine.

    Args:
        business_context: Business configuration (hours, services, etc.)
        calendar_service: Calendar integration (or stub if None)
        email_service: Email integration (or stub if None)
        sms_service: SMS integration (or stub if None)
        telephony_provider: For call transfer operations
        call: The current call object (for transfers)
        intent_parser: Optional LLM-based intent parser callback.
            Signature: async (user_input, state, context, business_context) -> (event, data)

    Returns:
        Configured AppointmentStateMachine instance
    """
    return AppointmentStateMachine(
        business_context=business_context,
        calendar_service=calendar_service,
        email_service=email_service,
        sms_service=sms_service,
        telephony_provider=telephony_provider,
        call=call,
        intent_parser=intent_parser,
    )
