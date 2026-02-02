"""
Graph-driven booking workflow demonstrating tool chaining.

This workflow shows how LangGraph orchestrates multi-step tool execution:
1. parse_request - LLM extracts structured data from user input
2. lookup_customer - Tool finds customer in CRM
3. check_availability - Tool checks calendar slots
4. book_appointment - Tool creates the booking
5. confirm_booking - LLM generates confirmation response

The graph handles routing; the LLM makes decisions at specific nodes.

Set USE_REAL_TOOLS=true to use actual Google Calendar integration.
"""

import logging
import os
import time
from typing import Literal, Optional

from langgraph.graph import END, StateGraph

from .state import BookingWorkflowState
from .workflow_state import get_workflow_state_manager

logger = logging.getLogger("atlas.agents.graphs.booking")

# Workflow type identifier for state persistence
BOOKING_WORKFLOW_TYPE = "booking"

# Toggle between mock and real tools
USE_REAL_TOOLS = os.environ.get("USE_REAL_TOOLS", "false").lower() == "true"


# =============================================================================
# Tool Functions - Mock or Real based on USE_REAL_TOOLS
# =============================================================================


def _get_real_tools():
    """Lazy import real tools to avoid circular imports."""
    from ...tools.scheduling import (
        lookup_customer_tool,
        check_availability_tool,
        book_appointment_tool,
    )
    return {
        "lookup_customer": lookup_customer_tool,
        "check_availability": check_availability_tool,
        "book_appointment": book_appointment_tool,
    }


async def tool_lookup_customer(
    name: Optional[str] = None,
    phone: Optional[str] = None,
) -> dict:
    """Look up customer in CRM by name or phone."""
    if USE_REAL_TOOLS:
        tools = _get_real_tools()
        result = await tools["lookup_customer"].execute({
            "name": name,
            "phone": phone,
            "include_history": True,
        })
        if result.success and result.data.get("found"):
            customer = result.data.get("customer", {})
            return {
                "found": True,
                "customer_id": customer.get("phone", ""),
                "name": customer.get("name"),
                "phone": customer.get("phone"),
                "email": customer.get("email"),
                "address": customer.get("address"),
            }
        return {"found": False}

    # Mock implementation for testing
    mock_customers = {
        "john smith": {
            "customer_id": "cust_001",
            "name": "John Smith",
            "phone": "555-1234",
            "email": "john@example.com",
            "address": "123 Main St",
        },
        "jane doe": {
            "customer_id": "cust_002",
            "name": "Jane Doe",
            "phone": "555-5678",
            "email": "jane@example.com",
            "address": "456 Oak Ave",
        },
    }

    if name:
        key = name.lower()
        if key in mock_customers:
            return {"found": True, **mock_customers[key]}

    if phone:
        for customer in mock_customers.values():
            if customer["phone"] == phone:
                return {"found": True, **customer}

    return {"found": False}


async def tool_check_availability(
    date: str,
    time_slot: Optional[str] = None,
    service_type: Optional[str] = None,
) -> dict:
    """Check calendar availability for requested date/time."""
    if USE_REAL_TOOLS:
        tools = _get_real_tools()
        result = await tools["check_availability"].execute({
            "date": date,
            "days_ahead": 7,
        })
        if result.success:
            slots = result.data.get("slots", [])
            if slots:
                # Check if requested time is in available slots
                for slot in slots:
                    if time_slot and time_slot.lower() in slot.get("display", "").lower():
                        # Extract just the time from the ISO start time
                        start_iso = slot.get("start", "")
                        return {
                            "available": True,
                            "confirmed_slot": slot.get("display"),
                            "confirmed_start": start_iso,  # ISO datetime for booking
                            "date": date,
                            "slot_data": slot,
                        }
                # Return alternatives
                return {
                    "available": False,
                    "alternatives": [s.get("display") for s in slots[:5]],
                }
            return {"available": False, "alternatives": []}
        return {"available": False, "alternatives": [], "error": result.error}

    # Mock implementation for testing
    from datetime import datetime, timedelta
    today = datetime.now().date()
    mock_available = {
        today.isoformat(): ["9:00 AM", "11:00 AM", "2:00 PM", "4:00 PM"],
        (today + timedelta(days=1)).isoformat(): ["10:00 AM", "1:00 PM", "3:00 PM"],
        (today + timedelta(days=2)).isoformat(): ["9:00 AM", "10:00 AM", "11:00 AM"],
    }

    available_slots = mock_available.get(date, [])

    if time_slot and time_slot in available_slots:
        return {"available": True, "confirmed_slot": time_slot, "date": date}
    elif available_slots:
        return {"available": False, "alternatives": [f"{date} {s}" for s in available_slots]}
    else:
        alternatives = []
        for alt_date, slots in mock_available.items():
            if slots:
                alternatives.append(f"{alt_date} {slots[0]}")
        return {"available": False, "alternatives": alternatives[:3]}


async def tool_book_appointment(
    customer_name: str,
    customer_phone: str,
    date: str,
    time_slot: str,
    service_type: Optional[str] = None,
    customer_email: Optional[str] = None,
    address: Optional[str] = None,
) -> dict:
    """Create appointment booking in system."""
    if USE_REAL_TOOLS:
        tools = _get_real_tools()
        result = await tools["book_appointment"].execute({
            "customer_name": customer_name,
            "customer_phone": customer_phone,
            "date": date,
            "time": time_slot,
            "service_type": service_type or "Cleaning Estimate",
            "customer_email": customer_email,
            "address": address,
        })
        if result.success:
            cal_event_id = str(result.data.get("calendar_event_id", ""))
            return {
                "success": True,
                "booking_id": result.data.get("appointment_id") or cal_event_id,
                "calendar_event_id": cal_event_id,
                "customer_name": customer_name,
                "date": result.data.get("start_time", date),
                "time": time_slot,
                "service": service_type or "Cleaning Estimate",
                "confirmation_number": cal_event_id[:8].upper() if cal_event_id else "N/A",
            }
        return {"success": False, "error": result.error, "message": result.message}

    # Mock implementation for testing
    import uuid
    booking_id = f"apt_{uuid.uuid4().hex[:8]}"
    return {
        "success": True,
        "booking_id": booking_id,
        "customer_name": customer_name,
        "date": date,
        "time": time_slot,
        "service": service_type or "general",
        "confirmation_number": booking_id.upper(),
    }


# =============================================================================
# Graph Nodes
# =============================================================================


async def check_continuation(state: BookingWorkflowState) -> BookingWorkflowState:
    """
    Check if this is a continuation of a saved workflow.

    Restores partial state from session.metadata if available.
    """
    session_id = state.get("session_id")
    if not session_id:
        return {**state, "is_continuation": False}

    manager = get_workflow_state_manager()
    saved = await manager.restore_workflow_state(session_id)

    if saved and saved.workflow_type == BOOKING_WORKFLOW_TYPE:
        logger.info(
            "Restored booking workflow: step=%s",
            saved.current_step,
        )
        return {
            **state,
            "is_continuation": True,
            "restored_from_step": saved.current_step,
            "customer_name": saved.partial_state.get("customer_name"),
            "customer_phone": saved.partial_state.get("customer_phone"),
            "customer_id": saved.partial_state.get("customer_id"),
            "customer_email": saved.partial_state.get("customer_email"),
            "requested_date": saved.partial_state.get("requested_date"),
            "requested_time": saved.partial_state.get("requested_time"),
            "service_type": saved.partial_state.get("service_type"),
            "needs_info": saved.partial_state.get("needs_info", []),
            "alternative_slots": saved.partial_state.get("alternative_slots", []),
        }

    return {**state, "is_continuation": False}


async def merge_continuation_input(state: BookingWorkflowState) -> BookingWorkflowState:
    """
    Merge new user input with restored partial state.

    Parses the new input and updates the relevant fields.
    """
    import re
    from datetime import datetime, timedelta

    start_time = time.perf_counter()
    input_text = state.get("input_text", "")
    input_lower = input_text.lower()
    restored_step = state.get("restored_from_step", "")
    needs_info = list(state.get("needs_info", []))

    # Parse name from input if needed
    if "customer_identifier" in needs_info or not state.get("customer_name"):
        name_match = re.search(
            r"(?:my name is|i'm|i am|this is|name is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            input_text,
            re.IGNORECASE,
        )
        if name_match:
            state = {**state, "customer_name": name_match.group(1).title()}
            if "customer_identifier" in needs_info:
                needs_info.remove("customer_identifier")

    # Parse phone from input if needed
    phone_match = re.search(r"(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})", input_text)
    if phone_match:
        state = {**state, "customer_phone": phone_match.group(1)}
        if "customer_identifier" in needs_info:
            needs_info.remove("customer_identifier")
        if "phone" in needs_info:
            needs_info.remove("phone")

    # Parse date from input if needed
    if "date" in needs_info or not state.get("requested_date"):
        today = datetime.now().date()
        if "tomorrow" in input_lower:
            state = {**state, "requested_date": (today + timedelta(days=1)).isoformat()}
            if "date" in needs_info:
                needs_info.remove("date")
        elif "today" in input_lower:
            state = {**state, "requested_date": today.isoformat()}
            if "date" in needs_info:
                needs_info.remove("date")

    # Parse time from input if needed
    if "time" in needs_info or not state.get("requested_time"):
        time_match = re.search(r"(\d{1,2})\s*(am|pm)", input_lower)
        if time_match:
            state = {
                **state,
                "requested_time": f"{time_match.group(1)}:00 {time_match.group(2).upper()}",
            }
            if "time" in needs_info:
                needs_info.remove("time")

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    step_timings = state.get("step_timings", {})
    step_timings["merge"] = elapsed_ms

    return {
        **state,
        "needs_info": needs_info,
        "current_step": "merge",
        "step_timings": step_timings,
    }


async def parse_request(state: BookingWorkflowState) -> BookingWorkflowState:
    """
    Parse user input to extract booking details.

    This node uses regex patterns to extract structured data.
    In production, this would call the LLM with a structured output prompt.
    """
    import re
    from datetime import datetime, timedelta

    start_time = time.perf_counter()
    input_text = state.get("input_text", "")
    input_lower = input_text.lower()

    extracted = {
        "customer_name": None,
        "customer_phone": None,
        "requested_date": None,
        "requested_time": None,
        "service_type": None,
    }

    # Extract phone number (various formats: 555-1234, 555-123-4567, (555) 123-4567)
    phone_patterns = [
        r"(?:phone\s+)?(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})",  # 555-123-4567
        r"(?:phone\s+)?(\d{3}[-.\s]?\d{4})",              # 555-1234
        r"\((\d{3})\)\s*(\d{3})[-.\s]?(\d{4})",           # (555) 123-4567
    ]
    for pattern in phone_patterns:
        phone_match = re.search(pattern, input_text)
        if phone_match:
            extracted["customer_phone"] = "".join(phone_match.groups())
            break

    # Extract name after "for" keyword
    name_match = re.search(
        r"(?:for|customer)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        input_text
    )
    if name_match:
        extracted["customer_name"] = name_match.group(1)

    # Extract date using keywords
    today = datetime.now().date()
    if "tomorrow" in input_lower:
        extracted["requested_date"] = (today + timedelta(days=1)).isoformat()
    elif "today" in input_lower:
        extracted["requested_date"] = today.isoformat()
    elif "next week" in input_lower:
        extracted["requested_date"] = (today + timedelta(days=7)).isoformat()
    else:
        # Try to find day of week
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for i, day in enumerate(days):
            if day in input_lower:
                days_ahead = (i - today.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7  # Next week if same day
                extracted["requested_date"] = (today + timedelta(days=days_ahead)).isoformat()
                break

    # Extract time with various formats
    time_patterns = [
        (r"(\d{1,2})\s*(?::|\.)?(\d{2})?\s*(am|pm)", lambda m: f"{m.group(1)}:{m.group(2) or '00'} {m.group(3).upper()}"),
        (r"(\d{1,2})\s*(am|pm)", lambda m: f"{m.group(1)}:00 {m.group(2).upper()}"),
    ]
    for pattern, formatter in time_patterns:
        match = re.search(pattern, input_lower)
        if match:
            extracted["requested_time"] = formatter(match)
            break

    # Extract service type
    service_keywords = {
        "cleaning estimate": "Cleaning Estimate",
        "deep clean": "Deep Clean",
        "regular clean": "Regular Cleaning",
        "move out": "Move Out Cleaning",
        "oil change": "Oil Change",
        "inspection": "Inspection",
        "repair": "Repair",
    }
    for keyword, service in service_keywords.items():
        if keyword in input_lower:
            extracted["service_type"] = service
            break

    # Determine what info is still needed
    needs_info = []
    if not extracted["customer_name"] and not extracted["customer_phone"]:
        needs_info.append("customer_identifier")
    if not extracted["requested_date"]:
        needs_info.append("date")
    if not extracted["requested_time"]:
        needs_info.append("time")

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    step_timings = state.get("step_timings", {})
    step_timings["parse"] = elapsed_ms

    return {
        **state,
        "customer_name": extracted["customer_name"],
        "customer_phone": extracted["customer_phone"],
        "requested_date": extracted["requested_date"],
        "requested_time": extracted["requested_time"],
        "service_type": extracted["service_type"],
        "needs_info": needs_info,
        "current_step": "parse",
        "step_timings": step_timings,
    }


async def lookup_customer(state: BookingWorkflowState) -> BookingWorkflowState:
    """
    Look up customer in CRM system.

    Tool execution node - no LLM needed.
    """
    start_time = time.perf_counter()

    result = await tool_lookup_customer(
        name=state.get("customer_name"),
        phone=state.get("customer_phone"),
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    step_timings = state.get("step_timings", {})
    step_timings["lookup"] = elapsed_ms

    if result.get("found"):
        return {
            **state,
            "customer_found": True,
            "customer_id": result.get("customer_id"),
            "customer_email": result.get("email"),
            "customer_address": result.get("address"),
            "current_step": "lookup",
            "step_timings": step_timings,
        }
    else:
        return {
            **state,
            "customer_found": False,
            "current_step": "lookup",
            "step_timings": step_timings,
        }


async def check_availability(state: BookingWorkflowState) -> BookingWorkflowState:
    """
    Check calendar availability for requested slot.

    Tool execution node - no LLM needed.
    """
    start_time = time.perf_counter()

    requested_date = state.get("requested_date", "")
    requested_time = state.get("requested_time")
    service_type = state.get("service_type")

    result = await tool_check_availability(
        date=requested_date,
        time_slot=requested_time,
        service_type=service_type,
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    step_timings = state.get("step_timings", {})
    step_timings["availability"] = elapsed_ms

    if result.get("available"):
        # Keep original date/time - don't overwrite with display string
        return {
            **state,
            "slot_available": True,
            "current_step": "availability",
            "step_timings": step_timings,
        }
    else:
        return {
            **state,
            "slot_available": False,
            "alternative_slots": result.get("alternatives", []),
            "current_step": "availability",
            "step_timings": step_timings,
        }


async def suggest_alternatives(state: BookingWorkflowState) -> BookingWorkflowState:
    """
    Generate response suggesting alternative time slots.

    Saves workflow state for multi-turn continuation.
    """
    start_time = time.perf_counter()

    alternatives = state.get("alternative_slots", [])

    if alternatives:
        alt_list = ", ".join(alternatives[:3])
        response = (
            f"I'm sorry, that time slot isn't available. "
            f"Here are some alternatives: {alt_list}. "
            f"Would any of these work for you?"
        )
    else:
        response = (
            "I'm sorry, I couldn't find any available slots for that date. "
            "Would you like to try a different day?"
        )

    # Save workflow state for multi-turn continuation
    session_id = state.get("session_id")
    if session_id:
        manager = get_workflow_state_manager()
        await manager.save_workflow_state(
            session_id=session_id,
            workflow_type=BOOKING_WORKFLOW_TYPE,
            current_step="suggest_alternatives",
            partial_state={
                "customer_name": state.get("customer_name"),
                "customer_phone": state.get("customer_phone"),
                "customer_id": state.get("customer_id"),
                "customer_email": state.get("customer_email"),
                "requested_date": state.get("requested_date"),
                "requested_time": state.get("requested_time"),
                "service_type": state.get("service_type"),
                "alternative_slots": alternatives,
                "needs_info": ["date", "time"],
            },
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    step_timings = state.get("step_timings", {})
    step_timings["suggest"] = elapsed_ms

    total_ms = sum(step_timings.values())

    return {
        **state,
        "response": response,
        "awaiting_user_input": True,
        "current_step": "suggest_alternatives",
        "step_timings": step_timings,
        "total_ms": total_ms,
    }


async def book_appointment(state: BookingWorkflowState) -> BookingWorkflowState:
    """
    Create the appointment booking.

    Tool execution node - no LLM needed.
    """
    start_time = time.perf_counter()

    result = await tool_book_appointment(
        customer_name=state.get("customer_name", ""),
        customer_phone=state.get("customer_phone", ""),
        date=state.get("requested_date", ""),
        time_slot=state.get("requested_time", ""),
        service_type=state.get("service_type"),
        customer_email=state.get("customer_email"),
        address=state.get("customer_address"),
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    step_timings = state.get("step_timings", {})
    step_timings["book"] = elapsed_ms

    if result.get("success"):
        return {
            **state,
            "booking_confirmed": True,
            "booking_id": result.get("booking_id"),
            "confirmation_details": result,
            "current_step": "book",
            "step_timings": step_timings,
        }
    else:
        return {
            **state,
            "booking_confirmed": False,
            "error": result.get("error", "Booking failed"),
            "current_step": "book",
            "step_timings": step_timings,
        }


async def confirm_booking(state: BookingWorkflowState) -> BookingWorkflowState:
    """
    Generate booking confirmation response.

    Clears workflow state on successful completion.
    """
    start_time = time.perf_counter()

    details = state.get("confirmation_details", {})

    response = (
        f"Your appointment has been booked. "
        f"Confirmation number: {details.get('confirmation_number', 'N/A')}. "
        f"Date: {details.get('date', 'N/A')} at {details.get('time', 'N/A')}. "
        f"Service: {details.get('service', 'general')}. "
        f"We'll send a reminder to your email."
    )

    # Clear workflow state - booking complete
    session_id = state.get("session_id")
    if session_id:
        manager = get_workflow_state_manager()
        await manager.clear_workflow_state(session_id)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    step_timings = state.get("step_timings", {})
    step_timings["confirm"] = elapsed_ms

    total_ms = sum(step_timings.values())

    return {
        **state,
        "response": response,
        "current_step": "complete",
        "step_timings": step_timings,
        "total_ms": total_ms,
    }


async def handle_missing_info(state: BookingWorkflowState) -> BookingWorkflowState:
    """
    Generate response asking for missing information.

    Saves workflow state to session.metadata for multi-turn continuation.
    """
    start_time = time.perf_counter()

    needs_info = state.get("needs_info", [])

    if "customer_identifier" in needs_info:
        response = "I'd be happy to help you book an appointment. Could you please tell me your name or phone number?"
    elif "date" in needs_info:
        response = "What date would you like to book your appointment for?"
    elif "time" in needs_info:
        response = "What time works best for you?"
    else:
        response = "I need a bit more information to complete your booking. What else can you tell me?"

    # Save workflow state for multi-turn continuation
    session_id = state.get("session_id")
    if session_id:
        manager = get_workflow_state_manager()
        await manager.save_workflow_state(
            session_id=session_id,
            workflow_type=BOOKING_WORKFLOW_TYPE,
            current_step="awaiting_info",
            partial_state={
                "customer_name": state.get("customer_name"),
                "customer_phone": state.get("customer_phone"),
                "requested_date": state.get("requested_date"),
                "requested_time": state.get("requested_time"),
                "service_type": state.get("service_type"),
                "needs_info": needs_info,
            },
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    step_timings = state.get("step_timings", {})
    step_timings["missing_info"] = elapsed_ms

    total_ms = sum(step_timings.values())

    return {
        **state,
        "response": response,
        "awaiting_user_input": True,
        "current_step": "awaiting_info",
        "step_timings": step_timings,
        "total_ms": total_ms,
    }


async def handle_customer_not_found(state: BookingWorkflowState) -> BookingWorkflowState:
    """
    Handle case where customer is not in CRM.

    Saves workflow state for multi-turn continuation.
    """
    start_time = time.perf_counter()

    customer_name = state.get("customer_name", "")

    response = (
        f"I don't see {customer_name if customer_name else 'you'} in our system. "
        f"That's okay - I can create a new customer profile. "
        f"Could you please confirm your phone number and email address?"
    )

    # Save workflow state for multi-turn continuation
    session_id = state.get("session_id")
    if session_id:
        manager = get_workflow_state_manager()
        await manager.save_workflow_state(
            session_id=session_id,
            workflow_type=BOOKING_WORKFLOW_TYPE,
            current_step="create_customer",
            partial_state={
                "customer_name": state.get("customer_name"),
                "customer_phone": state.get("customer_phone"),
                "requested_date": state.get("requested_date"),
                "requested_time": state.get("requested_time"),
                "service_type": state.get("service_type"),
                "needs_info": ["phone", "email"],
            },
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    step_timings = state.get("step_timings", {})
    step_timings["not_found"] = elapsed_ms

    total_ms = sum(step_timings.values())

    return {
        **state,
        "response": response,
        "awaiting_user_input": True,
        "current_step": "create_customer",
        "step_timings": step_timings,
        "total_ms": total_ms,
    }


# =============================================================================
# Routing Functions
# =============================================================================


def route_after_check_continuation(
    state: BookingWorkflowState,
) -> Literal["merge_continuation", "parse_request"]:
    """
    Route after checking for continuation.

    If continuing, merge new input with saved state.
    Otherwise, start fresh with parse_request.
    """
    if state.get("is_continuation"):
        return "merge_continuation"
    return "parse_request"


def route_after_merge(
    state: BookingWorkflowState,
) -> Literal["lookup_customer", "check_availability", "handle_missing_info"]:
    """
    Route after merging continuation input.

    Determines next step based on what info we have and where we left off.
    """
    needs_info = state.get("needs_info", [])
    restored_step = state.get("restored_from_step", "")

    # Still missing customer identifier
    if "customer_identifier" in needs_info:
        return "handle_missing_info"

    # If we came from suggest_alternatives and now have date/time, check availability
    if restored_step == "suggest_alternatives":
        if not needs_info or ("date" not in needs_info and "time" not in needs_info):
            return "check_availability"
        return "handle_missing_info"

    # If we came from create_customer and now have phone, proceed to availability
    if restored_step == "create_customer":
        if state.get("customer_name") and state.get("customer_phone"):
            return "check_availability"
        return "handle_missing_info"

    # Default: lookup customer if we have identifier
    if state.get("customer_name") or state.get("customer_phone"):
        return "lookup_customer"

    return "handle_missing_info"


def route_after_parse(
    state: BookingWorkflowState,
) -> Literal["lookup_customer", "handle_missing_info"]:
    """
    Route after parsing: if we have customer info, look them up.
    Otherwise, ask for missing information.
    """
    needs_info = state.get("needs_info", [])

    # If we're missing customer identifier, ask for it first
    if "customer_identifier" in needs_info:
        return "handle_missing_info"

    # Otherwise, proceed to lookup
    return "lookup_customer"


def route_after_lookup(
    state: BookingWorkflowState,
) -> Literal["check_availability", "handle_customer_not_found"]:
    """
    Route after customer lookup: if found OR we have enough info for new customer,
    proceed to availability check. Otherwise, ask for more info.
    """
    # Existing customer found - proceed
    if state.get("customer_found"):
        return "check_availability"

    # New customer but we have name AND phone - proceed with booking
    if state.get("customer_name") and state.get("customer_phone"):
        return "check_availability"

    # Need more info for new customer
    return "handle_customer_not_found"


def route_after_availability(
    state: BookingWorkflowState,
) -> Literal["book_appointment", "suggest_alternatives"]:
    """
    Route after availability check: if slot available, book it.
    Otherwise, suggest alternatives.
    """
    if state.get("slot_available"):
        return "book_appointment"
    return "suggest_alternatives"


# =============================================================================
# Graph Builder
# =============================================================================


def build_booking_graph() -> StateGraph:
    """
    Build the booking workflow graph with multi-turn support.

    Flow:
        check_continuation
            |
            ├── [continuation] → merge_continuation → route_after_merge
            |
            └── [new] → parse_request
                            |
                            ├── [missing info] → handle_missing_info → END (saves state)
                            |
                            └── [has customer] → lookup_customer
                                                    |
                                                    ├── [not found] → handle_customer_not_found → END
                                                    |
                                                    └── [found] → check_availability
                                                                    |
                                                                    ├── [unavailable] → suggest_alternatives → END
                                                                    |
                                                                    └── [available] → book_appointment → confirm_booking → END
    """
    graph = StateGraph(BookingWorkflowState)

    # Add nodes - including new continuation nodes
    graph.add_node("check_continuation", check_continuation)
    graph.add_node("merge_continuation", merge_continuation_input)
    graph.add_node("parse_request", parse_request)
    graph.add_node("lookup_customer", lookup_customer)
    graph.add_node("check_availability", check_availability)
    graph.add_node("book_appointment", book_appointment)
    graph.add_node("confirm_booking", confirm_booking)
    graph.add_node("suggest_alternatives", suggest_alternatives)
    graph.add_node("handle_missing_info", handle_missing_info)
    graph.add_node("handle_customer_not_found", handle_customer_not_found)

    # Set entry point to check_continuation
    graph.set_entry_point("check_continuation")

    # Route after checking for continuation
    graph.add_conditional_edges(
        "check_continuation",
        route_after_check_continuation,
        {
            "merge_continuation": "merge_continuation",
            "parse_request": "parse_request",
        },
    )

    # Route after merging continuation input
    graph.add_conditional_edges(
        "merge_continuation",
        route_after_merge,
        {
            "lookup_customer": "lookup_customer",
            "check_availability": "check_availability",
            "handle_missing_info": "handle_missing_info",
        },
    )

    # Route after parsing new request
    graph.add_conditional_edges(
        "parse_request",
        route_after_parse,
        {
            "lookup_customer": "lookup_customer",
            "handle_missing_info": "handle_missing_info",
        },
    )

    graph.add_conditional_edges(
        "lookup_customer",
        route_after_lookup,
        {
            "check_availability": "check_availability",
            "handle_customer_not_found": "handle_customer_not_found",
        },
    )

    graph.add_conditional_edges(
        "check_availability",
        route_after_availability,
        {
            "book_appointment": "book_appointment",
            "suggest_alternatives": "suggest_alternatives",
        },
    )

    # Add terminal edges
    graph.add_edge("book_appointment", "confirm_booking")
    graph.add_edge("confirm_booking", END)
    graph.add_edge("suggest_alternatives", END)
    graph.add_edge("handle_missing_info", END)
    graph.add_edge("handle_customer_not_found", END)

    return graph


def compile_booking_graph():
    """Compile the booking graph for execution."""
    graph = build_booking_graph()
    return graph.compile()


# =============================================================================
# Convenience Functions
# =============================================================================


async def run_booking_workflow(
    input_text: str,
    session_id: Optional[str] = None,
) -> BookingWorkflowState:
    """
    Run the booking workflow with user input.

    Args:
        input_text: User's booking request
        session_id: Optional session identifier

    Returns:
        Final workflow state with response
    """
    compiled_graph = compile_booking_graph()

    initial_state: BookingWorkflowState = {
        "input_text": input_text,
        "session_id": session_id,
        "step_timings": {},
    }

    # Run the graph
    final_state = await compiled_graph.ainvoke(initial_state)

    return final_state


