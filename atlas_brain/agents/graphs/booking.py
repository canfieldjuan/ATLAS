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
import re
import time
from typing import Any, Literal, Optional

from langgraph.graph import END, StateGraph

from .state import BookingWorkflowState
from .workflow_state import get_workflow_state_manager
from ...services import llm_registry
from ...services.protocols import Message

logger = logging.getLogger("atlas.agents.graphs.booking")

# Workflow type identifier for state persistence
BOOKING_WORKFLOW_TYPE = "booking"

# Toggle between mock and real tools (configured via ATLAS_WORKFLOW_USE_REAL_TOOLS)
from ...config import settings as _settings
USE_REAL_TOOLS = _settings.workflows.use_real_tools
logger.info("Booking workflow tools: %s", "real" if USE_REAL_TOOLS else "mock")

# Sequential field collection order
BOOKING_FIELDS_ORDER = ["name", "address", "date", "time"]

# Map field names to BookingWorkflowState keys
FIELD_STATE_MAP = {
    "name": "customer_name",
    "address": "customer_address",
    "date": "requested_date",
    "time": "requested_time",
}

# Field-specific extraction prompts
def _get_field_prompts() -> dict[str, str]:
    """Build field prompts with today's date for relative date resolution."""
    from datetime import date
    today = date.today()
    today_str = today.strftime("%A, %Y-%m-%d")  # e.g. "Monday, 2026-02-09"
    return {
        "name": "Extract the customer name from the user input. Reply with ONLY the name, nothing else. If no name found, reply NONE.",
        "address": "Extract the address from the user input. Reply with ONLY the address, nothing else. If no address found, reply NONE.",
        "date": f"Extract the date from the user input. Today is {today_str}. Convert relative dates like 'Thursday', 'tomorrow', 'next week' to YYYY-MM-DD format. Reply with ONLY the date in YYYY-MM-DD format, nothing else. If no date found, reply NONE.",
        "time": "Extract the time from the user input. Convert to HH:MM AM/PM format. Reply with ONLY the time, nothing else. If no time found, reply NONE.",
    }


async def extract_field_with_llm(
    field_name: str,
    user_input: str,
) -> Optional[str]:
    """
    Extract a specific field from user input using LLM.

    Args:
        field_name: The field to extract (name, address, date, time)
        user_input: The user's response text

    Returns:
        Extracted value or None if not found
    """
    llm = llm_registry.get_active()
    if llm is None:
        logger.warning("LLM not available for field extraction")
        return None

    prompt = _get_field_prompts().get(field_name)
    if not prompt:
        logger.warning("No prompt defined for field: %s", field_name)
        return None

    messages = [
        Message(role="system", content=prompt),
        Message(role="user", content=user_input),
    ]

    try:
        result = llm.chat(messages=messages, max_tokens=50, temperature=0.1)
        response = result.get("response", "").strip()

        # Strip <think>...</think> tags (Qwen3 models)
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        if response.upper() == "NONE" or not response:
            logger.debug("LLM extraction for %s returned NONE", field_name)
            return None

        logger.info("LLM extracted %s: %s", field_name, response)
        return response

    except Exception as e:
        logger.error("LLM extraction failed for %s: %s", field_name, e)
        return None


async def extract_all_fields_with_llm(
    user_input: str,
    missing_fields: list[str],
) -> dict[str, str]:
    """
    Extract ALL missing fields from user input in a single LLM call.

    Args:
        user_input: The user's response text
        missing_fields: List of field names to extract (name, address, date, time)

    Returns:
        Dict mapping field names to extracted values
    """
    llm = llm_registry.get_active()
    if llm is None or not missing_fields:
        return {}

    from datetime import date
    today = date.today()
    today_str = today.strftime("%A, %Y-%m-%d")
    field_descriptions = {
        "name": "customer name",
        "address": "street address",
        "date": f"date (today is {today_str}; convert relative dates like 'Thursday', 'tomorrow' to YYYY-MM-DD format)",
        "time": "time (convert to HH:MM AM/PM format)",
    }
    fields_list = "\n".join(
        f"- {f}: {field_descriptions.get(f, f)}" for f in missing_fields
    )

    prompt = (
        f"Extract the following fields from the user input. "
        f"For each field found, reply with one line in the format 'field: value'. "
        f"Only include fields that are clearly present. Do NOT guess.\n\n"
        f"Fields to extract:\n{fields_list}\n\n"
        f"Reply with ONLY the extracted lines, nothing else. "
        f"If no fields are found, reply NONE."
    )

    messages = [
        Message(role="system", content=prompt),
        Message(role="user", content=user_input),
    ]

    try:
        result = llm.chat(messages=messages, max_tokens=100, temperature=0.1)
        raw = result.get("response", "").strip()

        # Strip <think>...</think> tags (Qwen3 models)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        if raw.upper() == "NONE" or not raw:
            return {}

        extracted = {}
        for line in raw.splitlines():
            line = line.strip()
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip().lower()
                value = value.strip()
                if key in missing_fields and value and value.upper() != "NONE":
                    extracted[key] = value

        logger.info("Greedy extraction: %s from %d missing fields", extracted, len(missing_fields))
        return extracted

    except Exception as e:
        logger.error("Greedy LLM extraction failed: %s", e)
        return {}


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
            "Restored booking workflow: step=%s collecting=%s",
            saved.current_step,
            saved.partial_state.get("collecting_field"),
        )
        return {
            **state,
            "is_continuation": True,
            "restored_from_step": saved.current_step,
            "customer_name": saved.partial_state.get("customer_name"),
            "customer_phone": saved.partial_state.get("customer_phone"),
            "customer_id": saved.partial_state.get("customer_id"),
            "customer_email": saved.partial_state.get("customer_email"),
            "customer_address": saved.partial_state.get("customer_address"),
            "requested_date": saved.partial_state.get("requested_date"),
            "requested_time": saved.partial_state.get("requested_time"),
            "service_type": saved.partial_state.get("service_type"),
            "needs_info": saved.partial_state.get("needs_info", []),
            "alternative_slots": saved.partial_state.get("alternative_slots", []),
            "collecting_field": saved.partial_state.get("collecting_field"),
        }

    return {**state, "is_continuation": False}


async def merge_continuation_input(state: BookingWorkflowState) -> BookingWorkflowState:
    """
    Merge new user input with restored partial state using greedy LLM extraction.

    Extracts ALL missing fields from a single user utterance, not just the
    one we were collecting. E.g. "My name is John and I live at 123 Main St"
    fills both name AND address in one turn.
    """
    start_time = time.perf_counter()
    input_text = state.get("input_text", "")
    collecting_field = state.get("collecting_field")
    restored_step = state.get("restored_from_step", "")

    # Compute all missing fields
    missing_fields = [
        f for f in BOOKING_FIELDS_ORDER
        if not state.get(FIELD_STATE_MAP[f])
    ]

    if missing_fields:
        # Greedy extraction: try to get ALL missing fields at once
        extracted = await extract_all_fields_with_llm(input_text, missing_fields)

        if extracted:
            for field_name, value in extracted.items():
                state_field = FIELD_STATE_MAP[field_name]
                state = {**state, state_field: value}
                logger.info("Greedy extracted %s=%s", state_field, value)

        # If collecting_field was set but greedy found nothing for it,
        # fall back to using raw input for that field
        if collecting_field and collecting_field in FIELD_STATE_MAP:
            state_field = FIELD_STATE_MAP[collecting_field]
            if not state.get(state_field):
                state = {**state, state_field: input_text.strip()}
                logger.info("Fallback raw input for %s=%s", state_field, input_text.strip())

    # Handle suggest_alternatives: user is providing new date/time to replace rejected ones
    if restored_step == "suggest_alternatives":
        # Clear old rejected values BEFORE extraction so failed extraction
        # doesn't silently re-use the already-rejected date/time
        state = {**state, "requested_date": None, "requested_time": None}
        extracted = await extract_all_fields_with_llm(input_text, ["date", "time"])
        if extracted:
            for field_name, value in extracted.items():
                state_field = FIELD_STATE_MAP[field_name]
                state = {**state, state_field: value}
                logger.info("Suggest-alt extracted %s=%s", state_field, value)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    step_timings = state.get("step_timings", {})
    step_timings["merge"] = elapsed_ms

    return {
        **state,
        "current_step": "merge",
        "collecting_field": None,  # Clear after processing
        "step_timings": step_timings,
    }


async def parse_request(state: BookingWorkflowState) -> BookingWorkflowState:
    """
    Parse user input to extract booking details.

    This node uses regex patterns to extract structured data.
    In production, this would call the LLM with a structured output prompt.
    """
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
                "customer_address": state.get("customer_address"),
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


async def generate_natural_prompt(
    state: BookingWorkflowState,
    still_missing: list[str],
) -> str:
    """
    Generate a natural, conversational prompt for remaining fields.

    Args:
        state: Current workflow state (has collected fields)
        still_missing: List of field names still needed

    Returns:
        Natural language prompt string
    """
    llm = llm_registry.get_active()
    if llm is None:
        # Fallback to simple template
        return _simple_missing_prompt(state, still_missing)

    # Build context of what's collected
    collected = {}
    field_labels = {
        "name": "name",
        "address": "address",
        "date": "preferred date",
        "time": "preferred time",
    }

    for f in BOOKING_FIELDS_ORDER:
        val = state.get(FIELD_STATE_MAP[f])
        if val:
            collected[field_labels[f]] = val

    missing_labels = [field_labels[f] for f in still_missing]

    collected_str = ", ".join(f"{k}: {v}" for k, v in collected.items()) if collected else "nothing yet"
    missing_str = " and ".join(missing_labels)

    prompt = (
        "You are a friendly scheduling assistant. "
        "The user is booking an appointment. "
        f"So far you have collected: {collected_str}. "
        f"You still need: {missing_str}. "
        "Write a brief, warm, conversational response (1-2 sentences) "
        "acknowledging what you have and asking for the remaining info. "
        "Be natural, not robotic. Do not use bullet points or lists."
    )

    messages = [
        Message(role="system", content=prompt),
        Message(role="user", content="Generate the prompt."),
    ]

    try:
        result = llm.chat(messages=messages, max_tokens=80, temperature=0.7)
        raw = result.get("response", "").strip()
        # Strip <think>...</think> tags (Qwen3 models)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if raw:
            return raw
    except Exception as e:
        logger.warning("Natural prompt generation failed: %s", e)

    return _simple_missing_prompt(state, still_missing)


def _simple_missing_prompt(state: BookingWorkflowState, still_missing: list[str]) -> str:
    """Fallback simple template when LLM is unavailable."""
    if "name" in still_missing:
        return "Sure, I can help you book an appointment. Can I get your name?"
    if "address" in still_missing:
        name = state.get("customer_name", "")
        return f"Got it, {name}. And what's the address for the appointment?"
    if "date" in still_missing:
        return "Perfect. What day works for you?"
    if "time" in still_missing:
        return "And what time would you prefer?"
    return "Great, I have everything. Let me check availability."


async def handle_missing_info(state: BookingWorkflowState) -> BookingWorkflowState:
    """
    Generate response asking for missing information sequentially.

    Asks for fields in order: name -> address -> date -> time
    Uses LLM for natural conversational prompts instead of hardcoded templates.
    Saves workflow state with collecting_field for LLM extraction on next turn.
    """
    start_time = time.perf_counter()

    # Compute fields still missing (in order)
    still_missing = [
        f for f in BOOKING_FIELDS_ORDER
        if not state.get(FIELD_STATE_MAP[f])
    ]

    # First missing field is the one we're collecting
    collecting_field = still_missing[0] if still_missing else None

    if still_missing:
        response = await generate_natural_prompt(state, still_missing)
    else:
        response = "Great, I have everything. Let me check availability."

    # Save workflow state for multi-turn continuation
    session_id = state.get("session_id")
    if session_id and collecting_field:
        manager = get_workflow_state_manager()
        await manager.save_workflow_state(
            session_id=session_id,
            workflow_type=BOOKING_WORKFLOW_TYPE,
            current_step="collecting_field",
            partial_state={
                "customer_name": state.get("customer_name"),
                "customer_phone": state.get("customer_phone"),
                "customer_address": state.get("customer_address"),
                "requested_date": state.get("requested_date"),
                "requested_time": state.get("requested_time"),
                "service_type": state.get("service_type"),
                "collecting_field": collecting_field,
            },
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    step_timings = state.get("step_timings", {})
    step_timings["missing_info"] = elapsed_ms

    return {
        **state,
        "response": response,
        "awaiting_user_input": collecting_field is not None,
        "current_step": "collecting_field" if collecting_field else "ready",
        "collecting_field": collecting_field,
        "step_timings": step_timings,
        "total_ms": sum(step_timings.values()),
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
                "customer_address": state.get("customer_address"),
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

    Sequential field collection: name -> address -> date -> time -> availability
    """
    restored_step = state.get("restored_from_step", "")

    # Handle legacy suggest_alternatives step
    if restored_step == "suggest_alternatives":
        if state.get("requested_date") and state.get("requested_time"):
            return "check_availability"
        return "handle_missing_info"

    # Check if all required fields are collected
    has_name = bool(state.get("customer_name"))
    has_address = bool(state.get("customer_address"))
    has_date = bool(state.get("requested_date"))
    has_time = bool(state.get("requested_time"))

    # If we have all fields, proceed to availability check
    if has_name and has_address and has_date and has_time:
        return "check_availability"

    # Otherwise, continue collecting fields
    return "handle_missing_info"


def route_after_parse(
    state: BookingWorkflowState,
) -> Literal["lookup_customer", "handle_missing_info"]:
    """
    Route after parsing: start sequential field collection.

    Always goes to handle_missing_info to begin collecting fields in order.
    """
    # Check if we already have all required fields (rare on first parse)
    has_name = bool(state.get("customer_name"))
    has_address = bool(state.get("customer_address"))
    has_date = bool(state.get("requested_date"))
    has_time = bool(state.get("requested_time"))

    if has_name and has_address and has_date and has_time:
        return "lookup_customer"

    # Start sequential collection
    return "handle_missing_info"


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


