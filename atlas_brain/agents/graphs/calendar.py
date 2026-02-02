"""
LangGraph Calendar Workflow.

Handles calendar event creation with multi-turn slot filling.
- create_event: Create a new calendar event
- query_events: Query upcoming calendar events

The LLM classifies intent once, then the graph routes to the appropriate action.
"""

import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional

import dateparser
from langgraph.graph import END, StateGraph

from .state import CalendarWorkflowState
from .workflow_state import get_workflow_state_manager

logger = logging.getLogger("atlas.agents.graphs.calendar")

# Workflow type constant for multi-turn support
CALENDAR_WORKFLOW_TYPE = "calendar"

# Environment flag for using real tools vs mocks
USE_REAL_TOOLS = os.environ.get("USE_REAL_TOOLS", "false").lower() == "true"


# =============================================================================
# Tool Wrappers (Real vs Mock)
# =============================================================================


async def tool_create_event(
    summary: str,
    start: datetime,
    end: datetime,
    location: Optional[str] = None,
    description: Optional[str] = None,
    calendar_id: Optional[str] = None,
) -> dict[str, Any]:
    """Create a calendar event."""
    if USE_REAL_TOOLS:
        from atlas_brain.tools.calendar import calendar_tool

        try:
            result = await calendar_tool.create_event(
                summary=summary,
                start=start,
                end=end,
                location=location,
                description=description,
                calendar_id=calendar_id,
            )

            return {
                "success": result.success,
                "event_id": result.data.get("event_id") if result.data else None,
                "error": result.error,
                "message": result.message,
            }

        except Exception as e:
            logger.exception("Error creating calendar event")
            return {"success": False, "error": str(e)}
    else:
        # Mock response
        return {
            "success": True,
            "event_id": f"mock-event-{int(time.time())}",
            "message": f"[MOCK] Created event: {summary}",
        }


async def tool_query_events(
    hours_ahead: int = 24,
    max_results: int = 10,
    calendar_name: Optional[str] = None,
) -> dict[str, Any]:
    """Query upcoming calendar events."""
    if USE_REAL_TOOLS:
        from atlas_brain.tools.calendar import calendar_tool

        try:
            params = {
                "hours_ahead": hours_ahead,
                "max_results": max_results,
            }
            if calendar_name:
                params["calendar_name"] = calendar_name

            result = await calendar_tool.execute(params)

            return {
                "success": result.success,
                "events": result.data.get("events", []) if result.data else [],
                "count": result.data.get("count", 0) if result.data else 0,
                "error": result.error,
                "message": result.message,
            }

        except Exception as e:
            logger.exception("Error querying calendar events")
            return {"success": False, "error": str(e), "events": [], "count": 0}
    else:
        # Mock response
        now = datetime.now(timezone.utc)
        return {
            "success": True,
            "events": [
                {
                    "id": "mock-1",
                    "summary": "Team Meeting",
                    "start": (now + timedelta(hours=2)).isoformat(),
                    "end": (now + timedelta(hours=3)).isoformat(),
                    "location": "Conference Room A",
                },
            ],
            "count": 1,
            "message": "You have 1 event coming up",
        }


# =============================================================================
# Helper Functions
# =============================================================================


def parse_datetime(when_text: str, tz_name: str = "America/Chicago") -> Optional[datetime]:
    """Parse natural language time expression."""
    settings_dict = {
        "PREFER_DATES_FROM": "future",
        "PREFER_DAY_OF_MONTH": "first",
        "RETURN_AS_TIMEZONE_AWARE": True,
        "TIMEZONE": tz_name,
    }

    text = when_text.strip().lower()

    # Normalize "next Tuesday" patterns
    weekdays = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    pattern = r"^next\s+(" + "|".join(weekdays) + r")\b(.*)$"
    match = re.match(pattern, text, re.IGNORECASE)
    if match:
        weekday = match.group(1)
        rest = match.group(2).strip()
        text = f"{weekday} {rest}" if rest else weekday

    result = dateparser.parse(text, settings=settings_dict)
    return result


def parse_duration(duration_text: str) -> Optional[timedelta]:
    """Parse duration string into timedelta."""
    text = duration_text.strip().lower()

    # Pattern: "X hours", "X hour", "X minutes", "X minute"
    hour_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)", text)
    min_match = re.search(r"(\d+)\s*(?:minutes?|mins?|m)", text)

    hours = float(hour_match.group(1)) if hour_match else 0
    minutes = int(min_match.group(1)) if min_match else 0

    if hours or minutes:
        return timedelta(hours=hours, minutes=minutes)

    # Default: 1 hour if no duration parsed
    return timedelta(hours=1)


def classify_calendar_intent(text: str) -> str:
    """
    Simple rule-based intent classification for calendar.

    Returns: "create_event", "query_events", or "unknown"
    """
    text_lower = text.lower().strip()

    # Create patterns
    create_patterns = [
        "create", "add", "schedule", "book", "set up", "new event",
        "put on", "calendar event", "add to calendar",
    ]
    for pattern in create_patterns:
        if pattern in text_lower:
            return "create_event"

    # Query patterns
    query_patterns = [
        "what", "show", "list", "upcoming", "my calendar", "my schedule",
        "do i have", "any events", "what's on", "check calendar",
    ]
    for pattern in query_patterns:
        if pattern in text_lower:
            return "query_events"

    # Default to create if contains time-like expressions with event-like words
    if any(word in text_lower for word in ["meeting", "appointment", "event"]):
        return "create_event"

    return "unknown"


def parse_create_intent(text: str) -> dict[str, Any]:
    """
    Parse a create event request to extract title, date, time, duration.

    Returns dict with 'title', 'date_time', 'duration', 'location' keys.
    """
    text_lower = text.lower().strip()

    result = {
        "title": None,
        "date_time": None,
        "duration": None,
        "location": None,
    }

    # Pattern: "create/add/schedule [event title] (at|on|for) [time]"
    # Try to extract title first - order matters, more specific patterns first
    title_patterns = [
        # Pattern with "called [title]" - most specific
        r"(?:meeting|event|appointment)\s+(?:called|for|about)\s+['\"]?(.+?)['\"]?\s+(?:at|on|tomorrow)",
        r"(?:meeting|event|appointment)\s+(?:called|for|about)\s+['\"]?(.+?)['\"]?$",
        # Pattern with time at end
        r"(?:create|add|schedule|book|set up)\s+(?:a\s+)?(?:calendar\s+)?(?:event\s+)?(?:for\s+|called\s+)?['\"]?(.+?)['\"]?\s+(?:at|on|for|tomorrow|next|in\s+\d)",
        # Generic pattern - least specific
        r"(?:create|add|schedule|book)\s+(?:a\s+)?(.+?)\s+(?:meeting|event|appointment)",
    ]

    for pattern in title_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result["title"] = match.group(1).strip()
            break

    # If no title found, try simpler extraction
    if not result["title"]:
        # Remove common prefixes
        cleaned = re.sub(
            r"^(create|add|schedule|book|set up)\s+(a\s+)?(calendar\s+)?(event\s+)?(for\s+)?",
            "",
            text_lower,
        ).strip()
        # Take text before time indicators
        time_indicators = ["at ", "on ", "tomorrow", "next ", "in "]
        for indicator in time_indicators:
            idx = cleaned.find(indicator)
            if idx > 0:
                result["title"] = cleaned[:idx].strip()
                break

    # Filter out generic words that aren't real titles
    generic_words = {
        "event", "calendar", "meeting", "appointment", "an", "the", "a", "new",
    }
    if result["title"] and result["title"].lower() in generic_words:
        result["title"] = None

    # Extract time/date
    time_patterns = [
        r"(tomorrow\s+at\s+\d+(?::\d+)?\s*(?:am|pm)?)",
        r"(at\s+\d+(?::\d+)?\s*(?:am|pm)?(?:\s+(?:today|tomorrow))?)",
        r"(next\s+\w+\s+at\s+\d+(?::\d+)?\s*(?:am|pm)?)",
        r"(on\s+\w+\s+at\s+\d+(?::\d+)?\s*(?:am|pm)?)",
        r"(in\s+\d+\s+(?:hours?|minutes?))",
        r"(today\s+at\s+\d+(?::\d+)?\s*(?:am|pm)?)",
    ]

    for pattern in time_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result["date_time"] = match.group(1).strip()
            break

    # Extract duration
    duration_patterns = [
        r"for\s+(\d+(?:\.\d+)?\s*(?:hours?|hrs?|minutes?|mins?))",
        r"(\d+(?:\.\d+)?\s*(?:hours?|hrs?|minutes?|mins?))\s+long",
    ]

    for pattern in duration_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result["duration"] = match.group(1).strip()
            break

    # Extract location
    location_patterns = [
        r"(?:at|in)\s+(?:the\s+)?([A-Z][a-zA-Z\s]+(?:Room|Office|Building|Center|Hall))",
        r"location[:\s]+([^,\.]+)",
    ]

    for pattern in location_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["location"] = match.group(1).strip()
            break

    return result


def format_time_for_speech(dt: datetime) -> str:
    """Format datetime for human-readable response."""
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    hour = dt.hour % 12 or 12
    minute = dt.strftime("%M")
    ampm = "AM" if dt.hour < 12 else "PM"
    time_str = f"{hour}:{minute} {ampm}"

    if dt.date() == now.date():
        return f"today at {time_str}"

    tomorrow = (now + timedelta(days=1)).date()
    if dt.date() == tomorrow:
        return f"tomorrow at {time_str}"

    diff = dt.date() - now.date()
    if diff.days < 7:
        return f"on {dt.strftime('%A')} at {time_str}"

    return f"on {dt.strftime('%B')} {dt.day} at {time_str}"


# =============================================================================
# Graph Nodes
# =============================================================================


async def check_continuation(state: CalendarWorkflowState) -> CalendarWorkflowState:
    """Check if this is a continuation of a saved workflow."""
    session_id = state.get("session_id")
    if not session_id:
        return state

    manager = get_workflow_state_manager()
    saved = await manager.restore_workflow_state(session_id)

    if saved and saved.workflow_type == CALENDAR_WORKFLOW_TYPE:
        if saved.is_expired():
            logger.info("Calendar workflow expired for session %s", session_id)
            await manager.clear_workflow_state(session_id)
            return state

        logger.info(
            "Continuing calendar workflow from step %s for session %s",
            saved.current_step,
            session_id,
        )
        return {
            **state,
            "is_continuation": True,
            "restored_from_step": saved.current_step,
            "intent": saved.partial_state.get("intent", "create_event"),
            "event_title": saved.partial_state.get("event_title"),
            "event_date": saved.partial_state.get("event_date"),
            "event_time": saved.partial_state.get("event_time"),
            "event_duration": saved.partial_state.get("event_duration"),
            "event_location": saved.partial_state.get("event_location"),
            "parsed_start_at": saved.partial_state.get("parsed_start_at"),
            "parsed_end_at": saved.partial_state.get("parsed_end_at"),
        }

    return state


async def merge_continuation_input(state: CalendarWorkflowState) -> CalendarWorkflowState:
    """Merge new user input with saved partial state."""
    input_text = state.get("input_text", "")

    # Parse the new input to extract any provided values
    parsed = parse_create_intent(input_text)
    new_title = parsed.get("title")
    new_date_time = parsed.get("date_time")
    new_duration = parsed.get("duration")
    new_location = parsed.get("location")

    # Get existing values from state
    event_title = state.get("event_title")
    parsed_start_at = state.get("parsed_start_at")
    parsed_end_at = state.get("parsed_end_at")
    event_location = state.get("event_location")

    # Merge: new values override if provided
    if new_title:
        event_title = new_title

    if new_date_time:
        dt = parse_datetime(new_date_time)
        if dt:
            parsed_start_at = dt.isoformat()
            # Calculate end time using duration
            duration = parse_duration(new_duration) if new_duration else timedelta(hours=1)
            end_dt = dt + duration
            parsed_end_at = end_dt.isoformat()

    # If input looks like a plain time expression (no event keywords)
    if not new_title and not new_date_time:
        text_lower = input_text.strip().lower()
        # Check if it looks like a time
        time_patterns = ["at ", "tomorrow", "next ", "in ", "pm", "am", ":"]
        if any(p in text_lower for p in time_patterns):
            dt = parse_datetime(input_text)
            if dt:
                parsed_start_at = dt.isoformat()
                duration = parse_duration(state.get("event_duration") or "1 hour")
                end_dt = dt + duration
                parsed_end_at = end_dt.isoformat()
        # Otherwise treat as title
        elif not event_title:
            event_title = input_text.strip()

    if new_location:
        event_location = new_location

    logger.info(
        "[CALENDAR] Merged continuation: title=%s, start=%s, end=%s",
        event_title,
        parsed_start_at,
        parsed_end_at,
    )

    return {
        **state,
        "event_title": event_title,
        "parsed_start_at": parsed_start_at,
        "parsed_end_at": parsed_end_at,
        "event_location": event_location,
    }


async def classify_intent(state: CalendarWorkflowState) -> CalendarWorkflowState:
    """Classify the calendar intent from user input."""
    start = time.time()

    input_text = state.get("input_text", "")
    intent = classify_calendar_intent(input_text)

    elapsed = (time.time() - start) * 1000
    step_timings = state.get("step_timings", {})
    step_timings["classify"] = elapsed

    logger.info("[CALENDAR] Classified intent: %s", intent)

    return {
        **state,
        "intent": intent,
        "current_step": "classify",
        "step_timings": step_timings,
    }


async def parse_create_request(state: CalendarWorkflowState) -> CalendarWorkflowState:
    """Parse a create event request."""
    start = time.time()

    # If this is a continuation, use merged values
    if state.get("is_continuation"):
        title = state.get("event_title")
        parsed_start_at = state.get("parsed_start_at")
        parsed_end_at = state.get("parsed_end_at")
        location = state.get("event_location")
    else:
        input_text = state.get("input_text", "")
        parsed = parse_create_intent(input_text)

        title = parsed.get("title")
        date_time = parsed.get("date_time")
        duration_str = parsed.get("duration")
        location = parsed.get("location")

        parsed_start_at = None
        parsed_end_at = None

        if date_time:
            dt = parse_datetime(date_time)
            if dt:
                parsed_start_at = dt.isoformat()
                duration = parse_duration(duration_str) if duration_str else timedelta(hours=1)
                end_dt = dt + duration
                parsed_end_at = end_dt.isoformat()

    elapsed = (time.time() - start) * 1000
    step_timings = state.get("step_timings", {})
    step_timings["parse"] = elapsed

    # Check if we need clarification
    needs_clarification = False
    clarification_prompt = None

    if not title:
        needs_clarification = True
        clarification_prompt = "What would you like to call this event?"
    elif not parsed_start_at:
        needs_clarification = True
        clarification_prompt = f"When should I schedule {title}?"

    logger.info(
        "[CALENDAR] Parsed create: title=%s, start=%s, end=%s",
        title, parsed_start_at, parsed_end_at,
    )

    # Save workflow state if clarification needed
    session_id = state.get("session_id")
    if needs_clarification and session_id:
        manager = get_workflow_state_manager()
        await manager.save_workflow_state(
            session_id=session_id,
            workflow_type=CALENDAR_WORKFLOW_TYPE,
            current_step="awaiting_info",
            partial_state={
                "intent": "create_event",
                "event_title": title,
                "event_location": location,
                "parsed_start_at": parsed_start_at,
                "parsed_end_at": parsed_end_at,
            },
        )
        logger.info("Saved calendar workflow state for session %s", session_id)

    return {
        **state,
        "event_title": title,
        "event_location": location,
        "parsed_start_at": parsed_start_at,
        "parsed_end_at": parsed_end_at,
        "needs_clarification": needs_clarification,
        "clarification_prompt": clarification_prompt,
        "current_step": "parse",
        "step_timings": step_timings,
    }


async def execute_create(state: CalendarWorkflowState) -> CalendarWorkflowState:
    """Execute calendar event creation."""
    start = time.time()

    title = state.get("event_title", "")
    parsed_start_at = state.get("parsed_start_at")
    parsed_end_at = state.get("parsed_end_at")
    location = state.get("event_location")

    if not title or not parsed_start_at or not parsed_end_at:
        return {
            **state,
            "event_created": False,
            "error": "Missing title or time",
            "current_step": "execute",
        }

    start_dt = datetime.fromisoformat(parsed_start_at)
    end_dt = datetime.fromisoformat(parsed_end_at)

    result = await tool_create_event(
        summary=title,
        start=start_dt,
        end=end_dt,
        location=location,
    )

    elapsed = (time.time() - start) * 1000
    step_timings = state.get("step_timings", {})
    step_timings["execute"] = elapsed

    logger.info("[CALENDAR] Create result: %s", result)

    # Clear workflow state on success
    session_id = state.get("session_id")
    if result.get("success") and session_id:
        manager = get_workflow_state_manager()
        await manager.clear_workflow_state(session_id)
        logger.info("Cleared calendar workflow state for session %s", session_id)

    return {
        **state,
        "event_created": result.get("success", False),
        "created_event_id": result.get("event_id"),
        "error": result.get("error"),
        "current_step": "execute",
        "step_timings": step_timings,
    }


async def execute_query(state: CalendarWorkflowState) -> CalendarWorkflowState:
    """Execute calendar query."""
    start = time.time()

    hours_ahead = state.get("hours_ahead", 24)
    max_results = state.get("max_results", 10)
    calendar_name = state.get("calendar_name")

    result = await tool_query_events(
        hours_ahead=hours_ahead,
        max_results=max_results,
        calendar_name=calendar_name,
    )

    elapsed = (time.time() - start) * 1000
    step_timings = state.get("step_timings", {})
    step_timings["execute"] = elapsed

    logger.info("[CALENDAR] Query result: count=%d", result.get("count", 0))

    return {
        **state,
        "events_queried": result.get("success", False),
        "event_list": result.get("events", []),
        "event_count": result.get("count", 0),
        "error": result.get("error"),
        "current_step": "execute",
        "step_timings": step_timings,
    }


async def generate_response(state: CalendarWorkflowState) -> CalendarWorkflowState:
    """Generate the final response based on the operation result."""
    start = time.time()

    intent = state.get("intent", "unknown")
    error = state.get("error")

    if error:
        response = f"Sorry, I couldn't do that: {error}"
    elif intent == "create_event":
        if state.get("needs_clarification"):
            response = state.get("clarification_prompt", "Could you clarify?")
        elif state.get("event_created"):
            title = state.get("event_title", "")
            parsed_start_at = state.get("parsed_start_at")
            if parsed_start_at:
                start_dt = datetime.fromisoformat(parsed_start_at)
                time_str = format_time_for_speech(start_dt)
                response = f"I've added {title} to your calendar {time_str}."
            else:
                response = f"I've added {title} to your calendar."
        else:
            response = "I wasn't able to create that event."

    elif intent == "query_events":
        if state.get("events_queried"):
            count = state.get("event_count", 0)
            events = state.get("event_list", [])

            if count == 0:
                response = "You don't have any upcoming events."
            elif count == 1:
                e = events[0]
                start_dt = datetime.fromisoformat(e["start"])
                time_str = format_time_for_speech(start_dt)
                response = f"You have {e['summary']} {time_str}."
            else:
                response = f"You have {count} events coming up. "
                for i, e in enumerate(events[:3], 1):
                    start_dt = datetime.fromisoformat(e["start"])
                    time_str = format_time_for_speech(start_dt)
                    response += f"{i}. {e['summary']} {time_str}. "
                if count > 3:
                    response += f"And {count - 3} more."
        else:
            response = "I couldn't retrieve your calendar events."

    else:
        response = "I can help you with your calendar. You can ask me to create an event or check your upcoming schedule."

    elapsed = (time.time() - start) * 1000
    step_timings = state.get("step_timings", {})
    step_timings["respond"] = elapsed

    total_ms = sum(step_timings.values())

    logger.info("[CALENDAR] Response: %s", response[:100])

    return {
        **state,
        "response": response,
        "current_step": "respond",
        "step_timings": step_timings,
        "total_ms": total_ms,
    }


# =============================================================================
# Routing Functions
# =============================================================================


def route_by_intent(state: CalendarWorkflowState) -> str:
    """Route to appropriate execution node based on intent."""
    intent = state.get("intent", "unknown")

    if intent == "create_event":
        return "parse_create"
    elif intent == "query_events":
        return "execute_query"
    else:
        return "respond"


def route_after_parse(state: CalendarWorkflowState) -> str:
    """Route after parsing create request."""
    if state.get("needs_clarification"):
        return "respond"
    return "execute_create"


def route_after_check_continuation(
    state: CalendarWorkflowState,
) -> Literal["merge_continuation", "classify"]:
    """Route after checking for continuation."""
    if state.get("is_continuation"):
        return "merge_continuation"
    return "classify"


def route_after_merge(
    state: CalendarWorkflowState,
) -> Literal["parse_create", "execute_create"]:
    """Route after merging continuation input."""
    # If we have both title and time, go to execute
    if state.get("event_title") and state.get("parsed_start_at"):
        return "execute_create"
    # Otherwise go to parse to check what's missing
    return "parse_create"


# =============================================================================
# Graph Builder
# =============================================================================


def build_calendar_graph() -> StateGraph:
    """Build the calendar workflow graph."""
    graph = StateGraph(CalendarWorkflowState)

    # Add nodes
    graph.add_node("check_continuation", check_continuation)
    graph.add_node("merge_continuation", merge_continuation_input)
    graph.add_node("classify", classify_intent)
    graph.add_node("parse_create", parse_create_request)
    graph.add_node("execute_create", execute_create)
    graph.add_node("execute_query", execute_query)
    graph.add_node("respond", generate_response)

    # Set entry point
    graph.set_entry_point("check_continuation")

    # Check for continuation first
    graph.add_conditional_edges(
        "check_continuation",
        route_after_check_continuation,
        {
            "merge_continuation": "merge_continuation",
            "classify": "classify",
        },
    )

    # After merging, route based on what we have
    graph.add_conditional_edges(
        "merge_continuation",
        route_after_merge,
        {
            "parse_create": "parse_create",
            "execute_create": "execute_create",
        },
    )

    # Add edges for normal flow
    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "parse_create": "parse_create",
            "execute_query": "execute_query",
            "respond": "respond",
        },
    )

    graph.add_conditional_edges(
        "parse_create",
        route_after_parse,
        {
            "respond": "respond",
            "execute_create": "execute_create",
        },
    )

    graph.add_edge("execute_create", "respond")
    graph.add_edge("execute_query", "respond")
    graph.add_edge("respond", END)

    return graph


def compile_calendar_graph():
    """Compile the calendar workflow graph."""
    graph = build_calendar_graph()
    return graph.compile()


async def run_calendar_workflow(
    input_text: str,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run the calendar workflow.

    Args:
        input_text: User's natural language request
        session_id: Optional session ID for context

    Returns:
        Final state with response and operation results
    """
    compiled = compile_calendar_graph()

    initial_state: CalendarWorkflowState = {
        "input_text": input_text,
        "session_id": session_id,
        "step_timings": {},
    }

    final_state = await compiled.ainvoke(initial_state)
    return final_state


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "build_calendar_graph",
    "compile_calendar_graph",
    "run_calendar_workflow",
    "CalendarWorkflowState",
    "CALENDAR_WORKFLOW_TYPE",
]
