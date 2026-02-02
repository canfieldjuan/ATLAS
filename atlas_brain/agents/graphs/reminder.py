"""
LangGraph Reminder Workflow.

Consolidates 3 reminder tools into a single workflow:
- ReminderTool (create)
- ListRemindersTool (list)
- CompleteReminderTool (complete/delete)

The LLM classifies intent once, then the graph routes to the appropriate action.
"""

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional
from uuid import UUID

import dateparser
from langgraph.graph import END, StateGraph

from .state import ReminderWorkflowState
from .workflow_state import get_workflow_state_manager

logger = logging.getLogger("atlas.agents.graphs.reminder")

# Workflow type constant for multi-turn support
REMINDER_WORKFLOW_TYPE = "reminder"

# Environment flag for using real tools vs mocks
USE_REAL_TOOLS = os.environ.get("USE_REAL_TOOLS", "false").lower() == "true"


# =============================================================================
# Tool Wrappers (Real vs Mock)
# =============================================================================


async def tool_create_reminder(
    message: str,
    due_at: datetime,
    repeat_pattern: Optional[str] = None,
) -> dict[str, Any]:
    """Create a reminder via the ReminderService."""
    if USE_REAL_TOOLS:
        from atlas_brain.services.reminders import get_reminder_service

        try:
            service = get_reminder_service()
            reminder = await service.create_reminder(
                message=message,
                due_at=due_at,
                repeat_pattern=repeat_pattern,
                source="voice",
            )

            if reminder:
                return {
                    "success": True,
                    "reminder_id": str(reminder.id),
                    "message": reminder.message,
                    "due_at": reminder.due_at.isoformat(),
                    "repeat_pattern": reminder.repeat_pattern,
                }
            else:
                return {"success": False, "error": "Failed to create reminder"}

        except Exception as e:
            logger.exception("Error creating reminder")
            return {"success": False, "error": str(e)}
    else:
        # Mock response
        return {
            "success": True,
            "reminder_id": "mock-reminder-123",
            "message": message,
            "due_at": due_at.isoformat(),
            "repeat_pattern": repeat_pattern,
        }


async def tool_list_reminders(limit: int = 10) -> dict[str, Any]:
    """List upcoming reminders."""
    if USE_REAL_TOOLS:
        from atlas_brain.services.reminders import get_reminder_service

        try:
            service = get_reminder_service()
            reminders = await service.list_reminders(limit=limit)

            reminder_list = []
            for r in reminders:
                due_at = r.due_at
                if due_at.tzinfo is None:
                    due_at = due_at.replace(tzinfo=timezone.utc)

                reminder_list.append({
                    "id": str(r.id),
                    "message": r.message,
                    "due_at": due_at.isoformat(),
                    "repeat_pattern": r.repeat_pattern,
                })

            return {
                "success": True,
                "reminders": reminder_list,
                "count": len(reminder_list),
            }

        except Exception as e:
            logger.exception("Error listing reminders")
            return {"success": False, "error": str(e), "reminders": [], "count": 0}
    else:
        # Mock response
        return {
            "success": True,
            "reminders": [
                {
                    "id": "mock-1",
                    "message": "Call mom",
                    "due_at": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
                    "repeat_pattern": None,
                },
                {
                    "id": "mock-2",
                    "message": "Take medicine",
                    "due_at": (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat(),
                    "repeat_pattern": "daily",
                },
            ],
            "count": 2,
        }


async def tool_complete_reminder(
    reminder_id: Optional[str] = None,
    complete_next: bool = False,
) -> dict[str, Any]:
    """Complete/dismiss a reminder."""
    if USE_REAL_TOOLS:
        from atlas_brain.services.reminders import get_reminder_service

        try:
            service = get_reminder_service()

            if complete_next and not reminder_id:
                # Complete the next upcoming reminder
                next_reminder = await service.get_next_reminder()
                if not next_reminder:
                    return {
                        "success": False,
                        "error": "No active reminders to complete",
                    }
                reminder_id = str(next_reminder.id)
                message = next_reminder.message
            else:
                message = None

            success = await service.complete_reminder(UUID(reminder_id))

            if success:
                return {
                    "success": True,
                    "completed_id": reminder_id,
                    "message": message,
                }
            else:
                return {
                    "success": False,
                    "error": "Reminder not found or already completed",
                }

        except ValueError:
            return {"success": False, "error": "Invalid reminder ID format"}
        except Exception as e:
            logger.exception("Error completing reminder")
            return {"success": False, "error": str(e)}
    else:
        # Mock response
        return {
            "success": True,
            "completed_id": reminder_id or "mock-1",
            "message": "Call mom",
        }


async def tool_delete_reminder(reminder_id: str) -> dict[str, Any]:
    """Delete a reminder permanently."""
    if USE_REAL_TOOLS:
        from atlas_brain.services.reminders import get_reminder_service

        try:
            service = get_reminder_service()
            success = await service.delete_reminder(UUID(reminder_id))

            if success:
                return {"success": True, "deleted_id": reminder_id}
            else:
                return {"success": False, "error": "Reminder not found"}

        except ValueError:
            return {"success": False, "error": "Invalid reminder ID format"}
        except Exception as e:
            logger.exception("Error deleting reminder")
            return {"success": False, "error": str(e)}
    else:
        # Mock response
        return {"success": True, "deleted_id": reminder_id}


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

    # Normalize "next Tuesday" patterns
    text = when_text.strip().lower()
    weekdays = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

    import re
    pattern = r"^next\s+(" + "|".join(weekdays) + r")\b(.*)$"
    match = re.match(pattern, text, re.IGNORECASE)
    if match:
        weekday = match.group(1)
        rest = match.group(2).strip()
        text = f"{weekday} {rest}" if rest else weekday

    # Handle "now" variants
    now_variants = ("right now", "immediately", "asap", "now")
    if text in now_variants:
        text = "in 5 seconds"

    result = dateparser.parse(text, settings=settings_dict)
    return result


def format_time_for_speech(due_at: datetime) -> str:
    """Format datetime for human-readable response."""
    now = datetime.now(timezone.utc)
    if due_at.tzinfo is None:
        due_at = due_at.replace(tzinfo=timezone.utc)

    diff = due_at - now

    # Format time part
    hour = due_at.hour % 12 or 12
    minute = due_at.strftime('%M')
    ampm = 'AM' if due_at.hour < 12 else 'PM'
    time_str = f"{hour}:{minute} {ampm}"

    # Same day
    if due_at.date() == now.date():
        return f"today at {time_str}"

    # Tomorrow
    tomorrow = (now + timedelta(days=1)).date()
    if due_at.date() == tomorrow:
        return f"tomorrow at {time_str}"

    # Within a week
    if diff.days < 7:
        return f"on {due_at.strftime('%A')} at {time_str}"

    # Further out
    return f"on {due_at.strftime('%B')} {due_at.day} at {time_str}"


def classify_reminder_intent(text: str) -> str:
    """
    Simple rule-based intent classification for reminders.

    Returns: "create", "list", "complete", "delete", or "unknown"
    """
    text_lower = text.lower().strip()

    # List patterns
    list_patterns = [
        "list", "show", "what are my", "my reminders", "upcoming reminders",
        "do i have any", "what reminders", "check reminders",
    ]
    for pattern in list_patterns:
        if pattern in text_lower:
            return "list"

    # Complete patterns
    complete_patterns = [
        "complete", "done", "finished", "dismiss", "mark as done",
        "i did", "already did", "check off",
    ]
    for pattern in complete_patterns:
        if pattern in text_lower:
            return "complete"

    # Delete patterns
    delete_patterns = [
        "delete", "remove", "cancel", "clear",
    ]
    for pattern in delete_patterns:
        if pattern in text_lower:
            return "delete"

    # Create patterns (default for "remind me", "set reminder", etc.)
    create_patterns = [
        "remind", "reminder", "set a reminder", "don't let me forget",
        "remember to", "alert me",
    ]
    for pattern in create_patterns:
        if pattern in text_lower:
            return "create"

    # Default to create if contains time-like expressions
    time_patterns = [
        "in ", "at ", "tomorrow", "tonight", "morning", "afternoon",
        "evening", "hour", "minute", "week", "day",
    ]
    for pattern in time_patterns:
        if pattern in text_lower:
            return "create"

    return "unknown"


def parse_create_intent(text: str) -> dict[str, Any]:
    """
    Parse a create reminder request to extract message and time.

    Returns dict with 'message' and 'when' keys.
    """
    text_lower = text.lower().strip()

    import re

    # Handle "now" variants - these should trigger immediately
    now_patterns = ["now", "right now", "immediately", "asap"]

    # Pattern: "remind me now to [message]" or "remind me right now to [message]"
    for now_word in now_patterns:
        pattern_now = rf"remind(?:\s+me)?\s+{now_word}\s+to\s+(.+)"
        match = re.search(pattern_now, text_lower)
        if match:
            message = match.group(1).strip()
            return {"message": message, "when": "now"}

    # Pattern: "remind me to [message] now"
    pattern_msg_now = r"remind(?:\s+me)?\s+to\s+(.+?)\s+(?:right\s+)?now$"
    match = re.search(pattern_msg_now, text_lower)
    if match:
        message = match.group(1).strip()
        return {"message": message, "when": "now"}

    # Pattern: "send me a reminder now" or "send a reminder" (no message = test reminder)
    if re.search(r"send\s+(?:me\s+)?a?\s*reminder", text_lower):
        # Check if "now" is present
        if any(now in text_lower for now in now_patterns):
            return {"message": "Test reminder", "when": "now"}
        # No time specified = now
        if not any(time_word in text_lower for time_word in ["in ", "at ", "tomorrow", "tonight", "morning"]):
            return {"message": "Test reminder", "when": "now"}

    # Pattern: "remind me to [message] (in|at|on|tomorrow|...) [time]"
    pattern1 = r"remind(?:\s+me)?\s+to\s+(.+?)\s+(in\s+\d+|at\s+\d|tomorrow|tonight|next\s+\w+|on\s+\w+)"
    match = re.search(pattern1, text_lower)
    if match:
        message = match.group(1).strip()
        when = match.group(2).strip()
        # Get full time expression from original text
        time_start = text_lower.find(when)
        when = text[time_start:].strip()
        return {"message": message, "when": when}

    # Pattern: "remind me (in|at|tomorrow) [time] to [message]"
    pattern2 = r"remind(?:\s+me)?\s+(in\s+\d+.*?|at\s+\d+.*?|tomorrow.*?)\s+to\s+(.+)"
    match = re.search(pattern2, text_lower)
    if match:
        when = match.group(1).strip()
        message = match.group(2).strip()
        return {"message": message, "when": when}

    # IMPORTANT: "from now" patterns must come BEFORE generic "set a reminder for X to Y"
    # Pattern: "set a reminder for [X] minutes/hours from now to [message]"
    pattern_set_from_now = r"set\s+a?\s*reminder\s+for\s+(\d+)\s*(minute|hour|second)s?\s+from\s+now\s+to\s+(.+)"
    match = re.search(pattern_set_from_now, text_lower)
    if match:
        amount = match.group(1)
        unit = match.group(2)
        message = match.group(3).strip()
        when = f"in {amount} {unit}s"
        return {"message": message, "when": when}

    # Pattern: "[X] minutes/hours from now to [message]"
    pattern_from_now = r"(\d+)\s*(minute|hour|second)s?\s+from\s+now\s+to\s+(.+)"
    match = re.search(pattern_from_now, text_lower)
    if match:
        amount = match.group(1)
        unit = match.group(2)
        message = match.group(3).strip()
        when = f"in {amount} {unit}s"
        return {"message": message, "when": when}

    # Pattern: "[X] minutes/hours from now" (without "to [message]")
    pattern_from_now_only = r"(\d+)\s*(minute|hour|second)s?\s+from\s+now"
    match = re.search(pattern_from_now_only, text_lower)
    if match:
        amount = match.group(1)
        unit = match.group(2)
        when = f"in {amount} {unit}s"
        # Extract message - everything before or after
        before = text_lower[:match.start()].strip()
        before = re.sub(r"^(set\s+a?\s*reminder\s+(?:for\s+)?|remind\s+me\s+to\s*)", "", before).strip()
        if before:
            return {"message": before, "when": when}
        return {"message": "Reminder", "when": when}

    # Pattern: "set a reminder for [time] to [message]" (generic, after specific patterns)
    pattern3 = r"set\s+a?\s*reminder\s+(?:for\s+)?(.+?)\s+to\s+(.+)"
    match = re.search(pattern3, text_lower)
    if match:
        when = match.group(1).strip()
        message = match.group(2).strip()
        return {"message": message, "when": when}

    # Fallback: try to find time expression and use rest as message
    time_keywords = [
        r"in\s+\d+\s*(?:seconds?|minutes?|hours?|days?|weeks?)",
        r"at\s+\d+(?::\d+)?\s*(?:am|pm)?",
        r"tomorrow(?:\s+(?:at|morning|afternoon|evening|night))?",
        r"tonight",
        r"next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
    ]

    for time_pattern in time_keywords:
        match = re.search(time_pattern, text_lower)
        if match:
            when = match.group(0)
            # Remove the time part and common filler words to get message
            message = text_lower.replace(when, "").strip()
            # Remove filler phrases at start
            filler_pattern = r"^(remind\s+me\s+to\s*|set\s+a?\s*reminder\s+(?:for\s+)?(?:to\s+)?|to\s+)"
            message = re.sub(filler_pattern, "", message)
            message = message.strip()
            # Clear if message is just leftover filler words
            if message in ("for", "to", "a", "the"):
                message = None
            return {"message": message if message else None, "when": when}

    # Fallback: "remind me to [message]" without time
    pattern_msg_only = r"remind(?:\s+me)?\s+to\s+(.+)"
    match = re.search(pattern_msg_only, text_lower)
    if match:
        message = match.group(1).strip()
        return {"message": message, "when": None}

    return {"message": None, "when": None}


def parse_complete_intent(text: str, reminder_list: list[dict]) -> dict[str, Any]:
    """
    Parse which reminder to complete.

    Returns dict with 'reminder_id' or 'by_index' or 'complete_next'.
    """
    text_lower = text.lower()

    # Check for "all" / "clear all"
    if "all" in text_lower:
        return {"complete_all": True}

    # Check for ordinal references ("first one", "second", etc.)
    ordinals = {
        "first": 0, "1st": 0,
        "second": 1, "2nd": 1,
        "third": 2, "3rd": 2,
        "fourth": 3, "4th": 3,
        "fifth": 4, "5th": 4,
        "last": -1,
    }

    for word, idx in ordinals.items():
        if word in text_lower:
            if idx == -1 and reminder_list:
                idx = len(reminder_list) - 1
            if 0 <= idx < len(reminder_list):
                return {"reminder_id": reminder_list[idx]["id"], "by_index": idx}

    # Default to completing the next upcoming reminder
    return {"complete_next": True}


# =============================================================================
# Graph Nodes
# =============================================================================


async def check_continuation(state: ReminderWorkflowState) -> ReminderWorkflowState:
    """Check if this is a continuation of a saved workflow."""
    session_id = state.get("session_id")
    if not session_id:
        return state

    manager = get_workflow_state_manager()
    saved = await manager.restore_workflow_state(session_id)

    if saved and saved.workflow_type == REMINDER_WORKFLOW_TYPE:
        if saved.is_expired():
            logger.info("Reminder workflow expired for session %s", session_id)
            await manager.clear_workflow_state(session_id)
            return state

        logger.info(
            "Continuing reminder workflow from step %s for session %s",
            saved.current_step,
            session_id,
        )
        return {
            **state,
            "is_continuation": True,
            "restored_from_step": saved.current_step,
            "intent": saved.partial_state.get("intent", "create"),
            "reminder_message": saved.partial_state.get("reminder_message"),
            "reminder_time": saved.partial_state.get("reminder_time"),
            "parsed_due_at": saved.partial_state.get("parsed_due_at"),
        }

    return state


async def merge_continuation_input(state: ReminderWorkflowState) -> ReminderWorkflowState:
    """Merge new user input with saved partial state."""
    input_text = state.get("input_text", "")
    restored_step = state.get("restored_from_step", "")

    # Parse the new input to extract any provided values
    parsed = parse_create_intent(input_text)
    new_message = parsed.get("message")
    new_when = parsed.get("when")

    # Merge with existing state
    reminder_message = state.get("reminder_message")
    reminder_time = state.get("reminder_time")
    parsed_due_at = state.get("parsed_due_at")

    # If we were waiting for a message and user provided one
    if not reminder_message and new_message:
        reminder_message = new_message

    # If we were waiting for a time and user provided one
    if not parsed_due_at and new_when:
        dt = parse_datetime(new_when)
        if dt:
            parsed_due_at = dt.isoformat()
        reminder_time = new_when

    # If input looks like a plain time expression (no reminder keywords)
    if not new_message and not new_when:
        text_lower = input_text.strip().lower()
        # Check if it looks like a time
        time_patterns = ["in ", "at ", "tomorrow", "tonight", "morning", "hour", "minute"]
        if any(p in text_lower for p in time_patterns):
            dt = parse_datetime(input_text)
            if dt:
                parsed_due_at = dt.isoformat()
                reminder_time = input_text
        # Otherwise treat as message
        elif not reminder_message:
            reminder_message = input_text.strip()

    logger.info(
        "[REMINDER] Merged continuation: message=%s, time=%s, due_at=%s",
        reminder_message,
        reminder_time,
        parsed_due_at,
    )

    return {
        **state,
        "reminder_message": reminder_message,
        "reminder_time": reminder_time,
        "parsed_due_at": parsed_due_at,
    }


async def classify_intent(state: ReminderWorkflowState) -> ReminderWorkflowState:
    """Classify the reminder intent from user input."""
    start = time.time()

    input_text = state.get("input_text", "")
    intent = classify_reminder_intent(input_text)

    elapsed = (time.time() - start) * 1000
    step_timings = state.get("step_timings", {})
    step_timings["classify"] = elapsed

    logger.info("[REMINDER] Classified intent: %s", intent)

    return {
        **state,
        "intent": intent,
        "current_step": "classify",
        "step_timings": step_timings,
    }


async def parse_create_request(state: ReminderWorkflowState) -> ReminderWorkflowState:
    """Parse a create reminder request."""
    start = time.time()

    # If this is a continuation, use merged values
    if state.get("is_continuation"):
        message = state.get("reminder_message")
        when = state.get("reminder_time")
        parsed_due_at = state.get("parsed_due_at")
    else:
        input_text = state.get("input_text", "")
        parsed = parse_create_intent(input_text)
        message = parsed.get("message")
        when = parsed.get("when")
        parsed_due_at = None
        if when:
            dt = parse_datetime(when)
            if dt:
                parsed_due_at = dt.isoformat()

    elapsed = (time.time() - start) * 1000
    step_timings = state.get("step_timings", {})
    step_timings["parse"] = elapsed

    # Check if we need clarification
    needs_clarification = False
    clarification_prompt = None

    if not message:
        needs_clarification = True
        clarification_prompt = "What would you like to be reminded about?"
    elif not parsed_due_at:
        needs_clarification = True
        clarification_prompt = f"When should I remind you to {message}?"

    logger.info("[REMINDER] Parsed create: message=%s, when=%s, due_at=%s", message, when, parsed_due_at)

    # Save workflow state if clarification needed
    session_id = state.get("session_id")
    if needs_clarification and session_id:
        manager = get_workflow_state_manager()
        await manager.save_workflow_state(
            session_id=session_id,
            workflow_type=REMINDER_WORKFLOW_TYPE,
            current_step="awaiting_info",
            partial_state={
                "intent": "create",
                "reminder_message": message,
                "reminder_time": when,
                "parsed_due_at": parsed_due_at,
            },
        )
        logger.info("Saved reminder workflow state for session %s", session_id)

    return {
        **state,
        "reminder_message": message,
        "reminder_time": when,
        "parsed_due_at": parsed_due_at,
        "needs_clarification": needs_clarification,
        "clarification_prompt": clarification_prompt,
        "current_step": "parse",
        "step_timings": step_timings,
    }


async def execute_create(state: ReminderWorkflowState) -> ReminderWorkflowState:
    """Execute reminder creation."""
    start = time.time()

    message = state.get("reminder_message", "")
    parsed_due_at = state.get("parsed_due_at")
    repeat_pattern = state.get("repeat_pattern")

    if not message or not parsed_due_at:
        return {
            **state,
            "reminder_created": False,
            "error": "Missing message or time",
            "current_step": "execute",
        }

    due_at = datetime.fromisoformat(parsed_due_at)

    result = await tool_create_reminder(
        message=message,
        due_at=due_at,
        repeat_pattern=repeat_pattern,
    )

    elapsed = (time.time() - start) * 1000
    step_timings = state.get("step_timings", {})
    step_timings["execute"] = elapsed

    logger.info("[REMINDER] Create result: %s", result)

    # Clear workflow state on success
    session_id = state.get("session_id")
    if result.get("success") and session_id:
        manager = get_workflow_state_manager()
        await manager.clear_workflow_state(session_id)
        logger.info("Cleared reminder workflow state for session %s", session_id)

    return {
        **state,
        "reminder_created": result.get("success", False),
        "created_reminder_id": result.get("reminder_id"),
        "error": result.get("error"),
        "current_step": "execute",
        "step_timings": step_timings,
    }


async def execute_list(state: ReminderWorkflowState) -> ReminderWorkflowState:
    """Execute reminder listing."""
    start = time.time()

    result = await tool_list_reminders(limit=10)

    elapsed = (time.time() - start) * 1000
    step_timings = state.get("step_timings", {})
    step_timings["execute"] = elapsed

    logger.info("[REMINDER] List result: count=%d", result.get("count", 0))

    return {
        **state,
        "reminders_listed": result.get("success", False),
        "reminder_list": result.get("reminders", []),
        "reminder_count": result.get("count", 0),
        "error": result.get("error"),
        "current_step": "execute",
        "step_timings": step_timings,
    }


async def execute_complete(state: ReminderWorkflowState) -> ReminderWorkflowState:
    """Execute reminder completion."""
    start = time.time()

    input_text = state.get("input_text", "")
    reminder_list = state.get("reminder_list", [])

    # If we don't have a reminder list yet, fetch it first
    if not reminder_list:
        list_result = await tool_list_reminders(limit=10)
        reminder_list = list_result.get("reminders", [])

    # Parse which reminder to complete
    parsed = parse_complete_intent(input_text, reminder_list)

    if parsed.get("complete_all"):
        # Complete all reminders
        completed_ids = []
        for r in reminder_list:
            result = await tool_complete_reminder(reminder_id=r["id"])
            if result.get("success"):
                completed_ids.append(r["id"])

        elapsed = (time.time() - start) * 1000
        step_timings = state.get("step_timings", {})
        step_timings["execute"] = elapsed

        return {
            **state,
            "reminder_completed": len(completed_ids) > 0,
            "completed_reminder_id": ",".join(completed_ids),
            "complete_all": True,
            "current_step": "execute",
            "step_timings": step_timings,
        }

    # Complete specific reminder
    reminder_id = parsed.get("reminder_id")
    complete_next = parsed.get("complete_next", False)

    result = await tool_complete_reminder(
        reminder_id=reminder_id,
        complete_next=complete_next,
    )

    elapsed = (time.time() - start) * 1000
    step_timings = state.get("step_timings", {})
    step_timings["execute"] = elapsed

    logger.info("[REMINDER] Complete result: %s", result)

    return {
        **state,
        "reminder_completed": result.get("success", False),
        "completed_reminder_id": result.get("completed_id"),
        "error": result.get("error"),
        "current_step": "execute",
        "step_timings": step_timings,
    }


async def execute_delete(state: ReminderWorkflowState) -> ReminderWorkflowState:
    """Execute reminder deletion."""
    start = time.time()

    input_text = state.get("input_text", "")
    reminder_list = state.get("reminder_list", [])

    # If we don't have a reminder list yet, fetch it first
    if not reminder_list:
        list_result = await tool_list_reminders(limit=10)
        reminder_list = list_result.get("reminders", [])

    # Parse which reminder to delete (reuse complete parsing)
    parsed = parse_complete_intent(input_text, reminder_list)

    reminder_id = parsed.get("reminder_id")

    if not reminder_id and reminder_list:
        # Default to first reminder
        reminder_id = reminder_list[0]["id"]

    if not reminder_id:
        return {
            **state,
            "reminder_deleted": False,
            "error": "No reminder specified to delete",
            "current_step": "execute",
        }

    result = await tool_delete_reminder(reminder_id)

    elapsed = (time.time() - start) * 1000
    step_timings = state.get("step_timings", {})
    step_timings["execute"] = elapsed

    logger.info("[REMINDER] Delete result: %s", result)

    return {
        **state,
        "reminder_deleted": result.get("success", False),
        "deleted_reminder_id": result.get("deleted_id"),
        "error": result.get("error"),
        "current_step": "execute",
        "step_timings": step_timings,
    }


async def generate_response(state: ReminderWorkflowState) -> ReminderWorkflowState:
    """Generate the final response based on the operation result."""
    start = time.time()

    intent = state.get("intent", "unknown")
    error = state.get("error")

    if error:
        response = f"Sorry, I couldn't do that: {error}"
    elif intent == "create":
        if state.get("needs_clarification"):
            response = state.get("clarification_prompt", "Could you clarify?")
        elif state.get("reminder_created"):
            message = state.get("reminder_message", "")
            parsed_due_at = state.get("parsed_due_at")
            if parsed_due_at:
                due_at = datetime.fromisoformat(parsed_due_at)
                time_str = format_time_for_speech(due_at)
                response = f"I'll remind you to {message} {time_str}."
                repeat = state.get("repeat_pattern")
                if repeat:
                    response += f" This will repeat {repeat}."
            else:
                response = f"I've set a reminder to {message}."
        else:
            response = "I wasn't able to create that reminder."

    elif intent == "list":
        if state.get("reminders_listed"):
            count = state.get("reminder_count", 0)
            reminders = state.get("reminder_list", [])

            if count == 0:
                response = "You don't have any upcoming reminders."
            elif count == 1:
                r = reminders[0]
                due_at = datetime.fromisoformat(r["due_at"])
                time_str = format_time_for_speech(due_at)
                response = f"You have one reminder: {r['message']} {time_str}."
            else:
                response = f"You have {count} reminders. "
                for i, r in enumerate(reminders[:3], 1):
                    due_at = datetime.fromisoformat(r["due_at"])
                    time_str = format_time_for_speech(due_at)
                    response += f"{i}. {r['message']} {time_str}. "
                if count > 3:
                    response += f"And {count - 3} more."
        else:
            response = "I couldn't retrieve your reminders."

    elif intent == "complete":
        if state.get("reminder_completed"):
            if state.get("complete_all"):
                response = "All reminders have been completed."
            else:
                response = "Reminder completed."
        else:
            response = "I couldn't complete that reminder."

    elif intent == "delete":
        if state.get("reminder_deleted"):
            response = "Reminder deleted."
        else:
            response = "I couldn't delete that reminder."

    else:
        response = "I'm not sure what you'd like me to do with reminders. You can ask me to set a reminder, list your reminders, or mark one as complete."

    elapsed = (time.time() - start) * 1000
    step_timings = state.get("step_timings", {})
    step_timings["respond"] = elapsed

    total_ms = sum(step_timings.values())

    logger.info("[REMINDER] Response: %s", response[:100])

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


def route_by_intent(state: ReminderWorkflowState) -> str:
    """Route to appropriate execution node based on intent."""
    intent = state.get("intent", "unknown")

    if intent == "create":
        return "parse_create"
    elif intent == "list":
        return "execute_list"
    elif intent == "complete":
        return "execute_complete"
    elif intent == "delete":
        return "execute_delete"
    else:
        return "respond"


def route_after_parse(state: ReminderWorkflowState) -> str:
    """Route after parsing create request."""
    if state.get("needs_clarification"):
        return "respond"
    return "execute_create"


def route_after_check_continuation(
    state: ReminderWorkflowState,
) -> Literal["merge_continuation", "classify"]:
    """Route after checking for continuation."""
    if state.get("is_continuation"):
        return "merge_continuation"
    return "classify"


def route_after_merge(
    state: ReminderWorkflowState,
) -> Literal["parse_create", "execute_create"]:
    """Route after merging continuation input."""
    # If we have both message and time, go to execute
    if state.get("reminder_message") and state.get("parsed_due_at"):
        return "execute_create"
    # Otherwise go to parse to check what's missing
    return "parse_create"


# =============================================================================
# Graph Builder
# =============================================================================


def build_reminder_graph() -> StateGraph:
    """Build the reminder workflow graph."""
    graph = StateGraph(ReminderWorkflowState)

    # Add nodes
    graph.add_node("check_continuation", check_continuation)
    graph.add_node("merge_continuation", merge_continuation_input)
    graph.add_node("classify", classify_intent)
    graph.add_node("parse_create", parse_create_request)
    graph.add_node("execute_create", execute_create)
    graph.add_node("execute_list", execute_list)
    graph.add_node("execute_complete", execute_complete)
    graph.add_node("execute_delete", execute_delete)
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
            "execute_list": "execute_list",
            "execute_complete": "execute_complete",
            "execute_delete": "execute_delete",
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
    graph.add_edge("execute_list", "respond")
    graph.add_edge("execute_complete", "respond")
    graph.add_edge("execute_delete", "respond")
    graph.add_edge("respond", END)

    return graph


def compile_reminder_graph():
    """Compile the reminder workflow graph."""
    graph = build_reminder_graph()
    return graph.compile()


async def run_reminder_workflow(
    input_text: str,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run the reminder workflow.

    Args:
        input_text: User's natural language request
        session_id: Optional session ID for context

    Returns:
        Final state with response and operation results
    """
    compiled = compile_reminder_graph()

    initial_state: ReminderWorkflowState = {
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
    "build_reminder_graph",
    "compile_reminder_graph",
    "run_reminder_workflow",
    "ReminderWorkflowState",
]
