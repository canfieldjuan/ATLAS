"""
ReceptionistAgent LangGraph implementation.

Business phone call handler with appointment booking flow.
Optimized for cleaning business workflow:
- 90-95% of callers want to book a FREE ESTIMATE
- Never quote prices - every home is different
- Calls typically last 3-4 minutes
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Literal, Optional

from langgraph.graph import END, StateGraph

from .state import ActionResult, ReceptionistAgentState
from ...utils.cuda_lock import get_cuda_lock

logger = logging.getLogger("atlas.agents.graphs.receptionist")


class ConversationPhase(Enum):
    """Simple conversation phases."""

    GREETING = auto()
    ANSWERING = auto()
    COLLECTING = auto()
    CONFIRMING = auto()
    COMPLETE = auto()
    TRANSFER = auto()


# Node functions


async def detect_intent(state: ReceptionistAgentState) -> ReceptionistAgentState:
    """Detect caller intent from input."""
    start_time = time.perf_counter()
    input_text = state["input_text"]
    text_lower = input_text.lower()
    current_phase = state.get("call_phase", "greeting")

    # Extract info from caller's response
    extracted = _extract_caller_info(text_lower, input_text, state)

    new_phase = current_phase
    action_type = "conversation"
    tools_to_call: list[str] = []

    if current_phase == "greeting":
        # First turn - detect what they want
        if _wants_estimate(text_lower):
            new_phase = "collecting"
            action_type = "tool_use"
            tools_to_call = ["check_availability"]
            logger.info("Intent: estimate booking")
        elif _wants_message(text_lower):
            action_type = "tool_use"
            tools_to_call = ["send_notification"]
            logger.info("Intent: leave message")
        elif _is_asking_questions(text_lower):
            new_phase = "answering"
            logger.info("Intent: asking questions")
        else:
            # Default assumption: they probably want an estimate
            new_phase = "collecting"
            logger.info("Intent: assumed estimate (default)")

    elif current_phase == "answering":
        if _wants_estimate(text_lower):
            new_phase = "collecting"
            action_type = "tool_use"
            tools_to_call = ["check_availability"]
            logger.info("Question phase → now wants estimate")
        elif _is_not_interested(text_lower):
            logger.info("Caller not interested, wrapping up")

    elif current_phase == "collecting":
        # Check if we have enough info to confirm
        if _has_enough_info(state):
            new_phase = "confirming"
            logger.info("Info collected, moving to confirmation")

    elif current_phase == "confirming":
        if _is_confirmation(text_lower):
            action_type = "tool_use"
            tools_to_call = ["book_appointment"]
            new_phase = "complete"
            logger.info("Booking confirmed")
        elif _is_rejection(text_lower):
            new_phase = "collecting"
            logger.info("Changes requested, back to collecting")

    classify_ms = (time.perf_counter() - start_time) * 1000

    return {
        **state,
        **extracted,
        "call_phase": new_phase,
        "action_type": action_type,
        "tools_to_call": tools_to_call,
        "classify_ms": classify_ms,
    }


async def execute_tools(state: ReceptionistAgentState) -> ReceptionistAgentState:
    """Execute phone tools with collected call context."""
    action_type = state.get("action_type", "conversation")
    if action_type != "tool_use":
        return state

    start_time = time.perf_counter()
    tools_to_call = state.get("tools_to_call", [])

    from ...tools import tool_registry

    result = ActionResult(success=True, message="")
    tool_results: dict[str, Any] = {}
    booking_id: Optional[str] = None
    booking_confirmed = False

    for tool_name in tools_to_call:
        try:
            params = {"query": state.get("input_text", "")}

            if tool_name == "send_notification":
                msg = (
                    f"Callback request from {state.get('customer_name', 'Unknown')} "
                    f"({state.get('caller_number', 'no phone')})"
                )
                params.update(
                    {
                        "message": msg,
                        "title": "Callback Request",
                        "priority": "high",
                    }
                )

            elif tool_name == "book_appointment":
                # Build booking params from collected info
                preferred_date = state.get("appointment_time", "")
                if "tomorrow" in preferred_date.lower():
                    book_date = datetime.now() + timedelta(days=1)
                elif "next week" in preferred_date.lower():
                    book_date = datetime.now() + timedelta(days=7)
                else:
                    book_date = datetime.now() + timedelta(days=1)

                # Default time based on preference
                if "morning" in preferred_date.lower():
                    book_time = "09:00"
                elif "afternoon" in preferred_date.lower():
                    book_time = "14:00"
                elif "evening" in preferred_date.lower():
                    book_time = "17:00"
                else:
                    book_time = "10:00"

                params.update(
                    {
                        "date": book_date.strftime("%Y-%m-%d"),
                        "time": book_time,
                        "customer_name": state.get("customer_name", "Customer"),
                        "customer_phone": state.get("caller_number", ""),
                        "service_type": "Free Estimate",
                        "address": state.get("customer_address", ""),
                    }
                )

            tool_result = await tool_registry.execute(tool_name, params)
            tool_results[tool_name] = {
                "success": tool_result.success,
                "data": tool_result.data,
                "message": tool_result.message,
            }

            if tool_name == "book_appointment" and tool_result.success:
                booking_confirmed = True
                booking_id = tool_result.data.get("appointment_id")

            if tool_result.success and tool_result.message:
                result.message = tool_result.message

        except Exception as e:
            logger.warning("Tool %s failed: %s", tool_name, e)
            tool_results[tool_name] = {"success": False, "error": str(e)}

    act_ms = (time.perf_counter() - start_time) * 1000

    return {
        **state,
        "action_result": result,
        "tool_results": tool_results,
        "booking_confirmed": booking_confirmed,
        "booking_id": booking_id,
        "act_ms": act_ms,
    }


async def generate_response(state: ReceptionistAgentState) -> ReceptionistAgentState:
    """Generate response using LLM with business context."""
    start_time = time.perf_counter()

    from ...services import llm_registry
    from ...services.protocols import Message

    llm = llm_registry.get_active()
    if llm is None:
        return {**state, "response": "Thank you for calling. How can I help you?"}

    # Build system prompt
    system_prompt = _build_system_prompt(state)

    # Add tool context if available
    action_result = state.get("action_result")
    if action_result and action_result.message:
        system_prompt += f"\n\nDATA:\n{action_result.message}"

    messages = [Message(role="system", content=system_prompt)]

    # Add current message
    messages.append(Message(role="user", content=state.get("input_text", "")))

    try:
        cuda_lock = get_cuda_lock()
        async with cuda_lock:
            llm_result = llm.chat(
                messages=messages,
                max_tokens=150,
                temperature=0.7,
            )
        response = llm_result.get("response", "").strip()
        if response:
            respond_ms = (time.perf_counter() - start_time) * 1000
            return {
                **state,
                "response": response,
                "respond_ms": respond_ms,
                "llm_input_tokens": llm_result.get("prompt_eval_count", 0),
                "llm_output_tokens": llm_result.get("eval_count", 0),
                "llm_system_prompt": system_prompt,
                "llm_history_count": 0,
                "llm_prompt_eval_duration_ms": llm_result.get("prompt_eval_duration_ms"),
                "llm_eval_duration_ms": llm_result.get("eval_duration_ms"),
                "llm_total_duration_ms": llm_result.get("total_duration_ms"),
                "llm_provider_request_id": llm_result.get("request_id") or llm_result.get("id"),
                "llm_has_response": True,
            }
    except Exception as e:
        logger.warning("LLM response failed: %s", e)

    respond_ms = (time.perf_counter() - start_time) * 1000
    return {
        **state,
        "response": "Thank you for calling. How can I help you today?",
        "respond_ms": respond_ms,
    }


# Helper functions


def _build_system_prompt(state: ReceptionistAgentState) -> str:
    """Build system prompt optimized for estimate booking."""
    phase = state.get("call_phase", "greeting")
    customer_name = state.get("customer_name")
    customer_address = state.get("customer_address")
    appointment_time = state.get("appointment_time")

    # Collected info summary
    info_parts = []
    if customer_name:
        info_parts.append(f"Name: {customer_name}")
    if customer_address:
        info_parts.append(f"Address: {customer_address}")
    if appointment_time:
        info_parts.append(f"Time: {appointment_time}")
    info_summary = "; ".join(info_parts) if info_parts else "No info collected yet"

    return f"""You are a professional virtual receptionist for a cleaning company.

## YOUR PRIMARY GOAL
Book FREE estimates. 90% of callers want this - get them scheduled quickly.

## WHAT YOU NEED TO COLLECT (for booking)
1. Their NAME
2. Service ADDRESS (where we'll do the estimate)
3. Preferred DAY and TIME (morning/afternoon works fine)

## HANDLING DIFFERENT CALLERS

### Ready to Book (90% of calls)
Get their name, address, and when works for them. Keep it moving.

### Info Seekers ("What services do you offer?")
Answer briefly: "We do residential and commercial cleaning - deep cleans, regular maintenance, move-in/move-out, and offices."
Then pivot: "Would you like to schedule a free estimate so we can see your space?"

### Price Shoppers ("How much do you charge?")
NEVER quote prices. Say something like:
- "Every home is different, so we do free in-person estimates. That way you get an accurate price, no surprises."
Then offer: "Want me to get you on the schedule?"

## CRITICAL RULES
- NEVER quote specific prices or ranges
- Keep responses SHORT (1-2 sentences)
- Be warm but efficient
- Always pivot back to offering the free estimate

## CURRENT CALL STATUS
Phase: {phase}
Info collected: {info_summary}
"""


def _wants_estimate(text: str) -> bool:
    """Check if caller wants to book an estimate."""
    keywords = [
        "estimate",
        "appointment",
        "schedule",
        "book",
        "available",
        "come out",
        "set up",
        "cleaning",
        "quote",
        "free estimate",
        "get started",
    ]
    return any(kw in text for kw in keywords)


def _wants_message(text: str) -> bool:
    """Check if caller wants to leave a message."""
    keywords = [
        "message",
        "call back",
        "callback",
        "leave",
        "tell them",
        "have them call",
        "speak to someone",
    ]
    return any(kw in text for kw in keywords)


def _is_asking_questions(text: str) -> bool:
    """Check if caller is asking about services/pricing."""
    price_keywords = [
        "how much",
        "price",
        "cost",
        "rate",
        "charge",
        "expensive",
        "affordable",
        "ballpark",
        "range",
    ]
    info_keywords = [
        "what services",
        "do you offer",
        "do you do",
        "what kind",
        "how does it work",
        "what's included",
        "what areas",
        "where do you",
        "how long",
    ]
    return any(kw in text for kw in price_keywords + info_keywords)


def _is_not_interested(text: str) -> bool:
    """Check if caller is just shopping / not interested."""
    keywords = [
        "just looking",
        "just checking",
        "not right now",
        "maybe later",
        "think about it",
        "get back to you",
        "too expensive",
        "can't afford",
        "no thanks",
        "not interested",
    ]
    return any(kw in text for kw in keywords)


def _is_confirmation(text: str) -> bool:
    """Check if caller is confirming."""
    keywords = [
        "yes",
        "yeah",
        "yep",
        "correct",
        "that's right",
        "sounds good",
        "perfect",
        "ok",
        "okay",
    ]
    return any(kw in text for kw in keywords)


def _is_rejection(text: str) -> bool:
    """Check if caller wants changes."""
    keywords = [r"\bno\b", r"\bchange\b", r"\bdifferent\b", r"\bactually\b", r"\bwait\b"]
    return any(re.search(kw, text, re.IGNORECASE) for kw in keywords)


def _has_enough_info(state: ReceptionistAgentState) -> bool:
    """Check if we have minimum info to book."""
    has_name = state.get("customer_name") is not None
    has_address = state.get("customer_address") is not None
    has_time = state.get("appointment_time") is not None
    return has_name and (has_address or has_time)


def _extract_caller_info(
    text_lower: str, text_original: str, state: ReceptionistAgentState
) -> dict[str, Any]:
    """Extract name, address, time preferences from caller's response."""
    extracted: dict[str, Any] = {}

    # Name detection
    name_triggers = ["my name is", "this is", "i'm ", "i am "]
    for trigger in name_triggers:
        if trigger in text_lower:
            idx = text_lower.find(trigger) + len(trigger)
            remaining = text_original[idx:].strip()
            words = remaining.split()[:3]
            if words:
                name = " ".join(words).rstrip(".,!?")
                if len(name) > 1:
                    extracted["customer_name"] = name
                    break

    # Address detection
    address_pattern = r"\d+\s+[\w\s]+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|way|court|ct|boulevard|blvd)"
    match = re.search(address_pattern, text_lower)
    if match:
        extracted["customer_address"] = text_original[match.start() : match.end()]

    # Time preference detection
    time_keywords = {
        "morning": "morning",
        "afternoon": "afternoon",
        "evening": "evening",
        "monday": "Monday",
        "tuesday": "Tuesday",
        "wednesday": "Wednesday",
        "thursday": "Thursday",
        "friday": "Friday",
        "saturday": "Saturday",
        "tomorrow": "tomorrow",
        "next week": "next week",
        "this week": "this week",
    }
    for kw, value in time_keywords.items():
        if kw in text_lower:
            current_time = state.get("appointment_time", "")
            if not current_time:
                extracted["appointment_time"] = value
            elif value not in current_time:
                extracted["appointment_time"] = f"{current_time} {value}"
            break

    # Phone detection
    phone_pattern = r"\d{3}[-.\s]?\d{3}[-.\s]?\d{4}"
    phone_match = re.search(phone_pattern, text_original)
    if phone_match and not state.get("caller_number"):
        extracted["caller_number"] = phone_match.group(0)

    return extracted


# Routing functions


def route_after_intent(
    state: ReceptionistAgentState,
) -> Literal["execute", "respond"]:
    """Route based on detected intent."""
    action_type = state.get("action_type", "conversation")
    if action_type == "tool_use":
        return "execute"
    return "respond"


# Build the graph


def build_receptionist_agent_graph() -> StateGraph:
    """
    Build the ReceptionistAgent LangGraph.

    Flow:
    detect_intent -> (execute -> respond) | respond
    """
    graph = StateGraph(ReceptionistAgentState)

    # Add nodes
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("execute", execute_tools)
    graph.add_node("respond", generate_response)

    # Set entry point
    graph.set_entry_point("detect_intent")

    # Add edges
    graph.add_conditional_edges(
        "detect_intent",
        route_after_intent,
        {
            "execute": "execute",
            "respond": "respond",
        },
    )

    graph.add_edge("execute", "respond")
    graph.add_edge("respond", END)

    return graph


# Compiled graph singleton
_receptionist_graph: Optional[StateGraph] = None


def get_receptionist_agent_graph() -> StateGraph:
    """Get or create the compiled ReceptionistAgent graph."""
    global _receptionist_graph
    if _receptionist_graph is None:
        _receptionist_graph = build_receptionist_agent_graph()
    return _receptionist_graph


class ReceptionistAgentGraph:
    """
    ReceptionistAgent using LangGraph for orchestration.

    Handles phone calls with appointment booking workflow.
    """

    def __init__(
        self,
        business_context: Optional[Any] = None,
        session_id: Optional[str] = None,
        caller_phone: Optional[str] = None,
    ):
        """Initialize ReceptionistAgent graph."""
        self._business_context = business_context
        self._session_id = session_id
        self._caller_phone = caller_phone
        self._graph = get_receptionist_agent_graph()
        self._compiled = self._graph.compile()

        # Persistent call state across turns
        self._call_state: dict[str, Any] = {
            "call_phase": "greeting",
            "customer_name": None,
            "customer_phone": caller_phone,
            "customer_address": None,
            "appointment_time": None,
            "booking_confirmed": False,
            "booking_id": None,
        }

    async def run(
        self,
        input_text: str,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Process phone call input and return result.

        Args:
            input_text: Caller input text
            session_id: Call/session identifier
            **kwargs: Additional context

        Returns:
            Result dict with response, booking status, etc.
        """
        start_time = time.perf_counter()

        # Build initial state from persistent call state
        initial_state: ReceptionistAgentState = {
            "input_text": input_text,
            "input_type": "voice",
            "session_id": session_id or self._session_id,
            "call_id": kwargs.get("call_id"),
            "caller_number": self._caller_phone or kwargs.get("caller_number"),
            "messages": [],
            "action_type": "conversation",
            "confidence": 0.8,
            "is_phone_call": True,
            "use_phone_tts": True,
            "tools_to_call": [],
            "tool_results": {},
            # Carry forward persistent state
            **self._call_state,
        }

        # Run the graph
        try:
            final_state = await self._compiled.ainvoke(initial_state)
        except Exception as e:
            logger.exception("Error running ReceptionistAgent graph: %s", e)
            final_state = {
                **initial_state,
                "response": "I'm sorry, could you repeat that?",
                "error": str(e),
            }

        # Update persistent call state
        self._call_state.update(
            {
                "call_phase": final_state.get("call_phase", "greeting"),
                "customer_name": final_state.get("customer_name"),
                "customer_address": final_state.get("customer_address"),
                "appointment_time": final_state.get("appointment_time"),
                "booking_confirmed": final_state.get("booking_confirmed", False),
                "booking_id": final_state.get("booking_id"),
            }
        )

        total_ms = (time.perf_counter() - start_time) * 1000

        return {
            "success": final_state.get("error") is None,
            "response_text": final_state.get("response", ""),
            "action_type": final_state.get("action_type", "conversation"),
            "call_phase": final_state.get("call_phase", "greeting"),
            "booking_confirmed": final_state.get("booking_confirmed", False),
            "booking_id": final_state.get("booking_id"),
            "error": final_state.get("error"),
            "timing": {
                "total": total_ms,
                "classify": final_state.get("classify_ms", 0),
                "act": final_state.get("act_ms", 0),
                "respond": final_state.get("respond_ms", 0),
            },
            "llm_meta": {
                "input_tokens": final_state.get("llm_input_tokens"),
                "output_tokens": final_state.get("llm_output_tokens"),
                "system_prompt": final_state.get("llm_system_prompt"),
                "history_count": final_state.get("llm_history_count", 0),
                "prompt_eval_duration_ms": final_state.get("llm_prompt_eval_duration_ms"),
                "eval_duration_ms": final_state.get("llm_eval_duration_ms"),
                "total_duration_ms": final_state.get("llm_total_duration_ms"),
                "provider_request_id": final_state.get("llm_provider_request_id"),
                "has_llm_response": final_state.get("llm_has_response", False),
            },
        }

    def reset_call(self, caller_phone: Optional[str] = None) -> None:
        """Reset call context for a new call."""
        self._caller_phone = caller_phone
        self._call_state = {
            "call_phase": "greeting",
            "customer_name": None,
            "customer_phone": caller_phone,
            "customer_address": None,
            "appointment_time": None,
            "booking_confirmed": False,
            "booking_id": None,
        }
        logger.info("Call context reset for new call")


# Factory functions


def get_receptionist_agent_langgraph(
    business_context: Optional[Any] = None,
    session_id: Optional[str] = None,
    caller_phone: Optional[str] = None,
) -> ReceptionistAgentGraph:
    """Get ReceptionistAgent using LangGraph."""
    return ReceptionistAgentGraph(
        business_context=business_context,
        session_id=session_id,
        caller_phone=caller_phone,
    )
