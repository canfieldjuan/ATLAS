"""
Receptionist Agent - Business phone call handler.

Separate from AtlasAgent to avoid conflicts with home assistant duties.
Handles appointment booking, service inquiries, and message taking.

Design Philosophy:
- 90-95% of callers want to book an estimate
- Calls last 3-4 minutes on average
- Never quote prices - always determined at estimate
- Every home is different - emphasize free in-person estimate
- Keep it simple: GREETING → COLLECT_INFO → CONFIRM → DONE
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from .base import BaseAgent, Timer
from .protocols import (
    ActResult,
    AgentContext,
    ThinkResult,
)

logger = logging.getLogger("atlas.agents.receptionist")

# CUDA lock shared with other agents
_cuda_lock: Optional[asyncio.Lock] = None


def _get_cuda_lock() -> asyncio.Lock:
    """Get or create global CUDA lock."""
    global _cuda_lock
    if _cuda_lock is None:
        _cuda_lock = asyncio.Lock()
    return _cuda_lock


class ConversationPhase(Enum):
    """Simple conversation phases - NOT a complex state machine."""
    GREETING = auto()       # Just answered, detecting intent
    ANSWERING = auto()      # Answering questions (services, pricing, how it works)
    COLLECTING = auto()     # Gathering info for estimate (name, address, availability)
    CONFIRMING = auto()     # Confirming appointment details
    COMPLETE = auto()       # Booked or message taken
    TRANSFER = auto()       # Needs human


@dataclass
class CallContext:
    """Context accumulated during a phone call."""
    phase: ConversationPhase = ConversationPhase.GREETING

    # Caller info
    caller_name: Optional[str] = None
    caller_phone: Optional[str] = None  # From caller ID

    # Estimate booking
    service_address: Optional[str] = None
    preferred_date: Optional[str] = None
    preferred_time: Optional[str] = None

    # Tracking
    turns: int = 0
    intent_detected: Optional[str] = None  # "estimate", "question", "message"

    def summary(self) -> str:
        """Summary of collected info."""
        parts = []
        if self.caller_name:
            parts.append(f"Name: {self.caller_name}")
        if self.service_address:
            parts.append(f"Address: {self.service_address}")
        if self.preferred_date:
            parts.append(f"Date: {self.preferred_date}")
        if self.preferred_time:
            parts.append(f"Time: {self.preferred_time}")
        return "; ".join(parts) if parts else "No info collected yet"


class ReceptionistAgent(BaseAgent):
    """
    Business receptionist agent for phone calls.

    Optimized for cleaning business workflow:
    - 90-95% of callers want to book a FREE ESTIMATE
    - Never quote prices over phone - every home is different
    - Calls typically last 3-4 minutes
    - Focus on getting: name, address, and preferred time

    This agent is SEPARATE from AtlasAgent and does not
    handle home automation, weather, or personal queries.
    """

    def __init__(
        self,
        business_context: Optional[Any] = None,
        session_id: Optional[str] = None,
    ):
        """
        Initialize receptionist agent.

        Args:
            business_context: BusinessContext with persona, services, etc.
            session_id: Call/session identifier
        """
        super().__init__(
            name="receptionist",
            description="Business phone receptionist agent",
        )

        self._capabilities = [
            "estimate_booking",
            "service_inquiry",
            "message_taking",
        ]

        self._session_id = session_id
        self._business_context = business_context
        self._call_context = CallContext()
        self._llm = None
        self._phone_tools = None

    @property
    def business_context(self) -> Optional[Any]:
        """Get current business context."""
        return self._business_context

    @business_context.setter
    def business_context(self, value: Any) -> None:
        """Set business context."""
        self._business_context = value

    def _get_llm(self):
        """Get or create LLM service."""
        if self._llm is None:
            from ..services import llm_registry
            self._llm = llm_registry.get_active()
        return self._llm

    def _get_tools(self):
        """Get main tool registry."""
        if self._phone_tools is None:
            from ..tools import tool_registry
            self._phone_tools = tool_registry
        return self._phone_tools

    def _build_system_prompt(self) -> str:
        """Build system prompt optimized for estimate booking."""
        ctx = self._business_context
        call_ctx = self._call_context

        if ctx is None:
            return (
                "You are a professional virtual receptionist for a cleaning company. "
                "Most callers want to schedule a FREE estimate. "
                "Get their name, address, and preferred day/time. "
                "Never quote prices - every home is different."
            )

        # Core prompt focused on estimate booking
        prompt = f"""You are {getattr(ctx, 'voice_name', 'the virtual receptionist')} for {ctx.name}.

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
- "It depends on the size and condition - we'd need to see it. The estimate is free and takes about 15 minutes."
Then offer: "Want me to get you on the schedule?"

### Just Looking / Not Ready
Be friendly, don't push. Say: "No problem! Feel free to call back when you're ready. We're happy to help."

## CRITICAL RULES
- NEVER quote specific prices or ranges
- Keep responses SHORT (1-2 sentences)
- Be warm but efficient
- Always pivot back to offering the free estimate

## CURRENT CALL STATUS
Phase: {call_ctx.phase.name}
Info collected: {call_ctx.summary()}
"""

        # Add service area if available
        if hasattr(ctx, "service_area") and ctx.service_area:
            prompt += f"\nService area: {ctx.service_area}"

        return prompt

    async def _do_think(
        self,
        context: AgentContext,
    ) -> ThinkResult:
        """
        Analyze caller input and decide action.

        Simple phase tracking:
        - GREETING: Detect intent (estimate, question, message)
        - COLLECTING: Extract name, address, time from responses
        - CONFIRMING: Validate details before booking
        """
        result = ThinkResult(
            action_type="conversation",
            confidence=0.8,
            needs_llm=True,
        )

        query_lower = context.input_text.lower()
        self._call_context.turns += 1

        # Extract info from caller's response (runs every turn)
        self._extract_caller_info(query_lower, context.input_text)

        # Phase-based logic
        phase = self._call_context.phase

        if phase == ConversationPhase.GREETING:
            # First turn - detect what they want
            if self._wants_estimate(query_lower):
                self._call_context.intent_detected = "estimate"
                self._call_context.phase = ConversationPhase.COLLECTING
                result.action_type = "tool_use"
                result.tools_to_call = ["check_availability"]
                logger.info("Intent: estimate booking")
            elif self._wants_message(query_lower):
                self._call_context.intent_detected = "message"
                result.action_type = "tool_use"
                result.tools_to_call = ["send_notification"]
                logger.info("Intent: leave message")
            elif self._is_asking_questions(query_lower):
                # Info seeker or price shopper - LLM answers from system prompt
                self._call_context.intent_detected = "question"
                self._call_context.phase = ConversationPhase.ANSWERING
                # No tool needed - business info is in system prompt
                logger.info("Intent: asking questions (info seeker)")
            else:
                # Default assumption: they probably want an estimate
                self._call_context.intent_detected = "estimate"
                self._call_context.phase = ConversationPhase.COLLECTING
                logger.info("Intent: assumed estimate (default)")

        elif phase == ConversationPhase.ANSWERING:
            # They asked questions - check if now ready to book
            if self._wants_estimate(query_lower):
                self._call_context.intent_detected = "estimate"
                self._call_context.phase = ConversationPhase.COLLECTING
                result.action_type = "tool_use"
                result.tools_to_call = ["check_availability"]
                logger.info("Question phase → now wants estimate")
            elif self._is_asking_questions(query_lower):
                # Still asking questions - LLM answers from system prompt
                logger.debug("Still in question phase")
            elif self._is_not_interested(query_lower):
                # They're just shopping, be polite
                logger.info("Caller not interested, wrapping up")
            # Otherwise LLM will answer and offer estimate

        elif phase == ConversationPhase.COLLECTING:
            # Check if we have enough info to confirm
            if self._has_enough_info():
                self._call_context.phase = ConversationPhase.CONFIRMING
                logger.info("Info collected, moving to confirmation")
            # Otherwise LLM will ask for missing info

        elif phase == ConversationPhase.CONFIRMING:
            # Check for confirmation or changes
            if self._is_confirmation(query_lower):
                result.action_type = "tool_use"
                result.tools_to_call = ["book_appointment"]
                self._call_context.phase = ConversationPhase.COMPLETE
                logger.info("Booking confirmed")
            elif self._is_rejection(query_lower):
                self._call_context.phase = ConversationPhase.COLLECTING
                logger.info("Changes requested, back to collecting")

        return result

    def _wants_estimate(self, text: str) -> bool:
        """Check if caller wants to book an estimate."""
        keywords = [
            "estimate", "appointment", "schedule", "book",
            "available", "come out", "set up", "cleaning",
            "quote", "free estimate", "get started",
        ]
        return any(kw in text for kw in keywords)

    def _wants_message(self, text: str) -> bool:
        """Check if caller wants to leave a message."""
        keywords = [
            "message", "call back", "callback", "leave",
            "tell them", "have them call", "speak to someone",
        ]
        return any(kw in text for kw in keywords)

    def _is_asking_questions(self, text: str) -> bool:
        """Check if caller is asking about services/pricing (info seeker)."""
        # Price shoppers
        price_keywords = [
            "how much", "price", "cost", "rate", "charge",
            "expensive", "affordable", "ballpark", "range",
            "per hour", "per square", "minimum",
        ]
        # Info seekers
        info_keywords = [
            "what services", "do you offer", "do you do",
            "what kind", "how does it work", "what's included",
            "what areas", "where do you", "how long",
            "tell me about", "more information",
        ]
        return any(kw in text for kw in price_keywords + info_keywords)

    def _is_not_interested(self, text: str) -> bool:
        """Check if caller is just shopping / not interested."""
        keywords = [
            "just looking", "just checking", "not right now",
            "maybe later", "think about it", "get back to you",
            "too expensive", "can't afford", "out of my budget",
            "no thanks", "not interested",
        ]
        return any(kw in text for kw in keywords)

    def _extract_caller_info(self, text_lower: str, text_original: str) -> None:
        """Extract name, address, time preferences from caller's response."""
        # Name detection (simple heuristics)
        name_triggers = ["my name is", "this is", "i'm ", "i am "]
        for trigger in name_triggers:
            if trigger in text_lower:
                idx = text_lower.find(trigger) + len(trigger)
                remaining = text_original[idx:].strip()
                # Take first 2-3 words as name
                words = remaining.split()[:3]
                if words:
                    name = " ".join(words).rstrip(".,!?")
                    if len(name) > 1:
                        self._call_context.caller_name = name
                        logger.debug("Extracted name: %s", name)
                        break

        # Address detection (looks for numbers + street patterns)
        import re
        address_pattern = r'\d+\s+[\w\s]+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|way|court|ct|boulevard|blvd)'
        match = re.search(address_pattern, text_lower)
        if match:
            self._call_context.service_address = text_original[match.start():match.end()]
            logger.debug("Extracted address: %s", self._call_context.service_address)

        # Time preference detection
        time_keywords = {
            "morning": "morning",
            "afternoon": "afternoon",
            "evening": "evening",
            "monday": "Monday", "tuesday": "Tuesday", "wednesday": "Wednesday",
            "thursday": "Thursday", "friday": "Friday", "saturday": "Saturday",
            "tomorrow": "tomorrow", "next week": "next week",
            "this week": "this week",
        }
        for kw, value in time_keywords.items():
            if kw in text_lower:
                if not self._call_context.preferred_date:
                    self._call_context.preferred_date = value
                elif not self._call_context.preferred_time:
                    self._call_context.preferred_time = value
                logger.debug("Extracted time pref: %s", value)

    def _has_enough_info(self) -> bool:
        """Check if we have minimum info to book."""
        ctx = self._call_context
        # Need at least name and either address or time
        has_name = ctx.caller_name is not None
        has_address = ctx.service_address is not None
        has_time = ctx.preferred_date is not None or ctx.preferred_time is not None
        return has_name and (has_address or has_time)

    def _is_confirmation(self, text: str) -> bool:
        """Check if caller is confirming."""
        keywords = ["yes", "yeah", "yep", "correct", "that's right", "sounds good", "perfect", "ok", "okay"]
        return any(kw in text for kw in keywords)

    def _is_rejection(self, text: str) -> bool:
        """Check if caller wants changes."""
        keywords = ["no", "change", "different", "actually", "wait", "not right"]
        return any(kw in text for kw in keywords)

    async def _do_act(
        self,
        context: AgentContext,
        think_result: ThinkResult,
    ) -> ActResult:
        """Execute phone tools with collected call context."""
        result = ActResult(
            success=True,
            action_type=think_result.action_type,
        )

        if think_result.action_type != "tool_use":
            return result

        start_time = time.perf_counter()
        tools = self._get_tools()
        call_ctx = self._call_context

        for tool_name in think_result.tools_to_call:
            try:
                # Build params from call context
                params = {
                    "query": context.input_text,
                    "caller_id": context.speaker_id or call_ctx.caller_phone,
                }

                if self._business_context:
                    params["context_id"] = self._business_context.id

                # Handle send_notification (callback request)
                if tool_name == "send_notification":
                    business_name = ""
                    if self._business_context:
                        business_name = self._business_context.name
                    msg = (
                        f"Callback request from {call_ctx.caller_name or 'Unknown'} "
                        f"({call_ctx.caller_phone or 'no phone'})"
                    )
                    params.update({
                        "message": msg,
                        "title": f"Callback - {business_name}",
                        "priority": "high",
                    })

                # Add collected info for booking
                elif tool_name == "book_appointment":
                    from datetime import datetime, timedelta

                    # Default to tomorrow if no date specified
                    if call_ctx.preferred_date:
                        if call_ctx.preferred_date == "tomorrow":
                            book_date = datetime.now() + timedelta(days=1)
                        elif call_ctx.preferred_date == "next week":
                            book_date = datetime.now() + timedelta(days=7)
                        else:
                            book_date = datetime.now() + timedelta(days=1)
                    else:
                        book_date = datetime.now() + timedelta(days=1)

                    # Default time based on preference
                    if call_ctx.preferred_time == "morning":
                        book_time = "09:00"
                    elif call_ctx.preferred_time == "afternoon":
                        book_time = "14:00"
                    elif call_ctx.preferred_time == "evening":
                        book_time = "17:00"
                    else:
                        book_time = "10:00"

                    params.update({
                        "date": book_date.strftime("%Y-%m-%d"),
                        "time": book_time,
                        "customer_name": call_ctx.caller_name or "Customer",
                        "customer_phone": call_ctx.caller_phone or "",
                        "service_type": "Free Estimate",
                        "address": call_ctx.service_address or "",
                    })

                tool_result = await tools.execute(tool_name, params)
                result.tool_results[tool_name] = {
                    "success": tool_result.success,
                    "data": tool_result.data,
                    "message": tool_result.message,
                }

                if tool_result.success and tool_result.message:
                    result.response_data["tool_context"] = (
                        f"[{tool_name}] {tool_result.message}"
                    )

            except Exception as e:
                logger.warning("Tool %s failed: %s", tool_name, e)
                result.tool_results[tool_name] = {
                    "success": False,
                    "error": str(e),
                }

        result.duration_ms = (time.perf_counter() - start_time) * 1000
        return result

    async def _do_respond(
        self,
        context: AgentContext,
        think_result: ThinkResult,
        act_result: Optional[ActResult],
    ) -> str:
        """Generate response using LLM with business context."""
        from ..services.protocols import Message

        llm = self._get_llm()
        if llm is None:
            return "Thank you for calling. How can I help you?"

        # Build messages
        system_prompt = self._build_system_prompt()

        # Add tool context if available
        if act_result and act_result.response_data.get("tool_context"):
            system_prompt += f"\n\nDATA:\n{act_result.response_data['tool_context']}"

        messages = [Message(role="system", content=system_prompt)]

        # Add conversation history
        if context.conversation_history:
            for turn in context.conversation_history[-6:]:
                messages.append(Message(
                    role=turn.get("role", "user"),
                    content=turn.get("content", ""),
                ))

        # Add current message
        messages.append(Message(role="user", content=context.input_text))

        # Call LLM
        try:
            cuda_lock = _get_cuda_lock()
            async with cuda_lock:
                llm_result = llm.chat(
                    messages=messages,
                    max_tokens=150,
                    temperature=0.7,
                )
            response = llm_result.get("response", "").strip()
            if response:
                return response
        except Exception as e:
            logger.warning("LLM response failed: %s", e)

        return "Thank you for calling. How can I help you today?"

    def reset_call(self, caller_phone: Optional[str] = None) -> None:
        """Reset call context for a new call."""
        self._call_context = CallContext()
        if caller_phone:
            self._call_context.caller_phone = caller_phone
        logger.info("Call context reset for new call")

    @property
    def call_context(self) -> CallContext:
        """Get current call context."""
        return self._call_context


# Factory functions

_receptionist_agent: Optional[ReceptionistAgent] = None


def get_receptionist_agent(
    business_context: Optional[Any] = None,
    session_id: Optional[str] = None,
    new_call: bool = False,
) -> ReceptionistAgent:
    """
    Get or create receptionist agent instance.

    Args:
        business_context: Business context for this call
        session_id: Call/session identifier
        new_call: If True, reset call context for a fresh conversation
    """
    global _receptionist_agent
    if _receptionist_agent is None:
        _receptionist_agent = ReceptionistAgent(
            business_context=business_context,
            session_id=session_id,
        )
    else:
        if business_context:
            _receptionist_agent.business_context = business_context
        if session_id:
            _receptionist_agent._session_id = session_id
        if new_call:
            _receptionist_agent.reset_call()
    return _receptionist_agent


def create_receptionist_agent(
    business_context: Optional[Any] = None,
    session_id: Optional[str] = None,
    caller_phone: Optional[str] = None,
) -> ReceptionistAgent:
    """
    Create a new receptionist agent instance for a phone call.

    Args:
        business_context: Business context (persona, services, etc.)
        session_id: Call/session identifier
        caller_phone: Caller ID phone number
    """
    agent = ReceptionistAgent(
        business_context=business_context,
        session_id=session_id,
    )
    if caller_phone:
        agent._call_context.caller_phone = caller_phone
    return agent


def reset_receptionist_agent() -> None:
    """Reset global receptionist agent."""
    global _receptionist_agent
    _receptionist_agent = None
