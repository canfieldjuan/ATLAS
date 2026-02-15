"""
Escalation evaluator for edge security events.

Classifies events as routine or escalation-worthy using rule-based logic
(no LLM). For escalations, synthesizes a TTS-ready alert via LLM + skill.
"""

import asyncio
import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional

logger = logging.getLogger("atlas.escalation")


@dataclass
class EscalationResult:
    """Result of escalation evaluation."""

    should_escalate: bool
    rule_name: Optional[str] = None
    priority: str = "routine"  # "routine" | "medium" | "high"
    context: dict[str, Any] = field(default_factory=dict)


class EscalationEvaluator:
    """
    Evaluates security events against escalation rules.

    Stateful -- maintains a sliding window of recent unknown face timestamps
    for rapid-unknowns detection.

    Rules:
        1. unknown_face + house EMPTY -> high escalation
        2. 3+ unknown_face in 60s -> high escalation (even if occupied)
        Everything else -> routine
    """

    def __init__(self) -> None:
        # maxlen caps memory under adversarial rapid-fire events
        self._unknown_timestamps: deque[float] = deque(maxlen=50)

    async def evaluate(
        self,
        event_type: str,
        message: dict[str, Any],
        node_id: str,
    ) -> EscalationResult:
        """
        Classify an event as routine or escalation.

        Pure logic, no LLM. Fast enough for inline use.
        """
        from ..config import settings

        config = settings.escalation
        if not config.enabled:
            return EscalationResult(should_escalate=False, priority="routine")

        # Only unknown_face and person_entered with is_known=False can escalate
        is_unknown = (
            event_type == "unknown_face"
            or (event_type == "person_entered" and not message.get("is_known", False))
        )

        if not is_unknown:
            return EscalationResult(should_escalate=False, priority="routine")

        # Get current occupancy from PresenceTracker
        occupancy_state = "unknown"
        occupants: list[str] = []
        try:
            from ..autonomous.presence import get_presence_tracker

            tracker = get_presence_tracker()
            occupancy_state = tracker.state.state.value
            occupants = list(tracker.state.occupants.keys())
        except Exception as e:
            logger.warning("Could not get occupancy state: %s", e)

        now = time.monotonic()

        # Track unknown face timestamps for rapid detection
        self._unknown_timestamps.append(now)
        # Prune expired entries
        window = config.rapid_unknowns_window_seconds
        while self._unknown_timestamps and (now - self._unknown_timestamps[0]) > window:
            self._unknown_timestamps.popleft()

        context = {
            "event_type": event_type,
            "node_id": node_id,
            "occupancy_state": occupancy_state,
            "occupants": occupants,
            "timestamp": time.time(),
            "message": {k: v for k, v in message.items() if k != "type"},
        }

        # Rule 1: unknown face + house EMPTY -> high escalation
        if config.unknown_empty_enabled and occupancy_state == "empty":
            logger.warning(
                "ESCALATION: unknown face while house empty (node=%s)", node_id,
            )
            self._unknown_timestamps.clear()
            return EscalationResult(
                should_escalate=True,
                rule_name="unknown_face_empty_house",
                priority="high",
                context=context,
            )

        # Rule 2: rapid unknowns (threshold+ in window) -> high escalation
        if len(self._unknown_timestamps) >= config.rapid_unknowns_threshold:
            logger.warning(
                "ESCALATION: %d unknown faces in %ds window (node=%s)",
                len(self._unknown_timestamps), window, node_id,
            )
            context["rapid_count"] = len(self._unknown_timestamps)
            context["window_seconds"] = window
            self._unknown_timestamps.clear()
            return EscalationResult(
                should_escalate=True,
                rule_name="rapid_unknown_faces",
                priority="high",
                context=context,
            )

        return EscalationResult(should_escalate=False, priority="routine")

    async def synthesize_and_send(
        self,
        result: EscalationResult,
        connection: Any,
    ) -> None:
        """
        LLM synthesis + send escalation_alert to edge.

        Called as a tracked background task via connection._spawn_task().
        """
        from ..config import settings

        config = settings.escalation
        text: Optional[str] = None

        try:
            # Lazy imports (same pattern as runner._synthesize_with_skill)
            from ..skills import get_skill_registry
            from ..services import llm_registry
            from ..services.protocols import Message
            from ..orchestration import cuda_lock

            skill = get_skill_registry().get(config.synthesis_skill)
            if skill is None:
                logger.warning(
                    "Escalation skill '%s' not found, sending raw alert",
                    config.synthesis_skill,
                )
                text = f"Security alert: {result.rule_name}"
            else:
                llm = llm_registry.get_active()
                if llm is None:
                    logger.warning("No active LLM for escalation synthesis")
                    text = f"Security alert: {result.rule_name}"
                else:
                    messages = [
                        Message(role="system", content=skill.content),
                        Message(
                            role="user",
                            content=json.dumps(result.context, indent=2, default=str),
                        ),
                    ]

                    # Prefer async chat (API backends) -- no cuda_lock needed,
                    # avoids blocking behind other LLM calls. Fall back to
                    # sync chat + cuda_lock for local GPU backends.
                    if hasattr(llm, "chat_async"):
                        raw = await llm.chat_async(
                            messages=messages,
                            max_tokens=config.synthesis_max_tokens,
                            temperature=config.synthesis_temperature,
                        )
                        text = raw.strip() if isinstance(raw, str) else raw.get("response", "").strip()
                    else:
                        async with cuda_lock:
                            llm_result = await asyncio.to_thread(
                                partial(
                                    llm.chat,
                                    messages=messages,
                                    max_tokens=config.synthesis_max_tokens,
                                    temperature=config.synthesis_temperature,
                                )
                            )
                        text = llm_result.get("response", "").strip()
                    # Strip <think> tags (Qwen3 models)
                    if text:
                        text = re.sub(
                            r"<think>.*?</think>", "", text, flags=re.DOTALL,
                        ).strip()

                    if not text:
                        text = f"Security alert: {result.rule_name}"

        except Exception:
            logger.exception("Escalation synthesis failed for rule '%s'", result.rule_name)
            text = f"Security alert: {result.rule_name}"

        # Get current occupancy for the alert payload
        try:
            from ..autonomous.presence import get_presence_tracker

            tracker = get_presence_tracker()
            occupancy_context = {
                "state": tracker.state.state.value,
                "occupants": list(tracker.state.occupants.keys()),
            }
        except Exception:
            occupancy_context = {"state": "unknown", "occupants": []}

        alert_msg = {
            "type": "escalation_alert",
            "priority": result.priority,
            "rule": result.rule_name,
            "text": text,
            "occupancy_context": occupancy_context,
        }

        try:
            await connection.send(alert_msg)
            logger.info(
                "Sent escalation_alert to edge: rule=%s, priority=%s",
                result.rule_name, result.priority,
            )
        except Exception:
            logger.exception("Failed to send escalation_alert to edge")
