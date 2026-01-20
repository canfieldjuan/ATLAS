"""
Intent Router for fast query classification.

Uses a DistilBERT model fine-tuned on the MASSIVE dataset to quickly
classify user queries into action categories (device, tool, conversation).
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

from ..config import settings

logger = logging.getLogger("atlas.services.intent_router")


@dataclass
class IntentRouteResult:
    """Result from intent routing."""

    action_category: str  # "device_command", "tool_use", "conversation"
    raw_label: str  # Original model label (e.g., "iot_hue_lighton")
    confidence: float
    route_time_ms: float = 0.0
    tool_name: Optional[str] = None  # Mapped tool name if applicable
    fast_path_ok: bool = False  # True if tool can execute without params


# Tools that can execute without parameters (fast path OK)
PARAMETERLESS_TOOLS = {
    "get_time",
    "get_weather",
    "get_calendar",
    "list_reminders",
    "get_traffic",
    "get_location",
}


# Map MASSIVE labels to our action categories and tool names
LABEL_TO_CATEGORY = {
    # Device commands (IoT)
    "iot_hue_lighton": ("device_command", None),
    "iot_hue_lightoff": ("device_command", None),
    "iot_hue_lightdim": ("device_command", None),
    "iot_hue_lightup": ("device_command", None),
    "iot_hue_lightchange": ("device_command", None),
    "iot_wemo_on": ("device_command", None),
    "iot_wemo_off": ("device_command", None),
    "iot_cleaning": ("device_command", None),
    "iot_coffee": ("device_command", None),
    # Audio/Volume (device commands)
    "audio_volume_up": ("device_command", None),
    "audio_volume_down": ("device_command", None),
    "audio_volume_mute": ("device_command", None),
    "audio_volume_other": ("device_command", None),
    # Tool queries - time
    "datetime_query": ("tool_use", "get_time"),
    "datetime_convert": ("tool_use", "get_time"),
    # Tool queries - weather
    "weather_query": ("tool_use", "get_weather"),
    # Tool queries - calendar/reminders
    "calendar_query": ("tool_use", "get_calendar"),
    "calendar_set": ("tool_use", "set_reminder"),
    "calendar_remove": ("tool_use", "get_calendar"),
    "alarm_set": ("tool_use", "set_reminder"),
    "alarm_query": ("tool_use", "list_reminders"),
    "alarm_remove": ("tool_use", "complete_reminder"),
    # Tool queries - traffic
    "transport_traffic": ("tool_use", "get_traffic"),
    # Tool queries - lists (map to reminders)
    "lists_createoradd": ("tool_use", "set_reminder"),
    "lists_query": ("tool_use", "list_reminders"),
    "lists_remove": ("tool_use", "complete_reminder"),
    # Conversation/General
    "general_greet": ("conversation", None),
    "general_joke": ("conversation", None),
    "general_quirky": ("conversation", None),
    # QA queries (conversation - needs LLM)
    "qa_factoid": ("conversation", None),
    "qa_definition": ("conversation", None),
    "qa_maths": ("conversation", None),
    "qa_currency": ("conversation", None),
    "qa_stock": ("conversation", None),
    # Media (conversation for now - could be device later)
    "play_music": ("conversation", None),
    "play_radio": ("conversation", None),
    "play_podcasts": ("conversation", None),
    "play_audiobook": ("conversation", None),
    "play_game": ("conversation", None),
    "music_query": ("conversation", None),
    "music_likeness": ("conversation", None),
    "music_dislikeness": ("conversation", None),
    "music_settings": ("conversation", None),
    # News/Social (conversation)
    "news_query": ("conversation", None),
    "social_query": ("conversation", None),
    "social_post": ("conversation", None),
    # Email (tool_use for send, conversation for others)
    "email_query": ("conversation", None),
    "email_sendemail": ("tool_use", "send_email"),
    "email_addcontact": ("conversation", None),
    "email_querycontact": ("conversation", None),
    # Recommendations (conversation)
    "recommendation_events": ("conversation", None),
    "recommendation_locations": ("conversation", None),
    "recommendation_movies": ("conversation", None),
    # Transport (conversation - no tool yet)
    "transport_query": ("conversation", None),
    "transport_taxi": ("conversation", None),
    "transport_ticket": ("conversation", None),
    # Food (conversation)
    "takeaway_query": ("conversation", None),
    "takeaway_order": ("conversation", None),
    "cooking_query": ("conversation", None),
    "cooking_recipe": ("conversation", None),
}


class IntentRouter:
    """
    Fast intent router using DistilBERT for query classification.

    Classifies queries into: device_command, tool_use, or conversation.
    """

    def __init__(self) -> None:
        """Initialize router (model loaded lazily on first use)."""
        self._classifier = None
        self._config = settings.intent_router

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._classifier is not None

    def _get_device(self) -> int:
        """Get device index for inference."""
        if self._config.device == "cpu":
            return -1
        if self._config.device == "cuda":
            return 0
        # Auto-detect
        try:
            import torch
            return 0 if torch.cuda.is_available() else -1
        except ImportError:
            return -1

    async def load(self) -> None:
        """Load the classification model."""
        if self._classifier is not None:
            logger.info("Intent router already loaded")
            return

        logger.info("Loading intent router model: %s", self._config.model_id)
        start = time.time()

        loop = asyncio.get_event_loop()

        def _load_model():
            from transformers import pipeline
            return pipeline(
                "text-classification",
                model=self._config.model_id,
                device=self._get_device(),
            )

        self._classifier = await loop.run_in_executor(None, _load_model)
        elapsed = time.time() - start
        device_str = "cuda" if self._get_device() >= 0 else "cpu"
        logger.info("Intent router loaded in %.2fs on %s", elapsed, device_str)

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._classifier is not None:
            del self._classifier
            self._classifier = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("Intent router unloaded")

    async def route(self, query: str) -> IntentRouteResult:
        """
        Classify a query into an action category.

        Args:
            query: User query text

        Returns:
            IntentRouteResult with category, label, and confidence
        """
        if not self._config.enabled:
            # Disabled - return conversation as fallback
            return IntentRouteResult(
                action_category="conversation",
                raw_label="disabled",
                confidence=0.0,
            )

        if self._classifier is None:
            await self.load()

        start = time.time()

        # Run classification in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._classifier(query)
        )

        route_time = (time.time() - start) * 1000

        # Extract label and score
        raw_label = result[0]["label"]
        confidence = result[0]["score"]

        # Map to our categories
        category, tool_name = LABEL_TO_CATEGORY.get(
            raw_label,
            ("conversation", None),  # Default fallback
        )

        # If confidence is below threshold, fallback to conversation
        if confidence < self._config.confidence_threshold:
            logger.debug(
                "Low confidence %.2f for '%s' -> %s, falling back to conversation",
                confidence, query[:30], raw_label,
            )
            category = "conversation"
            tool_name = None

        logger.info(
            "Route: '%s' -> %s/%s (conf=%.2f, %.0fms)",
            query[:30], category, raw_label, confidence, route_time,
        )

        return IntentRouteResult(
            action_category=category,
            raw_label=raw_label,
            confidence=confidence,
            route_time_ms=route_time,
            tool_name=tool_name,
            fast_path_ok=tool_name in PARAMETERLESS_TOOLS if tool_name else False,
        )


# Module-level singleton
_router: Optional[IntentRouter] = None


def get_intent_router() -> IntentRouter:
    """Get or create the global intent router instance."""
    global _router
    if _router is None:
        _router = IntentRouter()
    return _router


async def route_query(query: str) -> IntentRouteResult:
    """Convenience function to route a query."""
    router = get_intent_router()
    return await router.route(query)
