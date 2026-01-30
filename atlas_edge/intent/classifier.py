"""
Edge Intent Classifier using DistilBERT.

Binary classification: handle_locally (device_command) vs escalate (conversation/tool).
Optimized for Jetson edge devices.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("atlas.edge.intent.classifier")


@dataclass
class ClassificationResult:
    """Result from intent classification."""

    action_category: str  # "device_command", "conversation", "tool_use"
    raw_label: str  # Original model label
    confidence: float
    route_time_ms: float = 0.0
    should_escalate: bool = False  # True if should send to brain


# Map model labels to action categories
LABEL_TO_CATEGORY = {
    # Device commands (handle locally)
    "iot_hue_lighton": ("device_command", False),
    "iot_hue_lightoff": ("device_command", False),
    "iot_hue_lightdim": ("device_command", False),
    "iot_hue_lightup": ("device_command", False),
    "iot_hue_lightchange": ("device_command", False),
    "iot_wemo_on": ("device_command", False),
    "iot_wemo_off": ("device_command", False),
    "iot_cleaning": ("device_command", False),
    "iot_coffee": ("device_command", False),
    "audio_volume_up": ("device_command", False),
    "audio_volume_down": ("device_command", False),
    "audio_volume_mute": ("device_command", False),
    "audio_volume_other": ("device_command", False),
    # Conversations (escalate to brain)
    "general_greet": ("conversation", True),
    "general_joke": ("conversation", True),
    "general_quirky": ("conversation", True),
    "qa_factoid": ("conversation", True),
    "qa_definition": ("conversation", True),
    "qa_maths": ("conversation", True),
    # Tool queries (escalate to brain)
    "datetime_query": ("tool_use", True),
    "weather_query": ("tool_use", True),
    "calendar_query": ("tool_use", True),
    "calendar_set": ("tool_use", True),
    "alarm_set": ("tool_use", True),
    "alarm_query": ("tool_use", True),
    "transport_traffic": ("tool_use", True),
    "email_sendemail": ("tool_use", True),
}


class EdgeIntentClassifier:
    """
    Fast intent classifier for edge devices.

    Uses DistilBERT to quickly determine if a query should be
    handled locally (device commands) or escalated to the brain
    (conversations, tools requiring reasoning).
    """

    def __init__(
        self,
        model_id: str = "qanastek/XLMRoberta-Alexa-Intents-Classification",
        device: str = "cuda",
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize the intent classifier.

        Args:
            model_id: HuggingFace model ID
            device: Device for inference (cuda, cpu, auto)
            confidence_threshold: Minimum confidence for local handling
        """
        self._model_id = model_id
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._classifier = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def _get_device_index(self) -> int:
        """Get device index for pipeline."""
        if self._device == "cpu":
            return -1
        if self._device == "cuda":
            return 0
        # Auto-detect
        try:
            import torch

            return 0 if torch.cuda.is_available() else -1
        except ImportError:
            return -1

    async def load(self) -> None:
        """Load the classification model."""
        if self._loaded:
            return

        logger.info("Loading intent classifier: %s", self._model_id)
        start = time.time()

        loop = asyncio.get_event_loop()

        def _load_model():
            from transformers import pipeline

            return pipeline(
                "text-classification",
                model=self._model_id,
                device=self._get_device_index(),
            )

        self._classifier = await loop.run_in_executor(None, _load_model)
        self._loaded = True

        elapsed = time.time() - start
        device_str = "cuda" if self._get_device_index() >= 0 else "cpu"
        logger.info("Intent classifier loaded in %.2fs on %s", elapsed, device_str)

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._classifier is not None:
            del self._classifier
            self._classifier = None
            self._loaded = False

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("Intent classifier unloaded")

    async def classify(self, query: str) -> ClassificationResult:
        """
        Classify a query into an action category.

        Args:
            query: User query text

        Returns:
            ClassificationResult with category and escalation decision
        """
        if not self._loaded:
            await self.load()

        start = time.time()

        # Run classification in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self._classifier(query)
        )

        route_time = (time.time() - start) * 1000

        # Extract label and score
        raw_label = result[0]["label"]
        confidence = result[0]["score"]

        # Map to our categories
        category, should_escalate = LABEL_TO_CATEGORY.get(
            raw_label,
            ("conversation", True),  # Default: escalate unknown
        )

        # Low confidence -> escalate
        if confidence < self._confidence_threshold:
            should_escalate = True
            logger.debug(
                "Low confidence %.2f for '%s', will escalate",
                confidence,
                query[:30],
            )

        logger.info(
            "Classified: '%s' -> %s (conf=%.2f, escalate=%s, %.0fms)",
            query[:30],
            category,
            confidence,
            should_escalate,
            route_time,
        )

        return ClassificationResult(
            action_category=category,
            raw_label=raw_label,
            confidence=confidence,
            route_time_ms=route_time,
            should_escalate=should_escalate,
        )

    def should_handle_locally(self, result: ClassificationResult) -> bool:
        """
        Determine if query should be handled locally.

        Args:
            result: Classification result

        Returns:
            True if should handle locally, False to escalate
        """
        return (
            result.action_category == "device_command"
            and result.confidence >= self._confidence_threshold
            and not result.should_escalate
        )


# Singleton instance
_classifier: Optional[EdgeIntentClassifier] = None


def get_intent_classifier() -> EdgeIntentClassifier:
    """Get or create global intent classifier."""
    global _classifier
    if _classifier is None:
        from ..config import settings

        _classifier = EdgeIntentClassifier(
            model_id=settings.intent.model_id,
            device=settings.intent.device,
            confidence_threshold=settings.intent.confidence_threshold,
        )
    return _classifier
