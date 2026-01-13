"""
Streaming intent detector for early classification of partial transcripts.

Classifies partial transcripts to enable early warmup of appropriate models.
Works with incomplete text from streaming STT.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger("atlas.orchestration.streaming_intent")


class IntentCategory(Enum):
    """High-level intent categories for routing."""
    DEVICE_COMMAND = "device_command"
    CONVERSATION = "conversation"
    QUESTION = "question"
    GREETING = "greeting"
    UNKNOWN = "unknown"


@dataclass
class StreamingIntent:
    """Result from streaming intent classification."""
    category: IntentCategory
    confidence: float
    matched_pattern: Optional[str] = None
    partial_text: str = ""

    def should_warmup_llm(self) -> bool:
        """Check if we should start warming up the LLM."""
        return (
            self.category in (IntentCategory.CONVERSATION, IntentCategory.QUESTION)
            and self.confidence >= 0.85
        )


class StreamingIntentDetector:
    """
    Detects intent from partial transcripts during streaming STT.

    Uses lightweight pattern matching optimized for incomplete text.
    Goal: Early detection to enable model warmup before full transcript.
    """

    # Device command signal words (start of command)
    DEVICE_START_PATTERNS = [
        r"^turn\s",
        r"^switch\s",
        r"^toggle\s",
        r"^dim\s",
        r"^brighten\s",
        r"^set\s",
        r"^lights?\s",
        r"^kill\s",
        r"^shut\s",
        r"^power\s",
        r"^enable\s",
        r"^disable\s",
        r"^mute\s",
        r"^unmute\s",
        r"^pause\s",
        r"^play\s",
        r"^stop\s",
        r"^volume\s",
    ]

    # Device keywords that appear anywhere
    DEVICE_KEYWORDS = [
        "light", "lights", "lamp", "lamps",
        "tv", "television", "roku",
        "switch", "fan", "thermostat",
        "speaker", "speakers",
    ]

    # Question patterns (start of question)
    QUESTION_START_PATTERNS = [
        r"^what\s",
        r"^who\s",
        r"^when\s",
        r"^where\s",
        r"^why\s",
        r"^how\s",
        r"^can\s+you\s+(?:tell|explain|describe|help)",
        r"^do\s+you\s+know",
        r"^is\s+(?:there|it|this)",
        r"^are\s+(?:there|they|you)",
        r"^could\s+you\s+(?:tell|explain)",
        r"^would\s+you\s+(?:tell|explain)",
    ]

    # Conversation/chat patterns
    CONVERSATION_PATTERNS = [
        r"^tell\s+me\s+(?:about|a)",
        r"^i\s+(?:want|need|think|feel|am|was|have)",
        r"^let'?s\s+(?:talk|chat|discuss)",
        r"^can\s+we\s+(?:talk|chat|discuss)",
        r"^(?:explain|describe|elaborate)",
        r"^help\s+me\s+(?:understand|with)",
    ]

    # Greeting patterns
    GREETING_PATTERNS = [
        r"^(?:hey|hi|hello|good\s+(?:morning|afternoon|evening))",
        r"^(?:thanks|thank\s+you)",
        r"^(?:bye|goodbye|see\s+you)",
    ]

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for performance."""
        self._device_start_re = [
            re.compile(p, re.IGNORECASE) for p in self.DEVICE_START_PATTERNS
        ]
        self._device_keywords_re = re.compile(
            r"\b(" + "|".join(self.DEVICE_KEYWORDS) + r")\b",
            re.IGNORECASE
        )
        self._question_start_re = [
            re.compile(p, re.IGNORECASE) for p in self.QUESTION_START_PATTERNS
        ]
        self._conversation_re = [
            re.compile(p, re.IGNORECASE) for p in self.CONVERSATION_PATTERNS
        ]
        self._greeting_re = [
            re.compile(p, re.IGNORECASE) for p in self.GREETING_PATTERNS
        ]

    def classify(self, partial_text: str) -> StreamingIntent:
        """
        Classify a partial transcript into an intent category.

        Args:
            partial_text: Partial transcript from streaming STT

        Returns:
            StreamingIntent with category and confidence
        """
        text = partial_text.strip().lower()

        if not text:
            return StreamingIntent(
                category=IntentCategory.UNKNOWN,
                confidence=0.0,
                partial_text=partial_text,
            )

        # Check for device command signals
        device_result = self._check_device_command(text)
        if device_result:
            return device_result

        # Check for question patterns
        question_result = self._check_question(text)
        if question_result:
            return question_result

        # Check for conversation patterns
        conversation_result = self._check_conversation(text)
        if conversation_result:
            return conversation_result

        # Check for greetings
        greeting_result = self._check_greeting(text)
        if greeting_result:
            return greeting_result

        # Unknown - not enough signal yet
        return StreamingIntent(
            category=IntentCategory.UNKNOWN,
            confidence=0.0,
            partial_text=partial_text,
        )

    def _check_device_command(self, text: str) -> Optional[StreamingIntent]:
        """Check if text looks like a device command."""
        # Check start patterns (high confidence)
        for pattern in self._device_start_re:
            if pattern.match(text):
                return StreamingIntent(
                    category=IntentCategory.DEVICE_COMMAND,
                    confidence=0.90,
                    matched_pattern=pattern.pattern,
                    partial_text=text,
                )

        # Check for device keywords anywhere (medium confidence)
        if self._device_keywords_re.search(text):
            # Higher confidence if combined with action words
            action_words = ["on", "off", "up", "down", "dim", "bright"]
            has_action = any(word in text for word in action_words)
            confidence = 0.85 if has_action else 0.70

            return StreamingIntent(
                category=IntentCategory.DEVICE_COMMAND,
                confidence=confidence,
                matched_pattern="device_keyword",
                partial_text=text,
            )

        return None

    def _check_question(self, text: str) -> Optional[StreamingIntent]:
        """Check if text looks like a question."""
        for pattern in self._question_start_re:
            if pattern.match(text):
                # Higher confidence for longer text
                word_count = len(text.split())
                confidence = min(0.90, 0.70 + (word_count * 0.05))

                return StreamingIntent(
                    category=IntentCategory.QUESTION,
                    confidence=confidence,
                    matched_pattern=pattern.pattern,
                    partial_text=text,
                )

        # Check for question marks (high confidence)
        if "?" in text:
            return StreamingIntent(
                category=IntentCategory.QUESTION,
                confidence=0.95,
                matched_pattern="question_mark",
                partial_text=text,
            )

        return None

    def _check_conversation(self, text: str) -> Optional[StreamingIntent]:
        """Check if text looks like a conversation request."""
        for pattern in self._conversation_re:
            if pattern.match(text):
                word_count = len(text.split())
                confidence = min(0.90, 0.75 + (word_count * 0.03))

                return StreamingIntent(
                    category=IntentCategory.CONVERSATION,
                    confidence=confidence,
                    matched_pattern=pattern.pattern,
                    partial_text=text,
                )

        return None

    def _check_greeting(self, text: str) -> Optional[StreamingIntent]:
        """Check if text is a greeting."""
        for pattern in self._greeting_re:
            if pattern.match(text):
                return StreamingIntent(
                    category=IntentCategory.GREETING,
                    confidence=0.90,
                    matched_pattern=pattern.pattern,
                    partial_text=text,
                )

        return None


# Global detector instance
streaming_intent_detector = StreamingIntentDetector()
