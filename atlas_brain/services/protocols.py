"""
Protocol definitions for AI model services.

These protocols define the interface that all VLM and STT implementations must follow,
enabling runtime model swapping and consistent behavior across different backends.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class ModelInfo:
    """Metadata about a loaded model."""
    name: str
    model_id: str
    is_loaded: bool
    device: str
    capabilities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "is_loaded": self.is_loaded,
            "device": self.device,
            "capabilities": self.capabilities,
        }


@dataclass
class InferenceMetrics:
    """Standard metrics returned by all inference operations."""
    duration_ms: float
    device: str
    memory_allocated_mb: float = 0.0
    memory_reserved_mb: float = 0.0
    memory_total_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_ms": self.duration_ms,
            "device": self.device,
            "memory_allocated_mb": self.memory_allocated_mb,
            "memory_reserved_mb": self.memory_reserved_mb,
            "memory_total_mb": self.memory_total_mb,
        }


@runtime_checkable
class VLMService(Protocol):
    """Protocol for Vision-Language Model services."""

    @property
    def model_info(self) -> ModelInfo:
        """Return metadata about the current model."""
        ...

    def load(self) -> None:
        """Load the model into memory."""
        ...

    def unload(self) -> None:
        """Unload the model from memory to free resources."""
        ...

    def process_text(self, query: str) -> dict[str, Any]:
        """Process a text-only query."""
        ...

    async def process_vision(
        self,
        image_bytes: bytes,
        prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Process an image with optional text prompt."""
        ...


@runtime_checkable
class STTService(Protocol):
    """Protocol for Speech-to-Text services."""

    @property
    def model_info(self) -> ModelInfo:
        """Return metadata about the current model."""
        ...

    def load(self) -> None:
        """Load the model into memory."""
        ...

    def unload(self) -> None:
        """Unload the model from memory to free resources."""
        ...

    async def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> dict[str, Any]:
        """Transcribe audio bytes to text."""
        ...


@dataclass
class Message:
    """A chat message for LLM conversation."""
    role: str  # "system", "user", "assistant"
    content: str


@runtime_checkable
class LLMService(Protocol):
    """Protocol for Large Language Model (reasoning) services."""

    @property
    def model_info(self) -> ModelInfo:
        """Return metadata about the current model."""
        ...

    def load(self) -> None:
        """Load the model into memory."""
        ...

    def unload(self) -> None:
        """Unload the model from memory to free resources."""
        ...

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Generate text from a prompt."""
        ...

    def chat(
        self,
        messages: list[Message],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Generate a response in a chat conversation."""
        ...


@runtime_checkable
class TTSService(Protocol):
    """Protocol for Text-to-Speech services."""

    @property
    def model_info(self) -> ModelInfo:
        """Return metadata about the current model."""
        ...

    def load(self) -> None:
        """Load the model into memory."""
        ...

    def unload(self) -> None:
        """Unload the model from memory to free resources."""
        ...

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> bytes:
        """Convert text to speech audio (WAV format)."""
        ...


@dataclass
class AudioEvent:
    """A detected audio event."""

    label: str  # e.g., "Doorbell", "Dog", "Glass breaking"
    confidence: float  # 0.0 - 1.0
    class_id: int  # YAMNet class ID
    timestamp_ms: float = 0.0  # When in the audio chunk this occurred

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "timestamp_ms": self.timestamp_ms,
        }


@runtime_checkable
class AudioEventService(Protocol):
    """Protocol for Audio Event Detection services (YAMNet, etc.)."""

    @property
    def model_info(self) -> ModelInfo:
        """Return metadata about the current model."""
        ...

    def load(self) -> None:
        """Load the model into memory."""
        ...

    def unload(self) -> None:
        """Unload the model from memory to free resources."""
        ...

    def classify(
        self,
        audio_bytes: bytes,
        top_k: int = 5,
        min_confidence: float = 0.1,
    ) -> list[AudioEvent]:
        """
        Classify audio and return detected events.

        Args:
            audio_bytes: Raw audio (16-bit, 16kHz mono PCM or WAV)
            top_k: Return top K predictions
            min_confidence: Minimum confidence threshold

        Returns:
            List of AudioEvent detections
        """
        ...

    def get_interesting_events(
        self,
        audio_bytes: bytes,
        event_filter: Optional[list[str]] = None,
    ) -> list[AudioEvent]:
        """
        Get only interesting/actionable events from audio.

        Filters for events like doorbell, alarm, glass breaking, etc.
        """
        ...
