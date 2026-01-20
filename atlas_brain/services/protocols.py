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
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: list = None  # For assistant messages that call tools
    tool_call_id: str = None  # For tool result messages


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

    def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """
        Generate a response with tool calling capability.

        Args:
            messages: Conversation messages
            tools: List of tool schemas in OpenAI format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Dict with 'response' text and optional 'tool_calls' list
        """
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


@dataclass
class SpeakerInfo:
    """Information about an enrolled speaker."""

    name: str  # Speaker's name/identifier
    embedding: Any  # Voice embedding (numpy array)
    enrolled_at: float = 0.0  # Unix timestamp
    sample_count: int = 1  # Number of samples used to create embedding

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enrolled_at": self.enrolled_at,
            "sample_count": self.sample_count,
        }


@dataclass
class SpeakerMatch:
    """Result of speaker identification."""

    name: str  # Matched speaker name, or "unknown"
    confidence: float  # Similarity score 0.0 - 1.0
    is_known: bool  # Whether this is an enrolled speaker

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "confidence": self.confidence,
            "is_known": self.is_known,
        }


@runtime_checkable
class SpeakerIDService(Protocol):
    """Protocol for Speaker Identification services (Resemblyzer, etc.)."""

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

    def get_embedding(self, audio_bytes: bytes) -> Any:
        """
        Extract speaker embedding from audio.

        Args:
            audio_bytes: Raw audio (16-bit, 16kHz mono PCM or WAV)

        Returns:
            Speaker embedding vector (numpy array)
        """
        ...

    def enroll(
        self,
        name: str,
        audio_bytes: bytes,
        merge_existing: bool = True,
    ) -> SpeakerInfo:
        """
        Enroll a speaker with a voice sample.

        Args:
            name: Speaker's name/identifier
            audio_bytes: Voice sample audio
            merge_existing: If True, merge with existing enrollment

        Returns:
            SpeakerInfo for the enrolled speaker
        """
        ...

    def identify(
        self,
        audio_bytes: bytes,
        threshold: float = 0.75,
    ) -> SpeakerMatch:
        """
        Identify the speaker in an audio sample.

        Args:
            audio_bytes: Voice sample to identify
            threshold: Minimum similarity to consider a match

        Returns:
            SpeakerMatch with the best matching speaker
        """
        ...

    def list_enrolled(self) -> list[SpeakerInfo]:
        """List all enrolled speakers."""
        ...

    def remove_speaker(self, name: str) -> bool:
        """Remove an enrolled speaker."""
        ...


@dataclass
class SegmentationResult:
    """Result from image/video segmentation."""

    masks: Any  # numpy array [N, H, W]
    scores: Any  # numpy array [N]
    labels: list[str]
    boxes: Any  # numpy array [N, 4] or None
    image_shape: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "masks": self.masks.tolist() if hasattr(self.masks, "tolist") else self.masks,
            "scores": self.scores.tolist() if hasattr(self.scores, "tolist") else self.scores,
            "labels": self.labels,
            "boxes": self.boxes.tolist() if self.boxes is not None and hasattr(self.boxes, "tolist") else self.boxes,
            "image_shape": self.image_shape,
        }


@runtime_checkable
class VOSService(Protocol):
    """Protocol for Video Object Segmentation services (SAM3, etc.)."""

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

    async def segment_image(
        self,
        image: Any,
        prompts: Optional[list[str]] = None,
        point_prompts: Optional[list[tuple[int, int]]] = None,
        box_prompts: Optional[list[tuple[int, int, int, int]]] = None,
    ) -> dict[str, Any]:
        """Segment objects in an image."""
        ...

    async def segment_video(
        self,
        video_path: str,
        prompts: Optional[list[str]] = None,
        frame_skip: int = 1,
    ) -> list[dict[str, Any]]:
        """Segment objects across video frames."""
        ...


@dataclass
class OmniResponse:
    """Response from an omni (speech-to-speech) model."""

    text: str  # Text response
    audio_bytes: Optional[bytes] = None  # WAV audio output
    audio_duration_sec: float = 0.0
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "has_audio": self.audio_bytes is not None,
            "audio_duration_sec": self.audio_duration_sec,
            "metrics": self.metrics,
        }


@runtime_checkable
class OmniService(Protocol):
    """Protocol for unified speech-to-speech (Omni) services.

    These services combine STT + LLM + TTS into a single model,
    providing end-to-end speech-in, speech-out capabilities.
    """

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

    async def speech_to_speech(
        self,
        audio_bytes: bytes,
        conversation_history: Optional[list[Message]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> "OmniResponse":
        """
        Process speech input and generate speech + text response.

        Args:
            audio_bytes: Input audio (WAV format)
            conversation_history: Previous conversation turns
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            OmniResponse with text and audio output
        """
        ...

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
    ) -> dict[str, Any]:
        """Transcribe audio to text (STT mode)."""
        ...

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> bytes:
        """Convert text to speech (TTS mode)."""
        ...

    async def chat(
        self,
        messages: list[Message],
        include_audio: bool = True,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> "OmniResponse":
        """Multi-turn text conversation with optional audio output."""
        ...
