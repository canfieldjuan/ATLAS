"""
Protocol definitions for AI model services.

These protocols define the interface that LLM implementations must follow,
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

