"""Standalone protocol/dataclass definitions for the LLM-infrastructure
package.

Mirrors ``atlas_brain.services.protocols`` exactly so callers that import
``Message``, ``ModelInfo``, ``InferenceMetrics``, or the ``LLMService``
Protocol from either source get identical types. This file is loaded by
``extracted_llm_infrastructure.services.protocols`` when
``EXTRACTED_LLM_INFRA_STANDALONE=1`` is set; otherwise the bridge stub
delegates to atlas_brain.

Pure Python -- no third-party deps.
"""

from __future__ import annotations

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
    tool_calls: list = None  # type: ignore[assignment]
    tool_call_id: str = None  # type: ignore[assignment]


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
        """Generate a response with tool calling capability."""
        ...
