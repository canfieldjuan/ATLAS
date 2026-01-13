"""
Base types and protocols for Atlas tools.

Tools are functions that retrieve information or perform actions
that are not device-specific (weather, traffic, calendar, etc.).
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolResult:
    """Result from executing a tool."""

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "message": self.message,
            "error": self.error,
        }


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    param_type: str
    description: str
    required: bool = False
    default: Any = None


@runtime_checkable
class Tool(Protocol):
    """Protocol for all Atlas tools."""

    @property
    def name(self) -> str:
        """Unique tool identifier."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description for LLM."""
        ...

    @property
    def parameters(self) -> list[ToolParameter]:
        """List of parameters this tool accepts."""
        ...

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters."""
        ...
