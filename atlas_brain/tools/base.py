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
        """Unique tool identifier (e.g., 'get_time', 'book_appointment')."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description for LLM."""
        ...

    @property
    def parameters(self) -> list[ToolParameter]:
        """List of parameters this tool accepts."""
        ...

    @property
    def aliases(self) -> list[str]:
        """
        Short names users might say to invoke this tool.

        Examples:
            - get_time: ["time", "current time", "what time"]
            - get_weather: ["weather", "forecast"]
            - book_appointment: ["appointment", "booking", "schedule"]

        The intent parser uses these to recognize tool requests.
        Default implementation returns empty list for backwards compatibility.
        """
        return []

    @property
    def category(self) -> str:
        """
        Tool category for grouping and filtering.

        Categories: utility, scheduling, device, notification, security
        Default is 'utility'.
        """
        return "utility"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters."""
        ...
