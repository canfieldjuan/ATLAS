"""
Tool registry for Atlas.

Manages registration and lookup of available tools.
"""

import logging
from typing import Optional

from .base import Tool, ToolResult

logger = logging.getLogger("atlas.tools.registry")


class ToolRegistry:
    """Central registry for all tools."""

    _instance: Optional["ToolRegistry"] = None

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    @classmethod
    def get_instance(cls) -> "ToolRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        if tool.name in self._tools:
            logger.warning("Tool %s already registered, overwriting", tool.name)
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_all(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        """List all tool names."""
        return list(self._tools.keys())

    async def execute(self, name: str, params: dict) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get(name)
        if not tool:
            return ToolResult(
                success=False,
                error="TOOL_NOT_FOUND",
                message=f"Tool not found: {name}",
            )
        try:
            return await tool.execute(params)
        except Exception as e:
            logger.exception("Error executing tool %s", name)
            return ToolResult(
                success=False,
                error="EXECUTION_ERROR",
                message=str(e),
            )


# Global registry instance
tool_registry = ToolRegistry.get_instance()
