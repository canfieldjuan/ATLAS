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
        self._alias_map: dict[str, str] = {}  # alias -> tool_name

    @classmethod
    def get_instance(cls) -> "ToolRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, tool: Tool) -> None:
        """Register a tool and its aliases."""
        if tool.name in self._tools:
            logger.warning("Tool %s already registered, overwriting", tool.name)
        self._tools[tool.name] = tool

        # Register aliases
        aliases = getattr(tool, "aliases", None)
        if aliases:
            for alias in aliases:
                alias_lower = alias.lower()
                if alias_lower in self._alias_map:
                    logger.debug(
                        "Alias '%s' already mapped to %s, remapping to %s",
                        alias, self._alias_map[alias_lower], tool.name
                    )
                self._alias_map[alias_lower] = tool.name

        logger.info("Registered tool: %s (aliases: %s)", tool.name, aliases or [])

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def resolve_alias(self, alias: str) -> Optional[str]:
        """Resolve an alias to a tool name. Returns None if not found."""
        alias_lower = alias.lower()
        # Check if it's already a tool name
        if alias_lower in self._tools:
            return alias_lower
        # Check alias map
        return self._alias_map.get(alias_lower)

    def get_by_alias(self, alias: str) -> Optional[Tool]:
        """Get a tool by alias or name."""
        tool_name = self.resolve_alias(alias)
        if tool_name:
            return self._tools.get(tool_name)
        return None

    def get_all_aliases(self) -> list[str]:
        """Get all registered aliases (for intent parser)."""
        return list(self._alias_map.keys())

    def get_alias_map(self) -> dict[str, str]:
        """Get the full alias -> tool_name mapping."""
        return dict(self._alias_map)

    def list_all(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        """List all tool names."""
        return list(self._tools.keys())

    def get_tools_by_names(self, names: list[str]) -> list[Tool]:
        """Get tools filtered by a list of names."""
        return [
            self._tools[name]
            for name in names
            if name in self._tools
        ]

    def get_tool_schemas(self) -> list[dict]:
        """Generate Ollama-compatible tool schemas for LLM tool calling."""
        schemas = []
        for tool in self._tools.values():
            properties = {}
            required = []
            for param in tool.parameters:
                prop_type = "string"
                if param.param_type == "int":
                    prop_type = "integer"
                elif param.param_type == "float":
                    prop_type = "number"
                elif param.param_type == "boolean":
                    prop_type = "boolean"
                properties[param.name] = {
                    "type": prop_type,
                    "description": param.description,
                }
                if param.required:
                    required.append(param.name)
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })
        return schemas

    def get_tool_schemas_filtered(self, tool_names: list[str]) -> list[dict]:
        """Generate tool schemas for specific tools only."""
        schemas = []
        for name in tool_names:
            tool = self._tools.get(name)
            if not tool:
                continue
            properties = {}
            required = []
            for param in tool.parameters:
                prop_type = "string"
                if param.param_type == "int":
                    prop_type = "integer"
                elif param.param_type == "float":
                    prop_type = "number"
                elif param.param_type == "boolean":
                    prop_type = "boolean"
                properties[param.name] = {
                    "type": prop_type,
                    "description": param.description,
                }
                if param.required:
                    required.append(param.name)
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })
        return schemas

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
