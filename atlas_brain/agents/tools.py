"""
Agent tools system.

Wraps IntentParser, ActionDispatcher, and built-in tools
to provide a unified tool interface for agents.
"""

import logging
from typing import Any, Optional

from .protocols import AgentTools as AgentToolsProtocol

logger = logging.getLogger("atlas.agents.tools")


# Tool name mapping - maps intent target_name to tool registry names
TOOL_MAP = {
    "time": "get_time",
    "weather": "get_weather",
    "traffic": "get_traffic",
    "location": "get_location",
    "calendar": "get_calendar",
    "reminder": "set_reminder",
    "reminders": "list_reminders",
    "complete_reminder": "complete_reminder",
}


class AtlasAgentTools:
    """
    Tools system for Atlas Agent.

    Provides unified access to:
    - Intent parsing (natural language → Intent)
    - Action execution (Intent → device actions)
    - Built-in tools (weather, traffic, time, location)

    This class wraps existing components rather than replacing them,
    allowing gradual migration to the agent architecture.
    """

    def __init__(
        self,
        intent_parser: Optional[Any] = None,
        action_dispatcher: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
    ):
        """
        Initialize agent tools.

        Args:
            intent_parser: IntentParser instance (lazy-loaded if None)
            action_dispatcher: ActionDispatcher instance (lazy-loaded if None)
            tool_registry: ToolRegistry instance (lazy-loaded if None)
        """
        self._intent_parser = intent_parser
        self._action_dispatcher = action_dispatcher
        self._tool_registry = tool_registry

    # Lazy loading of dependencies

    def _get_intent_parser(self) -> Any:
        """Get or create IntentParser."""
        if self._intent_parser is None:
            from ..capabilities.intent_parser import intent_parser
            self._intent_parser = intent_parser
        return self._intent_parser

    def _get_action_dispatcher(self) -> Any:
        """Get or create ActionDispatcher."""
        if self._action_dispatcher is None:
            from ..capabilities.actions import action_dispatcher
            self._action_dispatcher = action_dispatcher
        return self._action_dispatcher

    def _get_tool_registry(self) -> Any:
        """Get or create ToolRegistry."""
        if self._tool_registry is None:
            from ..tools import tool_registry
            self._tool_registry = tool_registry
        return self._tool_registry

    # Intent parsing

    async def parse_intent(
        self,
        query: str,
    ) -> Optional[Any]:
        """
        Parse intent from natural language query.

        Args:
            query: Natural language input (e.g., "turn on the living room lights")

        Returns:
            Intent object if parsed, None if not a device command
        """
        try:
            parser = self._get_intent_parser()
            intent = await parser.parse(query)
            return intent

        except Exception as e:
            logger.warning("Intent parsing failed: %s", e)
            return None

    def is_device_command(self, query: str) -> bool:
        """
        Quick check if query looks like a device command.

        Uses keyword matching - faster than full parse.
        """
        query_lower = query.lower()

        # Device action keywords
        action_keywords = [
            "turn on", "turn off", "switch on", "switch off",
            "dim", "brighten", "set", "toggle",
            "lights", "light", "lamp", "tv", "television",
        ]

        return any(kw in query_lower for kw in action_keywords)

    # Action execution

    async def execute_intent(
        self,
        intent: Any,
    ) -> dict[str, Any]:
        """
        Execute a parsed intent via ActionDispatcher.

        Args:
            intent: Intent object from parse_intent()

        Returns:
            Dictionary with success, message, and any data
        """
        try:
            dispatcher = self._get_action_dispatcher()
            result = await dispatcher.dispatch_intent(intent)

            return {
                "success": result.success,
                "message": result.message,
                "data": result.data,
                "error": result.error,
            }

        except Exception as e:
            logger.warning("Intent execution failed: %s", e)
            return {
                "success": False,
                "message": f"Execution failed: {e}",
                "error": "EXECUTION_ERROR",
            }

    async def execute_action(
        self,
        capability_id: str,
        action: str,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Execute a direct action on a capability.

        Args:
            capability_id: ID of the capability/device
            action: Action name (e.g., "turn_on", "set_brightness")
            params: Action parameters

        Returns:
            Dictionary with success, message, and any data
        """
        try:
            from ..capabilities.actions import ActionRequest

            dispatcher = self._get_action_dispatcher()
            request = ActionRequest(
                capability_id=capability_id,
                action=action,
                params=params or {},
            )
            result = await dispatcher.dispatch(request)

            return {
                "success": result.success,
                "message": result.message,
                "data": result.data,
                "error": result.error,
            }

        except Exception as e:
            logger.warning("Action execution failed: %s", e)
            return {
                "success": False,
                "message": f"Execution failed: {e}",
                "error": "EXECUTION_ERROR",
            }

    # Built-in tools

    async def execute_tool(
        self,
        tool_name: str,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Execute a built-in tool.

        Args:
            tool_name: Name of the tool (e.g., "get_weather", "get_traffic")
            params: Tool parameters

        Returns:
            Dictionary with success, message, data, and error
        """
        try:
            registry = self._get_tool_registry()
            result = await registry.execute(tool_name, params or {})

            return {
                "success": result.success,
                "message": result.message,
                "data": result.data,
                "error": result.error,
            }

        except Exception as e:
            logger.warning("Tool execution failed: %s", e)
            return {
                "success": False,
                "message": f"Tool execution failed: {e}",
                "error": "TOOL_ERROR",
            }

    async def execute_tool_by_intent(
        self,
        target_name: str,
        parameters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Execute a tool based on intent target_name.

        Maps intent target names to tool registry names and executes.

        Args:
            target_name: Tool name from intent (e.g., "time", "weather")
            parameters: Optional parameters from intent

        Returns:
            Tool result dictionary
        """
        # Map intent target_name to tool registry name
        tool_name = TOOL_MAP.get(target_name, target_name)
        params = parameters or {}

        # Weather and traffic need location
        if tool_name in ("get_weather", "get_traffic"):
            try:
                loc_result = await self.execute_tool("get_location", {})
                if loc_result["success"] and loc_result.get("data"):
                    params["latitude"] = loc_result["data"].get("latitude")
                    params["longitude"] = loc_result["data"].get("longitude")
            except Exception as e:
                logger.warning("Location fetch failed: %s", e)

        return await self.execute_tool(tool_name, params)

    def list_tools(self) -> list[str]:
        """List available tool names."""
        try:
            registry = self._get_tool_registry()
            return registry.list_names()
        except Exception:
            return list(TOOL_MAP.values())

    # Capability listing

    def list_capabilities(self) -> list[dict[str, Any]]:
        """
        List all available capabilities/devices.

        Returns:
            List of capability info dictionaries
        """
        try:
            dispatcher = self._get_action_dispatcher()
            capabilities = dispatcher.registry.list_all()

            return [
                {
                    "id": cap.id,
                    "name": cap.name,
                    "type": cap.capability_type.value if hasattr(cap.capability_type, "value") else str(cap.capability_type),
                    "actions": cap.supported_actions,
                }
                for cap in capabilities
            ]

        except Exception as e:
            logger.warning("Failed to list capabilities: %s", e)
            return []

    def get_capability(self, capability_id: str) -> Optional[dict[str, Any]]:
        """
        Get info about a specific capability.

        Args:
            capability_id: ID of the capability

        Returns:
            Capability info dictionary, or None if not found
        """
        try:
            dispatcher = self._get_action_dispatcher()
            cap = dispatcher.registry.get(capability_id)

            if cap:
                return {
                    "id": cap.id,
                    "name": cap.name,
                    "type": cap.capability_type.value if hasattr(cap.capability_type, "value") else str(cap.capability_type),
                    "actions": cap.supported_actions,
                }
            return None

        except Exception as e:
            logger.warning("Failed to get capability %s: %s", capability_id, e)
            return None


# Global tools instance
_agent_tools: Optional[AtlasAgentTools] = None


def get_agent_tools() -> AtlasAgentTools:
    """Get or create the global agent tools instance."""
    global _agent_tools
    if _agent_tools is None:
        _agent_tools = AtlasAgentTools()
    return _agent_tools


def reset_agent_tools() -> None:
    """Reset the global agent tools instance."""
    global _agent_tools
    _agent_tools = None
