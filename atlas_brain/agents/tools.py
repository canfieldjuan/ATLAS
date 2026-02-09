"""
Agent tools system.

Wraps IntentParser, ActionDispatcher, and built-in tools
to provide a unified tool interface for agents.
"""

import logging
from typing import Any, Optional, TYPE_CHECKING

from .protocols import AgentTools as AgentToolsProtocol

if TYPE_CHECKING:
    from ..services.intent_router import IntentRouteResult

logger = logging.getLogger("atlas.agents.tools")


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

    async def route_intent(self, query: str) -> "IntentRouteResult":
        """
        Fast intent routing using semantic embeddings.

        Classifies queries into: device_command, tool_use, or conversation.
        Falls back to conversation if router is disabled or on error.

        Args:
            query: User query text

        Returns:
            IntentRouteResult with action_category, raw_label, confidence
        """
        try:
            from ..services.intent_router import get_intent_router
            router = get_intent_router()
            return await router.route(query)
        except Exception as e:
            logger.warning("Intent routing failed: %s", e)
            # Return fallback result
            from ..services.intent_router import IntentRouteResult
            return IntentRouteResult(
                action_category="conversation",
                raw_label="error",
                confidence=0.0,
            )

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

    async def _enrich_location(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Inject user's real-time location from HA for location-dependent tools."""
        if tool_name not in ("get_weather", "get_traffic"):
            return params
        if params.get("latitude") and params.get("longitude"):
            return params  # Already has coordinates
        try:
            loc_result = await self.execute_tool("get_location", {})
            if loc_result["success"] and loc_result.get("data"):
                lat = loc_result["data"].get("latitude")
                lon = loc_result["data"].get("longitude")
                if lat and lon:
                    params["latitude"] = lat
                    params["longitude"] = lon
                    logger.info("Location resolved from HA: %.4f, %.4f", lat, lon)
        except Exception as e:
            logger.warning("Location enrichment failed: %s", e)
        return params

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
            resolved_params = await self._enrich_location(tool_name, params or {})
            result = await registry.execute(tool_name, resolved_params)

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
        # Resolve alias via registry
        registry = self._get_tool_registry()
        tool_name = registry.resolve_alias(target_name) or target_name
        params = await self._enrich_location(tool_name, parameters or {})

        # Map intent parameter names to tool parameter names for booking
        if tool_name == "book_appointment":
            param_map = {
                "name": "customer_name",
                "phone": "customer_phone",
                "when": "date",
                "person": "customer_name",
            }
            mapped_params = {}
            for key, value in params.items():
                mapped_key = param_map.get(key, key)
                mapped_params[mapped_key] = value
            # Split "when" into date and time if needed
            if "date" in mapped_params and "time" not in mapped_params:
                when_val = mapped_params.get("date", "")
                if " at " in when_val:
                    date_part, time_part = when_val.split(" at ", 1)
                    mapped_params["date"] = date_part
                    mapped_params["time"] = time_part
                else:
                    # Try to extract time from end of string
                    # Patterns: "Monday 2pm", "tomorrow 10am", "next Tuesday morning"
                    import re
                    time_pattern = r'\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)|morning|afternoon|evening|noon)$'
                    match = re.search(time_pattern, when_val, re.IGNORECASE)
                    if match:
                        time_val = match.group(1).strip().lower()
                        # Convert descriptive times to specific hours for dateparser
                        time_map = {
                            "morning": "9am",
                            "afternoon": "2pm",
                            "evening": "6pm",
                            "noon": "12pm",
                        }
                        mapped_params["time"] = time_map.get(time_val, time_val)
                        mapped_params["date"] = when_val[:match.start()].strip()
                    else:
                        # No time found - use the whole string as date
                        # The booking tool will ask for time if needed
                        pass
            params = mapped_params

        return await self.execute_tool(tool_name, params)

    def list_tools(self) -> list[str]:
        """List available tool names."""
        try:
            registry = self._get_tool_registry()
            return registry.list_names()
        except Exception:
            return []

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
