"""
Action dispatch framework for capabilities.

Handles both direct action requests and intent-based dispatch from the LLM.
"""

import logging
from typing import Any, Optional

from pydantic import BaseModel

from .protocols import ActionResult, Capability
from .registry import CapabilityRegistry

logger = logging.getLogger("atlas.capabilities.actions")


class ActionRequest(BaseModel):
    """Request to execute an action on a capability."""
    capability_id: str
    action: str
    params: dict[str, Any] = {}


class Intent(BaseModel):
    """
    Parsed intent from natural language.

    The VLM extracts this structure from user queries like
    "turn on the living room lights" or "set thermostat to 72".
    """
    action: str  # e.g., "turn_on", "set_temperature"
    target_type: Optional[str] = None  # e.g., "light", "thermostat"
    target_name: Optional[str] = None  # e.g., "living room", "bedroom"
    target_id: Optional[str] = None  # Direct capability ID if known
    parameters: dict[str, Any] = {}  # e.g., {"brightness": 100}
    confidence: float = 0.0
    raw_query: str = ""


class ActionDispatcher:
    """
    Dispatches actions to capabilities.

    Handles both direct action requests and intent-based dispatch
    from the VLM.
    """

    def __init__(self, registry: Optional[CapabilityRegistry] = None):
        self.registry = registry or CapabilityRegistry.get_instance()

    async def dispatch(self, request: ActionRequest) -> ActionResult:
        """
        Execute an action directly on a capability.

        Args:
            request: The action request with capability_id, action, and params

        Returns:
            ActionResult from the capability
        """
        capability = self.registry.get(request.capability_id)
        if not capability:
            logger.warning("Capability not found: %s", request.capability_id)
            return ActionResult(
                success=False,
                message=f"Capability not found: {request.capability_id}",
                error="CAPABILITY_NOT_FOUND",
            )

        if request.action not in capability.supported_actions:
            logger.warning(
                "Action '%s' not supported by %s (supported: %s)",
                request.action,
                request.capability_id,
                capability.supported_actions,
            )
            return ActionResult(
                success=False,
                message=f"Action '{request.action}' not supported by {request.capability_id}",
                error="ACTION_NOT_SUPPORTED",
            )

        logger.info(
            "Dispatching %s.%s(%s)",
            request.capability_id,
            request.action,
            request.params,
        )

        try:
            result = await capability.execute_action(request.action, request.params)
            logger.info("Action result: %s", result)
            return result
        except Exception as e:
            logger.exception("Error executing action %s on %s", request.action, request.capability_id)
            return ActionResult(
                success=False,
                message=f"Error executing action: {e}",
                error="EXECUTION_ERROR",
            )

    async def dispatch_intent(self, intent: Intent) -> ActionResult:
        """
        Dispatch an action based on a parsed intent.

        Resolves the target capability from the intent's target information.
        """
        # If direct ID is provided, use it
        if intent.target_id:
            capability = self.registry.get(intent.target_id)
        else:
            # Resolve by type and name
            capability = self._resolve_capability(intent)

        if not capability:
            return ActionResult(
                success=False,
                message=f"Could not resolve target for intent: {intent.action}",
                error="TARGET_NOT_RESOLVED",
            )

        # Map generic "query" action to device-specific read action
        action = intent.action
        if action == "query":
            action = "get_state"

        request = ActionRequest(
            capability_id=capability.id,
            action=action,
            params=intent.parameters,
        )
        return await self.dispatch(request)

    def _resolve_capability(self, intent: Intent) -> Optional[Capability]:
        """
        Resolve a capability from intent target info.

        Uses simple substring matching - can be enhanced with fuzzy matching.
        """
        from .protocols import CapabilityType

        # Type aliases for common names
        TYPE_ALIASES = {
            "tv": CapabilityType.MEDIA_PLAYER,
            "television": CapabilityType.MEDIA_PLAYER,
            "roku": CapabilityType.MEDIA_PLAYER,
            "media": CapabilityType.MEDIA_PLAYER,
        }

        for cap in self.registry.list_all():
            # Match by type if specified
            if intent.target_type:
                target_type_lower = intent.target_type.lower()

                # Check aliases first
                if target_type_lower in TYPE_ALIASES:
                    expected_type = TYPE_ALIASES[target_type_lower]
                    if cap.capability_type != expected_type:
                        continue
                else:
                    # Try direct type match
                    try:
                        expected_type = CapabilityType(target_type_lower)
                        if cap.capability_type != expected_type:
                            continue
                    except ValueError:
                        # Unknown type, skip type check
                        pass

            # Match by name if specified (case-insensitive substring)
            if intent.target_name:
                target_lower = intent.target_name.lower()
                cap_name_lower = cap.name.lower()
                # Check for substring match or common aliases
                if (target_lower in cap_name_lower or
                    cap_name_lower in target_lower or
                    (target_lower in ["tv", "television", "roku"] and "tv" in cap_name_lower)):
                    return cap

        # No match found
        return None


# Global dispatcher instance
action_dispatcher = ActionDispatcher()
