"""
Local Action Dispatcher for edge devices.

Executes device commands locally via Home Assistant REST API,
generating appropriate responses.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

from ..intent.actions import ActionIntent

logger = logging.getLogger("atlas.edge.capabilities.dispatcher")


@dataclass
class ActionResult:
    """Result from executing an action."""

    success: bool
    message: str = ""
    entity_id: Optional[str] = None
    action: str = ""
    error: Optional[str] = None


class LocalActionDispatcher:
    """
    Local action dispatcher for edge devices.

    Handles:
    - Entity name to ID resolution
    - Home Assistant service calls
    - Response generation
    """

    def __init__(self):
        """Initialize the dispatcher."""
        self._ha = None

    async def _get_ha(self):
        """Get Home Assistant client."""
        if self._ha is None:
            from .homeassistant import get_homeassistant

            self._ha = await get_homeassistant()
        return self._ha

    async def execute(self, intent: ActionIntent) -> ActionResult:
        """
        Execute an action intent.

        Args:
            intent: Parsed action intent

        Returns:
            ActionResult with success status and message
        """
        ha = await self._get_ha()

        # Resolve entity ID if not already set
        entity_id = intent.target_id
        if not entity_id and intent.target_name:
            # Map target_type to HA domain
            domain_map = {
                "light": "light",
                "switch": "switch",
                "fan": "fan",
                "scene": "scene",
                "media_player": "media_player",
                "device": None,  # Any domain
            }
            domain = domain_map.get(intent.target_type)
            entity_id = ha.resolve_entity_id(intent.target_name, domain)

            if not entity_id:
                return ActionResult(
                    success=False,
                    message=f"I couldn't find a device called {intent.target_name}.",
                    action=intent.action,
                    error="entity_not_found",
                )

            intent.target_id = entity_id

        if not entity_id:
            return ActionResult(
                success=False,
                message="I'm not sure which device you mean.",
                action=intent.action,
                error="no_entity",
            )

        # Get service path and data
        service_path = intent.get_service_path()
        service_data = intent.get_service_data()

        if not service_path:
            return ActionResult(
                success=False,
                message=f"I don't know how to {intent.action} that device.",
                action=intent.action,
                error="unknown_action",
            )

        # Execute the service
        try:
            await ha.call_service(service_path, service_data)

            # Generate response
            message = self._generate_response(intent, success=True)

            return ActionResult(
                success=True,
                message=message,
                entity_id=entity_id,
                action=intent.action,
            )

        except Exception as e:
            logger.error("Action failed: %s -> %s", service_path, e)

            message = self._generate_response(intent, success=False, error=str(e))

            return ActionResult(
                success=False,
                message=message,
                entity_id=entity_id,
                action=intent.action,
                error=str(e),
            )

    def _generate_response(
        self,
        intent: ActionIntent,
        success: bool,
        error: Optional[str] = None,
    ) -> str:
        """
        Generate natural response for action result.

        Args:
            intent: The executed intent
            success: Whether action succeeded
            error: Error message if failed

        Returns:
            Response text
        """
        target = intent.target_name or "device"
        action = intent.action

        if success:
            # Success responses
            responses = {
                "turn_on": f"Done. The {target} is on.",
                "turn_off": f"Done. The {target} is off.",
                "toggle": f"Done. Toggled the {target}.",
                "set_brightness": f"Done. Set the {target} to {intent.parameters.get('brightness', 'the requested')} percent.",
                "brighten": f"Done. Brightened the {target}.",
                "dim": f"Done. Dimmed the {target}.",
                "play": f"Playing on {target}.",
                "pause": f"Paused {target}.",
                "stop": f"Stopped {target}.",
                "mute": f"Muted {target}.",
                "unmute": f"Unmuted {target}.",
                "set_volume": f"Set volume to {intent.parameters.get('volume', 'the requested')} percent.",
                "volume_up": "Volume up.",
                "volume_down": "Volume down.",
            }
            return responses.get(action, "Done.")

        else:
            # Failure responses
            if "unavailable" in str(error).lower():
                return f"The {target} doesn't seem to be available right now."
            elif "not found" in str(error).lower():
                return f"I couldn't find a device called {target}."
            else:
                return f"Sorry, I couldn't control the {target}."


# Singleton instance
_dispatcher: Optional[LocalActionDispatcher] = None


def get_dispatcher() -> LocalActionDispatcher:
    """Get or create global dispatcher."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = LocalActionDispatcher()
    return _dispatcher
