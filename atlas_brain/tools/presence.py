"""
Presence-aware tools for device control.

These tools hide location from the LLM - the LLM just says "turn on the lights"
and the tool resolves which lights based on user's current room.

This is the key architectural decision: location is an implementation detail
of the tool layer, not something the LLM reasons about.
"""

import logging
from typing import Any, Optional

from .base import Tool, ToolParameter, ToolResult

logger = logging.getLogger("atlas.tools.presence")


class LightsNearUserTool:
    """
    Control lights near the user's current location.

    The LLM doesn't need to know which room - just the intent.
    Location resolution happens internally via PresenceService.
    """

    name = "lights_near_user"
    description = (
        "Control lights in the user's current location. "
        "Use this when the user says 'turn on the lights' without specifying a room. "
        "Automatically determines which lights to control based on where the user is."
    )
    parameters = [
        ToolParameter(
            name="action",
            param_type="string",
            description="Action to perform: 'on', 'off', or 'toggle'",
            required=True,
        ),
        ToolParameter(
            name="brightness",
            param_type="integer",
            description="Brightness level 0-100 (optional, only for 'on' action)",
            required=False,
        ),
    ]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute light control near user."""
        action = params.get("action", "on").lower()
        brightness = params.get("brightness")

        # Get user's current room from presence service
        try:
            from ..presence import get_presence_service
            presence = get_presence_service()
            room_id = presence.get_current_room()

            if not room_id:
                return ToolResult(
                    success=False,
                    error="Could not determine your current location. Please specify which lights.",
                    message="I'm not sure where you are right now.",
                )

            # Get lights in this room
            lights = presence.get_devices_near_user(device_type="lights")
            room_state = presence.get_room_state(room_id)
            room_name = room_state.room_name if room_state else room_id

            if not lights:
                return ToolResult(
                    success=False,
                    error=f"No lights configured for {room_name}",
                    message=f"I don't see any lights registered in the {room_name}.",
                )

        except Exception as e:
            logger.error("Presence service error: %s", e)
            return ToolResult(
                success=False,
                error=str(e),
                message="I couldn't determine your location.",
            )

        # Execute action on lights via Home Assistant
        try:
            from ..capabilities.backends.homeassistant import HomeAssistantBackend
            from ..config import settings

            if not settings.homeassistant.enabled:
                return ToolResult(
                    success=False,
                    error="Home Assistant not configured",
                )

            backend = HomeAssistantBackend(
                base_url=settings.homeassistant.url,
                access_token=settings.homeassistant.token,
            )
            await backend.connect()

            # Map action to HA service
            if action == "on":
                service = "light/turn_on"
                payload = {"entity_id": lights}
                if brightness is not None:
                    payload["brightness_pct"] = brightness
            elif action == "off":
                service = "light/turn_off"
                payload = {"entity_id": lights}
            elif action == "toggle":
                service = "light/toggle"
                payload = {"entity_id": lights}
            else:
                return ToolResult(
                    success=False,
                    error=f"Unknown action: {action}",
                )

            await backend.send_command(service, payload)
            await backend.disconnect()

            # Build response
            light_names = ", ".join(l.replace("light.", "").replace("_", " ") for l in lights)
            action_past = "turned on" if action == "on" else "turned off" if action == "off" else "toggled"

            return ToolResult(
                success=True,
                data={
                    "room": room_name,
                    "lights": lights,
                    "action": action,
                    "brightness": brightness,
                },
                message=f"I've {action_past} the lights in the {room_name}.",
            )

        except Exception as e:
            logger.error("Failed to control lights: %s", e)
            return ToolResult(
                success=False,
                error=str(e),
                message="I couldn't control the lights.",
            )


class MediaNearUserTool:
    """
    Control media players near the user's current location.
    """

    name = "media_near_user"
    description = (
        "Control TV or media player in the user's current location. "
        "Use this when the user says 'turn on the TV' without specifying a room."
    )
    parameters = [
        ToolParameter(
            name="action",
            param_type="string",
            description="Action: 'on', 'off', 'play', 'pause', 'stop'",
            required=True,
        ),
    ]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute media control near user."""
        action = params.get("action", "on").lower()

        try:
            from ..presence import get_presence_service
            presence = get_presence_service()
            room_id = presence.get_current_room()

            if not room_id:
                return ToolResult(
                    success=False,
                    error="Could not determine your current location.",
                    message="I'm not sure where you are right now.",
                )

            media_players = presence.get_devices_near_user(device_type="media_players")
            room_state = presence.get_room_state(room_id)
            room_name = room_state.room_name if room_state else room_id

            if not media_players:
                return ToolResult(
                    success=False,
                    error=f"No media players in {room_name}",
                    message=f"I don't see any TV or media player in the {room_name}.",
                )

        except Exception as e:
            logger.error("Presence service error: %s", e)
            return ToolResult(
                success=False,
                error=str(e),
            )

        try:
            from ..capabilities.backends.homeassistant import HomeAssistantBackend
            from ..config import settings

            backend = HomeAssistantBackend(
                base_url=settings.homeassistant.url,
                access_token=settings.homeassistant.token,
            )
            await backend.connect()

            # Map action to HA service
            action_map = {
                "on": "media_player/turn_on",
                "off": "media_player/turn_off",
                "play": "media_player/media_play",
                "pause": "media_player/media_pause",
                "stop": "media_player/media_stop",
            }

            service = action_map.get(action)
            if not service:
                return ToolResult(
                    success=False,
                    error=f"Unknown action: {action}",
                )

            await backend.send_command(service, {"entity_id": media_players})
            await backend.disconnect()

            action_past = {
                "on": "turned on",
                "off": "turned off",
                "play": "started playing",
                "pause": "paused",
                "stop": "stopped",
            }.get(action, action)

            return ToolResult(
                success=True,
                data={"room": room_name, "media_players": media_players, "action": action},
                message=f"I've {action_past} the TV in the {room_name}.",
            )

        except Exception as e:
            logger.error("Failed to control media: %s", e)
            return ToolResult(
                success=False,
                error=str(e),
            )


class SceneNearUserTool:
    """
    Set a lighting scene in the user's current location.
    """

    name = "scene_near_user"
    description = (
        "Set a lighting scene or mood in the user's current location. "
        "Use for requests like 'make it cozy' or 'I'm going to watch a movie'."
    )
    parameters = [
        ToolParameter(
            name="scene",
            param_type="string",
            description="Scene name: 'bright', 'dim', 'cozy', 'movie', 'focus', 'relax', 'off'",
            required=True,
        ),
    ]

    # Scene definitions: brightness and color temp
    SCENES = {
        "bright": {"brightness_pct": 100, "color_temp_kelvin": 5000},
        "focus": {"brightness_pct": 100, "color_temp_kelvin": 5500},
        "dim": {"brightness_pct": 30, "color_temp_kelvin": 3000},
        "cozy": {"brightness_pct": 40, "color_temp_kelvin": 2700},
        "relax": {"brightness_pct": 50, "color_temp_kelvin": 2700},
        "movie": {"brightness_pct": 10, "color_temp_kelvin": 2700},
        "off": None,  # Special case - turn off
    }

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute scene near user."""
        scene = params.get("scene", "").lower()

        if scene not in self.SCENES:
            return ToolResult(
                success=False,
                error=f"Unknown scene: {scene}",
                message=f"I don't know the '{scene}' scene. Try: bright, dim, cozy, movie, focus, relax, or off.",
            )

        try:
            from ..presence import get_presence_service
            presence = get_presence_service()
            room_id = presence.get_current_room()

            if not room_id:
                return ToolResult(
                    success=False,
                    error="Could not determine location",
                    message="I'm not sure where you are.",
                )

            lights = presence.get_devices_near_user(device_type="lights")
            room_state = presence.get_room_state(room_id)
            room_name = room_state.room_name if room_state else room_id

            if not lights:
                return ToolResult(
                    success=False,
                    error=f"No lights in {room_name}",
                )

        except Exception as e:
            return ToolResult(success=False, error=str(e))

        try:
            from ..capabilities.backends.homeassistant import HomeAssistantBackend
            from ..config import settings

            backend = HomeAssistantBackend(
                base_url=settings.homeassistant.url,
                access_token=settings.homeassistant.token,
            )
            await backend.connect()

            scene_config = self.SCENES[scene]

            if scene_config is None:
                # Turn off
                await backend.send_command("light/turn_off", {"entity_id": lights})
            else:
                payload = {"entity_id": lights, **scene_config}
                await backend.send_command("light/turn_on", payload)

            await backend.disconnect()

            return ToolResult(
                success=True,
                data={"room": room_name, "scene": scene, "lights": lights},
                message=f"I've set the {room_name} to {scene} mode.",
            )

        except Exception as e:
            logger.error("Failed to set scene: %s", e)
            return ToolResult(success=False, error=str(e))


class WhereAmITool:
    """
    Tell the user their current detected location.

    This is for debugging/transparency - lets user know what room
    the system thinks they're in.
    """

    name = "where_am_i"
    description = (
        "Tell the user which room the system detects them in. "
        "Use when user asks 'where am I?' or 'what room am I in?'"
    )
    parameters = []

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Report user's current location."""
        try:
            from ..presence import get_presence_service
            presence = get_presence_service()
            user_presence = presence.get_user_presence()

            if not user_presence or not user_presence.current_room:
                return ToolResult(
                    success=True,
                    data={"room": None, "status": "unknown"},
                    message="I'm not sure which room you're in right now.",
                )

            room_name = user_presence.current_room_name or user_presence.current_room
            confidence = user_presence.confidence
            source = user_presence.source.value if user_presence.source else "unknown"

            return ToolResult(
                success=True,
                data={
                    "room": user_presence.current_room,
                    "room_name": room_name,
                    "confidence": confidence,
                    "source": source,
                },
                message=f"You're in the {room_name} (detected via {source}, {confidence:.0%} confidence).",
            )

        except Exception as e:
            logger.error("Presence error: %s", e)
            return ToolResult(success=False, error=str(e))


# Tool instances for registration
lights_near_user = LightsNearUserTool()
media_near_user = MediaNearUserTool()
scene_near_user = SceneNearUserTool()
where_am_i = WhereAmITool()

# All presence tools
PRESENCE_TOOLS = [
    lights_near_user,
    media_near_user,
    scene_near_user,
    where_am_i,
]
