"""
Media player device implementations.

Provides control for TVs, media players, and streaming devices
through Home Assistant.
"""

from typing import Any

from ..backends.homeassistant import HomeAssistantBackend
from ..protocols import ActionResult, CapabilityType, MediaPlayerState


class HomeAssistantMediaPlayer:
    """Media player capability backed by Home Assistant.

    Supports media_player.* entities (TVs, streaming devices, receivers, etc.).
    """

    SUPPORTED_ACTIONS = [
        "turn_on", "turn_off", "toggle",
        "volume_up", "volume_down", "mute", "unmute",
        "play", "pause", "stop",
        "select_source", "get_state",
    ]

    def __init__(
        self,
        entity_id: str,
        name: str,
        backend: HomeAssistantBackend,
    ):
        self._entity_id = entity_id
        self._name = name
        self._backend = backend
        self._domain = entity_id.split(".", 1)[0]  # "media_player"

    @property
    def id(self) -> str:
        return self._entity_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def capability_type(self) -> CapabilityType:
        return CapabilityType.MEDIA_PLAYER

    @property
    def supported_actions(self) -> list[str]:
        return self.SUPPORTED_ACTIONS

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self._entity_id,
            "name": self._name,
            "type": self.capability_type.value,
            "supported_actions": self.supported_actions,
        }

    async def get_state(self) -> MediaPlayerState:
        """Get state from Home Assistant."""
        raw = await self._backend.get_state(self._entity_id)
        attrs = raw.get("attributes", {})
        return MediaPlayerState(
            is_on=raw.get("state") not in ("off", "unavailable", "unknown"),
            volume_level=attrs.get("volume_level"),
            is_volume_muted=attrs.get("is_volume_muted", False),
            source=attrs.get("source"),
            media_title=attrs.get("media_title"),
            app_name=attrs.get("app_name"),
            online=raw.get("state") != "unavailable",
            last_updated=raw.get("last_updated"),
        )

    async def execute_action(self, action: str, params: dict[str, Any]) -> ActionResult:
        """Execute an action on the media player via Home Assistant."""
        # Check entity availability before sending commands
        try:
            raw = await self._backend.get_state(self._entity_id)
            entity_state = raw.get("state", "unknown")
            if entity_state == "unavailable":
                return ActionResult(
                    success=False,
                    message=f"{self._name} is unavailable (device offline or unreachable)",
                    error="DEVICE_UNAVAILABLE",
                )
        except Exception:
            pass  # Proceed anyway if state check fails

        base_payload = {"entity_id": self._entity_id}

        if action == "turn_on":
            await self._backend.send_command(
                f"{self._domain}/turn_on", base_payload,
            )
            return ActionResult(success=True, message=f"{self._name} turning on")

        elif action == "turn_off":
            await self._backend.send_command(
                f"{self._domain}/turn_off", base_payload,
            )
            return ActionResult(success=True, message=f"{self._name} turning off")

        elif action == "toggle":
            await self._backend.send_command(
                f"{self._domain}/toggle", base_payload,
            )
            return ActionResult(success=True, message=f"{self._name} toggled")

        elif action == "volume_up":
            await self._backend.send_command(
                f"{self._domain}/volume_up", base_payload,
            )
            return ActionResult(success=True, message=f"{self._name} volume up")

        elif action == "volume_down":
            await self._backend.send_command(
                f"{self._domain}/volume_down", base_payload,
            )
            return ActionResult(success=True, message=f"{self._name} volume down")

        elif action == "mute":
            is_muted = params.get("is_volume_muted", True)
            await self._backend.send_command(
                f"{self._domain}/volume_mute",
                {**base_payload, "is_volume_muted": is_muted},
            )
            state_str = "muted" if is_muted else "unmuted"
            return ActionResult(success=True, message=f"{self._name} {state_str}")

        elif action == "unmute":
            await self._backend.send_command(
                f"{self._domain}/volume_mute",
                {**base_payload, "is_volume_muted": False},
            )
            return ActionResult(success=True, message=f"{self._name} unmuted")

        elif action == "play":
            await self._backend.send_command(
                f"{self._domain}/media_play", base_payload,
            )
            return ActionResult(success=True, message=f"{self._name} playing")

        elif action == "pause":
            await self._backend.send_command(
                f"{self._domain}/media_pause", base_payload,
            )
            return ActionResult(success=True, message=f"{self._name} paused")

        elif action == "stop":
            await self._backend.send_command(
                f"{self._domain}/media_stop", base_payload,
            )
            return ActionResult(success=True, message=f"{self._name} stopped")

        elif action == "select_source":
            source = params.get("source")
            if not source:
                return ActionResult(
                    success=False,
                    message="source parameter required",
                    error="MISSING_PARAM",
                )
            await self._backend.send_command(
                f"{self._domain}/select_source",
                {**base_payload, "source": source},
            )
            return ActionResult(
                success=True,
                message=f"{self._name} source set to {source}",
            )

        elif action == "get_state":
            state = await self.get_state()
            status = "on" if state.is_on else "off"
            extra = ""
            if state.app_name:
                extra = f" ({state.app_name})"
            elif state.media_title:
                extra = f" ({state.media_title})"
            return ActionResult(
                success=True,
                message=f"{self._name} is {status}{extra}",
                data={
                    "is_on": state.is_on,
                    "volume_level": state.volume_level,
                    "is_volume_muted": state.is_volume_muted,
                    "source": state.source,
                    "media_title": state.media_title,
                    "app_name": state.app_name,
                },
            )

        else:
            return ActionResult(
                success=False,
                message=f"Unknown action: {action}",
                error="UNKNOWN_ACTION",
            )
