"""
Light device implementations.

Provides light control through MQTT and Home Assistant backends.
"""

from typing import Any, Optional

from ..backends.homeassistant import HomeAssistantBackend
from ..backends.mqtt import MQTTBackend
from ..protocols import ActionResult, CapabilityType, LightState


class BaseLightCapability:
    """Base class for light capabilities with shared functionality."""

    SUPPORTED_ACTIONS = ["turn_on", "turn_off", "toggle", "set_brightness"]

    @property
    def capability_type(self) -> CapabilityType:
        return CapabilityType.LIGHT

    @property
    def supported_actions(self) -> list[str]:
        return self.SUPPORTED_ACTIONS

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.capability_type.value,
            "supported_actions": self.supported_actions,
        }

    async def execute_action(self, action: str, params: dict[str, Any]) -> ActionResult:
        """Route action to specific method."""
        if action == "turn_on":
            return await self.turn_on(params.get("brightness"))
        elif action == "turn_off":
            return await self.turn_off()
        elif action == "toggle":
            state = await self.get_state()
            if state.is_on:
                return await self.turn_off()
            else:
                return await self.turn_on()
        elif action == "set_brightness":
            brightness = params.get("brightness", 255)
            return await self.turn_on(brightness)
        else:
            return ActionResult(
                success=False,
                message=f"Unknown action: {action}",
                error="UNKNOWN_ACTION",
            )


class MQTTLight(BaseLightCapability):
    """
    Light capability backed by MQTT.

    Compatible with devices like Tasmota, Shelly, or custom ESP devices.
    """

    def __init__(
        self,
        device_id: str,
        name: str,
        backend: MQTTBackend,
        command_topic: str,
        state_topic: str,
    ):
        self._id = device_id
        self._name = name
        self._backend = backend
        self._command_topic = command_topic
        self._state_topic = state_topic
        self._state = LightState()

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    async def get_state(self) -> LightState:
        """Query state from MQTT (uses cached state from subscriptions)."""
        raw_state = await self._backend.get_state(self._state_topic)
        if raw_state:
            self._state.is_on = raw_state.get("POWER", "OFF") == "ON"
            self._state.brightness = raw_state.get("Dimmer")
            self._state.online = True
        return self._state

    async def turn_on(self, brightness: Optional[int] = None) -> ActionResult:
        """Turn on the light with optional brightness."""
        payload = {"POWER": "ON"}
        if brightness is not None:
            payload["Dimmer"] = brightness

        await self._backend.send_command(self._command_topic, payload)
        self._state.is_on = True
        if brightness:
            self._state.brightness = brightness

        return ActionResult(
            success=True,
            message=f"{self.name} turned on",
            data={"brightness": brightness},
        )

    async def turn_off(self) -> ActionResult:
        """Turn off the light."""
        await self._backend.send_command(self._command_topic, {"POWER": "OFF"})
        self._state.is_on = False

        return ActionResult(
            success=True,
            message=f"{self.name} turned off",
        )


class HomeAssistantLight(BaseLightCapability):
    """Light capability backed by Home Assistant."""

    def __init__(
        self,
        entity_id: str,
        name: str,
        backend: HomeAssistantBackend,
    ):
        self._entity_id = entity_id
        self._name = name
        self._backend = backend

    @property
    def id(self) -> str:
        return self._entity_id

    @property
    def name(self) -> str:
        return self._name

    async def get_state(self) -> LightState:
        """Get state from Home Assistant."""
        raw = await self._backend.get_state(self._entity_id)
        return LightState(
            is_on=raw.get("state") == "on",
            brightness=raw.get("attributes", {}).get("brightness"),
            online=True,
            last_updated=raw.get("last_updated"),
            attributes=raw.get("attributes", {}),
        )

    async def turn_on(self, brightness: Optional[int] = None) -> ActionResult:
        """Turn on the light with optional brightness."""
        payload: dict[str, Any] = {"entity_id": self._entity_id}
        if brightness is not None:
            payload["brightness"] = brightness

        await self._backend.send_command("light/turn_on", payload)
        return ActionResult(
            success=True,
            message=f"{self.name} turned on",
            data={"brightness": brightness},
        )

    async def turn_off(self) -> ActionResult:
        """Turn off the light."""
        await self._backend.send_command(
            "light/turn_off",
            {"entity_id": self._entity_id},
        )
        return ActionResult(
            success=True,
            message=f"{self.name} turned off",
        )
