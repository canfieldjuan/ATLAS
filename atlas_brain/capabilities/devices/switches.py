"""
Switch device implementations.

Provides switch/relay control through MQTT and Home Assistant backends.
"""

from typing import Any

from ..backends.homeassistant import HomeAssistantBackend
from ..backends.mqtt import MQTTBackend
from ..protocols import ActionResult, CapabilityType, SwitchState


class BaseSwitchCapability:
    """Base class for switch capabilities with shared functionality."""

    SUPPORTED_ACTIONS = ["turn_on", "turn_off", "toggle"]

    @property
    def capability_type(self) -> CapabilityType:
        return CapabilityType.SWITCH

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
            return await self.turn_on()
        elif action == "turn_off":
            return await self.turn_off()
        elif action == "toggle":
            state = await self.get_state()
            if state.is_on:
                return await self.turn_off()
            else:
                return await self.turn_on()
        else:
            return ActionResult(
                success=False,
                message=f"Unknown action: {action}",
                error="UNKNOWN_ACTION",
            )


class MQTTSwitch(BaseSwitchCapability):
    """
    Switch capability backed by MQTT.

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
        self._state = SwitchState()

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    async def get_state(self) -> SwitchState:
        """Query state from MQTT (uses cached state from subscriptions)."""
        raw_state = await self._backend.get_state(self._state_topic)
        if raw_state:
            self._state.is_on = raw_state.get("POWER", "OFF") == "ON"
            self._state.online = True
        return self._state

    async def turn_on(self) -> ActionResult:
        """Turn on the switch."""
        await self._backend.send_command(self._command_topic, {"POWER": "ON"})
        self._state.is_on = True

        return ActionResult(
            success=True,
            message=f"{self.name} turned on",
        )

    async def turn_off(self) -> ActionResult:
        """Turn off the switch."""
        await self._backend.send_command(self._command_topic, {"POWER": "OFF"})
        self._state.is_on = False

        return ActionResult(
            success=True,
            message=f"{self.name} turned off",
        )


class HomeAssistantSwitch(BaseSwitchCapability):
    """Switch capability backed by Home Assistant.

    Supports switch.*, input_boolean.*, and other binary entities.
    """

    def __init__(
        self,
        entity_id: str,
        name: str,
        backend: HomeAssistantBackend,
    ):
        self._entity_id = entity_id
        self._name = name
        self._backend = backend
        # Extract domain for service calls (e.g., "switch" or "input_boolean")
        self._domain = entity_id.split(".", 1)[0]

    @property
    def id(self) -> str:
        return self._entity_id

    @property
    def name(self) -> str:
        return self._name

    async def get_state(self) -> SwitchState:
        """Get state from Home Assistant."""
        raw = await self._backend.get_state(self._entity_id)
        return SwitchState(
            is_on=raw.get("state") == "on",
            online=True,
            last_updated=raw.get("last_updated"),
        )

    async def turn_on(self) -> ActionResult:
        """Turn on the switch."""
        await self._backend.send_command(
            f"{self._domain}/turn_on",
            {"entity_id": self._entity_id},
        )
        return ActionResult(
            success=True,
            message=f"{self.name} turned on",
        )

    async def turn_off(self) -> ActionResult:
        """Turn off the switch."""
        await self._backend.send_command(
            f"{self._domain}/turn_off",
            {"entity_id": self._entity_id},
        )
        return ActionResult(
            success=True,
            message=f"{self.name} turned off",
        )
