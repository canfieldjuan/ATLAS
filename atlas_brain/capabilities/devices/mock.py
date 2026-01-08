"""
Mock device implementations for testing.

These devices simulate real devices without requiring backends.
"""

import logging
from typing import Any, Optional

from ..protocols import ActionResult, CapabilityType, LightState, SwitchState

logger = logging.getLogger("atlas.capabilities.devices.mock")


class MockLight:
    """
    Mock light for testing the action pipeline.

    Simulates a dimmable light without requiring MQTT or Home Assistant.
    """

    SUPPORTED_ACTIONS = ["turn_on", "turn_off", "toggle", "set_brightness"]

    def __init__(self, device_id: str, name: str):
        self._id = device_id
        self._name = name
        self._state = LightState(is_on=False, brightness=255, online=True)

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

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
            "state": {
                "is_on": self._state.is_on,
                "brightness": self._state.brightness,
            },
        }

    async def get_state(self) -> LightState:
        """Return current simulated state."""
        return self._state

    async def execute_action(self, action: str, params: dict[str, Any]) -> ActionResult:
        """Execute an action on the mock light."""
        if action == "turn_on":
            brightness = params.get("brightness")
            self._state.is_on = True
            if brightness is not None:
                self._state.brightness = brightness
            logger.info("[MOCK] %s turned ON (brightness=%s)", self.name, self._state.brightness)
            return ActionResult(
                success=True,
                message=f"{self.name} turned on",
                data={"brightness": self._state.brightness},
            )

        elif action == "turn_off":
            self._state.is_on = False
            logger.info("[MOCK] %s turned OFF", self.name)
            return ActionResult(
                success=True,
                message=f"{self.name} turned off",
            )

        elif action == "toggle":
            self._state.is_on = not self._state.is_on
            state_str = "on" if self._state.is_on else "off"
            logger.info("[MOCK] %s toggled to %s", self.name, state_str)
            return ActionResult(
                success=True,
                message=f"{self.name} toggled {state_str}",
                data={"is_on": self._state.is_on},
            )

        elif action == "set_brightness":
            brightness = params.get("brightness", 255)
            self._state.brightness = brightness
            self._state.is_on = True
            logger.info("[MOCK] %s brightness set to %d", self.name, brightness)
            return ActionResult(
                success=True,
                message=f"{self.name} brightness set to {brightness}",
                data={"brightness": brightness},
            )

        return ActionResult(
            success=False,
            message=f"Unknown action: {action}",
            error="UNKNOWN_ACTION",
        )


class MockSwitch:
    """
    Mock switch for testing.

    Simulates a simple on/off switch.
    """

    SUPPORTED_ACTIONS = ["turn_on", "turn_off", "toggle"]

    def __init__(self, device_id: str, name: str):
        self._id = device_id
        self._name = name
        self._state = SwitchState(is_on=False, online=True)

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

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
            "state": {"is_on": self._state.is_on},
        }

    async def get_state(self) -> SwitchState:
        return self._state

    async def execute_action(self, action: str, params: dict[str, Any]) -> ActionResult:
        """Execute an action on the mock switch."""
        if action == "turn_on":
            self._state.is_on = True
            logger.info("[MOCK] %s turned ON", self.name)
            return ActionResult(
                success=True,
                message=f"{self.name} turned on",
            )

        elif action == "turn_off":
            self._state.is_on = False
            logger.info("[MOCK] %s turned OFF", self.name)
            return ActionResult(
                success=True,
                message=f"{self.name} turned off",
            )

        elif action == "toggle":
            self._state.is_on = not self._state.is_on
            state_str = "on" if self._state.is_on else "off"
            logger.info("[MOCK] %s toggled to %s", self.name, state_str)
            return ActionResult(
                success=True,
                message=f"{self.name} toggled {state_str}",
                data={"is_on": self._state.is_on},
            )

        return ActionResult(
            success=False,
            message=f"Unknown action: {action}",
            error="UNKNOWN_ACTION",
        )


def register_test_devices() -> list[str]:
    """
    Register mock devices for testing.

    Returns list of registered device IDs.
    """
    from ..registry import capability_registry

    devices = [
        MockLight("living_room_light", "Living Room Light"),
        MockLight("bedroom_light", "Bedroom Light"),
        MockLight("kitchen_light", "Kitchen Light"),
        MockSwitch("garage_door", "Garage Door"),
        MockSwitch("fan_switch", "Ceiling Fan"),
    ]

    registered = []
    for device in devices:
        capability_registry.register(device)
        registered.append(device.id)

    logger.info("Registered %d test devices", len(registered))
    return registered
