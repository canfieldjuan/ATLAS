"""
Media device implementations.

Provides control for TVs, media players, and streaming devices.
"""

from typing import Any, Optional

from ..backends.roku import RokuBackend
from ..protocols import ActionResult, CapabilityType


class RokuTV:
    """Roku TV/streaming device capability."""

    SUPPORTED_ACTIONS = [
        "turn_on",
        "turn_off",
        "toggle",
        "get_state",
        "launch_app",
        "list_apps",
    ]

    def __init__(
        self,
        device_id: str,
        name: str,
        backend: RokuBackend,
    ):
        self._id = device_id
        self._name = name
        self._backend = backend

    @property
    def id(self) -> str:
        return self._id

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
            "id": self._id,
            "name": self._name,
            "type": "media_player",  # Report as media_player
            "supported_actions": self.supported_actions,
        }

    async def get_state(self) -> dict[str, Any]:
        """Get current power state."""
        power_mode = await self._backend.get_power_state()
        is_on = power_mode == "PowerOn"
        active_app = await self._backend.get_active_app() if is_on else None

        return {
            "is_on": is_on,
            "power_mode": power_mode,
            "active_app": active_app,
            "online": True,
        }

    async def execute_action(self, action: str, params: dict[str, Any]) -> ActionResult:
        """Execute an action on the Roku."""
        if action == "turn_on":
            success = await self._backend.power_on()
            return ActionResult(
                success=success,
                message=f"{self._name} turning on" if success else "Failed to power on",
            )

        elif action == "turn_off":
            success = await self._backend.power_off()
            return ActionResult(
                success=success,
                message=f"{self._name} turning off" if success else "Failed to power off",
            )

        elif action == "toggle":
            success = await self._backend.power_toggle()
            return ActionResult(
                success=success,
                message=f"{self._name} toggled" if success else "Failed to toggle power",
            )

        elif action == "get_state":
            state = await self.get_state()
            status = "on" if state["is_on"] else "off"
            app_info = ""
            if state.get("active_app"):
                app_info = f" (playing {state['active_app']['name']})"
            return ActionResult(
                success=True,
                message=f"{self._name} is {status}{app_info}",
                data=state,
            )

        elif action == "launch_app":
            app_id = params.get("app_id")
            if not app_id:
                return ActionResult(
                    success=False,
                    message="app_id parameter required",
                    error="MISSING_PARAM",
                )
            success = await self._backend.launch_app(app_id)
            return ActionResult(
                success=success,
                message=f"Launched app {app_id}" if success else "Failed to launch app",
            )

        elif action == "list_apps":
            apps = await self._backend.get_apps()
            return ActionResult(
                success=True,
                message=f"Found {len(apps)} apps",
                data={"apps": apps},
            )

        else:
            return ActionResult(
                success=False,
                message=f"Unknown action: {action}",
                error="UNKNOWN_ACTION",
            )
