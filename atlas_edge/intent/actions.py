"""
Action intent dataclass for edge devices.

Defines the structure for parsed intents that can be executed locally.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ActionIntent:
    """
    Parsed action intent for edge execution.

    Represents a device command or local action that can be
    executed without brain connectivity.
    """

    # Core intent fields
    action: str  # turn_on, turn_off, toggle, set_brightness, etc.
    target_type: str  # light, switch, fan, media_player, scene, etc.
    target_name: Optional[str] = None  # kitchen, living room, etc.
    target_id: Optional[str] = None  # entity_id if resolved

    # Parameters for the action
    parameters: dict[str, Any] = field(default_factory=dict)

    # Metadata
    confidence: float = 0.0
    raw_query: str = ""
    source: str = "pattern"  # "pattern", "classifier", "brain"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action,
            "target_type": self.target_type,
            "target_name": self.target_name,
            "target_id": self.target_id,
            "parameters": self.parameters,
            "confidence": self.confidence,
            "raw_query": self.raw_query,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionIntent":
        """Create from dictionary."""
        return cls(
            action=data.get("action", ""),
            target_type=data.get("target_type", "device"),
            target_name=data.get("target_name"),
            target_id=data.get("target_id"),
            parameters=data.get("parameters", {}),
            confidence=data.get("confidence", 0.0),
            raw_query=data.get("raw_query", ""),
            source=data.get("source", "unknown"),
        )

    def is_device_command(self) -> bool:
        """Check if this is a device command."""
        device_actions = {
            "turn_on",
            "turn_off",
            "toggle",
            "set_brightness",
            "set_temperature",
            "set_volume",
            "volume_up",
            "volume_down",
            "play",
            "pause",
            "stop",
            "mute",
            "unmute",
            "brighten",
            "dim",
        }
        return self.action in device_actions

    def get_service_path(self) -> Optional[str]:
        """
        Get Home Assistant service path for this intent.

        Returns:
            Service path like "light/turn_on" or None
        """
        # Map actions to HA services
        action_map = {
            ("light", "turn_on"): "light/turn_on",
            ("light", "turn_off"): "light/turn_off",
            ("light", "toggle"): "light/toggle",
            ("light", "set_brightness"): "light/turn_on",
            ("light", "brighten"): "light/turn_on",
            ("light", "dim"): "light/turn_on",
            ("switch", "turn_on"): "switch/turn_on",
            ("switch", "turn_off"): "switch/turn_off",
            ("switch", "toggle"): "switch/toggle",
            ("fan", "turn_on"): "fan/turn_on",
            ("fan", "turn_off"): "fan/turn_off",
            ("fan", "toggle"): "fan/toggle",
            ("scene", "turn_on"): "scene/turn_on",
            ("media_player", "turn_on"): "media_player/turn_on",
            ("media_player", "turn_off"): "media_player/turn_off",
            ("media_player", "play"): "media_player/media_play",
            ("media_player", "pause"): "media_player/media_pause",
            ("media_player", "stop"): "media_player/media_stop",
            ("media_player", "mute"): "media_player/volume_mute",
            ("media_player", "unmute"): "media_player/volume_mute",
            ("media_player", "set_volume"): "media_player/volume_set",
            ("media_player", "volume_up"): "media_player/volume_up",
            ("media_player", "volume_down"): "media_player/volume_down",
            ("device", "turn_on"): "homeassistant/turn_on",
            ("device", "turn_off"): "homeassistant/turn_off",
            ("device", "toggle"): "homeassistant/toggle",
        }

        return action_map.get((self.target_type, self.action))

    def get_service_data(self) -> dict[str, Any]:
        """
        Get Home Assistant service data for this intent.

        Returns:
            Service data dict with entity_id and parameters
        """
        data = {}

        if self.target_id:
            data["entity_id"] = self.target_id

        # Add action-specific parameters
        if self.action == "set_brightness":
            brightness = self.parameters.get("brightness", 100)
            data["brightness_pct"] = min(100, max(0, brightness))

        elif self.action == "brighten":
            # Relative brightness increase
            amount = self.parameters.get("amount", 10)
            data["brightness_step_pct"] = amount

        elif self.action == "dim":
            # Relative brightness decrease
            amount = self.parameters.get("amount", 10)
            data["brightness_step_pct"] = -amount

        elif self.action == "set_volume":
            volume = self.parameters.get("volume", 50)
            data["volume_level"] = min(1.0, max(0.0, volume / 100.0))

        elif self.action in ("mute", "unmute"):
            data["is_volume_muted"] = self.action == "mute"

        return data
