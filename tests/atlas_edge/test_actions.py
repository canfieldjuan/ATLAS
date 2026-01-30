"""
Tests for edge device ActionIntent dataclass.

Tests the ActionIntent structure and its methods.
"""

import pytest

from atlas_edge.intent.actions import ActionIntent


class TestActionIntent:
    """Tests for ActionIntent dataclass."""

    # --- Basic construction ---

    def test_basic_creation(self):
        """Test basic ActionIntent creation."""
        intent = ActionIntent(
            action="turn_on",
            target_type="light",
            target_name="kitchen",
        )
        assert intent.action == "turn_on"
        assert intent.target_type == "light"
        assert intent.target_name == "kitchen"

    def test_default_values(self):
        """Test default values are set correctly."""
        intent = ActionIntent(action="turn_on", target_type="light")
        assert intent.target_name is None
        assert intent.target_id is None
        assert intent.parameters == {}
        assert intent.confidence == 0.0
        assert intent.raw_query == ""
        assert intent.source == "pattern"

    def test_custom_values(self):
        """Test custom values are preserved."""
        intent = ActionIntent(
            action="set_brightness",
            target_type="light",
            target_name="bedroom",
            target_id="light.bedroom_main",
            parameters={"brightness": 75},
            confidence=0.95,
            raw_query="dim bedroom to 75",
            source="classifier",
        )
        assert intent.target_id == "light.bedroom_main"
        assert intent.parameters["brightness"] == 75
        assert intent.confidence == 0.95
        assert intent.source == "classifier"

    # --- to_dict / from_dict ---

    def test_to_dict(self):
        """Test conversion to dictionary."""
        intent = ActionIntent(
            action="turn_off",
            target_type="switch",
            target_name="garage",
            confidence=0.9,
        )
        data = intent.to_dict()
        assert data["action"] == "turn_off"
        assert data["target_type"] == "switch"
        assert data["target_name"] == "garage"
        assert data["confidence"] == 0.9

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "action": "toggle",
            "target_type": "fan",
            "target_name": "office",
            "parameters": {"speed": "high"},
            "confidence": 0.85,
        }
        intent = ActionIntent.from_dict(data)
        assert intent.action == "toggle"
        assert intent.target_type == "fan"
        assert intent.target_name == "office"
        assert intent.parameters["speed"] == "high"

    def test_from_dict_defaults(self):
        """Test from_dict uses defaults for missing keys."""
        data = {"action": "turn_on"}
        intent = ActionIntent.from_dict(data)
        assert intent.action == "turn_on"
        assert intent.target_type == "device"  # default
        assert intent.target_name is None
        assert intent.source == "unknown"  # default for from_dict

    def test_roundtrip(self):
        """Test to_dict -> from_dict roundtrip."""
        original = ActionIntent(
            action="set_volume",
            target_type="media_player",
            target_name="living room tv",
            target_id="media_player.living_room_tv",
            parameters={"volume": 30},
            confidence=0.88,
            raw_query="volume to 30",
            source="pattern",
        )
        data = original.to_dict()
        restored = ActionIntent.from_dict(data)
        assert restored.action == original.action
        assert restored.target_type == original.target_type
        assert restored.target_name == original.target_name
        assert restored.target_id == original.target_id
        assert restored.parameters == original.parameters

    # --- is_device_command ---

    def test_is_device_command_true(self):
        """Test is_device_command returns True for device actions."""
        device_actions = [
            "turn_on", "turn_off", "toggle",
            "set_brightness", "brighten", "dim",
            "set_volume", "volume_up", "volume_down",
            "play", "pause", "stop", "mute", "unmute",
        ]
        for action in device_actions:
            intent = ActionIntent(action=action, target_type="device")
            assert intent.is_device_command() is True, f"{action} should be device cmd"

    def test_is_device_command_false(self):
        """Test is_device_command returns False for non-device actions."""
        non_device_actions = ["query", "search", "unknown", "chat"]
        for action in non_device_actions:
            intent = ActionIntent(action=action, target_type="device")
            assert intent.is_device_command() is False

    # --- get_service_path ---

    def test_service_path_light_on(self):
        """Test service path for light turn_on."""
        intent = ActionIntent(action="turn_on", target_type="light")
        assert intent.get_service_path() == "light/turn_on"

    def test_service_path_light_off(self):
        """Test service path for light turn_off."""
        intent = ActionIntent(action="turn_off", target_type="light")
        assert intent.get_service_path() == "light/turn_off"

    def test_service_path_light_toggle(self):
        """Test service path for light toggle."""
        intent = ActionIntent(action="toggle", target_type="light")
        assert intent.get_service_path() == "light/toggle"

    def test_service_path_brightness(self):
        """Test service path for set_brightness uses turn_on."""
        intent = ActionIntent(action="set_brightness", target_type="light")
        assert intent.get_service_path() == "light/turn_on"

    def test_service_path_switch(self):
        """Test service path for switch."""
        intent = ActionIntent(action="turn_on", target_type="switch")
        assert intent.get_service_path() == "switch/turn_on"

    def test_service_path_fan(self):
        """Test service path for fan."""
        intent = ActionIntent(action="turn_off", target_type="fan")
        assert intent.get_service_path() == "fan/turn_off"

    def test_service_path_scene(self):
        """Test service path for scene."""
        intent = ActionIntent(action="turn_on", target_type="scene")
        assert intent.get_service_path() == "scene/turn_on"

    def test_service_path_media_play(self):
        """Test service path for media play."""
        intent = ActionIntent(action="play", target_type="media_player")
        assert intent.get_service_path() == "media_player/media_play"

    def test_service_path_media_pause(self):
        """Test service path for media pause."""
        intent = ActionIntent(action="pause", target_type="media_player")
        assert intent.get_service_path() == "media_player/media_pause"

    def test_service_path_media_volume(self):
        """Test service path for volume set."""
        intent = ActionIntent(action="set_volume", target_type="media_player")
        assert intent.get_service_path() == "media_player/volume_set"

    def test_service_path_generic_device(self):
        """Test service path for generic device."""
        intent = ActionIntent(action="turn_on", target_type="device")
        assert intent.get_service_path() == "homeassistant/turn_on"

    def test_service_path_unknown(self):
        """Test service path for unknown combination."""
        intent = ActionIntent(action="unknown_action", target_type="unknown_type")
        assert intent.get_service_path() is None

    # --- get_service_data ---

    def test_service_data_with_entity_id(self):
        """Test service data includes entity_id."""
        intent = ActionIntent(
            action="turn_on",
            target_type="light",
            target_id="light.kitchen",
        )
        data = intent.get_service_data()
        assert data["entity_id"] == "light.kitchen"

    def test_service_data_without_entity_id(self):
        """Test service data without entity_id."""
        intent = ActionIntent(action="turn_on", target_type="light")
        data = intent.get_service_data()
        assert "entity_id" not in data

    def test_service_data_brightness(self):
        """Test service data for set_brightness."""
        intent = ActionIntent(
            action="set_brightness",
            target_type="light",
            target_id="light.bedroom",
            parameters={"brightness": 75},
        )
        data = intent.get_service_data()
        assert data["entity_id"] == "light.bedroom"
        assert data["brightness_pct"] == 75

    def test_service_data_brightness_clamped(self):
        """Test brightness is clamped to 0-100."""
        intent = ActionIntent(
            action="set_brightness",
            target_type="light",
            parameters={"brightness": 150},
        )
        data = intent.get_service_data()
        assert data["brightness_pct"] == 100

        intent2 = ActionIntent(
            action="set_brightness",
            target_type="light",
            parameters={"brightness": -10},
        )
        data2 = intent2.get_service_data()
        assert data2["brightness_pct"] == 0

    def test_service_data_brighten(self):
        """Test service data for brighten action."""
        intent = ActionIntent(
            action="brighten",
            target_type="light",
            parameters={"amount": 20},
        )
        data = intent.get_service_data()
        assert data["brightness_step_pct"] == 20

    def test_service_data_dim(self):
        """Test service data for dim action."""
        intent = ActionIntent(
            action="dim",
            target_type="light",
            parameters={"amount": 15},
        )
        data = intent.get_service_data()
        assert data["brightness_step_pct"] == -15

    def test_service_data_volume(self):
        """Test service data for set_volume."""
        intent = ActionIntent(
            action="set_volume",
            target_type="media_player",
            parameters={"volume": 50},
        )
        data = intent.get_service_data()
        assert data["volume_level"] == 0.5

    def test_service_data_volume_clamped(self):
        """Test volume is clamped to 0.0-1.0."""
        intent = ActionIntent(
            action="set_volume",
            target_type="media_player",
            parameters={"volume": 150},
        )
        data = intent.get_service_data()
        assert data["volume_level"] == 1.0

    def test_service_data_mute(self):
        """Test service data for mute."""
        intent = ActionIntent(action="mute", target_type="media_player")
        data = intent.get_service_data()
        assert data["is_volume_muted"] is True

    def test_service_data_unmute(self):
        """Test service data for unmute."""
        intent = ActionIntent(action="unmute", target_type="media_player")
        data = intent.get_service_data()
        assert data["is_volume_muted"] is False
