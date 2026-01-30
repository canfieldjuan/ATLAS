"""
Tests for edge device pattern-based intent parsing.

Tests the DevicePatternParser for device command recognition.
"""

import pytest

from atlas_edge.intent.patterns import (
    DevicePatternParser,
    ParsedDeviceIntent,
    get_pattern_parser,
)


class TestDevicePatternParser:
    """Tests for DevicePatternParser."""

    @pytest.fixture
    def parser(self):
        """Create a fresh parser instance."""
        return DevicePatternParser()

    # --- Turn on/off patterns ---

    def test_turn_on_light(self, parser):
        """Test basic turn on light command."""
        result = parser.parse("turn on the kitchen light")
        assert result is not None
        assert result.action == "turn_on"
        assert result.target_type == "light"
        assert result.target_name == "kitchen"

    def test_turn_off_light(self, parser):
        """Test basic turn off light command."""
        result = parser.parse("turn off the living room lights")
        assert result is not None
        assert result.action == "turn_off"
        assert result.target_type == "light"
        assert "living room" in result.target_name

    def test_turn_on_without_article(self, parser):
        """Test turn on without 'the' article."""
        result = parser.parse("turn on bedroom light")
        assert result is not None
        assert result.action == "turn_on"
        assert "bedroom" in result.target_name

    def test_switch_on(self, parser):
        """Test switch on variant."""
        result = parser.parse("switch on the hallway light")
        assert result is not None
        assert result.action == "turn_on"
        assert result.target_type == "light"

    def test_flip_off(self, parser):
        """Test flip off variant."""
        result = parser.parse("flip off the garage light")
        assert result is not None
        assert result.action == "turn_off"

    def test_lights_on_reverse_order(self, parser):
        """Test 'kitchen lights on' word order."""
        result = parser.parse("kitchen lights on")
        assert result is not None
        assert result.action == "turn_on"
        assert result.target_type == "light"
        assert result.target_name == "kitchen"

    def test_lights_off_reverse_order(self, parser):
        """Test 'bedroom light off' word order."""
        result = parser.parse("bedroom light off")
        assert result is not None
        assert result.action == "turn_off"

    # --- Dim/brightness patterns ---

    def test_dim_to_percentage(self, parser):
        """Test dim to specific percentage."""
        result = parser.parse("dim the living room to 50%")
        assert result is not None
        assert result.action == "set_brightness"
        assert result.target_type == "light"
        assert result.target_name == "living room"
        assert result.parameters.get("brightness") == 50

    def test_dim_without_percent_sign(self, parser):
        """Test dim without percent sign."""
        result = parser.parse("dim the kitchen to 30")
        assert result is not None
        assert result.action == "set_brightness"
        assert result.parameters.get("brightness") == 30

    def test_set_brightness(self, parser):
        """Test set brightness command."""
        result = parser.parse("set the bedroom brightness to 75%")
        assert result is not None
        assert result.action == "set_brightness"
        assert result.parameters.get("brightness") == 75

    def test_brighten_light(self, parser):
        """Test brighten command."""
        result = parser.parse("brighten the office")
        assert result is not None
        assert result.action == "brighten"
        assert result.target_name == "office"
        assert result.parameters.get("amount") == 10  # default

    def test_brighten_by_amount(self, parser):
        """Test brighten by specific amount."""
        result = parser.parse("brighten the kitchen by 20%")
        assert result is not None
        assert result.action == "brighten"
        assert result.parameters.get("amount") == 20

    def test_dim_light(self, parser):
        """Test dim command without target percentage."""
        result = parser.parse("dim the bedroom")
        assert result is not None
        assert result.action == "dim"
        assert result.target_name == "bedroom"

    # --- Toggle patterns ---

    def test_toggle_light(self, parser):
        """Test toggle command."""
        result = parser.parse("toggle the porch light")
        assert result is not None
        assert result.action == "toggle"
        assert result.target_type == "light"
        assert result.target_name == "porch"

    # --- Fan patterns ---

    def test_turn_on_fan(self, parser):
        """Test turn on fan command."""
        result = parser.parse("turn on the bedroom fan")
        assert result is not None
        assert result.action == "turn_on"
        assert result.target_type == "fan"
        assert result.target_name == "bedroom"

    def test_fan_off_reverse_order(self, parser):
        """Test 'kitchen fan off' word order."""
        result = parser.parse("kitchen fan off")
        assert result is not None
        assert result.action == "turn_off"
        assert result.target_type == "fan"

    # --- Switch patterns ---

    def test_turn_on_switch(self, parser):
        """Test turn on switch command."""
        result = parser.parse("turn on the garage switch")
        assert result is not None
        assert result.action == "turn_on"
        assert result.target_type == "switch"

    # --- Scene patterns ---

    def test_activate_scene(self, parser):
        """Test activate scene command."""
        result = parser.parse("activate the movie scene")
        assert result is not None
        assert result.action == "turn_on"
        assert result.target_type == "scene"
        assert result.target_name == "movie"

    def test_set_scene(self, parser):
        """Test set scene command."""
        result = parser.parse("set the dinner scene")
        assert result is not None
        assert result.target_type == "scene"

    def test_scene_to_name(self, parser):
        """Test 'scene to X' pattern."""
        result = parser.parse("scene to relaxation")
        assert result is not None
        assert result.target_type == "scene"
        assert result.target_name == "relaxation"

    # --- Media player patterns ---

    def test_pause_media(self, parser):
        """Test pause command."""
        result = parser.parse("pause the living room tv")
        assert result is not None
        assert result.action == "pause"
        assert result.target_type == "media_player"

    def test_play_media(self, parser):
        """Test play command."""
        result = parser.parse("play the bedroom speaker")
        assert result is not None
        assert result.action == "play"
        assert result.target_type == "media_player"

    def test_mute_media(self, parser):
        """Test mute command."""
        result = parser.parse("mute the tv")
        assert result is not None
        assert result.action == "mute"
        assert result.target_type == "media_player"

    def test_turn_on_tv(self, parser):
        """Test turn on tv command."""
        result = parser.parse("turn on the tv")
        assert result is not None
        assert result.action == "turn_on"
        assert result.target_type == "media_player"

    def test_turn_on_tv_in_room(self, parser):
        """Test turn on tv in specific room."""
        result = parser.parse("turn on the tv in the bedroom")
        assert result is not None
        assert result.action == "turn_on"
        assert result.target_type == "media_player"
        assert result.target_name == "bedroom"

    # --- Volume patterns ---

    def test_set_volume(self, parser):
        """Test set volume command."""
        result = parser.parse("volume to 50%")
        assert result is not None
        assert result.action == "set_volume"
        assert result.parameters.get("volume") == 50

    def test_set_volume_on_device(self, parser):
        """Test set volume on specific device."""
        result = parser.parse("set volume to 30 on the bedroom speaker")
        assert result is not None
        assert result.action == "set_volume"
        assert result.parameters.get("volume") == 30
        assert result.target_name == "bedroom speaker"

    def test_volume_up(self, parser):
        """Test volume up command."""
        result = parser.parse("volume up")
        assert result is not None
        assert result.action == "volume_up"
        assert result.target_type == "media_player"

    def test_turn_volume_down(self, parser):
        """Test turn volume down command."""
        result = parser.parse("turn volume down")
        assert result is not None
        assert result.action == "volume_down"

    # --- Fallback generic pattern ---

    def test_generic_turn_on(self, parser):
        """Test generic turn on fallback."""
        result = parser.parse("turn on the humidifier")
        assert result is not None
        assert result.action == "turn_on"
        assert result.target_type == "device"
        assert result.target_name == "humidifier"

    # --- Negative tests ---

    def test_empty_string(self, parser):
        """Test empty string returns None."""
        result = parser.parse("")
        assert result is None

    def test_whitespace_only(self, parser):
        """Test whitespace only returns None."""
        result = parser.parse("   ")
        assert result is None

    def test_no_match(self, parser):
        """Test non-matching query returns None."""
        result = parser.parse("what is the weather today")
        assert result is None

    def test_partial_match_fails(self, parser):
        """Test partial command doesn't match."""
        result = parser.parse("turn the light")  # missing on/off
        assert result is None

    # --- can_parse method ---

    def test_can_parse_true(self, parser):
        """Test can_parse returns True for valid command."""
        assert parser.can_parse("turn on the kitchen light") is True

    def test_can_parse_false(self, parser):
        """Test can_parse returns False for invalid command."""
        assert parser.can_parse("hello how are you") is False

    def test_can_parse_empty(self, parser):
        """Test can_parse returns False for empty string."""
        assert parser.can_parse("") is False

    # --- Confidence ---

    def test_confidence_value(self, parser):
        """Test parsed intent has confidence value."""
        result = parser.parse("turn on the light")
        assert result is not None
        assert result.confidence == 0.9

    # --- Case insensitivity ---

    def test_case_insensitive_upper(self, parser):
        """Test parser handles uppercase."""
        result = parser.parse("TURN ON THE KITCHEN LIGHT")
        assert result is not None
        assert result.action == "turn_on"

    def test_case_insensitive_mixed(self, parser):
        """Test parser handles mixed case."""
        result = parser.parse("Turn Off The Living Room Lights")
        assert result is not None
        assert result.action == "turn_off"


class TestGetPatternParser:
    """Tests for pattern parser singleton."""

    def test_returns_parser(self):
        """Test get_pattern_parser returns a parser."""
        parser = get_pattern_parser()
        assert isinstance(parser, DevicePatternParser)

    def test_singleton_same_instance(self):
        """Test singleton returns same instance."""
        parser1 = get_pattern_parser()
        parser2 = get_pattern_parser()
        assert parser1 is parser2


class TestParsedDeviceIntent:
    """Tests for ParsedDeviceIntent dataclass."""

    def test_default_parameters(self):
        """Test default parameters is empty dict."""
        intent = ParsedDeviceIntent(
            action="turn_on",
            target_type="light",
        )
        assert intent.parameters == {}

    def test_parameters_preserved(self):
        """Test custom parameters are preserved."""
        intent = ParsedDeviceIntent(
            action="set_brightness",
            target_type="light",
            target_name="kitchen",
            parameters={"brightness": 50},
        )
        assert intent.parameters["brightness"] == 50
