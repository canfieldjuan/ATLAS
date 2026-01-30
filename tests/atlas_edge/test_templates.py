"""
Tests for edge device response templates.

Tests the ResponseTemplates class for generating user responses.
"""

import pytest

from atlas_edge.responses.templates import (
    ResponseTemplates,
    get_templates,
    OFFLINE_FALLBACK_MESSAGE,
    BRAIN_UNAVAILABLE_MESSAGE,
    ACTION_SUCCESS_TEMPLATES,
    ACTION_FAILURE_TEMPLATES,
)


class TestResponseTemplates:
    """Tests for ResponseTemplates class."""

    @pytest.fixture
    def short_templates(self):
        """Create templates with short responses."""
        return ResponseTemplates(use_short_responses=True)

    @pytest.fixture
    def long_templates(self):
        """Create templates with long responses."""
        return ResponseTemplates(use_short_responses=False)

    # --- Success responses ---

    def test_success_turn_on_short(self, short_templates):
        """Test short success response for turn_on."""
        response = short_templates.success_response("turn_on", "kitchen")
        assert response == "Done."

    def test_success_turn_on_long(self, long_templates):
        """Test long success response for turn_on."""
        response = long_templates.success_response("turn_on", "kitchen")
        assert "kitchen" in response
        assert "on" in response.lower()

    def test_success_turn_off_short(self, short_templates):
        """Test short success response for turn_off."""
        response = short_templates.success_response("turn_off", "bedroom")
        assert response == "Done."

    def test_success_turn_off_long(self, long_templates):
        """Test long success response for turn_off."""
        response = long_templates.success_response("turn_off", "bedroom")
        assert "bedroom" in response
        assert "off" in response.lower()

    def test_success_toggle(self, short_templates):
        """Test success response for toggle."""
        response = short_templates.success_response("toggle", "porch")
        assert response == "Done."

    def test_success_brightness_with_param(self, long_templates):
        """Test brightness response with parameter."""
        response = long_templates.success_response(
            "set_brightness",
            "office",
            brightness=75,
        )
        assert "75" in response

    def test_success_volume_with_param(self, long_templates):
        """Test volume response with parameter."""
        response = long_templates.success_response(
            "set_volume",
            "tv",
            volume=50,
        )
        assert "50" in response

    def test_success_play(self, short_templates):
        """Test success response for play."""
        response = short_templates.success_response("play", "speaker")
        assert "play" in response.lower()

    def test_success_pause(self, short_templates):
        """Test success response for pause."""
        response = short_templates.success_response("pause", "tv")
        assert "pause" in response.lower()

    def test_success_mute(self, short_templates):
        """Test success response for mute."""
        response = short_templates.success_response("mute", "tv")
        assert "mute" in response.lower()

    def test_success_volume_up(self, short_templates):
        """Test success response for volume_up."""
        response = short_templates.success_response("volume_up")
        assert "volume" in response.lower()

    def test_success_unknown_action(self, short_templates):
        """Test success response for unknown action defaults to Done."""
        response = short_templates.success_response("unknown_action", "device")
        assert response == "Done."

    def test_success_no_target(self, short_templates):
        """Test success response without target uses default."""
        response = short_templates.success_response("turn_on")
        assert response == "Done."

    # --- Failure responses ---

    def test_failure_entity_not_found(self, short_templates):
        """Test failure response for entity_not_found."""
        response = short_templates.failure_response("entity_not_found", "garage")
        assert "garage" in response
        assert "find" in response.lower() or "couldn" in response.lower()

    def test_failure_unavailable(self, short_templates):
        """Test failure response for unavailable device."""
        response = short_templates.failure_response("unavailable", "thermostat")
        assert "thermostat" in response
        assert "available" in response.lower()

    def test_failure_timeout(self, short_templates):
        """Test failure response for timeout."""
        response = short_templates.failure_response("timeout", "smart lock")
        assert "smart lock" in response
        assert "respond" in response.lower() or "time" in response.lower()

    def test_failure_permission(self, short_templates):
        """Test failure response for permission error."""
        response = short_templates.failure_response("permission", "alarm")
        assert "alarm" in response
        assert "permission" in response.lower()

    def test_failure_default(self, short_templates):
        """Test failure response for unknown error type."""
        response = short_templates.failure_response("unknown_error", "device")
        assert "device" in response
        assert "sorry" in response.lower() or "couldn" in response.lower()

    def test_failure_no_target(self, short_templates):
        """Test failure response without target uses default."""
        response = short_templates.failure_response("entity_not_found")
        assert "device" in response

    # --- Offline responses ---

    def test_offline_with_devices(self, short_templates):
        """Test offline response when devices still work."""
        response = short_templates.offline_response(can_do_devices=True)
        assert "brain" in response.lower() or "offline" in response.lower()
        assert "device" in response.lower()

    def test_offline_without_devices(self, short_templates):
        """Test offline response when devices don't work."""
        response = short_templates.offline_response(can_do_devices=False)
        assert "sorry" in response.lower()
        assert "later" in response.lower()

    # --- Device not found ---

    def test_device_not_found(self, short_templates):
        """Test device not found response."""
        response = short_templates.device_not_found("smart toaster")
        assert "smart toaster" in response
        assert "find" in response.lower() or "couldn" in response.lower()

    # --- Confirmation ---

    def test_confirmation_lights_on(self, short_templates):
        """Test confirmation for lights on."""
        response = short_templates.confirmation("lights_on")
        assert "light" in response.lower()
        assert "on" in response.lower()

    def test_confirmation_lights_off(self, short_templates):
        """Test confirmation for lights off."""
        response = short_templates.confirmation("lights_off")
        assert "light" in response.lower()
        assert "off" in response.lower()

    def test_confirmation_all_off(self, short_templates):
        """Test confirmation for all off."""
        response = short_templates.confirmation("all_off")
        assert "off" in response.lower()

    def test_confirmation_unknown(self, short_templates):
        """Test confirmation for unknown key."""
        response = short_templates.confirmation("unknown_key")
        assert response == "Done."


class TestGetTemplates:
    """Tests for get_templates singleton."""

    def test_returns_templates(self):
        """Test get_templates returns ResponseTemplates instance."""
        templates = get_templates()
        assert isinstance(templates, ResponseTemplates)

    def test_singleton_same_instance(self):
        """Test singleton returns same instance."""
        templates1 = get_templates()
        templates2 = get_templates()
        assert templates1 is templates2


class TestTemplateConstants:
    """Tests for template constants."""

    def test_offline_message_exists(self):
        """Test offline fallback message is defined."""
        assert OFFLINE_FALLBACK_MESSAGE
        assert "brain" in OFFLINE_FALLBACK_MESSAGE.lower()

    def test_brain_unavailable_exists(self):
        """Test brain unavailable message is defined."""
        assert BRAIN_UNAVAILABLE_MESSAGE
        assert "brain" in BRAIN_UNAVAILABLE_MESSAGE.lower()

    def test_success_templates_coverage(self):
        """Test success templates cover common actions."""
        required_actions = [
            "turn_on", "turn_off", "toggle",
            "set_brightness", "brighten", "dim",
            "play", "pause", "stop",
            "mute", "unmute", "set_volume",
        ]
        for action in required_actions:
            assert action in ACTION_SUCCESS_TEMPLATES, f"Missing: {action}"

    def test_failure_templates_coverage(self):
        """Test failure templates cover common errors."""
        required_errors = [
            "entity_not_found", "unavailable",
            "timeout", "permission", "default",
        ]
        for error in required_errors:
            assert error in ACTION_FAILURE_TEMPLATES, f"Missing: {error}"
