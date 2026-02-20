"""Tests for the Atlas modes module.

Covers ModeType enum, ModeConfig dataclass, MODE_CONFIGS registry,
config helper functions, ModeManager state machine, and singleton lifecycle.
"""

import time
from unittest.mock import patch, MagicMock

import pytest

from atlas_brain.modes.config import (
    ModeType,
    ModeConfig,
    MODE_CONFIGS,
    SHARED_TOOLS,
    get_mode_config,
    get_mode_tools,
    get_all_tools,
)
from atlas_brain.modes.manager import (
    ModeManager,
    MODE_SWITCH_PATTERN,
    get_mode_manager,
    reset_mode_manager,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_singleton():
    """Reset the singleton before and after every test."""
    reset_mode_manager()
    yield
    reset_mode_manager()


@pytest.fixture
def manager():
    """Fresh ModeManager with default (HOME) mode."""
    return ModeManager()


# ---------------------------------------------------------------------------
# TestModeType
# ---------------------------------------------------------------------------


class TestModeType:
    """Tests for the ModeType enum."""

    def test_all_mode_values(self):
        expected = {
            "home": ModeType.HOME,
            "receptionist": ModeType.RECEPTIONIST,
            "comms": ModeType.COMMS,
            "security": ModeType.SECURITY,
            "chat": ModeType.CHAT,
        }
        actual = {m.value: m for m in ModeType}
        assert actual == expected

    def test_mode_from_string(self):
        assert ModeType("home") is ModeType.HOME
        assert ModeType("receptionist") is ModeType.RECEPTIONIST
        assert ModeType("comms") is ModeType.COMMS
        assert ModeType("security") is ModeType.SECURITY
        assert ModeType("chat") is ModeType.CHAT

    def test_mode_invalid_value(self):
        with pytest.raises(ValueError):
            ModeType("invalid")


# ---------------------------------------------------------------------------
# TestModeConfig
# ---------------------------------------------------------------------------


class TestModeConfig:
    """Tests for mode configuration data and helper functions."""

    def test_all_modes_have_config(self):
        for mode in ModeType:
            assert mode in MODE_CONFIGS, f"Missing config for {mode.value}"

    def test_shared_tools_content(self):
        assert SHARED_TOOLS == ["get_weather", "get_traffic", "get_location", "get_time"]
        assert len(SHARED_TOOLS) == 4

    def test_home_mode_tools(self):
        home_cfg = MODE_CONFIGS[ModeType.HOME]
        assert home_cfg.tools == [
            "lights_near_user",
            "media_near_user",
            "scene_near_user",
            "where_am_i",
        ]

    def test_chat_mode_empty_tools(self):
        chat_cfg = MODE_CONFIGS[ModeType.CHAT]
        assert chat_cfg.tools == []

    def test_get_mode_config_valid(self):
        cfg = get_mode_config(ModeType.SECURITY)
        assert cfg is MODE_CONFIGS[ModeType.SECURITY]
        assert cfg.name == "security"

    def test_get_mode_config_fallback(self):
        """If .get() misses (e.g. a fabricated key), fallback is CHAT config."""
        # MODE_CONFIGS.get uses ModeType keys; passing a non-ModeType
        # would miss. We simulate by calling the dict .get directly.
        sentinel = "not_a_mode"
        fallback = MODE_CONFIGS.get(sentinel, MODE_CONFIGS[ModeType.CHAT])
        assert fallback is MODE_CONFIGS[ModeType.CHAT]

    def test_get_mode_tools_with_shared(self):
        tools = get_mode_tools(ModeType.HOME, include_shared=True)
        # Must contain mode-specific tools followed by shared tools
        home_tools = MODE_CONFIGS[ModeType.HOME].tools
        assert tools[:len(home_tools)] == home_tools
        assert tools[len(home_tools):] == SHARED_TOOLS

    def test_get_mode_tools_without_shared(self):
        tools = get_mode_tools(ModeType.HOME, include_shared=False)
        assert tools == MODE_CONFIGS[ModeType.HOME].tools
        for st in SHARED_TOOLS:
            assert st not in tools

    def test_get_all_tools_contains_shared_and_specific(self):
        all_tools = get_all_tools()
        all_set = set(all_tools)
        # Every shared tool is present
        for t in SHARED_TOOLS:
            assert t in all_set
        # Every mode-specific tool is present
        for config in MODE_CONFIGS.values():
            for t in config.tools:
                assert t in all_set
        # No duplicates (returned as a set internally)
        assert len(all_tools) == len(all_set)


# ---------------------------------------------------------------------------
# TestModeManager
# ---------------------------------------------------------------------------


class TestModeManager:
    """Tests for the ModeManager state machine."""

    def test_default_mode_is_home(self, manager):
        assert manager.current_mode is ModeType.HOME

    def test_custom_default_mode(self):
        mgr = ModeManager(default_mode=ModeType.CHAT)
        assert mgr.current_mode is ModeType.CHAT

    def test_current_config_matches_mode(self, manager):
        cfg = manager.current_config
        assert cfg is get_mode_config(ModeType.HOME)
        assert cfg.name == "home"

    def test_previous_mode_starts_none(self, manager):
        assert manager.previous_mode is None

    def test_switch_mode_changes_mode(self, manager):
        manager.switch_mode(ModeType.SECURITY)
        assert manager.current_mode is ModeType.SECURITY

    def test_switch_mode_tracks_previous(self, manager):
        manager.switch_mode(ModeType.COMMS)
        assert manager.previous_mode is ModeType.HOME

    def test_switch_mode_same_mode_returns_false(self, manager):
        result = manager.switch_mode(ModeType.HOME)
        assert result is False

    def test_switch_mode_different_returns_true(self, manager):
        result = manager.switch_mode(ModeType.RECEPTIONIST)
        assert result is True

    def test_workflow_active_default_false(self, manager):
        assert manager.has_active_workflow is False

    def test_set_workflow_active(self, manager):
        manager.set_workflow_active(True)
        assert manager.has_active_workflow is True
        manager.set_workflow_active(False)
        assert manager.has_active_workflow is False

    def test_update_activity_updates_timestamp(self, manager):
        old_ts = manager._last_activity
        # Ensure enough time passes for a measurable difference
        time.sleep(0.01)
        manager.update_activity()
        assert manager._last_activity > old_ts

    # -- Timeout tests (require patching settings) --

    def test_check_timeout_no_timeout_in_default_mode(self, manager):
        """Already in default mode (home) -- timeout should not trigger."""
        mock_modes = MagicMock()
        mock_modes.default_mode = "home"
        mock_modes.timeout_seconds = 0  # immediate timeout
        with patch("atlas_brain.modes.manager.settings") as mock_settings:
            mock_settings.modes = mock_modes
            # Force elapsed time
            manager._last_activity = time.time() - 9999
            result = manager.check_timeout()
        assert result is False
        assert manager.current_mode is ModeType.HOME

    def test_check_timeout_no_timeout_when_workflow_active(self, manager):
        """Workflow active prevents timeout even if time exceeds limit."""
        manager.switch_mode(ModeType.RECEPTIONIST)
        manager.set_workflow_active(True)
        mock_modes = MagicMock()
        mock_modes.default_mode = "home"
        mock_modes.timeout_seconds = 1
        with patch("atlas_brain.modes.manager.settings") as mock_settings:
            mock_settings.modes = mock_modes
            manager._last_activity = time.time() - 100
            result = manager.check_timeout()
        assert result is False
        assert manager.current_mode is ModeType.RECEPTIONIST

    def test_check_timeout_triggers_after_elapsed(self, manager):
        """Mode should revert to default after inactivity exceeds timeout."""
        manager.switch_mode(ModeType.SECURITY)
        mock_modes = MagicMock()
        mock_modes.default_mode = "home"
        mock_modes.timeout_seconds = 5
        with patch("atlas_brain.modes.manager.settings") as mock_settings:
            mock_settings.modes = mock_modes
            manager._last_activity = time.time() - 10
            result = manager.check_timeout()
        assert result is True
        assert manager.current_mode is ModeType.HOME

    def test_check_timeout_does_not_trigger_within_limit(self, manager):
        """No timeout if elapsed time is within the limit."""
        manager.switch_mode(ModeType.COMMS)
        mock_modes = MagicMock()
        mock_modes.default_mode = "home"
        mock_modes.timeout_seconds = 600
        with patch("atlas_brain.modes.manager.settings") as mock_settings:
            mock_settings.modes = mock_modes
            manager._last_activity = time.time()
            result = manager.check_timeout()
        assert result is False
        assert manager.current_mode is ModeType.COMMS

    def test_check_timeout_exception_returns_false(self, manager):
        """If _check_timeout_inner raises, check_timeout returns False."""
        with patch.object(manager, "_check_timeout_inner", side_effect=RuntimeError("boom")):
            result = manager.check_timeout()
        assert result is False

    # -- switch_mode_by_name --

    def test_switch_mode_by_name_valid(self, manager):
        result = manager.switch_mode_by_name("security")
        assert result is True
        assert manager.current_mode is ModeType.SECURITY

    def test_switch_mode_by_name_invalid(self, manager):
        result = manager.switch_mode_by_name("nonexistent")
        assert result is False
        assert manager.current_mode is ModeType.HOME

    def test_switch_mode_by_name_case_insensitive(self, manager):
        result = manager.switch_mode_by_name("  COMMS  ")
        assert result is True
        assert manager.current_mode is ModeType.COMMS

    # -- parse_mode_switch (regex + aliases) --

    def test_parse_mode_switch_transition(self, manager):
        mode = manager.parse_mode_switch("transition to home mode")
        assert mode is ModeType.HOME

    def test_parse_mode_switch_switch_to(self, manager):
        mode = manager.parse_mode_switch("switch to security mode")
        assert mode is ModeType.SECURITY

    def test_parse_mode_switch_go_to(self, manager):
        mode = manager.parse_mode_switch("go to comms mode")
        assert mode is ModeType.COMMS

    def test_parse_mode_switch_atlas_prefix(self, manager):
        mode = manager.parse_mode_switch("atlas switch to chat mode")
        assert mode is ModeType.CHAT

    def test_parse_mode_switch_alias_business_to_receptionist(self, manager):
        mode = manager.parse_mode_switch("switch to business mode")
        assert mode is ModeType.RECEPTIONIST

    def test_parse_mode_switch_alias_personal_to_comms(self, manager):
        mode = manager.parse_mode_switch("go to personal mode")
        assert mode is ModeType.COMMS

    def test_parse_mode_switch_alias_camera_to_security(self, manager):
        mode = manager.parse_mode_switch("switch to camera mode")
        assert mode is ModeType.SECURITY

    def test_parse_mode_switch_alias_conversation_to_chat(self, manager):
        mode = manager.parse_mode_switch("transition to conversation mode")
        assert mode is ModeType.CHAT

    def test_parse_mode_switch_no_match(self, manager):
        result = manager.parse_mode_switch("just a normal sentence")
        assert result is None

    def test_parse_mode_switch_unknown_mode(self, manager):
        result = manager.parse_mode_switch("switch to spaceship mode")
        assert result is None

    # -- tools and model --

    def test_get_tools_for_current_mode(self, manager):
        tools = manager.get_tools_for_current_mode()
        expected = get_mode_tools(ModeType.HOME, include_shared=True)
        assert tools == expected

    def test_get_model_preference(self, manager):
        assert manager.get_model_preference() == "qwen3:8b"
        manager.switch_mode(ModeType.CHAT)
        assert manager.get_model_preference() == "cloud"

    # -- detect_mode_hint --

    def test_detect_mode_hint_home_keywords(self, manager):
        hint = manager.detect_mode_hint("please turn on the lights")
        assert hint is ModeType.HOME

    def test_detect_mode_hint_security_keywords(self, manager):
        hint = manager.detect_mode_hint("check the security camera for motion alerts")
        assert hint is ModeType.SECURITY

    def test_detect_mode_hint_no_match(self, manager):
        hint = manager.detect_mode_hint("hello how are you today")
        assert hint is None


# ---------------------------------------------------------------------------
# TestModeSingleton
# ---------------------------------------------------------------------------


class TestModeSingleton:
    """Tests for the module-level singleton lifecycle."""

    def test_get_mode_manager_creates_singleton(self):
        mgr = get_mode_manager()
        assert isinstance(mgr, ModeManager)

    def test_get_mode_manager_returns_same_instance(self):
        mgr1 = get_mode_manager()
        mgr2 = get_mode_manager()
        assert mgr1 is mgr2

    def test_reset_mode_manager_clears(self):
        mgr1 = get_mode_manager()
        reset_mode_manager()
        mgr2 = get_mode_manager()
        assert mgr1 is not mgr2
