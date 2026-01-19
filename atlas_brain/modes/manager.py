"""
Mode manager for Atlas.

Handles mode switching, tool filtering, and model preferences.
"""

import logging
import re
import time
from typing import Optional

from ..config import settings

from .config import (
    ModeType,
    ModeConfig,
    MODE_CONFIGS,
    SHARED_TOOLS,
    get_mode_config,
    get_mode_tools,
)

logger = logging.getLogger("atlas.modes.manager")

# Pattern for explicit mode switch commands
MODE_SWITCH_PATTERN = re.compile(
    r"(?:atlas\s+)?(?:transition|switch|go)\s+to\s+(\w+)\s+mode",
    re.IGNORECASE,
)


class ModeManager:
    """
    Manages Atlas operating modes.

    Handles:
    - Current mode tracking
    - Mode switching (explicit commands)
    - Tool filtering based on current mode
    - Model preferences per mode
    """

    def __init__(self, default_mode: ModeType = ModeType.HOME):
        self._current_mode = default_mode
        self._previous_mode: Optional[ModeType] = None
        self._last_activity: float = time.time()
        self._workflow_active: bool = False
        logger.info("ModeManager initialized with mode: %s", default_mode.value)

    @property
    def current_mode(self) -> ModeType:
        """Get current operating mode."""
        return self._current_mode

    @property
    def current_config(self) -> ModeConfig:
        """Get configuration for current mode."""
        return get_mode_config(self._current_mode)

    @property
    def previous_mode(self) -> Optional[ModeType]:
        """Get previous mode (before last switch)."""
        return self._previous_mode

    def switch_mode(self, new_mode: ModeType) -> bool:
        """
        Switch to a new mode.

        Returns True if mode changed, False if already in that mode.
        """
        if new_mode == self._current_mode:
            logger.debug("Already in %s mode", new_mode.value)
            return False

        self._previous_mode = self._current_mode
        self._current_mode = new_mode
        logger.info(
            "Mode switched: %s -> %s",
            self._previous_mode.value,
            new_mode.value,
        )
        return True

    @property
    def has_active_workflow(self) -> bool:
        """Check if there is an active workflow that should prevent timeout."""
        return self._workflow_active

    def set_workflow_active(self, active: bool) -> None:
        """Set workflow active state (prevents timeout during multi-turn workflows)."""
        self._workflow_active = active
        if active:
            logger.debug("Workflow active - timeout disabled")
        else:
            logger.debug("Workflow complete - timeout enabled")

    def update_activity(self) -> None:
        """Update last activity timestamp (call on each user interaction)."""
        self._last_activity = time.time()

    def check_timeout(self) -> bool:
        """
        Check if mode should timeout and fall back to default.

        Returns True if timeout triggered and mode switched.
        Does NOT timeout if:
        - Already in HOME (default) mode
        - Workflow is active
        """
        # Get default mode from config
        try:
            default_mode = ModeType(settings.modes.default_mode)
        except ValueError:
            default_mode = ModeType.HOME

        if self._current_mode == default_mode:
            return False

        if self._workflow_active:
            return False

        timeout_seconds = settings.modes.timeout_seconds
        elapsed = time.time() - self._last_activity

        if elapsed > timeout_seconds:
            logger.info(
                "Mode timeout after %.0fs inactivity - switching to %s",
                elapsed,
                default_mode.value,
            )
            self.switch_mode(default_mode)
            return True

        return False

    def switch_mode_by_name(self, mode_name: str) -> bool:
        """Switch mode by name string."""
        mode_name = mode_name.lower().strip()
        try:
            new_mode = ModeType(mode_name)
            return self.switch_mode(new_mode)
        except ValueError:
            logger.warning("Unknown mode: %s", mode_name)
            return False

    def parse_mode_switch(self, text: str) -> Optional[ModeType]:
        """
        Check if text contains a mode switch command.

        Matches patterns like:
        - "Atlas transition to schedule mode"
        - "switch to business mode"
        - "go to home mode"

        Returns the requested mode or None.
        """
        match = MODE_SWITCH_PATTERN.search(text)
        if not match:
            return None

        mode_name = match.group(1).lower()

        # Handle aliases
        aliases = {
            # Home
            "smart home": "home",
            "device": "home",
            "devices": "home",
            # Receptionist (business)
            "business": "receptionist",
            "scheduling": "receptionist",
            "appointment": "receptionist",
            "appointments": "receptionist",
            # Comms (personal)
            "communications": "comms",
            "personal": "comms",
            "friends": "comms",
            "family": "comms",
            # Security
            "camera": "security",
            "cameras": "security",
            "surveillance": "security",
            # Chat
            "conversation": "chat",
            "general": "chat",
        }
        mode_name = aliases.get(mode_name, mode_name)

        try:
            return ModeType(mode_name)
        except ValueError:
            logger.warning("Unrecognized mode in switch command: %s", mode_name)
            return None

    def get_tools_for_current_mode(self) -> list[str]:
        """Get list of tool names available in current mode."""
        return get_mode_tools(self._current_mode, include_shared=True)

    def get_model_preference(self) -> Optional[str]:
        """Get preferred model for current mode."""
        return self.current_config.model_preference

    def detect_mode_hint(self, text: str) -> Optional[ModeType]:
        """
        Detect which mode might be relevant based on keywords.

        This is for hinting, not automatic switching.
        Returns None if no strong hint detected.
        """
        text_lower = text.lower()

        # Score each mode by keyword matches
        scores: dict[ModeType, int] = {}
        for mode, config in MODE_CONFIGS.items():
            score = sum(1 for kw in config.keywords if kw in text_lower)
            if score > 0:
                scores[mode] = score

        if not scores:
            return None

        # Return highest scoring mode
        return max(scores, key=scores.get)


# Global singleton instance
_mode_manager: Optional[ModeManager] = None


def get_mode_manager() -> ModeManager:
    """Get or create the global mode manager."""
    global _mode_manager
    if _mode_manager is None:
        _mode_manager = ModeManager()
    return _mode_manager


def reset_mode_manager() -> None:
    """Reset the global mode manager."""
    global _mode_manager
    _mode_manager = None
