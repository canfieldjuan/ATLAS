"""
Atlas Modes - Modular operating modes with tool groupings.

Each mode has:
- Specific set of tools optimized for the use case
- Optional model preference (smaller models for simple tasks)
- Keywords for mode detection hints
"""

from .config import (
    ModeType,
    ModeConfig,
    MODE_CONFIGS,
    SHARED_TOOLS,
    get_mode_config,
    get_mode_tools,
    get_all_tools,
)
from .manager import (
    ModeManager,
    get_mode_manager,
    reset_mode_manager,
)

__all__ = [
    # Config
    "ModeType",
    "ModeConfig",
    "MODE_CONFIGS",
    "SHARED_TOOLS",
    "get_mode_config",
    "get_mode_tools",
    "get_all_tools",
    # Manager
    "ModeManager",
    "get_mode_manager",
    "reset_mode_manager",
]
