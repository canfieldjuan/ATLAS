"""
Home mode tools for smart home device control.

These tools hide location from the LLM - the tool resolves which devices
based on user's current room.
"""

from .presence import (
    LightsNearUserTool,
    lights_near_user,
    MediaNearUserTool,
    media_near_user,
    SceneNearUserTool,
    scene_near_user,
    WhereAmITool,
    where_am_i,
)

__all__ = [
    "LightsNearUserTool",
    "lights_near_user",
    "MediaNearUserTool",
    "media_near_user",
    "SceneNearUserTool",
    "scene_near_user",
    "WhereAmITool",
    "where_am_i",
]
