"""
Atlas Tools - Information query tools.

Tools are functions that retrieve information (weather, traffic, etc.)
as opposed to device control capabilities.
"""

from .base import Tool, ToolParameter, ToolResult
from .registry import ToolRegistry, tool_registry
from .weather import WeatherTool, weather_tool
from .traffic import TrafficTool, traffic_tool
from .location import LocationTool, location_tool

# Register tools on import
tool_registry.register(weather_tool)
tool_registry.register(traffic_tool)
tool_registry.register(location_tool)

__all__ = [
    "Tool",
    "ToolParameter",
    "ToolResult",
    "ToolRegistry",
    "tool_registry",
    "WeatherTool",
    "weather_tool",
    "TrafficTool",
    "traffic_tool",
    "LocationTool",
    "location_tool",
]
