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
from .time import TimeTool, time_tool
from .calendar import CalendarTool, calendar_tool
from .reminder import (
    ReminderTool,
    reminder_tool,
    ListRemindersTool,
    list_reminders_tool,
    CompleteReminderTool,
    complete_reminder_tool,
)

# Register tools on import
tool_registry.register(weather_tool)
tool_registry.register(traffic_tool)
tool_registry.register(location_tool)
tool_registry.register(time_tool)
tool_registry.register(calendar_tool)
tool_registry.register(reminder_tool)
tool_registry.register(list_reminders_tool)
tool_registry.register(complete_reminder_tool)

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
    "TimeTool",
    "time_tool",
    "CalendarTool",
    "calendar_tool",
    "ReminderTool",
    "reminder_tool",
    "ListRemindersTool",
    "list_reminders_tool",
    "CompleteReminderTool",
    "complete_reminder_tool",
]
