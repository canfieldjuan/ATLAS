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
from .notify import NotifyTool, notify_tool
from .email import (
    EmailTool,
    email_tool,
    EstimateEmailTool,
    estimate_email_tool,
    ProposalEmailTool,
    proposal_email_tool,
)
from .scheduling import (
    CheckAvailabilityTool,
    check_availability_tool,
    BookAppointmentTool,
    book_appointment_tool,
    CancelAppointmentTool,
    cancel_appointment_tool,
    RescheduleAppointmentTool,
    reschedule_appointment_tool,
    LookupCustomerTool,
    lookup_customer_tool,
)
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

# Register tools on import
tool_registry.register(weather_tool)
tool_registry.register(traffic_tool)
tool_registry.register(location_tool)
tool_registry.register(time_tool)
tool_registry.register(calendar_tool)
tool_registry.register(reminder_tool)
tool_registry.register(list_reminders_tool)
tool_registry.register(complete_reminder_tool)
tool_registry.register(notify_tool)
tool_registry.register(email_tool)
tool_registry.register(estimate_email_tool)
tool_registry.register(proposal_email_tool)
tool_registry.register(check_availability_tool)
tool_registry.register(book_appointment_tool)
tool_registry.register(cancel_appointment_tool)
tool_registry.register(reschedule_appointment_tool)
tool_registry.register(lookup_customer_tool)
tool_registry.register(lights_near_user)
tool_registry.register(media_near_user)
tool_registry.register(scene_near_user)
tool_registry.register(where_am_i)

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
    "NotifyTool",
    "notify_tool",
    "EmailTool",
    "email_tool",
    "EstimateEmailTool",
    "estimate_email_tool",
    "ProposalEmailTool",
    "proposal_email_tool",
    "CheckAvailabilityTool",
    "check_availability_tool",
    "BookAppointmentTool",
    "book_appointment_tool",
    "CancelAppointmentTool",
    "cancel_appointment_tool",
    "RescheduleAppointmentTool",
    "reschedule_appointment_tool",
    "LookupCustomerTool",
    "lookup_customer_tool",
    "LightsNearUserTool",
    "lights_near_user",
    "MediaNearUserTool",
    "media_near_user",
    "SceneNearUserTool",
    "scene_near_user",
    "WhereAmITool",
    "where_am_i",
]
