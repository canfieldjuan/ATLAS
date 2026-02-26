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
from .calendar import CalendarTool, calendar_tool, CreateCalendarEventTool, create_calendar_event_tool
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
    QueryEmailHistoryTool,
    query_email_history_tool,
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
    WhoIsHereTool,
    who_is_here,
)
from .security import (
    # Camera tools
    ListCamerasTool,
    list_cameras_tool,
    GetCameraStatusTool,
    get_camera_status_tool,
    StartRecordingTool,
    start_recording_tool,
    StopRecordingTool,
    stop_recording_tool,
    PTZControlTool,
    ptz_control_tool,
    # Detection tools
    GetCurrentDetectionsTool,
    get_current_detections_tool,
    QueryDetectionsTool,
    query_detections_tool,
    GetPersonAtLocationTool,
    get_person_at_location_tool,
    GetMotionEventsTool,
    get_motion_events_tool,
    # Access control tools
    ListZonesTool,
    list_zones_tool,
    GetZoneStatusTool,
    get_zone_status_tool,
    ArmZoneTool,
    arm_zone_tool,
    DisarmZoneTool,
    disarm_zone_tool,
)
from .display import (
    ShowCameraFeedTool,
    show_camera_feed_tool,
    CloseCameraFeedTool,
    close_camera_feed_tool,
)
from .digest import RunDigestTool, digest_tool

# Register tools on import
# NOTE: Read-only tools are registered for fast-path execution. Scheduling, reminder,
# calendar, and email tools are also registered for LLM tool calling in their workflows.
tool_registry.register(weather_tool)
tool_registry.register(traffic_tool)
tool_registry.register(location_tool)
tool_registry.register(time_tool)
# Read-only workflow tools - safe for direct execution (no multi-turn state needed)
tool_registry.register(calendar_tool)       # get_calendar - read-only query
tool_registry.register(list_reminders_tool) # list_reminders - read-only query
tool_registry.register(notify_tool)
tool_registry.register(lights_near_user)
tool_registry.register(media_near_user)
tool_registry.register(scene_near_user)
tool_registry.register(where_am_i)
tool_registry.register(who_is_here)
# Security - Camera tools
tool_registry.register(list_cameras_tool)
tool_registry.register(get_camera_status_tool)
tool_registry.register(start_recording_tool)
tool_registry.register(stop_recording_tool)
tool_registry.register(ptz_control_tool)
# Security - Detection tools
tool_registry.register(get_current_detections_tool)
tool_registry.register(query_detections_tool)
tool_registry.register(get_person_at_location_tool)
tool_registry.register(get_motion_events_tool)
# Security - Access control tools
tool_registry.register(list_zones_tool)
tool_registry.register(get_zone_status_tool)
tool_registry.register(arm_zone_tool)
tool_registry.register(disarm_zone_tool)
# Display tools
tool_registry.register(show_camera_feed_tool)
tool_registry.register(close_camera_feed_tool)
# Digest tools
tool_registry.register(digest_tool)
# Scheduling tools - available for LLM tool calling in booking conversations
tool_registry.register(check_availability_tool)
tool_registry.register(book_appointment_tool)
tool_registry.register(lookup_customer_tool)
tool_registry.register(cancel_appointment_tool)
tool_registry.register(reschedule_appointment_tool)
# Reminder tools - available for LLM tool calling in reminder conversations
tool_registry.register(reminder_tool)
tool_registry.register(complete_reminder_tool)
# Calendar tools - create event for LLM tool calling in calendar conversations
tool_registry.register(create_calendar_event_tool)
# Email tools - available for LLM tool calling in email conversations
tool_registry.register(email_tool)
tool_registry.register(estimate_email_tool)
tool_registry.register(proposal_email_tool)
tool_registry.register(query_email_history_tool)

# WORKFLOW_ONLY_TOOLS is defined in services/intent_router.py (the routing module)
# and re-exported here for convenience.  See that module for the full explanation.
from ..services.intent_router import WORKFLOW_ONLY_TOOLS  # noqa: E402

__all__ = [
    "Tool",
    "ToolParameter",
    "ToolResult",
    "ToolRegistry",
    "tool_registry",
    "WORKFLOW_ONLY_TOOLS",
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
    "CreateCalendarEventTool",
    "create_calendar_event_tool",
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
    "QueryEmailHistoryTool",
    "query_email_history_tool",
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
    "WhoIsHereTool",
    "who_is_here",
    # Security - Camera tools
    "ListCamerasTool",
    "list_cameras_tool",
    "GetCameraStatusTool",
    "get_camera_status_tool",
    "StartRecordingTool",
    "start_recording_tool",
    "StopRecordingTool",
    "stop_recording_tool",
    "PTZControlTool",
    "ptz_control_tool",
    # Security - Detection tools
    "GetCurrentDetectionsTool",
    "get_current_detections_tool",
    "QueryDetectionsTool",
    "query_detections_tool",
    "GetPersonAtLocationTool",
    "get_person_at_location_tool",
    "GetMotionEventsTool",
    "get_motion_events_tool",
    # Security - Access control tools
    "ListZonesTool",
    "list_zones_tool",
    "GetZoneStatusTool",
    "get_zone_status_tool",
    "ArmZoneTool",
    "arm_zone_tool",
    "DisarmZoneTool",
    "disarm_zone_tool",
    # Display tools
    "ShowCameraFeedTool",
    "show_camera_feed_tool",
    "CloseCameraFeedTool",
    "close_camera_feed_tool",
    # Digest tool
    "RunDigestTool",
    "digest_tool",
]
