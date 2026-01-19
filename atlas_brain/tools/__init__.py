"""
Atlas Tools - Organized by category for mode-based filtering.

Tool categories:
- shared: Available in all modes (time, weather, location, traffic)
- home: Smart home device control (presence-aware)
- scheduling: Calendar, reminders, appointments
- security: Cameras, detection, access control
- comms: Email, notifications, messaging
- display: Monitor and TV display control
"""

from .base import Tool, ToolParameter, ToolResult
from .registry import ToolRegistry, tool_registry

# Shared tools (all modes)
from .shared import (
    TimeTool, time_tool,
    WeatherTool, weather_tool,
    LocationTool, location_tool,
    TrafficTool, traffic_tool,
)

# Home mode tools
from .home import (
    LightsNearUserTool, lights_near_user,
    MediaNearUserTool, media_near_user,
    SceneNearUserTool, scene_near_user,
    WhereAmITool, where_am_i,
)

# Scheduling tools (receptionist + comms modes)
from .scheduling import (
    CalendarTool, calendar_tool,
    ReminderTool, reminder_tool,
    ListRemindersTool, list_reminders_tool,
    CompleteReminderTool, complete_reminder_tool,
    CheckAvailabilityTool, check_availability_tool,
    BookAppointmentTool, book_appointment_tool,
    CancelAppointmentTool, cancel_appointment_tool,
    RescheduleAppointmentTool, reschedule_appointment_tool,
    LookupCustomerTool, lookup_customer_tool,
)

# Communications tools (receptionist + comms modes)
from .comms import (
    EmailTool, email_tool,
    EstimateEmailTool, estimate_email_tool,
    ProposalEmailTool, proposal_email_tool,
    NotifyTool, notify_tool,
)

# Security tools
from .security import (
    Camera, Detection, SecurityZone,
    ListCamerasTool, list_cameras_tool,
    GetCameraStatusTool, get_camera_status_tool,
    StartRecordingTool, start_recording_tool,
    StopRecordingTool, stop_recording_tool,
    PTZControlTool, ptz_control_tool,
    GetCurrentDetectionsTool, get_current_detections_tool,
    QueryDetectionsTool, query_detections_tool,
    GetPersonAtLocationTool, get_person_at_location_tool,
    GetMotionEventsTool, get_motion_events_tool,
    ListZonesTool, list_zones_tool,
    GetZoneStatusTool, get_zone_status_tool,
    ArmZoneTool, arm_zone_tool,
    DisarmZoneTool, disarm_zone_tool,
)

# Display tools
from .display import (
    ShowCameraFeedTool, show_camera_feed_tool,
    CloseCameraFeedTool, close_camera_feed_tool,
)

# Register all tools
# Shared
tool_registry.register(time_tool)
tool_registry.register(weather_tool)
tool_registry.register(location_tool)
tool_registry.register(traffic_tool)

# Home
tool_registry.register(lights_near_user)
tool_registry.register(media_near_user)
tool_registry.register(scene_near_user)
tool_registry.register(where_am_i)

# Scheduling
tool_registry.register(calendar_tool)
tool_registry.register(reminder_tool)
tool_registry.register(list_reminders_tool)
tool_registry.register(complete_reminder_tool)
tool_registry.register(check_availability_tool)
tool_registry.register(book_appointment_tool)
tool_registry.register(cancel_appointment_tool)
tool_registry.register(reschedule_appointment_tool)
tool_registry.register(lookup_customer_tool)

# Communications
tool_registry.register(email_tool)
tool_registry.register(estimate_email_tool)
tool_registry.register(proposal_email_tool)
tool_registry.register(notify_tool)

# Security - Camera
tool_registry.register(list_cameras_tool)
tool_registry.register(get_camera_status_tool)
tool_registry.register(start_recording_tool)
tool_registry.register(stop_recording_tool)
tool_registry.register(ptz_control_tool)

# Security - Detection
tool_registry.register(get_current_detections_tool)
tool_registry.register(query_detections_tool)
tool_registry.register(get_person_at_location_tool)
tool_registry.register(get_motion_events_tool)

# Security - Access control
tool_registry.register(list_zones_tool)
tool_registry.register(get_zone_status_tool)
tool_registry.register(arm_zone_tool)
tool_registry.register(disarm_zone_tool)

# Display
tool_registry.register(show_camera_feed_tool)
tool_registry.register(close_camera_feed_tool)

__all__ = [
    # Base
    "Tool",
    "ToolParameter",
    "ToolResult",
    "ToolRegistry",
    "tool_registry",
    # Shared
    "TimeTool", "time_tool",
    "WeatherTool", "weather_tool",
    "LocationTool", "location_tool",
    "TrafficTool", "traffic_tool",
    # Home
    "LightsNearUserTool", "lights_near_user",
    "MediaNearUserTool", "media_near_user",
    "SceneNearUserTool", "scene_near_user",
    "WhereAmITool", "where_am_i",
    # Scheduling
    "CalendarTool", "calendar_tool",
    "ReminderTool", "reminder_tool",
    "ListRemindersTool", "list_reminders_tool",
    "CompleteReminderTool", "complete_reminder_tool",
    "CheckAvailabilityTool", "check_availability_tool",
    "BookAppointmentTool", "book_appointment_tool",
    "CancelAppointmentTool", "cancel_appointment_tool",
    "RescheduleAppointmentTool", "reschedule_appointment_tool",
    "LookupCustomerTool", "lookup_customer_tool",
    # Communications
    "EmailTool", "email_tool",
    "EstimateEmailTool", "estimate_email_tool",
    "ProposalEmailTool", "proposal_email_tool",
    "NotifyTool", "notify_tool",
    # Security
    "Camera", "Detection", "SecurityZone",
    "ListCamerasTool", "list_cameras_tool",
    "GetCameraStatusTool", "get_camera_status_tool",
    "StartRecordingTool", "start_recording_tool",
    "StopRecordingTool", "stop_recording_tool",
    "PTZControlTool", "ptz_control_tool",
    "GetCurrentDetectionsTool", "get_current_detections_tool",
    "QueryDetectionsTool", "query_detections_tool",
    "GetPersonAtLocationTool", "get_person_at_location_tool",
    "GetMotionEventsTool", "get_motion_events_tool",
    "ListZonesTool", "list_zones_tool",
    "GetZoneStatusTool", "get_zone_status_tool",
    "ArmZoneTool", "arm_zone_tool",
    "DisarmZoneTool", "disarm_zone_tool",
    # Display
    "ShowCameraFeedTool", "show_camera_feed_tool",
    "CloseCameraFeedTool", "close_camera_feed_tool",
]
